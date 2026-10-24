// src/core/mtp.cpp - see include/strata/core/mtp.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/core/on_device.hpp"

#include "strata/core/native_head.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/platform/memory.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/ngram.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <vector>
#if defined(_WIN32)
#include <intrin.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#endif

#ifdef STRATA_NATIVE_EXPERTS
#include "ggml.h"   // --mtp-q4: the head rows' and projections' formats, dequantized and requantized at load
#endif

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;

// #783 PR-i: STRATA_MTP_CATCHUP_ALL=1 catches the drafter's K/V up for the whole verified window, rejected rows included
bool mtp_catchup_all() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MTP_CATCHUP_ALL");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}
constexpr int GGML_Q8_0 = 8;
constexpr int GGML_Q4_0 = 2;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void **h, void **d) try {
    /*
    DPCT1048: The original value cudaHostAllocMapped is not meaningful in the
    migrated code and was removed or replaced with 0. You may need to check the
    migrated code.
    */
    if (DPCT_CHECK_ERROR(*h = strata::host_malloc_polled(
                             bytes, dpct::get_in_order_queue())) !=
        0 || *h == nullptr) return false;   // the host polls what the draft steps write: uncached host memory (sycl_queue.hpp)
    std::memset(*h, 0, bytes);
    return DPCT_CHECK_ERROR(*d = (void *)*h) == 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) { return qsa_shapes(g); }

// 64-bit seek/tell on a `FILE*`: `fseek`/`ftell` take a 32-bit `long` on Windows and would wrap past 2 GiB.
#if defined(_WIN32)
#define STRATA_FILE_SEEK64(f, o, w) _fseeki64((f), (long long) (o), (w))
#define STRATA_FILE_TELL64(f) _ftelli64(f)
#else
#define STRATA_FILE_SEEK64(f, o, w) fseeko((f), (off_t) (o), (w))
#define STRATA_FILE_TELL64(f) ftello(f)
#endif

bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    // Loader fix (0.1.15+loaderfix.2): `ifstream::read` reaches the disk as 4095-byte reads on MSVC - the same
    // split `load_experts_ranges` had - so the drafter's 111 MiB dense blob paid 4 KiB per operation on a cold
    // start.  `fread` passes a request bigger than the stream buffer straight to `ReadFile()`.
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    // A `FILE*` has no destructor and the short-read path below returns early: the guard closes on every path.
    struct Closer {
        FILE* f;
        ~Closer() { if (f != nullptr) std::fclose(f); }
    } closer{f};
    if (STRATA_FILE_SEEK64(f, 0, SEEK_END) != 0) return false;
    const long long n = (long long) STRATA_FILE_TELL64(f);
    if (n < 0 || STRATA_FILE_SEEK64(f, 0, SEEK_SET) != 0) return false;
#if !defined(_WIN32)
    strata::platform::advise_willneed(fileno(f), 0, (uint64_t) n);
#endif
    out.resize((size_t) n);
    return n == 0 || std::fread(out.data(), 1, (size_t) n, f) == (size_t) n;
}

}  // namespace

MtpDrafter::~MtpDrafter() {
    if (cs_) cs_->wait();
    for (auto &e : prefill_exec_) if (e) delete (e);
    for (auto &e : prefill_dev_exec_) if (e) delete (e);
    if (pf_dev_) sycl::free(pf_dev_, dpct::get_in_order_queue());
    if (dense4_) sycl::free(dense4_, dpct::get_in_order_queue());
    for (auto &e : round_exec_) if (e) delete (e);
    for (auto &e : step_exec_) if (e) delete (e);
    for (auto &e : round_exec_c_) if (e) delete (e);
    for (auto &e : step_exec_c_) if (e) delete (e);
    if (cparams_) sycl::free(cparams_, dpct::get_in_order_queue());
    if (cring_) sycl::free(cring_, dpct::get_in_order_queue());
    if (dinv_) sycl::free(dinv_, dpct::get_in_order_queue());
    if (cscratch_) sycl::free(cscratch_, dpct::get_in_order_queue());
    if (h_cparams_) sycl::free(h_cparams_, dpct::get_in_order_queue());
    if (h_chist_) sycl::free(h_chist_, dpct::get_in_order_queue());
    if (cs_) dpct::get_current_device().destroy_queue(cs_);
    if (side_) dpct::get_current_device().destroy_queue(side_);
    if (sh_fork_) dpct::destroy_event(sh_fork_);
    if (sh_join_) dpct::destroy_event(sh_join_);
    if (owns_weights_ && dense_) sycl::free(dense_, dpct::get_in_order_queue());
    if (owns_weights_ && experts_)
        sycl::free(experts_, dpct::get_in_order_queue());
    if (state_arena_) sycl::free(state_arena_, dpct::get_in_order_queue());
    if (arena_) sycl::free(arena_, dpct::get_in_order_queue());
    if (head_logits_) sycl::free(head_logits_, dpct::get_in_order_queue());
    if (owns_draft_head_ && dhead_) {
      strata::kernels::native_q6_k_unpack(dhead_);
      sycl::free(dhead_, dpct::get_in_order_queue());
    }
    if (owns_draft_head_ && dvocab_)
        sycl::free(dvocab_, dpct::get_in_order_queue());
    if (ev_chain_) dpct::destroy_event(ev_chain_);
    for (dpct::event_ptr e : ev_step_) if (e) dpct::destroy_event(e);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_row_, h_out_, h_prob_, h_force_};
    for (void *h : hosts) if (h) strata::host_free_polled(h, dpct::get_in_order_queue());
}

const float* MtpDrafter::f32(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "f32") return (const float*) (dense_ + t.off);
    return nullptr;
}
const uint16_t* MtpDrafter::bf16(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "bf16") return (const uint16_t*) (dense_ + t.off);
    return nullptr;
}
const void* MtpDrafter::q8(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "q8_0") return dense_ + t.off;
    return nullptr;
}
const void* MtpDrafter::wq(const char* name, int& type) const {
    if (dense4_ != nullptr)
        for (const auto& e : q4_off_)
            if (e.first == name) { type = GGML_Q4_0; return dense4_ + e.second; }
    type = GGML_Q8_0;
    return q8(name);
}

// --mtp-q4: every Q8_0 tensor of dense.bin as Q4_0 (ggml's reference quantizer), in one device buffer
bool MtpDrafter::make_q4_dense(const std::vector<uint8_t>& blob, std::string& err) {
#ifdef STRATA_NATIVE_EXPERTS
    uint64_t total = 0;
    std::vector<std::pair<const Tensor*, uint64_t>> plan;
    for (const auto& t : tensors_) {
        if (t.kind != "q8_0" || t.cols % 32 != 0) continue;
        plan.push_back({&t, total});
        total += ((uint64_t) t.rows * (uint64_t) (t.cols / 32) * 18 + 255) & ~255ull;
    }
    std::vector<uint8_t> host((size_t) total);
    std::vector<float> row;
    const auto* q8t = ggml_get_type_traits(GGML_TYPE_Q8_0);
    for (const auto& [t, off] : plan) {
        row.resize((size_t) t->cols);
        const uint64_t in_row = (uint64_t) (t->cols / 32) * 34, out_row = (uint64_t) (t->cols / 32) * 18;
        for (int64_t r = 0; r < t->rows; ++r) {
            q8t->to_float(blob.data() + t->off + (uint64_t) r * in_row, row.data(), t->cols);
            ggml_quantize_chunk(GGML_TYPE_Q4_0, row.data(), host.data() + off + (uint64_t) r * out_row, 0, 1, t->cols,
                                nullptr);
        }
        q4_off_.push_back({t->name, off});
    }
    dense4_ = (uint8_t*) strata::malloc_device_guarded((size_t) total, dpct::get_in_order_queue());   // SYCL port: cudaMalloc/cudaMemcpy
    if (dense4_ == nullptr) { err = "mtp: the Q4_0 projections do not fit"; return false; }
    dpct::get_in_order_queue().memcpy(dense4_, host.data(), (size_t) total).wait();
    vram_ += total;
    std::fprintf(stderr, "strata mtp: --mtp-q4: %zu projections as Q4_0 (%.1f MiB)\n", plan.size(), (double) total / 1048576.0);
    return true;
#else
    (void) blob;
    err = "mtp: --mtp-q4 needs a build with native experts (ggml)";
    return false;
#endif
}

// --mtp-q4: the draft head's subset rows as Q4_0 (dequantized from the main head's format)
bool MtpDrafter::make_q4_head(std::string& err) {
#ifdef STRATA_NATIVE_EXPERTS
    const int64_t N = g_->n_embd;
    const auto* tt = ggml_get_type_traits((ggml_type) head_->type());
    if (tt == nullptr || tt->to_float == nullptr || N % 32 != 0) { err = "mtp: --mtp-q4 cannot read the head's format"; return false; }
    const size_t in_row = head_->row_bytes(), out_row = (size_t) (N / 32) * 18;
    std::vector<uint8_t> src((size_t) n_dvocab_ * in_row), dst((size_t) n_dvocab_ * out_row);
    try { dpct::get_in_order_queue().memcpy(src.data(), dhead_, src.size()).wait(); }   // SYCL port: cudaMemcpy
    catch (sycl::exception const&) { err = "mtp: reading the draft head"; return false; }
    std::vector<float> row((size_t) N);
    for (int64_t r = 0; r < n_dvocab_; ++r) {
        tt->to_float(src.data() + (size_t) r * in_row, row.data(), N);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, row.data(), dst.data() + (size_t) r * out_row, 0, 1, N, nullptr);
    }
    // the Q4_0 rows replace the subset in place (they are smaller)
    try { dpct::get_in_order_queue().memcpy(dhead_, dst.data(), dst.size()).wait(); }
    catch (sycl::exception const&) { err = "mtp: writing the draft head"; return false; }
    dhead_type_ = GGML_Q4_0;
    std::fprintf(stderr, "strata mtp: --mtp-q4: draft head over %lld tokens as Q4_0 (%.1f MiB read per step, was %.1f)\n",
                 (long long) n_dvocab_, (double) dst.size() / 1048576.0, (double) src.size() / 1048576.0);
    return true;
#else
    err = "mtp: --mtp-q4 needs a build with native experts (ggml)";
    return false;
#endif
}

bool MtpDrafter::load(const std::string &rt_dir, const ModelGeometry &g,
                      SessionState &ss, int max_t, std::string &err,
                      int64_t window, const MtpDrafter *shared) try {
    device_ =
        dpct::get_current_device_id(); // a layer split's last stage on another
                                       // GPU: the drafter lives there
    g_ = &g;
    ss_ = &ss;
    if (ple_ss_ == nullptr) ple_ss_ = &ss;
    max_t_ = max_t;
    rt_dir_ = rt_dir;
    if (max_t < 1 || max_t > strata::kernels::kVerifyMaxT) { err = "mtp: max_t out of range"; return false; }
    if (shared != nullptr) {
        if (shared->device_ != device_ || shared->g_ != &g || shared->dense_ == nullptr ||
            shared->experts_ == nullptr || shared->rt_dir_ != rt_dir) {
            err = "mtp: incompatible shared weights";
            return false;
        }
        dense_ = shared->dense_;
        experts_ = shared->experts_;
        tensors_ = shared->tensors_;
        owns_weights_ = false;
        owns_draft_head_ = false;
    }
    // Loader fix (0.1.15+loaderfix.2): the two reads below are the whole “drafter files” cost; reporting
    // them apart from the rest of the stage is what makes the next regression visible.
    const auto t_files = std::chrono::steady_clock::now();
    // ---- the index and the dense weights
    if (shared == nullptr) {
        std::ifstream idx(rt_dir + "/dense.txt");
        if (!idx) { err = "mtp: cannot open " + rt_dir + "/dense.txt (run tools/mtp_rt.py)"; return false; }
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty()) continue;
            std::istringstream is(line);
            Tensor t;
            is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
            if (!is) { err = "mtp: malformed dense.txt line: " + line; return false; }
            tensors_.push_back(t);
        }
        std::vector<uint8_t> blob;
        if (!read_file(rt_dir + "/dense.bin", blob)) { err = "mtp: cannot read dense.bin"; return false; }
        const dpct::err0 alloc =
            DPCT_CHECK_ERROR(dense_ = (uint8_t *)strata::malloc_device_guarded(
                                 blob.size(), dpct::get_in_order_queue()));
        /*
        DPCT1000: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (alloc != 0) {
            size_t free_bytes = 0, total_bytes = 0;
            /*
            DPCT1106: 'cudaMemGetInfo' was migrated with the Intel
            extensions for device information which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const dpct::err0 info =
                DPCT_CHECK_ERROR(dpct::get_current_device().get_memory_info(
                    free_bytes, total_bytes));
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1001: The statement could not be removed.
            */
            err = "mtp: dense weights allocation failed (" +
                  std::string(dpct::get_error_string_dummy(alloc)) +
                  "), requested " + std::to_string(blob.size() >> 20) +
                  " MiB, CUDA0 free " +
                  (info == 0 ? std::to_string(free_bytes >> 20) + " MiB"
                             : "unknown");
            return false;
        }
        dpct::get_in_order_queue().memcpy(dense_, blob.data(), blob.size()).wait();
        vram_ += blob.size();
        if (q4_ && !make_q4_dense(blob, err)) return false;
    }
    // ---- the 512 routed experts, one blob each
    if (shared == nullptr) {
        const uint64_t bytes = (uint64_t) g.n_expert * strata::kernels::cpu::BLOB;
        // Loader fix (0.1.15+loaderfix.2): each 64 MiB read below reached the disk as ~16k 4095-byte reads under
        // MSVC's `basic_filebuf::xsgetn`, which is what made 675 MiB of drafter experts take minutes.
        FILE* f = std::fopen((rt_dir + "/experts.bin").c_str(), "rb");
        if (f == nullptr) { err = "mtp: cannot open experts.bin"; return false; }
        struct Closer {
            FILE* f;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } closer{f};
        if (DPCT_CHECK_ERROR(experts_ = (uint8_t *)strata::malloc_device_guarded(
                                 bytes, dpct::get_in_order_queue())) != 0) {
            err = "mtp: the 512 experts do not fit in VRAM"; return false;
        }
#if !defined(_WIN32)
        strata::platform::advise_willneed(fileno(f), 0, bytes);
#endif
        std::vector<uint8_t> chunk(64u << 20);
        for (uint64_t off = 0; off < bytes;) {
            const uint64_t n = std::min<uint64_t>(chunk.size(), bytes - off);
            if (std::fread(chunk.data(), 1, (size_t) n, f) != (size_t) n) { err = "mtp: experts.bin is truncated"; return false; }
            dpct::get_in_order_queue()
                .memcpy(experts_ + off, chunk.data(), n)
                .wait();
            off += n;
        }
        vram_ += bytes;
    }
    const char* required[] = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                              "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.shared_expert.gate_proj.weight",
                              "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"};
    for (const char* n : required) if (!q8(n)) { err = std::string("mtp: ") + n + " is missing (q8_0)"; return false; }

    // ---- the layer's own K/V (dense attention: no indexer state is read)
    const strata::kernels::QsaShapes s = shapes_of(g);
    const int64_t max_cells = ss.qsa_states[ss.qsa_primary()].max_cells;
    // KV streaming: the drafter only reads its last `window` cells, so with streaming on its K/V is a ring of the
    // window (plus the cells a round writes ahead of its queries) over a host copy, refilled on a resume. The host copy
    // is pinned after the expert arena has pinned what it could: if it does not fit, the K/V stays whole in VRAM.
    int64_t ring = (window > 0 && window < max_cells) ? window + 4 * (int64_t) max_t + 64 : 0;
    // K8V4 never applies to the drafter: its own attention paths (below, and verify.cpp) handle whole formats
    // only, whatever ring shape it takes (0, a window, or the -1 fully-resident fallback).
    const bool kv_hybrid_was = qsa_kv_hybrid();
    const bool kv_int8_was = qsa_kv_int8();
    qsa_set_kv_hybrid(false);
    if (kv_hybrid_was) qsa_set_kv_int8(true);   // the drafter under --kv k8v4: plain INT8
    uint64_t sb = qsa_state_bytes(g, max_cells, false, ring);
    if (DPCT_CHECK_ERROR(state_arena_ = (void *)strata::malloc_device_guarded(
                             sb, dpct::get_in_order_queue())) != 0) {
        err = "mtp: the K/V state does not fit"; return false;
    }
    if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[ss.qsa_primary()], ring) == 0) {
        if (st_.kv_mode == 0) { err = "mtp: state init failed"; return false; }
        std::fprintf(stderr, "strata mtp: no pinned RAM left for the draft layer's K/V copy; keeping it in VRAM\n");
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        sycl::free(state_arena_, dpct::get_in_order_queue());
        st_ = QsaState{};
        ring = -1;   // fully resident
        sb = qsa_state_bytes(g, max_cells, false, ring);
        if (DPCT_CHECK_ERROR(state_arena_ = (void *)strata::malloc_device_guarded(
                                 sb, dpct::get_in_order_queue())) != 0) {
            err = "mtp: the K/V state does not fit"; return false;
        }
        if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[ss.qsa_primary()], ring) == 0) { err = "mtp: state init failed"; return false; }
    }
    qsa_set_kv_int8(kv_int8_was);
    qsa_set_kv_hybrid(kv_hybrid_was);
    qsa_state_zero(st_, g, nullptr);
    dpct::get_current_device().queues_wait_and_throw();
    vram_ += sb;

    // ---- buffers
    window_ = (window > 0 && window < max_cells) ? window : 0;
    cap_ = (((window_ > 0 ? window_ : max_cells) + 63) / 64) * 64;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);
    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t R2 = 2 * T;   // step/pos rows: T catch-up rows + up to T-2 chain steps
    bool ok = mapped(T * 4 + 64, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(R2 * 4 * 4 + 64, (void**) &h_step_, (void**) &m_step_) &&
              mapped(R2 * NH * 4 + 64, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped(64, (void**) &h_row_, (void**) &m_row_) &&
              mapped(T * 4 + 64, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * 4 + 64, (void**) &h_prob_, (void**) &m_prob_) &&
              (!force_on_ || mapped(64, (void**) &h_force_, (void**) &m_force_));
    if (!ok) { err = "mtp: mapped staging failed"; return false; }
    if (h_force_ != nullptr) for (int j = 0; j < 16; ++j) h_force_[j] = -1;
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(R2 * 4); pos_ = b.take<int32_t>(R2 * NH); row_ = b.take<int32_t>(4);
        ident_ = b.take<int32_t>(T * (uint64_t) cap_);
        Rin_ = b.take<float>(T * HC * N); R_ = b.take<float>(T * HC * N);
        emb_ = b.take<float>(T * N); en_ = b.take<float>(T * N); e2_ = b.take<float>(T * N);
        hn_ = b.take<float>(T * HC * N); h2_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); bo_ = b.take<float>(T * N);
        xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes((int) (NH * HD), 8));
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>((uint64_t) attn_scratch_floats_);   // the full layer runs one row at a time
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); y_ = b.take<float>(T * N);
        sample_ = b.take<float>(T * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        grp_ptr_ = b.take<unsigned long long>(T * K); grp_start_ = b.take<int32_t>(T * K + 1);
        grp_counts_ = b.take<int32_t>(4);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        hit_scratch_ = b.take<uint8_t>(strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff));
        sh_scratch_ = (float*) b.take<uint8_t>(strata::kernels::shared_expert_scratch_bytes(g.n_ff));
        x_bf16_ = b.take<uint16_t>(N);
        out_ids_ = b.take<int32_t>(T + 4);
        probs_ = b.take<float>(T + 4);
        arg_scratch_ = b.take<uint8_t>(strata::kernels::argmax_rows_scratch_bytes((int) T));
        top_scratch_ = b.take<uint8_t>(strata::kernels::row_top_prob_scratch_bytes((int) T));
        dummy_inj_ = b.take<float>(HC);
    };
    Bump count;
    carve(count);
    if (DPCT_CHECK_ERROR(arena_ = (void *)strata::malloc_device_guarded(
                             count.used, dpct::get_in_order_queue())) != 0) {
        err = "mtp: buffers do not fit"; return false;
    }
    strata::big_fill_zero(dpct::get_in_order_queue(), arena_, count.used);
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    vram_ += count.used;
    {
        std::vector<int32_t> id((size_t) (T * (uint64_t) cap_));
        for (uint64_t t = 0; t < T; ++t)
            for (int64_t i = 0; i < cap_; ++i) id[(size_t) (t * (uint64_t) cap_ + (uint64_t) i)] = (int32_t) i;
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        dpct::get_in_order_queue().memcpy(ident_, id.data(), id.size() * 4).wait();
    }
    // --pipeline-windows 2 (the forcing graphs): the chain shares its card with stage 1's windows and the next launch
    // on stage 0 waits for it, so its stream gets the highest priority there (STRATA_MTP_PRIORITY=0: the default)
    bool prio = false;
#if !defined(STRATA_USE_HIP)   // (HIP: the default priority)
    static const bool hi_prio = [] { const char* v = std::getenv("STRATA_MTP_PRIORITY"); return v == nullptr || std::atoi(v) != 0; }();
    int prio_lo = 0, prio_hi = 0;
    /*
    DPCT1027: The call to cudaDeviceGetStreamPriorityRange was replaced with
    0 because SYCL currently does not support get queue priority range.
    */
    if (force_on_ && hi_prio && 0 == 0)
        /*
        DPCT1025: The SYCL queue is created ignoring the flag and priority
        options.
        */
        prio = DPCT_CHECK_ERROR(
                   cs_ = dpct::get_current_device().create_queue(true)) == 0;
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
#endif
    /*
    DPCT1025: The SYCL queue is created ignoring the flag and priority
    options.
    */
    if (!prio && DPCT_CHECK_ERROR(cs_ = dpct::get_current_device().create_queue(
                                      true)) != 0) {
      err = "mtp: stream"; return false;
    }
    /*
    DPCT1025: The SYCL queue is created ignoring the flag and priority
    options.
    */
    if (DPCT_CHECK_ERROR(
            side_ = dpct::get_current_device().create_queue(true)) != 0 ||
        DPCT_CHECK_ERROR(sh_fork_ = new sycl::event()) != 0 ||
        DPCT_CHECK_ERROR(sh_join_ = new sycl::event()) != 0) {
        err = "mtp: streams";
        return false;
    }
    const double files_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_files).count();
    if (shared != nullptr) {
        std::fprintf(stderr, "strata mtp: shared draft weights, %.0f MiB of private state and buffers\n",
                     (double) vram_ / 1048576.0);
    } else {
        std::fprintf(stderr, "strata mtp: draft layer loaded, %.0f MiB of VRAM (experts %.0f, dense %.0f), files read in %.2f s (%.0f MiB/s)\n",
                     (double) vram_ / 1048576.0, (double) g.n_expert * strata::kernels::cpu::BLOB / 1048576.0,
                     (double) tensors_.back().off / 1048576.0, files_s,
                     files_s > 0 ? ((double) g.n_expert * strata::kernels::cpu::BLOB + (double) tensors_.back().off) /
                                       1048576.0 / files_s : 0.0);
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

namespace {
// STRATA_MTP_FULL_HEAD=1 (a diagnostic): draft over the whole vocabulary (the main head itself) instead of
// rt/draft_vocab.bin's subset - the reference head, to measure what the subset costs in acceptance
bool full_head_env() {
    static const bool on = [] { const char* v = std::getenv("STRATA_MTP_FULL_HEAD"); return v && v[0] == '1'; }();
    return on;
}
}  // namespace

bool MtpDrafter::top2_env() {
    static const bool on = [] { const char* v = std::getenv("STRATA_MTP_TOP2"); return v && v[0] == '1'; }();
    return on;
}

// STRATA_MTP_TOP2=1 (a diagnostic for tree drafts): draft j's runner-up under the draft layer, from the head logits
// the step just wrote (row 0), on the host.  Slow; only for measuring what a second branch would have caught.
void MtpDrafter::record_top2(int j) try {
    const int64_t nv = dhead_ != nullptr ? n_dvocab_ : n_vocab_;
    if ((int64_t) lg_host_.size() < nv) lg_host_.resize((size_t) nv);
    if ((int) top2_.size() < max_t_) top2_.assign((size_t) max_t_, -1);
    if (j < 0 || j >= max_t_) return;
    if (DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(lg_host_.data(), head_logits_,
                                     (size_t)nv * sizeof(float))
                             .wait()) != 0) {
        top2_[(size_t) j] = -1;
        return;
    }
    int64_t b1 = -1, b2 = -1;
    for (int64_t i = 0; i < nv; ++i) {
        const float v = lg_host_[(size_t) i];
        if (b1 < 0 || v > lg_host_[(size_t) b1]) { b2 = b1; b1 = i; }
        else if (b2 < 0 || v > lg_host_[(size_t) b2]) b2 = i;
    }
    top2_[(size_t) j] = b2 < 0 ? -1 : dhead_ != nullptr ? dvocab_host_[(size_t) b2] : (int32_t) b2;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

uint64_t MtpDrafter::bind_bytes(uint64_t head_row_bytes, int64_t n_vocab) const {
    uint64_t bytes = head_logits_ ? 0 : (uint64_t) max_t_ * (uint64_t) n_vocab * sizeof(float);
    if (dhead_ == nullptr && owns_draft_head_ && !full_head_env()) {
        if (FILE* f = std::fopen(vocab_file().c_str(), "rb")) {
            std::fseek(f, 0, SEEK_END);
            const long size = std::ftell(f);
            std::fclose(f);
            if (size >= 4 && size % 4 == 0) bytes += (uint64_t) (size / 4) * head_row_bytes + (uint64_t) size;
        }
    }
    // coupled draft sampling (STRATA_SPEC_COUPLED=1 only): an upper bound - the id -> subset map, the penalty ring,
    // the split scratch (64 lists of 64 entries at most) and the parameters
    if (coupled_draft_env() && cparams_ == nullptr)
        bytes += (uint64_t) n_vocab * 4 + (uint64_t) (kCoupledHistCap + max_t_) * 4 + 64ull * 64ull * 8ull + 4096;
    return bytes;
}

bool MtpDrafter::setup_coupled(std::string& err) try {
    const int64_t nv = dhead_ != nullptr ? n_dvocab_ : n_vocab_;
    const size_t scratch = strata::kernels::coupled_draft_scratch_bytes((int) nv);
    if (scratch == 0) {
        std::fprintf(stderr, "strata mtp: STRATA_SPEC_COUPLED: %lld draft logits are too wide for the coupled sampler; "
                             "argmax drafts\n", (long long) nv);
        return true;
    }
    sycl::queue& q = dpct::get_in_order_queue();
    const size_t ring = (size_t) (kCoupledHistCap + max_t_) * sizeof(int32_t);
    cparams_ = sycl::malloc_device<strata::kernels::SamplerParams>(1, q);
    cring_ = (int32_t*) sycl::malloc_device(ring, q);
    cscratch_ = sycl::malloc_device(scratch, q);
    if (cparams_ == nullptr || cring_ == nullptr || cscratch_ == nullptr ||
        !mapped(sizeof(strata::kernels::SamplerParams), (void**) &h_cparams_, (void**) &m_cparams_) ||
        !mapped((size_t) kCoupledHistCap * sizeof(int32_t), (void**) &h_chist_, (void**) &m_chist_)) {
        err = "mtp: the coupled draft sampler's buffers do not fit";
        return false;
    }
    q.memset(cring_, 0xff, ring).wait();   // -1: no token
    std::fill(h_chist_, h_chist_ + kCoupledHistCap, -1);
    vram_ += sizeof(strata::kernels::SamplerParams) + ring + scratch;
    if (dhead_ != nullptr) {   // token id -> subset index (-1: not in the draft head), for the penalties
        std::vector<int32_t> sub((size_t) n_dvocab_), inv((size_t) n_vocab_, -1);
        q.memcpy(sub.data(), dvocab_, sub.size() * sizeof(int32_t)).wait();
        for (size_t i = 0; i < sub.size(); ++i)
            if (sub[i] >= 0 && sub[i] < n_vocab_ && inv[(size_t) sub[i]] < 0) inv[(size_t) sub[i]] = (int32_t) i;
        dinv_ = (int32_t*) sycl::malloc_device(inv.size() * sizeof(int32_t), q);
        if (dinv_ == nullptr) {
            err = "mtp: the coupled draft sampler's token map does not fit";
            return false;
        }
        q.memcpy(dinv_, inv.data(), inv.size() * sizeof(int32_t)).wait();
        vram_ += inv.size() * sizeof(int32_t);
    }
    if (DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()) !=
        0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        err = std::string("mtp: coupled setup: ") + dpct::get_error_string_dummy(0);
        return false;
    }
    coupled_ok_ = true;
    std::fprintf(stderr, "strata mtp: coupled draft sampling on (STRATA_SPEC_COUPLED): sampled requests draft with the "
                         "target's chain and Philox draw over %lld tokens\n", (long long) nv);
    return true;
} catch (sycl::exception const& e) {
    err = std::string("mtp: coupled setup: ") + e.what();
    return false;
}

void MtpDrafter::set_draft_sampling(const strata::kernels::SamplerParams& sp) {
    coupled_active_ = coupled_ok_ && !sp.greedy && sp.temperature > 0.0f;
    if (!coupled_active_) return;
    *h_cparams_ = sp;   // read by the next round graph (after the previous one has synced)
    std::fill(h_chist_, h_chist_ + kCoupledHistCap, -1);
}

void MtpDrafter::set_draft_history(const int32_t* tail, int64_t n_tail, int32_t next) {
    if (!coupled_active_) return;
    const int h = coupled_hist_len(h_cparams_->penalty_last_n, kCoupledHistCap);
    coupled_hist_base(tail, n_tail, next, h, h_chist_ + (kCoupledHistCap - h));
}

namespace {
// #474: what to change when the draft head's token subset does not fit the VRAM left - a smaller subset that setup
// ships (data/draft_vocab_*.bin; their token counts below), and how to pick it.  The start used to stop at "the draft
// head does not fit" with no hint.  Text only, after the failure: nothing changes for a start that fits.
void draft_head_hint(int64_t n_tokens, int64_t row_bytes) {
    size_t free_b = 0, total_b = 0;
    bool have_free = true;   // SYCL port: the device's own free-memory figure
    try { dpct::get_current_device().get_memory_info(free_b, total_b); } catch (...) { have_free = false; }
    auto mib = [&](int64_t n) { return (double) (n * row_bytes) / 1048576.0; };
    std::string free_s;
    if (have_free) {
        char b[64];
        std::snprintf(b, sizeof(b), " and %.0f MiB is free", (double) free_b / 1048576.0);
        free_s = b;
    }
    std::fprintf(stderr, "strata mtp: the draft head over %lld tokens needs %.0f MiB of VRAM%s.\n",
                 (long long) n_tokens, mib(n_tokens), free_s.c_str());
    struct Subset { const char* name; int64_t tokens; const char* what; };
    static const Subset smaller[] = {{"cyrillic", 58963, "English, code and the Cyrillic script"},
                                     {"en", 40525, "English and code"}};
    std::string opts;
    for (const Subset& s : smaller)
        if (s.tokens < n_tokens) {
            char b[160];
            std::snprintf(b, sizeof(b), "%s--draft-vocab %s (%s, ~%.0f MiB)", opts.empty() ? "" : " or ", s.name,
                          s.what, mib(s.tokens));
            opts += b;
        }
    if (!opts.empty())
        std::fprintf(stderr, "strata mtp: hint: a smaller draft vocabulary needs less VRAM: %s. Start once with it - "
                             "START-HERE.bat --draft-vocab en (Windows) or ./setup.sh --draft-vocab en - and the model "
                             "keeps it (\"draft_vocab\" in its strata-*.json config); or a smaller --context in "
                             "setup.\n",
                     opts.c_str());
    else
        std::fprintf(stderr, "strata mtp: hint: this is already the smallest shipped draft vocabulary: a smaller "
                             "--context (or closing what else uses the GPU) leaves it room.\n");
}
}  // namespace

bool MtpDrafter::bind(const WeightTable &wt, const NativeHead *head,
                      const float *window_R, std::string &err,
                      const MtpDrafter *shared) try {
    const OnDevice on_device(device_);
    wt_ = &wt;
    head_ = head;
    window_R_ = window_R;
    const WeightRef* wo = wt.find("output.weight");
    if (!wo) { err = "mtp: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;
    if (head == nullptr || !head->loaded()) { err = "mtp: the draft layer needs the native head (--native)"; return false; }
    if (head_logits_ == nullptr &&
        DPCT_CHECK_ERROR(head_logits_ = sycl::malloc_device<float>(
                             (size_t)max_t_ * (size_t)n_vocab_,
                             dpct::get_in_order_queue())) != 0) {
        err = "mtp: the draft logits do not fit";
        return false;
    }
    if (shared != nullptr) {
        if (shared->head_ != head || shared->device_ != device_ ||
            shared->n_vocab_ != n_vocab_) {
            err = "mtp: incompatible shared draft head";
            return false;
        }
        dhead_ = shared->dhead_;
        dhead_type_ = shared->dhead_type_;   // the subset head's type (as src/core/mtp.cpp since #1110): -1 failed every --batch-mtp step
        dvocab_ = shared->dvocab_;
        n_dvocab_ = shared->n_dvocab_;
        owns_draft_head_ = false;
    }
    // the draft head's token subset, when tools/draft_vocab.py wrote one
    if (dhead_ == nullptr && shared == nullptr && !full_head_env()) {
        std::vector<uint8_t> raw;
        if (read_file(vocab_file(), raw) && raw.size() >= 4 && raw.size() % 4 == 0) {
            n_dvocab_ = (int64_t) (raw.size() / 4);
            const int64_t row_bytes = (int64_t) head->row_bytes();   // a vocabulary row of the native head
            if (DPCT_CHECK_ERROR(dvocab_ = (int32_t *)sycl::malloc_device(
                                     raw.size(), dpct::get_in_order_queue())) !=
                    0 ||
                DPCT_CHECK_ERROR(dhead_ = (uint8_t *)sycl::malloc_device(
                                     (size_t)(n_dvocab_ * row_bytes),
                                     dpct::get_in_order_queue())) != 0 ||
                dvocab_ == nullptr || dhead_ == nullptr) {   // Level Zero returns null when the VRAM is full
                err = "mtp: the draft head does not fit";
                draft_head_hint(n_dvocab_, row_bytes);
                return false;
            }
            dpct::get_in_order_queue().memcpy(dvocab_, raw.data(), raw.size()).wait();
            dvocab_host_.resize((size_t) n_dvocab_);
            std::memcpy(dvocab_host_.data(), raw.data(), raw.size());
            bool planar_gathered = false;
            if (head->type() == 14 && strata::kernels::native_q6_k_is_planar(head->weights()) && !q4_head_) {
                // STRATA_Q6_PLANAR: the LM head is stored as planes; the subset is gathered from them, planar
                void* pd = strata::kernels::native_q6_k_planar_gather(head->weights(), (int) (row_bytes / 210) * 256, dvocab_,
                                                                      n_dvocab_, "MTP draft head");
                if (pd != nullptr) {
                    sycl::free(dhead_, dpct::get_in_order_queue());
                    dhead_ = (uint8_t*) pd;
                    planar_gathered = true;
                }
            }
            if (!planar_gathered)
                strata::kernels::gather_rows((const uint8_t*) head->weights(), row_bytes, dvocab_, n_dvocab_, dhead_, nullptr);
            dpct::get_current_device().queues_wait_and_throw();
            vram_ += (uint64_t) (n_dvocab_ * row_bytes) + raw.size();
            std::fprintf(stderr, "strata mtp: draft head over %lld tokens (%.1f MiB)\n", (long long) n_dvocab_,
                         (double) (n_dvocab_ * row_bytes) / 1048576.0);
            dhead_type_ = head->type();
            if (q4_head_ && !make_q4_head(err)) return false;
            if (dhead_type_ == 14 && strata::kernels::native_q6_k_packed_enabled())   // STRATA_Q6_PACKED=1
                strata::kernels::native_q6_k_pack(dhead_, (int) (row_bytes / 210) * 256, (int) n_dvocab_,
                                                  "MTP draft head");
            if (!planar_gathered && dhead_type_ == 14 && !q4_head_ && !strata::kernels::native_q6_k_packed_enabled() &&
                strata::kernels::native_q6_k_planar_enabled())   // STRATA_Q6_PLANAR
                dhead_ = (uint8_t*) strata::kernels::native_q6_k_planar_adopt(dhead_, (int) (row_bytes / 210) * 256,
                                                                              (int) n_dvocab_, "MTP draft head");
        }
    }
    if (coupled_draft_env() && cparams_ == nullptr && !setup_coupled(err)) return false;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// The layer for T rows.  full = false stops after the K/V append (the prompt only needs the cache).
bool MtpDrafter::record_forward(int T, int step_row0, dpct::queue_ptr cs,
                                std::string &err) {
    // step_row0 >= 0: the full layer on step rows [step_row0, +T); step_row0 < 0: K/V only on rows [-1 - step_row0, +T)
    const bool full = step_row0 >= 0;
    if (full && T != 1) { err = "mtp: the full layer runs one row at a time (its attention scratch is sized for one)"; return false; }
    if (!record_front(T, full ? step_row0 : -1 - step_row0, cs, err)) return false;
    return !full || record_rest(step_row0, cs, err);
}

bool MtpDrafter::record_front(int T, int row0, dpct::queue_ptr cs,
                              std::string &err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const int64_t N = g.n_embd, HC = g.hc, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const QsaShapes s = shapes_of(g);
    const int32_t* step = step_ + row0 * 4;
    const int32_t* pos = pos_ + row0 * NH;
    // --mtp-q4: each projection's format and weights for this pass (Q8_0 unless the Q4_0 copies exist)
    int wt_fc_embedding, wt_fc_hidden, wt_k_proj, wt_v_proj;
    const void* wp_fc_embedding = wq("fc_embedding.weight", wt_fc_embedding);
    const void* wp_fc_hidden = wq("fc_hidden.weight", wt_fc_hidden);
    const void* wp_k_proj = wq("self_attn.k_proj.weight", wt_k_proj);
    const void* wp_v_proj = wq("self_attn.v_proj.weight", wt_v_proj);
    try {
        // ---- the two input branches
        const WeightRef* we = wt_->find("token_embd.weight");
        if (!we) { err = "mtp: token_embd.weight is missing"; return false; }
        if (const NativeEmbed* ne = native_embed()) {   // plan v0.3 P6: the GGUF-form table
            ne->gather_dev(tok_, T, emb_, cs);
        } else {
            const auto* codes = (const uint8_t*) we->data;
            const auto* scales = (const float*) (codes + we->codes_bytes);
            const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
            embedding_gather_dev(codes, scales, offsets, tok_, T, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                 (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), emb_, cs);
        }
        native_qsa_rms_norm_weighted(emb_, f32("pre_fc_norm_embedding.weight"), en_, (int) N, T, EPS, cs);
        native_quantize_q8_1(en_, xq_, (int) N, T, cs);
        native_mmvq(wt_fc_embedding, wp_fc_embedding, xq_, e2_, (int) N, (int) N, T, cs);
        if (hnorm_stream_)   // --mtp-hnorm stream: one RMS per stream, each scaled by its slice of the weight
            native_qsa_rms_norm_grouped(Rin_, f32("pre_fc_norm_hidden.weight"), hn_, (int) N, (int) HC, (int) (T * HC), EPS, cs);
        else
            native_qsa_rms_norm_weighted(Rin_, f32("pre_fc_norm_hidden.weight"), hn_, (int) (HC * N), T, EPS, cs);
        for (int c0 = 0; c0 < T * HC; c0 += 8) {
            const int nc = (int) std::min<int64_t>(8, T * HC - c0);
            native_quantize_q8_1(hn_ + (size_t) c0 * N, xq_, (int) N, nc, cs);
            native_mmvq(wt_fc_hidden, wp_fc_hidden, xq_, h2_ + (size_t) c0 * N, (int) N, (int) N, nc, cs);
        }
        add_streams_broadcast(h2_, e2_, R_, N, (int) HC, T, cs);
        // ---- the attention hyper-connection
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = false;
                fa[t].w_norm = f32("attn_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("attn_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("attn_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("attn_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        // ---- attention: K/V into the layer's own cache
        native_quantize_q8_1(mixed_, xq_, (int) N, T, cs);
        native_mmvq(wt_k_proj, wp_k_proj, xq_, kcur_, (int) N, (int) (NKV * HD), T, cs);
        native_mmvq(wt_v_proj, wp_v_proj, xq_, vcur_, (int) N, (int) (NKV * HD), T, cs);
        // #783 PR-d (stuchapin909): the T tokens' K norm, K/V rotation and K/V append each run once over all T rows
        // (every row is independent; the rope keeps its per-token positions). STRATA_NO_BATCH_KV_STEP=1 appends per token.
        static const bool no_batch_kv = [] {
            const char* v = std::getenv("STRATA_NO_BATCH_KV_STEP");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        // #783 PR-f (stuchapin909): the per-head RMSNorm and the rope fused (bit-identical to the pair, rope_parity
        // check 6; STRATA_NO_NORM_ROPE=1 keeps the two; off on HIP until its parity check passes). K rows of one token
        // are consecutive in `pos`, tokens are NH apart, so a one-token window is the fusable case.
        const bool fuse_nr = native_rope_enabled() && native_norm_rope_usable((int) HD, (int) s.n_rot);
        if (T == 1 && fuse_nr) {
            native_qsa_rms_norm_rope(kcur_, (int) HD, f32("self_attn.k_norm.weight"), kcur_, (int) NKV, (int) HD,
                                     (int) s.n_rot, EPS, rope_scaling(), pos, cs);
        } else {
            native_qsa_rms_norm_weighted(kcur_, f32("self_attn.k_norm.weight"), kcur_, (int) HD, (int) (T * NKV), EPS, cs);
            for (int t = 0; t < T; ++t) {
                float* kc = kcur_ + t * NKV * HD;
                if (native_rope_enabled()) native_rope_apply(kc, kc, (int) NKV, (int) HD, (int) s.n_rot, rope_scaling(), pos + t * NH, cs);
                else rope_neox_apply(kc, kc, (int) NKV, (int) HD, (int) s.n_rot, st_.cos_tab, st_.sin_tab, pos + t * NH, cs);
            }
        }
        if (st_.kv_rot) {   // rotated K and V (kv_q4.hpp): Q4_0, and INT8 with STRATA_KV_ROT=1
            fwht256_inplace_cuda(kcur_, (int64_t) T * NKV, cs);
            fwht256_inplace_cuda(vcur_, (int64_t) T * NKV, cs);
        }
        // stored in the state's own format (#293 appended rotated INT8 K/V as Q4_0, into pools INT8 never has)
        if (!no_batch_kv && st_.kv_q4)
            kv_append_q4_steps(st_.k_q4, st_.v_q4, st_.page_table, step, 4, T, kcur_, vcur_, s, cs, &st_.host);
        else if (!no_batch_kv && st_.kv_int8)
            kv_append_q8_steps(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step, 4, kcur_, vcur_,
                               (int) (NKV * HD), T, s, cs, &st_.host);
        else
            for (int t = 0; t < T; ++t) {
                if (st_.kv_q4)
                    kv_append_q4_step(st_.k_q4, st_.v_q4, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                      vcur_ + t * NKV * HD, s, cs, &st_.host);
                else if (st_.kv_int8)
                    kv_append_q8_step(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step + t * 4,
                                      kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st_.host);
                else
                    kv_append_step(st_.k_pool, st_.v_pool, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                   vcur_ + t * NKV * HD, s, cs, &st_.host);
            }
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}

void MtpDrafter::norm_rope(float *data, const float *gamma, int rows, int cols,
                           const int32_t *p, dpct::queue_ptr cs) {
    using namespace strata::kernels;
    const QsaShapes s = shapes_of(*g_);
    native_qsa_rms_norm_weighted(data, gamma, data, cols, rows, EPS, cs);
    if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), p, cs);
    else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st_.cos_tab, st_.sin_tab, p, cs);
}

bool MtpDrafter::record_rest(int step_row, dpct::queue_ptr cs,
                             std::string &err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const int T = 1;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, NH = g.n_head, HD = g.head_dim;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const int32_t* step = step_ + step_row * 4;
    const int32_t* pos = pos_ + step_row * NH;
    // --mtp-q4: the q and o projections' format and weights (Q8_0 unless the Q4_0 copies exist)
    int wt_q_proj, wt_o_proj;
    const void* wp_q_proj = wq("self_attn.q_proj.weight", wt_q_proj);
    const void* wp_o_proj = wq("self_attn.o_proj.weight", wt_o_proj);
    const bool fuse_nr = native_rope_enabled() && native_norm_rope_usable((int) HD, (int) s.n_rot);   // #783 PR-f
    try {
        // ---- dense attention over every cell
        native_mmvq(wt_q_proj, wp_q_proj, xq_, qfull_, (int) N, (int) (NH * 2 * HD), T, cs);
        // all T tokens' q rows at once (the rows of token t sit at pos[t * NH ..], so row r reads pos[r])
        if (fuse_nr) {
            native_qsa_rms_norm_rope(qfull_, (int) (2 * HD), f32("self_attn.q_norm.weight"), qcur_, (int) (T * NH),
                                     (int) HD, (int) s.n_rot, EPS, rope_scaling(), pos, cs);
        } else {
            /*
            DPCT1124: cudaMemcpy2DAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            if (DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(
                    qcur_, (size_t)HD * 4, qfull_, (size_t)HD * 2 * 4,
                    (size_t)HD * 4, (size_t)(T * NH), dpct::device_to_device,
                    *cs)) != 0) {
                err = "mtp: q split failed";
                return false;
            }
            norm_rope(qcur_, f32("self_attn.q_norm.weight"), (int) (T * NH), (int) HD, pos, cs);
        }
        if (st_.kv_rot) fwht256_inplace_cuda(qcur_, (int64_t) T * NH, cs);
        const QsaAttnPools pools = qsa_attn_pools(st_);
        if (window_ > 0) window_ids(const_cast<int32_t*>(step), T, (int) window_, ident_, cap_, cs);
        qsa_decode_attn_batch(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn_, T, cs);
        if (st_.kv_rot) fwht256_inplace_cuda(attn_, (int64_t) T * NH, cs);
        native_qsa_gate_apply(attn_, qfull_, attn32_, (int) (T * NH), (int) HD, cs);
        native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), T, cs);
        native_mmvq(wt_o_proj, wp_o_proj, xq_, bo_, (int) (NH * HD), (int) N, T, cs);
        // ---- the MLP hyper-connection (the attention write folded in)
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = true;
                fa[t].bo_prev = bo_ + t * N; fa[t].inj_prev = inj_ + t * HC;
                fa[t].w_norm = f32("mlp_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("mlp_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("mlp_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("mlp_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj2_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        // ---- MoE: router, the 512 resident experts, the shared expert (on a branch beside them on CUDA), the
        // combine, the write
        NativeSharedWeights nsw;
        nsw.gate_data = wq("mlp.shared_expert.gate_proj.weight", nsw.gate_type);
        nsw.up_data = wq("mlp.shared_expert.up_proj.weight", nsw.up_type);
        nsw.down_data = wq("mlp.shared_expert.down_proj.weight", nsw.down_type);
        nsw.q8_1 = xq_;   // the routed experts read hit_xq_
        const SForm none{};
        const bool need_bf16_x = !shared_expert_native_bf16_enabled();
        static const bool fuse_head_gr = [] {
            const char* v = std::getenv("STRATA_FUSE_HEAD_GR");
            return v != nullptr && std::atoi(v) != 0;
        }();
        static const bool head_mix_multi_on = [] {
#if defined(STRATA_USE_HIP)
            return false;
#else
            const char* v = std::getenv("STRATA_HEAD_MIX_MULTI");
            return v == nullptr || std::atoi(v) != 0;
#endif
        }();
        // the shared expert only reads mixed_ / xq_ and writes shared_: a branch beside the router and the routed
        // experts, joined before the combine (CUDA; HIP keeps one stream: STRATA_MTP_SHARED_BRANCH=0 does too)
        static const bool branch_on = [] {
#if defined(STRATA_USE_HIP)
            return false;
#else
            const char* v = std::getenv("STRATA_MTP_SHARED_BRANCH");
            return v == nullptr || std::atoi(v) != 0;
#endif
        }();
        dpct::queue_ptr sh_cs = cs;
        if (branch_on) {
            /*
            DPCT1024: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            if (DPCT_CHECK_ERROR(dpct::sync_barrier(sh_fork_, cs)) != 0 ||
                DPCT_CHECK_ERROR(
                    (side_)->ext_oneapi_submit_barrier({*sh_fork_})) != 0) {
                err = "mtp: the shared expert's branch";
                return false;
            }
            sh_cs = side_;
        }
        auto shared_rows = [&]() {
            for (int t = 0; t < T; ++t) {
                if (need_bf16_x) f32_to_bf16_bulk(mixed_ + t * N, x_bf16_, N, sh_cs);
                shared_expert(nullptr, nullptr, x_bf16_, none, nullptr, nullptr, nullptr, none, nullptr, nullptr, nullptr,
                              none, nullptr, nullptr, nullptr, bf16("mlp.shared_expert_gate.weight"), sh_scratch_,
                              shared_ + t * N, N, g.n_ff, 32, sh_cs, mixed_ + t * N, &nsw);
            }
        };
        if (branch_on) {
            shared_rows();
            /*
            DPCT1024: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            if (DPCT_CHECK_ERROR(dpct::sync_barrier(sh_join_, side_)) != 0) {
              err = "mtp: the shared expert's branch"; return false;
            }
        }
        // #783 PR-c (stuchapin909): a multi-token window routes in two launches (one router GEMV reading the weight once,
        // one top-10 over all rows - each row bitwise the single-token call) and combines in one
        if (T > 1 && native_router_enabled() && g.n_expert == 512 && K == 10) {
            bf16_gemv_fp32_mmvf_multi(mixed_, N, bf16("mlp.gate.weight"), logits_, g.n_expert, (int) N, (int) g.n_expert, T, cs);
            native_router_top10_multi(logits_, ids_, w_, T, cs);
        } else {
        for (int t = 0; t < T; ++t) {
            bf16_gemv_fp32_mmvf(mixed_ + t * N, bf16("mlp.gate.weight"), logits_ + t * g.n_expert, (int) N, (int) g.n_expert, cs);
            if (native_router_enabled() && g.n_expert == 512 && K == 10) native_router_top10(logits_ + t * g.n_expert, ids_ + t * K, w_ + t * K, cs);
            else router_top10(logits_ + t * g.n_expert, 1, (int) g.n_expert, (int) K, ids_ + t * K, w_ + t * K, cs);
        }
        }
        moe_group_resident(ids_, (int) (T * K), (int) K, experts_, (int64_t) strata::kernels::cpu::BLOB, grp_ptr_,
                           grp_start_, grp_counts_, hit_dst_, hit_slot_, cs);
        quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, (int64_t) T * N, cs);
        moe_grouped_s2(grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, (int64_t) T * K, (int64_t) T * K, hit_xq_,
                       hit_xs_, hit_scratch_, parts_, cs);
        if (branch_on) {
            if (DPCT_CHECK_ERROR(cs->ext_oneapi_submit_barrier({*sh_join_})) !=
                0) {
              err = "mtp: the shared expert's join"; return false;
            }
        } else {
            shared_rows();
        }
        static const bool no_multi_gr = [] {   // #783 PR-g: STRATA_NO_MULTI_GR=1 keeps the per-token GR calls
            const char* v = std::getenv("STRATA_NO_MULTI_GR");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        for (int t = 0; t < T; ++t) {
            if (!(T > 1 && native_moe_combine_enabled())) {
                if (native_moe_combine_enabled())
                    native_moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
                else
                    moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
            }
            if (!fuse_head_gr && no_multi_gr)
                gr_write(R_ + (size_t) t * HC * N, y_ + t * N, inj2_ + t * HC, gs, R_ + (size_t) t * HC * N, cs);
        }
        if (T > 1 && native_moe_combine_enabled())
            native_moe_combine_multi(parts_, w_, shared_, y_, N, K, T, cs);
        // #783 PR-g (stuchapin909): the T residual writes in one launch (each token's R, block output and injection are
        // its own, so doing them after the loop changes nothing)
        if (!fuse_head_gr && !no_multi_gr)
            gr_write_multi(R_, y_, inj2_, gs, R_, T, cs);
        // ---- the final mixer and the main model's head
        if (fuse_head_gr) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = true;
                fa[t].bo_prev = y_ + t * N; fa[t].inj_prev = inj2_ + t * HC;
                fa[t].w_norm = f32("hyper_connection_mixer.hc_norm.weight");
                fa[t].w_down = bf16("hyper_connection_mixer.input_mix_weight_down.weight");
                fa[t].w_up = bf16("hyper_connection_mixer.input_mix_weight_up.weight");
                fa[t].w_inject = nullptr;
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = dummy_inj_; fa[t].mixed = sample_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        } else if (head_mix_multi_on) {
            // the window's rows in one read (CUDA; HIP keeps gr_read per row): bitwise the same sums
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = false;
                fa[t].w_norm = f32("hyper_connection_mixer.hc_norm.weight");
                fa[t].w_down = bf16("hyper_connection_mixer.input_mix_weight_down.weight");
                fa[t].w_up = bf16("hyper_connection_mixer.input_mix_weight_up.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC; fa[t].mixed = sample_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        } else {
            for (int t = 0; t < T; ++t)
                gr_read(R_ + (size_t) t * HC * N, f32("hyper_connection_mixer.hc_norm.weight"),
                        bf16("hyper_connection_mixer.input_mix_weight_down.weight"),
                        bf16("hyper_connection_mixer.input_mix_weight_up.weight"), nullptr, EPS, gs, ss.block.gr,
                        sample_ + t * N, dummy_inj_, cs);
        }
        native_quantize_q8_1(sample_, xq_, (int) N, T, cs);
        const bool sub = dhead_ != nullptr;
        const int64_t nv = sub ? n_dvocab_ : n_vocab_;
        native_mmvq(sub ? dhead_type_ : head_->type(), sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N,
                    (int) nv, T, cs);
        if (coupled_rec_) {
            // coupled draft sampling: the target's chain and Philox draw for the row that will verify this draft
            // (counter = this cell + 1, from its step record), penalties over the ring; T == 1 (full layer)
            coupled_draft_sample(head_logits_, (int) nv, sub ? dvocab_ : nullptr, sub ? dinv_ : nullptr, (int) n_vocab_,
                                 cparams_, cring_, kCoupledHistCap, coupled_j_, step, cscratch_, out_ids_, probs_, cs);
            return true;
        }
        if (argmax_rows_wanted()) {
            argmax_rows(head_logits_, T, (int) nv, arg_scratch_, out_ids_, cs);
        } else {
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            sample_tokens(head_logits_, T, (int) nv, nullptr, 0, sp, out_ids_, cs);
        }
        if (multi_block_head_ops()) row_top_prob_split(head_logits_, T, (int) nv, out_ids_, probs_, top_scratch_, cs);
        else row_top_prob(head_logits_, T, (int) nv, out_ids_, probs_, cs);
        if (sub) map_ids(out_ids_, dvocab_, T, cs);
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}

namespace {
bool finish_capture(dpct::queue_ptr cs, bool ok,
                    dpct::experimental::command_graph_exec_ptr &exec,
                    const char *what, std::string &err) try {
    dpct::experimental::command_graph_ptr graph = nullptr;
    const dpct::err0 ce =
        DPCT_CHECK_ERROR(dpct::experimental::end_recording(cs, &graph));
    if (!ok) {
        if (graph) delete (graph);
        return false;
    }
    if (ce != 0 ||
        DPCT_CHECK_ERROR(
            exec = new sycl::ext::oneapi::experimental::command_graph<
                sycl::ext::oneapi::experimental::graph_state::executable>(
                graph->finalize())) != 0) {
        if (graph) delete (graph);
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        err = std::string("mtp: ") + what +
              " capture: " + dpct::get_error_string_dummy(ce);
        return false;
    }
    delete (graph);
    // an explicit upload: the first launch's implicit one blocked behind a device-side spin (verify.cpp)
    /*
    DPCT1007: Migration of cudaGraphUpload is not supported.
    */
    // no cudaGraphUpload on SYCL: a finalized command_graph is already resident
    cs->wait();
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
}  // namespace

bool MtpDrafter::capture_prefill(int T, std::string &err) try {
    if (prefill_exec_[T]) return true;
    using namespace strata::kernels;
    if (DPCT_CHECK_ERROR(dpct::experimental::begin_recording(cs_)) != 0) {
        err = "mtp: begin capture"; return false;
    }
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) T * g_->n_head, cs_);
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T)
    return finish_capture(cs_, ok, prefill_exec_[T], "prefill", err);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool MtpDrafter::capture_prefill_dev(int T, std::string &err) try {
    if (prefill_dev_exec_[T]) return true;
    if (DPCT_CHECK_ERROR(dpct::experimental::begin_recording(cs_)) != 0) {
        err = "mtp: begin capture"; return false;
    }
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T); tok_/step_/pos_ filled before launch
    return finish_capture(cs_, ok, prefill_dev_exec_[T], "prefill (device inputs)", err);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool MtpDrafter::capture_round(int T, bool coupled, std::string &err) try {
    dpct::experimental::command_graph_exec_ptr& exec = coupled ? round_exec_c_[T] : round_exec_[T];
    if (exec) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    if (DPCT_CHECK_ERROR(dpct::experimental::begin_recording(cs_)) != 0) {
        err = "mtp: begin capture"; return false;
    }
    bool ok = true;
    // coupled: the request's chain and the penalty history's base, for this round's drafts
    if (coupled) coupled_draft_stage(m_cparams_, m_chist_, cparams_, cring_, kCoupledHistCap, cs_);
    // the catch-up: the layer's front for the window's T cells (their K/V), then its rest for row a only, on row
    // a's intermediates copied to row 0, at the cell the host staged in step row 2*max_t - 1; the draft chain is
    // one graph per step (`capture_step`) so the host can stop it when a draft is unlikely
    const int ra = 2 * max_t_ - 1;
    const int64_t N = g_->n_embd, NH = g_->n_head;
#if defined(STRATA_USE_HIP)   // AMD keeps its separate copies
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) 2 * T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) 2 * T * NH, cs_);
    copy_i32_from_mapped(row_, m_row_, 2, cs_);
    copy_from_mapped(Rin_, window_R_, (int64_t) T * HCN, cs_);
    copy_i32_from_mapped(step_ + ra * 4, m_step_ + ra * 4, 4, cs_);
    copy_i32_from_mapped(pos_ + ra * NH, m_pos_ + ra * NH, NH, cs_);
#else   // the draft round's seven inputs in one launch
    const MappedCopy in[7] = {{tok_, m_tok_, T}, {step_, m_step_, (int64_t) 2 * T * 4}, {pos_, m_pos_, 2 * T * NH},
                              {row_, m_row_, 2}, {Rin_, window_R_, T * HCN}, {step_ + ra * 4, m_step_ + ra * 4, 4},
                              {pos_ + ra * NH, m_pos_ + ra * NH, NH}};
    copy_from_mapped_multi(in, 7, cs_);
#endif
    ok = record_front(T, 0, cs_, err);
    if (ok) {
        if (T > 1) {
            copy_row_to_first(row_, R_, HCN, inj_, g_->hc, mixed_, N, cs_);
            native_quantize_q8_1(mixed_, xq_, (int) N, 1, cs_);
        }
        coupled_rec_ = coupled;
        coupled_j_ = 0;
        ok = record_rest(ra, cs_, err);
        coupled_rec_ = false;
    }
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, 0, cs_, probs_, m_prob_);
    if (ok && force_on_ && !coupled) force_token(tok_, m_force_, 0, cs_);   // chain_launch: step 1's input
    return finish_capture(cs_, ok, exec, coupled ? "round (coupled)" : "round", err);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Chain step j (1..max_t-2): one row at the cell staged in step row `max_t + j - 1`, from the previous step's
// residual and token (left in Rin_[0] / tok_[0] by mtp_select); draft j and its probability to the mapped outputs.
bool MtpDrafter::capture_step(int j, bool coupled, std::string &err) try {
    dpct::experimental::command_graph_exec_ptr& exec = coupled ? step_exec_c_[j] : step_exec_[j];
    if (exec) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    const int row = max_t_ + j - 1;
    if (DPCT_CHECK_ERROR(dpct::experimental::begin_recording(cs_)) != 0) {
        err = "mtp: begin capture"; return false;
    }
#if defined(STRATA_USE_HIP)
    copy_i32_from_mapped(step_ + row * 4, m_step_ + row * 4, 4, cs_);
    copy_i32_from_mapped(pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head, cs_);
#else
    const MappedCopy in[2] = {{step_ + row * 4, m_step_ + row * 4, 4},
                              {pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head}};
    copy_from_mapped_multi(in, 2, cs_);
#endif
    coupled_rec_ = coupled;
    coupled_j_ = j;
    bool ok = record_forward(1, row, cs_, err);
    coupled_rec_ = false;
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, j, cs_, probs_, m_prob_);
    if (ok && force_on_ && !coupled) force_token(tok_, m_force_, j, cs_);   // chain_launch: step j+1's input
    return finish_capture(cs_, ok, exec, coupled ? "step (coupled)" : "step", err);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void MtpDrafter::kv_restore(int64_t upto) {
    const OnDevice on_device(device_);
    if (st_.kv_mode != 2 || upto <= 0) return;
    // the ring's blocks below `upto`, from the host copy: a checkpoint resume may have left later cells in them
    const strata::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t b1 = (upto + s.page_size - 1) / s.page_size, b0 = std::max<int64_t>(0, b1 - st_.n_slots);
    strata::kernels::kv_ring_restore(qsa_attn_pools(st_), st_.host, qsa_kv_format(st_), b0, b1, st_.n_slots, s, cs_);
    cs_->wait();
}

bool MtpDrafter::prepare_prefill(std::string &err) try {
    const OnDevice on_device(device_);
    const int64_t per_row = 1 + 4 + g_->n_head;
    if (pf_cap_ < (int64_t) max_t_ * per_row) {
        if (pf_dev_) sycl::free(pf_dev_, dpct::get_in_order_queue());
        pf_dev_ = nullptr;
        pf_cap_ = 0;
        if (DPCT_CHECK_ERROR(pf_dev_ = sycl::malloc_device<int32_t>(
                                 (size_t)(max_t_ * per_row),
                                 dpct::get_in_order_queue())) != 0) {
            err = "mtp prefill: the input records do not fit";
            return false;
        }
        pf_cap_ = max_t_ * per_row;
    }
    for (int T = 1; T <= max_t_; ++T)
        if (!capture_prefill_dev(T, err)) return false;   // (STRATA_MTP_PREFILL_SYNC's loop syncs anyway)
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool MtpDrafter::prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err,
                         bool sync) {
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    const int64_t HCN = g_->hc * g_->n_embd;
    // cells the window can never reach again need no K/V
    const int64_t first_needed = (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0;
    // E-4: every group's token / step / position records uploaded at once; each group is then device copies and a
    // graph on the one stream, with a single sync at the end (a group of <= max_t rows used to be staged in mapped
    // memory and synced before the next: ~5,500 host round trips on a 32K prompt).  The same work in the same
    // order.  STRATA_MTP_PREFILL_SYNC=1 keeps the old loop, =0 forces E-4.
    // HIP defaults to the old loop.  This pass runs whenever E-9 (Prefill::draft_kv) declines, which it does for the
    // drafter's ring (KV streaming, --kv-resident), and on gfx1201 / ROCm 7.2.4 the E-4 queue (a graph launch and four
    // device copies per group, ~2,000 groups per 8192-token chunk, no sync) sometimes never completes: the prompt hangs
    // in hipGraphLaunch or the final sync until the watchdog ends the engine.  R9700, full IQ3_XXS, --kv-resident
    // 32768, mixed load (chats, 32K and 7K prompts, deep follow-ups): E-4 hung in the first round on 2 of 2 tries,
    // the old loop ran 30 of 30 rounds clean, prompt speed unchanged.
    static const bool per_group_sync = [] {
        if (const char* v = std::getenv("STRATA_MTP_PREFILL_SYNC")) return std::atoi(v) != 0;
#if defined(STRATA_USE_HIP)
        return true;
#else
        return false;
#endif
    }();
    const int64_t NHp = g_->n_head, per_row = 1 + 4 + NHp;
    if (!per_group_sync && n > 0) {
        if (pf_cap_ < n * per_row) {
            if (pf_dev_) sycl::free(pf_dev_, dpct::get_in_order_queue());
            pf_dev_ = nullptr;
            pf_cap_ = 0;
            if (DPCT_CHECK_ERROR(pf_dev_ = sycl::malloc_device<int32_t>(
                                     (size_t)(n * per_row),
                                     dpct::get_in_order_queue())) != 0) {
                err = "mtp prefill: the input records do not fit";
                return false;
            }
            pf_cap_ = n * per_row;
        }
        std::vector<int32_t> rec((size_t) (n * per_row));
        int32_t* tk = rec.data();
        int32_t* stp = tk + n;
        int32_t* ps = stp + 4 * n;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t cell = cell0 + i;
            tk[i] = next_tokens[i];
            stp[i * 4 + 0] = (int32_t) cell;
            stp[i * 4 + 1] = (int32_t) (cell + 1);
            stp[i * 4 + 2] = (int32_t) ((cell + 1) / 4);
            stp[i * 4 + 3] = (int32_t) (cell + 1);
            for (int64_t h = 0; h < NHp; ++h) ps[i * NHp + h] = (int32_t) cell;
        }
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        if (DPCT_CHECK_ERROR(cs_->memcpy(pf_dev_, rec.data(),
                                         rec.size() * sizeof(int32_t))) != 0) {
            err = "mtp prefill: the input records upload failed";
            return false;
        }
        const int32_t* d_tk = pf_dev_;
        const int32_t* d_stp = d_tk + n;
        const int32_t* d_ps = d_stp + 4 * n;
        for (int64_t c = 0; c < n; c += max_t_) {
            const int T = (int) std::min<int64_t>(max_t_, n - c);
            if (cell0 + c + T <= first_needed) continue;
            if (!capture_prefill_dev(T, err)) return false;
            /*
            DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            if (DPCT_CHECK_ERROR(cs_->memcpy(tok_, d_tk + c, (size_t)T * 4)) !=
                    0 ||
                /*
                DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy
                API. While the origin API might be synchronous, it depends on
                the type of operand memory, so you may need to call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                DPCT_CHECK_ERROR(
                    cs_->memcpy(step_, d_stp + c * 4, (size_t)T * 16)) != 0 ||
                /*
                DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy
                API. While the origin API might be synchronous, it depends on
                the type of operand memory, so you may need to call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                DPCT_CHECK_ERROR(cs_->memcpy(pos_, d_ps + c * NHp,
                                             (size_t)(T * NHp) * 4)) != 0 ||
                /*
                DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy
                API. While the origin API might be synchronous, it depends on
                the type of operand memory, so you may need to call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                DPCT_CHECK_ERROR(
                    cs_->memcpy(Rin_, R_rows + (size_t)c * HCN,
                                (size_t)T * HCN * sizeof(float))) != 0 ||
                DPCT_CHECK_ERROR(
                    (cs_)->ext_oneapi_graph(*prefill_dev_exec_[T])) != 0) {
                /*
                DPCT1009: SYCL reports errors using exceptions and does not
                use error codes. Please replace the
                "get_error_string_dummy(...)" with a real error-handling
                function.
                */
                /*
                DPCT1010: SYCL uses exceptions to report errors and does not
                use the error codes. The cudaGetLastError function call was
                replaced with 0. You need to rewrite this code.
                */
                err = std::string("mtp prefill: ") +
                      dpct::get_error_string_dummy(0);
                return false;
            }
        }
        if (!sync) {   // the caller orders on stream(); the records are staged (a pageable copy returns once taken)
            (void)DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty()));
            ms_prefill += ms_since(t0);
            return true;
        }
        if (DPCT_CHECK_ERROR(cs_->wait()) != 0) {
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not use
            the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            err =
                std::string("mtp prefill: ") + dpct::get_error_string_dummy(0);
            return false;
        }
        ms_prefill += ms_since(t0);
        return true;
    }
    for (int64_t c = 0; c < n; c += max_t_) {
        const int T = (int) std::min<int64_t>(max_t_, n - c);
        if (cell0 + c + T <= first_needed) continue;
        if (!capture_prefill(T, err)) return false;
        for (int t = 0; t < T; ++t) {
            const int64_t cell = cell0 + c + t;
            h_tok_[t] = next_tokens[c + t];
            h_step_[t * 4 + 0] = (int32_t) cell;
            h_step_[t * 4 + 1] = (int32_t) (cell + 1);
            h_step_[t * 4 + 2] = (int32_t) ((cell + 1) / 4);
            h_step_[t * 4 + 3] = (int32_t) (cell + 1);
            for (int64_t h = 0; h < g_->n_head; ++h) h_pos_[t * g_->n_head + h] = (int32_t) cell;
        }
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        if (DPCT_CHECK_ERROR(cs_->memcpy(Rin_, R_rows + (size_t)c * HCN,
                                         (size_t)T * HCN * sizeof(float))) !=
                0 ||
            DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(*prefill_exec_[T])) != 0 ||
            DPCT_CHECK_ERROR(cs_->wait()) != 0) {
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not use
            the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            err =
                std::string("mtp prefill: ") + dpct::get_error_string_dummy(0);
            return false;
        }
    }
    ms_prefill += ms_since(t0);
    return true;
}

bool MtpDrafter::draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
                       float* probs, float min_p, int* n_drafts) {
    const OnDevice on_device(device_);
    if (T < 1 || T > max_t_ || a < 0 || a >= T) { err = "mtp: draft arguments out of range"; return false; }
    if (chain_live_) { err = "mtp: a pipelined chain is still in flight"; return false; }
    // #783 PR-i (stuchapin909): the round runs for the cells up to the accepted row a (T = a + 1): the K/V of the rejected
    // rows is not caught up - no later read reaches a cell past the one being drafted and the next round writes it first
    if (!mtp_catchup_all()) T = a + 1;
    const bool cp = coupled_active_;   // coupled draft sampling for this request: its own graphs
    if (!capture_round(T, cp, err)) return false;
    const int max_steps = std::min(max_t_ - 1, max_drafts_);
    for (int j = 1; j < max_steps; ++j)
        if (!capture_step(j, cp, err)) return false;
    const Clock::time_point t0 = Clock::now();
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {
        h_step_[row * 4 + 0] = (int32_t) cell;
        h_step_[row * 4 + 1] = (int32_t) (cell + 1);
        h_step_[row * 4 + 2] = (int32_t) ((cell + 1) / 4);
        h_step_[row * 4 + 3] = (int32_t) (cell + 1);
        for (int64_t h = 0; h < NH; ++h) h_pos_[row * NH + h] = (int32_t) cell;
    };
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        put(t, p + t);
    }
    put(2 * max_t_ - 1, coupled_draft_cell(p, a, 0));   // p + a: draft 0's cell
    for (int j = 1; j < max_steps; ++j)
        put(max_t_ + j - 1, coupled_draft_cell(p, a, j));
    for (int j = 0; j < max_steps; ++j) ((volatile int32_t*) h_out_)[j] = -1;
    h_row_[0] = a;
    h_row_[1] = 0;
    if (h_force_ != nullptr) for (int j = 0; j < 16; ++j) h_force_[j] = -1;   // the forcing kernels: no-ops here
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!stage_source_R(T, err)) return false;

    SessionState* const pss = ple_ss_ ? ple_ss_ : ss_;
    const bool do_ple = pss != nullptr && pss->ple.ready() && pss->ple.table != nullptr;
    int32_t ple_prev[2] = {do_ple ? pss->ple_prev[0] : 0, do_ple ? pss->ple_prev[1] : 0};
    auto prefetch_ple = [&](int32_t tok) {
        if (!do_ple || tok < 0) return;
        uint32_t rows16[strata::kernels::PLE_N_HEADS];
        strata::kernels::ngram_rows(&tok, ple_prev, 1, pss->ple.consts, rows16);
        ple_prev[0] = ple_prev[1];
        ple_prev[1] = tok;
        pss->ple.table->prefetch_rows(rows16);
    };

    int n = 0;
    if (min_p <= 0.0f && max_steps > 0) {
        if (DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(
                *(cp ? round_exec_c_[T] : round_exec_[T]))) != 0) {
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not use
            the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            err = std::string("mtp draft: ") + dpct::get_error_string_dummy(0);
            return false;
        }
        for (int j = 1; j < max_steps; ++j) {
            if (DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(
                    *(cp ? step_exec_c_[j] : step_exec_[j]))) != 0) {
                /*
                DPCT1009: SYCL reports errors using exceptions and does not
                use error codes. Please replace the
                "get_error_string_dummy(...)" with a real error-handling
                function.
                */
                /*
                DPCT1010: SYCL uses exceptions to report errors and does not
                use the error codes. The cudaGetLastError function call was
                replaced with 0. You need to rewrite this code.
                */
                err = std::string("mtp draft step: ") + dpct::get_error_string_dummy(0);
                return false;
            }
        }
        (void)DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty()));
        prefetch_ple(tokens[a]);
        for (int j = 0; j + 1 < max_steps; ++j) {
            uint32_t spins = 0;
            while (((volatile int32_t*) h_out_)[j] < 0) {
#if defined(_WIN32) || defined(__x86_64__)
                _mm_pause();
#endif
                if ((++spins & 1023u) == 0 &&
                    ((cs_)->ext_oneapi_empty() ? 0 : 1) != 1) break;   /* SYCL port: DPCT_CHECK_ERROR of ext_oneapi_empty() was always 0, so the wait gave up after 1024 spins and read a draft the GPU had not written */
            }
            drafts[j] = ((volatile int32_t*) h_out_)[j];
            if (probs) probs[j] = ((volatile float*) h_prob_)[j];
            prefetch_ple(drafts[j]);
        }
        if (DPCT_CHECK_ERROR(cs_->wait()) != 0) {
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not use
            the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            err = std::string("mtp draft: ") + dpct::get_error_string_dummy(0);
            return false;
        }
        const int last = max_steps - 1;
        drafts[last] = ((volatile int32_t*) h_out_)[last];
        if (probs) probs[last] = ((volatile float*) h_prob_)[last];
        prefetch_ple(drafts[last]);
        n = max_steps;
    } else if (max_steps > 0) {
        if (DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(
                *(cp ? round_exec_c_[T] : round_exec_[T]))) != 0) {
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not use
            the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            err = std::string("mtp draft: ") + dpct::get_error_string_dummy(0);
            return false;
        }
        (void)DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty()));
        prefetch_ple(tokens[a]);
        auto wait_step = [&](int j) {
            try {
        uint32_t spins = 0;
            while (((volatile int32_t*) h_out_)[j] < 0) {
#if defined(_WIN32) || defined(__x86_64__)
                _mm_pause();
#endif
                if ((++spins & 1023u) == 0 &&
                    ((cs_)->ext_oneapi_empty() ? 0 : 1) != 1) break;   /* SYCL port: DPCT_CHECK_ERROR of ext_oneapi_empty() was always 0, so the wait gave up after 1024 spins and read a draft the GPU had not written */
            }
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        wait_step(0);
        drafts[0] = ((volatile int32_t*) h_out_)[0];
        float pj = ((volatile float*) h_prob_)[0];
        if (probs) probs[0] = pj;
        n = 1;
        for (int j = 1; j < max_steps && pj >= min_p; ++j) {
            if (DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(
                    *(cp ? step_exec_c_[j] : step_exec_[j]))) != 0) {
                /*
                DPCT1009: SYCL reports errors using exceptions and does not
                use error codes. Please replace the
                "get_error_string_dummy(...)" with a real error-handling
                function.
                */
                /*
                DPCT1010: SYCL uses exceptions to report errors and does not
                use the error codes. The cudaGetLastError function call was
                replaced with 0. You need to rewrite this code.
                */
                err = std::string("mtp draft step: ") + dpct::get_error_string_dummy(0);
                return false;
            }
            (void)DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty()));
            prefetch_ple(drafts[j - 1]);
            wait_step(j);
            drafts[j] = ((volatile int32_t*) h_out_)[j];
            pj = ((volatile float*) h_prob_)[j];
            if (probs) probs[j] = pj;
            ++n;
        }
        prefetch_ple(drafts[n - 1]);
        if (DPCT_CHECK_ERROR(cs_->wait()) != 0) {
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not use
            the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            err = std::string("mtp draft: ") + dpct::get_error_string_dummy(0);
            return false;
        }
    }
    for (int j = n; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }
    if (n_drafts) *n_drafts = n;
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

bool MtpDrafter::stage_source_R(int T, std::string &err) try {
    if (src_R_ == nullptr || src_R_ == window_R_) return true;
    /*
    DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    if (DPCT_CHECK_ERROR(cs_->memcpy((void *)window_R_, src_R_,
                                     (size_t)T * (size_t)(g_->hc * g_->n_embd) *
                                         sizeof(float))) != 0) {
        err = "mtp: staging the window's residual rows failed";
        return false;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool MtpDrafter::prepare_chain(std::string &err) try {
    const OnDevice on_device(device_);
    if (!force_on_ || h_force_ == nullptr) { err = "mtp: chains need set_force_capture before load"; return false; }
    for (int T = 1; T <= max_t_; ++T)
        if (!capture_round(T, false, err)) return false;
    for (int j = 1; j <= max_t_ - 2; ++j)
        if (!capture_step(j, false, err)) return false;
    if (ev_chain_ == nullptr &&
        DPCT_CHECK_ERROR(ev_chain_ = new sycl::event()) != 0) {
        err = "mtp: event creation failed";
        return false;
    }
    for (dpct::event_ptr &e : ev_step_)
        if (e == nullptr && DPCT_CHECK_ERROR(e = new sycl::event()) != 0) {
            err = "mtp: event creation failed";
            return false;
        }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool MtpDrafter::chain_launch(int T, const int32_t *tokens, int64_t p, int a,
                              const int32_t *force, int n_force, int n_out,
                              int n_early, std::string &err) try {
    const OnDevice on_device(device_);
    if (chain_live_) { err = "mtp: a chain is already in flight"; return false; }
    if (!force_on_ || ev_chain_ == nullptr) { err = "mtp: chains need prepare_chain"; return false; }
    if (T < 1 || T > max_t_ || a < 0 || a >= T || n_out < 1 || n_out > max_t_ - 1 || n_force < 0 || n_force > n_out ||
        n_force > 16) {
        err = "mtp: chain arguments out of range";
        return false;
    }
    if (round_exec_[T] == nullptr) { err = "mtp: chain graphs not prepared"; return false; }
    for (int j = 1; j < n_out; ++j)
        if (step_exec_[j] == nullptr) { err = "mtp: chain graphs not prepared"; return false; }
    const Clock::time_point t0 = Clock::now();
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {
        h_step_[row * 4 + 0] = (int32_t) cell;
        h_step_[row * 4 + 1] = (int32_t) (cell + 1);
        h_step_[row * 4 + 2] = (int32_t) ((cell + 1) / 4);
        h_step_[row * 4 + 3] = (int32_t) (cell + 1);
        for (int64_t h = 0; h < NH; ++h) h_pos_[row * NH + h] = (int32_t) cell;
    };
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        put(t, p + t);
    }
    put(2 * max_t_ - 1, p + a);                                      // output 0's cell
    for (int j = 1; j < n_out; ++j) put(max_t_ + j - 1, p + a + j);  // every step's cell, staged at once
    for (int j = 0; j < 16; ++j) h_force_[j] = j < n_force ? force[j] : -1;
    h_row_[0] = a;
    h_row_[1] = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!stage_source_R(T, err)) return false;
    chain_early_ = std::max(0, std::min(n_early, n_out));
    bool ok = DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(*round_exec_[T])) == 0;
    /*
    DPCT1024: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    if (ok && chain_early_ >= 1) ok =
        DPCT_CHECK_ERROR(dpct::sync_barrier(ev_step_[0], cs_)) == 0;
    for (int j = 1; ok && j < n_out; ++j) {
        ok = DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(*step_exec_[j])) == 0;
        /*
        DPCT1024: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        if (ok && j < chain_early_) ok =
            DPCT_CHECK_ERROR(dpct::sync_barrier(ev_step_[j], cs_)) == 0;
    }
    /*
    DPCT1024: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    if (!ok || DPCT_CHECK_ERROR(dpct::sync_barrier(ev_chain_, cs_)) != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        err = std::string("mtp chain: ") + dpct::get_error_string_dummy(0);
        return false;
    }
    (void)DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty())); // WDDM: submit now
    steps_seen_ = 0;
    chain_live_ = true;
    chain_n_ = n_out;
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int MtpDrafter::chain_outputs_ready(std::string &err) try {
    if (!chain_live_) return chain_n_;
    const OnDevice on_device(device_);
    while (steps_seen_ < chain_early_) {
        const dpct::err0 q = dpct::sycl_event_query(ev_step_[steps_seen_]);
        /*
        DPCT1001: The statement could not be removed.
        */
        /*
        DPCT1002: Special case error handling if-stmt was detected. You may
        need to rewrite this code.
        */
        if (q == 1) break;
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        /*
        DPCT1000: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (q != 0) {
          err = std::string("mtp chain: ") + dpct::get_error_string_dummy(q);
          return -1;
        }
        chain_tok_[steps_seen_] = ((volatile int32_t*) h_out_)[steps_seen_];
        chain_prob_[steps_seen_] = ((volatile float*) h_prob_)[steps_seen_];
        ++steps_seen_;
    }
    return steps_seen_;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int MtpDrafter::chain_poll(std::string &err) try {
    if (!chain_live_) return 1;
    const OnDevice on_device(device_);
    const dpct::err0 q = dpct::sycl_event_query(ev_chain_);
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1002: Special case error handling if-stmt was detected. You may need
    to rewrite this code.
    */
    if (q == 1) return 0;
    chain_live_ = false;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (q != 0) {
      err = std::string("mtp chain: ") + dpct::get_error_string_dummy(q);
      return -1;
    }
    for (int j = 0; j < chain_n_; ++j) {
        chain_tok_[j] = ((volatile int32_t*) h_out_)[j];
        chain_prob_[j] = ((volatile float*) h_prob_)[j];
    }
    for (int j = chain_n_; j < 8; ++j) { chain_tok_[j] = 0; chain_prob_[j] = 0.0f; }
    return 1;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool MtpDrafter::draft_first(int T, const float *R_row, int32_t token,
                             int64_t cell, int32_t *drafts, std::string &err,
                             float *probs, float min_p, int *n_drafts) try {
    const OnDevice on_device(device_);
    // row 0 is the real pair; rows 1.. repeat it and only write cells the next round overwrites
    const int64_t HCN = g_->hc * g_->n_embd;
    for (int t = 0; t < T; ++t)
        if (DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                (void *)(window_R_ + (size_t)t * HCN), R_row,
                (size_t)HCN * sizeof(float)).wait()) != 0) {
            err = "mtp: staging the first residual failed";
            return false;
        }
    std::vector<int32_t> toks((size_t) T, token);
    return draft(T, toks.data(), cell, 0, drafts, err, probs, min_p, n_drafts);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::core
