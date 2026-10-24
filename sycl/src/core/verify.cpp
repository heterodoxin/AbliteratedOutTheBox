// src/core/verify.cpp - see include/strata/core/verify.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/dma_batch.hpp"
#if defined(_WIN32)
#include <intrin.h>
#endif

#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <immintrin.h>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
const bool g_dbg = std::getenv("STRATA_VERIFY_DEBUG") != nullptr;
inline bool g_lfuse() { static const bool on = [] { const char* v = std::getenv("STRATA_LFUSE"); return v != nullptr && v[0] == '1'; }(); return on; }
inline bool g_lfuse_gate() { static const bool on = [] { const char* v = std::getenv("STRATA_LFUSE_GATE"); return v == nullptr || v[0] != '0'; }(); return on; }
inline bool g_lfuse_pair() { static const bool on = [] { const char* v = std::getenv("STRATA_LFUSE_PAIR"); return v == nullptr || v[0] != '0'; }(); return on; }
inline bool g_qdedup() { static const bool on = [] { const char* v = std::getenv("STRATA_VERIFY_QDEDUP"); return v != nullptr && std::atoi(v) != 0; }(); return on; }
// S26 STRATA_QFUSE=1: activation q8_1 images written by their producers (the GDN output norm) - the same bytes
inline bool g_qfuse() { static const bool on = [] { const char* v = std::getenv("STRATA_QFUSE"); return v != nullptr && std::atoi(v) != 0; }(); return on; }
#define VDBG(...) do { if (g_dbg) { std::fprintf(stderr, "verify dbg: " __VA_ARGS__); std::fflush(stderr); } } while (0)

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

// #649 (HIP, opt-in A/Bs for the gfx1030 verify timeouts; CUDA never reads them)
//   STRATA_VERIFY_COHERENT=1  the handshake words and rows in explicitly coherent (fine-grained) host memory
//   STRATA_DOORBELL_STORE=1   the GPU stores each step's ring instead of read-modify-writing it over PCIe
bool env_on(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && e[0] != 0 && e[0] != '0';
}
#if defined(STRATA_USE_HIP)
const bool g_coherent = env_on("STRATA_VERIFY_COHERENT");
const bool g_doorbell_store = env_on("STRATA_DOORBELL_STORE");
#endif
const bool g_trace = env_on("STRATA_VERIFY_TRACE");
// STRATA_VERIFY_STEPPED=1 (opt-in, Arc A-series): see Verifier::step_hook_.  STRATA_VERIFY_PIPE=0 serves without the
// helper thread (serial segments) for A/B runs.
const bool g_stepped = env_on("STRATA_VERIFY_STEPPED");

// STRATA_PIPE_HELPER_SIBLING=1 (opt-in, stepped mode): the host thread is pinned to one logical CPU and a std::thread
// inherits that mask, so the pipelined serve's pool helper and the main thread's flag-A wait share one CPU and the
// early segment is mostly submitted only after the pool finished.  With this set the helper runs on the host CPU's SMT
// sibling instead.  -1: keep the inherited mask (unset, not pinned to one CPU, or no sibling).
int pipe_helper_cpu() {
#if defined(__linux__)
    static const int cpu = [] {
        if (!env_on("STRATA_PIPE_HELPER_SIBLING")) return -1;
        cpu_set_t s;
        CPU_ZERO(&s);
        if (pthread_getaffinity_np(pthread_self(), sizeof s, &s) != 0 || CPU_COUNT(&s) != 1) {
            std::fprintf(stderr, "strata verify: STRATA_PIPE_HELPER_SIBLING: host thread is not pinned to one CPU; unchanged\n");
            return -1;
        }
        int host = -1;
        for (int c = 0; c < CPU_SETSIZE && host < 0; ++c)
            if (CPU_ISSET(c, &s)) host = c;
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", host);
        int sib = -1;
        if (FILE* f = std::fopen(path, "r")) {
            char buf[128] = {};
            if (std::fgets(buf, sizeof buf, f) != nullptr) {
                for (char* p = buf; *p != 0 && sib < 0;) {   // "0,8" or "0-1"
                    char* end = nullptr;
                    const long a = std::strtol(p, &end, 10);
                    if (end == p) break;
                    long b = a;
                    if (*end == '-') { p = end + 1; b = std::strtol(p, &end, 10); }
                    for (long c = a; c <= b && sib < 0; ++c)
                        if (c != host) sib = (int) c;
                    p = (*end == ',') ? end + 1 : end;
                    if (*end != ',') break;
                }
            }
            std::fclose(f);
        }
        std::fprintf(stderr, "strata verify: pipelined pool helper on CPU %d (host thread CPU %d)%s\n", sib, host,
                     sib < 0 ? ": no SMT sibling, unchanged" : "");
        return sib;
    }();
    return cpu;
#else
    return -1;
#endif
}
const bool g_no_multi_gr = env_on("STRATA_NO_MULTI_GR");   // #783 PR-g: per-token GR reads/writes and generic-T kernels again
const bool g_no_batch_kv = env_on("STRATA_NO_BATCH_KV_STEP");   // #783 PR-d: per-token K/V and indexer appends again

// The shared expert runs on its own stream, forked off and joined back per layer (`sh_fork` in record_window).  The
// overlap pays off on CUDA, where it was added (cfd3b72).  On HIP a cross-stream event dependency costs more than the
// shared expert takes to run, so there the fork starts off: same binary, same 48,067-token prompt, three runs each on a
// Radeon AI PRO R9700 (gfx1201, ROCm 6.4.3), the fork off decodes at 62.9 t/s against 43.4 t/s with it on.  The
// profiler turns the fork off to time the stages, so a profiled run never showed that.  #816 reports the same fall on a
// gfx1030 (RX 6800, Windows, 42.5 -> 56.4 t/s).  STRATA_SH_STREAM overrides either default: `=1` forks, `=0` does not.
// Strix Halo (gfx1151, unified memory) is the exception: the fork pays there (+1.8% UD-IQ4_XS, +6.7% UD-Q4_K_XL decode
// at 8K, ids identical), so the gfx1151 arch table sets STRATA_SH_STREAM=1.  Read on first use, after that table ran.
bool sh_stream_on() {
    static const bool on = [] {
        if (const char* v = std::getenv("STRATA_SH_STREAM")) return v[0] != '0';
#if defined(STRATA_USE_HIP)
        return false;
#else
        return true;
#endif
    }();
    return on;
}

// A one-token window always keeps its token, so its graph advances the sequence state itself (the GDN conv history
// and recurrence state) and Verifier::commit launches no commit graph after it (eddoursul's fork, F7).  CUDA only:
// HIP keeps the commit graph after every window (STRATA_ONE_TOKEN_COMMIT=0 does too).
// The head's hyper-connection read for the window's tokens in one launch set (CUDA; HIP keeps gr_read per token;
// STRATA_HEAD_MIX_MULTI=0 does too): bitwise the same sums.
bool head_mix_multi_enabled() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HEAD_MIX_MULTI");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
#endif
}

bool one_token_self_commit() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    static const bool on = [] {
        const char* v = std::getenv("STRATA_ONE_TOKEN_COMMIT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
#endif
}

// True when the GPU runs under the i915 kernel driver (Arc Alchemist: A310-A770), read from sysfs.
bool intel_i915_gpu() { return strata::intel_gpu_driver() == "i915"; }

// How long the host waits for a layer's doorbell before it gives the window up (#267).  20 s by default; the first
// window of a run on a card that JIT-compiles its kernels (no AOT: an Arc A750 needs FP64 emulation) can take longer
// to start, so STRATA_RING_TIMEOUT_S raises it.
std::chrono::seconds strata_ring_timeout() {
    static const long s = [] {
        const char* v = std::getenv("STRATA_RING_TIMEOUT_S");
        const long n = v ? std::atol(v) : 20;
        return n >= 1 ? n : 20L;
    }();
    return std::chrono::seconds(s);
}

// STRATA_VERIFY_STEPPED: no kernel of the window reads a host store while it runs, and no host reads a kernel's store
// while it runs - every flag is raised before the segment that waits on it is submitted, and the host reads a
// segment's rings and payload after the segment drained. So the window's mapped buffers need no uncached host memory
// (sycl_queue.hpp) and its doorbell publishes need no uncached payload stores (sycl_doorbell.hpp). Both cost the
// window time on an A770 (xe, Coder IQ1_M, 8K prompt): the GPU waited ~70 ms a window for the CPU with them, ~42
// without; decode 22.8-25.0 -> 26.9-28.9 tok/s. STRATA_VERIFY_HOST_UNCACHED=1 keeps both.
bool window_host_uncached() {
    static const bool v = !g_stepped || env_on("STRATA_VERIFY_HOST_UNCACHED");
    return v;
}

bool mapped(size_t bytes, void **h, void **d) try {
    /*
    DPCT1048: The original value cudaHostAllocMapped is not meaningful in the
    migrated code and was removed or replaced with 0. You may need to check the
    migrated code.
    */
    // polled by the window's kernels: uncached host memory (sycl_queue.hpp); a buffer only read or written between
    // kernels (the stepped window's payload) is ordinary host USM
    if (DPCT_CHECK_ERROR(*h = window_host_uncached() ? strata::host_malloc_polled(bytes, dpct::get_in_order_queue())
                                     : sycl::malloc_host(bytes, dpct::get_in_order_queue())) !=
        0 || *h == nullptr) return false;
    std::memset(*h, 0, bytes);
    return DPCT_CHECK_ERROR(*d = (void *)*h) == 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) { return qsa_shapes(g); }

const WeightRef* need(const LayerView& v, const char* suffix, std::string& err) {
    const WeightRef* r = v.get(suffix);
    if (r == nullptr && err.empty()) err = v.name(suffix) + " is missing";
    return r;
}

bool native_of(const WeightRef* w, const std::string& name, std::string& err) {
    if (w == nullptr) return false;
    if (w->native_data == nullptr) {
        err = "verify: " + name + " is not served natively (run with --native)";
        return false;
    }
    return true;
}

}  // namespace

namespace {
std::atomic<const Verifier*> g_diag_verifier{nullptr};
void diag_active_verifier(std::FILE* f) {
    if (const Verifier* v = g_diag_verifier.load()) v->diag(f);
}
// #267: every live verifier (a layer split has one per stage), for the release before the engine ends
constexpr int kLiveMax = 16;
std::atomic<Verifier*> g_live[kLiveMax];
void release_live_verifiers(std::FILE* f) {
    for (auto& slot : g_live)
        if (Verifier* v = slot.load()) {
            const Clock::time_point t0 = Clock::now();
            const bool done = v->release_gpu_waits(5000);
            if (f != nullptr)
                std::fprintf(f, "strata: released the verify window's GPU waits (#267): the GPU %s\n",
                             done ? ("finished in " + std::to_string((long long) ms_since(t0)) + " ms").c_str()
                                  : "did not finish within 5 s");
        }
    if (f != nullptr) std::fflush(f);
}
// #649: the host side of STRATA_VERIFY_TRACE - a ring of the last kTraceN handshake events of every verifier
struct TraceEv {
    int64_t t_ns;
    const void* who;
    int64_t window, step, layer, aux;
    uint32_t seq, flag, a, b;
    const char* what;
};
constexpr uint64_t kTraceN = 4096;
TraceEv g_trace_ring[kTraceN];
std::atomic<uint64_t> g_trace_next{0};
int64_t trace_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
std::string released_note(bool drained) {
    return drained ? "; its GPU waits were released and the GPU finished (#267)"
                   : "; its GPU waits were released but the GPU did not finish within 5 s (#267)";
}
// #267 test hook: STRATA_TEST_VERIFY_STALL=N withholds the last layer's flag in the N-th window (1-based), so the
// GPU spins on a flag nobody raises - the bounded window wait and the release are then what ends it.  Unset: never.
const int64_t g_test_stall = [] {
    const char* e = std::getenv("STRATA_TEST_VERIFY_STALL");
    return e != nullptr ? (int64_t) std::atoll(e) : (int64_t) 0;
}();
}  // namespace

bool Verifier::release_gpu_waits(int timeout_ms) try {
    released_.store(true);
    trace_ev("RELEASE", -1, -1, timeout_ms);
    // the words the spin kernels read (wait_flag_ge, wait_flag_ge_or) are mapped host memory, so a store here
    // reaches them with no API call; UINT32_MAX is past every ring.  (E-6's skip words are device memory, but
    // wait_flag_ge_or also returns on its flag.)  A host function raising flag B later only raises.
    for (uint32_t* p : {h_flag_, h_flagA_, h_flagB_})
        if (p != nullptr) *(volatile uint32_t*) p = UINT32_MAX;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    // SYCL port: no cudaStreamQuery; a queue with nothing left to run reports ext_oneapi_empty()
    for (dpct::queue_ptr q : {cs_, copy_}) {
        if (q == nullptr) continue;
        while (!q->ext_oneapi_empty()) {
            if (ms_since(t0) > timeout_ms) {
                trace_ev("RELEASE-NOT-DRAINED", -1, -1, (int64_t) ms_since(t0));
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    trace_ev("RELEASE-DRAINED", -1, -1, (int64_t) ms_since(t0));   // aux: ms the GPU took to finish once released
    return true;
} catch (...) {
    return false;   // the runtime may already be tearing down (the watchdog's path ends the engine)
}

void Verifier::trace_ev(const char* what, int64_t step, int64_t layer, int64_t aux) const {
    if (!g_trace) return;
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    TraceEv& e = g_trace_ring[g_trace_next.fetch_add(1) % kTraceN];
    e = {trace_now_ns(), this, windows, step, layer, aux, rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_), what};
}

void Verifier::trace_dump(std::FILE* f) const {
    if (!g_trace || f == nullptr) return;
    static const char* names[kProfPer] = {"pre", "hc-read0", "qkv gemv", "conv", "ab", "z", "rec", "kv-idx",
                                          "k/v rope", "kv append", "q", "scores+topk", "kv-resolve", "attention",
                                          "gate", "", "out-proj", "router+ring", "shared", "waitA", "VRAM hits", "waitB",
                                          "PCIe grp", "waitCPU", "combine", "", "head", "", "", "", "", "", ""};
    const uint64_t end = g_trace_next.load();
    const uint64_t n = end < 160 ? end : 160;
    std::fprintf(f, "strata verify trace (#649): the last %llu handshake events (host clock, us before the last; "
                    "words: seq = GPU rang, flag = CPU served, A = plan, B = copies)\n", (unsigned long long) n);
    const int64_t t_last = n ? g_trace_ring[(end - 1) % kTraceN].t_ns : 0;
    for (uint64_t i = end - n; i < end; ++i) {
        const TraceEv& e = g_trace_ring[i % kTraceN];
        if (e.what == nullptr) continue;
        std::fprintf(f, "  %10.1f %-20s v%p window %lld step %lld layer %lld aux %lld | seq %u flag %u A %u B %u\n",
                     (double) (t_last - e.t_ns) / 1000.0, e.what, e.who, (long long) e.window, (long long) e.step,
                     (long long) e.layer, (long long) e.aux, e.seq, e.flag, e.a, e.b);
    }
    if (trace_h_ == nullptr || g_ == nullptr) return;
    // the breadcrumbs: every stamp point the GPU passed in this window (ns of the GPU clock after the window's first)
    unsigned long long t0 = ~0ull;
    for (size_t i = 0; i < trace_n_; ++i) {
        const unsigned long long t = *(const volatile unsigned long long*) (trace_h_ + i);
        if (t != 0 && t < t0) t0 = t;
    }
    if (t0 == ~0ull) {
        std::fprintf(f, "strata verify trace: no GPU breadcrumb in this window (the GPU never started it)\n");
        return;
    }
    int64_t last_l = -1;
    for (int64_t l = 0; l <= g_->n_layers; ++l)
        for (int i = 0; i < kProfPer; ++i)
            for (int grp = 0; grp < 2; ++grp)
                if (trace_h_[(size_t) ((l * kProfPer + i) * 2 + grp)] != 0) last_l = l;
    std::fprintf(f, "strata verify trace: GPU breadcrumbs (us after the window's first), the GPU got as far as layer "
                    "%lld:\n", (long long) last_l);
    for (int64_t l = std::max<int64_t>(0, last_l - 2); l <= std::min<int64_t>(g_->n_layers, last_l + 1); ++l)
        for (int grp = 0; grp < 2; ++grp) {
            std::string line;
            char b[64];
            for (int i = 0; i < kProfPer; ++i) {
                const unsigned long long t = trace_h_[(size_t) ((l * kProfPer + i) * 2 + grp)];
                if (t == 0) continue;
                std::snprintf(b, sizeof b, " %s(%d)=%.1f", names[i], i, (double) (t - t0) / 1000.0);
                line += b;
            }
            if (!line.empty())
                std::fprintf(f, "  layer %lld group %d:%s\n", (long long) l, grp, line.c_str());
        }
    std::fflush(f);
}

void Verifier::diag(std::FILE* f) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    // #251: outside a verify stage these are the LAST window's numbers (it finished), not the stalled work's
    const char* where = progress().where.load();
    const bool current = where != nullptr && std::strncmp(where, "verify window", 13) == 0;
    std::fprintf(f, "  verify window%s: %d tokens at position %lld, host at layer step %u; the GPU rang %u; flags: "
                    "served %u, plan (A) %u, copies (B) %u\n", current ? "" : " (last window, not the current stage)",
                 last_t_, (long long) last_pos0_, cur_layer_ + 1, rd(h_seq_), rd(h_flag_), rd(h_flagA_), rd(h_flagB_));
    trace_dump(f);   // #649: STRATA_VERIFY_TRACE=1 only
}

Verifier::~Verifier() try {
    // SYCL port: at process exit the Level Zero context can already be gone (serve mode's end), and a destructor
    // that throws aborts the process (exit 139); the waits and frees below are best-effort then.
    const Verifier* self = this;
    g_diag_verifier.compare_exchange_strong(self, nullptr);
    for (auto& slot : g_live) {
        Verifier* me = this;
        slot.compare_exchange_strong(me, nullptr);
    }
    if (cs_) cs_->wait();
    if (sh_cs_) sh_cs_->wait();
    for (auto& e : exec_)
        if (e) delete (e);
    for (auto& e : exec_nr_)
        if (e) delete (e);
    for (auto& v : seg_)
        for (auto* e : v)
            if (e) delete e;
    if (commit_exec_) delete (commit_exec_);
    for (auto& kv : exec_bm_)
        if (kv.second) delete (kv.second);
    for (auto& kv : commit_bm_)
        if (kv.second) delete (kv.second);
    if (arena_b_) sycl::free(arena_b_, dpct::get_in_order_queue());
    if (h_commitb_) sycl::free(h_commitb_, dpct::get_in_order_queue());
    if (qcnt_) sycl::free(qcnt_, dpct::get_in_order_queue());
    if (cs_ && cs_ != ext_stream_) dpct::get_current_device().destroy_queue(
        cs_); // set_stream: the stage's stream, shared, not ours
    if (sh_cs_) dpct::get_current_device().destroy_queue(sh_cs_);
    if (copy_) { copy_->wait(); dpct::get_current_device().destroy_queue(copy_); }
    if (commit_done_) dpct::destroy_event(commit_done_);
    if (ev_done_) dpct::destroy_event(ev_done_);
    if (ev_commit_) dpct::destroy_event(ev_commit_);
    if (prof_pin_) sycl::free(prof_pin_, dpct::get_in_order_queue());
    if (ev_fork_) dpct::destroy_event(ev_fork_);
    if (ev_join_) dpct::destroy_event(ev_join_);
    if (arena_) sycl::free(arena_, dpct::get_in_order_queue());
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_commit_, h_ple_, h_out_, h_x_, h_ids_, h_w_, h_seq_, h_flag_, h_ymiss_,
                     h_flagA_, h_plan_, h_flagB_, h_plan_err_};
    for (void* h : hosts)
        if (h) strata::host_free_polled(h, dpct::get_in_order_queue());
} catch (...) {
}


bool Verifier::init(const WeightTable &wt, const ModelGeometry &g,
                    SessionState &ss, const VerifyHits &hits,
                    const NativeHead *head, int max_t, std::string &err) try {
    g_diag_verifier.store(this);
    diag_verify_fn().store(&diag_active_verifier);
    for (auto& slot : g_live) {
        Verifier* none = nullptr;
        if (slot.load() == this || slot.compare_exchange_strong(none, this)) break;
    }
    release_gpu_fn().store(&release_live_verifiers);
    device_ = dpct::get_current_device_id(); // a layer split's stage on another
                                             // GPU: its streams, graphs and
                                             // buffers live there
    strata::kernels::fused_gr_check();   // once per card: which bitwise-equal hyper-connection read runs there
    wt_ = &wt;
    g_ = &g;
    ss_ = &ss;
    hits_ = hits;
    head_ = head;
    max_t_ = max_t;
    sampling_.greedy = true;      // a fresh verifier samples greedily until set_sampling says otherwise
    sampling_.temperature = 0.0f;
    if (max_t < 2 || max_t > strata::kernels::kVerifyMaxT || max_t > strata::kernels::cpu::MAXT) {
        err = "verify: the window must hold 2.." + std::to_string(strata::kernels::kVerifyMaxT) + " tokens";
        return false;
    }
    if (hits.d_res == nullptr || hits.cache_base == nullptr || hits.blob <= 0) {
        err = "verify: needs the profile-filled VRAM expert tier (--expert-profile and --expert-cache); with "
              "--expert-cache auto, no VRAM was left for it - the 'no VRAM is left for the expert cache' line above "
              "says how much is short and what makes room (a smaller --max-context, --kv q4_0, setup --draft-vocab en, "
              "images on the CPU)";
        return false;
    }
    std::string why;
    if (!layer_verify_compatible(why)) {
        err = "verify: " + why + " (the verify window reproduces the default native decode path)";
        return false;
    }
    if (!strata::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr) || ss.k != 10 || g.ssm_state_size != 128 ||
        g.ssm_d_conv != 4) {
        err = "verify: geometry differs from the artifact's";
        return false;
    }
    if (le_ < 0) le_ = g.n_layers;
    if (lb_ < 0 || lb_ >= le_ || le_ > g.n_layers || (lb_ > 0 && hand_in_ == nullptr) ||
        (le_ < g.n_layers && hand_out_ == nullptr)) {
        err = "verify: the stage's layer range or its hand-off buffers are wrong";
        return false;
    }
    const WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { err = "verify: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;

    const strata::kernels::QsaShapes s = shapes_of(g);
    cap_ = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    max_blocks_ = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);

    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t C = (uint64_t) g.ssm_conv_channels, ZV = (uint64_t) g.ssm_value_dim, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t IQ = (uint64_t) g.idx_q_heads, ID = (uint64_t) g.idx_key_dim;
    const uint64_t nG = (uint64_t) g.n_gdn_layers(), nQ = (uint64_t) g.n_qsa_layers();
    const uint64_t HS = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
    const uint64_t TS = (uint64_t) (s.idx_block - 1) * ID;
    const int max_in = (int) std::max<uint64_t>(std::max<uint64_t>(N, ZV), NH * HD);

    // ---- mapped staging
    bool ok = mapped(T * 4, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(T * strata::kernels::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) &&
              mapped(T * (NH + NKV + IQ) * 4, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped((2 + T) * 4 + 16, (void**) &h_commit_, (void**) &m_commit_) &&
              mapped(T * N * 4, (void**) &h_ple_, (void**) &m_ple_) &&
              mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * N * 4, (void**) &h_x_, (void**) &m_x_) &&
              mapped(T * K * 4, (void**) &h_ids_, (void**) &m_ids_) &&
              mapped(T * K * 4, (void**) &h_w_, (void**) &m_w_) &&
              mapped(64, (void**) &h_seq_, (void**) &m_seq_) &&
              mapped(64, (void**) &h_flag_, (void**) &m_flag_) &&
              mapped(64, (void**) &h_flagA_, (void**) &m_flagA_) &&
              mapped(64, (void**) &h_flagB_, (void**) &m_flagB_) &&
              mapped(64, (void**) &h_plan_err_, (void**) &m_plan_err_) &&
              mapped(T * K * N * 4, (void**) &h_ymiss_, (void**) &m_ymiss_);
    if (!ok) { err = "verify: mapped staging allocation failed"; return false; }
    // the GPU plan: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)
    {
        const int64_t cap = (int64_t) (T * K);
        const int64_t i32 = 4 + (cap + 1) + cap + cap;
        const int64_t ptr_off = (i32 + 1) & ~1ll;
        plan_i32_ = ptr_off + 4 * cap + (cap + 1) + 1;
        if (!mapped((size_t) plan_i32_ * 4 * 2 + 64, (void**) &h_plan_, (void**) &m_plan_)) {
            err = "verify: mapped plan allocation failed";
            return false;
        }
        sink_.counts = h_plan_;
        sink_.start = h_plan_ + 4;
        sink_.dst = sink_.start + cap + 1;
        sink_.tok = sink_.dst + cap;
        sink_.ptr = (unsigned long long*) (h_plan_ + ptr_off);
        sink_.ptr2 = sink_.ptr + cap;
        sink_.start2 = h_plan_ + ptr_off + 4 * cap;
        sink_.cap = cap;
        sink_.publish = &Verifier::publish_plan;
        sink_.fetch = &Verifier::fetch_dma;
        sink_.ctx = this;
    }

    // APR: the rotated copy of the expert input (nat_xr_, T x N floats) only where a layer's gate/up is
    // Hadamard-folded; its one reader is that layer's f.had_gu branch in the expert-input step below
    bool had_gu_any = false;
    {
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (size_t l = 0; lay.native && l < lay.fmt.size(); ++l) had_gu_any = had_gu_any || lay.fmt[l].had_gu;
    }
    // ---- the device arena: the same sequence counted, then carved
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(T * strata::kernels::kStepCount);
        pos_ = b.take<int32_t>(T * (NH + NKV + IQ)); commit_ = b.take<int32_t>(2 + T);
        ple_ = b.take<float>(T * N); emb_ = b.take<float>(T * N); R_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); bo_ = b.take<float>(T * N);
        inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
        sh_xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes(max_in, (int) T));
        qkv_L_ = b.take<float>(nG * T * C); h_L_ = b.take<float>(nG * T * C);
        gate_L_ = b.take<float>(nG * T * HV); beta_L_ = b.take<float>(nG * T * HV);
        z_ = b.take<float>(T * ZV); y_ = b.take<float>(T * ZV); y_dummy_ = b.take<float>(T * ZV);
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        idx_raw_L_ = b.take<float>(nQ * T * ID); qidx_ = b.take<float>(T * IQ * ID);
        scores_ = b.take<float>(T * (uint64_t) max_blocks_); sel_ = b.take<int32_t>(T * (uint64_t) cap_);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>(T * (uint64_t) attn_scratch_floats_);
        tail_snap_ = b.take<float>(nQ * TS);
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); hit_out_ = b.take<float>(T * K * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        plan_ = b.take<int32_t>(2 * ((uint64_t) plan_i32_ + 16));
        staging_ = b.take<uint8_t>((uint64_t) kStagingBlobs * strata::kernels::cpu::expert_layout().max_blob);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        nat_xr_ = had_gu_any ? b.take<float>(T * N) : nullptr;
        hit_scratch_ = b.take<uint8_t>(std::max<uint64_t>(
            strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff),
            strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff)));
        head_mixed_ = b.take<float>(T * N); head_inj_ = b.take<float>(HC);
        sh_bf16_ = b.take<uint16_t>(T * N); sh_gate_ = b.take<float>(T * (uint64_t) g.n_ff);
        sh_up_ = b.take<float>(T * (uint64_t) g.n_ff); sh_g_ = b.take<float>(T + 4);
        head_logits_ = b.take<float>(T * (uint64_t) n_vocab_);
        arg_scratch_ = b.take<uint8_t>(strata::kernels::argmax_rows_scratch_bytes((int) T));
        one_ = b.take<int32_t>(4);
        hist_snap_ = b.take<float>(T * HS);
        ple_key_ = b.take<float>(T * (uint64_t) strata::kernels::NG_HC_DIM); ple_val_ = b.take<float>(T * N);
    };
    Bump count;
    carve(count);
    if (DPCT_CHECK_ERROR(arena_ = (void *)strata::malloc_device_guarded(
                             count.used, dpct::get_in_order_queue())) != 0) {
        err = "verify: the device arena (" + std::to_string(count.used >> 20) + " MiB) does not fit";
        return false;
    }
    strata::big_fill_zero(dpct::get_in_order_queue(), arena_, count.used);
    if (g_trace && trace_h_ == nullptr) {   // #649: the breadcrumbs, mapped so they read while the GPU hangs
        trace_n_ = (size_t) (g.n_layers + 1) * kProfPer * 2;
        if (!mapped(trace_n_ * 8, (void**) &trace_h_, (void**) &trace_m_)) {
            trace_h_ = trace_m_ = nullptr;
            trace_n_ = 0;
        }
#if defined(STRATA_USE_HIP)
        std::fprintf(stderr, "strata verify trace (#649): on; coherent words %s, doorbell %s, HIP_HOST_COHERENT=%s "
                             "HSA_ENABLE_SDMA=%s GPU_MAX_HW_QUEUES=%s\n", g_coherent ? "explicit" : "default",
                     g_doorbell_store ? "stored" : "incremented", std::getenv("HIP_HOST_COHERENT") ? std::getenv("HIP_HOST_COHERENT") : "-",
                     std::getenv("HSA_ENABLE_SDMA") ? std::getenv("HSA_ENABLE_SDMA") : "-",
                     std::getenv("GPU_MAX_HW_QUEUES") ? std::getenv("GPU_MAX_HW_QUEUES") : "-");
#else
        std::fprintf(stderr, "strata verify trace (#649): on\n");
#endif
    }
    prof_on_ = std::getenv("STRATA_VERIFY_PROFILE") != nullptr;
    if (prof_on_) {
        const size_t np = (size_t) g.n_layers * kProfPer + 4;
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        if (DPCT_CHECK_ERROR(prof_ = (unsigned long long *)sycl::malloc_device(
                                 np * 8, dpct::get_in_order_queue())) != 0) {
            prof_on_ = false; prof_ = nullptr;
        } else {
            dpct::get_in_order_queue().memset(prof_, 0, np * 8).wait();
            prof_h_.assign(np, 0);
        }
    }
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    {
        const int32_t one = 1;
        if (DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memcpy(one_, &one, sizeof one)
                                 .wait()) != 0) {
            err = "verify: the arena could not be set";
            return false;
        }
    }
    {   // APR (prism.hadamard, hadamard.txt): the sign vectors on this device, before any graph capture
        had_sx_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) g.n_embd), (int) g.n_embd);
        had_sh_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) g.n_ff), (int) g.n_ff);
    }
    sink_.staging = (unsigned long long) staging_;
    sink_.staging_cap = kStagingBlobs;
    (void) TS;
    /*
    DPCT1025: The SYCL queue is created ignoring the flag and priority
    options.
    */
    if (DPCT_CHECK_ERROR(
            copy_ = dpct::get_current_device().create_queue(true)) != 0) {
        err = "verify: copy stream create failed";
        return false;
    }
    if (ext_stream_ != nullptr) cs_ =
        ext_stream_; // set_stream (pipelined windows): the stage's shared
                     // stream
    /*
    DPCT1025: The SYCL queue is created ignoring the flag and priority
    options.
    */
    else if (DPCT_CHECK_ERROR(
                 cs_ = dpct::get_current_device().create_queue(true)) != 0)
        cs_ = &dpct::get_in_order_queue();
    /*
    DPCT1025: The SYCL queue is created ignoring the flag and priority
    options.
    */
    if (cs_ == &dpct::get_in_order_queue() ||
        DPCT_CHECK_ERROR(
            sh_cs_ = dpct::get_current_device().create_queue(true)) != 0) {
        err = "verify: stream create failed";
        return false;
    }
    if (DPCT_CHECK_ERROR(commit_done_ = new sycl::event()) != 0 ||
        DPCT_CHECK_ERROR(ev_fork_ = new sycl::event()) != 0 ||
        DPCT_CHECK_ERROR(ev_join_ = new sycl::event()) != 0) {
        err = "verify: event create failed";
        return false;
    }
    // Check whether 100% of experts across [lb_, le_) are resident in this stage's VRAM cache.
    // When true, every layer plans on device and writes directly into parts_ without any CPU doorbells,
    // wait_flag_ge spins, PCIe empty launches, or moe_hit_add copies.
    all_resident_ = false;
    ar_off_ = false;
    h_res_ = hits.h_res;
    if (h_plan_err_ != nullptr) *(volatile uint32_t*) h_plan_err_ = 0;
    if (hits.h_res != nullptr && hits.d_res != nullptr && hits.cache_base != nullptr) {
        const char* v_ar = std::getenv("STRATA_VERIFY_ALL_RESIDENT");
        if (v_ar == nullptr || std::atoi(v_ar) != 0) {
            bool all_ok = true;
            for (int64_t l = lb_; l < le_ && all_ok; ++l) {
                for (int64_t e = 0; e < g.n_expert; ++e) {
                    if (hits.h_res[l * g.n_expert + e] < 0) {
                        all_ok = false;
                        break;
                    }
                }
            }
            all_resident_ = all_ok;
        }
    }
    // E-6: a layer whose routed experts are all resident is planned on the device (STRATA_VERIFY_DEVICE_PLAN=1: on;
    // exact, but neutral on RIBPC 1-2 GPUs: off by default)
    {
        const char* v = std::getenv("STRATA_VERIFY_DEVICE_PLAN");
        // (not with set_always_publish: the device table may lag the host's while windows are in flight)
        device_plan_ = !all_resident_ && !always_publish_ && (v != nullptr && std::atoi(v) != 0);
    }
    // (halo's STRATA_VERIFY_RESIDENT=1 is this window's all_resident_ graph above, which 0.1.39 has on by default)
    if (g_qfuse()) {   // S26: the HC read's q8_1 group counters, zeroed once (each launch leaves them zero)
        if (DPCT_CHECK_ERROR(qcnt_ = sycl::malloc_device<unsigned int>(
                                 (size_t)(g.n_embd / 32),
                                 dpct::get_in_order_queue())) != 0 ||
            DPCT_CHECK_ERROR(
                dpct::get_in_order_queue()
                    .memset(qcnt_, 0,
                            sizeof(unsigned) * (size_t)(g.n_embd / 32))
                    .wait()) != 0) {
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            if (qcnt_) sycl::free(qcnt_, dpct::get_in_order_queue());
            qcnt_ = nullptr;
        }
    }
    if (all_resident_ || device_plan_) {
        bool ok2 = true;
        if (device_plan_)
            ok2 = DPCT_CHECK_ERROR(skip_ = (uint32_t *)sycl::malloc_device(
                                       64, dpct::get_in_order_queue())) == 0 &&
                  DPCT_CHECK_ERROR(
                      dpct::get_in_order_queue().memset(skip_, 0, 64).wait()) ==
                      0;
        if (ok2 && hits.slot_off != nullptr && hits.n_slots > 0) {
            ok2 = DPCT_CHECK_ERROR(
                      slot_off_d_ = sycl::malloc_device<unsigned long long>(
                          (size_t)hits.n_slots, dpct::get_in_order_queue())) ==
                      0 &&
                  DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                       .memcpy(slot_off_d_, hits.slot_off,
                                               (size_t)hits.n_slots *
                                                   sizeof(unsigned long long))
                                       .wait()) == 0;
        }
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        if (!ok2) {; all_resident_ = false; device_plan_ = false; }
    }
    if (std::getenv("STRATA_VERIFY_DEBUG") != nullptr) {   // SYCL port: the per-layer residual ladder (token 0)
        dbgR_ = (float*) sycl::malloc_device((size_t) g.n_layers * g.n_embd * 4, dpct::get_in_order_queue());
        dbgM_ = (float*) sycl::malloc_device((size_t) g.n_layers * g.n_embd * 4, dpct::get_in_order_queue());
    }
    std::fprintf(stderr, "strata verify: window up to %d tokens, %.1f MiB of device buffers%s\n", max_t,
                 (double) count.used / 1048576.0,
                 all_resident_ ? " (100% VRAM resident: zero-doorbell graph)" : "");
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

const float* Verifier::final_R(int t) const { return R_ + (size_t) t * (size_t) (g_->hc * g_->n_embd); }

// ================================ THE WINDOW, AS CAPTURED ================================
//
// Plan v0.3 P6 (split window): with `groups_ == 2` the window's tokens are cut into two groups A = [0, T/2 up) and
// B = the rest, and the stream is ordered
//
//     pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | post(1,A) pre(2,A) | ...
//
// so the CPU computes A's experts of layer l while the GPU runs B's mixer and router of layer l, and B's experts
// while the GPU combines A and runs A's layer l+1.  B's mixer only needs A's mixer of the same layer (K/V, GDN
// state), never A's experts, so nothing waits that did not wait before.  Every token's arithmetic is unchanged.
bool Verifier::record_window(int T, dpct::queue_ptr cs, std::string &err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    // 0.1.43 (STRATA_VERIFY_LEAN=0 turns it off): with the device plan under STRATA_VERIFY_NO_HOST no host ever answers a doorbell, so
    // a layer's doorbell publish, the three flag waits, the plan copy, the PCIe-share group and its staging kernels are nodes of
    // the graph that only wait for each other or do nothing: ~9 nodes (~1.4 ms) per layer-window less.  The VRAM / mirror groups, the
    // combine and the answers are the same kernels on the same data.
    static const bool lean_env = [] { const char* v = std::getenv("STRATA_VERIFY_LEAN"); return !(v && v[0] == '0'); }();
    static const bool no_host_env = [] { const char* v = std::getenv("STRATA_VERIFY_NO_HOST"); return v && *v && std::strcmp(v, "0") != 0; }();
    // Not for a batch window (batch_rec_): unless the stage is all-resident, run_slot_rows / batch_poll wait for each
    // layer's doorbell, so a lean batch graph finishes with the ring silent ("verify batch: layer 0 never rang (graph
    // finished)", then the engine dies). Batch windows keep the doorbell nodes; the solo window stays lean.
    const bool lean = lean_env && no_host_env && device_plan_ && !batch_rec_ && !step_hook_ && !step_hook2_ && !remote_opt_ && !always_publish_ && !g_stepped;
    const WeightTable& wt = *wt_;
    SessionState& ss = *ss_;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    const int64_t HV = g.ssm_v_heads, HK = g.ssm_k_heads, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const int64_t IQ = g.idx_q_heads, ID = g.idx_key_dim, NE = g.n_expert, MT = max_t_;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    const int64_t TS = (s.idx_block - 1) * ID;
    const bool ple_on = ss.ple.ready() && ple_stage();
    auto Rt = [&](int t) { return R_ + (size_t) t * HC * N; };
    const int G = (split_ && T >= 2 && !batch_rec_) ? 2 : 1;   // a batch window is one group
    const bool self_commit = T == 1 && !batch_rec_ && !g_qfuse() && one_token_self_commit();   // see Verifier::commit
    static const bool dec_batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    // SYCL port, STRATA_VERIFY_EAGER=1: the window runs on the queue instead of as a graph, and each stage stamp is
    // a host-side wait + clock (there is no %globaltimer here), so the existing stage profiler reports ms per stage.
    static const bool eager = std::getenv("STRATA_VERIFY_EAGER") != nullptr;
    auto stamp = [&](int64_t l, int i, int grp) {
        if (trace_m_ != nullptr) gpu_stamp(trace_m_, (int) ((l * kProfPer + i) * 2 + grp), cs);   // #649
        if (!prof_on_ || grp != 0) return;
        if (eager) { cs->wait(); prof_h_[(size_t) (l * kProfPer + i)] = (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count(); }
        else gpu_stamp(prof_, (int) (l * kProfPer + i), cs);
    };
    const int tb_[2] = {0, (T + 1) / 2}, te_[2] = {G == 2 ? (T + 1) / 2 : T, T};
    // S26 STRATA_LFUSE=1: where the shared expert's gate / scale fusions apply (the paths they replace are the ones taken)
    auto lfuse_on = [&](int n) {
        return g_lfuse() && ar_on() && native_moe_combine_enabled() && dec_batch && n > 1 && n <= 8 &&
               shared_expert_native_bf16_enabled();
    };
    bool sg_gated_[2] = {false, false};   // per group: the combine applies the shared gate
    if (!batch_rec_) groups_[T] = G;
    const int64_t nQall = g.n_qsa_layers();
    // a batch window: row t is slot t, whose state lives in its own session
    auto slot_ss = [&](int t) -> SessionState& { return batch_rec_ ? *slots_[(size_t) brow_[t]] : ss; };
    const int hrow0 = batch_rec_ ? row_base_ : 0;   // a slot group's own hand-off rows

    // ---- the window's inputs, from mapped staging
    copy_i32_from_mapped(tok_, m_tok_, T, cs);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, cs);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) MT * (NH + NKV + IQ), cs);
    // per-ROW positions of the K rows [t][NKV] and the indexer query rows [t][IQ] (for batched RoPE)
    const int32_t* pos_k = pos_ + MT * NH;
    const int32_t* pos_i = pos_ + MT * (NH + NKV);

    // ---- the embeddings, broadcast to the hc streams - or, in a later stage of a layer split, the previous stage's
    // residual, pending write and inject (see set_stage)
    const int64_t HB = Verifier::handoff_floats(g);
    if (lb_ > 0) {
        const float* hin = hand_in_ + (size_t) hrow0 * HB;   // the group's rows: [R][bo][inj] contiguous
        copy_from_mapped(R_, hin, (int64_t) T * HC * N, cs);
        copy_from_mapped(bo_, hin + (size_t) T * HC * N, (int64_t) T * N, cs);
        copy_from_mapped(inj2_, hin + (size_t) T * (HC + 1) * N, (int64_t) T * HC, cs);
    } else if (const NativeEmbed* ne = native_embed()) {       // plan v0.3 P6: the GGUF-form table
        ne->gather_dev(tok_, T, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    } else {
        const WeightRef* w = wt.find("token_embd.weight");
        if (w == nullptr || w->codebook_iq4nl || (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
            err = "verify: token_embd.weight is missing or not an S2/S4/S8 tensor";
            return false;
        }
        const auto* codes = (const uint8_t*) w->data;
        const auto* scales = (const float*) (codes + w->codes_bytes);
        const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
        const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
        const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
        embedding_gather_dev(codes, scales, offsets, tok_, T, w->ne0, w->code_bits, w->code_bias, w->group_elems,
                             row_codes, row_groups, emb_, cs);
        broadcast_streams(emb_, R_, N, (int) HC, T, cs);
    }

    // per-layer state indices (GDN and QSA layers are numbered separately)
    std::vector<int64_t> gdn_idx((size_t) g.n_layers, -1), qsa_idx((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }

    // ---------------------------------------------------------------- pre(l, group): up to the ring
    auto pre = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        stamp(l, 0, grp);
        const LayerView v(wt, l);
        const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
        const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
        for (int h = 0; h < 2; ++h) {
            wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
            wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
            wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
            wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
            if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
        }
        // the previous layer's FFN write, folded into this layer's first read (a control vector after it has
        // already applied it)
        bool pending = l > 0 && !cvec().covers(l - 1);
        if (l == 1 && ple_on) {
            if (grp == 0) {
                if (ar_on()) wait_flag_ge(m_flag_, 1, cs);
                copy_from_mapped(ple_, m_ple_, (int64_t) T * N, cs);
            }
            float* normalized = (float*) ((uint8_t*) ss.ple.scratch + ple_block_scratch_bytes());
            static const bool ple_batch_env = [] {
                const char* e = std::getenv("STRATA_PLE_BATCH");
                return !e || e[0] != '0';
            }();
            const bool ple_batch_kv = ple_batch_env && dec_batch && n > 1 && ss.ple.w.key_bf16 != nullptr &&
                                      ss.ple.w.value_bf16 != nullptr && ple_native_bf16_enabled() &&
                                      ple_native_postops_enabled();
            // S25 (STRATA_PLE_BATCH=1): the key and value projections depend only on each row's n-gram embedding, so
            // the group's rows run through the multi-row forms of the same kernels (the weights read once); every
            // row's arithmetic is the single-row call's. The history-dependent rest stays row by row.
            static const bool ple_batch = [] { const char* e = std::getenv("STRATA_PLE_BATCH"); return e && e[0] == '1'; }();
            PleWeights pw = ss.ple.w;
            const bool nkey = pw.key_native_data != nullptr && pw.key_bf16 == nullptr;
            const bool batched = !ple_batch_kv && ple_batch && n > 1 && (pw.key_bf16 != nullptr || nkey) && ple_native_bf16_enabled();
            if (ple_batch_kv) {
                try {
                    bf16_gemv_fp32_mmvf_multi(ple_ + (size_t) tb * N, N, ss.ple.w.key_bf16,
                                              xn_ + (size_t) tb * HC * N, HC * N, N, HC * N, n, cs);
                    bf16_gemv_fp32_mmvf_multi(ple_ + (size_t) tb * N, N, ss.ple.w.value_bf16,
                                              z_ + (size_t) tb * N, N, N, N, n, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE multi: ") + e.what();
                    return false;
                }
            } else if (batched) {
                try {
                    if (pw.key_bf16 != nullptr)
                        bf16_gemv_fp32_mmvf_multi(ple_ + tb * N, N, pw.key_bf16, ple_key_ + (size_t) tb * NG_HC_DIM,
                                                  NG_HC_DIM, N, NG_HC_DIM, n, cs);
                    else {
                        native_quantize_q8_1(ple_ + tb * N, xq_, (int) N, n, cs);
                        native_mmvq(pw.key_native_type, pw.key_native_data, xq_, ple_key_ + (size_t) tb * NG_HC_DIM,
                                    (int) N, NG_HC_DIM, n, cs);
                    }
                    bf16_gemv_fp32_mmvf_multi(ple_ + tb * N, N, pw.value_bf16, ple_val_ + tb * N, N, N, N, n, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE batch: ") + e.what();
                    return false;
                }
            }
            // #783 PR-g (stuchapin909): the window's residual writes in one launch (STRATA_NO_MULTI_GR=1: per token)
            const bool multi_write = dec_batch && !g_no_multi_gr && n > 1;
            if (multi_write) gr_write_multi(Rt(tb), bo_ + tb * N, inj2_ + tb * HC, gs, Rt(tb), n, cs);
            for (int t = tb; t < te; ++t) {
                if (batched) { pw.pre_key = ple_key_ + (size_t) t * NG_HC_DIM; pw.pre_value = ple_val_ + t * N; }
                if (!multi_write) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                PleOut po;
                po.normalized = normalized;
                po.result = Rt(t);
                float* hist = batch_rec_ ? slot_ss(t).ple_hist : ss.ple.hist;   // the row's own history
                try {
                    if (ple_batch_kv) {
                        ple_block_projected(xn_ + (size_t) t * HC * N, z_ + (size_t) t * N, Rt(t), hist,
                                            ss.ple.w, po, ss.ple.scratch, cs);
                    } else {
                        ple_block(ple_ + t * N, Rt(t), hist, pw, po, ss.ple.scratch, cs);
                    }
                    ple_history_advance(hist, normalized, cs);
                } catch (const std::exception& e) {
                    err = std::string("verify PLE: ") + e.what();
                    return false;
                }
                copy_from_mapped(hist_snap_ + (size_t) t * HS, hist, HS, cs);
            }
            pending = false;
        }
        auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = tb; t < te; ++t) {
                FusedGrArgs& a = fa[t - tb];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj_prev + t * HC;
                a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
                a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
                a.q8_down = (const uint8_t*) wd[half]->hc_q8; a.q8_up = (const uint8_t*) wu[half]->hc_q8;
                a.q8_inject = (const uint8_t*) wi[half]->hc_q8;
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.inject_out = inj_out + t * HC; a.mixed = mixed_ + t * N;
                if (qcnt_ != nullptr) {   // S26 STRATA_QFUSE: the consumer's q8_1 image written by the read itself
                    a.q8_cnt = qcnt_;
                    if (half == 0) a.q8_mixed = xq_ + (size_t) (t - tb) * (N / 32) * 36;
                    else if (strata::kernels::cpu::expert_layout().native) a.q8_mixed = nat_xq_ + (size_t) t * (N / 32) * 36;
                }
            }
            return fused_gr_read_multi(fa, n, xn_ + (size_t) tb * HC * N, cs, (prof_on_ && grp == 0) ? prof_ : nullptr,
                                (int) (l * kProfPer + (half == 0 ? 27 : 30)));
        };
        const bool q8_attn = gr_read_group(0, pending, inj2_, inj_);   // true: xq_ holds mixed's q8_1 (STRATA_QFUSE)
        stamp(l, 1, grp);
        float* xm = mixed_ + tb * N;
        try {
            if (!is_qsa_layer(g, l)) {
                // ======================= GDN =======================
                const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                *wsa = need(v, "ssm_a", err);
                if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
                    !native_of(wout, v.name("ssm_out.weight"), err))
                    return false;
                const int64_t gi = gdn_idx[(size_t) l];
                float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                float* qkv = qkv_L_ + (size_t) gi * MT * C;
                float* hb = h_L_ + (size_t) gi * MT * C;
                float* gate = gate_L_ + (size_t) gi * MT * HV;
                float* beta = beta_L_ + (size_t) gi * MT * HV;
                if (!q8_attn) native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                native_mmvq(wqkv->native_type, wqkv->native_data, xq_, qkv + (size_t) tb * C, (int) N, (int) C, n, cs);
                stamp(l, 2, grp);
                if (batch_rec_) {   // contiguous rows may be proposals for the same slot
                    for (int t = tb; t < te;) {
                        const int first = t;
                        while (t < te && brow_[t] == brow_[first]) ++t;
                        SessionState& sx = slot_ss(first);
                        const float* cx = sx.gdn_state + (size_t) (gi - sx.gdn_ord0) * gdn_floats +
                                          (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                        gdn_conv_l2_multi(cx, qkv + (size_t) first * C, (const float*) wc->data,
                                          hb + (size_t) first * C, (int) C, (int) (2 * HK), EPS, t - first, cs, 0);
                    }
                } else
                // a one-token window keeps its token: it advances the conv history and the state itself (commit)
                gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs, tb,
                                  self_commit);
                stamp(l, 3, grp);
                gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                             (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N, (int) HV,
                             n, cs);
                stamp(l, 4, grp);
                native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n, cs);
                stamp(l, 5, grp);
                // the recurrence from the untouched state over tokens [0, te); outputs only for this group's
                if (batch_rec_) {   // each slot's recurrence over its own proposed-token group
                    for (int t = tb; t < te;) {
                        const int first = t;
                        while (t < te && brow_[t] == brow_[first]) ++t;
                        SessionState& sx = slot_ss(first);
                        float* stx = sx.gdn_state + (size_t) (gi - sx.gdn_ord0) * gdn_floats;
                        gdn_step_norm_multi(stx, hb + (size_t) first * C, (int) C, gate + (size_t) first * HV,
                                            beta + (size_t) first * HV, z_ + (size_t) first * ZV,
                                            (const float*) wnm->data, EPS, y_ + (size_t) first * ZV,
                                            (int) HK, (int) HV, t - first, nullptr, cs, 0);
                    }
                } else
                gdn_step_norm_multi(state, hb, (int) C, gate, beta, z_, (const float*) wnm->data, EPS, y_, (int) HK,
                                    (int) HV, te, self_commit ? one_ : nullptr, cs, tb, g_qfuse() ? (void*) xq_ : nullptr);
                stamp(l, 6, grp);
                if (!g_qfuse()) native_quantize_q8_1(y_ + (size_t) tb * ZV, xq_, (int) ZV, n, cs);   // STRATA_QFUSE: done above
                native_mmvq(wout->native_type, wout->native_data, xq_, bo_ + tb * N, (int) ZV, (int) N, n, cs);
            } else {
                // ======================= QSA =======================
                const int64_t qi = qsa_idx[(size_t) l];
                const QsaState& st = ss.qsa_states[qi];
                const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                                *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                                *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                                *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
                if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
                    !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
                    return false;
                // #783 PR-f (stuchapin909): the per-head RMSNorm and the rope in one launch, bit-identical to the pair
                // (rope_parity check 6); STRATA_NO_NORM_ROPE=1 keeps the two; off on HIP until its parity check passes
                const bool fuse_nr = native_qsa_enabled() && native_rope_enabled() && native_norm_rope_usable((int) HD, (int) s.n_rot);
                auto norm_rope = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos) {
                    if (fuse_nr && native_norm_rope_usable(cols, (int) s.n_rot)) {
                        native_qsa_rms_norm_rope(data, cols, (const float*) norm->data, data, rows, cols, (int) s.n_rot, EPS,
                                                 rope_scaling(), pos, cs);
                        return;
                    }
                    if (native_qsa_enabled()) native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, EPS, cs);
                    else rms_norm_weighted(data, (const float*) norm->data, rows, cols, EPS, cs);
                    if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), pos, cs);
                    else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st.cos_tab, st.sin_tab, pos, cs);
                };
                float* idx_raw = idx_raw_L_ + (size_t) qi * MT * ID;
                // the per-token GEMVs / norms / RoPEs / copies of this layer as one launch over the
                // window's rows each - row-wise identical arithmetic (STRATA_DEC_BATCH=0: token by token)
                const bool qb = dec_batch && n > 1 && native_qsa_enabled() && native_rope_enabled() && !st.kv_q4;
                if (!q8_attn) native_quantize_q8_1(xm, xq_, (int) N, n, cs);
                if (qb) bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wik->data, idx_raw + tb * ID, ID, N, ID, n, cs);
                else for (int t = tb; t < te; ++t)
                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wik->data, idx_raw + t * ID, (int) N, (int) ID, cs);
                stamp(l, 7, grp);
                native_mmvq(wk->native_type, wk->native_data, xq_, kcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD), n, cs);
                native_mmvq(wv->native_type, wv->native_data, xq_, vcur_ + tb * NKV * HD, (int) N, (int) (NKV * HD), n, cs);
                if (qb) norm_rope(kcur_ + tb * NKV * HD, wkn, (int) (n * NKV), (int) HD, pos_k + tb * NKV);
                else for (int t = tb; t < te; ++t) norm_rope(kcur_ + t * NKV * HD, wkn, (int) NKV, (int) HD, pos_ + t * NH);
                if (st.kv_rot) {   // K and V rotated before they are stored (kv_q4.hpp)
                    fwht256_inplace_cuda(kcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                } else if (st.kv_hybrid) {   // K8V4: only V is rotated
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                }
                stamp(l, 8, grp);
                if (batch_rec_) {   // snapshot each slot's indexer tail before its first proposed row
                    for (int t = tb; t < te; ++t)
                        if (t == tb || brow_[t] != brow_[t - 1])
                            copy_from_mapped(tail_snap_b_ + ((size_t) brow_[t] * nQall + qi) * TS,
                                             slot_ss(t).qsa_states[qi].idx_tail, TS, cs);
                } else
                if (grp == 0) copy_from_mapped(tail_snap_ + (size_t) qi * TS, st.idx_tail, TS, cs);
                // #783 PR-d (stuchapin909): the window's K/V cells in one launch per pool instead of one per token (a
                // batch's rows each have their own K/V, so that case keeps the loop)
                if (dec_batch && !g_no_batch_kv && !batch_rec_) {
                    const int32_t* step_b = step_ + tb * kStepCount;
                    const float* kc_b = kcur_ + tb * NKV * HD;
                    const float* vc_b = vcur_ + tb * NKV * HD;
                    if (st.kv_hybrid) {   // K8V4: the unused half's lanes folded onto the used pool (layer.cpp)
                        kv_append_q8_steps(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_b, kStepCount,
                                           kc_b, kc_b, (int) (NKV * HD), n, s, cs, nullptr);
                        kv_append_q4_steps(st.v_q4, st.v_q4, st.page_table, step_b, kStepCount, n, vc_b, vc_b, s, cs,
                                           nullptr);
                    } else if (st.kv_q4)
                        kv_append_q4_steps(st.k_q4, st.v_q4, st.page_table, step_b, kStepCount, n, kc_b, vc_b, s, cs,
                                           &st.host);
                    else if (st.kv_int8)
                        kv_append_q8_steps(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_b, kStepCount,
                                           kc_b, vc_b, (int) (NKV * HD), n, s, cs, &st.host);
                    else
                        for (int t = tb; t < te; ++t)
                            kv_append_step(st.k_pool, st.v_pool, st.page_table, step_ + t * kStepCount,
                                           kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st.host);
                } else
                for (int t = tb; t < te; ++t) {
                    const QsaState& st = slot_ss(t).qsa_states[qi];   // the row's own K/V (ss's outside a batch)
                    const int32_t* step_t = step_ + t * kStepCount;
                    if (st.kv_hybrid) {   // K8V4: the unused half's lanes folded onto the used pool (layer.cpp)
                        const KvHostPools hk = kv_hybrid_k_half(st.host), hv = kv_hybrid_v_half(st.host);
                        const bool mirror = st.host.present();   // streamed: the host copy too
                        kv_append_q8_step(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, kcur_ + t * NKV * HD, s, cs, mirror ? &hk : nullptr);
                        kv_append_q4_step(st.v_q4, st.v_q4, st.page_table, step_t, vcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, mirror ? &hv : nullptr);
                    } else if (st.kv_q4)
                        kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step_t, kcur_ + t * NKV * HD,
                                          vcur_ + t * NKV * HD, s, cs, &st.host);
                    else if (st.kv_int8)
                        kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_t,
                                          kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st.host);
                    else
                        kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, kcur_ + t * NKV * HD,
                                       vcur_ + t * NKV * HD, s, cs, &st.host);
                }
                if (dec_batch && !g_no_batch_kv && !batch_rec_) {
                    const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    native_qsa_indexer_append_steps(idx_raw + tb * ID, step_ + tb * kStepCount + kStepPos, kStepCount,
                                                    n, 0, (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                    rope_scaling(), cs);
                } else {
                    for (int t = tb; t < te; ++t) {
                        const QsaState& sx = slot_ss(t).qsa_states[qi];
                        const QsaIndexerBuffers ib{sx.idx_tail, sx.idx_dead, sx.idx_pooled, sx.idx_block_pos};
                        native_qsa_indexer_append(idx_raw + t * ID, step_ + t * kStepCount + kStepPos, 0,
                                                  (const float*) wikn->data, EPS, ib, s, sx.max_cells,
                                                  rope_scaling(), cs);
                    }
                }
                stamp(l, 9, grp);
                native_mmvq(wq->native_type, wq->native_data, xq_, qfull_ + tb * NH * 2 * HD, (int) N, (int) (NH * 2 * HD),
                            n, cs);
                if (qb) {
                    if (fuse_nr) {   // the q/gate split reads q straight out of the q|gate rows (stride 2 * HD)
                        native_qsa_rms_norm_rope(qfull_ + tb * NH * 2 * HD, (int) (2 * HD), (const float*) wqn->data,
                                                 qcur_ + tb * NH * HD, (int) (n * NH), (int) HD, (int) s.n_rot, EPS,
                                                 rope_scaling(), pos_ + tb * NH, cs);
                    } else {
                    /*
                    DPCT1124: cudaMemcpy2DAsync is migrated to asynchronous
                    memcpy API. While the origin API might be synchronous, it
                    depends on the type of operand memory, so you may need to
                    call wait() on event return by memcpy API to ensure
                    synchronization behavior.
                    */
                    if (DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(
                            qcur_ + tb * NH * HD, (size_t)HD * 4,
                            qfull_ + tb * NH * 2 * HD, (size_t)HD * 2 * 4,
                            (size_t)HD * 4, (size_t)(n * NH),
                            dpct::device_to_device, *cs)) != 0) {
                        err = "verify: the q/gate split failed";
                        return false;
                    }
                    norm_rope(qcur_ + tb * NH * HD, wqn, (int) (n * NH), (int) HD, pos_ + tb * NH);
                    }
                    if (st.kv_rot) fwht256_inplace_cuda(qcur_ + tb * NH * HD, (int64_t) n * NH, cs);   // <Hq, Hk> = <q, k>
                    bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID, IQ * ID,
                                              N, IQ * ID, n, cs);
                    norm_rope(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ);
                } else {
                for (int t = tb; t < te; ++t) {
                    float* qc = qcur_ + t * NH * HD;
                    if (fuse_nr) {
                        native_qsa_rms_norm_rope(qfull_ + t * NH * 2 * HD, (int) (2 * HD), (const float*) wqn->data, qc,
                                                 (int) NH, (int) HD, (int) s.n_rot, EPS, rope_scaling(), pos_ + t * NH, cs);
                    } else {
                    /*
                    DPCT1124: cudaMemcpy2DAsync is migrated to asynchronous
                    memcpy API. While the origin API might be synchronous, it
                    depends on the type of operand memory, so you may need to
                    call wait() on event return by memcpy API to ensure
                    synchronization behavior.
                    */
                    if (DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(
                            qc, (size_t)HD * 4, qfull_ + t * NH * 2 * HD,
                            (size_t)HD * 2 * 4, (size_t)HD * 4, (size_t)NH,
                            dpct::device_to_device, *cs)) != 0) {
                        err = "verify: the q/gate split failed";
                        return false;
                    }
                    norm_rope(qc, wqn, (int) NH, (int) HD, pos_ + t * NH);
                    }
                    if (st.kv_rot) fwht256_inplace_cuda(qc, NH, cs);   // <Hq, Hk> = <q, k>
                }
                for (int t = tb; t < te; ++t) {
                    float* qx = qidx_ + t * IQ * ID;
                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wiq->data, qx, (int) N, (int) (IQ * ID), cs);
                    norm_rope(qx, wiqn, (int) IQ, (int) ID, pos_ + t * NH);
                }
                }
                stamp(l, 10, grp);
                if (batch_rec_) {   // each row selects and attends over its own slot's K/V
                    for (int t = tb; t < te; ++t) {
                        const QsaState& sx = slot_ss(t).qsa_states[qi];
                        qsa_block_scores(sx.idx_pooled, sx.idx_dead, qidx_ + t * IQ * ID, step_ + t * kStepCount, 1,
                                         max_blocks_, s, scores_ + (size_t) t * max_blocks_, cs);
                        qsa_block_topk(scores_ + (size_t) t * max_blocks_, step_ + t * kStepCount, 1, max_blocks_, cap_, s,
                                       sel_ + (size_t) t * cap_, cs);
                        qsa_kv_resolve(sx, *g_, sel_ + (size_t) t * cap_, step_ + t * kStepCount, 1, cap_, cs);
                        const QsaAttnPools px = qsa_attn_pools(sx);
                        qsa_decode_attn_batch(qcur_ + t * NH * HD, px, sel_ + (size_t) t * cap_, step_ + t * kStepCount,
                                              cap_, s, attn_scratch_ + (size_t) t * attn_scratch_floats_,
                                              attn_ + t * NH * HD, 1, cs);
                    }
                } else {
                qsa_block_scores(st.idx_pooled, st.idx_dead, qidx_ + tb * IQ * ID, step_ + tb * kStepCount, n, max_blocks_,
                                 s, scores_ + (size_t) tb * max_blocks_, cs);
                qsa_block_topk(scores_ + (size_t) tb * max_blocks_, step_ + tb * kStepCount, n, max_blocks_, cap_, s,
                               sel_ + (size_t) tb * cap_, cs);
                stamp(l, 11, grp);
                // KV streaming: the n selections' blocks resident (device-side, inside the graph)
                qsa_kv_resolve(st, *g_, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, n, cap_, cs);
                stamp(l, 12, grp);
                const QsaAttnPools pools = qsa_attn_pools(st);
                qsa_decode_attn_batch(qcur_ + tb * NH * HD, pools, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, cap_,
                                      s, attn_scratch_ + (size_t) tb * attn_scratch_floats_, attn_ + tb * NH * HD, n, cs);
                }
                stamp(l, 13, grp);
                if (st.kv_rot || st.kv_hybrid) fwht256_inplace_cuda(attn_ + tb * NH * HD, (int64_t) n * NH, cs);   // back: H^-1 = H
                if (qb) native_qsa_gate_apply(attn_ + tb * NH * HD, qfull_ + tb * NH * 2 * HD, attn32_ + tb * NH * HD,
                                              (int) (n * NH), (int) HD, cs);
                else
                for (int t = tb; t < te; ++t) {
                    if (native_qsa_enabled())
                        native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD,
                                              (int) NH, (int) HD, cs);
                    else
                        qsa_gate_apply_f32(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, s, attn32_ + t * NH * HD, cs);
                }
                stamp(l, 14, grp);
                native_quantize_q8_1(attn32_ + tb * NH * HD, xq_, (int) (NH * HD), n, cs);
                native_mmvq(wo->native_type, wo->native_data, xq_, bo_ + tb * N, (int) (NH * HD), (int) N, n, cs);
            }
        } catch (const std::exception& e) {
            err = "verify layer " + std::to_string(l) + ": " + e.what();
            return false;
        }
        stamp(l, 16, grp);
        const bool q8_ffn = gr_read_group(1, true, inj_, inj2_);   // true: nat_xq_ holds the MoE input's q8_1 (STRATA_QFUSE)
        // with CPU experts in the window the shared expert forks off once the doorbell is published (it no longer
        // competes with it for the GPU's first microseconds); STRATA_SH_FORK_LATE=0 forks at the top as before
        static const bool sh_fork_late_env = [] {
            const char* e = std::getenv("STRATA_SH_FORK_LATE");
            return !e || e[0] != '0';
        }();
        const bool sh_fork = sh_stream_on() && !prof_on_ && !g_stepped &&
                             sh_cs_ != &dpct::get_in_order_queue() &&
                             ev_fork_ != nullptr && ev_join_ != nullptr;
        const bool sh_fork_late = sh_fork && sh_fork_late_env && !ar_on();
        dpct::queue_ptr sh_stream = sh_fork ? sh_cs_ : cs;
        if (sh_fork && !sh_fork_late) {
            dpct::sync_barrier(ev_fork_, cs);
            (sh_cs_)->ext_oneapi_submit_barrier({*ev_fork_});
        }
        // the window's rows routed in 2 launches (one router GEMV reading the weight once, one
        // top-10) instead of 2 per token; every row's arithmetic is the single-token call's (STRATA_DEC_BATCH=0: old)
        const WeightRef* w_router = v.get("ffn_gate_inp.weight");
        // S26 STRATA_LFUSE=1: the shared expert's gate row rides in the router GEMV launch, its sigmoid + row scale
        // move into the combine, gate + up share one launch (all bitwise; resident combine path, 2-8 rows only)
        const WeightRef* w_sgi = lfuse_on(n) && g_lfuse_gate() ? v.get("ffn_gate_inp_shexp.weight") : nullptr;
        bool sg_ready = false;
        if (dec_batch && n > 1 && w_router != nullptr && native_router_enabled() && (NE == 512 || NE == 256) && K == 10) {
            try {   // SYCL port: the fused multi-token router for 256 experts as well (it was 512-only)
                if (w_sgi != nullptr)
                    sg_ready = bf16_gemv_fp32_mmvf_multi_aux(mixed_ + tb * N, N, (const uint16_t*) w_router->data,
                                                             logits_ + tb * NE, NE, N, NE, n,
                                                             (const uint16_t*) w_sgi->data, sh_g_ + tb, 1, cs);
                if (!sg_ready)
                bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) w_router->data, logits_ + tb * NE, NE, N,
                                          NE, n, cs);
                native_router_top10_multi_ne(logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, n, (int) NE, cs);
            } catch (const std::exception& e) { err = "verify router: " + std::string(e.what()); return false; }
#if defined(STRATA_HIP_GFX906)
        } else if (dec_batch) {
            // gfx906: the Coder's 256-expert router (the native router is 512 x 10 only) for the whole window in one
            // multi-column BF16 projection and one top-k, bitwise the per-token calls (route_window_parity); ~3 ms
            // of a ~55 ms verify window on 2x MI50.  STRATA_ROUTE_PER_TOKEN=1: the per-token calls.
            if (!moe_route_window(wt, g, l, K, ss.moe, mixed_ + tb * N, logits_ + tb * NE, ids_ + tb * K, w_ + tb * K, n,
                                  cs, err))
                return false;
#endif
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.logits = logits_ + t * NE; mb.ids = ids_ + t * K; mb.weights = w_ + t * K;
            if (!moe_route(wt, g, l, K, mb, mixed_ + t * N, cs, err, nullptr)) return false;
        }
        if (ar_on()) {
            resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                          hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                          plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, nullptr, 0, cs, m_plan_err_);
        } else {
            if (device_plan_)   // E-6: every routed expert resident: this group's plan without the host
                resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                              hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                              plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, skip_ + grp,
                              (uint32_t) ((l - lb_) * G + grp + 1), cs);
#if defined(STRATA_USE_HIP)
            if (g_doorbell_store)   // #649 A/B: the step's ring stored, not incremented over PCIe
                doorbell_publish_value(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
                                       m_ids_ + tb * K, m_w_ + tb * K, m_seq_, (uint32_t) ((l - lb_) * G + grp + 1), cs);
            else
#endif
            // #578: the helper GPUs reduce with the routing weights - publish them; set_always_publish: the rows
            // whatever the device's residency table says (pipelined windows)
            if (remote_opt_ || always_publish_)
                doorbell_publish(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
                                 m_ids_ + tb * K, m_w_ + tb * K, m_seq_, cs);
            else if (!lean) {   // (lean: nobody reads the doorbell)
                const int32_t* layer_res = hits_.d_res != nullptr ? (hits_.d_res + l * g.n_expert) : nullptr;
                doorbell_publish_res(xm, ids_ + tb * K, layer_res, (int) g.n_expert, (int64_t) n * N, (int64_t) n * K,
                                     m_x_ + tb * N, m_ids_ + tb * K, m_seq_, cs);
            }
            if (sh_fork_late) {
                dpct::sync_barrier(ev_fork_, cs);
                (sh_cs_)->ext_oneapi_submit_barrier({*ev_fork_});
            }
            if (step_hook_ && !step_hook_(l, grp)) { if (err.empty()) err = "verify: stepped serve failed"; return false; }
        }
        stamp(l, 17, grp);
        bool qdedup = false;   // STRATA_VERIFY_QDEDUP took effect: the experts' q8_1 image of xm is already made
        {
            const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err), *wsg = need(v, "ffn_gate_shexp.weight", err),
                            *wsu = need(v, "ffn_up_shexp.weight", err), *wsd = need(v, "ffn_down_shexp.weight", err);
            if (!wgi || !wsg || !wsu || !wsd) return false;
            if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) || !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
                !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
                return false;
            NativeSharedWeights nsw;
            nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
            nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
            nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
            nsw.q8_1 = sh_fork ? sh_xq_ : xq_;
            // STRATA_VERIFY_QDEDUP=1 (not with the forked shared-expert stream, which quantizes into its own buffer): the
            // experts' q8_1 image of xm is made first and the shared expert's gate/up read it (quantize_q8_1_rows and
            // native_quantize_q8_1 write the same bytes)
            qdedup = g_qdedup() && !sh_fork && strata::kernels::cpu::expert_layout().native;
            if (qdedup && !q8_ffn) quantize_q8_1_rows(xm, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
            if (!shared_expert_native_bf16_enabled()) {
                if (dec_batch) f32_to_bf16_bulk(mixed_ + tb * N, sh_bf16_ + tb * N, (int64_t) n * N, sh_stream);   // contiguous rows
                else for (int t = tb; t < te; ++t) f32_to_bf16_bulk(mixed_ + t * N, sh_bf16_ + t * N, N, sh_stream);
            }
            try {
                shared_expert_multi(n, xm, sh_bf16_ + tb * N, nsw, (const uint16_t*) wgi->data, sh_gate_ + (size_t) tb * g.n_ff,
                                    sh_up_ + (size_t) tb * g.n_ff, sh_g_ + tb, shared_ + tb * N, N, g.n_ff, sh_stream,
                                    qdedup ? (const void*) (nat_xq_ + (size_t) tb * (N / 32) * 36) : nullptr,
                                    (sg_ready ? 1 : 0) | (lfuse_on(n) && g_lfuse_pair() ? 2 : 0));
                sg_gated_[grp] = sg_ready;
            } catch (const std::exception& e) {
                err = std::string("verify shared expert: ") + e.what();
                return false;
            }
            if (sh_fork) dpct::sync_barrier(ev_join_, sh_cs_);
        }
        if (strata::kernels::cpu::expert_layout().native) {
            // APR: Hadamard-folded gate/up take H (s * x); xm itself stays as it is (the shared expert above and the
            // doorbell copy for the CPU pool, which rotates its own in native_quant_act). The q8_1 image of xm that
            // STRATA_QFUSE / STRATA_VERIFY_QDEDUP may already have written is of the unrotated input, so a folded layer
            // always rewrites it from the rotated rows (same stream, after the shared expert has read it).
            const auto& f = strata::kernels::cpu::expert_layout().fmt[(size_t) l];
            if (f.had_gu) {   // nat_xr_ exists: init carved it because this layer is folded (had_gu_any)
                hadamard_rows(xm, nat_xr_ + (size_t) tb * N, n, N, f.had_block, had_sx_, cs);
                quantize_q8_1_rows(nat_xr_ + (size_t) tb * N, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
            } else if (!qdedup && !q8_ffn) {
                quantize_q8_1_rows(xm, n, N, nat_xq_ + (size_t) tb * (N / 32) * 36, cs);
            }
        } else
            quantize_q8_0_scaled(xm, hit_xq_ + (size_t) tb * (N / 32) * 34, hit_xs_ + (size_t) tb * (N / 32), (int64_t) n * N, cs);
        stamp(l, 18, grp);
        return true;
    };

    // ---------------------------------------------------------------- post(l, group): experts, combine
    static const bool fuse_head_gr_env = [] {
        const char* e = std::getenv("STRATA_FUSE_HEAD_GR");
        return e && e[0] == '1';
    }();
    const bool fuse_head_gr = fuse_head_gr_env && (le_ == g.n_layers) && (head_ != nullptr && head_->loaded()) &&
                              !cvec().covers(g.n_layers - 1);
    auto post = [&](int64_t l, int grp) -> bool {
        const int tb = tb_[grp], te = te_[grp], n = te - tb;
        const uint32_t ring = (uint32_t) ((l - lb_) * G + grp + 1);
        const int64_t cap = (int64_t) n * K, capx = (int64_t) max_t_ * K;
        int32_t* pl = plan_ + (size_t) grp * (size_t) (plan_i32_ + 16);
        const int32_t* p_counts = pl;
        const int32_t* p_start = pl + 4;
        const int32_t* p_dst = p_start + capx + 1;
        const int32_t* p_tok = p_dst + capx;
        const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const unsigned long long* p_ptr = (const unsigned long long*) (pl + ptr_off);
        const unsigned long long* p_ptr2 = p_ptr + capx;
        const int32_t* p_start2 = pl + ptr_off + 4 * capx;
        float* hit_out = hit_out_ + (size_t) tb * K * N;
        float* parts_out = parts_ + (size_t) tb * K * N;
        const auto& lay = strata::kernels::cpu::expert_layout();
        // plan v0.3 P6: the VRAM groups now; the PCIe groups once the copy engine has landed them in staging.
        // `gy`: the native launch's groups side by side (0: cap, one block row per possible group).
        auto grouped = [&](const unsigned long long* gp, const int32_t* gs, const int32_t* gn, int64_t gy, float* dst_buf) {
            if (lay.native) {
                // the layer's GGUF formats (i-quant gate/up, Q2_0 / IQ4_NL down)
                const auto& f = lay.fmt[(size_t) l];
                NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
                if (f.had_d) { L.had_block = f.had_block; L.d_signs = had_sh_; }   // APR: h is rotated in there
                native_expert_grouped(L, gp, gs, gn, p_dst, p_tok, cap, cap,
                                      nat_xq_ + (size_t) tb * (N / 32) * 36, hit_scratch_, dst_buf, cs, gy);
            } else {
                moe_grouped_s2(gp, gs, gn, p_dst, p_tok, cap, cap, hit_xq_ + (size_t) tb * (N / 32) * 34,
                               hit_xs_ + (size_t) tb * (N / 32), hit_scratch_, dst_buf, cs);
            }
        };
        if (ar_on()) {
            stamp(l, 19, grp);
            grouped(p_ptr, p_start, p_counts, 0, parts_out);
            stamp(l, 20, grp);
        } else {
            if (device_plan_) {   // E-6: skipped when the device planned this group (all its experts resident)
                if (!lean) wait_flag_ge_or(m_flagA_, ring, skip_ + grp, cs);
                if (!lean) copy_i32_from_mapped_unless(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, skip_ + grp, ring, cs);
            } else {
                wait_flag_ge(m_flagA_, ring, cs);                  // the pool published this group's GPU plan
                copy_i32_from_mapped(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, cs);
            }
            stamp(l, 19, grp);
            grouped(p_ptr, p_start, p_counts, 0, hit_out);
            stamp(l, 20, grp);
            // stepped mode: the plan copy + VRAM groups end a segment, so they run while the CPU computes its rows
            if (step_hook2_ && !step_hook2_(l, grp)) { if (err.empty()) err = "verify: stepped split failed"; return false; }
            if (device_plan_) { if (!lean) wait_flag_ge_or(m_flagB_, ring, skip_ + grp, cs); }
            else wait_flag_ge(m_flagB_, ring, cs);                 // the PCIe share is in staging (DMA) or mapped
            if (!lean && sink_.pcie_mode == 2) {                 // stage it with a copy kernel, then point at staging
                const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
                uint8_t* stage = staging_ + (size_t) (grp * per) * lay.max_blob;
                fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, cs);
                rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), cs);
            }
            stamp(l, 21, grp);
            // the PCIe share is pcie_frac of the misses: a few groups when the cache is cold, usually none (always none at
            // pcie_frac 0), so its launch is kPcieGroupRows block rows striding over the groups, not cap of them
            if (!lean) grouped(p_ptr2, p_start2, p_counts + 2, kPcieGroupRows, hit_out);   // (lean: the device plan leaves it empty)
            stamp(l, 22, grp);
            if (device_plan_) {   // no CPU share when the device planned the group: its rows are zeros
                if (!lean) wait_flag_ge_or(m_flag_, ring, skip_ + grp, cs);
                copy_or_zero_from_mapped(parts_out, m_ymiss_ + (size_t) tb * K * N, (long long) n * K * N,
                                         skip_ + grp, ring, cs);
            } else {
                wait_flag_ge(m_flag_, ring, cs);               // the CPU's share is in the mapped rows
                stamp(l, 23, grp);
                if (remote_opt_)   // #578: the helper's rows come back reduced; skip them as well
                    remote_opt_->copy_rows(parts_out, m_ymiss_ + (size_t) tb * K * N, tb, n, p_dst, p_counts + 1, cs);
                else if (dec_batch)   // only the CPU rows cross PCIe (p_dst[0, counts[1]) = the GPU's own rows)
                    copy_rows_from_mapped(parts_out, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K, N,
                                          p_dst, p_counts + 1, cs);
                else
                    copy_from_mapped(parts_out, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K * N, cs);
            }
            moe_hit_add(parts_out, hit_out, p_dst, p_counts + 1, cap, N, cs);
        }
        // Same condition as `sh_fork` above, which is per group and out of scope here: wait only when the fork
        // actually ran (a wait on an event never recorded is a no-op, but saying it outright reads better).
        if (sh_stream_on() && !prof_on_ &&
            sh_cs_ != &dpct::get_in_order_queue() && ev_fork_ != nullptr &&
            ev_join_ != nullptr) {
            cs->ext_oneapi_submit_barrier({*ev_join_});
        }
        if (sg_gated_[grp]) {   // STRATA_LFUSE: the shared expert's gate was not applied: the combine applies it (all rows are hits)
            try {
                native_moe_combine_multi_hits_gated(parts_out, w_ + tb * K, shared_ + tb * N, sh_g_ + tb, bo_ + tb * N, N, K, n, cs);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else
        if (dec_batch && n > 1 && native_moe_combine_enabled()) {   // one launch for the window's rows
            try {
                native_moe_combine_multi(parts_ + (size_t) tb * K * N, w_ + tb * K, shared_ + tb * N, bo_ + tb * N, N, K, n, cs);
            } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
        } else
        for (int t = tb; t < te; ++t) {
            MoEBuffers mb = ss.moe;
            mb.weights = w_ + t * K; mb.shared = shared_ + t * N;
            if (!moe_combine_parts(g, l, K, mb, parts_ + (size_t) t * K * N, bo_ + t * N, cs, err)) return false;
        }
        if (remote_opt_) remote_opt_->combine(bo_ + tb * N, y_dummy_ + tb * N, tb, n,
                              device_plan_ ? skip_ + grp : nullptr, ring, cs);
        stamp(l, 24, grp);
        if (l == g.n_layers - 1) {
            if (!fuse_head_gr) {
                if (dec_batch && !g_no_multi_gr) gr_write_multi(Rt(tb), bo_ + tb * N, inj2_ + tb * HC, gs, Rt(tb), n, cs);
                else for (int t = tb; t < te; ++t) gr_write(Rt(t), bo_ + t * N, inj2_ + t * HC, gs, Rt(t), cs);
                if (cvec().covers(l)) cvec_apply(Rt(tb), l, n, HC * N, nullptr, 0, nullptr, 0, false, cs);
            }
        } else if (cvec().covers(l)) {
            cvec_apply(Rt(tb), l, n, HC * N, bo_ + tb * N, N, inj2_ + tb * HC, HC, true, cs);
        }
        return true;
    };

    for (int grp = 0; grp < G; ++grp)
        if (!pre(lb_, grp)) return false;
    for (int64_t l = lb_; l < le_; ++l)
        for (int grp = 0; grp < G; ++grp) {
            if (!post(l, grp)) return false;
            if (dbgR_ && grp == 0) {   // recorded into the window graph as memcpy nodes
                cs->memcpy(dbgR_ + (size_t) l * N, Rt(0), (size_t) N * 4);
                cs->memcpy(dbgM_ + (size_t) l * N, mixed_, (size_t) N * 4);
            }
            if (l + 1 < le_ && !pre(l + 1, grp)) return false;
        }
    if (le_ < g.n_layers) {   // a layer split's earlier stage: hand the residual on, no head
        float* hout = hand_out_ + (size_t) hrow0 * HB;
        copy_from_mapped(hout, R_, (int64_t) T * HC * N, cs);
        copy_from_mapped(hout + (size_t) T * HC * N, bo_, (int64_t) T * N, cs);
        copy_from_mapped(hout + (size_t) T * (HC + 1) * N, inj2_, (int64_t) T * HC, cs);
        return true;
    }

    // ---- the head, T columns, and the argmax of each
    stamp(g.n_layers, 0, 0);
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "verify: an output_hc_* weight is missing"; return false; }
        // S25 (STRATA_HC_Q8=1): the final mixer from the GGUF's Q8_0 projections too, through the same multi read
        // (no pending write, no inject)
        const bool mix_q8 = hd->hc_q8 != nullptr && hu->hc_q8 != nullptr && head_ != nullptr && head_->loaded();
        if (mix_q8) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                FusedGrArgs& a = fa[t];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = false;
                a.w_norm = (const float*) hn->data; a.w_down = (const uint16_t*) hd->data;
                a.w_up = (const uint16_t*) hu->data; a.w_inject = nullptr;
                a.q8_down = (const uint8_t*) hd->hc_q8; a.q8_up = (const uint8_t*) hu->hc_q8;
                a.eps = EPS; a.lo = lo_ + t * g.hc_lr; a.rs = rs_ + t * HC;
                a.mixed = head_mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        else if (fuse_head_gr) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                FusedGrArgs& a = fa[t];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = true;
                a.bo_prev = bo_ + t * N; a.inj_prev = inj2_ + t * HC;
                a.w_norm = (const float*) hn->data;
                a.w_down = (const uint16_t*) hd->data;
                a.w_up = (const uint16_t*) hu->data;
                a.w_inject = nullptr;
                a.eps = EPS;
                a.lo = lo_ + t * g.hc_lr;
                a.rs = rs_ + t * HC;
                a.inject_out = head_inj_;
                a.mixed = head_mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        } else if (head_mix_multi_enabled() && head_ != nullptr && head_->loaded()) {
            // the final mixer, the window's tokens in one read (fused_gr_read_multi without the pending write: the
            // last layer's write was done above; its sums are gr_read's)
            if (hn->kind != WeightKind::F32 || hd->kind != WeightKind::Bf16InF32 || hu->kind != WeightKind::Bf16InF32) {
                err = "verify: the output_hc_* weights have the wrong engine forms";
                return false;
            }
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = Rt(t); fa[t].R_out = Rt(t); fa[t].apply = false;
                fa[t].w_norm = (const float*) hn->data; fa[t].w_down = (const uint16_t*) hd->data;
                fa[t].w_up = (const uint16_t*) hu->data; fa[t].eps = EPS;
                fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC; fa[t].mixed = head_mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        } else {
            for (int t = 0; t < T; ++t) {
                BlockBuffers bb = ss.block;
                bb.R = Rt(t);
                bb.mixed = head_mixed_ + t * N;
                if (head_ != nullptr && head_->loaded()) {
                    if (!lm_head_mix(wt, g, bb, cs, err)) return false;
                } else if (!lm_head(wt, g, bb, head_logits_ + (size_t) t * n_vocab_, cs, err)) {
                    return false;
                }
            }
        }
        if (head_ != nullptr && head_->loaded()) {
            try {
                native_quantize_q8_1(head_mixed_, xq_, (int) N, T, cs);
                native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, (int) N, (int) n_vocab_, T, cs);
            } catch (const std::exception& e) {
                err = std::string("verify head: ") + e.what();
                return false;
            }
        }
        // Greedy, the default, is recorded here as before (no extra launch or sync per window). A request that
        // samples or penalizes is sampled again host-side after the replay (run()) with its own parameters and a
        // fresh draw counter: a captured sampler would bake them in and replay the same draws forever.
        if (argmax_rows_wanted()) {   // sm_80 to sm_89: the pick over many blocks a row (bitwise the one-block kernel's)
            argmax_rows(head_logits_, T, (int) n_vocab_, arg_scratch_, m_out_, cs);
        } else {
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            sample_tokens(head_logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
        }
    }
    stamp(g.n_layers, 1, 0);
    return true;
}

std::string Verifier::profile_report() {
    if (!prof_on_ || prof_windows_ == 0) return std::string();
    static const char* names[kProfPer] = {"-", "hc-read0", "q8+qkv/q-idx gemv", "conv", "ab", "z", "rec", "q8+kv-idx",
                                          "k/v+norm-rope", "kv+idx append", "q+q-idx", "scores+topk", "kv-resolve",
                                          "attention", "gate", "", "out-proj", "hc-read1+router", "shared+quant",
                                          "waitA", "VRAM hits", "waitB", "PCIe grp", "waitCPU", "copy+combine",
                                          "(gap)", "head", "  hc0 norm", "  hc0 down", "  hc0 up", "", "", ""};
    std::string out;
    char b[80];
    double total = 0;
    for (int k = 0; k < 2; ++k) {
        out += k == 0 ? " GDN layers:" : " | QSA layers:";
        for (int i = 0; i < kProfPer; ++i) {
            if (prof_sum_[k][i] <= 0) continue;
            total += prof_sum_[k][i];
            std::snprintf(b, sizeof b, " %s %.2f", names[i], prof_sum_[k][i] / 1e6 / (double) prof_windows_);
            out += b;
        }
    }
    std::snprintf(b, sizeof b, " | total %.2f ms/window over %lld windows", total / 1e6 / (double) prof_windows_, (long long) prof_windows_);
    out += b;
    for (auto& r : prof_sum_) for (double& d : r) d = 0;
    prof_windows_ = 0;
    return out;
}

// #871: is the zero-doorbell graph right for the next window?  Only while every expert of [lb_, le_) is in VRAM now
// (the host table is the one the device copy was uploaded from; a loan, a VRAM shrink or a swap in flight has
// marked some -1).  Not: the doorbell graph, which asks the pool for what the device does not hold.
void Verifier::refresh_ar() {
    if (!all_resident_ || h_res_ == nullptr || g_ == nullptr) return;
    const int64_t ne = g_->n_expert;
    const int32_t* r = h_res_ + lb_ * ne;
    const int64_t n = (le_ < 0 ? g_->n_layers : le_) * ne - lb_ * ne;
    int32_t lo = 0;
    for (int64_t i = 0; i < n; ++i) lo = std::min(lo, r[i]);
    const bool off = lo < 0;
    if (off != ar_off_) {
        ar_off_ = off;
        std::fprintf(stderr, "strata verify: layers [%lld, %lld) %s\n", (long long) lb_, (long long) (le_ < 0 ? g_->n_layers : le_),
                     off ? "have experts out of VRAM (a prompt loan, a shrink or a swap): the doorbell graph runs the windows"
                         : "are 100% VRAM resident again: the zero-doorbell graph");
    }
}

bool Verifier::capture_segments(int T, std::string& err) try {
    if (!seg_[T].empty()) return true;
    // every segment's payload is read after the segment drained: its publishes store plainly (elementwise.dp.cpp)
    struct PlainPayload {
        PlainPayload() { strata::kernels::doorbell_plain_payload(!window_host_uncached()); }
        ~PlainPayload() { strata::kernels::doorbell_plain_payload(false); }
    } plain_payload;
    namespace sx = sycl::ext::oneapi::experimental;
    auto close_segment = [&]() -> bool {
        dpct::experimental::command_graph_ptr gph = nullptr;
        dpct::experimental::end_recording(cs_, &gph);
        if (gph == nullptr) { err = "verify: segment capture produced no graph"; return false; }
        seg_[T].push_back(new sx::command_graph<sx::graph_state::executable>(gph->finalize()));
        delete gph;
        return true;
    };
    auto boundary = [&](int64_t l, int grp, char kind) -> bool {
        if (!close_segment()) return false;
        seg_at_[T].push_back({l, grp});
        seg_kind_[T].push_back(kind);
        dpct::experimental::begin_recording(cs_);
        return true;
    };
    dpct::experimental::begin_recording(cs_);
    step_hook_ = [&](int64_t l, int grp) { return boundary(l, grp, 1); };
    static const bool split_on = [] { const char* e = std::getenv("STRATA_VERIFY_SPLIT"); return !(e && e[0] == '0'); }();
    if (split_on) step_hook2_ = [&](int64_t l, int grp) { return boundary(l, grp, 0); };
    std::string rerr;
    const bool ok = record_window(T, cs_, rerr);
    step_hook_ = nullptr;
    step_hook2_ = nullptr;
    const bool closed = close_segment();
    if (!ok || !closed) {
        for (auto* e : seg_[T]) delete e;
        seg_[T].clear(); seg_at_[T].clear(); seg_kind_[T].clear();
        if (err.empty()) err = "verify: segment capture: " + rerr;
        return false;
    }
    if (std::getenv("STRATA_VERIFY_NODES") != nullptr)
        std::fprintf(stderr, "strata verify: the %d-token window as %zu graph segments\n", T, seg_[T].size());
    return true;
}
catch (sycl::exception const& exc) {
    err = std::string("verify: segment capture: ") + exc.what();
    return false;
}

bool Verifier::capture(int T, std::string &err) try {
    // stepped mode: the doorbell window as segments; the all-resident (zero-doorbell) graph has no serves and is
    // captured whole below, its one host flag raised before launch (run)
    if (g_stepped && !ar_on() && std::getenv("STRATA_VERIFY_EAGER") == nullptr) return capture_segments(T, err);
    dpct::experimental::command_graph_exec_ptr &exec_t =
        ar_off_ ? exec_nr_[T] : exec_[T];
    if (exec_t != nullptr) return true;
    if (std::getenv("STRATA_VERIFY_EAGER") != nullptr) return true;   // SYCL port: no graph, run() replays the body
    if (DPCT_CHECK_ERROR(dpct::experimental::begin_recording(cs_)) != 0) {
        err = "verify: begin capture failed";
        return false;
    }
    std::string rerr;
    const bool ok = record_window(T, cs_, rerr);
    dpct::experimental::command_graph_ptr graph = nullptr;
    const dpct::err0 ce =
        DPCT_CHECK_ERROR(dpct::experimental::end_recording(cs_, &graph));
    if (!ok) {
        if (graph) delete (graph);
        err = rerr;
        return false;
    }
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (ce != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use error
        codes. Please replace the "get_error_string_dummy(...)" with a real
        error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        err = std::string("verify: end capture: ") +
              dpct::get_error_string_dummy(ce);
        return false;
    }
    if (std::getenv("STRATA_VERIFY_NODES") != nullptr) {   // what the window graph holds
        size_t nn = 0;
        dpct::experimental::get_nodes(graph, nullptr, &nn);
        std::vector<dpct::experimental::node_ptr> nodes(nn);
        dpct::experimental::get_nodes(graph, nodes.data(), &nn);
        std::map<std::string, int> kinds;
        for (dpct::experimental::node_ptr nd : nodes) {
            sycl::ext::oneapi::experimental::node_type ty;
            ty = nd->get_type();
            std::string name = "type" + std::to_string((int) ty);
            if (ty == sycl::ext::oneapi::experimental::node_type::kernel) {
                name = "kernel";   // SYCL: no kernel-name introspection on graph nodes
            } else if (ty == sycl::ext::oneapi::experimental::node_type::memcpy)
                name = "memcpy";
            else if (ty == sycl::ext::oneapi::experimental::node_type::memset)
                name = "memset";
            ++kinds[name];
        }
        std::vector<std::pair<int, std::string>> v;
        for (auto& [k2, c] : kinds) v.push_back({c, k2});
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "strata verify: the %d-token window graph has %zu nodes:", T, nn);
        for (size_t i = 0; i < v.size() && i < 40; ++i) std::fprintf(stderr, " %d x %.60s;", v[i].first, v[i].second.c_str());
        std::fprintf(stderr, "\n");
    }
    const dpct::err0 ie = DPCT_CHECK_ERROR(
        exec_t = new sycl::ext::oneapi::experimental::command_graph<
            sycl::ext::oneapi::experimental::graph_state::executable>(
            graph->finalize()));
    delete (graph);
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (ie != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use error
        codes. Please replace the "get_error_string_dummy(...)" with a real
        error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        err = std::string("verify: instantiate: ") +
              dpct::get_error_string_dummy(ie);
        return false;
    }
    /*
    DPCT1007: Migration of cudaGraphUpload is not supported.
    */
    const dpct::err0 ue = 0;   // no cudaGraphUpload on SYCL: a finalized command_graph is already resident
    const dpct::err0 us = DPCT_CHECK_ERROR(cs_->wait());
    std::fprintf(
        stderr,
        "strata verify: captured the %d-token window (upload %s, sync %s)\n", T,
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use error
        codes. Please replace the "get_error_string_dummy(...)" with a real
        error-handling function.
        */
        dpct::get_error_string_dummy(ue), dpct::get_error_string_dummy(us));
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool Verifier::capture_commit(std::string &err) try {
    if (commit_exec_ != nullptr) return true;
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ID = g.idx_key_dim, MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (DPCT_CHECK_ERROR(dpct::experimental::begin_recording(cs_)) != 0) {
        err = "verify: begin commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        for (int64_t l = lb_; l < le_ && ok; ++l) {
            const LayerView v(*wt_, l);
            if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                const float* qkv = qkv_L_ + (size_t) gdn_index * MT * C;
                gdn_conv_commit(conv, qkv, (int) C, commit_, cs_);
                gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C, (int) C, gate_L_ + (size_t) gdn_index * MT * HV,
                                    beta_L_ + (size_t) gdn_index * MT * HV, z_, (const float*) wnm->data, EPS, y_dummy_,
                                    (int) g.ssm_k_heads, (int) HV, (int) MT, commit_, cs_,
                                    (int) MT);   // S26: t_out_begin = MT - the replay's outputs (y_dummy_) are never read
                ++gdn_index;
            } else {
                const QsaState& st = ss.qsa_states[qsa_index];
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                copy_from_mapped(st.idx_tail, tail_snap_ + (size_t) qsa_index * TS, TS, cs_);
                const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                if (!g_no_batch_kv) {
                    native_qsa_indexer_append_steps(idx_raw_L_ + (size_t) qsa_index * MT * ID, commit_ + 2, 1, (int) MT, 0,
                                                    (const float*) wikn->data, EPS, ib, s, st.max_cells, rope_scaling(), cs_);
                } else {
                    for (int64_t t = 0; t < MT; ++t)
                        native_qsa_indexer_append(idx_raw_L_ + (size_t) (qsa_index * MT + t) * ID, commit_ + 2 + t, 0,
                                                  (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                  rope_scaling(), cs_);
                }
                ++qsa_index;
            }
        }
        if (ok && ss.ple.ready() && ple_stage()) copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify commit: ") + e.what();
        ok = false;
    }
    dpct::experimental::command_graph_ptr graph = nullptr;
    const dpct::err0 ce =
        DPCT_CHECK_ERROR(dpct::experimental::end_recording(cs_, &graph));
    if (!ok) {
        if (graph) delete (graph);
        return false;
    }
    if (ce != 0 ||
        DPCT_CHECK_ERROR(
            commit_exec_ = new sycl::ext::oneapi::experimental::command_graph<
                sycl::ext::oneapi::experimental::graph_state::executable>(
                graph->finalize())) != 0) {
        if (graph) delete (graph);
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use error
        codes. Please replace the "get_error_string_dummy(...)" with a real
        error-handling function.
        */
        err = std::string("verify: commit capture: ") +
              dpct::get_error_string_dummy(ce);
        return false;
    }
    delete (graph);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void Verifier::stage_inputs(int T, const int32_t* tokens, int64_t pos0) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const QsaShapes s = shapes_of(g);
    int32_t* const pk = h_pos_ + (size_t) max_t_ * g.n_head;
    int32_t* const pi = pk + (size_t) max_t_ * g.n_head_kv;
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos0 + t, s);
        const int32_t pos_t = (int32_t) (pos0 + t);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = pos_t;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = pos_t;
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = pos_t;
    }
    *(volatile uint32_t*) h_seq_ = 0;
    for (int _i = 1; _i <= 8; ++_i) ((volatile uint32_t*) h_seq_)[_i] = 0;   /* + the payload tags/checksums */
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = T;
    last_pos0_ = pos0;
    for (int t = 0; t < T; ++t) last_tokens_[t] = tokens[t];
    staged_ = true;
}

void Verifier::collect_profile() {
    if (std::getenv("STRATA_VERIFY_EAGER") == nullptr)   // eager (SYCL port): prof_h_ already holds the host clocks
        dpct::get_in_order_queue()
            .memcpy(prof_h_.data(), prof_, prof_h_.size() * 8)
            .wait();
    accumulate_profile(prof_h_.data());
}

void Verifier::accumulate_profile(const unsigned long long* stamps) {
    const ModelGeometry& g = *g_;
    const int64_t L = g.n_layers;
    auto at = [&](int64_t l, int i) { return stamps[(size_t) (l * kProfPer + i)]; };
    // D8: the derived columns below read stamps the hc-read kernels write themselves or the next
    // layer's first.  A slot no kernel stamped is 0 and its unsigned difference wrapped to ~1e19 ns -
    // which is why (gap)/head/hc0 printed ~1e14 ms per window.  A missing or out-of-order stamp now
    // contributes nothing.
    const auto gap = [](unsigned long long to, unsigned long long from) {
        return (from != 0 && to != 0 && to >= from) ? (double) (to - from) : 0.0;
    };
    // only this stage's layers [lb_, le_) are ever stamped (a layer split); the head only on the last stage
    for (int64_t l = lb_; l < le_; ++l) {
        const int kind = is_qsa_layer(g, l) ? 1 : 0;
        unsigned long long prev = at(l, 0);
        for (int i = 1; i <= 24; ++i) {
            const unsigned long long x = at(l, i);
            if (x == 0 || x < prev) continue;
            prof_sum_[kind][i] += (double) (x - prev);
            prev = x;
        }
        if (l + 1 < le_) prof_sum_[kind][25] += gap(at(l + 1, 0), at(l, 24));
        const double dn = gap(at(l, 27), at(l, 0)), dd = gap(at(l, 28), at(l, 27)), du = gap(at(l, 1), at(l, 28));
        if (dn > 0 && dd > 0 && du > 0) {   // the split exists: show it split, not twice
            prof_sum_[kind][27] += dn;      // hc-read0: norm
            prof_sum_[kind][28] += dd;      //           down
            prof_sum_[kind][29] += du;      //           up (through both halves, as before)
            prof_sum_[kind][1] -= gap(at(l, 1), at(l, 0));   // (hc-read0 shown split)
        }
    }
    if (le_ == L) prof_sum_[0][26] += gap(at(L, 1), at(L, 0));
    ++prof_windows_;
}

bool Verifier::run(int T, const int32_t *tokens, int64_t pos0, PoolMultiFn pool,
                   void *user, int32_t *out, std::string &err) try {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    last_batch_ = false;
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    if (pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells) { err = "verify: the window runs past the context"; return false; }
    refresh_ar();
    if (!capture(T, err) || !capture_commit(err)) return false;
    VDBG("captured; staging\n");
    const Clock::time_point t0 = Clock::now();
    if (!staged_) stage_inputs(T, tokens, pos0);
    staged_ = false;
    const bool do_ple = ss.ple.ready() && ple_stage();
    uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];
    if (do_ple) {
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, ple_rows + t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
            ss.ple.table->prefetch_rows(ple_rows + t * PLE_N_HEADS);
        }
    }
    if (trace_h_ != nullptr) std::memset(trace_h_, 0, trace_n_ * 8);   // #649: this window's breadcrumbs only
    std::atomic_thread_fence(std::memory_order_seq_cst);
    trace_ev("WINDOW", -1, -1, pos0 * 16 + T);
    ms_host += ms_since(t0);
    VDBG("staged; launching\n");
    static Clock::time_point t_prev_end;   // SYCL port timing: where does a round's wall clock go?
    const Clock::time_point t_launch = Clock::now();
    if (ar_on() && h_plan_err_ != nullptr) *(volatile uint32_t*) h_plan_err_ = 0;
    {   // SYCL port: on an Arc A750 the first window after other GPU work (a prefill, the previous request) never
        // started when it was launched behind that work still in flight on another queue (no GPU breadcrumb, 20 s
        // timeout, the engine dies). Letting the other queues drain before every window removes it (9 of 9 three-request runs pass; before, almost all
        // failed; the gap rule alone (=2) did not help, so it is the windows after each other, not only the first). The cost is
        // the overlap of a window with the previous commit's tail.  STRATA_WINDOW_SYNC=1 always drains, =0 never, =2 the gap rule; the default is 1 on an i915 card, 0 elsewhere.
        // #2194: the same never-starting first window on an Arc A770 under xe (main thread in this function's stepped
        // serve, queue::wait, no xe reset) - an Alchemist card, whatever its driver. Reproduced on the A750 with
        // STRATA_WINDOW_SYNC=0 (stepped, --resident-experts, IQ3_XXS): 1 of 6 runs hung there (same stack), against 0 of 11
        // with the drain. So the default drain is per GPU generation now: every Alchemist card (i915 or xe), as before
        // every i915 card; Battlemage and the others are unchanged (0).
        static const char* wv = std::getenv("STRATA_WINDOW_SYNC");
        static const int wmode = wv ? std::atoi(wv)
                                    : ((intel_i915_gpu() || strata::intel_gpu_gen(cs_->get_device()) == strata::IntelGpuGen::Alchemist) ? 1 : 0);
        const bool first_or_gap = t_prev_end.time_since_epoch().count() == 0 || (t_launch - t_prev_end) > std::chrono::milliseconds(20);
        if (wmode == 1 || (wmode == 2 && first_or_gap)) dpct::get_current_device().queues_wait_and_throw();
    }
    // STRATA_VERIFY_STEPPED (Arc A-series): the doorbell window runs as segments with the CPU experts served at the
    // queue boundaries between them (no mid-kernel host signalling); the all-resident graph gets its one flag (the PLE
    // rows are in place) before launch.
    const bool stepped_run = g_stepped && !ar_on() && std::getenv("STRATA_VERIFY_EAGER") == nullptr && !seg_[T].empty();
    const bool stepped_ar = g_stepped && ar_on() && !(g_test_stall > 0 && windows + 1 == g_test_stall);
    if (stepped_ar) {
        if (do_ple) {
            const Clock::time_point tp = Clock::now();
            if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            ms_host += ms_since(tp);
        }
        *(volatile uint32_t*) h_flag_ = 1;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    dpct::err0 le = 0;
    if (stepped_run) {
        if (remote_opt_) { err = "verify: STRATA_VERIFY_STEPPED does not serve --remote-expert-opt"; return false; }
        const int Gs = groups_[T] > 0 ? groups_[T] : 1;
        double st_wait = 0, st_pool = 0;
        std::vector<char> submitted(seg_[T].size(), 0);
        // one serve: segment i drained (the ring's payload is in mapped memory); run the pool - on a helper thread
        // when `nxt` is a split segment, which is submitted as soon as the pool publishes its plan (flag A) - then
        // raise flags B and M before the following segment is submitted
        auto serve = [&](int64_t l, int grp, size_t nxt) -> bool {
            const Clock::time_point a = Clock::now();
            cs_->wait();
            const Clock::time_point b = Clock::now();
            const uint32_t want = (uint32_t) ((l - lb_) * Gs + grp + 1);
            if (*(volatile uint32_t*) h_seq_ < want) {
                err = "verify (stepped): layer " + std::to_string(l) + " did not ring after its queue drained (seq " +
                      std::to_string(*(volatile uint32_t*) h_seq_) + ", want " + std::to_string(want) + ")";
                return false;
            }
            cur_layer_ = want - 1;
            set_plan_slot(grp);
            const int tb = grp == 0 ? 0 : (T + 1) / 2, n = (Gs == 2 ? (grp == 0 ? (T + 1) / 2 : T) : T) - tb;
            auto run_pool = [&]() {
                if (pool != nullptr)
                    pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                         h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
            };
            auto empty_plan = [&]() {   // the pool did not publish a plan: an empty one
                sink_.counts[0] = 0; sink_.counts[1] = 0; sink_.counts[2] = 0;
                sink_.start[0] = 0; sink_.start2[0] = 0;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                *(volatile uint32_t*) h_flagA_ = want;
            };
            // segment nxt ends at a split point: it is the plan copy + VRAM groups, whose only host wait is flag A
            const bool split = nxt < seg_kind_[T].size() && seg_kind_[T][nxt] == 0;
            static const bool pipe = [] { const char* e = std::getenv("STRATA_VERIFY_PIPE"); return !(e && e[0] == '0'); }();
            if (pipe && split) {
                std::atomic<bool> done{false};
                std::string herr;
                try {
                    const int helper_cpu = pipe_helper_cpu();
                    std::thread th([&, helper_cpu]() {
#if defined(__linux__)
                        if (helper_cpu >= 0) {
                            cpu_set_t s;
                            CPU_ZERO(&s);
                            CPU_SET(helper_cpu, &s);
                            (void) pthread_setaffinity_np(pthread_self(), sizeof s, &s);
                        }
#endif
                        try { run_pool(); } catch (const std::exception& e) { herr = e.what(); }
                        done.store(true, std::memory_order_release);
                    });
                    while (!done.load(std::memory_order_acquire) && *(volatile uint32_t*) h_flagA_ != want)
                        std::this_thread::yield();
                    if (done.load(std::memory_order_acquire) && *(volatile uint32_t*) h_flagA_ != want) empty_plan();
                    cs_->ext_oneapi_graph(*seg_[T][nxt]);   // flag A is up: its waits pass
                    submitted[nxt] = 1;
                    th.join();
                } catch (const std::exception& e) { err = std::string("verify: pipelined serve: ") + e.what(); return false; }
                if (!herr.empty()) { err = "verify: pool: " + herr; return false; }
            } else {
                run_pool();
                std::atomic_thread_fence(std::memory_order_seq_cst);
                if (*(volatile uint32_t*) h_flagA_ != want) empty_plan();
            }
            copy_->wait();   // fetch_dma's copies and its flag-B host task
            std::atomic_thread_fence(std::memory_order_seq_cst);
            raise_flag(h_flagB_, want);
            if (want == 1 && do_ple) {   // layer 1's pre copies h_ple_ after layer 0's wait on the M flag
                const Clock::time_point tp = Clock::now();
                if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
                ms_host += ms_since(tp);
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            *(volatile uint32_t*) h_flag_ = want;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            st_wait += std::chrono::duration<double, std::milli>(b - a).count();
            st_pool += ms_since(b);
            return true;
        };
        try {
            for (size_t i = 0; i < seg_[T].size() && le == 0; ++i) {
                const bool has_entry = i < seg_at_[T].size();
                if (!submitted[i]) cs_->ext_oneapi_graph(*seg_[T][i]);   // (a split segment: its serve submitted it)
                if (has_entry && seg_kind_[T][i] == 1 && !serve(seg_at_[T][i].first, seg_at_[T][i].second, i + 1)) {
                    if (err.empty()) err = "verify: stepped serve failed";
                    le = 1;
                }
            }
        } catch (const std::exception& e) { err = std::string("verify (stepped): ") + e.what(); le = 1; }
        ms_wait += st_wait;
        ms_pool += st_pool;
        if (le != 0) return false;
    } else {
        le = (std::getenv("STRATA_VERIFY_EAGER") != nullptr)
                 ? (record_window(T, cs_, err) ? 0 : 1)   // SYCL port: eager replay of the window body
                 : DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(*(ar_off_ ? exec_nr_[T] : exec_[T])));
    }
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    trace_ev("LAUNCHED", -1, -1, (int64_t) le);
    if (le != 0) {
        err =
            std::string("verify: launch: ") + dpct::get_error_string_dummy(le);
        return false;
    }
    (void)DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty()));
    VDBG("launched\n");
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    // SYCL port: with every routed expert resident the GPU takes its own plan (`device_plan_`) and skips every
    // host flag, so the host has nothing to do per layer - and on this platform a kernel's writes to host-mapped
    // memory are not reliably visible while the graph runs (measured: the ring is seen late or not at all),
    // so waiting on them per layer fails. STRATA_VERIFY_NO_HOST=1 waits for the whole window instead. Only for
    // an all-resident cache: a missed expert would leave the GPU waiting for a plan that never comes.
    static const bool no_host = [] { const char* v = std::getenv("STRATA_VERIFY_NO_HOST"); return v && *v && std::strcmp(v, "0") != 0; }();
    const int64_t steps = (le_ - lb_) * G;
    const bool test_stall = g_test_stall > 0 && windows + 1 == g_test_stall;   // #267 test hook (off: false)
    if (stepped_run || stepped_ar) {
        // the stepped window already served every layer (or the all-resident one had its flag before launch)
    } else if (ar_on() && !test_stall) {
        if (do_ple) {
            const Clock::time_point tp = Clock::now();
            if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            *flag = 1;
            ms_host += ms_since(tp);
        } else {
            *flag = 1;
        }
    } else
    for (int64_t k = 0; !no_host && k < steps; ++k) {
        const int64_t l = lb_ + k / G;
        const int grp = (int) (k % G);
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify window: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                // SYCL port fix: cudaStreamQuery's answer (0 finished, 1 still running) was lost in the migration
                // (DPCT_CHECK_ERROR of ext_oneapi_empty() is always 0), so ANY wait over 2 ms was reported as
                // "graph finished". Layers whose GPU time exceeds 2 ms (host-mirrored experts read over PCIe) failed that way.
                const dpct::err0 q = cs_->ext_oneapi_empty() ? 0 : 1;
                if (q != 1 && *seq < want) {
                    trace_ev("NEVER-RANG", k, l, (int64_t) q);
                    trace_dump(stderr);
                    err = "verify: layer " + std::to_string(l) +
                          " never rang (" +
                          /*
                          DPCT1009: SYCL reports errors using exceptions and
                          does not use error codes. Please replace the
                          "get_error_string_dummy(...)" with a real
                          error-handling function.
                          */
                          (q == 0
                               ? std::string("graph finished")
                               : std::string(dpct::get_error_string_dummy(q))) +
                          ")";
                    return false;
                }
            }
            if (now - a > strata_ring_timeout()) {
                // #267: the caller ends the engine; no spin kernel may outlive it
                trace_ev("TIMEOUT", k, l, (int64_t) !cs_->ext_oneapi_empty());   // SYCL port: 1 = still running
                if (g_trace) {
                    diag(stderr);   // the words and the breadcrumbs before the release ...
                    const bool drained = release_gpu_waits(5000);
                    trace_dump(stderr);   // ... and after it: did the GPU move once its waits were raised?
                    err = "verify: timed out at layer " + std::to_string(l) + released_note(drained);
                    return false;
                }
                err = "verify: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
                return false;
            }
        }
        const Clock::time_point b = Clock::now();
        if (g_trace) trace_ev("RANG", k, l, (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
        VDBG("layer %lld rang\n", (long long) l);
        cur_layer_ = want - 1;
        set_plan_slot(grp);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        // the ring can arrive before its payload (elementwise.hpp): the pool must read a whole one
        if (!strata::kernels::doorbell_wait_payload(h_seq_, h_x_ + (size_t) tb * g.n_embd, (int64_t) n * g.n_embd,
                                                    h_ids_ + (size_t) tb * ss.k, h_w_ + (size_t) tb * ss.k,
                                                    (int64_t) n * ss.k, want)) {
            err = "verify: layer " + std::to_string(l) + " rang but its payload never arrived whole";
            return false;
        }
        progress_at("verify window: the CPU experts of layer", l);
        if (remote_opt_) remote_opt_->begin(h_w_ + (size_t) tb * ss.k, tb, n);
        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        if (remote_opt_) remote_opt_->end();
        VDBG("layer %lld served\n", (long long) l);
        if (g_trace) trace_ev(*(volatile uint32_t*) h_flagA_ == want ? "SERVED" : "SERVED-NO-PLAN-YET", k, l,
                              (int64_t) ms_since(b));   // aux: ms the CPU experts took
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        // Layer 1's pre(1, 0) copies h_ple_ -> ple_ after Layer 0's wait_flag_ge(m_flag_, 1).
        // By collecting PLE here at k == 0 (after publishing flagA/flagB for Layer 0 so the GPU can run
        // Layer 0's VRAM experts, and before raising *flag = 1), the NVMe PLE reads overlap with both
        // MtpDrafter::draft and Layer 0's attention + router + expert execution!
        if (k == 0 && do_ple) {
            const Clock::time_point tp = Clock::now();
            if (!ss.ple.table->gather_batch(ple_rows, (size_t) T, h_ple_, err)) return false;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            ms_host += ms_since(tp);
        }
        if (!(test_stall && k + 1 == steps)) *flag = want;
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
    }
    // (#646 staged the next stage's inputs here; 0.1.39b keeps the layer split's order: each stage stages its own)
    progress_at("verify window: waiting for the GPU to finish the window (flags A/B/M raised)", (int64_t) T);
    // #267: a window the GPU never finishes (a spin kernel that never sees its flag) holds the host here; the stall
    // watchdog then releases every verifier's GPU waits (release_live_verifiers) before it ends the engine, so no
    // spin kernel outlives the process.  The wait itself stays a blocking sync (a polling wait cost decode upstream).
    trace_ev("SYNC", -1, -1, 0);
    const dpct::err0 se = DPCT_CHECK_ERROR(cs_->wait());
    trace_ev("SYNCED", -1, -1, (int64_t) se);
    const Clock::time_point t_done = Clock::now();
    if (std::getenv("STRATA_VERIFY_DEBUG") != nullptr) {
        std::fprintf(stderr, "verify dbg: T=%d window: since previous window %.1f ms, staging %.1f ms, gpu %.1f ms\n", T,
                     t_prev_end.time_since_epoch().count() ? std::chrono::duration<double, std::milli>(t0 - t_prev_end).count() : 0.0,
                     std::chrono::duration<double, std::milli>(t_launch - t0).count(),
                     std::chrono::duration<double, std::milli>(t_done - t_launch).count());
    }
    t_prev_end = t_done;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (se != 0) {
        err = std::string("verify: ") + dpct::get_error_string_dummy(se);
        return false;
    }
    if (no_host && std::getenv("STRATA_VERIFY_DEBUG") != nullptr) {   // SYCL port: what the window left behind
        const int Gd = groups_[T] > 0 ? groups_[T] : 1;
        std::vector<uint32_t> sk((size_t) Gd, 0);
        std::vector<int32_t> id0((size_t) ss.k, 0);
        std::vector<int32_t> res0((size_t) 16, 0);
        cs_->memcpy(sk.data(), skip_, sk.size() * 4).wait();
        cs_->memcpy(id0.data(), ids_, id0.size() * 4).wait();
        cs_->memcpy(res0.data(), hits_.d_res, res0.size() * 4).wait();
        std::fprintf(stderr, "verify dbg: window T=%d: host seq=%u (expected %lld), skip[]=", T, *(volatile uint32_t*) h_seq_,
                     (long long) ((le_ - lb_) * Gd));
        for (int i = 0; i < Gd; ++i) std::fprintf(stderr, " %u", sk[(size_t) i]);
        std::vector<float> w0((size_t) ss.k, 0.f);
        cs_->memcpy(w0.data(), w_, w0.size() * 4).wait();
        std::fprintf(stderr, "; layer-0 weights:");
        for (size_t i = 0; i < w0.size(); ++i) std::fprintf(stderr, " %.3g", w0[i]);
        std::fprintf(stderr, "; layer-0 ids:");
        for (size_t i = 0; i < id0.size(); ++i) std::fprintf(stderr, " %d", id0[i]);
        std::fprintf(stderr, "; d_res[0..15]:");
        for (size_t i = 0; i < res0.size(); ++i) std::fprintf(stderr, " %d", res0[i]);
        std::fprintf(stderr, "; out[0..3]: %d %d %d %d\n", ((volatile int32_t*) h_out_)[0], ((volatile int32_t*) h_out_)[1],
                     ((volatile int32_t*) h_out_)[2], ((volatile int32_t*) h_out_)[3]);
        auto stat = [&](const char* what, const float* dev, size_t n) {   // token 0's vector: where do the zeros begin?
            std::vector<float> v(n, 0.f);
            cs_->memcpy(v.data(), dev, n * 4).wait();
            double sum = 0, mx = 0; size_t nonfinite = 0;
            for (float x : v) { if (!std::isfinite(x)) { ++nonfinite; continue; } sum += std::fabs(x); mx = std::max(mx, (double) std::fabs(x)); }
            std::fprintf(stderr, "verify dbg:   %-8s n=%zu mean|.|=%.4g max|.|=%.4g nonfinite=%zu first4=%g %g %g %g\n", what, n,
                         sum / (double) n, mx, nonfinite, v[0], v[1], v[2], v[3]);
        };
        if (dbgR_) {
            const size_t N = (size_t) g.n_embd;
            std::vector<float> r((size_t) g.n_layers * N), m((size_t) g.n_layers * N);
            cs_->memcpy(r.data(), dbgR_, r.size() * 4).wait();
            cs_->memcpy(m.data(), dbgM_, m.size() * 4).wait();
            for (int64_t l = 0; l < g.n_layers; ++l) {
                double sr = 0, sm = 0; size_t nr = 0, nm = 0;
                for (size_t i = 0; i < N; ++i) {
                    const float x = r[(size_t) l * N + i], y = m[(size_t) l * N + i];
                    if (std::isfinite(x)) sr += std::fabs(x); else ++nr;
                    if (std::isfinite(y)) sm += std::fabs(y); else ++nm;
                }
                std::fprintf(stderr, "verify dbg:   layer %2lld %s  R mean|.|=%.4g nonfinite=%zu   mixed mean|.|=%.4g nonfinite=%zu\n",
                             (long long) l, is_qsa_layer(g, l) ? "QSA" : "GDN", sr / N, nr, sm / N, nm);
            }
        }
        {   // the device plan of the last layer group and the MoE parts it produced (token 0)
            std::vector<int32_t> pc(4, 0);
            cs_->memcpy(pc.data(), plan_, 16).wait();
            std::fprintf(stderr, "verify dbg:   plan (last layer, grp 0): groups=%d entries=%d pcie=%d\n", pc[0], pc[1], pc[2]);
        }
        stat("parts_", parts_, (size_t) ss.k * g.n_embd);
        stat("hit_out_", hit_out_, (size_t) ss.k * g.n_embd);
        stat("shared_", shared_, (size_t) g.n_embd);
        stat("emb_", emb_, (size_t) g.n_embd);
        stat("ple_", ple_, (size_t) g.n_embd);
        stat("R_", R_, (size_t) g.n_embd);
        stat("mixed_", mixed_, (size_t) g.n_embd);
    }
    if (ar_on() && h_plan_err_ != nullptr && *(volatile uint32_t*) h_plan_err_ != 0) {   // #871: refresh_ar saw the table whole
        *(volatile uint32_t*) h_plan_err_ = 0;
        err = "verify: the all-resident plan met an expert that is not in VRAM (the residency table changed during the window)";
        return false;
    }
    commit_pending_ = false;
    progress_at("verify window: waiting for the expert copies", (int64_t) T);
    if (copy_used_) {
        copy_->wait(); // no host function of this window may raise flag B in
                       // the next one
        copy_used_ = false;
    }
    if (prof_on_ && G == 1) {   // the window's GPU stage stamps
        collect_profile();
        if (std::getenv("STRATA_VERIFY_EAGER") != nullptr)   // SYCL port: the stage table per window, host-clocked
            std::fprintf(stderr, "verify stages T=%d (ms/window):%s\n", T, profile_report().c_str());
    }
    // ---- a sampled or penalized request: the head's sampling again, host-side so its parameters are this call's
    // own (a captured kernel would replay the same draws forever).  Row t's draw is Philox(seed, pos0 + t): tied to
    // the POSITION it samples, not to how the text was cut into windows, so a seed replays the same text whatever
    // the drafts were. Exact: a rejected row's draw is discarded, and no kept decision depends on a reused draw.
    if (le_ < g.n_layers) {   // a layer split's earlier stage: the hand-off is written (synced above)
        ++windows;
        return next_ == nullptr || next_->run(T, tokens, pos0, pool, next_user_, out, err);
    }
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    if (head_sampling_ && (sampled || hist_d_ != nullptr)) {
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) pos0;
        sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
        if (DPCT_CHECK_ERROR(cs_->wait()) !=
            0) { // m_out_ is the mapped h_out_: synced, it is readable
            err = "verify: the head sampling failed";
            return false;
        }
    }
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {   // debug: the first non-finite head
        static bool reported = false;
        if (!reported) {
            std::vector<float> h((size_t) T * (size_t) n_vocab_);
            dpct::get_in_order_queue()
                .memcpy(h.data(), head_logits_, h.size() * 4)
                .wait();
            for (int t = 0; t < T && !reported; ++t) {
                int64_t bad = 0;
                for (int64_t v = 0; v < n_vocab_; ++v) bad += !std::isfinite(h[(size_t) t * n_vocab_ + v]);
                if (bad) {
                    reported = true;
                    std::fprintf(stderr, "strata dbg: verify window at position %lld, row %d: %lld of %lld logits non-finite "
                                         "(token out %d)\n", (long long) pos0, t, (long long) bad, (long long) n_vocab_, out[t]);
                }
            }
        }
    }
    VDBG("window done\n");
    ++windows;
    progress_at("decode");
    progress_beat();
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void Verifier::set_plan_slot(int grp) {
    const int64_t cap = sink_.cap;
    int32_t* base = h_plan_ + (size_t) grp * (size_t) plan_i32_;
    const int64_t i32 = 4 + (cap + 1) + cap + cap;
    const int64_t ptr_off = (i32 + 1) & ~1ll;
    sink_.counts = base;
    sink_.start = base + 4;
    sink_.dst = sink_.start + cap + 1;
    sink_.tok = sink_.dst + cap;
    sink_.ptr = (unsigned long long*) (base + ptr_off);
    sink_.ptr2 = sink_.ptr + cap;
    sink_.start2 = base + ptr_off + 4 * cap;
    const int G = last_batch_ ? 1 : (groups_[last_t_] > 0 ? groups_[last_t_] : 1);
    const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
    sink_.staging = (unsigned long long) (staging_ + (size_t) (grp * per) * strata::kernels::cpu::expert_layout().max_blob);
    sink_.staging_cap = per;
}

// Flag B only rises: a host function of an earlier layer may run after a later layer already raised it directly.
void Verifier::raise_flag(uint32_t* flag, uint32_t value) {
    volatile long* f = (volatile long*) flag;
#if defined(_WIN32)
    long cur = *f;
    while ((uint32_t) cur < value) {
        const long prev = _InterlockedCompareExchange(f, (long) value, cur);
        if (prev == cur) break;
        cur = prev;
    }
#else
    uint32_t cur = __atomic_load_n((uint32_t*) flag, __ATOMIC_SEQ_CST);
    while (cur < value && !__atomic_compare_exchange_n((uint32_t*) flag, &cur, value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {}
#endif
}

// Plan v0.3 P6: the PCIe share by DMA.  The copy engine moves the blobs while the CPU computes its own share and the
// GPU its VRAM experts; a host function raises flag B when they have landed (the graph waits for it before the PCIe
// groups).  Staging is split between the two token groups of a split window.
void Verifier::fetch_dma(void *ctx, const uint8_t *const *src, int n,
                         size_t bytes) try {
    Verifier* v = (Verifier*) ctx;
    const uint32_t want = v->cur_layer_ + 1;
    if (n <= 0) { raise_flag(v->h_flagB_, want); return; }
    v->copy_used_ = true;
    uint8_t* stage = (uint8_t*) v->sink_.staging;                  // this group's half in a split window
    // STRATA_DMA_BATCH (midhatn's #807, F12): the group's independent uploads as one cudaMemcpyBatchAsync.  Mode 0 is
    // the loop the engine always ran, errors unchecked (it never stops the engine).  A batch mode whose uploads
    // cannot be submitted stops the engine: flag B must not release the consumers to expert weights that were never
    // copied.
    const int batch_mode = strata::core::dma_batch_mode();
    if (batch_mode == 0) {
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        for (int i = 0; i < n; ++i)
            v->copy_->memcpy(stage + (size_t)i * bytes, src[i], bytes);
    } else {
        const auto copied = strata::core::copy_expert_blobs(stage, src, n, bytes, v->copy_, batch_mode);
        /*
        DPCT1000: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (copied != 0) {
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            std::fprintf(stderr, "strata DMA: expert upload failed: %s\n",
                         dpct::get_error_string_dummy(copied));
            /*
            DPCT1001: The statement could not be removed.
            */
            std::fflush(stderr);
            v->release_gpu_waits(5000);
            std::abort();
        }
    }
    FlagSet& fs = v->flag_sets_[v->cur_layer_ % (sizeof v->flag_sets_ / sizeof v->flag_sets_[0])];
    fs.flag = v->h_flagB_;
    fs.value = want;
  FlagSet* fsp = &fs;
  v->copy_->submit([&](sycl::handler &cgh) {
    cgh.host_task([=]() { raise_flag(fsp->flag, fsp->value); });
  });
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void Verifier::publish_plan(void* ctx) {
    Verifier* v = (Verifier*) ctx;
    _mm_sfence();
    *(volatile uint32_t*) v->h_flagA_ = v->cur_layer_ + 1;
}

bool Verifier::window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                               std::string& err) try {
    if (next_ != nullptr) return next_->window_logprobs(targets, T, pos0, extra_id, out, err);
    const OnDevice on_device(device_);
    if (head_logits_ == nullptr || n_vocab_ <= 0 || T <= 0 || targets == nullptr || out == nullptr) {
        err = "window_logprobs: no head logits for this window";
        return false;
    }
    // run() synchronized cs_ before returning, so the head of this window is complete
    std::vector<float> h((size_t) T * (size_t) n_vocab_);
    cs_->memcpy(h.data(), head_logits_, h.size() * sizeof(float)).wait();
    for (int t = 0; t < T; ++t) {
        const float* row = h.data() + (size_t) t * (size_t) n_vocab_;
        const int32_t tgt = targets[t];
        if (tgt < 0 || (int64_t) tgt >= n_vocab_) continue;
        int64_t top = 0;
        for (int64_t v = 1; v < n_vocab_; ++v)
            if (row[v] > row[top]) top = v;
        const double maxv = row[top];
        const bool has_extra = extra_id >= 0 && (int64_t) extra_id < n_vocab_;
        double sum = 0.0, sum_without = 0.0;   // the second skips extra_id: no cancellation when it holds ~all mass
        for (int64_t v = 0; v < n_vocab_; ++v) {
            const double e = std::exp((double) row[v] - maxv);
            sum += e;
            if (v != (int64_t) extra_id) sum_without += e;
        }
        const double lse = maxv + std::log(sum);
        const double extra = has_extra ? (double) row[extra_id] - lse : NAN;
        const double without = has_extra && tgt != extra_id && sum_without > 0.0
                                   ? (double) row[tgt] - (maxv + std::log(sum_without)) : NAN;
        std::fprintf(out, "%lld\t%d\t%.9f\t%lld\t%.9f\t%d\t%.9f\t%.9f", (long long) (pos0 + t), (int) tgt,
                     (double) row[tgt] - lse, (long long) top, maxv - lse, (int) (top == (int64_t) tgt), extra,
                     without);
        // STRATA_LOGPOS_TOPK=K: the K most likely tokens and their log-probabilities too (`id:logprob`), for a
        // top-k comparison with another engine on the same tokens (docs/UNSLOTH_Q4.md)
        static const int topk = [] {
            const char* v = std::getenv("STRATA_LOGPOS_TOPK");
            return v != nullptr ? std::max(0, std::min(256, std::atoi(v))) : 0;
        }();
        if (topk > 0) {
            std::vector<int32_t> order((size_t) n_vocab_);
            for (int64_t v = 0; v < n_vocab_; ++v) order[(size_t) v] = (int32_t) v;
            std::partial_sort(order.begin(), order.begin() + topk, order.end(),
                              [&](int32_t a, int32_t b) { return row[a] > row[b]; });
            for (int j = 0; j < topk; ++j)
                std::fprintf(out, "\t%d:%.6f", order[(size_t) j], (double) row[order[(size_t) j]] - lse);
        }
        std::fprintf(out, "\n");
    }
    std::fflush(out);
    return true;
} catch (sycl::exception const& e) {
    err = std::string("window_logprobs: ") + e.what();
    return false;
}

namespace { bool g_commit_async = false; }
void Verifier::set_commit_async(bool on) { g_commit_async = on && std::getenv("STRATA_COMMIT_SYNC") == nullptr; }

bool Verifier::commit(int n_keep, std::string &err) try {
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    const Clock::time_point t0 = Clock::now();
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    if (last_t_ == 1 && one_token_self_commit()) {
        // a one-token window has advanced the state itself (record_window): no commit graph
    } else {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const dpct::err0 le =
            DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(*commit_exec_));
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
        if (le != 0) {
          err = std::string("verify: commit launch: ") +
                dpct::get_error_string_dummy(le);
          return false;
        }
        // set_commit_async: no wait here - the next window runs on the same stream after it, and the drafter (its own
        // stream) reads only this window's final rows and its own K/V. h_commit_ is next written after the next window's
        // results are read, i.e. after this graph has run.  Everything else waits on commit_done_ (wait_commit).
        if (!g_commit_async || next_ != nullptr) {
            const dpct::err0 se = DPCT_CHECK_ERROR(cs_->wait());
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
            if (se != 0) {
              err = std::string("verify: commit: ") +
                    dpct::get_error_string_dummy(se);
              return false;
            }
        } else {
            /*
            DPCT1024: The original code returned the error code that was
            further consumed by the program logic. This original code was
            replaced with 0. You may need to rewrite the program logic consuming
            the error code.
            */
            const dpct::err0 re =
                DPCT_CHECK_ERROR(dpct::sync_barrier(commit_done_, cs_));
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
            if (re != 0) {
              err = std::string("verify: commit event: ") +
                    dpct::get_error_string_dummy(re);
              return false;
            }
            (void)DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty()));
            commit_pending_ = true;
        }
    }
    if (ple_stage())   // stages that share one session must advance it once
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    ms_commit += ms_since(t0);
    return next_ == nullptr || next_->commit(n_keep, err);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool Verifier::wait_commit(std::string &err) try {
    if (commit_pending_) {
        const OnDevice on_device(device_);
        commit_pending_ = false;
        const dpct::err0 se = DPCT_CHECK_ERROR(commit_done_->wait_and_throw());
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
        if (se != 0) {
          err = std::string("verify: commit: ") +
                dpct::get_error_string_dummy(se);
          return false;
        }
    }
    return next_ == nullptr || next_->wait_commit(err);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// SYCL port: capture every window graph now (one per window size) instead of on first use, so the first request
// does not pay for them (a 2,400-node graph takes tens of ms to finalize on this backend).
bool Verifier::warm(std::string &err) {
    for (int T = 1; T <= max_t_; ++T)
        if (!capture(T, err)) return false;
    return true;
}

// ================================ BATCH WINDOWS (see init_slots) ================================

bool Verifier::init_slots(const std::vector<SessionState *> &slots,
                          std::string &err) try {
    const OnDevice on_device(device_);
    if (g_ == nullptr || ss_ == nullptr) { err = "verify: init_slots before init"; return false; }
    if (slots.empty()) {
        err = "verify: init_slots needs at least one session";
        return false;
    }
    for (SessionState* x : slots) {
        if (x == nullptr || x->layer_lo != ss_->layer_lo || x->layer_hi != ss_->layer_hi ||
            x->max_cells != ss_->max_cells || x->gdn_ord0 != ss_->gdn_ord0 || x->qsa_ord0 != ss_->qsa_ord0) {
            err = "verify: a slot's session is not carved like the stage's own (layer range, context)";
            return false;
        }
    }
    const strata::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t S = (int64_t) slots.size(), CB = 2 + max_t_;
    const int64_t TS = (s.idx_block - 1) * g_->idx_key_dim, nQ = g_->n_qsa_layers();
    void* d = nullptr;
    if (!mapped((size_t) (S * CB * 4 + 16), (void**) &h_commitb_, (void**) &m_commitb_)) {
        err = "verify: the batch commit staging failed";
        return false;
    }
    const uint64_t a = ((uint64_t) S * CB * 4 + 255) & ~255ull;
    if (DPCT_CHECK_ERROR(
            d = (void *)sycl::malloc_device(
                a + (uint64_t)S * std::max<int64_t>(nQ, 1) * TS * 4,
                dpct::get_in_order_queue())) != 0) {
        err = "verify: the batch buffers do not fit";
        return false;
    }
    arena_b_ = d;
    commitb_ = (int32_t*) d;
    tail_snap_b_ = (float*) ((uint8_t*) d + a);
    slots_ = slots;
    slot_sp_.assign(slots.size(), sampling_);   // greedy until set_slot_sampling
    std::fprintf(stderr, "strata verify: batch windows of up to %lld sequences (layers [%lld, %lld))\n", (long long) S,
                 (long long) lb_, (long long) le_);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool Verifier::capture_batch(const int *rows, int S, int hbase,
                             std::string &err) try {
    dpct::experimental::command_graph_exec_ptr &ex =
        exec_bm_[bkey(rows, S, hbase)];
    if (ex != nullptr) return true;
    if (DPCT_CHECK_ERROR(dpct::experimental::begin_recording(cs_)) != 0) {
        err = "verify: begin batch capture failed";
        return false;
    }
    batch_rec_ = true;
    row_base_ = hbase;
    for (int t = 0; t < S; ++t) brow_[t] = rows[t];
    std::string rerr;
    const bool ok = record_window(S, cs_, rerr);
    batch_rec_ = false;
    row_base_ = 0;
    dpct::experimental::command_graph_ptr graph = nullptr;
    const dpct::err0 ce =
        DPCT_CHECK_ERROR(dpct::experimental::end_recording(cs_, &graph));
    if (!ok || ce != 0) {
        if (graph) delete (graph);
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use error
        codes. Please replace the "get_error_string_dummy(...)" with a real
        error-handling function.
        */
        err = !ok ? rerr
                  : std::string("verify: end batch capture: ") +
                        dpct::get_error_string_dummy(ce);
        return false;
    }
    const dpct::err0 ie = DPCT_CHECK_ERROR(
        ex = new sycl::ext::oneapi::experimental::command_graph<
            sycl::ext::oneapi::experimental::graph_state::executable>(
            graph->finalize()));
    delete (graph);
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (ie != 0) {
      err = std::string("verify: batch instantiate: ") +
            dpct::get_error_string_dummy(ie);
      return false;
    }
    /*
    DPCT1007: Migration of cudaGraphUpload is not supported.
    */
    // no cudaGraphUpload on SYCL: a finalized command_graph is already resident
    cs_->wait();
    std::string list;
    for (int t = 0; t < S; ++t) list += (t ? "," : "") + std::to_string(rows[t]);
    std::fprintf(stderr, "strata verify: captured the batch window over slots %s\n", list.c_str());
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool Verifier::capture_commit_batch(const int *rows, int S, int hbase,
                                    std::string &err) try {
    dpct::experimental::command_graph_exec_ptr &cex =
        commit_bm_[bkey(rows, S, hbase)];
    if (cex != nullptr) return true;
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ZV = g.ssm_value_dim, ID = g.idx_key_dim, MT = max_t_;
    const int64_t CB = 2 + MT, nQ = g.n_qsa_layers();
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (DPCT_CHECK_ERROR(dpct::experimental::begin_recording(cs_)) != 0) {
        err = "verify: begin batch commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        for (int t = 0; t < S; ++t)
            if (t == 0 || rows[t] != rows[t - 1])
                copy_i32_from_mapped(commitb_ + (size_t) rows[t] * CB, m_commitb_ + (size_t) rows[t] * CB, CB, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        for (int64_t l = lb_; l < le_ && ok; ++l) {
            const LayerView v(*wt_, l);
            if (!is_qsa_layer(g, l)) {
                const WeightRef* wnm = need(v, "ssm_norm.weight", err);
                if (!wnm) { ok = false; break; }
                for (int t = 0; t < S;) {
                    const int first = t;
                    while (t < S && rows[t] == rows[first]) ++t;
                    SessionState& sx = *slots_[(size_t) rows[first]];
                    const int32_t* keep = commitb_ + (size_t) rows[first] * CB;
                    float* state = sx.gdn_state + (size_t) (gdn_index - sx.gdn_ord0) * gdn_floats;
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    gdn_conv_commit(conv, qkv_L_ + (size_t) gdn_index * MT * C + (size_t) first * C,
                                    (int) C, keep, cs_);
                    gdn_step_norm_multi(state, h_L_ + (size_t) gdn_index * MT * C + (size_t) first * C, (int) C,
                                        gate_L_ + (size_t) gdn_index * MT * HV + (size_t) first * HV,
                                        beta_L_ + (size_t) gdn_index * MT * HV + (size_t) first * HV,
                                        z_ + (size_t) first * ZV, (const float*) wnm->data, EPS,
                                        y_dummy_ + (size_t) first * ZV, (int) g.ssm_k_heads,
                                        (int) HV, t - first, keep, cs_,
                                        t - first > 1 ? t - first : 0);   // a 1-row group keeps the 0.1.39 kernel
                }
                ++gdn_index;
            } else {
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn) { ok = false; break; }
                for (int t = 0; t < S;) {
                    const int first = t;
                    while (t < S && rows[t] == rows[first]) ++t;
                    const QsaState& st = slots_[(size_t) rows[first]]->qsa_states[qsa_index];
                    copy_from_mapped(st.idx_tail, tail_snap_b_ + ((size_t) rows[first] * nQ + qsa_index) * TS,
                                     TS, cs_);
                    const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    for (int u = first; u < t; ++u)
                        native_qsa_indexer_append(idx_raw_L_ + (size_t) (qsa_index * MT + u) * ID,
                                                  commitb_ + (size_t) rows[first] * CB + 2 + (u - first),
                                                  0, (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                  rope_scaling(), cs_);
                }
                ++qsa_index;
            }
        }
        if (ok && ss_->ple.ready() && ple_stage())
            for (int t = 0; t < S;) {
                const int first = t;
                while (t < S && rows[t] == rows[first]) ++t;
                copy_indexed(slots_[(size_t) rows[first]]->ple_hist, hist_snap_ + (size_t) first * HS,
                             HS, commitb_ + (size_t) rows[first] * CB + 1, HS, cs_);
            }
    } catch (const std::exception& e) {
        err = std::string("verify batch commit: ") + e.what();
        ok = false;
    }
    dpct::experimental::command_graph_ptr graph = nullptr;
    const dpct::err0 ce =
        DPCT_CHECK_ERROR(dpct::experimental::end_recording(cs_, &graph));
    if (!ok) {
        if (graph) delete (graph);
        return false;
    }
    if (ce != 0 ||
        DPCT_CHECK_ERROR(
            cex = new sycl::ext::oneapi::experimental::command_graph<
                sycl::ext::oneapi::experimental::graph_state::executable>(
                graph->finalize())) != 0) {
        if (graph) delete (graph);
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use error
        codes. Please replace the "get_error_string_dummy(...)" with a real
        error-handling function.
        */
        err = std::string("verify: batch commit capture: ") +
              dpct::get_error_string_dummy(ce);
        return false;
    }
    delete (graph);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool Verifier::stage_batch(const int *rows, int S, int hbase,
                           const int32_t *tokens, const int64_t *pos,
                           std::string &err) try {
    using namespace strata::kernels;
    if (S < 1 || S > max_t_ || hbase < 0 ||
        (next_ != nullptr && hbase + S > (int) slots_.size())) {
        err = "verify: batch rows out of range (init_slots)";
        return false;
    }
    for (int t = 0; t < S; ++t) {
        if (rows[t] < 0 || rows[t] >= (int) slots_.size()) {
            err = "verify: a batch row's slot is out of range";
            return false;
        }
        for (int u = 0; u < t - 1; ++u)
            if (rows[u] == rows[t] && rows[t - 1] != rows[t]) {
                err = "verify: a slot's proposed rows must be contiguous";
                return false;
            }
        if (t > 0 && rows[t] == rows[t - 1] && pos[t] != pos[t - 1] + 1) {
            err = "verify: proposed rows must have consecutive positions";
            return false;
        }
        if (t > 0 && rows[t] == rows[t - 1] && next_ != nullptr) {
            err = "verify: grouped slot rows do not support a layer split yet";
            return false;
        }
    }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    const ModelGeometry& g = *g_;
    for (int t = 0; t < S; ++t)
        if (pos[t] < 0 || pos[t] + 1 > slots_[(size_t) rows[t]]->max_cells) {
            err = "verify: slot " + std::to_string(rows[t]) + " runs past its context";
            return false;
        }
    refresh_ar();
    // --batch-mtp only (limit 0 = 0.1.39: no eviction): slot rotation creates new layouts; bound the captured graph
    // pairs, evicting the least recently used layout.
    if (batch_graph_limit_ > 0) {
        const auto key = bkey(rows, S, hbase);
        if (exec_bm_.find(key) == exec_bm_.end() && exec_bm_.size() >= batch_graph_limit_) {
            if (DPCT_CHECK_ERROR(cs_->wait()) != 0) {
                err = "verify: synchronizing before batch graph eviction failed";
                return false;
            }
            auto old = exec_bm_.begin();
            uint64_t oldest = UINT64_MAX;
            for (auto it = exec_bm_.begin(); it != exec_bm_.end(); ++it) {
                const auto u = bm_used_.find(it->first);
                const uint64_t t = u == bm_used_.end() ? 0 : u->second;
                if (t < oldest) { oldest = t; old = it; }
            }
            const auto old_key = old->first;
            if (old->second) delete (old->second);
            exec_bm_.erase(old);
            bm_used_.erase(old_key);
            auto commit_old = commit_bm_.find(old_key);
            if (commit_old != commit_bm_.end()) {
                if (commit_old->second) delete (commit_old->second);
                commit_bm_.erase(commit_old);
            }
        }
        bm_used_[key] = ++bm_tick_;
    }
    if (!capture_batch(rows, S, hbase, err) || !capture_commit_batch(rows, S, hbase, err)) return false;
    const Clock::time_point t0 = Clock::now();
    const QsaShapes s = shapes_of(g);
    for (int t = 0; t < S; ++t) {
        h_tok_[t] = tokens[t];
        qsa_step_fill(h_step_ + t * kStepCount, pos[t], s);
        for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) pos[t];
        int32_t* pk = h_pos_ + (size_t) max_t_ * g.n_head;
        int32_t* pi = pk + (size_t) max_t_ * g.n_head_kv;
        for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = (int32_t) pos[t];
        for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = (int32_t) pos[t];
    }
    if (ss_->ple.ready() && ple_stage()) {
        uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];   // (not `rows`: that is the slots of the window's rows)
        for (int t = 0; t < S; ++t) {
            const SessionState& sx = *slots_[(size_t) rows[t]];
            int32_t prev[2] = {sx.ple_prev[0], sx.ple_prev[1]};
            if (t > 0 && rows[t] == rows[t - 1]) {
                prev[0] = t > 1 && rows[t - 2] == rows[t] ? tokens[t - 2] : sx.ple_prev[1];
                prev[1] = tokens[t - 1];
            }
            ngram_rows(&tokens[t], prev, 1, ss_->ple.consts, ple_rows + t * PLE_N_HEADS);
        }
        if (!ss_->ple.table->gather_batch(ple_rows, (size_t) S, h_ple_, err)) return false;
    }
    // Each slot owns a contiguous group. The default commit keeps all its rows; speculative
    // decoding may change the prefix length after comparing the draft with these picks.
    const int64_t CB = 2 + max_t_;
    for (int t = 0; t < S;) {
        const int first = t;
        while (t < S && rows[t] == rows[first]) ++t;
        int32_t* c = h_commitb_ + (size_t) rows[first] * CB;
        c[0] = t - first;
        c[1] = t - first - 1;
        for (int64_t j = 0; j < max_t_; ++j)
            c[2 + j] = j < t - first ? (int32_t) pos[first + j] : -1;
    }
    *(volatile uint32_t*) h_seq_ = 0;
    for (int _i = 1; _i <= 8; ++_i) ((volatile uint32_t*) h_seq_)[_i] = 0;   /* + the payload tags/checksums */
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = S;
    for (int t = 0; t < S; ++t) last_rows_[t] = rows[t];
    row_base_ = hbase;   // the graphs' key (and nothing else) reads it until the next stage_batch
    last_batch_ = true;
    for (int t = 0; t < S; ++t) { last_tokens_[t] = tokens[t]; last_pos_b_[t] = pos[t]; }
    ms_host += ms_since(t0);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool Verifier::run_slots(int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user, int32_t* out,
                         std::string& err) {
    int rows[8] = {};
    for (int t = 0; t < S && t < 8; ++t) rows[t] = t;
    return run_slot_rows(rows, S, tokens, pos, pool, user, out, err);
}

bool Verifier::run_slot_rows(const int *rows, int S, const int32_t *tokens,
                             const int64_t *pos, PoolMultiFn pool, void *user,
                             int32_t *out, std::string &err) try {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    if (!stage_batch(rows, S, 0, tokens, pos, err)) return false;
    const dpct::err0 le =
        DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(*exec_bm_[bkey(rows, S, 0)]));
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (le != 0) {
      err = std::string("verify: batch launch: ") +
            dpct::get_error_string_dummy(le);
      return false;
    }
    (void)DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty()));
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int64_t steps = le_ - lb_;
    // #646: a stage whose every expert is resident plans on the device and raises no host doorbells, so the
    // per-layer spin below has nothing to wait for: without the PLE flag the graph finishes with the ring silent
    // ("verify batch: layer K never rang (graph finished)" - K is that stage's first layer), and with it the graph
    // sits on the PLE wait until the 20 s timeout.  As run() does: raise the PLE flag the graph's first wait reads,
    // let the graph run to the end, and skip the host's per-layer service (there is nothing to serve).
    if (all_resident_) {
        if (ss_->ple.ready() && ple_stage()) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            *flag = 1;
        }
        const dpct::err0 se = DPCT_CHECK_ERROR(cs_->wait());
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
        if (se != 0) {
          err =
              std::string("verify batch: ") + dpct::get_error_string_dummy(se);
          return false;
        }
        copy_->wait();
        if (prof_on_) collect_profile();
        ++windows;
        if (le_ < g.n_layers) return next_ == nullptr || next_->run_slot_rows(rows, S, tokens, pos, pool, next_user_, out, err);
        if (!sample_rows(S, err)) return false;
        for (int t = 0; t < S; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
        progress_at("decode");
        progress_beat();
        return true;
    }
    for (int64_t k = 0; k < steps; ++k) {
        const int64_t l = lb_ + k;
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify batch: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const dpct::err0 q = cs_->ext_oneapi_empty() ? 0 : 1;
                if (q != 1 && *seq < want) {
                    err = "verify batch: layer " + std::to_string(l) +
                          " never rang (" +
                          /*
                          DPCT1009: SYCL reports errors using exceptions and
                          does not use error codes. Please replace the
                          "get_error_string_dummy(...)" with a real
                          error-handling function.
                          */
                          (q == 0
                               ? std::string("graph finished")
                               : std::string(dpct::get_error_string_dummy(q))) +
                          ")";
                    return false;
                }
            }
            if (now - a > strata_ring_timeout()) {
                err = "verify batch: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
                return false;
            }
        }
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot(0);
        progress_at("verify batch: the CPU experts of layer", l);
        if (pool != nullptr) pool(user, h_x_, h_ids_, S, ss_->k, h_ymiss_, l);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *flag = want;
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
    }
    const dpct::err0 se = DPCT_CHECK_ERROR(cs_->wait());
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (se != 0) {
      err = std::string("verify batch: ") + dpct::get_error_string_dummy(se);
      return false;
    }
    copy_->wait();
    if (prof_on_) collect_profile();
    ++windows;
    if (le_ < g.n_layers) return next_ == nullptr || next_->run_slot_rows(rows, S, tokens, pos, pool, next_user_, out, err);
    if (!sample_rows(S, err)) return false;
    for (int t = 0; t < S; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    progress_at("decode");
    progress_beat();
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool Verifier::commit_slots(std::string& err) {
    std::vector<int> keep(slots_.size(), 0);
    for (int t = 0; t < last_t_; ++t) ++keep[(size_t) last_rows_[t]];
    return commit_slot_prefixes(keep.data(), err);
}

bool Verifier::commit_slot_prefixes(const int *keep, std::string &err) try {
    const OnDevice on_device(device_);
    if (!last_batch_ || last_t_ < 1) { err = "verify: commit_slots without a batch window"; return false; }
    const int S = last_t_;
    const Clock::time_point t0 = Clock::now();
    const int64_t CB = 2 + max_t_;
    // One prefix per slot: rejected draft rows must not enter recurrent state.
    for (int t = 0; t < S;) {
        const int first = t;
        while (t < S && last_rows_[t] == last_rows_[first]) ++t;
        const int n = keep[last_rows_[first]];
        if (n < 1 || n > t - first) {
            err = "verify: accepted prefix is outside its slot group";
            return false;
        }
        int32_t* c = h_commitb_ + (size_t) last_rows_[first] * CB;
        c[0] = n;
        c[1] = n - 1;
        for (int j = 0; j < max_t_; ++j)
            c[2 + j] = j < n ? (int32_t) last_pos_b_[first + j] : -1;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const dpct::err0 le = DPCT_CHECK_ERROR(
        (cs_)->ext_oneapi_graph(*commit_bm_[bkey(last_rows_, S, row_base_)]));
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (le != 0) {
      err = std::string("verify: batch commit launch: ") +
            dpct::get_error_string_dummy(le);
      return false;
    }
    const dpct::err0 se = DPCT_CHECK_ERROR(cs_->wait());
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (se != 0) {
      err = std::string("verify: batch commit: ") +
            dpct::get_error_string_dummy(se);
      return false;
    }
    if (ple_stage())
        for (int t = 0; t < S;) {
            const int first = t;
            while (t < S && last_rows_[t] == last_rows_[first]) ++t;
            SessionState& sx = *slots_[(size_t) last_rows_[first]];
            for (int u = first; u < first + keep[last_rows_[first]]; ++u) {
                sx.ple_prev[0] = sx.ple_prev[1];
                sx.ple_prev[1] = last_tokens_[u];
            }
        }
    ms_commit += ms_since(t0);
    return next_ == nullptr || next_->commit_slot_prefixes(keep, err);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool Verifier::sample_rows(int S, std::string &err) try {
    bool any = false;
    for (int t = 0; t < S; ++t) {
        strata::kernels::SamplerParams sp = slot_sp_[(size_t) last_rows_[t]];
        if (sp.greedy || sp.temperature <= 0.0f) continue;
        sp.counter = (uint64_t) last_pos_b_[t];   // Philox(seed, position): the solo window's draw for this position
        sp.penalty_last_n = 0;
        strata::kernels::sample_tokens(head_logits_ + (size_t) t * (size_t) n_vocab_, 1, (int) n_vocab_, nullptr, 0, sp,
                                       m_out_ + t, cs_);
        any = true;
    }
    if (any && DPCT_CHECK_ERROR(cs_->wait()) != 0) {
      err = "verify batch: the row sampling failed"; return false;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool Verifier::batch_launch(int base, int S, const int32_t *tokens,
                            const int64_t *pos, std::string &err) try {
    const OnDevice on_device(device_);
    if (b_running_) { err = "verify: batch_launch while this stage is busy"; return false; }
    int rows[8] = {};
    for (int t = 0; t < S && t < 8; ++t) rows[t] = base + t;
    if (!stage_batch(rows, S, base, tokens, pos, err)) return false;
    dpct::err0 le = DPCT_CHECK_ERROR(
        (cs_)->ext_oneapi_graph(*exec_bm_[bkey(rows, S, base)]));
    if (le == 0) le = DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(*commit_bm_[bkey(
        rows, S, base)])); // right behind it: every row is kept
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (le != 0) {
      err = std::string("verify: batch launch: ") +
            dpct::get_error_string_dummy(le);
      return false;
    }
    (void)DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty()));
    if (ple_stage())   // the host's side of the commit (the hash's last two tokens)
        for (int t = 0; t < S; ++t) {
            SessionState& sx = *slots_[(size_t) rows[t]];
            sx.ple_prev[0] = sx.ple_prev[1];
            sx.ple_prev[1] = tokens[t];
        }
    b_running_ = true;
    b_k_ = 0;
    b_steps_ = le_ - lb_;
    b_last_ = Clock::now();
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int Verifier::batch_poll(PoolMultiFn pool, void *user, std::string &err) try {
    if (!b_running_) return 1;
    const OnDevice on_device(device_);
    volatile uint32_t* const seq = h_seq_;
    const int S = last_t_;
    // #646: an all-resident stage's graph raises no host doorbells (see run_slot_rows): nothing to serve per layer,
    // so the poll is just "has the graph finished" - except the PLE flag, which the graph's first wait reads and
    // only the host can raise (the same raise run_slot_rows makes).
    if (all_resident_ && ss_->ple.ready() && ple_stage()) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        *h_flag_ = 1;
    }
    while (!all_resident_ && b_k_ < b_steps_) {
        const uint32_t want = (uint32_t) (b_k_ + 1);
        if (*seq < want) {
            const auto now = Clock::now();
            if (now - b_last_ > std::chrono::milliseconds(2)) {
                const dpct::err0 q = cs_->ext_oneapi_empty() ? 0 : 1;
                if (q != 1 && *seq < want) {
                    err = "verify batch: layer " + std::to_string(lb_ + b_k_) +
                          " never rang (" +
                          /*
                          DPCT1009: SYCL reports errors using exceptions and
                          does not use error codes. Please replace the
                          "get_error_string_dummy(...)" with a real
                          error-handling function.
                          */
                          (q == 0
                               ? std::string("graph finished")
                               : std::string(dpct::get_error_string_dummy(q))) +
                          ")";
                    b_running_ = false;
                    return -1;
                }
                if (now - b_last_ > strata_ring_timeout()) {
                    err = "verify batch: timed out at layer " + std::to_string(lb_ + b_k_) + released_note(release_gpu_waits(5000));
                    b_running_ = false;
                    return -1;
                }
            }
            return 0;
        }
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot(0);
        if (pool != nullptr) pool(user, h_x_, h_ids_, S, ss_->k, h_ymiss_, lb_ + b_k_);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *(volatile uint32_t*) h_flag_ = want;
        ms_pool += ms_since(b);
        b_last_ = Clock::now();
        ++b_k_;
    }
    const dpct::err0 q = cs_->ext_oneapi_empty() ? 0 : 1;
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1002: Special case error handling if-stmt was detected. You may need
    to rewrite this code.
    */
    if (q == 1) return 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (q != 0) {
      err = std::string("verify batch: ") + dpct::get_error_string_dummy(q);
      b_running_ = false;
      return -1;
    }
    const dpct::err0 qc = DPCT_CHECK_ERROR(
        ((copy_)->ext_oneapi_empty())); // no host function of this window may
                                        // raise flag B in the next
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1002: Special case error handling if-stmt was detected. You may need
    to rewrite this code.
    */
    if (qc == 1) return 0;
    if (prof_on_) collect_profile();
    if (last_stage()) {
        if (!sample_rows(S, err)) { b_running_ = false; return -1; }
        for (int t = 0; t < S; ++t) b_out_[t] = ((volatile int32_t*) h_out_)[t];
    }
    ++windows;
    b_running_ = false;
    progress_beat();
    return 1;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool Verifier::copy_logits(int t, float* host) const {
    if (next_ != nullptr) return next_->copy_logits(t, host);   // a layer split: the head is on the last stage
    if (head_logits_ == nullptr || host == nullptr || t < 0 || n_vocab_ <= 0) return false;
    try {
        cs_->memcpy(host, head_logits_ + (size_t) t * (size_t) n_vocab_, (size_t) n_vocab_ * sizeof(float)).wait();
    } catch (sycl::exception const&) {
        return false;
    }
    return true;
}

// ================================ PIPELINED WINDOWS (see verify.hpp) ================================
//
// The window is run()'s, step for step: the same graph, the same staging, the same per-layer service (the PLE rows
// gathered while layer 0 is served, the zero-doorbell graph's one flag), the same commit graph.  Only the waits are
// split up: the host polls instead of spinning, so it can serve the other stage's window in between.

namespace {
double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }
}  // namespace

void Verifier::diag_pipelined(std::FILE* f, const char* name) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    auto ev = [](dpct::event_ptr e) {
        try {
    if (e == nullptr) return "none";
        const dpct::err0 q = dpct::sycl_event_query(e);
        return q == 0 ? "done" : q == 1 ? "PENDING" : "error";
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    };
    std::fprintf(f, "  %s: %s T=%d pos %lld served %lld/%lld; GPU rang %u, flags served %u A %u B %u; window event %s, "
                    "commit event %s (commit launched %d)\n", name, fl_active_ ? "IN FLIGHT" : "idle", last_t_,
                 (long long) last_pos0_, (long long) fl_k_, (long long) fl_total_, rd(h_seq_), rd(h_flag_), rd(h_flagA_),
                 rd(h_flagB_), ev(ev_done_), ev(ev_commit_), (int) commit_live_);
}

bool Verifier::capture_all(std::string &err) try {
    const OnDevice on_device(device_);
    if (g_ == nullptr) { err = "verify: capture_all before init"; return false; }
    if (remote_opt_ != nullptr) { err = "verify: pipelined windows do not serve --remote-expert-opt"; return false; }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    for (int T = 1; T <= max_t_; ++T)
        if (!capture(T, err)) return false;
    if (!capture_commit(err)) return false;
    if ((ev_done_ == nullptr &&
         DPCT_CHECK_ERROR(ev_done_ = new sycl::event()) != 0) ||
        (ev_commit_ == nullptr &&
         DPCT_CHECK_ERROR(ev_commit_ = new sycl::event()) != 0)) {
        err = "verify: event create failed";
        return false;
    }
    if (prof_on_ && prof_pin_ == nullptr &&
        /*
        DPCT1048: The original value cudaHostAllocDefault is not meaningful in
        the migrated code and was removed or replaced with 0. You may need to
        check the migrated code.
        */
        DPCT_CHECK_ERROR(prof_pin_ = (unsigned long long *)sycl::malloc_host(
                             prof_h_.size() * 8, dpct::get_in_order_queue())) !=
            0) {
        prof_pin_ = nullptr;   // the pipelined windows go unprofiled
                               /*
                               DPCT1026: The call to cudaGetLastError was removed because this
                               functionality is redundant in SYCL.
                               */
    }
    pl_ple_rows_.assign((size_t) strata::kernels::kVerifyMaxT * strata::kernels::PLE_N_HEADS, 0u);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// The host staging of a pipelined window: run()'s (stage_inputs), and its PLE rows from `ple_prev` with their pages
// prefetched (they are gathered into the mapped rows when layer 0 is served, as run() does).
void Verifier::pl_stage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2]) {
    using namespace strata::kernels;
    stage_inputs(T, tokens, pos0);
    staged_ = false;   // a later run() stages its own window
    SessionState& ss = *ss_;
    fl_ple_ = ss.ple.ready() && ple_stage();
    if (fl_ple_) {
        int32_t prev[2] = {ple_prev[0], ple_prev[1]};
        for (int t = 0; t < T; ++t) {
            ngram_rows(&tokens[t], prev, 1, ss.ple.consts, pl_ple_rows_.data() + t * PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
            ss.ple.table->prefetch_rows(pl_ple_rows_.data() + t * PLE_N_HEADS);
        }
    }
    pl_prev_[0] = ple_prev[0];
    pl_prev_[1] = ple_prev[1];
}

bool Verifier::prestage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err) {
    if (fl_active_) { err = "verify: a window is in flight on this verifier"; return false; }
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    if (pl_ple_rows_.empty()) { err = "verify: pipelined window not prepared (capture_all)"; return false; }
    const Clock::time_point t0 = Clock::now();
    pl_stage(T, tokens, pos0, ple_prev);
    pl_prestaged_ = true;
    ms_host += ms_since(t0);
    return true;
}

bool Verifier::pl_launch(int T, const int32_t *tokens, int64_t pos0,
                         std::string &err) try {
    const OnDevice on_device(device_);
    if (fl_active_) { err = "verify: a window is already in flight on this verifier"; return false; }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU (#267); restart the engine"; return false; }
    if (T < 1 || T > max_t_) { err = "verify: window size out of range"; return false; }
    SessionState& ss = *ss_;
    if (pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells) { err = "verify: the window runs past the context"; return false; }
    if (exec_[T] == nullptr || commit_exec_ == nullptr || ev_done_ == nullptr || pl_ple_rows_.empty()) {
        err = "verify: pipelined window not prepared (capture_all)";
        return false;
    }
    const Clock::time_point t0 = Clock::now();
    last_batch_ = false;
    bool staged = pl_prestaged_ && last_t_ == T && last_pos0_ == pos0;
    for (int t = 0; staged && t < T; ++t) staged = last_tokens_[t] == tokens[t];
    if (staged && ss.ple.ready() && ple_stage())
        staged = pl_prev_[0] == ss.ple_prev[0] && pl_prev_[1] == ss.ple_prev[1];
    pl_prestaged_ = false;
    if (!staged) pl_stage(T, tokens, pos0, ss.ple_prev);
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    fl_T_ = T;
    fl_k_ = 0;
    fl_total_ = (le_ - lb_) * G;
    fl_prof_ = prof_on_ && G == 1 && prof_pin_ != nullptr;
    if (trace_h_ != nullptr) std::memset(trace_h_, 0, trace_n_ * 8);   // #649: this window's breadcrumbs only
    std::atomic_thread_fence(std::memory_order_seq_cst);
    trace_ev("WINDOW (pipelined)", -1, -1, pos0 * 16 + T);
    ms_host += ms_since(t0);
    const dpct::err0 le = DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(*exec_[T]));
    trace_ev("LAUNCHED", -1, -1, (int64_t) le);
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be rewritten.
    */
    if (le != 0) {
      err = std::string("verify: launch: ") + dpct::get_error_string_dummy(le);
      return false;
    }
    /*
    DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    if (fl_prof_) cs_->memcpy(prof_pin_, prof_, prof_h_.size() * 8);
    /*
    DPCT1024: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    if (DPCT_CHECK_ERROR(dpct::sync_barrier(ev_done_, cs_)) != 0) {
      err = "verify: event record failed"; return false;
    }
    (void)DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty())); // WDDM: submit now
    fl_active_ = true;
    fl_since_ms_ = fl_flush_ms_ = fl_launch_ms_ = now_ms();
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int Verifier::service(PoolMultiFn pool, void *user, std::string &err) try {
    if (!fl_active_) return 1;
    if (fl_k_ >= fl_total_) return 1;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const int T = fl_T_;
    // the PLE rows into the mapped staging (layer 1's pre() copies them once flag 1 is up), as run() does at k == 0
    auto gather_ple = [&]() -> bool {
        if (!fl_ple_) return true;
        const Clock::time_point tp = Clock::now();
        if (!ss.ple.table->gather_batch(pl_ple_rows_.data(), (size_t) T, h_ple_, err)) return false;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        ms_host += ms_since(tp);
        return true;
    };
    if (all_resident_) {   // the zero-doorbell graph never rings: it waits only for flag 1 (stage 0's PLE rows)
        if (!gather_ple()) return -1;
        *(volatile uint32_t*) h_flag_ = 1;
        fl_k_ = fl_total_;
        progress_tick();
        return 1;
    }
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    while (fl_k_ < fl_total_) {
        const int64_t l = lb_ + fl_k_ / G;
        const uint32_t want = (uint32_t) (fl_k_ + 1);
        if (*(volatile uint32_t*) h_seq_ < want) {
            const double now = now_ms();
            if (now - fl_flush_ms_ > 2.0) {   // flush WDDM and notice a dead graph, as run() does
                fl_flush_ms_ = now;
                const dpct::err0 q = dpct::sycl_event_query(ev_done_);
                if (q != 1 && *(volatile uint32_t *)h_seq_ < want) {
                    trace_ev("NEVER-RANG", fl_k_, l, (int64_t) q);
                    err = "verify: layer " + std::to_string(l) +
                          " never rang (" +
                          /*
                          DPCT1009: SYCL reports errors using exceptions and
                          does not use error codes. Please replace the
                          "get_error_string_dummy(...)" with a real
                          error-handling function.
                          */
                          (q == 0
                               ? std::string("graph finished")
                               : std::string(dpct::get_error_string_dummy(q))) +
                          ")";
                    return -1;
                }
                (void)DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty()));
            }
            if (now - fl_since_ms_ > 20000.0) {   // #267: no spin kernel may outlive the engine
                trace_ev("TIMEOUT", fl_k_, l, 0);
                err = "verify: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
                return -1;
            }
            return 0;
        }
        const Clock::time_point b = Clock::now();
        ms_wait += now_ms() - fl_since_ms_;
        const int grp = (int) (fl_k_ % G);
        cur_layer_ = want - 1;
        set_plan_slot(grp);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        // the ring can arrive before its payload (elementwise.hpp): the pool must read a whole one
        if (!strata::kernels::doorbell_wait_payload(h_seq_, h_x_ + (size_t) tb * g.n_embd, (int64_t) n * g.n_embd,
                                                    h_ids_ + (size_t) tb * ss.k, h_w_ + (size_t) tb * ss.k,
                                                    (int64_t) n * ss.k, want)) {
            err = "verify: layer " + std::to_string(l) + " rang but its payload never arrived whole";
            return -1;
        }
        progress_at("verify window (pipelined): the CPU experts of layer", l);
        if (pool != nullptr)
            pool(user, h_x_ + (size_t) tb * g.n_embd, h_ids_ + (size_t) tb * ss.k, n, ss.k,
                 h_ymiss_ + (size_t) tb * ss.k * g.n_embd, l);
        if (g_trace) trace_ev(*(volatile uint32_t*) h_flagA_ == want ? "SERVED" : "SERVED-NO-PLAN-YET", fl_k_, l,
                              (int64_t) ms_since(b));
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        if (fl_k_ == 0 && !gather_ple()) return -1;
        *(volatile uint32_t*) h_flag_ = want;
        ++fl_k_;
        ms_pool += ms_since(b);
        fl_since_ms_ = fl_flush_ms_ = now_ms();
    }
    return 1;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool Verifier::done(std::string &err) try {
    if (!fl_active_ || fl_k_ < fl_total_) return false;
    const OnDevice on_device(device_);
    const dpct::err0 q = dpct::sycl_event_query(ev_done_);

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
      err = std::string("verify: ") + dpct::get_error_string_dummy(q);
      return false;
    }
    if (copy_used_) {   // no host function of this window may raise flag B in the next one
        if (((copy_)->ext_oneapi_empty() ? 0 : 1) == 1) return false;
        copy_used_ = false;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool Verifier::pl_finish(int32_t *out, std::string &err) try {
    using namespace strata::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    fl_active_ = false;
    if (fl_prof_) accumulate_profile(prof_pin_);
    ++windows;
    trace_ev("DONE (pipelined)", -1, -1, (int64_t) (now_ms() - fl_launch_ms_));
    if (le_ < g.n_layers) return true;   // an earlier stage: the hand-off is written
    const int T = fl_T_;
    const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
    if (head_sampling_ && (sampled || hist_d_ != nullptr)) {   // run()'s host-side sampling, Philox(seed, pos0 + t)
        SamplerParams sp = sampling_;
        sp.counter = (uint64_t) last_pos0_;
        sample_tokens(head_logits_, T, (int) n_vocab_, hist_d_, hist_len_, sp, m_out_, cs_);
        if (DPCT_CHECK_ERROR(cs_->wait()) != 0) {
            err = "verify: the head sampling failed";
            return false;
        }
    }
    if (out != nullptr)
        for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    progress_beat();
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool Verifier::pl_commit_async(int n_keep, std::string &err) try {
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    if (commit_exec_ == nullptr || ev_commit_ == nullptr) { err = "verify: pipelined commit not prepared"; return false; }
    const Clock::time_point t0 = Clock::now();
    if (commit_live_) {   // its graph reads the words below when it starts: the previous one has (a window ago)
        dpct::err0 q;
        while ((q = dpct::sycl_event_query(ev_commit_)) == 1) _mm_pause();
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
          err =
              std::string("verify: commit: ") + dpct::get_error_string_dummy(q);
          return false;
        }
    }
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const dpct::err0 le =
        DPCT_CHECK_ERROR((cs_)->ext_oneapi_graph(*commit_exec_));
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
    if (le != 0) {
      err = std::string("verify: commit launch: ") +
            dpct::get_error_string_dummy(le);
      return false;
    }
    /*
    DPCT1024: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    if (DPCT_CHECK_ERROR(dpct::sync_barrier(ev_commit_, cs_)) != 0) {
      err = "verify: event record failed"; return false;
    }
    (void)DPCT_CHECK_ERROR(((cs_)->ext_oneapi_empty()));
    commit_live_ = true;
    if (ple_stage())
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    ms_commit += ms_since(t0);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void Verifier::absorb_stats(Verifier& o) {
    ms_wait += o.ms_wait; ms_pool += o.ms_pool; ms_host += o.ms_host; ms_commit += o.ms_commit;
    windows += o.windows;
    o.ms_wait = o.ms_pool = o.ms_host = o.ms_commit = 0;
    o.windows = 0;
    for (int k = 0; k < 2; ++k)
        for (int i = 0; i < kProfPer; ++i) { prof_sum_[k][i] += o.prof_sum_[k][i]; o.prof_sum_[k][i] = 0; }
    prof_windows_ += o.prof_windows_;
    o.prof_windows_ = 0;
}

}  // namespace strata::core
