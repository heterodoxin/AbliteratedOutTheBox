#define DPCT_COMPAT_RT_VERSION 12080
// src/program/generate.cpp - P2.S6: `strata generate`.
//
// THE DRIVER, and the first program in this project that answers a question.
// Everything below it is a component; this is the thing that composes them into
// a token:
//
//     embed_row(token)  ->  48 captured layer graphs (the CPU expert pool
//     behind the doorbell)  -> lm_head(R)        ->  sample        ->
//     embed_row(next)  ->  ...
//
// WHAT IT IS NOT.  There is no tokenizer here.  `pack/full/tokenizer/` and
// `tools/strata_tokenizer.py` exist, and a C++ BPE is Phase 1's deliverable
// rather than this program's, so the prompt arrives as IDS via
// `--tokens`.  That is not a placeholder: it is exactly what Gate C1 needs,
// because C1 compares logits against llama.cpp on the SAME ids, and a tokenizer
// on only one side of that comparison is a second variable.
//
// AND IT IS PHASE 2, so hit rate is `h = 0` and the number it prints is slow on
// purpose
// (`phase-2-correct-engine.md:5-9`).  What it is FOR is the honest tok/s figure
// and the logit dump.

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/sycl_doorbell.hpp"
#include "strata/core/arch_defaults.hpp"
#include "strata/core/dma_batch.hpp"
#include "strata/core/device.hpp"
#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/gguf_expert_source.hpp"
namespace strata::prefill { void set_nonresident_share(double share); }   // SYCL port: prefill.cpp
#include "strata/kernels/resident_plan_mirror.hpp"
#include "strata/core/pinned.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/program/split_rules.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/mtp.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/program/logits_selection.hpp"
#include "strata/program/conv_cache.hpp"
#include "strata/program/message_boundary.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/draft_source.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/device.hpp"
#include "strata/core/emulate.hpp"
#ifndef NOMINMAX
#define NOMINMAX   // gguf_reader.hpp includes windows.h
#endif
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#include <io.h>
#else
#include <unistd.h>
#include <cerrno>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#endif

#include <array>
#include <chrono>
#include <algorithm>
#include <memory>
#include <iostream>
#include <functional>
#include <future>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <new>
#include <charconv>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>

namespace {
// Windows' WDDM driver model: native Windows, or WSL2 (its GPU goes through /dev/dxg to the Windows driver).  There,
// pinning a large arena into two CUDA contexts leaves WDDM refusing every later allocation (the 5080 + 3090 rig);
// a Linux driver has no such limit (#253: the 8 GiB cap cost a 4090 + 3060 split 3x of its prompt speed).
bool under_wddm() {
#ifdef _WIN32
    return true;
#else
    static const bool dxg = std::filesystem::exists("/dev/dxg");
    return dxg;
#endif
}

// #620 #486: " (N MiB of M MiB VRAM free on this GPU)" for an allocation's failure message (clears the error first)
std::string vram_free_note() try {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    (void)0;
    size_t free_b = 0, total_b = 0;
    /*
    DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    if (DPCT_CHECK_ERROR(
            dpct::get_current_device().get_memory_info(free_b, total_b)) != 0) {
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use
        the error codes. The cudaGetLastError function call was replaced with 0.
        You need to rewrite this code.
        */
        (void)0;
        return {};
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, " (%llu MiB of %llu MiB VRAM free on this GPU)", (unsigned long long) (free_b >> 20),
                  (unsigned long long) (total_b >> 20));
    return buf;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// perf-review D-4: the lent slots are refilled with queued copies and one wait; STRATA_REFILL_BLOCKING=1 waits on each
// A file-backed arena on AMD (STRATA_ARENA_MMAP; the engine raises ROCclr's pin-in-place threshold, see main): an
// adaptive swap copies one expert straight from the mapped file.  Through ROCclr's staging that copy blocks the host
// (decode windows of 150+ ms); instead the batch's pages are locked here for the async copies and unlocked when they
// have landed (unpin_blobs), so the release that follows can hand them back.  Neighbouring experts share a page and
// a copy from a range only PART of which is registered fails ("invalid argument"), so the page ranges are merged
// first and each merged span is registered once.  A span that does not register is copied through the staging
// path.  Portable: a layer split copies to either card.
// Whether pin_blobs locks anything (AMD on Linux with STRATA_ARENA_MMAP=1): the callers only gather the swaps' spans
// then - `blob()` is not free (it counts reads, and the file tier may assemble the blob), so the default runs skip it.
bool pin_blobs_on() {
#if (defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)) && !defined(_WIN32)
    static const bool on = std::getenv("STRATA_ARENA_MMAP") && std::getenv("STRATA_ARENA_MMAP")[0] == '1';
    return on;
#else
    return false;
#endif
}
// #884 (HIP, opt-in STRATA_HIP_ADAPT_KERNEL_COPY=1): an adaptive swap's host-to-device copy as a kernel on the refill stream
// instead of a copy-engine (SDMA) memcpy.  2x RX 6900 XT hung after a long prompt only with the MMQ prompt path, adaptive swaps
// AND SDMA together (any one off avoids it; the barrier packet that never passes is a ROCm-side problem, #884).  This keeps
// the swaps off SDMA without turning SDMA off for the prompt read (HSA_ENABLE_SDMA=0 costs 10-24% of it).  Needs the source to
// be mapped, 16-byte aligned memory (the registered arena); anything else takes the normal cudaMemcpyAsync.
dpct::err0 adapt_copy_h2d(void *dst, const void *src, size_t bytes,
                          dpct::queue_ptr st) try {
#if defined(STRATA_USE_HIP)
    static const bool kernel = [] {
        const char* e = std::getenv("STRATA_HIP_ADAPT_KERNEL_COPY");
        const bool on = e != nullptr && e[0] == '1';
        if (on) std::fprintf(stderr, "strata: STRATA_HIP_ADAPT_KERNEL_COPY=1: adaptive swaps copy with a kernel, not SDMA (#884)\n");
        return on;
    }();
    if (kernel && bytes % 16 == 0 && ((uintptr_t) src & 15) == 0 && ((uintptr_t) dst & 15) == 0) {
        void* dsrc = nullptr;
        if (cudaHostGetDevicePointer(&dsrc, const_cast<void*>(src), 0) == cudaSuccess && dsrc != nullptr) {
            strata::kernels::copy_from_mapped((float*) dst, (const float*) dsrc, (int64_t) (bytes / 4), (void*) st);
            return cudaGetLastError();
        }
        (void) cudaGetLastError();   // not registered / not mapped: the ordinary copy
    }
#endif
    /*
    DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    return DPCT_CHECK_ERROR(st->memcpy(dst, src, bytes));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
int pin_blobs(std::vector<std::pair<uintptr_t, uintptr_t>> r, std::vector<void*>& live) {
    int failed = 0;   // ranges left pageable (a copy from them still works, through the driver's staging)
#if (defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)) && !defined(_WIN32)
    if (!pin_blobs_on() || r.empty()) return 0;
    for (auto& [a, e] : r) { a &= ~(uintptr_t) 4095; e = (e + 4095) & ~(uintptr_t) 4095; }
    std::sort(r.begin(), r.end());
    size_t k = 0;
    for (size_t i = 1; i < r.size(); ++i) {
        if (r[i].first <= r[k].second) r[k].second = std::max(r[k].second, r[i].second);
        else r[++k] = r[i];
    }
    r.resize(k + 1);
    for (const auto& [a, e] : r) {
        if (cudaHostRegister((void*) a, (size_t) (e - a), cudaHostRegisterReadOnly | cudaHostRegisterPortable) ==
            cudaSuccess)
            live.push_back((void*) a);
        else {
            (void) cudaGetLastError();
            ++failed;
        }
    }
#else
    (void) r; (void) live;
#endif
    return failed;
}
void unpin_blobs(std::vector<void*>& v) {
    /*
    DPCT1027: The call to cudaHostUnregister was replaced with 0 because
    SYCL currently does not support registering of existing host memory for use
    by device. Use USM to allocate memory for use by host and device.
    */
    for (void *p : v)(void) 0;
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    (void)0;
    v.clear();
}

// STRATA_RSS_TRACE=1: the process's mapped file pages (the arena) at a startup step - who reads the arena back
void rss_probe(const char* where) {
    static const bool on = std::getenv("STRATA_RSS_TRACE") != nullptr;
    if (!on) return;
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    long long kb = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::strncmp(line, "RssFile:", 8) == 0) { kb = std::atoll(line + 8); break; }
    std::fclose(f);
    std::fprintf(stderr, "strata rss: %s: RssFile %.2f GiB\n", where, (double) kb / (1024.0 * 1024.0));
    std::fflush(stderr);
}

bool refill_blocking() {
    static const bool v = std::getenv("STRATA_REFILL_BLOCKING") != nullptr;
    return v;
}

using Clock = std::chrono::steady_clock;

// --pipeline-windows: every STRATA_PIPELINE_* test and tuning variable (THETA, FORCE_MISS, SWITCH, LOG, TRACE, DOOM_SKIP,
// LOOKUP_PON, PRESTAGE, AGREE, SNAP_OVERLAP) is read only with STRATA_PIPELINE_DEBUG=1; without it the defaults apply.
static const char* pipe_dbg_env(const char* name) {
    static const bool on = [] { const char* v = std::getenv("STRATA_PIPELINE_DEBUG"); return v != nullptr && v[0] != 0 && v[0] != '0'; }();
    return on ? std::getenv(name) : nullptr;
}

// The resident RAM mode and the adaptive tier.  A swap copies `in` (held in RAM) into the slot of `out` (held only
// by that slot).  Before the slot is overwritten, `out`'s bytes are copied back from it into an exchange buffer, so
// the CPU computes `out` from RAM while the swap is in flight; when the swap has landed, `commit_exchanges` moves
// them into `in`'s place in RAM.  The RAM copy then again holds exactly the experts no core slot does, and no swap
// reads the file.  Swaps that need no exchange (`out` in the lend region is held in RAM already; or `in` is not) go
// on as before; ones beyond the buffers' room wait for a later round.  Runs on the adaptive tier's thread while the
// GPU commits and drafts: the copies back are on its stream, and waited for before the refills are queued.
//
// With a layer split, `host_res` holds slot numbers of WHICHEVER cache owns the layer, so the copy back must read the
// owning card's cache on that card's stream: `locate(layer)` names them (cache, stream, device; -1 = CUDA0).  Reading
// a later stage's slot number out of CUDA0's cache copied another expert's bytes into RAM - the resident mode with a
// split then computed some experts from the wrong weights after the first adaptive swaps (garbage output).
struct SwapHome {
    strata::core::ExpertCache* cache;
    dpct::queue_ptr stream;
    int dev;
};
template <class Swap, class Locate>
bool resident_stage_swaps(strata::core::FileExpertSource &src,
                          const std::vector<int32_t> &host_res,
                          int64_t n_expert, std::vector<Swap> &swaps,
                          Locate locate) try {
    if (!src.complement_ready() || swaps.empty()) return true;
    struct Staged { int32_t layer, in, out; int64_t q; };
    std::vector<Staged> staged;
    std::vector<SwapHome> used;
    std::vector<Swap> kept;
    kept.reserve(swaps.size());
    for (const Swap& s : swaps) {
        if (!src.has_resident(s.layer, s.in) || src.has_resident(s.layer, s.out)) { kept.push_back(s); continue; }
        const int64_t q = (int64_t) staged.size();
        if (q >= src.exchange_capacity()) continue;
        const int32_t slot = host_res[(size_t) s.layer * (size_t) n_expert + (size_t) s.out];
        if (slot < 0) continue;
        const SwapHome home = locate(s.layer);
        const strata::core::OnDevice on(home.dev);
        // the evicted expert's bytes go back into exchange buffer q BEFORE its slot is overwritten
        // (CUDA: cudaMemcpyAsync D2H on the owning card's stream; the dpct migration had dropped the call)
        try {
            home.stream->memcpy(src.exchange_buffer(q), home.cache->device_slot(slot),
                                (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer));
        } catch (sycl::exception const& e) {
            std::fprintf(stderr, "strata: copy back of layer %d slot %d (device %d) failed: %s\n", (int) s.layer,
                         (int) slot, home.dev, e.what());
            return false;
        }
        bool seen = false;
        for (const SwapHome& u : used) seen = seen || u.stream == home.stream;
        if (!seen) used.push_back(home);
        staged.push_back({s.layer, s.in, s.out, q});
        kept.push_back(s);
    }
    if (!staged.empty()) {
        for (const SwapHome& u : used) {
            const strata::core::OnDevice on(u.dev);
            try {
                u.stream->wait();
            } catch (sycl::exception const& e) {
                std::fprintf(stderr, "strata: copy back sync (device %d) failed: %s\n", u.dev, e.what());
                return false;
            }
        }
        // STRATA_CHECK_SWAPS=1 (a test aid): each evicted expert's exchange buffer must hold the model file's bytes
        if (const char* chk = std::getenv("STRATA_CHECK_SWAPS"); chk != nullptr && chk[0] != '\0' && chk[0] != '0') {
            static std::atomic<int64_t> n_ok{0}, n_bad{0};
            std::vector<uint8_t> ref;
            int64_t bad = 0;
            for (const Staged& x : staged) {
                ref.resize((size_t) strata::kernels::cpu::expert_layout().blob_bytes(x.layer));
                if (!src.copy_blob_from_file(x.layer, x.out, ref.data()) ||
                    std::memcmp(ref.data(), src.exchange_buffer(x.q), ref.size()) != 0) {
                    ++bad;
                    std::fprintf(stderr, "strata: STRATA_CHECK_SWAPS: layer %d expert %d: the exchange buffer differs from the file\n",
                                 (int) x.layer, (int) x.out);
                } else {
                    ++n_ok;
                }
            }
            n_bad += bad;
            std::fprintf(stderr, "strata: STRATA_CHECK_SWAPS: %zu staged, %lld differ (total %lld ok, %lld differ)\n",
                         staged.size(), (long long) bad, (long long) n_ok.load(), (long long) n_bad.load());
        }
        for (const Staged& x : staged)
            if (!src.stage_exchange(x.layer, x.in, x.out, x.q)) {
                std::fprintf(stderr, "strata: stage_exchange refused layer %d in %d out %d\n", (int) x.layer,
                             (int) x.in, (int) x.out);
                return false;
            }
    }
    swaps.swap(kept);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

struct Options {
    std::string pack = "pack/full";
    std::vector<int64_t> tokens;      // the prompt, PRE-TOKENIZED
    int64_t max_new = 16;
    int64_t max_context = 4096;
    bool greedy = true;
    uint64_t seed = 0;
    int top_k = 20;
    float top_p = 0.95f;
    float temperature = 1.0f;
    std::string dump_logits;          // one line of logits per generated position
    int64_t logits_stride = 1;        // storage selection; all prompt tokens remain conditioned
    /// **THE RESIDUAL, SO THE HEAD CAN BE CHECKED WITHOUT THE LAYERS.**
    ///
    /// C1 fails (LEDGER L116) and the pipeline is `embed -> 48 layers -> head`.  Dumping `R` splits it in half:
    /// the head is one norm, two bf16 projections and one 794 MB GEMV, all of which can be recomputed in Python
    /// from the manifest.  If Python agrees with the engine on the same `R`, the head is right and the layers
    /// are wrong; if it disagrees, the head is wrong.  Nothing else in the engine can be split that cheaply.
    std::string ple_gguf;              // the ORIGINAL second GGUF shard: the PLE table is not in the pack
    bool no_ple = false;              // explicit diagnostic ablation; never a normal inference default
    bool stream_token = false;        // R2.6 experiment: ordered work on the session stream
    bool check_logits = false;        // optional full-vocabulary finite scan
    bool gr_fp32_activations = false;  // pinned CUDA single-token BF16 activation contract
    bool gr_native_mmvf = false;       // pinned projection reduction tree as well as FP32 inputs
    bool native_bf16 = false;          // SSM gates, router and indexer projections only
    bool native_bf16_extra = false;    // PLE value and shared expert scalar gate
    bool native_ple_key = false;       // unchanged Q2_0 key and CUDA Q8_1 activations
    bool native_moe_combine = false;   // pinned fused CUDA weighted reduction
    bool native_gdn = false;          // pinned CUDA recurrence and preprocessing
    bool native_flash_attn_short = false; // diagnostic pinned attention, context <=256
    bool native_qsa_indexer = false;  // pinned F16 key cache and F32 pooling
    bool native_qsa = false;          // pinned F32 QSA norms and gate
    /// THE CONTEXT EXTENSION (rope scaling, rope_scaling.hpp).  These knobs resolve to ONE process config,
    /// set once before `session_init` builds the rope table and the graphs capture the kernels.  There is
    /// deliberately no per-request form: K sits in the cache POST-RoPE, so one cache must never mix two
    /// scalings.  Precedence: an EXPLICIT flag over the model file's rope keys over the struct defaults -
    /// `none` and `1` are explicit values (the opt-outs), the absent flag is not.
    std::string rope_scaling;           ///< --rope-scaling none|linear|yarn (llama.cpp's names); empty = the flag is absent
    double rope_scale = 0;              ///< --rope-scale F: the extension factor; 0 = the flag is absent (the model file's, else 1 = off)
    double rope_freq_base = 0;          ///< --rope-freq-base N: 0 = the model's (1e7)
    double rope_freq_scale = 0;         ///< --rope-freq-scale F: the raw ggml knob; 0 = 1/--rope-scale
    double yarn_orig_ctx = 0;           ///< --yarn-orig-ctx N: 0 = the model's, else 262144
    double yarn_ext_factor = -1.0;      ///< --yarn-ext-factor F: <0 = auto (1 for yarn, 0 otherwise)
    double yarn_attn_factor = 1.0;      ///< --yarn-attn-factor F
    double yarn_beta_fast = 32.0;       ///< --yarn-beta-fast F
    double yarn_beta_slow = 1.0;        ///< --yarn-beta-slow F
    bool native_rope = false;         // pinned text-only CUDA rotary arithmetic
    bool native_ple_postops = false;  // pinned PLE postprojection arithmetic
    bool native_router = false;       // pinned fused 512-expert top-10 router
    bool cpu_oracle_q8_0 = false;      // pinned x86 activation scales/codes at both expert stages
    std::string native_head_gguf;      // native output.weight experiment; same model shard as the pack
    std::vector<std::string> native_dense_gguf; // repeat for native GDN/QSA projection shards
    /// Every shard of --native's model (strata::gguf_split_paths: the metadata shard first; a missing shard is an
    /// error) and of --native-head-gguf's (the same list unless that names another model).
    std::vector<std::string> native_shards, native_head_shards;
    /// Plan v0.3 P1: the whole native arithmetic set as ONE switch (model shard 1). It enables exactly the
    /// combination recorded in bench/results/2026-09-23-attention-ple plus the native indexer, and never the
    /// <=256-token attention adapter. It becomes the default once P0 shows it is not slower.
    std::string native_preset;
    /// The token embedding from this GGUF instead of --native's (tools/embd_bf16_pack.py: BF16 as shipped)
    std::string embd_gguf;
    /// Plan v0.3 P2: how the n-gram table is read. Direct (default) = unbuffered SSD reads, table never in RAM.
    std::string ple_io = "direct";
    bool ple_io_explicit = false;      ///< --ple-io was given: the engine only warns when direct measures slow
    int64_t ple_row_cache = 1 << 20;   ///< bounded row cache (rows of 90 B); 0 disables
    int ple_inflight = 256;   // the prompt path reads a chunk's rows at once: 64 left the SSD half idle (32K: 303 -> 189 ms)
    double ple_delay_us = 0;           ///< fault injection: every row read completes no earlier than this
    bool ple_sync_submit = false;      ///< A/B arm: submit reads on the token thread, no I/O worker
    std::string kv = "fp16";           ///< plan v0.3 P7: KV storage, fp16 (default) or int8 (half the VRAM)
    int64_t kv_resident = 0;           ///< KV streaming: resident cells per QSA layer (0: all in VRAM)
    bool kv_grow = false;              ///< the elastic K/V (--kv-grow, STRATA_KV_GROW=1; kvg_ensure)
    std::string dump_residual;
    /// The head input, `bb.mixed`.  It exists so the head can be SPLIT: steps 1-4 (the per-stream norm, the two
    /// bf16 projections and the stream mean) recompute cheaply in Python, and only the 794 MB GEMV does not.
    std::string dump_mixed;
    /// One residual snapshot per layer per position: `n_layers * hc * n_embd` floats per position, appended in
    /// position order.  This is the C1 BISECTION LADDER - it is what `llama-debug --tensor-filter l_last` prints
    /// for the reference, so the first layer whose `sum` diverges is the layer that holds the bug.  It needs the
    /// captured path (the expert pool only exists there), so it is refused with `--no-capture`.
    std::string dump_layers;
    /// `2 * n_embd + 2 * hc` floats per layer per position: the attention half's block output, the MoE half's,
    /// and the two injection vectors.  It separates `linear_attn_out-<l>` from `ffn_out-<l>`, which the residual
    /// ladder cannot.  **CAPTURED INTO THE LAYER GRAPHS**, so it must be armed before `session_capture`.
    std::string dump_halves;
    /// P0.S8's routing trace, and a prerequisite the Phase 3 plan names explicitly.  One record per layer per
    /// position: `int32 layer, int32 k, k int32 ids, k float weights`.  It is what a hit-rate curve for a
    /// candidate VRAM expert cache is computed from, and it needs no new kernels - the doorbell already
    /// publishes exactly this much to pinned memory.
    std::string dump_routing;
    bool no_capture = false;          // run the layers directly instead of replaying graphs
    bool no_pool = false;             // skip the CPU expert pool: the GPU-only floor
    bool sync_every_layer = false;
    /// Per-stage CUDA-event timings inside the layer halves.  `--no-capture` only: an event recorded inside a
    /// stream capture is silently dropped, so the captured path cannot carry this.
    bool stage_timing = false;
    /// Launch the 48 captured `pre` graphs back to back with no host work between them and report the pure GPU
    /// time per token.  This is the only measurement that separates host-bound from GPU-bound, because the
    /// stage events include every gap where the GPU waited for the host.
    bool graph_only = false;
    bool gpu_only_full = false;   ///< R0.3: pre + post + head, the true per-token GPU floor
    int pool_workers = 0;         ///< R2.2: 0 = "all physical cores minus the host's"; >0 overrides
    /// #272: the pool's core layout; `all` (the default) is the layout it always had, auto / p-cores are opt-in
    strata::kernels::cpu::PoolAffinity pool_affinity = strata::kernels::cpu::PoolAffinity::All;
    /// --host-core first|last (STRATA_HOST_CORE): the host thread's core (see HostCore in pool.hpp)
    std::string host_core;
    /// --pool-cores LIST (STRATA_POOL_CORES; experimental, off by default): the logical processors the pool workers are pinned to
    std::string pool_cores;
    /// R2.2's first half, as an A/B arm.  **ON by default**, because the measurement that justifies it is the
    /// pool's own drain: 33.7 GB/s against 5/6 x 44.14 = 36.8 for five workers, on a machine whose sixth core
    /// is reserved for a host thread that has nothing to do while the drain runs.
    bool no_host_worker = false;
    bool coupled_draft = strata::core::coupled_draft_env(); ///< Coupled draft sampling for MTP drafter under sampling
    bool mmap_experts = false;    ///< R2.1: opt OUT of the resident arena, back to MapViewOfFile
    std::string shared_expert_arena; ///< Linux: optional file backing for the resident arena shared by processes
    bool resident_cpu_experts = false; ///< mmap-backed static-cache misses copied into ordinary RAM
    bool stream_experts = false;  ///< SYCL port: no host arena; blobs read from the GGUF on demand (all experts in VRAM)
    /// `--resident-experts` (the low-RAM PC's resident mode, chosen by setup): `--resident-cpu-experts` with the copy
    /// page-locked when the driver allows (else locked in the working set), 4 GiB of RAM headroom, and plain mmap
    /// (with a warning) when even the experts no slot holds do not fit.
    bool resident_pin = false;
    uint64_t resident_headroom = 8ull << 30;
    bool resident_soft = false;
    bool resident_cpu_explicit = false;   ///< #384: --resident-cpu-experts given by itself (not only implied)
    /// CS-T `--resident-budget-gib N`: the resident mode with a RAM budget - the N GiB of experts the GPU cache does
    /// not hold that the expert profile ranks hottest are copied into RAM, the rest are read from the files in place
    /// (the GGUF shards when the pack has no experts.bin).  0 = the whole complement (--resident-experts).
    uint64_t resident_budget = 0;
    /// R4: slots of VRAM-resident experts.  **0 = off, and off is the default.**
    /// **THE COMMENT THAT USED TO BE HERE WAS FALSE AND ROUND 328 MEASURED IT.**  It said "the cache has no
    /// consumer yet - `moe_hit_grouped_s2` does not exist - so switching it on costs the fill traffic and
    /// saves nothing".  The kernel exists (`src/kernels/cuda/s2_expert_grouped.cu`), it is wired at line ~660
    /// via `expert_hit_run`, and switching the cache on **does** move work off the CPU pool: the drain fell
    /// **19.076 -> 10.312 ms/token** at 4096 per-layer slots, for **-2.7 ms/token** end to end.  What was
    /// true is that the ADMISSION POLICY gave every slot to the first position, which is why the earlier
    /// measurement found nothing - see `expert_cache_per_layer`.
    int expert_cache = 0;
    std::array<int, 3> expert_cache_remote{}; ///< CUDA1..3 slots; CUDA0 keeps dense/state/MTP
    std::array<bool, 3> expert_cache_remote_auto{};
    bool remote_expert_opt = false; ///< optional optimization of the existing remote-expert decode path
    std::string expert_cache_remote_placement = "stripe"; ///< stripe experts or assign complete layers to CUDA1..3
    bool expert_cache_cpu_order = false;
    /// **R4.2g.  ROUND 328 MEASURED THAT THE GLOBAL ADMISSION POLICY CANNOT WORK, AND THIS IS THE FIX.**
    /// The default policy hands out slots in arrival order from one counter shared by all 48 layers, so the
    /// first `n_slots` distinct pairs - about 26 LAYERS OF POSITION 0 - take every slot and hits are confined
    /// to them.  Measured at 256 slots: **1781 of 60000 = 2.97%**, against **21.4%** for 8 slots per layer and
    /// **70.4%** for 64, from `Memory/cache_allocation.py` on the same run's routing.  Off by default.
    bool expert_cache_per_layer = false;
    /// Multi-GPU: a second expert tier on CUDA device `peer_device` (-1 = off), `peer_reserve_mib` left free
    /// there, `peer_slots` caps its size (0 = as many as fit), `peer_adapt_swaps` per adaptive round
    /// (-1 = adapt_swaps).
    int peer_device = -1;
    int peer_reserve_mib = 600;
    int64_t peer_slots = 0;
    int peer_adapt_swaps = -1;
    /// the prompt path's rows per layer the peer computes: -1 = half of chunk x top-k, 0 = the prompt path stays on
    /// the primary
    int64_t peer_prefill_rows = -1;
    /// The PLE gather's prefetch, as an A/B arm.  The gather measured 2.10-2.61 ms/token because its sixteen
    /// row reads are sixteen SEPARATE page faults into a 26.8 GB mapping; see `ple_prefetch_enable`.
    bool no_ple_prefetch = false;
    /// R4.2e: a `profile.bin` from `tools/make_profile.py`.  **When given, it decides residency instead of the
    /// compulsory-miss policy**, which is the whole point: a profile ranked by routing frequency over a whole
    /// trace is what the plan's `h = 0.6447` refers to, and compulsory-miss measured 0.4864 because it fills
    /// with whatever the prompt touched FIRST.  Empty means no profile.
    std::string expert_profile;
    /// #477 (--serve, opt-in): where to save what the adaptive tier learned, as a profile `--expert-profile` reads
    /// (the resident experts first, then the routing counted since the start); on QUIT and every
    /// `expert_profile_save_min` minutes between requests.  Empty (the default): nothing is counted or written.
    std::string expert_profile_save;
    double expert_profile_save_min = 10.0;
    /// R4.2d: **ON by default**, because the measurement is unambiguous and the alternative is known-broken.
    /// Without it, 17 of 10,562 layers had the hit work done when the pool returned; with it, 9,190.  The
    /// A/B arm is `--no-hit-poke`.
    bool no_hit_poke = false;
    /// R0.9: capture each layer as THREE graphs and time them from outside the capture, which is the only
    /// valid way to get a per-stage table on the real graph.  Prints and exits; it is a measurement, not a run.
    bool gpu_stages = false;
    bool stats = false;
    bool shared_late = false;          ///< plan v0.3 P3 A/B: shared expert inside post[l] (old order)
    bool keep_canonical = false;       ///< plan v0.3 P1 A/B: load canonical copies of natively served tensors
    bool no_token_graph = false;       ///< plan v0.3 P3 A/B: two graphs per layer instead of one per token
    bool no_fused_gr = false;          ///< plan v0.3 P3 A/B: the six-kernel native gr_read + separate gr_write
    bool no_fast_attn = false;         ///< plan v0.3 P3 A/B: gather + one-block-per-head QSA attention
    bool no_publish_kernel = false;    ///< plan v0.3 P3 A/B: memcpy nodes for the doorbell and QSA step
    bool no_fused_gdn = false;         ///< plan v0.3 P3 A/B: llama.cpp-layout GDN step + separate out norm
    bool no_fast_select = false;       ///< plan v0.3 P7 A/B: FP64 row scores + bit-serial cell top-k
    /// Plan v0.3 P4: `--expert-cache auto` sizes the VRAM tier from what is free after the weights, the session
    /// and the KV state, minus this reserve for the graphs, the hit scratch and the head.
    int vram_reserve_mib = 700;
    bool vram_reserve_given = false;   ///< --vram-reserve-mib on the command line (#496: no smaller automatic reserve)
    /// #533 (opt-in): the expert cache in physical segments (CUDA virtual memory management), so the serve loop's
    /// `VRAM <reserve_mib>` command can give part of it back to other programs and take it again.  Off: one cudaMalloc.
    bool vram_elastic = false;
    int64_t vram_segment_mib = 512;
    /// `--vram-reserve-later-mib N`: the reserve on a layer split's later cards (default: the same as the first).
    /// A card that drives no display needs less than the one the monitors are on.
    int vram_reserve_later_mib = -1;
    /// Plan v0.3 P5: batched prompt processing in chunks of this many tokens (0 = the token path).
    int64_t prefill_chunk = 0;
    /// `--prefill auto`: the largest chunk (up to 8192) whose buffers the expert cache can lend.  Every expert a chunk
    /// routes to is streamed once per chunk, so a bigger chunk streams fewer bytes per token (the "ubatch" effect).
    bool prefill_auto = false;
    /// #282, opt-in: the largest chunk `--prefill auto` may take - 8192 by default; `--prefill auto:16384` or
    /// `auto:32768` (or STRATA_PREFILL_AUTO_MAX) lets it go further, never past the context
    int64_t prefill_auto_max = 8192;
    bool no_split_rows = false;        ///< plan v0.3 P4 A/B: one whole expert per pool thread
    /// Plan v0.3 P5: the prompt path borrows the top expert-cache slots for its buffers and refills them after
    /// the prompt (default); `--no-prefill-borrow` reserves the buffers' VRAM for the whole session instead.
    bool no_prefill_borrow = false;
    /// Plan v0.3 P5 validation: batch only positions [0, P) and run the rest of the prompt through the token path
    /// (teacher-forced), so the logits of positions >= P - which depend on the batched state - can be scored
    /// against the oracle at many positions.  0 = the whole prompt but the last position.
    int64_t prefill_until = 0;
    /// Plan v0.3 P6: after every processed position, append the residual after the last layer (hc x n_embd
    /// floats, the MTP draft head's input) to this file.  Token path only.
    std::string dump_final_r;
    /// Plan v0.3 P6: speculative decoding with a verify window of this many tokens (the last accepted token and
    /// spec-1 drafts); 0 = plain decode.  `spec_oracle` drafts from a token file (the expected continuation, for
    /// the exactness test); `spec_corrupt` N > 0 replaces every Nth draft with a wrong token.
    int spec = 0;
    /// --serve runs up to this many sequences at once, one token each per batch window (0 = off).  A request
    /// sent as `BGEN <slot> <max_new> ...` reads its prompt and its first token through the usual path, then
    /// continues in slot <slot> of the batch windows (`BT <slot> <id>` lines, then `BDONE <slot> ...`).
    int batch = 0;
    /// With an explicit --layer-split, every GPU loads only its own layers' dense weights instead of the whole
    /// model's: the VRAM they took goes back to the expert cache (opt-in)
    bool trim_stage_weights = false;
    /// The --batch slots in this many groups pipelined through the stages of a layer split (stage k
    /// runs one group while stage k+1 runs another).  1 = every slot in one window, stage after stage.
    int batch_groups = 1;
    bool batch_mtp = false;      ///< --batch-mtp / STRATA_BATCH_MTP=1 (opt-in): one MTP proposal per batch slot
    std::string spec_oracle;
    int spec_corrupt = 0;
    /// Benchmarks and A/B checks (eddoursul's fork, F19): the run emits this continuation (token ids) instead of
    /// the window's argmax, and a draft is accepted when it matches it, so runs with different speculation
    /// settings process the same text.
    std::string spec_follow;
    /// Per verify window: a line with its size, a hash of its final residual rows and its argmaxes.
    std::string window_hashes;
    /// Plan v0.3 P6: the MTP draft layer's runtime directory (tools/mtp_rt.py); drafts come from it.
    std::string mtp;
    int64_t mtp_window = 32768;   ///< the draft layer attends to the last N cells (0 = every cell)
    /// Plan v0.3 P6: the share (0..1) of each layer's distinct missed experts the GPU reads over PCIe from the
    /// pinned arena while the CPU computes the rest (verify windows).
    double pcie_frac = -1.0;   ///< < 0: the model's default (0.2 direct for the Q2_0 pack, 0.55 DMA for native packs)
    bool pcie_frac_auto = false;     ///< STRATA_PCIE_FRAC_DEFAULT=auto, single card, no --pcie-frac: the share is refined in the first decode windows (PcieAuto)
    double pcie_probe_gbps = 0.0;    ///< the start-up probe's reading (for PcieAuto)
    std::string pcie_mode = "auto";   ///< auto | dma | kernel | direct
    /// Plan v0.3 P6: every `adapt_every` rounds, swap up to `adapt_swaps` of the most-routed missing experts into
    /// the VRAM tier in place of the least-routed resident ones (decayed counts).  0 = static residency.
    int adapt_every = 4;
    float adapt_decay = 0.7f;   ///< the usage counts are multiplied by this after each adaptation (--adapt-decay)
    /// --adapt-async 1 (--serve with the resident RAM mode; opt-in): the adaptive tier's rounds advance between verify
    /// windows on a helper thread instead of one window waiting for a whole round (see the tier in the --serve block).
    /// Not bit-exact run to run.  0 = the blocking tier.
    int adapt_async = 0;
    /// Plan v0.3 P6: a draft enters the verify window only while every draft before it (and itself) has at least
    /// this probability under the draft layer; 0 = always --spec-1 drafts.
    double spec_min_p = 0.0;
    /// Stop when the model emits an end-of-turn token (<|endoftext|> 248044, <|im_end|> 248046, or --eos-ids).
    bool stop_eos = false;
    std::vector<int64_t> eos_ids = {248044, 248046};
    bool spec_split = false;   ///< opt-in split verify window (the overlap study: exact, ~7% slower)
    /// --serve, multi-GPU layer split: "K" or "K1,K2,.." (the first layer of each later stage) or "auto" (placed
    /// from each GPU's free VRAM); empty = one GPU
    std::string layer_split;
    /// the later stages' devices "D1,D2,.." (default: the next visible GPUs; "0" with one K: both stages on this
    /// GPU, sharing everything - the bit-exact A/B of the hand-off)
    std::string split_device;
    /// --split-skip-if-fits (opt-in; the server passes it for a config's "split_skip_if_fits"): with
    /// --layer-split auto, run on CUDA0 alone when it holds every profiled expert pair plus the whole session (KV),
    /// the drafter and the reserve - a split then only adds hand-offs (two RDNA4 cards: 1,384 vs 1,794 tok/s at 4K)
    bool split_skip_if_fits = false;
    /// Plan v0.3 P8: stay resident and take requests on stdin (see the --serve block in main).
    bool serve = false;
    /// The vision path: keep a per-cell (t, h, w) rotary position table so --serve can take GENI requests.
    bool vision = false;
    int adapt_swaps = 96;
    /// --serve: how many conversation checkpoints to keep between requests (0 = every request reads its whole
    /// prompt again, the v0.1.2 behaviour).  One is the GDN recurrence of the 36 layers, the QSA indexer tails and
    /// the PLE history (~118 MB of host RAM); the KV cache itself is positional and stays where it is.
    int prompt_cache = 6;
    int64_t conversation_cache_mib = 0; // opt-in host RAM for independent conversations
    int conversation_cache_slots = 4;
    int64_t conversation_cache_min_free_mib = 2560;
    /// --serve SAVE: disk space a session file must leave free where it is written (MiB; 0 = no check)
    int64_t session_min_free_mib = 4096;
    /// --serve: also keep a checkpoint every N freshly read prompt tokens (0 = only at the last turn boundary)
    int64_t prompt_cache_every = 16384;
    bool prompt_cache_tail = false;   // optional extra checkpoint at an existing near-tail chunk boundary
    /// --serve: a prompt read from token 0 is also checkpointed at its first turn boundary - the end of the system
    /// prompt, which every chat of the same client shares - when that is at least N tokens (0 = never)
    int64_t prompt_cache_root = 2048;
    /// --serve: the token that opens a chat turn (<|im_start|>).  The last one in a prompt is where the chat's
    /// history ends and the new assistant turn begins, which is the checkpoint the next request can reuse.
    int64_t turn_token = 248045;
    /// --serve (#458, opt-in): the role token of a short turn the server puts right before the new assistant turn
    /// (the "system" of a trailing reasoning-effort turn).  When the turn before the last <|im_start|> opens with it,
    /// the checkpoint goes in front of that turn instead, so it holds the conversation and not the effort text.
    /// -1 = off (the checkpoint at the last <|im_start|>, as before).
    int64_t tail_role_token = -1;
    /// --serve: a text part of the prompt of at most N tokens (a chat message, the assistant header, a short tool
    /// result) goes through the verify windows, S tokens at a time, instead of the batched prompt path (0 = always
    /// the batched path)
    int64_t short_read = 64;
    /// --pipeline-windows N (--serve, a layer split on exactly two GPUs; opt-in): 1 = a prompt's short reads through
    /// the verify windows run with the stages overlapped (stage 0 reads window K+1 while stage 1 reads window K);
    /// 2 = the decode windows too: stage 0 runs window K+1 speculatively while stage 1 verifies window K (rolled back
    /// when K is not accepted as guessed).  0 = off (the stages take turns).
    int pipeline_windows = 0;
    /// The suffix drafter (prompt lookup): when the text being written repeats an earlier stretch of the context (code
    /// edits, quoted input, tool-call JSON) by at least this many tokens, the window may be filled with what followed
    /// it there instead of the MTP's drafts, where the MTP's own first guess agrees and the draft policy expects it to
    /// pay (strata/spec/draft_policy.hpp).  On by default; 0 = MTP only.
    int suffix_draft = 3;
    /// The MTP's own window cap (0 = --spec): with --spec 6 --mtp-max-t 4 the long windows come from suffix matches.
    int mtp_max_t = 0;
    /// --lookup-chain K (opt-in, 0 = off): prompt lookup CHAINED after the MTP's proposal.  The suffix drafter matches
    /// the context followed by the MTP's drafts and appends up to K of the tokens that followed that match to the same
    /// verify window (the window grows to at most 8).  Greedy verification keeps the output; its own counts are kept.
    int lookup_chain = 0;
    int lookup_chain_min = 3;   ///< --lookup-chain-min M: the shortest match (in tokens, drafts included) it extends on
    /// --mtp-hnorm stream (opt-in): the draft layer's pre_fc_norm_hidden normalizes each hyper-connection stream on
    /// its own (llama.cpp's qwen4exp MTP graph) instead of one RMS over all four (the default, tools/mtp_probe.py).
    bool mtp_hnorm_stream = false;
    /// --mtp-q4 (opt-in): the draft layer's projections and draft head from 4-bit copies (fewer bytes per draft step)
    std::string mtp_q4;   ///< "" (off), all, proj (the projections only) or head (the draft head only)
    /// --mtp-draft-vocab FILE (opt-in): the draft head's token subset from FILE instead of <mtp>/draft_vocab.bin
    std::string mtp_draft_vocab;
    /// A control vector on the residual stream (strata/kernels/cvec.hpp), with llama.cpp's flags: the
    /// `experimental-speed-projection` profile passes `--control-vector-scaled FILE:1.0 --control-vector-layer-range
    /// 4 44 --cvec-mode project --cvec-dir per-layer`.  None by default; --serve switches a loaded one per request.
    std::vector<std::pair<std::string, float>> cvec_files;
    int cvec_first = -1, cvec_last = -1;   ///< llama.cpp's defaults: 1 .. the last layer
    int cvec_mode = 1;                     ///< 0 = project, 1 = add (llama.cpp's default)
    int cvec_single = -1;                  ///< --cvec-dir single:L (project mode): layer L's direction everywhere
};

void usage() {
    std::fprintf(stderr,
                 "strata generate --pack DIR --tokens \"1,2,3\" [options]\n"
                 "\n"
                 "  --pack DIR           the pack directory (default pack/full)\n"
                 "  --tokens LIST        the prompt as comma-separated token IDS (required)\n"
                 "  --tokens-file PATH   pretokenized prompt, commas or whitespace (alternative to --tokens)\n"
                 "  --ple-gguf PATH      required PLE table (original second GGUF shard); with --native, the model's\n"
                 "                       shard that holds per_layer_token_embd.weight when not given\n"
                 "  --no-ple             explicit diagnostic ablation of the PLE layer\n"
                 "  --ple-io direct|mmap|ram  n-gram table reads (plan v0.3 P2). direct (default): unbuffered SSD\n"
                 "                       reads, the table never enters RAM or the file cache; mmap: A/B arm;\n"
                 "                       ram: mmap with the whole table locked in RAM at start (Linux/macOS)\n"
                 "  --ple-row-cache N    bounded cache of fetched rows, 90 B each (default 1048576; 0 = off)\n"
                 "  --ple-inflight N     outstanding SSD reads (default 256)\n"
                 "  --ple-delay-us U     fault injection: each row read completes no earlier than U us\n"
                 "  --ple-sync-submit    A/B arm: submit table reads on the token thread (default: an I/O thread)\n"
                 "  --kv fp16|int8       KV storage (plan v0.3 P7): int8 codes + fp16 scale per 64 values, half the\n"
                 "                       VRAM; default fp16 until gate G-C accepts int8\n"
                 "  --kv q4_0            4-bit K/V after a Hadamard rotation (PR #21): half of int8's memory,\n"
                 "                       slightly lower precision (see bench/results/2026-09-27-kv-q4)\n"
                 "  --kv k8v4            hybrid: INT8 K (exact attention scores) + rotated Q4_0 V, 816 B/cell\n"
                 "                       (vs int8's 1,056); streams with --kv-resident too\n"
                 "  --kv-resident N      KV streaming: keep N cells of each QSA layer in VRAM (min 20480) and the\n"
                 "                       whole K/V in pinned RAM; the freed VRAM goes to expert slots. 0 (default):\n"
                 "                       all of it in VRAM. A context of N cells or fewer is not streamed\n"
                 "  --kv-grow            the K/V takes VRAM only for the cells the requests reach and the expert\n"
                 "                       cache holds the rest, giving slots back as the context grows (one GPU, with\n"
                 "                       --expert-profile; STRATA_KV_GROW=1/0 also). Default: the whole --max-context\n"
                 "                       allocated at start\n"
                 "  --stream-token       enqueue token work on the session stream (experimental)\n"
                 "  --check-logits       copy and check all logits in the stream-token path\n"
                 "  --gr-fp32-activations  experimental CUDA-oracle GR activation precision\n"
                 "  --gr-native-mmvf      experimental pinned GR norm/projections; implies FP32 activations\n"
                 "  --native-bf16         experimental CUDA-oracle SSM/router/indexer BF16 projections\n"
                 "  --native-bf16-extra   experimental CUDA-oracle PLE/shared gate BF16 projections\n"
                 "  --native-ple-key      experimental native PLE key; requires --native-dense-gguf\n"
                 "  --native-moe-combine  experimental pinned CUDA routed/shared combination\n"
                 "  --native-gdn          experimental pinned CUDA GDN norms/gates/recurrence\n"
                 "  --native-flash-attn-short  diagnostic pinned vector attention; --max-context <=256\n"
                 "  --native-qsa-indexer  experimental pinned indexer key cache and pooling\n"
                 "  --native-qsa          experimental pinned QSA normalization and output gate\n"
                 "  --native-rope         experimental pinned text-only CUDA rotary arithmetic\n"
                 "  --native-ple-postops  experimental pinned PLE postprojection arithmetic\n"
                 "  --native-router       experimental pinned CUDA 512-expert top-10 routing\n"
                 "  --cpu-oracle-q8-0     experimental pinned CPU expert quantization and dot reduction\n"
                 "  --native SHARD1      every full-context native path at once (plan v0.3 P1): stream-token,\n"
                 "                       GR MMVF, BF16, head, dense + PLE key, MoE combine, GDN, router, QSA,\n"
                 "                       indexer, RoPE, PLE postops, and the CPU q8_0 contract unless the\n"
                 "                       expert cache is on. Individual --native-* flags stay for A/B.\n"
                 "  --native-head-gguf PATH  native Q5_K head from model shard 1; requires --stream-token\n"
                 "  --embd-gguf PATH     the token embedding from this GGUF instead of --native's (tools/embd_bf16_pack.py:\n"
                 "                       BF16 as the checkpoint ships it; mapped host memory, no VRAM)\n"
                 "  --native-dense-gguf PATH native GDN/QSA/shared projections; repeat for each source model shard\n"
                 "  --expert-cache-cpu-order  experimental GPU expert reduction matching CPU order\n"
                 "  --max-new N          tokens to generate (default 16)\n"
                 "  --max-context N      KV/state capacity (default 4096)\n"
                 "  --rope-scaling T     extend the context past the trained one: none, linear\n"
                 "                       (position interpolation) or yarn - llama.cpp's types and names.\n"
                 "                       Default: the model file's rope keys, else none. Fixed at startup:\n"
                 "                       K in the cache is post-RoPE, so one run one scaling\n"
                 "  --rope-scale F       the extension factor for linear/yarn (default: the model file's\n"
                 "                       factor, else 1 = off)\n"
                 "  --rope-freq-base N   the raw ggml knobs: the frequency base (0 = the model's 1e7) and\n"
                 "  --rope-freq-scale F  the angle shrink (0 = 1/--rope-scale)\n"
                 "  --yarn-orig-ctx N    the trained context the correction targets (0 = 262144)\n"
                 "  --yarn-ext-factor F --yarn-attn-factor F --yarn-beta-fast F --yarn-beta-slow F\n"
                 "                       YaRN's knobs; defaults: -1 (auto: 1 for yarn), 1, 32, 1\n"
                 "  --greedy             argmax (the default)\n"
                 "  --seed S             enable sampling with this Philox seed\n"
                 "  --top-k N --top-p F --temperature F\n"
                 "  --dump-logits PATH   write one line of raw logits per position\n"
                 "  --logits-stride N    store every Nth row plus final input (default 1); N>1 requires --max-new 1\n"
                 "  --dump-residual PATH write the final R (hc x n_embd, f32) for head bisection\n"
                 "  --dump-layers PATH   write R after EVERY layer, per position: the C1 bisection ladder\n"
                 "  --dump-halves PATH   write both halves' block_out and inject per layer: the half bisection\n"
                 "  --dump-routing PATH  write the routed expert ids and weights per layer per position (P0.S8)\n"
                 "  --no-capture         run the layers directly instead of replaying graphs\n"
                 "  --shared-late        A/B: shared expert after the CPU pool (default: overlapped with it)\n"
                 "  --keep-canonical     A/B: also load canonical copies of natively served tensors (more VRAM)\n"
                 "  --vision             --serve takes images too (GENI requests; embeddings from strata-vision)\n"
                 "  --prompt-cache N     --serve: keep N conversation checkpoints between requests (default 6, ~118 MB\n"
                 "                       of RAM each; 0 = read every prompt from the start)\n"
                 "  --conversation-cache-mib N  --serve: RAM budget for parked conversations (default 0 = off)\n"
                 "  --conversation-cache-slots N  --serve: at most N parked conversations (default 4)\n"
                 "  --conversation-cache-min-free-mib N  --serve: physical RAM floor when parking or restoring a\n"
                 "                       session file (default 2560)\n"
                 "  --session-min-free-mib N  --serve: disk space a SAVE must leave free (default 4096; 0 = no check)\n"
                 "  --vram-elastic       --serve (#533, opt-in, NVIDIA): the expert cache in segments (--vram-segment-mib,\n"
                 "                       default 512), so the command `VRAM <reserve_mib>` (the server's POST /v1/vram) can\n"
                 "                       give VRAM back to other programs between requests and take it back later\n"
                 "  --batch N / --slots N  --serve (opt-in): up to N requests decode together in batch slots (2..8\n"
                 "                       normally; --batch-mtp waves more through eight-row windows),\n"
                 "                       each slot with its own session (VRAM like the main one); docs/BATCHING.md\n"
                 "  --batch-mtp          --batch (opt-in, one GPU, needs --mtp and --spec): each slot also verifies one MTP\n"
                 "                       proposal per window (STRATA_BATCH_MTP=1 does the same); needs VRAM per slot\n"
                 "  --batch-groups G     --batch with a layer split: the slots in G groups pipelined through the GPUs\n"
                 "  --pipeline-windows N --serve with a layer split on two GPUs (opt-in): one conversation's verify windows\n"
                 "                       with the GPUs overlapped - 1 = the prompt's short reads, 2 = decode as well\n"
                 "                       (stage 0 runs the next window while stage 1 verifies this one); docs/MULTI_GPU.md\n"
                 "  --trim-stage-weights an explicit --layer-split: each GPU loads only its own layers' dense weights\n"
                 "  --prompt-cache-every N  --serve: also checkpoint every N fresh prompt tokens (default 16384, 0 = off)\n"
                 "  --prompt-cache-tail  --serve, single GPU: one extra checkpoint near the prompt's end, at an existing\n"
                 "                       chunk boundary (default off; requires --prompt-cache and --prompt-cache-every > 0)\n"
                 "  --turn-token ID      --serve: the token that opens a chat turn (default 248045, <|im_start|>)\n"
                 "  --tail-role-token ID --serve: a turn of this role right before the last turn is left out of\n"
                 "                       the conversation checkpoint (a trailing effort turn; default -1 = off)\n"
                 "  --short-read N       --serve: read at most N fresh text tokens through the decode windows instead\n"
                 "                       of the batched prompt path (default 64, 0 = off)\n"
                 "  --suffix-draft N     prompt lookup: draft from an earlier repeat of the last N+ tokens of context\n"
                 "                       when it pays (default 3; 0 = MTP only)\n"
                 "  --mtp-max-t M        cap the MTP's windows at M tokens (0 = --spec; longer ones come from suffixes)\n"
                 "  --lookup-chain K     opt-in: after the MTP's drafts, add up to K prompt-lookup drafts that continue\n"
                 "                       them (the window grows to at most 8; default 0 = off)\n"
                 "  --lookup-chain-min M  the shortest context match --lookup-chain extends on (default 3)\n"
                 "  --mtp-hnorm pooled|stream  the draft layer's hidden-input norm: one RMS over all four streams\n"
                 "                       (default) or one per stream (llama.cpp's MTP graph)\n"
                 "  --mtp-q4 all|proj|head  opt-in: the draft layer's projections and/or draft head as 4-bit copies\n"
                 "  --mtp-draft-vocab FILE  opt-in: the draft head's token subset (default <mtp>/draft_vocab.bin)\n"
                 "  --control-vector-scaled FILE:SCALE[,...]  a control vector GGUF on the residual stream (llama.cpp's\n"
                 "                       format; --control-vector FILE = scale 1).  --serve: requests switch it (cvec=0|1)\n"
                 "  --control-vector-layer-range A B  the layers it follows (inclusive; default 1 .. the last)\n"
                 "  --cvec-mode add|project  h += s v (default) or h -= s (h.v) v with v unit\n"
                 "  --cvec-dir per-layer|single:L  each layer's own direction (default) or layer L's everywhere (project)\n"
                 "  --no-token-graph     A/B: two graphs per layer (the host launches each) instead of one per token\n"
                 "  --no-fused-gr        A/B: the six-kernel hyper-connection read and a separate write (native)\n"
                 "  --prefill CHUNK      batched prompt processing in chunks of CHUNK tokens (needs --native); auto =\n"
                 "                       the largest chunk up to 8192 whose buffers the expert cache can lend;\n"
                 "                       auto:16384 / auto:32768 (or STRATA_PREFILL_AUTO_MAX) allow bigger ones\n"
                 "  --no-pool            skip the CPU expert pool (the GPU-only floor)\n"
                 "  --sync-every-layer   debug: synchronise after every layer\n"
                 "  --ple-gguf PATH      the n-gram/PLE shard.  WITHOUT IT LAYER 1's PLE IS SILENTLY SKIPPED,\n"
                 "                       which changes every number downstream - pass it for any real run\n"
                 "  --dump-mixed PATH    write the post-attention residual (n_embd, f32)\n"
                 "  --stage-timing       per-stage KERNEL-COUNT shares.  NOT a time profile: an uncaptured\n"
                 "                       event interval includes host gaps, so run with --gpu-only-full first\n"
                 "  --graph-only         MEASURE: replay the 48 `pre` graphs only.  OMITS the 48 `post` graphs\n"
                 "                       and the LM head, so it is NOT the GPU floor (R0.3, Memory/ERRORS.md A4)\n"
                 "  --gpu-only-full      MEASURE: replay pre+post for all 48 layers plus the LM head, no pool.\n"
                 "                       THE TRUE PER-TOKEN GPU FLOOR.  Quote this one, not --graph-only.\n"
                 "  --stats              print the per-stage breakdown\n"
                 "  --spec-follow PATH   benchmarks: emit this continuation (token ids) instead of the argmax and\n"
                 "                       accept the drafts that match it, so speculation settings compare on the\n"
                 "                       same text (with --spec; not --serve)\n"
                 "  --window-hashes PATH per verify window, a line with its index, position, size, a 64-bit hash of\n"
                 "                       its final residual rows and its argmaxes: two builds that do the same\n"
                 "                       arithmetic write the same file (use with --spec-follow, --adapt-every 0)\n"
                 "  --gpu-stages         R0.9: capture the layer as three graphs (mixer / ffn+router / post)\n"
                 "                       and time them from OUTSIDE the capture.  The per-stage table on the\n"
                 "                       real graph that --stage-timing cannot give.  Prints and exits.\n"
                 "  --expert-profile P   R4.2e: pre-load the VRAM tier from a `profile.bin` (see\n"
                 "                       tools/make_profile.py) instead of admitting on first use.\n"
                 "  --expert-profile-save P  --serve, #477: save what the adaptive tier learned (the experts in\n"
                 "                       VRAM, then the routing counted since the start) as a profile at P, on\n"
                 "                       QUIT and every --expert-profile-save-every MIN minutes (default 10;\n"
                 "                       0 = on QUIT only) between requests; start from it with --expert-profile P\n"
                 "  --no-hit-poke        R4.2d's A/B arm.  The hit path pokes the driver once right after its\n"
                 "                       launch so the GPU starts while the CPU pool runs; without it the work\n"
                 "                       waits for the next driver entry and does not overlap at all.\n"
                 "  --no-ple-prefetch     A/B arm: read the PLE table's sixteen rows one at a time, instead of\n"
                 "                       issuing them in one PrefetchVirtualMemory call.\n"
                 "  --expert-cache N     R4: keep N expert blobs resident in VRAM and compute their rows on the\n"
                 "                       GPU via `moe_hit_grouped_s2`.  DEFAULT 0.  Measured at 4096 slots\n"
                 "                       with --expert-cache-per-layer: 54.4%% hits, CPU pool drain 19.1 -> 10.3\n"
                 "                       ms/token, -2.7 ms/token end to end.\n"
                 "  --expert-cache-device1 N|auto  fill CUDA1 with N experts or as many as fit\n"
                 "  --expert-cache-device2 N|auto  fill CUDA2 with more experts\n"
                 "  --expert-cache-device3 N|auto  fill CUDA3 with more experts\n"
                 "  --remote-expert-opt   enable remote-expert decode optimizations (serve mode)\n"
                 "  --expert-cache-remote-placement stripe|layer  distribute expert ranks or whole\n"
                 "                       layers across CUDA1..3 (default: stripe)\n"
                 "  --peer-device N      a second GPU as an adaptive expert-cache tier (rows over P2P; it also\n"
                 "                       computes its experts' prompt rows).  Not with --layer-split or\n"
                 "                       --expert-cache-device1..3.  Default output unchanged without it.\n"
                 "  --peer-reserve-mib N  VRAM the peer tier leaves free on its card (default 600)\n"
                 "  --peer-slots N       expert slots on the peer (default: what fits)\n"
                 "  --peer-adapt-swaps N  peer cache swaps per adaptation step (default: --adapt-swaps)\n"
                 "  --peer-prefill-rows N  prompt rows per layer the peer computes (default half of chunk x top-k;\n"
                 "                       0 = the prompt path stays on the primary)\n"
                 "  --expert-cache-per-layer  R4.2g: give each layer its OWN slots instead of letting the first\n"
                 "                       position take all of them.  The default policy fills in arrival order\n"
                 "                       from one shared counter, so 256 slots went to ~26 layers of position 0\n"
                 "                       and measured **2.97%%**.  Per-layer, the same routing gives 21.4%% at 8\n"
                 "                       slots/layer and 70.4%% at 64.  On a native pack, N is still the budget of N\n"
                 "                       largest blobs; each layer's slots are that layer's own blob.\n"
                 "  --no-host-worker     R2.2: the A/B arm.  By default the HOST THREAD joins the drain, so the\n"
                 "                       pool is six threads on six cores instead of five plus an idle core;\n"
                 "                       this flag restores the five-worker form for comparison on `pool phases`.\n"
                 "  --pool-workers N     R2.2: CPU expert pool worker count.  Default 0 = every physical core\n"
                 "                       except the one the host loop spins on (with --pool-affinity auto or\n"
                 "                       p-cores on a hybrid CPU: P-cores minus 1).  A sweep is how the pool's\n"
                 "                       deviation from `cpu_s2` is attributed.\n"
                 "  --host-core WHERE    The host thread's core: first (the default: the first physical core, as always) or\n"
                 "                       last (the last physical core, the workers on the others).  Windows sends a GPU's\n"
                 "                       interrupts to one logical processor, usually the first, and every copy that lands\n"
                 "                       raises one: a spinning host there waits for them.  Moves threads only, never a\n"
                 "                       result.  Not on hybrid CPUs (STRATA_HOST_CORE sets it too).\n"
                 "  --pool-cores LIST    Experimental, off by default: pin the CPU workers to exactly these logical\n"
                 "                       processors (0,2,4,6 or 2-14:2).  Two entries on one core, or the host's own, are\n"
                 "                       kept with a warning.  STRATA_POOL_CORES sets it too.\n"
                 "  --pool-affinity MODE Worker CPU affinity: all (default: one worker per physical core, as\n"
                 "                       always), auto (hybrid CPUs: P-cores first, then their SMT siblings,\n"
                 "                       then E-cores) or p-cores (P-cores and their siblings only).\n"
                 "  --coupled-draft      enable coupled draft sampling for MTP drafter under sampling (STRATA_SPEC_COUPLED)\n"
                 "  --no-coupled-draft   disable coupled draft sampling (propose argmax drafts)\n"
                 "  --mmap-experts       R2.1: opt OUT of the resident expert arena, back to MapViewOfFile.\n"
                 "  --stream-experts     no host arena: experts read from the GGUF on demand (needs a native pack and\n"
                 "                       a VRAM cache that holds every expert; for machines with less RAM than experts)\n"
                 "                       The A/B arm: the mmap's rate depends on the OS page cache holding\n"
                 "                       34 GB, and measured 71.97 vs 34.78 ms/token cold vs warm.\n"
                 "  --shared-expert-arena FILE  Linux: back the resident arena with one MAP_SHARED file.\n"
                 "                       Put this file on /dev/shm, not ordinary SSD storage.\n"
                 "                       A small header binds an existing backing file to the same pack.\n"
                 "  --resident-cpu-experts  with mmap and a static profile, keep the experts the GPU cache does not\n"
                 "                       hold resident in ordinary RAM (and the prompt path's lend region as far as\n"
                 "                       RAM allows); adaptive swaps exchange them, so none is read from the file again.\n"
                 "  --resident-experts   the low-RAM PC's resident mode (setup): --mmap-experts --resident-cpu-experts\n"
                 "                       with the copy page-locked when possible, 4 GiB headroom, plain mmap if it\n"
                 "                       does not fit.  Same answers as --mmap-experts for the same placement.\n"
                 "  --adapt-async 1      --serve with the resident RAM mode (opt-in): the adaptive tier's swaps\n"
                 "                       advance between verify windows instead of a window waiting for a whole\n"
                 "                       round.  Not bit-exact run to run.  0 (default) = the blocking tier.\n");
}

// --lookup-chain: the context's last tokens for the draft sources (at most 64), the MTP's pending drafts last
int chain_tail(const strata::spec::SuffixDrafter& sfx, const int32_t* pending, int n_pending, std::vector<int32_t>& tail) {
    const std::vector<int32_t>& h = sfx.history();
    const int keep = std::max(0, std::min((int) h.size(), 64 - n_pending));
    tail.assign(h.end() - keep, h.end());
    tail.insert(tail.end(), pending, pending + n_pending);
    return (int) tail.size();
}

// --lookup-chain: the registered extra draft sources (strata/spec/draft_source.hpp) see the committed context too
void extra_sources_reset() {
    for (auto& src : strata::spec::extra_draft_sources()) src->reset();
}
void extra_sources_append(const int32_t* t, size_t n) {
    for (auto& src : strata::spec::extra_draft_sources()) src->append(t, n);
}

bool parse_i64_list(const char* s, std::vector<int64_t>& out, std::string& err) {
    out.clear();
    std::string text(s);
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream input(text);
    std::string token;
    while (input >> token) {
        int32_t id = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || id < 0) {
            err = "invalid token id: expected an integer in [0, 2147483647]";
            out.clear();
            return false;
        }
        out.push_back(id);
    }
    if (out.empty()) { err = "token list was empty"; return false; }
    return true;
}

/// The pool's adapter plus the wall-clock it spent, so the report can say how much of the token was the CPU.
struct Drive {
    strata::core::ExpertDispatch d;
    double cpu_ms = 0;
    int64_t calls = 0;
    /// THE ROUTING TRACE, which is P0.S8 and a stated prerequisite of Phase 3.  `drive_pool` is called once
    /// per layer from the main loop - the workers live inside `expert_pool_dispatch` - so a single FILE* here
    /// needs no locking.  `d.layers` is the CURRENT layer on entry (the adapter increments it as it walks the
    /// blob), which is why the layer index comes from there rather than from a counter of our own.
    std::FILE* routing = nullptr;
};

void drive_pool(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd, int64_t k,
                float* out) {
    Drive* t = (Drive*) user;
    const Clock::time_point a = Clock::now();
    strata::core::expert_pool_dispatch(&t->d, x_f, ids, weights, n_embd, k, out);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // THE ROUTING TRACE.  Written AFTER the dispatch so the layer index is still this layer's: `d.layers` is
    // advanced by the adapter as it consumes the blob, and reading it after the call is the same value the
    // dispatch used.  Record = int32 layer, int32 k, k int32 ids, k float weights.
    if (t->routing != nullptr) {
        // **`d.layers` HAS ALREADY BEEN ADVANCED BY THE TIME THIS RUNS, AND THE FIRST TRACE WAS OFF BY ONE
        // BECAUSE OF IT.**  The adapter walks the blob by incrementing `d.layers` as it consumes each layer's
        // experts, so after the dispatch it holds the NEXT layer's index.  `tools/make_profile.py` caught it
        // with a bounds check when the trace turned out to span 1..48 instead of 0..47.  The hit-rate CURVE was
        // unaffected - it is a per-layer split, and shifting every layer by one preserves both metrics - but
        // anything keyed on the layer index, which is exactly what a cache profile is, would have been wrong.
        const int32_t layer_idx = (int32_t) (t->d.layers - 1);
        if (layer_idx < 0 || layer_idx >= 48) {
            std::fprintf(stderr, "strata generate: the routing trace saw layer %d, outside 0..47\n", layer_idx);
            return;
        }
        const int32_t rec[2] = {layer_idx, (int32_t) k};
        std::fwrite(rec, sizeof rec, 1, t->routing);
        std::fwrite(ids, sizeof(int32_t), (size_t) k, t->routing);
        std::fwrite(weights, sizeof(float), (size_t) k, t->routing);
    }
}

/// Plan v0.3 P6: the pool for a verify window.
void drive_pool_multi(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer) {
    Drive* t = (Drive*) user;
    t->d.layers = layer;
    const Clock::time_point a = Clock::now();
    strata::core::expert_pool_dispatch_multi(t->d, x_f, ids, n_tok, k, out);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // the routing trace for the serve path: the same record format drive_pool writes (layer, k, ids, weights),
    // one record per token.  The multi dispatch fuses the router weights into the kernel and does not surface
    // them, so records carry unit weights: tools/make_profile.py ranks pairs by routed frequency, which is the
    // signal that matters; a one-shot --dump-routing run records true weights if a weighted ranking is wanted.
    if (t->routing != nullptr && layer >= 0 && layer < 48) {
        for (int64_t tok = 0; tok < n_tok; ++tok) {
            const int32_t rec[2] = {(int32_t) layer, (int32_t) k};
            std::fwrite(rec, sizeof rec, 1, t->routing);
            std::fwrite(ids + tok * k, sizeof(int32_t), (size_t) k, t->routing);
            static const float one[64] = {};   // k <= 64 in a verify window; zeros read as unit weights
            std::fwrite(one, sizeof(float), (size_t) k, t->routing);
        }
    }
}

/// Layer split: every verify stage shares one Drive (its counters, usage and failure flags); the GPU plan, the expert
/// cache and the PCIe share the pool uses for a layer are those of the stage that runs it.
struct SplitDrive {
    static constexpr int kMax = 8;
    Drive* base = nullptr;
    int n = 0;                                    ///< stages
    int64_t end[kMax] = {};                       ///< stage i runs the layers from end[i - 1] (0) below end[i]
    strata::core::GpuPlanSink* plan[kMax] = {};
    const uint8_t* cache_base[kMax] = {};
    const uint64_t* cache_slot_off[kMax] = {};
    int pcie_num[kMax] = {};
};
void drive_pool_split(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer) {
    SplitDrive* s = (SplitDrive*) user;
    int st = 0;
    while (st + 1 < s->n && layer >= s->end[st]) ++st;
    Drive& d = *s->base;
    d.d.plan = s->plan[st];
    d.d.cache_base = s->cache_base[st];
    d.d.cache_slot_off = s->cache_slot_off[st];
    d.d.pcie_num = s->pcie_num[st];
    drive_pool_multi(s->base, x_f, ids, n_tok, k, out, layer);
}

/// Layer split across GPUs: a later stage on its own device, with its own copy of the dense weights, a session, an
/// expert cache for its layers, a verify window and a prompt path; the last one also holds the head (the drafter
/// lives on its device too).
struct GpuStage {
    int dev = 0;
    int64_t lb = 0, le = 0;
    double pcie_frac = 0.0;
    strata::core::WeightTable wt;
    strata::core::NativeDense dense;
    strata::core::NativeHead head;
    strata::core::SessionState ss;
    dpct::queue_ptr stream = &dpct::get_in_order_queue();
    strata::core::ExpertCache cache;
    std::vector<std::pair<int32_t, int32_t>> profile;   ///< its layers' share of the profile, hottest first
    int32_t* d_res = nullptr;                            ///< the residency table on its device
    strata::core::Verifier ver;
    strata::core::Verifier ver_b;                        ///< --pipeline-windows: the odd windows' verifier
    strata::prefill::Prefill sp;
    dpct::queue_ptr adapt_stream = &dpct::get_in_order_queue();
    dpct::event_ptr adapt_ev = nullptr;
    bool adapt_live = false;                             ///< swaps of this request are in flight on it
    int32_t* mrope = nullptr;                            ///< --vision: the image-position table on its device
};

// ---- issue #31: what the watchdog prints before it stops a stalled engine
struct MemSample {
    unsigned long long faults = 0, rss_mib = 0, avail_mib = 0, commit_mib = 0;   // faults: hard (Linux) / all (Windows)
};
MemSample mem_sample() {
    MemSample m;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
        m.faults = pmc.PageFaultCount;
        m.rss_mib = pmc.WorkingSetSize >> 20;
        m.commit_mib = pmc.PagefileUsage >> 20;
    }
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) m.avail_mib = ms.ullAvailPhys >> 20;
#else
    if (std::FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[4096];
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        std::fclose(f);
        const char* s = std::strrchr(buf, ')');   // fields after the command name: 3 state ... 12 majflt
        for (int field = 2; s && field < 12; ++field) s = std::strchr(s + 1, ' ');
        if (s) m.faults = std::strtoull(s + 1, nullptr, 10);
    }
    auto kb = [](const char* path, const char* key) -> unsigned long long {
        unsigned long long v = 0;
        if (std::FILE* f = std::fopen(path, "r")) {
            char line[256];
            const size_t kl = std::strlen(key);
            while (std::fgets(line, sizeof line, f))
                if (std::strncmp(line, key, kl) == 0) { v = std::strtoull(line + kl, nullptr, 10); break; }
            std::fclose(f);
        }
        return v;
    };
    m.rss_mib = kb("/proc/self/status", "VmRSS:") >> 10;
    m.commit_mib = kb("/proc/self/status", "VmSwap:") >> 10;
    m.avail_mib = kb("/proc/meminfo", "MemAvailable:") >> 10;
#endif
    return m;
}

// the stage the watchdog names: "<where> <detail>", and the prompt chunk a batched read is in (#251)
std::string stage_text() {
    const strata::core::Progress& p = strata::core::progress();
    std::string s = std::string(p.where.load()) + " " + std::to_string((long long) p.detail.load());
    if (const int64_t c = p.chunk.load(); c >= 0) s += " of the prompt chunk from token " + std::to_string((long long) c);
    return s;
}

#if !defined(_WIN32)
// #605: whether `path` is on a rotational disk (1), not (0) or unknown (-1), from sysfs; `dev` is the disk's name
int on_rotational_disk(const std::string& path, std::string& dev) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) return -1;
    char sys[64];
    std::snprintf(sys, sizeof sys, "/sys/dev/block/%u:%u", (unsigned) major(st.st_dev), (unsigned) minor(st.st_dev));
    std::error_code ec;
    std::filesystem::path p = std::filesystem::canonical(sys, ec);
    for (int up = 0; !ec && up < 2; ++up, p = p.parent_path()) {   // a partition has its disk's queue
        std::ifstream q(p / "queue" / "rotational");
        int r = -1;
        if (q >> r) {
            dev = p.filename().string();
            return r;
        }
    }
    return -1;
}

// #605: the engine's threads in uninterruptible sleep (state D - almost always waiting for the disk), and where
// (wchan): 16 threads in blk_io_schedule is a disk that cannot keep up, not a deadlock in the engine
void disk_wait_threads(std::FILE* f) {
    std::map<std::string, int> where;
    int waiting = 0, threads = 0;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator("/proc/self/task", ec)) {
        ++threads;
        std::ifstream st(e.path() / "stat");
        std::string line;
        std::getline(st, line);
        const size_t rp = line.rfind(')');
        if (rp == std::string::npos || rp + 2 >= line.size() || line[rp + 2] != 'D') continue;
        ++waiting;
        std::ifstream w(e.path() / "wchan");
        std::string wc;
        std::getline(w, wc);
        ++where[wc.empty() || wc == "0" ? "?" : wc];
    }
    if (threads == 0) return;
    std::string list;
    for (const auto& [name, n] : where) list += (list.empty() ? " (" : ", ") + name + " x" + std::to_string(n);
    if (!list.empty()) list += ")";
    std::fprintf(f, "  threads waiting on the disk (state D): %d of %d%s%s\n", waiting, threads, list.c_str(),
                 waiting > 0 ? " - the engine is waiting for the drive: a rotational or failing disk with --ple-io "
                               "direct (#605: --ple-io ram), or a drive too slow for the reads asked of it" : "");
}
#endif

// --pipeline-windows: what the watchdog prints about the pipelined decode loop (set while it runs)
struct PlDiag {
    strata::core::Verifier* v[2][2] = {};
    std::atomic<int> a_seq{0}, a_bits{0}, b_bits{0}, d_bits{0}, chain_kind{0}, doomed{0}, ending{0}, chain_live{0};
    std::atomic<long long> iters{0};
};
PlDiag g_pl_diag;
void pl_diag_print(std::FILE* f) {
    PlDiag& d = g_pl_diag;
    auto bits = [](int b, char* o) {
        std::snprintf(o, 16, "%c%c%c%c%c%c", b & 1 ? 'R' : '-', b & 2 ? 'L' : '-', b & 4 ? 'F' : '-', b & 8 ? 'C' : '-',
                      b & 16 ? 'S' : '-', b & 32 ? 'V' : '-');
        return (const char*) o;
    };
    char ba[16], bb[16], bd[16];
    std::fprintf(f, "  pipelined loop: %lld iterations; A seq %d [%s] B [%s] D [%s] (Ready Launched Finished Committed "
                    "S1-launched S1-done); chain kind %d live %d; doomed %d ending %d\n", d.iters.load(), d.a_seq.load(),
                 bits(d.a_bits.load(), ba), bits(d.b_bits.load(), bb), bits(d.d_bits.load(), bd), d.chain_kind.load(),
                 d.chain_live.load(), d.doomed.load(), d.ending.load());
    const char* names[2][2] = {{"stage 0 even", "stage 0 odd"}, {"stage 1 even", "stage 1 odd"}};
    for (int st = 0; st < 2; ++st)
        for (int par = 0; par < 2; ++par)
            if (d.v[st][par] != nullptr) d.v[st][par]->diag_pipelined(f, names[st][par]);
}

void stall_report(std::FILE* f, uint64_t layers_during) {
    strata::core::Progress& p = strata::core::progress();
    std::fprintf(f, "strata serve: stall report (engine %s): stage \"%s\" for %lld s; %llu layers served since the "
                    "last finished step (0 = stopped, more = slow)\n", STRATA_VERSION, stage_text().c_str(),
                 (long long) ((strata::core::progress_now_ms() - p.since_ms.load()) / 1000),
                 (unsigned long long) layers_during);
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::fprintf(f, "  2 s later:\n");
        }
        if (auto fn = strata::core::diag_pool_fn().load()) fn(f);
        if (auto fn = strata::core::diag_verify_fn().load()) fn(f);
        if (auto fn = strata::core::diag_pipeline_fn().load()) fn(f);
#if !defined(_WIN32)
        disk_wait_threads(f);
#endif
        const MemSample m = mem_sample();
        std::fprintf(f, "  memory: %llu MiB resident, %llu MiB %s, %llu MiB RAM available; %llu %s\n", m.rss_mib,
                     m.commit_mib,
#if defined(_WIN32)
                     "committed", m.avail_mib, m.faults, "page faults so far"
#else
                     "in swap", m.avail_mib, m.faults, "major page faults so far"
#endif
        );
        std::fflush(f);
    }
#if defined(_WIN32)
    // every thread's stack, to read against this build's PDB: a few MB beside the engine's working directory
    if (HMODULE dbg = LoadLibraryA("dbghelp.dll")) {
        using Fn = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, int, void*, void*, void*);
        if (auto write = (Fn) GetProcAddress(dbg, "MiniDumpWriteDump")) {
            char path[64];
            std::snprintf(path, sizeof path, "strata-stall-%lu.dmp", (unsigned long) GetCurrentProcessId());
            HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (h != INVALID_HANDLE_VALUE) {
                const int kThreadInfo = 0x1000;   // MiniDumpWithThreadInfo; MiniDumpNormal = 0
                const BOOL ok = write(GetCurrentProcess(), GetCurrentProcessId(), h, kThreadInfo, nullptr, nullptr, nullptr);
                CloseHandle(h);
                char full[MAX_PATH];
                if (!GetFullPathNameA(path, MAX_PATH, full, nullptr)) std::snprintf(full, sizeof full, "%s", path);
                std::fprintf(f, "  %s the thread stacks to %s (attach it to the issue)\n", ok ? "wrote" : "could not write",
                             full);
            }
        }
    }
#endif
}

/// STRATA_TRACE=1: the VRAM left at a step of the startup (finds what fills the card after the cache is sized)
void mem_mark(const char* where) {
    static const bool on = std::getenv("STRATA_TRACE") != nullptr;
    if (!on) return;
    size_t free_b = 0, total_b = 0;
    /*
    DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    dpct::get_current_device().get_memory_info(free_b, total_b);
    std::fprintf(stderr, "strata trace: %lld MiB free after %s\n", (long long) (free_b >> 20), where);
}

/// #463's A/B: STRATA_ADAPT_NOWAIT=1 lets a verify window start before the adaptive tier's copies have landed (0.1.37)
bool adapt_nowait() {
    static const bool v = [] { const char* e = std::getenv("STRATA_ADAPT_NOWAIT"); return e && e[0] == '1'; }();
    return v;
}

/// #463 without the stall: a verify window takes the adaptive tier's previous swaps only once they are this many windows
/// old, and then waits for their copies (which have landed by then).  The window that first computes a swapped-in expert on
/// the GPU depends only on the window count, as with #463's wait, so the answers are as reproducible.  Default 1 = #463 and
/// 0.1.39 (the very next window waits for the copies: ~4.6 ms per window on an RTX 2060 SUPER over PCIe 3.0 x8); lag 2 is
/// opt-in (STRATA_ADAPT_LAG=2).  The batch paths, --pipeline-windows and the peer tier are not lagged.  (#764)
int adapt_lag() {
    static const int v = [] { const char* e = std::getenv("STRATA_ADAPT_LAG"); return e ? std::max(1, std::atoi(e)) : 1; }();
    return v;
}

/// --adapt-async: one background job at a time (the asynchronous adaptive tier's copies and memcpys).  `post` hands
/// it over, `idle` says whether it has finished; the owner never posts while a job runs.
class JobThread {
public:
    JobThread() : th_([this] { loop(); }) {}
    ~JobThread() {
        { std::lock_guard<std::mutex> lk(mu_); quit_ = true; }
        cv_.notify_one();
        th_.join();
    }
    JobThread(const JobThread&) = delete;
    JobThread& operator=(const JobThread&) = delete;
    void post(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            job_ = std::move(f);
            has_ = true;
            done_.store(false, std::memory_order_release);
        }
        cv_.notify_one();
    }
    bool idle() const { return done_.load(std::memory_order_acquire); }
    void wait() const { while (!idle()) std::this_thread::yield(); }
private:
    // The second logical processor of the last physical core that has two: an SMT sibling no pool worker or host loop
    // is pinned to (the pool takes the first of every core but the first, the host the first core), or -1.
    static int smt_spare_cpu() {
#if defined(_WIN32)
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
        if (len == 0) return -1;
        std::vector<char> buf(len);
        if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(),
                                              &len))
            return -1;
        int best = -1, core = 0;
        for (const char* p = buf.data(); p < buf.data() + len;
             p += ((const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p)->Size, ++core) {
            const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
            const GROUP_AFFINITY& ga = e->Processor.GroupMask[0];
            int seen = 0;
            for (int bit = 0; bit < 64; ++bit)
                if ((ga.Mask & (1ull << bit)) && ++seen == 2 && core > 0) best = ga.Group * 64 + bit;
        }
        return best;
#else
        return -1;
#endif
    }
    void loop() {
        // the job's memcpys stay off the host's core and the pool's: by default the spare SMT sibling of the last
        // two-thread core (Windows); STRATA_ADAPT_JOB_CPU=<logical cpu>, or -1 for no pinning
        const char* c = std::getenv("STRATA_ADAPT_JOB_CPU");
        const int cpu = c != nullptr ? std::atoi(c) : smt_spare_cpu();
        if (cpu >= 0) (void) strata::kernels::cpu::pin_current_thread(cpu);
        for (;;) {
            std::function<void()> f;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return has_ || quit_; });
                if (!has_) return;
                f = std::move(job_);
                has_ = false;
            }
            f();
            done_.store(true, std::memory_order_release);
        }
    }
    std::mutex mu_;
    std::condition_variable cv_;
    std::function<void()> job_;
    bool has_ = false, quit_ = false;
    std::atomic<bool> done_{true};
    std::thread th_;   // last: started once the members above exist
};

// ---- --serve's conversation cache.  A chat or an agent sends the whole conversation again with every request, and
// reading it again is what made a long session wait minutes for every turn.  What a sequence leaves behind splits in
// two, and only one half needs copying:
//   * POSITIONAL state - the KV cache of the 12 QSA layers and their pooled indexer keys, the draft layer's KV.  A
//     cell is written once for its position and read only by later positions (the block scores take `dead` for the
//     block being filled, never its pooled row), so rewinding to a position just means writing from there again.
//   * RUNNING state - the 36 GDN recurrences and conv histories, each QSA layer's indexer tail (the unfinished
//     block's raw keys) and the PLE's normalized history.  Each describes "everything so far" and cannot be
//     rewound, so a checkpoint is a copy of exactly these: ~118 MB, the same set the verifier snapshots to roll
//     back rejected drafts.
// A checkpoint is only valid while the positional cells below it still hold ITS tokens, so the serve loop keeps
// just the checkpoints that are prefixes of the tokens the session holds now.
using ImgKey = strata::core::ConversationImageKey;
using ConvCheckpoint = strata::core::ConversationCheckpoint;

uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

using ConvStateSizes = strata::core::ConversationStateSizes;

ConvStateSizes conv_state_sizes(const strata::core::ModelGeometry& g, const strata::core::SessionState& ss) {
    ConvStateSizes z;
    std::string error;
    // a split stage's session owns only its layer range's state (see SessionState's carve note); the geometry
    // and the carve have already passed engine validation
    strata::core::conversation_session_sizes(g, ss, z, error);
    return z;
}

/// Copies the running state out (this session's carve only).  The caller has synchronized the device.
bool checkpoint_save(ConvCheckpoint& c, const strata::core::SessionState& ss, const strata::core::ModelGeometry& g) {
    std::string error;
    if (strata::core::conversation_checkpoint_save(c, ss, g, error)) return true;
    std::fprintf(stderr, "strata serve: checkpoint save: %s\n", error.c_str());   // the caller's ERR has no reason
    return false;
}

/// Puts a checkpoint's running state back; the positional cells below it are the caller's to guarantee.
bool checkpoint_restore(const ConvCheckpoint& c, strata::core::SessionState& ss, const strata::core::ModelGeometry& g) {
    std::string error;
    if (strata::core::conversation_checkpoint_restore(c, ss, g, error)) return true;
    std::fprintf(stderr, "strata serve: checkpoint restore: %s\n", error.c_str());
    return false;
}

// --control-vector-scaled: llama.cpp's `common_control_vector_load` (every file's `direction.<l>` times its scale,
// summed; layer 0 has none) and `llama_adapter_cvec::apply` with the projection-mode patch (project: the unit
// direction and its norm as the scale), into the tables `cvec_upload` takes.  `summary` is what INFO reports;
// `digest` identifies the uploaded tables, mode and range exactly (a session file is bound to it).
bool load_control_vectors(const Options& o, const strata::core::ModelGeometry& g, std::string& summary,
                          uint64_t& digest, std::string& err) {
    const int64_t L = g.n_layers, N = g.n_embd;
    std::vector<float> data((size_t) (L * N), 0.0f);
    std::vector<bool> have((size_t) L, false);
    for (const auto& [path, scale] : o.cvec_files) {
        try {
            strata::GgufFile f(path);
            const strata::MetaValue* arch = f.get("general.architecture");
            if (arch == nullptr || arch->s != "controlvector") {
                err = path + ": not a control vector GGUF (general.architecture is not 'controlvector')";
                return false;
            }
            const strata::MetaValue* hint = f.get("controlvector.model_hint");
            if (hint != nullptr && hint->s != "qwen4exp")
                std::fprintf(stderr, "strata generate: %s was made for '%s', not qwen4exp\n", path.c_str(), hint->s.c_str());
            int found = 0;
            for (const strata::TensorInfo& t : f.tensors()) {
                if (t.name.rfind("direction.", 0) != 0) continue;
                const long l = std::strtol(t.name.c_str() + 10, nullptr, 10);
                if (l < 1 || l >= L) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp
                if (t.type != 0 || t.elements() != (uint64_t) N) {
                    err = path + ": " + t.name + " must be " + std::to_string((long long) N) + " f32";
                    return false;
                }
                const float* src = reinterpret_cast<const float*>(f.tensor_data(t));
                for (int64_t j = 0; j < N; ++j) data[(size_t) (l * N + j)] += scale * src[j];
                have[(size_t) l] = true;
                ++found;
            }
            if (found == 0) { err = path + ": no direction.<layer> tensors"; return false; }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }
    const int first = o.cvec_first <= 0 ? 1 : o.cvec_first;
    const int last = (o.cvec_last <= 0 || o.cvec_last >= L) ? (int) L - 1 : o.cvec_last;
    const int single = o.cvec_mode == 0 ? o.cvec_single : -1;
    if (single >= 0 && (single >= L || !have[(size_t) single])) {
        err = "--cvec-dir single:" + std::to_string(single) + ": the vector has no direction for that layer";
        return false;
    }
    std::vector<float> dir((size_t) (L * N), 0.0f), s((size_t) L, 0.0f);
    int steered = 0;
    for (int64_t l = first; l <= last; ++l) {
        const int64_t src = single >= 0 ? single : l;
        if (!have[(size_t) src]) continue;
        const float* d = data.data() + (size_t) (src * N);
        if (o.cvec_mode == 0) {
            double nrm = 0.0;
            for (int64_t j = 0; j < N; ++j) nrm += (double) d[j] * d[j];
            nrm = std::sqrt(nrm);
            if (nrm <= 0.0) continue;
            s[(size_t) l] = (float) nrm;
            for (int64_t j = 0; j < N; ++j) dir[(size_t) (l * N + j)] = (float) (d[j] / nrm);
        } else {
            s[(size_t) l] = 1.0f;
            std::copy(d, d + N, dir.begin() + (size_t) (l * N));
        }
        ++steered;
    }
    if (steered == 0) { err = "the control vector has no direction in layers " + std::to_string(first) + ".." + std::to_string(last); return false; }
    if (!strata::kernels::cvec_upload(dir, s, o.cvec_mode, first, last, N, g.hc, err)) return false;
    {
        strata::core::SessionIdentityBuilder b(0x43564543ull);   // "CVEC"
        b.i64("mode", o.cvec_mode);
        b.i64("first", first);
        b.i64("last", last);
        b.i64("single", single);
        b.bytes("dir", dir.data(), dir.size() * sizeof(float));
        b.bytes("scale", s.data(), s.size() * sizeof(float));
        digest = b.digest();
        if (digest == 0) digest = 1;   // 0 means "none loaded"
    }
    summary = std::string(o.cvec_mode == 0 ? "project" : "add") + ":" + std::to_string(first) + "-" + std::to_string(last) +
              (single >= 0 ? ":single" + std::to_string(single) : "");
    // the line llama.cpp's patched build prints, so a log shows the same thing
    std::fprintf(stderr, "strata generate: control vector mode = %s, dir = %s, layers %d..%d (%d steered)\n",
                 o.cvec_mode == 0 ? "project" : "add", single >= 0 ? "single" : "per-layer", first, last, steered);
    return true;
}

// The effective host->device bandwidth of the PCIe link: copies from pinned host memory, as the expert arena's
// reads are.  The native default share (0.55) was measured on x16 links (~26-28 GB/s); a x8 card in a x8 slot
// carries about half of that.  Returns < 0 when the probe cannot run (then the caller keeps the default).
//
// #485: a single timed burst read an x16 PCIe 4 link (RTX A3000 laptop) at 18.5, 6.9 and 5.8 GB/s in three starts -
// a link still in a low-power state, other DMA, a context not yet up to speed - and the low reading set the share.
// Such a disturbance only ever slows a copy: nothing makes a correctly timed copy faster than the link carries.  So
// the same 1 GiB is now copied as four bursts of 256 MiB, timed one by one, and the fastest is the link's figure (the
// median would still follow a disturbance that lasts through half the bursts).  Each burst takes ~10 ms on an x16
// PCIe 4 link, so the probe takes no longer than the one 1 GiB burst did.  `samples`, when given, gets every
// burst's reading for the log.
double probe_pcie_h2d_gbps(std::string *samples = nullptr, bool pin_source = false) try {
    constexpr size_t kBytes = 256ull << 20;
    constexpr int kBursts = 4;
    void* h = nullptr;
    void* d = nullptr;
    // PR #1669: a pageable source made the copy engine fault on a layer split (VM_NOT_FOUND), so a split pins it. A single card keeps the
    // pageable source of 0.1.41: pinned reads 1.7x faster, which moves the default pcie_frac (A750 0.28 -> 0.47, 10% slower decode).
    if (DPCT_CHECK_ERROR(h = pin_source ? (void *)sycl::malloc_host(kBytes, dpct::get_in_order_queue()) : (void *)malloc(kBytes)) != 0 || h == nullptr) return -1.0;
    if (DPCT_CHECK_ERROR(d = (void *)sycl::malloc_device(
                             kBytes, dpct::get_in_order_queue())) != 0) {
        if (pin_source) sycl::free(h, dpct::get_in_order_queue()); else free(h);
        return -1.0;
    }
    std::memset(h, 0, kBytes);   // fault the pages in before timing
    dpct::get_in_order_queue().memcpy(
        d, h, kBytes).wait(); // warmup: context up, copy engine primed
    float ms[kBursts] = {};
#if defined(STRATA_USE_HIP) && defined(_WIN32)
    // Windows HIP: the events do not bracket the copies there (an RX 6800 read 3,300-26,000 GB/s, so every link kept
    // the x16 share), so each burst is timed on the host between two synchronizes: 256 MiB takes ~10 ms, so the
    // synchronize around it hardly matters
    bool ok = cudaDeviceSynchronize() == cudaSuccess;
    for (int b = 0; b < kBursts && ok; ++b) {
        const Clock::time_point t0 = Clock::now();
        cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);
        ok = cudaDeviceSynchronize() == cudaSuccess;
        ms[b] = (float) std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }
#else
    // the bursts run back to back on the stream, an event between each two
    dpct::event_ptr ev[kBursts + 1];
    int n_ev = 0;
    while (n_ev <= kBursts &&
           DPCT_CHECK_ERROR(ev[n_ev] = new sycl::event()) == 0)++ n_ev;
    bool ok = n_ev == kBursts + 1;
    if (ok) {
        dpct::sync_barrier(ev[0]);
        for (int b = 0; b < kBursts; ++b) {
            /*
            DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            dpct::get_in_order_queue().memcpy(d, h, kBytes).wait();
            dpct::sync_barrier(ev[b + 1]);
        }
        ok = DPCT_CHECK_ERROR(ev[kBursts]->wait_and_throw()) == 0;
        for (int b = 0; b < kBursts && ok; ++b) ok =
            DPCT_CHECK_ERROR(
                ms[b] =
                    (ev[b + 1]
                         ->get_profiling_info<
                             sycl::info::event_profiling::command_end>() -
                     ev[b]
                         ->get_profiling_info<
                             sycl::info::event_profiling::command_start>()) /
                    1000000.0f) == 0;
    }
    for (int i = 0; i < n_ev; ++i) dpct::destroy_event(ev[i]);
#endif
    double bw = -1.0;
    if (samples != nullptr) samples->clear();
    for (int b = 0; b < kBursts && ok; ++b) {
        const double s = ms[b] > 0.01f ? (double) kBytes / (ms[b] * 1e-3) / 1e9 : -1.0;
        bw = std::max(bw, s);
        if (samples != nullptr) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%s%.1f", b == 0 ? "" : " ", s);
            *samples += buf;
        }
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    if (!ok)(void) 0;
    sycl::free(d, dpct::get_in_order_queue());
    if (pin_source) sycl::free(h, dpct::get_in_order_queue()); else free(h);
    return bw;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// The PCIe share of the missed experts for a link measured at `gbps`: `base` (the share measured on x16 links) from
// 20 GB/s up, and below that in proportion to the bandwidth, so the time the link spends on its share stays about
// what the x16 share costs.  Continuous (#485): before, 19.9 GB/s gave 0.42 and 20.0 the full 0.55 (and 4.0 GB/s
// gave 0.08, 3.9 none), so a reading near either edge moved the share by a quarter of its range.  From 20 GB/s up
// (an x16 PCIe 4/5 link: ~26-28 GB/s) the share is unchanged.
double pcie_frac_for_gbps(double gbps, double base, double ref_gbps = 20.0) {
    return gbps <= 0.0 ? base : base * std::min(1.0, gbps / ref_gbps);
}

// 0.1.42 (CUDA) / 0.1.48 (SYCL, opt-in STRATA_PCIE_FRAC_DEFAULT=auto): the share refined from what the CPU pool measures in
// the first decode windows.  The balance share of the missed experts is link / (link + 1.85 x CPU rate); the link is the
// start-up probe, the CPU rate the expert bytes over the pool's wall time per missed expert (ms_run / multi_misses).
// Three updates at most (after 48, then 96 and 96 more windows), each rounded to 0.05.
struct PcieAuto {
    static constexpr double kPoolFactor = 1.85;
    bool on = false;
    double link_gbps = 0.0, blob_bytes = 0.0;
    double mark_ms = 0.0;
    int64_t mark_jobs = 0;
    double acc_ms = 0.0;
    int64_t acc_jobs = 0;
    int acc_win = 0, updates = 0;
    void mark(double ms_run, int64_t jobs) { mark_ms = ms_run; mark_jobs = jobs; }
    /// After a decode window: the new share, or -1 while there is nothing to change.  `tc_us`/`rate` get the reading.
    double window(double ms_run, int64_t jobs, double cur, double* tc_us, double* rate) {
        if (!on) return -1.0;
        acc_ms += ms_run - mark_ms;
        acc_jobs += jobs - mark_jobs;
        mark(ms_run, jobs);
        ++acc_win;
        if (acc_win < (updates == 0 ? 48 : 96)) return -1.0;
        if (acc_jobs < 300) {          // too few missed experts to read the pool's speed (everything hits): look again
            if (acc_win >= 2000) on = false;
            return -1.0;
        }
        const double tc = acc_ms * 1000.0 / (double) acc_jobs;
        acc_ms = 0.0;
        acc_jobs = 0;
        acc_win = 0;
        if (++updates >= 3) on = false;
        if (!(tc > 1.0) || !(blob_bytes > 0.0) || !(link_gbps > 0.0)) return -1.0;
        const double r = blob_bytes / (tc * 1e-6) / 1e9;
        const double f = std::min(0.60, std::max(0.10, std::round(link_gbps / (link_gbps + kPoolFactor * r) * 20.0) / 20.0));
        *tc_us = tc;
        *rate = r;
        return std::fabs(f - cur) < 0.025 ? -1.0 : f;
    }
};

}  // namespace

// #1425 #1549 #1629: --ple-io direct (the default) does unbuffered random reads of the n-gram table; on some drives
// (DRAM-less NVMe, Windows unbuffered I/O) they run at 34 MB/s or 4-8 tok/s and a prompt reads layer 1 for minutes.
// Rows per second of a short cold probe: a few batches of random rows through the same batched reader a prompt chunk
// uses, stopped after ~0.6 s. Returns <= 0 when nothing could be measured.
static double ple_direct_rows_per_s(strata::kernels::PleTable& t) {
    const uint64_t n = t.rows();
    if (n < 4096) return -1.0;
    constexpr size_t kTok = 4;   // 64 rows a batch
    std::vector<uint32_t> rows(kTok * 16);
    std::vector<float> out(kTok * 2560);
    uint64_t x = 0x9E3779B97F4A7C15ull ^ n;
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t done = 0;
    double el = 0;
    for (int b = 0; b < 64; ++b) {
        for (auto& r : rows) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; r = (uint32_t) (x % n); }
        std::string e;
        if (!t.gather_batch(rows.data(), kTok, out.data(), e)) return -1.0;
        done += rows.size();
        el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el > 0.6) break;
    }
    return el > 0 ? (double) done / el : -1.0;
}

// #1980: a start-up refusal ends the process here instead of returning from main.  Returning ran main's destructors
// (the stages' caches, queues and Level Zero objects) in an order the UR loader did not survive: 2x Arc Pro B60, every
// REFUSED start segfaulted in libur_loader after its message, and a trimmed one also left an xe page fault (bcs CAT error)
// on the second card that hung the next start.  The device work queued so far finishes first; nothing is freed.
[[noreturn]] static void refuse_exit(int code) {
    std::fflush(stdout);
    std::fflush(stderr);
    try {
        for (unsigned int d = 0; d < dpct::device_count(); ++d) dpct::get_device(d).queues_wait_and_throw();
    } catch (...) {
    }
    std::fflush(stderr);
    std::_Exit(code);
}

int main(int argc, char **argv) try {
    // **UNBUFFERED, BECAUSE THE INTERESTING OUTPUT IS THE OUTPUT BEFORE A CRASH.**  `stdout` redirected to a
    // pipe or a file is block-buffered, so a program that dies loses every line it had already printed - which
    // turns "it crashed at step 7" into "it crashed somewhere", and the difference is a debugging session.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // `strata --version` / `-V` (also `strata generate --version`): the version and nothing else, before any GPU is touched, as the CUDA
    // and HIP engines have answered it. `strata update` runs it as the engine's smoke test, and the Intel package build runs it with
    // an empty environment.
    for (int i = 1; i < argc && i < 3; ++i) {
        if (std::strcmp(argv[i], "--version") == 0 || std::strcmp(argv[i], "-V") == 0) {
            std::printf("strata %s\n", STRATA_VERSION);
            return 0;
        }
    }
#if (defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)) && !defined(_WIN32)
    // AMD, a file-backed arena (STRATA_ARENA_MMAP): ROCclr copies a pageable source of 1 MiB or more by locking its
    // pages in place (a GPU userptr), and keeps them - so every expert the VRAM fill copied from the mapped
    // experts.bin stayed resident, the release of the VRAM-held ones was undone by the driver at once, and a
    // 32 GB PC ran with ~0.3 GB free: the kernel swapped, and a turn's checkpoint took 25-80 s.  Above this size
    // ROCclr stages through its own pinned buffers instead; decode is unchanged (49.06 vs 49.16 ms/window).
    if (std::getenv("STRATA_ARENA_MMAP") && std::getenv("STRATA_ARENA_MMAP")[0] == '1')
        setenv("GPU_PINNED_MIN_XFER_SIZE", "1000000", 0);   // MiB; an explicit setting wins
#endif
    // Load every CUDA kernel when the context is created, before the expert cache takes the free VRAM.  With the
    // default lazy loading, a kernel first used mid-prompt (MMQ for IQ3_XXS at 64K+ on a 12 GB card) found no VRAM
    // left for its code and the engine ended ("out of memory: cudaFuncSetAttribute").  Costs ~30 MB of VRAM.
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#endif
    }
    Options o;
    bool borrow_explicit = false;   // SYCL port: --prefill-borrow / --no-prefill-borrow given (else chosen by context)
    bool have_tokens = false;
    bool have_logits_stride = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        bool parsed = true;
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--pack") o.pack = next("--pack");
        else if (a == "--tokens") {
            if (have_tokens) { std::fprintf(stderr, "supply one token input only\n"); return 2; }
            std::string e;
            if (!parse_i64_list(next("--tokens"), o.tokens, e)) { std::fprintf(stderr, "%s\n", e.c_str()); return 2; }
            have_tokens = true;
        }
        else if (a == "--tokens-file") {
            if (have_tokens) { std::fprintf(stderr, "supply one token input only\n"); return 2; }
            std::ifstream input(next("--tokens-file"));
            if (!input) { std::fprintf(stderr, "cannot open token file\n"); return 2; }
            std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            if (input.bad()) { std::fprintf(stderr, "cannot read token file\n"); return 2; }
            std::string error;
            if (text.find('\0') != std::string::npos || !parse_i64_list(text.c_str(), o.tokens, error)) {
                std::fprintf(stderr, "malformed token file: %s\n", error.c_str()); return 2;
            }
            have_tokens = true;
        }
        else if (a == "--max-new") o.max_new = std::atoll(next("--max-new"));
        else if (a == "--max-context") o.max_context = std::atoll(next("--max-context"));
        else if (a == "--rope-scaling") o.rope_scaling = next("--rope-scaling");
        else if (a == "--rope-scale") o.rope_scale = std::atof(next("--rope-scale"));
        else if (a == "--rope-freq-base") o.rope_freq_base = std::atof(next("--rope-freq-base"));
        else if (a == "--rope-freq-scale") o.rope_freq_scale = std::atof(next("--rope-freq-scale"));
        else if (a == "--yarn-orig-ctx") o.yarn_orig_ctx = std::atof(next("--yarn-orig-ctx"));
        else if (a == "--yarn-ext-factor") o.yarn_ext_factor = std::atof(next("--yarn-ext-factor"));
        else if (a == "--yarn-attn-factor") o.yarn_attn_factor = std::atof(next("--yarn-attn-factor"));
        else if (a == "--yarn-beta-fast") o.yarn_beta_fast = std::atof(next("--yarn-beta-fast"));
        else if (a == "--yarn-beta-slow") o.yarn_beta_slow = std::atof(next("--yarn-beta-slow"));
        else if (a == "--greedy") o.greedy = true;
        else if (a == "--seed") { o.seed = (uint64_t) std::atoll(next("--seed")); o.greedy = false; }
        else if (a == "--top-k") o.top_k = std::atoi(next("--top-k"));
        else if (a == "--top-p") o.top_p = (float) std::atof(next("--top-p"));
        else if (a == "--temperature") o.temperature = (float) std::atof(next("--temperature"));
        else if (a == "--dump-logits") o.dump_logits = next("--dump-logits");
        else if (a == "--logits-stride") {
            if (have_logits_stride) { std::fprintf(stderr, "--logits-stride must be supplied only once\n"); return 2; }
            if (!strata::program::logits_selection::parse_stride(next("--logits-stride"), o.logits_stride)) {
                std::fprintf(stderr, "--logits-stride requires a positive decimal int64\n"); return 2;
            }
            have_logits_stride = true;
        }
        else if (a == "--dump-residual") o.dump_residual = next("--dump-residual");
        else if (a == "--dump-mixed") o.dump_mixed = next("--dump-mixed");
        else if (a == "--dump-layers") o.dump_layers = next("--dump-layers");
        else if (a == "--dump-halves") o.dump_halves = next("--dump-halves");
        else if (a == "--dump-routing") o.dump_routing = next("--dump-routing");
        else if (a == "--ple-gguf") o.ple_gguf = next("--ple-gguf");
        else if (a == "--no-ple") o.no_ple = true;
        else if (a == "--ple-io") { o.ple_io = next("--ple-io"); o.ple_io_explicit = true; }
        else if (a == "--ple-row-cache") o.ple_row_cache = std::atoll(next("--ple-row-cache"));
        else if (a == "--ple-inflight") o.ple_inflight = std::atoi(next("--ple-inflight"));
        else if (a == "--ple-delay-us") o.ple_delay_us = std::atof(next("--ple-delay-us"));
        else if (a == "--ple-sync-submit") o.ple_sync_submit = true;
        else if (a == "--kv") o.kv = next("--kv");
        else if (a == "--kv-resident") o.kv_resident = std::atoll(next("--kv-resident"));
        else if (a == "--kv-grow") o.kv_grow = true;
        else if (a == "--no-kv-grow") o.kv_grow = false;
        else if (a == "--stream-token") o.stream_token = true;
        else if (a == "--check-logits") o.check_logits = true;
        else if (a == "--gr-fp32-activations") o.gr_fp32_activations = true;
        else if (a == "--gr-native-mmvf") o.gr_native_mmvf = true;
        else if (a == "--native-bf16") o.native_bf16 = true;
        else if (a == "--native-bf16-extra") o.native_bf16_extra = true;
        else if (a == "--native-ple-key") o.native_ple_key = true;
        else if (a == "--native-moe-combine") o.native_moe_combine = true;
        else if (a == "--native-gdn") o.native_gdn = true;
        else if (a == "--native-flash-attn-short") o.native_flash_attn_short = true;
        else if (a == "--native-qsa-indexer") o.native_qsa_indexer = true;
        else if (a == "--native-qsa") o.native_qsa = true;
        else if (a == "--native-rope") o.native_rope = true;
        else if (a == "--native-ple-postops") o.native_ple_postops = true;
        else if (a == "--native-router") o.native_router = true;
        else if (a == "--cpu-oracle-q8-0") o.cpu_oracle_q8_0 = true;
        else if (a == "--native") o.native_preset = next("--native");
        else if (a == "--native-head-gguf") o.native_head_gguf = next("--native-head-gguf");
        else if (a == "--embd-gguf") o.embd_gguf = next("--embd-gguf");
        else if (a == "--native-dense-gguf") o.native_dense_gguf.push_back(next("--native-dense-gguf"));
        else if (a == "--no-capture") o.no_capture = true;
        else if (a == "--no-pool") o.no_pool = true;
        else if (a == "--sync-every-layer") o.sync_every_layer = true;
        else if (a == "--stage-timing") o.stage_timing = true;
        else if (a == "--graph-only") o.graph_only = true;
        else if (a == "--gpu-only-full") o.gpu_only_full = true;
        else if (a == "--pool-workers") o.pool_workers = std::atoi(next("--pool-workers"));
        else if (a == "--pool-cores") o.pool_cores = next("--pool-cores");
        else if (a == "--host-core") {
            o.host_core = next("--host-core");
            if (o.host_core != "first" && o.host_core != "last") {
                std::fprintf(stderr, "strata generate: unknown --host-core value '%s' (expected first or last)\n", o.host_core.c_str());
                return 2;
            }
        }
        else if (a == "--pool-affinity") {
            const std::string v = next("--pool-affinity");
            if (v == "auto") o.pool_affinity = strata::kernels::cpu::PoolAffinity::Auto;
            else if (v == "p-cores" || v == "pcores") o.pool_affinity = strata::kernels::cpu::PoolAffinity::PCores;
            else if (v == "all") o.pool_affinity = strata::kernels::cpu::PoolAffinity::All;
            else {
                std::fprintf(stderr, "strata generate: unknown --pool-affinity value '%s' (expected auto, p-cores, or all)\n", v.c_str());
                return 2;
            }
        }
        else if (a == "--no-host-worker") o.no_host_worker = true;
        else if (a == "--no-ple-prefetch") o.no_ple_prefetch = true;
        else if (a == "--coupled-draft") o.coupled_draft = true;
        else if (a == "--no-coupled-draft") o.coupled_draft = false;
        else parsed = false;
        // The chain continues here in a second statement: one chain of 120+ `else if` passed MSVC's limit of 128
        // nested blocks (C1061).  The order of the tests and what each does are unchanged.
        if (!parsed) {
        if (a == "--expert-cache") {
            const std::string v = next("--expert-cache");
            o.expert_cache = (v == "auto") ? -1 : std::atoi(v.c_str());
        }
        else if (a == "--expert-cache-device1" || a == "--expert-cache-device2" || a == "--expert-cache-device3") {
            const size_t r = (size_t) (a.back() - '1');
            const std::string v = next(a.c_str());
            o.expert_cache_remote_auto[r] = v == "auto";
            o.expert_cache_remote[r] = v == "auto" ? std::numeric_limits<int>::max() : std::atoi(v.c_str());
        }
        else if (a == "--remote-expert-opt") o.remote_expert_opt = true;
        else if (a == "--expert-cache-remote-placement")
            o.expert_cache_remote_placement = next("--expert-cache-remote-placement");
        else if (a == "--vram-reserve-mib") { o.vram_reserve_mib = std::atoi(next("--vram-reserve-mib")); o.vram_reserve_given = true; }
        else if (a == "--vram-reserve-later-mib")
            o.vram_reserve_later_mib = std::atoi(next("--vram-reserve-later-mib"));
        else if (a == "--prefill") {
            const std::string v = next("--prefill");
            o.prefill_auto = v == "auto" || v.rfind("auto:", 0) == 0;
            // #282: auto:N (N = 16384 or 32768) or STRATA_PREFILL_AUTO_MAX lets auto take chunks above 8192
            const char* env_max = std::getenv("STRATA_PREFILL_AUTO_MAX");
            const long long want_max = v.rfind("auto:", 0) == 0 ? std::atoll(v.c_str() + 5)
                                     : (o.prefill_auto && env_max != nullptr ? std::atoll(env_max) : 8192);
            o.prefill_auto_max = want_max >= 32768 ? 32768 : want_max >= 16384 ? 16384 : 8192;
            o.prefill_chunk = o.prefill_auto ? o.prefill_auto_max : std::atoll(v.c_str());
        }
        else if (a == "--no-split-rows") o.no_split_rows = true;
        else if (a == "--no-prefill-borrow") { o.no_prefill_borrow = true; borrow_explicit = true; }
        else if (a == "--prefill-borrow") { o.no_prefill_borrow = false; borrow_explicit = true; }
        else if (a == "--vram-elastic") o.vram_elastic = true;
        else if (a == "--vram-segment-mib") o.vram_segment_mib = std::atoll(next("--vram-segment-mib"));
        else if (a == "--prefill-until") o.prefill_until = std::atoll(next("--prefill-until"));
        else if (a == "--dump-final-r") o.dump_final_r = next("--dump-final-r");
        else if (a == "--spec") o.spec = std::atoi(next("--spec"));
        else if (a == "--batch") o.batch = std::atoi(next("--batch"));
        else if (a == "--slots") o.batch = std::atoi(next("--slots"));   // the same as --batch
        else if (a == "--trim-stage-weights") o.trim_stage_weights = true;
        else if (a == "--batch-groups") o.batch_groups = std::atoi(next("--batch-groups"));
        else if (a == "--batch-mtp") o.batch_mtp = true;
        else if (a == "--spec-oracle") o.spec_oracle = next("--spec-oracle");
        else if (a == "--spec-corrupt") o.spec_corrupt = std::atoi(next("--spec-corrupt"));
        else if (a == "--spec-follow") o.spec_follow = next("--spec-follow");
        else if (a == "--window-hashes") o.window_hashes = next("--window-hashes");
        else if (a == "--mtp") o.mtp = next("--mtp");
        else if (a == "--mtp-window") o.mtp_window = std::atoll(next("--mtp-window"));
        else if (a == "--pcie-frac") o.pcie_frac = std::atof(next("--pcie-frac"));
        else if (a == "--adapt-every") o.adapt_every = std::atoi(next("--adapt-every"));
        else if (a == "--adapt-decay") o.adapt_decay = (float) std::atof(next("--adapt-decay"));
        else if (a == "--adapt-async") o.adapt_async = std::atoi(next("--adapt-async")) != 0 ? 1 : 0;
        else if (a == "--spec-min-p") o.spec_min_p = std::atof(next("--spec-min-p"));
        else if (a == "--stop-eos") o.stop_eos = true;
        else if (a == "--spec-split") o.spec_split = true;
        else if (a == "--layer-split") o.layer_split = next("--layer-split");
        else if (a == "--split-device") o.split_device = next("--split-device");
        else if (a == "--split-skip-if-fits") o.split_skip_if_fits = true;
        else if (a == "--pcie-mode") o.pcie_mode = next("--pcie-mode");
        else if (a == "--serve") o.serve = true;
        else if (a == "--vision") o.vision = true;
        else if (a == "--prompt-cache") o.prompt_cache = std::max(0, std::atoi(next("--prompt-cache")));
        else if (a == "--conversation-cache-mib" || a == "--conversation-cache-slots" ||
                 a == "--conversation-cache-min-free-mib" || a == "--session-min-free-mib") {
            const std::string value = next(a.c_str());
            int64_t number = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
            const int64_t limit = a == "--conversation-cache-slots" ? INT32_MAX : INT64_MAX / (1024 * 1024);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || number < 0 || number > limit) {
                std::fprintf(stderr, "%s needs a nonnegative integer within range\n", a.c_str());
                return 2;
            }
            if (a == "--conversation-cache-mib") o.conversation_cache_mib = number;
            else if (a == "--conversation-cache-min-free-mib") o.conversation_cache_min_free_mib = number;
            else if (a == "--session-min-free-mib") o.session_min_free_mib = number;
            else o.conversation_cache_slots = (int) number;
        }
        else if (a == "--prompt-cache-tail") o.prompt_cache_tail = true;
        else if (a == "--prompt-cache-every") o.prompt_cache_every = std::max(0LL, std::atoll(next("--prompt-cache-every")));
        else if (a == "--prompt-cache-root") o.prompt_cache_root = std::max(0LL, std::atoll(next("--prompt-cache-root")));
        else if (a == "--turn-token") o.turn_token = std::atoll(next("--turn-token"));
        else if (a == "--tail-role-token") o.tail_role_token = std::atoll(next("--tail-role-token"));
        else if (a == "--short-read") o.short_read = std::max(0LL, std::atoll(next("--short-read")));
        else if (a == "--pipeline-windows") o.pipeline_windows = std::max(0, std::min(2, std::atoi(next("--pipeline-windows"))));
        else if (a == "--suffix-draft") o.suffix_draft = std::max(0, std::atoi(next("--suffix-draft")));
        else if (a == "--mtp-max-t") o.mtp_max_t = std::max(0, std::atoi(next("--mtp-max-t")));
        else if (a == "--lookup-chain") o.lookup_chain = std::clamp(std::atoi(next("--lookup-chain")), 0, 7);
        else if (a == "--lookup-chain-min") o.lookup_chain_min = std::max(3, std::atoi(next("--lookup-chain-min")));
        else if (a == "--mtp-hnorm") {
            const std::string v = next("--mtp-hnorm");
            if (v != "pooled" && v != "stream") {
                std::fprintf(stderr, "strata generate: --mtp-hnorm takes pooled or stream\n");
                return 2;
            }
            o.mtp_hnorm_stream = v == "stream";
        }
        else if (a == "--mtp-q4") {
            o.mtp_q4 = next("--mtp-q4");
            if (o.mtp_q4 != "all" && o.mtp_q4 != "proj" && o.mtp_q4 != "head") {
                std::fprintf(stderr, "strata generate: --mtp-q4 takes all, proj or head\n");
                return 2;
            }
        }
        else if (a == "--mtp-draft-vocab") o.mtp_draft_vocab = next("--mtp-draft-vocab");
        else if (a == "--control-vector") o.cvec_files.push_back({next("--control-vector"), 1.0f});
        else if (a == "--control-vector-scaled") {
            // FILE:SCALE, comma-separated; the LAST colon splits, so a Windows path (C:\...) keeps its drive
            std::stringstream list(next("--control-vector-scaled"));
            std::string item;
            while (std::getline(list, item, ',')) {
                const size_t colon = item.rfind(':');
                char* end = nullptr;
                const float sc = colon == std::string::npos ? 0.0f : std::strtof(item.c_str() + colon + 1, &end);
                if (colon == std::string::npos || colon == 0 || end == item.c_str() + colon + 1 || *end != '\0') {
                    std::fprintf(stderr, "--control-vector-scaled: expected FILE:SCALE, got '%s'\n", item.c_str());
                    return 2;
                }
                o.cvec_files.push_back({item.substr(0, colon), sc});
            }
        }
        else if (a == "--control-vector-layer-range") {
            o.cvec_first = std::atoi(next("--control-vector-layer-range"));
            o.cvec_last = std::atoi(next("--control-vector-layer-range"));
        }
        else if (a == "--cvec-mode") {
            const std::string m = next("--cvec-mode");
            if (m == "project") o.cvec_mode = 0;
            else if (m == "add") o.cvec_mode = 1;
            else { std::fprintf(stderr, "--cvec-mode: add or project, got '%s'\n", m.c_str()); return 2; }
        }
        else if (a == "--cvec-dir") {
            const std::string d = next("--cvec-dir");
            if (d == "per-layer") o.cvec_single = -1;
            else if (d.rfind("single:", 0) == 0) o.cvec_single = std::atoi(d.c_str() + 7);
            else { std::fprintf(stderr, "--cvec-dir: per-layer or single:L, got '%s'\n", d.c_str()); return 2; }
        }
        else if (a == "--no-spec-split") o.spec_split = false;
        else if (a == "--eos-ids") {
            std::string e;
            if (!parse_i64_list(next("--eos-ids"), o.eos_ids, e)) { std::fprintf(stderr, "--eos-ids: %s\n", e.c_str()); return 2; }
            o.stop_eos = true;
        }
        else if (a == "--adapt-swaps") o.adapt_swaps = std::atoi(next("--adapt-swaps"));
        else if (a == "--expert-cache-cpu-order") o.expert_cache_cpu_order = true;
        else if (a == "--expert-cache-per-layer") o.expert_cache_per_layer = true;
        else if (a == "--peer-device") o.peer_device = std::atoi(next("--peer-device"));
        else if (a == "--peer-reserve-mib") o.peer_reserve_mib = std::atoi(next("--peer-reserve-mib"));
        else if (a == "--peer-slots") o.peer_slots = std::atoll(next("--peer-slots"));
        else if (a == "--peer-adapt-swaps") o.peer_adapt_swaps = std::atoi(next("--peer-adapt-swaps"));
        else if (a == "--peer-prefill-rows") o.peer_prefill_rows = std::atoll(next("--peer-prefill-rows"));
        else if (a == "--no-hit-poke") o.no_hit_poke = true;
        else if (a == "--expert-profile") o.expert_profile = next("--expert-profile");
        else if (a == "--expert-profile-save") o.expert_profile_save = next("--expert-profile-save");
        else if (a == "--expert-profile-save-every")
            o.expert_profile_save_min = std::atof(next("--expert-profile-save-every"));
        else if (a == "--gpu-stages") o.gpu_stages = true;
        else if (a == "--mmap-experts") o.mmap_experts = true;
        else if (a == "--shared-expert-arena") o.shared_expert_arena = next("--shared-expert-arena");
        else if (a == "--resident-cpu-experts") o.resident_cpu_experts = o.resident_cpu_explicit = true;
        else if (a == "--stream-experts") o.stream_experts = true;
        else if (a == "--resident-experts") {
            o.mmap_experts = o.resident_cpu_experts = o.resident_pin = o.resident_soft = true;
            o.resident_headroom = 4ull << 30;
            // A/B arms: STRATA_RESIDENT_PIN=0 keeps the copy pageable (the --resident-cpu-experts form);
            // STRATA_RESIDENT_HEADROOM_GIB=N leaves N GiB of the available RAM free instead of 4
            if (const char* v = std::getenv("STRATA_RESIDENT_PIN"); v != nullptr && std::string(v) == "0")
                o.resident_pin = false;
            if (const char* v = std::getenv("STRATA_RESIDENT_HEADROOM_GIB"); v != nullptr && std::atof(v) >= 0.0)
                o.resident_headroom = (uint64_t) (std::atof(v) * 1073741824.0);
        }
        else if (a == "--resident-budget-gib") {
            const double gib = std::atof(next("--resident-budget-gib"));
            if (!(gib > 0.0)) { std::fprintf(stderr, "strata generate: --resident-budget-gib needs N > 0\n"); return 2; }
            o.resident_budget = (uint64_t) (gib * 1073741824.0);
            o.mmap_experts = o.resident_cpu_experts = o.resident_pin = true;
            o.resident_headroom = 4ull << 30;
            if (const char* v = std::getenv("STRATA_RESIDENT_PIN"); v != nullptr && std::string(v) == "0")
                o.resident_pin = false;
            if (const char* v = std::getenv("STRATA_RESIDENT_HEADROOM_GIB"); v != nullptr && std::atof(v) >= 0.0)
                o.resident_headroom = (uint64_t) (std::atof(v) * 1073741824.0);
        }
        else if (a == "--stats") o.stats = true;
        else if (a == "--shared-late") o.shared_late = true;
        else if (a == "--keep-canonical") o.keep_canonical = true;
        else if (a == "--no-token-graph") o.no_token_graph = true;
        else if (a == "--no-fused-gr") o.no_fused_gr = true;
        else if (a == "--no-fast-attn") o.no_fast_attn = true;
        else if (a == "--no-publish-kernel") o.no_publish_kernel = true;
        else if (a == "--no-fused-gdn") o.no_fused_gdn = true;
        else if (a == "--no-fast-select") o.no_fast_select = true;
        else {
            // An unknown flag is an ERROR and not a warning: a typo'd `--max-neww` that silently generated 16
            // tokens would look like a working run.
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage();
            return 2;
        }
        }
    }
#if defined(STRATA_USE_HIP)
    {   // gfx1151 (Strix Halo): the switches that are exact there are on by default (strata/core/arch_defaults.hpp); before
        // any engine code reads its switches (they are read at first use), and the user's own settings are kept
        cudaDeviceProp ad{};
        if (cudaGetDeviceProperties(&ad, 0) == cudaSuccess) strata::core::apply_arch_defaults(ad.gcnArchName);
        else (void) cudaGetLastError();
    }
#endif
    strata::core::set_coupled_draft(o.coupled_draft);
    {   // --host-core / STRATA_HOST_CORE, before the pool and the session pin any thread
        std::string hc = o.host_core;
        if (hc.empty())
            if (const char* e = std::getenv("STRATA_HOST_CORE")) hc = e;
        if (hc == "last") strata::kernels::cpu::set_host_core(strata::kernels::cpu::HostCore::Last);
    }
    {   // --pool-cores / STRATA_POOL_CORES (experimental, off by default), before the pool is built; a list that cannot be read is
        // said and left out (recommend, never force)
        std::string pc = o.pool_cores;
        if (pc.empty())
            if (const char* e = std::getenv("STRATA_POOL_CORES")) pc = e;
        if (!pc.empty()) {
            std::vector<int> cpus;
            std::string perr;
            if (strata::kernels::cpu::parse_pool_cores(pc, cpus, perr)) strata::kernels::cpu::set_pool_cores(cpus);
            else std::fprintf(stderr, "strata generate: --pool-cores: %s; the pool uses its default placement\n", perr.c_str());
        }
    }
    strata::core::set_peer_portable(o.peer_device >= 1);   // multi-GPU: the Portable flag on mapped host buffers only with a peer device (before any allocation)
    if (o.serve && o.conversation_cache_mib > 0 && (o.prompt_cache == 0 || o.conversation_cache_slots == 0))
        std::fprintf(stderr, "strata serve: warning: conversation caching is disabled by %s\n",
                     o.prompt_cache == 0 ? "--prompt-cache 0" : "--conversation-cache-slots 0");
    // parking with --layer-split saves every stage (SavedConversation::stage_images)
    // Layer split (multi-GPU): the later stages run layers [K_i, K_i+1) on their own GPUs (--split-device, default
    // the next visible ones); "auto" places the K from each GPU's free VRAM once the weights are in (below).  KV
    // streaming and images (the server's GPU encoder) run across a split, also with parking and --batch.  Across
    // GPUs, not yet: control vectors, the helper caches (--expert-cache-remote), and lending cache slots to the
    // prompt path (each stage's prompt path has its own buffers).
    // --pcie-frac given: that one share is every stage's (the stages' own link probes are skipped), so a split over
    // a fast and a slow link cannot set the two apart from the command line (#485)
    const bool pcie_given = o.pcie_frac >= 0.0;
    std::vector<int64_t> split_at;
    std::vector<int> split_devs;
    bool split_auto = false, split_same = false;
    if (!o.layer_split.empty()) {
        int n_dev = 1;
        if (DPCT_CHECK_ERROR(n_dev = dpct::device_count()) != 0 || n_dev < 1)
            n_dev = 1;
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        auto ints = [](const std::string &str, auto &out) -> bool {
            using V = typename std::decay_t<decltype(out)>::value_type;
            size_t a = 0;
            while (a < str.size()) {
                size_t b = str.find(',', a);
                if (b == std::string::npos) b = str.size();
                const std::string t = str.substr(a, b - a);
                if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos) return false;
                out.push_back((V) std::atoll(t.c_str()));
                a = b + 1;
            }
            return !out.empty();
        };
        split_auto = o.layer_split == "auto";
        bool ok = o.serve && (split_auto || ints(o.layer_split, split_at));
        if (ok && !o.split_device.empty()) ok = ints(o.split_device, split_devs);
        else if (ok)
            for (int d = 1; d < n_dev && (split_auto || split_devs.size() < split_at.size()); ++d) split_devs.push_back(d);
        if (ok && !split_auto && split_devs.empty() && split_at.size() == 1) split_devs.push_back(0);   // one GPU
        split_same = ok && split_devs.size() == 1 && split_devs[0] == 0 && !split_auto;
        if (split_same && o.conversation_cache_mib > 0 && o.conversation_cache_slots > 0 && o.prompt_cache > 0) {
            std::fprintf(stderr, "strata serve: conversation parking does not support a layer split on one GPU "
                                 "(--split-device 0); disable parking with --conversation-cache-mib 0\n");
            return 2;
        }
        if (ok && split_auto && split_devs.empty()) {
            std::fprintf(stderr, "strata generate: --layer-split auto: one GPU visible, so no split\n");
            o.layer_split.clear();
            split_auto = false;
        } else if (ok) {
            ok = (split_auto || split_at.size() == split_devs.size()) && split_devs.size() < (size_t) SplitDrive::kMax;
            for (size_t i = 0; ok && i < split_at.size(); ++i) ok = split_at[i] >= 2 && (i == 0 || split_at[i] > split_at[i - 1]);
            for (size_t i = 0; ok && !split_same && i < split_devs.size(); ++i) {
                ok = split_devs[i] > 0 && split_devs[i] < n_dev;
                for (size_t j = 0; ok && j < i; ++j) ok = split_devs[i] != split_devs[j];
            }
        }
        if (!ok) {
            std::fprintf(stderr, "strata generate: --layer-split K[,K2..]|auto needs --serve, rising K from 2, and one "
                                 "distinct GPU per K in --split-device (1..%d; or 0 with one K: the same GPU). K is the "
                                 "FIRST LAYER of each later GPU, not a count of layers per card: \"24,36,42\" for 4 "
                                 "GPUs, not \"24,12,6,6\" (got \"%s\")\n", n_dev - 1, o.layer_split.c_str());
            return 2;
        }
    }
    bool multi_gpu = !split_devs.empty() && !split_same;   // cleared by --split-skip-if-fits before any stage loads
    bool split_own_auto = false;   // #340: the split keeps own prompt buffers by its rule (not --no-prefill-borrow)
    if (o.mmap_experts && !o.shared_expert_arena.empty()) {
        std::fprintf(stderr, "strata generate: --shared-expert-arena backs the resident arena and cannot be used with --mmap-experts\n");
        return 2;
    }
    if (o.resident_cpu_experts && (!o.mmap_experts || o.expert_profile.empty())) {
        std::fprintf(stderr, "strata generate: --resident-cpu-experts requires --mmap-experts and a static --expert-profile\n");
        return 2;
    }
    const bool remote_caches = o.expert_cache_remote[0] > 0 || o.expert_cache_remote[1] > 0 ||
                               o.expert_cache_remote[2] > 0;
    // A layer split keeps the resident RAM mode: every stage's GPU cache is left out of the RAM copy, and an adaptive
    // swap copies an evicted expert back from the card that owns its layer (resident_stage_swaps).
    if (o.resident_cpu_experts && remote_caches) {
        std::fprintf(stderr, "strata generate: --resident-cpu-experts does not support remote expert caches\n");
        return 2;
    }
    // the helper-GPU expert caches (--expert-cache-remote, docs/SECOND_GPU.md): CUDA1..3 on one GPU; with a layer
    // split, the visible GPUs no stage runs on, in order
    int remote_dev[3] = {1, 2, 3};
    if (multi_gpu) {
        if (o.expert_profile.empty()) {
            std::fprintf(stderr, "strata generate: a layer split across GPUs needs --expert-profile\n");
            return 2;
        }
        // A STAGE'S PROMPT PATH BORROWS FROM THAT STAGE'S OWN EXPERT CACHE.  This used to set
        // `no_prefill_borrow = true` - "each stage's prompt path has its own buffers" - which is true, but it is
        // a reason to give each stage its own LOAN, not a reason to make every stage withhold a chunk-sized
        // reserve from its cache for the whole session.  Forced on, it also collapsed `--prefill auto` to 2048
        // (below) and skipped the lend arm, so a stage paid for its prompt buffers twice over: once in VRAM it
        // never got back, once in the smaller chunk.  At `--prefill 5524` that reserve is 3.8 GiB per stage,
        // more than either 8 GB card had - which is how adding two GPUs to the two 12 GB ones lost 260K.
        // The loan is the tail of the stage's own cache (see `PfPart` in the serve block); outside the prompt
        // that tail is expert cache, so a large chunk costs a stage nothing permanent.
        // #340 - BUT A LOAN IS NOT FREE PER REQUEST: every stage streams the lent experts during the prompt and
        // copies them back after it, so on cards that hold (nearly) every expert of their layers a 2K prompt read
        // 37% slower than 0.1.29's own buffers (2x RX 9070 XT / R9700: 1570 -> 990 tok/s; 2x A5000 in #340: -35%).
        // So a split keeps 0.1.29's own buffers (2048-token chunks, priced into the split search) when they are a
        // small part of every card - at most 12% of its VRAM (16 GB and larger cards) - and borrows on smaller cards,
        // where the reserve would cost the cache (and the context) the paragraph above is about.  Measured with own
        // buffers on the 9070 XT + R9700: 2K 1588 tok/s, 16K 2006 (0.1.29 1568 / 1935, 0.1.31 993 / 1852).
        // OPT-IN (the owner's choice for 0.1.32): own buffers change which experts a full card keeps resident, so the
        // split's output differs from 0.1.31's; the default borrows as 0.1.31 did - with the concurrent refill and the
        // 96-blob split ring that alone gave 2K +27%, 16K +5% over 0.1.31, decode unchanged, output identical.
        // STRATA_SPLIT_OWN=auto: the 12% rule above; 1: own buffers on any cards; unset/0: borrow.
        // --split-skip-if-fits keeps the loans (it can fall back to one GPU, which must stay as it is).
        const char* own_env = std::getenv("STRATA_SPLIT_OWN");
        if (!o.no_prefill_borrow && !o.split_skip_if_fits && own_env != nullptr && own_env[0] != '0') {
            const char* v = std::string(own_env) == "auto" ? nullptr : own_env;
            bool own = v ? v[0] == '1' : true;
            // the buffers of a 2048-token chunk (or the --prefill one) + a 96-blob ring (as split_pf_mib prices them)
            const int64_t own_chunk = o.prefill_auto || o.prefill_chunk <= 0 ? 2048 : o.prefill_chunk;
            const int64_t reserve_mib = 160 + (own_chunk * 680) / 1024 + 96 * 4;
            std::string why;
            if (!v) {
                std::vector<int> devs = {0};
                for (const int d : split_devs) devs.push_back(d);
                for (const int d : devs) {
                    dpct::device_info prop{};
                    /*
                    DPCT1026: The call to cudaGetLastError was removed
                    because this functionality is redundant in SYCL.
                    */
                    if (DPCT_CHECK_ERROR(
                            dpct::get_device(d).get_device_info(prop)) != 0) {
                        ; own = false; break;
                    }
                    const int64_t total_mib =
                        (int64_t)(prop.get_global_mem_size() >> 20);
                    if (reserve_mib * 100 > 12 * total_mib) {
                        own = false;
                        why = "CUDA" + std::to_string(d) + " has " + std::to_string(total_mib) + " MiB";
                        break;
                    }
                }
            }
            if (own) {
                o.no_prefill_borrow = true;
                split_own_auto = true;
                std::fprintf(stderr, "strata generate: layer split: every stage keeps its own prompt buffers (~%lld MiB "
                                     "each, %lld-token chunks)%s\n", (long long) reserve_mib, (long long) own_chunk,
                             v ? " (STRATA_SPLIT_OWN=1)" : "");
            } else if (!why.empty()) {
                std::fprintf(stderr, "strata generate: layer split: the prompt paths borrow from the caches (%s)\n",
                             why.c_str());
            }
        }
        int n_vis = 1;
        if (DPCT_CHECK_ERROR(n_vis = dpct::device_count()) != 0 || n_vis < 1)
            n_vis = 1;
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        int next_free = 1;
        for (int r = 0; r < 3; ++r) {
            if (o.expert_cache_remote[(size_t) r] <= 0) continue;
            while (next_free < n_vis &&
                   std::find(split_devs.begin(), split_devs.end(), next_free) != split_devs.end()) ++next_free;
            if (next_free >= n_vis) {
                std::fprintf(stderr, "strata generate: --expert-cache-remote with a layer split needs a GPU that runs no "
                                     "stage (%d visible, %zu used by the split)\n", n_vis, split_devs.size() + 1);
                return 2;
            }
            remote_dev[r] = next_free++;
        }
        std::string devs;
        for (const int d : split_devs) devs += (devs.empty() ? "" : ",") + std::to_string(d);
        std::fprintf(stderr, "strata generate: layer split across %zu GPUs: CUDA0, then CUDA%s (split %s)\n",
                     split_devs.size() + 1, devs.c_str(), o.layer_split.c_str());
    }
    // SYCL port (the B70, 2026-09-30): lending cache slots to the prompt path costs ~1 s per prompt (2,184 tokens:
    // 610 vs 792 tok/s) but is what keeps every expert in VRAM at a long context - 80,000 tokens without it: the
    // reserve for the KV and the chunk buffers evicts ~1,900 experts, prompt 790 tok/s and decode 2 tok/s after it;
    // with it: 1,062 tok/s and 39.5 tok/s after it. So by default only above a 32K context; the flags still decide.
    // (0.1.31-0.1.32: borrowing hung in the first chunk of a long prompt; the prompt path's stager waited on the copy
    // queue's events from its own threads, which the Level Zero v2 adapter did not survive once its ring wrapped -
    // prefill.cpp, Stager::issued_one. Fixed 2026-10-01: a 40K prompt borrowing reads at 1,144 tok/s and decodes at
    // 69 tok/s after it, against 1,201 / 65 with its own buffers.)
    //
    // 2026-10-08 (the A770): borrowing is BROKEN on Alchemist. Its prompt buffers are carved from the top of the
    // ~10 GiB expert-cache allocation, and oneMKL's BF16 GEMM reads a wrong tile when an operand lies more than 4 GiB
    // into its allocation (docs/INTEL.md); the borrowed operands hit that, so from the first full prefill chunk the
    // router read garbage and prefill failed ("prefill: routed id out of range") or the engine segfaulted (measured:
    // a 236-token prompt, mirrored and mmap configs alike). Keep Alchemist's prompt path on its own buffers (its own
    // allocation, reserved exactly by owned_prefill_mib) and keep an automatic chunk under 4 GiB, until the SYCL BLAS
    // stages operands as the CUDA/HIP shims do. Battlemage keeps the old default (its long prompts were measured
    // correct with borrowing). --prefill-borrow overrides, for testing the staging fix. A layer split keeps its rule.
    if (!borrow_explicit) {
        static const bool alchemist = [] {
            try {
                // PCI device ids, the same table setup_intel.py uses: 56a0 A770, 56a1 A750, 56a2 A580, 56a5 A380,
                // 56a6 A310, 5690 A770M
                const unsigned id = dpct::get_current_device().get_info<sycl::ext::intel::info::device::device_id>();
                // Only a card with room for an expert cache past 4 GiB can hit it: on an 8 GB A750 the cache was 0.34 GiB
                // and the own-buffers reservation cost experts (prompt 72 -> 52 tok/s, decode -6%, 4K prompt, 4 rounds)
                if (dpct::get_current_device().get_info<sycl::info::device::global_mem_size>() < (12ull << 30)) return false;
                switch (id & 0xffffu) {
                case 0x56a0: case 0x56a1: case 0x56a2: case 0x56a5: case 0x56a6: case 0x5690: return true;
                default: return false;
                }
            } catch (...) { return false; }
        }();
        if (!multi_gpu && alchemist) {
            o.no_prefill_borrow = true;
            if (o.prefill_auto && o.prefill_chunk > 2048) o.prefill_chunk = o.prefill_auto_max = 2048;
        } else {
            o.no_prefill_borrow = o.max_context <= 32768;
        }
    }
#if defined(STRATA_USE_HIP)
    {
        // every GPU this run uses must be an architecture the binary has code for (a gfx1100 build on a gfx1201
        // card would otherwise fail later with "invalid device function")
        std::vector<int> used{0};
        if (multi_gpu) used.insert(used.end(), split_devs.begin(), split_devs.end());
        for (int r = 0; r < 3; ++r)
            if (o.expert_cache_remote[(size_t) r] > 0) used.push_back(remote_dev[r]);
        for (const int d : used) {
            if (const std::string why = strata::core::gpu_arch_problem(d); !why.empty()) {
                std::fprintf(stderr, "strata generate: %s\n", why.c_str());
                return 1;
            }
        }
    }
#endif
    if (o.prefill_auto && (o.no_prefill_borrow || o.expert_profile.empty())) {
        o.prefill_auto = false;       // nothing to lend from: the buffers are reserved for the session, so keep them small
        o.prefill_chunk = 2048;
    }
    if (!have_tokens && o.serve) {   // plan v0.3 P8: requests bring their own tokens
        o.tokens = {248045};
        o.max_new = 1;
        have_tokens = true;
        o.stop_eos = true;
    }
    if (!have_tokens) {
        std::fprintf(stderr, "strata generate: --tokens is required (this build has no tokenizer; see the "
                             "header of src/program/generate.cpp)\n");
        usage();
        return 2;
    }

    if ((o.ple_io != "direct" && o.ple_io != "mmap" && o.ple_io != "ram") || o.ple_row_cache < 0 || o.ple_inflight < 1 ||
        o.ple_inflight > 1024 || !(o.ple_delay_us >= 0)) {
        std::fprintf(stderr, "strata generate: invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us\n");
        return 2;
    }
#if defined(_WIN32)
    if (o.ple_io == "ram") {
        std::fprintf(stderr, "strata generate: --ple-io ram is not available on Windows (no mlock); use --ple-io mmap\n");
        return 2;
    }
#endif
    if (o.kv == "q4") o.kv = "q4_0";
    if (o.kv != "fp16" && o.kv != "int8" && o.kv != "q4_0" && o.kv != "k8v4") {
        std::fprintf(stderr, "strata generate: --kv must be fp16, int8, q4_0 or k8v4\n");
        return 2;
    }
    strata::core::qsa_set_kv_int8(o.kv == "int8");
    strata::core::qsa_set_kv_q4(o.kv == "q4_0");   // PR #21: 4-bit codes after a Hadamard rotation (kv_q4.hpp)
    // STRATA_KV_ROT=1: INT8 K/V through the Hadamard rotation --kv q4_0 already uses. Opt-in: first-token KL to
    // fp16 K/V improved on an NVFP4 pack (0.0066 -> 0.0051) but not on IQ2_XS (0.0022 -> 0.0054)
    const char* kv_rot = std::getenv("STRATA_KV_ROT");
    strata::core::qsa_set_kv_int8_rotate(kv_rot != nullptr && kv_rot[0] == '1');
    if (kv_rot != nullptr && kv_rot[0] == '1' && o.kv == "int8")
        std::fprintf(stderr, "strata generate: STRATA_KV_ROT=1: INT8 K/V through the Hadamard rotation (opt-in)\n");
    strata::core::qsa_set_kv_hybrid(o.kv == "k8v4");   // K8V4: INT8 K + rotated Q4_0 V, 816 B/cell
    if (o.kv_resident < 0) {
        std::fprintf(stderr, "strata generate: --kv-resident must be >= 0\n");
        return 2;
    }
    strata::core::qsa_set_kv_resident(o.kv_resident);
    // Prompt lookup (the suffix drafter, on by default): the MTP keeps its --spec windows and a lookup window may be
    // up to 2 tokens longer; the draft policy (strata/spec/draft_policy.hpp) takes one only where it pays. Code
    // edits +6-11%, ordinary text unchanged (bench/results/2026-09-27-spec). --suffix-draft 0 turns it off.
    if (o.suffix_draft > 0 && o.spec >= 2 && o.mtp_max_t == 0) {
        o.mtp_max_t = o.spec;
        o.spec = std::min(o.spec + 2, 8);   // kVerifyMaxT
    }
    // --lookup-chain (opt-in): the MTP keeps its windows; a lookup chained after its drafts may lengthen one by up to K
    if (o.lookup_chain > 0 && o.spec >= 2) {
        if (o.mtp_max_t == 0) o.mtp_max_t = o.spec;
        o.spec = std::max(o.spec, std::min(o.mtp_max_t + o.lookup_chain, 8));   // kVerifyMaxT
    }
    strata::spin_drafts_need_long_bound(o.spec >= 2 && o.mtp.empty());   // spin_max: drafting without MTP waits long
    strata::core::layer_set_shared_early(!o.shared_late);
    if (!o.native_preset.empty()) {
        try {
            // every shard of the model (<name>-0000N-of-0000M.gguf beside --native).  A missing shard is an error
            // here: it used to be skipped, leaving a model with some tensors absent and a later error, or none.
            o.native_shards = strata::gguf_split_paths(o.native_preset);
            // a model family Strata recognises but has no engine path for yet (Qwen3.6-35B-A3B) is refused here,
            // with a message that says so, before anything else about the file is judged
            {
                const strata::GgufFile meta(o.native_shards.front());
                const std::string why = strata::model_family_not_supported(strata::model_family(meta));
                if (strata::model_family(meta) != strata::ModelFamily::Unknown && !why.empty()) {
                    std::fprintf(stderr, "strata generate: --native %s: %s\n", o.native_preset.c_str(), why.c_str());
                    return 2;
                }
            }
            // --ple-gguf defaults to the shard that holds the PLE table, found by name: shard 2 of the ISTA files
            // and of Unsloth's UD-Q4_K_XL, shard 1 of Swift's
            if (o.ple_gguf.empty() && !o.no_ple) {
                const strata::GgufModel model(o.native_shards);
                size_t at = 0;
                if (model.find("per_layer_token_embd.weight", &at) != nullptr) o.ple_gguf = o.native_shards[at];
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata generate: --native %s: %s\n", o.native_preset.c_str(), e.what());
            return 2;
        }
        if (o.no_ple || o.ple_gguf.empty()) {
            std::fprintf(stderr, "strata generate: --native requires --ple-gguf (the PLE key is native too), and no "
                                 "shard of the model holds per_layer_token_embd.weight\n");
            return 2;
        }
        o.stream_token = true;
        o.gr_native_mmvf = true;
        o.native_bf16 = o.native_bf16_extra = true;
        o.native_ple_key = o.native_moe_combine = o.native_gdn = o.native_router = true;
        o.native_qsa = o.native_qsa_indexer = o.native_rope = o.native_ple_postops = true;
        if (o.native_head_gguf.empty()) o.native_head_gguf = o.native_preset;
        if (o.native_dense_gguf.empty()) {
            // every shard of the model (<name>-0000N-of-0000M.gguf beside --native), then the PLE shard: a split
            // may put any layer in any shard (Swift's GGUFs: layers 13-47 in shard 2, the PLE table in shard 1)
            o.native_dense_gguf = o.native_shards;
            // a PLE-only table (tools/ple_fp8_pack.py: architecture strata-ple) holds no projections
            bool ple_only = false;
            try {
                strata::GgufFile pg(o.ple_gguf);
                if (const strata::MetaValue* v = pg.get("general.architecture")) ple_only = v->s == "strata-ple";
            } catch (const std::exception&) {}
            if (!ple_only &&
                std::find(o.native_dense_gguf.begin(), o.native_dense_gguf.end(), o.ple_gguf) == o.native_dense_gguf.end())
                o.native_dense_gguf.push_back(o.ple_gguf);
        }
        // Plan v0.3 (24 Sep): the CPU experts stay on the VNNI kernel.  The llama.cpp-CPU-exact q8_0 contract
        // cost 27.0 vs 17.2 ms/token of pool time and G-C does not need it; `--cpu-oracle-q8-0` still selects it.
    }
    if (o.logits_stride > 1 && (o.max_new != 1 || o.dump_logits.empty())) {
        std::fprintf(stderr, "strata generate: --logits-stride > 1 requires --max-new 1 and --dump-logits\n");
        return 2;
    }
    if (o.no_ple && !o.ple_gguf.empty()) {
        std::fprintf(stderr, "strata generate: --no-ple and --ple-gguf are mutually exclusive\n");
        return 2;
    }
    if (o.native_ple_postops && o.no_ple) {
        std::fprintf(stderr, "strata generate: --native-ple-postops requires PLE enabled\n");
        return 2;
    }
    if (!o.no_ple && o.ple_gguf.empty()) {
        std::fprintf(stderr, "strata generate: --ple-gguf is required; --no-ple explicitly enables a diagnostic ablation\n");
        return 2;
    }
    // P7 audit: positions, cells and pooled-block indices are cast to int32 on the device path.
    if (o.max_context > 2147483647LL - 8) {
        std::fprintf(stderr, "strata generate: --max-context must be below 2^31\n");
        return 2;
    }
    if (o.max_new <= 0 || o.max_context <= 0 || o.max_new > o.max_context ||
        o.tokens.size() > (size_t) (o.max_context - o.max_new)) {
        std::fprintf(stderr, "strata generate: positive --max-new and --max-context must fit the prompt and generation\n");
        return 2;
    }
    // THE ROPE KNOBS (rope_scaling.hpp).  Anything invalid dies here, at second zero, rather than becoming a
    // NaN angle inside one of the twelve QSA layers.  Only the RANGES are checked - the config itself is
    // resolved after the model file has had its say, right before session_init.
    strata::kernels::RopeScaling rope_cfg;   // type filled here; the rest at the resolution below
    {
        using RST = strata::kernels::RopeScalingType;
        // an absent --rope-scaling (the empty default) leaves the type to the model file's rope keys,
        // resolved below; anything present must be one of the three names
        if (o.rope_scaling == "none") rope_cfg.type = RST::None;
        else if (o.rope_scaling == "linear") rope_cfg.type = RST::Linear;
        else if (o.rope_scaling == "yarn") rope_cfg.type = RST::YaRN;
        else if (!o.rope_scaling.empty()) {
            std::fprintf(stderr, "strata generate: --rope-scaling must be none, linear or yarn (got '%s')\n",
                         o.rope_scaling.c_str());
            return 2;
        }
        // every knob FINITE first: `atof("nan")` is NaN, and a NaN passes every range comparison below
        for (const double v : {o.rope_scale, o.rope_freq_base, o.rope_freq_scale, o.yarn_orig_ctx, o.yarn_ext_factor,
                               o.yarn_attn_factor, o.yarn_beta_fast, o.yarn_beta_slow})
            if (!std::isfinite(v)) {
                std::fprintf(stderr, "strata generate: a rope scaling knob is not a finite number (%g)\n", v);
                return 2;
            }
        // 0 is the absent default; an explicit factor must extend, not shrink
        if (o.rope_scale != 0 && o.rope_scale < 1.0) {
            std::fprintf(stderr, "strata generate: --rope-scale %g must be >= 1 (it extends the context, not shrinks it)\n",
                         o.rope_scale);
            return 2;
        }
        if (o.rope_freq_base != 0 && o.rope_freq_base <= 1.0) {
            std::fprintf(stderr, "strata generate: --rope-freq-base must be a base above 1 (0 = the model's)\n");
            return 2;
        }
        if (o.rope_freq_scale < 0 || o.yarn_orig_ctx < 0 || o.yarn_ext_factor < -1.0 || o.yarn_attn_factor <= 0 ||
            o.yarn_beta_fast <= 0 || o.yarn_beta_slow <= 0) {
            std::fprintf(stderr, "strata generate: invalid rope scaling knob (see usage: --yarn-ext-factor <0 = auto, "
                                 "--yarn-orig-ctx 0 = default, the rest positive)\n");
            return 2;
        }
    }
    if (!std::isfinite(o.temperature) || o.temperature < 0 || !std::isfinite(o.top_p) ||
        o.top_p <= 0 || o.top_p > 1 || o.top_k < 0 || o.expert_cache < -1 ||
        o.pool_workers < 0 || std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                           [](int slots) { return slots < 0; }) ||
        (o.expert_cache_remote[1] > 0 && o.expert_cache_remote[0] == 0) ||
        (o.expert_cache_remote[2] > 0 && o.expert_cache_remote[1] == 0)) {
        std::fprintf(stderr, "strata generate: invalid sampling or resource parameter\n");
        return 2;
    }
    if (o.expert_cache_remote_placement != "stripe" && o.expert_cache_remote_placement != "layer") {
        std::fprintf(stderr, "strata generate: --expert-cache-remote-placement must be stripe or layer\n");
        return 2;
    }
    if (o.remote_expert_opt && !o.serve) {
        std::fprintf(stderr, "strata generate: --remote-expert-opt requires --serve\n");
        return 2;
    }
    // the peer tier is the second card's only user: a layer split or a remote expert cache would put a second engine
    // part (and a second copy of the same experts) on it
    if (o.peer_device >= 1 && (!o.layer_split.empty() || o.expert_cache_remote[0] > 0)) {
        std::fprintf(stderr, "strata generate: --peer-device cannot be combined with %s\n",
                     !o.layer_split.empty() ? "--layer-split (use one or the other)"
                                            : "--expert-cache-device1..3 (the peer tier already caches experts there)");
        return 2;
    }

    if (o.native_flash_attn_short && o.max_context > 256) {
        std::fprintf(stderr, "strata generate: --native-flash-attn-short requires --max-context <=256\n");
        return 2;
    }
    if (o.native_flash_attn_short && (o.gpu_only_full || o.graph_only || o.gpu_stages)) {
        std::fprintf(stderr, "strata generate: --native-flash-attn-short requires the normal decode loop for status validation\n");
        return 2;
    }
    // The whole-model graph measurements replay every layer through CUDA0's session, which a layer split carves to
    // CUDA0's own range - the answer would read another stage's state.  Refused here rather than at the call, so
    // the reason is visible before 55 GB is loaded.
    if (multi_gpu && (o.gpu_only_full || o.gpu_stages)) {
        std::fprintf(stderr, "strata generate: --gpu-only-full and --gpu-stages replay the whole model through one "
                             "session, which a layer split does not have; run them without --layer-split\n");
        return 2;
    }
    if (!o.native_head_gguf.empty()) {
        try {
            o.native_head_shards = o.native_head_gguf == o.native_preset ? o.native_shards
                                                                         : strata::gguf_split_paths(o.native_head_gguf);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata generate: --native-head-gguf %s: %s\n", o.native_head_gguf.c_str(), e.what());
            return 2;
        }
    }
    if (o.native_ple_key && (o.native_dense_gguf.empty() || o.no_ple)) {
        std::fprintf(stderr, "strata generate: --native-ple-key requires PLE and --native-dense-gguf\n");
        return 2;
    }
    if (o.cpu_oracle_q8_0 && (o.expert_cache != 0 || !o.expert_profile.empty())) {
        std::fprintf(stderr, "strata generate: --cpu-oracle-q8-0 cannot be combined with --expert-cache or --expert-profile until the GPU expert contract matches\n");
        return 2;
    }

    // **BEFORE ANYTHING ELSE.**  The CPU expert kernel is AVX-512 (VNNI + VBMI) and its translation unit is
    // compiled `/arch:AVX512`, so on a CPU without those features it does not fail - it executes an illegal
    // instruction at some unpredictable token.  Refusing at second zero is the whole point of P2.S3's check.
    strata::kernels::cpu::expert_set_oracle_q8_0(o.cpu_oracle_q8_0);
    // ... and nothing runs on a CPU without AVX2: every CPU expert kernel is AVX2 at least (the AVX-512 ones are
    // chosen above it), and so is ggml-cpu in the release build, which the native pack's layout load initializes
    // next.  Refused here, by name, rather than an illegal instruction in the first expert.
    // The experimental older-CPU build (STRATA_ISA_FLOOR=avx|none, compiled on that PC; #394 #595 #623) has ggml-cpu
    // for that floor, so a native pack's experts run there on ggml-cpu (every AVX2 kernel is behind cpu_avx2_ok).
    const char* isa_floor = strata::kernels::cpu::isa_floor_build();
    if (!strata::kernels::cpu::cpu_avx2_ok() && isa_floor[0] == '\0') {
        std::fprintf(stderr, "strata generate: this CPU (%s) does not support AVX2 with FMA and F16C, which this engine's "
                             "CPU expert kernels need; Strata runs on Intel Haswell (2013), AMD Zen (2017) or newer. "
                             "Older CPUs are EXPERIMENTAL and slow: setup compiles an engine for them on this PC "
                             "(STRATA_ISA_FLOOR, see \"Older CPUs\" in docs/INSTALL.md)\n",
                     strata::kernels::cpu::cpu_name().c_str());
        return 2;
    }
    if (isa_floor[0] != '\0') {
        const bool avx_floor = std::strcmp(isa_floor, "avx") == 0;
        if (avx_floor ? !strata::kernels::cpu::cpu_avx1_ok() : !strata::kernels::cpu::cpu_sse42_ok()) {
            std::fprintf(stderr, "strata generate: this engine is the older-CPU build for %s, which this CPU (%s) does "
                                 "not have%s\n", avx_floor ? "AVX" : "SSE4.2 with POPCNT",
                         strata::kernels::cpu::cpu_name().c_str(),
                         avx_floor ? "; compile it with STRATA_ISA_FLOOR=none" : "");
            return 2;
        }
        if (!strata::kernels::cpu::cpu_avx2_ok())
            std::fprintf(stderr, "strata generate: EXPERIMENTAL older-CPU build (ggml-cpu for %s): this CPU has no AVX2, "
                                 "so the CPU's experts run on ggml-cpu's kernels - expect it to be slow\n", isa_floor);
        else
            std::fprintf(stderr, "strata generate: this is the older-CPU build (ggml-cpu for %s); this CPU has AVX2, "
                                 "and the normal build is faster on it\n", isa_floor);
    }

    std::string err;
    if (!o.native_head_gguf.empty() && !o.stream_token) {
        std::fprintf(stderr, "--native-head-gguf requires --stream-token\n");
        return 2;
    }
    // Plan v0.3 P6: where the experts live.  A native pack (tools/iq_pack.py: the IQ2_XS / IQ3_XXS files) keeps
    // every quantized tensor in its GGUF form, so it needs --native (the dense projections, head and embedding
    // come from the model file) and runs its experts in verify windows only (--spec).
    {
        const strata::core::ModelGeometry g0;
        if (!strata::kernels::cpu::expert_layout_load(o.pack, g0.n_layers, g0.n_expert, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        // Every layer's formats must have GPU expert kernels and a prompt-path dequantizer, checked here, before
        // anything is allocated: an unsupported down type used to exit from inside the first verify window, and
        // an unsupported dequant type left the prompt path's fp16 buffer unwritten.
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (int64_t l = 0; lay.native && l < (int64_t) lay.fmt.size(); ++l) {
            const auto& f = lay.fmt[(size_t) l];
            if (!strata::kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff)) {
                std::fprintf(stderr, "strata generate: layer %lld's experts are %s/%s (ggml types %d/%d), which this "
                                     "engine has no GPU kernels for\n", (long long) l,
                             strata::ggml_type_name((uint32_t) f.gu_type), strata::ggml_type_name((uint32_t) f.d_type),
                             f.gu_type, f.d_type);
                return 1;
            }
        }
        // APR (prism.hadamard) packs (Gyro): every expert path rotates its activations (CPU pool: native_quant_act / _h;
        // GPU verify and the prompt's FP16 chain: hadamard_rows).  Validated on the Arc A750 (SYCL, JIT build).
        if (strata::kernels::cpu::hadamard_spec().any())
            std::fprintf(stderr, "strata generate: APR pack: Hadamard-folded experts (block %d)\n",
                         strata::kernels::cpu::hadamard_spec().block);
    }
    const bool native_pack = strata::kernels::cpu::expert_layout().native;
    // STRATA_EARLY_REMOTE_CONTEXTS=1: create EVERY secondary context here, like CUDA1's.  Under WSL2 the driver's
    // pinned/mapped host budget (dxg gpadl, ~1 GiB) is spent by CUDA0's weights and MTP before the later loop runs,
    // and a new context then fails with cudaErrorMemoryAllocation (CUDA2: "cudaSetDevice(2) failed: out of memory").
    const char* early_env = std::getenv("STRATA_EARLY_REMOTE_CONTEXTS");
    const bool early_remote = early_env && early_env[0] == '1';
    for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0 && (r == 0 || early_remote)) {
        // Keep CUDA1's proven startup order: initialise its context before
        // allocating GPU0 weights or mapping the large host expert arena.
        double free_gib = 0;
        if (!strata::core::RemoteExperts::preflight(remote_dev[r], free_gib, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: CUDA%d context ready, %.2f GiB free before expert arena registration\n",
                     remote_dev[r], free_gib);
    }
    // plan v0.3 P6: the PCIe share of the missed experts, measured per kind of pack (the paper, finding on PCIe).
    // PR #44: a x8 link carries half of what the native default assumes - the GPU's SMs read that share over the
    // link (the copy kernel, since 0.1.14), so on a slower link it must shrink or the window waits for it.  The
    // real H2D bandwidth is probed once; from 20 GB/s up (x16 PCIe 4/5) the measured default stays.  The canonical
    // pack's 0.2 was never measured against the link, so it is left alone.  `--calibrate` measures it outright.
    const bool pcie_explicit = o.pcie_frac >= 0.0;
    if (o.pcie_frac < 0.0) {
        const double base = native_pack ? 0.55 : 0.2;
        std::string bursts;
        const double bw = native_pack ? probe_pcie_h2d_gbps(&bursts, multi_gpu) : -1.0;
        if (!native_pack) {
            o.pcie_frac = base;
        } else if (bw > 0.0) {
            // Gyro's trellis experts: the CPU pool decodes them at ~8 GB/s (48 cycles a block on AVX2, compute bound, a third of the
            // i-quants' rate), so the link carries a bigger share of the misses than the 20 GB/s reference says.  Measured on the
            // Arc A750 (10.5 GB/s link): 0.29 -> 12.8 tok/s, 0.45 -> 15.5, 0.60 -> 15.2 (5 interleaved pairs each).  The reference
            // for these packs is 12 GB/s: 10.5 GB/s gives 0.48; from 12 GB/s up the measured 0.55 stays.
            bool trellis = false;
            {
                const auto& lay = strata::kernels::cpu::expert_layout();
                for (size_t l = 0; lay.native && l < lay.fmt.size(); ++l) trellis = trellis || lay.fmt[l].gu_type >= 144;
            }
            o.pcie_frac = pcie_frac_for_gbps(bw, base, trellis ? 12.0 : 20.0);
            std::fprintf(stderr, "strata generate: PCIe probe: %.1f GB/s host->device (best of %s) -> pcie_frac %.2f "
                                 "(default %.2f)\n", bw, bursts.c_str(), o.pcie_frac, base);
            {   // PcieAuto on SYCL: opt-in (STRATA_PCIE_FRAC_DEFAULT=auto), one card, not the xe NO_HOST path
                const char* pfd = std::getenv("STRATA_PCIE_FRAC_DEFAULT");
                o.pcie_frac_auto = pfd != nullptr && std::strcmp(pfd, "auto") == 0 && o.layer_split.empty();
                o.pcie_probe_gbps = bw;
            }
        } else {
            o.pcie_frac = base;
            std::fprintf(stderr, "strata generate: PCIe probe failed -> pcie_frac default %.2f\n", base);
        }
    }
    // SYCL port, Arc on the xe driver (Arc Pro B60/B65/B70, B580): the GPU page-faults when a kernel reads ordinary
    // (pageable) host memory, and the card times the job out (seen on a B70: Timedout job, Fault response -EINVAL,
    // faulted address in the CPU's mmap range). Without --stream-experts the experts outside the VRAM cache stay in a
    // pageable arena that the PCIe share (--pcie-frac) hands to the GPU. Recommend, never force: the default share
    // becomes 0 (the CPU computes those misses) and an explicit --pcie-frac is kept with a warning.
    if (strata::intel_gpu_driver() == "xe" && !o.stream_experts) {
        if (!pcie_explicit) {
            o.pcie_frac = 0.0;
            std::fprintf(stderr, "strata generate: xe driver without --stream-experts: the GPU faults on pageable host "
                                 "memory, so --pcie-frac defaults to 0 here (the CPU computes the experts outside VRAM). "
                                 "Use --stream-experts, as setup configures it, to let the GPU read a pinned mirror\n");
        } else if (o.pcie_frac > 0.0) {
            std::fprintf(stderr, "strata generate: WARNING: --pcie-frac %.2f on the xe driver without --stream-experts "
                                 "makes the GPU read pageable host memory; that page-faulted and hung an Arc Pro B70 "
                                 "(Timedout job, device lost). Use --stream-experts, as setup configures it, or "
                                 "--pcie-frac 0\n", o.pcie_frac);
        }
    }
    // the canonical Q2_0 pack's CPU kernels are AVX-512 only; a native pack runs on AVX2 CPUs as well
    if (!native_pack) strata::kernels::cpu::cpu_require_expert_support();
    else if (!strata::kernels::cpu::cpu_avx512_ok())
        std::fprintf(stderr, "strata generate: this CPU has no AVX-512: the expert kernels run on %s "
                             "(multi-token for the i-quant gate/up rows)\n",
                     !strata::kernels::cpu::cpu_avx2_ok() ? "ggml-cpu vec_dot (no AVX2: the older-CPU build)"
                     : std::getenv("STRATA_NO_IQ256") == nullptr ? "AVX-2" : "ggml-cpu vec_dot (STRATA_NO_IQ256 set)");
    // the compiled shape; a native model's file replaces it below with its own metadata (read_geometry), which a
    // pruned variant (GSQ-RCO Coder, fewer than 512 experts) differs from
    strata::core::ModelGeometry g;
    int64_t K = g.n_expert_used;
    // THE ROPE CONFIG RESOLVES HERE, BEFORE ANY WEIGHT MOVES - the CLI and the model file have both spoken,
    // and `session_init` below builds the rope table from it and captures the kernels reading its constants
    // (rope_scaling.hpp); the only hard constraint is "set before that", and dying on a bad rope key beats
    // scanning gigabytes of shards first.  Precedence: an EXPLICIT flag over the model file's rope keys over
    // the struct defaults.  The empty --rope-scaling and the 0 --rope-scale mean the flag is absent, so the
    // model file decides; an explicit value - `none` and `1` included, the opt-outs - wins over the model file.
    {
        // The model file's rope keys (llama.cpp's names under the arch prefix), when it carries any - the
        // artifact today ships none, so this is a no-op defaults channel for future fine-tunes.
        std::string gguf_rope_type;
        double gguf_rope_base = 0, gguf_rope_factor = 0, gguf_rope_orig_ctx = 0;
        if (!o.native_preset.empty()) {
            // The geometry is the model file's (core::geometry_from_gguf: every "<arch>.*" shape key), checked
            // against the shapes this engine's kernels are built for.  The expert count and top-k are run-time
            // sizes: a pruned variant (GSQ-RCO Coder) ships fewer experts than the canonical 512x10.
            try {
                strata::GgufFile model_gguf(o.native_shards.front());   // the metadata shard
                std::string gerr;
                if (!strata::core::geometry_from_gguf(model_gguf, g, K, gerr)) {
                    std::fprintf(stderr, "strata generate: %s\n", gerr.c_str());
                    return 1;
                }
                const std::string mismatch = strata::core::compiled_geometry_mismatch(g);
                if (!mismatch.empty()) {
                    std::fprintf(stderr, "strata generate: %s: %s\n", o.native_preset.c_str(), mismatch.c_str());
                    return 1;
                }
                const std::string p = std::string(strata::model_family_arch(g.family)) + ".";
                if (const strata::MetaValue* v = model_gguf.get(p + "rope.freq_base")) gguf_rope_base = v->num();
                if (const strata::MetaValue* v = model_gguf.get(p + "rope.scaling.type")) gguf_rope_type = v->s;
                if (const strata::MetaValue* v = model_gguf.get(p + "rope.scaling.factor")) gguf_rope_factor = v->num();
                if (const strata::MetaValue* v = model_gguf.get(p + "rope.scaling.original_context_length"))
                    gguf_rope_orig_ctx = v->num();
            } catch (const std::exception& e) {
                std::fprintf(stderr, "strata generate: reading the model's geometry from %s: %s\n",
                             o.native_preset.c_str(), e.what());
                return 1;
            }
        }
        using RST = strata::kernels::RopeScalingType;
        if (!o.rope_scaling.empty()) {
            // the early validation pinned the spelling; `none` here is the CLI opting OUT of the model file's keys
            if (o.rope_scaling == "linear") rope_cfg.type = RST::Linear;
            else if (o.rope_scaling == "yarn") rope_cfg.type = RST::YaRN;
            else rope_cfg.type = RST::None;
        } else if (!gguf_rope_type.empty()) {
            if (gguf_rope_type == "linear") rope_cfg.type = RST::Linear;
            else if (gguf_rope_type == "yarn") rope_cfg.type = RST::YaRN;
            else if (gguf_rope_type != "none") {
                std::fprintf(stderr, "strata generate: %s carries rope.scaling.type '%s' - none, linear or yarn only\n",
                             o.native_preset.c_str(), gguf_rope_type.c_str());
                return 2;
            }
        }
        if (o.rope_scale > 0) rope_cfg.factor = o.rope_scale;            // an explicit factor, 1 included
        else if (gguf_rope_factor > 1.0) rope_cfg.factor = gguf_rope_factor;
        if (o.rope_freq_base > 0) rope_cfg.freq_base = o.rope_freq_base;
        else if (gguf_rope_base > 1.0) rope_cfg.freq_base = gguf_rope_base;
        if (o.yarn_orig_ctx > 0) rope_cfg.orig_ctx = o.yarn_orig_ctx;
        else if (gguf_rope_orig_ctx >= 1) rope_cfg.orig_ctx = gguf_rope_orig_ctx;
        rope_cfg.freq_scale_in = o.rope_freq_scale;
        rope_cfg.ext_factor = o.yarn_ext_factor >= 0 ? o.yarn_ext_factor
                                                     : (rope_cfg.type == RST::YaRN ? 1.0 : 0.0);
        rope_cfg.attn_factor = o.yarn_attn_factor;
        rope_cfg.beta_fast = o.yarn_beta_fast;
        rope_cfg.beta_slow = o.yarn_beta_slow;
        if (rope_cfg.type == RST::None) {
            // none is the trained rotation, exactly: the scaling knobs are inert (the table builder and
            // `kernel_args` ignore them), and resetting them keeps the logged/queried config honest.  Only the
            // frequency base survives - it is the rotation itself, not a scaling knob.
            const bool knobs = o.rope_scale > 1.0 || o.rope_freq_scale > 0 || o.yarn_ext_factor > 0 ||
                               o.yarn_attn_factor != 1.0;
            const double base = rope_cfg.freq_base;
            rope_cfg = strata::kernels::RopeScaling{};
            rope_cfg.freq_base = base;
            if (knobs)
                std::fprintf(stderr, "strata generate: note: no rope scaling is active (none), so --rope-scale, "
                                     "--rope-freq-scale and the --yarn-* knobs have no effect\n");
        }
        // THE RESOLVED CONFIG IS VALIDATED AS A WHOLE, with the one rule every rotation site also applies
        // (rope_scaling.hpp): the CLI ranges above cannot see a model-file value, nor a combination such as a
        // --rope-freq-scale that turns the resolved factor non-finite.
        if (const char* why = strata::kernels::rope_scaling_invalid(rope_cfg)) {
            std::fprintf(stderr, "strata generate: invalid rope scaling configuration: %s (type %s, factor %g, "
                                 "freq_scale %g, base %g, original context %g)\n",
                         why, rope_cfg.type == RST::YaRN ? "yarn" : rope_cfg.type == RST::Linear ? "linear" : "none",
                         rope_cfg.factor, rope_cfg.freq_scale(), rope_cfg.freq_base, rope_cfg.orig_ctx);
            return 2;
        }
        strata::kernels::rope_scaling_set(rope_cfg);
        if (rope_cfg.type != RST::None) {
            const char* tn = rope_cfg.type == RST::YaRN ? "yarn" : "linear";
            std::fprintf(stderr,
                         "strata generate: rope scaling %s, factor %.6g (freq_scale %.6g, base %.6g, mscale %.6f), "
                         "--max-context %lld against a trained context of %.0f\n",
                         tn, rope_cfg.factor, rope_cfg.freq_scale(), rope_cfg.freq_base, rope_cfg.mscale(),
                         (long long) o.max_context, rope_cfg.orig_ctx);
            if ((double) o.max_context <= rope_cfg.orig_ctx)
                std::fprintf(stderr,
                             "strata generate: note: the context is within the trained %.0f - no position needs the "
                             "extension, and the resolved scaling still applies to every angle\n",
                             rope_cfg.orig_ctx);
            // only when there IS a magnitude correction: YaRN's log term (ext_factor != 0) or an explicit
            // --yarn-attn-factor; plain linear (mscale 1) has none, and saying otherwise was TODO 22
            if (rope_cfg.mscale() != 1.0)
                std::fprintf(stderr,
                             "strata generate: note: the %s magnitude correction scales cos and sin by %.6f "
                             "at every position\n",
                             rope_cfg.type == RST::YaRN ? "YaRN" : "--yarn-attn-factor", rope_cfg.mscale());
        }
    }
    strata::core::NativeEmbed native_embed;
    if (native_pack) {
        if (o.native_preset.empty() || o.spec < 2 || o.keep_canonical ||
            (o.prefill_chunk <= 0 && o.tokens.size() > 1)) {
            std::fprintf(stderr, "strata generate: %s is a native (IQ) pack: it needs --native SHARD1, --spec T (T >= 2) "
                                 "and --prefill CHUNK\n", o.pack.c_str());
            return 2;
        }
        const auto embed_t0 = std::chrono::steady_clock::now();
        if (!native_embed.load(o.embd_gguf.empty() ? o.native_shards : std::vector<std::string>{o.embd_gguf}, g.n_embd,
                               g.n_vocab, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        strata::core::set_native_embed(&native_embed);
        std::fprintf(stderr, "strata generate: native pack: %s experts (largest blob %.2f MB), token embedding "
                             "%s in mapped host memory (%.0f MiB, %.1f s)\n",
                     o.pack.c_str(), (double) strata::kernels::cpu::expert_layout().max_blob / 1e6,
                     strata::ggml_type_name((uint32_t) native_embed.type()), (double) native_embed.bytes() / 1048576.0,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - embed_t0).count());
    }
    // Plan v0.3 P1: tensors served in native form are not also loaded in canonical form (~2.7 GB of VRAM back
    // to the expert cache with --native).  `--keep-canonical` loads both, as before.
    std::set<std::string> skip;
    if (!o.keep_canonical) {
        if (!o.native_dense_gguf.empty() &&
            !strata::core::NativeDense::served_names(o.native_dense_gguf, o.native_ple_key, skip, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (!o.native_head_gguf.empty()) skip.insert("output.weight");
        // the PLE module validates its canonical key at construction (8 MB); a native pack has none to load
        if (!native_pack) skip.erase("blk.1.ple_key.weight");
        // #326: a --compat-bf16 pack (OrcaRouter IQ3_XXS) keeps its BF16 key in the arena
        if (native_pack && !strata::core::NativeDense::keep_unquantized_ple_key(o.pack, skip, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (native_pack) skip.insert("token_embd.weight");
    }
    // Layer split: every GPU holds only the dense weights of ITS layers (the PLE tensors stay everywhere).
    // Without this each card keeps a full copy (~3.4 GB for the Coder) that its stage never reads - VRAM the
    // expert cache wants.  Opt-in, STRATA_STAGE_TRIM=1, on HIP and CUDA alike: measured on 2x MI50 (PR #639),
    // and the default waits for its author's re-run of the release branch on his cards.
    const std::set<std::string> skip_base = skip;
    // --trim-stage-weights (PR #559) is the same switch as STRATA_STAGE_TRIM=1 (PR #639)
    const bool trim_asked = o.trim_stage_weights || [] {
        const char* v = std::getenv("STRATA_STAGE_TRIM");
        return v != nullptr && v[0] != 0 && std::string(v) != "0";
    }();
    // CUDA0'S ARENA IS BUILT BEFORE THE SPLIT IS KNOWN.  The prompt path's PLE tables are read out of it, here,
    // and `auto` does not choose its split points until 590 lines below - so CUDA0 can still only trim an
    // explicit --layer-split.  It is also the cheapest card to leave alone: it runs layers 0..split_at[0] on
    // the device with the most room, and on the 4-way IQ3_S rig the trim buys it 4598 -> 4608 slots.  The VRAM
    // is on the later cards (CUDA2 1241 -> 2485, CUDA3 533 -> 1772), and THEIR weights load after the search,
    // so they trim under `auto` too - see the block past it.
    const bool stage_trim = trim_asked && multi_gpu && !split_auto && !split_at.empty();
    if (o.trim_stage_weights && !multi_gpu)
        std::fprintf(stderr, "strata generate: --trim-stage-weights needs a layer split across GPUs (ignored)\n");
    // Only `--layer-split auto` WITH the trim asked loads the later stages' weights after the split search (they can
    // only be trimmed once the split is known).  Every other run keeps the 0.1.39 order: weights first, then the
    // search - so without the trim nothing about a split moves.
    const bool late_weights = trim_asked && multi_gpu && split_auto;
    // keep_routers: CUDA0 with --mmap-experts keeps every layer's router (ffn_gate_inp, ~2.5 MiB a layer) - the file
    // tier's routing-aware prefetch (RouterLookahead, below) copies all of them from CUDA0's arena, and would
    // otherwise turn itself off
    auto add_foreign = [&](int64_t lb, int64_t le, std::set<std::string>& out, bool keep_routers) {
        std::FILE* f = std::fopen((o.pack + "/index.txt").c_str(), "rb");
        if (!f) return;
        char line[1024], name[256];
        while (std::fgets(line, sizeof line, f)) {
            if (line[0] == '#' || std::sscanf(line, "%255s", name) != 1) continue;
            const std::string n = name;
            if (n.rfind("blk.", 0) != 0 || n.find("ple") != std::string::npos) continue;
            if (keep_routers && n.ends_with(".ffn_gate_inp.weight")) continue;
            const int64_t l = std::atoll(name + 4);
            if (l < lb || l >= le) out.insert(n);
        }
        std::fclose(f);
    };
    if (stage_trim) {
        add_foreign(0, split_at[0], skip, o.mmap_experts);
        strata::core::NativeDense::set_layer_range(0, (int) split_at[0]);
        std::fprintf(stderr, "strata generate: layer split: CUDA0 loads the dense weights of layers 0-%lld only\n",
                     (long long) split_at[0] - 1);
    }
    uint64_t pool_bytes = 0;
    if (!strata::core::WeightTable::pool_bytes(o.pack, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    void* arena = nullptr;
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (const dpct::err0 ce =
            DPCT_CHECK_ERROR(arena = (void *)strata::malloc_device_guarded(
                                 pool_bytes, dpct::get_in_order_queue()));
        ce != 0) {
        // #486: the arena is the first large allocation and its size does not depend on the context, so what is
        // missing is held by something else: say how much was free
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        size_t free_b = 0, total_b = 0;
        /*
        DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions
        for device information which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        dpct::get_current_device().get_memory_info(free_b, total_b);
        std::fprintf(
            stderr,
            "strata generate: cudaMalloc(%llu) for the weight arena failed "
            "(%s): %llu MiB of %llu "
            "MiB VRAM free on this GPU. The arena is allocated first, before "
            "the KV and expert "
            "caches: another program (or an engine that is still exiting) "
            "holds the rest - "
            "nvidia-smi / rocm-smi lists them\n",
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            (unsigned long long)pool_bytes, dpct::get_error_string_dummy(ce),
            (unsigned long long)(free_b >> 20),
            (unsigned long long)(total_b >> 20));
        return 1;
    }
    auto load_t0 = std::chrono::steady_clock::now();
    auto load_s = [&] {
        const auto t = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t - load_t0).count();
        load_t0 = t;
        return s;
    };
    strata::core::WeightTable wt;
    if (!wt.load(o.pack, arena, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "strata generate: %llu MiB of weights loaded from %s in %.1f s (%zu canonical tensors "
                         "skipped: served natively)\n",
                 (unsigned long long) (pool_bytes >> 20), o.pack.c_str(), load_s(), skip.size());

    strata::core::NativeDense native_dense;
    if (!o.native_dense_gguf.empty()) {
        if (!native_dense.load(o.native_dense_gguf, wt, err, o.native_ple_key)) {
            std::fprintf(stderr, "strata generate: native dense projections: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: %zu native projection matrices, %.2f MiB of weights, in %.1f s\n",
                     native_dense.tensor_count(), (double) native_dense.weight_bytes() / (1024.0 * 1024.0), load_s());
    }

    strata::kernels::gr_set_fp32_activations(o.gr_fp32_activations);
    strata::kernels::gr_set_native_mmvf(o.gr_native_mmvf);
    // Plan v0.3 P3: the fused hyper-connection read rides the native (FP32-activation) contract; the per-stage
    // and dump measurements need the unfused layout of R, so they keep the old kernels.
    strata::core::layer_set_fast_attn(!o.no_fast_attn);
    strata::core::layer_set_publish_kernel(!o.no_publish_kernel);
    strata::core::layer_set_fused_gdn(!o.no_fused_gdn);
    strata::core::layer_set_fast_select(!o.no_fast_select);
    strata::core::layer_set_fused_gr(o.gr_native_mmvf && !o.no_fused_gr && !o.gpu_stages && o.dump_layers.empty() &&
                                     o.dump_halves.empty() && !o.stage_timing);
    strata::core::layer_set_native_bf16(o.native_bf16);
    strata::core::layer_set_native_flash_attn_short(o.native_flash_attn_short);
    strata::kernels::ple_set_native_bf16(o.native_bf16_extra);
    strata::kernels::shared_expert_set_native_bf16(o.native_bf16_extra);
    strata::kernels::native_moe_combine_set_enabled(o.native_moe_combine);
    strata::kernels::native_gdn_set_enabled(o.native_gdn);
    strata::kernels::native_router_set_enabled(o.native_router);
    strata::kernels::native_qsa_set_enabled(o.native_qsa);
    strata::kernels::native_qsa_indexer_set_enabled(o.native_qsa_indexer);
    strata::kernels::native_rope_set_enabled(o.native_rope);
    // The vision path: every rope kernel reads a cell's (t, h, w) from this table (strata/kernels/mrope.hpp).  It is
    // the identity until an image request, and it is set here, before any CUDA graph captures a rope kernel.
    int32_t* d_mrope = nullptr;
    std::vector<int32_t> mrope_host;
    if (o.vision) {
        const int64_t cells = o.max_context + 64;
        mrope_host.resize((size_t) cells * 3);
        for (int64_t c = 0; c < cells; ++c)
            mrope_host[(size_t) c * 3] = mrope_host[(size_t) c * 3 + 1] = mrope_host[(size_t) c * 3 + 2] = (int32_t) c;
        if (DPCT_CHECK_ERROR(
                d_mrope = sycl::malloc_device<int32_t>(
                    mrope_host.size(), dpct::get_in_order_queue())) != 0 ||
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                d_mrope, mrope_host.data(),
                mrope_host.size() * sizeof(int32_t)).wait()) != 0) {
            std::fprintf(stderr, "strata generate: cannot allocate the image position table\n");
            return 1;
        }
        strata::kernels::mrope_table_set(d_mrope);
    }
    strata::kernels::ple_set_native_postops(o.native_ple_postops);
    // before session_init: every graph captured from here on has the vector's kernels where it applies
    std::string cvec_summary = "0";
    uint64_t cvec_digest = 0;   // a session file is bound to the loaded vector (0: none)
    if (!o.cvec_files.empty()) {
        std::string ce;
        if (!load_control_vectors(o, g, cvec_summary, cvec_digest, ce)) {
            std::fprintf(stderr, "strata generate: control vector: %s\n", ce.c_str());
            return 2;
        }
    }
    if (o.max_context < (int64_t) o.tokens.size() + o.max_new) {
        std::fprintf(stderr, "strata generate: --max-context %lld cannot hold %zu prompt + %lld new tokens\n",
                     (long long) o.max_context, o.tokens.size(), (long long) o.max_new);
        return 2;
    }

    strata::core::SessionState ss;
    void* sbuf = nullptr;   // allocated after the layer-split search, sized to CUDA0's own layer range (the carve)
    // **THE ENGINE RAN ON THE LEGACY DEFAULT STREAM, WHICH ON WDDM IS THE SLOW PATH.**  All four session
    // calls - `session_capture`, `session_replay`, `session_token` and `session_loop` - were handed `nullptr`,
    // i.e. stream 0.  `bench/micro/kernel_costs.cu` measures what that costs: EVERY kernel it launches through
    // a wrapper comes back at 28-31 us REGARDLESS OF SIZE, `scale_inplace` on 2,048 floats and `silu_inplace`
    // on 10,240 floats being indistinguishable, which is a fixed per-launch cost and not execution.
    // `bench/micro/graph_node_cost.cu` measures the same kernels on a real stream at 3.63 us ungrapped and
    // 0.805 us inside a graph.  **That is an ~8x penalty on every launch in the engine.**
    dpct::queue_ptr main_stream = &dpct::get_in_order_queue();
    /*
    DPCT1025: The SYCL queue is created ignoring the flag and priority
    options.
    */
    if (DPCT_CHECK_ERROR(
            main_stream = dpct::get_current_device().create_queue(true)) != 0) {
        std::fprintf(stderr, "strata generate: cannot create the main stream\n");
        return 1;
    }
    void* const main_cs = (void*) main_stream;

    // ---- **THE HALF-LEVEL DUMP HAS TO BE ARMED BEFORE `session_capture`, AND THE LADDER MUST NOT BE.**  The
    // half copies are issued from inside `block_layer_pre`/`block_layer_post`, so they are only ever enqueued
    // while a graph is being CAPTURED - arming `ss.block.dump` afterwards would produce a file of zeros that
    // reads exactly like a wrong answer.  The ladder is the opposite: `session_loop` enqueues it per token on
    // the replay stream, so it must be armed after capture to stay out of the graph.
    const uint64_t half_stride = (uint64_t) 2 * g.n_embd + (uint64_t) 2 * g.hc +
                                 (uint64_t) g.n_head * g.head_dim +
                                 (uint64_t) 5 * g.n_head_kv * g.head_dim + 8;
    std::FILE* half_dump = nullptr;
    float* half_stage = nullptr;
    if (!o.dump_halves.empty()) {
        half_dump = std::fopen(o.dump_halves.c_str(), "wb");
        if (half_dump == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_halves.c_str());
            return 1;
        }
        const size_t n = (size_t) g.n_layers * (size_t) half_stride;
        /*
        DPCT1048: The original value cudaHostAllocDefault is not meaningful
        in the migrated code and was removed or replaced with 0. You may need to
        check the migrated code.
        */
        if (DPCT_CHECK_ERROR(half_stage = sycl::malloc_host<float>(
                                 n, dpct::get_in_order_queue())) != 0) {
            std::fprintf(stderr, "strata generate: cannot pin the half-dump staging buffer\n");
            return 1;
        }
        ss.block.dump = half_stage;
    }

    strata::core::Doorbell db;
    if (strata::core::doorbell_init(g, K, db) == 0) {
        std::fprintf(stderr, "strata generate: doorbell_init failed\n");
        return 1;
    }
    ss.db = &db;

    // ================================ THE PLE ================================
    //
    // **ITS ABSENCE IS WHY GATE C1 FAILED** (LEDGER L123): layer 1 carries six `blk.1.ple_*` tensors, the whole
    // module was built and parity-tested, and nothing called it.  Everything below is construction - the table
    // is a mapping of the ORIGINAL second GGUF shard, the six weights are already loaded in the arena, and the
    // three buffers are the only allocation.
    strata::kernels::PleTable ple_table;
    std::vector<float> ple_emb_host((size_t) strata::kernels::NG_N_EMBD);
    float* ple_emb_dev = nullptr;
    float* ple_scratch = nullptr;
    if (!o.ple_gguf.empty()) {
        strata::kernels::PleIoOptions pio;
        pio.mode = o.ple_io == "mmap" || o.ple_io == "ram" ? strata::kernels::PleIo::Mmap : strata::kernels::PleIo::Direct;
        pio.lock = o.ple_io == "ram";
        const auto tpl = Clock::now();
        pio.max_inflight = (uint32_t) o.ple_inflight;
        pio.cache_rows = (uint64_t) o.ple_row_cache;
        pio.io_thread = !o.ple_sync_submit;
        // Keep the SSD awake while rows are being asked for (PleReader::set_keepalive): some SSDs stall the first
        // reads 50-150 ms after ~250 ms without a command.  STRATA_SSD_KEEPALIVE = ms without a read before one
        // page is read anyway (default 100, 0 = off), STRATA_SSD_KEEPALIVE_WINDOW = seconds after the last row
        // request that this goes on (default 60; then the SSD may sleep until the next request).
        {
            const char* ka = std::getenv("STRATA_SSD_KEEPALIVE");
            const char* kw = std::getenv("STRATA_SSD_KEEPALIVE_WINDOW");
            pio.keepalive_ms = ka != nullptr && *ka ? std::clamp(std::atof(ka), 0.0, 10000.0) : 100.0;
            pio.keepalive_window_s = kw != nullptr && *kw ? std::clamp(std::atof(kw), 1.0, 86400.0) : 60.0;
            if (pio.mode != strata::kernels::PleIo::Direct || !pio.io_thread) pio.keepalive_ms = 0;
        }
        if (!ple_table.open(o.ple_gguf, err, pio)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        // Recommend, never force: when the default direct reads measure slow, switch to the mapped reads (40-100x
        // faster on the reporters' drives); an explicit --ple-io direct is kept and only warned about.
        if (pio.mode == strata::kernels::PleIo::Direct && std::getenv("STRATA_PLE_PROBE") == nullptr) {
            constexpr double kSlowRowsPerS = 15000.0;   // ~60 MB/s of 4 KiB pages; a healthy NVMe is >100k
            const double rps = ple_direct_rows_per_s(ple_table);
            if (rps >= kSlowRowsPerS)
                std::fprintf(stderr, "strata generate: PLE direct read probe: %.0f rows/s (fine)\n", rps);
            if (rps > 0 && rps < kSlowRowsPerS) {
                // #1984 #1971 #2040: on no platform is the switch made on its own.  Windows: the mapped reads stalled prompts (the first chunk waits for 16 serial page faults
                // a token from a cold table with no progress mark).  Linux SATA SSD: the 15K rows/s bar is NVMe-calibrated, a SATA SSD
                // tops out at 10-14K with direct's 16 reads in flight and mapped reads were 7x slower.  The switch is only made when asked for
                // (STRATA_PLE_AUTO_MMAP=1, the 0.1.42 behaviour); otherwise the slow probe warns and direct is kept.
                bool auto_mmap = !o.ple_io_explicit;
                if (const char* am = std::getenv("STRATA_PLE_AUTO_MMAP"); am == nullptr || std::atoi(am) == 0) auto_mmap = false;
                if (auto_mmap) {
                    std::fprintf(stderr, "strata generate: --ple-io direct measures slow here (%.0f rows/s, about %.0f MB/s of "
                                         "random 4 KiB reads): using --ple-io mmap instead (the OS file cache holds the pages; "
                                         "--ple-io direct forces the old mode, --ple-io ram keeps the table in RAM)\n",
                                 rps, rps * 4096.0 / 1e6);
                    ple_table.close();
                    pio.mode = strata::kernels::PleIo::Mmap;
                    pio.keepalive_ms = 0;
                    if (!ple_table.open(o.ple_gguf, err, pio)) {
                        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                        return 1;
                    }
                } else {
                    std::fprintf(stderr, "strata generate: WARNING: --ple-io direct measures slow here (%.0f rows/s, about %.0f "
                                         "MB/s of random 4 KiB reads): a prompt can spend minutes reading layer 1 and trip the "
                                         "stall watchdog (#1425, #1629). --ple-io mmap was 40-100x faster on some DRAM-less NVMe drives but 7x slower on a SATA SSD (#2040), so it is not chosen for you; "
                                         "--ple-io ram needs the RAM for the table%s\n", rps, rps * 4096.0 / 1e6,
                                 o.ple_io_explicit ? "" : " (kept direct: pass --ple-io mmap or --ple-io ram to change it; in a config file's \"args\" they are two entries: \"--ple-io\", \"mmap\")");
                }
            }
        }
#if !defined(_WIN32)
        if (pio.mode == strata::kernels::PleIo::Direct) {
            // #605: --ple-io direct's random reads (up to --ple-inflight at once) are for SSDs; a rotational disk
            // can take longer than the stall watchdog's 60 s for one prompt chunk's rows.  Recommended, not forced.
            std::string dev;
            if (!under_wddm() && on_rotational_disk(o.ple_gguf, dev) == 1)   // (WSL's virtual disk says rotational)
                std::fprintf(stderr, "strata generate: WARNING: the n-gram table (%s) is on a rotational disk (%s): "
                                     "--ple-io direct reads it at random and can stall a prompt for minutes (#605). "
                                     "--ple-io ram (it then needs RAM for the table) or the model on an SSD is "
                                     "recommended\n", o.ple_gguf.c_str(), dev.c_str());
        }
#endif
        if (pio.lock)
            std::fprintf(stderr, "strata generate: PLE table %s (--ple-io ram) in %.1f s\n",
                         ple_table.locked() ? "locked in RAM" : "loaded (not locked)",
                         std::chrono::duration<double>(Clock::now() - tpl).count());
        if (pio.keepalive_ms > 0)
            std::fprintf(stderr, "strata generate: the SSD is kept awake while rows are read: one page of the table after "
                                 "%.0f ms without a read, until %.0f s after the last request "
                                 "(STRATA_SSD_KEEPALIVE=0 turns it off)\n", pio.keepalive_ms, pio.keepalive_window_s);
        else if (pio.mode == strata::kernels::PleIo::Direct)
            std::fprintf(stderr, "strata generate: SSD keep-alive off: the SSD may fall asleep between reads\n");
        const strata::core::WeightRef* wk = wt.find("blk.1.ple_key.weight");
        const strata::core::WeightRef* wv = wt.find("blk.1.ple_value.weight");
        const strata::core::WeightRef* wnk = wt.find("blk.1.ple_norm_key.weight");
        const strata::core::WeightRef* wnq = wt.find("blk.1.ple_norm_query.weight");
        const strata::core::WeightRef* wnc = wt.find("blk.1.ple_norm_conv.weight");
        const strata::core::WeightRef* wc = wt.find("blk.1.ple_conv1d.weight");
        if (!wk || !wv || !wnk || !wnq || !wnc || !wc) {
            std::fprintf(stderr, "strata generate: the pack has no blk.1.ple_* tensors, so the PLE cannot be "
                                 "wired - and running without it is a DIFFERENT MODEL (LEDGER L123)\n");
            return 1;
        }
        // `ple_key` is S2 and the loader has already widened its scales to f32, so the two planes are located
        // by the sizes the `WeightRef` records rather than re-derived - the same rule `plane_ptrs` follows.
        if (!wk->quantized()) {
            // plan v0.3 P6: the IQ model files' BF16 key (the pack's extra.bin, raw BF16)
            ss.ple.w.key_bf16 = (const uint16_t*) wk->data;
        } else if (wk->data != nullptr) {
            ss.ple.w.key_codes = (const uint8_t*) wk->data;
            ss.ple.w.key_scales = (const float*) ((const uint8_t*) wk->data + wk->codes_bytes);
        }
        if (o.native_ple_key && wk->quantized()) {
            if (!wk->native_data || (wk->native_type != 42 && wk->native_type != 18 && wk->native_type != 23 &&
                                     wk->native_type != 8) || !wk->native_q8_1) {
                std::fprintf(stderr, "strata generate: native PLE key is absent or incompatible\n");
                return 1;
            }
            ss.ple.w.key_native_data = wk->native_data;
            ss.ple.w.key_native_type = wk->native_type;
            ss.ple.w.key_native_q8_1 = wk->native_q8_1;
        }
        ss.ple.w.value_bf16 = (const uint16_t*) wv->data;
        ss.ple.w.norm_key = (const float*) wnk->data;
        ss.ple.w.norm_query = (const float*) wnq->data;
        ss.ple.w.norm_conv = (const float*) wnc->data;
        // The conv1d kernel reads F16, so this cast is a claim about the pack's storage.  A checkpoint that keeps
        // the tensor F32 (Q8_0, UD-Q4_K_XL) would hand the kernel the low halves of the f32 words - not an error,
        // a plausible wrong layer-1 routing.  tools/iq_pack.py narrows it (index kind 3); a pack that did not is
        // refused here (#255, gopinath87607).  F16 is index kind 5 or 3 (F16InF32) in a native pack and kind 0
        // (verbatim 2-byte F16) in the canonical Q2_0 pack; BF16 (kind 4) has the same size and is not F16.
        const bool f16 = wc->kind == strata::core::WeightKind::F16InF32 ||
                         (wc->kind == strata::core::WeightKind::Verbatim && wc->code_bits == 0);
        if (!f16 || wc->bytes != (uint64_t) wc->elements * 2) {
            std::fprintf(stderr, "strata generate: blk.1.ple_conv1d.weight is not stored as F16 (pack index kind %d, "
                                 "%llu B for %lld values); the PLE conv1d kernel reads F16 - repack with "
                                 "tools/iq_pack.py\n",
                         (int) wc->kind, (unsigned long long) wc->bytes, (long long) wc->elements);
            return 1;
        }
        ss.ple.w.conv1d_f16 = (const uint16_t*) wc->data;
        ss.ple.consts = strata::kernels::ple_artifact_consts();
        if (o.ple_delay_us > 0) ple_table.set_injected_delay_us(o.ple_delay_us);
        ss.ple.table = &ple_table;
        ss.ple.token = &ss.ple_token;
        ss.ple.prev = ss.ple_prev;
        // `ss.ple.hist` and the ready() check wait for `session_init`, which carves the history - the session is
        // now allocated after the layer-split search (see the carve), and the wiring lands there
        ss.ple.emb_host = ple_emb_host.data();
        if (DPCT_CHECK_ERROR(ple_emb_dev = (float *)sycl::malloc_device(
                                 (size_t)strata::kernels::NG_N_EMBD * 4,
                                 dpct::get_in_order_queue())) != 0 ||
            DPCT_CHECK_ERROR(ple_scratch = (float *)sycl::malloc_device(
                                 strata::core::ple_run_scratch_bytes(),
                                 dpct::get_in_order_queue())) != 0) {
            std::fprintf(stderr, "strata generate: the PLE buffers failed\n");
            return 1;
        }
        ss.ple.emb_dev = ple_emb_dev;
        ss.ple.scratch = ple_scratch;
    } else {
        std::fprintf(stderr,
                     "strata generate: PLE OFF by explicit --no-ple diagnostic request.\n"
                     "  The tokens below are NOT this model's; this is only useful for A/B measurement.\n");
    }

    float* d_parts = nullptr;
    if (DPCT_CHECK_ERROR(
            d_parts = (float *)sycl::malloc_device(
                (size_t)K * g.n_embd * 4, dpct::get_in_order_queue())) != 0 ||
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memset(d_parts, 0, (size_t)K * g.n_embd * 4)
                             .wait()) != 0) {
        std::fprintf(stderr, "strata generate: the parts buffer failed\n");
        return 1;
    }

    // ---- --split-skip-if-fits: before any later stage loads, does CUDA0 alone hold every profiled pair?  What it
    // still has to allocate on one GPU is the whole session (the KV of every layer), the drafter and the head
    // (kDrafterMib below), the verify windows and the reserve; the prompt path borrows from the cache.  If the
    // profile's pairs fit in what is left, a split would only add the hand-offs: run on CUDA0 alone.
    if (multi_gpu && split_auto && o.split_skip_if_fits) {
        std::vector<std::pair<int32_t, int32_t>> prof;
        int64_t pslots = 0;
        std::string perr;
        const bool remote = o.expert_cache_remote[0] > 0 || o.expert_cache_remote[1] > 0 || o.expert_cache_remote[2] > 0;
        if (remote || o.expert_profile.empty() ||
            !strata::core::read_expert_profile(o.expert_profile, g.n_layers, g.n_expert, prof, pslots, perr)) {
            std::fprintf(stderr, "strata generate: --split-skip-if-fits: %s; the split stays\n",
                         remote ? "remote expert caches are in use" : perr.empty() ? "no expert profile" : perr.c_str());
        } else {
            const auto& lay = strata::kernels::cpu::expert_layout();
            int64_t pairs_bytes = 0;
            for (const auto& pr : prof)
                pairs_bytes += native_pack ? ((int64_t) lay.blob_bytes(pr.first) + 255) / 256 * 256 : (int64_t) lay.max_blob;
            size_t fb = 0, tb = 0;
            /*
            DPCT1106: 'cudaMemGetInfo' was migrated with the Intel
            extensions for device information which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            dpct::get_current_device().get_memory_info(fb, tb);
            const int64_t session = (int64_t) strata::core::session_bytes(g, o.max_context, K, 0, g.n_layers);
            const int64_t held_back = session + (((int64_t) o.vram_reserve_mib + 1000 + 96) << 20);   // + drafter/head, windows
            const int64_t room = (int64_t) fb - held_back;
            dpct::device_info dp{};
            dpct::get_device(0).get_device_info(dp);
            if (pairs_bytes <= room) {
                std::fprintf(stderr,
                             "strata generate: layer split skipped "
                             "(--split-skip-if-fits): CUDA0 (%s) holds all "
                             "%zu profiled pairs (%.2f GiB) with the session "
                             "(%.2f GiB, %lld-token context), "
                             "the drafter and the reserve: %.2f GiB free, %.2f "
                             "GiB to spare - one GPU\n",
                             dp.get_name(), prof.size(),
                             (double)pairs_bytes / 1073741824.0,
                             (double)session / 1073741824.0,
                             (long long)o.max_context,
                             (double)fb / 1073741824.0,
                             (double)(room - pairs_bytes) / 1073741824.0);
                multi_gpu = false;
                split_auto = false;
                split_devs.clear();
                split_at.clear();
                o.layer_split.clear();
            } else {
                std::fprintf(stderr,
                             "strata generate: --split-skip-if-fits: CUDA0 "
                             "(%s) would hold only %.2f of the "
                             "profile's %.2f GiB (%.2f GiB free, %.2f GiB for "
                             "the session, drafter and "
                             "reserve): the split stays\n",
                             dp.get_name(),
                             (double)std::max<int64_t>(room, 0) / 1073741824.0,
                             (double)pairs_bytes / 1073741824.0,
                             (double)fb / 1073741824.0,
                             (double)held_back / 1073741824.0);
            }
        }
    }
    strata::core::Verifier::set_commit_async(!multi_gpu);   // see Verifier::set_commit_async
    // --pipeline-windows (opt-in): one conversation's windows with the two stages of a layer split overlapped.  Decided
    // here, before any stage sizes its expert cache (the second verifier per stage and the snapshots are allocated
    // after the caches, so their room is kept out of them).  What it does not support turns it off, said once.
    if (o.pipeline_windows > 0) {
        const bool helpers = o.expert_cache_remote[0] > 0 || o.expert_cache_remote[1] > 0 || o.expert_cache_remote[2] > 0;
        const char* off = !o.serve ? "it needs --serve"
                        : split_same ? "it needs each stage of the layer split on its own GPU"
                        : !multi_gpu ? "it needs a layer split on two GPUs"
                        : split_devs.size() != 1 ? "it needs a layer split into exactly two stages"
                        : o.mtp.empty() || o.spec < 2 ? "it needs the MTP drafter (--mtp, --spec)"
                        : o.batch != 0 ? "not with --batch slots"
                        : o.peer_device >= 1 ? "not with --peer-device"
                        : helpers ? "not with the helper caches (--expert-cache-device1..3, --remote-expert-opt)"
                        : nullptr;
        if (off != nullptr) {
            std::fprintf(stderr, "strata generate: --pipeline-windows %d is off: %s\n", o.pipeline_windows, off);
            o.pipeline_windows = 0;
        }
    }
    // ---- layer split across GPUs: each later stage's own copy of the dense weights, its session and (the last) the
    // head, made on its device before the host arena is mapped (as the drafter below, for the same WDDM reason)
    std::vector<std::unique_ptr<GpuStage>> stages;
    // One stage's dense weights on its device (the caller has made it current): `trim` = only layers [lb, le).
    auto load_stage_weights = [&](GpuStage &st, int64_t lb, int64_t le,
                                  bool trim) -> bool {
        try {
    std::set<std::string> skip_s = skip_base;
        uint64_t pool_s = pool_bytes;
        if (trim) {
            add_foreign(lb, le, skip_s, false);
            strata::core::NativeDense::set_layer_range((int) lb, (int) le);
            if (!strata::core::WeightTable::pool_bytes(o.pack, pool_s, err, &skip_s)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return false;
            }
            std::fprintf(stderr, "strata generate: layer split: CUDA%d loads the dense weights of layers %lld-%lld only\n",
                         st.dev, (long long) lb, (long long) le - 1);
        } else {
            skip_s = skip;
        }
        void* arena_s = nullptr;
        if (DPCT_CHECK_ERROR(arena_s = (void *)strata::malloc_device_guarded(
                                 pool_s, dpct::get_in_order_queue())) != 0 ||
            !st.wt.load(o.pack, arena_s, pool_s, err,
                        skip_s.empty() ? nullptr : &skip_s)) {
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            size_t free_b = 0, total_b = 0; // #486: what that card had free
            /*
            DPCT1106: 'cudaMemGetInfo' was migrated with the Intel
            extensions for device information which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            dpct::get_current_device().get_memory_info(free_b, total_b);
            std::fprintf(stderr, "strata generate: layer split, CUDA%d weights: %s (%llu MiB needed, %llu MiB of %llu "
                                 "MiB free on that card)\n", st.dev,
                         err.empty() ? "the weight arena does not fit" : err.c_str(),
                         (unsigned long long) (pool_s >> 20), (unsigned long long) (free_b >> 20),
                         (unsigned long long) (total_b >> 20));
            return false;
        }
        if (!o.native_dense_gguf.empty() && !st.dense.load(o.native_dense_gguf, st.wt, err, o.native_ple_key)) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d native dense projections: %s\n", st.dev,
                         err.c_str());
            return false;
        }
        return true;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    };
    for (size_t i = 0; multi_gpu && i < split_devs.size(); ++i) {
        stages.push_back(std::make_unique<GpuStage>());
        GpuStage& st = *stages.back();
        st.dev = split_devs[i];
        double free_gib = 0;
        if (!strata::core::RemoteExperts::preflight(st.dev, free_gib, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        const strata::core::OnDevice on(st.dev);
        // Later stages' weights load here (0.1.39), except `--layer-split auto` with the trim asked: those load after
        // the split search (see late_weights), so `auto` can trim them too.
        if (!late_weights) {
            const int64_t lb = stage_trim ? split_at[i] : 0;
            const int64_t le = stage_trim ? (i + 1 < split_at.size() ? split_at[i + 1] : g.n_layers) : 0;
            if (!load_stage_weights(st, lb, le, stage_trim)) return 1;
        }
        // THE SESSION AND THE HEAD WAIT FOR THE SPLIT SEARCH.  `session_bytes` prices a stage's session by its
        // LAYER RANGE (the carve - every stage used to hold all 48 layers' state whatever it ran), so the
        // sessions are allocated after the search below has set `st.lb`/`st.le`; the last stage's head follows.
        /*
        DPCT1025: The SYCL queue is created ignoring the flag and priority
        options.
        */
        if (DPCT_CHECK_ERROR(
                st.stream = dpct::get_current_device().create_queue(true)) !=
                0 ||
            /*
            DPCT1025: The SYCL queue is created ignoring the flag and
            priority options.
            */
            DPCT_CHECK_ERROR(
                st.adapt_stream =
                    dpct::get_current_device().create_queue(true)) != 0 ||
            DPCT_CHECK_ERROR(st.adapt_ev = new sycl::event()) != 0) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: its streams failed\n", st.dev);
            return 1;
        }
        // a control vector (the speed projection): its tables on this device too - the stage's layers apply it here
        if (!strata::kernels::cvec_replicate(err)) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: %s\n", st.dev, err.c_str());
            return 1;
        }
        // --vision: this device's image-position table (the identity until a picture request), read by every rope
        // kernel its stage runs - set before any of its graphs is captured
        if (o.vision) {
            if (DPCT_CHECK_ERROR(
                    st.mrope = sycl::malloc_device<int32_t>(
                        mrope_host.size(), dpct::get_in_order_queue())) != 0 ||
                /*
                DPCT1114: cudaMemcpy is migrated to asynchronization
                memcpy, assuming in the original code the source host memory is
                pageable memory. If the memory is not pageable, call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                    st.mrope, mrope_host.data(),
                    mrope_host.size() * sizeof(int32_t)).wait()) != 0) {
                std::fprintf(stderr, "strata generate: layer split, CUDA%d: the image position table failed\n", st.dev);
                return 1;
            }
            strata::kernels::mrope_table_set(st.mrope);
        }
        // its own PCIe share of the missed experts (the same rule as CUDA0's above: its link is probed).  A given
        // --pcie-frac is every stage's share and skips these probes (pcie_given); there is no per-stage setting yet.
        st.pcie_frac = o.pcie_frac;
        if (!pcie_given && native_pack) {
            std::string bursts;
            const double bw = probe_pcie_h2d_gbps(&bursts, true);
            if (bw > 0.0) st.pcie_frac = pcie_frac_for_gbps(bw, 0.55);
            std::fprintf(stderr, "strata generate: layer split: CUDA%d PCIe probe %.1f GB/s (best of %s) -> pcie_frac "
                                 "%.2f\n", st.dev, bw, bursts.c_str(), st.pcie_frac);
        }
        if (!late_weights) {
            size_t fb = 0, tb = 0;
            /*
            DPCT1106: 'cudaMemGetInfo' was migrated with the Intel
            extensions for device information which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            dpct::get_current_device().get_memory_info(fb, tb);
            std::fprintf(stderr, "strata generate: layer split: CUDA%d holds its weights; %.2f GiB free (its session "
                                 "follows the split search)\n", st.dev, (double) fb / 1073741824.0);
        }
    }
    GpuStage* const last_st = stages.empty() ? nullptr : stages.back().get();

    std::vector<std::pair<int32_t, int32_t>> profile;
    if (!o.expert_profile.empty()) {
        int64_t pslots = 0;
        if (!strata::core::read_expert_profile(o.expert_profile, g.n_layers, g.n_expert, profile, pslots, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        // An explicit number truncates the ranked list ("what would 2,000 slots give" without rebuilding the
        // file).  `--expert-cache 0` used to take the count the profile was built for; the profile now ranks
        // every pair (issue #46: a card that holds more than the old 8,000 used to stop there), so it means auto.
        if (o.expert_cache == 0) o.expert_cache = -1;
        // Multi-GPU: STRATA_PEER_HOT=<f> gives the peer card a share f of the HOT pairs, so both cards
        // compute routed experts every layer (the primary alone did ~25 of ~30 per layer-window).  Of the first
        // STRATA_PEER_HOT_AT (default 8700, ~ the primary's slots) ranks, every pair with floor((r+1)f) > floor(rf)
        // moves to just after that point: the primary fills past them, the peer (which takes what the primary does
        // not hold, in order) gets them first.
        if (o.peer_device >= 1) {   // default 0.45 (measured: 0.3-0.6 all better than 0; 0 = off, e.g. for the gate)
            const char* ph = std::getenv("STRATA_PEER_HOT");
            const double f = ph ? std::atof(ph) : 0.45;
            const char* pa = std::getenv("STRATA_PEER_HOT_AT");
            const size_t at = std::min(profile.size(), (size_t) (pa ? std::atoll(pa) : 8700));
            if (f > 0.0 && f < 1.0 && at > 0) {
                std::vector<std::pair<int32_t, int32_t>> keep, moved;
                for (size_t r = 0; r < at; ++r) {
                    const bool to_peer = (int64_t) ((double) (r + 1) * f) > (int64_t) ((double) r * f);
                    (to_peer ? moved : keep).push_back(profile[r]);
                }
                const size_t n_moved = moved.size();
                // the primary's share continues with the ranks after `at` until it is full; then the moved ones
                std::vector<std::pair<int32_t, int32_t>> out;
                out.reserve(profile.size());
                out.insert(out.end(), keep.begin(), keep.end());
                const size_t fill = std::min(profile.size(), at + n_moved);   // what the primary still takes
                out.insert(out.end(), profile.begin() + (long) at, profile.begin() + (long) fill);
                out.insert(out.end(), moved.begin(), moved.end());
                out.insert(out.end(), profile.begin() + (long) fill, profile.end());
                profile.swap(out);
                std::fprintf(stderr, "strata generate: STRATA_PEER_HOT %.2f: %zu of the first %zu ranked pairs moved "
                                     "behind rank %zu (the peer's)\n", f, n_moved, at, fill);
            }
        }
        std::fprintf(stderr, "strata generate: profile %s: %zu ranked pairs, built for %lld slots\n",
                     o.expert_profile.c_str(), profile.size(), (long long) pslots);
    }
    // #477: the whole ranking as loaded, the prior of --expert-profile-save's order (a layer split keeps only
    // CUDA0's pairs in `profile` below).  Empty without --expert-profile-save.
    std::vector<std::pair<int32_t, int32_t>> profile_loaded;
    if (!o.expert_profile_save.empty()) profile_loaded = profile;
    // ---- layer split across GPUs: "auto" places the split points by a cost model of one decode window, measured on
    // the 5080 + 3090 rig (bench/results/2026-09-29-layer-split):
    //   - every layer costs its GPU a time inversely proportional to SMs x clock (0.33 ms on an RTX 5080, 0.50 on a
    //     3090: the per-layer round trip and kernels, not the bytes - both cards have ~950 GB/s);
    //   - an expert no cache holds costs ~190 ms per unit of routed mass: the CPU pool in decode and the PCIe stream
    //     in prompts (fitted: the sweep's best K, 26-28, is where one more layer on the faster card stops paying
    //     for the ~0.1% of the mass it pushes out of its cache);
    //   - which experts a cache holds: its layers' profiled pairs, hottest first, until its free VRAM (less the
    //     reserve, the prompt path's buffers and, on a later GPU, 1 GiB for its windows and the drafter) is used;
    //     the routed mass a cache holds follows the COVERAGE CURVE below, not the (r+1)^-1.2 this used to use.
    // Up to 4 GPUs every placement is tried; beyond, the layers are shared in proportion to speed.
    // A FOUR-WAY PLACEMENT USED TO BE THE PROPORTIONAL GUESS AND NOTHING ELSE.  That formula reads only
    // `layer_ms`, so on four GPUs the free VRAM a placement leaves for expert caches - the whole point of
    // the search - never entered it.  It does now; see the ns == 4 branch below.
    // STRATA_SPLIT_MISS_MS tunes the miss cost (a slower CPU: higher).
    // THE PROMPT PATH'S BUFFERS ARE BORROWED FROM THE CACHE, NOT WITHHELD BESIDE IT.  With borrowing the cache
    // is sized first and at full size, and the prompt path is laid out in the tail of it (`Prefill::relayout`),
    // so it withholds no VRAM of its own and this reserve is zero.  Only without borrowing - no profile to fill
    // a cache from, or --no-prefill-borrow - do the buffers take a reserve, and then this estimate stands in
    // for buffers that cannot be priced exactly yet because the sessions do not exist.  `plan_lend` uses the
    // exact `Prefill::bytes_needed` as soon as it can.
    const bool pf_borrow = !o.no_prefill_borrow && !o.expert_profile.empty();
    // (#340: the estimate predates the streamed ring: from 1024-token chunks the prompt path also holds a ring of
    // whole expert blobs, which a split without borrowing sizes at 96 (Prefill::set_ring_override below) and books
    // here - without it a `--no-prefill-borrow` split filled the cards and the draft head no longer fit)
    const int64_t split_ring_mib =
        (multi_gpu && !pf_borrow && o.prefill_chunk >= 1024)
            ? (int64_t) ((96ull * (uint64_t) strata::kernels::cpu::expert_layout().max_blob + (1ull << 20) - 1) >> 20)
            : 0;
    if (split_ring_mib > 0) strata::prefill::Prefill::set_ring_override(96);
    const int64_t split_pf_mib =
        (o.prefill_chunk > 0 && !pf_borrow) ? 160 + (o.prefill_chunk * 680) / 1024 + split_ring_mib : 0;
    // ---- WHAT A STAGE RESERVES, AND ON WHICH STAGE.  The flat 1 GiB this used to withhold from EVERY stage
    // after the first was booked "for its windows and the drafter", but the windows measure 75 MiB ("window up
    // to 6 tokens, 74.1 MiB of device buffers", on every boot) and the drafter is loaded on ONE stage - the
    // last one, which is also the only one that holds the head.  On the two identical 8 GB cards that GiB was
    // the entire difference between CUDA0's cache and CUDA1's: 814 slots against 188, 2026-09-30.  Both of
    // those allocations are already made before a stage's cache is sized, so what has to be held back here is
    // the windows and - only on the stage that carries them - the drafter and the head.
    const int64_t kWindowMib = 96;       // the verify windows; 75 MiB measured, rounded up
    const int64_t kDrafterMib = 1000;    // the MTP drafter (839 MiB) + the head, on the last stage only
    // --pipeline-windows: a second verifier on every stage (its arena alone is 70-75 MiB and its graphs come on top:
    // 64 MiB left two RTX 5070 Ti under Linux short of it) and, with the decode pipelined, two copies of the first
    // stage's GDN state (sized from its session where the first card's cache is sized; this is the split search's
    // estimate).  Allocated after the caches, so kept out of them.
    const int64_t kPipeWindowMib = o.pipeline_windows > 0 ? 160 : 0;
    const int64_t kPipeSnapMib = o.pipeline_windows >= 2 ? 96 : 0;   // ~3 MiB per GDN layer, twice, ~15 layers
    auto gdn_snapshot_bytes = [&](const strata::core::SessionState& s0) -> size_t {
        return (size_t) s0.gdn_alloc * ((size_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                        (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1)) * sizeof(float);
    };
    // #340: with the own prompt buffers chosen by the split's rule (not asked for with --no-prefill-borrow) the
    // boundary is searched as the borrowing configuration would (no reserve): the reserve then only makes the caches
    // smaller, which measured cost no decode (K=28 on 9070 XT + R9700: 58.4 tok/s own vs 58.5 borrowing), while a
    // search with the reserve moved the boundary to K=32 and decode to 54.8. STRATA_SPLIT_OWN_PLACE=reserve: the
    // search sees the reserve.
    // ---- `--layer-split auto` with the trim asked: a later stage's weights load after this search, so its arena is
    // not in `cudaMemGetInfo` yet.  Price it from the pack index as the full (untrimmed) copy the search has always
    // seen on the card (`pool_bytes` reads the index and allocates nothing), so the split it picks is the split it
    // has always picked: this buys cache, not a different placement.  Every other run has its weights loaded: 0.
    std::vector<uint64_t> stage_pool(stages.size() + 1, 0);
    if (late_weights)
        for (size_t i = 1; i < stage_pool.size(); ++i) stage_pool[i] = pool_bytes;
    static const bool place_with_reserve = [] {
        const char* v = std::getenv("STRATA_SPLIT_OWN_PLACE");
        return v != nullptr && std::string(v) == "reserve";
    }();
    auto stage_room = [&](int dev, bool later, bool drafter,
                          bool search = false) -> int64_t {
        try {
    const strata::core::OnDevice on(dev);
        size_t fb = 0, tb = 0;
        dpct::get_current_device().get_memory_info(fb, tb);   // #423 (tmking01): dpct dropped cudaMemGetInfo here
        /*
        DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions
        for device information which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */

        const int64_t pf = search && split_own_auto && !place_with_reserve ? 0 : split_pf_mib;
        const int64_t base_reserve =
            later && o.vram_reserve_later_mib >= 0 ? o.vram_reserve_later_mib : o.vram_reserve_mib;
        const int64_t reserve = (base_reserve + pf + (later ? kWindowMib : kPipeSnapMib) +
                                 kPipeWindowMib + (drafter ? kDrafterMib : 0)) << 20;
        return std::max<int64_t>((int64_t) fb - reserve, 0);
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    };
    if (multi_gpu && split_auto) {
        const auto& lay = strata::kernels::cpu::expert_layout();
        const int ns = (int) stages.size() + 1;
        std::vector<int64_t> cap((size_t) ns), used((size_t) ns);
        std::vector<double> layer_ms((size_t) ns);
        for (int i = 0; i < ns; ++i) {
            const int dev = i == 0 ? 0 : stages[(size_t) i - 1]->dev;
            cap[(size_t) i] = stage_room(i == 0 ? -1 : dev, i > 0, i + 1 == ns, true);
            if (late_weights) cap[(size_t) i] = std::max<int64_t>(cap[(size_t) i] - (int64_t) stage_pool[(size_t) i], 0);
            int sms = 0, khz = 0;
            sms = dpct::get_device(dev).get_max_compute_units();
            if (DPCT_CHECK_ERROR(
                    khz = dpct::get_device(dev).get_max_clock_frequency()) !=
                    0 ||
                khz <= 0) khz = 1800000;
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            const double speed =
                std::max(1.0, (double)sms * (double)khz / 1e6); // SMs x GHz
            layer_ms[(size_t) i] = 0.33 * (84.0 * 2.617) / speed;
            std::fprintf(stderr, "strata generate: layer split auto: CUDA%d %d SMs at %.2f GHz -> %.2f ms per layer, "
                                 "%.2f GiB free before its session carve\n", dev, sms, khz / 1e6, layer_ms[(size_t) i],
                         (double) cap[(size_t) i] / 1073741824.0);
        }
        const double miss_ms = std::getenv("STRATA_SPLIT_MISS_MS") ? std::atof(std::getenv("STRATA_SPLIT_MISS_MS")) : 190.0;
        // ---- THE COVERAGE CURVE: HOW MUCH ROUTED MASS A CACHE HOLDS, AND WHY IT IS NO LONGER A POWER LAW.
        //
        // `mass[r]` used to be (r+1)^-1.2, fitted to the 5080+3090 sweep.  That fit is still right ON THE RIG IT
        // WAS FITTED TO - at 20,000 pairs held it predicts 99.35% of the routed mass and that rig measured 99.4%.
        // It is wrong on the IQ3_S 4-way rig, badly: it claims the top 5,805 pairs hold 94.9% of the mass, and
        // that rig measures 69.2% (four boots, 14.9M decode lookups).  No exponent reconciles them, because the
        // curve is a property of the MODEL'S ROUTING, not a universal constant - and the profile file carries
        // only a RANKING, with no counts, so it cannot be read off the data either.  (tools/make_profile.py has
        // the frequencies, and drops them; see the note there.)
        //
        // What the two rigs do agree on is that the hit rate follows the HELD COUNT and almost nothing else: two
        // different splits held 14,541 and 14,384 pairs and measured 92.2% and 89.8% of the routed mass.  So the
        // curve is taken over the held FRACTION f, as H(f) = 1 - (1 - f)^b, and each rank's mass is that curve's
        // increment, which makes any prefix of N pairs sum to exactly H(N/total) while a cache holding
        // colder-than-prefix pairs still scores below it.
        //
        // b = 3 IS CHOSEN TO LEAVE THE PREVIOUSLY VALIDATED REGIME WHERE IT WAS.  At 20,000 held (f = 0.8138) it
        // gives 99.35%, against the old fit's 99.35% and that rig's measured 99.4% - the 2-GPU placement is
        // untouched.  At f = 0.59 it gives 93.2% against 92.2% measured.  Below f ~ 0.5 the two curves diverge,
        // and that is the whole point: it is exactly the regime where the old one was wrong, and where it told
        // the search that stranding a card's VRAM costs almost nothing.
        //
        // THE CURVE IS A FIT TO ONE MODEL'S TRAFFIC.  STRATA_SPLIT_COVER_B overrides b for a model whose routing
        // is sharper or flatter, and STRATA_SPLIT_MISS_MS still scales the miss cost it feeds.
        const size_t n_ranked = profile.size();
        const double cover_b = std::getenv("STRATA_SPLIT_COVER_B") ? std::atof(std::getenv("STRATA_SPLIT_COVER_B")) : 3.0;
        std::vector<double> mass(n_ranked);
        double total_mass = 0;
        // Up to 3 GPUs the 0.1.39 curve (r+1)^-1.2 is kept, so 2- and 3-GPU placements do not move (the new curve is a
        // fit to a 4-way rig and was not measured on them); STRATA_SPLIT_COVER_B asks for the coverage curve at any
        // GPU count.
        const bool coverage_curve = ns >= 4 || std::getenv("STRATA_SPLIT_COVER_B") != nullptr;
        for (size_t r = 0; r < n_ranked; ++r) {
            if (!coverage_curve) {
                total_mass += (mass[r] = std::pow((double) r + 1.0, -1.2));
                continue;
            }
            const double hi = std::pow(1.0 - (double) r / (double) n_ranked, cover_b);
            const double lo = std::pow(1.0 - (double) (r + 1) / (double) n_ranked, cover_b);
            total_mass += (mass[r] = hi - lo);
        }
        auto cost = [&](int64_t l) -> int64_t {
            return native_pack ? ((int64_t) lay.blob_bytes(l) + 255) / 256 * 256 : (int64_t) lay.max_blob;
        };
        // the predicted window time (ms) of a placement, and the routed mass its caches hold
        auto predict = [&](const std::vector<int64_t>& at, double& held_mass, int64_t& held) -> double {
            // THE CARVE, PRICED: a placement gives stage i the layers [lb, le), and that range's session is a
            // real cost on its device - subtracted here so the search knows what it leaves for experts.  This
            // is why the sessions are allocated after the search: `session_bytes` is pure arithmetic.
            std::vector<int64_t> capr((size_t) ns);
            for (int i = 0; i < ns; ++i) {
                const int64_t lb = i == 0 ? 0 : at[(size_t) i - 1];
                const int64_t le = i + 1 < ns ? at[(size_t) i] : g.n_layers;
                capr[(size_t) i] = cap[(size_t) i] - (int64_t) strata::core::session_bytes(g, o.max_context, K, lb, le);
            }
            std::fill(used.begin(), used.end(), 0);
            held_mass = 0;
            held = 0;
            std::vector<bool> full((size_t) ns, false);
            for (size_t r = 0; r < profile.size(); ++r) {
                const int64_t l = profile[r].first;
                int st = 0;
                while (st + 1 < ns && l >= at[(size_t) st]) ++st;
                if (full[(size_t) st]) continue;
                if (used[(size_t) st] + cost(l) > capr[(size_t) st]) { full[(size_t) st] = true; continue; }   // as the fill
                used[(size_t) st] += cost(l);
                held_mass += mass[r];
                ++held;
            }
            held_mass /= std::max(total_mass, 1e-9);
            double ms = miss_ms * (1.0 - held_mass);
            for (int i = 0; i < ns; ++i) {
                const int64_t lb = i == 0 ? 0 : at[(size_t) i - 1], le = i + 1 < ns ? at[(size_t) i] : g.n_layers;
                ms += (double) (le - lb) * layer_ms[(size_t) i];
            }
            return ms;
        };
        std::vector<int64_t> best, at((size_t) ns - 1);
        double best_ms = 1e30, best_mass = 0;
        int64_t best_held = 0, tried = 0;
        // #1440: of the placements within STRATA_SPLIT_HEADROOM_TOL (default 0.2%) of the best predicted time, the one
        // whose tightest card keeps the most VRAM beyond all its layers' experts (split_rules::pick_with_headroom).
        // On by default here (a full xe card moves buffers to system memory instead of failing); STRATA_SPLIT_HEADROOM=0
        // keeps the first best, as before.
        static const bool split_headroom = [] {
            const char* v = std::getenv("STRATA_SPLIT_HEADROOM");
            return v == nullptr || std::atoi(v) != 0;
        }();
        const double headroom_tol =
            std::getenv("STRATA_SPLIT_HEADROOM_TOL") ? std::atof(std::getenv("STRATA_SPLIT_HEADROOM_TOL")) : 0.002;
        std::vector<int64_t> need_prefix((size_t) g.n_layers + 1, 0);
        for (int64_t l = 0; l < g.n_layers; ++l)
            need_prefix[(size_t) l + 1] = need_prefix[(size_t) l] + (int64_t) g.n_expert * cost(l);
        auto range_session = [&](int64_t lb, int64_t le) -> int64_t {
            return (int64_t) strata::core::session_bytes(g, o.max_context, K, lb, le);
        };
        std::vector<strata::program::split_rules::Placement> scored;
        std::vector<double> scored_mass;
        std::vector<int64_t> scored_held;
        auto score = [&](const std::vector<int64_t>& cuts, double ms, double hm, int64_t held) {
            if (!split_headroom) return;
            scored.push_back({cuts, ms, strata::program::split_rules::min_spare(cuts, g.n_layers, cap, need_prefix,
                                                                                range_session)});
            scored_mass.push_back(hm);
            scored_held.push_back(held);
        };
        auto consider = [&]() {
            double hm = 0;
            int64_t held = 0;
            const double ms = predict(at, hm, held);
            ++tried;
            score(at, ms, hm, held);
            if (ms < best_ms) { best = at; best_ms = ms; best_mass = hm; best_held = held; }
        };
        const int64_t L = g.n_layers;
        // share the layers in proportion to speed: the fallback beyond four GPUs, and the only placement a
        // model too small to hold four stages has (see the ns == 4 branch).
        auto proportional = [&]() {
            double total = 0;
            for (const double c : layer_ms) total += 1.0 / c;
            double acc = 0;
            for (int i = 0; i + 1 < ns; ++i) {
                acc += 1.0 / layer_ms[(size_t) i];
                at[(size_t) i] = std::clamp<int64_t>((int64_t) std::llround(acc / total * (double) L),
                                                     i == 0 ? 2 : at[(size_t) i - 1] + 1, L - (ns - 1 - i));
            }
            consider();
        };
        if (ns == 2) {
            for (int64_t k = 2; k < L; ++k) { at[0] = k; consider(); }
        } else if (ns == 3) {
            for (int64_t k1 = 2; k1 + 1 < L; ++k1)
                for (int64_t k2 = k1 + 1; k2 < L; ++k2) { at[0] = k1; at[1] = k2; consider(); }
        } else if (ns == 4) {
            // ---- EVERY FOUR-WAY PLACEMENT IS TRIED.  This branch used to share the layers in proportion to
            // speed and score that ONE candidate, and that formula reads only `layer_ms` - never the free VRAM.
            // So on a four-GPU rig cache capacity did not influence the placement AT ALL, and the card with
            // the most room left could be handed the fewest layers: on the 3060/3060/5060/5060 box CUDA1 ended
            // up with 1,841 MiB no cache could use while CUDA2 ran 15 layers it could only hold a third of.
            // All C(L-3,3) placements - 15,180 at 48 layers - are now scored.
            //
            // A STAGE'S FILL DEPENDS ONLY ON ITS OWN RANGE AND ITS OWN FREE VRAM.  `predict` hands each ranked
            // pair to the one stage that owns its layer and each stage stops at its first pair that does not
            // fit, so the four fills are independent and a placement's held mass is a SUM over its four
            // ranges.  That is what makes this affordable: each distinct range is walked once and memoised,
            // 1,121 walks instead of 15,180 - measured at 14 ms against 740 ms, for the same answer.
            // `le` runs to L inclusive, so a row is L+1 wide - keying on a stride of L would put the last
            // stage's [lb, L) one past the end of the vector.
            //
            // NOTE: on v0.1.39 the search only runs under `--layer-split auto`, where every card still holds
            // the whole model's dense weights, so `cap[i]` is already net of them and the stage's room is what
            // `predict` subtracts - the session.  (On the branch this came from the carve reached `auto` too,
            // and each stage's own weight bytes were taken off here as well.)
            const size_t stride = (size_t) L + 1;
            std::vector<double> held_mass_at((size_t) ns * (size_t) L * stride, -1.0);
            std::vector<int64_t> held_at(held_mass_at.size(), 0);
            auto range_hold = [&](int i, int64_t lb, int64_t le) -> size_t {
                const size_t key = ((size_t) i * (size_t) L + (size_t) lb) * stride + (size_t) le;
                if (held_mass_at[key] < 0.0) {
                    // one stage's own carve, exactly as `predict` prices it
                    const int64_t room = cap[(size_t) i] -
                        (int64_t) strata::core::session_bytes(g, o.max_context, K, lb, le);
                    double hm = 0;
                    int64_t used = 0, cnt = 0;
                    for (size_t r = 0; r < profile.size(); ++r) {
                        const int64_t l = profile[r].first;
                        if (l < lb || l >= le) continue;
                        if (used + cost(l) > room) break;   // the fill stops exactly where predict's does
                        used += cost(l);
                        hm += mass[r];
                        ++cnt;
                    }
                    held_mass_at[key] = hm;
                    held_at[key] = cnt;
                }
                return key;
            };
            const double denom = std::max(total_mass, 1e-9);
            for (int64_t k1 = 2; k1 + 2 < L; ++k1)
                for (int64_t k2 = k1 + 1; k2 + 1 < L; ++k2)
                    for (int64_t k3 = k2 + 1; k3 < L; ++k3) {
                        const size_t a = range_hold(0, 0, k1), b = range_hold(1, k1, k2);
                        const size_t c = range_hold(2, k2, k3), d = range_hold(3, k3, L);
                        const double hm = held_mass_at[a] + held_mass_at[b] + held_mass_at[c] + held_mass_at[d];
                        const double ms = miss_ms * (1.0 - hm / denom) +
                                          (double) k1 * layer_ms[0] + (double) (k2 - k1) * layer_ms[1] +
                                          (double) (k3 - k2) * layer_ms[2] + (double) (L - k3) * layer_ms[3];
                        ++tried;
                        score({k1, k2, k3}, ms, hm / denom, held_at[a] + held_at[b] + held_at[c] + held_at[d]);
                        if (ms < best_ms) {
                            best = {k1, k2, k3};
                            best_ms = ms;
                            best_mass = hm / denom;
                            best_held = held_at[a] + held_at[b] + held_at[c] + held_at[d];
                        }
                    }
            // Four stages into fewer than five layers: nothing was enumerated, so keep the old guess rather
            // than leave `best` empty - an empty split would leave every stage's lb/le unset.
            if (best.empty()) proportional();
        } else {
            // Beyond four GPUs the placements outnumber any budget worth spending at startup (C(46,4) is
            // 163,185 at five).  Nothing on this rig reaches it.
            proportional();
        }
        if (split_headroom && scored.size() > 1) {
            const size_t first = strata::program::split_rules::fastest(scored);
            const size_t pick = strata::program::split_rules::pick_with_headroom(scored, headroom_tol);
            if (pick != first && scored[first].cuts == best) {
                auto fmt_k = [](const std::vector<int64_t>& v) {
                    std::string s;
                    for (const int64_t k : v) s += (s.empty() ? "" : ",") + std::to_string(k);
                    return s;
                };
                std::fprintf(stderr, "strata generate: layer split auto: K=%s instead of K=%s (predicted %.1f vs %.1f ms): its "
                                     "tightest card keeps %.2f GiB beyond all its layers' experts, against %.2f GiB "
                                     "(#1440: a full card can page to system memory); STRATA_SPLIT_HEADROOM=0 keeps K=%s\n",
                             fmt_k(scored[pick].cuts).c_str(), fmt_k(best).c_str(), scored[pick].ms, best_ms,
                             (double) scored[pick].min_spare / 1073741824.0,
                             (double) scored[first].min_spare / 1073741824.0, fmt_k(best).c_str());
                best = scored[pick].cuts;
                best_ms = scored[pick].ms;
                best_mass = scored_mass[pick];
                best_held = scored_held[pick];
            }
        }
        split_at = best;
        std::string ks;
        for (const int64_t k : split_at) ks += (ks.empty() ? "" : ",") + std::to_string(k);
        std::fprintf(stderr, "strata generate: layer split auto: K=%s - predicted %.1f ms per decode window; the caches "
                             "hold %lld of %zu profiled pairs (~%.1f%% of the routed mass), best of %lld placements\n",
                     ks.c_str(), best_ms, (long long) best_held, profile.size(), 100.0 * best_mass, (long long) tried);
    }
    for (size_t i = 0; i < split_at.size(); ++i)
        if (split_at[i] >= g.n_layers) {
            std::fprintf(stderr, "strata generate: --layer-split: layer %lld is past the last (%lld)\n",
                         (long long) split_at[i], (long long) (g.n_layers - 1));
            return 2;
        }
    // the stage that runs a layer (0: CUDA0's)
    auto stage_of = [&](int64_t l) -> int {
        int st = 0;
        while (st < (int) split_at.size() && l >= split_at[(size_t) st]) ++st;
        return st;
    };
    // the whole ranking, hottest first, before each stage takes its own layers' share: the resident RAM copy ranks
    // the experts no GPU holds over ALL the layers (with CUDA0's share alone, a copy that did not fit kept only layers
    // 0..K-1's experts and read every later layer's from the files)
    std::vector<std::pair<int32_t, int32_t>> profile_all;
    if (multi_gpu) {
        profile_all = profile;
        std::vector<std::pair<int32_t, int32_t>> mine;
        for (const auto& pr : profile) {
            const int st = stage_of(pr.first);
            (st == 0 ? mine : stages[(size_t) st - 1]->profile).push_back(pr);
        }
        profile.swap(mine);
        for (size_t i = 0; i < stages.size(); ++i) {
            stages[i]->lb = split_at[i];
            stages[i]->le = i + 1 < stages.size() ? split_at[i + 1] : g.n_layers;
        }
    }

    // ---- `--layer-split auto` with the trim asked: THE STAGES' WEIGHTS, NOW THAT THE SPLIT IS KNOWN.  Each later card
    // loads the dense weights of the layers it runs and nothing else (before this they loaded in the loop above, a
    // full copy per card: on the 4-way IQ3_S rig CUDA1 3650 expert slots, CUDA2 1241, CUDA3 533 and a 768-token prompt
    // chunk; carved 4608 / 2485 / 1772 and 6144).  The search priced these arenas from the pack index (`stage_pool`).
    if (late_weights)
        for (size_t i = 0; i < stages.size(); ++i) {
            GpuStage& st = *stages[i];
            const strata::core::OnDevice on(st.dev);
            if (!load_stage_weights(st, st.lb, st.le, true)) return 1;
            size_t fb = 0, tb = 0;
            /*
            DPCT1106: 'cudaMemGetInfo' was migrated with the Intel
            extensions for device information which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            dpct::get_current_device().get_memory_info(fb, tb);
            std::fprintf(stderr, "strata generate: layer split: CUDA%d holds its weights; %.2f GiB free (its session "
                                 "follows)\n", st.dev, (double) fb / 1073741824.0);
        }

    // ---- CUDA0's session, and the stages' sessions: sized to each device's own layer range (the carve).  A
    // stage that runs [lb, le) carves only those layers' GDN rows and QSA pools - before the carve every stage
    // held all 48 layers' state whatever layers it ran, which is the same disease the chunked-QSA-prefill PR
    // fixed in llama.cpp: allocation sized by the whole model instead of the device's own work.
    {
        const strata::core::OnDevice on0(0);
        const int64_t hi0 = multi_gpu ? split_at[0] : -1;
        // the elastic K/V (--kv-grow, see kvg_ensure): one GPU, the whole K/V in VRAM (no streaming), a profiled cache
        // that can give slots up, and every expert in RAM for the CPU to compute the ones it gives up
        {
            const char* ev = std::getenv("STRATA_KV_GROW");
            bool remote = false;
            for (const int r : o.expert_cache_remote) remote = remote || r > 0;
            const bool asked = ev != nullptr && ev[0] != '\0' ? ev[0] != '0' : o.kv_grow;
            const bool on = asked && !multi_gpu && o.kv_resident <= 0 &&
                            !o.expert_profile.empty() && !o.resident_cpu_experts && o.expert_cache != 0 && !remote &&
                            strata::core::vmm_available() &&
                            // the batch slots carve their own K/V and --vram-elastic's cache is not one VMM range
                            o.batch == 0 && !o.vram_elastic && o.peer_device < 0;
            if (asked && !on)
                std::fprintf(stderr, "strata generate: --kv-grow is off (one GPU, a profile, the whole K/V in VRAM, "
                                     "every expert in RAM, no --batch, --vram-elastic or --peer-device)\n");
            const char* iv = std::getenv("STRATA_KV_GROW_INIT");
            strata::core::qsa_set_kv_elastic(on, iv != nullptr && std::atoll(iv) > 0 ? std::atoll(iv) : 16384);
            strata::core::ExpertCache::set_vmm(on);
        }
        if (DPCT_CHECK_ERROR(
                sbuf = (void *)sycl::malloc_device(
                    strata::core::session_bytes(g, o.max_context, K, 0, hi0),
                    dpct::get_in_order_queue())) != 0) {
            std::fprintf(stderr, "strata generate: session state allocation failed\n");
            return 1;
        }
        if (strata::core::session_init(g, o.max_context, K, sbuf, ss, 0, hi0) == 0) {
            std::fprintf(stderr, "strata generate: session_init failed\n");
            return 1;
        }
        if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1)
            std::fprintf(stderr, "strata generate: KV streaming: %lld of %lld cells per QSA layer in VRAM, the K/V in "
                                 "%.2f GiB of pinned RAM\n",
                         (long long) (ss.qsa_states[ss.qsa_primary()].n_slots * 4),
                         (long long) o.max_context, (double) strata::core::qsa_kv_host_bytes() / 1073741824.0);
        // the PLE block above built everything but the history, which session_init has just carved
        ss.ple.hist = ss.ple_hist;
        // (#167) generate mode starts from an empty sequence, and nothing else zeroes this state before the prompt
        // path or the verifier reads it (--serve zeroes it per request when nothing is reused)
        strata::core::session_zero(ss, g, nullptr, main_cs);
        if (DPCT_CHECK_ERROR(
                dpct::get_current_device().queues_wait_and_throw()) != 0) {
            std::fprintf(stderr, "strata generate: zeroing the session state failed\n");
            return 1;
        }
        if (!o.ple_gguf.empty()) {
            if (!ss.ple.ready()) {
                std::fprintf(stderr, "strata generate: the PLE run is not ready after construction\n");
                return 1;
            }
            std::fprintf(stderr, "strata generate: PLE on, table %llu rows of %s\n",
                         (unsigned long long) ple_table.rows(), o.ple_gguf.c_str());
        }
    }
    for (size_t i = 0; i < stages.size(); ++i) {
        GpuStage& st = *stages[i];
        const strata::core::OnDevice on(st.dev);
        void* sbuf_s = nullptr;
        if (DPCT_CHECK_ERROR(sbuf_s = (void *)sycl::malloc_device(
                                 strata::core::session_bytes(g, o.max_context,
                                                             K, st.lb, st.le),
                                 dpct::get_in_order_queue())) != 0 ||
            strata::core::session_init(g, o.max_context, K, sbuf_s, st.ss,
                                       st.lb, st.le) == 0) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: the session state failed\n", st.dev);
            return 1;
        }
        const bool last = i + 1 == stages.size();
        const strata::core::WeightRef* wo_s = st.wt.find("output.weight");
        if (wo_s == nullptr ||
            (last && !o.native_head_gguf.empty() && !st.head.load(o.native_head_shards, g.n_embd, wo_s->ne1, err))) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d head: %s%s\n", st.dev,
                         wo_s == nullptr ? "output.weight is missing" : err.c_str(),
                         wo_s == nullptr ? "" : vram_free_note().c_str());
            return 1;
        }
        size_t fb = 0, tb = 0;
        /*
        DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions
        for device information which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        dpct::get_current_device().get_memory_info(fb, tb);
        std::fprintf(stderr, "strata generate: layer split: CUDA%d holds its weights, session [%lld, %lld)%s; "
                             "%.2f GiB free\n", st.dev, (long long) st.lb, (long long) st.le,
                     last ? " and the head" : "", (double) fb / 1073741824.0);
    }

    // --batch: every stage carves o.batch more sessions like its own (same layer range and context), one per
    // slot of the batch windows.  Before the expert cache is sized, so `--expert-cache auto` leaves them room.
    // Recommend, never force: a count the engine cannot run is said and adjusted (or batching left off) - the
    // engine still starts and serves one request at a time.
    // --batch-mtp / STRATA_BATCH_MTP=1 (opt-in): a slot also verifies one MTP proposal per window.  Recommend, never
    // force: when it cannot run it is said and left off (plain batching still starts); default off = exactly 0.1.39.
    const char* batch_mtp_env = std::getenv("STRATA_BATCH_MTP");
    bool batch_mtp = o.batch_mtp || (batch_mtp_env != nullptr && batch_mtp_env[0] != '\0' && batch_mtp_env[0] != '0');
    if (batch_mtp) {
        const char* why = o.batch < 2 ? "it needs --batch 2 or more" : o.mtp.empty() ? "it needs --mtp"
                        : o.spec < 2 ? "it needs --spec T (T >= 2)" : (multi_gpu || split_same || !stages.empty())
                        ? "it is for one GPU (no layer split or helper) for now" : !o.serve ? "it needs --serve" : nullptr;
        if (why != nullptr) {
            std::fprintf(stderr, "strata generate: WARNING: --batch-mtp is off: %s\n", why);
            batch_mtp = false;
        }
    }
    std::vector<std::vector<std::unique_ptr<strata::core::SessionState>>> bslot_ss;
    if (o.batch != 0) {
        const int cap = strata::kernels::kVerifyMaxT;
        const char* off = !o.serve ? "it needs --serve" : o.batch < 2 ? "it needs 2 or more slots"
                        : split_same ? "it needs each stage of a layer split on its own GPU" : nullptr;
        if (off != nullptr) {
            std::fprintf(stderr, "strata generate: WARNING: --batch %d is off: %s\n", o.batch, off);
            o.batch = 0;
        } else if (o.batch > cap && !batch_mtp) {
            std::fprintf(stderr, "strata generate: WARNING: --batch %d: a batch window holds at most %d rows, so %d "
                                 "slots\n", o.batch, cap, cap);
            o.batch = cap;
        }
    }
    if (o.batch > 0 && o.batch_groups > 1 && (stages.empty() || o.batch % o.batch_groups != 0)) {
        std::fprintf(stderr, "strata generate: WARNING: --batch-groups %d needs a layer split and to divide --batch %d; "
                             "one group\n", o.batch_groups, o.batch);
        o.batch_groups = 1;
    }
    if (o.batch > 0) {
        bslot_ss.resize(1 + stages.size());
        int fit = o.batch;   // the slots every stage could carve
        uint64_t bytes0 = 0;
        for (size_t k = 0; k < bslot_ss.size(); ++k) {
            const int dev = k == 0 ? 0 : stages[k - 1]->dev;
            const int64_t lo = k == 0 ? 0 : stages[k - 1]->lb;
            const int64_t hi = k == 0 ? (multi_gpu ? split_at[0] : -1) : stages[k - 1]->le;
            const strata::core::OnDevice on_k(dev);
            const uint64_t bytes = strata::core::session_bytes(g, o.max_context, K, lo, hi);
            if (k == 0) bytes0 = bytes;
            for (int b = 0; b < fit; ++b) {
                auto u = std::make_unique<strata::core::SessionState>();
                void* buf = nullptr;
                if (DPCT_CHECK_ERROR(buf = (void *)sycl::malloc_device(
                                         bytes, dpct::get_in_order_queue())) !=
                        0 ||
                    strata::core::session_init(g, o.max_context, K, buf, *u, lo,
                                               hi) == 0) {
                    /*
                    DPCT1026: The call to cudaGetLastError was removed
                    because this functionality is redundant in SYCL.
                    */
                    if (buf != nullptr)
                        sycl::free(buf, dpct::get_in_order_queue());
                    size_t fb = 0, tb = 0;
                    /*
                    DPCT1106: 'cudaMemGetInfo' was migrated with the Intel
                    extensions for device information which may not be supported
                    by all compilers or runtimes. You may need to adjust the
                    code.
                    */
                    dpct::get_current_device().get_memory_info(fb, tb);
                    std::fprintf(stderr, "strata generate: WARNING: --batch %d: slot %d's session does not fit on CUDA%d "
                                         "(%.2f GiB each, %.2f GiB free); %d slot%s\n", o.batch, b, dev,
                                 (double) bytes / 1073741824.0, (double) fb / 1073741824.0, b, b == 1 ? "" : "s");
                    fit = b;
                    break;
                }
                strata::core::session_zero(*u, g, nullptr, nullptr);
                bslot_ss[k].push_back(std::move(u));
            }
            dpct::get_current_device().queues_wait_and_throw();
            size_t fb = 0, tb = 0;
            /*
            DPCT1106: 'cudaMemGetInfo' was migrated with the Intel
            extensions for device information which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            dpct::get_current_device().get_memory_info(fb, tb);
            std::fprintf(stderr, "strata generate: --batch: %d slot sessions on CUDA%d (%.2f GiB each); %.2f GiB free\n",
                         (int) bslot_ss[k].size(), dev, (double) bytes / 1073741824.0, (double) fb / 1073741824.0);
        }
        for (auto& v : bslot_ss) if ((int) v.size() > fit) v.resize((size_t) fit);   // the same count on every stage
        if (fit < 2) {
            std::fprintf(stderr, "strata generate: WARNING: --batch is off: not two slot sessions fit (a smaller "
                                 "--max-context or KV streaming, --kv-resident, makes them smaller)\n");
            bslot_ss.clear();
            o.batch = 0;
        } else {
            o.batch = fit;
            if (o.batch_groups > 1 && o.batch % o.batch_groups != 0) o.batch_groups = 1;
            // the sessions' VRAM is the expert cache's: say what it costs (docs/BATCHING.md has the measured trade)
            std::fprintf(stderr, "strata generate: --batch %d: the slot sessions take %.2f GiB of VRAM on CUDA0 that the "
                                 "expert cache would otherwise hold\n", o.batch,
                         (double) bytes0 * o.batch / 1073741824.0);
        }
    }

    // Secure MTP's CUDA0 allocations before the large host arena is registered with both CUDA contexts.
    // In particular WDDM can refuse the draft weights after mapping tens of GiB of host pages.
    strata::core::MtpDrafter mtp;
    // the draft layer's geometry (the canonical model's MTP head), `static` because MtpDrafter keeps a reference; the batch
    // slots' draft-KV copies use this one too
    static const strata::core::ModelGeometry draft_geometry{};
    std::vector<std::unique_ptr<strata::core::MtpDrafter>> slot_mtp;
    if (!o.mtp.empty()) {
        if (o.spec < 2) {
            std::fprintf(stderr, "strata generate: --mtp is ignored without --spec T (T >= 2)\n");
            o.mtp.clear();
        }
        if (!o.mtp.empty()) mtp.set_prompt_len((int64_t) o.tokens.size());
        mtp.set_hnorm_per_stream(o.mtp_hnorm_stream);   // --mtp-hnorm stream (opt-in), before any capture
        mtp.set_q4(o.mtp_q4 == "all" || o.mtp_q4 == "proj", o.mtp_q4 == "all" || o.mtp_q4 == "head");   // before load
        if (!o.mtp_draft_vocab.empty()) mtp.set_draft_vocab(o.mtp_draft_vocab);
        // the draft layer is the canonical model's MTP head (512 experts) even when the target is pruned,
        // so it always sees the canonical geometry; `static` because MtpDrafter keeps a reference
        // with a layer split across GPUs the drafter reads the last stage's residual: it lives on that device
        const strata::core::OnDevice on_mtp(last_st ? last_st->dev : -1);
        // --pipeline-windows 2: 8 rows (the chain runs on past the window in flight: its drafts, its bonus token and
        // the next window's drafts), and the round/step graphs carry the forcing kernel
        const int mtp_t = o.pipeline_windows >= 2 ? strata::kernels::kVerifyMaxT : o.spec;
        if (o.pipeline_windows >= 2) mtp.set_force_capture(true);
        if (!o.mtp.empty() && !mtp.load(o.mtp, draft_geometry, last_st ? last_st->ss : ss, mtp_t, err, o.mtp_window)) { std::fprintf(stderr, "strata generate: %s%s\n", err.c_str(), vram_free_note().c_str()); return 1; }
        mtp.set_ple_session(&ss);
        if (batch_mtp) {
            for (int b = 0; b < o.batch; ++b) {
                auto d = std::make_unique<strata::core::MtpDrafter>();
                if (!d->load(o.mtp, draft_geometry, *bslot_ss[0][(size_t) b], o.spec, err, o.mtp_window, &mtp)) {
                    std::fprintf(stderr, "strata generate: batch MTP slot %d: %s%s\n", b, err.c_str(),
                                 vram_free_note().c_str());
                    return 1;
                }
                d->set_max_drafts(1);   // the first candidate verifies one proposal per slot
                d->set_ple_session(bslot_ss[0][(size_t) b].get());
                slot_mtp.push_back(std::move(d));
            }
        }
    }
    // THE HEAD BEFORE THE CACHE, AND BEFORE THE ARENA.  The expert cache takes what is free minus the reserve, so
    // everything allocated after it comes out of the reserve.  The native head (~0.5 GB with IQ3_S) was loaded after
    // it and ate most of the 700 MiB: 128K IQ3_S ended with 30 MiB free, the driver paged, and a request stalled for
    // good at its first verify window.  Loaded first, the cache is sized around it.  #620: and before the expert
    // arena registers tens of GiB of host pages, like the drafter above - WDDM then refused the head's cudaMalloc on
    // a 16 GB card with the desktop on the iGPU ("native head upload: out of memory" with GiBs free).
    const strata::core::WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { std::fprintf(stderr, "strata generate: output.weight is missing\n"); return 1; }
    const int64_t n_vocab = wo->ne1;
    strata::core::NativeHead native_head;
    if (!o.native_head_gguf.empty() && !multi_gpu) {   // a layer split's head is on its last stage
        const auto head_t0 = std::chrono::steady_clock::now();
        if (!native_head.load(o.native_head_shards, g.n_embd, n_vocab, err)) {
            std::fprintf(stderr, "strata generate: %s%s\n", err.c_str(), vram_free_note().c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: experimental native Q5_K head, %llu bytes, in %.1f s\n",
                     (unsigned long long) native_head.weight_bytes(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - head_t0).count());
    }
    std::vector<float> logits((size_t) n_vocab);
    float* d_logits = nullptr;
    if (DPCT_CHECK_ERROR(
            d_logits = (float *)sycl::malloc_device(
                (size_t)n_vocab * 4, dpct::get_in_order_queue())) != 0) {
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use
        the error codes. The cudaGetLastError function call was replaced with 0.
        You need to rewrite this code.
        */
        (void)0;
        std::fprintf(stderr, "strata generate: the logits buffer failed%s\n", vram_free_note().c_str());
        return 1;
    }
    // Create the additional contexts after MTP has secured CUDA0 memory, but
    // before the host arena maps its expert pages into their address spaces.
    for (int r = 1; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0 && !early_remote) {
        double free_gib = 0;
        if (!strata::core::RemoteExperts::preflight(remote_dev[r], free_gib, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: CUDA%d context ready, %.2f GiB free before expert arena registration\n",
                     remote_dev[r], free_gib);
    }

    // ---- the CPU expert pool
    //
    // R2.1: the experts are loaded into a RESIDENT ARENA by default.  The mmap path is kept behind
    // `--mmap-experts` because it is the A/B arm, not because it is competitive.
    //
    // The reasoning is the review's C1 and it is now measured on both sides.  `FileExpertSource` maps the 34 GB
    // file, and mapped file pages are the first thing the OS reclaims; the engine's rate then depends on whether
    // the standby list happens to hold `experts.bin`, which is why two consecutive runs of the SAME BINARY with
    // the SAME FLAGS measured 71.97 and 34.78 ms/token in the pool (7.54 vs 12.18 tok/s).  The arena is
    // anonymous memory the engine owns, and the pool runs at 19.41 ms/token - 1.79x better than the warm mmap
    // and 3.7x better than the cold one.
    //
    // IT IS NOT PINNED, and that is reported rather than hidden: `cudaHostRegister` on 31.64 GiB fails with
    // "out of memory" (you cannot pin 34 of 63 GB) and the arena falls back to 4 KB anonymous pages.  That is
    // fine for the CPU pool - which is all that exists today - and NOT fine for Phase 3, whose cache fills and
    // CPU/PCIe miss split need the GPU to DMA out of this arena.  Read `note()` when that lands.
    //
    // The earlier "the arena does not fit" conclusion was WRONG and is worth recording: the failure was a stale
    // CUDA error left set by the failed `cudaHostRegister` and read later by `gr_read`'s launch check.  See the
    // note in `pinned.cu`.
    strata::core::FileExpertSource src;
    {   // the card, and whether this build has code for it (a binary built for other GPUs fails at its first kernel
        // otherwise, after the whole expert arena has loaded) - before the arena starts loading
        int dev = 0;
        dpct::device_info p{};
        const bool named =
            DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) == 0 &&
            DPCT_CHECK_ERROR(dpct::get_device(dev).get_device_info(p)) == 0;
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use
        the error codes. The cudaGetLastError function call was replaced with 0.
        You need to rewrite this code.
        */
        if (!named) 0;
        const char *name =
            named && p.get_name()[0] ? p.get_name() : "(an unnamed GPU)";
#if defined(STRATA_USE_HIP)
        std::fprintf(stderr, "strata generate: GPU %d: %s (%s)\n", dev, name, named ? p.gcnArchName : "?");
#if defined(_WIN32)
        // #468 #461: which HIP runtime was loaded - the bundled one beside the exe, or an AMD driver's System32 copy
        if (HMODULE h = GetModuleHandleA("amdhip64_7.dll")) {
            char path[MAX_PATH] = {};
            if (GetModuleFileNameA(h, path, MAX_PATH) > 0)
                std::fprintf(stderr, "strata generate: HIP runtime %s\n", path);
        }
#endif
#else
        std::fprintf(
            stderr, "strata generate: GPU %d: %s, compute capability %d.%d%s\n",
            dev, name,
            /*
            DPCT1005: The SYCL device version is different from CUDA
            Compute Compatibility. You may need to rewrite this code.
            */
            strata::cc_major_of(p.get_major_version()),
            strata::cc_minor_of(p.get_minor_version()),
            strata::emulated_cc()
                ? " (STRATA_EMULATE_CC: a test mode, the card is emulated)"
                : "");
#if 0 // the SYCL port has no CUDA runtime to mismatch (#542): dpct::get_major_version is the device version
#if !defined(STRATA_HIP_GFX906) // a CUDA toolkit mismatch; the gfx906 HIP build
                                // has no DPCT_COMPAT_RT_VERSION
        {   // #542: a build whose libcudart is older than its headers (a CUDA 13 kit with a dangling libcudart.so that
            // CMake resolved to the system's CUDA 12 one) reads cudaDeviceProp shifted - silently, and slowly
            int rt = 0;
            /*
            DPCT1043: The version-related API is different in SYCL. An
            initial code was generated, but you need to adjust it.
            */
            if (DPCT_CHECK_ERROR(rt = dpct::get_major_version(
                                     dpct::get_current_device())) == 0 &&
                rt / 1000 != DPCT_COMPAT_RT_VERSION / 1000)
                std::fprintf(
                    stderr,
                    "strata generate: WARNING: this engine was compiled with "
                    "CUDA %d.%d headers but "
                    "loaded a CUDA %d.%d runtime (libcudart): GPU properties "
                    "can read wrong and some "
                    "kernels go unused. Rebuild it against one toolkit (cmake "
                    "-DCUDAToolkit_ROOT=<the "
                    "toolkit>, with its libcudart.so present) (#542)\n",
                    DPCT_COMPAT_RT_VERSION / 1000,
                    DPCT_COMPAT_RT_VERSION % 1000 / 10, rt / 1000,
                    rt % 1000 / 10);
        }
#endif
#endif // #if 0
#endif
        const std::string e = strata::core::device_code_error();
        if (!e.empty()) {
            std::fprintf(
                stderr,
                "strata generate: this engine has no code for %s (sm_%d%d): %s "
                "- rebuild it for this "
                /*
                DPCT1005: The SYCL device version is different from CUDA
                Compute Compatibility. You may need to rewrite this code.
                */
                "card (setup does: START-HERE.bat --setup)\n",
                name, p.get_major_version(), p.get_minor_version(), e.c_str());
            return 1;
        }
    }
    strata::core::ArenaExpertSource arena_src;
    strata::core::GgufExpertSource gguf_src;
    strata::core::ExpertSource* srcp = nullptr;
    if (o.stream_experts && !o.mmap_experts) {
        // the SYCL port: a card that holds every expert needs no host copy of them, and a 23 GiB machine cannot
        // hold one. The profile fill reads each blob once from the shard; nothing reads from here afterwards
        // unless a slot is lent to the prefill and refilled (the experts no slot holds: the pinned host mirror).
        if (!native_pack || o.native_preset.empty()) {
            std::fprintf(stderr, "strata generate: --stream-experts needs a native (IQ) pack and --native SHARD1\n");
            return 2;
        }
        if (!gguf_src.open(o.native_preset, g.n_layers, g.n_expert, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: experts streamed from the GGUF on demand (--stream-experts): no host "
                             "arena; every expert must fit the VRAM cache\n");
        srcp = &gguf_src;
    } else if (o.mmap_experts) {
        // FileExpertSource maps the pack's experts.bin: a canonical pack has it; a native (IQ) pack has it when
        // built with `tools/iq_pack.py --experts-bin` (the per-layer blob sizes of its layout, PR #121).  The low-RAM
        // mode: the experts come from the file through the OS cache instead of a pinned copy in RAM, for a PC whose
        // GPU holds most of them but whose RAM cannot hold them all.
        // CS-T: a native pack without experts.bin maps the model's GGUF shards instead (native_experts.txt's spans,
        // checked against the files first) - no 30-77 GB copy of the experts on the disk
        src.set_gguf(o.native_preset);
        if (const char* v = std::getenv("STRATA_FETCH_THREADS"); v != nullptr && std::atoi(v) > 0)
            src.set_fetch_threads(std::atoi(v));
        if (!src.open(o.pack, g.n_layers, g.n_expert, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: experts via mmap (--mmap-experts; %s)\n",
                     src.gguf_mode() ? "the GGUF shards in place, no experts.bin" : "the A/B arm of R2.1");
        // #286: with a RAM budget the hottest experts live in it, and the rest are read from the drive unbuffered
        // when the file cache could not keep them beside the budget anyway (a 32 GB PC) - the mapped reads' page
        // faults are small requests on the critical path, and their pages take the RAM the budget was sized for
        // #773: without a RAM budget there is no RAM copy and the file cache IS the expert tier (measured on a
        // 32 GB, 2 x 16 GB rig: forcing unbuffered reads there re-read 163-629 GB from the drive and halved the
        // speed), so STRATA_UNBUFFERED_LOAD=1 is not honoured in that mode
        if (o.resident_budget > 0) {
            std::string why;
            const bool ub = src.set_unbuffered(o.resident_budget, why);
            std::fprintf(stderr, "strata generate: the file tier reads %s (%s)\n",
                         ub ? "unbuffered" : "through the file cache", why.c_str());
        } else if (const char* env = std::getenv("STRATA_UNBUFFERED_LOAD"); env != nullptr && env[0] != '\0' &&
                   env[0] != '0') {
            std::fprintf(stderr, "strata generate: the file tier reads through the file cache (STRATA_UNBUFFERED_LOAD=%s "
                                 "ignored: without --resident-budget-gib the file cache holds the experts)\n", env);
        }
        srcp = &src;
    } else {
        arena_src.set_gguf(o.native_preset);   // plan v0.3 P6: a native pack may take its experts from shard 1
        // Under WDDM (Windows, WSL2), a multi-GPU run (a layer split, or remote experts) starts with at most 8 GiB of
        // mapped host pages: pinning all of it into two contexts leaves WDDM refusing every later allocation -
        // measured on the 5080 + 3090 rig: cudaMemGetInfo and the next cudaMalloc fail.  Unregistered layers remain
        // in the resident arena; their streamed experts go through the pinned staging ring.  A Linux driver has no
        // such limit, so there the whole arena is pinned (#253).  STRATA_ARENA_PIN_GIB overrides both ways.
        const int pin_env = strata::core::arena_pin_cap_gib();   // -1 unset, -2 "auto" (#243, Windows sliced pin)
        const bool pin_wddm_cap = pin_env < 0 && (o.expert_cache_remote[0] > 0 || multi_gpu) && under_wddm();
        const uint64_t pin_limit = pin_env >= 0 ? (uint64_t) pin_env << 30 : pin_wddm_cap ? (8ull << 30) : 0;
        if (pin_env >= 0)
            std::fprintf(stderr, "strata generate: STRATA_ARENA_PIN_GIB=%d: %s\n", pin_env,
                         pin_env == 0 ? "the whole expert arena is pinned" : "the expert arena's pinning is capped");
        else if (pin_wddm_cap)
            std::fprintf(stderr, "strata generate: multi-GPU under WDDM: at most 8 GiB of the expert arena is pinned "
                                 "(STRATA_ARENA_PIN_GIB changes it)\n");
        if (!arena_src.open(o.pack, g.n_layers, g.n_expert, /*threads=*/6, err, pin_limit,
                            o.shared_expert_arena)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (!arena_src.ram_warning().empty())   // #633: said before the load's numbers, which it explains
            std::fprintf(stderr, "strata generate: WARNING: %s\n", arena_src.ram_warning().c_str());
        std::fprintf(stderr, "strata generate: expert arena: %s\n", arena_src.note().c_str());
        std::fprintf(stderr, "strata generate: loaded %.2f GiB at %.2f GiB/s\n",
                     (double) strata::kernels::cpu::expert_layout().total / (1024.0 * 1024 * 1024),
                     arena_src.load_gib_per_second());
        // A rate under ~0.2 GiB/s is not the hardware.  Task Scheduler / service contexts throttle this
        // read+fill about 24x (measured 0.05 vs 1.42 GiB/s for the same binary, args and cache state; the
        // scheduler's defaults - Below normal priority and a least-privilege token - were the only
        // difference between the runs).  Say so instead of letting the user blame the disk; see
        // docs/DETAILS.md, "Running it at startup (Task Scheduler)".
#ifdef _WIN32   // a Windows launch context; elsewhere a load this slow is the disk
        if (arena_src.load_gib_per_second() > 0.0 && arena_src.load_gib_per_second() < 0.2) {
            std::fprintf(stderr,
                         "strata generate: hint: ~24x below what this hardware streams from a normal "
                         "launch. If Strata is started by Task Scheduler or a service, register the task "
                         "with Priority 4 (Normal) and 'Run with highest privileges' - the scheduler's "
                         "defaults (Below normal + a least-privilege token) throttle the load. See "
                         "docs/DETAILS.md ('Running it at startup').\n");
        }
#endif
        srcp = &arena_src;
    }
    strata::kernels::cpu::ExpertPool pool(o.pool_workers, /*pin=*/true, /*host_works=*/!o.no_host_worker, o.pool_affinity);
    if (pool.is_hybrid() && pool.affinity() != strata::kernels::cpu::PoolAffinity::All) {
        const char* aff_str = pool.affinity() == strata::kernels::cpu::PoolAffinity::PCores ? "p-cores" :
                              pool.affinity() == strata::kernels::cpu::PoolAffinity::All ? "all" : "auto";
        std::fprintf(stderr, "strata generate: hybrid CPU detected (%d P-cores / %d threads, %d E-cores), pool workers: %d, affinity: %s\n",
                     pool.p_cores(), pool.p_threads(), pool.e_cores(), pool.workers(), aff_str);
    }
    {   // where the threads run (--host-core)
        const strata::kernels::cpu::CpuTopology ht = strata::kernels::cpu::detect_cpu_topology(true, o.pool_affinity);
        std::string on;
        for (int i = 0; i < pool.workers() && i < (int) ht.worker_cores.size(); ++i)
            on += (i > 0 ? "," : "") + std::to_string(ht.worker_cores[(size_t) i]);
        std::fprintf(stderr, "strata generate: %d pool workers on logical processors %s, the host thread on %d%s "
                             "(--host-core %s)\n", pool.workers(), on.c_str(), ht.host_core,
                     pool.host_works() ? " (draining too)" : "",
                     strata::kernels::cpu::host_core_setting() == strata::kernels::cpu::HostCore::Last ? "last" : "first");
        // the same facts in plain words, and whether each worker has a physical core of its own
        std::fprintf(stderr, "strata generate: %s\n", pool.placement_line().c_str());
    }
    if (o.no_ple_prefetch) strata::kernels::ple_prefetch_enable(false);
    // ---- R4's slot storage.  Allocated AFTER the weights and the session, so `cudaMemGetInfo` inside `open`
    // sees the memory this process actually has left rather than the card's idle figure - and refuses with both
    // numbers if the slots do not fit, instead of handing back a cache smaller than it was asked for.
    mem_mark("the weights, the session and the drafter");
    strata::core::ExpertCache xcache;
    // THE HEAD BEFORE THE CACHE: the native head and the logits are allocated above, before the expert arena (#620)
    const bool auto_cache = o.expert_cache < 0;
    bool reserve_adapted = false;   // #496: the auto sizing lowered the reserve so a small card's cache fits
    // --pipeline-windows: what it allocates on CUDA0 after the cache (see kPipeWindowMib)
    const int64_t pipe_first = o.pipeline_windows <= 0 ? 0
        : (kPipeWindowMib << 20) + (o.pipeline_windows >= 2 ? 2 * (int64_t) gdn_snapshot_bytes(ss) : 0);
    if (pipe_first > 0)
        std::fprintf(stderr, "strata generate: --pipeline-windows %d: %lld MiB of CUDA0 kept out of the expert cache "
                             "(the second verifier%s)\n", o.pipeline_windows, (long long) (pipe_first >> 20),
                     o.pipeline_windows >= 2 ? ", the GDN snapshots" : "");
    // The MiB the prompt path's OWN buffers (no loan) take, which the cache sizing leaves out.  0.1.39's rule is
    // 160 + chunk * 680 / 1024 (~16 GiB at 24576 against ~2.4 GiB really allocated); STRATA_OWNED_PRICE=exact prices the
    // real allocation instead (Prefill::bytes_needed_owned: 2 MiB pages, the ring in one piece, as PR #796 measured)
    // and gives that cache the difference.  A recommendation to try, so it is not the default.
    auto owned_prefill_mib = [&]() -> int64_t {
        if (!(o.prefill_chunk > 0 && !pf_borrow)) return 0;
        static const bool exact = [] { const char* v = std::getenv("STRATA_OWNED_PRICE"); return v != nullptr && std::string(v) == "exact"; }();
        if (!exact) return 160 + (o.prefill_chunk * 680) / 1024;
        const int64_t mib = ((int64_t) strata::prefill::Prefill::bytes_needed_owned(g, ss, o.prefill_chunk) + (1 << 20) - 1) >> 20;
        std::fprintf(stderr, "strata generate: STRATA_OWNED_PRICE=exact: the prompt path's own buffers for a %lld-token chunk: %lld MiB (the 0.1.39 rule: %lld MiB)\n",
                     (long long) o.prefill_chunk, (long long) mib, (long long) (160 + (o.prefill_chunk * 680) / 1024));
        return mib + 64;   // a margin for the allocator
    };
    if (o.expert_cache < 0) {
        size_t free_b = 0, total_b = 0;
        free_b = strata::core::device_free_bytes(); (void) total_b;
        // Plan v0.3 P5: the batched prompt path's chunk buffers are allocated later, so they are reserved here -
        // under WDDM an over-subscribed allocation does not fail, it pages to system memory and crawls.
        // (with borrowing - the default with a profile - the prompt path lends cache slots instead; `pf_borrow` is
        // the predicate a local `borrow` was here, hoisted above so both cache-size branches read the same one)
        const int64_t prefill_mib = owned_prefill_mib();
        // the draft layer's head and logits are allocated when it binds, after this: 0.1.27's CJK subset made them
        // ~110-180 MiB larger, and out of the reserve they left 16 GB cards below the stall line (#199)
        int64_t mtp_bind = (!o.mtp.empty() && native_head.loaded())
                               ? (int64_t) mtp.bind_bytes(native_head.row_bytes(), n_vocab) : 0;
        for (const auto& d : slot_mtp)
            mtp_bind += (int64_t) d->bind_bytes(native_head.row_bytes(), n_vocab);
        const int64_t reserve = (((int64_t) o.vram_reserve_mib + prefill_mib) << 20) + mtp_bind + pipe_first;
        const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        int64_t slots = ((int64_t) free_b - reserve) / blob;
        if (!profile.empty()) slots = std::min<int64_t>(slots, (int64_t) profile.size());
        o.expert_cache = (int) std::max<int64_t>(slots, 0);
        std::fprintf(stderr, "strata generate: expert cache auto: %.2f GiB free, %d MiB reserved (+%lld MiB for the "
                             "draft head) -> %d slots\n",
                     (double) free_b / 1073741824.0, o.vram_reserve_mib, (long long) (mtp_bind >> 20), o.expert_cache);
        // #496: the verify window cannot start without a cache (#174), and a cache too small to lend the prompt path
        // a 256-token chunk's buffers (plus the 128 slots a loan leaves; one slot without --prefill) makes it
        // allocate its own on top - more than the reserve.  When the default reserve leaves less than that (a 6 GB
        // card), the reserve shrinks to what leaves exactly that cache, down to kSmallReserveMib: what is allocated
        // after the cache - the prompt path's own part, the verify buffers, the draft head - comes out of the reserve,
        // and below ~550 MiB a card ends with less than the 256 MiB the serve check calls LOW (IQ3_XXS, 32K, a 300 MiB
        // reserve: 5 MiB left), so the cache gets no more than it needs, and the serve check says so plainly when it
        // ends LOW (`reserve_adapted`).  A reserve given on the command line is kept.  No slot at all: the start
        // stops, saying what is short and what makes room.  A card the default reserve leaves that much is sized as
        // before.
        constexpr int kSmallReserveMib = 300;
        const int64_t min_slots = (o.prefill_chunk > 0 && pf_borrow)
            ? ((int64_t) strata::prefill::Prefill::bytes_needed(g, ss, 256) + blob - 1) / blob + 128 : 1;
        if (o.expert_cache < min_slots && !o.vram_reserve_given && o.vram_reserve_mib > kSmallReserveMib) {
            // the largest reserve (in MiB) that still leaves min_slots
            const int64_t fit_mib = ((int64_t) free_b - mtp_bind - pipe_first - min_slots * blob) / (1 << 20) - prefill_mib;
            if (fit_mib >= kSmallReserveMib) {
                const int r = (int) std::min<int64_t>(fit_mib, o.vram_reserve_mib);
                int64_t s2 = ((int64_t) free_b - ((((int64_t) r + prefill_mib) << 20) + mtp_bind + pipe_first)) / blob;
                if (!profile.empty()) s2 = std::min<int64_t>(s2, (int64_t) profile.size());
                std::fprintf(stderr, "strata generate: expert cache auto: the %d MiB reserve leaves too few slots on "
                                     "this card (a working cache needs %lld): a %d MiB reserve instead -> %lld slots\n",
                             o.vram_reserve_mib, (long long) min_slots, r, (long long) s2);
                o.vram_reserve_mib = r;
                o.expert_cache = (int) s2;
                reserve_adapted = true;
            }
        }
        if (o.expert_cache == 0) {
            // what is short, and what makes room: the numbers a small card picks from
            const int64_t at_reserve = o.vram_reserve_given ? o.vram_reserve_mib
                                                            : std::min(o.vram_reserve_mib, kSmallReserveMib);
            const int64_t need_b = (((int64_t) at_reserve + prefill_mib) << 20) + mtp_bind + pipe_first + min_slots * blob;
            const int64_t short_mib = std::max<int64_t>(1, (need_b - (int64_t) free_b + (1 << 20) - 1) >> 20);
            const int64_t session_mib =
                (int64_t) (strata::core::session_bytes(g, o.max_context, K, 0, g.n_layers) >> 20);
            const std::string reserve_tip =
                o.vram_reserve_given && o.vram_reserve_mib > kSmallReserveMib
                    ? ", a smaller --vram-reserve-mib (" + std::to_string(o.vram_reserve_mib) + " now; " +
                          std::to_string(kSmallReserveMib) + " is enough on a small card)"
                    : std::string();
            std::fprintf(stderr, "strata generate: no VRAM is left for the expert cache: it needs at least %lld slots "
                                 "(%lld MiB), about %lld MiB more than this card has free. To make room: a smaller "
                                 "--max-context (the session, mostly its KV cache, takes %lld MiB at %lld tokens), "
                                 "--kv q4_0, the English draft subset (setup --draft-vocab en; the draft head takes "
                                 "%lld MiB now)%s, images on the CPU, or close other programs that use the GPU\n",
                         (long long) min_slots, (long long) ((min_slots * blob) >> 20), (long long) short_mib,
                         (long long) session_mib, (long long) o.max_context, (long long) (mtp_bind >> 20),
                         reserve_tip.c_str());
        }
    } else if (multi_gpu && o.expert_cache > 0) {
        // an explicit cache size leaves room for the prompt path's buffers and the reserve, or the first prompt
        // fails with "device buffers ... do not fit" (with borrowing - the default with a profile - the path lends
        // slots instead and `prefill_mib` is 0, so only the reserve is checked)
        size_t free_b = 0, total_b = 0;
        free_b = strata::core::device_free_bytes(); (void) total_b;
        const int64_t prefill_mib = owned_prefill_mib();
        const int64_t reserve = (((int64_t) o.vram_reserve_mib + prefill_mib) << 20) + pipe_first;
        const int64_t fit = std::max<int64_t>(((int64_t) free_b - reserve) / (int64_t) strata::kernels::cpu::expert_layout().max_blob, 0);
        if (o.expert_cache > fit) {
            // a WARNING that names the knob: the user asked for this size, and gets fewer slots
            std::fprintf(stderr, "strata generate: WARNING: layer split: --expert-cache %d leaves no room for the "
                                 "prompt path's buffers (%lld MiB) and the %d MiB reserve on CUDA0: %lld slots instead "
                                 "(a smaller --vram-reserve-mib leaves more of them)\n", o.expert_cache,
                         (long long) prefill_mib, o.vram_reserve_mib, (long long) fit);
            o.expert_cache = (int) fit;
        }
    }
    // plan v0.3 P6: a native pack's blobs differ per layer, so the same VRAM holds more experts than slots of
    // the largest blob would. The shared cache below sizes slots in profile order.
    // #369: that list does not match --expert-cache-per-layer. A range is layer * quota + n, so a size cut for
    // one layer can land in another layer's slot. Every expert in a layer is one size, so the per-layer list is
    // `quota` copies of that layer's own blob, in layer order. `--expert-cache N` stays the budget of N largest
    // blobs, and the quota is however many copies of every layer fit in it.
    std::vector<int64_t> sized_slots;
    uint64_t per_layer_bytes = 0;
    if (native_pack && o.expert_cache > 0 && o.expert_cache_per_layer) {
        size_t free_b = 0, total_b = 0;
        /*
        DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions
        for device information which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        dpct::get_current_device().get_memory_info(free_b, total_b);
        const auto& lay = strata::kernels::cpu::expert_layout();
        const int asked = o.expert_cache;
        const uint64_t budget = (uint64_t) asked * lay.max_blob;
        size_t free_room = free_b > ((size_t) o.vram_reserve_mib << 20) ? free_b - ((size_t) o.vram_reserve_mib << 20) : 0;
        const uint64_t cap = std::min<uint64_t>(budget, (uint64_t) free_room);
        uint64_t sum = 0;
        for (int64_t l = 0; l < g.n_layers; ++l)
            sum += (lay.blob_bytes(l) + 255) / 256 * 256;
        int64_t q = sum > 0 ? (int64_t) (cap / sum) : 0;
        if (q > g.n_expert) q = g.n_expert;
        if (q > 0) {
            per_layer_bytes = sum;
            sized_slots.reserve((size_t) q * (size_t) g.n_layers);
            for (int64_t l = 0; l < g.n_layers; ++l) {
                const int64_t b = (int64_t) lay.blob_bytes(l);
                for (int64_t s = 0; s < q; ++s) sized_slots.push_back(b);
            }
            o.expert_cache = (int) sized_slots.size();
            std::fprintf(stderr, "strata generate: per-layer slots use each layer's blob: %d uniform slots "
                                 "(%.2f GiB) -> %d slots, %lld per layer (%.2f GiB)\n",
                         asked, (double) budget / 1073741824.0, o.expert_cache, (long long) q,
                         (double) ((uint64_t) q * per_layer_bytes) / 1073741824.0);
        }
    } else if (native_pack && o.expert_cache > 0 && !profile.empty()) {
        size_t free_b = 0, total_b = 0;
        free_b = strata::core::device_free_bytes(); (void) total_b;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const uint64_t budget = (uint64_t) o.expert_cache * lay.max_blob;   // what the uniform sizing granted
        uint64_t used = 0;
        const size_t keep_free = ((size_t) o.vram_reserve_mib << 20) + (size_t) pipe_first;
        size_t free_room = free_b > keep_free ? free_b - keep_free : 0;
        const uint64_t cap = std::min<uint64_t>(budget, (uint64_t) free_room);
        for (const auto& pr : profile) {
            const uint64_t b = (lay.blob_bytes(pr.first) + 255) / 256 * 256;
            if (used + b > cap) break;
            used += b;
            sized_slots.push_back((int64_t) lay.blob_bytes(pr.first));
        }
        // #831: the sized-slots loop walks the profile's pairs, so the profile's length is the ceiling; an explicit
        // --expert-cache above it used to be cut with no word.  A warning only (what is allocated does not change).
        if (!auto_cache && o.expert_cache > 0 && sized_slots.size() == profile.size() && (size_t) o.expert_cache > profile.size())
            std::fprintf(stderr, "strata generate: WARNING: --expert-cache %d is more than the %zu pairs of the profile %s: "
                                 "%zu slots (the profile is the ceiling on a native pack; a profile that ranks every "
                                 "(layer, expert) pair lifts it - tools/make_profile.py)\n", o.expert_cache,
                         profile.size(), o.expert_profile.c_str(), profile.size());
        o.expert_cache = (int) sized_slots.size();
    }
    // #533: --vram-elastic: the cache in physical segments (the VRAM command resizes it between requests).  One GPU,
    // no helper caches, serve mode: anything else keeps the one cudaMalloc, said once.
    if (o.vram_elastic) {
        const char* why = !o.serve ? "it works between requests of --serve"
                        : multi_gpu ? "a layer split has a cache per GPU"
                        : std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                      [](int n) { return n > 0; }) ? "the helper caches on other GPUs"
                        : o.vram_segment_mib < 64 ? "--vram-segment-mib is below 64" : nullptr;
#if defined(STRATA_USE_HIP)
        if (why == nullptr) why = "it is NVIDIA-only for now";
#endif
        if (why != nullptr) {
            std::fprintf(stderr, "strata generate: --vram-elastic is off: %s\n", why);
            o.vram_elastic = false;
        } else {
            xcache.set_segment_bytes(o.vram_segment_mib << 20);
        }
    }
    if (o.expert_cache > 0) {
        // keep the first `keep_bytes` of the cache (the profile's hottest experts first); false when nothing is left
        auto shrink_to = [&](int64_t keep_bytes) -> bool {
            if (keep_bytes <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            if (per_layer_bytes > 0) {
                // Cutting the list short would put one layer's blob in the next layer's range.
                const auto& lay = strata::kernels::cpu::expert_layout();
                int64_t q = keep_bytes / (int64_t) per_layer_bytes;
                if (q > g.n_expert) q = g.n_expert;
                sized_slots.clear();
                if (q <= 0) { o.expert_cache = 0; return false; }
                for (int64_t l = 0; l < g.n_layers; ++l) {
                    const int64_t b = (int64_t) lay.blob_bytes(l);
                    for (int64_t s = 0; s < q; ++s) sized_slots.push_back(b);
                }
                o.expert_cache = (int) sized_slots.size();
                return true;
            }
            if (!sized_slots.empty()) {
                int64_t used = 0;
                size_t keep = 0;
                while (keep < sized_slots.size() && used + (sized_slots[keep] + 255) / 256 * 256 <= keep_bytes)
                    used += (sized_slots[keep++] + 255) / 256 * 256;
                sized_slots.resize(keep);
                o.expert_cache = (int) keep;
            } else {
                o.expert_cache = (int) (keep_bytes / (int64_t) strata::kernels::cpu::expert_layout().max_blob);
            }
            if (o.expert_cache <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            return true;
        };
        auto cache_bytes = [&]() -> int64_t {
            if (sized_slots.empty()) return (int64_t) o.expert_cache * (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            int64_t b = 0;
            for (const int64_t s : sized_slots) b += (s + 255) / 256 * 256;
            return b;
        };
        // With `--expert-cache auto` the reserve must still be free once the slots are WRITTEN: under WDDM an
        // allocation is not resident until it is touched, and the free figure read before it can be ~1 GB too
        // high.  A cache sized from it filled the card to 0 MiB, the driver then paged, and a request that needed a
        // page back while the verify graph spun on a host flag never finished.  So the slots are zeroed and the
        // free figure read again; while it is short of the reserve the cache is reopened smaller.
        // STRATA_TEST_CACHE_FAIL=N: the first N opens fail as an out-of-commit cudaMalloc does (tests the retry)
        int fake_fails = std::getenv("STRATA_TEST_CACHE_FAIL") ? std::atoi(std::getenv("STRATA_TEST_CACHE_FAIL")) : 0;
        int failed = 0;
        int zero_reads = 0;
        for (int attempt = 0;; ++attempt) {
            bool ok = false;
            if (fake_fails > 0) {
                --fake_fails;
                err = "ExpertCache: cudaMalloc failed: out of memory (STRATA_TEST_CACHE_FAIL)";
            } else {
                ok = sized_slots.empty()
                    ? xcache.open(o.expert_cache, g.n_layers, g.n_expert, (int64_t) strata::kernels::cpu::expert_layout().max_blob, err)
                    : xcache.open_sized(sized_slots, g.n_layers, g.n_expert, err);
            }
            if (!ok) {
                // Issue #60: on Windows a device allocation is also charged to the system commit (RAM + page file),
                // so with a small page file the cache's one big cudaMalloc fails while the VRAM is free.  An auto
                // cache then tries three quarters of the size, a few times, instead of stopping the engine.
                char commit[96] = "";
#if defined(_WIN32)
                MEMORYSTATUSEX ms{};
                ms.dwLength = sizeof ms;
                if (GlobalMemoryStatusEx(&ms))
                    std::snprintf(commit, sizeof commit, " (Windows has %.1f GiB of commit left: RAM + page file)",
                                  (double) ms.ullAvailPageFile / 1073741824.0);
#endif
                if (auto_cache && failed < 8 && shrink_to(cache_bytes() / 4 * 3)) {
                    ++failed;
                    std::fprintf(stderr, "strata generate: %s%s; trying a smaller expert cache: %d slots\n", err.c_str(),
                                 commit, o.expert_cache);
                    continue;
                }
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
#if defined(_WIN32)
                std::fprintf(stderr, "strata generate: on Windows the graphics card's memory also needs room in the page "
                                     "file: set it to \"System managed\" (System > About > Advanced system settings > "
                                     "Performance > Advanced > Virtual memory), or lower --expert-cache\n");
#endif
                return 1;
            }
            if (!auto_cache || attempt - failed >= 6) break;
            if (uint8_t* slot0 = xcache.device_slot(0)) strata::big_fill_zero(dpct::get_in_order_queue(), slot0, (size_t)xcache.bytes());
            dpct::get_current_device().queues_wait_and_throw();
            size_t free_b = 0, total_b = 0;
            free_b = strata::core::device_free_bytes(); (void) total_b;
            const int64_t want = ((int64_t) o.vram_reserve_mib << 20) + pipe_first;
            if ((int64_t) free_b >= want - (64ll << 20)) break;
            // short by (want - free); a figure of 0 only says "at least": the first two such reads give back 1 GiB
            // each (under WDDM the free figure read before the allocation runs ~0.7 GiB high), later ones a quarter
            int64_t give = want - (int64_t) free_b + (64ll << 20);
            if (free_b < ((size_t) 16 << 20))
                give = std::max<int64_t>(give, ++zero_reads <= 2 ? 1ll << 30 : xcache.bytes() / 4);
            const int64_t keep_bytes = xcache.bytes() - give;
            std::fprintf(stderr, "strata generate: only %lld MiB free once the slots are written (reserve %d MiB); "
                                 "shrinking the expert cache\n", (long long) (free_b >> 20), o.vram_reserve_mib);
            xcache.close();
            if (!shrink_to(keep_bytes)) break;
        }
        if (failed > 0 && o.expert_cache > 0)
            std::fprintf(stderr, "strata generate: expert cache: %d slots (%.2f GiB) after %d smaller tries - a bigger "
                                 "page file lets it use more of the free VRAM\n",
                         o.expert_cache, (double) xcache.bytes() / 1073741824.0, failed);
    }
    if (o.expert_cache > 0) {
        std::fprintf(stderr, "strata generate: expert cache %lld slots, %.2f GiB of VRAM; policy is\n",
                     (long long) xcache.slots(), xcache.gib());
        mem_mark("opening the expert cache");
        xcache.set_per_layer_admission(o.expert_cache_per_layer);
        // Round 328 warned here that the GPU hit path was wrong (tokens diverged from a cache-off run from
        // token 0). That fault was fixed long since (native_expert_parity, expert_parity, the grouped kernels'
        // tests), and the warning outlived it (issue #23). What remains is rounding: a GPU expert and the CPU's
        // compute the same quantized expert with different float order, so a near-tie can flip. Measured teacher-
        // forced on 2,557 tokens (bench/results/2026-09-27-cache-parity): 95-98% same top-1, and perplexity equal
        // (on - off = -0.005 +- 0.005 nats). Neither output is more correct than the other.
        std::fprintf(stderr,
                     "strata generate: the GPU computes the experts in the cache; it rounds differently from the CPU,\n"
                     "                 so a reply can differ slightly from a run without the cache (same quality:\n"
                     "                 bench/results/2026-09-27-cache-parity).\n");
        if (o.expert_cache_per_layer) {
            int64_t lo = 0, hi = 0;
            xcache.layer_slot_range(0, lo, hi);
            std::fprintf(stderr, "                 R4.2g PER-LAYER: each layer owns %lld slots (%lld..%lld).\n",
                         (long long) (hi - lo), (long long) lo, (long long) (hi - 1));
        } else if (profile.empty()) {
            std::fprintf(stderr, "                 compulsory-miss (fills with whatever the run routes first).\n");
        } else {
            std::fprintf(stderr, "                 PROFILE, ranked by routing frequency, no eviction.\n");
        }
    }
    // ---- R4.2e: fill the tier from the profile.  This is the only place the plan is applied, and it runs
    // ONCE: with `slots` pairs and `slots` slots the cache is full when this returns, so the decode-time
    // admission finds no room and every non-profiled expert stays a CPU miss.  That is what makes the profile
    // the policy rather than a hint.
    int64_t prefilled = 0;
    std::function<void(const char*)> verify_all_slots;   // STRATA_VERIFY_ALL_SLOTS (debug), set by the pipelined fill
    if (!profile.empty() && srcp != nullptr) {
        // #369 (dag08): per layer, a full layer skips only its own pairs - each layer takes its hottest experts until
        // its range is full (one full layer used to end the whole fill, leaving most layers empty)
        const bool per_layer = xcache.per_layer_admission();
        const int64_t want = per_layer ? (int64_t) profile.size()
                                       : std::min<int64_t>((int64_t) profile.size(), xcache.slots());
        const auto tfill = std::chrono::steady_clock::now();
        uint64_t fill_bytes = 0;
        // SYCL port: from the GGUF (--stream-experts) the fill is a pipeline. The serial form read each expert's three
        // slices, copied it, and waited for the copy before the next read (~18,000 experts, neither the SSD nor PCIe
        // kept busy). Here the slots are admitted first (the same profile order, so the same placement), the reads go
        // in file order by worker threads into page-locked batches, and a batch's copies run while the next batch is
        // read. STRATA_FILL_SERIAL=1: the serial form.
        const bool piped = srcp == &gguf_src && !per_layer && std::getenv("STRATA_FILL_SERIAL") == nullptr;
        if (piped) {
            struct Fill { int32_t slot; int32_t l, e; };
            std::vector<Fill> fills;
            fills.reserve((size_t) want);
            for (int64_t i = 0; i < want; ++i) {
                const int32_t slot = xcache.admit(profile[(size_t) i].first, profile[(size_t) i].second);
                if (slot == strata::core::kNotResident) break;
                fills.push_back({slot, (int32_t) profile[(size_t) i].first, (int32_t) profile[(size_t) i].second});
            }
            std::sort(fills.begin(), fills.end(), [](const Fill& a, const Fill& b) {
                return a.l != b.l ? a.l < b.l : a.e < b.e;   // file order: each tensor read front to back
            });
            const auto& lay = strata::kernels::cpu::expert_layout();
            constexpr size_t kBatch = 64;
            const size_t blob_cap = (size_t) lay.max_blob;
            sycl::queue& q = dpct::get_in_order_queue();
            uint8_t* pin = sycl::malloc_host<uint8_t>(2 * kBatch * blob_cap, q);
            if (pin == nullptr) {
                std::fprintf(stderr, "strata generate: the profile fill could not pin %zu MiB\n",
                             (2 * kBatch * blob_cap) >> 20);
                return 1;
            }
            const int nthreads = std::max(2, std::min(8, (int) std::thread::hardware_concurrency()));
            sycl::event done[2];
            bool read_ok = true;
            for (size_t b0 = 0, k = 0; b0 < fills.size() && read_ok; b0 += kBatch, ++k) {
                const size_t n = std::min(kBatch, fills.size() - b0);
                uint8_t* set = pin + (k % 2) * kBatch * blob_cap;
                done[k % 2].wait();                       // the copies that last read this half are finished
                std::atomic<size_t> next{0};
                std::atomic<bool> ok{true};
                std::vector<std::thread> ts;
                for (int t = 0; t < nthreads; ++t)
                    ts.emplace_back([&] {
                        for (size_t j; (j = next.fetch_add(1)) < n;) {
                            const Fill& f = fills[b0 + j];
                            if (!gguf_src.read_into(f.l, f.e, set + j * blob_cap, blob_cap)) ok = false;
                        }
                    });
                for (auto& t : ts) t.join();
                if (!ok) { read_ok = false; break; }
                for (size_t j = 0; j < n; ++j) {
                    const Fill& f = fills[b0 + j];
                    const size_t nb = (size_t) lay.blob_bytes(f.l);
                    done[k % 2] = q.memcpy(xcache.device_slot(f.slot), set + j * blob_cap, nb);
                    fill_bytes += nb;
                }
            }
            q.wait();
            sycl::free(pin, q);
            if (!read_ok) {
                std::fprintf(stderr, "strata generate: the profile fill could not read an expert from the GGUF\n");
                return 1;
            }
            prefilled = (int64_t) fills.size();
            if (std::getenv("STRATA_VERIFY_ALL_SLOTS") != nullptr) {   // debug: every filled slot read back and compared with the GGUF
                auto fl = std::make_shared<std::vector<Fill>>(fills);
                verify_all_slots = [&, fl, blob_cap](const char* when) {
                    sycl::queue& vq = dpct::get_in_order_queue();
                    vq.wait();
                    std::vector<uint8_t> want_b(blob_cap), got_b(blob_cap);
                    int64_t bad = 0, first_bad = -1;
                    for (size_t i = 0; i < fl->size(); ++i) {
                        const Fill& f = (*fl)[i];
                        const size_t nb = (size_t) lay.blob_bytes(f.l);
                        if (!gguf_src.read_into(f.l, f.e, want_b.data(), blob_cap)) { ++bad; continue; }
                        vq.memcpy(got_b.data(), xcache.device_slot(f.slot), nb).wait();
                        if (std::memcmp(want_b.data(), got_b.data(), nb) != 0) {
                            if (first_bad < 0) first_bad = (int64_t) i;
                            ++bad;
                            size_t nd = 0, fd = nb, ld = 0;
                            for (size_t x = 0; x < nb; ++x)
                                if (want_b[x] != got_b[x]) { ++nd; if (fd == nb) fd = x; ld = x; }
                            if (std::getenv("STRATA_VERIFY_FIND") != nullptr && bad <= 2 && fd + 4160 < nb) {   // whose blob is the wrong data?
                                const uint8_t* probe = got_b.data() + fd + 4096;
                                std::vector<uint8_t> other(blob_cap);
                                for (size_t j = 0; j < fl->size(); ++j) {
                                    const Fill& g2 = (*fl)[j];
                                    if (!gguf_src.read_into(g2.l, g2.e, other.data(), blob_cap)) continue;
                                    const size_t nb2 = (size_t) lay.blob_bytes(g2.l);
                                    const void* hit = memmem(other.data(), nb2, probe, 64);
                                    if (hit != nullptr)
                                        std::fprintf(stderr, "    the wrong data at byte %zu is expert (layer %d, expert %d), slot %d (fill %zu), byte %zu of its blob\n",
                                                     fd + 4096, (int) g2.l, (int) g2.e, (int) g2.slot, j, (size_t) ((const uint8_t*) hit - other.data()));
                                }
                            }
                            if (const char* dd = std::getenv("STRATA_VERIFY_DUMP"); dd != nullptr && bad == 1) {
                                const std::string base = std::string(dd) + "/slot" + std::to_string(f.slot);
                                if (FILE* fo = std::fopen((base + ".got").c_str(), "wb")) { std::fwrite(got_b.data(), 1, nb, fo); std::fclose(fo); }
                                if (FILE* fo = std::fopen((base + ".want").c_str(), "wb")) { std::fwrite(want_b.data(), 1, nb, fo); std::fclose(fo); }
                            }
                            std::fprintf(stderr, "strata generate: STRATA_VERIFY_ALL_SLOTS (%s): fill %zu (layer %d expert %d) slot %d at %p: %zu of %zu bytes differ, first at %zu, last at %zu\n",
                                         when, i, (int) f.l, (int) f.e, (int) f.slot, (const void*) xcache.device_slot(f.slot), nd, nb, fd, ld);
                        }
                    }
                    std::fprintf(stderr, "strata generate: STRATA_VERIFY_ALL_SLOTS (%s): %lld of %zu slots differ from the GGUF (first at fill %lld)\n",
                                 when, (long long) bad, fl->size(), (long long) first_bad);
                };
                verify_all_slots("after the fill");
            }
        }
        // #286: an unbuffered file tier reads the pairs in batches of 64, the next batch while this one is copied
        std::future<void> ahead;
        auto read_batch = [&](int64_t at) { src.prefetch_pairs(profile.data() + at, std::min<int64_t>(64, want - at)); };
        // mapped reads: advise the next `fill_ahead` pairs so their reads overlap; STRATA_FILL_AHEAD=0 turns it off
        int64_t fill_ahead = 0;
        if (!piped && !per_layer && srcp == &src && !src.unbuffered()) {
            const char* v = std::getenv("STRATA_FILL_AHEAD");
            fill_ahead = v == nullptr ? 256 : std::max(0, std::atoi(v));
            if (fill_ahead > 0 && !src.advise_pairs(profile.data(), std::min<int64_t>(fill_ahead, want))) fill_ahead = 0;
            std::fprintf(stderr, "strata generate: the profile fill asks for %lld pairs ahead (STRATA_FILL_AHEAD)\n",
                         (long long) fill_ahead);
        }
        for (int64_t i = 0; i < want && !piped; ++i) {
            if (!per_layer && srcp == &src && src.unbuffered() && i % 64 == 0) {
                if (ahead.valid()) ahead.get();
                else read_batch(i);
                if (i + 64 < want) ahead = std::async(std::launch::async, read_batch, i + 64);
            }
            if (fill_ahead > 0 && i + fill_ahead < want) (void) src.advise_pairs(profile.data() + i + fill_ahead, 1);
            const int32_t slot = xcache.admit(profile[(size_t) i].first, profile[(size_t) i].second);
            if (slot == strata::core::kNotResident) {
                if (per_layer) continue;
                break;
            }
            const uint8_t* b = srcp->blob(profile[(size_t) i].first, profile[(size_t) i].second);
            if (b == nullptr || !xcache.fill_slot_blocking(slot, b, err,
                    (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first))) {
                std::fprintf(stderr, "strata generate: the profile fill failed at pair %lld: %s\n",
                             (long long) i, err.c_str());
                return 1;
            }
            ++prefilled;
            fill_bytes += (uint64_t) strata::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first);
        }
        const double fill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tfill).count();
        std::fprintf(stderr, "strata generate: profile fill: %.2f GiB in %.1f s (%.2f GB/s, %s)\n",
                     (double) fill_bytes / 1073741824.0, fill_s, fill_s > 0 ? (double) fill_bytes / 1e9 / fill_s : 0.0,
                     piped ? "pipelined" : "serial");
        // **AND ONE SLOT IS READ BACK AND COMPARED.**  A residency table that is right about indices and wrong
        // about bytes produces a plausible token, which is this project's most expensive failure mode; the
        // cache's own `verify_slot` is the check and it costs one 1.38 MB D2H at startup.
        if (prefilled > 0 && !xcache.verify_slot(xcache.slot_of(profile[0].first, profile[0].second),
                                srcp->blob(profile[0].first, profile[0].second), err,
                                (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(profile[0].first))) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        mem_mark("the profile fill");
        std::fprintf(stderr, "strata generate: pre-filled %lld of %lld slots from the profile in %.1f s (%.0f MB/s); "
                             "slot 0 verified\n",
                     (long long) prefilled, (long long) (per_layer ? xcache.slots() : want), fill_s,
                     fill_s > 0 ? (double) fill_bytes / 1e6 / fill_s : 0.0);
    }
    // SYCL port, plan item 2: the experts that did not fit VRAM, mirrored once in pinned host memory the GPU reads
    // over PCIe (--stream-experts has no host copy otherwise: each routed miss was an SSD read). The share of misses
    // the GPU takes is --pcie-frac; STRATA_MIRROR_MIB caps the mirror (default: MemAvailable less 4 GiB), 0 = off.
    unsigned long long* mirror_table_d = nullptr;   // [n_layers][n_expert] device-readable mirror addresses (0 = none)
    int64_t unmirrored_misses = 0;
    if (o.stream_experts && srcp == &gguf_src && o.expert_cache > 0) {
        // #1054, #1440: with a layer split the mirror holds only the first GPU's layers. The pinned memory belongs to
        // this GPU's context, and the later stages' caches do not exist yet (every later-stage expert would count as a
        // miss: 20 GiB of RAM for experts the other card then holds, or one 39 GiB pinned allocation that fails).
        const int64_t mirror_end = multi_gpu && !split_at.empty() ? split_at[0] : g.n_layers;
        std::vector<std::pair<int64_t, int64_t>> miss;
        size_t n_true_miss = 0;   // the experts no slot holds; what follows them in `miss` are the lend candidates (below)
        for (const auto& pr : profile)   // the profile's order: the most-routed misses first, if the cap is reached
            if (pr.first < mirror_end && xcache.slot_of(pr.first, pr.second) == strata::core::kNotResident)
                miss.push_back({pr.first, pr.second});
        for (int64_t l = 0; l < mirror_end; ++l)                     // pairs the profile does not list at all
            for (int64_t e = 0; e < g.n_expert; ++e)
                if (xcache.slot_of(l, e) == strata::core::kNotResident &&
                    std::find(miss.begin(), miss.end(), std::pair<int64_t, int64_t>{l, e}) == miss.end())
                    miss.push_back({l, e});
        n_true_miss = miss.size();
        // 0.1.43: the experts in the last slots - the ones a prompt's loan takes (Prefill::relayout lends the TAIL of the
        // cache) - are mirrored too, after the real misses (the cap drops them first).  A lent expert is streamed during
        // the prompt and put back after it; from the GGUF that was a pread at ~2.3 GB/s per expert (~1 s for a 4K prompt's
        // 930 slots, 0.26 s for a 600-token prompt's 284), from the pinned mirror it is a DMA.  Their mirror entries are never
        // read by a decode window: it only looks at a mirror address for an expert that has no slot.
        // STRATA_LEND_MIRROR=0 turns it off; =N: the last N slots (default 10% of the cache with a loan, at most 1,200).
        if (!o.no_prefill_borrow && o.prefill_chunk > 0 && !multi_gpu) {
            const char* lm = std::getenv("STRATA_LEND_MIRROR");
            const int64_t want_tail = lm ? std::atoll(lm) : std::min<int64_t>(xcache.slots() / 10, 1200);
            if (want_tail > 0) {
                const int64_t first_tail = xcache.slots() - std::min<int64_t>(want_tail, xcache.slots());
                for (int64_t l = 0; l < mirror_end; ++l)
                    for (int64_t e = 0; e < g.n_expert; ++e)
                        if (const int64_t sl = xcache.slot_of(l, e); sl != strata::core::kNotResident && sl >= first_tail)
                            miss.push_back({l, e});
            }
        }
        uint64_t avail = 0;
        if (FILE* f = std::fopen("/proc/meminfo", "r")) {
            char key[64]; unsigned long long kb = 0;
            while (std::fscanf(f, "%63s %llu kB", key, &kb) == 2)
                if (std::strcmp(key, "MemAvailable:") == 0) { avail = kb << 10; break; }
            std::fclose(f);
        }
        const char* mv = std::getenv("STRATA_MIRROR_MIB");
        const uint64_t cap = mv ? (uint64_t) std::atoll(mv) << 20 : (avail > (4ull << 30) ? avail - (4ull << 30) : 0);
        if (!miss.empty() && cap > 0) {
            const auto tm = Clock::now();
            const int64_t got = gguf_src.mirror(miss, cap, 8, err);
            if (got < 0) {
                std::fprintf(stderr, "strata generate: mirroring the experts missing from VRAM: %s (they stay on the SSD)\n",
                             err.c_str());
                err.clear();
            } else {
                std::fprintf(stderr, "strata generate: %lld of %zu experts missing from VRAM (+ %zu in the prompt loan's slots) mirrored in pinned host memory "
                                     "(%.2f GiB, %.1f s); the GPU reads them over PCIe\n", (long long) got, n_true_miss, miss.size() - n_true_miss,
                             (double) gguf_src.mirrored_bytes() / 1073741824.0,
                             std::chrono::duration<double>(Clock::now() - tm).count());
                // the device-built verify plan's view of it: each (layer, expert) -> its mirror address, 0 = none
                std::vector<unsigned long long> tab((size_t) (g.n_layers * g.n_expert), 0ull);
                for (int64_t l = 0; l < g.n_layers; ++l)
                    for (int64_t e = 0; e < g.n_expert; ++e)
                        if (gguf_src.pinned(l, e))
                            tab[(size_t) (l * g.n_expert + e)] = (unsigned long long) gguf_src.device_alias(l, e);
                mirror_table_d = sycl::malloc_device<unsigned long long>(tab.size(), dpct::get_in_order_queue());
                dpct::get_in_order_queue().memcpy(mirror_table_d, tab.data(), tab.size() * sizeof(unsigned long long)).wait();
            }
        }
        unmirrored_misses = (int64_t) n_true_miss - (int64_t) (gguf_src.mirrored_bytes() ? std::count_if(miss.begin(), miss.begin() + (ptrdiff_t) n_true_miss,
            [&](const std::pair<int64_t, int64_t>& pr) { return gguf_src.pinned(pr.first, pr.second); }) : 0);
        const bool no_host = [] { const char* v = std::getenv("STRATA_VERIFY_NO_HOST"); return v && *v && std::strcmp(v, "0") != 0; }();
        if (unmirrored_misses > 0 && no_host) {
            std::fprintf(stderr, "strata generate: REFUSED: %lld experts are neither in VRAM nor mirrored; with STRATA_VERIFY_NO_HOST "
                                 "the device plan cannot run them and generation would lack a safe host fallback - raise "
                                 "STRATA_MIRROR_MIB or the free RAM, or lower --max-context\n", (long long) unmirrored_misses);
            refuse_exit(2);
        }
        // The same rule after the start: the mirror holds the start's misses only, so an expert an adaptive swap moves
        // out of VRAM would be in neither, and the device plan would wait for a host that never serves it (every token
        // 0 after a --batch window had counted misses and the tier swapped).
        if (!miss.empty() && no_host && o.adapt_every > 0 && o.adapt_swaps > 0) {
            std::fprintf(stderr, "strata generate: adaptive swaps off: with STRATA_VERIFY_NO_HOST an expert they move out of "
                                 "VRAM would be neither there nor mirrored\n");
            o.adapt_every = 0;
        }
    }

    for (auto& stp : stages) {
        GpuStage& st = *stp;
        const auto& lay = strata::kernels::cpu::expert_layout();
        // the drafter and the head are already allocated by now (they load above, before this), so what is left
        // to hold back is the windows - and `free_b` has already lost the drafter.
        const int64_t room = stage_room(st.dev, true, false);
        const strata::core::OnDevice on(st.dev);
        {
            size_t fb = 0, tb = 0;
            /*
            DPCT1106: 'cudaMemGetInfo' was migrated with the Intel
            extensions for device information which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            dpct::get_current_device().get_memory_info(fb, tb);
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: %.2f GiB free of %.2f, room for experts %.2f GiB\n",
                         st.dev, (double) fb / 1073741824.0, (double) tb / 1073741824.0, (double) room / 1073741824.0);
        }
        std::vector<int64_t> sized;
        int64_t used = 0;
        for (const auto& pr : st.profile) {
            const int64_t b = native_pack ? ((int64_t) lay.blob_bytes(pr.first) + 255) / 256 * 256 : (int64_t) lay.max_blob;
            if (used + b > room) break;
            used += b;
            sized.push_back((int64_t) lay.blob_bytes(pr.first));
        }
        // #841: a stage's cache can fail to open with the VRAM free (the card's allocation is also charged to the
        // Windows commit, or the free figure ran high): it is tried again with 90% of the slots, up to 3 times, each
        // try said.  Nothing changes when the first open works.
        bool stage_open = false;
        for (int attempt = 0; !sized.empty(); ++attempt) {
            stage_open = native_pack ? st.cache.open_sized(sized, g.n_layers, g.n_expert, err)
                                     : st.cache.open((int64_t) sized.size(), g.n_layers, g.n_expert, (int64_t) lay.max_blob, err);
            if (stage_open || attempt >= 3) break;
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not
            use the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            (void)0;
            const size_t keep = sized.size() * 9 / 10;
            std::fprintf(stderr, "strata generate: layer split, CUDA%d expert cache: %s; trying %zu of %zu slots "
                                 "(try %d of 3)\n", st.dev, err.c_str(), keep, sized.size(), attempt + 1);
            if (keep == 0) break;
            st.cache.close();
            sized.resize(keep);
        }
        if (!stage_open) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d expert cache: %s\n", st.dev,
                         sized.empty() ? "no room" : err.c_str());
            return 1;
        }
        int64_t filled = 0;
        for (const auto& pr : st.profile) {
            if (filled >= st.cache.slots()) break;
            const int32_t slot = st.cache.admit(pr.first, pr.second);
            if (slot == strata::core::kNotResident) break;
            const uint8_t* b = srcp->blob(pr.first, pr.second);
            if (b == nullptr || !st.cache.fill_slot_blocking(slot, b, err, (int64_t) lay.blob_bytes(pr.first))) {
                std::fprintf(stderr, "strata generate: layer split, CUDA%d profile fill failed at pair %lld: %s\n",
                             st.dev, (long long) filled, err.c_str());
                return 1;
            }
            ++filled;
        }
        if (filled == 0 || !st.cache.verify_slot(st.cache.slot_of(st.profile[0].first, st.profile[0].second),
                                                 srcp->blob(st.profile[0].first, st.profile[0].second), err,
                                                 (int64_t) lay.blob_bytes(st.profile[0].first))) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d expert cache: %s\n", st.dev,
                         filled == 0 ? "nothing filled" : err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: layer split: CUDA%d runs layers %lld-%lld, expert cache %lld slots "
                             "(%.2f GiB), %lld of its %zu profiled pairs; slot 0 verified\n",
                     st.dev, (long long) st.lb, (long long) (st.le - 1), (long long) st.cache.slots(), st.cache.gib(),
                     (long long) filled, st.profile.size());
    }
    if (multi_gpu)
        std::fprintf(stderr, "strata generate: layer split: CUDA0 runs layers 0-%lld\n", (long long) (split_at[0] - 1));
    // #1980: the host mirror and the "REFUSED" count above cover the first stage only. A later stage whose cache does not
    // hold all of its layers' experts has no mirror entry for the rest, and with STRATA_VERIFY_NO_HOST no host answers for
    // them: the device plan reads a bad address and the card is lost at the first verify window (2x Arc Pro B60, xe CAT
    // fault). Refuse at the start with the same message, and say how to get a complete load.
    if (multi_gpu) {
        const bool no_host_split = [] { const char* v = std::getenv("STRATA_VERIFY_NO_HOST"); return v && *v && std::strcmp(v, "0") != 0; }();
        int64_t later_misses = 0;
        for (auto& stp : stages) {
            const GpuStage& st = *stp;
            later_misses += std::max<int64_t>(0, (st.le - st.lb) * g.n_expert - (int64_t) st.cache.slots());
        }
        if (later_misses > 0 && no_host_split) {
            std::fprintf(stderr, "strata generate: REFUSED: %lld experts of the later layer-split stages are neither in VRAM nor mirrored "
                                 "(the host mirror covers the first stage only); with STRATA_VERIFY_NO_HOST the device plan cannot run them "
                                 "and the card is lost at the first verify window - set STRATA_STAGE_TRIM=1 (with --layer-split N) to free "
                                 "each card's duplicate dense weights, lower --max-context or --vram-reserve-mib, or unset "
                                 "STRATA_VERIFY_NO_HOST\n", (long long) later_misses);
            refuse_exit(2);
        }
        if (later_misses > 0)
            std::fprintf(stderr, "strata generate: layer split: %lld experts of the later stages are not in VRAM (served by the host)\n",
                         (long long) later_misses);
    }

    std::array<strata::core::RemoteExperts, 3> remote_experts;
    std::unique_ptr<strata::core::RemoteExpertOpt> remote_opt;
    if (o.remote_expert_opt && o.expert_cache_remote[0] > 0) remote_opt = std::make_unique<strata::core::RemoteExpertOpt>();
    const bool multi_remote = o.expert_cache_remote[1] > 0 || o.expert_cache_remote[2] > 0;
    if (o.expert_cache_remote[0] > 0) {
        if (o.expert_cache <= 0 || profile.empty() || o.no_pool) {
            std::fprintf(stderr, "strata generate: remote experts need --expert-profile, "
                                 "a CUDA0 expert cache and the expert pool\n");
            return 2;
        }
        std::vector<std::pair<int32_t, int32_t>> ranked = profile;
        if (!stages.empty()) {   // a layer split: CUDA0's share of the profile, then the later stages' pairs no cache holds
            for (auto& st : stages)
                for (const auto& pr : st->profile)
                    if (st->cache.slot_of(pr.first, pr.second) < 0) ranked.push_back(pr);
        }
        if (multi_remote || std::any_of(o.expert_cache_remote_auto.begin(), o.expert_cache_remote_auto.end(),
                                        [](bool automatic) { return automatic; })) {
            // The shipped frequency profile names only 8000 of 24576 experts. Once exhausted,
            // fill remaining VRAM from unranked pairs in expert-then-layer order: this spreads
            // the tail across all layers instead of concentrating it on layer zero.
            std::vector<uint8_t> seen((size_t) g.n_layers * (size_t) g.n_expert, 0);
            for (const auto& pair : ranked)
                if (pair.first >= 0 && pair.first < g.n_layers && pair.second >= 0 && pair.second < g.n_expert)
                    seen[(size_t) pair.first * (size_t) g.n_expert + (size_t) pair.second] = 1;
            for (int64_t e = 0; e < g.n_expert; ++e)
                for (int64_t l = 0; l < g.n_layers; ++l)
                    if (!seen[(size_t) l * (size_t) g.n_expert + (size_t) e])
                        ranked.emplace_back((int32_t) l, (int32_t) e);
            std::fprintf(stderr, "strata generate: remote ranking: %zu profiled pairs, "
                                 "%zu other pairs to fill CUDA1..3\n", profile.size(), ranked.size() - profile.size());
        }
        std::array<std::vector<std::pair<int32_t, int32_t>>, 3> by_device;
        if (multi_remote) {
            // Either stripe experts for parallel GPU work, or give each layer one
            // secondary GPU to reduce switches and transfers over shared USB4.
            std::vector<uint8_t> assigned((size_t) g.n_layers * (size_t) g.n_expert, 0);
            const int devices = 1 + (o.expert_cache_remote[1] > 0) + (o.expert_cache_remote[2] > 0);
            int next = 0;
            for (const auto& pair : ranked) {
                if (pair.first < 0 || pair.first >= g.n_layers || pair.second < 0 || pair.second >= g.n_expert ||
                    xcache.slot_of(pair.first, pair.second) >= 0) continue;
                const size_t index = (size_t) pair.first * (size_t) g.n_expert + (size_t) pair.second;
                if (assigned[index]) continue;
                int target = -1;
                if (o.expert_cache_remote_placement == "layer") {
                    target = pair.first % devices;
                    // Other layers' owners may still have room: keep scanning ranks.
                    if (by_device[(size_t) target].size() >=
                        (size_t) o.expert_cache_remote[(size_t) target]) continue;
                } else {
                    for (int i = 0; i < devices; ++i) {
                        const int r = (next + i) % devices;
                        if (by_device[(size_t) r].size() < (size_t) o.expert_cache_remote[(size_t) r]) {
                            target = r;
                            break;
                        }
                    }
                    if (target < 0) break;
                }
                assigned[index] = 1;
                by_device[(size_t) target].push_back(pair);
                next = (target + 1) % devices;
            }
            std::fprintf(stderr, "strata generate: remote ranks %s across %d CUDA devices\n",
                         o.expert_cache_remote_placement == "layer" ? "grouped by layer" : "striped", devices);
        } else {
            by_device[0] = std::move(ranked);
        }
        std::vector<uint8_t> claimed((size_t) g.n_layers * (size_t) g.n_expert, 0);
        for (auto& st : stages)   // a layer split: what a stage's cache holds is no helper's
            for (const auto& pr : st->profile)
                if (st->cache.slot_of(pr.first, pr.second) >= 0)
                    claimed[(size_t) pr.first * (size_t) g.n_expert + (size_t) pr.second] = 1;
        for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0) {
            if (remote_opt) remote_opt->attach(remote_experts[(size_t) r]);
            if (!remote_experts[(size_t) r].open(remote_dev[r], o.expert_cache_remote[(size_t) r],
                     g.n_layers, g.n_expert, by_device[(size_t) r], xcache, *srcp, claimed, err,
                     o.expert_cache_remote_auto[(size_t) r])) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "strata generate: CUDA%d: %lld additional experts, %.2f GiB; "
                                 "results return through pinned host rows\n", remote_dev[r],
                         (long long) remote_experts[(size_t) r].resident(), remote_experts[(size_t) r].gib());
        }
    }

    // ---- Multi-GPU: the second GPU's expert tier, filled with the ranked pairs the primary does not hold
    strata::core::PeerExperts peer;
    if (o.peer_device >= 1) {
        if (profile.empty() || srcp == nullptr || o.expert_cache <= 0) {
            std::fprintf(stderr, "strata generate: --peer-device needs --expert-profile and the expert cache\n");
            return 1;
        }
        const auto tp0 = Clock::now();
        if (!peer.open(o.peer_device, profile, xcache, *srcp, g.n_layers, g.n_expert, o.peer_reserve_mib, o.peer_slots,
                       err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: peer GPU %d: %lld experts, %.2f GiB (filled in %.1f s); with the primary's "
                             "%lld that is %lld of %lld on the GPUs\n", o.peer_device, (long long) peer.resident(), peer.gib(),
                     std::chrono::duration<double>(Clock::now() - tp0).count(), (long long) xcache.slots(),
                     (long long) (peer.resident() + xcache.slots()), (long long) (g.n_layers * g.n_expert));
    }

    if (remote_opt && !remote_opt->init(err)) { std::fprintf(stderr, "strata generate: %s\n", err.c_str()); return 1; }
    Drive drive;
    // #731 (opt-in, STRATA_DISJOINT_ADAPT=1): the adaptive tiers leave an expert a helper GPU holds out of the primary's
    // promotion candidates (it would sit in both caches).  Asked live, from the helper's own cache (RemoteExperts::holds,
    // #854), so an expert the helper's tier swaps in or out later is followed - never a copy taken at load.
    const bool disjoint_adapt = [] { const char* v = std::getenv("STRATA_DISJOINT_ADAPT"); return v != nullptr && std::atoi(v) != 0; }();
    auto helper_holds = [&](int64_t l, int32_t e) -> bool {
        if (!disjoint_adapt) return false;
        for (int r = 0; r < drive.d.remote_count; ++r)
            if (drive.d.remote[r]->holds(l, e)) return true;
        return false;
    };
    for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
        drive.d.remote[drive.d.remote_count++] = &remote_experts[(size_t) r];
    drive.d.peer = peer.valid() ? &peer : nullptr;
    drive.d.hit_cpu_order = o.expert_cache_cpu_order;
    drive.d.split_rows = !o.no_split_rows;
    drive.d.pool = &pool;
    drive.d.src = srcp;
    drive.d.n_expert = g.n_expert;
    drive.d.jobs.resize((size_t) K);
    // CS-T: routing-aware prefetch of the file tier (the GGUF in place): the next layer's router on this layer's MoE
    // input predicts its experts and their pages are warmed meanwhile.  It only warms pages; STRATA_LOOKAHEAD=0 is
    // the A/B arm, STRATA_LOOKAHEAD_K the experts per token (default 10).
    strata::core::RouterLookahead lookahead;
    if (srcp == &src && src.warms() && [] { const char* v = std::getenv("STRATA_LOOKAHEAD"); return v == nullptr || std::atoi(v) != 0; }()) {
        std::vector<std::vector<uint16_t>> routers((size_t) g.n_layers);
        bool ok = true;
        for (int64_t l = 0; l < g.n_layers && ok; ++l) {
            const strata::core::WeightRef* w = wt.find("blk." + std::to_string(l) + ".ffn_gate_inp.weight");
            ok = w != nullptr && w->kind == strata::core::WeightKind::Bf16InF32 &&
                 w->bytes == (uint64_t) (g.n_expert * g.n_embd) * 2;
            if (!ok) break;
            routers[(size_t) l].resize((size_t) (g.n_expert * g.n_embd));
            ok = DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                      .memcpy(routers[(size_t)l].data(),
                                              w->data, (size_t)w->bytes)
                                      .wait()) == 0;
        }
        const char* kv = std::getenv("STRATA_LOOKAHEAD_K");
        if (ok && lookahead.start(std::move(routers), g.n_embd, g.n_expert, kv ? std::atoi(kv) : 10, &src, err)) {
            drive.d.lookahead = &lookahead;
            std::fprintf(stderr, "strata generate: routing-aware prefetch of the file tier on (the next layer's router)\n");
        } else {
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not
            use the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            (void)0;
            std::fprintf(stderr, "strata generate: routing-aware prefetch off (%s)\n",
                         ok ? err.c_str() : "the routers are not BF16 in the arena");
            err.clear();
        }
    }
    // ---- R4.2c: THE HIT PATH.  Every one of these is required for `hits_ready()`, which is all-or-nothing on
    // purpose: a half-configured hit path would compute some experts twice and others not at all, and a token
    // built on that is wrong rather than refused.
    void* hit_scratch = nullptr;
    int32_t* d_hit_slot = nullptr;
    int32_t* d_hit_dst = nullptr;
    uint8_t* d_hit_q8 = nullptr;
    float* d_hit_q8_scale = nullptr;   ///< R4.2h: the fp32 activation scales the CPU path also uses
    float* d_hit_out = nullptr;
    if (o.expert_cache > 0 && !o.no_pool) {
        const uint64_t sb = strata::kernels::moe_hit_grouped_scratch_bytes(K, g.n_embd, strata::kernels::cpu::FF);
        if (DPCT_CHECK_ERROR(hit_scratch = (void *)sycl::malloc_device(
                                 (size_t)sb, dpct::get_in_order_queue())) !=
                0 ||
            DPCT_CHECK_ERROR(d_hit_slot = sycl::malloc_device<int32_t>(
                                 (size_t)K, dpct::get_in_order_queue())) != 0 ||
            DPCT_CHECK_ERROR(d_hit_dst = sycl::malloc_device<int32_t>(
                                 (size_t)K, dpct::get_in_order_queue())) != 0 ||
            DPCT_CHECK_ERROR(d_hit_q8 = (uint8_t *)sycl::malloc_device(
                                 (size_t)(g.n_embd / 32) * 34,
                                 dpct::get_in_order_queue())) != 0 ||
            // R4.2h: the fp32 activation scales.  Without this the GPU's hits
            // use the block's fp16 `d` while the CPU's misses use
            // `ActQ::scale`, which is fp32 - a 4.761e-04 relative disagreement
            // on every chunk, and the reason enabling the cache changed the
            // tokens.
            DPCT_CHECK_ERROR(d_hit_q8_scale = sycl::malloc_device<float>(
                                 (size_t)(g.n_embd / 32),
                                 dpct::get_in_order_queue())) != 0 ||
            DPCT_CHECK_ERROR(d_hit_out = (float *)sycl::malloc_device(
                                 (size_t)K * g.n_embd * 4,
                                 dpct::get_in_order_queue())) != 0) {
            std::fprintf(stderr, "strata generate: the R4 hit path could not allocate its device buffers\n");
            return 1;
        }
        drive.d.cache = &xcache;
        drive.d.cache_stream = main_cs;
        drive.d.cache_base = (const uint8_t*) xcache.device_slot(0);
        drive.d.cache_blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        drive.d.cache_slot_off = xcache.slot_offsets();
        drive.d.hit_scratch = hit_scratch;
        drive.d.parts_out = d_parts;
        drive.d.hit_out = d_hit_out;
        drive.d.parts_elems = K * g.n_embd;
        drive.d.mixed = ss.block.mixed;
        drive.d.x_q8_0_hit = d_hit_q8;
        drive.d.x_q8_0_hit_scale = d_hit_q8_scale;
        drive.d.d_slot = d_hit_slot;
        drive.d.d_dst = d_hit_dst;
        drive.d.h_slot.resize((size_t) K);
        dpct::event_ptr hit_done = nullptr;
        if (DPCT_CHECK_ERROR(hit_done = new sycl::event()) != 0) {
            std::fprintf(stderr, "strata generate: the hit path could not create its probe event\n");
            return 1;
        }
        drive.d.hit_done = (void*) hit_done;
        drive.d.hit_poke = !o.no_hit_poke;
        drive.d.h_dst.resize((size_t) K);
        mem_mark("the R4 hit path");
        std::fprintf(stderr, "strata generate: R4 hit path ON - resident experts are computed on the GPU\n");
    }
    // ---- P0.S8: the routing trace.  Only meaningful with the pool running, because the ids arrive through
    // the doorbell that the pool consumes - so `--no-pool` is refused rather than silently producing an empty
    // file that would read as "the router selected nothing".
    // ---- PER-STAGE TIMING.  `--no-capture` only: an event recorded inside a stream capture is silently
    // dropped, so a captured graph cannot carry these events and the numbers would be zeros that read as
    // "every stage is free".  Refusing is the fix.
    if (o.stage_timing) {
        if (!o.no_capture) {
            std::fprintf(stderr, "strata generate: --stage-timing records CUDA events inside the layer path, "
                                 "and an event record inside a stream capture is silently dropped. Pass "
                                 "--no-capture as well.\n");
            return 2;
        }
        if (!strata::core::stage_timing_enable()) {
            std::fprintf(stderr, "strata generate: stage_timing_enable failed\n");
            return 1;
        }
        strata::core::stage_timing_name(0, "gr_read (attn)");
        strata::core::stage_timing_name(1, "attention block");
        strata::core::stage_timing_name(2, "gr_write (attn)");
        strata::core::stage_timing_name(3, "gr_read (ffn)");
        strata::core::stage_timing_name(4, "moe_route");
        strata::core::stage_timing_name(5, "moe_finish");
        strata::core::stage_timing_name(6, "gr_write (ffn)");
        // The GDN block's internals.  It is 36 of the 48 layers, 0.96 ms each, and its entire weight traffic
        // is ~26 MB - so ~0.11 ms at the measured read rate.  ~13 tiny latency-bound launches live in it and
        // a single "attention block" number cannot say which one costs anything.
        strata::core::stage_timing_name(8, "  gdn: quantize x");
        strata::core::stage_timing_name(9, "  gdn: qkv gemv");
        strata::core::stage_timing_name(10, "  gdn: conv+silu");
        strata::core::stage_timing_name(11, "  gdn: l2 norms");
        strata::core::stage_timing_name(12, "  gdn: alpha/beta/gate");
        strata::core::stage_timing_name(13, "  gdn: gdn_step");
        strata::core::stage_timing_name(14, "  gdn: z + out_norm");
        strata::core::stage_timing_name(15, "  gdn: out gemv");
    }
    std::FILE* routing = nullptr;
    if (!o.dump_routing.empty()) {
        if (o.no_pool) {
            std::fprintf(stderr, "strata generate: --dump-routing needs the expert pool; the routed ids reach "
                                 "the host through the doorbell the pool reads. Drop --no-pool.\n");
            return 2;
        }
        routing = std::fopen(o.dump_routing.c_str(), "wb");
        if (routing == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_routing.c_str());
            return 1;
        }
        drive.routing = routing;
    }
    strata::core::PoolFn pool_fn = o.no_pool ? nullptr : &drive_pool;
    // The hit hook rides the same switch as the pool: with no pool there is no `parts` staging to
    // write into, and a hit path with nowhere to write is a wrong token rather than an error.
    strata::core::HitFn hit_fn =
        (o.no_pool || o.expert_cache <= 0) ? nullptr : &strata::core::expert_hit_run;
    void* pool_user = o.no_pool ? nullptr : (void*) &drive;
    std::fprintf(stderr, "strata generate: %d expert-pool workers%s%s\n", pool.workers(),
                 pool.host_works() ? " + the host thread" : "",
                 o.no_pool ? " (UNUSED: --no-pool)" : "");

    // **THE MISALIGNMENT WARNING THAT STOOD HERE IS GONE, BECAUSE THE MISALIGNMENT IS FIXED.**
    //
    // It said the tokens were not the model's, and it was true: `session_loop` handed layer `l`'s expert
    // outputs to layer `l+1`, which multiplied them by layer `l+1`'s router weights (LEDGER L100).  The loop
    // now runs a captured PAIR per layer - `pre[l]` ending with the router and the doorbell, then the CPU
    // pool, then `post[l]` which combines those experts with THAT layer's weights - so layer `l`'s experts meet
    // layer `l`'s routing.  The generated ids changed the moment it landed, which is what a correctness fix
    // looks like from the outside.
    //
    // The cost is real and is recorded rather than hidden: the window for the CPU pool is now whatever GPU
    // work follows the ring inside `pre[l]`, which is the shared expert and nothing else - 0.038 ms against
    // 0.514 ms of CPU work per layer.  A per-layer CPU expert pool cannot be hidden behind a strictly serial
    // residual chain; the CPU term is answered by Phase 3's VRAM expert cache, not by this pipeline.

    // ---- the graphs
    strata::core::SessionGraphs gr;
    if (!o.no_capture && !native_pack) {   // plan v0.3 P6: a native pack runs verify windows only
        // a layer split's CUDA0 session owns only [0, split_at[0]), so its graphs cover that range; the
        // whole-model replay paths (`session_loop`, the plain generate loop) refuse rather than read another
        // stage's state - a split runs its layers on the stages' verifiers (serve) or prefill stage chain
        if (!strata::core::session_capture(wt, g, ss, d_parts, gr, err, /*split=*/o.gpu_stages, 0,
                                           multi_gpu ? split_at[0] : -1)) {
            std::fprintf(stderr, "strata generate: session_capture: %s\n", err.c_str());
            return 1;
        }
    }

    // **`--no-capture` AND THE EXPERTS ARE MUTUALLY EXCLUSIVE, AND SILENTLY SO.**
    //
    // The CPU expert pool is wired into `session_loop` - the host loop around the captured graphs - and
    // `session_token` has no pool hook at all.  So `--no-capture` did not merely change HOW the layers were
    // launched: it ran the whole model with `parts` left at whatever the buffer held, which is ZERO, and the
    // only symptom was `expert blobs 0` in a stats line nobody had to read.  A run that silently omits the
    // routed experts is not a slow measurement of this model, it is a measurement of a different model.
    //
    // Refusing is the fix.  `--no-pool` is the explicit way to say "I want the GPU-only floor".
    if (o.no_capture && !o.no_pool) {
        std::fprintf(stderr,
                     "strata generate: --no-capture runs `session_token`, which has NO CPU expert pool hook, so "
                     "the routed experts would silently contribute nothing. Pass --no-pool as well if the "
                     "GPU-only floor is what you want.\n");
        return 2;
    }
    // The ladder is written by `session_loop`, and `session_token` does not touch the staging buffer at all - so
    // accepting the flag there would produce a file of uninitialised memory, which reads as a wrong answer rather
    // than as a mistake.  `--no-capture` without `--no-pool` is already refused above, so this catches the pair.
    if (o.no_capture && !o.dump_layers.empty()) {
        std::fprintf(stderr,
                     "strata generate: --dump-layers is written by `session_loop`; `--no-capture` runs "
                     "`session_token` instead, which never fills the staging buffer. Drop one of the two.\n");
        return 2;
    }
    if (o.no_capture && !o.dump_halves.empty()) {
        std::fprintf(stderr,
                     "strata generate: --dump-halves is CAPTURED into the layer graphs, so it needs the "
                     "captured path; `--no-capture` never records it. Drop one of the two.\n");
        return 2;
    }

    mem_mark("the expert cache and the graphs");
    std::fprintf(stderr, "strata generate: session is up (engine %s)\n", STRATA_VERSION);
    if (verify_all_slots) verify_all_slots("session up");
    auto run_head = [&](void* stream) -> bool {
        if (!native_head.loaded())
            return strata::core::lm_head(wt, g, ss.block, d_logits, stream, err);
        return strata::core::lm_head_mix(wt, g, ss.block, stream, err) &&
               native_head.run(ss.block.mixed, d_logits, stream, err);
    };
    float* d_emb = nullptr;
    if (DPCT_CHECK_ERROR(
            d_emb = (float *)sycl::malloc_device(
                (size_t)g.n_embd * 4, dpct::get_in_order_queue())) != 0) {
        std::fprintf(stderr, "strata generate: the embedding buffer failed\n");
        return 1;
    }
    // **`sample_tokens` TAKES DEVICE POINTERS.**  It is a kernel launch; `logits` and `out` are both read and
    // written on the device.  Passing `logits.data()` - the host vector - faults inside the kernel and the
    // error surfaces at the NEXT synchronising call, which here was the next token's `embed_row`, reporting an
    // illegal access on a weight plane.  Nothing in the parameter names said device.
    int* d_next = nullptr;
    if (DPCT_CHECK_ERROR(d_next = sycl::malloc_device<int>(
                             1, dpct::get_in_order_queue())) != 0) {
        std::fprintf(stderr, "strata generate: the sampler output buffer failed\n");
        return 1;
    }

    // **`R` IS BOTH THE INPUT AND THE OUTPUT, SO THE NEW TOKEN'S EMBEDDING HAS TO REPLACE THE OLD RESIDUAL.**
    // At `pos == 0` that is `session_zero`, which is the reference's own initial condition - the embedding
    // broadcast to all `hc` streams.  After that `session_zero` would also wipe the recurrence, so the
    // broadcast is done directly.  Getting this wrong is invisible for exactly one token.
    void* token_stream = o.stream_token ? main_cs : nullptr;
    auto put_input = [&](int64_t tok, int64_t pos) -> bool {
        try {
    if (!strata::core::embed_row(wt, g, tok, d_emb, token_stream, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return false;
        }
        if (pos == 0) {
            strata::core::session_zero(ss, g, d_emb, token_stream);
        } else {
            for (int64_t c = 0; c < g.hc; ++c)
                /*
                DPCT1124: cudaMemcpyAsync is migrated to asynchronous
                memcpy API. While the origin API might be synchronous, it
                depends on the type of operand memory, so you may need to call
                wait() on event return by memcpy API to ensure synchronization
                behavior.
                */
                if (DPCT_CHECK_ERROR(strata::q_of(token_stream)->memcpy(
                        ss.R + (size_t)c * g.n_embd, d_emb,
                        (size_t)g.n_embd * 4)) != 0) {
                    std::fprintf(stderr, "strata generate: the residual broadcast failed\n");
                    return false;
                }
        }
        return o.stream_token ||
               DPCT_CHECK_ERROR(
                   dpct::get_current_device().queues_wait_and_throw()) == 0;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    };

    strata::kernels::SamplerParams sp;
    sp.greedy = o.greedy;
    sp.seed = o.seed;
    sp.top_k = o.top_k;
    sp.top_p = o.top_p;
    sp.temperature = o.temperature;
    // what this run actually samples with (the speculative loop below gets the same parameters); serve samples
    // per request instead
    if (!o.serve) {
        if (sp.greedy || sp.temperature <= 0.0f)
            std::fprintf(stderr, "strata generate: sampling greedy\n");
        else
            std::fprintf(stderr, "strata generate: sampling temperature=%g top_k=%d top_p=%g seed=%llu\n",
                         (double) sp.temperature, sp.top_k > 0 && sp.top_k < 64 ? sp.top_k : 64, (double) sp.top_p,
                         (unsigned long long) sp.seed);
    }

    std::FILE* dump = nullptr;
    // The logits header is written WITH THE FIRST ROW, not at open: a native pack never reaches the
    // per-token dump site, and a header promising rows that were never written is worse than no file.
    int32_t hdr[2] = {0, 0};
    bool hdr_written = false;
    const int64_t dump_positions = (int64_t) o.tokens.size() - 1 + o.max_new;
    if (!o.dump_logits.empty()) {
        if (dump_positions > INT32_MAX || n_vocab > INT32_MAX) {
            std::fprintf(stderr, "strata generate: logits dump dimensions exceed int32\n");
            return 2;
        }
        dump = std::fopen(o.dump_logits.c_str(), "wb");
        if (dump == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_logits.c_str());
            return 1;
        }
        // **THE COUNT IS `n_prompt - 1 + max_new`, NOT `n_prompt + max_new`.**  The loop writes one row per
        // position from 0, and it stops once `produced` holds `max_new` tokens - and `produced` only starts
        // receiving at position `n_prompt - 1`.  So a 5-token prompt with `--max-new 6` writes 10 rows, and the
        // header used to claim 11.  A header that describes a different file from the one written is the same
        // class of defect as a self-check that verifies the wrong invariant: anything reading the count instead
        // of the size gets a wrong answer that looks authoritative.  `tools/logits_identical.py` caught it by
        // parsing the header and refusing the file.
        const int32_t n_rows = (int32_t) strata::program::logits_selection::row_count(dump_positions, o.logits_stride);
        hdr[0] = n_vocab; hdr[1] = n_rows;
    }

    // ---- THE C1 ORACLE: ONE RESIDUAL SNAPSHOT PER LAYER PER POSITION, so the engine can be bisected against
    // `llama-debug`'s `l_last-<il>` node instead of against a single end-to-end perplexity.  The buffer is
    // PINNED because `session_loop` enqueues a device-to-host copy into it after every layer and the transfer
    // would otherwise be staged through a pageable bounce buffer on the critical path.
    std::FILE* layer_dump = nullptr;
    float* layer_stage = nullptr;
    const size_t layer_floats = (size_t) (g.n_layers + 1) * (size_t) g.hc * (size_t) g.n_embd;
    if (!o.dump_layers.empty()) {
        layer_dump = std::fopen(o.dump_layers.c_str(), "wb");
        if (layer_dump == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_layers.c_str());
            return 1;
        }
        /*
        DPCT1048: The original value cudaHostAllocDefault is not meaningful
        in the migrated code and was removed or replaced with 0. You may need to
        check the migrated code.
        */
        if (DPCT_CHECK_ERROR(layer_stage = sycl::malloc_host<float>(
                                 layer_floats, dpct::get_in_order_queue())) !=
            0) {
            std::fprintf(stderr, "strata generate: cannot pin the layer-dump staging buffer\n");
            return 1;
        }
    }

    // ---- prefill is the DECODE PATH ONE TOKEN AT A TIME, which `phase-2-correct-engine.md:12-13` says is
    // fine here: "process the prompt through the decode-style graphs in small batches; a 19K-token prompt will
    // take minutes".  A real batched prefill is P2.S6's other half and is not this.
    //
    // **THE LOOP IS `feed -> 48 layers -> head -> sample -> feed`, AND THE FIRST GENERATED TOKEN COMES FROM THE
    // LAST *PROMPT* POSITION.**  The first version sampled only on the decode positions, so `produced` was
    // still empty when the first generated position asked for `produced.back()` - an out-of-bounds read on an
    // empty vector.  Teacher forcing below is what makes the distinction unnecessary to special-case: for every
    // position before the last prompt one, the next input is the PROMPT's next token, and after that it is the
    // sampled one.
    std::vector<int64_t> produced;
    double total_ms = 0;
    double prefill_ms = 0;   // positions 0 .. n_prompt-2: prompt tokens that only condition
    const Clock::time_point t_start = Clock::now();
    double ttft_ms = 0;
    const int64_t n_prompt = (int64_t) o.tokens.size();
    int64_t tok = o.tokens[0];

    // ---- THE PURE-GPU MEASUREMENT.  `session_replay` launches all 48 `pre` graphs back to back on one stream
    // with NO host work between them - no doorbell poll, no pool, no parts copy - so what it times is the GPU
    // executing the layer sequence and nothing else.  It had been declared, defined and never called since the
    // day it was written.
    //
    // **THIS IS THE MEASUREMENT THAT SAYS WHETHER THE ENGINE IS HOST-BOUND OR GPU-BOUND**, and the stage table
    // cannot answer it: those events measure the interval between two marks on a stream, which includes every
    // gap where the GPU sat idle waiting for the host to enqueue the next kernel.  In `--no-capture` those gaps
    // are the host's launch latency and they are proportional to the KERNEL COUNT rather than to any work, so
    // the no-capture stage shares are shares of kernel count - which is why the attention block, with the most
    // kernels, looks like 55% of the token there.
    // ---- R0.9: THE PER-STAGE TABLE ON THE CAPTURED GRAPH.
    if (o.gpu_stages) {
        strata::core::doorbell_reset(db);
        double mix = 0, ffn = 0, post = 0;
        if (!strata::core::session_replay_stages(g, 0, 0, ss, gr, main_cs, mix, ffn, post, err)) {
            // #610: a native (IQ) pack captures no per-layer graphs; the serve path times its window's stages
            std::fprintf(stderr, "strata generate: %s%s\n", err.c_str(),
                         native_pack ? " - a native (IQ) pack has no per-layer graphs to replay. Its decode window's GPU "
                                       "stages are printed per request by the serve path instead: STRATA_VERIFY_PROFILE=1 "
                                       "STRATA_DECODE_TIMING=1 with --serve (docs/DETAILS.md, \"Where a decode window's "
                                       "time goes\")" : "");
            return 1;
        }
        const int reps = 20;
        double t_mix = 0, t_ffn = 0, t_post = 0;
        for (int r = 0; r < reps; ++r) {
            if (!strata::core::session_replay_stages(g, 0, 0, ss, gr, main_cs, mix, ffn, post, err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            t_mix += mix;
            t_ffn += ffn;
            t_post += post;
        }
        const double tot = t_mix + t_ffn + t_post;
        std::printf("\nper-stage GPU time on the CAPTURED graph, one token over %lld layers\n",
                    (long long) g.n_layers);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "mixer (gr_read+attn+gr_write)",
                    t_mix / reps, t_mix / reps / (double) g.n_layers, 100.0 * t_mix / tot);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "ffn front + router",
                    t_ffn / reps, t_ffn / reps / (double) g.n_layers, 100.0 * t_ffn / tot);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "post (moe_finish+gr_write)",
                    t_post / reps, t_post / reps / (double) g.n_layers, 100.0 * t_post / tot);
        std::printf("  %-22s %9.3f ms/token\n", "sum of the three", tot / reps);

        // ---- AND THE MIXER BY LAYER KIND, because 36 of the 48 are GDN and 12 are QSA and a total cannot
        // separate them.  Round 309's uncaptured table put GDN at 10.88 ms for 36 layers against QSA's 4.56 for
        // 12, which would make the recurrence the largest single R3 target - and that table had `moe_finish`
        // wrong by 4x, so the ratio is re-derived here from the captured graph rather than inherited.
        {
            // **ACCUMULATED OVER `reps`, NOT MEASURED ONCE AND THEN DIVIDED.**  The first version called the
            // per-layer replay a single time and printed `gdn / reps`, which reported GDN at 0.480 ms/token
            // against a mixer total of 14.039 - a factor of exactly `reps`, and the tell was that
            // 0.480 + 0.220 = 0.700 = 14.039 / 20.
            std::vector<double> acc((size_t) g.n_layers, 0.0), per;
            double f2 = 0, p2 = 0;
            for (int r = 0; r < reps; ++r) {
                if (!strata::core::session_replay_stages_per_layer(g, 0, 0, ss, gr, main_cs, per, f2, p2, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
                for (int64_t l = 0; l < g.n_layers; ++l) acc[(size_t) l] += per[(size_t) l];
            }
            double gdn = 0, qsa = 0, worst = 0;
            int64_t ng = 0, nq = 0, worst_l = 0;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                const double v = acc[(size_t) l] / reps;
                if (strata::core::is_qsa_layer(g, l)) { qsa += v; ++nq; }
                else { gdn += v; ++ng; }
                if (v > worst) { worst = v; worst_l = l; }
            }
            std::printf("\n  the mixer by layer kind, averaged over %d runs\n", reps);
            std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  (%lld layers)\n", "GDN layers",
                        gdn, ng ? gdn / (double) ng : 0.0, (long long) ng);
            std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  (%lld layers)\n", "QSA layers",
                        qsa, nq ? qsa / (double) nq : 0.0, (long long) nq);
            std::printf("  %-22s layer %lld at %.3f ms\n", "worst mixer layer", (long long) worst_l, worst);
            std::printf("  %-22s %9.3f ms/token (must equal the mixer above)\n", "GDN + QSA", gdn + qsa);
        }

        // ================================ R0.11: THE FIVE STAGES SEPARATELY ================================
        //
        // Prefixes 1..5 are replayed per layer with the residual restored between them, and consecutive
        // differences are the per-stage times.  **THE CHECK IS THAT THE FIVE SUM TO THE THREE-GRAPH TOTAL** -
        // the same independent-restatement test that caught round 320's divide-by-reps bug, and it is the only
        // reason to believe a table built out of differences.
        {
            std::vector<double> acc5(5, 0.0), per, s5;
            for (int r = 0; r < reps; ++r) {
                if (!strata::core::session_replay_stage_prefixes(g, 0, 0, ss, gr, main_cs, s5, per, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
                for (int k = 0; k < 5; ++k) acc5[(size_t) k] += s5[(size_t) k];
            }
            static const char* sn[5] = {"0 gr_read (attn)", "1 attention", "2 gr_write (attn)",
                                        "3 gr_read (ffn)", "4 moe_route (router)"};
            const char* kind[5] = {"GR", "ATTN", "GR", "GR", "ROUTER"};
            double tot5 = 0, gr_ms = 0;
            std::printf("\n  the five stages separately, by differencing prefixes\n");
            std::printf("  %-24s %-8s %10s %10s %8s\n", "stage", "kind", "ms/token", "ms/layer", "share");
            for (int k = 0; k < 5; ++k) {
                const double v = acc5[(size_t) k] / reps;
                tot5 += v;
                if (k != 1 && k != 4) gr_ms += v;
                std::printf("  %-24s %-8s %10.3f %10.4f %7.1f%%\n", sn[k], kind[k], v,
                            v / (double) g.n_layers, 0.0);
            }
            for (int k = 0; k < 5; ++k) {
                const double v = acc5[(size_t) k] / reps;
                (void) v;
            }
            std::printf("  %-24s %-8s %10.3f\n", "sum of the five", "", tot5);
            std::printf("  %-24s %-8s %10.3f   <- R3.3's target is <= 3 ms for all four passes\n",
                        "GR passes (0,2,3)", "GR", gr_ms);
            std::printf("\n  the three-graph total above was %.3f ms/token; the five must account for it.\n",
                        tot / reps);

            // ================================ R3.5c: THE SAME TABLE, A DIFFERENT WAY ================================
            //
            // The differencing table above mixes five graphs per layer, so stage 4's interval carries the launch
            // of the FULL five-stage graph while stage 3's carries a four-stage one.  This sweep launches ONE
            // graph type per layer and nothing else, so that bias cannot exist.  **If the two disagree, the
            // difference IS the bias and this one is right** - and stage 4 is the router, so it is exactly the
            // number that must not be wrong.
            {
                double sweep[6] = {0, 0, 0, 0, 0, 0};
                for (int k = 1; k <= 5; ++k) {
                    double acc = 0, one = 0;
                    for (int r = 0; r < reps; ++r) {
                        if (!strata::core::session_replay_stage_sweep(g, 0, 0, ss, gr, main_cs, k, one, err)) {
                            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                            return 1;
                        }
                        acc += one;
                    }
                    sweep[k] = acc / reps;
                }
                std::printf("\n  the same five stages, by SWEEPING each prefix back to back (no graph switching)\n");
                std::printf("  %-24s %10s %10s %12s\n", "stage", "prefix sweep", "difference", "bias");
                static const char* sn2[5] = {"0 gr_read (attn)", "1 attention", "2 gr_write (attn)",
                                             "3 gr_read (ffn)", "4 moe_route (router)"};
                for (int k = 0; k < 5; ++k) {
                    const double sw = sweep[k + 1] - sweep[k];
                    const double df = acc5[(size_t) k] / reps;
                    std::printf("  %-24s %10.3f %10.3f %11.1f%%\n", sn2[k], sw, df,
                                sw != 0.0 ? 100.0 * (df / sw - 1.0) : 0.0);
                }
                std::printf("  %-24s %10.3f   (full pre, 48 layers)\n", "prefix 5 total", sweep[5]);
            }
        }
        std::printf("\n  compare `--gpu-only-full`, which replays the same work as TWO graphs per layer.  The\n");
        std::printf("  three sum slightly above it because each launch carries the driver's gap.\n");
        strata::core::session_graphs_free(gr);
        strata::core::doorbell_free(db);
        sycl::free(d_next, dpct::get_in_order_queue());
        return 0;
    }

    if (o.graph_only) {
        strata::core::doorbell_reset(db);
        // one warm pass so the first launch does not pay for page mapping
        if (!strata::core::session_replay(g, 0, 0, ss, gr, main_cs, err)) {
            std::fprintf(stderr, "strata generate: session_replay warm: %s\n", err.c_str());
            return 1;
        }
        if (DPCT_CHECK_ERROR(
                dpct::get_current_device().queues_wait_and_throw()) != 0) {
            std::fprintf(stderr, "strata generate: session_replay warm faulted\n");
            return 1;
        }
        const int reps = 20;
        const Clock::time_point t0 = Clock::now();
        for (int r = 0; r < reps; ++r) {
            if (!strata::core::session_replay(g, 0, 0, ss, gr, main_cs, err)) {
                std::fprintf(stderr, "strata generate: session_replay: %s\n", err.c_str());
                return 1;
            }
        }
        if (DPCT_CHECK_ERROR(
                dpct::get_current_device().queues_wait_and_throw()) != 0) {
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not
            use the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            std::fprintf(stderr,
                         "strata generate: session_replay faulted: %s\n",
                         dpct::get_error_string_dummy(0));
            return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / (double) reps;
        std::printf("pre graphs only   %8.2f ms per token over %lld layers  ->  %.2f tok/s of GPU work\n", ms,
                    (long long) g.n_layers, ms > 0 ? 1000.0 / ms : 0.0);
        std::printf("                  %8.3f ms per layer\n", ms / (double) g.n_layers);
        return 0;
    }

    // ---- R0.3: THE TRUE PER-TOKEN GPU FLOOR.
    //
    // `--graph-only` above launches ONLY `gr.execs[l]`, the `pre` graphs.  It omits the 48 `post` graphs - the
    // shared expert, the combine and the second `gr_write` - and the LM head.  Everything this project published
    // as "39.8 ms pure GPU" came from that loop while being described as the whole GPU, which also made the
    // "host's share = 49.4 - 39.8 = 9.6 ms" figure wrong by however much the missing work costs.  See
    // Memory/ERRORS.md A4/A5.
    //
    // This flag is what "pure GPU" has to mean, and it replaces that number everywhere.  No pool runs, so
    // `parts` keeps whatever the buffer holds and the timing is GPU work alone.
    if (o.gpu_only_full) {
        strata::core::doorbell_reset(db);
        const int reps = 20;
        double ms_layers = 0, ms_head = 0;
        for (int r = -1; r < reps; ++r) {          // r == -1 is the warm pass, not counted
            const Clock::time_point t0 = Clock::now();
            if (!strata::core::session_replay_full(g, 0, 0, ss, gr, main_cs, err)) {
                std::fprintf(stderr, "strata generate: session_replay_full: %s\n", err.c_str());
                return 1;
            }
            // The sync is INSIDE the interval on purpose: it is the wait for the GPU, so t1 - t0 is GPU time.
            if (DPCT_CHECK_ERROR(strata::q_of(main_cs)->wait()) != 0) {
                std::fprintf(
                    stderr,
                    "strata generate: gpu-only-full layers faulted: %s\n",
                    /*
                    DPCT1009: SYCL reports errors using exceptions and does
                    not use error codes. Please replace the
                    "get_error_string_dummy(...)" with a real error-handling
                    function.
                    */
                    /*
                    DPCT1010: SYCL uses exceptions to report errors and
                    does not use the error codes. The cudaGetLastError function
                    call was replaced with 0. You need to rewrite this code.
                    */
                    dpct::get_error_string_dummy(0));
                return 1;
            }
            const Clock::time_point t1 = Clock::now();
            if (!run_head(main_cs)) {
                std::fprintf(stderr, "strata generate: gpu-only-full lm_head: %s\n", err.c_str());
                return 1;
            }
            if (DPCT_CHECK_ERROR(strata::q_of(main_cs)->wait()) != 0) {
                std::fprintf(
                    stderr, "strata generate: gpu-only-full head faulted: %s\n",
                    /*
                    DPCT1009: SYCL reports errors using exceptions and does
                    not use error codes. Please replace the
                    "get_error_string_dummy(...)" with a real error-handling
                    function.
                    */
                    /*
                    DPCT1010: SYCL uses exceptions to report errors and
                    does not use the error codes. The cudaGetLastError function
                    call was replaced with 0. You need to rewrite this code.
                    */
                    dpct::get_error_string_dummy(0));
                return 1;
            }
            const Clock::time_point t2 = Clock::now();
            if (r < 0) continue;
            ms_layers += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ms_head += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        ms_layers /= (double) reps;
        ms_head /= (double) reps;
        const double ms = ms_layers + ms_head;
        std::printf("GPU floor pre+post+head %7.2f ms per token  ->  %.2f tok/s of GPU work\n", ms,
                    ms > 0 ? 1000.0 / ms : 0.0);
        std::printf("                  %8.3f ms layers (%lld x pre+post)\n", ms_layers, (long long) g.n_layers);
        std::printf("                  %8.3f ms per layer\n", ms_layers / (double) g.n_layers);
        std::printf("                  %8.3f ms LM head\n", ms_head);
        return 0;
    }

    // ---- **THE TOKEN PATH ALLOCATES NOTHING (P2.T10, review finding H3).**
    //
    // `session_loop` used to allocate its pinned staging buffer, its probe event and its host pin ON EVERY
    // TOKEN, and `cudaFreeHost` at the end of each call implicitly synchronises the device - so every token
    // finished with a device-wide sync nobody asked for.  The scratch is created once here and reused; it also
    // owns the host pin for the whole session rather than taking and releasing it per token.
    strata::core::SessionLoopScratch loop_scratch;
    struct ScratchFree {
        strata::core::SessionLoopScratch* p;
        ~ScratchFree() { if (p != nullptr) p->free(); }
    } scratch_free{&loop_scratch};
    // Initialised unconditionally, including under --no-pool: the loop validates the scratch it is handed, so
    // passing a default-constructed one is an error rather than a fallback.  (It was, and the guard caught it -
    // which is the point of the guard.)  One allocation at setup either way.
    if (!loop_scratch.init((size_t) K * g.n_embd * 4, err)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    // Plan v0.3 P3: the whole token as ONE graph whenever nothing needs a host step between the ring and post[l]
    // (the VRAM expert tier and the per-layer dumps do).  `--no-token-graph` keeps two graphs per layer.
    strata::core::TokenGraph tgraph;
    struct TokenGraphFree {
        strata::core::TokenGraph* p;
        ~TokenGraphFree() { strata::core::token_graph_free(*p); }
    } tgraph_free{&tgraph};
    // Plan v0.3 P4: with a PROFILE-filled cache the residency is static, so the hit decision moves onto the
    // device and the token graph keeps it.  (A cache filled on demand still needs the per-layer host path.)
    std::vector<int32_t> host_res;
    // The residency table is read by kernels on non-blocking streams, which do not wait for the legacy stream a plain
    // cudaMemcpy runs on, and a pageable copy can return before its DMA has landed: each upload waits for its own copy
    // (the current device's legacy stream, stream 0 - nothing else; HIP has no cudaStreamLegacy name, #550).
    auto res_put = [&host_res](int32_t *dst) -> dpct::err0 {
        // SYCL port: the in-order queue's memcpy, waited for: the copy is complete when this returns
        try {
            dpct::get_in_order_queue().memcpy(dst, host_res.data(), host_res.size() * sizeof(int32_t)).wait();
            return 0;
        } catch (sycl::exception const& exc) {
            std::cerr << exc.what() << " (res_put)" << std::endl;
            return 1;
        }
    };
    int32_t* d_res = nullptr;
    int32_t* d_hit_count = nullptr;
    strata::core::TokenHits thits;
    const bool graph_hits = hit_fn != nullptr && !profile.empty() && !o.no_pool;
    if (graph_hits && !o.no_capture && !o.no_token_graph && layer_dump == nullptr && half_dump == nullptr) {
        host_res.assign((size_t) (g.n_layers * g.n_expert), strata::core::kNotResident);
        int64_t resident = 0;
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const int st = multi_gpu ? stage_of(l) : 0;
                const int32_t slot = st > 0 ? stages[(size_t) st - 1]->cache.slot_of(l, e) : xcache.slot_of(l, e);
                host_res[(size_t) (l * g.n_expert + e)] = slot;
                if (slot != strata::core::kNotResident) ++resident;
            }
        if (DPCT_CHECK_ERROR(
                d_res = sycl::malloc_device<int32_t>(
                    host_res.size(), dpct::get_in_order_queue())) != 0 ||
            DPCT_CHECK_ERROR(d_hit_count = sycl::malloc_device<int32_t>(
                                 1, dpct::get_in_order_queue())) != 0 ||
            res_put(d_res) != 0) {
            std::fprintf(stderr, "strata generate: the device residency table could not be staged\n");
            return 1;
        }
        thits.d_res = d_res;
        thits.n_expert = g.n_expert;
        // a file-backed arena (STRATA_ARENA_MMAP): the experts no GPU holds are the ones the CPU pool and the
        // prompt path will read - start reading them now instead of faulting them in 4 KB at a time mid-request
        // ... and the ones a GPU holds are handed back first (STRATA_ARENA_RELEASE=0 keeps them): the fill read the
        // whole arena, and left mapped and referenced it crowds every other allocation into swap
        {
            static const bool keep = std::getenv("STRATA_ARENA_RELEASE") && std::getenv("STRATA_ARENA_RELEASE")[0] == '0';
            rss_probe("before the release");
            uint64_t released = 0;
            if (!keep)
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] != strata::core::kNotResident)
                        released += srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
            if (released > 0)
                std::fprintf(stderr, "strata generate: expert arena: %.2f GiB of VRAM-held experts handed back to the OS\n",
                             (double) released / (1024.0 * 1024.0 * 1024.0));
            rss_probe("after the release");
            int64_t pf = 0;
            for (size_t i = 0; i < host_res.size(); ++i)
                if (host_res[i] == strata::core::kNotResident) {
                    srcp->prefetch((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
                    ++pf;
                }
            (void) pf;
            rss_probe("after the prefetch hints");
        }
        for (auto& st : stages) {   // layer split across GPUs: the same table on every device
            const strata::core::OnDevice on(st->dev);
            if (DPCT_CHECK_ERROR(
                    st->d_res = sycl::malloc_device<int32_t>(
                        host_res.size(), dpct::get_in_order_queue())) != 0 ||
                res_put(st->d_res) != 0) {
                std::fprintf(stderr, "strata generate: layer split: CUDA%d residency table failed\n", st->dev);
                return 1;
            }
        }
        thits.cache_base = drive.d.cache_base;
        thits.blob = drive.d.cache_blob;
        thits.d_slot = drive.d.d_slot;
        thits.d_dst = drive.d.d_dst;
        thits.d_count = d_hit_count;
        thits.x_q8 = drive.d.x_q8_0_hit;
        thits.x_scale = drive.d.x_q8_0_hit_scale;
        thits.scratch = drive.d.hit_scratch;
        thits.hit_out = drive.d.hit_out;
        drive.d.host_res = host_res.data();
        std::fprintf(stderr, "strata generate: token graph hit path: %lld resident experts, decided on the device\n",
                     (long long) resident);
    }
    if (!o.no_capture && !o.no_token_graph && layer_dump == nullptr && half_dump == nullptr &&
        (hit_fn == nullptr || thits.on()) && !native_pack && !multi_gpu) {   // a split's token graph cannot span stages
        if (!strata::core::session_capture_token(wt, g, ss, d_parts, loop_scratch.y_miss, loop_scratch.parts_bytes,
                                                 tgraph, err, thits.on() ? &thits : nullptr)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: token graph captured (48 layers, one launch per token)\n");
    }

    // ================================ WHERE THE HOST TERM GOES, PER TOKEN ================================
    //
    // **`--gpu-only-full` MEASURES THE 48 LAYER GRAPHS AND THE LM HEAD AND NOTHING ELSE.**  It never enters
    // this loop, so it does not run `ple_stage_token`, `embed_row`, the whole-vocabulary logits readback, the
    // NaN scan or the sampler - and `--no-pool --stats` against that floor was being read as "the per-layer
    // round trip costs 12.4 ms" when an unknown part of it is per TOKEN, not per layer.  That is the same error
    // the review catalogued as A4/A5, one level down: a difference between two measurements attributed to a
    // mechanism that neither of them isolates.
    //
    // Six accumulators, because the six have different fixes.  Reported in `--stats` as ms/token.  **The
    // boundary after the layer loop is the one that matters**: without it the head's interval swallows all 48
    // layers and the report reads as "head = 50 ms", which is not a thing that can happen to 1.4 ms of GPU
    // work.  That is not hypothetical - it is what the first version of this printed.
    double ms_ple = 0, ms_embed = 0, ms_layers = 0, ms_head = 0, ms_readback = 0, ms_sample = 0;
    int64_t phase_tokens = 0;

    // ================================ plan v0.3 P8: THE PERSISTENT ENGINE (--serve) ================================
    //
    // The weights, the expert arena and the VRAM tier load once; then requests arrive on stdin, one per line,
    //
    //     GEN <max_new> <id,id,...>
    //
    // and each generated token is written to stdout as `T <id>` as soon as its verify window is done, followed by
    //
    //     DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <stop|length|cancel> <drafts accepted>
    //          <drafts offered> <prompt tokens reused> ... <prompt tokens read>   (see the DONE line below; #471)
    //
    // Before that, `RESUME <n>` (n prompt tokens are not read again), `PP <position> <prompt_tokens> <ms> <tok/s>`
    // after every prompt chunk, and `REUSED <n>` once the prompt is read.  (`ERR <message>` instead when a request
    // cannot run; `STOP` ends the running request at its next step; `QUIT` ends the process.)  A request continues
    // from the live session or the longest conversation checkpoint its prompt starts with (see ConvCheckpoint),
    // otherwise from an empty sequence (`session_zero`); the rest of the prompt goes through the batched prompt path
    // and its last token through the first verify window - the path all three model files share.  Decoding is greedy.
    // The expert-cache slots (from the end of the cache) that hold the prompt path's buffers for a chunk, and the
    // bytes from the first of them to the end.
    // the share of expert bytes the arena could pin (sizes the prompt path's streamed ring and its lend cap)
    if (srcp != nullptr && o.prefill_chunk > 0) {
        uint64_t pinned = 0, total = 0;
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const uint64_t b = lay.blob_bytes(l);
                total += b;
                if (srcp->pinned(l, e)) pinned += b;
            }
        strata::prefill::Prefill::set_pinned_share(total ? (double) pinned / (double) total : 1.0);
        // SYCL port: the pairs outside the VRAM cache decide the prompt path's stream-all walk (prefill.cpp ring_slots)
        int64_t nonres = 0;
        for (const int32_t r : host_res) nonres += r < 0;
        strata::prefill::set_nonresident_share(host_res.empty() ? 1.0 : (double) nonres / (double) host_res.size());
    }
    // bytes -> slots for CUDA0's cache: exact when it knows its per-slot offsets (a native pack's blobs differ
    // per layer), otherwise max_blob each.
    auto slots_from_bytes = [&](uint64_t need) -> int64_t {
        const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        int64_t k = (int64_t) ((need + (uint64_t) blob - 1) / (uint64_t) blob);
        if (xcache.slot_offsets() != nullptr) {   // sized slots: take slots from the end until they hold `need`
            k = 0;
            while (k < xcache.slots() &&
                   (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[xcache.slots() - k]) < need) ++k;
        }
        return k;
    };
    // ...and its inverse, for the ring's room: the bytes the cache's last `k` slots hold.  k can arrive <= 0: the
    // caller's lend budget is `min(slots - 128, pct * slots / 100)` and goes negative on a cache smaller than 128
    // slots, where `slots - k` would index slot_offsets() past its end.  Such a cache lends nothing, so say so.
    auto bytes_from_slots = [&](int64_t k) -> uint64_t {
        if (k <= 0) return 0;
        if (xcache.slot_offsets() != nullptr)
            return (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[xcache.slots() - k]);
        return (uint64_t) k * (uint64_t) strata::kernels::cpu::expert_layout().max_blob;
    };
    auto lend_slots = [&](int64_t c) -> int64_t {
        return slots_from_bytes(strata::prefill::Prefill::bytes_needed(g, ss, c));
    };
    // `lend_bytes` went with the single-cache serve loan: a participant's loan is priced by `part_bytes` from its
    // OWN cache, and the only other user of the old helper was the serve path's own relayout.
    auto request_chunk = [](int64_t tokens, int64_t max_chunk) -> int64_t {
        if (tokens <= 0 || max_chunk <= 0) return 0;
        const int64_t rounded = tokens > std::numeric_limits<int64_t>::max() - 255
                                    ? tokens
                                    : ((tokens + 255) / 256) * 256;
        return std::min(max_chunk, rounded);
    };
    // STRATA_PREFILL_EQUAL=1 (opt-in, PR #693): a segment of `tokens` reads in n = ceil(tokens / max_chunk) chunks of EQUAL
    // size (ceil(tokens / n) rounded up to 256) instead of full chunks and a short last one.  A chunk of
    // stream_all_min_tokens() or more streams every expert the GPU does not hold, whatever its length, so a prompt's cost
    // is its number of such chunks and equal chunks borrow no more slots than that count needs.  A last chunk below
    // stream_all_min_tokens() stays as it is (it moves only the experts its own tokens route to).  Used ONLY where a
    // segment's chunk is chosen (the serve lend, generate's prompt); `request_chunk` is unchanged because
    // STRATA_SPLIT_SMALL_OWN uses it too.  The chunk geometry changes the rounding, so it is off by default.
    static const bool equal_chunks = [] { const char* e = std::getenv("STRATA_PREFILL_EQUAL"); return e != nullptr && e[0] == '1'; }();
    auto equal_chunk = [&](int64_t tokens, int64_t max_chunk) -> int64_t {
        const int64_t plain = request_chunk(tokens, max_chunk);
        if (!equal_chunks || tokens <= 0 || max_chunk <= 0) return plain;
        const int64_t n = tokens / max_chunk + (tokens % max_chunk != 0);
        const int64_t last = tokens - (n - 1) * max_chunk;
        if (n <= 1 || last < strata::prefill::Prefill::stream_all_min_tokens()) return plain;
        const int64_t per = tokens / n + (tokens % n != 0);
        const int64_t rounded = per > std::numeric_limits<int64_t>::max() - 255 ? per : ((per + 255) / 256) * 256;
        return std::min(max_chunk, rounded);
    };
    // The prompt path's chunk and the slots it borrows for its buffers: the requested chunk halved until it fits,
    // or with --prefill auto the largest chunk whose buffers take at most kAutoLendPct % of the slots (a
    // lent slot's expert is streamed during the prompt and refilled after it; measured on a 12 GB card, 32K Q2_0
    // prompt: 4096 791 tok/s, 6144 878, 8192 973 with 69% of the slots lent).  A request lends only what its own
    // prompt needs (Prefill::relayout), so a big chunk costs short prompts nothing.  0 = none fits.
    // at 8192-token chunks nearly every expert streams anyway, so a lent slot costs little: 90% when the
    // copies are DMA from pinned RAM (Q2_0 8192 + a 384-slot ring: 1283 tok/s), 85% when host copies are the
    // limit (lending more only streams more through them).  STRATA_PREFILL_LEND_PCT overrides (tuning).
    // Hoisted out of plan_lend: the serve path's per-stage loans obey the same cap, one participant at a time.
    const int64_t kAutoLendPct = [] {
        const char* v = std::getenv("STRATA_PREFILL_LEND_PCT");
        return v ? (int64_t) std::atoi(v)
                 : (int64_t) (strata::prefill::Prefill::pinned_share() >= 0.9 ? 90 : 85);
    }();
    // The largest chunk --prefill auto may pick: the operator's ceiling (--prefill auto:N) and never past the
    // context, but a bare `auto` always reaches 8192.  32768 and 16384 stay opt-in (#282): a 32K prompt with
    // IQ2_XS (RTX 5090, 64K context) read at 5,624 tok/s in 8192-token chunks and 6,465 in one 32768 chunk (40K
    // -> 18K experts streamed; an NVFP4 pack at 262K: 3,535 -> 5,201).
    const int64_t auto_ceiling = std::max<int64_t>(8192, std::min<int64_t>(o.prefill_auto_max, o.max_context));
    // The auto scan used to walk a fixed list of sizes - 32768, 16384, 8192, 6144, ... - and take the first that
    // fit.  That list is coarse exactly where a rig needs it: this one affords ~8,700 tokens and was handed 8192,
    // and 8704 is not on it.  `bytes_needed` is a sum of (T x positive constant) terms plus a max of such sums,
    // so it rises monotonically with T, and so does every test the scan applies - which makes the largest size
    // that fits a bisection on the 256-token grid the prompt path already works on (`request_chunk` rounds up to
    // it).  log2(32768/256) = 7 probes, against the list's 10, at one `bytes_needed` per probe.
    auto biggest_chunk = [](int64_t ceiling, auto&& ok) -> int64_t {
        if (ceiling < 256) return 0;
        int64_t lo = 1, hi = ceiling / 256, best = 0;   // n = T / 256, and ok() is monotone in n
        while (lo <= hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            if (ok(mid * 256)) { best = mid * 256; lo = mid + 1; } else hi = mid - 1;
        }
        return best;
    };
    auto plan_lend = [&](int64_t& chunk) -> int64_t {
        auto slots_for = lend_slots;
        // 0.1.39's list with its ring (also the size up to which a prompt keeps that ring under #583)
        auto old_rule = [&]() -> int64_t {
            strata::prefill::Prefill::set_ring_budget(0, 0);
            static constexpr int64_t kAutoChunks[] = {32768, 16384, 8192, 6144, 4096, 3072, 2048, 1024, 512, 256};
            for (const int64_t c : kAutoChunks) {
                // above 8192: only when asked for, and only when a prompt of the context can use it
                if (c > 8192 && (c > o.prefill_auto_max || c > o.max_context)) continue;
                const int64_t k = slots_for(c);
                if (k + 128 <= xcache.slots() && k * 100 <= kAutoLendPct * xcache.slots()) return c;
            }
            return 0;
        };
        if (o.prefill_auto && !strata::prefill::Prefill::ring_bytes_enabled()) {   // STRATA_RING_BYTES=0: 0.1.39's list
            const int64_t c = old_rule();
            if (c > 0) { chunk = c; return slots_for(c); }
            return 0;
        }
        if (o.prefill_auto) {   // the default since 0.1.39b (#583)
            // The chunk and the ring are one budget, and the ring is the better buy.  Measured on this 4-way
            // IQ3_S rig (2x RTX 3060 + 2x RTX 5060, CUDA3 lending at its 90% cap) on a 120K prompt: 8960 tokens
            // with the 17-slot ring that leaves reads at 963 tok/s, 8192/130 at 1,008, 7168 with the ring full at
            // its 199-slot byte budget at 1,000, and 5632/199 at 915.  A ring slot is worth ~0.53 tok/s there and
            // a chunk token ~0.05, so the 69 slots between a full ring and 8192's 130 are worth more than the
            // 512 chunk tokens they cost - and once the ring IS full, shrinking the chunk further buys nothing.
            // So: the largest chunk that still leaves the ring full.  Only a rig where no chunk at all can afford
            // one falls back to the old rule, which takes the largest chunk whose ring clears kRingMin (the
            // value at or below which ring_slots() returns STAGE and streaming is off).
            constexpr int64_t kRingMin = 16;
            const int64_t small = old_rule();   // 0.1.39's chunk (and its ring for the prompts that fit it)
            const int64_t ring_max = strata::prefill::Prefill::ring_cap_for(small);
            const int64_t budget = std::min(xcache.slots() - 128, kAutoLendPct * xcache.slots() / 100);
            // The room is a BYTE budget.  A ring slot is max_blob bytes (`carve` lays out one whole blob each),
            // while these cache slots hold their own layer's blob, which is smaller than max_blob unless the cache
            // happens to hold the pack's biggest layer - on the 4-way IQ3_S rig that is 2.15 MiB a slot against a
            // 2.54 MiB max_blob.  Counting the room in slots priced the ring at 0.85x what it costs, `fits` then
            // rejected the whole chunk instead of shrinking the ring, and the scan stopped a step short: measured
            // against chunk 6144/ring 199, 8192 was refused where the same bytes afford a ~87-slot ring.
            const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            const uint64_t avail = bytes_from_slots(budget);
            auto room_of = [&](int64_t t) -> int64_t {
                const uint64_t nonring = strata::prefill::Prefill::bytes_needed_no_ring(g, ss, t);
                return std::min((int64_t) ((avail - std::min(avail, nonring)) / (uint64_t) blob), ring_max);
            };
            // `room_of` is capped at ring_max, so "the ring is full" is exactly room == ring_max, and both that
            // test and the ones below it only get harder as t grows - the bisection stays valid.
            auto scan = [&](int64_t floor) -> int64_t {
                return biggest_chunk(auto_ceiling, [&](int64_t t) -> bool {
                    const int64_t room = room_of(t);
                    if (room < floor) return false;
                    strata::prefill::Prefill::set_ring_budget((int) room, 0);
                    // ring_slots() now returns `room` - unless STRATA_PREFILL_RING overrides it, in which case the
                    // chunk has to fit THAT ring, which is the pre-fix rule and the A/B arm
                    const int64_t k = slots_for(t);
                    return k + 128 <= xcache.slots() && k * 100 <= kAutoLendPct * xcache.slots();
                });
            };
            int64_t c = scan(ring_max);
            if (c == 0 && ring_max < strata::prefill::Prefill::ring_default_slots()) c = scan(kRingMin);
            // a prompt that fits 0.1.39's chunk keeps 0.1.39's ring (one chunk: the smaller ring only slowed it)
            if (small >= c) {   // the scan bought nothing: 0.1.39's choice
                strata::prefill::Prefill::set_ring_budget(0, 0);
                if (small > 0) { chunk = small; return slots_for(small); }
                return 0;
            }
            strata::prefill::Prefill::set_ring_budget((int) room_of(c), small);
            chunk = c;
            return slots_for(c);
        }
        for (int64_t c = chunk; c >= 256; c /= 2) {
            const int64_t k = slots_for(c);
            if (k + 128 <= xcache.slots()) { chunk = c; return k; }
        }
        return 0;
    };
    // ---- THE ELASTIC K/V (--kv-grow; vmm.hpp).  Allocated for the whole --max-context up front, the K/V of a 262K
    // context (3.35 GiB with the drafter's at int8) takes VRAM the expert cache could hold more experts in.  Now the K/V pools hold physical memory only for the cells
    // the requests reach.  When a request needs more, slots just below the prompt path's loan give up their experts
    // (the CPU computes those from RAM, like any miss) and their whole chunks are mapped into the K/V; a later short
    // request hands them back and the slots are refilled with the profile's hottest experts.  Neither side's
    // addresses move, so every captured graph stays valid.  The context is never smaller: --max-context cells always
    // fit, the cache simply holds fewer experts while a long one is live.
    struct KvGrow {
        bool on = false;
        int64_t top = 0, lo = 0;          // slots [lo, top) hold no expert; their whole chunks may be with the K/V
        int64_t floor = 128;              // the cache keeps at least these slots
        int64_t step = 8192;              // the K/V grows in whole steps of cells
        int64_t cells = 0;                // what every K/V pool holds now
        std::vector<strata::core::VmmChunk> spare;   // out of the cache, not (yet) in the K/V
        int64_t grows = 0, trims = 0, fresh = 0, evicted = 0, refilled = 0;
    } kvg;
    auto kvg_start = [&](int64_t top) {
        kvg.on = strata::core::qsa_kv_elastic() && xcache.vmm_range() != nullptr && d_res != nullptr &&
                 !host_res.empty() && srcp != nullptr && top > kvg.floor;
        if (!kvg.on) return;
        kvg.top = kvg.lo = top;
        kvg.cells = strata::core::qsa_kv_elastic_cells();
        if (const char* v = std::getenv("STRATA_KV_GROW_STEP"); v != nullptr && std::atoll(v) > 0) kvg.step = std::atoll(v);
        // tests: the cache keeps this many slots (its size: the K/V grows into new VRAM only)
        if (const char* v = std::getenv("STRATA_KV_GROW_FLOOR"); v != nullptr && std::atoll(v) > 0) kvg.floor = std::atoll(v);
        std::fprintf(stderr, "strata generate: elastic K/V: %lld of %lld cells in VRAM (%.2f of %.2f GiB); the expert "
                             "cache (%lld slots) gives the K/V room below slot %lld as the context grows\n",
                     (long long) kvg.cells, (long long) o.max_context,
                     (double) strata::core::qsa_kv_elastic_mapped_bytes() / 1073741824.0,
                     (double) strata::core::qsa_kv_elastic_full_bytes() / 1073741824.0, (long long) xcache.slots(),
                     (long long) top);
    };
    // Room for `cells` cells (rounded up to a step).  `quiesce` must leave nothing running on the device and land
    // the adaptive tier's swaps (a swap in flight could still be writing a slot given up here).
    auto kvg_ensure = [&](int64_t cells, const std::function<void()>& quiesce) -> bool {
        if (!kvg.on || cells <= kvg.cells) return true;
        const int64_t target = std::min<int64_t>(o.max_context, (cells + kvg.step - 1) / kvg.step * kvg.step);
        const int64_t need = strata::core::qsa_kv_elastic_need(target);
        if (need == 0) { kvg.cells = strata::core::qsa_kv_elastic_cells(); return true; }
        quiesce();
        strata::core::VmmRange& r = *xcache.vmm_range();
        const uint64_t G = strata::core::vmm_granularity();
        std::vector<int32_t> owner;   // slot -> residency index
        // The slots given up are the ones just below the prompt path's loan, and the loan is most of the cache: they
        // hold experts of middling heat.  So an expert there that is hotter than the coldest one in the loan region
        // moves into that one's slot (a device copy), and the coldest is given up instead - the cache then holds what
        // a smaller cache would.  Heat: the adaptive tier's routing counts, then the profile's rank.
        std::vector<int32_t> rank;
        std::vector<std::pair<float, int32_t>> cold;   // the loan region's residents, hottest first
        auto heat = [&](int32_t i) -> float {
            const float u = drive.d.usage.empty() ? 0.0f : drive.d.usage[(size_t) i];
            return u * 1048576.0f - (float) rank[(size_t) i];
        };
        const auto& lay = strata::kernels::cpu::expert_layout();
        int64_t gave = 0, moved = 0;
        auto freeable = [&]() -> int64_t {   // mapped chunks wholly inside [lo, top): a chunk a live slot shares stays
            const int64_t c0 = (int64_t) ((xcache.slot_offset(kvg.lo) + G - 1) / G);
            const int64_t c1 = (int64_t) (xcache.slot_offset(kvg.top) / G);
            int64_t k = 0;
            for (int64_t c = c0; c < c1; ++c) k += r.mapped(c) ? 1 : 0;
            return k;
        };
        while ((int64_t) kvg.spare.size() + freeable() < need && kvg.lo > kvg.floor) {
            if (owner.empty()) {
                owner.assign((size_t) xcache.slots(), -1);
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= 0 && host_res[i] < xcache.slots()) owner[(size_t) host_res[i]] = (int32_t) i;
                rank.assign(host_res.size(), (int32_t) profile.size());
                for (size_t r = 0; r < profile.size(); ++r)
                    rank[(size_t) profile[r].first * (size_t) g.n_expert + (size_t) profile[r].second] = (int32_t) r;
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= kvg.top) cold.emplace_back(heat((int32_t) i), (int32_t) i);
                std::sort(cold.begin(), cold.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            }
            --kvg.lo;
            ++gave;
            if (const int32_t oi = owner[(size_t) kvg.lo]; oi >= 0) {
                const int64_t layer = oi / g.n_expert;
                int32_t vi = -1;
                if (!cold.empty() && cold.back().first < heat(oi)) {
                    vi = cold.back().second;
                    const int32_t vs = host_res[(size_t) vi];
                    if (xcache.slot_offset(vs + 1) - xcache.slot_offset(vs) <
                            lay.blob_bytes(layer) ||
                        DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                            xcache.device_slot(vs),
                            xcache.device_slot((int32_t)kvg.lo),
                            (size_t)lay.blob_bytes(layer)).wait()) != 0)
                        vi = -1;
                    else {
                        cold.pop_back();
                        host_res[(size_t) oi] = vs;
                        host_res[(size_t) vi] = strata::core::kNotResident;
                        ++moved;
                    }
                }
                if (vi < 0) host_res[(size_t) oi] = strata::core::kNotResident;
                ++kvg.evicted;
            }
        }
        // a device-to-device cudaMemcpy does not wait for the copy: the moves still read the slots given up, and a
        // chunk unmapped under a copy faults it (the error then sticks to whatever runs next)
        if (gave > 0 &&
            DPCT_CHECK_ERROR(
                dpct::get_current_device().queues_wait_and_throw()) != 0) {
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not
            use the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            std::fprintf(stderr,
                         "strata: moving experts for the K/V failed: %s\n",
                         dpct::get_error_string_dummy(0));
            return false;
        }
        {
            const int64_t c0 = (int64_t) ((xcache.slot_offset(kvg.lo) + G - 1) / G);
            const int64_t c1 = (int64_t) (xcache.slot_offset(kvg.top) / G);
            for (int64_t c = c0; c < c1; ++c)
                if (r.mapped(c))
                    if (const strata::core::VmmChunk h = r.unmap(c)) kvg.spare.push_back(h);
        }
        if (gave > 0 &&
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                d_res, host_res.data(), host_res.size() * sizeof(int32_t)).wait()) !=
                0) {
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not
            use the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            std::fprintf(stderr,
                         "strata: the residency table upload failed: %s\n",
                         dpct::get_error_string_dummy(0));
            return false;
        }
        int64_t fresh = 0;
        const bool ok = strata::core::qsa_kv_elastic_grow(target, [&]() -> strata::core::VmmChunk {
            if (kvg.spare.empty()) { ++fresh; return 0; }   // the cache is at its floor: new memory
            const strata::core::VmmChunk h = kvg.spare.back();
            kvg.spare.pop_back();
            return h;
        });
        kvg.fresh += fresh;
        kvg.cells = strata::core::qsa_kv_elastic_cells();
        ++kvg.grows;
        std::fprintf(stderr, "strata: K/V grown to %lld cells (%.2f GiB); the expert cache gave %lld slots for it, "
                             "%lld hotter experts moved to colder ones' slots (%lld of %lld slots hold experts)%s\n",
                     (long long) kvg.cells, (double) strata::core::qsa_kv_elastic_mapped_bytes() / 1073741824.0,
                     (long long) gave, (long long) moved, (long long) (kvg.lo + (xcache.slots() - kvg.top)),
                     (long long) xcache.slots(),
                     fresh > 0 ? " - and new VRAM, the cache being at its floor" : "");
        if (!ok)
            std::fprintf(
                stderr, "strata: the K/V could not grow to %lld cells (%s)\n",
                (long long)target,
                /*
                DPCT1009: SYCL reports errors using exceptions and does not
                use error codes. Please replace the
                "get_error_string_dummy(...)" with a real error-handling
                function.
                */
                /*
                DPCT1010: SYCL uses exceptions to report errors and does
                not use the error codes. The cudaGetLastError function call was
                replaced with 0. You need to rewrite this code.
                */
                dpct::get_error_string_dummy(0));
        return ok;
    };
    // A request that needs far fewer cells than the K/V holds gives the rest back: the slots refill with the
    // profile's hottest experts the GPU does not hold.  Run on a quiet device (as kvg_ensure's `quiesce`).
    auto kvg_trim = [&](int64_t cells) -> bool {
        if (!kvg.on) return true;
        const int64_t target = std::max<int64_t>(kvg.step, (cells + kvg.step - 1) / kvg.step * kvg.step);
        if (kvg.cells < target + 2 * kvg.step || kvg.lo >= kvg.top) return true;
        strata::core::qsa_kv_elastic_shrink(target, [&](strata::core::VmmChunk h) { kvg.spare.push_back(h); });
        kvg.cells = strata::core::qsa_kv_elastic_cells();
        strata::core::VmmRange& r = *xcache.vmm_range();
        const uint64_t G = strata::core::vmm_granularity();
        const int64_t c1 = (int64_t) (xcache.slot_offset(kvg.top) / G);
        {   // one run from the lowest unmapped chunk, as long as the chunks handed back last (one access call)
            const int64_t c0 = (int64_t) (xcache.slot_offset(kvg.lo) / G);
            int64_t cend = c0, need = (int64_t) kvg.spare.size();
            while (cend < c1 && need > 0) need -= r.mapped(cend++) ? 0 : 1;
            if (!r.map_range(c0, cend, [&]() -> strata::core::VmmChunk {
                    if (kvg.spare.empty()) return 0;
                    const strata::core::VmmChunk h = kvg.spare.back();
                    kvg.spare.pop_back();
                    return h;
                }))
                std::fprintf(stderr, "strata: the expert cache could not take its VRAM back from the K/V\n");
        }
        const int64_t lo0 = kvg.lo;
        while (kvg.lo < kvg.top) {   // a slot whose chunks are all mapped again holds an expert again
            const int64_t a = (int64_t) (xcache.slot_offset(kvg.lo) / G);
            const int64_t b = (int64_t) ((xcache.slot_offset(kvg.lo + 1) + G - 1) / G);
            bool all = true;
            for (int64_t c = a; c < b && all; ++c) all = r.mapped(c);
            if (!all) break;
            ++kvg.lo;
        }
        // the refills on one stream and one sync (a blocking copy each was a driver round trip per expert)
        const auto& lay = strata::kernels::cpu::expert_layout();
        std::vector<std::pair<size_t, int32_t>> filled;   // (residency index, slot), resident once the copies landed
        size_t pi = 0;
        for (int64_t s = lo0; s < kvg.lo; ++s) {
            const uint64_t room = xcache.slot_offset(s + 1) - xcache.slot_offset(s);
            for (; pi < profile.size(); ++pi) {
                const int32_t l = profile[pi].first, e = profile[pi].second;
                const size_t i = (size_t) l * (size_t) g.n_expert + (size_t) e;
                if (host_res[i] >= 0 || lay.blob_bytes(l) > room) continue;
                const uint8_t* b = srcp->blob(l, e);
                if (b == nullptr) continue;
                /*
                DPCT1124: cudaMemcpyAsync is migrated to asynchronous
                memcpy API. While the origin API might be synchronous, it
                depends on the type of operand memory, so you may need to call
                wait() on event return by memcpy API to ensure synchronization
                behavior.
                */
                dpct::get_in_order_queue().memcpy(
                    xcache.device_slot((int32_t)s), b,
                    (size_t)lay.blob_bytes(l)).wait();
                filled.emplace_back(i, (int32_t) s);
                ++pi;
                break;
            }
        }
        if (DPCT_CHECK_ERROR(
                dpct::get_current_device().queues_wait_and_throw()) != 0) {
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not
            use the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            std::fprintf(stderr,
                         "strata: refilling the expert cache failed: %s\n",
                         dpct::get_error_string_dummy(0));
            return false;
        }
        for (const auto& [i, s] : filled) host_res[i] = s;
        kvg.refilled += (int64_t) filled.size();
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        if (DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                d_res, host_res.data(), host_res.size() * sizeof(int32_t)).wait()) !=
            0)
            return false;
        for (const strata::core::VmmChunk h : kvg.spare) strata::core::vmm_chunk_free(h);   // new ones, if any
        kvg.spare.clear();
        ++kvg.trims;
        std::fprintf(stderr, "strata: K/V trimmed to %lld cells; %lld slots back to the expert cache, refilled from "
                             "the profile\n", (long long) kvg.cells, (long long) (kvg.lo - lo0));
        return true;
    };
    // ---- the resident RAM mode (--resident-experts / --resident-cpu-experts): the experts the GPU cache does not
    // hold are copied from experts.bin into RAM once, so no decode or prompt step reads the file (the plain mmap
    // mode reads them through the OS file cache, which a small-RAM PC keeps giving back to the SSD).  Built here,
    // after the prompt path's lend plan is known: the slots it may lend (the cache's last ones) have their experts
    // streamed during a prompt and copied back after it, so those are kept in RAM too as far as RAM allows.  The
    // bytes are the file's bytes and the placement is the same, so the answers are the plain mmap mode's; the
    // share-of-pinned figure above (which sizes the prompt path) is left as the mmap mode's for the same reason.
    // --adapt-async (opt-in): where the asynchronous adaptive tier does not apply, the blocking tier runs, said once.
    // It exchanges experts between the GPU caches and the resident RAM copy only (the copy is checked once it is
    // built, below).  Not beside the --batch slots (their windows run between a request's prompt chunks), a peer
    // tier or the helper GPUs' tier, which pick their swaps from the same table.  (A pipelined decode loop that
    // adapts beside windows in flight needs the blocking tier too: such a mode belongs on this list.)
    auto adapt_async_off = [&](const char* why) {
        std::fprintf(stderr, "strata generate: --adapt-async 1 is off (%s): the blocking adaptive tier\n", why);
        o.adapt_async = 0;
    };
    if (o.adapt_async) {
        const char* why = !o.serve ? "it needs --serve"
                        : o.adapt_every <= 0 || o.adapt_swaps <= 0 ? "the adaptive tier is off"
                        : !o.resident_cpu_experts ? "it needs the resident RAM mode, --resident-experts"
                        : o.batch > 0 ? "not with --batch slots"
                        : o.pipeline_windows > 0 ? "not with --pipeline-windows (its windows are in flight while the tier adapts)"
                        : peer.valid() ? "not with --peer-device"
                        : remote_opt ? "not with --remote-expert-opt"
                        : nullptr;
        if (why != nullptr) adapt_async_off(why);
    }
    if (o.resident_cpu_experts) {
        int64_t lend_from = -1;
        if (o.prefill_chunk > 0 && !o.no_prefill_borrow && d_res != nullptr && xcache.slots() > 0) {
            int64_t chunk = o.prefill_chunk;
            const int64_t k = plan_lend(chunk);
            if (k > 0) lend_from = xcache.slots() - k;
        }
        // a layer split: the experts every later stage's cache holds are left out of the RAM copy, as CUDA0's are
        // (with them the copy keeps no lend region: pin_cache_complement turns the loan off)
        std::vector<std::pair<int32_t, int32_t>> stage_pairs;
        for (auto& st : stages)
            for (int64_t l = st->lb; l < st->le; ++l)
                for (int64_t e = 0; e < g.n_expert; ++e)
                    if (st->cache.slot_of(l, e) >= 0) stage_pairs.emplace_back((int32_t) l, (int32_t) e);
        const std::vector<std::pair<int32_t, int32_t>>& rank_all = profile_all.empty() ? profile : profile_all;
        bool resident_ok = src.pin_cache_complement(xcache, err, o.resident_pin, stage_pairs, lend_from,
                                                    o.resident_headroom, o.resident_budget, &rank_all);
        std::string whole_err;
        if (!resident_ok && o.resident_soft) {
            // #467: the whole complement does not fit - keep what does, the hottest by the profile, through the #403
            // budget path (sized by available memory) instead of none: the misses outside it read the same file bytes
            // the mmap fallback reads, so the answers are unchanged.  Nothing pinned: the old fallback below.
            whole_err = err;
            resident_ok = src.pin_cache_complement(xcache, err, o.resident_pin, stage_pairs, -1, o.resident_headroom,
                                                   strata::core::FileExpertSource::kResidentWhatFits, &rank_all);
            if (resident_ok)
                std::fprintf(stderr, "strata generate: WARNING: the whole resident RAM mode does not fit (%s); %.2f "
                                     "GiB of the experts the GPU does not hold, the hottest by the expert profile, are "
                                     "kept in RAM and the rest are read from the model folder through the OS file "
                                     "cache\n",
                             whole_err.c_str(), (double) src.resident_bytes() / 1073741824.0);
            else
                err = whole_err + "; " + err;
        }
        if (resident_ok) {
            // --adapt-async: twice the exchange buffers - the second half bounces the copies in whose source is not
            // page-locked (an asynchronous copy instead of the driver's staged, synchronous one)
            if (o.adapt_async && !src.reserve_exchanges(2 * std::min<int64_t>(o.adapt_swaps, 96), err))
                adapt_async_off(err.c_str());
            // pageable buffers would make every copy of a round synchronous (the driver stages them): that is the
            // blocking tier's cost, so the asynchronous tier is not worth running - said, and the blocking tier stays
            if (o.adapt_async && !src.exchange_pinned())
                adapt_async_off("the exchange buffers could not be page-locked");
            if (o.adapt_every > 0 && o.adapt_swaps > 0 &&
                !src.reserve_exchanges(std::min<int64_t>(o.adapt_swaps, 96), err)) {
                std::fprintf(stderr, "strata generate: CPU expert residency: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "strata generate: resident RAM mode: %.2f GiB of experts in RAM (%s), %lld in the GPU "
                                 "cache; adaptive swaps %s\n",
                         (double) src.resident_bytes() / 1073741824.0,
                         src.complement_pinned() ? "page-locked" : src.locked_bytes() > 0 ? "locked" : "pageable",
                         (long long) xcache.resident(),
                         o.adapt_every > 0 && o.adapt_swaps > 0 ? "exchange them with the GPU cache (no file reads)"
                                                                : "off");
        } else if (o.resident_soft) {
            std::fprintf(stderr, "strata generate: WARNING: the resident RAM mode does not fit (%s); the experts the "
                                 "GPU does not hold are read from the model folder through the OS file cache "
                                 "(--mmap-experts), which is slower when the RAM cannot keep them\n", err.c_str());
        } else if (o.resident_budget > 0) {
            // #403: a RAM budget that cannot be kept is not a reason to stop - the experts it would have held are
            // read from the files like the ones outside it (pin_cache_complement leaves nothing half-built)
            std::fprintf(stderr, "strata generate: WARNING: the RAM budget (--resident-budget-gib) cannot be kept (%s); "
                                 "every expert the GPU does not hold is read from the model files %s "
                                 "(--mmap-experts), which is slower\n", err.c_str(),
                         src.unbuffered() ? "with unbuffered I/O" : "through the OS file cache");
        } else {
            std::fprintf(stderr, "strata generate: CPU expert residency: %s\n", err.c_str());
            return 1;
        }
        // #577: the unbuffered choice above was made before the RAM copy existed, from the budget asked for; now the
        // copy is built, decide again from the RAM it really holds and the expert bytes outside it (on a 96 GB PC
        // the file cache keeps those, and every refill after a prompt read the drive instead).  Windows and Linux: the
        // unbuffered reads exist there
#if defined(_WIN32) || defined(__linux__)
        if (o.mmap_experts && o.resident_budget > 0) {
            std::string why;
            const bool was = src.unbuffered();
            const bool ub = src.recheck_unbuffered(why);
            std::fprintf(stderr, "strata generate: the file tier reads %s%s (%s)\n",
                         ub ? "unbuffered" : "through the file cache", ub == was ? "" : " (changed)", why.c_str());
        }
#endif
    }
    if (o.adapt_async && !src.complement_ready()) adapt_async_off("the resident RAM mode is not running");
    if (o.serve) {
        if (o.spec < 2 || o.prefill_chunk <= 0 ||
            (graph_hits && (thits.d_res == nullptr || host_res.empty()))) {
            std::fprintf(stderr, "strata serve: needs --spec T and --prefill CHUNK (and a fillable "
                                 "--expert-cache; the graphed hit path additionally needs --expert-profile P)\n");
            return 2;
        }
        // STRATA_VOLTA_BUILD backport (see docs/ and ninfer-flash-next-v100-backport): --mtp is optional here,
        // mirroring the plain `generate` command's own use_mtp handling below. Without it, drafting falls back
        // to the suffix/prompt-lookup drafter (DraftPolicy still picks Lookup vs plain per round) or, when
        // neither fires, plain one-token-per-round decoding - slower, never wrong: the verify window still
        // confirms every emitted token against the real model regardless of where the draft came from.
        const bool use_mtp = !o.mtp.empty();
        strata::prefill::Prefill sp;
        void* borrow = nullptr;
        uint64_t borrow_bytes = 0;
        int32_t lend_first = -1;          // the first slot the prompt path may borrow (its largest chunk)
        // ---- WHO BORROWS, AND FROM WHOSE CACHE.  One entry per prompt path: CUDA0's (layers [0, split_at[0]),
        // which is the whole model without a split) borrowing the tail of CUDA0's cache, then one per stage
        // borrowing the tail of ITS OWN cache.  A loan is sized by the exact `Prefill::bytes_needed` for the
        // chunk, is laid out by `Prefill::relayout`, and is refilled before any window reads - so outside the
        // prompt the whole cache is expert cache.  THIS IS THE POINT OF THE STRUCT: the loan used to exist only
        // for CUDA0, and `no_prefill_borrow` made every stage instead withhold a chunk-sized reserve from its
        // cache for the entire session, which is what cost the 4-way its context (see the note at the top of the
        // layer-split block).  A stage may only lend the rows for ITS OWN layers: `host_res` is one table whose
        // slot values are indices into whichever cache owns the layer, so a loan that marked rows by slot number
        // alone would hand CUDA0 a slot belonging to another stage's cache.
        struct PfPart {
            strata::core::ExpertCache* cache = nullptr;
            const strata::core::SessionState* ses = nullptr;
            strata::prefill::Prefill* sp = nullptr;
            int dev = -1;                  // -1: leave the device alone (CUDA0)
            int64_t lb = 0, le = 0;        // the layers whose rows this cache holds - the only rows it may lend
            int32_t first = -1;            // the first slot it may lend, for the chunk that was chosen
            int32_t first_now = -1;        // where its buffers are laid out now
            int64_t lent_chunk = 0;
            std::vector<std::pair<int32_t, int32_t>> lent;
        };
        // bytes -> slots for one cache: exact when it knows its per-slot offsets (a native pack's blobs differ
        // per layer), otherwise max_blob each.
        auto cache_slots_for = [&](const strata::core::ExpertCache& xc, uint64_t need) -> int64_t {
            if (xc.slot_offsets() != nullptr) {   // sized slots: from the end until they hold `need`
                int64_t k = 0;
                while (k < xc.slots() &&
                       (uint64_t) (xc.bytes() - (int64_t) xc.slot_offsets()[xc.slots() - k]) < need) ++k;
                return k;
            }
            const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            return (int64_t) ((need + (uint64_t) blob - 1) / (uint64_t) blob);
        };
        auto part_slots = [&](const PfPart& p, int64_t c) -> int64_t {
            return cache_slots_for(*p.cache, strata::prefill::Prefill::bytes_needed(g, *p.ses, c));
        };
        auto part_bytes = [&](const PfPart& p, int32_t first) -> uint64_t {
            strata::core::ExpertCache& xc = *p.cache;
            return xc.slot_offsets() ? (uint64_t) (xc.bytes() - (int64_t) xc.slot_offsets()[first])
                                     : (uint64_t) (xc.slots() - first) *
                                           (uint64_t) strata::kernels::cpu::expert_layout().max_blob;
        };
        // #340: a layer split whose caches already hold most experts streams few of them through the prompt path, so
        // the 384-slot ring (sized for a card that streams nearly every expert of a chunk) only makes every stage's
        // loan bigger: 96 slots (0.1.30's ring here) when >= 75% of the (layer, expert) pairs are resident.  One GPU
        // keeps the pinned-share rule.  STRATA_SPLIT_RING=N: N slots on a split; 0: the pinned-share rule.
        if (multi_gpu && !host_res.empty()) {
            int64_t res_n = 0;
            for (const int32_t r : host_res) res_n += r >= 0;
            const double res_share = (double) res_n / (double) host_res.size();
            const char* v = std::getenv("STRATA_SPLIT_RING");
            const int ring = v ? std::atoi(v) : (res_share >= 0.75 ? 96 : 0);
            if (ring > 0) {
                strata::prefill::Prefill::set_ring_override(ring);
                std::fprintf(stderr, "strata serve: layer split: %.0f%% of the experts resident, the prompt path's "
                                     "streamed ring %d slots\n", 100.0 * res_share, ring);
            }
        }
        std::vector<PfPart> pf_parts;
        // a cache too small to lend the prompt path its buffers would make it allocate them on top - on a card
        // whose cache already filled its reserve, that is the over-subscription the auto sizing avoids - so the
        // chunk is the largest one EVERY participant can lend (a smaller chunk only reads slower)
        if (pf_borrow && d_res != nullptr) {
            pf_parts.push_back({&xcache, &ss, &sp, -1, 0, multi_gpu ? split_at[0] : g.n_layers, -1, -1, 0, {}});
            for (auto& st : stages)
                pf_parts.push_back({&st->cache, &st->ss, &st->sp, st->dev, st->lb, st->le, -1, -1, 0, {}});
            // The two tests plan_lend makes for CUDA0 alone, one participant at a time: a loan must leave the
            // 128-slot floor.  The percentage cap is an AUTO-chunk rule and only the auto scan applies it - an
            // explicit --prefill is the operator's number, and a loan of it only has to fit.  With one participant
            // (no split) this reduces to plan_lend exactly, so the single-GPU loan is unchanged from main.
            auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
                const int64_t k = part_slots(p, c);
                if (k <= 0 || k + 128 > p.cache->slots()) return false;
                return !(cap && k * 100 > kAutoLendPct * p.cache->slots());
            };
            // `only`: CUDA0's cache alone (#448: what one GPU would choose, for the log below); null: every one
            auto fits = [&](int64_t c, bool cap, const PfPart* only = nullptr) -> bool {
                if (only != nullptr) return fits_one(*only, c, cap);
                for (const PfPart& p : pf_parts)
                    if (!fits_one(p, c, cap)) return false;
                return true;
            };
            // The chunk and the ring are one budget.  The ring used to be a fixed 384 slots taken off the top
            // before the chunk was considered: 506 MiB on a Q2_0 pack, where it was tuned, but 1,912 MiB on
            // Q8_0, more than half an 8 GB card's cache - which is what kept that card at a 1024-token chunk.
            // Size the chunk first and hand the ring what the smallest participant leaves over, up to the ring's
            // byte budget - but hand it ALL of it, as in plan_lend: the ring is the better buy per byte (the
            // measurements are there), so the chunk to take is the largest one that still leaves the ring full,
            // not the largest one that leaves it anything at all.  kRingMin is the fallback floor for a rig where
            // no chunk can afford a full ring; ring_slots() returns STAGE at or below it, which turns streaming
            // off.
            constexpr int64_t kRingMin = 16;
            // The room is a BYTE budget, as in plan_lend: a ring slot is max_blob bytes, but a cache slot holds
            // its own layer's blob, which on this rig averages 0.85 of it.  In slots the ring looked 18% cheaper
            // than it is, and a chunk whose ring only fitted after that discount was refused outright - the scan
            // stopped one step short while the chunk it did take had room to spare for the cap-sized ring.
            const int64_t kBlob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            auto ring_room = [&](int64_t c, const PfPart* only, int64_t cap) -> int64_t {
                int64_t room = cap;
                for (const PfPart& p : pf_parts) {
                    if (only != nullptr && &p != only) continue;
                    // clamped at 0: a cache under 128 slots has a negative lend budget, and `slots - budget` is
                    // then `slots`, which is off_[slots] - the end of the prefix sum, so that cache lends 0 bytes
                    // rather than reading one entry past its offsets
                    const int64_t budget = std::max<int64_t>(0, std::min(p.cache->slots() - 128,
                                                                        kAutoLendPct * p.cache->slots() / 100));
                    const uint64_t avail = part_bytes(p, (int32_t) (p.cache->slots() - budget));
                    const uint64_t nonring = strata::prefill::Prefill::bytes_needed_no_ring(g, *p.ses, c);
                    room = std::min(room, (int64_t) ((avail - std::min(avail, nonring)) / (uint64_t) kBlob));
                }
                return room;
            };
            // The largest chunk on the 256-token grid that `only` - or, with null, every stage - can lend with the
            // ring still at its full byte budget (ring_room is capped at ring_max, so that is room == ring_max),
            // under the operator's ceiling (auto_ceiling keeps a bare `auto` at 8192, #282).  Failing that, the
            // largest chunk whose ring clears kRingMin.
            // 0.1.39's list with its ring: the chunk it would pick, and the size up to which a prompt keeps that ring
            auto old_pick = [&](const PfPart* only) -> int64_t {
                strata::prefill::Prefill::set_ring_budget(0, 0);
                static constexpr int64_t kAutoChunks[] = {32768, 16384, 8192, 6144, 4096, 3072, 2048, 1024, 512, 256};
                for (const int64_t c : kAutoChunks) {
                    if (c > 8192 && (c > o.prefill_auto_max || c > o.max_context)) continue;   // #282, as plan_lend
                    if (fits(c, true, only)) return c;
                }
                return 0;
            };
            auto scan = [&](const PfPart* only) -> int64_t {
                if (!o.prefill_auto) {   // an explicit --prefill is the operator's number, and a loan of it only has to fit
                    for (int64_t c = o.prefill_chunk; c >= 256; c /= 2)
                        if (fits(c, false, only)) return c;
                    return 0;
                }
                if (!strata::prefill::Prefill::ring_bytes_enabled()) return old_pick(only);   // STRATA_RING_BYTES=0
                const int64_t small = old_pick(only);
                const int64_t cap = strata::prefill::Prefill::ring_cap_for(small);
                auto probe = [&](int64_t floor) -> int64_t {
                    return biggest_chunk(auto_ceiling, [&](int64_t t) -> bool {
                        const int64_t room = ring_room(t, only, cap);
                        if (room < floor) return false;
                        strata::prefill::Prefill::set_ring_budget((int) room, 0);
                        // ring_slots() now returns `room` - unless STRATA_PREFILL_RING overrides it, in which
                        // case the chunk has to fit THAT ring, which is the pre-fix rule and the A/B arm
                        return fits(t, true, only);
                    });
                };
                int64_t c = probe(cap);
                if (c == 0 && cap < strata::prefill::Prefill::ring_default_slots()) c = probe(kRingMin);
                return std::max(c, small);   // never a smaller chunk than 0.1.39's
            };
            const int64_t chunk = scan(nullptr);
            // #448: a small card in a layer split caps every stage's chunk (an RTX 3080's 512-slot cache held a
            // 32 GB card's split to 512 tokens: prompts 6.2x slower, decode the same).  Named when it bites, so the
            // regression is one log line: each stage that cannot fund the chunk CUDA0 alone would read in.
            const int64_t alone = pf_parts.size() > 1 ? scan(&pf_parts[0]) : chunk;
            // both scans left the ring override on their own last trial: put it back on the chunk that won, which
            // is what `init` lays out and what the INFO line reports
            if (chunk > 0 && o.prefill_auto && strata::prefill::Prefill::ring_bytes_enabled()) {
                // a prompt that fits 0.1.39's chunk keeps 0.1.39's ring (one chunk: the smaller ring only slowed
                // it); the byte-budget ring is for the chunks past it
                const int64_t small = old_pick(nullptr);
                strata::prefill::Prefill::set_ring_budget(
                    small >= chunk ? 0 : (int) ring_room(chunk, nullptr, strata::prefill::Prefill::ring_cap_for(small)),
                    small);
            }
            if (chunk > 0 && alone > chunk) {
                for (size_t i = 1; i < pf_parts.size(); ++i) {
                    const PfPart& p = pf_parts[i];
                    if (fits_one(p, alone, o.prefill_auto)) continue;
                    const int dev = p.dev < 0 ? 0 : p.dev;
                    dpct::device_info prop{};
                    if (DPCT_CHECK_ERROR(
                            dpct::get_device(dev).get_device_info(prop)) != 0) {
                        /*
                        DPCT1010: SYCL uses exceptions to report errors and
                        does not use the error codes. The cudaGetLastError
                        function call was replaced with 0. You need to rewrite
                        this code.
                        */
                        (void)0;
                        prop.get_name()[0] = 0;
                    }
                    const std::string pct =
                        o.prefill_auto ? ", and lend at most " + std::to_string(kAutoLendPct) + "%" : "";
                    std::fprintf(stderr,
                                 "strata serve: WARNING: prompt chunk %lld "
                                 "tokens, not %lld: CUDA%d (%s) "
                                 "has %lld expert-cache slots, and a "
                                 "%lld-token chunk borrows %lld of them "
                                 "(it must keep 128%s) - prompts read slower "
                                 "than on CUDA0 alone (#448)\n",
                                 (long long)chunk, (long long)alone, dev,
                                 prop.get_name(), (long long)p.cache->slots(),
                                 (long long)alone,
                                 (long long)part_slots(p, alone), pct.c_str());
                    // the helper tiers start at CUDA1 without a split (and are enabled in order)
                    std::fprintf(stderr, "strata serve:   a card this small can serve as a helper expert cache "
                                         "instead of a split stage: without --layer-split, with %s "
                                         "(docs/SECOND_GPU.md)\n",
                                 dev == 1 ? "--expert-cache-device1 N" : "--expert-cache-device1..3 N, in order");
                }
            }
            if (chunk > 0) {
                if (o.prefill_auto)
                    // the RESOLVED ring, not the room offered: ring_slots() clamps to [16, RING_MAX] and a chunk
                    // under stream_all_min() gets STAGE, and STRATA_PREFILL_RING overrides both.  This is what
                    // `init` lays out, and what the INFO line reports to the Monitor tab.
                    std::fprintf(stderr, "strata serve: prompt chunk auto: %lld tokens, a %lld-slot ring\n",
                                 (long long) chunk,
                                 (long long) strata::prefill::Prefill::ring_slots_for(chunk));
                else if (chunk != o.prefill_chunk)
                    std::fprintf(stderr, "strata serve: prompt chunk %lld -> %lld tokens so its buffers fit in "
                                         "every expert cache\n", (long long) o.prefill_chunk, (long long) chunk);
                o.prefill_chunk = chunk;
                for (PfPart& p : pf_parts) {
                    p.first = (int32_t) (p.cache->slots() - part_slots(p, chunk));
                    p.first_now = p.first;
                }
                // #340: a split stage whose card still has room for the chunk's buffers (its cache already holds
                // every expert of its layers, so auto stopped short of its VRAM) keeps them as its own instead of
                // borrowing cache slots: nothing of it is lent, streamed during the prompt or refilled after it.
                // Room = the buffers + 1.5 GiB (the verify windows, the draft head, hipBLAS/cuBLAS workspaces made
                // after this).  Layer splits only - one GPU keeps its loan exactly as before.
                // STRATA_SPLIT_OWN_BUFFERS=0: every stage borrows (0.1.30/0.1.31).
                static const bool own_ok = [] {
                    const char* v = std::getenv("STRATA_SPLIT_OWN_BUFFERS");
                    return v == nullptr || v[0] != '0';
                }();
                if (pf_parts.size() > 1 && own_ok) {
                    for (PfPart& p : pf_parts) {
                        const strata::core::OnDevice on(p.dev);
                        size_t fb = 0, tb = 0;
                        /*
                        DPCT1010: SYCL uses exceptions to report errors and
                        does not use the error codes. The cudaGetLastError
                        function call was replaced with 0. You need to rewrite
                        this code.
                        */
                        /*
                        DPCT1106: 'cudaMemGetInfo' was migrated with the
                        Intel extensions for device information which may not be
                        supported by all compilers or runtimes. You may need to
                        adjust the code.
                        */
                        if (DPCT_CHECK_ERROR(
                                dpct::get_current_device().get_memory_info(
                                    fb, tb)) != 0) {
                            (void)0; continue;
                        }
                        const uint64_t need = strata::prefill::Prefill::bytes_needed(g, *p.ses, chunk);
                        if ((uint64_t) fb >= need + (3ull << 29)) {
                            std::fprintf(stderr, "strata serve:   CUDA%d keeps its own prompt buffers (%.2f GiB of "
                                                 "%.2f GiB free): no loan\n", p.dev < 0 ? 0 : p.dev,
                                         (double) need / 1073741824.0, (double) fb / 1073741824.0);
                            p.first = -1;
                            p.first_now = -1;
                        }
                    }
                }
                lend_first = pf_parts[0].first;
                borrow = lend_first >= 0 ? xcache.device_slot(lend_first) : nullptr;
                borrow_bytes = lend_first >= 0 ? part_bytes(pf_parts[0], lend_first) : 0;
            } else if (o.prefill_auto) {
                o.prefill_chunk = 1024;   // nothing lendable: small buffers of its own
            } else if (pf_parts.size() > 1) {
                // An explicit chunk no stage can lend in full.  main falls back to the prompt path's own buffers
                // here and so do we, rather than refusing to start - but say what every stage has, because a split
                // stage that has to allocate these on top of a cache that already filled its VRAM will not fit,
                // and `init` would otherwise report only that the buffers do not fit.
                std::fprintf(stderr, "strata serve: no stage can lend the prompt path its %lld-token buffers, so "
                                     "each stage allocates its own:\n", (long long) o.prefill_chunk);
                for (const PfPart& p : pf_parts)
                    std::fprintf(stderr, "strata serve:   CUDA%d has %lld slots, and a %lld-token chunk needs "
                                         "the last %lld of them\n", p.dev < 0 ? 0 : p.dev,
                                 (long long) p.cache->slots(), (long long) o.prefill_chunk,
                                 (long long) part_slots(p, o.prefill_chunk));
            }
        } else if (o.prefill_auto && d_res == nullptr) {
            o.prefill_chunk = 1024;       // #85: no expert cache at all (a full 8 GB card): small buffers of its own
        }
        bool any_loan = borrow != nullptr;
        for (size_t i = 1; i < pf_parts.size(); ++i) any_loan = any_loan || pf_parts[i].first >= 0;
        if (any_loan) {
            if (borrow != nullptr)
                std::fprintf(stderr, "strata serve: the prompt path borrows %lld CUDA0 cache slots (%.2f GiB)\n",
                             (long long) (xcache.slots() - lend_first), (double) borrow_bytes / 1073741824.0);
            for (size_t i = 1; i < pf_parts.size(); ++i)   // one loan per stage, from that stage's own cache
                if (pf_parts[i].first >= 0) std::fprintf(stderr, "strata serve:   CUDA%d prompt path borrows %lld of its %lld slots (%.2f GiB)\n",
                             pf_parts[i].dev, (long long) (pf_parts[i].cache->slots() - pf_parts[i].first),
                             (long long) pf_parts[i].cache->slots(),
                             (double) part_bytes(pf_parts[i], pf_parts[i].first) / 1073741824.0);
        } else {
            std::fprintf(stderr, "strata serve: the prompt path allocates its own buffers (too few cache slots to borrow)\n");
        }
        rss_probe("the prompt path set up");
        // layer split across GPUs: a prompt path per stage, each handing its chunk's rows to the next.
        // THE CHUNK STEPS DOWN INSTEAD OF EXITING.  A split's stage caches are sized after the arena is registered, and
        // the prompt path's own buffers (no loan) are not priced into them: with the whole arena pinned (#253) a
        // `--prefill auto` split could stop at start with "device buffers for a chunk of 2048 tokens do not fit".  A
        // chunk that does not fit is tried again one size smaller, down to 512 tokens (a smaller chunk only reads slower).
        auto init_prompt_paths = [&]() -> int {   // 0: ready; 1: failed (err set); 2: failed with "do not fit"
            for (size_t i = 0; i < stages.size(); ++i) {
                GpuStage& st = *stages[i];
                st.sp.set_stage(st.lb, i + 1 < stages.size() ? st.le : -1, i + 1 < stages.size() ? &stages[i + 1]->sp : nullptr);
                const strata::core::OnDevice on(st.dev);
                void* sb = nullptr;              // this stage's own loan, out of its own cache
                uint64_t sbb = 0;
                // `first < 0`: no loan was taken (nothing was lendable), so this stage allocates its own buffers
                if (i + 1 < pf_parts.size() && pf_parts[i + 1].first >= 0) {
                    sb = st.cache.device_slot(pf_parts[i + 1].first);
                    sbb = part_bytes(pf_parts[i + 1], pf_parts[i + 1].first);
                }
                if (!st.sp.init(st.wt, g, st.ss, srcp, &st.cache, host_res.data(), o.prefill_chunk, (void*) st.stream,
                                err, sb, sbb)) {
                    err = "layer split, CUDA" + std::to_string(st.dev) + " prompt path: " + err;
                    return err.find("do not fit") != std::string::npos ? 2 : 1;
                }
            }
            if (multi_gpu) sp.set_stage(0, split_at[0], &stages[0]->sp);
            if (!sp.init(wt, g, ss, srcp, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes))
                return err.find("do not fit") != std::string::npos ? 2 : 1;
            return 0;
        };
        static constexpr int64_t kStepChunks[] = {6144, 4096, 3072, 2048, 1536, 1024, 512};
        {
            // First by arithmetic: a prompt path without a loan allocates its buffers, so the chunk must leave
            // headroom on that device (a chunk that fits to the last MiB left hipBLAS nothing: its GEMMs then
            // failed to launch on gfx1201 and the prompt hung).  `bytes_needed` is the same count `init` makes.
            const int64_t kHeadroom = 512ll << 20;
            auto own_fits = [&](int64_t c, int &dev_out, int64_t &need_out,
                                int64_t &free_out) -> bool {
                try {
            for (size_t i = 0; i <= stages.size(); ++i) {
                    const bool loan = i == 0 ? borrow != nullptr : (i < pf_parts.size() && pf_parts[i].first >= 0);
                    if (loan) continue;
                    const int dev = i == 0 ? -1 : stages[i - 1]->dev;
                    const strata::core::OnDevice on(dev);
                    size_t fb = 0, tb = 0;
                    /*
                    DPCT1106: 'cudaMemGetInfo' was migrated with the Intel
                    extensions for device information which may not be supported
                    by all compilers or runtimes. You may need to adjust the
                    code.
                    */
                    dpct::get_current_device().get_memory_info(fb, tb);
                    const int64_t need = (int64_t) strata::prefill::Prefill::bytes_needed(g, i == 0 ? ss : stages[i - 1]->ss, c);
                    if (need + kHeadroom > (int64_t) fb) {
                        dev_out = dev < 0 ? 0 : dev; need_out = need; free_out = (int64_t) fb;
                        return false;
                    }
                }
                return true;
            }
            catch (sycl::exception const &exc) {
              std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                        << ", line:" << __LINE__ << std::endl;
              std::exit(1);
            }
            };
            int dev = 0;
            int64_t need = 0, fb = 0;
            if (!own_fits(o.prefill_chunk, dev, need, fb)) {
                int64_t c = 0;
                for (const int64_t s : kStepChunks) {
                    int d2 = 0;
                    int64_t n2 = 0, f2 = 0;
                    if (s < o.prefill_chunk && own_fits(s, d2, n2, f2)) { c = s; break; }
                }
                if (c > 0) {
                    std::fprintf(stderr, "strata serve: a %lld-token chunk's prompt buffers need %lld MiB on CUDA%d, "
                                         "%lld MiB free: %lld-token chunks\n", (long long) o.prefill_chunk,
                                 (long long) (need >> 20), dev, (long long) (fb >> 20), (long long) c);
                    o.prefill_chunk = c;
                }
            }
        }
        for (;;) {
            const int r = init_prompt_paths();
            if (r == 0) break;
            int64_t next = 0;
            for (const int64_t c : kStepChunks)
                if (c < o.prefill_chunk) { next = c; break; }
            if (r == 2 && next > 0) {
                std::fprintf(stderr, "strata serve: %s: trying a %lld-token chunk\n", err.c_str(), (long long) next);
                // start the prompt paths over (a failed init may hold buffers), and consume the failed allocation's
                // error: it is sticky, and the next launch check would report "out of memory" for a kernel
                for (auto& stp : stages) {
                    const strata::core::OnDevice on(stp->dev);
                    stp->sp.reset();
                    /*
                    DPCT1010: SYCL uses exceptions to report errors and
                    does not use the error codes. The cudaGetLastError function
                    call was replaced with 0. You need to rewrite this code.
                    */
                    (void)0;
                }
                sp.reset();
                /*
                DPCT1010: SYCL uses exceptions to report errors and does
                not use the error codes. The cudaGetLastError function call was
                replaced with 0. You need to rewrite this code.
                */
                (void)0;
                o.prefill_chunk = next;
                if (any_loan) {                  // smaller loans for the smaller chunk
                    for (PfPart& p : pf_parts)
                        if (p.first >= 0) {
                            p.first = (int32_t) (p.cache->slots() - part_slots(p, next));
                            p.first_now = p.first;
                        }
                    lend_first = pf_parts[0].first;
                    borrow = lend_first >= 0 ? xcache.device_slot(lend_first) : nullptr;
                    borrow_bytes = lend_first >= 0 ? part_bytes(pf_parts[0], lend_first) : 0;
                }
                err.clear();
                continue;
            }
            std::fprintf(stderr, "strata serve: %s\n", err.c_str());
            if (err.find("fit") != std::string::npos)   // #85: say what frees VRAM
                std::fprintf(stderr, "strata serve: the GPU has too little free VRAM for the prompt path: turn images "
                                     "off (setup: --vision no), close other programs using the GPU, use a shorter "
                                     "context, or read prompts in smaller chunks (--prefill 512)\n");
            return 1;
        }
        if (peer.valid() && o.peer_prefill_rows != 0) {
            const int64_t rows = o.peer_prefill_rows > 0 ? o.peer_prefill_rows : o.prefill_chunk * K / 2;
            if (!sp.set_peer(&peer, rows, err)) {
                std::fprintf(stderr, "strata serve: %s - the prompt path stays on the primary GPU\n", err.c_str());
                err.clear();
            }
        }
        // a layer split: a one-chunk prompt runs the stages one after the other, so each stage's prompt path gets the
        // next stage's GPU (the last one the first's) to stream and compute a share of its experts meanwhile
        if (multi_gpu && !stages.empty()) {
            std::vector<std::pair<strata::prefill::Prefill*, int>> paths{{&sp, 0}};
            for (const auto& st : stages) paths.emplace_back(&st->sp, st->dev);
            for (size_t i = 0; i < paths.size(); ++i) {
                const strata::core::OnDevice on(paths[i].second);
                if (!paths[i].first->set_stage_helper(paths[(i + 1) % paths.size()].first, err)) {
                    std::fprintf(stderr, "strata serve: %s - stage %zu reads its prompts alone\n", err.c_str(), i);
                    err.clear();
                }
            }
        }
        mem_mark("the head and the prompt path");
        // #340: STRATA_SPLIT_SMALL_OWN=S (tokens): on a layer split, every stage that borrows keeps the slots for an
        // S-token chunk's buffers for the whole session (0.1.29's own buffers, carved from the tail of its cache):
        // a request of at most S prompt tokens then lends, streams and refills nothing, a longer one lends (and
        // refills) only the slots above them.  Their experts stay non-resident (the CPU / PCIe path computes them).
        // STRATA_SPLIT_SMALL_MAX=M: a request of at most M prompt tokens reads in S-token chunks on those stages
        // (several chunks, still nothing lent) instead of borrowing for a bigger one.
        int64_t split_small = 0, split_small_max = 0;
        if (multi_gpu && !pf_parts.empty() && !host_res.empty()) {
            const char* v = std::getenv("STRATA_SPLIT_SMALL_OWN");
            const int64_t S = v ? request_chunk(std::atoll(v), o.prefill_chunk) : 0;
            const char* vm = std::getenv("STRATA_SPLIT_SMALL_MAX");
            split_small = S;
            split_small_max = S > 0 ? std::max<int64_t>(S, vm ? std::atoll(vm) : S) : 0;
            int64_t evicted = 0;
            for (PfPart& p : pf_parts) {
                if (S <= 0 || p.first < 0) continue;
                const int32_t first_s = std::max<int32_t>(p.first, (int32_t) (p.cache->slots() - part_slots(p, S)));
                int64_t n = 0;
                for (int64_t l = p.lb; l < p.le; ++l)
                    for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                        int32_t& r = host_res[(size_t) (l * g.n_expert + ex)];
                        if (r >= first_s) { r = strata::core::kNotResident; ++n; }
                    }
                evicted += n;
                std::fprintf(stderr, "strata serve:   CUDA%d keeps %lld slots for %lld-token prompts (%lld experts "
                                     "no longer resident)\n", p.dev < 0 ? 0 : p.dev,
                             (long long) (p.cache->slots() - first_s), (long long) S, (long long) n);
            }
            if (evicted > 0) {
                if (d_res != nullptr)
                    res_put(d_res);
                for (auto& st : stages) {
                    const strata::core::OnDevice on(st->dev);
                    res_put(st->d_res);
                }
            }
        }
        kvg_start(borrow != nullptr ? (int64_t) lend_first : xcache.slots());
        // the penalty-history buffer: one row per verify-window row (`penalty_rows`), each the last
        // `penalty_last_n` tokens that row's pick follows, -1 padded in front.  Allocated once at the cap for
        // the widest window; a request without penalties gets a null buffer and takes the byte-for-byte
        // neutral path (no upload, no buffer handed to the sampler).
        constexpr int kPenaltyWindowCap = 4096;
        constexpr size_t kHistSlots = (size_t) kPenaltyWindowCap * (size_t) strata::kernels::kVerifyMaxT;
        int32_t* d_hist = nullptr;
        std::vector<int32_t> hist_stage(kHistSlots, -1);
        const int hist_dev = last_st ? last_st->dev : -1;   // with the head: the last stage's device
        if (const strata::core::OnDevice on_h(hist_dev);
            DPCT_CHECK_ERROR(d_hist = sycl::malloc_device<int32_t>(
                                 kHistSlots, dpct::get_in_order_queue())) !=
            0) {
            std::fprintf(stderr, "strata serve: the penalty-history allocation failed\n");
            return 1;
        }
        strata::core::Verifier ver;
        strata::core::VerifyHits vh;
        int32_t* usage_dev_srv = nullptr;
        vh.d_res = thits.d_res;
        // 0.1.44 --expert-profile-save under STRATA_VERIFY_NO_HOST (no adaptive tier, so no host-side counts): the device
        // plan counts every routed expert of the verify windows and the saved profile is ranked from those counts.
        if (mirror_table_d && !o.expert_profile_save.empty()) {
            usage_dev_srv = sycl::malloc_device<int32_t>((size_t) (g.n_layers * g.n_expert), dpct::get_in_order_queue());
            dpct::get_in_order_queue().memset(usage_dev_srv, 0, (size_t) (g.n_layers * g.n_expert) * 4).wait();
        }
        if (mirror_table_d) strata::kernels::resident_plan_set_mirror(thits.d_res, mirror_table_d, usage_dev_srv, (long long) (g.n_layers * g.n_expert));
        vh.h_res = host_res.empty() ? nullptr : host_res.data();
        vh.cache_base = thits.cache_base;
        vh.blob = thits.blob;
        vh.slot_off = xcache.slot_offsets();   // E-6: the device plan's pointers
        vh.n_slots = xcache.slots();
        // Layer split: `ver` runs layers [0, K1) and hands its residual to the next stage's verifier, and so on; the
        // last runs the head.  The hand-offs are mapped pinned memory, portable: a stage on another GPU reads it.
        // (--split-device 0: the second stage on this GPU, sharing its weights, session and cache - the A/B.)
        strata::core::Verifier ver_same;
        SplitDrive split_drive;
        auto stage_ver = [&](int st) -> strata::core::Verifier& {
            return st == 0 ? ver : split_same ? ver_same : stages[(size_t) st - 1]->ver;
        };
        auto pcie_num_of = [](double f) { return std::max(0, std::min(256, (int) (f * 256.0 + 0.5))); };
        const int n_stages = split_devs.empty() ? 1 : (int) split_at.size() + 1;
        // --pipeline-windows (decided where the split was): two verifiers per stage, one for the even windows and one
        // for the odd ones, sharing one stream of their card, and a hand-off per parity - so stage 0 can run window
        // K+1 while stage 1 still reads window K's hand-off
        const bool pipe = o.pipeline_windows > 0 && n_stages == 2 && !split_same && stages.size() == 1;
        strata::core::Verifier ver_b;
        SplitDrive split_drive_b;
        dpct::queue_ptr pl_stream[2] = {&dpct::get_in_order_queue(),
                                        &dpct::get_in_order_queue()};
        if (pipe) {
            /*
            DPCT1025: The SYCL queue is created ignoring the flag and
            priority options.
            */
            bool ok_p =
                DPCT_CHECK_ERROR(
                    pl_stream[0] =
                        dpct::get_current_device().create_queue(true)) == 0;
            {
                const strata::core::OnDevice on(stages[0]->dev);
                /*
                DPCT1025: The SYCL queue is created ignoring the flag and
                priority options.
                */
                ok_p =
                    ok_p &&
                    DPCT_CHECK_ERROR(
                        pl_stream[1] =
                            dpct::get_current_device().create_queue(true)) == 0;
            }
            if (!ok_p) { std::fprintf(stderr, "strata serve: --pipeline-windows: stream creation failed\n"); return 1; }
            for (strata::core::Verifier* v : {&ver, &ver_b}) {
                v->set_stream(pl_stream[0]);
                v->set_always_publish(true);
            }
            for (strata::core::Verifier* v : {&stages[0]->ver, &stages[0]->ver_b}) {
                v->set_stream(pl_stream[1]);
                v->set_always_publish(true);
            }
        }
        if (n_stages > 1) {
            const size_t hb = (size_t) strata::kernels::kVerifyMaxT *
                              (size_t) strata::core::Verifier::handoff_floats(g) * sizeof(float);
            std::vector<float*> hand((size_t) n_stages - 1, nullptr);
            for (float*& h : hand) {
                float* hh = nullptr;
                /*
                DPCT1048: The original value cudaHostAllocMapped is not
                meaningful in the migrated code and was removed or replaced with
                0. You may need to check the migrated code.
                */
                /*
                DPCT1048: The original value cudaHostAllocPortable is not
                meaningful in the migrated code and was removed or replaced with
                0. You may need to check the migrated code.
                */
                if (DPCT_CHECK_ERROR(hh = (float *)sycl::malloc_host(
                                         hb, dpct::get_in_order_queue())) !=
                        0 ||
                    DPCT_CHECK_ERROR(*(void **)&h = (float *)hh) != 0) {
                    std::fprintf(stderr, "strata serve: the layer-split hand-off allocation failed\n");
                    return 1;
                }
                std::memset(hh, 0, hb);
            }
            split_drive.base = &drive;
            split_drive.n = n_stages;
            for (int st = 0; st < n_stages; ++st) {
                stage_ver(st).set_stage(st == 0 ? 0 : split_at[(size_t) st - 1], st + 1 < n_stages ? split_at[(size_t) st] : -1,
                                        st == 0 ? nullptr : hand[(size_t) st - 1], st + 1 < n_stages ? hand[(size_t) st] : nullptr);
                split_drive.end[st] = st + 1 < n_stages ? split_at[(size_t) st] : g.n_layers;
                split_drive.cache_base[st] = drive.d.cache_base;
                split_drive.cache_slot_off[st] = drive.d.cache_slot_off;
                split_drive.pcie_num[st] = pcie_num_of(o.pcie_frac);
            }
            for (int st = 1; st < n_stages; ++st) {
                bool ok_s = false;
                if (split_same) {
                    ok_s = ver_same.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, std::max(o.spec, o.batch), err);
                } else {
                    GpuStage& gs = *stages[(size_t) st - 1];
                    const strata::core::OnDevice on(gs.dev);
                    strata::core::VerifyHits vs;
                    vs.d_res = gs.d_res;
                    vs.h_res = host_res.empty() ? nullptr : host_res.data();
                    vs.cache_base = gs.cache.device_slot(0);
                    vs.blob = thits.blob;
                    vs.slot_off = gs.cache.slot_offsets();
                    vs.n_slots = gs.cache.slots();
                    gs.ver.set_remote_expert_opt(remote_opt.get());
                    ok_s = gs.ver.init(gs.wt, g, gs.ss, vs, gs.head.loaded() ? &gs.head : nullptr, std::max(o.spec, o.batch), err);
                    split_drive.cache_base[st] = gs.cache.device_slot(0);
                    split_drive.cache_slot_off[st] = gs.cache.slot_offsets();
                    split_drive.pcie_num[st] = pcie_num_of(gs.pcie_frac);
                }
                if (!ok_s) {
                    std::fprintf(stderr, "strata serve: layer split, stage %d: %s\n", st + 1, err.c_str());
                    return 1;
                }
            }
            for (int st = 0; st + 1 < n_stages; ++st) stage_ver(st).set_next(&stage_ver(st + 1), &split_drive);
            std::string plan_s = "0-" + std::to_string(split_at[0] - 1) + " (CUDA0)";
            for (int st = 1; st < n_stages; ++st)
                plan_s += ", " + std::to_string(split_at[(size_t) st - 1]) + "-" + std::to_string(split_drive.end[st] - 1) +
                          " (CUDA" + std::to_string(split_same ? 0 : stages[(size_t) st - 1]->dev) + ")";
            std::fprintf(stderr, "strata serve: layer split: layers %s, one hand-off per window\n", plan_s.c_str());
        }
        // --pipeline-windows: the odd windows' verifiers and their hand-off (initialized before `ver`, which stays the
        // watchdog's verifier), and the drafter's own row buffer, which either parity's rows are copied into
        float* pl_mtp_R = nullptr;
        if (pipe) {
            const size_t hb = (size_t) strata::kernels::kVerifyMaxT *
                              (size_t) strata::core::Verifier::handoff_floats(g) * sizeof(float);
            float *hh = nullptr, *hand_b = nullptr;
            /*
            DPCT1048: The original value cudaHostAllocMapped is not
            meaningful in the migrated code and was removed or replaced with 0.
            You may need to check the migrated code.
            */
            /*
            DPCT1048: The original value cudaHostAllocPortable is not
            meaningful in the migrated code and was removed or replaced with 0.
            You may need to check the migrated code.
            */
            if (DPCT_CHECK_ERROR(hh = (float *)sycl::malloc_host(
                                     hb, dpct::get_in_order_queue())) != 0 ||
                DPCT_CHECK_ERROR(*(void **)&hand_b = (float *)hh) != 0) {
                std::fprintf(stderr, "strata serve: --pipeline-windows: the second hand-off allocation failed\n");
                return 1;
            }
            std::memset(hh, 0, hb);
            GpuStage& gs = *stages[0];
            ver_b.set_stage(0, split_at[0], nullptr, hand_b);
            gs.ver_b.set_stage(split_at[0], -1, hand_b, nullptr);
            {
                const strata::core::OnDevice on(gs.dev);
                strata::core::VerifyHits vs;
                vs.d_res = gs.d_res;
                vs.h_res = host_res.empty() ? nullptr : host_res.data();
                vs.cache_base = gs.cache.device_slot(0);
                vs.blob = thits.blob;
                vs.slot_off = gs.cache.slot_offsets();
                vs.n_slots = gs.cache.slots();
                if (!gs.ver_b.init(gs.wt, g, gs.ss, vs, gs.head.loaded() ? &gs.head : nullptr, o.spec, err)) {
                    std::fprintf(stderr, "strata serve: --pipeline-windows: the second verifier on CUDA%d (the later "
                                         "card): %s (raise --vram-reserve-mib by ~200, or --pipeline-windows 0)\n",
                                 gs.dev, err.c_str());
                    return 1;
                }
                if (DPCT_CHECK_ERROR(pl_mtp_R = sycl::malloc_device<float>(
                                         (size_t)strata::kernels::kVerifyMaxT *
                                             (size_t)(g.hc * g.n_embd),
                                         dpct::get_in_order_queue())) != 0) {
                    std::fprintf(stderr, "strata serve: --pipeline-windows: the drafter's row buffer does not fit\n");
                    return 1;
                }
            }
            if (!ver_b.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.spec, err)) {
                std::fprintf(stderr, "strata serve: --pipeline-windows: the second verifier on CUDA0 (the first card): "
                                     "%s (raise --vram-reserve-mib by ~200, or --pipeline-windows 0)\n", err.c_str());
                return 1;
            }
            ver_b.set_next(&gs.ver_b, &split_drive_b);   // the setters reach it; the pipeline never chains run/commit
        }
        ver.set_remote_expert_opt(remote_opt.get());
        if (batch_mtp) ver.set_batch_graph_limit(64);   // --batch-mtp only: slot rotation makes many layouts (LRU); 0.1.39 keeps all
        if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr,
                      batch_mtp ? strata::kernels::kVerifyMaxT : std::max(o.spec, o.batch), err) ||
            (use_mtp && !mtp.bind(last_st ? last_st->wt : wt, last_st ? &last_st->head : &native_head,
                                  pipe ? pl_mtp_R : ver.final_R_all(), err))) {
            std::fprintf(stderr, "strata serve: %s\n", err.c_str());
            return 1;
        }
        if (pipe) mtp.set_source_R(ver.final_R_all());   // the serial decode's rows (the last stage's even verifier)
        auto free_slot_mtp_rows = [](float *p) {
                                                 try {
        if (p != nullptr) (void) DPCT_CHECK_ERROR(sycl::free(p, dpct::get_in_order_queue())); }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        std::vector<std::unique_ptr<float, decltype(free_slot_mtp_rows)>> slot_mtp_rows;
        if (batch_mtp) {
            for (int b = 0; b < o.batch; ++b) {
                float* r = nullptr;
                const size_t bytes = (size_t) o.spec * (size_t) g.hc * (size_t) g.n_embd * sizeof(float);
                if (DPCT_CHECK_ERROR(r = (float *)sycl::malloc_device(
                                         bytes, dpct::get_in_order_queue())) !=
                    0) {
                    std::fprintf(stderr, "strata serve: batch MTP slot %d residual buffer does not fit%s\n",
                                 b, vram_free_note().c_str());
                    return 1;
                }
                slot_mtp_rows.emplace_back(r, free_slot_mtp_rows);
                if (!slot_mtp[(size_t) b]->bind(wt, &native_head, r, err, &mtp)) {
                    std::fprintf(stderr, "strata serve: batch MTP slot %d: %s%s\n", b, err.c_str(),
                                 vram_free_note().c_str());
                    return 1;
                }
            }
        }
        // each stage's verifier gets its stage's slot sessions
        if (o.batch > 0) {
            for (size_t k = 0; k < bslot_ss.size(); ++k) {
                std::vector<strata::core::SessionState*> ptrs;
                for (auto& u : bslot_ss[k]) ptrs.push_back(u.get());
                strata::core::Verifier& vk = k == 0 ? ver : stages[k - 1]->ver;
                const strata::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
                if (!vk.init_slots(ptrs, err)) {
                    std::fprintf(stderr, "strata serve: --batch, stage %zu: %s\n", k + 1, err.c_str());
                    return 1;
                }
            }
        }
        for (int st = 0; st < n_stages && n_stages > 1; ++st) {
            split_drive.plan[st] = stage_ver(st).plan_sink();
            if (st > 0) {
                stage_ver(st).set_split(o.spec_split);
                stage_ver(st).set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
            }
        }
        if (pipe) {   // the odd windows' pool routing: the same as the even ones', with their plans
            split_drive_b = split_drive;
            split_drive_b.plan[0] = ver_b.plan_sink();
            split_drive_b.plan[1] = stages[0]->ver_b.plan_sink();
            stages[0]->ver_b.set_split(o.spec_split);
            stages[0]->ver_b.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
            ver_b.set_split(o.spec_split);
            ver_b.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
        }
        // the pool the verify windows call: with a layer split, the wrapper that routes each layer to its stage
        const strata::core::PoolMultiFn win_pool_fn = n_stages > 1 ? &drive_pool_split : &drive_pool_multi;
        void* const win_pool_user = n_stages > 1 ? (void*) &split_drive : (void*) &drive;
        mem_mark("the verifier and the drafter's binding");
        ver.set_split(o.spec_split);
        // auto: the copy kernel for every pack.  DMA (the native packs' default until 0.1.13) has the host call
        // cudaMemcpyAsync + cudaLaunchHostFunc inside a verify window while the GPU spins on the flag they raise;
        // issue #31's thread dumps show the host stuck in that cudaMemcpyAsync on a driver lock for good.  The copy
        // kernel needs no host CUDA call there, and costs ~1-3% decode on IQ3_S (45.3 -> 44.8 tok/s, 8 requests).
        ver.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
        // --pipeline-windows 2: stage 0's GDN state before a speculative commit, restored when the guess was wrong.
        // Window W reads the state its snapshot must hold and no window writes it (only commits do), so it is copied
        // on a side stream while W runs, into the buffer of W's parity, and only W's speculative commit waits for it
        // (STRATA_PIPELINE_SNAP_OVERLAP=0: one buffer, copied on the critical path).
        const size_t pl_snap_bytes = pipe ? gdn_snapshot_bytes(ss) : 0;
        float* pl_snap2[2] = {nullptr, nullptr};
        dpct::queue_ptr pl_snap_stream = &dpct::get_in_order_queue();
        dpct::event_ptr pl_snap_pre[2] = {nullptr, nullptr},
                        pl_snap_ev[2] = {nullptr, nullptr};
        bool pl_snap_overlap = false;
        if (pipe && o.pipeline_windows >= 2) {
            if (DPCT_CHECK_ERROR(
                    pl_snap2[0] = (float *)sycl::malloc_device(
                        pl_snap_bytes, dpct::get_in_order_queue())) != 0) {
                std::fprintf(stderr, "strata serve: --pipeline-windows 2: the GDN snapshot (%.1f MiB) does not fit; the "
                                     "decode stays serial\n", (double) pl_snap_bytes / 1048576.0);
                /*
                DPCT1026: The call to cudaGetLastError was removed because
                this functionality is redundant in SYCL.
                */
                pl_snap2[0] = nullptr;
            } else if ([] { const char* v = pipe_dbg_env("STRATA_PIPELINE_SNAP_OVERLAP"); return v == nullptr || std::atoi(v) != 0; }()) {
                pl_snap_overlap =
                    DPCT_CHECK_ERROR(
                        pl_snap2[1] = (float *)sycl::malloc_device(
                            pl_snap_bytes, dpct::get_in_order_queue())) == 0 &&
                    /*
                    DPCT1025: The SYCL queue is created ignoring the flag
                    and priority options.
                    */
                    DPCT_CHECK_ERROR(
                        pl_snap_stream =
                            dpct::get_current_device().create_queue(true)) == 0;
                for (int i = 0; i < 2 && pl_snap_overlap; ++i)
                    pl_snap_overlap =
                        DPCT_CHECK_ERROR(pl_snap_pre[i] = new sycl::event()) ==
                            0 &&
                        DPCT_CHECK_ERROR(pl_snap_ev[i] = new sycl::event()) ==
                            0;
                /*
                DPCT1010: SYCL uses exceptions to report errors and does
                not use the error codes. The cudaGetLastError function call was
                replaced with 0. You need to rewrite this code.
                */
                if (!pl_snap_overlap) 0;
            }
        }
        if (pipe)
            std::fprintf(stderr, "strata serve: --pipeline-windows %d: two verifiers per stage, the stages overlap across "
                                 "windows%s\n", o.pipeline_windows,
                         pl_snap2[0] == nullptr ? "" : pl_snap_overlap ? " (decode too; the GDN snapshots beside the window)"
                                                                       : " (decode too; the GDN snapshot on the critical path)");
        // the verifiers by [stage][parity], their pool routing, and the drafter's events (in a pipelined prompt read, a
        // stage-1 window waits until the drafter has read the rows of the same parity's previous window)
        strata::core::Verifier* PV[2][2] = {{&ver, &ver_b}, {pipe ? &stages[0]->ver : nullptr, pipe ? &stages[0]->ver_b : nullptr}};
        SplitDrive* PSD[2] = {&split_drive, &split_drive_b};
        dpct::event_ptr pl_mtp_ev[2] = {nullptr, nullptr};
        bool pl_mtp_live[2] = {false, false};
        bool pl_prepared = false;
        // every graph captured and the drafter's prompt path sized while nothing is in flight (a capture syncs its stream)
        auto pl_prepare = [&](std::string &e) -> bool {
            try {
        if (pl_prepared) return true;
            for (int st = 0; st < 2; ++st)
                for (int par = 0; par < 2; ++par) {
                    const strata::core::OnDevice on(PV[st][par]->device());
                    if (!PV[st][par]->capture_all(e)) return false;
                }
            if (!mtp.prepare_prefill(e)) return false;
            if (o.pipeline_windows >= 2 && pl_snap2[0] != nullptr && !mtp.prepare_chain(e)) return false;
            const strata::core::OnDevice on(mtp.device());
            for (dpct::event_ptr &ev : pl_mtp_ev)
                if (ev == nullptr &&
                    DPCT_CHECK_ERROR(ev = new sycl::event()) != 0) {
                    e = "pipeline: event creation failed";
                    return false;
                }
            pl_prepared = true;
            return true;
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        // INVARIANT: nothing pipelined is in flight outside read_windows_pl and the pipelined decode loop - both end
        // here (a checkpoint, a park, the next request and the serial paths all find both stages and the drafter idle).
        // Every window in flight is served to its end (its picks discarded), then both stages and the drafter idle.
        auto pl_drain = [&]() {
            try {
        std::string e2;
            for (int round = 0; round < 10000000; ++round) {
                bool any = false;
                for (int st = 0; st < 2; ++st)
                    for (int par = 0; par < 2; ++par) {
                        strata::core::Verifier& v = *PV[st][par];
                        if (!v.in_flight()) continue;
                        any = true;
                        if (v.service(&drive_pool_split, PSD[par], e2) < 0) return;
                        if (v.done(e2)) v.pl_finish(nullptr, e2);
                        else if (!e2.empty()) return;
                    }
                if (!any) break;
            }
            for (int st = 0; st < 2; ++st) {
                const strata::core::OnDevice on(PV[st][0]->device());
                PV[st][0]->stream()->wait();
            }
            const strata::core::OnDevice on(mtp.device());
            mtp.stream()->wait();
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        std::vector<int64_t> cur;
        // ---- the conversation cache (see ConvCheckpoint).  `live` is what the session holds right now: the tokens
        // it has consumed, so a request that starts with exactly them continues without any copy.  `checks` are the
        // saved points; every one of them is a prefix of `live` (the loop drops the rest), so they form a chain -
        // the radix cache's tree collapsed onto the one branch of history whose cells the session holds.  The
        // chain's root is the deepest point every request so far shared (the end of the system prompt, in
        // practice); the retention policy pins it and rotates the rest LRU (conv_cache.hpp), so a NEW chat that
        // shares that prefix mounts through it instead of reading it again.
        std::vector<int32_t> live;
        std::vector<ImgKey> live_imgs, req_imgs;
        bool live_ok = false;
        std::vector<ConvCheckpoint> checks;
        uint64_t check_clock = 0;   // the checkpoints' LRU clock; creation and every use advance it
        int64_t tail_ckpt_len = -1;   // --prompt-cache-tail: the length of the one tail checkpoint alive (-1 = none)
        bool cvec_cached = true;   // the control vector's state the live session and the checkpoints were read with
        // without --mtp the conversation cache stays on: a parked image carries the draft layer's K/V only when there is one
        // (the snapshots accept a null draft)
        strata::core::ConversationCache conversations(
            o.prompt_cache > 0 ? (size_t) o.conversation_cache_mib * 1024 * 1024 : 0,
            (size_t) o.conversation_cache_slots);
        // Disk sessions: what a session file is bound to.  The model fingerprint samples every model input this
        // engine loaded, by role (conversation_file.hpp), once; the config fingerprint covers the RESOLVED settings
        // that change what the saved bytes mean - the rope (K is cached post-RoPE), the loaded control vector, the
        // K/V format and the arithmetic switches.  Sampling, seeds, draft tuning and the expert tier are not in it.
        std::optional<uint64_t> model_fp, config_fp;
        auto session_identity = [&](strata::core::SessionFileIdentity& id, std::string& e,
                                    const std::function<void()>& next_file) -> bool {
            if (!model_fp) {
                // what the loaders actually resolved: the CLI inputs, the pack's files and - from the expert source
                // itself - every file its experts were read from (native_experts.txt can name GGUFs per layer/role)
                strata::core::SessionInputs in;
                in.native_shards = o.native_shards;
                in.native_dense = o.native_dense_gguf;
                in.native_head = o.native_head_shards;
                in.embedding = o.embd_gguf;
                in.ple = o.ple_gguf;
                in.pack = o.pack;
                in.mtp = o.mtp.empty() ? std::string() : o.mtp + "/dense.bin";
                if (srcp != nullptr) in.experts = srcp->model_inputs();
                std::vector<strata::core::SessionModelFile> files = strata::core::session_model_inputs(in);
                // the pack's other files the loader reads (weights.cpp, expert_layout.cpp)
                if (!o.pack.empty()) {
                    for (const char* n : {"index.txt", "dense.bin"}) files.push_back({std::string("pack ") + n, o.pack + "/" + n});
                    for (const char* n : {"embd.bin", "extra.bin", "manifest.json"})
                        files.push_back({std::string("pack ") + n, o.pack + "/" + n, true});
                    if (srcp == nullptr) files.push_back({"pack experts.bin", o.pack + "/experts.bin", true});
                }
                if (!o.mtp.empty()) {
                    for (const char* n : {"dense.txt", "experts.bin"}) files.push_back({std::string("mtp ") + n, o.mtp + "/" + n});
                    files.push_back({"mtp draft_vocab.bin", o.mtp + "/draft_vocab.bin", true});
                }
                uint64_t fp = 0;
                // each file is one blocking read of at most 2 MiB (head and tail): announced before it starts
                if (next_file) next_file();
                if (!strata::core::session_model_fingerprint(files, fp, e, next_file))
                    return false;
                model_fp = fp;
            }
            if (!config_fp) {
                strata::core::SessionConfig c;
                c.engine_version = STRATA_VERSION;
#if defined(STRATA_USE_HIP)
                c.backend = "hip";
#else
                c.backend = "cuda";
#endif
                c.kv = o.kv;
                c.max_context = o.max_context;
                c.kv_resident = o.kv_resident;
                c.mtp_window = o.mtp.empty() ? -1 : o.mtp_window;
                const char* rot = std::getenv("STRATA_KV_ROT");
                c.kv_rot = rot != nullptr && rot[0] == '1';
                c.rope = {(int64_t) rope_cfg.type, rope_cfg.freq_base, rope_cfg.factor, rope_cfg.freq_scale(),
                          rope_cfg.orig_ctx, rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast,
                          rope_cfg.beta_slow};
                c.cvec = cvec_digest;
                c.switches = {
                    {"no_ple", o.no_ple}, {"native_bf16", o.native_bf16}, {"native_bf16_extra", o.native_bf16_extra},
                    {"native_ple_key", o.native_ple_key}, {"native_moe_combine", o.native_moe_combine},
                    {"native_gdn", o.native_gdn}, {"native_flash_attn_short", o.native_flash_attn_short},
                    {"native_qsa_indexer", o.native_qsa_indexer}, {"native_qsa", o.native_qsa},
                    {"native_rope", o.native_rope}, {"native_ple_postops", o.native_ple_postops},
                    {"native_router", o.native_router}, {"cpu_oracle_q8_0", o.cpu_oracle_q8_0},
                    {"gr_fp32_activations", o.gr_fp32_activations}, {"gr_native_mmvf", o.gr_native_mmvf},
                    {"native_preset", !o.native_preset.empty()}, {"shared_late", o.shared_late},
                    {"keep_canonical", o.keep_canonical}, {"no_fused_gr", o.no_fused_gr},
                    {"no_fast_attn", o.no_fast_attn}, {"no_fused_gdn", o.no_fused_gdn},
                    {"no_fast_select", o.no_fast_select}, {"vision", o.vision}};
                config_fp = strata::core::session_config_fingerprint(c);
            }
            id.model = *model_fp;
            id.config = *config_fp;
            return true;
        };
        // Save only on a switch/rewind, not on each continuing request. No graph
        // addresses change: all parked images live in ordinary host vectors.
        auto park_current_body = [&](size_t held) -> bool {
            if (!conversations.enabled() || !live_ok || live.empty()) return true;
            // #342: before make_room evicts oldest-first, the copies of this conversation a turn back go (they hold
            // nothing the outgoing chain does not, apart from the tail this conversation rewrote)
            if (const size_t dropped = conversations.drop_superseded(live, live_imgs, checks, cvec_cached))
                std::fprintf(stderr, "strata serve: conversation cache: dropped %zu superseded cop%s of this "
                             "conversation; parked=%zu\n", dropped, dropped == 1 ? "y" : "ies", conversations.size());
            // A layer split parks one image per stage: the first stage's (with the draft layer's K/V) and one per
            // later stage (without it - the draft ring is saved once).  The checkpoints are MOVED apart into their
            // stage parts for the capture and put back together whichever way this ends (no running state copied).
            const size_t n_st = stages.size();
            // the draft layer's K/V lives on the last stage's GPU (the drafter is loaded there): with a split it is
            // saved with that stage's image, under its device; stage 0's image holds none
            const strata::core::QsaState* draft0 = n_st > 0 || !use_mtp ? nullptr : &mtp.kv_state();   // no --mtp: no draft K/V (snapshots accept null)
            auto draft_of = [&](size_t k) -> const strata::core::QsaState* { return use_mtp && k + 1 == n_st ? &mtp.kv_state() : nullptr; };
            strata::core::ConversationCheckpointSplit cs;
            if (n_st > 0) cs = strata::core::conversation_checkpoints_split(std::move(checks), n_st);   // #752: two-phase; a bad_alloc leaves `checks` whole
            struct MergeBack {
                strata::core::ConversationCheckpointSplit& cs; std::vector<ConvCheckpoint>& checks; bool on;
                ~MergeBack() { if (on && !strata::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear(); }
            } merge_back{cs, checks, n_st > 0};
            const strata::core::ConversationView view{live, live_imgs, n_st > 0 ? cs.stage0 : checks, cvec_cached};
            auto reuse = conversations.take_reuse();
            std::vector<strata::core::ConversationKvReuse> stage_reuse = std::move(reuse.stages);
            stage_reuse.resize(n_st);
            reuse.stages.clear();
            // the later stages' estimates, each with its own retained K/V when it still fits
            size_t stage_estimate = 0, stage_retained = 0;
            for (size_t k = 0; k < n_st; ++k) {
                auto& st = stages[k];
                const strata::core::OnDevice on(st->dev);
                const strata::core::ConversationView view_k{live, live_imgs, cs.parts[k], cvec_cached};
                size_t b = 0;
                if (!strata::core::conversation_snapshot_bytes(view_k, st->ss, g, draft_of(k), b, err)) {
                    std::fprintf(stderr, "strata serve: conversation cache: skip parking (stage CUDA%d: %s)\n",
                                 st->dev, err.c_str());
                    err.clear();
                    return true;
                }
                if (!stage_reuse[k].kv.empty()) {
                    size_t r = 0;
                    if (strata::core::conversation_snapshot_capture_bytes(stage_reuse[k], view_k, st->ss, g, draft_of(k),
                                                                          r, err)) {
                        b = r;
                        stage_retained += stage_reuse[k].bytes();
                    } else {
                        stage_reuse[k] = {};
                        err.clear();
                    }
                }
                stage_estimate += b;
            }
            size_t estimate = 0;
            if (!strata::core::conversation_snapshot_bytes(view, ss, g, draft0, estimate, err)) {
                std::fprintf(stderr, "strata serve: conversation cache: skip parking (%s)\n", err.c_str());
                err.clear(); // A recoverable miss must not poison the batched draft prefill's error channel.
                return true;
            }
            const size_t fresh_estimate = estimate;
            if (!reuse.kv.empty() && !strata::core::conversation_snapshot_capture_bytes(
                    reuse, view, ss, g, draft0, estimate, err)) {
                reuse = {};
                estimate = fresh_estimate;
                err.clear();
            }
            estimate += stage_estimate;
            // A park carrying its own retained K/V replaces memory the cache
            // already held, so capacity is make_room's call - it runs next either
            // way, and put()'s accounting still bounds the budget. The with-reuse
            // estimate must stay uncapped: it counts the retained buffers'
            // capacity and directories, and put() charges that same true size -
            // a capped figure would under-evict and overfill the budget.
            if (!conversations.make_room(estimate, held)) {
                std::fprintf(stderr, "strata serve: conversation cache: skip parking (snapshot %zu MiB exceeds available budget)\n",
                             estimate >> 20);
                return true;
            }
            const auto t0 = Clock::now();
            try {
                const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib * 1024 * 1024;
                const size_t retained = reuse.bytes() + stage_retained;
                const size_t additional = estimate > retained ? estimate - retained : 0;
                if (!strata::core::conversation_memory_admit(strata::core::available_host_bytes(),
                        additional, floor)) {
                    std::fprintf(stderr, "strata serve: conversation cache: skip parking (physical RAM admission; need %zu MiB plus %lld MiB floor, or telemetry unavailable)\n",
                                 additional >> 20, (long long) o.conversation_cache_min_free_mib);
                    return true;
                }
                strata::core::SavedConversation image;
                size_t reused_bytes = 0;
                if (!strata::core::conversation_snapshot_save(image, view, ss, g, draft0, err,
                        std::move(reuse), &reused_bytes)) return false;
                for (size_t k = 0; k < n_st; ++k) {   // the later stages (the last one with the draft layer's K/V)
                    auto& st = stages[k];
                    const strata::core::OnDevice on(st->dev);
                    if (DPCT_CHECK_ERROR(dpct::get_current_device()
                                             .queues_wait_and_throw()) != 0) {
                        err = "stage sync failed"; return false;
                    }
                    const strata::core::ConversationView view_k{live, live_imgs, cs.parts[k], cvec_cached};
                    strata::core::SavedConversation part;
                    if (!strata::core::conversation_snapshot_save(part, view_k, st->ss, g, draft_of(k), err,
                            std::move(stage_reuse[k]), &reused_bytes))
                        return false;
                    image.stage_images.push_back(std::move(part));
                }
                if (!strata::core::conversation_memory_admit(strata::core::available_host_bytes(), 0, floor)) {
                    std::fprintf(stderr, "strata serve: conversation cache: skip parking (physical RAM floor after capture, or telemetry unavailable)\n");
                    return true;
                }
                const size_t snapshot_bytes = image.bytes();
                const bool stored = conversations.put(std::move(image), held);
                std::fprintf(stderr, "strata serve: conversation cache: %s %zu tokens in %.1f ms; parked=%zu bytes=%zu evictions=%zu snapshot_bytes=%zu reused_kv_bytes=%zu\n",
                             stored ? "parked" : "skipped", live.size(),
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                             conversations.size(), conversations.bytes(), conversations.evictions(), snapshot_bytes, reused_bytes);
            } catch (const std::bad_alloc&) {
                // The active state has not been touched. Continue with normal
                // prompt processing rather than killing a serving process.
                std::fprintf(stderr, "strata serve: conversation cache: allocation failed; skip parking\n");
            }
            return true;
        };
        // #752: the whole park (the checkpoint split included) is under one try, so a host OOM skips parking instead of
        // ending the server
        auto park_current = [&](size_t held) -> bool {
            try { return park_current_body(held); }
            catch (const std::bad_alloc&) {
                std::fprintf(stderr, "strata serve: conversation cache: allocation failed; skip parking\n");
                return true;
            }
        };
        int64_t pp_total = 0, pp_from = 0, pp_next_check = 0;
        bool pp_tail_saved = false;
        // #471: the position the prompt pass has read up to (a chunk's or a window's end): what a request cancelled
        // mid-read reports as read, instead of the whole prompt
        int64_t pp_reached = 0;
        Clock::time_point pp_t0 = Clock::now();
        auto imgs_below = [&](const std::vector<ImgKey>& all, int64_t L) {
            std::vector<ImgKey> v;
            for (const ImgKey& k : all) if (k.start < L) v.push_back(k);
            return v;
        };
        // a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed
        // A layer split's mid-prompt checkpoints: when the last stage reports a chunk, the earlier ones already read
        // the next, so each stage saves its own part when IT reaches a checkpoint position (the same rule as below:
        // every `prompt_cache_every` tokens from where the request resumed), and the last stage puts them together.
        std::mutex part_mu;
        std::map<int64_t, std::vector<ConvCheckpoint>> part_at;   // position -> one part per stage
        std::vector<int64_t> part_next(stages.size() + 1, INT64_MAX);
        // a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed.  `parts`: the stages'
        // states saved at L (a split's mid-prompt checkpoint); without, they are read now (everything is at L)
        // #613: why the last checkpoint_at failed - a failed device sync is the GPU itself (a hang Windows then resets),
        // not the checkpoint, and the error says so
        std::string ckpt_why;
        auto gpu_sync_ok = [&]() -> bool {
            try {
        const dpct::err0 e = DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw());
            if (e == 0) return true;
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            ckpt_why = std::string(": the GPU stopped responding before it (") +
                       dpct::get_error_string_dummy(e) +
                       ") - a GPU hang; on Windows the driver is then reset";
            return false;
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        auto checkpoint_at = [&](int64_t L, std::vector<ConvCheckpoint>* parts = nullptr, bool as_tail = false) -> bool {
            ckpt_why.clear();
            if (o.prompt_cache <= 0 || L < 1) return true;
            for (ConvCheckpoint& c : checks)
                if ((int64_t) c.ids.size() == L) {
                    c.used = ++check_clock;
                    if (!as_tail && L == tail_ckpt_len) tail_ckpt_len = -1;   // a periodic one lands here: a normal item now
                    return true;
                }
            ConvCheckpoint c;
            c.ids.assign(cur.begin(), cur.begin() + L);
            c.imgs = imgs_below(req_imgs, L);
            if (parts != nullptr) {
                if (parts->size() != stages.size() + 1) return false;
                c.gdn = std::move((*parts)[0].gdn);
                c.ple = std::move((*parts)[0].ple);
                c.tails = std::move((*parts)[0].tails);
                c.dead = std::move((*parts)[0].dead);
                c.block_pos = std::move((*parts)[0].block_pos);
                for (size_t i = 1; i < parts->size(); ++i) c.stage_parts.push_back(std::move((*parts)[i]));
            } else {
                if (!gpu_sync_ok() || !checkpoint_save(c, ss, g)) return false;
                for (auto& st : stages) {   // a layer split's later stages: their sessions' part
                    const strata::core::OnDevice on(st->dev);
                    ConvCheckpoint part;
                    part.ids = c.ids;
                    if (!gpu_sync_ok() || !checkpoint_save(part, st->ss, g)) return false;
                    c.stage_parts.push_back(std::move(part));
                }
            }
            c.used = ++check_clock;
            if (as_tail) {   // only one tail checkpoint stays alive: the previous request's goes
                const int64_t old_tail = tail_ckpt_len;
                checks.erase(std::remove_if(checks.begin(), checks.end(), [&](const ConvCheckpoint& k) {
                                 return old_tail >= 0 && (int64_t) k.ids.size() == old_tail;
                             }), checks.end());
                tail_ckpt_len = L;
            } else if (L == tail_ckpt_len) {
                tail_ckpt_len = -1;   // the flagged tail was dropped and a normal checkpoint takes its length
            }
            checks.push_back(std::move(c));
            while ((int) checks.size() > o.prompt_cache) {
                std::vector<uint64_t> stamps;
                std::vector<char> is_tail;
                stamps.reserve(checks.size());
                for (const ConvCheckpoint& k : checks) {
                    stamps.push_back(k.used);
                    is_tail.push_back(tail_ckpt_len >= 0 && (int64_t) k.ids.size() == tail_ckpt_len);
                }
                // the tail checkpoint serves only a branch of the last request: it leaves before a periodic one
                std::unique_ptr<bool[]> tail_flags(new bool[is_tail.size()]);
                for (size_t i = 0; i < is_tail.size(); ++i) tail_flags[i] = is_tail[i] != 0;
                const size_t victim = strata::program::conv_cache::eviction_victim(
                    stamps.data(), stamps.size(), o.prompt_cache, tail_flags.get());
                checks.erase(checks.begin() + (std::ptrdiff_t) victim);
            }
            return true;
        };
        sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
            std::vector<int32_t> nxt((size_t) T);
            for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) cur[(size_t) (p0 + t + 1)];
            // E-9: batched through the prompt path when it can (one GPU: a layer split's drafter is on the last stage)
            const bool batched = use_mtp && !multi_gpu && sp.draft_kv(mtp, R_rows, nxt.data(), T, p0, e);
            if (!e.empty() || (use_mtp && !batched && !mtp.prefill(R_rows, nxt.data(), T, p0, e))) return false;
            if (use_mtp && std::getenv("STRATA_SNAPSHOT_VERIFY") != nullptr)
                std::fprintf(stderr, "strata serve: DRAFT_PREFILL path=%s mode=%d cells=%lld\n",
                             batched ? "batched" : "token", mtp.kv_state().kv_mode, (long long) T);
            // progress for the server window: PP <position reached> <prompt tokens> <ms> <fresh tokens/s>
            const int64_t done = p0 + T;
            pp_reached = done;
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) pp_total, ms,
                        ms > 0.0 ? 1000.0 * (double) (done - pp_from) / ms : 0.0);
            strata::core::progress_at("reading the prompt (batched), done up to token", done);
            strata::core::progress_beat();
            std::fflush(stdout);
            const bool periodic_checkpoint = o.prompt_cache_every > 0 && done >= pp_next_check;
            // Keep the prefill's existing chunk geometry. One extra near-tail checkpoint
            // can serve a branch whose shared prefix ends before the final cached state.
            const bool tail_checkpoint = o.prompt_cache_tail && o.prompt_cache > 0 && o.prompt_cache_every > 0 &&
                !multi_gpu && !pp_tail_saved && pp_total - done > std::max<int64_t>(1, o.short_read) &&
                pp_total - done <= T;   // this chunk's real size (--prefill auto chunks differ from o.prefill_chunk)
            if (periodic_checkpoint || tail_checkpoint) {
                bool saved = false;
                if (multi_gpu) {   // the stages' parts, saved when each of them read this chunk
                    std::vector<ConvCheckpoint> parts;
                    {
                        std::lock_guard<std::mutex> lk(part_mu);
                        auto it = part_at.find(done);
                        if (it != part_at.end()) parts = std::move(it->second);
                        part_at.erase(part_at.begin(), part_at.upper_bound(done));
                    }
                    bool complete = parts.size() == stages.size() + 1;
                    for (const ConvCheckpoint& k : parts) complete = complete && !k.gdn.empty();
                    saved = !complete || checkpoint_at(done, &parts);   // an incomplete set: no checkpoint here
                } else {
                    saved = checkpoint_at(done, nullptr, tail_checkpoint && !periodic_checkpoint);
                }
                if (!saved) { e = "saving a conversation checkpoint failed" + ckpt_why; return false; }
                if (periodic_checkpoint) pp_next_check = done + o.prompt_cache_every;
                if (tail_checkpoint) pp_tail_saved = true;
            }
            return true;
        };
        if (multi_gpu) {   // the batched prompt is reported by its last stage (the drafter's rows are there)
            stages.back()->sp.on_chunk = std::move(sp.on_chunk);
            sp.on_chunk = nullptr;
            for (size_t i = 0; i <= stages.size(); ++i) {
                strata::prefill::Prefill& stage_sp = i == 0 ? sp : stages[i - 1]->sp;
                strata::core::SessionState& stage_ss = i == 0 ? ss : stages[i - 1]->ss;
                stage_sp.on_stage_chunk = [&, i](int64_t done, std::string& e) -> bool {
                    if (o.prompt_cache <= 0 || o.prompt_cache_every <= 0 || done < part_next[i]) return true;
                    part_next[i] = done + o.prompt_cache_every;
                    ConvCheckpoint part;   // this stage's state at `done` (its stream is synchronized)
                    part.ids.assign(cur.begin(), cur.begin() + done);
                    if (!checkpoint_save(part, stage_ss, g)) { e = "saving a checkpoint part failed"; return false; }
                    std::lock_guard<std::mutex> lk(part_mu);
                    auto& v = part_at[done];
                    v.resize(stages.size() + 1);
                    v[i] = std::move(part);
                    return true;
                };
            }
        }
        drive.d.plan = ver.plan_sink();
        drive.d.pcie_num = std::max(0, std::min(256, (int) (o.pcie_frac * 256.0 + 0.5)));
        const bool all_experts_resident = !host_res.empty() &&
            std::all_of(host_res.begin(), host_res.end(), [](int32_t r) { return r >= 0; });
        if (o.adapt_every > 0 && o.adapt_swaps > 0 && !all_experts_resident)
            drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
        PcieAuto pcie_auto;
        if (o.pcie_frac_auto && !all_experts_resident && o.batch == 0 && n_stages == 1) {
            const auto& pl = strata::kernels::cpu::expert_layout();
            double sum = 0.0;
            for (int64_t l = 0; l < g.n_layers; ++l) sum += (double) pl.blob_bytes(l);
            pcie_auto.on = true;
            pcie_auto.link_gbps = o.pcie_probe_gbps;
            pcie_auto.blob_bytes = g.n_layers > 0 ? sum / (double) g.n_layers : 0.0;
        }
        auto pcie_auto_step = [&](double& req_pcie_frac) {
            if (!pcie_auto.on) return;
            double tc_us = 0.0, rate = 0.0;
            const double nf = req_pcie_frac == o.pcie_frac
                                  ? pcie_auto.window(drive.d.ms_run, drive.d.multi_misses, o.pcie_frac, &tc_us, &rate)
                                  : (pcie_auto.mark(drive.d.ms_run, drive.d.multi_misses), -1.0);
            if (nf < 0.0) return;
            std::fprintf(stderr, "strata serve: PCIe share: the CPU pool takes %.0f us per missed expert (%.1f GB/s of expert bytes), "
                                 "the link %.1f GB/s -> pcie_frac %.2f (was %.2f; --pcie-frac N fixes it)\n",
                         tc_us, rate, pcie_auto.link_gbps, nf, o.pcie_frac);
            o.pcie_frac = nf;
            req_pcie_frac = nf;
            drive.d.pcie_num = std::max(0, std::min(256, (int) (nf * 256.0 + 0.5)));
        };
        // #477 --expert-profile-save: what the adaptive tier learned, kept across restarts (opt-in; off: `heat` stays
        // empty and nothing below runs).  It needs the adaptive tier's counts and the residency table.
        std::vector<double> heat;
        if (!o.expert_profile_save.empty() && !all_experts_resident) {
            if (usage_dev_srv != nullptr) {
                // the device plan's routing counts (STRATA_VERIFY_NO_HOST): read when the profile is written
            } else if (drive.d.usage.empty() || host_res.empty())
                std::fprintf(stderr, "strata serve: --expert-profile-save needs the adaptive tier (--adapt-every and "
                                     "--adapt-swaps above 0) and --expert-profile: nothing will be saved\n");
            else
                heat.assign(drive.d.usage.size(), 0.0);
        }
        Clock::time_point profile_saved_at = Clock::now();
        dpct::queue_ptr adapt_stream = &dpct::get_in_order_queue();
        /*
        DPCT1025: The SYCL queue is created ignoring the flag and priority
        options.
        */
        if (DPCT_CHECK_ERROR(
                adapt_stream = dpct::get_current_device().create_queue(true)) !=
            0) {
            std::fprintf(stderr, "strata serve: cannot create the refill stream\n");
            return 1;
        }
        // plan v0.3 P6: swaps in flight - (residency index, slot) admitted when adapt_ev has completed
        std::vector<std::pair<int32_t, int32_t>> pending;
        int pending_age = 0;   // the windows the pending swaps have waited (adapt_lag)
        std::vector<void*> pin_live;   // the swaps' locked arena pages (pin_blob), unlocked once they have landed
        dpct::event_ptr adapt_ev = nullptr;
        adapt_ev = new sycl::event();
        // a layer split's later stages keep a copy of the residency table on their devices, and swap on their own
        auto res_upload = [&]() {
            try {
        if (d_res != nullptr)
                res_put(d_res);
            for (auto& st : stages) {
                const strata::core::OnDevice on(st->dev);
                res_put(st->d_res);
            }
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        auto apply_pending = [&](bool wait) {
            try {
        if (peer.valid()) peer.apply_pending(wait);
            if (pending.empty()) return;
            if (wait) adapt_ev->wait_and_throw();
            else if (adapt_ev->get_info<
                         sycl::info::event::command_execution_status>() !=
                     sycl::info::event_command_status::complete) return;
            for (auto& st : stages)
                if (st->adapt_live) {
                    if (wait) st->adapt_ev->wait_and_throw();
                    else if ((st->adapt_ev)
                                 ->get_info<sycl::info::event::
                                                command_execution_status>() !=
                             sycl::info::event_command_status::complete) return;
                }
            for (auto& st : stages) st->adapt_live = false;
            unpin_blobs(pin_live);
            src.commit_exchanges();   // the resident RAM mode: the evicted experts take their places in RAM
            for (const auto& [i, slot] : pending) {
                host_res[(size_t) i] = slot;
                srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);   // in VRAM now: RAM not needed
            }
            pending.clear();
            pending_age = 0;
            res_upload();
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        // --pipeline-windows: set while the tier adapts beside windows in flight (see its use in adapt)
        std::function<void()> adapt_fence;
        // the VRAM tier follows the conversation (the same rule as the speculative loop below)
        auto adapt = [&]() -> bool {
            if (!pending.empty()) return true;   // the previous swaps are still in flight
            if (remote_opt && !remote_opt->adapt(drive.d.usage, host_res, pending, o.adapt_swaps, *srcp)) return false;
            struct Swap { float gain; int32_t layer, in, out; };
            std::vector<Swap> swaps;
            std::vector<std::pair<float, int32_t>> cand, vict;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                cand.clear();
                vict.clear();
                const float* u = drive.d.usage.data() + l * g.n_expert;
                const int32_t* r = host_res.data() + l * g.n_expert;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    if (r[e] < 0) { if (u[e] >= 2.0f && !(peer.valid() && peer.has(l, e)) && !(remote_opt && remote_opt->owns(l, e)) && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
                    else vict.emplace_back(u[e], e);
                }
                if (cand.empty() || vict.empty()) continue;
                std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
                const size_t nc = std::min(cand.size(), vict.size());
                std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                                  [](auto& a, auto& b) { return a.first < b.first; });
                for (size_t i = 0; i < nc; ++i) {
                    if (cand[i].first < vict[i].first + 1.5f) break;
                    swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
                }
            }
            std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
            if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
            // the copies back read the cache of the card that owns each layer (see resident_stage_swaps)
            auto home_of = [&](int64_t layer) -> SwapHome {
                const int stn = multi_gpu ? stage_of(layer) : 0;
                if (stn > 0) {
                    GpuStage* gs = stages[(size_t) stn - 1].get();
                    return {&gs->cache, gs->adapt_stream, gs->dev};
                }
                return {&xcache, adapt_stream, -1};
            };
            if (!resident_stage_swaps(src, host_res, g.n_expert, swaps, home_of)) return false;
            if (pin_blobs_on()) {   // lock the batch's source pages (a file-backed arena on AMD; see pin_blobs)
                std::vector<std::pair<uintptr_t, uintptr_t>> spans;
                for (const Swap& s : swaps)
                    if (const uint8_t* b = srcp->blob(s.layer, s.in))
                        spans.emplace_back((uintptr_t) b,
                                           (uintptr_t) b + strata::kernels::cpu::expert_layout().blob_bytes(s.layer));
                pin_blobs(std::move(spans), pin_live);
            }
            // --pipeline-windows: windows are in flight while the tier adapts.  The evicted experts stop being planned
            // for the GPU first (the pool computes them on the CPU from the rows every layer publishes, see
            // Verifier::set_always_publish); then every copy into their slots waits until all the work the stages'
            // streams hold now has completed: any window that may still read those slots was launched before this.
            std::vector<int32_t> fenced_slots;
            if (adapt_fence && !swaps.empty()) {
                for (const Swap& s : swaps) {
                    const size_t out = (size_t) s.layer * g.n_expert + s.out;
                    fenced_slots.push_back(host_res[out]);
                    host_res[out] = strata::core::kNotResident;
                }
                std::atomic_thread_fence(std::memory_order_seq_cst);
                adapt_fence();
            }
            bool main_live = false;
            // STRATA_DMA_BATCH (F12): the main device's copies as one cudaMemcpyBatchAsync (mode 0: one each)
            const int dma_mode = multi_gpu ? 0 : strata::core::dma_batch_mode();
            std::vector<void*> cp_dst;
            std::vector<const void*> cp_src;
            std::vector<size_t> cp_bytes;
            for (size_t si = 0; si < swaps.size(); ++si) {
                const Swap& s = swaps[si];
                const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
                const int32_t slot = fenced_slots.empty() ? host_res[out] : fenced_slots[si];
                const uint8_t* b = srcp->blob(s.layer, s.in);
                const int stn = multi_gpu ? stage_of(s.layer) : 0;   // the swap stays in the layer's own cache
                GpuStage* gs = stn > 0 ? stages[(size_t) stn - 1].get() : nullptr;
                const strata::core::OnDevice on(gs ? gs->dev : -1);
                if (dma_mode != 0 && slot >= 0 && b != nullptr) {
                    cp_dst.push_back(xcache.device_slot(slot));
                    cp_src.push_back(b);
                    cp_bytes.push_back((size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer));
                    main_live = true;
                    host_res[out] = strata::core::kNotResident;
                    srcp->prefetch(s.layer, s.out);
                    pending.emplace_back((int32_t) in, slot);
                    continue;
                }
                if (slot < 0 || b == nullptr ||
                    adapt_copy_h2d(gs ? gs->cache.device_slot(slot)
                                      : xcache.device_slot(slot),
                                   b,
                                   (size_t)strata::kernels::cpu::expert_layout()
                                       .blob_bytes(s.layer),
                                   gs ? gs->adapt_stream : adapt_stream) != 0) {
                    std::fprintf(stderr,
                                 "strata serve: adaptive swap copy failed "
                                 "(layer %d, slot %d, pinned %d): %s\n",
                                 /*
                                 DPCT1009: SYCL reports errors using
                                 exceptions and does not use error codes. Please
                                 replace the "get_error_string_dummy(...)" with
                                 a real error-handling function.
                                 */
                                 /*
                                 DPCT1010: SYCL uses exceptions to report
                                 errors and does not use the error codes. The
                                 cudaGetLastError function call was replaced
                                 with 0. You need to rewrite this code.
                                 */
                                 (int)s.layer, (int)slot,
                                 pin_live.empty() ? 0 : 1,
                                 dpct::get_error_string_dummy(0));
                    return false;
                }
                if (gs) gs->adapt_live = true;
                else main_live = true;
                host_res[out] = strata::core::kNotResident;   // evicted now: the CPU computes it meanwhile
                srcp->prefetch(s.layer, s.out);   // a file-backed arena released its pages: read them back ahead
                pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
            }
            if (!cp_dst.empty() &&
                strata::core::copy_blobs(cp_dst.data(), cp_src.data(),
                                         cp_bytes.data(), cp_dst.size(),
                                         adapt_stream, dma_mode) != 0) {
                std::fprintf(
                    stderr,
                    "strata serve: adaptive swap copy failed (batched): %s\n",
                    /*
                    DPCT1009: SYCL reports errors using exceptions and does
                    not use error codes. Please replace the
                    "get_error_string_dummy(...)" with a real error-handling
                    function.
                    */
                    /*
                    DPCT1010: SYCL uses exceptions to report errors and
                    does not use the error codes. The cudaGetLastError function
                    call was replaced with 0. You need to rewrite this code.
                    */
                    dpct::get_error_string_dummy(0));
                return false;
            }
            if (!swaps.empty()) dpct::sync_barrier(adapt_ev, adapt_stream);
            (void) main_live;
            for (auto& st : stages)
                if (st->adapt_live) {
                    const strata::core::OnDevice on(st->dev);
                    dpct::sync_barrier(st->adapt_ev, st->adapt_stream);
                }
            // #477: the routing counted since the start (each count adds up to 1 / (1 - --adapt-decay) over its
            // decays: the sum is proportional to the routing itself) - only with --expert-profile-save, else `heat`
            // is empty
            for (size_t i = 0; i < heat.size(); ++i) heat[i] += (double) drive.d.usage[i];
            if (peer.valid()) {                   // multi-GPU: the peer takes the next most-routed CPU misses
                std::vector<int32_t> r0 = host_res;
                for (const auto& [i, slot] : pending) r0[(size_t) i] = slot;   // swapped into the primary already
                std::string perr;
                if (!peer.adapt(drive.d.usage.data(), r0.data(),
                                o.peer_adapt_swaps >= 0 ? o.peer_adapt_swaps : o.adapt_swaps, perr)) {
                    std::fprintf(stderr, "strata serve: %s\n", perr.c_str());
                    return false;
                }
            }
            for (float& v : drive.d.usage) v *= o.adapt_decay;
            return true;
        };
        // ---- --adapt-async 1: the asynchronous adaptive tier (opt-in; the resident RAM mode, see the checks where
        // the RAM copy is built).  adapt() runs a whole round - choose, copy the evicted slots back with a stream sync,
        // copy the new experts in from the RAM copy - on a thread the decode loop joins after the draft: ~24 ms every
        // 4th window with ~90 swaps.  Here a round moves through four steps, each started by `adapt_tick` between
        // windows once the previous step's copies (events) and host work (the job thread) are done:
        //   1. job: choose (adapt()'s rule, over snapshots of the routing counts and the residency) and copy the
        //      evicted slots back into exchange buffers (D2H), each from the cache of the card that owns its layer;
        //   2. stage the exchanges and mark the evicted experts non-resident (the CPU reads them from the exchange
        //      buffers from the next window on); job: copy the new experts in (straight from a page-locked RAM copy,
        //      else through a pinned bounce buffer), on the owning card's refill stream;
        //   3. mark the new experts resident; job: move the evicted blobs into the RAM places the new ones left;
        //   4. flip the copy's offsets.
        // A slot is overwritten only after its old expert stopped being planned for the GPU (step 2, between windows,
        // with the device copies of the residency table uploaded as apply_pending does), and a RAM place only once its
        // expert is resident (step 3), so no window computes from memory that is changing.  A request starts with the
        // round drained (adapt_tick(true)), before its prompt lends any slot.  Not bit-exact run to run: which window
        // first computes a swapped-in expert on the GPU (which rounds differently from the CPU) depends on when its
        // copy lands.  `j`: the swap's exchange buffer (its copy back) and bounce buffer (a_cap + j).
        struct ASwap { int32_t layer, in, out, slot; int home; int64_t j; bool exchange; };
        struct AHome {
            strata::core::ExpertCache *cache; dpct::queue_ptr stream;
            dpct::event_ptr ev;
            int dev;
            bool used;
        };
        enum class AState { Idle, CopyBack, SwapIn, Commit };
        AState astate = AState::Idle;
        std::vector<ASwap> aswaps;
        std::vector<float> a_usage;
        std::vector<int32_t> a_res;
        int64_t a_win = 0, a_last = 0, a_rounds = 0, a_swapped = 0;
        double a_ms = 0;                  // wall time from a round's start to its flip, summed (the "ms/round" stat)
        Clock::time_point a_t0;
        std::atomic<bool> a_err{false};
        std::vector<AHome> ahomes;   // [0] = CUDA0's cache, [k] = layer split stage k's
        std::unique_ptr<JobThread> ajob;
        const int64_t a_cap = std::min<int64_t>(o.adapt_swaps, 96);   // as the blocking tier's exchange buffers
        if (o.adapt_async && !drive.d.usage.empty() && src.complement_ready() && src.exchange_capacity() >= 2 * a_cap) {
            ahomes.push_back({&xcache, adapt_stream, adapt_ev, 0, false});
            for (auto& st : stages) ahomes.push_back({&st->cache, st->adapt_stream, st->adapt_ev, st->dev, false});
            ajob = std::make_unique<JobThread>();
            std::fprintf(stderr, "strata serve: adaptive tier asynchronous (--adapt-async 1): up to %lld swaps per round, "
                                 "a round every %d windows\n", (long long) a_cap, o.adapt_every);
        }
        auto a_flush = [&]() { // the job's copies: an event after them on every
                               // stream used, submitted now (WDDM)
            try {
        for (AHome &h : ahomes)
                if (h.used) {
                    const strata::core::OnDevice on(h.dev);
                    /*
                    DPCT1024: The original code returned the error code
                    that was further consumed by the program logic. This
                    original code was replaced with 0. You may need to rewrite
                    the program logic consuming the error code.
                    */
                    if (DPCT_CHECK_ERROR(dpct::sync_barrier(h.ev, h.stream)) !=
                        0) a_err = true;
                    (void)DPCT_CHECK_ERROR(((h.stream)->ext_oneapi_empty()));
                }
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        auto a_choose = [&]() { // step 1, on the job thread
            try {
        struct C { float gain; int32_t layer, in, out; };
            std::vector<C> all;
            std::vector<std::pair<float, int32_t>> cand, vict;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                cand.clear();
                vict.clear();
                const float* u = a_usage.data() + l * g.n_expert;
                const int32_t* r = a_res.data() + l * g.n_expert;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    if (r[e] < 0) { if (u[e] >= 2.0f && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
                    else vict.emplace_back(u[e], e);
                }
                if (cand.empty() || vict.empty()) continue;
                std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
                const size_t nc = std::min(cand.size(), vict.size());
                std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                                  [](auto& a, auto& b) { return a.first < b.first; });
                for (size_t i = 0; i < nc; ++i) {
                    if (cand[i].first < vict[i].first + 1.5f) break;
                    all.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
                }
            }
            std::sort(all.begin(), all.end(), [](const C& a, const C& b) { return a.gain > b.gain; });
            aswaps.clear();
            for (const C& c : all) {
                if ((int64_t) aswaps.size() >= a_cap) break;
                const int32_t slot = a_res[(size_t) c.layer * g.n_expert + c.out];
                if (slot < 0) continue;
                // an exchange (as resident_stage_swaps): `in` from the RAM copy, `out` held by its slot alone takes its
                // place there; otherwise `out` is in the copy already (the lend region) or `in` is not (a RAM budget)
                const bool x = src.has_resident(c.layer, c.in) && !src.has_resident(c.layer, c.out);
                aswaps.push_back({c.layer, c.in, c.out, slot, multi_gpu ? stage_of(c.layer) : 0, (int64_t) aswaps.size(), x});
            }
            for (AHome& h : ahomes) h.used = false;
            for (const ASwap& w : aswaps) {
                if (!w.exchange) continue;
                AHome& h = ahomes[(size_t) w.home];
                const strata::core::OnDevice on(h.dev);
                /*
                DPCT1000: Error handling if-stmt was detected but could not
                be rewritten.
                */
                /*
                DPCT1124: cudaMemcpyAsync is migrated to asynchronous
                memcpy API. While the origin API might be synchronous, it
                depends on the type of operand memory, so you may need to call
                wait() on event return by memcpy API to ensure synchronization
                behavior.
                */
                if (const dpct::err0 e = DPCT_CHECK_ERROR(h.stream->memcpy(
                        src.exchange_buffer(w.j), h.cache->device_slot(w.slot),
                        (size_t)strata::kernels::cpu::expert_layout()
                            .blob_bytes(w.layer)));
                    e != 0) {
                    std::fprintf(stderr,
                                 "strata serve: adaptive tier: copy back of "
                                 "layer %d slot %d (CUDA%d) failed: %s\n",
                                 /*
                                 DPCT1009: SYCL reports errors using
                                 exceptions and does not use error codes. Please
                                 replace the "get_error_string_dummy(...)" with
                                 a real error-handling function.
                                 */
                                 (int)w.layer, (int)w.slot, h.dev,
                                 dpct::get_error_string_dummy(e));
                    /*
                    DPCT1001: The statement could not be removed.
                    */
                    a_err = true;
                }
                h.used = true;
            }
            a_flush();
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        auto a_copy_in = [&]() { // step 2, on the job thread
            try {
        for (AHome &h : ahomes) h.used = false;
            for (const ASwap& w : aswaps) {
                AHome& h = ahomes[(size_t) w.home];
                const size_t nb = (size_t) strata::kernels::cpu::expert_layout().blob_bytes(w.layer);
                const uint8_t* b = src.pinned(w.layer, w.in) ? src.resident_blob(w.layer, w.in) : nullptr;
                if (b == nullptr) {   // pageable, or not in the RAM copy: through a bounce buffer (an async copy)
                    uint8_t* bb = src.exchange_buffer(a_cap + w.j);
                    if (bb == nullptr || !src.copy_blob(w.layer, w.in, bb)) {
                        std::fprintf(stderr, "strata serve: adaptive tier: layer %d expert %d could not be read\n",
                                     (int) w.layer, (int) w.in);
                        a_err = true;
                        continue;
                    }
                    b = bb;
                }
                const strata::core::OnDevice on(h.dev);
                /*
                DPCT1000: Error handling if-stmt was detected but could not
                be rewritten.
                */
                /*
                DPCT1124: cudaMemcpyAsync is migrated to asynchronous
                memcpy API. While the origin API might be synchronous, it
                depends on the type of operand memory, so you may need to call
                wait() on event return by memcpy API to ensure synchronization
                behavior.
                */
                if (const dpct::err0 e = DPCT_CHECK_ERROR(
                        h.stream->memcpy(h.cache->device_slot(w.slot), b, nb));
                    e != 0) {
                    std::fprintf(stderr,
                                 "strata serve: adaptive tier: copy of layer "
                                 "%d into slot %d (CUDA%d) failed: %s\n",
                                 /*
                                 DPCT1009: SYCL reports errors using
                                 exceptions and does not use error codes. Please
                                 replace the "get_error_string_dummy(...)" with
                                 a real error-handling function.
                                 */
                                 (int)w.layer, (int)w.slot, h.dev,
                                 dpct::get_error_string_dummy(e));
                    /*
                    DPCT1001: The statement could not be removed.
                    */
                    a_err = true;
                }
                h.used = true;
            }
            a_flush();
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        auto a_ready =
            [&](bool wait) -> bool { // the job and its copies are done
            try {
        if (wait) ajob->wait();
            else if (!ajob->idle()) return false;
            for (AHome& h : ahomes)
                if (h.used) {
                    const strata::core::OnDevice on(h.dev);
                    if (wait) {
                        if (DPCT_CHECK_ERROR(h.ev->wait_and_throw()) != 0)
                            a_err = true;
                    } else if (const dpct::err0 q =
                                   dpct::sycl_event_query(h.ev);
                               q == 1) return false;
                    /*
                    DPCT1001: The statement could not be removed.
                    */
                    /*
                    DPCT1000: Error handling if-stmt was detected but could
                    not be rewritten.
                    */
                    else if (q != 0) a_err = true;
                }
            return true;
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        // between windows: `drain` finishes the round in flight and starts none.  False when a copy failed.
        auto adapt_tick = [&](bool drain) -> bool {
            if (!ajob) return true;
            if (!drain) ++a_win;   // one tick per window
            for (;;) {
                if (a_err.load()) return false;
                switch (astate) {
                case AState::Idle:
                    if (drain || a_win - a_last < (int64_t) o.adapt_every) return true;
                    a_last = a_win;
                    ++a_rounds;
                    a_t0 = Clock::now();
                    a_usage = drive.d.usage;
                    a_res = host_res;
                    // #477: the routing counted, as adapt() counts it (--expert-profile-save; else `heat` is empty)
                    for (size_t i = 0; i < heat.size(); ++i) heat[i] += (double) drive.d.usage[i];
                    for (float& v : drive.d.usage) v *= o.adapt_decay;
                    ajob->post(a_choose);
                    astate = AState::CopyBack;
                    return true;
                case AState::CopyBack: {   // step 2: the evicted experts leave the GPU's plan
                    if (!a_ready(drain)) return true;
                    if (a_err.load()) return false;
                    size_t k = 0;
                    for (size_t i = 0; i < aswaps.size(); ++i) {
                        const ASwap w = aswaps[i];
                        const size_t out = (size_t) w.layer * g.n_expert + w.out;
                        if (host_res[out] != w.slot) continue;   // the table moved since the snapshot: not this one
                        if (w.exchange && !src.stage_exchange(w.layer, w.in, w.out, w.j)) {
                            std::fprintf(stderr, "strata serve: adaptive tier: stage_exchange refused layer %d in %d out %d\n",
                                         (int) w.layer, (int) w.in, (int) w.out);
                            return false;
                        }
                        host_res[out] = strata::core::kNotResident;   // the CPU computes it from the next window on
                        srcp->prefetch(w.layer, w.out);
                        aswaps[k++] = w;
                    }
                    aswaps.resize(k);
                    res_upload();
                    ajob->post(a_copy_in);
                    astate = AState::SwapIn;
                    if (!drain) return true;
                    continue;
                }
                case AState::SwapIn:   // step 3: the new experts are in their slots
                    if (!a_ready(drain)) return true;
                    if (a_err.load()) return false;
                    for (const ASwap& w : aswaps) {
                        host_res[(size_t) w.layer * g.n_expert + w.in] = w.slot;
                        srcp->release(w.layer, w.in);
                    }
                    res_upload();
                    a_swapped += (int64_t) aswaps.size();
                    ajob->post([&] { src.commit_copies(); });
                    astate = AState::Commit;
                    if (!drain) return true;
                    continue;
                case AState::Commit:   // step 4
                    if (!a_ready(drain)) return true;
                    if (a_err.load()) return false;
                    src.commit_flip();
                    a_ms += std::chrono::duration<double, std::milli>(Clock::now() - a_t0).count();
                    aswaps.clear();
                    astate = AState::Idle;
                    continue;   // a round due now starts at this same boundary
                }
            }
        };
        // #477: write the learned profile (between requests and at QUIT: a prompt's lent slots are back by then).
        // A swap still in flight counts as done - its expert is resident once the copy lands.  `why`: for the log.
        auto save_profile = [&](const char* why) {
            if (usage_dev_srv != nullptr) {
                // The cache and the mirror are fixed (no swaps under NO_HOST), so every pair is ranked by its device
                // count mixed with the start's ranking (a short conversation must not wipe the prior).
                const size_t n = (size_t) (g.n_layers * g.n_expert);
                std::vector<int32_t> uh(n);
                dpct::get_in_order_queue().memcpy(uh.data(), usage_dev_srv, n * 4).wait();
                double total = 0;
                for (int32_t c : uh) total += (double) c;
                // prior mass by rank: exponential with lambda 2 (the shipped ranking holds ~70% of the routing in its
                // first 47% of pairs, section 8 of notes-143-intel), worth `prior_entries` routed entries (800k: about one 150-token x 8 prompt session; 60k lost 36% on a different kind of text, 800k lost 18% and kept +13..19% on the trained kind)
                const double lam = 2.0;
                double prior_entries = 800000.0;
                if (const char* pe = std::getenv("STRATA_PROFILE_PRIOR_ENTRIES")) prior_entries = std::max(0.0, std::atof(pe));
                std::vector<double> hv(n, 0.0);
                for (size_t i = 0; i < n; ++i) hv[i] = (double) uh[i];
                for (size_t r = 0; r < profile_loaded.size(); ++r) {
                    const auto [l, e] = profile_loaded[r];
                    if (l < 0 || l >= g.n_layers || e < 0 || e >= g.n_expert) continue;
                    const double x = (double) r / (double) n;
                    hv[(size_t) (l * g.n_expert + e)] += prior_entries * lam * std::exp(-lam * x) / (1.0 - std::exp(-lam)) / (double) n;
                }
                std::vector<uint8_t> none(n, 0);
                std::string e;
                const auto ranked = strata::core::rank_learned_profile(g.n_layers, g.n_expert, none, hv, profile_loaded);
                if (strata::core::write_expert_profile(o.expert_profile_save, g.n_layers, g.n_expert, ranked, e))
                    std::fprintf(stderr, "strata serve: expert profile saved to %s (%s, %.0f routed entries counted on the device)\n",
                                 o.expert_profile_save.c_str(), why, total);
                else
                    std::fprintf(stderr, "strata serve: the expert profile was not saved: %s\n", e.c_str());
                profile_saved_at = Clock::now();
                return;
            }
            if (heat.empty()) return;
            std::vector<uint8_t> resident(host_res.size(), 0);
            for (size_t i = 0; i < host_res.size(); ++i) resident[i] = host_res[i] >= 0;
            for (const auto& p : pending) resident[(size_t) p.first] = 1;
            if (astate == AState::SwapIn)   // --adapt-async: the same for a round past its step 2
                for (const ASwap& w : aswaps) resident[(size_t) w.layer * g.n_expert + w.in] = 1;
            std::string e;
            const auto ranked = strata::core::rank_learned_profile(g.n_layers, g.n_expert, resident, heat,
                                                                   profile_loaded);
            if (strata::core::write_expert_profile(o.expert_profile_save, g.n_layers, g.n_expert, ranked, e))
                std::fprintf(stderr, "strata serve: expert profile saved to %s (%s)\n", o.expert_profile_save.c_str(),
                             why);
            else
                std::fprintf(stderr, "strata serve: the expert profile was not saved: %s\n", e.c_str());
            profile_saved_at = Clock::now();
        };
        // stdin is read on its own thread, so a STOP line reaches a request that is still running (the client went
        // away, or pressed Esc): the flag is checked between prompt chunks and between verify windows.
        // 0.1.44: the reader is detached and may sit in read(2) past QUIT (the server writes more, or closes stdin, while
        // main returns and the process exits), so what it touches is shared with it, never main's locals: a line or
        // EOF after QUIT used to lock a destroyed mutex (undefined behaviour; an abort on macOS).
        struct StdinShared {
            std::atomic<bool> stop_req{false};
            std::mutex mu;
            std::condition_variable cv;
            std::deque<std::string> lines;
            bool eof = false;
        };
        const auto in_shared = std::make_shared<StdinShared>();
        std::atomic<bool>& stop_req = in_shared->stop_req;
        std::mutex& in_mu = in_shared->mu;
        std::condition_variable& in_cv = in_shared->cv;
        std::deque<std::string>& in_lines = in_shared->lines;
        bool& in_eof = in_shared->eof;
        std::thread([st = in_shared] {
            // read(2) on the descriptor, not std::cin: glibc's exit() flushes every stdio stream and waits for
            // stdin's lock, which getline holds while it waits for input - an engine ending on an error (every
            // std::exit) would hang in exit() on Linux, and the server would wait for it forever
            std::string l, buf;
            char chunk[4096];
            auto getline_fd = [&](std::string& out) -> bool {
                for (;;) {
                    const size_t nlpos = buf.find('\n');
                    if (nlpos != std::string::npos) {
                        out.assign(buf, 0, nlpos);
                        buf.erase(0, nlpos + 1);
                        return true;
                    }
#if defined(_WIN32)
                    const int n = _read(0, chunk, (unsigned) sizeof chunk);
#else
                    const ssize_t n = ::read(0, chunk, sizeof chunk);
                    if (n < 0 && errno == EINTR) continue;
#endif
                    if (n <= 0) {
                        if (buf.empty()) return false;
                        out.swap(buf);
                        buf.clear();
                        return true;
                    }
                    buf.append(chunk, (size_t) n);
                }
            };
            while (getline_fd(l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { st->stop_req.store(true); continue; }
                std::lock_guard<std::mutex> lk(st->mu);
                st->lines.push_back(l);
                st->cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(st->mu);
            st->eof = true;
            st->cv.notify_one();
        }).detach();
        auto next_line = [&](std::string& out) -> bool {
            std::unique_lock<std::mutex> lk(in_mu);
            in_cv.wait(lk, [&] { return !in_lines.empty() || in_eof; });
            if (in_lines.empty()) return false;
            out = std::move(in_lines.front());
            in_lines.pop_front();
            return true;
        };
        sp.should_stop = [&] { return stop_req.load(); };
        // STRATA_TRACE=1: one stderr line per step of a request (the log shows where a request stops)
        const bool trace = std::getenv("STRATA_TRACE") != nullptr;
        auto tr = [&](const char* what, long long a = -1, long long b = -1) {
            if (!trace) return;
            std::fprintf(stderr, "strata trace: %s %lld %lld\n", what, a, b);
            std::fflush(stderr);
        };
        {
            // what is left once everything is allocated: under WDDM a GPU filled to the brim does not fail, it pages -
            // and a page-in while the verify graph spins on a host flag stalls the request for good
            size_t free_b = 0, total_b = 0;
            /*
            DPCT1106: 'cudaMemGetInfo' was migrated with the Intel
            extensions for device information which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            dpct::get_current_device().get_memory_info(free_b, total_b);
            // below ~256 MiB a later allocation (a first-used window's buffers, the desktop, another program) can make
            // the driver page GPU memory, and a verify graph spinning on a host flag then never finishes
            const int64_t free_mib = (int64_t) (free_b >> 20);
            if (free_mib >= 256) {
                std::fprintf(stderr, "strata serve: %lld MiB of VRAM free with everything loaded\n", (long long) free_mib);
                rss_probe("serving");
            } else if (reserve_adapted) {
                // #496: the reserve was already lowered to make the cache fit - a bigger one would leave it no room
                std::fprintf(stderr, "strata serve: WARNING: %lld MiB of VRAM free with everything loaded - this card "
                                     "only just fits the model (the VRAM reserve was lowered to %d MiB so the expert "
                                     "cache fits): requests may stall. Close other programs that use the GPU, or lower "
                                     "--max-context\n", (long long) free_mib, o.vram_reserve_mib);
            } else {
                std::fprintf(stderr, "strata serve: %lld MiB of VRAM free with everything loaded - LOW: requests may stall;"
                                     " add --vram-reserve-mib %lld to the config's args (or lower --max-context)\n",
                             (long long) free_mib, (long long) (o.vram_reserve_mib + 512 - free_mib));
            }
        }
        // what the server's Monitor tab shows (servers before 0.1.8 skip unknown lines until READY)
        {
            size_t free_b = 0, total_b = 0;
            /*
            DPCT1106: 'cudaMemGetInfo' was migrated with the Intel
            extensions for device information which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            dpct::get_current_device().get_memory_info(free_b, total_b);
            // "Experts in VRAM" is every tier, not one card's.  This used to report `xcache` alone, so a layer split
            // showed CUDA0's cache as if it were the whole GPU's: a 4-GPU 256K run read 3327 experts when the four
            // cards held 13320, and the Monitor tab was wrong by 4x for every multi-GPU config.  The tiers are
            // disjoint by construction (remote_experts.cpp skips any pair a stage already claimed), so they add.
            const int64_t slots_primary = (int64_t) xcache.slots();
            const int64_t mib_primary = (int64_t) (xcache.bytes() >> 20);
            int64_t slots_all = slots_primary, mib_all = mib_primary;
            for (const auto& st : stages) {
                slots_all += (int64_t) st->cache.slots();
                mib_all += (int64_t) (st->cache.bytes() >> 20);
            }
            for (int r = 0; r < 3; ++r)
                if (o.expert_cache_remote[(size_t) r] > 0) {
                    slots_all += remote_experts[(size_t) r].resident();
                    mib_all += (int64_t) (remote_experts[(size_t) r].gib() * 1024.0);
                }
            std::printf("INFO context=%lld kv=%s kv_resident=%lld expert_slots=%lld expert_cache_mib=%lld "
                        "expert_slots_primary=%lld expert_cache_primary_mib=%lld spec=%d "
                        "mtp_max=%d lookup=%d vram_free_mib=%lld cvec=%s arena_mib=%lld pool_workers=%d pcie_frac=%.2f "
                        "spec_min_p=%.2f conversation_cache_mib=%lld conversation_cache_slots=%d "
                        "conversation_cache_min_free_mib=%lld tail_role_token=%lld vram_elastic=%d%s engine=" STRATA_VERSION "\n",
                        (long long) o.max_context, o.kv.c_str(),
                        (long long) (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1
                                         ? ss.qsa_states[ss.qsa_primary()].n_slots * 4 : 0),
                        (long long) slots_all, (long long) mib_all,
                        (long long) slots_primary, (long long) mib_primary,
                        o.spec, o.mtp_max_t,
                        o.suffix_draft, (long long) (free_b >> 20), cvec_summary.c_str(),
                        (long long) ((o.mmap_experts ? src.resident_bytes() : strata::kernels::cpu::expert_layout().total) >> 20),
                        pool.workers(), o.pcie_frac,
                        o.spec_min_p, (long long) o.conversation_cache_mib, o.conversation_cache_slots,
                        (long long) o.conversation_cache_min_free_mib, (long long) o.tail_role_token, xcache.segmented() ? 1 : 0,
                        o.batch > 0 ? (" batch_slots=" + std::to_string(o.batch) +
                                       " slot_cache=" + std::to_string(o.prompt_cache > 0 ? 1 : 0)).c_str() : "");
        }
        // issue #29: a request whose heartbeat (tokens, prompt chunks, verify windows) stops for this long is stuck on
        // a flag nobody will raise - end the engine with where it was, so the server starts it again instead of the
        // GPU spinning forever.  STRATA_WATCHDOG_S=0 turns it off.  Issue #31: before it does, it reports what every
        // part was doing (stall_report), so one occurrence says where the wait is.
        {
            const char* ws = std::getenv("STRATA_WATCHDOG_S");
            const int limit = ws ? std::atoi(ws) : 60;   // one step (a prompt layer, a verify window) takes seconds
            if (limit > 0)
                std::thread([limit] {
                    strata::core::Progress& p = strata::core::progress();
                    uint64_t last = p.beats.load(), ticks_at = p.ticks.load();
                    auto since = std::chrono::steady_clock::now();
                    for (;;) {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        const auto now = std::chrono::steady_clock::now();
                        const uint64_t b = p.beats.load();
                        if (!p.busy.load() || b != last) { last = b; ticks_at = p.ticks.load(); since = now; continue; }
                        if (now - since < std::chrono::seconds(limit)) continue;
                        // a blocking step's explicit allowance (session files): still within it, not yet stuck
                        if (strata::core::progress_now_ms() < p.allow_until_ms.load()) continue;
                        std::fprintf(stderr, "strata serve: no progress for %d s during a request (%s) - stopping "
                                             "the engine so the server starts it again (issue #29)\n",
                                     limit, stage_text().c_str());
                        stall_report(stderr, p.ticks.load() - ticks_at);
                        strata::core::release_gpu_waits(stderr);   // #267: no spin kernel outlives the process
                        std::fflush(stderr);
                        std::abort();
                    }
                }).detach();
        }
        std::printf("READY %lld stop\n", (long long) o.max_context);   // "stop": this engine honours STOP
        std::fflush(stdout);
        std::string line;
        int64_t rounds = 0;
        const int S = o.spec;
        const int S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, S) : S;   // the MTP's windows; suffixes go up to S
        if (use_mtp && S_mtp < S) mtp.set_max_drafts(S_mtp - 1);
        // --pipeline-windows 2 loads the drafter with 8 rows: the serial loop's chain keeps its own length
        else if (use_mtp && mtp.max_t() > S_mtp) mtp.set_max_drafts(S_mtp - 1);
        strata::spec::SuffixDrafter sfx(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
        strata::spec::DraftPolicy policy(S);   // MTP or lookup window, learned over the whole process
        // The vision path (--vision): GENI <max_new> <embeddings file> <id,id,...> carries images.  The file is one
        // or more strata-vision records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats) in prompt order;
        // each image's rows go to its run of <|image_pad|> tokens, whose M-RoPE positions are mtmd's: t = p,
        // h = p + y, w = p + x, and the text after the image continues at p + max(nx, ny).
        constexpr int64_t kImagePad = 248056;   // qwen4exp.ple.image_token_id: the PLE hash reads it for image cells
        bool mrope_identity = true;
        std::vector<float> img_rows;
        std::vector<const float*> row_ptr;
        // ---- #533: VRAM <reserve_mib>, between requests, only with --vram-elastic.  It shrinks the expert cache
        // until that much VRAM is free for other programs (a game, a CAD session), or grows it back towards its full
        // size when more than that is free.  Never on its own: only this command (the server's POST /v1/vram) moves
        // it.  A shrink gives back the cache's LAST segments: the experts in their slots become CPU misses, as any
        // expert outside the cache is (no output changes beyond what a smaller cache decodes), the prompt path's
        // loan moves down to the end of what is left (a smaller chunk when it no longer fits).  A grow maps them
        // again and puts each slot's expert back (or, when the adaptive tier already brought that one back, the
        // most-routed missing expert of the same layer); the adaptive tier carries on from there.
        std::vector<std::pair<int32_t, int32_t>> vram_evicted;   // (residency index, slot) the shrinks took
        const int64_t chunk_full = o.prefill_chunk;              // the loan's chunk with the whole cache
        auto relend = [&]() {   // the prompt path's loan: the end of the slots left, its chunk as large as fits
            if (pf_parts.empty() || pf_parts[0].first < 0) return;
            PfPart& p = pf_parts[0];
            const int64_t live = xcache.slots();
            auto fits = [&](int64_t c) {
                const int64_t k = part_slots(p, c);
                return k > 0 && k + 128 <= live && !(o.prefill_auto && k * 100 > kAutoLendPct * live);
            };
            int64_t c = chunk_full;
            while (c > 256 && !fits(c)) c = std::max<int64_t>(256, c / 2 / 256 * 256);
            o.prefill_chunk = c;
            p.first = (int32_t) std::max<int64_t>(0, live - part_slots(p, c));
            p.first_now = -1;   // laid out again at the next loan
        };
        auto vram_command = [&](const std::string &cmd,
                                std::string &e) -> bool {
            // `VRAM` alone: the reserve the engine started with (--vram-reserve-mib)
            try {
        char *end = nullptr;
            const bool bare = cmd.find_first_not_of(' ', 4) == std::string::npos;
            const long long reserve = bare ? (long long) o.vram_reserve_mib : std::strtoll(cmd.c_str() + 4, &end, 10);
            if (!bare && (end == cmd.c_str() + 4 || reserve < 0)) { e = "expected: VRAM [reserve_mib]"; return false; }
            if (!xcache.segmented()) {
                e = "VRAM needs an engine started with --vram-elastic (one NVIDIA GPU, --serve)";
                return false;
            }
            if (src.complement_ready()) {
                e = "VRAM: not with the resident low-RAM mode (the cache holds experts RAM does not)";
                return false;
            }
            if (peer.valid()) { e = "VRAM: not with --peer-device"; return false; }
            for (const PfPart& p : pf_parts)
                if (!p.lent.empty()) { e = "VRAM: a prompt's loan is still out"; return false; }
            if (!ver.wait_commit(e)) return false;
            apply_pending(true);                     // the adaptive tier's swaps in flight land first
            if (DPCT_CHECK_ERROR(
                    dpct::get_current_device().queues_wait_and_throw()) != 0) {
                /*
                DPCT1009: SYCL reports errors using exceptions and does not
                use error codes. Please replace the
                "get_error_string_dummy(...)" with a real error-handling
                function.
                */
                /*
                DPCT1010: SYCL uses exceptions to report errors and does
                not use the error codes. The cudaGetLastError function call was
                replaced with 0. You need to rewrite this code.
                */
                e = std::string("VRAM: ") + dpct::get_error_string_dummy(0);
                return false;
            }
            const auto t0 = Clock::now();
            size_t free_b = 0, total_b = 0;
            /*
            DPCT1106: 'cudaMemGetInfo' was migrated with the Intel
            extensions for device information which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            dpct::get_current_device().get_memory_info(free_b, total_b);
            const int64_t want_free = (int64_t) reserve << 20, mapped = xcache.mapped_bytes();
            const int64_t before = xcache.slots();
            // what stays at least: the prompt path's smallest loan (256 tokens) and the 128 slots it must leave
            const int64_t floor_slots = std::min<int64_t>(xcache.full_slots(),
                128 + (pf_parts.empty() || pf_parts[0].first < 0 ? 0 : part_slots(pf_parts[0], 256)));
            std::string note;
            if ((int64_t) free_b < want_free) {
                const int64_t floor_b = xcache.bytes_of(floor_slots);
                int64_t keep = mapped - (want_free - (int64_t) free_b);
                if (keep < floor_b) {
                    keep = floor_b;
                    note = " (the cache keeps its smallest size: the prompt path's buffers)";
                }
                if (!xcache.shrink(keep, e)) return false;
                const int64_t live = xcache.slots();
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= live) {
                        vram_evicted.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = strata::core::kNotResident;
                    }
                res_upload();
            } else if (mapped < xcache.full_bytes()) {
                std::string gerr;
                if (!xcache.grow(mapped + ((int64_t) free_b - want_free), gerr)) note = " (" + gerr + ")";
                const int64_t live = xcache.slots();
                std::string ferr;
                std::vector<std::pair<int32_t, int32_t>> keep_out;
                for (const auto& [i, slot] : vram_evicted) {
                    if (slot >= live) { keep_out.emplace_back(i, slot); continue; }
                    const int64_t layer = i / g.n_expert;
                    int64_t pick = i;
                    if (host_res[(size_t) i] >= 0) {   // back already (the adaptive tier): the layer's most-routed miss
                        pick = -1;
                        float best = -1.0f;
                        for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                            const size_t j = (size_t) (layer * g.n_expert + ex);
                            if (host_res[j] >= 0) continue;
                            const float u = drive.d.usage.empty() ? 0.0f : drive.d.usage[j];
                            if (u > best) { best = u; pick = (int64_t) j; }
                        }
                    }
                    if (pick < 0) continue;
                    const uint8_t* b = srcp->blob(layer, pick % g.n_expert);
                    if (b == nullptr || !xcache.fill_slot_queued(slot, b, ferr,
                            (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(layer))) {
                        e = "VRAM: refilling the cache failed: " + ferr;
                        return false;
                    }
                    host_res[(size_t) pick] = slot;
                }
                vram_evicted.swap(keep_out);
                if (!xcache.sync_queued(ferr)) { e = "VRAM: " + ferr; return false; }
                res_upload();
            }
            relend();
            /*
            DPCT1106: 'cudaMemGetInfo' was migrated with the Intel
            extensions for device information which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            dpct::get_current_device().get_memory_info(free_b, total_b);
            std::fprintf(stderr, "strata serve: VRAM %lld MiB kept free: the expert cache %lld -> %lld of %lld slots "
                                 "(%.2f of %.2f GiB), %lld MiB free now, prompt chunk %lld, in %.0f ms%s\n",
                         reserve, (long long) before, (long long) xcache.slots(), (long long) xcache.full_slots(),
                         (double) xcache.mapped_bytes() / 1073741824.0, (double) xcache.full_bytes() / 1073741824.0,
                         (long long) (free_b >> 20), (long long) o.prefill_chunk,
                         std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), note.c_str());
            std::printf("VRAM reserve_mib=%lld expert_slots=%lld expert_slots_full=%lld expert_cache_mib=%lld "
                        "expert_cache_full_mib=%lld vram_free_mib=%lld prompt_chunk=%lld\n", reserve,
                        (long long) xcache.slots(), (long long) xcache.full_slots(),
                        (long long) (xcache.mapped_bytes() >> 20), (long long) (xcache.full_bytes() >> 20),
                        (long long) (free_b >> 20), (long long) o.prefill_chunk);
            return true;
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        // ---- --batch: the slots of the batch windows
        struct BSlot {
            bool active = false, stop = false;
            int32_t x = 0;                 ///< the token the next window feeds (not yet in the slot's state)
            std::array<int32_t, strata::kernels::kVerifyMaxT> draft{}; ///< the slot's next MTP proposal
            bool draft_ready = false;
            int64_t p = 0;                 ///< its position
            int64_t produced = 0, max_new = 0;
            Clock::time_point t0;
            // the slot's own conversation cache: the tokens its sessions hold (the prompt, then every token a window
            // fed).  Kept when the request ends (`cached`), so the next turn of that conversation continues from it
            // instead of reading its history again (see `slot_source` in the request path).
            std::vector<int32_t> ids;
            bool cached = false;           ///< idle, and its sessions still hold `ids`
            bool cvec = true;              ///< the control vector setting `ids` were read with
            bool img = false;              ///< the conversation has pictures (never reused from the slot)
            // the checkpoint its prompt read took at the last turn boundary (the history before the new turn's
            // header): a client's next turn renders the history again WITHOUT this reply's thinking, so it matches
            // the slot's tokens only up to there - the slot's K/V up to it plus this state continue from it
            std::vector<ConvCheckpoint> checks;
            // a prompt read that gave way to a waiting request (BYIELD): `ids` is the part read so far, and the same
            // request continues from it (the read goes on with the same chunks; `from0`: it had started at token 0)
            bool partial = false, partial_from0 = false;
        };
        std::vector<BSlot> bs((size_t) std::max(o.batch, 0));
        // timing of the batch windows since the slots were last all idle (one stderr line then)
        double bt_run = 0, bt_commit = 0, bt_emit = 0;
        double bt_wait0 = 0, bt_pool0 = 0;
        int64_t bt_miss0 = 0, bt_hits0 = 0, bt_pcie0 = 0;
        int64_t bt_windows = 0, bt_rows = 0, bt_tokens = 0;
        Clock::time_point bt_start = Clock::now();
        int admit_slot = -1;               ///< the slot the request being read will continue in (BGEN)
        long long admit_max_new = 0;
        auto batch_on = [&] { for (const BSlot& b : bs) if (b.active) return true; return false; };
        auto try_next_line = [&](std::string& out) -> bool {
            std::lock_guard<std::mutex> lk(in_mu);
            if (in_lines.empty()) return false;
            out = std::move(in_lines.front());
            in_lines.pop_front();
            return true;
        };
        // the session a request just left behind (its prompt) -> slot b's sessions, on every stage
        auto copy_to_slot = [&](int b, const std::vector<int32_t> &ids,
                                std::string &e) -> bool {
            try {
        const int64_t upto = (int64_t)ids.size();
            for (size_t k = 0; k < bslot_ss.size(); ++k) {
                strata::core::SessionState& from = k == 0 ? ss : stages[k - 1]->ss;
                strata::core::SessionState& to = *bslot_ss[k][(size_t) b];
                const strata::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
                if (DPCT_CHECK_ERROR(
                        dpct::get_current_device().queues_wait_and_throw()) !=
                    0) {
                    e = "batch admission: device sync failed"; return false;
                }
                strata::core::ConversationCheckpoint ck;
                ck.ids = ids;
                if (!strata::core::conversation_checkpoint_save(ck, from, g, e) ||
                    !strata::core::conversation_checkpoint_restore(ck, to, g, e))
                    return false;
                for (int64_t j = 0; j < from.qsa_alloc; ++j) {
                    strata::core::ConversationKv img;
                    if (!strata::core::conversation_kv_save(img, from.qsa_states[from.qsa_ord0 + j], g, upto, true, e))
                        return false;
                    if (DPCT_CHECK_ERROR(dpct::get_current_device()
                                             .queues_wait_and_throw()) != 0) {
                        e = "batch admission: device sync failed"; return false;
                    }
                    if (!strata::core::conversation_kv_restore(img, to.qsa_states[to.qsa_ord0 + j], g, upto, true, e))
                        return false;
                }
                if (DPCT_CHECK_ERROR(
                        dpct::get_current_device().queues_wait_and_throw()) !=
                    0) {
                    e = "batch admission: device sync failed"; return false;
                }
            }
            if (batch_mtp) {
                // Admission first builds the solo draft KV; copy it into the slot before drafting.
                strata::core::ConversationKv image;
                if (!mtp.idle(e) || !slot_mtp[(size_t)b]->idle(e) ||
                    !strata::core::conversation_kv_save(image, mtp.kv_state(),
                                                        draft_geometry, upto,
                                                        false, e) ||
                    !strata::core::conversation_kv_restore(
                        image, slot_mtp[(size_t)b]->kv_state(), draft_geometry,
                        upto, false, e) ||
                    DPCT_CHECK_ERROR(
                        dpct::get_current_device().queues_wait_and_throw()) !=
                        0)
                    return false;
                slot_mtp[(size_t) b]->set_prompt_len(upto);
            }
            return true;
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        // slot b's sessions -> the main ones (the reverse of copy_to_slot): a request that continues the
        // conversation an idle slot holds reads on from there.  Every stage, the same calls.
        // `at`: one of the slot's checkpoints - only the K/V up to it is copied and its state restored instead
        auto copy_from_slot = [&](int b, const ConvCheckpoint *at,
                                  std::string &e) -> bool {
            try {
        const int64_t upto = at != nullptr ? (int64_t)at->ids.size()
                                           : (int64_t)bs[(size_t)b].ids.size();
            if (at != nullptr && at->stage_parts.size() != stages.size()) { e = "a checkpoint without its stage parts"; return false; }
            for (size_t k = 0; k < bslot_ss.size(); ++k) {
                strata::core::SessionState& from = *bslot_ss[k][(size_t) b];
                strata::core::SessionState& to = k == 0 ? ss : stages[k - 1]->ss;
                const strata::core::OnDevice on_k(k == 0 ? 0 : stages[k - 1]->dev);
                if (DPCT_CHECK_ERROR(
                        dpct::get_current_device().queues_wait_and_throw()) !=
                    0) {
                    e = "batch slot restore: device sync failed"; return false;
                }
                if (at != nullptr) {
                    if (!strata::core::conversation_checkpoint_restore(k == 0 ? *at : at->stage_parts[k - 1], to, g, e))
                        return false;
                } else {
                    strata::core::ConversationCheckpoint ck;
                    ck.ids = bs[(size_t) b].ids;
                    if (!strata::core::conversation_checkpoint_save(ck, from, g, e) ||
                        !strata::core::conversation_checkpoint_restore(ck, to, g, e))
                        return false;
                }
                for (int64_t j = 0; j < from.qsa_alloc; ++j) {
                    strata::core::ConversationKv img;
                    if (!strata::core::conversation_kv_save(img, from.qsa_states[from.qsa_ord0 + j], g, upto, true, e))
                        return false;
                    if (DPCT_CHECK_ERROR(dpct::get_current_device()
                                             .queues_wait_and_throw()) != 0) {
                        e = "batch slot restore: device sync failed";
                        return false;
                    }
                    if (!strata::core::conversation_kv_restore(img, to.qsa_states[to.qsa_ord0 + j], g, upto, true, e))
                        return false;
                }
                if (DPCT_CHECK_ERROR(
                        dpct::get_current_device().queues_wait_and_throw()) !=
                    0) {
                    e = "batch slot restore: device sync failed"; return false;
                }
            }
            if (batch_mtp) {
                // A returning solo request resumes from its slot's draft KV.
                strata::core::ConversationKv image;
                if (!slot_mtp[(size_t)b]->idle(e) || !mtp.idle(e) ||
                    !strata::core::conversation_kv_save(
                        image, slot_mtp[(size_t)b]->kv_state(), draft_geometry,
                        upto, false, e) ||
                    !strata::core::conversation_kv_restore(
                        image, mtp.kv_state(), draft_geometry, upto, false,
                        e) ||
                    DPCT_CHECK_ERROR(
                        dpct::get_current_device().queues_wait_and_throw()) !=
                        0)
                    return false;
                mtp.set_prompt_len(upto);
            }
            return true;
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        // one batch window over the active slots only (row t is the t-th active slot): an idle slot is not touched,
        // so it keeps the conversation it holds
        auto batch_step = [&]() -> bool {
            int S = 0;
            int rows[strata::kernels::kVerifyMaxT] = {};
            int32_t tok[strata::kernels::kVerifyMaxT] = {}, outb[strata::kernels::kVerifyMaxT] = {};
            int64_t pos[strata::kernels::kVerifyMaxT] = {};
            int first[strata::kernels::kVerifyMaxT] = {}, active[strata::kernels::kVerifyMaxT] = {};
            static size_t next_slot = 0;
            // Each MTP slot uses two rows; rotate slots when more than four are active.
            int A = 0;
            for (size_t offset = 0; offset < bs.size() && S + (batch_mtp ? 2 : 1) <= strata::kernels::kVerifyMaxT; ++offset) {
                const int b = (int) ((next_slot + offset) % bs.size());
                if (bs[(size_t) b].active) {
                    first[A] = S;
                    active[A++] = b;
                    rows[S] = b;
                    tok[S] = bs[(size_t) b].x;
                    pos[S] = bs[(size_t) b].p;
                    ++S;
                    if (batch_mtp) {
                        rows[S] = b;
                        if (!bs[(size_t) b].draft_ready) {
                            err = "batch MTP: a live slot has no draft";
                            return false;
                        }
                        tok[S] = bs[(size_t) b].draft[0];
                        pos[S] = bs[(size_t) b].p + 1;
                        ++S;
                    }
                }
            }
            if (batch_mtp && A > 0) next_slot = ((size_t) active[A - 1] + 1) % bs.size();
            if (S == 0) return true;
            const bool was_busy = strata::core::progress().busy.load();
            strata::core::progress().busy.store(true);
            drive.d.layers = 0;
            drive.d.experts = 0;
            drive.d.failed = false;
            apply_pending(false);
            if (bt_windows == 0) {
                bt_start = Clock::now();
                bt_wait0 = ver.ms_wait; bt_pool0 = ver.ms_pool;
                bt_miss0 = drive.d.multi_misses; bt_hits0 = drive.d.cache_hits; bt_pcie0 = drive.d.pcie_experts;
            }
            const Clock::time_point w0 = Clock::now();
            if (!ver.run_slot_rows(rows, S, tok, pos, win_pool_fn, win_pool_user, outb, err) || drive.d.failed) {
                std::printf("ERR %s\n", drive.d.failed && drive.d.fail ? drive.d.fail : err.c_str());
                return false;
            }
            const Clock::time_point w1 = Clock::now();
            // Accept the proposal only when the target picked it and there is room to emit both tokens.
            std::vector<int> keep(bs.size(), 0);
            for (int a = 0; a < A; ++a) {
                const int b = active[a], i = first[a];
                const BSlot& sl = bs[(size_t) b];
                const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outb[i]) != o.eos_ids.end();
                keep[b] = batch_mtp && outb[i] == tok[i + 1] && !eos && !sl.stop &&
                          sl.produced + 2 <= sl.max_new && sl.p + 3 <= o.max_context ? 2 : 1;
            }
            if (!(batch_mtp ? ver.commit_slot_prefixes(keep.data(), err) : ver.commit_slots(err))) {
                std::printf("ERR %s\n", err.c_str());
                return false;
            }
            const Clock::time_point w2 = Clock::now();
            auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
            bt_run += msd(w0, w1);
            bt_commit += msd(w1, w2);
            ++bt_windows;
            for (int a = 0; a < A; ++a) bt_rows += keep[active[a]];
            for (int t = 0; t < A; ++t) {
                const int b = active[t];
                BSlot& sl = bs[(size_t) b];
                for (int j = 0; j < keep[b]; ++j) {
                    const int32_t y = outb[first[t] + j];
                    sl.ids.push_back(tok[first[t] + j]);
                    std::printf("BT %d %d\n", b, (int) y);
                    ++sl.produced;
                    const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
                    const char* fin = eos ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                    : sl.p + 2 > o.max_context ? "length" : nullptr;
                    if (fin != nullptr) {
                        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
                        std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms);
                        sl.active = false;
                        sl.cached = o.prompt_cache > 0 && !sl.img;
                        break;
                    }
                    sl.x = y;
                    sl.p += 1;
                }
                if (batch_mtp && sl.active) {
                    const size_t stride = (size_t) g.hc * (size_t) g.n_embd;
                    if (DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                            slot_mtp_rows[(size_t)b].get(),
                            ver.final_R_all() + (size_t)first[t] * stride,
                            2 * stride * sizeof(float)).wait()) != 0 ||
                        !slot_mtp[(size_t)b]->draft(2, outb + first[t],
                                                    pos[first[t]], keep[b] - 1,
                                                    sl.draft.data(), err)) {
                        std::printf("ERR batch MTP slot %d: %s\n", b, err.c_str());
                        return false;
                    }
                    sl.draft_ready = true;
                }
            }
            std::fflush(stdout);
            bt_emit += msd(w2, Clock::now());
            strata::core::progress().busy.store(was_busy);
            if (!batch_on() && bt_windows > 0) {
                const double w = (double) bt_windows, wall = msd(bt_start, Clock::now());
                const double L = (double) g.n_layers;
                std::fprintf(stderr, "strata batch: %lld windows, avg %.2f rows, %.2f ms/window = run %.2f (CUDA0 GPU-reach "
                                     "wait %.2f + CPU experts %.2f) + commit %.2f + emit %.2f; per layer-window: CPU experts "
                                     "%.2f, VRAM hits %.2f, PCIe %.2f; %.1f rows/s over %.0f ms of wall time (admissions "
                                     "included)\n",
                             (long long) bt_windows, bt_rows / w, (bt_run + bt_commit + bt_emit) / w, bt_run / w,
                             (ver.ms_wait - bt_wait0) / w, (ver.ms_pool - bt_pool0) / w, bt_commit / w, bt_emit / w,
                             (drive.d.multi_misses - bt_miss0) / (w * L), (drive.d.cache_hits - bt_hits0) / (w * L),
                             (drive.d.pcie_experts - bt_pcie0) / (w * L), 1000.0 * bt_rows / std::max(wall, 1e-9), wall);
                for (size_t k = 0; k <= stages.size(); ++k) {
                    const std::string pr = (k == 0 ? ver : stages[k - 1]->ver).profile_report();
                    if (!pr.empty()) std::fprintf(stderr, "strata batch GPU stages, stage %zu (ms/window):%s\n", k + 1, pr.c_str());
                }
                bt_run = bt_commit = bt_emit = 0;
                bt_windows = bt_rows = bt_tokens = 0;
            }
            return true;
        };
        // ---- --batch-groups: the slot groups pipelined through the stages (--batch-groups G > 1 with a layer split)
        const int n_pipe = (int) stages.size() + 1;
        const bool piped = o.batch > 0 && o.batch_groups > 1 && n_pipe > 1;
        const int GS = piped ? o.batch / o.batch_groups : o.batch;
        struct PGroup {
            bool inflight = false;
            int stage = 0;                  ///< the stage it runs on or waits for
            int32_t tok[strata::kernels::kVerifyMaxT] = {};
            int64_t pos[strata::kernels::kVerifyMaxT] = {};
            int64_t since = 0;              ///< tick it started waiting (fairness)
        };
        std::vector<PGroup> pg((size_t) (piped ? o.batch_groups : 0));
        std::vector<int> stage_group((size_t) n_pipe, -1);
        int64_t pipe_tick = 0, rr = 0;
        auto stage_verifier = [&](int k) -> strata::core::Verifier& { return k == 0 ? ver : stages[(size_t) k - 1]->ver; };
        auto pipe_inflight = [&] { for (const PGroup& x : pg) if (x.inflight) return true; return false; };
        auto group_active = [&](int gi) {
            for (int t = 0; t < GS; ++t) if (bs[(size_t) (gi * GS + t)].active) return true;
            return false;
        };
        // one turn: serve every running stage, hand finished groups on, start what can start
        auto pump = [&](bool may_start) -> bool {
            ++pipe_tick;
            if (bt_windows == 0 && !pipe_inflight()) {
                bt_start = Clock::now();
                bt_wait0 = ver.ms_wait; bt_pool0 = ver.ms_pool;
                bt_miss0 = drive.d.multi_misses; bt_hits0 = drive.d.cache_hits; bt_pcie0 = drive.d.pcie_experts;
            }
            for (int k = 0; k < n_pipe; ++k) {
                const int gi = stage_group[(size_t) k];
                if (gi < 0) continue;
                strata::core::Verifier& vk = stage_verifier(k);
                const int r = vk.batch_poll(win_pool_fn, win_pool_user, err);
                if (r < 0) { std::printf("ERR %s\n", err.c_str()); return false; }
                if (r == 0) continue;
                stage_group[(size_t) k] = -1;
                PGroup& G = pg[(size_t) gi];
                if (k + 1 < n_pipe) { G.stage = k + 1; G.since = pipe_tick; continue; }
                // the last stage: the group's picks
                const int32_t* outb = vk.batch_out();
                for (int t = 0; t < GS; ++t) {
                    BSlot& sl = bs[(size_t) (gi * GS + t)];
                    if (!sl.active) continue;
                    const int32_t y = outb[t];
                    std::printf("BT %d %d\n", gi * GS + t, (int) y);
                    ++sl.produced;
                    ++bt_rows;
                    const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
                    const char* fin = eos ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                    : sl.p + 2 > o.max_context ? "length" : nullptr;
                    sl.ids.push_back(sl.x);
                    if (fin != nullptr) {
                        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
                        std::printf("BDONE %d %lld %s %.1f\n", gi * GS + t, (long long) sl.produced, fin, ms);
                        sl.active = false;
                        sl.cached = false;   // the pipeline's pad rows: a pipelined slot is not reused as a cache
                    } else {
                        sl.x = y;
                        sl.p += 1;
                    }
                }
                std::fflush(stdout);
                ++bt_windows;
                G.inflight = false;
            }
            // start waiting groups on free stages, the longest waiting first; a new step on stage 0 round-robin
            for (int k = n_pipe - 1; k >= 0; --k) {
                if (stage_group[(size_t) k] >= 0) continue;
                int pick = -1;
                if (k > 0) {
                    for (int gi = 0; gi < (int) pg.size(); ++gi)
                        if (pg[(size_t) gi].inflight && pg[(size_t) gi].stage == k &&
                            (pick < 0 || pg[(size_t) gi].since < pg[(size_t) pick].since))
                            pick = gi;
                    if (pick < 0) continue;
                } else {
                    if (!may_start) continue;
                    for (int j = 0; j < (int) pg.size() && pick < 0; ++j) {
                        const int gi = (int) ((rr + j) % (int64_t) pg.size());
                        if (!pg[(size_t) gi].inflight && group_active(gi)) pick = gi;
                    }
                    if (pick < 0) continue;
                    rr = pick + 1;
                    PGroup& G = pg[(size_t) pick];
                    for (int t = 0; t < GS; ++t) {
                        BSlot& sl = bs[(size_t) (pick * GS + t)];
                        G.tok[t] = sl.active ? sl.x : 0;
                        G.pos[t] = sl.active ? sl.p : 0;
                        if (!sl.active) sl.cached = false;   // its pad row writes its state
                    }
                    G.inflight = true;
                    G.stage = 0;
                    drive.d.failed = false;
                    apply_pending(false);
                }
                PGroup& G = pg[(size_t) pick];
                strata::core::progress().busy.store(true);
                if (!stage_verifier(k).batch_launch(pick * GS, GS, G.tok, G.pos, err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return false;
                }
                stage_group[(size_t) k] = pick;
            }
            if (drive.d.failed) { std::printf("ERR %s\n", drive.d.fail ? drive.d.fail : "the expert pool failed"); return false; }
            if (!pipe_inflight() && !batch_on() && bt_windows > 0) {   // all idle: the timing line, as batch_step
                const double w = (double) bt_windows, wall = std::chrono::duration<double, std::milli>(Clock::now() - bt_start).count();
                std::fprintf(stderr, "strata batch (pipelined, %d groups of %d): %lld group-steps, %lld rows in %.0f ms = "
                                     "%.1f rows/s (admissions included)\n", o.batch_groups, GS, (long long) bt_windows,
                             (long long) bt_rows, wall, 1000.0 * bt_rows / std::max(wall, 1e-9));
                for (int k = 0; k < n_pipe; ++k) {
                    const std::string pr = stage_verifier(k).profile_report();
                    if (!pr.empty()) std::fprintf(stderr, "strata batch GPU stages, stage %d (ms/window):%s\n", k + 1, pr.c_str());
                }
                bt_windows = bt_rows = 0;
                strata::core::progress().busy.store(false);
            }
            return true;
        };
        auto pipe_drain = [&]() -> bool {
            while (pipe_inflight())
                if (!pump(false)) return false;
            return true;
        };
        for (;;) {
            if (batch_on() || (piped && pipe_inflight())) {
                if (!try_next_line(line)) {
                    if (!(piped ? pump(true) : batch_step())) return 1;
                    continue;
                }
                // a request reads its prompt through every stage: the groups in flight finish first
                if (piped && line.rfind("BSTOP ", 0) != 0 && !pipe_drain()) return 1;
            } else if (!next_line(line)) {
                break;
            }
            // --batch: BSTOP <slot> ends that slot at its next batch window; BGEN <slot> <max_new> ... reads the
            // request's prompt and first token as a GEN 1, then continues it in that slot
            if (line.rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(line.c_str() + 6);
                if (b >= 0 && b < (int) bs.size()) bs[(size_t) b].stop = true;
                continue;
            }
            if (line.rfind("BYIELD", 0) == 0) continue;   // for a prompt read that has ended meanwhile
            if (line.rfind("VRAM", 0) == 0) {   // #533 (above): between requests, not a request
                std::string verr;
                if (batch_on()) verr = "VRAM: not while batch slots are decoding";
                else vram_command(line, verr);
                if (!verr.empty()) std::printf("ERR %s\n", verr.c_str());
                std::fflush(stdout);
                continue;
            }
            admit_slot = -1;
            if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
                const bool bimg = line.rfind("BGENI ", 0) == 0;
                char* e1 = nullptr;
                const long b = std::strtol(line.c_str() + (bimg ? 6 : 5), &e1, 10);
                char* e2 = nullptr;
                const long long mn = std::strtoll(e1, &e2, 10);
                if (bs.empty() || b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || mn < 1 || e2 == e1) {
                    std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", o.batch);
                    std::fflush(stdout);
                    continue;
                }
                admit_slot = (int) b;
                admit_max_new = mn;
                line = std::string(bimg ? "GENI 1" : "GEN 1") + e2;
            }
            // #477: every --expert-profile-save-every minutes, before the next request (at QUIT: after the loop)
            if ((!heat.empty() || usage_dev_srv != nullptr) && line != "QUIT" && o.expert_profile_save_min > 0 &&
                Clock::now() - profile_saved_at >= std::chrono::duration<double>(o.expert_profile_save_min * 60.0))
                save_profile("periodic");
            if (line == "QUIT") break;
            // the watchdog watches a request from here until this iteration ends, whichever way it ends
            struct BusyScope {
                BusyScope() {
                    strata::core::progress().allow_until_ms.store(0);
                    strata::core::progress().busy.store(true);
                    strata::core::progress_at("request");
                }
                ~BusyScope() {
                    strata::core::progress().busy.store(false);
                    strata::core::progress().allow_until_ms.store(0);
                    strata::core::progress_at("idle");
                }
            } busy_scope;
            stop_req.store(false);   // a STOP that arrived between requests is stale
            err.clear();
            // Disk sessions: SAVE <path> | RESTORE <path>, between requests (the path runs to the end of the line,
            // UTF-8).  The file holds what a parked conversation holds (conversation_file.hpp).  Answers: SAVED /
            // RESTORED <tokens> <bytes> <ms>; SERR <invalid|storage|memory|io> <published 0|1> <reason> for a refusal
            // or failure that leaves the engine in step (the live session as it was, unless published=1 says the new
            // file already replaced the old one); FATAL <reason> (and the engine exits) for a restore transfer that
            // failed after the device state was touched.  On the way: SESSION <done> <total> after every block of
            // the file moved (at most 16 MiB), and SWAIT <phase> <seconds> before a step that blocks in one call
            // (fingerprint file, state capture, file flush, rename + folder flush, validation, device transfer):
            // that step is allowed <seconds> (session_phase_limit_s) by the watchdog and by the server, no more.
            // (A verifier commit that fails before the command still answers ERR and exits, as for a request.)
            if (line.rfind("SAVE ", 0) == 0 || line.rfind("RESTORE ", 0) == 0) {
                const bool save = line[0] == 'S';
                const std::string path = line.substr(save ? 5 : 8);
                const auto t0 = Clock::now();
                auto ms = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); };
                auto refuse = [&](const std::string& why, strata::core::SessionError kind = strata::core::SessionError::invalid,
                                  bool published = false) {
                    std::fprintf(stderr, "strata serve: session %s %s: %s\n", save ? "save" : "restore", path.c_str(),
                                 why.c_str());
                    std::string one = why;   // one protocol line
                    for (char& c : one) if (c == '\n' || c == '\r') c = ' ';
                    std::printf("SERR %s %d %s\n", strata::core::session_error_name(kind), published ? 1 : 0, one.c_str());
                    std::fflush(stdout);
                    err.clear();
                };
                auto moving = [&](uint64_t done, uint64_t total) {
                    strata::core::progress_at(save ? "writing a session file, MiB" : "reading a session file, MiB",
                                              (int64_t) (done >> 20));
                    strata::core::progress_allow(0);   // a block moved: the ordinary watchdog limit again, and a beat
                    std::printf("SESSION %llu %llu\n", (unsigned long long) done, (unsigned long long) total);
                    std::fflush(stdout);
                };
                // a step that blocks in one call: its explicit allowance, on both sides, announced before it starts
                auto blocking = [&](const char* phase, uint64_t bytes) {
                    const int64_t s = strata::core::session_phase_limit_s(bytes);
                    strata::core::progress_at(phase);
                    strata::core::progress_allow(s);
                    std::printf("SWAIT %s %lld\n", phase, (long long) s);
                    std::fflush(stdout);
                };
                strata::core::progress_at(save ? "saving a session" : "restoring a session");
                if (path.empty()) { refuse("missing path"); continue; }
                if (!stages.empty() || multi_gpu) { refuse("session files do not support --layer-split"); continue; }
                if (o.peer_device >= 1) { refuse("session files do not support --peer-device"); continue; }
                if (o.batch > 0) { refuse("session files do not support --batch (parallel requests)"); continue; }
                if (o.prompt_cache <= 0) { refuse("session files need --prompt-cache > 0"); continue; }
                if (!ver.wait_commit(err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                strata::core::SessionFileIdentity id;
                try {
                    if (!session_identity(id, err, [&] { blocking("fingerprint", 2u << 20); })) {
                        refuse(err, strata::core::SessionError::io);
                        continue;
                    }
                } catch (const std::bad_alloc&) {
                    refuse("session identity: out of memory", strata::core::SessionError::memory);
                    continue;
                } catch (const std::exception& e) {
                    refuse(std::string("session identity: ") + e.what(), strata::core::SessionError::io);
                    continue;
                }
                if (save) {
                    if (!live_ok || live.empty()) { refuse("no complete session to save"); continue; }
                    size_t bytes = 0, kept = 0;
                    double capture_ms = 0;
                    try {
                        // RAM PREFLIGHT for the save (not a reservation), BEFORE anything is copied: the deepest
                        // checkpoint is chosen by reference, and the copies it will take (the list below and the one
                        // in `meta`), the copied live running state with its tokens and images, the K/V source
                        // directory and the 16 MiB write buffer are asked for above the parking floor.  Only the
                        // deepest checkpoint goes to disk (the next turn's resume point); the K/V is streamed from
                        // the authoritative pools straight into the file, without a full host capture.
                        strata::core::SessionSaveLive sl;
                        {
                            strata::core::ConversationStateSizes z;
                            std::string why;
                            uint64_t state = UINT64_MAX;   // unknown sizes: the preflight refuses rather than guesses
                            if (strata::core::conversation_session_sizes(g, ss, z, why)) {
                                const uint64_t q = (uint64_t) std::max<int64_t>(ss.qsa_alloc, 0);
                                const uint64_t per = (uint64_t) z.tail + z.dead + z.block_pos;
                                if (q == 0 || per <= (UINT64_MAX - z.gdn - z.ple) / q) state = z.gdn + z.ple + q * per;
                            }
                            sl.state_bytes = state;
                            sl.tokens = live.size();
                            sl.images = live_imgs.size();
                            sl.kv_layers = (uint64_t) std::max<int64_t>(ss.qsa_alloc, 0) + 1;
                        }
                        const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib << 20;
                        auto admit = [&](uint64_t need, std::string& why) {
                            const auto avail = strata::core::available_host_bytes();
                            if (strata::core::conversation_memory_admit(avail, need, floor)) return true;
                            why = "not enough RAM to save the session (" +
                                  (need == UINT64_MAX ? std::string("unknown") : std::to_string(need >> 20)) +
                                  " MiB plus a floor of " + std::to_string((long long) o.conversation_cache_min_free_mib) +
                                  " MiB needed, " + (avail ? std::to_string(*avail >> 20) + " MiB available)"
                                                           : "RAM telemetry unavailable)");
                            return false;
                        };
                        std::vector<ConvCheckpoint> disk_checks;
                        std::string why;
                        if (!strata::core::session_save_checkpoints(checks, sl, admit, disk_checks, why)) {
                            refuse(why, strata::core::SessionError::memory);
                            continue;
                        }
                        kept = disk_checks.size();
                        const strata::core::ConversationView view{live, live_imgs, disk_checks, cvec_cached};
                        strata::core::SavedConversation meta;
                        std::vector<strata::core::SessionKvSource> sources;
                        // the live running state comes off the device in one synchronous copy
                        blocking("capture", sl.state_bytes == UINT64_MAX ? 0 : sl.state_bytes);
                        if (!strata::core::conversation_snapshot_sources(meta, sources, view, ss, g, mtp.kv_state(),
                                                                         err)) {
                            refuse(err, strata::core::SessionError::io);
                            continue;
                        }
                        capture_ms = ms();
                        strata::core::SessionWriteOptions wo;
                        wo.min_free_bytes = (uint64_t) o.session_min_free_mib << 20;
                        wo.progress = moving;
                        wo.phase = blocking;
                        strata::core::SessionStatus st;
                        if (!strata::core::session_file_write(path, meta, sources, id, bytes, err, wo, &st)) {
                            refuse(err, st.error, st.published);
                            continue;
                        }
                        if (st.dir_flush_unsupported)
                            std::fprintf(stderr, "strata serve: session saved to %s; this filesystem cannot flush a "
                                                 "folder, so the new name is not flushed\n", path.c_str());
                    } catch (const std::bad_alloc&) {
                        refuse("not enough RAM to save the session", strata::core::SessionError::memory);
                        continue;
                    }
                    std::fprintf(stderr, "strata serve: session saved %zu tokens, %zu of %zu checkpoints, %zu bytes to %s "
                                 "in %.1f ms (state %.1f ms)\n", live.size(), kept, checks.size(), bytes,
                                 path.c_str(), ms(), capture_ms);
                    std::printf("SAVED %zu %zu %.1f\n", live.size(), bytes, ms());
                } else {
                    strata::core::SavedConversation image;
                    size_t bytes = 0;
                    try {
                        // bounds this session can ever restore (geometry, layer range, tokens, checkpoints, every
                        // state and K/V part), checked as the file is parsed and before each array is allocated; the
                        // file size follows from them.  The RAM the parse needs at its peak is asked first.
                        strata::core::SessionReadLimits limits;
                        limits.progress = moving;
                        const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib << 20;
                        limits.admit = [floor, &o](uint64_t need, std::string& why) {
                            const auto avail = strata::core::available_host_bytes();
                            if (strata::core::conversation_memory_admit(avail, need, floor)) return true;
                            why = "not enough RAM to read it (" + std::to_string(need >> 20) + " MiB plus a floor of " +
                                  std::to_string((long long) o.conversation_cache_min_free_mib) + " MiB needed, " +
                                  (avail ? std::to_string(*avail >> 20) + " MiB available)" : "RAM telemetry unavailable)");
                            return false;
                        };
                        if (!strata::core::conversation_session_read_limits(
                                limits, ss, g, mtp.kv_state(), (uint64_t) o.max_context,
                                (uint64_t) std::max(o.prompt_cache, 1), err)) {
                            refuse(err, strata::core::SessionError::io);
                            continue;
                        }
                        strata::core::SessionStatus st;
                        if (!strata::core::session_file_read(path, id, image, bytes, err, limits, &st)) {
                            refuse(err, st.error);
                            continue;
                        }
                    } catch (const std::bad_alloc&) {
                        refuse("not enough RAM to read the session", strata::core::SessionError::memory);
                        continue;
                    }
                    const double read_ms = ms();
                    // the whole image against this engine, still without any device write
                    blocking("validate", bytes);
                    if (!strata::core::conversation_snapshot_validate(image, ss, g, mtp.kv_state(), err)) {
                        refuse(err);
                        continue;
                    }
                    // the elastic K/V (--kv-grow) maps only the cells it has grown to: make room for the file's cells
                    if (!kvg_ensure((int64_t) image.live.ids.size() + 256, [&] {
                                                                                 dpct::get_current_device()
                                                                                     .queues_wait_and_throw();
                                                                                     apply_pending(
                                                                                         true); })) {
                        refuse("the K/V cannot grow to the saved conversation: no VRAM is left", strata::core::SessionError::memory);
                        continue;
                    }
                    conversations.take_reuse();   // retained K/V described the outgoing session
                    live_ok = false;
                    // host -> device in synchronous copies of the whole state: one bounded allowance
                    blocking("transfer", bytes);
                    if (strata::core::conversation_snapshot_restore(image, ss, g, mtp.kv_state(), err) !=
                        strata::core::ConversationRestore::restored) {
                        // validated above: a failure here is a transfer failure, after device writes began - never
                        // decode from a partial state; the server starts the engine again
                        std::fprintf(stderr, "strata serve: session restore %s: transfer failed: %s\n", path.c_str(),
                                     err.c_str());
                        std::printf("FATAL restoring the session file failed after the device state was changed: %s\n",
                                    err.c_str());
                        std::fflush(stdout);
                        return 1;
                    }
                    live = std::move(image.live.ids);
                    live_imgs = std::move(image.live.imgs);
                    checks = std::move(image.checkpoints);
                    for (const ConvCheckpoint& c : checks) check_clock = std::max(check_clock, c.used);
                    cvec_cached = image.cvec;
                    live_ok = true;
                    std::fprintf(stderr, "strata serve: session restored %zu tokens, %zu checkpoints, %zu bytes from %s "
                                 "in %.1f ms (read+check %.1f ms)\n", live.size(), checks.size(), bytes, path.c_str(),
                                 ms(), read_ms);
                    std::printf("RESTORED %zu %zu %.1f\n", live.size(), bytes, ms());
                }
                std::fflush(stdout);
                continue;
            }
            const bool geni = line.rfind("GENI ", 0) == 0;
            if (!geni && line.rfind("GEN ", 0) != 0) {
                std::printf("ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>\n");
                continue;
            }
            char* endp = nullptr;
            const long long max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &endp, 10);
            // optional sampling keys between max_new and the ids: temperature=F, top_p=F, top_k=N, min_p=F,
            // penalty_last_n=N, penalty_repeat=F, penalty_freq=F, penalty_present=F, seed=N (text requests
            // only).  Absent keys keep today's behavior: greedy, no penalties.
            float req_temperature = 0.0f, req_top_p = 1.0f;
            int req_top_k = 20;   // the sampler's own default; the sampled path REQUIRES top_k in 1..64
            unsigned long long req_seed = 0;
            float req_min_p = 0.0f, req_penalty_repeat = 1.0f, req_penalty_freq = 0.0f, req_penalty_present = 0.0f;
            int req_penalty_last_n = 0;
            int req_cvec = 1;   // cvec=0|1: a loaded control vector for this request (on when absent)
            // SYCL port: logprobs=K (0..20) - after each "T id" an "LP logprob id:logprob ..." line with the token's
            // log-probability and the K most likely tokens', from the verify window's head logits (before sampling,
            // penalties and temperature). -1 (absent): no LP lines; a server that does not ask never sees one.
            int req_logprobs = -1;
            // ckpt=0: a one-shot call whose turn no later request extends.  No checkpoint at its last turn boundary
            // (so no split there) nor every --prompt-cache-every tokens, and its session is neither continued nor
            // parked after it.  It still resumes from a checkpoint it matches, and still saves the system-prompt root
            // when that reaches --prompt-cache-root.  Absent = checkpointed as before.
            int req_ckpt = 1;
            // tuning keys (setup's calibration measures settings without restarting the engine): the PCIe share of
            // the missed experts and the draft-probability floor, for this request only
            // #2228: img_keys=<hex>,<hex>,... (GENI): each picture's key from the server (its normalized bytes' hash), in
            // prompt order; the conversation cache matches a picture by it and its grid instead of its embedding floats
            std::vector<uint64_t> req_img_keys;
            double req_pcie_frac = o.pcie_frac, req_spec_min_p = o.spec_min_p;
            if (endp != nullptr) {   // GENI takes the same keys (#75: image requests were always greedy); its
                                     // embedding file path is the first token without an =
                for (;;) {
                    while (*endp == ' ') ++endp;
                    const char* start = endp;
                    while (*endp != '\0' && *endp != ' ') ++endp;
                    if (endp == start) break;
                    const std::string tok(start, (size_t) (endp - start));
                    const size_t eq = tok.find('=');
                    if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
                    const std::string key = tok.substr(0, eq);
                    const float fv = std::strtof(tok.c_str() + eq + 1, nullptr);
                    if (key == "cvec") req_cvec = std::atoi(tok.c_str() + eq + 1);
                    else if (key == "ckpt") req_ckpt = std::atoi(tok.c_str() + eq + 1) != 0;
                    else if (key == "temperature") req_temperature = fv;
                    else if (key == "top_p") req_top_p = fv;
                    else if (key == "top_k") req_top_k = std::atoi(tok.c_str() + eq + 1);
                    else if (key == "min_p") req_min_p = fv;
                    else if (key == "penalty_last_n") req_penalty_last_n = std::atoi(tok.c_str() + eq + 1);
                    else if (key == "penalty_repeat") req_penalty_repeat = fv;
                    else if (key == "penalty_freq") req_penalty_freq = fv;
                    else if (key == "penalty_present") req_penalty_present = fv;
                    else if (key == "seed") req_seed = std::strtoull(tok.c_str() + eq + 1, nullptr, 10);
                    else if (key == "logprobs") req_logprobs = std::clamp(std::atoi(tok.c_str() + eq + 1), -1, 20);
                    else if (key == "pcie_frac") req_pcie_frac = std::clamp((double) fv, 0.0, 1.0);
                    else if (key == "spec_min_p") req_spec_min_p = std::clamp((double) fv, 0.0, 1.0);
                    else if (key == "img_keys") {
                        req_img_keys.clear();
                        const char* q = tok.c_str() + eq + 1;
                        while (*q != '\0') {
                            char* qe = nullptr;
                            const unsigned long long v = std::strtoull(q, &qe, 16);
                            if (qe == q) { req_img_keys.clear(); break; }
                            req_img_keys.push_back((uint64_t) v);
                            q = *qe == ',' ? qe + 1 : qe;
                            if (*qe != ',' && *qe != '\0') { req_img_keys.clear(); break; }
                        }
                    }
                    // unknown keys are skipped: the ids start at the first token without '='
                }
            }
            std::string emb_path;
            if (geni && endp != nullptr) {
                while (*endp == ' ') ++endp;
                char* gap = std::strchr(endp, ' ');
                if (gap != nullptr) { emb_path.assign(endp, (size_t) (gap - endp)); endp = gap; }
            }
            std::vector<int64_t> ids;
            std::string pe;
            if (max_new < 1 || endp == nullptr || (geni && emb_path.empty()) || !parse_i64_list(endp, ids, pe)) {
                std::printf("ERR bad request: %s\n", pe.empty() ? "max_new" : pe.c_str());
                continue;
            }
            const int64_t n = (int64_t) ids.size();
            req_imgs.clear();
            if (geni && !o.vision) { std::printf("ERR this engine was started without --vision\n"); continue; }
            if (geni || !mrope_identity) {
                // positions for every cell this request can reach; the identity again for a text request
                std::string ve;
                row_ptr.assign((size_t) n, nullptr);
                const int64_t cells = (int64_t) mrope_host.size() / 3;
                auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
                    mrope_host[(size_t) c * 3] = (int32_t) t;
                    mrope_host[(size_t) c * 3 + 1] = (int32_t) h;
                    mrope_host[(size_t) c * 3 + 2] = (int32_t) w;
                };
                if (!geni) {
                    for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
                } else {
                    struct Img { int64_t n, nx, ny; size_t off; };
                    std::vector<Img> imgs;
                    img_rows.clear();
                    std::FILE* f = std::fopen(emb_path.c_str(), "rb");
                    if (!f) ve = "cannot open " + emb_path;
                    while (f && ve.empty()) {
                        int32_t hdr[5];
                        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
                        if (got == 0) break;
                        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
                            (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != (int32_t) g.n_embd) {
                            ve = "bad embeddings file (expected strata-vision records of width " +
                                 std::to_string((long long) g.n_embd) + ")";
                            break;
                        }
                        const size_t off = img_rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
                        img_rows.resize(off + cnt);
                        if (std::fread(img_rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
                        imgs.push_back({hdr[1], hdr[2], hdr[3], off});
                    }
                    if (f) std::fclose(f);
                    int64_t p = 0, i = 0;
                    size_t k = 0;
                    while (ve.empty() && i < n) {
                        if (ids[(size_t) i] != kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
                        if (k >= imgs.size()) { ve = "the prompt has more images than the embeddings file"; break; }
                        const Img& im = imgs[k++];
                        {   // what the conversation cache compares: a picture is its grid and its embeddings
                            const int64_t grid[3] = {im.n, im.nx, im.ny};
                            uint64_t h = fnv1a(grid, sizeof grid);
                            if (req_img_keys.size() == imgs.size())   // #2228: the server's key for picture k
                                h = fnv1a(&req_img_keys[k - 1], sizeof(uint64_t), h);
                            else
                                h = fnv1a(img_rows.data() + im.off, (size_t) im.n * (size_t) g.n_embd * sizeof(float), h);
                            req_imgs.push_back({i, h});
                        }
                        for (int64_t j = 0; j < im.n && ve.empty(); ++j)
                            if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                                ve = "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                                     " rows but fewer <|image_pad|> tokens";
                        for (int64_t j = 0; j < im.n && ve.empty(); ++j) {
                            const int64_t y = j / im.nx, x = j % im.nx;
                            put(i + j, p, p + y, p + x);
                            row_ptr[(size_t) (i + j)] = img_rows.data() + im.off + (size_t) j * (size_t) g.n_embd;
                        }
                        i += im.n;
                        p += std::max(im.nx, im.ny);
                    }
                    if (ve.empty() && k != imgs.size()) ve = "the embeddings file has more images than the prompt";
                    if (ve.empty() && n > 0 && ids[(size_t) (n - 1)] == kImagePad) ve = "the prompt cannot end in an image";
                    for (int64_t c = n; ve.empty() && c < cells; ++c) put(c, p + (c - n), p + (c - n), p + (c - n));
                }
                tr("positions built", (long long) img_rows.size());
                dpct::get_current_device().queues_wait_and_throw();
                tr("device idle");
                // CUDA0's table and, with a layer split, every later stage's (each device reads its own)
                auto upload_mrope = [&]() -> bool {
                    try {
                bool ok = DPCT_CHECK_ERROR(
                              dpct::get_in_order_queue()
                                  .memcpy(d_mrope, mrope_host.data(),
                                          mrope_host.size() * sizeof(int32_t))
                                  .wait()) == 0;
                    for (auto& st : stages) {
                        const strata::core::OnDevice on(st->dev);
                        dpct::get_current_device().queues_wait_and_throw();
                        ok =
                            ok &&
                            DPCT_CHECK_ERROR(
                                dpct::get_in_order_queue()
                                    .memcpy(st->mrope, mrope_host.data(),
                                            mrope_host.size() * sizeof(int32_t))
                                    .wait()) == 0;
                    }
                    return ok;
                }
                catch (sycl::exception const &exc) {
                  std::cerr << exc.what()
                            << "Exception caught at file:" << __FILE__
                            << ", line:" << __LINE__ << std::endl;
                  std::exit(1);
                }
                };
                if (ve.empty() && !upload_mrope()) ve = "the image position upload failed";
                if (!ve.empty()) {
                    // leave the table as the identity so the next text request is untouched
                    for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
                    upload_mrope();
                    mrope_identity = true;
                    std::printf("ERR %s\n", ve.c_str());
                    std::fflush(stdout);
                    continue;
                }
                mrope_identity = !geni;
            }
            sp.embd_rows = geni ? row_ptr.data() : nullptr;
            if (n + max_new + 8 > o.max_context) {
                std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) n,
                            (long long) max_new, (long long) o.max_context);
                continue;
            }
            bool bad = false;
            for (int64_t t : ids) bad = bad || t < 0 || t >= n_vocab;
            if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); continue; }
            std::array<int64_t, 3> remote_before{};
            std::array<int64_t, 3> launches_before{};
            std::array<uint64_t, 3> compact_before{}, full_before{};
            // ms_begin/ms_wait are cumulative since boot; the log line used to print them next to per-request deltas,
            // so the host time read as if it belonged to this request.  Take deltas here like every other column.
            std::array<double, 3> begin_before{}, wait_before{};
            for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
            {
                remote_before[(size_t) r] = remote_experts[(size_t) r].computed();
                launches_before[(size_t) r] = remote_experts[(size_t) r].launched_layers();
                compact_before[(size_t) r] = remote_experts[(size_t) r].returned_bytes();
                full_before[(size_t) r] = remote_experts[(size_t) r].full_row_bytes();
                begin_before[(size_t) r] = remote_experts[(size_t) r].ms_begin();
                wait_before[(size_t) r] = remote_experts[(size_t) r].ms_wait();
            }
            cur = ids;
            const Clock::time_point r0 = Clock::now();
            // ---- where this request starts reading: the live session, or a checkpoint, whose tokens AND pictures are
            // exactly the start of this prompt - at most n - 1 of them, the last token is always the first window
            auto starts_with = [&](const std::vector<int32_t>& pre, const std::vector<ImgKey>& pre_imgs) -> bool {
                const int64_t L = (int64_t) pre.size();
                if (L < 1 || L > n - 1) return false;
                for (int64_t i = 0; i < L; ++i)
                    if ((int32_t) ids[(size_t) i] != pre[(size_t) i]) return false;
                return imgs_below(req_imgs, L) == pre_imgs;
            };
            const bool want_cvec = strata::kernels::cvec().loaded() ? req_cvec != 0 : true;
            // the last request's final commit may still be running on the verifier's stream (set_commit_async):
            // everything below reads, restores or zeroes the session from other streams and the host (the end of the
            // last request waited already; this covers a request that ended on an error path)
            if (!ver.wait_commit(err)) {
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            auto kv_quiesce = [&] {
                                    try {
            dpct::get_current_device().queues_wait_and_throw(); apply_pending(true); }
            catch (sycl::exception const &exc) {
              std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                        << ", line:" << __LINE__ << std::endl;
              std::exit(1);
            }
            };
            if (kvg.on) {   // the elastic K/V: this prompt's cells (it gives back what it does not need below)
                // a failed growth may leave the tier half-changed: the engine stops (as for any CUDA failure)
                if (!kvg_ensure(n + 256, kv_quiesce)) {
                    std::printf("ERR the K/V cannot grow to this prompt: no VRAM is left\n");
                    std::fflush(stdout);
                    return 1;
                }
            }
            int64_t resume = 0;
            bool from_live = false;
            if (o.prompt_cache > 0 && want_cvec == cvec_cached) {
                if (live_ok && starts_with(live, live_imgs)) { resume = (int64_t) live.size(); from_live = true; }
                for (const ConvCheckpoint& c : checks)
                    if ((int64_t) c.ids.size() > resume && starts_with(c.ids, c.imgs)) {
                        resume = (int64_t) c.ids.size();
                        from_live = false;
                    }
            }
            // --batch: an idle slot that holds the start of this prompt (the conversation it served last) is a
            // source too - its sessions are copied back below, so only the new part is read
            int slot_source = -1;
            int64_t slot_tokens = 0;
            bool resumed_from0 = false;   // a prompt read parked in a slot (BYIELD) that had started at token 0
            const ConvCheckpoint* slot_ck = nullptr;   // the slot's checkpoint the prompt continues from (else its end)
            if (o.prompt_cache > 0 && req_imgs.empty())
                for (int b = 0; b < (int) bs.size(); ++b) {
                    const BSlot& sl = bs[(size_t) b];
                    if (sl.active || !sl.cached || sl.cvec != want_cvec) continue;
                    if ((int64_t) sl.ids.size() > std::max(resume, slot_tokens) && starts_with(sl.ids, {})) {
                        slot_source = b;
                        slot_tokens = (int64_t) sl.ids.size();
                        slot_ck = nullptr;
                    }
                    for (const ConvCheckpoint& c : sl.checks)
                        if ((int64_t) c.ids.size() > std::max(resume, slot_tokens) && starts_with(c.ids, c.imgs)) {
                            slot_source = b;
                            slot_tokens = (int64_t) c.ids.size();
                            slot_ck = &c;
                        }
                }
            const auto parked = conversations.best(ids, req_imgs, want_cvec);
            std::optional<strata::core::SavedConversation> incoming;
            if (parked.tokens > std::max(resume, slot_tokens)) incoming.emplace(conversations.take(parked.index));
            if (incoming) slot_source = -1;
            // Reject the entire image before parking/overwriting the outgoing
            // state. Invalid entries can safely fall back to its existing prefix.
            if (incoming && incoming->stage_images.size() != stages.size()) {
                std::fprintf(stderr, "strata serve: conversation cache: discard snapshot (stage count)\n");
                incoming.reset();
            }
            if (incoming) for (size_t i = 0; i < stages.size(); ++i) {
                const strata::core::OnDevice on(stages[i]->dev);
                if (!strata::core::conversation_snapshot_validate(incoming->stage_images[i], stages[i]->ss, g,
                        use_mtp && i + 1 == stages.size() ? &mtp.kv_state() : nullptr, err)) {   // the draft: the last stage's
                    std::fprintf(stderr, "strata serve: conversation cache: discard invalid stage snapshot (%s)\n",
                                 err.c_str());
                    incoming.reset();
                    err.clear();
                    break;
                }
            }
            if (incoming && !strata::core::conversation_snapshot_validate(*incoming, ss, g,
                    use_mtp && stages.empty() ? &mtp.kv_state() : nullptr, err)) {
                std::fprintf(stderr, "strata serve: conversation cache: discard invalid snapshot (%s)\n", err.c_str());
                incoming.reset();
                err.clear();
            }
            // Preserve the outgoing branch before any checkpoint rewind, reset,
            // or incoming restore overwrites the positional state it requires.
            if ((!from_live || incoming || slot_source >= 0) && !park_current(incoming ? incoming->bytes() : 0)) {
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            if (slot_source >= 0) {
                const auto t0 = Clock::now();
                if (!copy_from_slot(slot_source, slot_ck, err)) {
                    // the main session may be half written: read this prompt from the start
                    std::fprintf(stderr, "strata batch: restoring slot %d failed (%s); reading the prompt\n",
                                 slot_source, err.c_str());
                    err.clear();
                    live.clear();
                    live_imgs.clear();
                    checks.clear();
                    resume = 0;
                    from_live = false;
                } else {
                    checks.clear();   // the main session's checkpoints were of the conversation it held before
                    // a read that gave way, the same request again (into its own slot): on with it - its segments and
                    // checkpoints as if it had not stopped
                    if (slot_ck == nullptr && bs[(size_t) slot_source].partial && admit_slot == slot_source) {
                        resumed_from0 = bs[(size_t) slot_source].partial_from0;
                        for (const ConvCheckpoint& c : bs[(size_t) slot_source].checks) checks.push_back(c);
                        bs[(size_t) slot_source].partial = false;
                    }
                    if (slot_ck != nullptr) {
                        live = slot_ck->ids;
                        checks.push_back(*slot_ck);   // the main session's chain has it now
                        checks.back().used = ++check_clock;
                    } else {
                        live = bs[(size_t) slot_source].ids;
                    }
                    live_imgs.clear();
                    resume = slot_tokens;
                    from_live = true;
                    std::fprintf(stderr, "strata batch: slot %d gave back %lld tokens of this conversation (%s) in "
                                 "%.1f ms\n", slot_source, (long long) slot_tokens,
                                 slot_ck != nullptr ? "its turn checkpoint" : "all it holds",
                                 std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                }
            }
            // the elastic K/V: a parked conversation is restored whole, and it may be longer than this prompt
            if (incoming && !kvg_ensure((int64_t) incoming->live.ids.size() + 256, kv_quiesce)) {
                std::printf("ERR the K/V cannot grow to the parked conversation: no VRAM is left\n");
                return 1;
            }
            if (incoming) {
                const auto t0 = Clock::now();
                if (strata::core::conversation_snapshot_restore(*incoming, ss, g,
                        use_mtp && stages.empty() ? &mtp.kv_state() : nullptr, err) !=
                    strata::core::ConversationRestore::restored) {
                    // Already prevalidated above: a failure here is fatal, never
                    // permission to decode from a partially restored session.
                    std::printf("ERR restoring parked conversation: %s\n", err.c_str());
                    return 1;
                }
                for (size_t i = 0; i < stages.size(); ++i) {   // the later stages' parts
                    const strata::core::OnDevice on(stages[i]->dev);
                    if (strata::core::conversation_snapshot_restore(incoming->stage_images[i], stages[i]->ss, g,
                            use_mtp && i + 1 == stages.size() ? &mtp.kv_state() : nullptr, err) !=
                        strata::core::ConversationRestore::restored) {
                        std::printf("ERR restoring parked conversation (stage CUDA%d): %s\n", stages[i]->dev,
                                    err.c_str());
                        return 1;
                    }
                    if (DPCT_CHECK_ERROR(dpct::get_current_device()
                                             .queues_wait_and_throw()) != 0) {
                        std::printf("ERR stage sync\n"); return 1;
                    }
                }
                if (use_mtp && std::getenv("STRATA_SNAPSHOT_VERIFY") != nullptr) {
                    uint64_t draft_hash = 0;
                    const auto& draft_kv = stages.empty() ? incoming->kv.back() : incoming->stage_images.back().kv.back();
                    const strata::core::OnDevice on_d(stages.empty() ? 0 : stages.back()->dev);
                    if (!strata::core::conversation_kv_verify(draft_kv, mtp.kv_state(), g,
                            int64_t(incoming->live.ids.size()), false, draft_hash, err)) {
                        std::printf("ERR verifying restored draft KV: %s\n", err.c_str());
                        return 1;
                    }
                    std::fprintf(stderr, "strata serve: SNAPSHOT_VERIFY draft=%016llx cells=%lld mode=%d source=%s resident=%lld\n",
                                 (unsigned long long) draft_hash, (long long) draft_kv.cells,
                                 mtp.kv_state().kv_mode, "ram",
                                 (long long) (mtp.kv_state().n_slots * strata::kernels::qsa_real_shapes().page_size));
                }
                live = std::move(incoming->live.ids);
                live_imgs = std::move(incoming->live.imgs);
                if (stages.empty()) {
                    checks = std::move(incoming->checkpoints);
                } else {   // give each checkpoint back its stage parts (moved, not copied); none if they do not line up
                    strata::core::ConversationCheckpointSplit cs;
                    cs.stage0 = std::move(incoming->checkpoints);
                    for (auto& si : incoming->stage_images) cs.parts.push_back(std::move(si.checkpoints));
                    if (!strata::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear();
                }
                cvec_cached = incoming->cvec;
                resume = parked.tokens;
                from_live = parked.live;
                if (std::getenv("STRATA_SNAPSHOT_FULL_CAPTURE") == nullptr) {
                    std::vector<std::vector<strata::core::ConversationKv>> stage_kv;   // every stage's, for its next park
                    for (auto& si : incoming->stage_images) stage_kv.push_back(std::move(si.kv));
                    conversations.retain(std::move(incoming->kv), int64_t(live.size()), std::move(stage_kv));
                }
                incoming.reset(); // Running-state/checkpoint copies are no longer needed.
                std::fprintf(stderr, "strata serve: conversation cache: restored %lld tokens (%s) in %.1f ms; parked=%zu bytes=%zu\n",
                             (long long) resume, from_live ? "live" : "checkpoint",
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                             conversations.size(), conversations.bytes());
            }
            if (want_cvec != cvec_cached) {
                live_ok = false;
                checks.clear();
                cvec_cached = want_cvec;
            }
            if (strata::kernels::cvec().loaded()) strata::kernels::cvec_set_enabled(want_cvec);
            // this request rewrites every cell from `resume` on, so a checkpoint past it (or not on this prompt's
            // path) no longer has its cells; the ones kept are prefixes of both the old tokens and the new
            checks.erase(std::remove_if(checks.begin(), checks.end(), [&](const ConvCheckpoint& c) {
                             return (int64_t) c.ids.size() > resume || !starts_with(c.ids, c.imgs);
                         }), checks.end());
            live_ok = false;   // until this request has finished, the session is in between
            // the elastic K/V: the outgoing session is parked and this request rewrites every cell from `resume` on,
            // so cells past this prompt's are no longer anyone's - far more than it needs go back to the cache
            if (kvg.on) {
                kv_quiesce();
                if (!kvg_trim(n + 256)) {
                    std::printf("ERR the K/V could not give its VRAM back to the expert cache\n");
                    std::fflush(stdout);
                    return 1;
                }
            }
            int64_t reread_to = -1;   // STRATA_CKPT_REREAD only: read [0, reread_to) again instead of restoring
            if (resume == 0) {
                strata::core::session_zero(ss, g, nullptr, main_cs);
                main_stream->wait();
                for (auto& st : stages) {
                    const strata::core::OnDevice on(st->dev);
                    strata::core::session_zero(st->ss, g, nullptr, (void*) st->stream);
                    st->stream->wait();
                }
                checks.clear();
            } else if (!from_live) {
                ConvCheckpoint* c = nullptr;
                for (ConvCheckpoint& k : checks) if ((int64_t) k.ids.size() == resume) c = &k;
                if (c != nullptr) c->used = ++check_clock;   // mounting through it is the use LRU counts
                static const bool reread = std::getenv("STRATA_CKPT_REREAD") != nullptr;
                if (reread && c != nullptr) {
                    // THE CHECK OF THE CHECKPOINT: instead of restoring it, read its tokens again from position 0 in
                    // one run (below, with the prompt path's slots lent like any read) - the same chunks the request
                    // that saved it read them in, when that request started at 0.  With the VRAM expert set fixed
                    // (--adapt-swaps 0) the answer must match the restored one token for token; anything the
                    // checkpoint missed shows up as a difference.
                    strata::core::session_zero(ss, g, nullptr, main_cs);
                    main_stream->wait();
                    for (auto& st : stages) {
                        const strata::core::OnDevice on(st->dev);
                        strata::core::session_zero(st->ss, g, nullptr, (void*) st->stream);
                        st->stream->wait();
                    }
                    reread_to = resume;
                    std::fprintf(stderr, "strata serve: STRATA_CKPT_REREAD: reading %lld tokens again instead of "
                                         "restoring\n", (long long) resume);
                } else if (c == nullptr || !checkpoint_restore(*c, ss, g) || c->stage_parts.size() != stages.size() ||
                           [&] {
                               for (size_t i = 0; i < stages.size(); ++i) {
                                   const strata::core::OnDevice on(stages[i]->dev);
                                   if (!checkpoint_restore(c->stage_parts[i], stages[i]->ss, g)) return true;
                               }
                               return false;
                           }()) {
                    std::printf("ERR restoring a conversation checkpoint failed\n");
                    return 1;
                }
            }
            // KV streaming: the drafter's ring may hold cells past `resume` from a longer turn; the main layers'
            // host copies and slots are always current (every writer writes both), so they need nothing
            if (use_mtp && resume > 0 && reread_to <= 0) mtp.kv_restore(resume);
            tr("request", n, geni ? 1 : 0);
            if (use_mtp) mtp.set_prompt_len(n);
            const int64_t read_from = reread_to > 0 ? 0 : resume;
            conversations.limit_reuse(read_from);
            pp_total = n;
            pp_from = read_from;
            pp_reached = read_from;
            pp_t0 = r0;
            pp_next_check = reread_to > 0 || !req_ckpt ? INT64_MAX : resume + o.prompt_cache_every;   // ckpt=0: none
            pp_tail_saved = reread_to > 0 || !req_ckpt;   // ckpt=0 (#861): no tail checkpoint either
            {
                std::lock_guard<std::mutex> lk(part_mu);
                part_at.clear();
                std::fill(part_next.begin(), part_next.end(), pp_next_check);
            }
            std::printf("RESUME %lld\n", (long long) resume);   // before reading: this many prompt tokens are reused
            strata::core::progress_at("reading the prompt, from token", read_from);
            std::fflush(stdout);
            // A SHORT PART OF THE PROMPT - the new message of a chat that continues from a checkpoint, the assistant
            // header - goes through the verify windows, S tokens at a time, as decode reads them.  The batched path
            // costs ~300 ms per run however few tokens it has (it streams every expert the chunk routes to that is
            // not in VRAM over PCIe), and it borrows slots it must refill after (~180 ms); a window costs ~16 ms a
            // token, with the misses on the CPU.  Each part below is decided on its own, so a long first message is
            // read batched and its header still goes through the windows.  Picture rows need the batched path.
            // STRATA_CKPT_REREAD compares a restored checkpoint with a batched re-read, so it keeps every read batched.
            static const bool no_short = std::getenv("STRATA_CKPT_REREAD") != nullptr;
            int64_t req_short_read = o.short_read;   // STRATA_PIPELINE_SWITCH may set it per request (short_read=)
            auto windows_ok = [&](int64_t a, int64_t b) -> bool {
                if (no_short || b - a > req_short_read) return false;
                if (sp.embd_rows != nullptr)
                    for (int64_t i = a; i < b; ++i)
                        if (sp.embd_rows[i] != nullptr) return false;
                return true;
            };
            // --pipeline-windows, testing: STRATA_PIPELINE_SWITCH=<file> is read at every request, "pw=<0|1|2>
            // theta=<f> force_miss=<k> short_read=<n>" - an A/B of the pipelined and the serial loops (and forced
            // rollbacks) on one server, with the same expert placement
            int pl_pw = pipe ? o.pipeline_windows : 0;
            float pl_theta = [] {   // a speculative window is launched only when its estimated chance is at least this
                const char* v = pipe_dbg_env("STRATA_PIPELINE_THETA");
                return v ? (float) std::atof(v) : 0.2f;
            }();
            int pl_force_miss = [] {   // exactness test: every k-th speculative window gets a wrong first token
                const char* v = pipe_dbg_env("STRATA_PIPELINE_FORCE_MISS");
                return v ? std::max(0, std::atoi(v)) : 0;
            }();
            if (static const char* sw = pipe_dbg_env("STRATA_PIPELINE_SWITCH"); sw != nullptr && pipe) {
                if (std::FILE* f = std::fopen(sw, "r")) {
                    char buf[256] = {};
                    const size_t nr = std::fread(buf, 1, sizeof buf - 1, f);
                    std::fclose(f);
                    buf[nr] = 0;
                    if (const char* q = std::strstr(buf, "pw=")) pl_pw = std::max(0, std::min(2, std::atoi(q + 3)));
                    if (const char* q = std::strstr(buf, "theta=")) pl_theta = (float) std::atof(q + 6);
                    if (const char* q = std::strstr(buf, "force_miss=")) pl_force_miss = std::max(0, std::atoi(q + 11));
                    if (const char* q = std::strstr(buf, "short_read=")) req_short_read = std::max(0LL, std::atoll(q + 11));
                    std::fprintf(stderr, "strata pipeline switch: pw=%d theta=%.3f force_miss=%d short_read=%lld\n", pl_pw,
                                 pl_theta, pl_force_miss, (long long) req_short_read);
                }
            }
            // --pipeline-windows: tokens [a, b) through the windows with the stages overlapped.  The tokens are known,
            // so every window commits whole: stage 0 queues window K, its commit and window K+1 back to back (each
            // parity's verifier keeps its own window), and stage 1 runs window K once stage 0 is done with it.  Every
            // stage runs the same windows in the same order as read_windows below, so the session ends the same.
            auto read_windows_pl = [&](int64_t a, int64_t b, std::string& e) -> bool {
                strata::core::progress_at("reading the prompt (pipelined verify windows), from token", a);
                if (!pl_prepare(e)) return false;
                for (int par = 0; par < 2; ++par) PV[0][par]->set_head_sampling(false);   // reaches stage 1 (set_next)
                struct Restore {
                    strata::core::Verifier* v[2];
                    ~Restore() { v[0]->set_head_sampling(true); v[1]->set_head_sampling(true); }
                } restore{{PV[0][0], PV[0][1]}};
                struct Win { int T; int64_t q; };
                std::vector<Win> wins;
                for (int64_t q = a; q < b;) {
                    const int T = (int) std::min<int64_t>(S, b - q);
                    wins.push_back({T, q});
                    q += T;
                }
                const size_t N = wins.size();
                std::vector<int32_t> w0((size_t) S), w1((size_t) S), nx((size_t) S);
                size_t l0 = 0, f0 = 0, l1 = 0, f1 = 0;   // stage 0 launched / finished, stage 1 launched / finished
                drive.d.layers = 0;
                drive.d.experts = 0;
                drive.d.failed = false;
                auto fail = [&](std::string why) -> bool {
                    pl_drain();
                    e = why.empty() ? std::string("a pipelined prompt window failed") : why;
                    return false;
                };
                while (f1 < N) {
                    if (stop_req.load()) return fail("cancelled");
                    // stage 0: up to two windows queued; window K+2 (K's parity) only once stage 1 has finished window
                    // K, whose hand-off K+2 overwrites
                    while (l0 < N && l0 < f0 + 2 && l0 < f1 + 2) {
                        const Win& w = wins[l0];
                        for (int t = 0; t < w.T; ++t) w0[(size_t) t] = (int32_t) cur[(size_t) (w.q + t)];
                        strata::core::Verifier& v = *PV[0][l0 & 1];
                        if (!v.pl_launch(w.T, w0.data(), w.q, e) || !v.pl_commit_async(w.T, e)) return fail(e);
                        ++l0;
                    }
                    if (f0 < l0) {
                        strata::core::Verifier& v = *PV[0][f0 & 1];
                        if (v.service(win_pool_fn, PSD[f0 & 1], e) < 0) return fail(e);
                        if (v.done(e)) {
                            if (!v.pl_finish(nullptr, e)) return fail(e);
                            ++f0;
                        } else if (!e.empty()) return fail(e);
                    }
                    if (l1 < f0 && l1 == f1) {   // stage 1: the next window, its stage 0 done
                        const Win& w = wins[l1];
                        for (int t = 0; t < w.T; ++t) w1[(size_t) t] = (int32_t) cur[(size_t) (w.q + t)];
                        strata::core::Verifier& v = *PV[1][l1 & 1];
                        if (pl_mtp_live[l1 & 1]) {   // the drafter has read this parity's previous rows first
                            const strata::core::OnDevice on(mtp.device());
                            (v.stream())
                                ->ext_oneapi_submit_barrier(
                                    {*pl_mtp_ev[l1 & 1]});
                        }
                        if (!v.pl_launch(w.T, w1.data(), w.q, e) || !v.pl_commit_async(w.T, e)) return fail(e);
                        ++l1;
                    }
                    if (f1 < l1) {
                        strata::core::Verifier& v = *PV[1][f1 & 1];
                        if (v.service(win_pool_fn, PSD[f1 & 1], e) < 0) return fail(e);
                        if (v.done(e)) {
                            if (!v.pl_finish(nullptr, e)) return fail(e);
                            const Win& w = wins[f1];
                            for (int t = 0; t < w.T; ++t) nx[(size_t) t] = (int32_t) cur[(size_t) (w.q + t + 1)];
                            if (!mtp.prefill(v.final_R(0), nx.data(), w.T, w.q, e, false)) return fail(e);
                            {
                                const strata::core::OnDevice on(mtp.device());
                                dpct::sync_barrier(pl_mtp_ev[f1 & 1],
                                                   mtp.stream());
                            }
                            pl_mtp_live[f1 & 1] = true;
                            pp_reached = w.q + w.T;   // #471
                            ++f1;
                        } else if (!e.empty()) return fail(e);
                    }
                    if (drive.d.failed) return fail(drive.d.fail ? drive.d.fail : "the expert pool failed");
                }
                pl_drain();   // nothing in flight: both stages' commits and the drafter's rows have landed
                pl_mtp_live[0] = pl_mtp_live[1] = false;
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) b, (long long) pp_total, ms,
                            ms > 0.0 ? 1000.0 * (double) (b - pp_from) / ms : 0.0);
                strata::core::progress_beat();
                std::fflush(stdout);
                return true;
            };
            // tokens [a, b) through the windows: commit all of them, then give the draft layer their residuals
            auto read_windows = [&](int64_t a, int64_t b, std::string& e) -> bool {
                // (STRATA_LOGPOS reads every window's logits: the serial loop)
                if (pipe && pl_pw >= 1 && std::getenv("STRATA_LOGPOS") == nullptr) return read_windows_pl(a, b, e);
                strata::core::progress_at("reading the prompt (verify windows), from token", a);   // #217: not "batched"
                // every token is committed and the picks are discarded: no head sampling (see set_head_sampling)
                struct NoHeadSampling {
                    strata::core::Verifier& v;
                    explicit NoHeadSampling(strata::core::Verifier& x) : v(x) { v.set_head_sampling(false); }
                    ~NoHeadSampling() { v.set_head_sampling(true); }
                } no_head_sampling(ver);
                std::vector<int32_t> win((size_t) S), outw((size_t) S), nxt((size_t) S);
                for (int64_t q = a; q < b;) {
                    if (stop_req.load()) { e = "cancelled"; return false; }
                    const int T = (int) std::min<int64_t>(S, b - q);
                    for (int t = 0; t < T; ++t) {
                        win[(size_t) t] = (int32_t) cur[(size_t) (q + t)];
                        nxt[(size_t) t] = (int32_t) cur[(size_t) (q + t + 1)];
                    }
                    drive.d.layers = 0;
                    drive.d.experts = 0;
                    drive.d.failed = false;
                    if (!ver.run(T, win.data(), q, win_pool_fn, win_pool_user, outw.data(), e) || drive.d.failed) {
                        if (drive.d.failed && drive.d.fail) e = drive.d.fail;
                        return false;
                    }
                    // STRATA_LOGPOS=<path>: the teacher-forced log-probability of every token read here (every
                    // token is committed and nxt[t] is the prompt's own next token), appended to <path>; the last
                    // column is the log-probability of STRATA_LOGPOS_EXTRA (default 248046, <|im_end|>), so the
                    // end-of-turn mass a chat model puts on raw text can be taken out of the measurement
                    static std::FILE* logpos = [] {
                        const char* p = std::getenv("STRATA_LOGPOS");
                        return p != nullptr ? std::fopen(p, "ab") : nullptr;
                    }();
                    static const int32_t logpos_extra = [] {
                        const char* p = std::getenv("STRATA_LOGPOS_EXTRA");
                        return p != nullptr ? (int32_t) std::atoi(p) : (int32_t) 248046;
                    }();
                    if (logpos != nullptr && !ver.window_logprobs(nxt.data(), T, q, logpos_extra, logpos, e))
                        return false;
                    if (!ver.commit(T, e) || (use_mtp && !mtp.prefill(ver.final_R_all(), nxt.data(), T, q, e))) return false;
                    q += T;
                    pp_reached = q;   // #471
                }
                // the batched prompt path (other streams), checkpoints and snapshots may follow: the last commit first
                if (!ver.wait_commit(e)) return false;
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) b, (long long) pp_total, ms,
                            ms > 0.0 ? 1000.0 * (double) (b - pp_from) / ms : 0.0);
                strata::core::progress_beat();
                std::fflush(stdout);
                return true;
            };
            // the batched path's slots are lent just before its first run and given back (refilled) before a window
            // reads - so the windows always see the whole expert cache - or once the prompt is read
            // Every participant gives its loan back here: the rows it lent are refilled into the SAME slots from
            // the arena, the residency table is restored, and one upload puts it on every device.  A stage refills
            // through its own cache and its own device - a slot refilled into the wrong cache would leave that
            // stage's cache holding an expert it does not own, which is silent and produces plausible tokens.
            // (split in two halves so a layer split can queue every stage's copies before it waits for any: #340)
            auto refill_issue = [&](PfPart& p, std::string& e) -> bool {
                tr("refill start", (long long) p.lent.size());
                const strata::core::OnDevice on(p.dev);
                size_t queued = 0;
                for (const auto& [i, slot] : p.lent) {   // D-4: queued, one wait (STRATA_REFILL_BLOCKING=1: each)
                    const uint8_t* b = srcp->blob(i / g.n_expert, i % g.n_expert);
                    const int64_t nb = (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
                    if (b == nullptr || !(refill_blocking() ? p.cache->fill_slot_blocking(slot, b, e, nb)
                                                            : p.cache->fill_slot_queued(slot, b, e, nb)))
                        return false;
                    host_res[(size_t) i] = slot;
                    // SYCL port: drain before GgufExpertSource's 512-buffer blob ring can wrap onto a queued copy
                    if (!refill_blocking() && ++queued % 256 == 0 && !p.cache->sync_queued(e)) return false;
                }
                return true;
            };
            auto refill_wait = [&](PfPart& p, std::string& e) -> bool {
                const strata::core::OnDevice on(p.dev);
                if (!p.cache->sync_queued(e)) return false;
#if defined(_WIN32)
                // The copies have landed: the file pages touched by the loan need not stay in the working set.
                for (const auto& [i, slot] : p.lent)
                    srcp->release(i / g.n_expert, i % g.n_expert);
#endif
                p.lent.clear();
                p.lent_chunk = 0;
                return true;
            };
            auto refill_one = [&](PfPart& p, std::string& e) -> bool {
                if (p.lent.empty()) return true;
                return refill_issue(p, e) && refill_wait(p, e);
            };
            // #340: with several stages every stage's copies go out first (each card on its own link), then each is
            // waited for - the same copies into the same slots, so the same cache; one stage: refill_one exactly.
            // STRATA_REFILL_SERIAL=1: stage after stage, as 0.1.30/0.1.31.
            static const bool refill_serial = std::getenv("STRATA_REFILL_SERIAL") != nullptr;
            auto refill = [&](std::string& e) -> bool {
                const auto t_rf = Clock::now();
                int64_t n_lent = 0, n_parts = 0;
                for (PfPart& p : pf_parts)
                    if (!p.lent.empty()) { n_lent += (int64_t) p.lent.size(); ++n_parts; }
                if (n_parts > 1 && !refill_serial) {
                    for (PfPart& p : pf_parts)
                        if (!p.lent.empty() && !refill_issue(p, e)) return false;
                    for (PfPart& p : pf_parts)
                        if (!p.lent.empty() && !refill_wait(p, e)) return false;
                } else {
                    for (PfPart& p : pf_parts)
                        if (!p.lent.empty() && !refill_one(p, e)) return false;
                }
                if (n_parts > 0) res_upload();
                if (trace && n_parts > 0) {
                    std::fprintf(stderr, "strata trace: refilled %lld slots on %lld stage(s) in %.1f ms\n",
                                 (long long) n_lent, (long long) n_parts,
                                 std::chrono::duration<double, std::milli>(Clock::now() - t_rf).count());
                    std::fflush(stderr);
                }
                return true;
            };
            // lend the slots `tokens` batched prompt tokens need: the prompt path's buffers for min(chunk, tokens
            // rounded up to 256), laid out in the last of the slots it may borrow - per participant, out of that
            // participant's own cache, and marking only that participant's own layers
            auto lend = [&](int64_t tokens, std::string& e) -> bool {
                if (pf_parts.empty()) return true;                     // its own buffers: nothing to lend
                const auto t_ln = Clock::now();
                // what this segment needs, capped by the configured chunk: a request lends only what its own
                // prompt needs, so a large chunk costs a short prompt nothing
                const int64_t want_full = equal_chunk(tokens, o.prefill_chunk);
                // #340: a short enough request reads in the stages' own S-token chunks (nothing lent)
                const int64_t want = split_small > 0 && tokens <= split_small_max ? std::min(want_full, split_small)
                                                                                 : want_full;
                if (want <= 0) {
                    e = "prefill: cannot lend buffers for an empty request segment";
                    return false;
                }
                bool any = false;
                for (PfPart& p : pf_parts) {
                    if (p.first < 0) continue;
                    if (!p.lent.empty()) {
                        if (want <= p.lent_chunk) continue;            // its current loan already covers this
                        // ONLY this participant's loan goes back: `refill` would return the other participants'
                        // loans too, and their buffers are still laid out in their caches - marking those slots
                        // resident again would hand the next window a prompt buffer in place of an expert
                        if (!refill_one(p, e)) return false;
                    }
                    const strata::core::OnDevice on(p.dev);
                    const int32_t first = std::max<int32_t>(p.first, (int32_t) (p.cache->slots() - part_slots(p, want)));
                    if (want != p.sp->chunk() || first != p.first_now) {
                        if (!p.sp->relayout(want, p.cache->device_slot(first), part_bytes(p, first), e)) return false;
                        p.first_now = first;
                    }
                    for (int64_t l = p.lb; l < p.le; ++l)              // THIS participant's layers only
                        for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                            const size_t i = (size_t) (l * g.n_expert + ex);
                            if (host_res[i] >= first) {
                                p.lent.emplace_back((int32_t) i, host_res[i]);
                                host_res[i] = strata::core::kNotResident;
                                any = true;
                            }
                        }
                    p.lent_chunk = want;
                }
                if (any) res_upload();
                if (trace) {
                    int64_t n_lent = 0;
                    for (const PfPart& p : pf_parts) n_lent += (int64_t) p.lent.size();
                    std::fprintf(stderr, "strata trace: lent %lld slots for %lld tokens in %.1f ms\n", (long long) n_lent,
                                 (long long) want, std::chrono::duration<double, std::milli>(Clock::now() - t_ln).count());
                    std::fflush(stderr);
                }
                return true;
            };
            // --batch: a prompt read while slots are decoding goes one prompt chunk at a time (the same chunks as one
            // run: each run starts where the last ended, at a multiple of the chunk), and between two chunks the slots
            // decode for a share of the time the chunk took (STRATA_BATCH_DECODE_SHARE, default 0.5), so a long
            // prompt does not stop the others.  The slots' windows then see the cache without the slots this prompt
            // borrowed (marked missing: those experts run on the CPU), never a prompt buffer.
            static const double decode_share = [] {
                const char* v = std::getenv("STRATA_BATCH_DECODE_SHARE");
                return v != nullptr ? std::max(0.0, std::atof(v)) : 0.5;
            }();
            bool batch_fatal = false;
            int64_t il_parts = 0;   // the slots' windows run between this prompt's chunks
            double il_ms = 0;
            int64_t yielded_at = -1;   // BYIELD: the read stopped here and its slot holds it
            int yielded_slot = -1;
            // `BYIELD <slot>` (the server: a shorter request is waiting): at the next chunk boundary the part read so
            // far is copied into <slot> (this admission's own, or a free one the server reserved for a solo request),
            // the request ends with `YIELDED <slot> <tokens>` + DONE cancel, and the server sends it again later: it
            // continues from the slot with the same chunks.  #656's cooperative preemption, with a slot as the park.
            auto read_part = [&](int64_t a0, int64_t b0, std::string& e) -> bool {
                // (a layer split reads its stages as a pipeline over one run's chunks: in pieces only beside slots)
                if (o.batch <= 0 || piped || (!stages.empty() && !batch_on())) return sp.run(ids.data() + a0, b0 - a0, a0, e);
                const int64_t C = std::max<int64_t>(sp.chunk(), 1);
                for (int64_t q = a0; q < b0;) {
                    const int64_t r = std::min(b0, q + C);
                    const auto tq = Clock::now();
                    if (!sp.run(ids.data() + q, r - q, q, e)) return false;
                    q = r;
                    if (q >= b0) continue;
                    int ys = -1;
                    {   // a BSTOP that came meanwhile ends its slot at its next window; a BYIELD is for this read
                        std::lock_guard<std::mutex> lk(in_mu);
                        for (auto it = in_lines.begin(); it != in_lines.end();) {
                            if (it->rfind("BSTOP ", 0) == 0) {
                                const int b = std::atoi(it->c_str() + 6);
                                if (b >= 0 && b < (int) bs.size()) bs[(size_t) b].stop = true;
                                it = in_lines.erase(it);
                            } else if (it->rfind("BYIELD ", 0) == 0) {
                                ys = std::atoi(it->c_str() + 7);
                                it = in_lines.erase(it);
                            } else {
                                ++it;
                            }
                        }
                    }
                    if (ys >= 0) {
                        // only where the rest is read the same way after it (more than a chunk and more than a short
                        // read left), text only, and into a slot that is not decoding
                        const bool can = ys < (int) bs.size() && !bs[(size_t) ys].active &&
                                         (admit_slot < 0 || ys == admit_slot) && req_imgs.empty() && o.prompt_cache > 0 &&
                                         b0 - q > std::max<int64_t>(C, o.short_read);
                        std::string ye;
                        const auto ty = Clock::now();
                        std::vector<int32_t> pre(ids.begin(), ids.begin() + q);
                        if (can && copy_to_slot(ys, pre, ye)) {
                            BSlot& sl = bs[(size_t) ys];
                            sl = BSlot{};
                            sl.ids = std::move(pre);
                            sl.cached = true;
                            sl.partial = true;
                            sl.partial_from0 = read_from == 0 || resumed_from0;
                            sl.cvec = cvec_cached;
                            for (const ConvCheckpoint& c : checks)   // the root / periodic checkpoints of this read
                                if ((int64_t) c.ids.size() <= q && std::equal(c.ids.begin(), c.ids.end(), ids.begin(),
                                        [](int32_t x, int64_t y) { return (int64_t) x == y; }))
                                    sl.checks.push_back(c);
                            yielded_at = q;
                            yielded_slot = ys;
                            std::fprintf(stderr, "strata batch: the prompt read gives way at %lld of %lld tokens; slot %d "
                                                 "holds it (copied in %.1f ms)\n", (long long) q, (long long) n, ys,
                                         std::chrono::duration<double, std::milli>(Clock::now() - ty).count());
                            e = "yield";
                            return false;
                        }
                        if (can) bs[(size_t) ys].cached = false;   // half written
                        std::fprintf(stderr, "strata batch: BYIELD %d not taken at %lld of %lld%s%s\n", ys, (long long) q,
                                     (long long) n, ye.empty() ? "" : ": ", ye.c_str());
                    }
                    if (decode_share <= 0.0 || !batch_on()) continue;
                    const double budget = decode_share * std::chrono::duration<double, std::milli>(Clock::now() - tq).count();
                    const auto td = Clock::now();
                    do {
                        if (!batch_step()) { batch_fatal = true; e = "a batch window failed"; return false; }
                    } while (batch_on() && std::chrono::duration<double, std::milli>(Clock::now() - td).count() < budget);
                    ++il_parts;
                    il_ms += std::chrono::duration<double, std::milli>(Clock::now() - td).count();
                    if (trace) {
                        std::fprintf(stderr, "strata trace: prompt chunk to %lld, then the slots decoded %.0f ms\n",
                                     (long long) q, std::chrono::duration<double, std::milli>(Clock::now() - td).count());
                        std::fflush(stderr);
                    }
                }
                return true;
            };
            apply_pending(true);
            if (!adapt_tick(true)) {   // --adapt-async: the round in flight lands before the prompt lends any slot
                std::printf("ERR an adaptive refill failed\n");
                return 1;
            }
            // per-request sampling for the verify window's head (greedy when temperature is absent)
            strata::kernels::SamplerParams req_sp;
            req_sp.greedy = req_temperature <= 0.0f;
            req_sp.temperature = req_temperature;
            req_sp.top_p = req_top_p;
            req_sp.top_k = req_top_k;
            req_sp.seed = req_seed ? req_seed
                                   : (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
            req_sp.min_p = std::clamp(req_min_p, 0.0f, 1.0f);
            req_sp.penalty_last_n = std::max(req_penalty_last_n, 0);
            req_sp.penalty_repeat = req_penalty_repeat;
            req_sp.penalty_freq = req_penalty_freq;
            req_sp.penalty_present = req_penalty_present;
            req_sp.counter = 0;
            ver.set_sampling(req_sp);
            if (pipe) ver_b.set_sampling(req_sp);   // (reaches the later stage's odd verifier)
            if (use_mtp) mtp.set_draft_sampling(req_sp);   // STRATA_SPEC_COUPLED=1: sampled drafts (a no-op otherwise)
            drive.d.pcie_num = std::max(0, std::min(256, (int) (req_pcie_frac * 256.0 + 0.5)));
            // a layer split: CUDA0's share as asked; a later GPU keeps its own (its link) unless the request sets one
            for (int st = 0; st < split_drive.n; ++st)
                split_drive.pcie_num[st] = (st == 0 || split_same || req_pcie_frac != o.pcie_frac)
                                               ? drive.d.pcie_num : pcie_num_of(stages[(size_t) st - 1]->pcie_frac);
            if (pipe) for (int st = 0; st < split_drive.n; ++st) split_drive_b.pcie_num[st] = split_drive.pcie_num[st];
            const int hist_n = std::min(req_sp.penalty_last_n, kPenaltyWindowCap);
            ver.set_history(hist_n > 0 ? d_hist : nullptr, hist_n);
            if (pipe) ver_b.set_history(hist_n > 0 ? d_hist : nullptr, hist_n);
            bool cancelled = false;
            tr("prompt start", n - 1);
            // The prompt is read in two parts when it has a turn boundary past `resume`: up to the last <|im_start|>
            // (the conversation so far), a checkpoint there, then the new turn's header.  The next request of the same
            // chat renders the same history - but not always the same header or the thinking of this reply - so that
            // checkpoint is the one it reuses.
            int64_t turn_at = -1;
            if (o.prompt_cache > 0 && o.turn_token >= 0)
                for (int64_t i = n - 1; i > resume; --i)
                    if (ids[(size_t) i] == o.turn_token) { turn_at = i; break; }
            // #458 (opt-in): a short turn of --tail-role-token's role right before the new assistant turn (the
            // server's trailing reasoning-effort turn) stays out of the checkpoint, so the next request - another
            // effort, or the next turn of the chat - finds the conversation without it
            if (turn_at > 0 && o.tail_role_token >= 0)
                for (int64_t i = turn_at - 1; i > resume; --i)
                    if (ids[(size_t) i] == o.turn_token) {
                        if (ids[(size_t) i + 1] == o.tail_role_token) turn_at = i;
                        break;
                    }
            // A prompt read from token 0 also stops at its FIRST turn boundary: the end of the system prompt (with
            // the tools), which every new chat of the same client shares.  That checkpoint becomes the chain's root,
            // which the retention policy pins (conv_cache.hpp), so the next new chat reads only what comes after it.
            // (PR #65, code-martin.)  Only for a system prompt of --prompt-cache-root tokens or more: a small one
            // is cheaper to read again than the extra part costs (~0.3 s).
            const int64_t last_turn = turn_at;   // ckpt=0 drops the split there, the root is still looked for before it
            if (!req_ckpt) turn_at = -1;
            int64_t root_at = -1;
            if (o.prompt_cache > 0 && o.turn_token >= 0 && o.prompt_cache_root > 0 && (read_from == 0 || resumed_from0))
                for (int64_t i = 1; i < last_turn; ++i)
                    if (ids[(size_t) i] == o.turn_token) {
                        if (i >= o.prompt_cache_root) root_at = i;
                        break;
                    }
            static const bool message_checkpoint = [] {
                const char* e = std::getenv("STRATA_CACHE_MESSAGE_BOUNDARY");
                return e != nullptr && std::atoi(e) != 0;
            }();
            // Only add a snapshot; the existing token and image checks still decide reuse.
            const int64_t message_at = message_checkpoint && !multi_gpu && o.prompt_cache > 0
                ? strata::program::message_checkpoint_boundary(ids, resume, turn_at, o.turn_token) : -1;
            int64_t at = read_from;
            for (const int64_t to : {reread_to, root_at, message_at, turn_at, n - 1}) {
                if (to <= at) continue;
                err.clear();
                const bool win = windows_ok(at, to);
                if (win && !refill(err)) {
                    std::printf("ERR refilling a lent slot failed: %s\n", err.c_str());
                    return 1;
                }
                if (!win && !lend(to - at, err)) {
                    std::printf("ERR lending the prompt path its slots failed: %s\n", err.c_str());
                    return 1;
                }
                const auto tsp = Clock::now();
                const bool sp_ok = win ? read_windows(at, to, err) : read_part(at, to, err);
                if (trace) {
                    std::fprintf(stderr, "strata trace: read %lld tokens (%s) in %.1f ms\n", (long long) (to - at),
                                 win ? "windows" : "batched",
                                 std::chrono::duration<double, std::milli>(Clock::now() - tsp).count());
                    std::fflush(stderr);
                }
                if (!sp_ok) {
                    if (yielded_at < 0 && (batch_fatal || !stop_req.load())) {
                        std::fprintf(stderr, "strata serve: %s\n", err.c_str());
                        std::printf("ERR %s\n", err.c_str());
                        // #224: a CUDA fault (an illegal address) poisons the context for the whole process, and
                        // unwinding the destructors on it could hang until the 60 s watchdog: leave at once
                        /*
                        DPCT1010: SYCL uses exceptions to report errors and
                        does not use the error codes. The cudaPeekAtLastError
                        function call was replaced with 0. You need to rewrite
                        this code.
                        */
                        if (0 != 0) {
                            std::fflush(stdout);
                            std::fflush(stderr);
                            std::_Exit(1);
                        }
                        return 1;
                    }
                    cancelled = true;   // stopped while reading the prompt: refill the lent slots below, then DONE cancel
                    break;
                }
                at = to;
                if ((to == turn_at || to == root_at || to == message_at) &&
                    !checkpoint_at(to, nullptr, to == message_at && to != turn_at && to != root_at)) {   // the tail kind leaves first
                    std::printf("ERR saving a conversation checkpoint failed%s\n", ckpt_why.c_str());
                    return 1;
                }
                if (trace && to == message_at)
                    std::fprintf(stderr, "strata serve: message boundary checkpoint: %lld tokens, %lld tail\n",
                                 (long long) message_at, (long long) (turn_at - message_at));
            }
            if (!refill(err)) {
                std::printf("ERR refilling a lent slot failed: %s\n", err.c_str());
                return 1;
            }
            tr("prompt done (slots refilled)");
            if (il_parts > 0)
                std::fprintf(stderr, "strata batch: the prompt was read in %lld parts, the slots decoding %.0f ms between "
                                     "them\n", (long long) il_parts + 1, il_ms);
            const double prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
            std::printf("REUSED %lld\n", (long long) resume);   // the prompt is read; the first window comes next
            std::fflush(stdout);
            // the verify windows: the first holds the last prompt token alone
            int64_t p = n - 1;
            int32_t x = (int32_t) ids[(size_t) (n - 1)];
            // (the drafter writes max_t - 1 drafts: 8 rows with --pipeline-windows 2)
            const size_t DS = (size_t) std::max(S, use_mtp ? mtp.max_t() : 0);
            std::vector<int32_t> drafts(DS, 0), window((size_t) S), outv((size_t) S);
            std::vector<float> dprob(DS, 0.0f);
            std::vector<int32_t> sbuf((size_t) S, 0);
            const bool sfx_on = o.suffix_draft > 0 || o.lookup_chain > 0;
            if (sfx_on) {
                sfx.reset();
                for (int64_t t : ids) sfx.append((int32_t) t);
            }
            if (o.lookup_chain > 0) {
                extra_sources_reset();
                for (int64_t t : ids) { const int32_t t32 = (int32_t) t; extra_sources_append(&t32, 1); }
            }
            bool first_window = true;
            int64_t produced_n = 0, sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;
            int64_t chain_windows = 0, chain_drafts = 0, chain_ok = 0;   // --lookup-chain's own counts
            int64_t t2_rej[8] = {}, t2_hit[8] = {};   // STRATA_MTP_TOP2: rejections inside the MTP drafts by depth, runner-up hits
            std::vector<int32_t> cbuf((size_t) S, 0), ctail;
            strata::spec::PromptLookupSource lookup_src(sfx);
            static const bool chain_fixed = std::getenv("STRATA_LOOKUP_CHAIN_FIXED") != nullptr;   // no policy gate
            int64_t draft_offered = 0, draft_accepted = 0;
            // what the session holds once this request is done: the prompt read so far, then every committed token
            std::vector<int32_t> consumed;
            consumed.reserve((size_t) (n + max_new + S));
            for (int64_t i = 0; i < n - 1; ++i) consumed.push_back((int32_t) ids[(size_t) i]);
            const char* finish = "length";
            const Clock::time_point d0 = Clock::now();
            // STRATA_DECODE_TIMING=1: where a request's decode time goes (one line per request)
            static const bool dec_timing = std::getenv("STRATA_DECODE_TIMING") != nullptr;
            struct DecSnap {
                double wait, pool, host, plan, actq, jobs, run;
                int64_t misses, entries, hits, pcie;
            };
            auto dec_snap = [&]() {
                return DecSnap{ver.ms_wait, ver.ms_pool, ver.ms_host, drive.d.ms_plan, drive.d.ms_actq, drive.d.ms_jobs,
                               drive.d.ms_run, drive.d.multi_misses, drive.d.multi_entries, drive.d.cache_hits,
                               drive.d.pcie_experts};
            };
            const DecSnap ds0 = dec_snap();
            double dt_run = 0, dt_commit = 0, dt_draft = 0;
            int64_t dec_windows = 0, dec_T = 0;
            // STRATA_FOLLOW=<file> (test path, off by default): this request decodes the token ids in <file> (whitespace
            // separated, read and deleted at the request's start) instead of the argmax, through the real verify windows
            // with the MTP drafts; STRATA_FOLLOW_LOGPOS=<path> appends, for every valid window row, the teacher-forced
            // log-probability of the forced token and the top STRATA_LOGPOS_TOPK log-probabilities
            std::vector<int32_t> fol;
            if (const char* fp = std::getenv("STRATA_FOLLOW")) {
                if (std::FILE* ff = std::fopen(fp, "r")) {
                    long long v;
                    while (std::fscanf(ff, "%lld", &v) == 1) fol.push_back((int32_t) v);
                    std::fclose(ff);
                    std::remove(fp);
                }
            }
            static std::FILE* fol_lp = [] {
                const char* p = std::getenv("STRATA_FOLLOW_LOGPOS");
                return p != nullptr ? std::fopen(p, "ab") : nullptr;
            }();
            const int64_t decode_hits0 = drive.d.cache_hits;
            pcie_auto.mark(drive.d.ms_run, drive.d.multi_misses);   // the prompt's windows are not the decode's
            // CS-T: the RAM and file tiers of this request (the mmap source; 0 with the arena)
            const int64_t ram0 = src.ram_reads(), files0 = src.file_reads();
            const uint64_t file_bytes0 = src.file_read_bytes();
            const int64_t decode_look0 = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;
            const int64_t offload0 = drive.d.offload_entries;   // #588
            if (cancelled) finish = "cancel";
            // ======== --pipeline-windows 2: the decode windows with the two cards overlapped ========
            // Stage 0 (the first card) runs window K+1 while stage 1 verifies window K, on the guess that K is accepted
            // whole and that its bonus token is the drafter's.  The drafter, teacher-forced through K+1's drafts,
            // guesses K+1's bonus and drafts K+2 while stage 1 runs K+1.  A wrong guess costs the speculative window:
            // stage 0's GDN state comes back from a snapshot and K's commit is replayed with its real count (the
            // window's own commit inputs are intact in its parity's verifier).  Stage 1 never runs a speculative window
            // and a window row computes what it would in any other window, so every emitted token is the serial loop's
            // (with STRATA_IQ_MT_MIN=1 bit for bit; sampled requests with the same Philox draw per position).
            bool pl_ran = false;
            const bool pl_want = pipe && pl_pw >= 2 && pl_snap2[0] != nullptr;
            const char* pl_serial = !pl_want ? nullptr
                                  : hist_n > 0 ? "repetition penalties (penalty_last_n)"
                                  : mtp.coupled() ? "coupled draft sampling" : nullptr;
            if (pl_serial != nullptr) {   // said once per reason
                static std::set<std::string> said;
                if (said.insert(pl_serial).second)
                    std::fprintf(stderr, "strata serve: --pipeline-windows: a request with %s decodes serially\n",
                                 pl_serial);
            }
            if (pl_want && pl_serial == nullptr && !cancelled && produced_n < max_new) {
                pl_ran = true;
                if (!pl_prepare(err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                struct PW {
                    int T = 0, seq = 0;
                    int64_t p = 0;
                    int32_t tok[8] = {};
                    float prob[8] = {};    // prob[i]: the drafter's probability of tok[i + 1]
                    float p_on = 0.0f;     // the chain's estimate that the window before is accepted whole, tok[0] its bonus
                    bool ready = false, launched = false, finished = false, committed = false, s1 = false, s1_done = false;
                    bool spec = false;     // launched speculatively (behind the window before, ahead of its verdict)
                    bool made = false;     // made from a chain (its outcome can be scored even when the gate held it)
                    bool sfx = false;      // a lookup window (the suffix drafter's drafts)
                    int sfx_match = 0;
                };
                auto V0 = [&](const PW& w) -> strata::core::Verifier& { return *PV[0][w.seq & 1]; };
                auto V1 = [&](const PW& w) -> strata::core::Verifier& { return *PV[1][w.seq & 1]; };
                auto SDf = [&](const PW& w) { return (void*) PSD[w.seq & 1]; };
                const float theta = pl_theta;
                const int force_miss = pl_force_miss;
                static const bool pl_log = pipe_dbg_env("STRATA_PIPELINE_LOG") != nullptr;
                const int dev0 = PV[0][0]->device();
                dpct::queue_ptr s0 = PV[0][0]->stream();
                // the snapshot of the GDN state window `seq` reads (overlapped: before its launch, on the side stream)
                auto snap_take = [&](int seq) -> bool {
                    try {
                if (!pl_snap_overlap) return true;
                    const strata::core::OnDevice on(dev0);
                    const int q = seq & 1;
                    /*
                    DPCT1024: The original code returned the error code
                    that was further consumed by the program logic. This
                    original code was replaced with 0. You may need to rewrite
                    the program logic consuming the error code.
                    */
                    return DPCT_CHECK_ERROR(
                               dpct::sync_barrier(pl_snap_pre[q], s0)) == 0 &&
                           DPCT_CHECK_ERROR(
                               pl_snap_stream->ext_oneapi_submit_barrier(
                                   {*pl_snap_pre[q]})) == 0 &&
                           /*
                           DPCT1124: cudaMemcpyAsync is migrated to
                           asynchronous memcpy API. While the origin API might
                           be synchronous, it depends on the type of operand
                           memory, so you may need to call wait() on event
                           return by memcpy API to ensure synchronization
                           behavior.
                           */
                           DPCT_CHECK_ERROR(pl_snap_stream->memcpy(
                               pl_snap2[q], ss.gdn_state, pl_snap_bytes)) ==
                               0 &&
                           /*
                           DPCT1024: The original code returned the error
                           code that was further consumed by the program logic.
                           This original code was replaced with 0. You may need
                           to rewrite the program logic consuming the error
                           code.
                           */
                           DPCT_CHECK_ERROR(dpct::sync_barrier(
                               pl_snap_ev[q], pl_snap_stream)) == 0 &&
                           (DPCT_CHECK_ERROR(
                                (pl_snap_stream->ext_oneapi_empty())),
                            true);
                }
                catch (sycl::exception const &exc) {
                  std::cerr << exc.what()
                            << "Exception caught at file:" << __FILE__
                            << ", line:" << __LINE__ << std::endl;
                  std::exit(1);
                }
                };
                PW A, B, D;              // A: the oldest window not verified; B: the next; D: a wrong speculative window
                bool doomed = false;     // D still runs: when it finishes, stage 0 goes back to undo_w's real commit
                PW undo_w;
                int undo_keep = 0;
                int32_t undo_ple[2] = {-1, -1};
                int chain_kind = 0;      // 1: after an on-path verdict (A's drafts forced); 2: fresh (A's drafts from it)
                bool early_used = false;
                bool b_done = false;     // B has been made from the chain in flight
                int chain_n = 0;
                int64_t pl_spec = 0, pl_on = 0, pl_undo = 0, pl_gate = 0, pl_fm = 0;
                std::vector<int32_t> outp(8, 0);
                bool ending = false;
                A.T = 1;
                A.p = p;
                A.tok[0] = x;
                A.ready = true;
                drive.d.layers = 0;
                drive.d.experts = 0;
                drive.d.failed = false;
                // the adaptive tier beside the pipeline: the serial loop's rule and adapt() on a thread of its own (its
                // copies wait for the windows in flight, which this loop keeps serving meanwhile)
                std::thread pl_adapt_thr;
                std::atomic<bool> pl_adapt_done{true};
                bool pl_adapt_ok = true;
                // an error ends the engine, as the serial loop's do: every GPU wait is released first (#267), so no
                // window in flight keeps spinning on a flag nobody raises, and the tier's thread can finish
                auto die = [&](const std::string& m) -> int {
                    std::fprintf(stderr, "strata serve: pipelined decode: %s\n", m.c_str());
                    strata::core::diag_pipeline_fn().store(nullptr);
                    strata::core::release_gpu_waits(stderr);
                    if (pl_adapt_thr.joinable()) pl_adapt_thr.join();
                    std::printf("ERR %s\n", m.c_str());
                    std::fflush(stdout);
                    return 1;
                };
                // STRATA_PIPELINE_TRACE=<file>: every event of the loop with its time (appended per request)
                static const char* pl_trace_path = pipe_dbg_env("STRATA_PIPELINE_TRACE");
                std::string pl_trace;
                const Clock::time_point pl_tr0 = Clock::now();
                auto tre = [&](const char* ev, int seq, int x0 = 0, int x1 = 0) {
                    if (pl_trace_path == nullptr) return;
                    char b[96];
                    std::snprintf(b, sizeof b, "%.3f %s %d %d %d\n",
                                  std::chrono::duration<double, std::milli>(Clock::now() - pl_tr0).count(), ev, seq, x0, x1);
                    pl_trace += b;
                };
                auto pump0 = [&](PW& w) -> bool {
                    if (!w.launched || w.finished) return true;
                    strata::core::Verifier& v = V0(w);
                    // a doomed window (its guess was wrong) gets empty plans: no expert work on the CPU or the GPU for
                    // its remaining layers, so it frees stage 0 and the pool sooner (its rows are discarded and the
                    // undo restores everything it wrote).  STRATA_PIPELINE_DOOM_SKIP=0: served in full.
                    static const bool doom_skip = [] { const char* e = pipe_dbg_env("STRATA_PIPELINE_DOOM_SKIP"); return e == nullptr || std::atoi(e) != 0; }();
                    const bool skip = doom_skip && doomed && &w == &D;
                    if (v.service(skip ? nullptr : &drive_pool_split, SDf(w), err) < 0) return false;
                    if (v.done(err)) {
                        if (!v.pl_finish(nullptr, err)) return false;
                        w.finished = true;
                        tre("F0", w.seq, w.T, w.spec);
                    }
                    return err.empty();
                };
                // the serial loop's window size over the chain's outputs from index `from` (`avail` of them)
                auto t_rule = [&](const float* pr, int from, int avail) {
                    int T = S_mtp;
                    if (req_spec_min_p > 0.0) {
                        T = 1;
                        while (T < S_mtp && from + T - 1 < avail && pr[from + T - 1] >= (float) req_spec_min_p) ++T;
                    }
                    return std::max(1, std::min(T, 1 + avail - from));
                };
                // B from the chain's outputs: its row 0 is the guess at index `base`, its drafts follow
                auto make_b = [&](const int32_t* oc, const float* op, int base, int avail, float pa) {
                    B = PW{};
                    if (base >= avail) return;
                    B.seq = A.seq + 1;
                    B.p = A.p + A.T;
                    B.tok[0] = oc[base];
                    B.T = t_rule(op, base + 1, avail);
                    for (int i = 1; i < B.T; ++i) {
                        B.tok[i] = oc[base + i];
                        B.prob[i - 1] = op[base + i];
                    }
                    B.p_on = pa * op[base];
                    if (force_miss > 0 && ++pl_fm % force_miss == 0) B.tok[0] = B.tok[0] == 0 ? 1 : 0;
                    B.ready = true;
                    B.made = true;
                };
                // events after all the work both stages' streams hold now, and whether they have passed
                dpct::event_ptr fence_ev[2] = {nullptr, nullptr},
                                exch_ev[2] = {nullptr, nullptr};
                {
                    const strata::core::OnDevice on0(dev0);
                    fence_ev[0] = new sycl::event();
                    exch_ev[0] = new sycl::event();
                }
                {
                    const strata::core::OnDevice on1(stages[0]->dev);
                    fence_ev[1] = new sycl::event();
                    exch_ev[1] = new sycl::event();
                }
                auto stage_record = [&](dpct::event_ptr *ev) {
                    try {
                {
                        const strata::core::OnDevice on(dev0);
                        dpct::sync_barrier(ev[0], s0);
                        (void)DPCT_CHECK_ERROR((s0->ext_oneapi_empty()));
                    }
                    const strata::core::OnDevice on(stages[0]->dev);
                    dpct::sync_barrier(ev[1], PV[1][0]->stream());
                    (void)DPCT_CHECK_ERROR(
                        ((PV[1][0]->stream())->ext_oneapi_empty()));
                }
                catch (sycl::exception const &exc) {
                  std::cerr << exc.what()
                            << "Exception caught at file:" << __FILE__
                            << ", line:" << __LINE__ << std::endl;
                  std::exit(1);
                }
                };
                auto stage_passed = [&](dpct::event_ptr *ev) {
                    try {
                return dpct::sycl_event_query(ev[0]) != 1 &&
                       dpct::sycl_event_query(ev[1]) != 1;
                }
                catch (sycl::exception const &exc) {
                  std::cerr << exc.what()
                            << "Exception caught at file:" << __FILE__
                            << ", line:" << __LINE__ << std::endl;
                  std::exit(1);
                }
                };
                // adapt()'s fence, on the tier's thread: it waits on the HOST.  A device-side wait (the refill streams
                // waiting for the stages' events) would queue copies behind windows that need this loop, beside it.
                adapt_fence = [&]() {
                    stage_record(fence_ev);
                    while (!stage_passed(fence_ev)) std::this_thread::yield();
                };
                src.prepare_overrides();
                // The swaps land in two steps.  Once their copies are done, the incoming experts are resident (the plans
                // from then on use their slots).  What gives their RAM back - the resident RAM mode's exchange, the
                // released or unpinned pages - waits until every window that may still pull them over PCIe (a plan
                // published before that point) has completed: both stages' streams are fenced there.
                bool exch_wait = false;
                std::vector<int32_t> pl_landed;
                auto pl_release = [&]() {
                    unpin_blobs(pin_live);
                    src.commit_exchanges();
                    for (const int32_t i : pl_landed) srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
                    pl_landed.clear();
                    exch_wait = false;
                };
                auto pl_apply = [&]() {
                    try {
                if (exch_wait) {
                        if (stage_passed(exch_ev)) pl_release();
                        return;
                    }
                    if (pending.empty() ||
                        adapt_ev->get_info<
                            sycl::info::event::command_execution_status>() !=
                            sycl::info::event_command_status::complete) return;
                    for (auto& st : stages)
                        if (st->adapt_live &&
                            (st->adapt_ev)
                                    ->get_info<
                                        sycl::info::event::
                                            command_execution_status>() !=
                                sycl::info::event_command_status::
                                    complete) return;
                    for (auto& st : stages) st->adapt_live = false;
                    for (const auto& [i, slot] : pending) {
                        host_res[(size_t) i] = slot;
                        pl_landed.push_back(i);
                    }
                    pending.clear();
                    pending_age = 0;
                    res_upload();
                    stage_record(exch_ev);
                    exch_wait = true;
                }
                catch (sycl::exception const &exc) {
                  std::cerr << exc.what()
                            << "Exception caught at file:" << __FILE__
                            << ", line:" << __LINE__ << std::endl;
                  std::exit(1);
                }
                };
                auto pl_adapt = [&]() -> bool {
                    if (pl_adapt_thr.joinable() && pl_adapt_done.load()) {
                        pl_adapt_thr.join();
                        if (!pl_adapt_ok) return false;
                    }
                    if (pl_adapt_thr.joinable()) return true;   // still adapting
                    pl_apply();
                    if (!drive.d.usage.empty() && o.adapt_every > 0 && (rounds % o.adapt_every) == 0 && pending.empty() &&
                        !exch_wait) {
                        pl_adapt_done = false;
                        pl_adapt_thr = std::thread([&] { pl_adapt_ok = adapt(); pl_adapt_done = true; });
                    }
                    return true;
                };
                // the serial loop's prompt lookup for a fresh window: where the text repeats an earlier stretch and the
                // drafter's own first guess agrees, the policy may take the lookup's continuation (up to S rows)
                std::vector<int32_t> pl_sbuf((size_t) (2 * S), 0);
                // a lookup window's continuation goes on past it: the next window (B) is taken from the same lookup and
                // speculated behind it (a copied stretch is mostly accepted whole).  STRATA_PIPELINE_LOOKUP_PON: B's
                // estimate (default 0.75; 0 = no lookup B)
                static const float pl_lookup_pon = [] {
                    const char* v = pipe_dbg_env("STRATA_PIPELINE_LOOKUP_PON");
                    return v ? (float) std::atof(v) : 0.75f;
                }();
                auto pick_lookup = [&](PW& w, const int32_t* chain0) {
                    if (o.suffix_draft <= 0 || w.seq == 0) return;
                    const int k = sfx.propose(2 * S - 1, pl_sbuf.data());
                    const int match = sfx.last_match();
                    if (k <= 0 || pl_sbuf[0] != chain0[0]) return;
                    const strata::spec::DraftPolicy::Pick pk = policy.choose(w.T, std::min(k, S - 1), match);
                    if (!pk.lookup) return;
                    w.T = pk.t;
                    for (int i = 1; i < w.T; ++i) {
                        w.tok[i] = pl_sbuf[(size_t) i - 1];
                        w.prob[i - 1] = 1.0f;
                    }
                    w.sfx = true;
                    w.sfx_match = match;
                    const int rest = k - (w.T - 1);   // lookup tokens past w: B's row 0 (w's bonus guess) and its drafts
                    if (&w == &A && rest >= 1 && pl_lookup_pon > 0.0f && !b_done) {
                        B = PW{};
                        B.seq = A.seq + 1;
                        B.p = A.p + A.T;
                        B.T = std::min(S, rest);
                        for (int i = 0; i < B.T; ++i) B.tok[i] = pl_sbuf[(size_t) (A.T - 1 + i)];
                        for (int i = 1; i < B.T; ++i) B.prob[i - 1] = 1.0f;
                        B.p_on = pl_lookup_pon;
                        B.ready = true;
                        B.made = true;
                        b_done = true;   // the chain in flight makes no B for this A
                        tre("CD", A.seq + 1, 3, (int) (1000.0f * B.p_on));
                    }
                };
                const Clock::time_point pl_t0 = Clock::now();
                for (int st = 0; st < 2; ++st)
                    for (int par = 0; par < 2; ++par) g_pl_diag.v[st][par] = PV[st][par];
                strata::core::diag_pipeline_fn().store(&pl_diag_print);
                auto pw_bits = [](const PW& w) {
                    return (w.ready ? 1 : 0) | (w.launched ? 2 : 0) | (w.finished ? 4 : 0) | (w.committed ? 8 : 0) |
                           (w.s1 ? 16 : 0) | (w.s1_done ? 32 : 0);
                };
                // per window class (0: fresh, 1: run speculatively ahead of its predecessor's verdict): the time from
                // the previous verdict, the tokens it emitted, how many
                double cls_ms[2] = {0, 0}, cls_tok[2] = {0, 0}, cls_n[2] = {0, 0};
                int64_t cal_n[10] = {}, cal_on[10] = {};   // per p_on decile: windows scored, on the path
                double last_verdict = 0.0;
                int64_t pl_disagree = 0, pl_late = 0;
                static const bool pl_prestage = [] {   // STRATA_PIPELINE_PRESTAGE=0: B staged at its launch
                    const char* v = pipe_dbg_env("STRATA_PIPELINE_PRESTAGE");
                    return v == nullptr || std::atoi(v) != 0;
                }();
                static const bool pl_agree = [] {   // STRATA_PIPELINE_AGREE=0: only the old chain's probabilities
                    const char* v = pipe_dbg_env("STRATA_PIPELINE_AGREE");
                    return v == nullptr || std::atoi(v) != 0;
                }();
                auto ms_now = [&]() { return std::chrono::duration<double, std::milli>(Clock::now() - pl_t0).count(); };
                while (true) {
                    g_pl_diag.iters.fetch_add(1, std::memory_order_relaxed);
                    g_pl_diag.a_seq.store(A.seq, std::memory_order_relaxed);
                    g_pl_diag.a_bits.store(pw_bits(A), std::memory_order_relaxed);
                    g_pl_diag.b_bits.store(pw_bits(B), std::memory_order_relaxed);
                    g_pl_diag.d_bits.store(pw_bits(D), std::memory_order_relaxed);
                    g_pl_diag.chain_kind.store(chain_kind, std::memory_order_relaxed);
                    g_pl_diag.chain_live.store(mtp.chain_live() ? 1 : 0, std::memory_order_relaxed);
                    g_pl_diag.doomed.store(doomed ? 1 : 0, std::memory_order_relaxed);
                    g_pl_diag.ending.store(ending ? 1 : 0, std::memory_order_relaxed);
                    // ---- every window in flight served
                    if ((doomed && !pump0(D)) || !pump0(A) || !pump0(B)) return die(err);
                    if (A.s1 && !A.s1_done) {
                        strata::core::Verifier& v = V1(A);
                        if (v.service(&drive_pool_split, SDf(A), err) < 0) return die(err);
                        if (v.done(err)) {
                            if (!v.pl_finish(outp.data(), err)) return die(err);
                            A.s1_done = true;
                            tre("F1", A.seq, A.T);
                        } else if (!err.empty()) return die(err);
                    }
                    if (drive.d.failed) return die(drive.d.fail ? drive.d.fail : "the expert pool failed");
                    // ---- a wrong speculative window has finished: stage 0 back to the verified window's real commit
                    if (doomed && D.finished) {
                        {
                            const strata::core::OnDevice on(dev0);
                            /*
                            DPCT1124: cudaMemcpyAsync is migrated to
                            asynchronous memcpy API. While the origin API might
                            be synchronous, it depends on the type of operand
                            memory, so you may need to call wait() on event
                            return by memcpy API to ensure synchronization
                            behavior.
                            */
                            if (DPCT_CHECK_ERROR(s0->memcpy(
                                    ss.gdn_state,
                                    pl_snap2[pl_snap_overlap ? (undo_w.seq & 1)
                                                             : 0],
                                    pl_snap_bytes)) != 0)
                                return die("restoring the GDN snapshot failed");
                        }
                        ss.ple_prev[0] = undo_ple[0];
                        ss.ple_prev[1] = undo_ple[1];
                        if (!V0(undo_w).pl_commit_async(undo_keep, err)) return die(err);
                        doomed = false;
                        ++pl_undo;
                        tre("U", undo_w.seq, undo_keep);
                    }
                    if (ending) {
                        if (!doomed) break;
                        continue;
                    }
                    // ---- the drafter's chain, read as its outputs land (an event after each): a fresh A once its size
                    // is decided, then B once ITS size is decided - its bonus guess and a draft below spec_min_p (or
                    // S_mtp - 1 drafts) - so stage 0 need not wait for the whole chain
                    if (chain_kind != 0) {
                        const int k_ready = mtp.chain_outputs_ready(err);
                        if (k_ready < 0) return die(err);
                        if (chain_kind == 2 && !early_used) {
                            // A's size is decided by its first outputs: a draft below spec_min_p ends it (the serial
                            // loop's rule), so A goes to stage 0 as soon as that is known, not after all S_mtp - 1
                            const int k = k_ready;
                            bool decided = k >= S_mtp - 1;
                            if (!decided && req_spec_min_p > 0.0)
                                for (int j = 0; j < k; ++j)
                                    if (mtp.chain_prob()[j] < (float) req_spec_min_p) { decided = true; break; }
                            if (decided) {
                                A.T = t_rule(mtp.chain_prob(), 0, std::min(k, S_mtp - 1));
                                for (int i = 1; i < A.T; ++i) {
                                    A.tok[i] = mtp.chain_tok()[i - 1];
                                    A.prob[i - 1] = mtp.chain_prob()[i - 1];
                                }
                                pick_lookup(A, mtp.chain_tok());
                                A.ready = true;
                                early_used = true;
                                tre("CE", A.seq, A.T);
                            }
                        }
                        const int r = mtp.chain_poll(err);
                        if (r < 0) return die(err);
                        const int k = r == 1 ? chain_n : k_ready;
                        const int32_t* oc = mtp.chain_tok();
                        const float* op = mtp.chain_prob();
                        if (r == 1 && chain_kind == 2 && !early_used) {
                            A.T = t_rule(op, 0, std::min(chain_n, S_mtp - 1));
                            for (int i = 1; i < A.T; ++i) {
                                A.tok[i] = oc[i - 1];
                                A.prob[i - 1] = op[i - 1];
                            }
                            pick_lookup(A, oc);
                            A.ready = true;
                            early_used = true;
                        }
                        if (!b_done && (chain_kind == 1 || early_used)) {
                            const int base = A.T - 1;                    // B's row 0: the guess of A's bonus
                            bool decided = r == 1 || k >= base + S_mtp;
                            if (!decided && k > base + 1 && req_spec_min_p > 0.0)
                                for (int j = base + 1; j < k; ++j)
                                    if (op[j] < (float) req_spec_min_p) { decided = true; break; }
                            if (decided) {
                                float pa = 1.0f;
                                if (chain_kind == 1) {
                                    // forced through A's drafts: its outputs there are the drafter's own picks from
                                    // A's verified predecessor, a better estimate of A's acceptance than the chain that
                                    // drafted A; a disagreement means A is unlikely to be accepted whole
                                    bool agree = true;
                                    for (int i = 0; i + 1 < A.T; ++i) {
                                        if (oc[i] != A.tok[i + 1]) agree = false;
                                        pa *= pl_agree ? op[i] : A.prob[i];
                                    }
                                    if (pl_agree && !agree) { pa = 0.0f; ++pl_disagree; }
                                } else {
                                    for (int i = 0; i + 1 < A.T; ++i) pa *= A.prob[i];
                                }
                                if (!A.sfx) make_b(oc, op, base, k, pa);   // a lookup window's bonus is not the chain's
                                // stage B now (its PLE rows from the tokens before it as they will be once A is
                                // committed whole), so its launch behind A is only the graph launch (never while a
                                // rollback is pending: B's verifier is the one the undo commit reads)
                                if (B.ready && B.p_on >= theta && pl_prestage && !V0(B).in_flight() && !doomed) {
                                    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
                                    for (int i = 0; i < A.T; ++i) { prev[0] = prev[1]; prev[1] = A.tok[i]; }
                                    if (!V0(B).prestage(B.T, B.tok, B.p, prev, err)) return die(err);
                                }
                                b_done = true;
                                tre("CD", A.seq, chain_kind, (int) (1000.0f * B.p_on));
                                if (pl_log)
                                    std::fprintf(stderr, "strata pipeline: chain %d for p=%lld T=%d: B T=%d p_on %.3f (%d "
                                                         "outputs)\n", chain_kind, (long long) A.p, A.T, B.T, B.p_on, k);
                            }
                        }
                        if (r == 1) chain_kind = 0;
                    }
                    // ---- stage 0: A (never while a wrong window still holds stage 0's state)
                    if (A.ready && !A.launched && !doomed) {
                        if (A.p + A.T > o.max_context) { ending = true; continue; }
                        if (!snap_take(A.seq)) return die("the GDN snapshot failed");
                        if (!V0(A).pl_launch(A.T, A.tok, A.p, err)) return die(err);
                        A.launched = true;
                        tre("L0", A.seq, A.T, 0);
                    }
                    // ---- stage 0: B, speculatively, right behind A
                    if (B.ready && !B.launched && A.finished && !A.committed && !doomed) {
                        if (B.p_on >= theta && B.p + B.T <= o.max_context) {
                            {
                                const strata::core::OnDevice on(dev0);
                                if (pl_snap_overlap
                                        ? DPCT_CHECK_ERROR(
                                              s0->ext_oneapi_submit_barrier(
                                                  {*pl_snap_ev[A.seq & 1]})) !=
                                              0
                                        /*
                                        DPCT1124: cudaMemcpyAsync is
                                        migrated to asynchronous memcpy API.
                                        While the origin API might be
                                        synchronous, it depends on the type of
                                        operand memory, so you may need to call
                                        wait() on event return by memcpy API to
                                        ensure synchronization behavior.
                                        */
                                        : DPCT_CHECK_ERROR(s0->memcpy(
                                              pl_snap2[0], ss.gdn_state,
                                              pl_snap_bytes)) != 0)
                                    return die("the GDN snapshot failed");
                            }
                            undo_ple[0] = ss.ple_prev[0];
                            undo_ple[1] = ss.ple_prev[1];
                            if (!V0(A).pl_commit_async(A.T, err) || !snap_take(B.seq) ||
                                !V0(B).pl_launch(B.T, B.tok, B.p, err))
                                return die(err.empty() ? std::string("the GDN snapshot failed") : err);
                            A.committed = true;
                            B.launched = true;
                            B.spec = true;
                            tre("L0", B.seq, B.T, 1);
                            ++pl_spec;
                        } else {
                            B.ready = false;   // not worth it: A's verdict decides the next window
                            ++pl_gate;
                        }
                    }
                    // ---- stage 1: A, once its stage 0 is done
                    if (A.finished && !A.s1) {
                        if (chain_kind == 1 && !B.ready) ++pl_late;   // stage 0 idles until the chain has B
                        if (!V1(A).pl_launch(A.T, A.tok, A.p, err)) return die(err);
                        A.s1 = true;
                        tre("L1", A.seq, A.T);
                    }
                    // ---- A's verdict
                    if (!A.s1_done) continue;
                    // the drafter must be idle before the next chain.  The chain needs nothing from the host, so this
                    // wait is bounded even with B (made from its first outputs) in flight on stage 0; in practice the
                    // chain ended long before a verdict that needs a whole stage-1 window.
                    while (chain_kind != 0 && mtp.chain_live())
                        if (mtp.chain_poll(err) < 0) return die(err);
                    if (chain_kind != 0 && !B.launched && !B.made) B = PW{};
                    chain_kind = 0;
                    int a = 0;
                    while (a < A.T - 1 && A.tok[a + 1] == outp[(size_t) a]) ++a;
                    if (!V1(A).pl_commit_async(a + 1, err)) return die(err);
                    for (int i = 0; i <= a; ++i) consumed.push_back(A.tok[i]);
                    draft_offered += A.T - 1;
                    draft_accepted += a;
                    ++rounds;
                    ++dec_windows;
                    dec_T += A.T;
                    bool eos = false;
                    for (int i = 0; i <= a && produced_n < max_new && !eos; ++i) {
                        std::printf("T %d\n", (int) outp[(size_t) i]);
                        strata::core::progress_beat();
                        ++produced_n;
                        if (o.suffix_draft > 0) sfx.append(outp[(size_t) i]);
                        eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outp[(size_t) i]) != o.eos_ids.end();
                    }
                    std::fflush(stdout);
                    if (eos) finish = "stop";
                    else if (stop_req.load()) finish = "cancel";
                    const bool last = eos || produced_n >= max_new || stop_req.load();
                    const bool on = B.launched && a == A.T - 1 && B.tok[0] == outp[(size_t) A.T - 1];
                    if (B.made) {   // the gate's calibration: would B have been on the path, by its estimate p_on
                        const bool would = a == A.T - 1 && B.tok[0] == outp[(size_t) A.T - 1];
                        const int bin = std::min(9, std::max(0, (int) (B.p_on * 10.0f)));
                        cal_n[bin] += 1;
                        cal_on[bin] += would ? 1 : 0;
                    }
                    tre("V", A.seq, a, on ? 1 : (B.launched ? -1 : 0));
                    {
                        const double now = ms_now();
                        const int c = A.spec ? 1 : 0;
                        if (A.sfx) { ++sfx_windows; sfx_drafts += A.T - 1; sfx_ok += a; }
                        if (A.seq > 0 && !last) policy.observe(A.sfx, A.T, a, A.sfx_match, now - last_verdict);
                        cls_ms[c] += now - last_verdict;
                        cls_tok[c] += a + 1;
                        cls_n[c] += 1;
                        last_verdict = now;
                    }
                    if (pl_log)
                        std::fprintf(stderr, "strata pipeline: verdict p=%lld T=%d a=%d B %s%s\n", (long long) A.p, A.T, a,
                                     B.launched ? "launched" : B.ready ? "ready" : "none", on ? " ON" : "");
                    x = outp[(size_t) a];
                    p = A.p + a + 1;
                    if (!last && !pl_adapt()) return die("an adaptive refill failed");
                    if (on && !last) {
                        ++pl_on;
                        // the chain over A, forced through B's drafts: B's bonus guess and the next window's drafts
                        mtp.set_source_R(V1(A).final_R(0));
                        chain_n = std::min(strata::kernels::kVerifyMaxT - 1, B.T + S_mtp - 1);
                        if (!mtp.chain_launch(A.T, outp.data(), A.p, a, B.tok + 1, B.T - 1, chain_n, chain_n, err))
                            return die(err);
                        chain_kind = 1;
                        tre("CL", A.seq, 1);
                        early_used = false;
                        b_done = false;
                        A = B;
                        B = PW{};
                    } else {
                        if (B.launched) {   // a wrong guess (or the end): undo once it has finished
                            doomed = true;
                            D = B;
                            undo_w = A;
                            undo_keep = a + 1;
                        } else if (!A.committed) {
                            if (!V0(A).pl_commit_async(a + 1, err)) return die(err);
                        }
                        // B is D now (or dropped): never pump it twice (whichever copy saw the window complete first
                        // would finish it, and the other would wait for it forever)
                        B = PW{};
                        if (!last) {
                            mtp.set_source_R(V1(A).final_R(0));
                            chain_n = strata::kernels::kVerifyMaxT - 1;
                            if (!mtp.chain_launch(A.T, outp.data(), A.p, a, nullptr, 0, chain_n, chain_n, err))
                                return die(err);
                            chain_kind = 2;
                            tre("CL", A.seq, 2);
                            early_used = false;
                            b_done = false;
                            PW nA;
                            nA.seq = A.seq + 1;
                            nA.p = A.p + a + 1;
                            nA.tok[0] = outp[(size_t) a];
                            A = nA;
                        }
                    }
                    if (last) ending = true;
                }
                pl_drain();   // nothing in flight: both stages' commits and the drafter have landed
                strata::core::diag_pipeline_fn().store(nullptr);
                if (pl_adapt_thr.joinable()) pl_adapt_thr.join();   // its copies waited only for finished work
                adapt_fence = nullptr;
                if (exch_wait) pl_release();   // every window has completed (drained)
                for (dpct::event_ptr ev : fence_ev) if (ev)
                    dpct::destroy_event(ev);
                for (dpct::event_ptr ev : exch_ev) if (ev)
                    dpct::destroy_event(ev);
                if (pl_trace_path != nullptr && !pl_trace.empty()) {
                    if (std::FILE* f = std::fopen(pl_trace_path, "a")) {
                        std::fprintf(f, "# request p0=%lld\n", (long long) (n - 1));
                        std::fwrite(pl_trace.data(), 1, pl_trace.size(), f);
                        std::fclose(f);
                    }
                }
                // the odd windows' counters and GPU profiles into the verifiers that report them
                PV[0][0]->absorb_stats(*PV[0][1]);
                PV[1][0]->absorb_stats(*PV[1][1]);
                if (!pl_adapt_ok) {
                    std::printf("ERR an adaptive refill failed\n");
                    return 1;
                }
                while (mtp.chain_live() && mtp.chain_poll(err) == 0) {}
                mtp.set_source_R(ver.final_R_all());   // the serial loop's rows again
                const double pl_ms = std::chrono::duration<double, std::milli>(Clock::now() - pl_t0).count();
                dt_run += pl_ms;
                if (dec_timing) {
                    auto avg = [](double s_, double n_) { return n_ > 0 ? s_ / n_ : 0.0; };
                    std::fprintf(stderr, "strata pipeline: %lld windows in %.0f ms (%.2f ms/window): %lld speculative, %lld on "
                                         "the path, %lld rolled back, %lld below the gate (theta %.2f)\n",
                                 (long long) dec_windows, pl_ms, avg(pl_ms, (double) dec_windows), (long long) pl_spec,
                                 (long long) pl_on, (long long) pl_undo, (long long) pl_gate, theta);
                    std::fprintf(stderr, "strata pipeline classes: fresh %.0f windows %.2f ms %.2f tok | speculative %.0f "
                                         "windows %.2f ms %.2f tok | forced-chain disagreements %lld, chain late %lld\n",
                                 cls_n[0], avg(cls_ms[0], cls_n[0]), avg(cls_tok[0], cls_n[0]), cls_n[1],
                                 avg(cls_ms[1], cls_n[1]), avg(cls_tok[1], cls_n[1]), (long long) pl_disagree,
                                 (long long) pl_late);
                    std::string cal;
                    char b[48];
                    for (int i = 0; i < 10; ++i)
                        if (cal_n[i] > 0) {
                            std::snprintf(b, sizeof b, " %.1f:%lld/%lld", i / 10.0, (long long) cal_on[i], (long long) cal_n[i]);
                            cal += b;
                        }
                    std::fprintf(stderr, "strata pipeline calibration (p_on decile: on/scored):%s\n", cal.c_str());
                }
            }
            while (!pl_ran && !cancelled && produced_n < max_new) {
                int T = use_mtp ? S_mtp : 1;   // no --mtp: one token a round unless a lookup draft fires
                if (use_mtp && req_spec_min_p > 0.0) {
                    T = 1;
                    while (T < S_mtp && dprob[(size_t) T - 1] >= (float) req_spec_min_p) ++T;
                }
                if (first_window) T = 1;
                // a repeat of earlier context (prompt lookup) where the MTP's own first guess agrees: the policy takes it
                // when its expected tokens per ms, from the measured acceptance and window costs, beat the MTP window's
                bool from_sfx = false;
                int sfx_match = 0;
                if (o.suffix_draft > 0 && !first_window) {
                    const int k = sfx.propose(S - 1, sbuf.data());
                    sfx_match = sfx.last_match();
                    if (k > 0 && (!use_mtp || sbuf[0] == drafts[0])) {
                        const strata::spec::DraftPolicy::Pick pk = policy.choose(T, k, sfx_match);
                        if (pk.lookup) { T = pk.t; from_sfx = true; }
                    }
                }
                // --lookup-chain: what followed an earlier occurrence of the context + the MTP's drafts, after them
                int chain_n = 0, cm = 0;
                if (o.lookup_chain > 0 && !first_window && !from_sfx && T < S) {
                    int csrc = -1;
                    const int nt = chain_tail(sfx, drafts.data(), T - 1, ctail);
                    chain_n = strata::spec::propose_from_sources(lookup_src, ctail.data(), nt, T - 1,
                                                                 std::min(o.lookup_chain, S - T), o.lookup_chain_min,
                                                                 cbuf.data(), &cm, &csrc);
                    // the policy keeps the chained tokens that pay at this machine's measured window costs
                    if (chain_n > 0 && !chain_fixed) {
                        double p_mtp = 1.0;
                        for (int i = 0; i < T - 1; ++i) p_mtp *= dprob[(size_t) i];
                        chain_n = policy.chain(T, p_mtp, chain_n, cm);
                    }
                }
                const int T_mtp = T;
                T += chain_n;
                const bool timed_round = !first_window;
                const Clock::time_point round0 = Clock::now();
                if (p + T > o.max_context) break;
                window[0] = x;
                for (int i = 1; i < T_mtp; ++i) window[(size_t) i] = from_sfx ? sbuf[(size_t) i - 1] : drafts[(size_t) i - 1];
                for (int i = 0; i < chain_n; ++i) window[(size_t) (T_mtp + i)] = cbuf[(size_t) i];
                drive.d.layers = 0;
                drive.d.experts = 0;
                drive.d.failed = false;
                // #463: the previous adapt round's copies land first - with a non-blocking query, whether a swapped-in
                // expert ran on the GPU or the CPU (they round differently) depended on the copy's timing
                // (STRATA_ADAPT_NOWAIT=1: 0.1.37's non-blocking query, the A/B)
                if (adapt_nowait()) apply_pending(false);
                else if (pending.empty() || ++pending_age >= adapt_lag()) apply_pending(true);
                if (!adapt_tick(false)) {   // --adapt-async: the round in flight moves on a step when it can
                    std::printf("ERR an adaptive refill failed\n");
                    return 1;
                }
                if (hist_n > 0) {
                    // the tails the penalties count over, ONE PER ROW: the tokens the state has consumed, the
                    // fed-back head `x` (it joins `consumed` only after this window commits), then the drafts
                    // before that row - what plain decode would have counted there.  (Until 0.1.19 only row 0
                    // was staged, and the drafted rows read unwritten slots.)
                    strata::kernels::penalty_rows(consumed.data(), (int64_t) consumed.size(), window.data(), T,
                                                  hist_n, hist_stage.data());
                    const strata::core::OnDevice on_h(hist_dev);
                    dpct::get_in_order_queue()
                        .memcpy(d_hist, hist_stage.data(),
                                (size_t)T * (size_t)hist_n * sizeof(int32_t))
                        .wait();
                }
                // the elastic K/V: the window's cells and the drafter's beyond them
                if (kvg.on && p + T + 64 > kvg.cells &&
                    !kvg_ensure(p + T + 64, [&] {
                                                  dpct::get_current_device()
                                                      .queues_wait_and_throw();
                                                      apply_pending(true); })) {
                    std::printf("ERR the K/V cannot grow: no VRAM is left\n");
                    return 1;
                }
                tr("window", p, T);
                const Clock::time_point tw0 = Clock::now();
                if (!ver.run(T, window.data(), p, win_pool_fn, win_pool_user, outv.data(), err) || drive.d.failed) {
                    std::printf("ERR %s\n", drive.d.failed && drive.d.fail ? drive.d.fail : err.c_str());
                    return 1;
                }
                int32_t fol_tgt[16];
                if (!fol.empty()) {   // STRATA_FOLLOW: the forced continuation stands in for the argmax
                    for (int i = 0; i < T && i < 16; ++i) {
                        const size_t k = (size_t) produced_n + (size_t) i;
                        fol_tgt[i] = k < fol.size() ? fol[k] : -1;
                        if (fol_tgt[i] >= 0) outv[(size_t) i] = fol_tgt[i];
                    }
                }
                int a = 0;
                while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
                if (!fol.empty() && fol_lp != nullptr && !ver.window_logprobs(fol_tgt, a + 1, produced_n, -1, fol_lp, err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
                if (chain_n > 0) { ++chain_windows; chain_drafts += chain_n; chain_ok += std::max(0, a - (T_mtp - 1)); }
                if (strata::core::MtpDrafter::top2_env() && !from_sfx && !first_window && a < T_mtp - 1 && a < 8) {
                    ++t2_rej[a];
                    if (mtp.top2(a) == outv[(size_t) a]) ++t2_hit[a];
                }
                const Clock::time_point tw1 = Clock::now();
                std::thread adapt_thr;   // the adaptive tier beside the commit and the draft (as in generate)
                bool adapt_ok = true;
                // (--adapt-async 1: the asynchronous tier above instead, ticked before the window)
                if (!drive.d.usage.empty() && !ajob && ((rounds + 1) % o.adapt_every) == 0)
                    adapt_thr = std::thread([&] { adapt_ok = adapt(); });
                if (!ver.commit(a + 1, err)) {
                    if (adapt_thr.joinable()) adapt_thr.join();
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                // the window's first a + 1 tokens are in the session now (the last output is not: it is next x)
                for (int i = 0; i <= a; ++i) consumed.push_back(window[(size_t) i]);
                draft_offered += T - 1;
                draft_accepted += a;
                first_window = false;
                bool eos = false;
                for (int i = 0; i <= a && produced_n < max_new && !eos; ++i) {
                    std::printf("T %d\n", (int) outv[(size_t) i]);
                    if (req_logprobs >= 0) {   // row i of this window's head logits is the distribution token i came from
                        static std::vector<float> lrow;
                        static std::vector<int32_t> lord;
                        lrow.resize((size_t) ver.vocab());
                        if (!ver.copy_logits(i, lrow.data())) {
                            std::printf("LP nan\n");
                        } else {
                            float mx = -INFINITY;
                            for (const float v : lrow) mx = std::max(mx, v);
                            double se = 0.0;
                            for (const float v : lrow) se += std::exp((double) v - mx);
                            const double lse = (double) mx + std::log(se);
                            std::printf("LP %.6f", (double) lrow[(size_t) outv[(size_t) i]] - lse);
                            if (req_logprobs > 0) {
                                lord.resize(lrow.size());
                                for (size_t v = 0; v < lrow.size(); ++v) lord[v] = (int32_t) v;
                                std::partial_sort(lord.begin(), lord.begin() + req_logprobs, lord.end(),
                                                  [&](int32_t x1, int32_t x2) { return lrow[(size_t) x1] > lrow[(size_t) x2]; });
                                for (int j = 0; j < req_logprobs; ++j)
                                    std::printf(" %d:%.6f", lord[(size_t) j], (double) lrow[(size_t) lord[(size_t) j]] - lse);
                            }
                            std::printf("\n");
                        }
                    }
                    strata::core::progress_beat();
                    ++produced_n;
                    if (sfx_on) sfx.append(outv[(size_t) i]);
                    if (o.lookup_chain > 0) extra_sources_append(&outv[(size_t) i], 1);
                    eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                }
                std::fflush(stdout);
                ++rounds;
                const Clock::time_point tw2 = Clock::now();
                // coupled drafts with penalties: the next window's row-0 history (`consumed` holds this window's
                // commit, outv[a] is its row 0) - the drafts extend it on the device as the verify rows will
                if (use_mtp && hist_n > 0 && mtp.coupled() && !eos && produced_n < max_new)
                    mtp.set_draft_history(consumed.data(), (int64_t) consumed.size(), outv[(size_t) a]);
                const bool drafted = !use_mtp || eos || produced_n >= max_new ||
                                     mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data(), (float) req_spec_min_p);
                {
                    const Clock::time_point tw3 = Clock::now();
                    auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
                    dt_run += msd(tw0, tw1); dt_commit += msd(tw1, tw2); dt_draft += msd(tw2, tw3);
                    ++dec_windows; dec_T += T;
                }
                pcie_auto_step(req_pcie_frac);   // PcieAuto (opt-in): the share refined from the CPU pool's speed
                if (adapt_thr.joinable()) adapt_thr.join();
                if (!adapt_ok) {
                    std::printf("ERR an adaptive refill failed\n");
                    return 1;
                }
                if (!drafted) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                if (timed_round && !eos && chain_n == 0)
                    policy.observe(from_sfx, T, a, sfx_match,
                                   std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
                else if (timed_round && !eos)   // a chained window: its MTP part, its chain, its cost
                    policy.observe_chain(T_mtp, chain_n, a, cm,
                                         std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
                if (eos) { finish = "stop"; break; }
                if (stop_req.load()) { finish = "cancel"; break; }
                x = outv[(size_t) a];
                p += a + 1;
            }
            const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
            // the last commit (set_commit_async): the session is complete before anything reads or copies it
            if (!ver.wait_commit(err)) {
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            if (dec_timing && dec_windows > 0) {
                const DecSnap d1 = dec_snap();
                const double w = (double) dec_windows, L = (double) g.n_layers;
                std::fprintf(stderr, "strata decode timing: %lld windows, avg T %.2f, %.2f tokens/window, %.2f ms/window = "
                                     "verify %.2f (GPU-reach wait %.2f + per-layer host %.2f [plan %.2f actq %.2f jobs %.2f "
                                     "CPU %.2f] + stage %.2f) + commit/emit %.2f + draft %.2f; per layer-window: CPU experts "
                                     "%.2f (%.2f entries), VRAM hits %.2f, PCIe %.2f\n",
                             (long long) dec_windows, dec_T / w, produced_n / w, decode_ms / w, dt_run / w,
                             (d1.wait - ds0.wait) / w, (d1.pool - ds0.pool) / w, (d1.plan - ds0.plan) / w,
                             (d1.actq - ds0.actq) / w, (d1.jobs - ds0.jobs) / w, (d1.run - ds0.run) / w,
                             (d1.host - ds0.host) / w, dt_commit / w, dt_draft / w, (d1.misses - ds0.misses) / (w * L),
                             (d1.entries - ds0.entries) / (w * L), (d1.hits - ds0.hits) / (w * L), (d1.pcie - ds0.pcie) / (w * L));
                for (int st = 0; st < n_stages; ++st) {   // every stage's GPU profile, not only the first card's
                    const std::string pr = stage_ver(st).profile_report();
                    if (pr.empty()) continue;
                    if (st == 0) std::fprintf(stderr, "strata decode GPU stages (ms/window):%s\n", pr.c_str());   // text as before
                    else std::fprintf(stderr, "strata decode GPU stages, stage %d (ms/window):%s\n", st, pr.c_str());
                }
            }
            if (!cancelled) {
                // a prompt stopped halfway leaves the session somewhere between two chunks: nothing to continue from
                // (the checkpoints taken while reading it are still good)
                live.swap(consumed);
                live_imgs = imgs_below(req_imgs, (int64_t) live.size());
                live_ok = o.prompt_cache > 0 && req_ckpt;   // ckpt=0: nothing to continue or park (#830)
            }
            static const bool state_hash = std::getenv("STRATA_STATE_HASH") != nullptr;
            if (state_hash && !cancelled && o.prompt_cache > 0) {   // every finished request, ckpt=0 too (parity gates)
                // DEBUG: a fingerprint of every part of the session over the positions it holds ([0, L)), and
                // separately of what lies past them in the last KV page (stale cells, fine unless something reads them)
                if (DPCT_CHECK_ERROR(
                        dpct::get_current_device().queues_wait_and_throw()) !=
                    0) {
                    std::printf("ERR synchronizing state fingerprint\n");
                    return 1;
                }
                const int64_t L = (int64_t) live.size();
                const strata::kernels::QsaShapes qs = strata::core::qsa_shapes(g);
                bool hash_ok = true;
                std::array<uint8_t, 65536> hash_buffer;
                auto hash_dev = [&](const void *p, size_t bytes, uint64_t h) {
                    try {
                for (size_t offset = 0; hash_ok && offset < bytes;) {
                        const size_t n = std::min(hash_buffer.size(), bytes - offset);
                        // VRAM or a streamed host copy, with fixed diagnostic workspace.
                        if (DPCT_CHECK_ERROR(
                                dpct::get_in_order_queue()
                                    .memcpy(hash_buffer.data(),
                                            static_cast<const uint8_t *>(p) +
                                                offset,
                                            n)
                                    .wait()) != 0) {
                            hash_ok = false;
                            break;
                        }
                        h = fnv1a(hash_buffer.data(), n, h);
                        offset += n;
                    }
                    return h;
                }
                catch (sycl::exception const &exc) {
                  std::cerr << exc.what()
                            << "Exception caught at file:" << __FILE__
                            << ", line:" << __LINE__ << std::endl;
                  std::exit(1);
                }
                };
                // the cells [c0, c1) of one int8 K or V pool ([page][kv_head][page_size][head_dim]), bytes per value `w`
                auto hash_cells = [&](const void* pool, int64_t per_cell, int64_t c0, int64_t c1, uint64_t h) {
                    const int64_t ps = qs.page_size;
                    for (int64_t pg = c0 / ps; pg * ps < c1; ++pg)
                        for (int64_t hd = 0; hd < qs.n_head_kv; ++hd) {
                            const int64_t a = std::max(c0, pg * ps) - pg * ps, e = std::min(c1, (pg + 1) * ps) - pg * ps;
                            const size_t off = (size_t) (((pg * qs.n_head_kv + hd) * ps + a) * per_cell);
                            h = hash_dev((const uint8_t*) pool + off, (size_t) ((e - a) * per_cell), h);
                        }
                    return h;
                };
                const ConvStateSizes z = conv_state_sizes(g, ss);
                uint64_t h_gdn = hash_dev(ss.gdn_state, z.gdn, 1469598103934665603ull);
                if (std::getenv("STRATA_STATE_HASH_GDN") != nullptr && ss.gdn_alloc > 0) {   // per GDN layer: which one differs first
                    const size_t per = z.gdn / (size_t) ss.gdn_alloc;
                    std::string s;
                    char b[8];
                    for (int64_t i = 0; i < ss.gdn_alloc; ++i) {
                        std::snprintf(b, sizeof(b), "%04llx ", (unsigned long long) (hash_dev((const uint8_t*) ss.gdn_state + i * per, per, 1469598103934665603ull) & 0xffff));
                        s += b;
                    }
                    std::fprintf(stderr, "strata serve: STATE_HASH_GDN %s\n", s.c_str());
                }
                uint64_t h_ple = hash_dev(ss.ple_hist, ss.ple_hist ? z.ple : 0, 1469598103934665603ull);
                uint64_t h_tail = 1469598103934665603ull, h_pool = h_tail, h_kv = h_tail, h_stale = h_tail;
                // pooled= keeps its 0.1.29 meaning: the completed rows [0, L / idx_block) only.  pooled_full= adds the
                // spare row at L / idx_block (the `dead` key the next block completion overwrites), which a
                // conversation restore writes back; dead= is the spare key itself
                uint64_t h_dead = h_tail, h_pool_full = h_tail;
                const int64_t kvb = qs.head_dim, scb = (qs.head_dim / 64) * 2;
                // a state's K/V arrays and their bytes per (cell, head) row: the host copy when it has one
                auto kv_arrays = [&](const strata::core::QsaState& st) {
                    const bool h = st.kv_mode != 0;
                    std::vector<std::pair<const void*, int64_t>> a;
                    if (st.kv_q4) {
                        const int64_t q4b = (int64_t) strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                        a = {{h ? st.host.k_q4 : st.k_q4, q4b}, {h ? st.host.v_q4 : st.v_q4, q4b}};
                    } else if (st.kv_hybrid) {
                        const int64_t q4b = (int64_t) strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                        a = {{h ? st.host.k_q : st.k_q, kvb}, {h ? st.host.v_q4 : st.v_q4, q4b},
                             {h ? st.host.k_scale : st.k_scale, scb}};
                    } else if (st.kv_int8) {
                        a = {{h ? st.host.k_q : st.k_q, kvb}, {h ? st.host.v_q : st.v_q, kvb},
                             {h ? st.host.k_scale : st.k_scale, scb}, {h ? st.host.v_scale : st.v_scale, scb}};
                    } else {
                        a = {{h ? st.host.k_pool : st.k_pool, qs.head_dim * 2},
                             {h ? st.host.v_pool : st.v_pool, qs.head_dim * 2}};
                    }
                    return a;
                };
                const int64_t end_cell = std::min<int64_t>(((L + qs.page_size - 1) / qs.page_size) * qs.page_size,
                                                           ss.max_cells);   // = the primary state's max_cells
                for (int64_t j = 0; j < ss.qsa_alloc; ++j) {   // this session's owned QSA ordinals only
                    const strata::core::QsaState& st = ss.qsa_states[ss.qsa_ord0 + j];
                    h_tail = hash_dev(st.idx_tail, z.tail, h_tail);
                    h_dead = hash_dev(st.idx_dead, z.dead, h_dead);
                    h_pool = hash_dev(st.idx_pooled, (size_t) (L / qs.idx_block) * qs.idx_dim * 4, h_pool);
                    h_pool_full = hash_dev(st.idx_pooled, (size_t) (L > 0 ? L / qs.idx_block + 1 : 0) * qs.idx_dim * 4,
                                           h_pool_full);
                    // KV streaming: the host copy is the identity layout and holds every cell
                    for (const auto& [pool, w] : kv_arrays(st)) {
                        h_kv = hash_cells(pool, w, 0, L, h_kv);
                        h_stale = hash_cells(pool, w, L, end_cell, h_stale);
                    }
                }
                uint64_t h_mtp = 1469598103934665603ull;
                if (use_mtp) {
                    const strata::core::QsaState& ms = mtp.kv_state();
                    const int64_t mL = std::min<int64_t>(L, ms.max_cells);
                    for (const auto& [pool, w] : kv_arrays(ms))
                        if (pool != nullptr) h_mtp = hash_cells(pool, w, 0, mL, h_mtp);
                }
                if (!hash_ok) {
                    std::printf("ERR reading state fingerprint\n");
                    return 1;
                }
                std::fprintf(stderr, "strata serve: STATE_HASH L=%lld gdn=%016llx ple=%016llx tail=%016llx pooled=%016llx "
                                     "kv=%016llx mtp=%016llx stale=%016llx dead=%016llx pooled_full=%016llx ple_prev=%d,%d\n", (long long) L,
                             (unsigned long long) h_gdn, (unsigned long long) h_ple, (unsigned long long) h_tail,
                             (unsigned long long) h_pool, (unsigned long long) h_kv, (unsigned long long) h_mtp,
                             (unsigned long long) h_stale, (unsigned long long) h_dead,
                             (unsigned long long) h_pool_full, ss.ple_prev[0], ss.ple_prev[1]);
            }
            const int64_t req_hits = drive.d.cache_hits - decode_hits0;
            const int64_t req_look = (drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused) - decode_look0;
            const int64_t req_offload = drive.d.offload_entries - offload0;
            // #471: the prompt tokens this request read - all the fresh ones, or as far as the prompt pass got when a
            // cancel stopped it part-way (a cancelled request used to be logged and counted as having read them all)
            const int64_t fresh = n - resume;
            const int64_t read_n = cancelled ? std::clamp<int64_t>(pp_reached - resume, 0, fresh) : fresh;
            // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused> [hits] [lookups]
            //      [RAM blobs] [file blobs] [file MB]   (CS-T tiers; appended, so an older server reads the rest)
            //      [prompt tokens read]   (#471: fewer than <prompt> - <reused> when a cancel stopped the read)
            //      [offloaded]   (#588: the decode's routed experts the GPU read over PCIe or another GPU computed;
            //                    not in [lookups])
            //      [lookup-chain accepted] [lookup-chain offered] [suffix accepted] [suffix offered] [windows]
            //      (only with --lookup-chain)
            char chain_txt[128] = "";
            if (o.lookup_chain > 0)
                std::snprintf(chain_txt, sizeof(chain_txt), " %lld %lld %lld %lld %lld", (long long) chain_ok,
                              (long long) chain_drafts, (long long) sfx_ok, (long long) sfx_drafts, (long long) dec_windows);
            if (yielded_at >= 0) std::printf("YIELDED %d %lld\n", yielded_slot, (long long) yielded_at);
            std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld%s\n",
                        (long long) produced_n,
                        (long long) n, prompt_ms, decode_ms, finish, (long long) draft_accepted, (long long) draft_offered,
                        (long long) resume, (long long) req_hits, (long long) req_look,
                        (long long) (src.ram_reads() - ram0), (long long) (src.file_reads() - files0),
                        (double) (src.file_read_bytes() - file_bytes0) / 1e6, (long long) read_n,
                        (long long) req_offload, chain_txt);
            std::fflush(stdout);
            if (admit_slot >= 0) {   // --batch: BADM <slot> <1 = continues in the batch windows | 0 = done>
                bool cont = !cancelled && produced_n == 1 && admit_max_new > 1 && std::strcmp(finish, "length") == 0 &&
                            (int64_t) live.size() == p;
                const auto tc0 = Clock::now();
                if (cont && !copy_to_slot(admit_slot, live, err)) {
                    std::fprintf(stderr, "strata serve: batch admission failed: %s\n", err.c_str());
                    bs[(size_t) admit_slot].cached = false;   // its sessions may be half written
                    cont = false;
                }
                if (cont) {
                    ver.set_slot_sampling(admit_slot, req_sp);   // the request's own sampling, row by row
                    BSlot& sl = bs[(size_t) admit_slot];
                    sl = BSlot{};
                    sl.active = true;
                    sl.x = x;
                    sl.p = p;
                    sl.produced = 1;
                    sl.max_new = admit_max_new;
                    sl.t0 = Clock::now();
                    sl.ids = live;
                    if (batch_mtp) {
                        if (!slot_mtp[(size_t) admit_slot]->draft_first(1, ver.final_R_all(), sl.x,
                                                                        sl.p - 1, sl.draft.data(), err)) {
                            std::fprintf(stderr, "strata batch: MTP admission for slot %d failed: %s\n",
                                         admit_slot, err.c_str());
                            return 1;
                        }
                        sl.draft_ready = true;
                    }
                    sl.cvec = cvec_cached;
                    sl.img = !live_imgs.empty();   // pictures: not matched again by tokens alone, so not cached
                    const ConvCheckpoint* best = nullptr;
                    for (const ConvCheckpoint& c : checks)
                        if (c.ids.size() < live.size() && (best == nullptr || c.ids.size() > best->ids.size()) &&
                            std::equal(c.ids.begin(), c.ids.end(), live.begin()))
                            best = &c;
                    if (best != nullptr && !sl.img) sl.checks.push_back(*best);
                    std::fprintf(stderr, "strata batch: slot %d takes %lld tokens (copied in %.1f ms)\n", admit_slot,
                                 (long long) live.size(), std::chrono::duration<double, std::milli>(Clock::now() - tc0).count());
                }
                std::printf("BADM %d %d\n", admit_slot, cont ? 1 : 0);
                std::fflush(stdout);
                admit_slot = -1;
            }
            if (drive.routing != nullptr) std::fflush(drive.routing);   // the routing trace survives a crash and is watchable mid-session
            // "12288 of 98179" when cancelled mid-read (#471), the rate from what was read
            char read_txt[64];
            if (cancelled)
                std::snprintf(read_txt, sizeof(read_txt), "%lld of %lld", (long long) read_n, (long long) fresh);
            else
                std::snprintf(read_txt, sizeof(read_txt), "%lld", (long long) fresh);
            std::fprintf(stderr, "strata serve: prompt %lld tokens = %lld reused + %s read in %.0f ms (%.1f tok/s), "
                                 "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld, %zu checkpoints%s\n",
                         (long long) n, (long long) resume, read_txt, prompt_ms,
                         prompt_ms > 0 ? 1000.0 * read_n / prompt_ms : 0.0, (long long) produced_n, decode_ms,
                         decode_ms > 0 ? 1000.0 * produced_n / decode_ms : 0.0, (long long) draft_accepted,
                         (long long) draft_offered, checks.size(), cancelled ? " (cancelled)" : "");
            // the VRAM share of the experts the pool looked up while decoding; experts it sent over PCIe for the GPU
            // to read (--pcie-frac) are in neither count - #588: so that share is said beside it (raising --pcie-frac
            // raises the hit rate while the PCIe reads may make the decode slower)
            if (req_look > 0) {
                char off[160] = "";
                if (req_offload > 0)
                    std::snprintf(off, sizeof off, "; %lld more read by the GPU over PCIe or from another GPU (%.1f%% of "
                                  "all %lld routed)", (long long) req_offload,
                                  100.0 * (double) req_offload / (double) (req_look + req_offload),
                                  (long long) (req_look + req_offload));
                std::fprintf(stderr, "strata serve: decode expert cache hit rate: %.1f%% (%lld hits / %lld lookups)%s\n",
                             100.0 * (double) req_hits / (double) req_look,
                             (long long) req_hits, (long long) req_look, off);
            }
            // the resident RAM mode, cumulative: experts read from experts.bin since the copy was made (what the plain
            // mmap mode reads through the OS file cache, from the SSD when the RAM could not keep it)
            if (src.complement_ready())
                std::fprintf(stderr, "strata serve: resident RAM: %.2f GiB of experts in RAM, %lld exchanged with the "
                                     "VRAM tier, %lld blob reads from the file\n",
                             (double) src.resident_bytes() / 1073741824.0, (long long) src.exchanges(),
                             (long long) src.file_reads());
            if (ajob)   // --adapt-async, cumulative (a round may still be in flight: it lands before the next request)
                std::fprintf(stderr, "strata serve: asynchronous adaptive tier: %lld rounds, %lld experts swapped in, "
                                     "%.1f ms per round (start to flip, between windows)\n",
                             (long long) a_rounds, (long long) a_swapped, a_rounds > 0 ? a_ms / (double) a_rounds : 0.0);
            if (src.exchange_rotation())
                std::fprintf(stderr, "strata serve: exchange rotation: %llu blocks, %llu host memcpy bytes avoided (cumulative payload)\n",
                             (unsigned long long)src.rotated_exchanges(),
                             (unsigned long long)src.avoided_exchange_copy_bytes());
            // CS-T: the tiers, cumulative - GPU cache hits (the decode lookups above), RAM copy, files (SSD / OS cache)
            if (srcp == &src)
                std::fprintf(stderr, "strata serve: expert tiers: GPU %lld hits this request; since the start RAM %lld blobs, files %lld blobs "
                                     "%.1f MB read%s\n", (long long) req_hits, (long long) src.ram_reads(),
                             (long long) src.file_reads(), (double) src.file_read_bytes() / 1e6,
                             src.gguf_mode() ? " (the GGUF in place)" : "");
            // STRATA_SPLIT_TIMING: where each verify stage's host time went, cumulative per window since the start
            // (waiting for its GPU to ring a layer, the CPU pool and plan per layer, staging the window)
            if (static const bool st_timing = std::getenv("STRATA_SPLIT_TIMING") != nullptr; st_timing)
                for (int st = 0; st < n_stages; ++st) {
                    const strata::core::Verifier& v = stage_ver(st);
                    const double w = v.windows > 0 ? (double) v.windows : 1.0;
                    std::fprintf(stderr, "strata serve: stage %d: %lld windows; per window: wait for the GPU %.3f ms, "
                                         "pool + plan %.3f ms, host staging %.3f ms, commit %.3f ms\n", st,
                                 (long long) v.windows, v.ms_wait / w, v.ms_pool / w, v.ms_host / w, v.ms_commit / w);
                }
            if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {
                // KV streaming, cumulative over the process: blocks the selections named vs blocks read from RAM
                // (this device's owned ordinals; a split's other stages hold theirs)
                uint64_t miss = 0, look = 0;
                bool over = false;
                for (int64_t j = 0; j < ss.qsa_alloc; ++j) {
                    const strata::kernels::KvStreamCounters c =
                        strata::kernels::kv_stream_counters(ss.qsa_states[ss.qsa_ord0 + j].map);
                    miss += c.misses; look += c.lookups; over = over || c.overflow;
                }
                std::fprintf(stderr, "strata serve: KV streaming: %.2f%% of %llu block reads hit VRAM, %.1f MiB read "
                                     "from RAM%s\n", look ? 100.0 * (double) (look - miss) / (double) look : 100.0,
                             (unsigned long long) look, (double) miss * 4224.0 / 1048576.0,
                             over ? " - OVERFLOW (too few resident cells)" : "");
            }
            if (sfx_windows > 0)
                std::fprintf(stderr, "strata serve: suffix drafts: %lld windows, %lld of %lld drafts accepted\n",
                             (long long) sfx_windows, (long long) sfx_ok, (long long) sfx_drafts);
            if (o.lookup_chain > 0)
                std::fprintf(stderr, "strata serve: lookup chain: %lld of %lld windows, %lld of %lld chained drafts accepted\n",
                             (long long) chain_windows, (long long) dec_windows, (long long) chain_ok, (long long) chain_drafts);
            if (strata::core::MtpDrafter::top2_env())
                std::fprintf(stderr, "strata serve: mtp top2: rejected/runner-up by depth %lld/%lld %lld/%lld %lld/%lld %lld/%lld "
                                     "of %lld windows\n", (long long) t2_rej[0], (long long) t2_hit[0], (long long) t2_rej[1],
                             (long long) t2_hit[1], (long long) t2_rej[2], (long long) t2_hit[2], (long long) t2_rej[3],
                             (long long) t2_hit[3], (long long) dec_windows);
            for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
                std::fprintf(stderr, "strata serve: CUDA%d: %lld expert entries, %lld active layer launches, %.1f MiB returned "
                                     "(%.1f MiB with full rows) in this request; host %.0f ms staging+launching, %.0f ms "
                                     "waiting for it in this request\n", r + 1,
                             (long long) (remote_experts[(size_t) r].computed() - remote_before[(size_t) r]),
                             (long long) (remote_experts[(size_t) r].launched_layers() - launches_before[(size_t) r]),
                             (double) (remote_experts[(size_t) r].returned_bytes() - compact_before[(size_t) r]) / 1048576.0,
                             (double) (remote_experts[(size_t) r].full_row_bytes() - full_before[(size_t) r]) / 1048576.0,
                             remote_experts[(size_t) r].ms_begin() - begin_before[(size_t) r],
                             remote_experts[(size_t) r].ms_wait() - wait_before[(size_t) r]);
        }
        save_profile("exit");   // #477: QUIT, or the server closed stdin
        // SYCL port: the requests are done and their output written; leave without unwinding the GPU objects (the OS
        // reclaims them). Their destructors ran against a runtime already shutting down and aborted (exit 139).
        try {
            dpct::get_current_device().queues_wait_and_throw();
            // Release the direct-access USM host mirror before process teardown.
            gguf_src.close();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata shutdown: mirror release failed: %s\n", e.what());
            std::fflush(stderr);
            std::_Exit(1);
        }
        std::fflush(stdout);
        std::fflush(stderr);
        std::_Exit(0);
    }

    // ---- plan v0.3 P5: the prompt's conditioning positions [0, n_prompt - 1) in batched chunks.  The token loop
    // then starts at the last prompt position, whose prediction is the first generated token.
    int64_t pos_start = 0;
    int64_t spec_pos = 0;   // plan v0.3 P6: where the speculative loop starts (0 = not used)
    strata::prefill::Prefill prefill;
    bool kvg_started = false;   // the elastic K/V took this run's cells
    double prefill_batched_ms = 0;
    std::FILE* final_r = o.dump_final_r.empty() ? nullptr : std::fopen(o.dump_final_r.c_str(), "wb");
    std::vector<float> final_r_host(final_r ? (size_t) (g.hc * g.n_embd) : 0);
    std::vector<std::pair<int32_t, int32_t>> lent;     // (residency index, slot) lent to the prompt path
    const int64_t n_batched = (o.prefill_until > 0 && o.prefill_until < n_prompt - 1) ? o.prefill_until : n_prompt - 1;
    if (o.prefill_chunk > 0 && n_prompt > 1) {
        void* borrow = nullptr;
        uint64_t borrow_bytes = 0;
        if (!o.no_prefill_borrow && !host_res.empty() && d_res != nullptr) {
            int64_t chunk = o.prefill_chunk;
            int64_t k = plan_lend(chunk);             // auto: the largest chunk that fits; fixed: halved to fit
            const int64_t request_sized = equal_chunk(n_batched, chunk);
            if (k > 0 && request_sized < chunk) {                     // no bigger than this prompt segment needs
                chunk = request_sized;
                k = lend_slots(chunk);
                if (k + 128 > xcache.slots()) k = 0;
                if (!o.prefill_auto) o.prefill_chunk = chunk;
            }
            if (o.prefill_auto) {
                o.prefill_chunk = k > 0 ? chunk : request_chunk(n_batched, 1024);
                std::fprintf(stderr, "strata generate: prompt chunk auto: %lld tokens\n", (long long) o.prefill_chunk);
            } else if (chunk != o.prefill_chunk) {
                k = 0;                                 // a fixed chunk that does not fit: its own buffers, as before
            }
            const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            if (k > 0) {   // the lent slots are refilled after the prompt
                const int32_t first = (int32_t) (xcache.slots() - k);
                // the elastic K/V first: its hot experts move into slots of the loan, which are refilled after it
                kvg_started = true;
                kvg_start(first);
                if (!kvg_ensure(n_prompt + o.max_new + 64, [] {
                                                                dpct::get_current_device().queues_wait_and_throw(); })) return 1;
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= first) {
                        lent.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = strata::core::kNotResident;
                    }
                res_put(d_res);
                borrow = xcache.device_slot(first);
                borrow_bytes = xcache.slot_offsets() ? (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[first])
                                                     : (uint64_t) k * (uint64_t) blob;
                std::fprintf(stderr, "strata generate: prompt path borrows %lld cache slots (%.2f GiB)\n", (long long) k,
                             (double) borrow_bytes / 1073741824.0);
            }
        }
        if (borrow == nullptr)
            std::fprintf(stderr, "strata generate: prompt path allocates its own buffers (no cache slots to borrow)\n");
        if (!prefill.init(wt, g, ss, srcp, o.expert_cache > 0 ? &xcache : nullptr,
                          host_res.empty() ? nullptr : host_res.data(), o.prefill_chunk, main_cs, err, borrow,
                          borrow_bytes)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (!o.mtp.empty()) {
            if (!mtp.bind(wt, &native_head, nullptr, err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            prefill.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
                // cell i pairs R_i with the token at i + 1 (every such token is in the prompt)
                std::vector<int32_t> nxt((size_t) T);
                for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) o.tokens[(size_t) (p0 + t + 1)];
                if (prefill.draft_kv(mtp, R_rows, nxt.data(), T, p0, e)) return true;   // E-9
                return e.empty() && mtp.prefill(R_rows, nxt.data(), T, p0, e);
            };
        }
        if (!kvg_started) {   // the elastic K/V: every cell this run can reach (no loan)
            kvg_started = true;
            kvg_start(xcache.slots());
            if (!kvg_ensure(n_prompt + o.max_new + 64, [] {
                                                            dpct::get_current_device().queues_wait_and_throw(); })) return 1;
        }
        const Clock::time_point tp0 = Clock::now();
        if (!prefill.run(o.tokens.data(), n_batched, 0, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        // refill the lent slots from the arena and give them back to the decode tier
        if (!lent.empty()) {
            const Clock::time_point tr = Clock::now();
            size_t queued = 0;
            for (const auto& [i, slot] : lent) {   // D-4: queued, one wait (STRATA_REFILL_BLOCKING=1: each)
                const uint8_t* b = srcp->blob(i / g.n_expert, i % g.n_expert);
                const int64_t nb = (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
                if (b == nullptr || !(refill_blocking() ? xcache.fill_slot_blocking(slot, b, err, nb)
                                                        : xcache.fill_slot_queued(slot, b, err, nb))) {
                    std::fprintf(stderr, "strata generate: refilling a lent slot failed: %s\n", err.c_str());
                    return 1;
                }
                host_res[(size_t) i] = slot;
                // SYCL port: GgufExpertSource::blob() hands out a ring of 512 buffers, valid until the next blob()
                // call; a queued copy reads its buffer later, so drain before the ring can wrap onto one (938 lent
                // slots refilled through 512 buffers gave ~426 slots another expert's bytes)
                if (!refill_blocking() && ++queued % 256 == 0 && !xcache.sync_queued(err)) {
                    std::fprintf(stderr, "strata generate: refilling the lent slots failed: %s\n", err.c_str());
                    return 1;
                }
            }
            if (!xcache.sync_queued(err)) {
                std::fprintf(stderr, "strata generate: refilling the lent slots failed: %s\n", err.c_str());
                return 1;
            }
#if defined(_WIN32)
            for (const auto& [i, slot] : lent)
                srcp->release(i / g.n_expert, i % g.n_expert);
#endif
            res_put(d_res);
            std::fprintf(stderr, "strata generate: %zu lent slots refilled in %.1f ms\n", lent.size(),
                         std::chrono::duration<double, std::milli>(Clock::now() - tr).count());
        }
        prefill_batched_ms = std::chrono::duration<double, std::milli>(Clock::now() - tp0).count();
        prefill_ms += prefill_batched_ms;
        pos_start = n_batched;
        tok = o.tokens[(size_t) pos_start];
        // the PLE window of the token path: the two tokens before `pos_start`
        ss.ple_prev[0] = pos_start >= 2 ? (int32_t) o.tokens[(size_t) (pos_start - 2)] : -1;
        ss.ple_prev[1] = pos_start >= 1 ? (int32_t) o.tokens[(size_t) (pos_start - 1)] : -1;
        const strata::prefill::PrefillStats& ps = prefill.stats();
        std::fprintf(stderr, "strata generate: prefill %lld tokens in %lld chunks, %.1f ms (%.1f tok/s); experts "
                             "streamed %lld (%lld by DMA, host %.1f ms), resident %lld; PLE %.1f ms\n",
                     (long long) ps.tokens, (long long) ps.chunks, ps.ms_total,
                     ps.ms_total > 0 ? 1000.0 * (double) ps.tokens / ps.ms_total : 0.0, (long long) ps.experts_streamed,
                     (long long) ps.experts_dma, ps.ms_experts_host, (long long) ps.experts_resident, ps.ms_ple);
    }

    if (!kvg_started) {   // the elastic K/V of a prompt that was not read in batches
        kvg_started = true;
        kvg_start(xcache.slots());
        if (!kvg_ensure(n_prompt + o.max_new + 64, [] {
                                                        dpct::get_current_device().queues_wait_and_throw(); })) return 1;
    }
    for (int64_t pos = pos_start;; ++pos) {
        // plan v0.3 P6: a native pack's last prompt token is the first verify window (T = 1)
        if (native_pack) { spec_pos = pos; break; }
        if (pos >= o.max_context) {
            std::fprintf(stderr, "strata generate: ran out of context at position %lld\n", (long long) pos);
            return 2;
        }
        // **THE TOKEN TIMER STARTS HERE, BEFORE ANY OF THE TOKEN'S WORK (A7).**  It used to start after
        // `put_input`/`embed_row`, which excluded the embedding and the PLE window advance from the reported
        // rate, and it stopped before the NaN scan, the logits dump and the sampler.  The published tok/s
        // figure is a WALL-CLOCK rate: everything one token costs, PLE advance to sampled id.  A rate that
        // excludes real per-token work is not a rate anyone can plan against.
        const Clock::time_point t0 = Clock::now();
        // **THE PLE'S TOKEN WINDOW ADVANCES HERE, ONCE PER TOKEN, AND `ple_stage_token` RUNS OUTSIDE THE
        // CAPTURE.**  Both are the driver's job: `ngram_rows` is a host hash over the last three tokens and the
        // table gather is a host read, so either one inside a captured graph would run once at capture time and
        // replay forever.  `ple_prev` is OLDEST FIRST and `-1` means "no predecessor", which `ngram_rows`
        // treats as the EOS cut - a sequence boundary.
        ss.ple_token = (int32_t) tok;
        Clock::time_point tp = Clock::now();
        // Plan v0.3 P2: the 16 SSD reads start here and complete while the embedding is staged; `ms_ple` is
        // the issue plus the time still spent WAITING afterwards, i.e. the part the embedding did not hide.
        if (ss.ple.ready() && !strata::core::ple_issue_token(ss.ple, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        {
            const Clock::time_point n = Clock::now();
            ms_ple += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        if (!put_input(tok, pos)) return 1;
        {
            const Clock::time_point n = Clock::now();
            ms_embed += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        if (ss.ple.ready() && !strata::core::ple_finish_token(ss.ple, token_stream, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        {
            const Clock::time_point n = Clock::now();
            ms_ple += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }

        if (pos % 256 == 0 || pos + 1 >= n_prompt - 1)
            std::fprintf(stderr, "strata generate: position %lld, token %lld%s\n", (long long) pos, (long long) tok,
                         pos < n_prompt ? " (prompt)" : "");
        // **`d.layers` IS THE BLOB'S LAYER AXIS, NOT A COUNTER.**  The adapter uses it to index
        // `experts.bin` as `layer * n_expert + expert`, so it MUST restart at 0 for every token.  Leaving it
        // running across tokens asks for layer 48 of a 48-layer file on the second token - which
        // `FileExpertSource` REFUSES rather than wrapping into layer 0's experts, and that refusal is the only
        // reason this was a clean error instead of a silently wrong second token.
        drive.d.layers = 0;
        drive.d.experts = 0;
        drive.d.failed = false;
        err.clear();
        if (o.no_capture) {
            if (!strata::core::session_token(wt, g, pos, /*pos_base=*/0, ss, d_parts, main_cs,
                                             o.sync_every_layer, err)) {
                std::fprintf(stderr, "strata generate: session_token: %s\n", err.c_str());
                return 1;
            }
        } else {
            strata::core::doorbell_reset(db);
            if (tgraph.captured) {
                if (!strata::core::session_run_token(g, pos, /*pos_base=*/0, ss, tgraph, pool_fn, pool_user,
                                                     loop_scratch.y_miss, main_cs, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
            } else if (!strata::core::session_loop(g, pos, /*pos_base=*/0, ss, gr, pool_fn, hit_fn, pool_user, /*overlap=*/true, main_cs,
                                        err, layer_stage, &loop_scratch)) {
                std::fprintf(stderr, "strata generate: session_loop: %s\n", err.c_str());
                return 1;
            }
        }
        if (final_r != nullptr) {
            dpct::get_in_order_queue()
                .memcpy(final_r_host.data(), ss.R,
                        final_r_host.size() * sizeof(float))
                .wait();
            const int64_t posrec[2] = {pos, tok};
            std::fwrite(posrec, sizeof posrec, 1, final_r);
            std::fwrite(final_r_host.data(), sizeof(float), final_r_host.size(), final_r);
        }
        if (drive.d.failed) {
            std::fprintf(stderr, "strata generate: the expert pool failed at layer %lld expert %lld: %s\n",
                         (long long) drive.d.fail_layer, (long long) drive.d.fail_expert,
                         drive.d.fail ? drive.d.fail : "(no message)");
            return 1;
        }
        {
            // **THE LAYER LOOP ITSELF, WHICH IS WHAT `--gpu-only-full` HAS TO BE COMPARED AGAINST.**
            const Clock::time_point n = Clock::now();
            ms_layers += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        // ---- the ladder for THIS position, in the order `session_loop` filled it: layer 0 first.
        if (layer_dump != nullptr) std::fwrite(layer_stage, sizeof(float), layer_floats, layer_dump);
        if (half_dump != nullptr) {
            std::fwrite(half_stage, sizeof(float), (size_t) g.n_layers * (size_t) half_stride, half_dump);
        }
        if (!run_head(token_stream)) {
            std::fprintf(stderr, "strata generate: lm_head: %s\n", err.c_str());
            return 1;
        }
        // **A CHECKPOINT AFTER THE HEAD, BECAUSE AN ASYNC FAULT IS STICKY AND LIES ABOUT WHERE IT HAPPENED.**
        // Measured, and it cost an hour: without this, `embed_row`'s D2H on the NEXT token reported "an illegal
        // memory access" at a plane offset that has nothing to do with the fault, and the layer that actually
        // faulted had completed its own error checks successfully - because its kernels had not run yet.  A
        // sticky error surfaces at the next SYNCHRONISING call, which is whatever happens to come next.
        if (!o.stream_token &&
            DPCT_CHECK_ERROR(
                dpct::get_current_device().queues_wait_and_throw()) != 0) {
            std::fprintf(
                stderr,
                "strata generate: the device faulted in lm_head at position "
                "%lld: %s\n",
                /*
                DPCT1009: SYCL reports errors using exceptions and does not
                use error codes. Please replace the
                "get_error_string_dummy(...)" with a real error-handling
                function.
                */
                /*
                DPCT1010: SYCL uses exceptions to report errors and does
                not use the error codes. The cudaGetLastError function call was
                replaced with 0. You need to rewrite this code.
                */
                (long long)pos, dpct::get_error_string_dummy(0));
            return 1;
        }
        {
            const Clock::time_point n = Clock::now();
            ms_head += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        const bool emit_logits = dump != nullptr &&
            strata::program::logits_selection::selected(pos, dump_positions, o.logits_stride);
        const bool read_logits = !o.stream_token || o.check_logits || emit_logits;
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        if (read_logits &&
            (DPCT_CHECK_ERROR(strata::q_of(token_stream)->memcpy(
                 logits.data(), d_logits, (size_t)n_vocab * 4)) != 0 ||
             DPCT_CHECK_ERROR(strata::q_of(token_stream)->wait()) != 0)) {
            std::fprintf(stderr, "strata generate: reading the logits back failed\n");
            return 1;
        }
        int bad = 0;
        if (read_logits) for (float v : logits) if (!std::isfinite(v)) ++bad;
        if (bad != 0) {
            std::fprintf(stderr, "strata generate: %d of %lld logits are not finite at position %lld\n", bad,
                         (long long) n_vocab, (long long) pos);
            return 1;
        }
        // the header with the first row (see `hdr`): a run that dumps nothing leaves an empty file
        if (emit_logits && !hdr_written && std::fwrite(hdr, sizeof hdr, 1, dump) != 1) {
            std::fprintf(stderr, "strata generate: cannot write logits header\n");
            std::fclose(dump);
            return 1;
        }
        if (emit_logits) hdr_written = true;
        if (emit_logits && std::fwrite(logits.data(), sizeof(float), (size_t) n_vocab, dump) != (size_t) n_vocab) {
            std::fprintf(stderr, "strata generate: cannot write logits at position %lld\n", (long long) pos);
            std::fclose(dump);
            return 1;
        }
        {
            // **993 KB OF SYNCHRONOUS D2H AND A 248,320-FLOAT HOST SCAN, EVERY TOKEN.**  (The review's notes
            // say 151,936 floats; the artifact's `output.weight` is 248,320 rows, so the real figure is 1.6x
            // that - a number nobody had checked because nothing measured this term.)  R2.6 asks for the dump
            // and the scan to be behind flags; round 36 did exactly that and measured it SLOWER, because on
            // this driver a large blocking readback is also what flushes the pipeline for the sampler that
            // follows.  Timed so the claim can be re-checked rather than remembered.
            const Clock::time_point n = Clock::now();
            ms_readback += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }

        int next = 0;
        // The draw is Philox(seed, position), as in a verify window (row t at pos0 draws pos0 + t): a seed gives
        // the same text whether a token comes from this path or from the speculative loop below.
        sp.counter = (uint64_t) pos;
        strata::kernels::sample_tokens(d_logits, 1, (int) n_vocab, nullptr, 0, sp, d_next, token_stream);
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        if (DPCT_CHECK_ERROR(strata::q_of(token_stream)->memcpy(
                &next, d_next, sizeof(int))) != 0 ||
            DPCT_CHECK_ERROR(strata::q_of(token_stream)->wait()) != 0) {
            std::fprintf(
                stderr,
                "strata generate: reading the sampled token back failed: %s\n",
                /*
                DPCT1009: SYCL reports errors using exceptions and does not
                use error codes. Please replace the
                "get_error_string_dummy(...)" with a real error-handling
                function.
                */
                /*
                DPCT1010: SYCL uses exceptions to report errors and does
                not use the error codes. The cudaGetLastError function call was
                replaced with 0. You need to rewrite this code.
                */
                dpct::get_error_string_dummy(0));
            return 1;
        }
        // The sampled-token synchronization also completes every captured QSA
        // status readback. Retain one status per layer so a later layer cannot
        // hide an earlier failure; no extra synchronization or token allocation.
        if (o.native_flash_attn_short) for (int64_t j = 0; j < ss.qsa_alloc; ++j) {
            const int64_t i = ss.qsa_ord0 + j;   // the global ordinal, as the session's carve names it
            const int32_t status = ss.qsa_states[i].host_step[strata::kernels::kStepCount];
            if (status != 0) {
                std::fprintf(stderr, "strata generate: native attention status %d at QSA layer %lld, position %lld\n",
                             status, (long long) i, (long long) pos);
                return 1;
            }
        }
        if (next < 0 || next >= n_vocab) {
            std::fprintf(stderr, "strata generate: the sampler returned %d, outside 0..%lld\n", next,
                         (long long) (n_vocab - 1));
            return 1;
        }
        {
            // **TWO DEVICE-WIDE SYNCS FOR FOUR BYTES.**  `sample_tokens(nullptr)` ends in
            // `cudaDeviceSynchronize()` (`sampler.cu:245`) and the blocking 4-byte read below is the second.
            const Clock::time_point n = Clock::now();
            ms_sample += std::chrono::duration<double, std::milli>(n - tp).count();
            ++phase_tokens;
        }
        // CHARGED HERE, AFTER THE SAMPLER, so the wall-clock rate covers the whole token including the embedding,
        // the NaN scan, the logits readback and the sample (A7).  Only DECODE positions count; prefill is
        // measured separately.
        if (pos >= n_prompt - 1) total_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        else prefill_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (pos == n_prompt - 1) ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
        // **THE PREDICTION AT THE LAST PROMPT POSITION *IS* THE FIRST GENERATED TOKEN.**  Sampling on every
        // position and recording only from `n_prompt - 1` onward is what keeps the two cases from needing
        // separate handling - and the version that "obviously" only samples after the prompt loses exactly one
        // token's worth of conditioning.
        if (pos >= n_prompt - 1) produced.push_back(next);
        if ((int64_t) produced.size() >= o.max_new) break;
        if (o.stop_eos && pos >= n_prompt - 1 &&
            std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) next) != o.eos_ids.end()) break;
        // TEACHER FORCING while the prompt lasts: the next input is the prompt's own next token, not the
        // model's guess.  Feeding the guess would make the run depend on the model's own errors from position
        // 1, which is a different (and worse) measurement of the same prompt.
        // and the window advances: the token just decoded becomes the newest predecessor.
        ss.ple_prev[0] = ss.ple_prev[1];
        ss.ple_prev[1] = (int32_t) tok;
        tok = (pos + 1 < n_prompt) ? o.tokens[(size_t) (pos + 1)] : next;
        // Plan v0.3 P6: from the first generated token on, the speculative loop below takes over.
        if (o.spec > 0 && pos >= n_prompt - 1) { spec_pos = pos + 1; break; }
    }

    // ================================ plan v0.3 P6: SPECULATIVE DECODING ================================
    //
    // Each round verifies [the last emitted token, drafts...] in one window; the window's argmax after token t
    // is exactly what greedy decode would emit there, so the first draft that differs ends the round and the
    // round emits (accepted drafts + 1) tokens.  `commit` keeps the state of the tokens that were emitted.
    const bool ended = o.stop_eos && !produced.empty() &&
                       std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) produced.back()) != o.eos_ids.end();
    if (spec_pos > 0 && (int64_t) produced.size() < o.max_new && !ended) {
        auto read_ids = [](const std::string& path, std::vector<int64_t>& ids) {
            std::ifstream in(path);
            std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            std::string e;
            return (bool) in && parse_i64_list(text.c_str(), ids, e);
        };
        std::vector<int64_t> oracle, follow;
        if (!o.spec_oracle.empty() && !read_ids(o.spec_oracle, oracle)) {
            std::fprintf(stderr, "strata generate: cannot read --spec-oracle %s\n", o.spec_oracle.c_str());
            return 2;
        }
        if (!o.spec_follow.empty() && !read_ids(o.spec_follow, follow)) {
            std::fprintf(stderr, "strata generate: cannot read --spec-follow %s\n", o.spec_follow.c_str());
            return 2;
        }
        // --spec-follow: the run ends with the continuation
        const int64_t max_new = follow.empty() ? o.max_new : std::min<int64_t>(o.max_new, (int64_t) follow.size());
        int64_t follow_differ = 0, follow_emitted = 0;
        if (thits.d_res == nullptr) {
            std::fprintf(stderr, "strata generate: --spec needs the device residency table (--expert-profile, "
                                 "--expert-cache and the token graph)\n");
            return 2;
        }
        mem_mark("the head and the prompt path");
        strata::core::Verifier ver;
        strata::core::VerifyHits vh;
        vh.d_res = thits.d_res;
        int32_t* usage_dev = nullptr;   // STRATA_USAGE_DUMP=PATH: the device plan counts every routed expert (measurement)
        const char* usage_dump = std::getenv("STRATA_USAGE_DUMP");
        if (mirror_table_d && usage_dump && *usage_dump) {
            usage_dev = sycl::malloc_device<int32_t>((size_t) (g.n_layers * g.n_expert), dpct::get_in_order_queue());
            dpct::get_in_order_queue().memset(usage_dev, 0, (size_t) (g.n_layers * g.n_expert) * 4).wait();
        }
        if (mirror_table_d) strata::kernels::resident_plan_set_mirror(thits.d_res, mirror_table_d, usage_dev, (long long) (g.n_layers * g.n_expert));
        vh.h_res = host_res.empty() ? nullptr : host_res.data();
        vh.cache_base = thits.cache_base;
        vh.blob = thits.blob;
        vh.slot_off = xcache.slot_offsets();   // E-6: the device plan's pointers
        vh.n_slots = xcache.slots();
        if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.spec, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        const bool use_mtp = !o.mtp.empty();
        if (use_mtp && !mtp.bind(wt, &native_head, ver.final_R_all(), err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        mem_mark("the verifier and the drafter's binding");
        ver.set_sampling(sp);   // the CLI's own sampling (until 0.1.19 this loop was always greedy); no penalties here
        if (use_mtp) mtp.set_draft_sampling(sp);   // STRATA_SPEC_COUPLED=1: sampled drafts (a no-op otherwise)
        ver.set_split(o.spec_split);
        // auto: the copy kernel for every pack.  DMA (the native packs' default until 0.1.13) has the host call
        // cudaMemcpyAsync + cudaLaunchHostFunc inside a verify window while the GPU spins on the flag they raise;
        // issue #31's thread dumps show the host stuck in that cudaMemcpyAsync on a driver lock for good.  The copy
        // kernel needs no host CUDA call there, and costs ~1-3% decode on IQ3_S (45.3 -> 44.8 tok/s, 8 requests).
        ver.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
        drive.d.plan = ver.plan_sink();
        drive.d.pcie_num = (int) (o.pcie_frac * 256.0 + 0.5);
        if (drive.d.pcie_num < 0) drive.d.pcie_num = 0;
        if (drive.d.pcie_num > 256) drive.d.pcie_num = 256;
        const int64_t pcie0 = drive.d.pcie_experts;
        // CS-T: the file tier since the decode began (the prompt path's copies are before this)
        const int64_t files0 = src.file_reads(), ram0 = src.ram_reads();
        const uint64_t fbytes0 = src.file_blob_bytes(), fall0 = src.file_read_bytes();
        const double fms0 = src.file_ms();
        if (o.adapt_every > 0 && o.adapt_swaps > 0) drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
        int64_t swaps_total = 0;
        double ms_adapt = 0;
        dpct::queue_ptr adapt_stream = &dpct::get_in_order_queue();
        /*
        DPCT1025: The SYCL queue is created ignoring the flag and priority
        options.
        */
        if (!drive.d.usage.empty() &&
            DPCT_CHECK_ERROR(
                adapt_stream = dpct::get_current_device().create_queue(true)) !=
                0) {
            std::fprintf(stderr, "strata generate: cannot create the refill stream\n");
            return 1;
        }
        // plan v0.3 P6: swaps in flight - (residency index, slot) admitted when adapt_ev has completed
        std::vector<std::pair<int32_t, int32_t>> pending;
        int pending_age = 0;   // the windows the pending swaps have waited (adapt_lag)
        std::vector<void*> pin_live;   // the swaps' locked arena pages (pin_blob), unlocked once they have landed
        dpct::event_ptr adapt_ev = nullptr;
        adapt_ev = new sycl::event();
        int64_t adapt_rounds = 0;   // counted here: `rounds` is declared below the adapt lambda
        auto apply_pending = [&](bool wait) {
            try {
        if (pending.empty()) return;
            // STRATA_TRACE_ADAPT=1: whether a round's copies had landed when the next window read the table
            static const bool trace_pending = std::getenv("STRATA_TRACE_ADAPT") != nullptr;
            if (wait) adapt_ev->wait_and_throw();
            else if (adapt_ev->get_info<
                         sycl::info::event::command_execution_status>() !=
                     sycl::info::event_command_status::complete) {
                if (trace_pending)
                    std::fprintf(stderr, "strata: PENDING not landed, %zu stay non-resident this window\n",
                                 pending.size());
                return;
            }
            if (trace_pending)
                std::fprintf(stderr, "strata: PENDING landed, %zu experts become resident\n", pending.size());
            unpin_blobs(pin_live);
            src.commit_exchanges();   // the resident RAM mode: the evicted experts take their places in RAM
            for (const auto& [i, slot] : pending) {
                host_res[(size_t) i] = slot;
                srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);   // in VRAM now: RAM not needed
            }
            pending.clear();
            pending_age = 0;
            if (d_res != nullptr)
                res_put(d_res);
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        // Plan v0.3 P6: the VRAM tier follows the conversation.  Candidates are missing experts routed at least
        // twice (decayed); each is paired with its layer's least-routed resident expert and swapped when it was
        // routed clearly more often.  Copies run between rounds, when the GPU is idle.
        auto adapt = [&]() -> bool {
            const Clock::time_point ta = Clock::now();
            ++adapt_rounds;
            // STRATA_TRACE_ADAPT: why an adapt round did or did not swap.  Default off, one getenv, and it
            // reports the only thing that can make an adapt round a coin flip: whether the PREVIOUS round's
            // asynchronous expert copies had landed by the time this round started.
            static const bool trace_adapt = std::getenv("STRATA_TRACE_ADAPT") != nullptr;
            if (!pending.empty()) {
                if (trace_adapt)
                    std::fprintf(stderr, "strata: ADAPT round=%lld SKIPPED, %zu swaps still in flight\n",
                                 (long long) adapt_rounds, pending.size());
                return true;   // the previous swaps are still in flight
            }
            if (trace_adapt) std::fprintf(stderr, "strata: ADAPT round=%lld considering\n", (long long) adapt_rounds);
            struct Swap { float gain; int32_t layer, in, out; };
            std::vector<Swap> swaps;
            std::vector<std::pair<float, int32_t>> cand, vict;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                cand.clear();
                vict.clear();
                const float* u = drive.d.usage.data() + l * g.n_expert;
                const int32_t* r = host_res.data() + l * g.n_expert;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    if (r[e] < 0) { if (u[e] >= 2.0f && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
                    else vict.emplace_back(u[e], e);
                }
                if (cand.empty() || vict.empty()) continue;
                std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
                const size_t nc = std::min(cand.size(), vict.size());
                std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                                  [](auto& a, auto& b) { return a.first < b.first; });
                for (size_t i = 0; i < nc; ++i) {
                    if (cand[i].first < vict[i].first + 1.5f) break;
                    swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
                }
            }
            std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
            if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
            // this (non-serve) path refills CUDA0's cache only, so its copies back read CUDA0's cache too
            if (!resident_stage_swaps(src, host_res, g.n_expert, swaps,
                                      [&](int64_t) -> SwapHome { return {&xcache, adapt_stream, -1}; })) {
                std::fprintf(stderr, "strata generate: an adaptive refill failed (copying evicted experts back)\n");
                return false;
            }
            if (pin_blobs_on()) {   // lock the batch's source pages (a file-backed arena on AMD; see pin_blobs)
                std::vector<std::pair<uintptr_t, uintptr_t>> spans;
                for (const Swap& s : swaps)
                    if (const uint8_t* b = srcp->blob(s.layer, s.in))
                        spans.emplace_back((uintptr_t) b,
                                           (uintptr_t) b + strata::kernels::cpu::expert_layout().blob_bytes(s.layer));
                pin_blobs(std::move(spans), pin_live);
            }
            // STRATA_DMA_BATCH (F12): the round's copies as one cudaMemcpyBatchAsync; mode 0 is the loop of one copy each
            const int dma_mode = strata::core::dma_batch_mode();
            std::vector<void*> cp_dst;
            std::vector<const void*> cp_src;
            std::vector<size_t> cp_bytes;
            for (const Swap& s : swaps) {
                const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
                const int32_t slot = host_res[out];
                const uint8_t* b = srcp->blob(s.layer, s.in);
                const size_t blob_n = (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer);
                // asynchronous: the copies run while the MTP drafts; the next window waits for them
                if (slot < 0 || b == nullptr) {
                    std::fprintf(stderr, "strata generate: an adaptive refill failed\n");
                    return false;
                }
                if (dma_mode != 0) {
                    cp_dst.push_back(xcache.device_slot(slot));
                    cp_src.push_back(b);
                    cp_bytes.push_back(blob_n);
                } else if (adapt_copy_h2d(xcache.device_slot(slot), b, blob_n,
                                          adapt_stream) != 0) {
                    std::fprintf(stderr, "strata generate: an adaptive refill failed\n");
                    return false;
                }
                host_res[out] = strata::core::kNotResident;   // evicted now: the CPU computes it meanwhile
                srcp->prefetch(s.layer, s.out);   // a file-backed arena released its pages: read them back ahead
                pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
            }
            if (!cp_dst.empty() &&
                strata::core::copy_blobs(cp_dst.data(), cp_src.data(),
                                         cp_bytes.data(), cp_dst.size(),
                                         adapt_stream, dma_mode) != 0) {
                std::fprintf(stderr, "strata generate: an adaptive refill failed\n");
                return false;
            }
            if (!swaps.empty()) dpct::sync_barrier(adapt_ev, adapt_stream);
            if (trace_adapt)
                std::fprintf(stderr, "strata: ADAPT round=%lld swapped %zu of %d slots, usage decayed\n",
                             (long long) adapt_rounds, swaps.size(), o.adapt_swaps);
            for (float& v : drive.d.usage) v *= o.adapt_decay;
            swaps_total += (int64_t) swaps.size();
            ms_adapt += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
            return true;
        };
        int64_t p = spec_pos;
        int32_t x = (int32_t) tok;
        std::vector<int32_t> drafts((size_t) o.spec, 0);
        std::vector<float> dprob((size_t) o.spec, 1.0f);
        std::vector<int64_t> window_hist((size_t) o.spec + 1, 0);
        // plan v0.3 P6: with a native pack the first window is the last prompt token alone (it produces the first
        // generated token and the MTP's first cell); otherwise the token loop already did that.
        bool first_window = native_pack;
        // SYCL port: every window graph and the drafter's graphs are captured here, once, instead of on first use in
        // the loop below (STRATA_WARM_GRAPHS=0: as before). A server pays this once per process, not per request.
        {
            static const bool warm_on = [] { const char* v = std::getenv("STRATA_WARM_GRAPHS"); return !(v && v[0] == '0'); }();
            if (warm_on) {
                const Clock::time_point tw = Clock::now();
                if (!ver.warm(err) || (use_mtp && !mtp.warm(err))) {
                    std::fprintf(stderr, "strata generate: warm-up capture: %s\n", err.c_str());
                    return 1;
                }
                std::fprintf(stderr, "strata generate: window and draft graphs captured in %.0f ms\n",
                             std::chrono::duration<double, std::milli>(Clock::now() - tw).count());
            }
        }
        if (use_mtp && !first_window &&
            !mtp.draft_first(o.spec, ss.R, x, p - 1, drafts.data(), err, dprob.data(), (float) o.spec_min_p)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::vector<int32_t> window((size_t) o.spec), outv((size_t) o.spec);
        std::vector<uint8_t> fdiff((size_t) o.spec, 0);
        std::FILE* hashes = o.window_hashes.empty() ? nullptr : std::fopen(o.window_hashes.c_str(), "w");
        if (!o.window_hashes.empty() && hashes == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.window_hashes.c_str());
            return 1;
        }
        std::vector<float> hash_rows;
        std::vector<int64_t> accepted_hist((size_t) o.spec, 0);
        int64_t rounds = 0, drafts_total = 0, drafts_ok = 0, corrupt_counter = 0;
        const int S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, o.spec) : o.spec;
        if (use_mtp && S_mtp < o.spec) mtp.set_max_drafts(S_mtp - 1);
        strata::spec::SuffixDrafter sfx(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
        strata::spec::DraftPolicy policy(o.spec);   // MTP or lookup window (see draft_policy.hpp)
        std::vector<int32_t> sbuf((size_t) o.spec, 0);
        int64_t sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;
        int64_t chain_windows = 0, chain_drafts = 0, chain_ok = 0;   // --lookup-chain's own counts
        std::vector<int32_t> cbuf((size_t) o.spec, 0), ctail;
        strata::spec::PromptLookupSource lookup_src(sfx);
        const bool sfx_on = o.suffix_draft > 0 || o.lookup_chain > 0;
        if (sfx_on) {
            for (int64_t t : o.tokens) sfx.append((int32_t) t);
            for (int64_t t : produced) sfx.append((int32_t) t);
        }
        if (o.lookup_chain > 0) {
            extra_sources_reset();
            for (int64_t t : o.tokens) { const int32_t t32 = (int32_t) t; extra_sources_append(&t32, 1); }
            for (int64_t t : produced) { const int32_t t32 = (int32_t) t; extra_sources_append(&t32, 1); }
        }
        const double pool_ms0 = drive.cpu_ms;
        const int64_t misses0 = drive.d.multi_misses, entries0 = drive.d.multi_entries;
        while ((int64_t) produced.size() < max_new) {
            const Clock::time_point t0 = Clock::now();
            int T = S_mtp;
            if (use_mtp && o.spec_min_p > 0.0) {
                T = 1;
                while (T < S_mtp && dprob[(size_t) T - 1] >= (float) o.spec_min_p) ++T;
            }
            if (first_window) T = 1;
            bool from_sfx = false;
            int sfx_match = 0;
            if (o.suffix_draft > 0 && !first_window) {
                const int k = sfx.propose(o.spec - 1, sbuf.data());
                sfx_match = sfx.last_match();
                if (k > 0 && (!use_mtp || sbuf[0] == drafts[0])) {
                    const strata::spec::DraftPolicy::Pick pk = policy.choose(T, k, sfx_match);
                    if (pk.lookup) { T = pk.t; from_sfx = true; }
                }
            }
            // --lookup-chain: what followed an earlier occurrence of the context + the MTP's drafts, after them
            int chain_n = 0, cm = 0;
            if (o.lookup_chain > 0 && use_mtp && !first_window && !from_sfx && T < o.spec) {
                int csrc = -1;
                const int nt = chain_tail(sfx, drafts.data(), T - 1, ctail);
                chain_n = strata::spec::propose_from_sources(lookup_src, ctail.data(), nt, T - 1,
                                                             std::min(o.lookup_chain, o.spec - T), o.lookup_chain_min,
                                                             cbuf.data(), &cm, &csrc);
                if (chain_n > 0 && std::getenv("STRATA_LOOKUP_CHAIN_FIXED") == nullptr) {
                    double p_mtp = 1.0;
                    for (int i = 0; i < T - 1; ++i) p_mtp *= dprob[(size_t) i];
                    chain_n = policy.chain(T, p_mtp, chain_n, cm);
                }
            }
            const int T_mtp = T;
            T += chain_n;
            const bool timed_round = !first_window;
            ++window_hist[(size_t) T];
            if (p + T > o.max_context) {
                std::fprintf(stderr, "strata generate: ran out of context at position %lld\n", (long long) p);
                return 2;
            }
            window[0] = x;
            for (int i = 1; i < T; ++i) {
                const size_t at = produced.size() - 1 + (size_t) i;
                int32_t d = from_sfx ? sbuf[(size_t) i - 1] : use_mtp ? drafts[(size_t) i - 1]
                                                    : at < oracle.size() ? (int32_t) oracle[at] : 0;
                if (i >= T_mtp) d = cbuf[(size_t) (i - T_mtp)];
                if (o.spec_corrupt > 0 && (++corrupt_counter % o.spec_corrupt) == 0) d = (d + 1) % (int32_t) n_vocab;
                window[(size_t) i] = d;
            }
            drive.d.layers = 0;
            drive.d.experts = 0;
            drive.d.failed = false;
            // #463: the previous adapt round's copies land first - with a non-blocking query, whether a swapped-in
            // expert ran on the GPU or the CPU (they round differently) depended on the copy's timing
            // (STRATA_ADAPT_NOWAIT=1: 0.1.37's non-blocking query, the A/B)
            if (adapt_nowait()) apply_pending(false);
                else if (pending.empty() || ++pending_age >= adapt_lag()) apply_pending(true);
            if (!ver.run(T, window.data(), p, &drive_pool_multi, &drive, outv.data(), err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            if (drive.d.failed) {
                std::fprintf(stderr, "strata generate: the expert pool failed at layer %lld expert %lld: %s\n",
                             (long long) drive.d.fail_layer, (long long) drive.d.fail_expert,
                             drive.d.fail ? drive.d.fail : "(no message)");
                return 1;
            }
            if (hashes != nullptr) {   // FNV-1a over the window's final residual rows (a test path: a plain copy)
                const size_t n_floats = (size_t) T * (size_t) (g.hc * g.n_embd);
                hash_rows.resize(n_floats);
                dpct::get_in_order_queue()
                    .memcpy(hash_rows.data(), ver.final_R_all(),
                            n_floats * sizeof(float))
                    .wait();
                uint64_t h = 1469598103934665603ull;
                const auto* bytes = (const uint8_t*) hash_rows.data();
                for (size_t i = 0; i < n_floats * sizeof(float); ++i) h = (h ^ bytes[i]) * 1099511628211ull;
                std::fprintf(hashes, "%lld %lld %d %016llx", (long long) rounds, (long long) p, T, (unsigned long long) h);
                for (int i = 0; i < T; ++i) std::fprintf(hashes, " %d", (int) outv[(size_t) i]);
                std::fprintf(hashes, "\n");
            }
            if (!follow.empty()) {   // the continuation stands in for the argmax
                const size_t k0 = produced.size();
                for (int i = 0; i < T && k0 + (size_t) i < follow.size(); ++i) {
                    fdiff[(size_t) i] = outv[(size_t) i] != (int32_t) follow[k0 + (size_t) i];
                    outv[(size_t) i] = (int32_t) follow[k0 + (size_t) i];
                }
            }
            int a = 0;
            while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
            if (std::getenv("STRATA_DBG_DRAFT") != nullptr) {   // the window the drafter built and what the target said
                std::fprintf(stderr, "draftdbg: window"); for (int i = 0; i < T; ++i) std::fprintf(stderr, " %d", (int) window[(size_t) i]);
                std::fprintf(stderr, " | target"); for (int i = 0; i < T; ++i) std::fprintf(stderr, " %d", (int) outv[(size_t) i]);
                std::fprintf(stderr, " | accepted %d\n", a);
            }
            if (first_window) {
                first_window = false;
                ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
                // Diagnostics: the first window runs the prompt's last token over the state the prompt path left
                // (keys, values, recurrent state), so its logits carry whatever that path did. A native pack never
                // runs the per-token loop --dump-logits reads; this is where prompt paths can be compared by output.
                if (const char* fl = std::getenv("STRATA_DUMP_FIRST_LOGITS")) {
                    std::vector<float> row((size_t) ver.vocab());
                    std::FILE* f = ver.copy_logits(0, row.data()) ? std::fopen(fl, "wb") : nullptr;
                    if (f == nullptr || std::fwrite(row.data(), sizeof(float), row.size(), f) != row.size())
                        std::fprintf(stderr, "strata generate: STRATA_DUMP_FIRST_LOGITS: cannot write %s\n", fl);
                    if (f) std::fclose(f);
                }
            }
            // plan v0.3 P6: the adaptive tier's host work (ranking, copy submission) runs on its own thread while the
            // GPU commits and drafts; it touches only the residency tables, which nothing reads until the next window
            std::thread adapt_thr;
            bool adapt_ok = true;
            if (!drive.d.usage.empty() && ((rounds + 1) % o.adapt_every) == 0)
                adapt_thr = std::thread([&] { adapt_ok = adapt(); });
            if (!ver.commit(a + 1, err)) {
                if (adapt_thr.joinable()) adapt_thr.join();
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            ++rounds;
            drafts_total += T - 1;
            drafts_ok += a;
            ++accepted_hist[(size_t) a];
            if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
            if (chain_n > 0) { ++chain_windows; chain_drafts += chain_n; chain_ok += std::max(0, a - (T_mtp - 1)); }
            bool eos = false;
            for (int i = 0; i <= a && (int64_t) produced.size() < max_new && !eos; ++i) {
                produced.push_back(outv[(size_t) i]);
                if (sfx_on) sfx.append(outv[(size_t) i]);
                if (o.lookup_chain > 0) extra_sources_append(&outv[(size_t) i], 1);
                eos = o.stop_eos && std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                if (!follow.empty()) { follow_differ += fdiff[(size_t) i]; ++follow_emitted; }
            }
            if (eos) {
                if (adapt_thr.joinable()) adapt_thr.join();
                if (!ver.wait_commit(err)) { std::fprintf(stderr, "strata generate: %s\n", err.c_str()); return 1; }
                total_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                break;
            }
            const bool drafted = !use_mtp || (int64_t) produced.size() >= max_new ||
                                 mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data(), (float) o.spec_min_p);
            if (adapt_thr.joinable()) adapt_thr.join();
            if (!adapt_ok) return 1;
            if (!drafted) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            x = outv[(size_t) a];
            p += a + 1;
            const double round_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            total_ms += round_ms;
            if (timed_round && chain_n == 0) policy.observe(from_sfx, T, a, sfx_match, round_ms);
            else if (timed_round) policy.observe_chain(T_mtp, chain_n, a, cm, round_ms);
            if (rounds % 64 == 0)
                std::fprintf(stderr, "strata generate: position %lld, %lld tokens, %lld rounds\n", (long long) p,
                             (long long) produced.size(), (long long) rounds);
        }
        // the last commit (set_commit_async) before anything reads the session again
        if (!ver.wait_commit(err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (hashes != nullptr) std::fclose(hashes);
        std::printf("%-24s %lld rounds of %d, drafts accepted %lld of %lld (%.3f), %.2f tokens per round\n",
                    "speculation", (long long) rounds, o.spec, (long long) drafts_ok, (long long) drafts_total,
                    drafts_total > 0 ? (double) drafts_ok / (double) drafts_total : 0.0,
                    rounds > 0 ? (double) (drafts_ok + rounds) / (double) rounds : 0.0);
        if (!follow.empty())
            std::printf("%-24s %lld of %lld emitted tokens differ from the argmax\n", "follow",
                        (long long) follow_differ, (long long) follow_emitted);
        if (o.spec_min_p > 0.0) {
            std::printf("%-24s", "window sizes");
            for (size_t i = 1; i < window_hist.size(); ++i) std::printf(" T%zu:%lld", i, (long long) window_hist[i]);
            std::printf("  (min draft probability %.2f)\n", o.spec_min_p);
        }
        if (o.suffix_draft > 0)
            std::printf("%-24s %lld windows, drafts accepted %lld of %lld\n", "suffix drafts", (long long) sfx_windows,
                        (long long) sfx_ok, (long long) sfx_drafts);
        if (o.lookup_chain > 0)
            std::printf("%-24s %lld windows, chained drafts accepted %lld of %lld\n", "lookup chain",
                        (long long) chain_windows, (long long) chain_ok, (long long) chain_drafts);
        std::printf("%-24s", "accepted per round");
        for (size_t i = 0; i < accepted_hist.size(); ++i) std::printf(" %zu:%lld", i, (long long) accepted_hist[i]);
        std::printf("\n");
        if (rounds > 0)
            std::printf("%-24s wait for rings %.3f  pool %.3f  host %.3f  commit %.3f ms/round; CPU experts %.2f "
                        "distinct / %.2f routed per layer\n",
                        "verify window", ver.ms_wait / rounds, ver.ms_pool / rounds, ver.ms_host / rounds,
                        ver.ms_commit / rounds,
                        (double) (drive.d.multi_misses - misses0) / (double) (rounds * g.n_layers),
                        (double) (drive.d.multi_entries - entries0) / (double) (rounds * g.n_layers));
        if (rounds > 0)
            std::printf("%-24s gate/up %.3f  quantize %.3f  down %.3f ms/round; %.1f GB/s over the rows phases; "
                        "CPU pool call %.3f ms/round\n", "pool multi", pool.ms_multi_gu / rounds,
                        pool.ms_multi_q / rounds, pool.ms_multi_down / rounds,
                        (double) pool.multi_bytes / 1e6 / std::max(1e-9, pool.ms_multi_gu + pool.ms_multi_down),
                        (drive.cpu_ms - pool_ms0) / rounds);
        if (rounds > 0)
            std::printf("%-24s plan %.3f  activation quantize %.3f  jobs %.3f  run %.3f ms/round\n", "dispatch",
                        drive.d.ms_plan / rounds, drive.d.ms_actq / rounds, drive.d.ms_jobs / rounds,
                        drive.d.ms_run / rounds);
        if (usage_dev != nullptr) {   // STRATA_USAGE_DUMP: the window's routing as make_profile.py reads it (layer, k=1, id, weight)
            std::vector<int32_t> uh((size_t) (g.n_layers * g.n_expert));
            dpct::get_in_order_queue().memcpy(uh.data(), usage_dev, uh.size() * 4).wait();
            if (FILE* uf = std::fopen(usage_dump, "wb")) {
                int64_t total = 0;
                for (int64_t l = 0; l < g.n_layers; ++l)
                    for (int64_t e = 0; e < g.n_expert; ++e)
                        for (int32_t c = 0; c < uh[(size_t) (l * g.n_expert + e)]; ++c, ++total) {
                            const int32_t hdr[3] = {(int32_t) l, 1, (int32_t) e};
                            const float wgt[2] = {0.0f, 0.0f};
                            std::fwrite(hdr, 4, 3, uf);
                            std::fwrite(wgt, 4, 1, uf);   // (layer, k) + k ids + k weights: 8 + 8 bytes
                        }
                std::fclose(uf);
                std::printf("%-24s %s (%lld routed entries counted on the device)\n", "usage dumped", usage_dump,
                            (long long) total);
            }
        }
        if (rounds > 0 && !drive.d.usage.empty())
            std::printf("%-24s %lld experts swapped into the VRAM tier (every %d rounds, %.3f ms/round)\n", "adaptive tier",
                        (long long) swaps_total, o.adapt_every, ms_adapt / rounds);
        if (src.complement_ready())
            std::printf("%-24s %.2f GiB of experts in RAM, %lld exchanged with the VRAM tier, %lld blob reads from "
                        "the file\n", "resident RAM", (double) src.resident_bytes() / 1073741824.0,
                        (long long) src.exchanges(), (long long) src.file_reads());
        // STRATA_CHECK_SWAPS=1 (a test aid): after the run every expert must be held (the GPU cache or the RAM copy),
        // and every RAM-copy blob must equal the model file's bytes (nothing lost, nothing stale after the swaps)
        if (src.complement_ready())
            if (const char* chk = std::getenv("STRATA_CHECK_SWAPS"); chk != nullptr && chk[0] != '\0' && chk[0] != '0') {
                int64_t held_ram = 0, held_gpu = 0, lost = 0, stale = 0;
                std::vector<uint8_t> ref;
                for (int64_t l = 0; l < g.n_layers; ++l)
                    for (int64_t e = 0; e < g.n_expert; ++e) {
                        const bool ram = src.has_resident(l, e);
                        const bool gpu = host_res[(size_t) l * (size_t) g.n_expert + (size_t) e] >= 0;
                        held_ram += ram;
                        held_gpu += gpu;
                        if (!ram && !gpu) { ++lost; continue; }
                        if (!ram) continue;
                        ref.resize((size_t) strata::kernels::cpu::expert_layout().blob_bytes(l));
                        if (!src.copy_blob_from_file(l, e, ref.data()) ||
                            std::memcmp(ref.data(), src.resident_blob(l, e), ref.size()) != 0) {
                            if (stale++ < 8)
                                std::fprintf(stderr, "strata: STRATA_CHECK_SWAPS: layer %d expert %d: the RAM copy differs from the file\n",
                                             (int) l, (int) e);
                        }
                    }
                std::printf("%-24s audit: %lld experts in RAM, %lld in the GPU cache, %lld held by neither, %lld RAM blobs differ from the file\n",
                            "STRATA_CHECK_SWAPS", (long long) held_ram, (long long) held_gpu, (long long) lost, (long long) stale);
            }
        if (srcp == &src) {  // CS-T: the RAM and file tiers (the GPU cache's share is the hit rate above)
            const double fms = src.file_ms() - fms0, fmb = (double) (src.file_blob_bytes() - fbytes0) / 1e6;
            std::printf("%-24s decode: RAM %lld blobs, files %lld blobs, %.1f MB read from the files (%.2f MB/round, "
                        "%.1f ms/round of reading, %.2f GB/s per reading thread)%s; prompt copies %.1f MB\n",
                        "expert tiers", (long long) (src.ram_reads() - ram0), (long long) (src.file_reads() - files0),
                        fmb, rounds > 0 ? fmb / rounds : 0.0, rounds > 0 ? fms / rounds : 0.0,
                        fms > 0 ? fmb / fms : 0.0, src.gguf_mode() ? " (the GGUF in place)" : "",
                        (double) (fall0 - fbytes0) / 1e6);
            if (drive.d.lookahead != nullptr)
                std::printf("%-24s %lld experts warmed, %lld of the file tier's %lld blob reads had been warmed (%.1f%%); "
                            "predictor %.1f ms/round on its thread, %lld layers skipped (still busy)\n", "routing prefetch",
                            (long long) src.warmed(), (long long) src.warmed_hits(),
                            (long long) (src.file_reads() - files0),
                            src.file_reads() > files0 ? 100.0 * (double) src.warmed_hits() / (double) (src.file_reads() - files0) : 0.0,
                            rounds > 0 ? drive.d.lookahead->busy_ms() / rounds : 0.0, (long long) drive.d.lookahead->skipped());
        }
        if (rounds > 0 && drive.d.pcie_num > 0)
            std::printf("%-24s %.2f distinct experts per layer read over PCIe (share %d/256 of the misses)\n",
                        "pcie experts", (double) (drive.d.pcie_experts - pcie0) / (double) (rounds * g.n_layers),
                        drive.d.pcie_num);
        (void) pool_ms0;
        if (use_mtp && rounds > 0)
            std::printf("%-24s %.3f ms/round drafting (%lld rounds), MTP prompt %.1f ms, %.0f MiB of VRAM\n", "mtp",
                        mtp.ms_draft / (double) mtp.rounds, (long long) mtp.rounds, mtp.ms_prefill,
                        (double) mtp.vram_bytes() / 1048576.0);
    }

    if (dump != nullptr && std::fclose(dump) != 0) {
        std::fprintf(stderr, "strata generate: cannot finish logits dump\n");
        return 1;
    }
    if (layer_dump != nullptr) {
        std::fclose(layer_dump);
        sycl::free(layer_stage, dpct::get_in_order_queue());
        std::printf("%-24s %s (%lld layers + the input x %d streams x %lld per position)\n", "layers dumped",
                    o.dump_layers.c_str(), (long long) g.n_layers, (int) g.hc, (long long) g.n_embd);
    }
    if (half_dump != nullptr) {
        std::fclose(half_dump);
        sycl::free(half_stage, dpct::get_in_order_queue());
        std::printf("%-24s %s (%lld layers x %llu per position)\n", "halves dumped", o.dump_halves.c_str(),
                    (long long) g.n_layers, (unsigned long long) half_stride);
    }
    if (routing != nullptr) {
        std::fclose(routing);
        drive.routing = nullptr;
        std::printf("%-24s %s (%lld records of layer, k, ids, weights)\n", "routing dumped",
                    o.dump_routing.c_str(), (long long) drive.calls);
    }
    if (o.stage_timing) strata::core::stage_timing_report(g.n_layers);

    const int64_t decoded = (int64_t) produced.size();
    std::printf("prompt  :");
    for (int64_t t : o.tokens) std::printf(" %lld", (long long) t);
    std::printf("\noutput  :");
    for (int64_t t : produced) std::printf(" %lld", (long long) t);
    std::printf("\n");
    const double decode_ms = decoded > 0 ? total_ms / (double) decoded : 0.0;
    std::printf("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s\n", "decode", (long long) decoded, total_ms,
                decode_ms > 0.0 ? 1000.0 / decode_ms : 0.0);
    if (n_prompt > 1)
        std::printf("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s  (time to first token %.1f ms)\n", "prefill",
                    (long long) (n_prompt - 1), prefill_ms,
                    prefill_ms > 0 ? 1000.0 * (double) (n_prompt - 1) / prefill_ms : 0.0, ttft_ms);
    if (!o.dump_mixed.empty()) {
        std::vector<float> mx((size_t) g.n_embd);
        if (DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                 .memcpy(mx.data(), ss.block.mixed,
                                         mx.size() * sizeof(float))
                                 .wait()) != 0) {
            std::fprintf(stderr, "strata generate: reading mixed back failed\n");
            return 1;
        }
        std::FILE* mf = std::fopen(o.dump_mixed.c_str(), "wb");
        if (mf == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_mixed.c_str());
            return 1;
        }
        std::fwrite(mx.data(), sizeof(float), mx.size(), mf);
        std::fclose(mf);
        double s2 = 0, mag = 0;
        for (float v : mx) { s2 += (double) v * (double) v; mag += std::fabs((double) v); }
        std::printf("%-24s %s (n_embd %lld, rms %.5g, mean|.| %.5g)\n", "mixed dumped", o.dump_mixed.c_str(),
                    (long long) g.n_embd, std::sqrt(s2 / (double) mx.size()), mag / (double) mx.size());
    }

    // ---- the residual, for bisecting the head against the layers (see `dump_residual`'s note)
    if (!o.dump_residual.empty()) {
        std::vector<float> R((size_t) g.hc * g.n_embd);
        if (DPCT_CHECK_ERROR(
                dpct::get_in_order_queue()
                    .memcpy(R.data(), ss.R, R.size() * sizeof(float))
                    .wait()) != 0) {
            std::fprintf(stderr, "strata generate: reading R back failed\n");
            return 1;
        }
        std::FILE* rf = std::fopen(o.dump_residual.c_str(), "wb");
        if (rf == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_residual.c_str());
            return 1;
        }
        const int32_t hdr[2] = {(int32_t) g.hc, (int32_t) g.n_embd};
        std::fwrite(hdr, sizeof hdr, 1, rf);
        std::fwrite(R.data(), sizeof(float), R.size(), rf);
        std::fclose(rf);
        double mag = 0, mx = 0;
        int bad = 0;
        for (float v : R) {
            if (!std::isfinite(v)) ++bad;
            else { mag += std::fabs((double) v); mx = std::max(mx, (double) std::fabs((double) v)); }
        }
        std::printf("%-24s %s (%d x %lld, nonfinite %d, mean|.| %.4g, max|.| %.4g)\n", "residual dumped",
                    o.dump_residual.c_str(), (int) g.hc, (long long) g.n_embd, bad, mag / (double) R.size(), mx);
    }

    if (o.stats) {
        std::printf("%-24s %.3f ms/token (WALL CLOCK: embed, layers, head, sample)\n", "  per token", decode_ms);
        // **THE PER-TOKEN HOST TERM, WHICH `--gpu-only-full` CANNOT SEE.**  That measurement never enters the
        // token loop, so it excludes all six of these.  On the 78-token fixture + 200 generated tokens the six
        // sum to ~9 ms of non-layer work against ~1.5 ms of actual head GPU work - 16% of the token, and it is
        // not the pool.
        if (phase_tokens > 0) {
            const double pt = (double) phase_tokens;
            std::printf("%-24s PLE %.3f  embed %.3f  LAYERS %.3f  head %.3f  readback %.3f  sample %.3f  "
                        "(sum %.3f of %.3f ms)\n",
                        "  token host phases", ms_ple / pt, ms_embed / pt, ms_layers / pt, ms_head / pt,
                        ms_readback / pt, ms_sample / pt,
                        (ms_ple + ms_embed + ms_layers + ms_head + ms_readback + ms_sample) / pt, decode_ms);
        }
        if (const std::string io = ple_table.io_report(); !io.empty()) std::printf("  %s\n", io.c_str());
        // **THE DENOMINATOR IS THE POSITIONS THE POOL ACTUALLY RAN ON, NOT THE DECODED TOKENS (A6).**
        // `drive_pool` is called once per layer per position and PREFILL runs the loop too, so accumulating
        // `cpu_ms` over prefill and then dividing by `decoded` inflates this figure.  `drive.calls / n_layers`
        // is the number of positions - the same correction the ring counters below already received, which is
        // why they print "of 192" rather than "240 of 192".
        const double pool_positions = g.n_layers > 0 ? (double) drive.calls / (double) g.n_layers : 0.0;
        std::printf("%-24s %.3f ms/token over %lld layers (%.0f positions, %lld dispatches)\n",
                    "  the CPU expert pool", pool_positions > 0.0 ? drive.cpu_ms / pool_positions : 0.0,
                    (long long) g.n_layers, pool_positions, (long long) drive.calls);
        // **AND WHERE INSIDE `run()` IT WENT.**  Three phases per layer and they were one number, which cannot
        // tell a pool that is slow at the WORK from one that is slow at the SYNCHRONISATION - opposite fixes.
        // Wait-for-park is expected to be ~0 (the workers re-parked at the end of the previous layer); the
        // question is whether the time is in the drain or in the re-park barrier.
        if (pool_positions > 0.0) {
            double wp = 0, dr = 0, rp = 0;
            pool.phase_ms(wp, dr, rp);
            const double per = pool_positions;
            std::printf("%-24s   wait-park %.3f  drain %.3f  re-park %.3f  ms/token\n",
                        "  pool phases", wp / per, dr / per, rp / per);
        }
        std::printf("%-24s %lld blobs read\n", "  expert blobs", (long long) srcp->reads());
        for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
            std::printf("  CUDA%d experts           %lld routed entries computed\n",
                        r + 1, (long long) remote_experts[(size_t) r].computed());
        // ---- **R4's DISPATCH MEASUREMENT: h, ON THE ENGINE'S OWN ROUTING.**  No offline trace, no corpus
        // question, no k-fold - these are the ids the router actually produced on this run.  Reported as
        // hits/lookups so it can be read directly as the h the cache would deliver, and alongside `refused`
        // so a full cache is visible rather than silently capping the rate.
        if (o.expert_cache > 0) {
            const int64_t look = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;
            const int64_t hl = drive.d.hit_ready + drive.d.hit_late;
            std::printf("%-24s %lld of %lld layers the hit work was DONE when the pool returned\n",
                        "  R4 overlap", (long long) drive.d.hit_ready, (long long) hl);
            std::printf("%-24s %lld of %lld = %.4f      (%lld admitted, %lld refused, cache %.4f%% full)\n",
                        "  R4 expert-cache hits", (long long) drive.d.cache_hits, (long long) look,
                        look > 0 ? (double) drive.d.cache_hits / (double) look : 0.0,
                        (long long) drive.d.cache_admitted, (long long) drive.d.cache_refused,
                        100.0 * (double) (drive.d.cache_admitted + drive.d.cache_hits > 0
                                              ? (double) xcache.resident() / (double) xcache.slots()
                                              : 0.0));
        }
        if (kvg.on)
            std::printf("%-24s %lld cells in VRAM, %lld grows, %lld trims, %lld experts given up, %lld refilled\n",
                        "  elastic K/V", (long long) kvg.cells, (long long) kvg.grows, (long long) kvg.trims,
                        (long long) kvg.evicted, (long long) kvg.refilled);
        if (tgraph.captured && tgraph.calls > 0) {
            const double per = (double) tgraph.calls;
            std::printf("%-24s wait for rings %.3f  pool %.3f ms/token  (%lld flushes over %lld positions)\n",
                        "  token graph", tgraph.ms_wait / per, tgraph.ms_pool / per, (long long) tgraph.flushes,
                        (long long) tgraph.calls);
        }
        if (gr.captured && gr.calls_total > 0) {
            // The counters are CUMULATIVE over every `session_loop` call, and PREFILL runs the loop too - so
            // the denominator is the number of positions, not the number of generated tokens.  Dividing by
            // `n_layers * decoded` printed "240 of 192", which is a reporting bug that looks like a ring
            // firing more often than it should.
            const int64_t positions = gr.calls_total;
            std::printf("%-24s %lld of %lld over %lld positions\n", "  rings seen MID-GRAPH",
                        (long long) gr.rings_mid_graph, (long long) (g.n_layers * positions),
                        (long long) positions);
            std::printf("%-24s %.3f ms of a %.3f ms layer\n", "  ring latency",
                        gr.ms_to_ring / (double) (g.n_layers * positions), decode_ms / (double) g.n_layers);
            // **THE ROUND TRIP, SPLIT AT THE RING.**  `ring latency` is the first half and stops when the ring
            // is seen; this is the second half - the driver calls after it, during which the GPU is IDLE
            // because `post[l]` has not been launched yet.  `--no-pool` is the arm that isolates it: 38.73
            // ms/token against a 26.32 ms pure-GPU floor is 12.4 ms of round trip with no expert work at all.
            //
            // Same denominator as the pool line above (the positions the loop actually ran on), so the two can
            // be added without one of them being inflated by prefill.
            const double perlap = (double) (g.n_layers * positions);
            std::printf("%-24s %.3f ms/token over %.0f positions (%.3f ms/layer, after the ring)\n",
                        "  host after ring", pool_positions > 0.0 ? gr.ms_host / pool_positions : 0.0,
                        pool_positions, gr.ms_host / perlap);
        }
    }

    if (dump != nullptr) std::printf("%-24s %s\n", "logits dumped", o.dump_logits.c_str());

    strata::core::session_graphs_free(gr);
    strata::core::doorbell_free(db);
    sycl::free(d_next, dpct::get_in_order_queue());
    sycl::free(d_logits, dpct::get_in_order_queue());
    sycl::free(d_emb, dpct::get_in_order_queue());
    sycl::free(d_parts, dpct::get_in_order_queue());
    sycl::free(sbuf, dpct::get_in_order_queue());
    sycl::free(arena, dpct::get_in_order_queue());
    return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
