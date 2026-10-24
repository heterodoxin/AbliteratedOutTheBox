// src/prefill/prefill.cpp - see include/strata/prefill/prefill.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/copy_queue.hpp"
#include "strata/prefill/xmx_moe.hpp"
#include "strata/prefill/esimd_moe.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/core/gguf_expert_source.hpp"
#include "../../../src/prefill/mmq_resident_sort.hpp"
#include "../../../src/prefill/wmma_gemm.h"
#include "strata/core/mtp.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/on_device.hpp"

#include "strata/core/layout.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/prefill/kernels.hpp"

#include <algorithm>
#include <set>
#include <map>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <sys/stat.h>
#include <tuple>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef STRATA_PREFILL_MMQ
// A build without the llama.cpp sources (no STRATA_NATIVE_EXPERTS): no MMQ, the FP16 expert path everywhere.
namespace strata::prefill::mmq {
bool built() { return false; }
bool supported(int) { return false; }
bool fits(int, int64_t) { return false; }
size_t matrix_bytes(int, int64_t, int64_t) { return 0; }
size_t q8_bytes(int64_t, int64_t) { return 0; }
void quantize(const float*, const int32_t*, void*, int, int64_t, int64_t, int64_t, void*, float*) {}
bool fp4_activations(int) { return false; }
Context::Context() {}
Context::~Context() {}
void Context::run(const Product&, void*) {}
void gather_native(const void*, const void*, size_t, const void*, size_t, void*, void*, void*, const void*, void*, size_t) {}
bool gather_native_group(const GatherGroup&, size_t, size_t, size_t, size_t, void*, size_t, void*, size_t, void*, size_t,
                         void*, size_t) { return false; }
void gather_strata_q2(const uint8_t*, void*, void*, void*) {}
void swiglu(const float*, float*, int64_t, int64_t, bool, void*) {}
void iota(int32_t*, int64_t, void*) {}
}  // namespace strata::prefill::mmq
#endif
#ifndef STRATA_PREFILL_FUSED
// #136: the fused int8 experts are CUDA-only (HIP and builds without MMQ keep the MMQ / FP16 paths)
namespace strata::prefill::fused {
bool built() { return false; }
bool available() { return false; }
bool enabled() { return false; }
bool requested() { return false; }
size_t act_bytes(int64_t, int64_t) { return 0; }
size_t group_bytes(int64_t, int) { return 0; }
void quantize_act(const float*, int64_t, int64_t, void*, void*) {}
void group(const int32_t*, int64_t, int, int, void*, int32_t*, int32_t*, void*) {}
void experts(const Batch&, int, int64_t, const void*, const void*, const int32_t*, void*, float*, void*) {}
bool native_supported(int, int) { return false; }
void quantize_act_native(const float*, int64_t, int64_t, void*, void*) {}
void experts_native(const Batch&, const NativeGeom&, int, int64_t, const void*, const void*, const int32_t*, void*,
                    float*, void*) {}
}  // namespace strata::prefill::fused
#endif

namespace strata::prefill {
namespace {

using Clock = std::chrono::steady_clock;

// 0.1.48, A750 deadlock: SYCL has no cudaHostRegister, so the expert arena is ordinary pageable memory even where the engine
// calls it "pinned" (pinned.dp.cpp registers nothing).  A DMA straight from it, `copy->memcpy(stage, arena_ptr, 2.3 MB)`, makes the
// Level Zero driver copy through its own staging buffers on the calling thread, and on some prompts that wait never ends (the
// thread spins in sched_yield inside libze_intel_gpu with the GPU idle; tools: /data/148-intel notes).  Only a USM host
// allocation (the --stream-experts pinned mirror, the B70's) is DMA'd directly; every other expert is copied into one of the
// stager's pinned buffers by its threads first, the path unpinned experts already take.  STRATA_PF_DMA_PAGEABLE=1 restores the
// direct copy from pageable memory.
inline bool dma_direct(const core::ExpertSource* src, int64_t layer, int64_t expert, const void* blob) {
    if (!src->pinned(layer, expert)) return false;
    static const bool pageable_ok = [] { const char* v = std::getenv("STRATA_PF_DMA_PAGEABLE"); return v && v[0] == '1'; }();
    if (pageable_ok) return true;
    if (blob == nullptr) return false;
    // the answer per 2 MiB bucket (a query costs ~2 us, 10k+ experts a chunk): "host USM" only when both ends of the bucket are, so a bucket
    // that straddles two allocations is never taken for USM
    static thread_local std::unordered_map<uintptr_t, bool> usm_bucket;
    const uintptr_t key = (uintptr_t) blob >> 21;
    const auto it = usm_bucket.find(key);
    if (it != usm_bucket.end()) return it->second;
    const auto ctx = dpct::get_in_order_queue().get_context();
    const bool usm = sycl::get_pointer_type((const void*) (key << 21), ctx) == sycl::usm::alloc::host &&
                     sycl::get_pointer_type((const void*) (((key + 1) << 21) - 1), ctx) == sycl::usm::alloc::host &&
                     sycl::get_pointer_type(blob, ctx) == sycl::usm::alloc::host;
    usm_bucket.emplace(key, usm);
    return usm;
}
constexpr float EPS = 1e-6f;
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10, NE = 512;
constexpr int64_t C = 10240, ZV = 6144, HV = 48;
// plan v0.3 P6: staging holds the largest blob of the pack (a native pack's blobs differ per layer)
inline int64_t MAXBLOB() { return (int64_t) strata::kernels::cpu::expert_layout().max_blob; }
constexpr int STAGE = 8;           // host->device expert staging ring (chunks below stream_all_min())
constexpr int kSplitHelpRing = 48;  // layer split help: the helper stage's ring slots it streams through (at most)
// Step 3: from this chunk size on, every non-resident expert of every layer streams in a fixed order through a
// ring_slots()-slot ring (nearly all 512 are routed at such a chunk), so the copy engine keeps working through the
// attention halves instead of waiting for each layer's routing.
// #828: STRATA_TEST_PAGEABLE=1 makes every pinned host allocation with a pageable fallback in Prefill::init fail, so the
// fallbacks (what an LXC container's memlock limit forces) run on a PC that can pin - for ASan / MALLOC_CHECK_=3 runs.
inline bool force_pageable() {
    static const bool v = [] { const char* e = std::getenv("STRATA_TEST_PAGEABLE"); return e != nullptr && e[0] == '1'; }();
    return v;
}
constexpr int RING_MAX = 1024;          // the arrays; the ring itself is ring_slots(), at most ring_cap()
// The chunk size from which every expert streams: 1024 since 0.1.30 (was 2048).  Measured on the 5070, Q2_0 / IQ2_XS,
// fixed cache: 1,500-token prompts 621 -> 785 / 612 -> 735 tok/s, 2,000 727 -> 934 / 712 -> 892, 4,000 (its last
// chunk) 779 -> 912 / 766 -> 844, the same output.  Below ~1,000 tokens the output changed on Q2_0 (a smaller chunk
// takes other kernels), so 1024 is the floor.  STRATA_PREFILL_STREAM_MIN overrides (A/B).
inline int64_t stream_all_min() {
    static const int64_t v = [] { const char* e = std::getenv("STRATA_PREFILL_STREAM_MIN"); return e ? (int64_t) std::atoll(e) : (int64_t) 1024; }();
    return v;
}
double g_pinned_share = 1.0;
// SYCL port: the share of (layer, expert) pairs the VRAM cache does not hold (set_nonresident_share, from generate.cpp)
double g_nonres_share = 0.0;
// The streamed ring: 384 slots when (nearly) every streamed expert is DMA'd from pinned RAM - measured on Q2_0,
// 8192-token chunks: 96 slots 1153 tok/s, 384 1294 (the next layer's experts arrive during its attention half) -
// and 96 when a large share goes through host copies (IQ3_S on 64 GB, a third unpinned: 96 slots 1216, 256 1070 -
// the host copies are the limit and the bigger ring only takes cache slots).  STRATA_PREFILL_RING overrides.
//
// Those are slots on the pack they were measured on, and a slot is one WHOLE expert blob - a Q2_0 blob is
// 1,382,400 B, so 384 slots is 506 MiB.  On a pack with bigger blobs the same slot count is a different amount of
// memory: 1,912 MiB on Q8_0 (5,222,400 B), which is more than half of an 8 GB card's expert cache and what held
// that card at a 1,024-token chunk when its own buffers would have fitted 6,144.  Measured on the 4-way rig, one
// env var and nothing else: chunk 1,024 -> 6,144, prefill 87 -> 402 tok/s.  So the budget is kept in BYTES and the
// slot count is derived from the pack (`ring_budget_slots`); Q2_0 still resolves to exactly 1024 (fused) and 384,
// so the pack all of this was tuned on does not move.
int g_ring_override = 0;   // #340: set by a layer split (Prefill::set_ring_override); 0 = the rule below
// #583 (0.1.39b): the auto chunk scan's byte-budget ring (Prefill::set_ring_budget), for requests whose chunk is above
// g_ring_small_max - the chunk 0.1.39's rule would have picked.  A prompt that fits that chunk keeps 0.1.39's ring:
// on one chunk the smaller ring only slowed it (RTX 5070, 4K prompts: Coder -15%, IQ3_S -5%, IQ3_XXS -2%).
int g_ring_budget = 0;
int64_t g_ring_small_max = 0;
// #136: the fused experts (STRATA_PF_FUSED=1) launch on a batch of a layer's streamed experts at once, so the ring
// should hold a whole layer's (~460 of 512 on Q2_0): with 384 slots a layer's last batch waits for slots its own
// first batch frees.  Measured on the 5070, Q2_0, the 4K / 32K code-agent prompts (one run each): fused at 384 slots
// +5% / +3% over MMQ, at 512 +19-22% / +11-12% (MMQ itself at 512: -2% / -1%).  P3: with the fused path's smaller
// buffers (moe_bufs) 1024 slots - two layers' experts - fit too; 2 pairs each, prompt tok/s at 512 / 1024 slots: 4K
// 1,512 / 1,605, 32K 2,485 / 2,660 (both chunk 8192), 128K with KV streaming 2,346 / 2,392 (the chunk falls from
// 8192 to 6144, but more experts stay resident).  A native pack likewise when the native kernels (moe_fused_iq.hpp)
// take any of its layers: IQ2_XS, 4K / 32K, their first version at 384 slots -3% / -6% against MMQ, at 512 +8% / 0%.
inline bool fused_ring() {
    if (!fused::enabled()) return false;
    if (core::peer_portable()) return false;   // multi-GPU: --peer-device keeps the MMQ path and its buffer sizes
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native) return true;
    // EVERY layer: fused_layout() shrinks the MoE buffers to the fused path's needs, so a layer the native kernels do
    // not cover (Unsloth UD-IQ4_XS's Q8_0 down projections) would run MMQ in them at the full chunk and overflow them
    // (garbage, an illegal memory access or a hung prompt on gfx1151).  A pack with such a layer keeps MMQ's buffers;
    // its covered layers still take the fused kernels.
    static const bool all = [&lay] {
        for (const auto& f : lay.fmt)
            if (!fused::native_supported(f.gu_type, f.d_type)) return false;
        return !lay.fmt.empty();
    }();
    return all;
}
// the largest ring: 512 slots; 1024 with the Q2_0 pack's fused experts (P3's smaller buffers, measured there) - the
// native packs' fused layers were measured at 512
inline int ring_cap() { return fused_ring() && !strata::kernels::cpu::expert_layout().native ? RING_MAX : 512; }
// The ring's budget in bytes: the measured slot counts above, at the blob size they were measured with.  It is
// bytes and not slots because a slot is one whole blob and the blob is the pack's - see the note above.
inline constexpr uint64_t Q2_0_BLOB = 1382400ull;   // the blob of the pack the ring was tuned on
inline uint64_t ring_bytes() {
    const uint64_t slots = fused_ring() ? 1024ull : 384ull;   // #136: a fused ring holds two layers' experts
    return (g_pinned_share >= 0.9 ? slots : slots / 4) * Q2_0_BLOB;
}
// ...and what that buys on THIS pack, never past ring_cap(): the slot count `init` lays out, and what the auto
// chunk scan treats as a full ring.  A pack whose blobs are larger than Q2_0's gets fewer slots for the same
// bytes, which is the point - the ring competes with the expert cache for the same VRAM.
// 0.1.39b: on by default (#583); STRATA_RING_BYTES=0 restores 0.1.39's ring, loan and auto chunk list.  It moves the
// prompt path's loan on a native pack (fewer ring slots, the rest kept as cache slots, a larger auto chunk), so a long
// prompt's experts are read through a different mix of resident and streamed groups and its bits differ from 0.1.39's
// (RTX 5070, IQ3_XXS, 32K prompt: +14% to +26%; teacher-forced against the FP16 prompt path in the same band).
inline bool ring_bytes_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_RING_BYTES"); return v == nullptr || v[0] != '0'; }();
    return on;
}
inline int ring_budget_slots() {
    const int64_t per = MAXBLOB();
    const int64_t n = per > 0 ? (int64_t) (ring_bytes() / (uint64_t) per) : 0;
    const int cap = ring_cap();
    return (int) (n <= 0 ? 0 : (n > cap ? cap : n));
}
inline int ring_slots(size_t T) {
    const char* v = std::getenv("STRATA_PREFILL_RING");
#if defined(STRATA_USE_HIP)
    // S6: with the opt-in RDNA4 matrix-core attention (STRATA_HIP_WMMA=1) a 96-slot ring: measured with it, 9070 XT
    // 4K prompts 718 -> 1,211 tok/s (16K 1,949 -> 2,032), R9700 4K 2,426 -> 2,483 (16K the same)
    static const bool wmma = [] {
        const char* e = std::getenv("STRATA_HIP_WMMA");
        return e != nullptr && e[0] == '1';
    }();
    if (!v && g_ring_override <= 0 && wmma) return (int64_t) T >= stream_all_min() ? 96 : STAGE;
#endif
    // the byte budget as this pack's slots: 1024 fused / 384 not on Q2_0, fewer on a pack with bigger blobs.  The
    // unpinned arm stays a slot count (96): it was measured where the host copies are the limit, and there the ring
    // is not what is competing for VRAM.
    // 0.1.39's ring (1024 fused / 384 pinned, 96 when a large share goes through host copies), and the #583 byte
    // budget the auto scan chose for chunks past the size 0.1.39's rule would have picked (set_ring_budget)
    const int pinned_ring = g_pinned_share >= 0.9 ? (fused_ring() ? 1024 : 384) : 96;
    const bool budget = ring_bytes_on() && g_ring_budget > 0 && (int64_t) T > g_ring_small_max;
    const int r = v ? std::atoi(v) : g_ring_override > 0 ? g_ring_override : budget ? g_ring_budget : pinned_ring;
    if (v && r == STAGE) return STAGE; // Explicit opt-in to routed-only staging, including large chunks.
    // SYCL port: the stream-all walk (every non-resident expert of every layer, copied ahead across layers) hung on the
    // B70 in its first large chunk until 2026-10-02 - the stager's and the PLE upload's host waits on queue events,
    // which the Level Zero v2 adapter did not survive (Stager::issued_one, ple_done_seq). It runs since, and reads a
    // 40K prompt 4.8% faster than routed-only staging (1,159 vs 1,106 tok/s, same output): the default again.
    // STRATA_PREFILL_STREAM_ALL=0 keeps routed-only staging, =1 forces the walk.
    // It copies EVERY non-resident expert of a layer, routed or not: with a big host tier (the IQ2_XS: a quarter of the
    // experts in the pinned mirror) that is several times the routed ones, and a 2,184-token prompt fell from ~570 to
    // 254 tok/s. So by default only while the VRAM holds more than 90% of the pairs (the Coder, and its lent slots).
    static const int stream_all_env = [] { const char* e = std::getenv("STRATA_PREFILL_STREAM_ALL"); return e ? (e[0] == '1' ? 1 : 0) : -1; }();
    const bool stream_all_ok = stream_all_env == 1 || (stream_all_env == -1 && g_nonres_share < 0.10);
    if (!stream_all_ok) return STAGE;
    const int big = r < 16 ? 16 : r > ring_cap() ? ring_cap() : r;
    return (int64_t) T >= stream_all_min() ? big : STAGE;
}
// A770 port: STRATA_PF_XMX=2 holds up to a group of staged experts until its launch, so routed-only staging needs
// that many more slots for the lookahead to keep copying (the slots are only reused once the group's event is in)
// 151-b70: the fused int8 DPAS expert products (prefill/esimd_moe.dp.cpp, Battlemage).  Unset STRATA_PREFILL_MMQ (alias STRATA_PF_I8) is
// auto on a Battlemage GPU (bmg-g21 / bmg-g31 by architecture id) whose kernels are in the binary, the FP16 chain on every other GPU
// (Alchemist included); =0 forces the FP16 chain, =1 the int8 chain everywhere it applies (IQ3_S / IQ3_XXS / IQ2_S gate/up with IQ4_NL /
// Q2_0 down, the 640 x 2560 expert), =auto the per-group choice from the crossover measured on this GPU.  The activations are quantized,
// so the bits differ from the FP16 path (docs/INTEL_INT8_EXPERTS.md: KL 0.0002 nats, top-1 99.7%).
static_assert(strata::kernels::kXmxGroupMax <= strata::prefill::i8::kMaxGroup, "the int8 path holds the grouped XMX path's experts: its Layout must be at least as big");
inline const char* i8_env() {
    const char* e = std::getenv("STRATA_PREFILL_MMQ");
    if (e == nullptr || e[0] == 0) e = std::getenv("STRATA_PF_I8");
    return (e != nullptr && e[0] != 0) ? e : nullptr;
}
std::atomic<bool> g_i8_failed{false};    // an int8 launch threw: the FP16 chain for the rest of the process
// The chain this queue's prompt experts take: 0 FP16, 1 int8, 2 auto.
inline int i8_mode(void* cs) {
    if (g_i8_failed.load(std::memory_order_relaxed)) return 0;
    const char* e = i8_env();
    if (e != nullptr) return strata::prefill::i8::choose_mode(e, false, false);
    return strata::prefill::i8::choose_mode(nullptr, strata::prefill::i8::battlemage(cs), strata::prefill::i8::available(cs));
}
// ring sizing happens before any queue is at hand: room for the int8 groups whenever the choice could be on
inline int stage_slots() {
    const char* e = i8_env();
    const bool i8 = e != nullptr ? strata::prefill::i8::choose_mode(e, false, false) != 0 : strata::prefill::i8::any_battlemage();
    return (strata::prefill::xmx::mode() == 2 || i8) ? STAGE + strata::prefill::xmx::kMaxGroup : STAGE;
}
constexpr int DQ = 2;              // dequantized-expert ring (FP16 gate/up + down)
// A770 port: STRATA_PF_XMX=1 dequantizes a whole group of experts, then one grouped XMX launch per product
inline int dq_slots() { return strata::prefill::xmx::mode() == 1 ? strata::prefill::xmx::kMaxGroup : DQ; }
// The BF16-weight projections (hyper-connection, SSM alpha/beta, indexer, router, shared gate, PLE key/value) take
// BF16 activations here and FP32 ones in decode. STRATA_PREFILL_BF16X2=1 adds each activation's BF16 remainder as a
// second GEMM (Y = W.hi + W.lo, ~16 mantissa bits): a router that picks its top 10 from the same x decode would.
// 2 = all but the hyper-connection's; 1 = the hyper-connection's too (its activations are 10240 wide and its up
// projection writes as much: slower); 0 (the default: opt-in, it changes the prompt path's numbers) = off.
// `f16_io`: the stage's FP16 prompt path (Prefill::init decides it once per stage, on the stage's own device): FP16
// activations carry 11 mantissa bits, so the BF16 low part does not apply there.
inline int bf16x2_mode(bool f16_io) {
    static const int v = [] {
        const char* e = std::getenv("STRATA_PREFILL_BF16X2");
        return e != nullptr ? std::atoi(e) : 0;
    }();
    return f16_io ? 0 : v;
}
inline bool bf16x2(bool f16_io) { return bf16x2_mode(f16_io) != 0; }
// S23 (opt-in STRATA_HC_UPMIX=1): the hyper-connection read's up projection and gr_mix_r as one kernel (gr_upmix, gfx11);
// STRATA_HC_UPMIX_CHECK=N also runs the default pair on the first N reads and reports the difference of `mixed`
static int64_t pf_switch_min_t() {   // S23: STRATA_PF_SWITCH_MIN_T=N - the rounding-level prompt switches only on chunks of
    // N or more tokens (shorter prompts keep the default numerics, and their outputs; the switches pay on long ones)
    static const int64_t v = [] { const char* e = std::getenv("STRATA_PF_SWITCH_MIN_T"); return e ? (int64_t) std::atoll(e) : (int64_t) 0; }();
    return v;
}
inline bool hc_upmix() {
    static const bool v = [] { const char* e = std::getenv("STRATA_HC_UPMIX"); return e != nullptr && e[0] == '1'; }();
    return v;
}
inline bool bf16x2_hc(bool f16_io) { return bf16x2_mode(f16_io) == 1; }
// S23 (opt-in STRATA_CVEC_FUSE=1): a steered layer's FFN write + control vector + the next half's norm in one pass
// over R (gr_write_cvec_norm_rs; bitwise the gr_write + cvec_apply + gr_norm_rs it replaces)
// S23 (opt-in STRATA_PF_HCDOWN=1, on chunks of STRATA_PF_SWITCH_MIN_T+ tokens): the hyper-connection read's down and
// inject projections as one WMMA GEMM over xn16 (strata_pf_hcdown_bf16), xn16 written with token stride
// 10240 + 64.  Rounding-level (another k order than hipBLASLt): quality-gated.
inline bool pf_hcdown() {
    static const bool v = [] { const char* e = std::getenv("STRATA_PF_HCDOWN"); return e && e[0] == '1'; }();
    return v;
}
constexpr int64_t XN_PAD = 64;
// S (opt-in STRATA_HCD_EXACT=1, on chunks of STRATA_PF_SWITCH_MIN_T+ tokens): the HC down projection by a WMMA kernel with
// hipBLASLt's own k order (bitwise: strata_pf_hcdown_exact_bf16), xn16 written with token stride 10240 + 64 (the inject
// projection stays on hipBLASLt, reading that stride)
inline bool hcd_exact() {
    static const bool v = [] { const char* e = std::getenv("STRATA_HCD_EXACT"); return e && e[0] == '1'; }();
    return v;
}
inline bool hc_pad() { return pf_hcdown() || hcd_exact(); }
inline bool cvec_fuse() {
    static const bool v = [] { const char* e = std::getenv("STRATA_CVEC_FUSE"); return e && e[0] == '1'; }();
    return v;
}
// S23 (opt-in STRATA_PF_PAD=1 with STRATA_PF_GEMM=1): the FP16 activations of the two K 6144 projections (ssm_out's
// y_h, attn_output's attn_h) with row stride 6144 + 64, so the GEMM's rows do not camp on the memory channels (the
// 4 KB-multiple stride; Gemm::native pads the weight the same way).  Same bits.
inline bool pf_pad() {
    static const bool v = [] {
        const char* p = std::getenv("STRATA_PF_PAD"); const char* g = std::getenv("STRATA_PF_GEMM");
        return p && p[0] == '1' && g && g[0] == '1';
    }();
    return v;
}
constexpr int64_t ZV_PAD = 64;

// F-1: STRATA_GR_UNFUSED=1 keeps the FP32 copy of the normalized rows (gr_norm + gr_mix), the A/B arm
inline bool gr_unfused() {
    static const bool v = [] { const char* e = std::getenv("STRATA_GR_UNFUSED"); return e && e[0] == '1'; }();
    return v;
}

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// Either cudaMalloc (owned, freed with the object) or a bump allocation from a borrowed region; with no base and
// no region it only counts, which is how `bytes_needed` sizes the region.
struct Alloc {
    uint8_t* base = nullptr;
    uint64_t cap = 0, used = 0;
    uint64_t failed_bytes = 0;          ///< the take that could not be allocated; the failure message names it
    uint64_t granule = 0;               ///< count_only: each take rounds up to this (cudaMalloc's 2 MiB pages: owned buffers)
    bool count_only = false;
    std::vector<void*>* owned = nullptr;
    template <typename T> T *take(size_t n, bool &ok) try {
        uint64_t bytes = ((uint64_t) n * sizeof(T) + 256 + 255) & ~255ull;
        if (count_only) {
            if (granule > 0) bytes = (bytes + granule - 1) / granule * granule;
            used += bytes;
            return nullptr;
        }
        if (base != nullptr) {
            if (used + bytes > cap) { ok = false; failed_bytes = bytes; return nullptr; }
            T* p = (T*) (base + used);
            used += bytes;
            return p;
        }
        void* p = nullptr;
        if (DPCT_CHECK_ERROR(p = (void *)strata::malloc_device_guarded(
                                 bytes, dpct::get_in_order_queue())) != 0) {
            ok = false; failed_bytes = bytes; return nullptr;
        }
        owned->push_back(p);
        used += bytes;
        return (T*) p;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
};

}  // namespace

// Step 4 of the prompt-speed plan: the experts the arena could not pin (a third of the streamed ones on IQ3_S) are
// copied into pinned buffers by these threads, ahead of the launches.  Copied in line by the launching thread they
// left the GPU without queued work while each ~2 MB memcpy ran (~15 s of a 32K prompt on IQ3_S).  Job j - a layer's
// j-th unpinned expert, in launch order - lands in host buffer j % kRing, which is free again once the DMA of job
// j - kRing (recorded by the launching thread, `issued`) is done.
struct Stager {
    // D-5: the pinned ring's depth (STRATA_STAGER_RING, default 16) - how far the host copies can run ahead of the
    // DMAs of the unpinned experts' blobs
    int kRing = 16;
    // SYCL port: a job may name the expert instead of a host pointer - the thread reads it from the GGUF itself
    // (`gsrc`). `from` set: the blob is copied by the source itself (CS-T: a GGUF read in place assembles it from its
    // three role slices; a pointer to it would not live as long as the queue)
    struct Job {
        const uint8_t* src; size_t bytes; core::ExpertSource* from = nullptr; int32_t l = 0, e = 0;
        const core::GgufExpertSource* gsrc = nullptr;
    };
    std::vector<uint8_t*> buf;
    std::vector<char> pinned;
    std::vector<std::vector<uint8_t>> pageable;   // the fallback when no more RAM can be pinned
    std::vector<dpct::event_ptr> dma_done;
    uint64_t* done_seq = nullptr;            // page-locked: the copy queue's last finished DMA per buffer
    uint64_t* src_seq = nullptr;             // page-locked: the values a copy-engine queue copies into done_seq (it cannot run a fill kernel)
    std::vector<uint64_t> want;              // the sequence number each buffer's last DMA writes
    uint64_t issue_seq = 0;
    std::vector<Job> jobs;
    std::unique_ptr<std::atomic<int>[]> ready;
    size_t ready_cap = 0;
    // gen << 32 | n << 16 | next index: a claim is a CAS on the generation it woke for (a thread late from the
    // previous layer can never take a job of this one - the expert pool's issue #29 lesson)
    std::atomic<uint64_t> head{0};
    std::atomic<int> issued{0}, active{0};
    uint32_t gen = 0;
    bool quit = false;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int device = 0;

    bool init(size_t blob_bytes, int nthreads) {
        if (const char* v = std::getenv("STRATA_STAGER_RING")) kRing = std::clamp(std::atoi(v), 2, 256);
        buf.assign((size_t) kRing, nullptr);
        pinned.assign((size_t) kRing, 0);
        dma_done.assign((size_t) kRing, nullptr);
        want.assign((size_t) kRing, 0);
        done_seq = sycl::malloc_host<uint64_t>((size_t) kRing, dpct::get_in_order_queue());
        if (done_seq == nullptr) return false;
        for (int i = 0; i < kRing; ++i) done_seq[i] = 0;
        src_seq = sycl::malloc_host<uint64_t>((size_t) 4 * kRing, dpct::get_in_order_queue());
        if (src_seq == nullptr) return false;
        for (int i = 0; i < 4 * kRing; ++i) src_seq[i] = 0;
        pageable.resize(kRing);
        for (int i = 0; i < kRing; ++i) {
            /*
            DPCT1048: The original value cudaHostAllocDefault is not
            meaningful in the migrated code and was removed or replaced with 0.
            You may need to check the migrated code.
            */
            pinned[i] = !force_pageable() &&
                        DPCT_CHECK_ERROR(
                            buf[i] = (unsigned char *)sycl::malloc_host(
                                blob_bytes, dpct::get_in_order_queue())) == 0;
            if (!pinned[i]) {
                /*
                DPCT1026: The call to cudaGetLastError was removed because
                this functionality is redundant in SYCL.
                */
                pageable[(size_t)i].resize(blob_bytes);
                buf[i] = pageable[(size_t) i].data();
            }
            if (DPCT_CHECK_ERROR(dma_done[i] = new sycl::event()) !=
                0) return false;
        }
        device = dpct::get_current_device_id();
        for (int t = 0; t < nthreads; ++t) threads.emplace_back([this] { work(); });
        return true;
    }
    ~Stager() {
        finish();
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto& t : threads) t.join();
        if (done_seq) sycl::free(done_seq, dpct::get_in_order_queue());
        if (src_seq) sycl::free(src_seq, dpct::get_in_order_queue());
        for (int i = 0; i < kRing; ++i) {
            if (dma_done[i]) dpct::destroy_event(dma_done[i]);
            if (buf[i] && pinned[i])
                sycl::free(buf[i], dpct::get_in_order_queue());
        }
    }
    void work() {
        /*
        DPCT1093: The "device" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        dpct::select_device(device);
        uint32_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            for (;;) {
                active.fetch_add(1, std::memory_order_acq_rel);
                const int j = claim(seen);
                if (j < 0) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
                const int b = j % kRing;
                if (j >= kRing)   // job j - kRing's DMA from this buffer is queued
                    while (issued.load(std::memory_order_acquire) <= j - kRing) std::this_thread::yield();
                // and done (#385) - for a generation's first kRing jobs that is the previous generation's last DMA from
                // the buffer, which nothing else waits for when a chunk ends without a sync (no MTP)
                while (!buffer_free(b)) std::this_thread::yield();
                const Job& jb = jobs[(size_t) j];
                if (jb.gsrc != nullptr) {
                    if (!jb.gsrc->read_into(jb.l, jb.e, buf[b], jb.bytes)) {
                        std::fprintf(stderr, "prefill: reading expert %d of layer %d from the GGUF failed\n", jb.e, jb.l);
                        std::memset(buf[b], 0, jb.bytes);
                    }
                } else if (jb.from == nullptr) {
                    std::memcpy(buf[b], jb.src, jb.bytes);
                } else if (!jb.from->copy_blob(jb.l, jb.e, buf[b])) {
                    std::fprintf(stderr, "prefill: the expert source could not copy expert %d of layer %d\n", jb.e, jb.l);
                    std::abort();
                }
                ready[(size_t) j].store(1, std::memory_order_release);
                active.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }
    int claim(uint32_t g) {
        uint64_t cur = head.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (cur >> 32) != g) return -1;
            const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
            if (j >= n) return -1;
            if (head.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel, std::memory_order_acquire)) return j;
        }
    }
    /// A layer's jobs; the previous layer's are finished (finish()).
    void start(std::vector<Job>&& js) {
        if (js.empty()) return;
        std::lock_guard<std::mutex> lk(mu);
        jobs = std::move(js);
        if (ready_cap < jobs.size()) {
            ready_cap = jobs.size() * 2;
            ready.reset(new std::atomic<int>[ready_cap]);
        }
        for (size_t i = 0; i < jobs.size(); ++i) ready[i].store(0, std::memory_order_relaxed);
        issued.store(0);
        ++gen;
        head.store((uint64_t) gen << 32 | (uint64_t) jobs.size() << 16, std::memory_order_release);
        cv.notify_all();
    }
    /// Job j's bytes, in a pinned buffer (waits for the copy).
    const uint8_t* wait(int j) {
        while (!ready[(size_t) j].load(std::memory_order_acquire)) std::this_thread::yield();
        return buf[j % kRing];
    }
    /// The launching thread queued job j's DMA on `copy`: its buffer is free once that is done.
    /// The launching thread queued job j's DMA on `copy`: its buffer is free once that is done.
    // SYCL port: the copy queue writes a sequence number into page-locked host memory after the DMA and the stager
    // thread polls it, instead of waiting on an event. A stager thread's host wait on the copy queue's events (the
    // CUDA form, cudaEventSynchronize) hung a lent-slot prompt in its first full chunk under the Level Zero v2 adapter
    // once the ring wrapped: no thread was waiting by then, the GPU sat busy - the event a host thread had waited on
    // was still in a queue's wait list (the v1 adapter and a 256-buffer ring, which never waits, both ran). 2026-10-01
    void issued_one(int j, dpct::queue_ptr copy) {
        const uint64_t s = ++issue_seq;
        want[(size_t) (j % kRing)] = s;
        if (strata::is_copy_engine_queue(copy)) {   // a copy-only command list runs no kernels (a fill is one): a copy of the value
            uint64_t* v = src_seq + (size_t) (s % (uint64_t) (4 * kRing));
            *v = s;
            copy->memcpy(done_seq + j % kRing, v, sizeof(uint64_t));
        } else {
            copy->fill<uint64_t>(done_seq + j % kRing, s, 1);
        }
        issued.store(j + 1, std::memory_order_release);
    }
    bool buffer_free(int b) const {
        return *(volatile const uint64_t*) (done_seq + b) >= want[(size_t) b];
    }
    /// No job is running after this (the end of a layer, or an early return in the middle of one).
    void finish() {
        head.store((uint64_t) gen << 32, std::memory_order_release);   // n = 0: nothing more to claim
        issued.store(1 << 30, std::memory_order_release);
        while (active.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
};

// multi-GPU: the peer GPU's share of a prompt chunk's experts.  Per MoE layer the primary copies its normed
// activations over P2P, the peer quantizes the rows routed to the experts it holds, runs the same MMQ products the
// primary would (gathered from its own slots), and copies the result rows back into the primary's Dm rows - the rows
// are laid out local-first, so the peer's are one contiguous block at the end.
struct PeerPrefill {
    core::PeerExperts* peer = nullptr;
    int dev = -1;
    int64_t cap_rows = 0, T_max = 0;
    dpct::queue_ptr s = &dpct::get_in_order_queue();
    dpct::event_ptr ev_in =
        nullptr; // on the primary: the activations and the row tables are ready
    dpct::event_ptr ev_done =
        nullptr; // on the peer: its rows have landed in the primary's Dm
    // multi-GPU: the result rows go back group by group on a second stream while the next group computes
    dpct::queue_ptr s_out = &dpct::get_in_order_queue();
    static constexpr int kGrpEv = NE / 16 + 2;
    dpct::event_ptr ev_grp[kGrpEv] = {};
    bool out_pending = false;       // s_out may still read Dm (the next layer's products wait for it)
    bool out_pipe = true;
    // multi-GPU COMPACT: the peer's per-row buffers hold one GROUP's rows instead of the layer's (a group
    // = up to 16 experts and at most G rows), the result rows double-buffered: ~0.9 GB -> ~0.26 GB on the helper
    // card at 8192-token chunks, and no row cap (every peer-held expert's rows go to the peer).
    // STRATA_PF_PEER_COMPACT=0: the layer-sized buffers (the A/B).
    bool compact = true;
    int64_t G = 0;
    void *Xq_g = nullptr, *Hq_g = nullptr;
    float *GU_g = nullptr, *H_g = nullptr, *Dm_b[2] = {};
    dpct::event_ptr ev_dm[2] = {};
    bool dm_live[2] = {};
    std::vector<std::pair<size_t, size_t>> groups;
    // multi-GPU PEER STREAMING: a share of the experts the primary would stream over ITS PCIe link in a
    // big chunk is streamed by the peer over its own link instead (into its own ring) and computed there - it halves
    // the primary's copy load and moves expert work to the card that idles through most of the MoE half.
    // STRATA_PF_PEER_STREAM = the share (0 = off), STRATA_PF_PEER_RING = its ring slots.
    double ps_frac = 0.0;
    int RP = 0;
    std::vector<uint8_t*> pstage;
    std::vector<dpct::event_ptr> pcopied, pused;
    std::vector<char> plive;
    dpct::queue_ptr s_cp = &dpct::get_in_order_queue();
    struct PsEntry { int32_t l, e; const uint8_t* blob; };
    std::vector<PsEntry> pseq;            // this chunk's peer-streamed experts, layer by layer in id order
    std::vector<size_t> pseq_start;
    std::vector<char> ps_flag;            // [layer * NE + e]: streamed by the peer in this chunk
    size_t p_issued = 0, pk = 0;
    int64_t ps_experts = 0;
    float *mixed = nullptr, *GU = nullptr, *H = nullptr, *Dm = nullptr;
    void *Xq = nullptr, *Hq = nullptr;
    int32_t *src = nullptr, *bounds = nullptr, *ident = nullptr;
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    std::unique_ptr<mmq::Context> ctx;
    mmq::Context* run_ctx = nullptr;      // the context its products run in: ctx, or a layer split helper stage's
    // Without P2P (a layer split's helper) the activations and the result rows go through MAPPED pinned host buffers: the device that needs them reads them with a copy kernel.  A copy-engine
    // upload would queue behind the expert blobs that device's streams already hold (up to a ring of them, ~70 us
    // each - the grouping tables' reason for mapped memory too), and cudaMemcpyPeerAsync would be staged by the
    // driver.  `back_at` / `back_rows`: the rows this layer sends back.
    bool p2p = true;
    // The row table and the group bounds go the same way: the primary's are pinned for its own device only (or
    // pageable), and a copy from them on the peer's stream would hold up the host until that stream is idle.
    float* host_x = nullptr;              // T x N activations (mapped, portable)
    float* host_rows = nullptr;           // the peer's result rows (mapped, portable)
    int32_t* host_src = nullptr;          // the peer's rows' tokens (mapped, portable)
    int32_t* host_bounds = nullptr;       // its group bounds (mapped, portable)
    size_t host_x_cap = 0, host_rows_cap = 0, host_src_cap = 0;   // elements
    static constexpr size_t kHostBounds = 2 * (NE + NE / 16 + 2) + 64;
    int64_t back_at = 0, back_rows = 0;
    std::vector<int32_t> bounds_host;
    std::vector<void*> owned;
    int64_t layers = 0, experts = 0, rows = 0, over_cap = 0;   // stats
    ~PeerPrefill() {
        if (dev < 0) return;
        int prev = 0;
        prev = dpct::get_current_device_id();
        /*
        DPCT1093: The "dev" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        dpct::select_device(dev);
        if (s) s->wait();
        ctx.reset();
        for (void *p : owned)
            DPCT_CHECK_ERROR(sycl::free(p, dpct::get_in_order_queue()));
        if (ev_done) dpct::destroy_event(ev_done);
        if (s_out) s_out->wait();
        for (dpct::event_ptr e : ev_grp) if (e) dpct::destroy_event(e);
        for (dpct::event_ptr e : ev_dm) if (e) dpct::destroy_event(e);
        if (s_cp) s_cp->wait();
        for (dpct::event_ptr e : pcopied) if (e) dpct::destroy_event(e);
        for (dpct::event_ptr e : pused) if (e) dpct::destroy_event(e);
        if (s_cp) dpct::get_current_device().destroy_queue(s_cp);
        if (s_out) dpct::get_current_device().destroy_queue(s_out);
        if (s) dpct::get_current_device().destroy_queue(s);
        /*
        DPCT1093: The "prev" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        dpct::select_device(prev);
        if (ev_in) dpct::destroy_event(ev_in);
        if (host_x) sycl::free(host_x, dpct::get_in_order_queue());
        if (host_rows) sycl::free(host_rows, dpct::get_in_order_queue());
        if (host_src) sycl::free(host_src, dpct::get_in_order_queue());
        if (host_bounds) sycl::free(host_bounds, dpct::get_in_order_queue());
    }
};

struct Prefill::Impl {
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    core::ExpertSource* src = nullptr;
    const core::ExpertCache* cache = nullptr;
    const int32_t* host_res = nullptr;
    int64_t T = 0, T_max = 0;
    bool borrowed = false;
    dpct::queue_ptr cs = &dpct::get_in_order_queue(),
                    copy = &dpct::get_in_order_queue();
    Gemm gemm;
    std::vector<void*> owned;
    // chunk buffers
    float *emb = nullptr, *R = nullptr, *xn = nullptr, *lo = nullptr, *gated = nullptr, *inj = nullptr;
    float* grs = nullptr;                    // F-1: the hyper-connection read's row scales (T x 4)
    uint16_t *xn16 = nullptr, *lo16 = nullptr;
    float* mixed = nullptr;
    uint16_t *mixed_bf = nullptr, *mixed_h = nullptr;
    // APR (prism.hadamard): per layer, this device's copies of the gate/up (n_embd) and down (n_ff) sign vectors
    // (nullptr: identity signs or not folded), resolved once in init so a prompt chunk reads a pointer
    std::vector<const float*> had_sx, had_sh;
    bool f16_io = false;   // HIP, STRATA_HIP_PROMPT_F16=1: this stage's 16-bit GEMMs run FP16 in and out (set in init)
    uint16_t *xn16_lo = nullptr, *lo16_lo = nullptr, *mixed_bf_lo = nullptr;   // bf16x2(): the BF16 GEMMs' low parts
    float* bo = nullptr;
    // GDN
    float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *hbuf = nullptr, *y = nullptr;
    uint16_t* y_h = nullptr;
    // QSA
    float *Kc = nullptr, *Vc = nullptr, *Qf = nullptr, *q = nullptr, *idx_raw = nullptr, *q_idx = nullptr, *attn = nullptr;
    uint16_t* attn_h = nullptr;
    int32_t* steps_dev = nullptr;
    std::vector<int32_t> steps_host;
    int32_t* sel_ids = nullptr;
    float* sel_scores = nullptr;          // [sel_batch, max_blocks]
    int64_t sel_batch = 256, max_blocks = 0;
    float* attn_scratch = nullptr;
    int64_t attn_batch = 32, cap = 0;
    // MoE
    float *logits = nullptr, *w = nullptr, *GU = nullptr, *Dm = nullptr, *sgate = nullptr, *sup = nullptr,
          *shared = nullptr, *sg = nullptr;
    int32_t *ids = nullptr, *slot_dev = nullptr, *src_dev = nullptr;
    uint16_t *Xs = nullptr, *Hh = nullptr, *sh_h = nullptr;
    // step 2b (MMQ): the activations quantized per layer, H in FP32 and its group's quantized rows, the identity
    // row map, the group bounds, the group buffers of gathered experts
    void *Xq = nullptr, *Hq = nullptr;
    float* H = nullptr;
    int32_t *ids_identity = nullptr, *bounds_dev = nullptr;
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    std::vector<int32_t> bounds_host;
    std::unique_ptr<mmq::Context> mmq_ctx;
    std::vector<int32_t> ids_host, slot_host, src_host, cnt, off;
    // The grouping tables in mapped pinned memory, [ids | slot | src] of T_max * K each, then the MMQ bounds: kernels
    // read and write them in place.  A cudaMemcpyAsync of them queues behind the expert blobs the copy stream already
    // holds (up to `ring` of them, ~70 us each), and the GPU idles meanwhile - measured 4.2 s of a 128K prompt's
    // 70 s at the default ring, 0.7 s with a 16-slot one.  STRATA_GROUP_COPY=1: the copies (the A/B arm).
    int32_t* grp_host = nullptr;
    int32_t* grp_dev = nullptr;          // its device alias
    size_t grp_n = 0, grp_tk = 0;        // int32s allocated; T_max * K (the offset of slot, and of src past it)
    uint16_t* dq_gu[strata::prefill::xmx::kMaxGroup] = {};
    uint16_t* dq_d[strata::prefill::xmx::kMaxGroup] = {};
    uint8_t* stage_dev[RING_MAX] = {};
    int ring = STAGE;                        // the slots of this layout's ring (ring_slots)
    // this layout's GU, H and Xq are the fused path's (moe_bufs' `fused`), so a layer that runs MMQ or the FP16 path may
    // only write stream_all_min() - 1 of its rows into them; fused_layout() keeps the two in step (#583, #954)
    bool fused_bufs = false;
    std::unique_ptr<Stager> stager;          // the unpinned experts' host copies (step 4)
    dpct::event_ptr copied[RING_MAX] = {}, used[RING_MAX] = {};
    bool stage_live[RING_MAX] = {};
    // the event that releases each ring slot: its own `used`, or - when an MMQ group is gathered in one launch - the
    // `used` of the last slot gathered with it, recorded once for all of them (a later record only waits longer)
    int used_of[RING_MAX] = {};
    // PLE
    float* ple_emb = nullptr;
    std::vector<float> ple_pageable[2];      // the fallback when no more RAM can be pinned
    float* ple_emb_host[2] = {};             // pinned, double-buffered: the next chunk's rows are read while this
    dpct::event_ptr ple_copied[2] =
        {}; // one runs; the event marks that buffer's upload done
    std::vector<uint32_t> ple_rows[2];
    // SYCL port: each buffer's upload is marked by a sequence number the queue writes into page-locked memory after
    // it, polled by the host - not a host wait on `ple_copied` (a host wait on a queue's event hung the prompt path
    // under the Level Zero v2 adapter once #374 moved it inside the layer loop; see Stager::issued_one)
    uint64_t* ple_done_seq = nullptr;
    uint64_t ple_want[2] = {};
    uint64_t ple_seq = 0;
    float* ple_norm = nullptr;
    uint8_t* region = nullptr;               // the attention/MoE scratch region (idle while the PLE block runs)
    uint64_t region_bytes = 0;
    PrefillStats* stats = nullptr;
    // KV streaming: one layer's whole K/V, staged from the host copy per layer and chunk (identity layout)
    strata::kernels::KvHostPools stage;
    int32_t* ident_table = nullptr;
    dpct::queue_ptr kv_copy = &dpct::get_in_order_queue();
    dpct::event_ptr kv_released = nullptr, kv_ready = nullptr;
    // layer split: the device, and the hand-off to the next stage (two pinned chunk buffers, used in turn)
    int device = -1;
    float* hand[2] = {};
    // C-4: the chunk's token ids on the device, for one batched embedding gather
    int32_t* tok_dev = nullptr;
    std::vector<int32_t> tok_host;
    std::unique_ptr<PeerPrefill> pp;         // multi-GPU: the peer GPU's expert share (set_peer)
    // layer split: the next stage's GPU as a stream-only peer (set_stage_helper); its buffers are that stage's own
    // prompt buffers, bound per prompt, and it takes the place of `pp` for a run it helps
    std::unique_ptr<PeerPrefill> help_pp;
};

namespace {
// the staging pool of a streamed session: every page of one layer (same sequence in init and bytes_needed)
// STRATA_KV_STAGE_OWN (A/B only): the staging pool gets its own allocation instead of borrowed expert slots, so a
// streamed run lends the prompt path exactly the slots a resident one does (a lent expert runs on the CPU, which
// rounds differently: without this an A/B compares two expert placements as well as two KV placements)
bool stage_own() { static const bool v = std::getenv("STRATA_KV_STAGE_OWN") != nullptr; return v; }
void take_stage(Alloc& o_borrowed, const core::SessionState& ss, const strata::kernels::QsaShapes& s,
                strata::kernels::KvHostPools& st, bool& ok) {
    const core::QsaState& q0 = ss.qsa_states[ss.qsa_primary()];
    if (q0.kv_mode != 1) return;
    if (stage_own() && o_borrowed.count_only) return;
    Alloc own;
    own.owned = o_borrowed.owned;
    Alloc& o = stage_own() ? own : o_borrowed;
    const size_t rows = (size_t) q0.n_pages * s.n_head_kv * s.page_size;
    if (q0.kv_hybrid) {   // K8V4: the three runs of kKvHybrid; pools_of() then reads as the hybrid pools (mode 3)
        st.k_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.k_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
        st.v_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
    } else if (q0.kv_q4) {
        st.k_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
        st.v_q4 = o.take<uint8_t>(rows * strata::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
    } else if (q0.kv_int8) {
        st.k_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.v_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.k_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
        st.v_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
    } else {
        st.k_pool = o.take<uint16_t>(rows * s.head_dim, ok);
        st.v_pool = o.take<uint16_t>(rows * s.head_dim, ok);
    }
}
strata::kernels::QsaAttnPools pools_of(const strata::kernels::KvHostPools& h, const int32_t* table) {
    strata::kernels::QsaAttnPools p;
    p.k_pool = h.k_pool; p.v_pool = h.v_pool; p.k_q = h.k_q; p.v_q = h.v_q; p.k_scale = h.k_scale; p.v_scale = h.v_scale;
    p.k_q4 = h.k_q4; p.v_q4 = h.v_q4;
    p.page_table = table;
    return p;
}
}  // namespace


// ------------------------------------------------------------------------------------------------ 151-b70: int8 or FP16 experts
// STRATA_PREFILL_MMQ=auto picks per group of experts from the measured cost on THIS GPU: at the first layer of a given expert
// format the int8 chain (quantize, gate/up, SwiGLU + quantize, down) and the FP16 chain (dequantize, two GEMMs, SwiGLU) are timed on
// synthetic experts (random blobs of the layer's format and shape) at 16, 32, 64, 128 and 256 rows per expert, and the crossover
// (the mean rows per expert where the FP16 chain becomes the faster one, interpolated on the cost ratio) is kept, with a flag for
// how clear the measurement was.  It is cached in a small text file (STRATA_CACHE_DIR, else ~/.cache/strata/i8-crossover.txt) keyed by
// device name, driver, formats and shape, and read back instead of measured when the key matches.
namespace {
struct I8Cross { double rows = 0; bool confident = false; bool measured = false; };

std::string i8_cache_file() {
    const char* d = std::getenv("STRATA_CACHE_DIR");
    std::string dir;
    if (d && d[0]) dir = d;
    else if (const char* h = std::getenv("HOME")) dir = std::string(h) + "/.cache/strata";
    else if (const char* a = std::getenv("LOCALAPPDATA")) dir = std::string(a) + "/strata";
    else return "";
    return dir + "/i8-crossover.txt";
}
bool i8_cache_read(const std::string& key, I8Cross& out) {
    const std::string path = i8_cache_file();
    if (path.empty()) return false;
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    char line[700];
    bool ok = false;
    while (std::fgets(line, sizeof line, f)) {
        char k[600]; double rows; int conf;
        if (std::sscanf(line, "%599[^\t]\t%lf\t%d", k, &rows, &conf) == 3 && key == k) { out.rows = rows; out.confident = conf != 0; ok = true; }
    }
    std::fclose(f);
    return ok;
}
void i8_cache_write(const std::string& key, const I8Cross& c) {
    const std::string path = i8_cache_file();
    if (path.empty()) return;
    const std::string dir = path.substr(0, path.rfind('/'));
    std::vector<std::string> keep;   // the other keys' lines
    if (std::FILE* f = std::fopen(path.c_str(), "r")) {
        char line[700];
        while (std::fgets(line, sizeof line, f)) {
            std::string s(line);
            if (s.compare(0, key.size() + 1, key + "\t") != 0) keep.push_back(s);
        }
        std::fclose(f);
    }
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        ::mkdir(dir.c_str(), 0755);
        f = std::fopen(path.c_str(), "w");
        if (!f) return;
    }
    for (auto& s : keep) std::fputs(s.c_str(), f);
    std::fprintf(f, "%s\t%.1f\t%d\n", key.c_str(), c.rows, c.confident ? 1 : 0);
    std::fclose(f);
}

template <class M>
I8Cross i8_calibrate(M& m, const strata::kernels::cpu::NativeFmt& f) {
    namespace i8 = strata::prefill::i8;
    sycl::queue& q = *m.cs;
    const std::string dev = q.get_device().template get_info<sycl::info::device::name>() + "|" +
                            q.get_device().template get_info<sycl::info::device::driver_version>();
    char keyb[600];
    std::snprintf(keyb, sizeof keyb, "%s|%d|%d|%lldx%lld", dev.c_str(), f.gu_type, f.d_type, (long long) f.n_ff, (long long) f.n_embd);
    const std::string key = keyb;
    I8Cross res;
    if (i8_cache_read(key, res)) {
        std::fprintf(stderr, "strata: int8 experts: crossover %.0f rows per expert (%s) from the cache for %s\n", res.rows,
                     res.confident ? "clear" : "not clear", key.c_str());
        return res;
    }
    constexpr int E = 8;
    const int rows_list[5] = {16, 32, 64, 128, 256};
    const int N = (int) f.n_embd, NF = (int) f.n_ff;
    double t8[5] = {}, t16[5] = {};
    const double facs[E] = {0.15, 0.35, 0.6, 0.85, 1.1, 1.4, 1.8, 2.45};
    double fac[E];
    std::vector<uint8_t*> blobs(E, nullptr);
    uint16_t *X = nullptr, *Hh = nullptr;
    float *GU = nullptr, *Dm = nullptr;
    bool ok = true;
    try {
        // rows per expert are uneven like a real group (a few big experts and many small ones): factors of the mean, mean 1
        for (int i = 0; i < E; ++i) { fac[i] = facs[i] / 1.0875; }
        size_t rmax = 0;
        for (int i = 0; i < E; ++i) rmax += (size_t) std::max(1.0, 256.0 * fac[i] + 0.5) + 8;   // rows of the largest sample (plus padding)
        for (int i = 0; i < E; ++i) blobs[i] = sycl::malloc_device<uint8_t>(f.bytes + 256, q);
        X = sycl::malloc_device<uint16_t>(rmax * N, q);
        Hh = sycl::malloc_device<uint16_t>(rmax * NF, q);
        GU = sycl::malloc_device<float>(rmax * 2 * NF, q);
        Dm = sycl::malloc_device<float>(rmax * N, q);
        for (int i = 0; i < E; ++i) ok = ok && blobs[i];
        ok = ok && X && Hh && GU && Dm;
        if (!ok) throw std::runtime_error("no VRAM for the measurement");
        // synthetic experts: random bytes with a sane fp16 scale at every block start (the cost does not depend on the values)
        std::vector<uint8_t> hb(f.bytes + 256);
        uint32_t s = 12345;
        auto rnd = [&] { s = s * 1664525u + 1013904223u; return s >> 8; };
        for (size_t i = 0; i < hb.size(); ++i) hb[i] = (uint8_t) rnd();
        const int gb = f.gu_type == 21 ? 110 : f.gu_type == 18 ? 98 : 82;
        for (size_t o = 0; o + 2 <= f.up_off * 2 && o < hb.size(); o += gb) { hb[o] = 0x00; hb[o + 1] = 0x2C; }
        for (size_t o = f.down_off; o + 2 <= f.bytes; o += 18) { hb[o] = 0x00; hb[o + 1] = 0x2C; }
        for (int i = 0; i < E; ++i) q.memcpy(blobs[i], hb.data(), f.bytes);
        std::vector<uint16_t> hx(rmax * N);
        for (auto& x : hx) x = (uint16_t) (0x3400 + (rnd() & 0x3ff));   // ~0.25..0.5, no zeros
        q.memcpy(X, hx.data(), hx.size() * 2);
        q.wait();
        auto timeit = [&](auto&& fn) {
            fn(); q.wait();
            const auto w0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < 0.02) { fn(); q.wait(); }
            double best = 1e30;
            for (int r = 0; r < 4; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                fn(); q.wait();
                best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            }
            return best;
        };
        for (int ri = 0; ri < 5; ++ri) {
            const int R = rows_list[ri];
            int32_t cnt[i8::kMaxGroup];
            for (int i = 0; i < E; ++i) cnt[i] = std::max(1, (int) (R * fac[i] + 0.5));
            i8::Layout lo;
            i8::make_layout(lo, cnt, E);
            i8::Workspace ws;
            if (!i8::workspace(lo, N, NF, ws, &q)) throw std::runtime_error("workspace");
            std::vector<const uint8_t*> pb(E);
            for (int i = 0; i < E; ++i) pb[i] = blobs[i];
            bool refused = false;
            t8[ri] = timeit([&] {
                i8::quantize_rows(X, lo, N, ws.xq, ws.dx, &q);
                if (!i8::gate_up(f.gu_type, pb.data(), f.up_off, NF, lo, N, ws.xq, ws.dx, GU, &q)) refused = true;
                i8::swiglu_quantize(GU, lo, NF, ws.hq, ws.dh, &q);
                if (!i8::down(f.d_type, pb.data(), f.down_off, N, lo, NF, ws.hq, ws.dh, Dm, &q)) refused = true;
            }) / E;
            if (refused) throw std::runtime_error("the int8 kernels refused this format");
            t16[ri] = timeit([&] {
                int64_t o = 0;
                for (int i = 0; i < E; ++i) {
                    const int qd = i % DQ;
                    strata::kernels::iq_dequant_expert_f16(f.gu_type, f.d_type, blobs[i], blobs[i] + f.up_off, blobs[i] + f.down_off, NF, N,
                                                           m.dq_gu[qd], m.dq_d[qd], &q);
                    m.gemm.f16(X + o * N, m.dq_gu[qd], GU + o * 2 * NF, cnt[i], 2 * NF, N);
                    strata::prefill::swiglu_interleaved(GU + o * 2 * NF, Hh + o * NF, cnt[i], &q);
                    m.gemm.f16(Hh + o * NF, m.dq_d[qd], Dm + o * N, cnt[i], N, NF);
                    o += cnt[i];
                }
            }) / E;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata: int8 experts: the crossover measurement failed (%s): the int8 path is used below 64 rows per expert\n", e.what());
        ok = false;
    }
    for (auto* b : blobs) if (b) sycl::free(b, q);
    if (X) sycl::free(X, q);
    if (Hh) sycl::free(Hh, q);
    if (GU) sycl::free(GU, q);
    if (Dm) sycl::free(Dm, q);
    if (!ok) { res.rows = 64; res.confident = false; return res; }
    // crossover: the first R where the FP16 chain is the faster one, interpolated in log(ratio) between the two bracketing sizes
    double ratio[5];
    for (int i = 0; i < 5; ++i) ratio[i] = t16[i] / t8[i];   // > 1: int8 faster
    int first = -1;
    for (int i = 0; i < 5; ++i) if (ratio[i] < 1.0) { first = i; break; }
    if (first < 0) res.rows = 512;                    // int8 wins everywhere measured
    else if (first == 0) res.rows = 0;                // FP16 wins even at 16 rows
    else {
        const double a = std::log(ratio[first - 1]), b = std::log(ratio[first]);
        const double t = a / (a - b);
        res.rows = std::exp(std::log((double) rows_list[first - 1]) + t * (std::log((double) rows_list[first]) - std::log((double) rows_list[first - 1])));
    }
    // clear: the ratio falls with the rows (within 5%) and the samples bracketing the crossover are at least 5% away from 1
    bool mono = true;
    for (int i = 1; i < 5; ++i) if (ratio[i] > ratio[i - 1] * 1.05) mono = false;
    res.confident = mono && (first < 0 ? ratio[4] > 1.05 : first == 0 ? ratio[0] < 0.95 : (ratio[first - 1] > 1.05 && ratio[first] < 0.95));
    res.measured = true;
    std::fprintf(stderr, "strata: int8 experts, measured here (%s): us per expert at 16/32/64/128/256 rows, int8 %.1f/%.1f/%.1f/%.1f/%.1f, FP16 %.1f/%.1f/%.1f/%.1f/%.1f -> crossover %.0f rows per expert (%s)\n",
                 key.c_str(), t8[0], t8[1], t8[2], t8[3], t8[4], t16[0], t16[1], t16[2], t16[3], t16[4], res.rows, res.confident ? "clear" : "not clear");
    i8_cache_write(key, res);
    return res;
}
}  // namespace
Prefill::Prefill() : impl_(new Impl) {}
Prefill::~Prefill() { release(); }

void Prefill::reset() {
    release();
    impl_.reset(new Impl);
    stats_ = PrefillStats{};
}

void Prefill::release() {
    if (!impl_) return;
    if (impl_->cs) impl_->cs->wait();
    if (impl_->copy) impl_->copy->wait();
    if (impl_->kv_copy) impl_->kv_copy->wait();
    for (int i = 0; i < RING_MAX; ++i) {
        if (impl_->copied[i]) dpct::destroy_event(impl_->copied[i]);
        if (impl_->used[i]) dpct::destroy_event(impl_->used[i]);
    }
    for (int b = 0; b < 2; ++b) {
        if (impl_->hand[b])
            sycl::free(impl_->hand[b], dpct::get_in_order_queue());
        if (impl_->ple_copied[b]) dpct::destroy_event(impl_->ple_copied[b]);
        if (b == 1 && impl_->ple_done_seq) { sycl::free(impl_->ple_done_seq, dpct::get_in_order_queue()); impl_->ple_done_seq = nullptr; }
        if (impl_->ple_emb_host[b] && impl_->ple_pageable[b].empty())
            sycl::free(impl_->ple_emb_host[b], dpct::get_in_order_queue());
    }
    if (impl_->copy) dpct::get_current_device().destroy_queue(impl_->copy);
    if (impl_->kv_copy) dpct::get_current_device().destroy_queue(
        impl_->kv_copy);
    if (impl_->kv_released) dpct::destroy_event(impl_->kv_released);
    if (impl_->kv_ready) dpct::destroy_event(impl_->kv_ready);
    if (impl_->grp_host)
        sycl::free(impl_->grp_host, dpct::get_in_order_queue());
    for (void *p : impl_->owned)
        DPCT_CHECK_ERROR(sycl::free(p, dpct::get_in_order_queue()));
}

namespace {
constexpr int64_t GEMM_SCRATCH = 32ll << 20;        // FP16 elements for the largest dequantized dense weight
constexpr size_t GEMM_WS = 32u << 20;               // cuBLAS workspace

// THE ATTENTION HALF AND THE MoE HALF SHARE THEIR BUFFERS.  A layer runs its attention (GDN or QSA), writes it back
// into the residual, and only then its MoE, so the three sets of scratch are never live at once: one region the size
// of the largest holds them all.  That is ~260 KB of the ~680 KB a prompt token cost - which is what lets a chunk
// grow (every expert is streamed once per chunk, so a bigger chunk streams fewer bytes per token).  The sizes are
// counted with the same `take` sequence `init` uses; a mismatch makes `init` fail with "do not fit", never overlap.
uint64_t gdn_set_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<float>(T * 2 * HV, ok); a.take<float>(T * HV, ok);
    a.take<float>(T * HV, ok); a.take<float>(T * C, ok); a.take<float>(T * ZV, ok);
    a.take<uint16_t>(T * (ZV + (pf_pad() ? ZV_PAD : 0)), ok);
    return a.used;
}
uint64_t qsa_set_bytes(size_t T, int64_t cap, int64_t max_blocks, int64_t sel_batch, int64_t attn_batch,
                       const strata::kernels::QsaShapes& s) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * 512, ok); a.take<float>(T * 512, ok); a.take<float>(T * 12288, ok); a.take<float>(T * ZV, ok);
    a.take<float>(T * 128, ok); a.take<float>(T * 512, ok); a.take<float>(T * ZV, ok);
    a.take<uint16_t>(T * (ZV + (pf_pad() ? ZV_PAD : 0)), ok);
    a.take<int32_t>(T * (size_t) cap, ok);
    a.take<float>((size_t) sel_batch * (size_t) max_blocks, ok);
    a.take<float>((size_t) attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(cap, s), ok);
    return a.used;
}
// Step 2b: which layers' experts go through MMQ (both weight types covered; the Strata Q2_0 pack always - its blob
// is converted to GGUF Q2_0 blocks on the gather), whether any layer keeps the FP16 path (IQ1_M), and the largest
// gate/up and down matrices a group buffer slot holds.  STRATA_PREFILL_MMQ=0: the FP16 path everywhere (the A/B).
constexpr int MMQ_GROUP = 16;                  // experts per MMQ launch (the gather is per expert, as blobs arrive)
// MMQ reads up to one 256-value tile past a matrix's last row when the row length is not a multiple of it (the down
// product: 640 values).  Those bytes meet zero activations, which is harmless only if they decode to finite numbers -
// llama.cpp zero-pads after every tensor, and so does a group buffer: this many zeroed bytes follow its last expert.
constexpr size_t MMQ_TAIL = 4096;
struct MmqPlan {
    bool any = false, fallback = true;
    std::vector<char> layer;                   // per layer: MMQ
    std::vector<char> fo;                      // per layer: no MMQ here but the native fused kernels cover its formats
                                               // (gfx11: UD-Q4_K_XL's Q4_K / Q5_K and Q5_1 / Q8_0 experts, STRATA_PF_FUSED=1)
    size_t gu_max = 0, d_max = 0;
};
const MmqPlan& mmq_plan() {
    static const MmqPlan plan = [] {
        MmqPlan p;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const char* env = std::getenv("STRATA_PREFILL_MMQ");
        const bool on = mmq::built() && (env == nullptr || std::atoi(env) != 0);
        const int64_t layers = lay.native ? (int64_t) lay.fmt.size() : lay.n_layers;
        p.layer.assign((size_t) std::max<int64_t>(layers, 0), 0);
        p.fo.assign(p.layer.size(), 0);
        p.fallback = !on || layers <= 0;
        for (int64_t l = 0; on && l < layers; ++l) {
            const int gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42, dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
            // #420: a tile on every GPU for these shapes (gate+up: 1280 rows, down: N rows), else the FP16 path
            if (!mmq::fits(gt, 1280) || !mmq::fits(dt, N)) {
                p.fallback = true;
                // the fused path's buffers (Xq, H) exist for it; a chunk it does not take (a small one) keeps the FP16 path
                if (on && lay.native && fused::native_supported(gt, dt)) { p.fo[(size_t) l] = 1; p.any = true; }
                continue;
            }
            p.layer[(size_t) l] = 1;
            p.any = true;
            p.gu_max = std::max(p.gu_max, mmq::matrix_bytes(gt, 1280, N));
            p.d_max = std::max(p.d_max, mmq::matrix_bytes(dt, N, 640));
        }
        return p;
    }();
    return plan;
}
// #136 P3: a layout whose chunks run the fused experts (STRATA_PF_FUSED=1, the Q2_0 pack, a streamed chunk of
// stream_all_min() tokens or more).  Its GU, H and Xq hold only the fused path's grouping tables, int8 H and per-token
// int8 activations, and Hq nothing: ~100 KB a token less than MMQ's FP32 GU / H and per-slot q8_1 rows, which is what
// lets a bigger chunk or ring fit in the slots the prompt path borrows.  A last chunk below stream_all_min() still runs
// MMQ in the same buffers, so each keeps MMQ's size for stream_all_min() - 1 tokens.  `src`: the layout streams experts
// (Prefill::init got an ExpertSource; without one no chunk takes the streamed walk, so no chunk is fused).
bool fused_layout(size_t T, bool src) {
    return src && fused_ring() && mmq_plan().any && ring_slots(T) > STAGE && (int64_t) T >= stream_all_min();
}
// The MoE buffers MMQ and the fused path share: GU and H in floats, Xq and Hq in bytes.  Without `fused` (the
// default): MMQ's, for T tokens.
struct MoeBufs { size_t gu, h, xq, hq; };
MoeBufs moe_bufs(size_t T, int64_t n_expert, bool fused) {
    if (!fused) return {T * K * 1280, T * K * 640, mmq::q8_bytes((int64_t) (T * K), N), mmq::q8_bytes((int64_t) (T * K), 640)};
    const size_t ts = (size_t) std::min<int64_t>((int64_t) T, stream_all_min() - 1);   // MMQ's last small chunk
    return {std::max(ts * K * 1280, (fused::group_bytes((int64_t) (T * K), (int) n_expert) + 3) / 4),
            std::max(ts * K * 640, (fused::act_bytes((int64_t) (T * K), 640) + 3) / 4),
            std::max(mmq::q8_bytes((int64_t) (ts * K), N), fused::act_bytes((int64_t) T, N)),
            mmq::q8_bytes((int64_t) (ts * K), 640)};
}
uint64_t moe_set_bytes(size_t T, int64_t n_expert, bool fused) {
    const MmqPlan& mp = mmq_plan();
    const MoeBufs mb = moe_bufs(T, n_expert, fused);
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * n_expert, ok); a.take<float>(T * K, ok); a.take<int32_t>(T * K, ok); a.take<int32_t>(T * K, ok);
    a.take<int32_t>(T * K, ok);
    if (mp.fallback) a.take<uint16_t>(T * K * N, ok);
    a.take<float>(mb.gu, ok);
    if (mp.fallback) a.take<uint16_t>(T * K * 640, ok);
    a.take<float>(T * K * N, ok); a.take<float>(T * 640, ok);
    a.take<float>(T * 640, ok); a.take<uint16_t>(T * 640, ok); a.take<float>(T * N, ok); a.take<float>(T, ok);
    if (mp.any) {
        a.take<uint8_t>(mb.xq, ok);
        a.take<float>(mb.h, ok);
        a.take<uint8_t>(mb.hq, ok);
    }
    return a.used;
}
}

bool Prefill::init(const core::WeightTable &wt, const core::ModelGeometry &g,
                   core::SessionState &ss, core::ExpertSource *src,
                   const core::ExpertCache *cache, const int32_t *host_res,
                   int64_t chunk, void *stream, std::string &err, void *borrow,
                   uint64_t borrow_bytes, bool borrow_spill) try {
    if (borrow_spill) {
        err = "prefill: a hybrid borrowed/owned layout (--prefill-head-alias) is not in the SYCL port";
        return false;
    }
    Impl& m = *impl_;
    m.wt = &wt; m.g = &g; m.ss = &ss; m.src = src; m.cache = cache; m.host_res = host_res;
    m.T = chunk; m.cs = strata::q_of(stream); m.stats = &stats_;
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K) {
        err = "prefill: geometry differs from the artifact's"; return false;
    }
    {   // APR: the sign vectors on this device, once (hadamard_device_signs takes a mutex and allocates its copy)
        const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
        m.had_sx.assign(lay.native ? lay.fmt.size() : 0, nullptr);
        m.had_sh.assign(m.had_sx.size(), nullptr);
        for (size_t l = 0; l < m.had_sx.size(); ++l) {
            if (lay.fmt[l].had_gu)
                m.had_sx[l] = strata::kernels::hadamard_device_signs(lay.fmt[l].gu_signs, (int) lay.fmt[l].n_embd);
            if (lay.fmt[l].had_d)
                m.had_sh[l] = strata::kernels::hadamard_device_signs(lay.fmt[l].d_signs, (int) lay.fmt[l].n_ff);
        }
    }
    m.device = dpct::get_current_device_id();
    if (static bool said_i8 = false; !said_i8 && i8_mode(m.cs) != 0) {
        said_i8 = true;
        std::fprintf(stderr, "strata: int8 prompt experts are on (%s); STRATA_PREFILL_MMQ=0 turns them off\n",
                     i8_env() == nullptr ? "Battlemage default, auto" : "STRATA_PREFILL_MMQ set");
    }
    if (stage_le_ < 0) stage_le_ = g.n_layers;
    if (stage_lb_ < 0 || stage_lb_ >= stage_le_ || stage_le_ > g.n_layers || (stage_le_ < g.n_layers) != (next_ != nullptr)) {
        err = "prefill: the stage's layer range is wrong";
        return false;
    }
    for (int b = 0; next_ != nullptr && b < 2; ++b)
        /*
        DPCT1048: The original value cudaHostAllocPortable is not meaningful
        in the migrated code and was removed or replaced with 0. You may need to
        check the migrated code.
        */
        if (!m.hand[b] &&
            DPCT_CHECK_ERROR(
                m.hand[b] = (float *)sycl::malloc_host(
                    (size_t)chunk * D * 4, dpct::get_in_order_queue())) != 0) {
            err = "prefill: the layer split's hand-off buffers";
            return false;
        }
    if (m.tok_dev == nullptr) {
        /*
        DPCT1000: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (const dpct::err0 e = DPCT_CHECK_ERROR(
                m.tok_dev = sycl::malloc_device<int32_t>(
                    (size_t)chunk, dpct::get_in_order_queue()));
            e != 0) {
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1001: The statement could not be removed.
            */
            err = std::string("prefill: the token id buffer (") +
                  dpct::get_error_string_dummy(e) + ")";
            return false;
        }
        m.owned.push_back(m.tok_dev);
        m.tok_host.resize((size_t) chunk);
    }
    /*
    DPCT1025: The SYCL queue is created ignoring the flag and priority
    options.
    */
    // 0.1.43: the expert copies on the GPU's copy engine when the device has one the SYCL runtime does not use itself
    // (the Level Zero v2 adapter runs every queue on the compute engine: ~0.8 s of a 4K prompt on a B70 is the compute
    // engine waiting for its own DMA).  nullptr: the ordinary in-order queue.  STRATA_COPY_ENGINE=0 turns it off.
    m.copy = strata::make_copy_engine_queue("the prompt path's expert");
    if (m.copy == nullptr &&
        DPCT_CHECK_ERROR(m.copy = dpct::get_current_device().create_queue(true)) != 0) {
        err = "prefill: copy stream"; return false;
    }
    const size_t T = (size_t) chunk;
    m.T_max = chunk;
    m.borrowed = borrow != nullptr;
    bool ok = true;
    // one-time: events, the stager, the host buffers (for the largest chunk), the identity page table
    for (int i = 0; i < ring_cap(); ++i) {
        if (DPCT_CHECK_ERROR(m.copied[i] = new sycl::event()) != 0) ok = false;
        if (DPCT_CHECK_ERROR(m.used[i] = new sycl::event()) != 0) ok = false;
    }
    if (!m.stager) {
        m.stager = std::make_unique<Stager>();
        const int hw = (int) std::thread::hardware_concurrency();
        const char* stv = std::getenv("STRATA_STAGER_THREADS");   // D-5: the host copy threads of unpinned blobs
        // A GGUF read in place (UD-Q4_K_XL beyond its RAM budget): most of a chunk's blobs are page faults on the
        // SSD, so the copies need many reads in flight - 32 threads and a 128-deep ring read a 4K chunk in 29 s
        // instead of 71 s on an RTX 5070 / NVMe PC (4 threads, 16 deep: the defaults, kept for every other source)
        bool files = false;
        for (int64_t l = 0; src != nullptr && !files && l < g.n_layers; ++l)
            for (int64_t e = 0; !files && e < g.n_expert; ++e) files = src->transient(l, e);
        const int threads = stv ? std::clamp(std::atoi(stv), 1, 32) : files ? 32 : std::max(2, std::min(4, hw / 4));
        if (files && std::getenv("STRATA_STAGER_RING") == nullptr) m.stager->kRing = 4 * threads;
        if (!m.stager->init((size_t) MAXBLOB(), threads)) ok = false;
    }
    m.steps_host.resize(T * strata::kernels::kStepCount);
    m.ids_host.resize(T * K); m.slot_host.resize(T * K); m.src_host.resize(T * K); m.cnt.resize(m.g->n_expert); m.off.resize(m.g->n_expert + 1);
    {
        const size_t need = 3 * T * K + (size_t) (2 * (m.g->n_expert + m.g->n_expert / MMQ_GROUP + 2));
        const char* gc = std::getenv("STRATA_GROUP_COPY");
        if (m.grp_n < need && !(gc && gc[0] == '1')) {
            if (m.grp_host) sycl::free(m.grp_host, dpct::get_in_order_queue());
            m.grp_host = m.grp_dev = nullptr;
            m.grp_n = m.grp_tk = 0;
            void *h = nullptr, *d = nullptr;
            if (!force_pageable() &&
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
                DPCT_CHECK_ERROR(h = (void *)sycl::malloc_host(
                                     need * 4, dpct::get_in_order_queue())) ==
                    0 &&
                DPCT_CHECK_ERROR(d = (void *)h) == 0) {
                m.grp_host = (int32_t*) h;
                m.grp_dev = (int32_t*) d;
                m.grp_n = need;
                m.grp_tk = T * K;
            } else {                              // the copies, as before
                if (h) sycl::free(h, dpct::get_in_order_queue());
                /*
                DPCT1026: The call to cudaGetLastError was removed because
                this functionality is redundant in SYCL.
                */
            }
        }
    }
    for (int b = 0; b < 2; ++b) {
        if (!m.ple_emb_host[b] &&
            /*
            DPCT1048: The original value cudaHostAllocDefault is not
            meaningful in the migrated code and was removed or replaced with 0.
            You may need to check the migrated code.
            */
            (force_pageable() ||
             DPCT_CHECK_ERROR(
                 m.ple_emb_host[b] = (float *)sycl::malloc_host(
                     (size_t)T * N * 4, dpct::get_in_order_queue())) != 0)) {
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            m.ple_pageable[b].resize(
                T * N); // pageable: the upload is staged before it returns
            m.ple_emb_host[b] = m.ple_pageable[b].data();
        }
        if (!m.ple_copied[b] &&
            DPCT_CHECK_ERROR(m.ple_copied[b] = new sycl::event()) != 0)
            ok = false;
        if (!m.ple_done_seq) {
            m.ple_done_seq = sycl::malloc_host<uint64_t>(2, dpct::get_in_order_queue());
            if (m.ple_done_seq) m.ple_done_seq[0] = m.ple_done_seq[1] = 0;
            else ok = false;
        }
        m.ple_rows[b].resize(T * strata::kernels::PLE_N_HEADS);
    }
    if (ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {   // KV streaming: the staging pool's identity page table
        const int64_t pages = ss.qsa_states[ss.qsa_primary()].n_pages;
        std::vector<int32_t> ident((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) ident[(size_t) i] = (int32_t) i;
        if (DPCT_CHECK_ERROR(
                m.ident_table = (int32_t *)sycl::malloc_device(
                    ident.size() * 4, dpct::get_in_order_queue())) != 0 ||
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                m.ident_table, ident.data(), ident.size() * 4).wait()) != 0)
            ok = false;
        else
            m.owned.push_back(m.ident_table);
    }
    if (!ok) { err = "prefill: host buffers or events for a chunk of " + std::to_string(chunk) + " tokens"; return false; }
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    {
        uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
        void* ws = o.take<uint8_t>(GEMM_WS, ok);
        if (!ok) { err = "prefill: GEMM scratch does not fit"; return false; }
        if (!m.gemm.init_external(stream, gs, GEMM_SCRATCH, ws, GEMM_WS, err)) return false;
        // this stage's device decides, once, here (init runs on the stage's own device): its gr_* kernels write the
        // image its GEMMs read, and set_act_f16 writes that device's symbol
        const core::OnDevice on_stage(m.device);
        m.f16_io = prompt_f16();
        m.gemm.set_f16_io(m.f16_io);
        set_act_f16(m.f16_io);
    }
    if (!carve(T, &o)) {
        size_t fb = 0, tb = 0;
        /*
        DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions
        for device information which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        dpct::get_current_device().get_memory_info(fb, tb);
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use
        the error codes. The cudaGetLastError function call was replaced with 0.
        You need to rewrite this code.
        */
        (void)0;
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit (" +
              std::to_string(fb >> 20) + " of " + std::to_string(tb >> 20) + " MiB free at the failure; " +
              std::to_string(o.used >> 20) + " MiB taken, the failing buffer wanted " +
              std::to_string(o.failed_bytes >> 20) + " MiB)";
        return false;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Every device buffer of a chunk of T tokens, from the Alloc `alloc` (after the GEMM scratch and workspace): `init`
// once, and `relayout` for a request's own chunk.  The order is `bytes_needed`'s.
bool Prefill::carve(size_t T, void* alloc) {
    Impl& m = *impl_;
    Alloc& o = *static_cast<Alloc*>(alloc);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    bool ok = true;
    m.emb = o.take<float>(T * N, ok); m.R = o.take<float>(T * D, ok);
    m.xn = gr_unfused() ? o.take<float>(T * D, ok) : nullptr;   // F-1: not needed (gr_mix_r reads R)
    m.grs = o.take<float>(T * HC, ok);
    m.xn16 = o.take<uint16_t>(T * (D + (hc_pad() ? XN_PAD : 0)), ok); m.lo = o.take<float>(T * LR, ok); m.lo16 = o.take<uint16_t>(T * LR, ok);
    m.gated = o.take<float>(T * D, ok); m.inj = o.take<float>(T * HC, ok);
    m.mixed = o.take<float>(T * N, ok); m.mixed_bf = o.take<uint16_t>(T * N, ok);
    m.mixed_h = o.take<uint16_t>(T * N, ok); m.bo = o.take<float>(T * N, ok);
    if (bf16x2_hc(m.f16_io)) { m.xn16_lo = o.take<uint16_t>(T * D, ok); m.lo16_lo = o.take<uint16_t>(T * LR, ok); }
    if (bf16x2(m.f16_io)) m.mixed_bf_lo = o.take<uint16_t>(T * N, ok);
    m.steps_dev = o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = core::qsa_shapes(g);
    m.cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    m.max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    {
        // one region for the attention half's and the MoE half's scratch (see gdn_set_bytes)
        const bool fz = fused_layout(T, m.src != nullptr);
        m.fused_bufs = fz;
        const MoeBufs mb = moe_bufs(T, m.g->n_expert, fz);
        const uint64_t region = std::max({gdn_set_bytes(T), qsa_set_bytes(T, m.cap, m.max_blocks, m.sel_batch,
                                                                           m.attn_batch, s), moe_set_bytes(T, m.g->n_expert, fz)});
        uint8_t* base = o.take<uint8_t>((size_t) region, ok);
        m.region = base;
        m.region_bytes = region;
        Alloc a;
        a.base = base; a.cap = region; a.owned = &m.owned;
        m.qkv = a.take<float>(T * C, ok); m.z = a.take<float>(T * ZV, ok); m.ab = a.take<float>(T * 2 * HV, ok);
        m.gate = a.take<float>(T * HV, ok); m.beta = a.take<float>(T * HV, ok); m.hbuf = a.take<float>(T * C, ok);
        m.y = a.take<float>(T * ZV, ok); m.y_h = a.take<uint16_t>(T * (ZV + (pf_pad() ? ZV_PAD : 0)), ok);
        Alloc b;
        b.base = base; b.cap = region; b.owned = &m.owned;
        m.Kc = b.take<float>(T * 512, ok); m.Vc = b.take<float>(T * 512, ok); m.Qf = b.take<float>(T * 12288, ok);
        m.q = b.take<float>(T * ZV, ok); m.idx_raw = b.take<float>(T * 128, ok); m.q_idx = b.take<float>(T * 512, ok);
        m.attn = b.take<float>(T * ZV, ok); m.attn_h = b.take<uint16_t>(T * (ZV + (pf_pad() ? ZV_PAD : 0)), ok);
        m.sel_ids = b.take<int32_t>(T * (size_t) m.cap, ok);
        m.sel_scores = b.take<float>((size_t) m.sel_batch * (size_t) m.max_blocks, ok);
        m.attn_scratch = b.take<float>((size_t) m.attn_batch * strata::kernels::qsa_decode_attn_scratch_floats(m.cap, s), ok);
        Alloc c;
        c.base = base; c.cap = region; c.owned = &m.owned;
        m.logits = c.take<float>(T * m.g->n_expert, ok); m.w = c.take<float>(T * K, ok); m.ids = c.take<int32_t>(T * K, ok);
        m.slot_dev = c.take<int32_t>(T * K, ok); m.src_dev = c.take<int32_t>(T * K, ok);
        const MmqPlan& mp = mmq_plan();
        m.Xs = mp.fallback ? c.take<uint16_t>(T * K * N, ok) : nullptr;
        m.GU = c.take<float>(mb.gu, ok);
        m.Hh = mp.fallback ? c.take<uint16_t>(T * K * 640, ok) : nullptr;
        m.Dm = c.take<float>(T * K * N, ok);
        m.sgate = c.take<float>(T * 640, ok); m.sup = c.take<float>(T * 640, ok); m.sh_h = c.take<uint16_t>(T * 640, ok);
        m.shared = c.take<float>(T * N, ok); m.sg = c.take<float>(T, ok);
        if (mp.any) {
            m.Xq = c.take<uint8_t>(mb.xq, ok);
            m.H = c.take<float>(mb.h, ok);
            m.Hq = c.take<uint8_t>(mb.hq, ok);
        }
        if (base == nullptr) ok = false;
    }
    for (int i = 0; i < dq_slots(); ++i) { m.dq_gu[i] = o.take<uint16_t>(1280 * 2560, ok); m.dq_d[i] = o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        m.ids_identity = o.take<int32_t>(T * K, ok);
        m.bounds_dev = o.take<int32_t>((size_t) (2 * (m.g->n_expert + m.g->n_expert / MMQ_GROUP + 2)), ok);
        m.grp_gu = o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        m.grp_d = o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL, ok);
        // (written at every run's start, not here: when serving, these are live expert-cache slots until a request
        // lends them - a write now would corrupt a resident expert)
        if (!m.mmq_ctx) m.mmq_ctx = std::make_unique<mmq::Context>();
    }
    m.ring = ring_slots(T);
    const int ring_n = std::max(m.ring, stage_slots());   // A770 port: STRATA_PF_XMX=2 holds a group of staged experts
    if (o.base == nullptr && ring_n > 0) {
        // OWNED buffers: the ring in ONE allocation.  384 separate 2.7 MiB cudaMallocs each round up to a 2 MiB page
        // (~1.3 MiB a slot, ~0.5 GiB in all) that no count ever saw.  A borrowed region keeps its per-slot layout (and
        // so its price, `bytes_needed`: the loans' slot counts do not move).
        uint8_t* ring_base = o.take<uint8_t>((size_t) ring_n * (size_t) MAXBLOB(), ok);
        for (int i = 0; ok && i < ring_n; ++i) {
            m.stage_dev[i] = ring_base + (size_t) i * (size_t) MAXBLOB();
            m.stage_live[i] = false;
            m.used_of[i] = i;
        }
    } else {
        for (int i = 0; i < ring_n; ++i) {
            m.stage_dev[i] = o.take<uint8_t>((size_t) MAXBLOB(), ok);
            m.stage_live[i] = false;                        // a new buffer: nothing of an earlier layout to wait for
            m.used_of[i] = i;
        }
    }
    m.ple_emb = o.take<float>(T * N, ok);
    m.ple_norm = o.take<float>((size_t) strata::kernels::NG_HC_DIM, ok);
    take_stage(o, ss, s, m.stage, ok);
    m.T = (int64_t) T;
    return ok;
}

bool Prefill::relayout(int64_t chunk, void *borrow, uint64_t borrow_bytes,
                       std::string &err) try {
    Impl& m = *impl_;
    if (!m.borrowed || borrow == nullptr || chunk <= 0 || chunk > m.T_max) {
        err = "prefill: relayout needs borrowed buffers and a chunk of at most " + std::to_string(m.T_max);
        return false;
    }
    if (DPCT_CHECK_ERROR(m.cs->wait()) != 0 ||
        DPCT_CHECK_ERROR(m.copy->wait()) != 0 ||
        (m.kv_copy && DPCT_CHECK_ERROR(m.kv_copy->wait()) != 0)) {
        err = "prefill: relayout: the stream failed";
        return false;
    }
    bool ok = true;
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    void* ws = o.take<uint8_t>(GEMM_WS, ok);
    if (ok) m.gemm.rebind(gs, GEMM_SCRATCH, ws, GEMM_WS);
    if (!ok || !carve((size_t) chunk, &o)) {
        size_t fb = 0, tb = 0;
        /*
        DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions
        for device information which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        dpct::get_current_device().get_memory_info(fb, tb);
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use
        the error codes. The cudaGetLastError function call was replaced with 0.
        You need to rewrite this code.
        */
        (void)0;
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit (" +
              std::to_string(fb >> 20) + " of " + std::to_string(tb >> 20) + " MiB free at the failure; " +
              std::to_string(o.used >> 20) + " MiB taken, the failing buffer wanted " +
              std::to_string(o.failed_bytes >> 20) + " MiB)";
        return false;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int64_t Prefill::chunk() const { return impl_->T; }

bool Prefill::draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                       std::string& err, const char** why) {
    Impl& m = *impl_;
    auto decline = [&](const char* reason) { if (why != nullptr) *why = reason; return false; };
    static const bool off = [] { const char* v = std::getenv("STRATA_MTP_BATCH"); return v != nullptr && v[0] == '0'; }();
    // A ring (KV streaming: the drafter's window, page p in slot p % n_slots over a host copy) takes the same appends
    // with its own page table and host copy, as a streamed main layer does; the cells written are those the window can
    // still reach (r0 below), which the ring holds, so no two of them share a slot.  STRATA_MTP_BATCH_RING=0: the
    // drafter's own pass for a ring (the A/B).
    static const bool ring_ok = [] { const char* v = std::getenv("STRATA_MTP_BATCH_RING"); return v == nullptr || v[0] != '0'; }();
    core::QsaState& st = mtp.kv_state_rw();
    if (off) return decline("STRATA_MTP_BATCH=0");
    if (n <= 0 || m.g == nullptr || m.region == nullptr) return decline("no prompt scratch region on this path");
    if (st.kv_mode != 0 && !(st.kv_mode == 2 && ring_ok))
        return decline(st.kv_mode == 2 ? "the drafter's K/V is a ring and STRATA_MTP_BATCH_RING=0" : "the drafter's K/V mode is not paged/ring");
    if (st.kv_hybrid) return decline("the drafter's K/V is hybrid (kv_hybrid)");
    if (mtp.device() != m.device) return decline("the drafter is on another device than this prefill path");
    const auto t0 = Clock::now();
    const core::ModelGeometry& g = *m.g;
    const int64_t Nn = g.n_embd, HCN = g.hc * g.n_embd, KV = g.n_head_kv * g.head_dim;
    constexpr int kQ8_0 = 8;   // GGML_TYPE_Q8_0
    const float* w_ne = mtp.tensor_f32("pre_fc_norm_embedding.weight");
    const void* w_fe = mtp.tensor_q8("fc_embedding.weight");
    const float* w_nh = mtp.tensor_f32("pre_fc_norm_hidden.weight");
    const void* w_fh = mtp.tensor_q8("fc_hidden.weight");
    const float* w_hn = mtp.tensor_f32("attn_hyper_connection.hc_norm.weight");
    const uint16_t* w_dn = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_down.weight");
    const uint16_t* w_up = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_up.weight");
    const void* w_k = mtp.tensor_q8("self_attn.k_proj.weight");
    const void* w_v = mtp.tensor_q8("self_attn.v_proj.weight");
    const float* w_kn = mtp.tensor_f32("self_attn.k_norm.weight");
    if (!w_ne || !w_fe || !w_nh || !w_fh || !w_hn || !w_dn || !w_up || !w_k || !w_v || !w_kn)
        return decline("a drafter tensor is missing (Q8_0 projections / norms)");
    const core::NativeEmbed* nemb = core::native_embed();
    const core::WeightRef* wemb = nemb ? nullptr : m.wt->find("token_embd.weight");
    if (!nemb && (wemb == nullptr || wemb->codebook_iq4nl || wemb->ne0 != g.n_embd || wemb->group_elems <= 0 ||
                  (wemb->code_bits != 2 && wemb->code_bits != 4 && wemb->code_bits != 8)))
        return decline("the token embedding is not gatherable on this path");
    // the cells the drafter's window can still reach
    const int64_t r0 = std::max<int64_t>(0, mtp.first_needed() - cell0);
    if (r0 >= n) return true;
    // per row: emb/e2 (N), en16 (N half), hn/h2/Rm/gated (HCN), hn16/xn16 (HCN half), lo (LR) + lo16, grs, mixed (N) +
    // mixed_h, K and V (KV each), the token id
    // E-9: the drafter's Q8_0 matrices through Q8_1 x Q8_0 MMQ - its own pass's integer dot products (mmvq), so
    // its K/V stay close to what the drafter computes itself; STRATA_MTP_BATCH_F16=1: FP16 GEMMs (the A/B)
    static const bool f16_only = [] { const char* v = std::getenv("STRATA_MTP_BATCH_F16"); return v && v[0] == '1'; }();
    const bool q8 = !f16_only && mmq::built() && mmq::fits(kQ8_0, Nn) && mmq::fits(kQ8_0, KV);   // #420
    const uint64_t per_row = 4 * (2 * Nn + 4 * HCN + LR + HC + Nn + 2 * KV + 1) + 2 * (Nn + 2 * HCN + LR + Nn) + 64 +
                             (q8 ? (uint64_t) mmq::q8_bytes(g.hc, Nn) + 4 * g.hc : 0);
    // SYCL port: B is the batch buffers' capacity, not the row count (each batch uses nb <= B rows), so a short
    // prompt gets 64-row buffers instead of the per-6-token fallback (19 tokens: 106 ms there, a few ms here)
    const int64_t cap = (int64_t) (m.region_bytes / per_row) & ~(int64_t) 63;
    if (cap < 64) return decline("too little scratch for a 64-row batch");
    int64_t B = std::min<int64_t>(std::max<int64_t>(n - r0, 64), cap);
    if (st.kv_mode == 2)   // #453: a ring: one batch's cells must not share a slot (a batch can straddle one page more)
        B = std::min<int64_t>(B, ((st.n_slots - 1) * strata::kernels::qsa_real_shapes().page_size) & ~(int64_t) 63);
    if (B < 64) return decline("too little scratch for a 64-row batch");
    uint8_t* q = m.region;
    auto carve = [&](size_t bytes) { void* p = q; q += (bytes + 255) & ~(size_t) 255; return p; };
    float* emb = (float*) carve((size_t) B * Nn * 4);
    float* e2 = (float*) carve((size_t) B * Nn * 4);
    uint16_t* en16 = (uint16_t*) carve((size_t) B * Nn * 2);
    float* hn = (float*) carve((size_t) B * HCN * 4);
    float* h2 = (float*) carve((size_t) B * HCN * 4);
    float* Rm = (float*) carve((size_t) B * HCN * 4);
    float* gated = (float*) carve((size_t) B * HCN * 4);
    uint16_t* hn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    uint16_t* xn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    float* lo = (float*) carve((size_t) B * LR * 4);
    uint16_t* lo16 = (uint16_t*) carve((size_t) B * LR * 2);
    float* grs = (float*) carve((size_t) B * HC * 4);
    float* mixed = (float*) carve((size_t) B * Nn * 4);
    uint16_t* mixed_h = (uint16_t*) carve((size_t) B * Nn * 2);
    float* Kc = (float*) carve((size_t) B * KV * 4);
    float* Vc = (float*) carve((size_t) B * KV * 4);
    int32_t* tok = (int32_t*) carve((size_t) B * 4);
    void* xq = q8 ? carve(mmq::q8_bytes(B * g.hc, Nn)) : nullptr;
    int32_t* ident = q8 ? (int32_t*) carve((size_t) B * g.hc * 4) : nullptr;
    int32_t* bnd = q8 ? (int32_t*) carve(16) : nullptr;
    if ((uint64_t) (q - m.region) > m.region_bytes) return decline("the batch buffers do not fit the scratch region");
    static const bool timing = std::getenv("STRATA_DRAFT_TIMING") != nullptr;   // debug: where this pass's time goes
    if (timing) m.cs->wait();
    const auto ti0 = Clock::now();
    if (!mtp.idle(err)) return false;   // the drafter's own stream (its graph uploads) before this writes its K/V
    const double ms_idle = ms_since(ti0);
    const auto tl0 = Clock::now();
    strata::kernels::QsaShapes s = core::qsa_shapes(g);
    std::vector<int32_t> tk((size_t) B);
    if (q8) {
        if (!m.mmq_ctx) m.mmq_ctx = std::make_unique<mmq::Context>();
        mmq::iota(ident, B * g.hc, m.cs);
    }
    // y[rows, n_out] = x[rows, k] . w^T for a Q8_0 matrix: MMQ from the FP32 rows (bounds slot 0: rows, 1: rows*hc)
    auto proj = [&](const float* x, const uint16_t* x16, const void* w, float* y, int64_t rows, int64_t n_out,
                    int64_t k, int slot) {
        if (!q8) { m.gemm.native(x16, kQ8_0, w, y, rows, n_out, k); return; }
        mmq::quantize(x, nullptr, xq, kQ8_0, k, k, rows, m.cs);
        mmq::Product p;
        p.w = w; p.type = kQ8_0; p.w_rows = n_out; p.w_cols = k; p.expert_bytes = mmq::matrix_bytes(kQ8_0, n_out, k);
        p.n = 1; p.xq = xq; p.bounds = bnd + 2 * slot; p.ids = ident; p.total_rows = rows; p.max_rows = rows;
        p.dst = y; p.ld_dst = n_out;
        m.mmq_ctx->run(p, m.cs);
    };
    for (int64_t b0 = r0; b0 < n; b0 += B) {
        const int64_t nb = std::min(B, n - b0), c0 = cell0 + b0;
        for (int64_t i = 0; i < nb; ++i) tk[(size_t) i] = next_tokens[b0 + i];
        const int32_t bh[4] = {0, (int32_t) nb, 0, (int32_t) (nb * g.hc)};
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        if (q8 && DPCT_CHECK_ERROR(m.cs->memcpy(bnd, bh, sizeof bh)) != 0) {
            err = "prefill: the draft bounds' upload failed";
            return false;
        }
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        if (DPCT_CHECK_ERROR(m.cs->memcpy(tok, tk.data(), (size_t)nb * 4)) !=
            0) {
            err = "prefill: the draft tokens' upload failed";
            return false;
        }
        // the input branches: the next token's embedding, and this cell's final residual rows
        if (nemb) {
            nemb->gather_dev(tok, nb, emb, m.cs);
        } else {
            const auto* codes = (const uint8_t*) wemb->data;
            const auto* scales = (const float*) (codes + wemb->codes_bytes);
            const auto* offsets = wemb->has_offset ? (const float*) (codes + wemb->codes_bytes + wemb->scales_bytes)
                                                   : nullptr;
            strata::kernels::embedding_gather_dev(codes, scales, offsets, tok, (int) nb, wemb->ne0, wemb->code_bits,
                                                  wemb->code_bias, wemb->group_elems,
                                                  (uint64_t) (wemb->ne0 / (8 / wemb->code_bits)),
                                                  (uint64_t) (wemb->ne0 / wemb->group_elems), emb, m.cs);
        }
        rms_rows(emb, w_ne, nb, Nn, Nn, EPS, m.cs);
        if (!q8) to_f16(emb, en16, nb * Nn, m.cs);
        proj(emb, en16, w_fe, e2, nb, Nn, Nn, 0);
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        m.cs->memcpy(hn, R_rows + (size_t)b0 * HCN, (size_t)nb * HCN * 4);
        if (mtp.hnorm_per_stream())   // --mtp-hnorm stream: as the drafter's own pass (mtp.cpp)
            strata::kernels::native_qsa_rms_norm_grouped(hn, w_nh, hn, (int) Nn, (int) g.hc, (int) (nb * g.hc), EPS, m.cs);
        else
            rms_rows(hn, w_nh, nb, HCN, HCN, EPS, m.cs);
        if (!q8) to_f16(hn, hn16, nb * HCN, m.cs);
        proj(hn, hn16, w_fh, h2, nb * g.hc, Nn, Nn, 1);   // every stream through fc_hidden
        strata::kernels::add_streams_broadcast(h2, e2, Rm, Nn, (int) g.hc, (int) nb, m.cs);
        // the attention hyper-connection's read (its mixed input only: this pass writes nothing back)
        gr_norm_rs(Rm, w_hn, EPS, grs, xn16, nb, m.cs);
        m.gemm.bf16(xn16, w_dn, lo, nb, LR, HCN);
        gr_silu(lo, lo16, nb, m.cs);
        m.gemm.bf16(lo16, w_up, gated, nb, HCN, LR);
        gr_mix_r(Rm, grs, w_hn, gated, mixed, nullptr, nb, m.cs, mixed_h);
        // K and V into the drafter's cache, as the prompt path's QSA layers append theirs
        proj(mixed, mixed_h, w_k, Kc, nb, KV, Nn, 0);
        proj(mixed, mixed_h, w_v, Vc, nb, KV, Nn, 0);
        rms_rows(Kc, w_kn, nb * g.n_head_kv, g.head_dim, g.head_dim, EPS, m.cs);
        rope(Kc, nb, g.n_head_kv, g.head_dim, KV, c0, strata::kernels::rope_scaling(), m.cs);
        if (st.kv_rot) {   // rotated as the drafter's own decode stores them (mtp.cpp)
            strata::kernels::fwht256_inplace_cuda(Kc, nb * g.n_head_kv, m.cs);
            strata::kernels::fwht256_inplace_cuda(Vc, nb * g.n_head_kv, m.cs);
        }
        if (st.kv_q4) {
            strata::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, c0, nb, Kc, Vc, s, m.cs, &st.host);
        } else {
            kv_append(Kc, Vc, nb, c0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs, &st.host);
        }
    }
    if (DPCT_CHECK_ERROR(m.cs->wait()) != 0) {
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
        err = std::string("prefill: the draft layer's K/V: ") +
              dpct::get_error_string_dummy(0);
        return false;
    }
    mtp.ms_prefill += ms_since(t0);
    if (timing)
        std::fprintf(stderr, "strata draft kv: %lld cells from %lld (first needed %lld), batch %lld: drafter idle %.1f ms, "
                     "the batches %.1f ms, all %.1f ms\n", (long long) n, (long long) cell0, (long long) (cell0 + r0),
                     (long long) B, ms_idle, ms_since(tl0), ms_since(t0));
    return true;
}
namespace {
// Layer split: STRATA_PREFILL_HELP=1 lends the idle stage's GPU (opt-in: the helped rows are computed in another MMQ
// grouping and round differently from the default, like STRATA_SPLIT_OWN). Unset or 0: every stage's prompt experts on
// its own GPU, as before.
bool split_help_env() {
    static const bool v = [] { const char* e = std::getenv("STRATA_PREFILL_HELP"); return e != nullptr && std::atoi(e) != 0; }();
    return v;
}
// The share of a stage's streamed experts the idle stage's GPU takes for a one-chunk prompt of T tokens (0: none).
// The streamed bytes are the same at any T; the hand-over (the activations there, the rows back) grows with T and is
// paid per layer whatever the share.  Measured on 2x RTX 3090 (UD-Q4_K_XL, PCIe 4.0 x16 each, no P2P), fixed shares
// 0.2 / 0.3 / 0.4 against none: 1.5K tokens 749 / 820 / 908 (none 675), 3K 1335 / 1437 / 1328 (1262), 4K 1622 / 1620 /
// 1440 (1609), 6K 1811 / 1660 / 1471 (2038), 8K 1895 / 1804 / 1663 (2230) - the best share falls with T and from ~4K
// none pays, which this rule follows (0.41 at 1.5K, 0.32 at 3K, off from ~3.3K).
// STRATA_PREFILL_HELP_FRAC sets the share for every T.
double split_help_frac(int64_t T) {
    static const double fixed = [] {
        const char* e = std::getenv("STRATA_PREFILL_HELP_FRAC");
        return e ? std::clamp(std::atof(e), 0.0, 1.0) : -1.0;
    }();
    if (fixed >= 0.0) return fixed;
    const double f = std::min(0.45, 0.5 - (double) T / 16384.0);
    return f >= 0.3 ? f : 0.0;
}
}  // namespace

bool Prefill::set_stage_helper(Prefill *helper, std::string &err) try {
    Impl& m = *impl_;
    m.help_pp.reset();
    if (helper == nullptr || helper == this || !split_help_env()) return true;
    const Impl& h = *helper->impl_;
    if (m.device < 0 || h.device < 0 || h.device == m.device || !mmq_plan().any) return true;
    auto pp = std::make_unique<PeerPrefill>();
    pp->peer = nullptr;   // stream-only: it holds no experts, it streams a share of this stage's
    pp->compact = true;
    pp->p2p = false;      // the mapped host route, whether or not the cards could reach each other (measured there)
    pp->out_pipe = true;
    bool ok;
    {
        const core::OnDevice on(m.device);
        ok = DPCT_CHECK_ERROR(pp->ev_in = new sycl::event()) == 0;
    }
    int prev = 0;
    prev = dpct::get_current_device_id();
    pp->dev = h.device;
    /*
    DPCT1093: The "pp->dev" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    dpct::select_device(pp->dev);
    /*
    DPCT1025: The SYCL queue is created ignoring the flag and priority
    options.
    */
    ok = ok &&
         DPCT_CHECK_ERROR(
             pp->s = dpct::get_current_device().create_queue(true)) == 0 &&
         /*
         DPCT1025: The SYCL queue is created ignoring the flag and priority
         options.
         */
         DPCT_CHECK_ERROR(
             pp->s_out = dpct::get_current_device().create_queue(true)) == 0 &&
         /*
         DPCT1025: The SYCL queue is created ignoring the flag and priority
         options.
         */
         DPCT_CHECK_ERROR(
             pp->s_cp = dpct::get_current_device().create_queue(true)) == 0 &&
         DPCT_CHECK_ERROR(pp->ev_done = new sycl::event()) == 0;
    for (int i = 0; ok && i < PeerPrefill::kGrpEv; ++i)
        ok = DPCT_CHECK_ERROR(pp->ev_grp[i] = new sycl::event()) == 0;
    for (int b = 0; ok && b < 2; ++b)
        ok = DPCT_CHECK_ERROR(pp->ev_dm[b] = new sycl::event()) == 0;
    // the ring's events; its slots are the helper stage's own ring, bound per prompt (bind_stage_helper)
    pp->pcopied.assign((size_t) kSplitHelpRing, nullptr);
    pp->pused.assign((size_t) kSplitHelpRing, nullptr);
    for (int i = 0; ok && i < kSplitHelpRing; ++i)
        ok =
            DPCT_CHECK_ERROR(pp->pcopied[(size_t)i] = new sycl::event()) == 0 &&
            DPCT_CHECK_ERROR(pp->pused[(size_t)i] = new sycl::event()) == 0;
    /*
    DPCT1093: The "prev" device may be not the one intended for use. Adjust
    the selected device if needed.
    */
    dpct::select_device(prev);
    if (!ok) { err = "prefill: the layer split's prompt help (streams and events)"; return false; }
    helper_ = helper;
    m.help_pp = std::move(pp);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// A one-chunk prompt of T tokens on a layer split: binds the stream-only peer to the helper stage's prompt buffers
// (that stage idles until this one hands its rows over).  False when it cannot help this prompt.
bool Prefill::bind_stage_helper(int64_t T) {
    Impl& m = *impl_;
    if (!m.help_pp || helper_ == nullptr || m.pp) return false;
    const double frac = split_help_frac(T);
    if (frac <= 0.0 || T < stream_all_min() || m.src == nullptr || m.ring <= STAGE || fused_ring()) return false;
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    for (int64_t l = stage_lb_; l < stage_le_; ++l) {
        if (!(mmq_plan().any && mmq_plan().layer[(size_t) l])) return false;   // the peer computes on MMQ only
        if (lay.native && fused::native_supported(lay.fmt[(size_t) l].gu_type, lay.fmt[(size_t) l].d_type)) return false;
    }
    Impl& h = *helper_->impl_;
    // the helper's buffers must be laid out for this chunk (the serve path lends every stage's at once)
    if (h.T < T || h.mixed == nullptr || h.Xq == nullptr || h.Hq == nullptr || h.GU == nullptr || h.H == nullptr ||
        h.Dm == nullptr || h.src_dev == nullptr || h.ids_identity == nullptr || h.bounds_dev == nullptr ||
        h.grp_gu == nullptr || h.grp_d == nullptr || !h.mmq_ctx || h.ring <= STAGE)
        return false;
    PeerPrefill& P = *m.help_pp;
    P.T_max = T;
    P.cap_rows = T * K;
    P.G = T;                                  // one expert never has more rows than the chunk has tokens
    P.mixed = h.mixed;
    P.Xq_g = h.Xq;                            // the helper's per-layer buffers hold T * K rows: a group's T fit
    P.Hq_g = h.Hq;
    P.GU_g = h.GU;
    P.H_g = h.H;
    P.Dm_b[0] = h.Dm;
    P.Dm_b[1] = h.Dm + (size_t) T * N;
    P.src = h.src_dev;
    P.ident = h.ids_identity;
    P.bounds = h.bounds_dev;
    P.grp_gu = h.grp_gu;
    P.grp_d = h.grp_d;
    P.run_ctx = h.mmq_ctx.get();
    P.RP = std::min(h.ring, kSplitHelpRing);
    P.pstage.assign(h.stage_dev, h.stage_dev + P.RP);
    P.plive.assign((size_t) P.RP, 0);
    P.dm_live[0] = P.dm_live[1] = false;
    P.out_pending = false;
    P.ps_frac = frac;
    if (!P.p2p) {   // grown once to the largest chunk it helped
        auto grow = [](auto *&p, size_t &cap, size_t n) {
            try {
        if (cap >= n) return true;
            if (p) sycl::free(p, dpct::get_in_order_queue());
            p = nullptr;
            cap = 0;
            /*
            DPCT1048: The original value cudaHostAllocPortable is not
            meaningful in the migrated code and was removed or replaced with 0.
            You may need to check the migrated code.
            */
            /*
            DPCT1048: The original value cudaHostAllocMapped is not
            meaningful in the migrated code and was removed or replaced with 0.
            You may need to check the migrated code.
            */
            if (DPCT_CHECK_ERROR(
                    p = (std::remove_reference_t<decltype(p)>)sycl::malloc_host(
                        n * sizeof(*p), dpct::get_in_order_queue())) != 0) {
                /*
                DPCT1026: The call to cudaGetLastError was removed because
                this functionality is redundant in SYCL.
                */
                p = nullptr;
                return false;
            }
            cap = n;
            return true;
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        size_t bounds_cap = P.host_bounds ? PeerPrefill::kHostBounds : 0;
        if (!grow(P.host_x, P.host_x_cap, (size_t) T * N) || !grow(P.host_rows, P.host_rows_cap, (size_t) T * K * N) ||
            !grow(P.host_src, P.host_src_cap, (size_t) T * K) || !grow(P.host_bounds, bounds_cap, PeerPrefill::kHostBounds))
            return false;
    }
    const core::OnDevice on(P.dev);
    mmq::iota(P.ident, T * K, P.s);           // the helper's row table: a refill may have overwritten its slots
    return true;
}

bool Prefill::set_peer(core::PeerExperts *peer, int64_t cap_rows,
                       std::string &err) try {
    Impl& m = *impl_;
    if (peer == nullptr || !peer->valid()) { m.pp.reset(); return true; }
    if (!peer->p2p()) { err = "prefill peer: the two GPUs cannot access each other (no P2P)"; return false; }
    if (!mmq_plan().any) { err = "prefill peer: needs the MMQ prompt path"; return false; }
    auto pp = std::make_unique<PeerPrefill>();
    pp->peer = peer;
    pp->T_max = m.T_max;
    pp->cap_rows = std::max<int64_t>(1, std::min<int64_t>(cap_rows, m.T_max * K));
    const int64_t R = pp->cap_rows;
    const MmqPlan& mp = mmq_plan();
    if (DPCT_CHECK_ERROR(pp->ev_in = new sycl::event()) != 0) {
        err = "prefill peer: event"; return false;
    }
    int prev = 0;
    prev = dpct::get_current_device_id();
    pp->dev = peer->device();
    /*
    DPCT1093: The "pp->dev" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    dpct::select_device(pp->dev);
    /*
    DPCT1025: The SYCL queue is created ignoring the flag and priority
    options.
    */
    bool ok = DPCT_CHECK_ERROR(
                  pp->s = dpct::get_current_device().create_queue(true)) == 0 &&
              DPCT_CHECK_ERROR(pp->ev_done = new sycl::event()) == 0;
    {
        const char* v = std::getenv("STRATA_PF_PEER_OUT_PIPE");   // =0: one copy after the last group (the A/B)
        pp->out_pipe = (v == nullptr || std::atoi(v) != 0) || pp->compact;   // compact always pipes
    }
    if (ok && pp->out_pipe) {
        /*
        DPCT1025: The SYCL queue is created ignoring the flag and priority
        options.
        */
        ok =
            DPCT_CHECK_ERROR(
                pp->s_out = dpct::get_current_device().create_queue(true)) == 0;
        for (int i = 0; ok && i < PeerPrefill::kGrpEv; ++i)
            ok = DPCT_CHECK_ERROR(pp->ev_grp[i] = new sycl::event()) == 0;
    }
    auto take = [&](size_t bytes) -> void * {
        try {
    void *p = nullptr;
        if (!ok ||
            DPCT_CHECK_ERROR(p = (void *)strata::malloc_device_guarded(
                                 bytes, dpct::get_in_order_queue())) != 0) {
            ok = false; return nullptr;
        }
        pp->owned.push_back(p);
        return p;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    };
    pp->mixed = (float*) take((size_t) m.T_max * N * 4);
    {
        const char* v = std::getenv("STRATA_PF_PEER_COMPACT");
        pp->compact = v == nullptr || std::atoi(v) != 0;
    }
    if (pp->compact) {
        pp->cap_rows = m.T_max * K;   // no row cap: only the row tables grow with it
        pp->G = m.T_max;              // one expert never has more rows than the chunk has tokens
        const int64_t Gr = pp->G;
        pp->Xq_g = take(mmq::q8_bytes(Gr, N));
        pp->Hq_g = take(mmq::q8_bytes(Gr, 640));
        pp->GU_g = (float*) take((size_t) Gr * 1280 * 4);
        pp->H_g = (float*) take((size_t) Gr * 640 * 4);
        pp->Dm_b[0] = (float*) take((size_t) Gr * N * 4);
        pp->Dm_b[1] = (float*) take((size_t) Gr * N * 4);
        for (int b = 0; b < 2 && ok; ++b)
            ok = DPCT_CHECK_ERROR(pp->ev_dm[b] = new sycl::event()) == 0;
        const char* fs = std::getenv("STRATA_PF_PEER_STREAM");
        pp->ps_frac = fs ? std::atof(fs) : 0.35;   // measured: 0.25-0.5 all ~1950-1970 at 32K, 0.35 best
        if (pp->ps_frac > 0.0 && !mp.fallback && m.T_max >= stream_all_min()) {
            pp->RP = 48;
            if (const char* pr = std::getenv("STRATA_PF_PEER_RING"); pr != nullptr) pp->RP = std::atoi(pr);
            pp->pstage.assign((size_t) pp->RP, nullptr);
            pp->pcopied.assign((size_t) pp->RP, nullptr);
            pp->pused.assign((size_t) pp->RP, nullptr);
            pp->plive.assign((size_t) pp->RP, 0);
            /*
            DPCT1025: The SYCL queue is created ignoring the flag and
            priority options.
            */
            ok = ok && DPCT_CHECK_ERROR(
                           pp->s_cp = dpct::get_current_device().create_queue(
                               true)) == 0;
            for (int i = 0; ok && i < pp->RP; ++i) {
                pp->pstage[(size_t) i] = (uint8_t*) take((size_t) MAXBLOB());
                ok = ok &&
                     DPCT_CHECK_ERROR(pp->pcopied[(size_t)i] =
                                          new sycl::event()) == 0 &&
                     DPCT_CHECK_ERROR(pp->pused[(size_t)i] =
                                          new sycl::event()) == 0;
            }
        } else {
            pp->ps_frac = 0.0;
        }
    } else {
        pp->Xq = take(mmq::q8_bytes(R, N));
        pp->Hq = take(mmq::q8_bytes(R, 640));
        pp->GU = (float*) take((size_t) R * 1280 * 4);
        pp->H = (float*) take((size_t) R * 640 * 4);
        pp->Dm = (float*) take((size_t) R * N * 4);
    }
    const int64_t Rt = pp->cap_rows;
    pp->src = (int32_t*) take((size_t) Rt * 4);
    pp->ident = (int32_t*) take((size_t) Rt * 4);
    pp->bounds = (int32_t*) take((size_t) (2 * (NE + NE / MMQ_GROUP + 2)) * 4);
    pp->grp_gu = (uint8_t*) take(MMQ_GROUP * mp.gu_max + MMQ_TAIL);
    pp->grp_d = (uint8_t*) take(MMQ_GROUP * mp.d_max + MMQ_TAIL);
    if (ok) {
        pp->ctx = std::make_unique<mmq::Context>();
        pp->run_ctx = pp->ctx.get();
        mmq::iota(pp->ident, pp->cap_rows, pp->s);
        ok = DPCT_CHECK_ERROR(pp->s->wait()) == 0;
    }
    size_t fb = 0, tb = 0;
    /*
    DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    dpct::get_current_device().get_memory_info(fb, tb);
    /*
    DPCT1093: The "prev" device may be not the one intended for use. Adjust
    the selected device if needed.
    */
    dpct::select_device(prev);
    if (!ok) { err = "prefill peer: the peer's buffers do not fit (raise --peer-reserve-mib or lower --peer-prefill-rows)"; return false; }
    std::fprintf(stderr, "strata prefill: peer GPU %d computes its experts' rows of each prompt chunk (up to %lld rows per "
                         "layer%s); %zu MiB left free on it\n", pp->dev, (long long) pp->cap_rows,
                 pp->compact ? (pp->ps_frac > 0.0 ? (", compact group buffers, streams " + std::to_string((int) (pp->ps_frac * 100 + 0.5)) +
                                                    "% of the primary's streamed experts through a " + std::to_string(pp->RP) + "-slot ring").c_str()
                                                 : ", compact group buffers") : "", fb >> 20);
    m.pp = std::move(pp);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void Prefill::set_pinned_share(double share) { g_pinned_share = share; }
void set_nonresident_share(double share) { g_nonres_share = share; }   // SYCL port (see ring_slots)
void Prefill::set_ring_override(int slots) { g_ring_override = slots > 0 ? slots : 0; }
void Prefill::set_ring_budget(int slots, int64_t small_max) {
    g_ring_budget = slots > 0 ? slots : 0;
    g_ring_small_max = small_max > 0 ? small_max : 0;
}
double Prefill::pinned_share() { return g_pinned_share; }
int64_t Prefill::stream_all_min_tokens() { return stream_all_min(); }

uint64_t Prefill::bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk, bool src) {
    return bytes_needed_impl(g, ss, chunk, false, src);
}

// What `init` really allocates when the prompt path owns its buffers (no loan): every cudaMalloc rounds up to a 2 MiB
// page, and the ring is one allocation (carve).  `bytes_needed` stays the borrowed region's sum.
uint64_t Prefill::bytes_needed_owned(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                      bool src) {
    return bytes_needed_impl(g, ss, chunk, true, src);
}

uint64_t Prefill::bytes_needed_impl(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                    bool owned_pages, bool src) {
    // the same allocation sequence as `init`, counted
    const size_t T = (size_t) chunk;
    bool ok = true;
    Alloc o;
    o.count_only = true;
    if (owned_pages) o.granule = 2ull << 20;
    o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    o.take<uint8_t>(GEMM_WS, ok);
    auto f = [&](size_t n) { o.take<float>(n, ok); };
    // `carve`'s order, buffer for buffer: emb, R, xn, grs, xn16, lo, lo16, gated, inj, mixed, mixed_bf, mixed_h,
    // bo.  This counted `xn` unconditionally (carve takes it only under STRATA_GR_UNFUSED) and never counted
    // `grs`.  Net over-count T*(D-HC)*4 bytes: 42 MB at a 1024-token chunk, 252 MB (48 Q8_0 slots) at 6144 - the
    // prompt path was told it had less room than it did.  Safe - the direction is over-estimating, and `take`
    // still bounds-checks - but it under-sizes every loan, so every chunk the scan picks is one step smaller.
    if (ring_bytes_on()) {
        f(T * N); f(T * D);
        if (gr_unfused()) f(T * D);
        f(T * HC);
    } else {   // STRATA_RING_BYTES=0: 0.1.39's count
        f(T * N); f(T * D); f(T * D);
    }
    o.take<uint16_t>(T * (D + (hc_pad() ? XN_PAD : 0)), ok); f(T * LR); o.take<uint16_t>(T * LR, ok);
    f(T * D); f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok); f(T * N);
    const bool f16_io = prompt_f16();   // the current device's mode (the stage's), as Prefill::init will decide it
    if (bf16x2_hc(f16_io)) { o.take<uint16_t>(T * D, ok); o.take<uint16_t>(T * LR, ok); }
    if (bf16x2(f16_io)) o.take<uint16_t>(T * N, ok);
    o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = core::qsa_shapes(g);
    const int64_t cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    const int64_t max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    o.take<uint8_t>((size_t) std::max({gdn_set_bytes(T), qsa_set_bytes(T, cap, max_blocks, 256, 32, s),
                                       moe_set_bytes(T, g.n_expert, fused_layout(T, src))}), ok);
    for (int i = 0; i < dq_slots(); ++i) { o.take<uint16_t>(1280 * 2560, ok); o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        o.take<int32_t>(T * K, ok);
        o.take<int32_t>((size_t) (2 * (g.n_expert + g.n_expert / MMQ_GROUP + 2)), ok);
        o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL, ok);
    }
    const int ring_n = std::max(ring_slots(T), stage_slots());
    if (owned_pages) {
        if (ring_n > 0) o.take<uint8_t>((size_t) ring_n * (size_t) MAXBLOB(), ok);   // one allocation
    } else {
        for (int i = 0; i < ring_n; ++i) o.take<uint8_t>((size_t) MAXBLOB(), ok);
    }
    f(T * N);
    f((size_t) strata::kernels::NG_HC_DIM);
    strata::kernels::KvHostPools stage;
    take_stage(o, ss, s, stage, ok);
    return o.used + (8u << 20);   // alignment slack
}

uint64_t Prefill::bytes_needed_no_ring(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                        bool src) {
    return bytes_needed(g, ss, chunk, src) - (uint64_t) ring_slots((size_t) chunk) * (uint64_t) MAXBLOB();
}

int64_t Prefill::ring_default_slots() {
    const int r = g_pinned_share >= 0.9 ? (fused_ring() ? 1024 : 384) : 96;
    return (int64_t) std::min(r, ring_cap());
}

int64_t Prefill::ring_cap_for(int64_t old_chunk) {
    // 0.1.39b (#583, measured on the RTX 5070): giving up ring slots for a bigger chunk paid where 0.1.39's ring held
    // the chunk at 4096 or less (IQ3_XXS 32K prompts +14% to +26%: 4096/384 -> 6656/56) and lost where 0.1.39 already
    // read 6144-token chunks (the Coder: 6144/384 -> 7936/199, 32K -12%).  From kKeepRingChunk on the ring keeps its
    // 0.1.39 size and the scan only looks for a bigger chunk next to it (IQ3_XXS unpinned 6144/96 -> 6912/96: +9%).
    constexpr int64_t kKeepRingChunk = 6144;
    return old_chunk >= kKeepRingChunk ? ring_default_slots() : ring_max_slots();
}

// the unpinned arm keeps its measured 96 (the PR's rule; the byte budget would have been ~49 slots on IQ3_S)
int64_t Prefill::ring_max_slots() {
    return g_pinned_share >= 0.9 ? (int64_t) ring_budget_slots() : (int64_t) std::min(96, ring_cap());
}

int64_t Prefill::ring_slots_for(int64_t chunk) { return ring_slots((size_t) chunk); }

bool Prefill::ring_bytes_enabled() { return ring_bytes_on(); }

namespace {

const core::WeightRef* need(const core::LayerView& v, const char* suffix, std::string& err) {
    const core::WeightRef* r = v.get(suffix);
    if (!r) err = v.name(suffix) + " is missing";
    return r;
}
bool native_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                 std::string& err, int64_t ldy = 0, int64_t ldx = 0) {
    if (!w->native_data) { err = "prefill: " + name + " has no native GGUF blocks (run with --native)"; return false; }
    gm.native(X, w->native_type, w->native_data, Y, T, w->ne1, w->ne0, ldy, 0.0f, ldx);
    return true;
}
bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
               std::string& err, int64_t ldy = 0, const uint16_t* X_lo = nullptr, int64_t ldx = 0) {
    if (w->kind != core::WeightKind::Bf16InF32 || !w->data) { err = "prefill: " + name + " is not a resident BF16 tensor"; return false; }
    gm.bf16(X, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy, 0.0f, ldx);
    if (X_lo) gm.bf16(X_lo, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy, 1.0f);
    return true;
}

}  // namespace

namespace {
// STRATA_PREFILL_TIMING=1: the prompt path's GPU time by phase.  Events are recorded on the compute stream in order;
// the time between two consecutive marks is charged to the phase of the first, so a gap where the GPU waits (for the
// host's expert grouping, or for an expert's copy) lands on the phase that was waiting.  Events are reused: the marks
// are folded at every MoE layer's host sync, after which all of them have completed.
enum PfPhase { kPfStart, kPfHc, kPfGdn, kPfQsa, kPfQsaIdx, kPfQsaSel, kPfQsaAttn, kPfRouter, kPfHostGroup, kPfGather,
               kPfWaitCopy, kPfDequant, kPfGemmGU, kPfGemmD, kPfCombine, kPfPle, kPfKvStage, kPfGdnConv, kPfGdnRec, kPfGdnOut,
               kPfCount };
const char* const kPfNames[kPfCount] = {"embed+steps", "hc read", "gdn", "qsa proj", "qsa indexer", "qsa select",
                                        "qsa attn", "router+shared", "host grouping", "gather", "wait copy", "dequant",
                                        "gemm gate/up", "gemm down", "combine", "ple", "kv stage", "gdn conv+gates",
                                        "gdn recurrence", "gdn out proj"};
struct PfTimer {
    bool on = std::getenv("STRATA_PREFILL_TIMING") != nullptr;
    std::vector<dpct::event_ptr> ev;
    std::vector<int> ph;
    size_t used = 0;
    double ms[kPfCount] = {};
    // STRATA_PREFILL_SYNC=1 (debug): wait for the GPU at every mark and log the phase that just finished, so a hang
    // names the stage that never completes
    bool sync = std::getenv("STRATA_PREFILL_SYNC") != nullptr;
    long long n_sync = 0;
    void mark(int phase, dpct::queue_ptr s) {
        if (sync) {
            const auto t0 = std::chrono::steady_clock::now();
            s->wait();
            std::fprintf(stderr, "strata prefill sync: mark %lld phase %s done (waited %.1f ms)\n", ++n_sync,
                         phase < kPfCount ? kPfNames[phase] : "?",
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        if (!on) return;
        if (used == ev.size()) {
            dpct::event_ptr e = nullptr;
            e = new sycl::event();
            ev.push_back(e);
            ph.push_back(0);
        }
        ph[used] = phase;
        dpct::sync_barrier(ev[used], s);
        ++used;
    }
    // every recorded mark has completed (the stream was synchronized): charge the gaps, keep the last mark
    void fold() try {
        if (!on || used < 2) return;
        for (size_t i = 0; i + 1 < used; ++i) {
            float t = 0.0f;
            if (DPCT_CHECK_ERROR(
                    t = (ev[i + 1]
                             ->get_profiling_info<
                                 sycl::info::event_profiling::command_end>() -
                         ev[i]
                             ->get_profiling_info<sycl::info::event_profiling::
                                                      command_start>()) /
                        1000000.0f) == 0) ms[ph[i]] += t;
        }
        std::swap(ev[0], ev[used - 1]);
        std::swap(ph[0], ph[used - 1]);
        used = 1;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    ~PfTimer() {
        for (dpct::event_ptr e : ev) DPCT_CHECK_ERROR(dpct::destroy_event(e));
    }
};
// multi-GPU: the peer's own timeline (STRATA_PREFILL_TIMING): marks on the peer stream, folded with the primary's
enum PePhase { kPeIdle, kPeMoeIn, kPeMoeGemm, kPeMoeOut, kPeCount };
const char* const kPeNames[kPeCount] = {"idle", "moe in", "moe gemm", "moe out"};
struct PeTimer {
    bool on = std::getenv("STRATA_PREFILL_TIMING") != nullptr;
    int dev = -1;
    std::vector<dpct::event_ptr> ev;
    std::vector<int> ph;
    size_t used = 0;
    double ms[kPeCount] = {};
    void mark(int phase, dpct::queue_ptr s) { // the peer device is current
        if (!on) return;
        if (used == ev.size()) {
            dpct::event_ptr e = nullptr;
            e = new sycl::event();
            ev.push_back(e);
            ph.push_back(0);
        }
        ph[used] = phase;
        dpct::sync_barrier(ev[used], s);
        ++used;
    }
    void fold() try { // every mark has completed
        if (!on || used < 2) return;
        for (size_t i = 0; i + 1 < used; ++i) {
            float t = 0.0f;
            if (DPCT_CHECK_ERROR(
                    t = (ev[i + 1]
                             ->get_profiling_info<
                                 sycl::info::event_profiling::command_end>() -
                         ev[i]
                             ->get_profiling_info<sycl::info::event_profiling::
                                                      command_start>()) /
                        1000000.0f) == 0) ms[ph[i]] += t;
        }
        std::swap(ev[0], ev[used - 1]);
        std::swap(ph[0], ph[used - 1]);
        used = 1;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    ~PeTimer() {
        if (dev < 0) return;
        int prev = 0;
        prev = dpct::get_current_device_id();
        /*
        DPCT1093: The "dev" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        dpct::select_device(dev);
        for (dpct::event_ptr e : ev) DPCT_CHECK_ERROR(dpct::destroy_event(e));
        /*
        DPCT1093: The "prev" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        dpct::select_device(prev);
    }
};
}  // namespace

bool Prefill::run_impl(const int64_t *tokens, int64_t n, int64_t pos0,
                       std::string &err) try {
    err.clear();
    Impl& m = *impl_;
    const core::OnDevice on_device(m.device);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    const auto t_start = Clock::now();
    const int64_t LB = stage_lb_, LE = stage_le_;
    // The direct successor's future lives on the Prefill object. Intermediate
    // stages therefore do not drain the complete remaining GPU chain here.
    double host_sync_ms = 0, host_chunk_ms = 0, host_setup_ms = 0;   // STRATA_PREFILL_TIMING: the host's share
    double grp_wait_ms = 0, grp_cpu_ms = 0;   // the per-layer grouping: the drain wait, the host's loops
    strata::kernels::QsaShapes s = core::qsa_shapes(g);
    // read once (STRATA_KV_PREFETCH=1/0: the next streamed layer's KV prefix is staged on its own stream)
    static const bool kv_prefetch_on = [] {
        const char* e = std::getenv("STRATA_KV_PREFETCH");
        return e ? std::atoi(e) != 0 : false;
    }();
    const bool kv_prefetch = kv_prefetch_on;
    if (kv_prefetch) {
        /*
        DPCT1025: The SYCL queue is created ignoring the flag and priority
        options.
        */
        if ((!m.kv_copy &&
             DPCT_CHECK_ERROR(
                 m.kv_copy = dpct::get_current_device().create_queue(true)) !=
                 0) ||
            (!m.kv_released &&
             DPCT_CHECK_ERROR(m.kv_released = new sycl::event()) != 0) ||
            (!m.kv_ready &&
             DPCT_CHECK_ERROR(m.kv_ready = new sycl::event()) != 0)) {
            err = "prefill: KV prefetch stream/events";
            return false;
        }
    }
    // An early return must drain DMA before the caller refills the borrowed expert slots.
    struct KvDrain {
        dpct::queue_ptr stream;
        ~KvDrain() { if (stream) stream->wait(); }
    } kv_drain{kv_prefetch ? m.kv_copy : &dpct::get_in_order_queue()};
    int64_t kv_prefetches = 0;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
    // layer split: a prompt of one chunk runs the stages one after the other, so the next stage's GPU idles while
    // this one reads; it streams and computes a share of this stage's experts then (set_stage_helper), in the
    // place of a --peer-device peer, and gives the stage back its own buffers when the run ends
    const bool single_chunk = hand_in_ == nullptr ? n <= m.T : single_chunk_;
    const bool helped = single_chunk && bind_stage_helper(n);
    if (helped) std::swap(m.pp, m.help_pp);
    struct HelpScope {
        Impl& m;
        bool on;
        ~HelpScope() {
            if (!on) return;
            const core::OnDevice d(m.pp->dev);   // its work is done before the stage hands over; settle its streams
            m.pp->s_cp->wait();
            m.pp->s_out->wait();
            m.pp->s->wait();
            std::swap(m.pp, m.help_pp);
        }
    } help_scope{m, helped};
    PfTimer pt;
    PeTimer pe;
    if (m.pp) pe.dev = m.pp->dev; else pe.on = false;
    const dpct::queue_ptr cs = strata::q_of(m.cs);
    // the MMQ row table lives in the borrowed cache slots, which the refill after a prompt overwrites with experts:
    // write it again for every prompt (a layout is reused as long as the chunk and the slots are the same)
    if (m.ids_identity != nullptr) mmq::iota(m.ids_identity, m.T * K, m.cs);
    // The PLE rows of a chunk are read from the model file on the host (an SSD read per missed row): the chunk
    // after this one is read on a thread while the GPU runs this one, into the other of two buffers.  The rows
    // depend only on the tokens (the two before a position name its n-grams), so this is the same data.
    const bool ple_on = ss.ple.ready() && LB <= 1 && 1 < LE;
    // the PLE block batched over the chunk: the pinned postops and a BF16 or GGUF-native key (else token by token);
    // STRATA_PLE_BATCH=0 keeps the per-token block (the A/B)
    static const bool ple_batch_env = [] {
        const char* v = std::getenv("STRATA_PLE_BATCH");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const bool ple_batch = ple_on && ple_batch_env && strata::kernels::ple_native_postops_enabled() &&
                           (ss.ple.w.key_bf16 != nullptr || ss.ple.w.key_native_data != nullptr) &&
                           m.region_bytes / ((uint64_t) (3 * strata::kernels::NG_HC_DIM + N + 4) * 4 + (uint64_t) N * 2 + 4096) >= 64;
    const int32_t prev0[2] = {prev[0], prev[1]};
    // SYCL port: a short first chunk so the GPU starts while the rest of the prompt's PLE rows are still being read
    // (27k random 4 KB reads per 2k tokens, 330 ms at the drive's ~85k IOPS, otherwise all before the first kernel).
    // STRATA_PREFILL_FIRST=<tokens> (0: off), default 256 when the prompt is longer than twice that.
    // Every chunk streams the experts the cache does not hold (a 4K prompt touches nearly all 512 per layer), over PCIe: with
    // part of the model in the host mirror a short first chunk costs a whole extra pass of that stream (an Arc Pro B70 on
    // IQ3_S: a 4,095-token prompt 784 -> 1,002 tok/s without it).  So the default is 256 only when the cache holds every
    // expert; STRATA_PREFILL_FIRST=<tokens> decides explicitly.
    const bool all_cached = m.cache != nullptr && m.g != nullptr && m.cache->slots() >= m.g->n_layers * m.g->n_expert;
    static const char* const first_env = std::getenv("STRATA_PREFILL_FIRST");
    const int64_t first_chunk = first_env ? std::atoll(first_env) : (all_cached ? 256 : 0);
    auto chunk_len = [&](int64_t c0) {
        if (c0 == 0 && first_chunk > 0 && first_chunk < m.T && n > 2 * first_chunk) return first_chunk;
        return std::min(m.T, n - c0);
    };
    auto ple_gather = [&m, &ss, tokens, n, prev0, &chunk_len](int64_t c0, int buf, std::string& e) -> bool {
        const int64_t T = chunk_len(c0);
        auto at = [&](int64_t i) { return i < 2 ? prev0[i] : (int32_t) tokens[i - 2]; };   // prev0, then the tokens
        int32_t pv[2] = {at(c0), at(c0 + 1)};
        for (int64_t t = 0; t < T; ++t) {
            const int32_t tok = (int32_t) tokens[c0 + t];
            strata::kernels::ngram_rows(&tok, pv, 1, ss.ple.consts,
                                        m.ple_rows[buf].data() + t * strata::kernels::PLE_N_HEADS);
            pv[0] = pv[1];
            pv[1] = tok;
        }
        return ss.ple.table->gather_batch(m.ple_rows[buf].data(), (size_t) T, m.ple_emb_host[buf], e);
    };
    std::string ple_next_err;
    std::future<bool> ple_next;             // declared after everything it reads: an early return waits for it
    int ple_buf = 0;

    for (int64_t c0 = 0; c0 < n; c0 += chunk_len(c0)) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        if (std::getenv("STRATA_TRACE")) { std::fprintf(stderr, "strata trace: prompt chunk %lld of %lld\n", (long long) c0, (long long) n); std::fflush(stderr); }
        const int64_t T = chunk_len(c0), p0 = pos0 + c0;
        core::progress_at("reading the prompt (batched): preparing the chunk from token", p0);   // #251
        ++stats_.chunks;
        pt.mark(kPfStart, cs);
        const auto tsetup = Clock::now();
        auto tlap = tsetup;   // STRATA_PREFILL_TIMING=1: where the host's chunk setup goes
        auto lap = [&](const char* what) {
            if (pt.on) { std::fprintf(stderr, "strata prefill timing: setup %s %.1f ms\n", what, ms_since(tlap)); tlap = Clock::now(); }
        };
        if (pt.on) { m.cs->wait(); lap("work queued before the chunk"); }
        // ---- embeddings, broadcast to the four streams - or, in a later stage of a layer split, the rows the
        // previous stage handed on
        if (hand_in_ != nullptr) {
            /*
            DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            if (DPCT_CHECK_ERROR(m.cs->memcpy(m.R, hand_in_ + (size_t)c0 * D,
                                              (size_t)T * D * 4)) != 0) {
                err = "prefill: the layer split's hand-off upload failed";
                return false;
            }
        }
        // C-4: the whole chunk's rows in one gather (the same per-element arithmetic as the per-token path, so the
        // same bits); a chunk with picture rows, or a token outside the table, takes the per-token path
        bool batched = hand_in_ == nullptr && m.tok_dev != nullptr;
        const core::NativeEmbed* nemb = core::native_embed();
        const core::WeightRef* wemb = nemb ? nullptr : m.wt->find("token_embd.weight");
        if (batched && nemb == nullptr &&
            (wemb == nullptr || wemb->codebook_iq4nl || wemb->ne0 != g.n_embd || wemb->group_elems <= 0 ||
             (wemb->code_bits != 2 && wemb->code_bits != 4 && wemb->code_bits != 8)))
            batched = false;
        for (int64_t t = 0; batched && t < T; ++t) {
            const int64_t tok = tokens[c0 + t];
            if ((embd_rows && embd_rows[p0 + t]) || tok < 0 || (wemb && tok >= wemb->ne1)) batched = false;
            else m.tok_host[(size_t) t] = (int32_t) tok;
        }
        if (batched) {
            /*
            DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            if (DPCT_CHECK_ERROR(m.cs->memcpy(m.tok_dev, m.tok_host.data(),
                                              (size_t)T * sizeof(int32_t))) !=
                0) {
                err = "prefill: the token id upload failed";
                return false;
            }
            lap("token ids uploaded");
            if (nemb) {
                nemb->gather_dev(m.tok_dev, T, m.emb, m.cs);
                if (pt.on) { m.cs->wait(); lap("embedding gather (waited)"); }
            } else {
                const auto* codes = (const uint8_t*) wemb->data;
                const auto* scales = (const float*) (codes + wemb->codes_bytes);
                const auto* offsets = wemb->has_offset ? (const float*) (codes + wemb->codes_bytes + wemb->scales_bytes)
                                                       : nullptr;
                strata::kernels::embedding_gather_dev(codes, scales, offsets, m.tok_dev, (int) T, wemb->ne0,
                                                      wemb->code_bits, wemb->code_bias, wemb->group_elems,
                                                      (uint64_t) (wemb->ne0 / (8 / wemb->code_bits)),
                                                      (uint64_t) (wemb->ne0 / wemb->group_elems), m.emb, m.cs);
            }
        }
        for (int64_t t = 0; hand_in_ == nullptr && !batched && t < T; ++t) {
            const float* row = embd_rows ? embd_rows[p0 + t] : nullptr;
            if (row) {
                /*
                DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy
                API. While the origin API might be synchronous, it depends on
                the type of operand memory, so you may need to call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                if (DPCT_CHECK_ERROR(
                        m.cs->memcpy(m.emb + t * N, row, (size_t)N * 4)) != 0) {
                    err = "prefill: the image embedding upload failed";
                    return false;
                }
            } else if (!core::embed_row(*m.wt, g, tokens[c0 + t], m.emb + t * N, m.cs, err)) {
                return false;
            }
        }
        if (hand_in_ == nullptr) gr_broadcast(m.emb, m.R, T, m.cs);
        lap("embeddings");
        // ---- the PLE rows of the whole chunk, one batched SSD request on a thread (see ple_gather).  A later chunk's
        // were read ahead during the previous chunk; the first chunk's are read beside layer 0 - they are needed from
        // layer 1 on, and gathering them here first left the GPU idle for the whole read (~0.4 s of a 32K prompt)
        if (ple_on && !ple_next.valid())
            ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c0, b = ple_buf] {
                return ple_gather(c0, b, ple_next_err);
            });
        bool ple_pending = ple_on;
        // the chunk's rows onto the device just before layer 1 reads them, and the next chunk's gather started
        auto ple_land = [&]() -> bool {
            if (!ple_pending) return true;
            ple_pending = false;
            const auto tp = Clock::now();
            if (!ple_next.get()) {
                err = ple_next_err;
                return false;
            }
            /*
            DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            if (DPCT_CHECK_ERROR(m.cs->memcpy(m.ple_emb,
                                              m.ple_emb_host[ple_buf],
                                              (size_t)T * N * 4)) != 0 ||
                /*
                DPCT1024: The original code returned the error code that was
                further consumed by the program logic. This original code was
                replaced with 0. You may need to rewrite the program logic
                consuming the error code.
                */
                DPCT_CHECK_ERROR((m.ple_want[ple_buf] = ++m.ple_seq,
                                  m.cs->fill<uint64_t>(m.ple_done_seq + ple_buf, m.ple_want[ple_buf], 1))) != 0) {
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
                err = std::string("prefill: the PLE rows' upload failed: ") +
                      dpct::get_error_string_dummy(0);
                return false;
            }
            if (c0 + T < n) {   // SYCL port: T, this chunk's length (the first chunk can be shorter: chunk_len)
                // the other buffer's upload (a chunk ago) is done before the SSD thread refills it
                if (DPCT_CHECK_ERROR([&] {
                        while (*(volatile const uint64_t*) (m.ple_done_seq + (ple_buf ^ 1)) < m.ple_want[ple_buf ^ 1])
                            std::this_thread::yield();
                    }()) != 0) {
                    /*
                    DPCT1009: SYCL reports errors using exceptions and does
                    not use error codes. Please replace the
                    "get_error_string_dummy(...)" with a real error-handling
                    function.
                    */
                    /*
                    DPCT1010: SYCL uses exceptions to report errors and does
                    not use the error codes. The cudaGetLastError function call
                    was replaced with 0. You need to rewrite this code.
                    */
                    err =
                        std::string("prefill: the PLE rows' upload failed: ") +
                        dpct::get_error_string_dummy(0);
                    return false;
                }
                ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c1 = c0 + T, b = ple_buf ^ 1] {
                    return ple_gather(c1, b, ple_next_err);
                });
            }
            ple_buf ^= 1;
            stats_.ms_ple += ms_since(tp);
            return true;
        };
        for (int64_t t = 0; t < T; ++t) { prev[0] = prev[1]; prev[1] = (int32_t) tokens[c0 + t]; }
        lap("PLE rows");
        // ---- the QSA step records of every position in the chunk
        for (int64_t t = 0; t < T; ++t) strata::kernels::qsa_step_fill(m.steps_host.data() + t * strata::kernels::kStepCount, p0 + t, s);
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        m.cs->memcpy(m.steps_dev, m.steps_host.data(),
                     (size_t)T * strata::kernels::kStepCount * 4);

        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < LB; ++l) (core::is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        lap("step records");
        int64_t kv_pending = -1;
        auto kv_prefetch_after = [&](int64_t layer, int64_t ordinal) -> bool {
            try {
        if (!kv_prefetch || p0 <= 0) return true;
            if (kv_pending >= 0) return true;   // one staging pool: the pending prefetch has not been consumed yet
            for (int64_t l = layer; l < LE; ++l) {
                if (!core::is_qsa_layer(g, l)) continue;
                const core::QsaState& state = ss.qsa_states[ordinal];
                if (state.kv_mode != 1) { ++ordinal; continue; }
                // Reuse one KV pool only after its last attention reader completes. The next layer's
                // [0, p0) host prefix is immutable throughout this chunk; its new rows are appended later.
                /*
                DPCT1024: The original code returned the error code that
                was further consumed by the program logic. This original code
                was replaced with 0. You may need to rewrite the program logic
                consuming the error code.
                */
                if (DPCT_CHECK_ERROR(dpct::sync_barrier(m.kv_released, cs)) !=
                        0 ||
                    DPCT_CHECK_ERROR((m.kv_copy)->ext_oneapi_submit_barrier(
                        {*m.kv_released})) != 0) {
                    err = "prefill: releasing KV staging pool";
                    return false;
                }
                strata::kernels::kv_stage_from_host(pools_of(m.stage, m.ident_table), state.host,
                    core::qsa_kv_format(state), (p0 + s.page_size - 1) / s.page_size, s, m.kv_copy);
                /*
                DPCT1010: SYCL uses exceptions to report errors and does
                not use the error codes. The cudaPeekAtLastError function call
                was replaced with 0. You need to rewrite this code.
                */
                if (0 != 0 ||
                    /*
                    DPCT1024: The original code returned the error code
                    that was further consumed by the program logic. This
                    original code was replaced with 0. You may need to rewrite
                    the program logic consuming the error code.
                    */
                    DPCT_CHECK_ERROR(
                        dpct::sync_barrier(m.kv_ready, m.kv_copy)) != 0) {
                    err = "prefill: enqueueing KV prefetch";
                    return false;
                }
                kv_pending = ordinal;
                ++kv_prefetches;
                break;
            }
            return true;
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        if (!kv_prefetch_after(LB, qsa_index)) return false;
        // step 3: this chunk's stream - every non-resident expert of every layer, layer by layer in id order (entry
        // k lands in ring slot k % ring); a copy is issued once the entry `ring` before it is consumed (its slot's
        // `used` event recorded), so the copy stream never waits on an event that is not queued yet
        const strata::kernels::cpu::ExpertLayout& lay0 = strata::kernels::cpu::expert_layout();
        const bool stream_all = m.ring > STAGE && T >= stream_all_min() && m.src != nullptr;
        const bool ps_on = stream_all && m.pp && m.pp->ps_frac > 0.0;
        if (m.pp && !ps_on) m.pp->ps_flag.clear();
        // multi-GPU: the peer's ring - issue its copies up to `limit` / give entries back (the peer device is current)
        auto p_issue_until = [&](size_t limit) {
            try {
        PeerPrefill &P = *m.pp;
            limit = std::min(limit, P.pseq.size());
            while (P.p_issued < limit) {
                const PeerPrefill::PsEntry& en = P.pseq[P.p_issued];
                const size_t sl = P.p_issued % (size_t) P.RP;
                if (P.plive[sl])(P.s_cp)->ext_oneapi_submit_barrier(
                    {*P.pused[sl]});
                /*
                DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy
                API. While the origin API might be synchronous, it depends on
                the type of operand memory, so you may need to call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                P.s_cp->memcpy(
                    P.pstage[sl], en.blob,
                    (size_t)strata::kernels::cpu::expert_layout().blob_bytes(
                        en.l));
                dpct::sync_barrier(P.pcopied[sl], P.s_cp);
                P.plive[sl] = 1;
                ++P.p_issued;
            }
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        auto p_release_to = [&](int64_t l, int32_t e_stop, dpct::queue_ptr ps) {
            try {
        PeerPrefill &P = *m.pp;
            while (P.pk < P.pseq_start[(size_t) l + 1] && P.pseq[P.pk].e < e_stop) {
                dpct::sync_barrier(P.pused[P.pk % (size_t)P.RP], ps);
                ++P.pk;
                p_issue_until(P.pk + (size_t) P.RP);
            }
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        struct StreamEntry { int32_t l, e; const uint8_t* blob; int job; };
        std::vector<StreamEntry> seq;
        std::vector<size_t> seq_start;
        size_t issued = 0, consumed = 0;
        static const bool plan_reads = [] { const char* v = std::getenv("STRATA_GGUF_PLAN_READ"); return v && v[0] == '1'; }();
        const core::GgufExpertSource* gsrc = plan_reads ? nullptr : dynamic_cast<const core::GgufExpertSource*>(m.src);
        if (stream_all) {
            seq_start.assign((size_t) g.n_layers + 1, 0);
            std::vector<Stager::Job> js;
            if (ps_on) {
                m.pp->pseq.clear();
                m.pp->pseq_start.assign((size_t) g.n_layers + 1, 0);
                m.pp->ps_flag.assign((size_t) g.n_layers * m.g->n_expert, 0);
                m.pp->p_issued = m.pp->pk = 0;
            }
            double ps_acc = 0.0;
            for (int64_t l = LB; l < LE; ++l) {
                seq_start[(size_t) l] = seq.size();
                if (ps_on) m.pp->pseq_start[(size_t) l] = m.pp->pseq.size();
                for (int32_t e = 0; e < m.g->n_expert; ++e) {
                    if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                    if (m.pp && m.pp->peer && m.pp->peer->has(l, e)) continue;   // multi-GPU: computed on (or read from) the peer
                    int job = -1;
                    const uint8_t* b = nullptr;
                    if (gsrc != nullptr) {   // read by the stager's thread when its turn comes (see Stager::Job)
                        job = (int) js.size();
                        js.push_back({nullptr, (size_t) lay0.blob_bytes(l), nullptr, (int32_t) l, e, gsrc});
                    } else if (m.src->transient(l, e)) {   // CS-T: copied by the source into the stager's buffer
                        job = (int) js.size();
                        js.push_back({nullptr, (size_t) lay0.blob_bytes(l), m.src, (int32_t) l, e});
                    } else {
                        b = m.src->blob(l, e);
                        if (!b) { err = "prefill: expert source has no blob"; return false; }
                        if (ps_on && m.src->pinned(l, e)) {   // multi-GPU: every ps_frac-th one goes to the peer's ring
                            ps_acc += m.pp->ps_frac;
                            if (ps_acc >= 1.0) {
                                ps_acc -= 1.0;
                                m.pp->pseq.push_back({(int32_t) l, e, b});
                                m.pp->ps_flag[(size_t) l * m.g->n_expert + e] = 1;
                                continue;
                            }
                        }
                        if (!dma_direct(m.src, l, e, b)) {
                            job = (int) js.size();
                            js.push_back({b, (size_t) lay0.blob_bytes(l)});
                        }
                    }
                    seq.push_back({(int32_t) l, e, b, job});
                }
            }
            for (int64_t l = LE; l <= g.n_layers; ++l) {
                seq_start[(size_t) l] = seq.size();
                if (ps_on) m.pp->pseq_start[(size_t) l] = m.pp->pseq.size();
            }
            m.stager->start(std::move(js));
        }
        struct StagerDone {
            Stager* st;
            ~StagerDone() { if (st) st->finish(); }
        } chunk_stager_done{stream_all ? m.stager.get() : nullptr};
        auto issue_until = [&](size_t limit) {
            try {
        limit = std::min(limit, seq.size());
            while (issued < limit) {
                const StreamEntry& en = seq[issued];
                const int sl = (int) (issued % (size_t) m.ring);
                const auto th = Clock::now();
                const size_t bytes = (size_t) lay0.blob_bytes(en.l);
                if (m.stage_live[sl])(m.copy)->ext_oneapi_submit_barrier(
                    {*m.used[m.used_of[sl]]});
                if (en.job < 0) {
                    /*
                    DPCT1124: cudaMemcpyAsync is migrated to asynchronous
                    memcpy API. While the origin API might be synchronous, it
                    depends on the type of operand memory, so you may need to
                    call wait() on event return by memcpy API to ensure
                    synchronization behavior.
                    */
                    m.copy->memcpy(m.stage_dev[sl], en.blob, bytes);
                    ++stats_.experts_dma;
                } else {
                    const uint8_t* hb = m.stager->wait(en.job);
                    /*
                    DPCT1124: cudaMemcpyAsync is migrated to asynchronous
                    memcpy API. While the origin API might be synchronous, it
                    depends on the type of operand memory, so you may need to
                    call wait() on event return by memcpy API to ensure
                    synchronization behavior.
                    */
                    m.copy->memcpy(m.stage_dev[sl], hb, bytes);
                    m.stager->issued_one(en.job, m.copy);
                }
                dpct::sync_barrier(m.copied[sl], m.copy);
                m.stage_live[sl] = true;
                stats_.ms_experts_host += ms_since(th);
                ++stats_.experts_streamed;
                ++issued;
            }
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        // D-5: the stream is issued by its own host thread, so the thread launching the layers' kernels never waits
        // behind a host copy of an unpinned blob (that wait left the GPU idle: the 'wait copy' / 'dequant' time of the
        // i-quant prompts).  The same copies in the same order into the same slots, and a slot is refilled only once
        // the compute stream has recorded that it is done with it: the same results.  STRATA_PREFILL_ISSUER=0: inline.
        static const bool issuer_on = [] {
            const char* v = std::getenv("STRATA_PREFILL_ISSUER");
            return v == nullptr || std::atoi(v) != 0;
        }();
        std::atomic<size_t> a_issued{0}, a_consumed{0};
        std::atomic<bool> a_stop{false};
        double iss_ms = 0;
        int64_t iss_streamed = 0, iss_dma = 0;
        std::thread issuer;
        struct IssuerJoin {
            std::atomic<bool>* stop;
            std::thread* t;
            ~IssuerJoin() { if (t->joinable()) { stop->store(true); t->join(); } }
        } issuer_join{&a_stop, &issuer};
        const bool threaded_issue = stream_all && issuer_on;
        if (threaded_issue) {
            issuer = std::thread([&] {
                const core::OnDevice od(m.device);
                for (size_t idx = 0; idx < seq.size(); ++idx) {
                    while (idx >= a_consumed.load(std::memory_order_acquire) + (size_t) m.ring) {
                        if (a_stop.load(std::memory_order_acquire)) return;
                        std::this_thread::yield();
                    }
                    const StreamEntry& en = seq[idx];
                    const int sl = (int) (idx % (size_t) m.ring);
                    const auto th = Clock::now();
                    const size_t bytes = (size_t) lay0.blob_bytes(en.l);
                    if (m.stage_live[sl])(m.copy)->ext_oneapi_submit_barrier(
                        {*m.used[m.used_of[sl]]});
                    if (en.job < 0) {
                        /*
                        DPCT1124: cudaMemcpyAsync is migrated to
                        asynchronous memcpy API. While the origin API might be
                        synchronous, it depends on the type of operand memory,
                        so you may need to call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        m.copy->memcpy(m.stage_dev[sl], en.blob, bytes);
                        ++iss_dma;
                    } else {
                        const uint8_t* hb = m.stager->wait(en.job);
                        /*
                        DPCT1124: cudaMemcpyAsync is migrated to
                        asynchronous memcpy API. While the origin API might be
                        synchronous, it depends on the type of operand memory,
                        so you may need to call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        m.copy->memcpy(m.stage_dev[sl], hb, bytes);
                        m.stager->issued_one(en.job, m.copy);
                    }
                    dpct::sync_barrier(m.copied[sl], m.copy);
                    m.stage_live[sl] = true;
                    iss_ms += ms_since(th);
                    ++iss_streamed;
                    a_issued.store(idx + 1, std::memory_order_release);
                }
            });
        } else if (stream_all) {
            issue_until((size_t) m.ring);   // layer 0's first experts, behind the embedding and the PLE
        }
        // the consumer's side: entry k's copy is on the copy stream (the thread issued it), then k is given back
        auto wait_issued = [&](size_t k) {
            if (!threaded_issue) return;
            while (a_issued.load(std::memory_order_acquire) <= k) std::this_thread::yield();
        };
        auto give_back = [&](size_t upto) {
            if (threaded_issue) a_consumed.store(upto, std::memory_order_release);
            else issue_until(upto + (size_t) m.ring);
        };
        lap("expert stream plan");
        if (ps_on) {
            int pd = 0;
            pd = dpct::get_current_device_id();
            /*
            DPCT1093: The "m.pp->dev" device may be not the one intended for
            use. Adjust the selected device if needed.
            */
            dpct::select_device(m.pp->dev);
            p_issue_until((size_t) m.pp->RP);
            /*
            DPCT1093: The "pd" device may be not the one intended for use.
            Adjust the selected device if needed.
            */
            dpct::select_device(pd);
        }
        host_setup_ms += ms_since(tsetup);
        bool normed = false;   // F-2: the previous half's write already normed R for this half (grs, xn16)
        // #579 #613 (opt-in diagnosis, STRATA_PF_STEP_SYNC=1): the compute and copy streams are waited for after each
        // step named below, a step that took over 250 ms is logged, and a stall's report names the step it is in.
        // Slower (a sync per step); the bytes are the same.
        static const bool step_sync = [] { const char* e = std::getenv("STRATA_PF_STEP_SYNC"); return e && e[0] == '1'; }();
        auto pf_step = [&](const char *what, int64_t layer) {
            try {
        if (!step_sync) return;
            core::progress_at(what, layer, p0);
            const auto ts = Clock::now();
            const dpct::err0 a = DPCT_CHECK_ERROR(m.cs->wait()),
                             b = DPCT_CHECK_ERROR(m.copy->wait());
            const double ms = ms_since(ts);
            if (ms > 250.0 || a != 0 || b != 0)
                std::fprintf(stderr,
                             "strata pf-step: chunk from token %lld, layer "
                             "%lld: %s took %.0f ms (%s / %s)\n",
                             /*
                             DPCT1009: SYCL reports errors using exceptions
                             and does not use error codes. Please replace the
                             "get_error_string_dummy(...)" with a real
                             error-handling function.
                             */
                             (long long)p0, (long long)layer, what, ms,
                             dpct::get_error_string_dummy(a),
                             dpct::get_error_string_dummy(b));
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        for (int64_t l = LB; l < LE; ++l) {
            core::progress_beat();   // the serve watchdog: a prompt chunk of 8192 tokens is still moving
            if (l > LB) pf_step("reading the prompt (batched, step sync): the experts and the rest of layer", l - 1);
            core::progress_at("reading the prompt (batched): layer", l, p0);   // #251: a stall names layer and chunk
            const core::LayerView v(*m.wt, l);
            if (l == std::max<int64_t>(LB, 1) && !ple_land()) return false;   // the PLE rows, read from layer 1 on
            // ---- the PLE block at layer 1, token by token (its conv reads the previous tokens' rows)
            if (l == 1 && ple_on && ple_batch) {
                // the whole chunk at once, in sub-batches carved from the idle scratch region: the key and value
                // projections as GEMMs (a token at a time they re-read ~52 MB of BF16 key per token on the IQ
                // files), the rest with the per-token kernels' arithmetic (native_ple_postops_batch)
                pt.mark(kPfPle, cs);
                const auto tp = Clock::now();
                const strata::kernels::PleWeights& pw = ss.ple.w;
                constexpr int64_t HD = strata::kernels::NG_HC_DIM;
                const uint64_t per_token = (uint64_t) (3 * HD + N + 4) * 4 + (uint64_t) N * (bf16x2(m.f16_io) ? 4 : 2) + 4096;
                const int64_t SB = std::min<int64_t>(T, (int64_t) (m.region_bytes / per_token));
                for (int64_t s0 = 0; s0 < T; s0 += SB) {
                    const int64_t nb = std::min(SB, T - s0);
                    uint8_t* q = m.region;
                    auto carve_f = [&](size_t n) { float* p = (float*) q; q += (n * 4 + 255) & ~(size_t) 255; return p; };
                    float* key = carve_f((size_t) nb * HD);
                    float* qn = carve_f((size_t) nb * HD);
                    float* gated = carve_f((size_t) nb * HD);
                    float* val = carve_f((size_t) nb * N);
                    float* gate = carve_f((size_t) nb * 4);
                    uint16_t* e16 = (uint16_t*) carve_f((size_t) nb * N / 2);
                    uint16_t* e16_lo = bf16x2(m.f16_io) ? (uint16_t*) carve_f((size_t) nb * N / 2) : nullptr;
                    const float* emb = m.ple_emb + s0 * N;
                    if (pw.key_bf16 != nullptr) {
                        to_bf16(emb, e16, nb * N, m.cs, e16_lo);
                        m.gemm.bf16(e16, pw.key_bf16, key, nb, HD, N);
                        if (e16_lo) m.gemm.bf16(e16_lo, pw.key_bf16, key, nb, HD, N, 0, 1.0f);
                    } else {
                        to_f16(emb, e16, nb * N, m.cs);
                        m.gemm.native(e16, pw.key_native_type, pw.key_native_data, key, nb, HD, N);
                        to_bf16(emb, e16, nb * N, m.cs, e16_lo);
                    }
                    m.gemm.bf16(e16, pw.value_bf16, val, nb, N, N);
                    if (e16_lo) m.gemm.bf16(e16_lo, pw.value_bf16, val, nb, N, N, 0, 1.0f);
                    try {
                        strata::kernels::native_ple_postops_batch(key, m.R + s0 * D, val, ss.ple.hist, pw, qn, gated,
                                                                  gate, (int) nb, m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                }
                stats_.ms_ple += ms_since(tp);
            } else if (l == 1 && ple_on) {
                pt.mark(kPfPle, cs);
                const auto tp = Clock::now();
                for (int64_t t = 0; t < T; ++t) {
                    strata::kernels::PleOut po;
                    po.normalized = m.ple_norm;
                    po.result = m.R + t * D;
                    try {
                        strata::kernels::ple_block(m.ple_emb + t * N, m.R + t * D, ss.ple.hist, ss.ple.w, po,
                                                   ss.ple.scratch, m.cs);
                    } catch (const std::exception& e) { err = std::string("prefill PLE: ") + e.what(); return false; }
                    strata::kernels::ple_history_advance(ss.ple.hist, m.ple_norm, m.cs);
                }
                stats_.ms_ple += ms_since(tp);
            }
            for (int half = 0; half < 2; ++half) {
                // ---- the hyper-connection read of this half
                const char* pre = half == 0 ? "hc_attn_" : "hc_ffn_";
                const std::string sn = std::string(pre) + "norm.weight", sd = std::string(pre) + "down.weight",
                                  su = std::string(pre) + "up.weight", si = std::string(pre) + "inject.weight";
                const core::WeightRef *wn = need(v, sn.c_str(), err), *wd = need(v, sd.c_str(), err),
                                      *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                if (!wn || !wd || !wu || !wi) return false;
                pt.mark(kPfHc, cs);
                // the xn16 token stride of this chunk (the writers of both halves and of the fused write-backs use it)
                const int64_t ldx = hc_pad() && T >= std::max<int64_t>(pf_switch_min_t(), 64) && !gr_unfused() &&
                                            !m.xn16_lo ? D + XN_PAD : D;
                if (gr_unfused()) gr_norm(m.R, (const float*) wn->data, EPS, m.xn, m.xn16, T, m.cs, m.xn16_lo);
                else if (!normed) gr_norm_rs(m.R, (const float*) wn->data, EPS, m.grs, m.xn16, T, m.cs, m.xn16_lo, ldx);
                normed = false;
                bool hcd = false;
                bool hdown = false;
                if (ldx != D && !pf_hcdown()) {   // STRATA_HCD_EXACT: the exact-order down kernel when hipBLASLt would take 1176 / 1177
                    hdown = wd->kind == core::WeightKind::Bf16InF32 && wd->data && wd->ne0 == D && wd->ne1 == LR &&
                            m.gemm.bf16_hcd_exact(m.xn16, ldx, (const uint16_t*) wd->data, m.lo, T, LR, D);
                } else if (ldx != D) {
                    if (wd->kind != core::WeightKind::Bf16InF32 || !wd->data || wd->ne0 != D || wd->ne1 != LR ||
                        wi->kind != core::WeightKind::Bf16InF32 || !wi->data || wi->ne0 != D || wi->ne1 != HC ||
                        !strata_pf_hcdown_bf16(m.xn16, ldx, (const uint16_t*) wd->data, (const uint16_t*) wi->data, LR,
                                               HC, m.lo, m.inj, T, D, m.cs)) {
                        err = "prefill: STRATA_PF_HCDOWN could not run the hyper-connection projections";
                        return false;
                    }
                    hcd = true;
                }
                if (!hcd && !hdown && !bf16_proj(m.gemm, wd, m.xn16, m.lo, T, sd, err, 0, m.xn16_lo, ldx != D ? ldx : 0)) return false;
                gr_silu(m.lo, m.lo16, T, m.cs, m.lo16_lo);
                bool upmixed = false;
                if (hc_upmix() && T >= pf_switch_min_t() && !gr_unfused() && !m.lo16_lo && !m.mixed_bf_lo &&
                    wu->kind == core::WeightKind::Bf16InF32 && wu->data && wu->ne0 == LR && wu->ne1 == D) {
                    static int checks = [] { const char* e = std::getenv("STRATA_HC_UPMIX_CHECK"); return e ? std::atoi(e) : 0; }();
                    if (checks > 0) {   // the default pair first, kept for the comparison
                        --checks;
                        if (!bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err)) return false;
                        gr_mix_r(m.R, m.grs, (const float*) wn->data, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h);
                        std::vector<float> ref((size_t) T * N), got((size_t) T * N);
                        /*
                        DPCT1124: cudaMemcpyAsync is migrated to
                        asynchronous memcpy API. While the origin API might be
                        synchronous, it depends on the type of operand memory,
                        so you may need to call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        m.cs->memcpy(ref.data(), m.mixed, ref.size() * 4);
                        upmixed = gr_upmix(m.lo16, (const uint16_t*) wu->data, m.R, m.grs, (const float*) wn->data,
                                           m.mixed, m.mixed_bf, m.mixed_h, T, m.cs);
                        /*
                        DPCT1124: cudaMemcpyAsync is migrated to
                        asynchronous memcpy API. While the origin API might be
                        synchronous, it depends on the type of operand memory,
                        so you may need to call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        m.cs->memcpy(got.data(), m.mixed, got.size() * 4);
                        m.cs->wait();
                        double e2 = 0, r2 = 0, emax = 0;
                        for (size_t i = 0; i < ref.size(); ++i) {
                            const double dd = (double) got[i] - ref[i];
                            e2 += dd * dd; r2 += (double) ref[i] * ref[i]; emax = std::max(emax, std::fabs(dd));
                        }
                        std::fprintf(stderr, "strata: STRATA_HC_UPMIX_CHECK layer %lld half %d T %lld: mixed rel RMS %.3e, "
                                     "max |diff| %.3e (%s)\n", (long long) l, half, (long long) T,
                                     std::sqrt(e2 / std::max(r2, 1e-30)), emax, upmixed ? "upmix" : "upmix refused");
                    } else {
                        upmixed = gr_upmix(m.lo16, (const uint16_t*) wu->data, m.R, m.grs, (const float*) wn->data,
                                           m.mixed, m.mixed_bf, m.mixed_h, T, m.cs);
                    }
                }
                if (!upmixed && !bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err, 0, m.lo16_lo)) return false;
                if (!hcd && !bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err, 0, m.xn16_lo, ldx != D ? ldx : 0)) return false;
                if (upmixed) {
                } else if (gr_unfused()) {
                    gr_mix(m.xn, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h, m.mixed_bf_lo);
                } else {
                    gr_mix_r(m.R, m.grs, (const float*) wn->data, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h,
                             m.mixed_bf_lo);
                }

                if (half == 0 && !core::is_qsa_layer(g, l)) {
                    // ======================= GDN =======================
                    const core::WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                                          *wo = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                                          *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                                          *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                                          *wsa = need(v, "ssm_a", err);
                    if (!wqkv || !wg || !wo || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
                    pt.mark(kPfGdn, cs);
                    float* state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                    if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
                    if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
                    pt.mark(kPfGdnConv, cs);   // "gdn" is the projections in; the rest on their own lines
                    gdn_gates(m.ab, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, T, m.cs);
                    gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs);
                    pt.mark(kPfGdnRec, cs);
                    const int64_t ld_y = pf_pad() && T >= std::max<int64_t>(pf_switch_min_t(), 64) ? ZV + ZV_PAD : 0;
                    gdn_recurrence(state, m.hbuf, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.y, m.y_h, T, m.cs,
                                   ld_y);
                    pt.mark(kPfGdnOut, cs);
                    if (!native_proj(m.gemm, wo, m.y_h, m.bo, T, v.name("ssm_out.weight"), err, 0, ld_y)) return false;
                    ++gdn_index;
                } else if (half == 0) {
                    // ======================= QSA =======================
                    const core::QsaState& st = ss.qsa_states[qsa_index];
                    const core::WeightRef *wq = need(v, "attn_q.weight", err), *wk = need(v, "attn_k.weight", err),
                                          *wv = need(v, "attn_v.weight", err), *wo = need(v, "attn_output.weight", err),
                                          *wik = need(v, "indexer.k_proj.weight", err),
                                          *wiq = need(v, "indexer.q_proj.weight", err),
                                          *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                                          *wiqn = need(v, "indexer.q_norm.weight", err),
                                          *wikn = need(v, "indexer.k_norm.weight", err);
                    if (!wq || !wk || !wv || !wo || !wik || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
                    pt.mark(kPfQsa, cs);
                    if (!native_proj(m.gemm, wk, m.mixed_h, m.Kc, T, v.name("attn_k.weight"), err)) return false;
                    if (!native_proj(m.gemm, wv, m.mixed_h, m.Vc, T, v.name("attn_v.weight"), err)) return false;
                    if (!native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err)) return false;
                    if (!bf16_proj(m.gemm, wik, m.mixed_bf, m.idx_raw, T, v.name("indexer.k_proj.weight"), err, 0, m.mixed_bf_lo)) return false;
                    if (!bf16_proj(m.gemm, wiq, m.mixed_bf, m.q_idx, T, v.name("indexer.q_proj.weight"), err, 0, m.mixed_bf_lo)) return false;
                    rms_rows(m.Kc, (const float*) wkn->data, T * 2, 256, 256, EPS, m.cs);
                    rope(m.Kc, T, 2, 256, 512, p0, strata::kernels::rope_scaling(), m.cs);
                    // KV streaming: this layer's cells [0, p0) come in from the host copy to the staging pool, and the
                    // chunk's cells go to the host copy, the staging pool, and the VRAM slots of resident blocks
                    const bool staged = st.kv_mode == 1;
                    if (staged) {
                        pt.mark(kPfKvStage, cs);
                        if (kv_pending == qsa_index) {
                            if (DPCT_CHECK_ERROR(cs->ext_oneapi_submit_barrier(
                                    {*m.kv_ready})) != 0) {
                                err = "prefill: waiting for KV prefetch";
                                return false;
                            }
                            kv_pending = -1;
                        } else strata::kernels::kv_stage_from_host(pools_of(m.stage, m.ident_table), st.host,
                                                            core::qsa_kv_format(st),
                                                            (p0 + s.page_size - 1) / s.page_size, s, m.cs);
                        pt.mark(kPfQsa, cs);
                        pf_step("reading the prompt (batched, step sync): the K/V staged from RAM at layer", l);
                    }
                    // #579 #613 (HIP, opt-in A/B, STRATA_KV_HOST_DMA=1): the append writes the staging pool and the
                    // resident slots only, and one DMA copies the chunk's blocks from the staging pool to the host copy
                    // - no kernel writes host memory over PCIe.  The bytes every reader sees are the same (the staged
                    // first block is complete; past the chunk's last cell nothing is read until a later append writes
                    // it).  CUDA: never.
#if defined(STRATA_USE_HIP)
                    static const bool kv_host_dma = [] { const char* e = std::getenv("STRATA_KV_HOST_DMA"); return e && e[0] == '1'; }();
#else
                    constexpr bool kv_host_dma = false;
#endif
                    const bool host_by_dma = staged && kv_host_dma;
                    const strata::kernels::KvHostPools* host_w = host_by_dma ? nullptr : &st.host;
                    if (st.kv_hybrid) {   // K8V4: K INT8 unrotated, V rotated Q4_0 (only V and the output rotate)
                        // streamed: each half writes its part of the host copy and of the staging pool
                        strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        const strata::kernels::KvHostPools hk = strata::kernels::kv_hybrid_k_half(st.host),
                                                           hv = strata::kernels::kv_hybrid_v_half(st.host),
                                                           sk = strata::kernels::kv_hybrid_k_half(m.stage),
                                                           sv = strata::kernels::kv_hybrid_v_half(m.stage);
                        const bool mirror = st.host.present();
                        kv_append(m.Kc, m.Kc, T, p0, st.page_table, s.page_size, nullptr, nullptr,
                                  st.k_q, st.k_q, st.k_scale, st.k_scale, m.cs, mirror ? &hk : nullptr,
                                  staged ? &sk : nullptr);
                        strata::kernels::kv_append_q4(st.v_q4, st.v_q4, st.page_table, p0, T, m.Vc, m.Vc, s, m.cs,
                                                      mirror ? &hv : nullptr, staged ? &sv : nullptr);
                    } else {
                        if (st.kv_rot) {   // rotated K and V (kv_q4.hpp), the queries below too, the output back
                            strata::kernels::fwht256_inplace_cuda(m.Kc, T * 2, m.cs);
                            strata::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
                        }
                        if (st.kv_q4)
                            strata::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, p0, T, m.Kc, m.Vc, s, m.cs,
                                                          host_w, staged ? &m.stage : nullptr);
                        else
                            kv_append(m.Kc, m.Vc, T, p0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs,
                                      host_w, staged ? &m.stage : nullptr);
                        if (host_by_dma)
                            strata::kernels::kv_unstage_to_host(pools_of(m.stage, m.ident_table), st.host,
                                                                core::qsa_kv_format(st), p0 / s.page_size,
                                                                (p0 + T + s.page_size - 1) / s.page_size, s, m.cs);
                    }
                    if (staged) pf_step("reading the prompt (batched, step sync): the K/V append at layer", l);
                    split_q(m.Qf, m.q, T, m.cs);
                    rms_rows(m.q, (const float*) wqn->data, T * 24, 256, 256, EPS, m.cs);
                    rope(m.q, T, 24, 256, 6144, p0, strata::kernels::rope_scaling(), m.cs);
                    if (st.kv_rot) strata::kernels::fwht256_inplace_cuda(m.q, T * 24, m.cs);
                    rms_rows(m.q_idx, (const float*) wiqn->data, T * 4, 128, 128, EPS, m.cs);
                    rope(m.q_idx, T, 4, 128, 512, p0, strata::kernels::rope_scaling(), m.cs);
                    // the indexer appends, token by token; then scores + selection for many queries at once:
                    // a query reads completed blocks (final once completed) and `dead` for its own tail block
                    const strata::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
                    pt.mark(kPfQsaIdx, cs);
                    // C-2: the chunk's appends in three launches instead of one per token (the same end state:
                    // the queries below read it only after the whole chunk is appended). STRATA_INDEXER_PER_TOKEN=1: the old
                    try {
                        static const bool per_token = std::getenv("STRATA_INDEXER_PER_TOKEN") != nullptr;
                        if (!per_token) {
                            strata::kernels::native_qsa_indexer_append_batch(m.idx_raw, T, p0, 0, (const float*) wikn->data,
                                                                             EPS, ib, s, st.max_cells,
                                                                             strata::kernels::rope_scaling(), m.cs);
                        }
                        for (int64_t t = 0; per_token && t < T; ++t) {
                            const int32_t* step_t = m.steps_dev + t * strata::kernels::kStepCount;
                            strata::kernels::native_qsa_indexer_append(m.idx_raw + t * 128, step_t + strata::kernels::kStepPos, 0,
                                                                       (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                                                       strata::kernels::rope_scaling(), m.cs);
                        }
                    } catch (const std::exception& e) { err = std::string("prefill indexer: ") + e.what(); return false; }
                    pt.mark(kPfQsaSel, cs);
                    for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                        const int64_t nb = std::min(m.sel_batch, T - t0);
                        const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                        // C-1: the grid reaches the batch's last query's n_bid (they rise with the position)
                        const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * strata::kernels::kStepCount +
                                                                                strata::kernels::kStepNBid)] + 1;
                        // the scores on tensor cores (3xTF32: FP32-level, not bitwise); STRATA_SELECT_OLD=1: the warp kernel
                        static const bool old_sel = std::getenv("STRATA_SELECT_OLD") != nullptr;
                        if (old_sel || !strata::kernels::qsa_block_scores_tc(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512,
                                                                             steps0, nb, m.max_blocks, s, m.sel_scores,
                                                                             m.cs, active))
                            strata::kernels::qsa_block_scores(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512, steps0, nb,
                                                              m.max_blocks, s, m.sel_scores, m.cs, active);
                        strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                        m.sel_ids + t0 * m.cap, m.cs, active);
                    }
                    // STRATA_SEL_OVERLAP (debug, D-1's question): how much do neighbouring queries' selections share?
                    // Per tile of 16 queries: the union of their selected cells against the sum of their widths.
                    if (static const bool ovl = std::getenv("STRATA_SEL_OVERLAP") != nullptr; ovl && qsa_index == 0) {
                        std::vector<int32_t> ids((size_t) (T * m.cap));
                        /*
                        DPCT1124: cudaMemcpyAsync is migrated to
                        asynchronous memcpy API. While the origin API might be
                        synchronous, it depends on the type of operand memory,
                        so you may need to call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        m.cs->memcpy(ids.data(), m.sel_ids, ids.size() * 4);
                        m.cs->wait();
                        double sum_w = 0, sum_u = 0;
                        for (int64_t t0 = 0; t0 + 16 <= T; t0 += 16) {
                            std::vector<int32_t> u;
                            for (int64_t t = t0; t < t0 + 16; ++t) {
                                const int64_t w = m.steps_host[(size_t) (t * strata::kernels::kStepCount + strata::kernels::kStepWidth)];
                                sum_w += (double) w;
                                u.insert(u.end(), ids.begin() + t * m.cap, ids.begin() + t * m.cap + w);
                            }
                            std::sort(u.begin(), u.end());
                            sum_u += (double) (std::unique(u.begin(), u.end()) - u.begin());
                        }
                        std::fprintf(stderr, "strata prefill: selection overlap at %lld: 16-query tiles read %.1f%% of the "
                                             "cells one query at a time does\n", (long long) p0, sum_w > 0 ? 100.0 * sum_u / sum_w : 0.0);
                    }
                    // STRATA_IDX_FP16_CHECK: would FP16 pooled indexer keys select the same cells? (the KV-streaming
                    // design's last question). Every query is selected again from the pooled keys and `dead` rounded
                    // to fp16 (exactly what an fp16 store reads back); the agreement with the fp32 selection is
                    // printed cumulatively after each chunk's last QSA layer. Debug: syncs per layer.
                    if (static const bool f16chk = std::getenv("STRATA_IDX_FP16_CHECK") != nullptr; f16chk) {
                        static float *pooled16 = nullptr, *dead16 = nullptr;
                        static int32_t* ids16 = nullptr;
                        static double shared = 0, cells = 0;
                        static long long queries = 0, same = 0, sel_queries = 0;
                        const int64_t rows = st.idx_pooled_rows;
                        if (pooled16 == nullptr &&
                            (DPCT_CHECK_ERROR(
                                 pooled16 = (float *)sycl::malloc_device(
                                     (size_t)rows * s.idx_dim * 4,
                                     dpct::get_in_order_queue())) != 0 ||
                             DPCT_CHECK_ERROR(
                                 dead16 = (float *)sycl::malloc_device(
                                     (size_t)s.idx_dim * 4,
                                     dpct::get_in_order_queue())) != 0 ||
                             DPCT_CHECK_ERROR(
                                 ids16 = (int32_t *)sycl::malloc_device(
                                     (size_t)(m.T * m.cap) * 4,
                                     dpct::get_in_order_queue())) != 0)) {
                            err = "STRATA_IDX_FP16_CHECK: no room for its buffers";
                            return false;
                        }
                        round_f16(st.idx_pooled, pooled16, rows * s.idx_dim, m.cs);
                        round_f16(st.idx_dead, dead16, s.idx_dim, m.cs);
                        for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
                            const int64_t nb = std::min(m.sel_batch, T - t0);
                            const int32_t* steps0 = m.steps_dev + t0 * strata::kernels::kStepCount;
                            strata::kernels::qsa_block_scores(pooled16, dead16, m.q_idx + t0 * 512, steps0, nb,
                                                              m.max_blocks, s, m.sel_scores, m.cs);
                            strata::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                                            ids16 + t0 * m.cap, m.cs);
                        }
                        std::vector<int32_t> a((size_t) (T * m.cap)), b((size_t) (T * m.cap));
                        /*
                        DPCT1124: cudaMemcpyAsync is migrated to
                        asynchronous memcpy API. While the origin API might be
                        synchronous, it depends on the type of operand memory,
                        so you may need to call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        m.cs->memcpy(a.data(), m.sel_ids, a.size() * 4);
                        /*
                        DPCT1124: cudaMemcpyAsync is migrated to
                        asynchronous memcpy API. While the origin API might be
                        synchronous, it depends on the type of operand memory,
                        so you may need to call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        m.cs->memcpy(b.data(), ids16, b.size() * 4);
                        m.cs->wait();
                        for (int64_t t = 0; t < T; ++t) {
                            const int64_t w = m.steps_host[(size_t) (t * strata::kernels::kStepCount + strata::kernels::kStepWidth)];
                            const int32_t *x = a.data() + t * m.cap, *y = b.data() + t * m.cap;
                            int64_t i = 0, j = 0, c = 0;
                            while (i < w && j < w) {
                                if (x[i] == y[j]) { ++c; ++i; ++j; } else if (x[i] < y[j]) ++i; else ++j;
                            }
                            ++queries;
                            same += c == w;
                            if (p0 + t + 1 > m.cap) { ++sel_queries; shared += (double) c; cells += (double) w; }
                        }
                        if (qsa_index + 1 == g.n_qsa_layers())
                            std::fprintf(stderr, "strata prefill: FP16 indexer keys: %lld of %lld selections identical; "
                                                 "where the selection is sparse, %.4f%% of cells shared (%lld queries)\n",
                                         same, queries, cells > 0 ? 100.0 * shared / cells : 100.0, sel_queries);
                    }
                    // STRATA_QSA_DUMP=<file>: append every QSA layer's selected cells for the prompt's last
                    // STRATA_QSA_DUMP_LAST (4096) positions - records of int32 {qsa layer, pos0, T, cap} + T*cap cells,
                    // for tools/qsa_locality.py (how local the sparse attention's reads are: the KV-streaming question)
                    if (static const char* dump = std::getenv("STRATA_QSA_DUMP"); dump != nullptr) {
                        static const long long last = std::getenv("STRATA_QSA_DUMP_LAST")
                                                          ? std::atoll(std::getenv("STRATA_QSA_DUMP_LAST")) : 4096;
                        if (p0 + T > pos0 + n - last) {
                            std::vector<int32_t> h((size_t) (T * m.cap));
                            /*
                            DPCT1124: cudaMemcpyAsync is migrated to
                            asynchronous memcpy API. While the origin API might
                            be synchronous, it depends on the type of operand
                            memory, so you may need to call wait() on event
                            return by memcpy API to ensure synchronization
                            behavior.
                            */
                            m.cs->memcpy(h.data(), m.sel_ids, h.size() * 4);
                            m.cs->wait();
                            if (std::FILE* f = std::fopen(dump, "ab")) {
                                const int32_t hdr[4] = {(int32_t) qsa_index, (int32_t) p0, (int32_t) T, (int32_t) m.cap};
                                std::fwrite(hdr, 4, 4, f);
                                std::fwrite(h.data(), 4, h.size(), f);
                                std::fclose(f);
                            }
                        }
                    }
                    const strata::kernels::QsaAttnPools pools = staged ? pools_of(m.stage, m.ident_table)
                                                                       : core::qsa_attn_pools(st);
                    pt.mark(kPfQsaAttn, cs);
                    // perf-review D-1: the whole chunk on tensor cores, one block per (query, KV head), FP32-level
                    // accuracy but not bitwise (qsa_prompt_attn.hpp). Q4_0 KV, or STRATA_PROMPT_ATTN_OLD=1: the
                    // decode kernel, 32 queries at a time (K8V4 runs the tensor kernel's mode 3: INT8 K,
                    // V dequantized from its q4_0 blocks to fp16 at gather)
                    static const bool old_attn = std::getenv("STRATA_PROMPT_ATTN_OLD") != nullptr;
                    // STRATA_DUMP_SEL=<file>: the selected cells of every prompt position of the first QSA layer of the
                    // last chunk (int32 T, cap, then T*cap ids and T widths) - the input to the grouped-gather study
                    if (static const char* dsel = std::getenv("STRATA_DUMP_SEL"); dsel && c0 + T >= n && qsa_index == 0) {
                        m.cs->wait();
                        std::vector<int32_t> ids_h((size_t) T * m.cap), st_h((size_t) T * strata::kernels::kStepCount);
                        m.cs->memcpy(ids_h.data(), m.sel_ids, ids_h.size() * 4).wait();
                        m.cs->memcpy(st_h.data(), m.steps_dev, st_h.size() * 4).wait();
                        if (FILE* f = std::fopen(dsel, "wb")) {
                            const int32_t hdr[2] = {(int32_t) T, (int32_t) m.cap};
                            std::fwrite(hdr, 4, 2, f);
                            std::fwrite(ids_h.data(), 4, ids_h.size(), f);
                            for (int64_t t = 0; t < T; ++t) {
                                const int32_t w = st_h[(size_t) t * strata::kernels::kStepCount + strata::kernels::kStepWidth];
                                std::fwrite(&w, 4, 1, f);
                            }
                            std::fclose(f);
                            std::fprintf(stderr, "strata: selection of %lld positions dumped to %s\n", (long long) T, dsel);
                        }
                    }
                    if (old_attn || !strata::kernels::qsa_prompt_attn_batch(m.q, pools, m.sel_ids, m.steps_dev, m.cap, s,
                                                                            m.attn, T, m.cs))
                        for (int64_t t0 = 0; t0 < T; t0 += m.attn_batch) {
                            const int64_t nb = std::min(m.attn_batch, T - t0);
                            strata::kernels::qsa_decode_attn_batch(m.q + t0 * ZV, pools, m.sel_ids + t0 * m.cap,
                                                                   m.steps_dev + t0 * strata::kernels::kStepCount, m.cap,
                                                                   s, m.attn_scratch, m.attn + t0 * ZV, nb, m.cs);
                        }
                    if (st.kv_rot || st.kv_hybrid) strata::kernels::fwht256_inplace_cuda(m.attn, T * 24, m.cs);
                    pt.mark(kPfQsa, cs);
                    const int64_t ld_a = pf_pad() && T >= std::max<int64_t>(pf_switch_min_t(), 64) ? ZV + ZV_PAD : 0;
                    gate_attn(m.attn, m.Qf, m.attn_h, T, m.cs, ld_a);
                    if (!native_proj(m.gemm, wo, m.attn_h, m.bo, T, v.name("attn_output.weight"), err, 0, ld_a)) return false;
                    ++qsa_index;
                    if (!kv_prefetch_after(l + 1, qsa_index)) return false;
                } else {
                    // ======================= MoE =======================
                    const core::WeightRef *wr = need(v, "ffn_gate_inp.weight", err),
                                          *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                                          *wsg = need(v, "ffn_gate_shexp.weight", err),
                                          *wsu = need(v, "ffn_up_shexp.weight", err),
                                          *wsd = need(v, "ffn_down_shexp.weight", err);
                    if (!wr || !wgi || !wsg || !wsu || !wsd) return false;
                    pt.mark(kPfRouter, cs);
                    if (!bf16_proj(m.gemm, wr, m.mixed_bf, m.logits, T, v.name("ffn_gate_inp.weight"), err, 0, m.mixed_bf_lo)) return false;
                    route(m.logits, m.ids, m.w, T, m.g->n_expert, m.cs);
                    // the shared expert and its scalar gate
                    if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err)) return false;
                    if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err)) return false;
                    swiglu_pair(m.sgate, m.sup, m.sh_h, T, m.cs);
                    if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err)) return false;
                    if (wgi->kind != core::WeightKind::Bf16InF32) { err = "prefill: shared gate is not BF16"; return false; }
                    m.gemm.bf16(m.mixed_bf, (const uint16_t*) wgi->data, m.sg, T, 1, N);
                    if (m.mixed_bf_lo) m.gemm.bf16(m.mixed_bf_lo, (const uint16_t*) wgi->data, m.sg, T, 1, N, 0, 1.0f);
                    // #136: STRATA_PF_FUSED=1 - the Q2_0 pack's experts on the fused int8 kernels (moe_fused.hpp),
                    // grouped on the GPU: no host sync.  Only where every expert's place is known before the routing -
                    // the streamed walk, in which every non-resident expert of the layer comes through the ring in id
                    // order - and where the MMQ buffers exist: they hold the fused path's own (the per-token int8
                    // activations in Xq, the int8 H in H, the grouping tables in GU).  Chunks below stream_all_min()
                    // keep MMQ; without the variable nothing here runs.  A native pack's layer takes the native kernels
                    // (moe_fused_iq.hpp) where they cover its two formats, else MMQ (or the FP16 path: IQ1_M).
                    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
                    const bool use_mmq = mmq_plan().any && mmq_plan().layer[(size_t) l];
                    // APR (prism.hadamard): folded gate/up take H (s * x) on the gathered fp16 rows, folded down H (s * h) on
                    // the fp16 SwiGLU output, both rotated in fp16 storage with fp32 arithmetic (as the CUDA FP16 path does).
                    // The fused int8 products and the MMQ path do not rotate; they take only formats of unfolded layers.
                    const bool had_gu = lay.native && lay.fmt[(size_t) l].had_gu;
                    const bool had_d = lay.native && lay.fmt[(size_t) l].had_d;
                    const int had_block = lay.native ? lay.fmt[(size_t) l].had_block : 0;
                    const float* had_sx = had_gu ? m.had_sx[(size_t) l] : nullptr;
                    const float* had_sh = had_d ? m.had_sh[(size_t) l] : nullptr;
                    auto had_h = [&](int64_t row0, int64_t nrows) {
                        if (had_d) strata::kernels::hadamard_rows_f16(m.Hh + row0 * 640, nrows, 640, had_block, had_sh, m.cs);
                    };
                    const int mmq_gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42;
                    const int mmq_dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
                    // --peer-device: MMQ only, whether or not the peer took the prompt path (set_peer can decline), as
                    // fused_ring() sized the ring and the buffers for it
                    const bool no_peer = !core::peer_portable();
                    const bool fused_only = mmq_plan().any && mmq_plan().fo[(size_t) l];
                    const bool fused_nat = (use_mmq || fused_only) && stream_all && no_peer && lay.native && fused::native_supported(mmq_gt, mmq_dt);
                    const bool fused_l = (use_mmq && stream_all && no_peer && !lay.native && fused::enabled()) || fused_nat;
                    // #583 / #954: the fused layout's GU/H/Xq hold the grouping tables and the int8 rows, sized for
                    // stream_all_min() - 1 tokens of MMQ's rows, so a layer that takes MMQ or the FP16 path at the FULL
                    // chunk would write T*K rows into a (stream_all_min() - 1)*K-row buffer: an illegal access, or a kernel
                    // that never returns (the first ck() to see it is `prefill mmq: iota`, one prompt later).  fused_layout()
                    // and fused_ring() keep the shrink and this decision in step; this is the backstop for anything that
                    // drifts: a clear error, never a hang.
                    // STRATA_DBG_FORCE_SLOW_LAYER=1 (a test of the backstop): layer 0 counts as one the fused path does not take
                    static const bool force_slow = [] { const char* e = std::getenv("STRATA_DBG_FORCE_SLOW_LAYER"); return e && e[0] == '1'; }();
                    if ((!fused_l || (force_slow && l == 0)) && m.fused_bufs && T >= stream_all_min()) {
                        err = "prefill: the fused layout's MoE buffers are too small for layer " + std::to_string(l) +
                              "'s expert path at a chunk of " + std::to_string(T) + " tokens";
                        return false;
                    }
                    size_t n_order = 0;                   // the routed experts (the debug report; unknown when fused)
                    bool peer_now = false;                // multi-GPU: the peer computed rows of this layer (MMQ path only)
                    if (fused_l) {
                        if (static bool said = false; !said) {
                            said = true;
                            std::fprintf(stderr, "strata: prompt experts on the fused int8 kernels (STRATA_PF_FUSED=1, "
                                                 "#136)\n");
                        }
                        pt.mark(kPfGather, cs);
                        // the layer's input to int8 once per token; the rows of each expert from the router's ids
                        if (fused_nat) fused::quantize_act_native(m.mixed, T, N, m.Xq, m.cs);
                        else fused::quantize_act(m.mixed, T, N, m.Xq, m.cs);
                        fused::group(m.ids, T * K, (int) K, (int) m.g->n_expert, m.GU, m.slot_dev, m.src_dev, m.cs);
                        // launches over the experts in id order, each at most kMaxBatch experts of which at most a
                        // third of the ring streamed: the next batch's blobs arrive while one computes.  An expert
                        // the routing did not pick has no tiles; its ring slot is given back with its batch.
                        const size_t per = (size_t) std::max(1, m.ring / 3);
                        size_t k = seq_start[(size_t) l];
                        const size_t kend = seq_start[(size_t) l + 1];
                        for (int32_t e = 0; e < m.g->n_expert;) {
                            fused::Batch b;
                            b.e0 = e;
                            const size_t k0 = k;
                            for (; e < m.g->n_expert && e - b.e0 < fused::kMaxBatch; ++e) {
                                if (k < kend && seq[k].e == e) {
                                    if (k - k0 == per) break;
                                    b.blob[e - b.e0] = m.stage_dev[k % (size_t) m.ring];
                                    ++k;
                                } else {                  // not streamed: resident (the walk streams all others)
                                    b.blob[e - b.e0] = m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]);
                                    // counted whether routed or not (the routing stays on the GPU): at a streamed
                                    // chunk's size (>= 1024 tokens x 10 of 512) nearly every expert is routed
                                    ++stats_.experts_resident;
                                }
                            }
                            b.e1 = e;
                            if (k > k0) {
                                // one copy stream, in order: the batch's last blob covers the others
                                pt.mark(kPfWaitCopy, cs);
                                wait_issued(k - 1);
                                (m.cs)->ext_oneapi_submit_barrier(
                                    {*m.copied[(k - 1) % (size_t)m.ring]});
                            }
                            pt.mark(kPfGemmGU, cs);
                            if (fused_nat) {
                                const auto& f = lay.fmt[(size_t) l];
                                const fused::NativeGeom ng{f.gu_type, f.d_type, f.gu_row, f.d_row, f.up_off, f.down_off};
                                fused::experts_native(b, ng, (int) m.g->n_expert, T * K, m.GU, m.Xq, m.src_dev, m.H,
                                                      m.Dm, m.cs);
                            } else {
                                fused::experts(b, (int) m.g->n_expert, T * K, m.GU, m.Xq, m.src_dev, m.H, m.Dm, m.cs);
                            }
                            for (size_t kk = k0; kk < k; ++kk) {
                                dpct::sync_barrier(m.used[kk % (size_t)m.ring],
                                                   m.cs);
                                m.used_of[kk % (size_t) m.ring] = (int) (kk % (size_t) m.ring);
                            }
                            if (k > k0) {
                                consumed = k;
                                give_back(consumed);
                            }
                        }
                    } else {
                        // group the (token, k) pairs by expert on the host
                        pt.mark(kPfHostGroup, cs);
                        // (the sync below also orders this layer's writes of slot/src/bounds after the previous
                        // layer's kernels that read them)
                        const bool grp_mapped = m.grp_host != nullptr;
                        int32_t* ids_h = grp_mapped ? m.grp_host : m.ids_host.data();
                        int32_t* slot_h = grp_mapped ? m.grp_host + m.grp_tk : m.slot_host.data();
                        int32_t* src_h = grp_mapped ? m.grp_host + 2 * m.grp_tk : m.src_host.data();
                        if (grp_mapped) copy_i32(m.grp_dev, m.ids, T * K, m.cs);
                        /*
                        DPCT1124: cudaMemcpyAsync is migrated to
                        asynchronous memcpy API. While the origin API might be
                        synchronous, it depends on the type of operand memory,
                        so you may need to call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        else m.cs->memcpy(m.ids_host.data(), m.ids,
                                          (size_t)T * K * 4);
                        // #579: a stall here is the GPU (this layer's attention and router, or the previous layer's
                        // work), not the host: the watchdog's report says so (only its text changes)
                        core::progress_at("reading the prompt (batched): waiting for the GPU (attention, router) at layer",
                                          l, p0);
                        m.cs->wait();
                        core::progress_at("reading the prompt (batched): layer", l, p0);
                        pt.fold();
                        if (pe.on) {   // the peer's marks so far are done: the primary waited for its last rows
                            /*
                            DPCT1093: The "pe.dev" device may be not the one
                            intended for use. Adjust the selected device if
                            needed.
                            */
                            /*
                            DPCT1093: The "pd" device may be not the one
                            intended for use. Adjust the selected device if
                            needed.
                            */
                            int pd = 0; pd = dpct::get_current_device_id();
                                dpct::select_device(pe.dev); m.pp->s->wait();
                                pe.fold(); dpct::select_device(pd);
                        }
                        std::fill(m.cnt.begin(), m.cnt.end(), 0);
                        for (int64_t i = 0; i < T * K; ++i) {
                            const int32_t e = ids_h[(size_t) i];
                            if (e < 0 || e >= m.g->n_expert) { err = "prefill: routed id out of range"; return false; }
                            ++m.cnt[(size_t) e];
                        }
                        // multi-GPU: the rows of the experts the peer computes go last, as one block [rows_local, T*K)
                        const bool pre_mmq = mmq_plan().any && mmq_plan().layer[(size_t) l];
                        std::vector<char> on_peer;
                        int64_t rows_local = T * K, rows_peer = 0;
                        if (m.pp && pre_mmq) {
                            on_peer.assign((size_t) m.g->n_expert, 0);
                            for (int32_t e = 0; e < m.g->n_expert; ++e) {
                                const int32_t c = m.cnt[(size_t) e];
                                if (c > 0 && !m.pp->ps_flag.empty() && m.pp->ps_flag[(size_t) l * m.g->n_expert + e]) {   // peer-streamed
                                    on_peer[(size_t) e] = 2;
                                    rows_peer += c;
                                    continue;
                                }
                                if (c == 0 || (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) ||
                                    !m.pp->peer || !m.pp->peer->has(l, e))
                                    continue;
                                if (rows_peer + c > m.pp->cap_rows) { ++m.pp->over_cap; continue; }
                                on_peer[(size_t) e] = 1;
                                rows_peer += c;
                            }
                            rows_local = T * K - rows_peer;
                        }
                        {
                            int32_t r = 0;
                            for (int32_t e = 0; e < m.g->n_expert; ++e)
                                if (on_peer.empty() || !on_peer[(size_t) e]) { m.off[(size_t) e] = r; r += m.cnt[(size_t) e]; }
                            // the peer's rows in order_peer's order (peer-held, then peer-streamed): its groups need them contiguous
                            for (int kind = 1; kind <= 2 && !on_peer.empty(); ++kind)
                                for (int32_t e = 0; e < m.g->n_expert; ++e)
                                    if (on_peer[(size_t) e] == kind) { m.off[(size_t) e] = r; r += m.cnt[(size_t) e]; }
                            m.off[(size_t) m.g->n_expert] = r;
                        }
                        std::vector<int32_t> fill(m.off.begin(), m.off.end() - 1);
                        for (int64_t i = 0; i < T * K; ++i) {
                            const int32_t e = ids_h[(size_t) i];
                            const int32_t p = fill[(size_t) e]++;
                            slot_h[(size_t) i] = p;
                            src_h[(size_t) p] = (int32_t) (i / K);
                        }
                        // the experts, in id order: resident ones from VRAM, the others through the staging ring
                        std::vector<int32_t> order, order_peer;
                        for (int32_t e = 0; e < m.g->n_expert; ++e)
                            if (m.cnt[(size_t) e] > 0 && (on_peer.empty() || on_peer[(size_t) e] != 2))
                                (!on_peer.empty() && on_peer[(size_t) e] ? order_peer : order).push_back(e);
                        for (int32_t e = 0; e < m.g->n_expert && !on_peer.empty(); ++e)   // the peer-streamed ones last:
                            if (on_peer[(size_t) e] == 2) order_peer.push_back(e);        // their copies get the most time
                        // Aurora (STRATA_MMQ_RESIDENT_SORT_NE=1, opt-in): a layer whose experts are ALL resident and
                        // run through MMQ groups takes them in row-count order, so each 16-expert group's max_rows
                        // (its padded tile rows) is close to its experts' own; slot/src/off are rebuilt coherently
                        // before the upload, the routed ids, weights and the within-expert row order stay as they are.
                        if (detail::mmq_resident_sort_eligible(detail::mmq_resident_sort_requested() && on_peer.empty(), use_mmq, lay.native,
                                m.cache != nullptr, m.host_res ? m.host_res + (size_t) l * m.g->n_expert : nullptr,
                                m.g->n_expert, stream_all, !stream_all || seq_start[(size_t) l] == seq_start[(size_t) l + 1]))
                            detail::mmq_resident_sort_rows(ids_h, T * K, (int32_t) K, m.cnt, m.off, order, slot_h, src_h);
                        if (grp_mapped) {
                            copy_i32(m.slot_dev, m.grp_dev + m.grp_tk, T * K, m.cs);
                            copy_i32(m.src_dev, m.grp_dev + 2 * m.grp_tk, T * K, m.cs);
                        } else {
                            /*
                            DPCT1124: cudaMemcpyAsync is migrated to
                            asynchronous memcpy API. While the origin API might
                            be synchronous, it depends on the type of operand
                            memory, so you may need to call wait() on event
                            return by memcpy API to ensure synchronization
                            behavior.
                            */
                            m.cs->memcpy(m.slot_dev, m.slot_host.data(),
                                         (size_t)T * K * 4);
                            /*
                            DPCT1124: cudaMemcpyAsync is migrated to
                            asynchronous memcpy API. While the origin API might
                            be synchronous, it depends on the type of operand
                            memory, so you may need to call wait() on event
                            return by memcpy API to ensure synchronization
                            behavior.
                            */
                            m.cs->memcpy(m.src_dev, m.src_host.data(),
                                         (size_t)T * K * 4);
                        }
                        n_order = order.size();
                        const size_t mmq_gub = use_mmq ? mmq::matrix_bytes(mmq_gt, 1280, N) : 0;
                        const size_t mmq_db = use_mmq ? mmq::matrix_bytes(mmq_dt, N, 640) : 0;
                        pt.mark(kPfGather, cs);
                        if (use_mmq) {
                            // step 2b: the layer's activations as q8_1 rows in expert order, straight from `mixed`
                            mmq::quantize(m.mixed, m.src_dev, m.Xq, mmq_gt, N, N, T * K, m.cs);
                            // each group's rows: absolute bounds (gate/up reads the layer's rows), relative ones (down
                            // reads the group's own quantized H)
                            const size_t n = order.size(), ng = (n + MMQ_GROUP - 1) / MMQ_GROUP;
                            m.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
                            for (size_t j = 0; j < n; ++j) m.bounds_host[j] = m.off[(size_t) order[j]];
                            m.bounds_host[n] = (int32_t) rows_local;
                            for (size_t g = 0; g < ng; ++g)
                                for (size_t i = 0; i <= MMQ_GROUP; ++i)
                                    m.bounds_host[n + 1 + g * (MMQ_GROUP + 1) + i] =
                                        m.bounds_host[std::min(n, g * MMQ_GROUP + i)] - m.bounds_host[g * MMQ_GROUP];
                            if (grp_mapped) {
                                int32_t* bh = m.grp_host + 3 * m.grp_tk;
                                std::memcpy(bh, m.bounds_host.data(), m.bounds_host.size() * 4);
                                copy_i32(m.bounds_dev, m.grp_dev + 3 * m.grp_tk, (int64_t) m.bounds_host.size(), m.cs);
                            } else {
                                /*
                                DPCT1124: cudaMemcpyAsync is migrated to
                                asynchronous memcpy API. While the origin API
                                might be synchronous, it depends on the type of
                                operand memory, so you may need to call wait()
                                on event return by memcpy API to ensure
                                synchronization behavior.
                                */
                                m.cs->memcpy(m.bounds_dev, m.bounds_host.data(),
                                             m.bounds_host.size() * 4);
                            }
                        } else {
                            gather_rows16(m.mixed_h, m.src_dev, m.Xs, T * K, N, m.cs);
                            // APR: Xs is this layer's private gathered copy (rotated in fp16 storage, fp32 arithmetic)
                            if (had_gu) strata::kernels::hadamard_rows_f16(m.Xs, T * K, N, had_block, had_sx, m.cs);
                        }
                        // multi-GPU: the peer's share, enqueued before the primary's own experts so both cards work at once
                        peer_now = use_mmq && !order_peer.empty();
                        if (peer_now) {
                            PeerPrefill& P = *m.pp;
                            /*
                            DPCT1124: cudaMemcpyAsync is migrated to
                            asynchronous memcpy API. While the origin API might
                            be synchronous, it depends on the type of operand
                            memory, so you may need to call wait() on event
                            return by memcpy API to ensure synchronization
                            behavior.
                            */
                            if (!P.p2p) m.cs->memcpy(P.host_x, m.mixed,
                                                     (size_t)T * N * 4);
                            dpct::sync_barrier(P.ev_in, m.cs);
                            int prevd = 0;
                            prevd = dpct::get_current_device_id();
                            /*
                            DPCT1093: The "P.dev" device may be not the one
                            intended for use. Adjust the selected device if
                            needed.
                            */
                            dpct::select_device(P.dev);
                            const dpct::queue_ptr ps = P.s;
                            ps->ext_oneapi_submit_barrier({*P.ev_in});
                            if (P.out_pending) {
                                ps->ext_oneapi_submit_barrier({*P.ev_done});
                                P.out_pending = false;
                            }
                            pe.mark(kPeMoeIn, ps);
                            /*
                            DPCT1124: cudaMemcpyPeerAsync is migrated to
                            asynchronous memcpy API. While the origin API might
                            be synchronous, it depends on the type of operand
                            memory, so you may need to call wait() on event
                            return by memcpy API to ensure synchronization
                            behavior.
                            */
                            if (P.p2p) dpct::async_dpct_memcpy(
                                P.mixed, P.dev, m.mixed, prevd,
                                (size_t)T * N * 4, *ps);
                            else copy_f32_wide(P.mixed, P.host_x, T * N, ps);
                            P.back_at = rows_local;
                            P.back_rows = rows_peer;
                            if (P.p2p) {
                                /*
                                DPCT1124: cudaMemcpyAsync is migrated to
                                asynchronous memcpy API. While the origin API
                                might be synchronous, it depends on the type of
                                operand memory, so you may need to call wait()
                                on event return by memcpy API to ensure
                                synchronization behavior.
                                */
                                ps->memcpy(
                                    P.src,
                                    src_h + rows_local, // multi-GPU: the mapped
                                                        // table when on
                                    (size_t)rows_peer * 4);
                            } else {
                                std::memcpy(P.host_src, src_h + rows_local, (size_t) rows_peer * 4);
                                copy_i32(P.src, P.host_src, rows_peer, ps);
                            }
                            const size_t n = order_peer.size();
                            if (P.compact) {
                                // groups of up to MMQ_GROUP experts and at most G rows, each computed from its own rows
                                P.groups.clear();
                                {
                                    size_t g0 = 0;
                                    int64_t gr = 0;
                                    for (size_t j = 0; j < n; ++j) {
                                        const int64_t c = m.cnt[(size_t) order_peer[j]];
                                        if (j > g0 && (j - g0 == (size_t) MMQ_GROUP || gr + c > P.G)) { P.groups.push_back({g0, j}); g0 = j; gr = 0; }
                                        gr += c;
                                    }
                                    if (n > g0) P.groups.push_back({g0, n});
                                }
                                const size_t ng = P.groups.size();
                                P.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
                                for (size_t j = 0; j < n; ++j) P.bounds_host[j] = m.off[(size_t) order_peer[j]] - (int32_t) rows_local;
                                P.bounds_host[n] = (int32_t) rows_peer;
                                for (size_t g2 = 0; g2 < ng; ++g2)
                                    for (size_t i = 0; i <= MMQ_GROUP; ++i)
                                        P.bounds_host[n + 1 + g2 * (MMQ_GROUP + 1) + i] =
                                            P.bounds_host[std::min(P.groups[g2].second, P.groups[g2].first + i)] -
                                            P.bounds_host[P.groups[g2].first];
                                if (P.p2p || P.bounds_host.size() > PeerPrefill::kHostBounds) {
                                    /*
                                    DPCT1124: cudaMemcpyAsync is migrated
                                    to asynchronous memcpy API. While the origin
                                    API might be synchronous, it depends on the
                                    type of operand memory, so you may need to
                                    call wait() on event return by memcpy API to
                                    ensure synchronization behavior.
                                    */
                                    ps->memcpy(P.bounds, P.bounds_host.data(),
                                               P.bounds_host.size() * 4);
                                } else {
                                    std::memcpy(P.host_bounds, P.bounds_host.data(), P.bounds_host.size() * 4);
                                    copy_i32(P.bounds, P.host_bounds, (int64_t) P.bounds_host.size(), ps);
                                }
                                pe.mark(kPeMoeGemm, ps);
                                const auto& f = lay.fmt[(size_t) l];
                                for (size_t g2 = 0; g2 < ng; ++g2) {
                                    const size_t j0 = P.groups[g2].first, j1 = P.groups[g2].second;
                                    const int ngx = (int) (j1 - j0);
                                    int64_t maxr = 0;
                                    for (size_t j = j0; j < j1; ++j) {
                                        const int32_t e = order_peer[j];
                                        maxr = std::max<int64_t>(maxr, m.cnt[(size_t) e]);
                                        const uint8_t* bd = nullptr;
                                        int psl = -1;
                                        if (on_peer[(size_t) e] == 2) {   // from the peer's ring
                                            p_release_to(l, e, ps);
                                            if (P.pk >= P.pseq_start[(size_t) l + 1] || P.pseq[P.pk].e != e) {
                                                err = "prefill: a peer-streamed expert is not next in the peer's ring";
                                                /*
                                                DPCT1093: The "prevd" device
                                                may be not the one intended for
                                                use. Adjust the selected device
                                                if needed.
                                                */
                                                dpct::select_device(prevd);
                                                return false;
                                            }
                                            psl = (int) (P.pk % (size_t) P.RP);
                                            ps->ext_oneapi_submit_barrier(
                                                {*P.pcopied[(size_t)psl]});
                                            bd = P.pstage[(size_t) psl];
                                            ++P.ps_experts;
                                        } else {
                                            bd = P.peer->slot_ptr(l, e);
                                        }
                                        const size_t q = j - j0;
                                        if (lay.native)
                                            mmq::gather_native(bd, bd + f.up_off, mmq_gub / 2, bd + f.down_off, mmq_db,
                                                               P.grp_gu + q * mmq_gub, P.grp_d + q * mmq_db, ps);
                                        else
                                            mmq::gather_strata_q2(bd, P.grp_gu + q * mmq_gub, P.grp_d + q * mmq_db, ps);
                                        if (psl >= 0) {   // the slot is free once gathered
                                            dpct::sync_barrier(
                                                P.pused[(size_t)psl], ps);
                                            ++P.pk;
                                            p_issue_until(P.pk + (size_t) P.RP);
                                        }
                                    }
                                    const int64_t r0 = P.bounds_host[j0], nr = P.bounds_host[j1] - r0;
                                    if (nr <= 0) continue;
                                    const int32_t* rel = P.bounds + n + 1 + g2 * (MMQ_GROUP + 1);
                                    ps->memset(P.grp_gu + (size_t)ngx * mmq_gub,
                                               0, MMQ_TAIL);
                                    ps->memset(P.grp_d + (size_t)ngx * mmq_db,
                                               0, MMQ_TAIL);
                                    mmq::quantize(P.mixed, P.src + r0, P.Xq_g, mmq_gt, N, N, nr, ps);
                                    mmq::Product gu;
                                    gu.w = P.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
                                    gu.n = ngx; gu.xq = P.Xq_g; gu.bounds = rel; gu.ids = P.ident;
                                    gu.total_rows = nr; gu.max_rows = maxr; gu.dst = P.GU_g; gu.ld_dst = 1280;
                                    P.run_ctx->run(gu, ps);
                                    mmq::swiglu(P.GU_g, P.H_g, nr, 640, !lay.native, ps);
                                    mmq::quantize(P.H_g, nullptr, P.Hq_g, mmq_dt, 640, 640, nr, ps);
                                    const int b = (int) (g2 & 1);
                                    if (P.dm_live[b])
                                        ps->ext_oneapi_submit_barrier(
                                            {*P.ev_dm[b]}); // its last rows
                                                            // have left
                                    mmq::Product dn;
                                    dn.w = P.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
                                    dn.n = ngx; dn.xq = P.Hq_g; dn.bounds = rel;
                                    dn.ids = P.ident; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = P.Dm_b[b];
                                    dn.ld_dst = N;
                                    P.run_ctx->run(dn, ps);
                                    dpct::sync_barrier(
                                        P.ev_grp[g2 % PeerPrefill::kGrpEv], ps);
                                    (P.s_out)->ext_oneapi_submit_barrier(
                                        {*P.ev_grp[g2 % PeerPrefill::kGrpEv]});
                                    if (P.p2p)
                                        /*
                                        DPCT1124: cudaMemcpyPeerAsync is
                                        migrated to asynchronous memcpy API.
                                        While the origin API might be
                                        synchronous, it depends on the type of
                                        operand memory, so you may need to call
                                        wait() on event return by memcpy API to
                                        ensure synchronization behavior.
                                        */
                                        dpct::async_dpct_memcpy(
                                            m.Dm +
                                                (size_t)(rows_local + r0) * N,
                                            prevd, P.Dm_b[b], P.dev,
                                            (size_t)nr * N * 4, *(P.s_out));
                                    else   // the primary reads them when it combines (below)
                                        /*
                                        DPCT1124: cudaMemcpyAsync is
                                        migrated to asynchronous memcpy API.
                                        While the origin API might be
                                        synchronous, it depends on the type of
                                        operand memory, so you may need to call
                                        wait() on event return by memcpy API to
                                        ensure synchronization behavior.
                                        */
                                        P.s_out->memcpy(
                                            P.host_rows + (size_t)r0 * N,
                                            P.Dm_b[b], (size_t)nr * N * 4);
                                    dpct::sync_barrier(P.ev_dm[b], P.s_out);
                                    P.dm_live[b] = true;
                                }
                            } else {
                                const size_t ng = (n + MMQ_GROUP - 1) / MMQ_GROUP;
                                P.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
                                for (size_t j = 0; j < n; ++j) P.bounds_host[j] = m.off[(size_t) order_peer[j]] - (int32_t) rows_local;
                                P.bounds_host[n] = (int32_t) rows_peer;
                                for (size_t g2 = 0; g2 < ng; ++g2)
                                    for (size_t i = 0; i <= MMQ_GROUP; ++i)
                                        P.bounds_host[n + 1 + g2 * (MMQ_GROUP + 1) + i] =
                                            P.bounds_host[std::min(n, g2 * MMQ_GROUP + i)] - P.bounds_host[g2 * MMQ_GROUP];
                                if (P.p2p || P.bounds_host.size() > PeerPrefill::kHostBounds) {
                                    /*
                                    DPCT1124: cudaMemcpyAsync is migrated
                                    to asynchronous memcpy API. While the origin
                                    API might be synchronous, it depends on the
                                    type of operand memory, so you may need to
                                    call wait() on event return by memcpy API to
                                    ensure synchronization behavior.
                                    */
                                    ps->memcpy(P.bounds, P.bounds_host.data(),
                                               P.bounds_host.size() * 4);
                                } else {
                                    std::memcpy(P.host_bounds, P.bounds_host.data(), P.bounds_host.size() * 4);
                                    copy_i32(P.bounds, P.host_bounds, (int64_t) P.bounds_host.size(), ps);
                                }
                                mmq::quantize(P.mixed, P.src, P.Xq, mmq_gt, N, N, rows_peer, ps);
                                pe.mark(kPeMoeGemm, ps);
                                const auto& f = lay.fmt[(size_t) l];
                                for (size_t j = 0; j < n; ++j) {
                                    const int32_t e = order_peer[j];
                                    const uint8_t* bd = P.peer->slot_ptr(l, e);
                                    const size_t q = j % MMQ_GROUP;
                                    if (lay.native)
                                        mmq::gather_native(bd, bd + f.up_off, mmq_gub / 2, bd + f.down_off, mmq_db,
                                                           P.grp_gu + q * mmq_gub, P.grp_d + q * mmq_db, ps);
                                    else
                                        mmq::gather_strata_q2(bd, P.grp_gu + q * mmq_gub, P.grp_d + q * mmq_db, ps);
                                    if (q + 1 < MMQ_GROUP && j + 1 < n) continue;
                                    const size_t j0 = j - q, g2 = j0 / MMQ_GROUP;
                                    const int ngx = (int) (q + 1);
                                    const int64_t r0 = P.bounds_host[j0], nr = P.bounds_host[j + 1] - r0;
                                    int64_t maxr = 0;
                                    for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order_peer[i]]);
                                    ps->memset(P.grp_gu + (size_t)ngx * mmq_gub,
                                               0, MMQ_TAIL);
                                    ps->memset(P.grp_d + (size_t)ngx * mmq_db,
                                               0, MMQ_TAIL);
                                    mmq::Product gu;
                                    gu.w = P.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
                                    gu.n = ngx; gu.xq = P.Xq; gu.bounds = P.bounds + j0; gu.ids = P.ident;
                                    gu.total_rows = rows_peer; gu.max_rows = maxr; gu.dst = P.GU; gu.ld_dst = 1280;
                                    P.run_ctx->run(gu, ps);
                                    mmq::swiglu(P.GU + r0 * 1280, P.H + r0 * 640, nr, 640, !lay.native, ps);
                                    mmq::quantize(P.H + r0 * 640, nullptr, P.Hq, mmq_dt, 640, 640, nr, ps);
                                    mmq::Product dn;
                                    dn.w = P.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
                                    dn.n = ngx; dn.xq = P.Hq; dn.bounds = P.bounds + n + 1 + g2 * (MMQ_GROUP + 1);
                                    dn.ids = P.ident; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = P.Dm + r0 * N;
                                    dn.ld_dst = N;
                                    P.run_ctx->run(dn, ps);
                                    if (P.out_pipe && nr > 0) {   // this group's rows go back while the next group computes
                                        dpct::sync_barrier(P.ev_grp[g2], ps);
                                        (P.s_out)->ext_oneapi_submit_barrier(
                                            {*P.ev_grp[g2]});
                                        /*
                                        DPCT1124: cudaMemcpyPeerAsync is
                                        migrated to asynchronous memcpy API.
                                        While the origin API might be
                                        synchronous, it depends on the type of
                                        operand memory, so you may need to call
                                        wait() on event return by memcpy API to
                                        ensure synchronization behavior.
                                        */
                                        dpct::async_dpct_memcpy(
                                            m.Dm +
                                                (size_t)(rows_local + r0) * N,
                                            prevd, P.Dm + (size_t)r0 * N, P.dev,
                                            (size_t)nr * N * 4, *(P.s_out));
                                    }
                                }
                            }
                            pe.mark(kPeMoeOut, ps);
                            if (P.out_pipe) {
                                dpct::sync_barrier(P.ev_done, P.s_out);
                                P.out_pending = true;
                            } else {
                                /*
                                DPCT1124: cudaMemcpyPeerAsync is migrated to
                                asynchronous memcpy API. While the origin API
                                might be synchronous, it depends on the type of
                                operand memory, so you may need to call wait()
                                on event return by memcpy API to ensure
                                synchronization behavior.
                                */
                                dpct::async_dpct_memcpy(
                                    m.Dm + (size_t)rows_local * N, prevd, P.Dm,
                                    P.dev, (size_t)rows_peer * N * 4, *ps);
                                dpct::sync_barrier(P.ev_done, ps);
                            }
                            pe.mark(kPeIdle, ps);
                            /*
                            DPCT1093: The "prevd" device may be not the one
                            intended for use. Adjust the selected device if
                            needed.
                            */
                            dpct::select_device(prevd);
                            ++P.layers;
                            P.experts += (int64_t) n;
                            P.rows += rows_peer;
                        }
                        if (ps_on) {   // multi-GPU: this layer's peer-ring entries the routing did not pick give their slots back
                            int pd = 0;
                            pd = dpct::get_current_device_id();
                            /*
                            DPCT1093: The "m.pp->dev" device may be not the
                            one intended for use. Adjust the selected device if
                            needed.
                            */
                            dpct::select_device(m.pp->dev);
                            p_release_to(l, (int32_t) m.g->n_expert, m.pp->s);
                            /*
                            DPCT1093: The "pd" device may be not the one
                            intended for use. Adjust the selected device if
                            needed.
                            */
                            dpct::select_device(pd);
                        }
                        // Stage ahead: the copy stream moves blobs host -> device while the compute stream works.
                        int stage_next = 0;
                        std::vector<int> stage_of(order.size(), -1);
                        // the unpinned ones are copied to pinned buffers by the stager's threads, in this order
                        std::vector<int> job_of(order.size(), -1);
                        if (!stream_all) {
                            std::vector<Stager::Job> js;
                            for (size_t j = 0; j < order.size(); ++j) {
                                const int32_t e = order[j];
                                if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                                { const uint8_t* pb = m.src->pinned(l, e) ? m.src->blob(l, e) : nullptr; if (dma_direct(m.src, l, e, pb)) continue; }
                                job_of[j] = (int) js.size();
                                if (gsrc != nullptr) {   // SYCL port: --stream-experts - the stager's thread reads it from the GGUF
                                js.push_back({nullptr, (size_t) lay.blob_bytes(l), nullptr, (int32_t) l, e, gsrc});
                                continue;
                            }
                            if (m.src->transient(l, e)) {   // CS-T: copied by the source
                                    js.push_back({nullptr, (size_t) lay.blob_bytes(l), m.src, (int32_t) l, e});
                                    continue;
                                }
                                const uint8_t* b = m.src->blob(l, e);
                                if (!b) { err = "prefill: expert source has no blob"; return false; }
                                js.push_back({b, (size_t) lay.blob_bytes(l)});
                            }
                            m.stager->start(std::move(js));
                        }
                        StagerDone stager_done{stream_all ? nullptr : m.stager.get()};
                        auto stage_one = [&](size_t j) -> bool {
                            try {
                        const int32_t e = order[j];
                            const bool resident = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                            if (resident) return true;
                            const int sl = stage_next;
                            stage_next = (stage_next + 1) % stage_slots();
                            const auto th = Clock::now();
                            const uint8_t* pb0 = m.src->pinned(l, e) ? m.src->blob(l, e) : nullptr;
                            const bool pinned = dma_direct(m.src, l, e, pb0);   // pinned: never transient
                            const uint8_t* b = pinned ? m.src->blob(l, e) : nullptr;
                            if (pinned && !b) { err = "prefill: expert source has no blob"; return false; }
                            if (pinned) {
                                // DMA straight from the page-locked arena: the copy stream only waits for the slot
                                if (m.stage_live[sl])(m.copy)
                                    ->ext_oneapi_submit_barrier(
                                        {*m.used[m.used_of[sl]]});
                                /*
                                DPCT1124: cudaMemcpyAsync is migrated to
                                asynchronous memcpy API. While the origin API
                                might be synchronous, it depends on the type of
                                operand memory, so you may need to call wait()
                                on event return by memcpy API to ensure
                                synchronization behavior.
                                */
                                m.copy->memcpy(m.stage_dev[sl], b,
                                               (size_t)lay.blob_bytes(l));
                                ++stats_.experts_dma;
                            } else {
                                // copied to a pinned buffer by the stager (waits only if it is behind), then DMA
                                const uint8_t* hb = m.stager->wait(job_of[j]);
                                if (m.stage_live[sl])(m.copy)
                                    ->ext_oneapi_submit_barrier(
                                        {*m.used[m.used_of[sl]]});
                                /*
                                DPCT1124: cudaMemcpyAsync is migrated to
                                asynchronous memcpy API. While the origin API
                                might be synchronous, it depends on the type of
                                operand memory, so you may need to call wait()
                                on event return by memcpy API to ensure
                                synchronization behavior.
                                */
                                m.copy->memcpy(m.stage_dev[sl], hb,
                                               (size_t)lay.blob_bytes(l));
                                m.stager->issued_one(job_of[j], m.copy);
                            }
                            dpct::sync_barrier(m.copied[sl], m.copy);
                            m.stage_live[sl] = true;
                            stage_of[j] = sl;
                            stats_.ms_experts_host += ms_since(th);
                            ++stats_.experts_streamed;
                            return true;
                        }
                        catch (sycl::exception const &exc) {
                          std::cerr << exc.what()
                                    << "Exception caught at file:" << __FILE__
                                    << ", line:" << __LINE__ << std::endl;
                          std::exit(1);
                        }
                        };
                        // In the streamed walk an MMQ group is gathered in ONE launch, after ONE wait on its last
                        // streamed copy (the copy stream is in order), and its ring slots are released by ONE event.  A
                        // wait and a record per expert put ~10 us of GPU bubble each on the compute stream under WDDM
                        // (226 ms of a 32K prompt on an NVFP4 pack).  The same bytes into the same group slots.  A ring
                        // entry the routing skipped inside an open group first gathers what the group holds so far
                        // (`flush`), so no more than a group's entries are ever held back from the issuer.
                        // STRATA_PREFILL_GROUP_GATHER=0: one gather, one wait and one record per expert.
                        static const bool group_env = [] {
                            const char* v = std::getenv("STRATA_PREFILL_GROUP_GATHER");
                            return v == nullptr || std::atoi(v) != 0;
                        }();
                        const bool group_gather = group_env && stream_all && use_mmq && lay.native &&
                                                  MMQ_GROUP <= mmq::kGatherGroupMax;
                        mmq::GatherGroup gg;
                        int gg_slots[MMQ_GROUP];
                        int gg_nslots = 0;   // ring slots gathered by the next flush
                        auto flush = [&]() {
                            try {
                        if (gg.n <= gg.first) return;
                            const auto& f = lay.fmt[(size_t) l];
                            if (gg_nslots > 0) {   // the copies land in order: the last one covers the others
                                pt.mark(kPfWaitCopy, cs);
                                (m.cs)->ext_oneapi_submit_barrier(
                                    {*m.copied[gg_slots[gg_nslots - 1]]});
                                pt.mark(kPfDequant, cs);
                            }
                            if (!mmq::gather_native_group(gg, f.up_off, mmq_gub / 2, f.down_off, mmq_db, m.grp_gu, mmq_gub,
                                                          m.grp_d, mmq_db, m.cs)) {
                                for (int i = gg.first; i < gg.n; ++i) {   // not 16-byte aligned: one at a time
                                    const uint8_t* b = gg.blob[i];
                                    mmq::gather_native(b, b + f.up_off, mmq_gub / 2, b + f.down_off, mmq_db,
                                                       m.grp_gu + i * mmq_gub, m.grp_d + i * mmq_db, m.cs);
                                }
                            }
                            if (gg_nslots > 0) {
                                const int rel = gg_slots[gg_nslots - 1];
                                dpct::sync_barrier(m.used[rel], m.cs);
                                for (int i = 0; i < gg_nslots; ++i) m.used_of[gg_slots[i]] = rel;
                            }
                            gg_nslots = 0;
                            gg.first = gg.n;
                        }
                        catch (sycl::exception const &exc) {
                          std::cerr << exc.what()
                                    << "Exception caught at file:" << __FILE__
                                    << ", line:" << __LINE__ << std::endl;
                          std::exit(1);
                        }
                        };
                        // A770 port, STRATA_PF_XMX=2 (streamed walk, native pack, FP16 path): the experts' quantized
                        // blobs are held until a group of kXmxGroupMax, then each product runs over the group in one
                        // launch that decodes the weights itself.  Their ring slots are released by ONE event after
                        // the group's down product, and nothing is given back to the issuer while any are held.
                        // (the formats the parity test covers: gate/up IQ3_XXS / IQ3_S / IQ2_S / IQ4_XS, down IQ4_NL / Q2_0;
                        // a layer in another format keeps the dequantize + oneMKL path)
                        auto xmx2_fmt = [&]() {
                            if (!lay.native) return false;
                            const int gt = lay.fmt[(size_t) l].gu_type, dt = lay.fmt[(size_t) l].d_type;
                            if ((gt == 144 || gt == 145) && (dt == 144 || dt == 146)) return true;   // Gyro's trellis experts (rotated above)
                            return (gt == 18 || gt == 21 || gt == 22 || gt == 23) && (dt == 20 || dt == 42);
                        };
                        const bool xmx2_only = strata::prefill::xmx::mode() == 2 && !use_mmq && xmx2_fmt();
                        // 151-b70: the fused int8 DPAS products (IQ3_S gate/up, IQ4_NL down, the 640 x 2560 expert), the same
                        // held group of staged experts as above
                        const int i8m = i8_mode(m.cs);
                        const bool i8on = !xmx2_only && i8m != 0 && !use_mmq && lay.native && !had_gu && !had_d &&
                                          (lay.fmt[(size_t) l].gu_type == 21 || lay.fmt[(size_t) l].gu_type == 18 || lay.fmt[(size_t) l].gu_type == 22) &&
                                          (lay.fmt[(size_t) l].d_type == 20 || lay.fmt[(size_t) l].d_type == 42) &&
                                          lay.fmt[(size_t) l].n_ff == 640 && lay.fmt[(size_t) l].n_embd == 2560 &&
                                          strata::prefill::i8::available(m.cs);
                        const bool xmx2 = xmx2_only || i8on;
                        if (i8_env() != nullptr && i8m != 0 && lay.native) {
                            static std::set<std::tuple<int, int, int, int, int>> said_fmt;
                            const auto& ff = lay.fmt[(size_t) l];
                            if (said_fmt.insert({ff.gu_type, ff.d_type, (int) ff.n_ff, (int) ff.n_embd, (int) i8on}).second)
                                std::fprintf(stderr, "strata: layer %lld experts gate/up type %d, down type %d, n_ff %d, n_embd %d -> int8 path %s\n",
                                             (long long) l, ff.gu_type, ff.d_type, (int) ff.n_ff, (int) ff.n_embd, i8on ? "yes" : "no");
                        }
                        if (strata::prefill::xmx::mode() == 2 && !xmx2) {
                            static bool said = false;
                            if (!said) {
                                said = true;
                                std::fprintf(stderr, "strata: STRATA_PF_XMX=2 not taken at layer %lld (stream_all %d, mmq %d, "
                                                     "native %d, formats %d/%d)\n", (long long) l, (int) stream_all,
                                             (int) use_mmq, (int) lay.native, lay.native ? lay.fmt[(size_t) l].gu_type : -1,
                                             lay.native ? lay.fmt[(size_t) l].d_type : -1);
                            }
                        }
                        const uint8_t* xg_g[strata::kernels::kXmxGroupMax];
                        const uint8_t* xg_u[strata::kernels::kXmxGroupMax];
                        const uint8_t* xg_d[strata::kernels::kXmxGroupMax];
                        int32_t xg_c[strata::kernels::kXmxGroupMax];
                        int xg_slots[strata::kernels::kXmxGroupMax];
                        int xg_n = 0, xg_ns = 0;
                        int64_t xg_r0 = 0;
                        auto xflush = [&]() {
                            if (xg_n == 0) return;
                            const auto& f = lay.fmt[(size_t) l];
                            int64_t nr = 0;
                            for (int i = 0; i < xg_n; ++i) nr += xg_c[i];
                            if (i8on) {
                                // STRATA_PREFILL_MMQ=1: always; =auto: while the group's mean rows per expert are under the crossover
                                // measured on this GPU (i8_calibrate); a group above it takes the FP16 chain below
                                bool done8 = false;
                                try {
                                bool use_i8 = true;
                                if (i8m == 2) {
                                    static std::map<std::tuple<const void*, int, int>, I8Cross> xover;   // per queue (a layer split has a card per queue)
                                    auto it = xover.find({(const void*) m.cs, f.gu_type, f.d_type});
                                    if (it == xover.end()) it = xover.emplace(std::make_tuple((const void*) m.cs, f.gu_type, f.d_type), i8_calibrate(m, f)).first;
                                    use_i8 = (double) nr / xg_n < it->second.rows;
                                    static long groups_i8 = 0, groups_fp = 0;
                                    (use_i8 ? groups_i8 : groups_fp) += 1;
                                    if (static bool said_n = false; !said_n && groups_i8 + groups_fp >= 500) {
                                        said_n = true;
                                        std::fprintf(stderr, "strata: int8 experts (auto): %ld groups took the int8 chain, %ld the FP16 chain\n", groups_i8, groups_fp);
                                    }
                                }
                                if (use_i8) {
                                    if (static bool said = false; !said) {
                                        said = true;
                                        std::fprintf(stderr, "strata: prompt experts on the fused int8 DPAS products (STRATA_PREFILL_MMQ=%s)\n",
                                                     i8m == 2 ? "auto" : "1");
                                    }
                                    namespace i8 = strata::prefill::i8;
                                    i8::Layout lo8;
                                    i8::make_layout(lo8, xg_c, xg_n);
                                    i8::Workspace ws8;
                                    pt.mark(kPfGemmGU, cs);
                                    if (i8::workspace(lo8, (int) N, f.n_ff, ws8, m.cs)) {
                                        i8::quantize_rows(m.Xs + xg_r0 * N, lo8, (int) N, ws8.xq, ws8.dx, m.cs);
                                        if (i8::gate_up(f.gu_type, xg_g, f.up_off, f.n_ff, lo8, (int) N, ws8.xq, ws8.dx, m.GU + xg_r0 * 1280, m.cs)) {
                                            i8::swiglu_quantize(m.GU + xg_r0 * 1280, lo8, f.n_ff, ws8.hq, ws8.dh, m.cs);
                                            pt.mark(kPfGemmD, cs);
                                            done8 = i8::down(f.d_type, xg_g, f.down_off, (int) N, lo8, f.n_ff, ws8.hq, ws8.dh, m.Dm + xg_r0 * N, m.cs);
                                        }
                                    }
                                    if (!done8) {
                                        if (static bool said_f = false; !said_f) {
                                            said_f = true;
                                            std::fprintf(stderr, "strata: the int8 expert products could not run (layer %lld, no workspace or a format they do not handle): the FP16 chain is used\n", (long long) l);
                                        }
                                    }
                                }
                                } catch (const std::exception& ex) {   // the kernels are not in this binary for this device, or a launch failed: the FP16 chain, for good
                                    done8 = false;
                                    if (!g_i8_failed.exchange(true))
                                        std::fprintf(stderr, "strata: the int8 expert kernels failed to launch (%s): the FP16 expert chain is used from here\n", ex.what());
                                }
                                if (!done8) {   // the FP16 chain: dequantize, two oneMKL GEMMs, SwiGLU, per expert
                                    int64_t o = xg_r0;
                                    for (int i = 0; i < xg_n; ++i) {
                                        const int qd = i % DQ;
                                        pt.mark(kPfDequant, cs);
                                        strata::kernels::iq_dequant_expert_f16(f.gu_type, f.d_type, xg_g[i], xg_u[i], xg_d[i], f.n_ff, f.n_embd,
                                                                               m.dq_gu[qd], m.dq_d[qd], m.cs);
                                        pt.mark(kPfGemmGU, cs);
                                        m.gemm.f16(m.Xs + o * N, m.dq_gu[qd], m.GU + o * 1280, xg_c[i], 1280, N);
                                        swiglu_interleaved(m.GU + o * 1280, m.Hh + o * 640, xg_c[i], m.cs);
                                        had_h(o, xg_c[i]);
                                        pt.mark(kPfGemmD, cs);
                                        m.gemm.f16(m.Hh + o * 640, m.dq_d[qd], m.Dm + o * N, xg_c[i], N, 640);
                                        o += xg_c[i];
                                    }
                                }
                                if (xg_ns > 0) {
                                    const int rel = xg_slots[xg_ns - 1];
                                    dpct::sync_barrier(m.used[rel], m.cs);
                                    for (int i = 0; i < xg_ns; ++i) m.used_of[xg_slots[i]] = rel;
                                }
                                xg_n = 0;
                                xg_ns = 0;
                                return;
                            }
                            if (static bool said = false; !said) {
                                said = true;
                                std::fprintf(stderr, "strata: prompt experts on the fused grouped XMX (STRATA_PF_XMX=2)\n");
                            }
                            // measured (xmx_group_parity, 16 experts): the fused launch wins below ~200 rows per
                            // expert, dequantize + oneMKL above (gate/up 0.67-0.76x at 320); STRATA_PF_XMX_ROWS moves it
                            static const int xg_max = [] {
                                const char* v = std::getenv("STRATA_PF_XMX_ROWS");
                                return v ? std::atoi(v) : 200;
                            }();
                            if (nr / xg_n < xg_max) {
                                pt.mark(kPfGemmGU, cs);
                                strata::kernels::iq_xmx_grouped(f.gu_type, xg_g, xg_u, xg_c, xg_n, m.Xs + xg_r0 * N,
                                                                m.GU + xg_r0 * 1280, 1280, (int) N, m.cs);
                                swiglu_interleaved(m.GU + xg_r0 * 1280, m.Hh + xg_r0 * 640, nr, m.cs);
                                had_h(xg_r0, nr);
                                pt.mark(kPfGemmD, cs);
                                strata::kernels::iq_xmx_grouped(f.d_type, xg_d, nullptr, xg_c, xg_n, m.Hh + xg_r0 * 640,
                                                                m.Dm + xg_r0 * N, (int) N, 640, m.cs);
                            } else {
                                int64_t o = xg_r0;
                                for (int i = 0; i < xg_n; ++i) {
                                    const int q = i % DQ;
                                    pt.mark(kPfDequant, cs);
                                    strata::kernels::iq_dequant_gu_f16(f.gu_type, xg_g[i], xg_u[i], f.n_ff, f.n_embd,
                                                                       m.dq_gu[q], m.cs);
                                    strata::kernels::iq_dequant_f16(f.d_type, xg_d[i], f.n_embd * f.n_ff, m.dq_d[q], m.cs);
                                    pt.mark(kPfGemmGU, cs);
                                    m.gemm.f16(m.Xs + o * N, m.dq_gu[q], m.GU + o * 1280, xg_c[i], 1280, N);
                                    swiglu_interleaved(m.GU + o * 1280, m.Hh + o * 640, xg_c[i], m.cs);
                                    had_h(o, xg_c[i]);
                                    pt.mark(kPfGemmD, cs);
                                    m.gemm.f16(m.Hh + o * 640, m.dq_d[q], m.Dm + o * N, xg_c[i], N, 640);
                                    o += xg_c[i];
                                }
                            }
                            if (xg_ns > 0) {
                                const int rel = xg_slots[xg_ns - 1];
                                dpct::sync_barrier(m.used[rel], m.cs);
                                for (int i = 0; i < xg_ns; ++i) m.used_of[xg_slots[i]] = rel;
                            }
                            xg_n = 0;
                            xg_ns = 0;
                        };
                        // one expert's products from its blob on the device; `slot` (a ring slot, or -1 for a resident
                        // expert) is released once the blob is read
                        auto compute = [&](size_t j, const uint8_t *blob_dev,
                                           int slot) -> bool {
                            try {
                        const int32_t e = order[j];
                            pt.mark(kPfDequant, cs);
                            if (use_mmq) {
                                // gather the expert into its group slot (GGUF blocks, unchanged or converted)
                                const size_t q = j % MMQ_GROUP;
                                if (group_gather) {
                                    gg.blob[q] = blob_dev;
                                    gg.n = (int) q + 1;
                                    if (slot >= 0) gg_slots[gg_nslots++] = slot;
                                    if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
                                    flush();
                                    gg = mmq::GatherGroup{};
                                } else if (lay.native) {
                                    const auto& f = lay.fmt[(size_t) l];
                                    mmq::gather_native(blob_dev, blob_dev + f.up_off, mmq_gub / 2, blob_dev + f.down_off,
                                                       mmq_db, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs);
                                } else {
                                    mmq::gather_strata_q2(blob_dev, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs);
                                }
                                if (slot >= 0 && !group_gather) {
                                    dpct::sync_barrier(m.used[slot], m.cs);
                                    m.used_of[slot] = slot;
                                }
                                if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
                                // the group's products: gate/up, swiglu, the group's H to q8_1, down
                                const size_t j0 = j - q, g = j0 / MMQ_GROUP, n = order.size();
                                const int ngx = (int) (q + 1);
                                const int64_t r0 = m.bounds_host[j0], nr = m.bounds_host[j + 1] - r0;
                                int64_t maxr = 0;
                                for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
                                pt.mark(kPfGemmGU, cs);
                                // the zeroed tail after the group's last expert (see MMQ_TAIL)
                                m.cs->memset(m.grp_gu + (size_t)ngx * mmq_gub,
                                             0, MMQ_TAIL);
                                m.cs->memset(m.grp_d + (size_t)ngx * mmq_db, 0,
                                             MMQ_TAIL);
                                mmq::Product gu;
                                gu.w = m.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
                                gu.n = ngx; gu.xq = m.Xq; gu.bounds = m.bounds_dev + j0; gu.ids = m.ids_identity;
                                gu.total_rows = T * K; gu.max_rows = maxr; gu.dst = m.GU; gu.ld_dst = 1280;
                                m.mmq_ctx->run(gu, m.cs);
                                mmq::swiglu(m.GU + r0 * 1280, m.H + r0 * 640, nr, 640, !lay.native, m.cs);
                                pt.mark(kPfGemmD, cs);
                                mmq::quantize(m.H + r0 * 640, nullptr, m.Hq, mmq_dt, 640, 640, nr, m.cs);
                                mmq::Product dn;
                                dn.w = m.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
                                dn.n = ngx; dn.xq = m.Hq; dn.bounds = m.bounds_dev + n + 1 + g * (MMQ_GROUP + 1);
                                dn.ids = m.ids_identity; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = m.Dm + r0 * N;
                                dn.ld_dst = N;
                                m.mmq_ctx->run(dn, m.cs);
                                return true;
                            }
                            if (xmx2) {
                                const auto& f = lay.fmt[(size_t) l];
                                if (xg_n == 0) xg_r0 = m.off[(size_t) e];
                                xg_g[xg_n] = blob_dev;
                                xg_u[xg_n] = blob_dev + f.up_off;
                                xg_d[xg_n] = blob_dev + f.down_off;
                                xg_c[xg_n] = m.cnt[(size_t) e];
                                ++xg_n;
                                if (slot >= 0) xg_slots[xg_ns++] = slot;
                                if (xg_n == strata::kernels::kXmxGroupMax || j + 1 == order.size()) xflush();
                                return true;
                            }
                            const bool xmx_on = strata::prefill::xmx::mode() == 1;
                            const int q = (int) (j % (size_t) dq_slots());
                            if (lay.native) {
                                // plan v0.3 P6: a native pack's layer, dequantized by llama.cpp's own formulas
                                const auto& f = lay.fmt[(size_t) l];
                                // 0.1.43: both matrices in one launch (the same values; STRATA_PF_DQ_FUSED=0: two launches)
                                static const bool dq_one = [] { const char* v = std::getenv("STRATA_PF_DQ_FUSED"); return !(v && v[0] == '0'); }();
                                if (dq_one) {
                                    strata::kernels::iq_dequant_expert_f16(f.gu_type, f.d_type, blob_dev, blob_dev + f.up_off,
                                                                           blob_dev + f.down_off, f.n_ff, f.n_embd, m.dq_gu[q],
                                                                           m.dq_d[q], m.cs);
                                } else {
                                    strata::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff, f.n_embd,
                                                                       m.dq_gu[q], m.cs);
                                    strata::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, m.dq_d[q], m.cs);
                                }
                            } else {
                                blob_dequant_f16(blob_dev, m.dq_gu[q], m.dq_d[q], m.cs);
                            }
                            if (slot >= 0) {
                                dpct::sync_barrier(m.used[slot], m.cs);
                                m.used_of[slot] = slot;
                            }
                            if (xmx_on) {
                                if (static bool said = false; !said) {
                                    said = true;
                                    std::fprintf(stderr, "strata: prompt experts on the grouped XMX GEMM (STRATA_PF_XMX=1)\n");
                                }
                                // A770 port: the group's experts sit in dq slots 0..q, their rows back to back from
                                // the first one's (the order is by id and so are the offsets)
                                if (q + 1 < dq_slots() && j + 1 < order.size()) return true;
                                const size_t j0 = j - (size_t) q;
                                const int ng = q + 1;
                                int32_t gc[strata::prefill::xmx::kMaxGroup];
                                const uint16_t* wg[strata::prefill::xmx::kMaxGroup];
                                const uint16_t* wd[strata::prefill::xmx::kMaxGroup];
                                int64_t nr = 0;
                                for (int i = 0; i < ng; ++i) {
                                    gc[i] = m.cnt[(size_t) order[j0 + (size_t) i]];
                                    wg[i] = m.dq_gu[i]; wd[i] = m.dq_d[i];
                                    nr += gc[i];
                                }
                                const int64_t r0 = m.off[(size_t) order[j0]];
                                pt.mark(kPfGemmGU, cs);
                                if (nr / ng < strata::prefill::xmx::gu_max_mean()) {
                                    strata::prefill::xmx::grouped_f16(m.Xs + r0 * N, wg, gc, ng, m.GU + r0 * 1280, 1280,
                                                                      (int) N, m.cs);
                                } else {
                                    int64_t o = r0;
                                    for (int i = 0; i < ng; ++i) {
                                        m.gemm.f16(m.Xs + o * N, wg[i], m.GU + o * 1280, gc[i], 1280, N);
                                        o += gc[i];
                                    }
                                }
                                swiglu_interleaved(m.GU + r0 * 1280, m.Hh + r0 * 640, nr, m.cs);
                                had_h(r0, nr);
                                pt.mark(kPfGemmD, cs);
                                strata::prefill::xmx::grouped_f16(m.Hh + r0 * 640, wd, gc, ng, m.Dm + r0 * N, (int) N, 640,
                                                                  m.cs);
                                return true;
                            }
                            const int64_t o0 = m.off[(size_t) e], ne = m.cnt[(size_t) e];
                            pt.mark(kPfGemmGU, cs);
                            m.gemm.f16(m.Xs + o0 * N, m.dq_gu[q], m.GU + o0 * 1280, ne, 1280, N);
                            swiglu_interleaved(m.GU + o0 * 1280, m.Hh + o0 * 640, ne, m.cs);
                            had_h(o0, ne);
                            pt.mark(kPfGemmD, cs);
                            m.gemm.f16(m.Hh + o0 * 640, m.dq_d[q], m.Dm + o0 * N, ne, N, 640);
                            return true;
                        }
                        catch (sycl::exception const &exc) {
                          std::cerr << exc.what()
                                    << "Exception caught at file:" << __FILE__
                                    << ", line:" << __LINE__ << std::endl;
                          std::exit(1);
                        }
                        };
                        if (!stream_all) {
                            size_t staged = 0;
                            const size_t lookahead = STAGE - 1;
                            for (size_t j = 0; j < order.size(); ++j) {
                                while (staged < order.size() && staged <= j + lookahead) {
                                    if (!stage_one(staged)) return false;
                                    ++staged;
                                }
                                const int32_t e = order[j];
                                if (stage_of[j] < 0) {
                                    ++stats_.experts_resident;
                                    if (!compute(j, m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]), -1)) return false;
                                } else {
                                    pt.mark(kPfWaitCopy, cs);
                                    (m.cs)->ext_oneapi_submit_barrier(
                                        {*m.copied[stage_of[j]]});
                                    if (!compute(j, m.stage_dev[stage_of[j]], stage_of[j])) return false;
                                }
                            }
                        } else {
                            // the streamed walk: this layer's entries [k, kend) in id order; an entry the routing did not
                            // pick only gives its slot back
                            size_t k = seq_start[(size_t) l];
                            const size_t kend = seq_start[(size_t) l + 1];
                            auto release_to = [&](int32_t e_stop) {
                                try {
                            while (k < kend && seq[k].e < e_stop) {
                                    if (gg_nslots > 0) flush();   // the open group's slots get their event first
                                    if (xg_n > 0) xflush();   // the open group's slots get their event first
                                    const int sl = (int) (k % (size_t) m.ring);
                                    dpct::sync_barrier(m.used[sl], m.cs);
                                    m.used_of[sl] = sl;
                                    consumed = ++k;
                                    give_back(consumed);
                                }
                            }
                            catch (sycl::exception const &exc) {
                              std::cerr
                                  << exc.what()
                                  << "Exception caught at file:" << __FILE__
                                  << ", line:" << __LINE__ << std::endl;
                              std::exit(1);
                            }
                            };
                            for (size_t j = 0; j < order.size(); ++j) {
                                const int32_t e = order[j];
                                release_to(e);
                                if (k < kend && seq[k].e == e) {
                                    const int sl = (int) (k % (size_t) m.ring);
                                    wait_issued(k);
                                    if (!group_gather) {
                                        pt.mark(kPfWaitCopy, cs);
                                        (m.cs)->ext_oneapi_submit_barrier(
                                            {*m.copied[sl]});
                                    }
                                    if (!compute(j, m.stage_dev[sl], sl)) return false;
                                    consumed = ++k;
                                    if ((!group_gather || gg_nslots == 0) && xg_ns == 0) give_back(consumed);   // its group was gathered
                                } else {
                                    const bool r0 = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                                    const uint8_t* bp = r0 ? m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e])
                                                           : (m.pp && m.pp->peer ? m.pp->peer->slot_ptr(l, e) : nullptr);   // over the cap: P2P
                                    if (bp == nullptr) { err = "prefill: an expert is neither resident, streamed nor on the peer"; return false; }
                                    ++stats_.experts_resident;
                                    if (!compute(j, bp, -1)) return false;
                                    if (group_gather && gg_nslots == 0) give_back(consumed);
                                }
                            }
                            release_to(m.g->n_expert);
                            xflush();
                            give_back(consumed);
                        }
                    }
                    pt.mark(kPfCombine, cs);
                    if (peer_now) {   // multi-GPU: the peer's rows are in Dm (or, without P2P, in host memory)
                        (m.cs)->ext_oneapi_submit_barrier({*m.pp->ev_done});
                        if (!m.pp->p2p)
                            copy_f32_wide(m.Dm + (size_t) m.pp->back_at * N, m.pp->host_rows, m.pp->back_rows * N, m.cs);
                    }
                    moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs);
                    // debug: STRATA_DBG_NAN=1 reports the first layer of a chunk whose MoE produced non-finite values
                    if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {
                        m.cs->wait();
                        auto bad = [&](const float *d, int64_t n) {
                            try {
                        std::vector<float> h((size_t)n);
                            dpct::get_in_order_queue()
                                .memcpy(h.data(), d, (size_t)n * 4)
                                .wait();
                            int64_t c = 0;
                            for (float v : h) c += !std::isfinite(v);
                            return c;
                        }
                        catch (sycl::exception const &exc) {
                          std::cerr << exc.what()
                                    << "Exception caught at file:" << __FILE__
                                    << ", line:" << __LINE__ << std::endl;
                          std::exit(1);
                        }
                        };
                        // (fused: GU and H hold the grouping tables and int8 rows, not floats)
                        const int64_t bgu = fused_l ? 0 : bad(m.GU, T * K * 1280), bdm = bad(m.Dm, T * K * N),
                                      bbo = bad(m.bo, T * N);
                        const int64_t bh = m.H && !fused_l ? bad(m.H, T * K * 640) : -1;
                        // the fp16 inputs of the experts' GEMMs: the gathered activations and the last dequantized
                        // gate/up (non-finite count, largest finite magnitude)
                        auto bad16 = [&](const uint16_t* d, int64_t n, double& mx) {
                            std::vector<uint16_t> h((size_t) n);
                            dpct::get_in_order_queue().memcpy(h.data(), d, (size_t) n * 2).wait();
                            int64_t c = 0;
                            mx = 0;
                            for (uint16_t v : h) {
                                if ((v & 0x7c00) == 0x7c00) { ++c; continue; }
                                const int e = (v >> 10) & 0x1f;
                                const double a = e ? std::ldexp(1.0 + (v & 0x3ff) / 1024.0, e - 15) : std::ldexp((v & 0x3ff) / 1024.0, -14);
                                mx = std::max(mx, a);
                            }
                            return c;
                        };
                        double mxs = 0, mxq = 0;
                        const int64_t bxs = (!use_mmq && m.Xs) ? bad16(m.Xs, T * K * N, mxs) : -1;
                        const int64_t bdq = (!use_mmq && m.dq_gu[0]) ? bad16(m.dq_gu[0], (int64_t) 1280 * N, mxq) : -1;
                        static int64_t reported = -1;
                        if ((bgu || bdm || bbo || bh > 0) && reported != stats_.chunks)
                            std::fprintf(stderr, "strata dbg: layer %lld fp16 inputs: activations %lld non-finite (max %.3g), "
                                         "dequantized gate/up %lld non-finite (max %.3g)\n", (long long) l, (long long) bxs, mxs,
                                         (long long) bdq, mxq);
                        if ((bgu || bdm || bbo || bh > 0) && reported != stats_.chunks) {
                            reported = stats_.chunks;
                            std::fprintf(stderr, "strata dbg: layer %lld (mmq %d, types %d/%d, %zu experts): non-finite GU %lld "
                                         "H %lld Dm %lld bo %lld of T %lld\n", (long long) l, (int) use_mmq, mmq_gt, mmq_dt,
                                         n_order, (long long) bgu, (long long) bh, (long long) bdm, (long long) bbo,
                                         (long long) T);
                        }
                    }
                }
                // ---- the hyper-connection write of this half; F-2: fused with the next half's norm when nothing else
                // touches R in between (not the stage's last half, not before the PLE block of layer 1, not under a
                // control vector)
                const int64_t nl = half == 0 ? l : l + 1;
                const bool steered = half == 1 && strata::kernels::cvec().covers(l);
                const float *cv_dir = nullptr, *cv_s = nullptr;
                const int* cv_on = nullptr;
                const bool cv_fused = steered && cvec_fuse() && !gr_unfused() && nl < LE && !(nl == 1 && ple_on) &&
                                      strata::kernels::cvec().n_embd == N && strata::kernels::cvec().hc == HC &&
                                      strata::kernels::cvec_tables(&cv_dir, &cv_s, &cv_on);
                const bool fuse = !gr_unfused() && nl < LE && !(half == 1 && nl == 1 && ple_on) && (!steered || cv_fused);
                const core::WeightRef* wnn = nullptr;
                if (fuse) {
                    const core::LayerView vn(*m.wt, nl);
                    wnn = need(vn, half == 0 ? "hc_ffn_norm.weight" : "hc_attn_norm.weight", err);
                    if (!wnn) return false;
                }
                if (wnn && cv_fused) {
                    // STRATA_CVEC_FUSE_CHECK=N: on the first N fused writes, also run the three-kernel path on a copy
                    // of R and report whether R, the row scales and the BF16 image are bit-identical
                    static int checks = [] { const char* e = std::getenv("STRATA_CVEC_FUSE_CHECK"); return e ? std::atoi(e) : 0; }();
                    if (checks > 0) {
                        --checks;
                        float *Rc = nullptr, *rc = nullptr;
                        uint16_t* xc = nullptr;
                        Rc = (float *)sycl::malloc_device(
                            (size_t)T * D * 4, dpct::get_in_order_queue());
                            rc = (float *)sycl::malloc_device(
                                (size_t)T * HC * 4, dpct::get_in_order_queue());
                        xc = (uint16_t *)sycl::malloc_device(
                            (size_t)T * D * 2, dpct::get_in_order_queue());
                        /*
                        DPCT1124: cudaMemcpyAsync is migrated to
                        asynchronous memcpy API. While the origin API might be
                        synchronous, it depends on the type of operand memory,
                        so you may need to call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        m.cs->memcpy(Rc, m.R, (size_t)T * D * 4);
                        gr_write(Rc, m.bo, m.inj, HC, T, m.cs);
                        strata::kernels::cvec_apply(Rc, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);
                        gr_norm_rs(Rc, (const float*) wnn->data, EPS, rc, xc, T, m.cs, nullptr, D);
                        gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, cv_dir + l * N, cv_s + l, cv_on,
                                              strata::kernels::cvec().mode, (const float*) wnn->data, EPS, m.grs,
                                              m.xn16, T, m.cs, m.xn16_lo, ldx);
                        std::vector<float> a((size_t) T * D), b((size_t) T * D), ra((size_t) T * HC), rb((size_t) T * HC);
                        std::vector<uint16_t> xa((size_t) T * D), xb((size_t) T * D);
                        /*
                        DPCT1124: cudaMemcpyAsync is migrated to
                        asynchronous memcpy API. While the origin API might be
                        synchronous, it depends on the type of operand memory,
                        so you may need to call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        m.cs->memcpy(a.data(), m.R, a.size() * 4);
                        /*
                        DPCT1124: cudaMemcpyAsync is migrated to
                        asynchronous memcpy API. While the origin API might be
                        synchronous, it depends on the type of operand memory,
                        so you may need to call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        m.cs->memcpy(b.data(), Rc, b.size() * 4);
                        /*
                        DPCT1124: cudaMemcpyAsync is migrated to
                        asynchronous memcpy API. While the origin API might be
                        synchronous, it depends on the type of operand memory,
                        so you may need to call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        m.cs->memcpy(ra.data(), m.grs, ra.size() * 4);
                        /*
                        DPCT1124: cudaMemcpyAsync is migrated to
                        asynchronous memcpy API. While the origin API might be
                        synchronous, it depends on the type of operand memory,
                        so you may need to call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        m.cs->memcpy(rb.data(), rc, rb.size() * 4);
                        /*
                        DPCT1124: cudaMemcpyAsync is migrated to
                        asynchronous memcpy API. While the origin API might be
                        synchronous, it depends on the type of operand memory,
                        so you may need to call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        m.cs->memcpy(xa.data(), m.xn16, xa.size() * 2);
                        /*
                        DPCT1124: cudaMemcpyAsync is migrated to
                        asynchronous memcpy API. While the origin API might be
                        synchronous, it depends on the type of operand memory,
                        so you may need to call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        m.cs->memcpy(xb.data(), xc, xb.size() * 2);
                        m.cs->wait();
                        sycl::free(Rc, dpct::get_in_order_queue());
                            sycl::free(rc, dpct::get_in_order_queue());
                            sycl::free(xc, dpct::get_in_order_queue());
                        std::fprintf(stderr, "strata: STRATA_CVEC_FUSE_CHECK layer %lld T %lld: R %s, rs %s, xn16 %s\n",
                                     (long long) l, (long long) T,
                                     std::memcmp(a.data(), b.data(), a.size() * 4) ? "DIFFERS" : "identical",
                                     std::memcmp(ra.data(), rb.data(), ra.size() * 4) ? "DIFFERS" : "identical",
                                     std::memcmp(xa.data(), xb.data(), xa.size() * 2) ? "DIFFERS" : "identical");
                    } else
                    gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, cv_dir + l * N, cv_s + l, cv_on,
                                          strata::kernels::cvec().mode, (const float*) wnn->data, EPS, m.grs, m.xn16,
                                          T, m.cs, m.xn16_lo, ldx);
                    normed = true;
                } else if (wnn) {
                    gr_write_norm_rs(m.R, m.bo, m.inj, HC, (const float*) wnn->data, EPS, m.grs, m.xn16, T, m.cs,
                                     m.xn16_lo, ldx);
                    normed = true;
                } else {
                    gr_write(m.R, m.bo, m.inj, HC, T, m.cs);
                }
                if (steered && !cv_fused)   // --control-vector-scaled
                    strata::kernels::cvec_apply(m.R, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);
            }
        }
        if (!ple_land()) return false;   // a stage that ends before layer 1: the rows land anyway, the next gather starts
        if (issuer.joinable()) {
            issuer.join();
            stats_.ms_experts_host += iss_ms;
            stats_.experts_streamed += iss_streamed;
            stats_.experts_dma += iss_dma;
        }
        stats_.tokens += T;
        core::progress_at("reading the prompt (batched): finishing the chunk from token", p0);
        pt.mark(kPfStart, cs);
        if (next_ != nullptr) {
            // The current hand-off slot was used two chunks ago.
            float* h = m.hand[hand_buf_];
            /*
            DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            if (DPCT_CHECK_ERROR(m.cs->memcpy(h, m.R, (size_t)T * D * 4)) !=
                    0 ||
                DPCT_CHECK_ERROR(m.cs->wait()) != 0) {
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
                err = std::string("prefill: the layer split's hand-off: ") +
                      dpct::get_error_string_dummy(0);
                return false;
            }
            // Wait only for the DIRECT successor's previous chunk. That successor
            // may already have forwarded its older chunk to later GPUs.
            if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
            if (next_run_.valid() && !next_run_.get()) { err = next_err_; return false; }
            next_err_.clear();
            next_->hand_in_ = h;
            next_->single_chunk_ = single_chunk;
            next_run_ = std::async(std::launch::async, [this, tokens, c0, T, p0] {
                return next_->run_impl(tokens + c0, T, p0, next_err_);
            });
            hand_buf_ ^= 1;
            continue;   // the last stage reports the chunk (on_chunk)
        }
        if (const char* dump = std::getenv("STRATA_PREFILL_DUMP_R")) {   // debug: the final residuals, every 64th
            m.cs->wait(); // position (A/B quality of this path)
            if (std::FILE* f = std::fopen(dump, c0 == 0 ? "wb" : "ab")) {
                std::vector<float> row((size_t) D);
                for (int64_t t = (64 - p0 % 64) % 64; t < T; t += 64) {
                    dpct::get_in_order_queue()
                        .memcpy(row.data(), m.R + t * D, (size_t)D * 4)
                        .wait();
                    const int64_t pos = p0 + t;
                    std::fwrite(&pos, sizeof pos, 1, f);
                    std::fwrite(row.data(), 4, row.size(), f);
                }
                std::fclose(f);
            }
        }
        if (const char* dump = std::getenv("STRATA_PREFILL_DUMP_R_ALL")) {
            // S25 (draft-layer distillation data, opt-in): every position's final multi-stream residual as BF16
            // (round-to-nearest-even), rows in position order, appended across chunks and requests: [n][hc*n_embd]
            m.cs->wait();
            if (std::FILE* f = std::fopen(dump, "ab")) {
                constexpr int64_t kRows = 512;
                std::vector<float> rows((size_t) (kRows * D));
                std::vector<uint16_t> out((size_t) (kRows * D));
                for (int64_t t0 = 0; t0 < T; t0 += kRows) {
                    const int64_t nr = std::min<int64_t>(kRows, T - t0);
                    dpct::get_in_order_queue()
                        .memcpy(rows.data(), m.R + t0 * D, (size_t)(nr * D) * 4)
                        .wait();
                    for (int64_t i = 0; i < nr * D; ++i) {
                        uint32_t u;
                        std::memcpy(&u, &rows[(size_t) i], 4);
                        u += 0x7fffu + ((u >> 16) & 1u);
                        out[(size_t) i] = (uint16_t) (u >> 16);
                    }
                    std::fwrite(out.data(), 2, (size_t) (nr * D), f);
                }
                std::fclose(f);
            }
        }
        if (on_chunk || on_stage_chunk) {
            const auto toc = Clock::now();
            if (DPCT_CHECK_ERROR(m.cs->wait()) != 0) {
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
                err =
                    std::string("prefill: ") + dpct::get_error_string_dummy(0);
                return false;
            }
            const auto toc2 = Clock::now();
            if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
            if (on_chunk && !on_chunk(m.R, T, p0, err)) return false;
            host_sync_ms += std::chrono::duration<double, std::milli>(toc2 - toc).count();
            host_chunk_ms += ms_since(toc2);
        }
    }
    // Do not drain the successor here. This is the overlap: an intermediate
    // stage can return while later GPUs are still processing the previous chunk.
    ss.ple_prev[0] = prev[0];
    ss.ple_prev[1] = prev[1];
    if (std::getenv("STRATA_DBG_NAN") != nullptr) {   // debug: the state the prompt leaves for the token path
        m.cs->wait();
        auto bad = [&](const float *d, int64_t n) {
            try {
        std::vector<float> h((size_t)n);
            dpct::get_in_order_queue()
                .memcpy(h.data(), d, (size_t)n * 4)
                .wait();
            int64_t c = 0;
            double mx = 0;
            for (float v : h) { c += !std::isfinite(v); if (std::isfinite(v)) mx = std::max(mx, (double) std::fabs(v)); }
            std::fprintf(stderr, " %lld non-finite (max |x| %.3g)", (long long) c, mx);
        }
        catch (sycl::exception const &exc) {
          std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                    << ", line:" << __LINE__ << std::endl;
          std::exit(1);
        }
        };
        const int64_t last = (n - 1) % m.T;
        std::fprintf(stderr, "strata dbg: prompt end: last residual row");
        bad(m.R + last * D, D);
        if (ss.ple.ready()) { std::fprintf(stderr, "; PLE history"); bad(ss.ple.hist, (int64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM); }
        std::fprintf(stderr, "; GDN state 0");
        bad(ss.gdn_state, 64 * 1024);
        std::fprintf(stderr, "\n");
    }
    if (DPCT_CHECK_ERROR(m.cs->wait()) != 0) {
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
        err = std::string("prefill: ") + dpct::get_error_string_dummy(0);
        return false;
    }
    // (PR #121) an expert copy that failed on the copy stream surfaces here, not in the next request
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (const dpct::err0 cst = DPCT_CHECK_ERROR(m.copy->wait()); cst != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        err = std::string("prefill: expert copy stream: ") +
              dpct::get_error_string_dummy(cst);
        return false;
    }
    if (kv_prefetch) {
        /*
        DPCT1000: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (const dpct::err0 cst = DPCT_CHECK_ERROR(m.kv_copy->wait());
            cst != 0) {
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1001: The statement could not be removed.
            */
            err = std::string("prefill: KV prefetch stream: ") +
                  dpct::get_error_string_dummy(cst);
            return false;
        }
        std::fprintf(stderr, "strata KV prefetch: %lld layer prefixes, single existing staging pool\n",
                     (long long) kv_prefetches);
    }
    stats_.ms_total += ms_since(t_start);
    if (pt.on) {
        pt.fold();
        double total = 0.0;
        for (double v : pt.ms) total += v;
        std::string line;
        char b[96];
        for (int i = 0; i < kPfCount; ++i) {
            if (pt.ms[i] <= 0.0) continue;
            std::snprintf(b, sizeof b, " %s %.0f (%.1f%%)", kPfNames[i], pt.ms[i], total > 0 ? 100.0 * pt.ms[i] / total : 0.0);
            line += b;
        }
        std::fprintf(stderr, "strata prefill timing: %lld tokens, GPU timeline %.0f ms, wall %.0f ms, host staging %.0f ms:%s\n",
                     (long long) n, total, ms_since(t_start), stats_.ms_experts_host, line.c_str());
        std::fprintf(stderr, "strata prefill timing: host: chunk setup (PLE rows, the expert stream plan) %.0f ms, "
                             "waiting for each chunk %.0f ms, after each chunk (the draft layer, progress) %.0f ms, "
                             "PLE %.0f ms\n", host_setup_ms, host_sync_ms, host_chunk_ms, stats_.ms_ple);
        if (pe.on) {
            /*
            DPCT1093: The "pe.dev" device may be not the one intended for
            use. Adjust the selected device if needed.
            */
            /*
            DPCT1093: The "pd" device may be not the one intended for use.
            Adjust the selected device if needed.
            */
            int pd = 0; pd = dpct::get_current_device_id();
                dpct::select_device(pe.dev); m.pp->s->wait(); pe.fold();
                dpct::select_device(pd);
            std::string pl;
            for (int i = 0; i < kPeCount; ++i) {
                std::snprintf(b, sizeof b, " %s %.0f", kPeNames[i], pe.ms[i]);
                pl += b;
            }
            std::fprintf(stderr, "strata prefill timing (peer GPU, ms):%s; MoE layers %lld, rows/layer %.0f of %lld cap, experts/layer %.0f, "
                                 "over the cap %lld\n", pl.c_str(), (long long) m.pp->layers,
                         m.pp->layers ? (double) m.pp->rows / m.pp->layers : 0.0, (long long) m.pp->cap_rows,
                         m.pp->layers ? (double) m.pp->experts / m.pp->layers : 0.0, (long long) m.pp->over_cap);
            if (m.pp->ps_frac > 0.0)
                std::fprintf(stderr, "strata prefill timing: peer-streamed experts %lld (%.0f per MoE layer)\n",
                             (long long) m.pp->ps_experts, m.pp->layers ? (double) m.pp->ps_experts / m.pp->layers : 0.0);
        }
    }
    if (std::getenv("STRATA_STATE_HASH_GDN") != nullptr) {   // debug: the GDN states as the prompt path leaves them
        m.cs->wait();
        std::vector<uint8_t> b((size_t) gdn_floats * 4);
        std::string line;
        char h[8];
        for (int64_t i = 0; i < ss.gdn_alloc; ++i) {
            dpct::get_in_order_queue()
                .memcpy(b.data(), ss.gdn_state + (size_t)i * gdn_floats,
                        b.size())
                .wait();
            uint64_t x = 1469598103934665603ull;
            for (uint8_t c : b) x = (x ^ c) * 1099511628211ull;
            std::snprintf(h, sizeof(h), "%04llx ", (unsigned long long) (x & 0xffff));
            line += h;
        }
        std::fprintf(stderr, "strata prefill: GDN_HASH %s\n", line.c_str());
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool Prefill::drain_pipeline(std::string& err) {
    bool ok = true;

    if (next_run_.valid()) {
        if (!next_run_.get()) {
            err = next_err_;
            ok = false;
        }
    }

    if (next_ != nullptr) {
        std::string tail_err;
        if (!next_->drain_pipeline(tail_err)) {
            if (ok) err = tail_err;
            ok = false;
        }
    }

    return ok;
}

bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    const bool body_ok = run_impl(tokens, n, pos0, err);

    std::string drain_err;
    const bool drain_ok = drain_pipeline(drain_err);

    if (!body_ok) return false;
    if (!drain_ok) {
        err = drain_err;
        return false;
    }
    return true;
}

}  // namespace strata::prefill
