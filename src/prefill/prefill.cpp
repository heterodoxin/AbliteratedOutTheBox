// src/prefill/prefill.cpp - see include/strata/prefill/prefill.hpp.
#include "strata/kernels/fused_gr.hpp"
#include "strata/prefill/prefill.hpp"
#if defined(STRATA_HAVE_DFLASH2)
#include "strata/kernels/dflash2.hpp"
#endif
#include "strata/prefill/warm_policy.hpp"
#include <atomic>
#if defined(__cpp_lib_atomic_wait)
#define STRATA_ATOMIC_WAIT 1   // #1488: std::atomic::wait / notify_all are a GCC 11 library feature; GCC 10 spins instead
#endif
#include "mmq_resident_sort.hpp"
#include "wmma_gemm.h"
#include "strata/core/mtp.hpp"
#include "strata/core/pinned_ring.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/on_device.hpp"
#include "strata/platform/aux_cpus.hpp"

#include "strata/core/layout.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/native_dense.hpp"
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

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <map>
#include <string>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
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
void quantize_scatter(const float*, const int32_t*, const int32_t*, void*, int, int64_t, int64_t, int64_t, int, void*) {}
Context::Context() {}
Context::~Context() {}
void Context::run(const Product&, void*) {}
void gather_native(const void*, const void*, size_t, const void*, size_t, void*, void*, void*, const void*, void*, size_t) {}
bool gather_native_group(const GatherGroup&, size_t, size_t, size_t, size_t, void*, size_t, void*, size_t, void*, size_t,
                         void*, size_t) { return false; }
void gather_strata_q2(const uint8_t*, void*, void*, void*) {}
void swiglu(const float*, float*, int64_t, int64_t, bool, void*) {}
void swiglu_scaled(const float*, float*, int64_t, int64_t, const int32_t*, int, const float*, int64_t, void*) {}
void down_row_scales(float*, const int32_t*, int, const float*, int64_t, void*) {}
void iota(int32_t*, int64_t, void*) {}
bool tab_built() { return false; }
bool tab_fits(int, int64_t) { return false; }
size_t overread_bytes(int, int64_t) { return 0; }
bool whole_tiles(int, int64_t, int) { return false; }
int direct_mode() { return 0; }
size_t direct_slot_guard() { return 0; }
void prep_group(const GuardTab&, size_t, void*, void*, size_t, void*) {}
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

#if defined(STRATA_GPU_METAL)
// The Metal backend reads a prompt through its decode window (metal/src/engine/window.cpp): init keeps the chunk, run
// hands over, draft_kv declines (the MTP hook reads the window's rows itself).
namespace strata::metal_engine {
bool window_ensure(const strata::core::ModelGeometry& g, strata::core::SessionState& ss, int64_t max_cells, std::string& err);
bool prompt_run(const int64_t* tokens, int64_t n, int64_t pos0, const float* const* embd_rows,
                const std::function<bool(const float*, int64_t, int64_t, std::string&)>& on_chunk, std::string& err);
}  // namespace strata::metal_engine
#endif

namespace strata::prefill {
namespace {

using Clock = std::chrono::steady_clock;
constexpr float EPS = 1e-6f;
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10, NE = 512;
constexpr int64_t C = 10240, ZV = 6144, HV = 48;
// plan v0.3 P6: staging holds the largest blob of the pack (a native pack's blobs differ per layer)
inline int64_t MAXBLOB() { return (int64_t) strata::kernels::cpu::expert_layout().max_blob; }
constexpr int STAGE = 8;           // host->device expert staging ring (chunks below stream_all_min())
// The staged walk's ring when its MMQ groups take the staged experts too: two groups, so a slot is refilled only after
// the group that read it was gathered (the walk stages at most STAGE_GRP - MMQ_GROUP - 1 entries ahead)
constexpr int STAGE_GRP = 32;
constexpr int kSplitHelpRing = 48;  // layer split help: the helper stage's ring slots it streams through (at most)
// STRATA_PREFILL_STREAM_AHEAD=0 keeps the previous routed-only upload schedule (A/B).
inline bool stream_ahead_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PREFILL_STREAM_AHEAD");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}
// Step 3: from this chunk size on, every non-resident expert of every layer streams in a fixed order through a
// ring_slots()-slot ring (nearly all 512 are routed at such a chunk), so the copy engine keeps working through the
// attention halves instead of waiting for each layer's routing.
// #828: STRATA_TEST_PAGEABLE=1 makes every pinned host allocation with a pageable fallback in Prefill::init fail, so the
// fallbacks (what an LXC container's memlock limit forces) run on a PC that can pin - for ASan / MALLOC_CHECK_=3 runs.
inline bool force_pageable() {
    static const bool v = [] { const char* e = std::getenv("STRATA_TEST_PAGEABLE"); return e != nullptr && e[0] == '1'; }();
    return v;
}
// #1057: the Stager threads wait by sleeping (atomic wait, blocking-sync events) instead of a yield spin.  On Linux the
// spinners starved a pinned host thread (DGX Spark IQ3_S 8K prompt 91 -> 1,222 tok/s with sleeping waits); on Windows
// the 5070 read an 8K Q2_0 prompt 1.2% slower with them (1,469.5 -> 1,452 tok/s, 10 pairs), so they stay spinning
// there.  Same output either way.  STRATA_STAGER_SLEEP=1/0 forces either.
inline bool stager_sleep() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_STAGER_SLEEP");
        if (e != nullptr && e[0] != '\0') return e[0] != '0';
#if defined(_WIN32)
        return false;
#else
        return true;
#endif
    }();
    return v;
}
constexpr int RING_MAX = 1024;          // the arrays; the ring itself is ring_slots(), at most ring_cap()
// The chunk size from which every expert streams: 1024 since 0.1.30 (was 2048).  Measured on the 5070, Q2_0 / IQ2_XS,
// fixed cache: 1,500-token prompts 621 -> 785 / 612 -> 735 tok/s, 2,000 727 -> 934 / 712 -> 892, 4,000 (its last
// chunk) 779 -> 912 / 766 -> 844, the same output.  Below ~1,000 tokens the output changed on Q2_0 (a smaller chunk
// takes other kernels), so 1024 is the floor.  STRATA_PREFILL_STREAM_MIN overrides (A/B).  With the CPU share armed
// (Prefill::arm_cpu_share) it is the share's own limit instead: the share applies to staged chunks only.
constexpr int64_t kStreamAllMinDefault = 1024;   // the default above (and the CPU share's default limit)
int64_t g_stream_min_share = 0;
inline int64_t stream_all_min() {
    static const int64_t env = [] { const char* e = std::getenv("STRATA_PREFILL_STREAM_MIN"); return e ? (int64_t) std::atoll(e) : (int64_t) -1; }();
    if (env >= 0) return env;
    return g_stream_min_share > 0 ? g_stream_min_share : kStreamAllMinDefault;
}
// STRATA_PREFILL_CPU_SHARE (opt-in; set_cpu_pool): a chunk below stream_all_min() hands the decode CPU pool - idle
// while a prompt is read - the non-resident experts few of its tokens route to, instead of streaming them over PCIe.
// `auto`: the share measured, each layer's CPU time per expert and the GPU's per streamed expert (CUDA events around
// its expert work, read after the next layer's routing sync - the host reaches the combine long before the GPU does)
// as running means, and the next layers hand the CPU g / (c + g) of them, where both sides end together - so a small
// CPU takes a small share - and only while the layers that share are measured cheaper than the ones that do not (the
// gate at cpu_maybe).  x: a fixed x.  Unset or 0: every expert on the GPU (the default).  RTX 5090 + 9950X3D,
// 600-token prompts: UD-Q4_K_XL 1,528-1,548 -> 1,296-1,369 ms (share 0.65), Q2_0 472-491 -> 445-448, IQ2_XS 507 -> 490
// (250 tokens 402 -> 368); IQ2_XS on 2 AVX2 workers 512 -> 487 (share 0.49).
// 0.1.41: ON BY DEFAULT where it was measured (CUDA builds, one GPU, no batch slots): STRATA_PREFILL_CPU_SHARE unset
// behaves as `auto` with STRATA_PREFILL_CPU_SHARE_MAX=1024 (chunks below 1024 tokens: -20..-35% at 512 and 1000 tokens
// on an RTX 3060, a Tesla P100 and an RTX 5070, 8/8 pairs each; mean KL against the share off about 0.004).
// STRATA_PREFILL_CPU_SHARE=0 turns it off: the output is then the 0.1.40.3 bytes.  g_share_default is set by
// Prefill::arm_cpu_share(.., true) from the paths that qualify; the layer split, batch slots and HIP stay off.
bool g_share_default = false;
inline double cpu_share_explicit() {   // -2: the variable is not set; -1: measured
    static const double v = [] {
        const char* e = std::getenv("STRATA_PREFILL_CPU_SHARE");
        if (e == nullptr) return -2.0;
        if (std::strcmp(e, "auto") == 0) return -1.0;
        return std::clamp(std::atof(e), 0.0, 1.0);
    }();
    return v;
}
inline double cpu_share_env() {   // -1: measured
    const double v = cpu_share_explicit();
    return v == -2.0 ? (g_share_default ? -1.0 : 0.0) : v;
}
inline bool cpu_share_on() { return cpu_share_env() != 0.0; }
// STRATA_PREFILL_CPU_SHARE_MAX (default 3072, at least 1024): with the share on, chunks below it are staged after their
// routing - so they can hand the CPU its share - instead of streaming every expert the cards do not hold.  The share pays
// on chunks up to ~3K tokens, where few tokens route to many of the streamed experts; 1024 keeps the old limit.
inline int64_t cpu_share_max() {
    static const int64_t v = [] {
        const char* e = std::getenv("STRATA_PREFILL_CPU_SHARE_MAX");
        return e ? std::max<int64_t>(kStreamAllMinDefault, (int64_t) std::atoll(e)) : (int64_t) 3072;
    }();
    static const bool explicit_max = std::getenv("STRATA_PREFILL_CPU_SHARE_MAX") != nullptr;
    if (g_share_default && cpu_share_explicit() == -2.0 && !explicit_max) return kStreamAllMinDefault;   // its measured range
    return v;
}
// On a layer split every stage may have the pool (set_cpu_pool), and the stages read different chunks at the same time:
// one stage at a time takes it for a chunk, the other reads its chunk with the GPU alone.
std::atomic<bool> g_cpu_pool_busy{false};
double g_pinned_share = 1.0;
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
    if (core::peer_portable()) return false;   // multi-GPU: --peer-device keeps the MMQ path and its buffer sizes
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
    // the Q2_0 pack: fused::enabled().  A native pack: native_supported() alone - it implies enabled() wherever the Q2_0
    // kernels exist, and on gfx12 (the native kernels only) enabled() is false while the native layers do run fused.
    if (!lay.native) return fused::enabled();
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
// #1454: `bo` shares `emb`'s storage (carve), which frees T * N floats per chunk.  The planner keeps counting them by
// default, so the auto chunk and the borrowed cache slots (and with them the prompt path's bits) are exactly 0.1.40.3's;
// STRATA_EMB_REUSE_ACCOUNT=1 lets it use the saved bytes: a larger chunk or more borrowed slots where VRAM is the limit
// (RTX 3060, IQ3_XXS: chunk 6400 -> 6656, prompt +3.9%), with other rounding in the prompt path.
inline bool emb_reuse_account() {
    static const bool on = [] { const char* v = std::getenv("STRATA_EMB_REUSE_ACCOUNT"); return v != nullptr && v[0] == '1'; }();
    return on;
}
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
    // S6: with the matrix-core attention (the gfx12 default since 148-amd; STRATA_HIP_WMMA=1 or auto elsewhere / forced, on a card that
    // has it) a 96-slot ring: measured with it, 9070 XT 4K prompts 718 -> 1,211 tok/s (16K 1,949 -> 2,032), R9700 4K 2,426 -> 2,483 (16K the same)
    static const bool wmma = strata::kernels::qsa_prompt_attn_matrix_cores_requested();
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
    const int big = r < 16 ? 16 : r > ring_cap() ? ring_cap() : r;
    return (int64_t) T >= stream_all_min() ? big : STAGE;
}
// The staging buffers a layout holds: the ring, and at least STAGE_GRP for the staged walk's grouped gather.  Counted
// and taken through this one place, so the bytes the prompt path borrows cover every buffer it writes.
inline int stage_slots(size_t T) { return std::max(ring_slots(T), STAGE_GRP); }
constexpr int DQ = 2;              // dequantized-expert ring (FP16 gate/up + down)
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
// STRATA_HANDOFF_PIPE=P (opt-in, 2..16; unset/0/1 = the one copy): a layer split hands a chunk's rows on in P row pieces.
// The stage records an event when its chunk's work is done and one after each piece's copy into the pinned hand-off
// buffer, waits on the host only for the first (what on_stage_chunk's checkpoint parts and the error check need), and
// starts the next stage at once: that stage's upload of piece i waits on piece i's event on the GPU, so its uploads
// overlap the remaining downloads (separate PCIe links; the 3090s here have no P2P) and the thread start is hidden
// behind them.  The same bytes land in the same rows before the next stage's first kernel: exact.  No device memory;
// the events and the existing two pinned buffers per stage only.
static constexpr int kHandPiecesMax = 16;
static int handoff_pipe_pieces() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_HANDOFF_PIPE");
        const int p = e ? std::atoi(e) : 0;
        return p >= 2 ? std::min(p, kHandPiecesMax) : 0;
    }();
    return v;
}
// rows [r0, r0 + nr) of piece i of a T-row chunk in P pieces (the same split on both sides of a hand-off)
static void handoff_piece(int64_t T, int P, int i, int64_t& r0, int64_t& nr) {
    const int64_t per = (T + P - 1) / P;
    r0 = std::min<int64_t>(T, (int64_t) i * per);
    nr = std::min<int64_t>(T, r0 + per) - r0;
}
// STRATA_HANDOFF_PIPE: the next stage's chunks run on one thread of the stage's own, started once and pinned once
// (std::async started a thread per chunk: ~1 ms from the hand-off to the next stage's first call, measured with nsys).
// The futures are the same (run_impl's result, or what it threw), so the chain's waits do not change.
// (STRATA_HANDOFF_PIPE_WORKER=0: a thread per chunk again, for an A/B of the two halves.)
static bool handoff_pipe_worker() {
    static const bool v = [] { const char* e = std::getenv("STRATA_HANDOFF_PIPE_WORKER"); return !e || std::atoi(e) != 0; }();
    return handoff_pipe_pieces() > 0 && v;
}
struct HandWorker {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::packaged_task<bool()>> jobs;
    bool stop = false;
    std::thread th;
    HandWorker() {
        th = std::thread([this] {
            strata::platform::name_this_thread("strata-hand");
            strata::aux_cpus::pin_current_thread();
            for (;;) {
                std::packaged_task<bool()> job;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [this] { return stop || !jobs.empty(); });
                    if (jobs.empty()) return;   // stopped, and every job taken has run (no broken promise)
                    job = std::move(jobs.front());
                    jobs.pop_front();
                }
                job();
            }
        });
    }
    std::future<bool> submit(std::packaged_task<bool()> job) {
        std::future<bool> f = job.get_future();
        {
            std::lock_guard<std::mutex> lk(mu);
            jobs.push_back(std::move(job));
        }
        cv.notify_one();
        return f;
    }
    ~HandWorker() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_one();
        if (th.joinable()) th.join();
    }
};

// S23 (opt-in STRATA_HC_UPMIX=1): the hyper-connection read's up projection and gr_mix_r as one kernel (gr_upmix: gfx11,
// and CUDA sm_80+ on chunks of 33+ tokens - there, with CUDA 13.3 on an RTX 5090, cuBLAS's up GEMM sums K in the fused
// kernel's order and the read is the default pair's bits: gr_upmix_parity, STRATA_HC_UPMIX_CHECK);
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
// CUDA: the fused read only on chunks of 33+ tokens - below that cuBLAS's up GEMM takes another kernel, rounding-level
// apart, so short chunks keep the default pair
inline int64_t hc_upmix_min_t() {
#if defined(__HIPCC__)
    return 0;
#else
    return 33;
#endif
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
        if (!(p && p[0] == '1' && g && g[0] == '1')) return false;
        // The padded stride pays only for strata_pf_gemm_f16_ld, which is built for the gfx11 matrix cores.  On another
        // card nothing takes a padded X (it used to end the engine: "a padded X ... needs STRATA_PF_PAD's path", #1237):
        // do not pad, and say so once.
        if (!strata_pf_padded_gemm_available()) {
            std::fprintf(stderr, "prefill gemm: STRATA_PF_PAD needs the padded gfx11 product, which this card does not run; "
                                 "not padding\n");
            return false;
        }
        return true;
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
    bool spill = false, spilled = false;
    std::vector<void*>* owned = nullptr;
    template <typename T> T* take(size_t n, bool& ok) {
        uint64_t bytes = ((uint64_t) n * sizeof(T) + 256 + 255) & ~255ull;
        if (count_only) {
            if (granule > 0) bytes = (bytes + granule - 1) / granule * granule;
            used += bytes;
            return nullptr;
        }
        if (base != nullptr && !spilled && used + bytes <= cap) {
            T* p = (T*) (base + used);
            used += bytes;
            return p;
        }
        if (base != nullptr && !spill) { ok = false; failed_bytes = bytes; return nullptr; }
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; failed_bytes = bytes; return nullptr; }
        spilled = true;
        owned->push_back(p);
        used += bytes;
        return (T*) p;
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
    // pp-opt: the jobs a thread claims at once (STRATA_STAGER_BATCH; set by init's caller, at most kMaxBatch and kRing)
    static constexpr int kMaxBatch = 32;
    int batch = 1;
    // `from` set: the blob is copied by the source itself (CS-T: a GGUF read in place assembles it from its three
    // role slices; a pointer to it would not live as long as the queue)
    struct Job { const uint8_t* src; size_t bytes; core::ExpertSource* from = nullptr; int32_t l = 0, e = 0; };
    std::vector<uint8_t*> buf;
    std::vector<char> pinned;
    std::vector<std::vector<uint8_t>> pageable;   // the fallback when no more RAM can be pinned
    std::vector<cudaEvent_t> dma_done;
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
        pageable.resize(kRing);
        for (int i = 0; i < kRing; ++i) {
            pinned[i] = !force_pageable() && cudaHostAlloc((void**) &buf[i], blob_bytes, cudaHostAllocDefault) == cudaSuccess;
            if (!pinned[i]) {
                cudaGetLastError();
                pageable[(size_t) i].resize(blob_bytes);
                buf[i] = pageable[(size_t) i].data();
            }
            if (cudaEventCreateWithFlags(&dma_done[i], stager_sleep() ? (cudaEventDisableTiming | cudaEventBlockingSync)
                                                                      : cudaEventDisableTiming) != cudaSuccess) return false;
        }
        cudaGetDevice(&device);
        for (int t = 0; t < nthreads; ++t)
            threads.emplace_back([this] {
                // --aux-cpus: the copy threads go to the spare CPUs with the other helpers; STRATA_AUX_STAGER=0 leaves them where
                // they were created (a small spare set would then hold all of them: it limits the copy rate of a long read)
                strata::platform::name_this_thread("strata-stage");
                static const bool keep = [] { const char* v = std::getenv("STRATA_AUX_STAGER"); return v != nullptr && std::atoi(v) == 0; }();
                if (keep) strata::aux_cpus::note_owned_thread();
                else strata::aux_cpus::pin_current_thread();
                work();
            });
        return true;
    }
    ~Stager() {
        finish();
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto& t : threads) t.join();
        for (int i = 0; i < kRing; ++i) {
            if (dma_done[i]) cudaEventDestroy(dma_done[i]);
            if (buf[i] && pinned[i]) cudaFreeHost(buf[i]);
        }
    }
    void work() {
        cudaSetDevice(device);
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
                // pp-opt: a run of up to `batch` consecutive jobs (batch <= kRing, so the earliest unfinished run
                // never waits for a buffer of its own)
                int k = 1;
                const int j = claim(seen, batch, k);
                if (j < 0) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
                for (int i = j; i < j + k; ++i) {
                    // The copies run ahead of the DMAs, so most of these threads spend most of a prompt waiting here: sleep
                    // (atomic wait, and a blocking-sync event below), not a yield spin - 32 spinners took every core
                    // (Linux; see stager_sleep for Windows).
                    if (i >= kRing) {   // job i - kRing's DMA from this buffer is queued
#if defined(STRATA_ATOMIC_WAIT)
                        if (stager_sleep())
                            for (int x; (x = issued.load(std::memory_order_acquire)) <= i - kRing;) issued.wait(x);
                        else
#endif
                            while (issued.load(std::memory_order_acquire) <= i - kRing) std::this_thread::yield();
                    }
                    // and done - for a generation's first kRing jobs that is the previous generation's last DMA from
                    // the buffer, which nothing else waits for when a chunk ends without a sync (no MTP) or the DMA
                    // was a ring entry the routing skipped (an event never recorded returns at once)
                    cudaEventSynchronize(dma_done[i % kRing]);
                }
                int32_t rl[kMaxBatch], re[kMaxBatch];
                uint8_t* rd[kMaxBatch];
                size_t nr = 0;
                core::ExpertSource* from = nullptr;
                for (int i = j; i < j + k; ++i) {
                    const Job& jb = jobs[(size_t) i];
                    if (jb.from == nullptr) { std::memcpy(buf[i % kRing], jb.src, jb.bytes); continue; }
                    from = jb.from;
                    rl[nr] = jb.l; re[nr] = jb.e; rd[nr] = buf[i % kRing]; ++nr;
                }
                if (nr > 0 && !from->copy_blobs(rl, re, rd, nr)) {
                    std::fprintf(stderr, "prefill: the expert source could not copy experts %d.. of layer %d\n", re[0], rl[0]);
                    std::abort();
                }
                for (int i = j; i < j + k; ++i) ready[(size_t) i].store(1, std::memory_order_release);
                active.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }
    int claim(uint32_t g, int want, int& k) {
        uint64_t cur = head.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (cur >> 32) != g) return -1;
            const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
            if (j >= n) return -1;
            k = std::min(want, n - j);
            if (head.compare_exchange_weak(cur, cur + (uint64_t) k, std::memory_order_acq_rel, std::memory_order_acquire))
                return j;
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
    void issued_one(int j, cudaStream_t copy) {
        cudaEventRecord(dma_done[j % kRing], copy);
        issued.store(j + 1, std::memory_order_release);
#if defined(STRATA_ATOMIC_WAIT)
        issued.notify_all();
#endif
    }
    /// No job is running after this (the end of a layer, or an early return in the middle of one).
    void finish() {
        head.store((uint64_t) gen << 32, std::memory_order_release);   // n = 0: nothing more to claim
        issued.store(1 << 30, std::memory_order_release);
#if defined(STRATA_ATOMIC_WAIT)
        issued.notify_all();
#endif
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
    cudaStream_t s = nullptr;
    cudaEvent_t ev_in = nullptr;    // on the primary: the activations and the row tables are ready
    cudaEvent_t ev_done = nullptr;  // on the peer: its rows have landed in the primary's Dm
    // multi-GPU: the result rows go back group by group on a second stream while the next group computes
    cudaStream_t s_out = nullptr;
    static constexpr int kGrpEv = NE / 16 + 2;
    cudaEvent_t ev_grp[kGrpEv] = {};
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
    cudaEvent_t ev_dm[2] = {};
    bool dm_live[2] = {};
    std::vector<std::pair<size_t, size_t>> groups;
    // multi-GPU PEER STREAMING: a share of the experts the primary would stream over ITS PCIe link in a
    // big chunk is streamed by the peer over its own link instead (into its own ring) and computed there - it halves
    // the primary's copy load and moves expert work to the card that idles through most of the MoE half.
    // STRATA_PF_PEER_STREAM = the share (0 = off), STRATA_PF_PEER_RING = its ring slots.
    double ps_frac = 0.0;
    int RP = 0;
    std::vector<uint8_t*> pstage;
    std::vector<cudaEvent_t> pcopied, pused;
    std::vector<char> plive;
    cudaStream_t s_cp = nullptr;
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
    // SUMS (the host route's default, STRATA_PF_PEER_SUMS=0: rows): the peer adds its rows into one weighted sum per
    // token (`sum`, from the routing weights `wk` and each row's routed pair `pair`) and sends that back (`host_sum`,
    // T x N) instead of its rows (rows_peer x N, up to K times more over the primary's link); the primary's combine
    // adds it to its own experts' rows (eddoursul/Strata's second-GPU prompt path, f8de703).
    bool sums = false;
    cudaStream_t x_stream = nullptr;      // primary: the host route's copies beside its compute stream
    cudaEvent_t ev_x = nullptr;
    bool f16 = false;                     // the sums mode's transfers in FP16 (STRATA_PF_PEER_F16=0: FP32)
    // the sums in one launch per group (STRATA_PF_PEER_GATHER=0: per expert): per group [tokens | starts | rows] at
    // adds_at[g], built on the host from the rows' pairs, uploaded once per layer
    bool gather = false;
    int32_t *adds = nullptr, *host_adds = nullptr;
    size_t adds_cap = 0;
    std::vector<size_t> adds_at;
    std::vector<int32_t> ntok, tcur, touched;
    uint16_t *sum16 = nullptr, *host_x16 = nullptr, *host_sum16 = nullptr;
    float *sum = nullptr, *wk = nullptr;
    int32_t* pair = nullptr;
    float *host_w = nullptr, *host_sum = nullptr;
    int32_t* host_pair = nullptr;
    std::vector<int32_t> bounds_host;
    core::PinnedRing up_ring;             // #1996: this device's pageable uploads (bounds, routed rows) go up from pinned slots
    // APR (prism.hadamard): per layer, this device's copy of the down projection's sign vector (nullptr: identity or
    // not folded).  The gate/up input arrives already rotated (the primary sends its rotated copy, m.Xr).
    std::vector<const float*> had_sh;
    std::vector<void*> owned;
    int64_t layers = 0, experts = 0, rows = 0, over_cap = 0;   // stats
    ~PeerPrefill() {
        if (dev < 0) return;
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(dev);
        if (s) cudaStreamSynchronize(s);
        ctx.reset();
        up_ring.release();
        for (void* p : owned) cudaFree(p);
        if (ev_done) cudaEventDestroy(ev_done);
        if (s_out) cudaStreamSynchronize(s_out);
        for (cudaEvent_t e : ev_grp) if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : ev_dm) if (e) cudaEventDestroy(e);
        if (s_cp) cudaStreamSynchronize(s_cp);
        for (cudaEvent_t e : pcopied) if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : pused) if (e) cudaEventDestroy(e);
        if (s_cp) cudaStreamDestroy(s_cp);
        if (s_out) cudaStreamDestroy(s_out);
        if (s) cudaStreamDestroy(s);
        cudaSetDevice(prev);
        if (ev_in) cudaEventDestroy(ev_in);
        if (x_stream) { cudaStreamSynchronize(x_stream); cudaStreamDestroy(x_stream); }
        if (ev_x) cudaEventDestroy(ev_x);
        if (host_x) cudaFreeHost(host_x);
        if (host_rows) cudaFreeHost(host_rows);
        if (host_src) cudaFreeHost(host_src);
        if (host_bounds) cudaFreeHost(host_bounds);
        if (host_w) cudaFreeHost(host_w);
        if (host_sum) cudaFreeHost(host_sum);
        if (host_pair) cudaFreeHost(host_pair);
        if (host_x16) cudaFreeHost(host_x16);
        if (host_sum16) cudaFreeHost(host_sum16);
        if (host_adds) cudaFreeHost(host_adds);
    }
};

// #1712: a chunk's small int32 tables (the QSA step records, the token ids) go up from a pinned buffer.  A cudaMemcpyAsync
// from pageable memory is staged by the runtime; on gfx1100 the first text prompt after a picture one was seen to read stale
// step records that way.  The event holds the next upload back until the last one has left the buffer.  No pinned RAM:
// the copy is made from the caller's memory, as before.
struct PinnedUpload {
    int32_t* p = nullptr;
    size_t n = 0;
    cudaEvent_t ev = nullptr;
    void init(size_t count) {
        if (p || force_pageable()) return;
        if (cudaHostAlloc((void**) &p, count * sizeof(int32_t), cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            p = nullptr;
            return;
        }
        n = count;
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            cudaFreeHost(p);
            p = nullptr;
            n = 0;
            ev = nullptr;
        }
    }
    bool upload(void* dst, const int32_t* src, size_t count, cudaStream_t s) {
        const size_t bytes = count * sizeof(int32_t);
        if (!p || count > n) return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s) == cudaSuccess;
        if (cudaEventSynchronize(ev) != cudaSuccess) return false;   // the previous upload is out of the buffer
        std::memcpy(p, src, bytes);
        return cudaMemcpyAsync(dst, p, bytes, cudaMemcpyHostToDevice, s) == cudaSuccess && cudaEventRecord(ev, s) == cudaSuccess;
    }
    void release() {
        if (ev) cudaEventDestroy(ev);
        if (p) cudaFreeHost(p);
        p = nullptr;
        ev = nullptr;
        n = 0;
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
    bool borrowed = false, borrow_spill = false;
    cudaStream_t cs = nullptr, copy = nullptr;
    Gemm gemm;
    std::vector<void*> owned;
    // chunk buffers
    float *emb = nullptr, *R = nullptr, *xn = nullptr, *lo = nullptr, *gated = nullptr, *inj = nullptr;
    float* grs = nullptr;                    // F-1: the hyper-connection read's row scales (T x 4)
    uint16_t *xn16 = nullptr, *lo16 = nullptr;
    float* mixed = nullptr;
    uint16_t *mixed_bf = nullptr, *mixed_h = nullptr;
    bool f16_io = false;   // HIP, STRATA_HIP_PROMPT_F16=1: this stage's 16-bit GEMMs run FP16 in and out (set in init)
    bool xn_f16 = false;   // Turing, BF16 on FP16 tensor cores: xn16 written as the FP16 values the GEMM would convert
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
    PinnedUpload steps_up;
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
    float* Xr = nullptr;   // APR: Hadamard-rotated copy of `mixed` for the MMQ path
    // step 2b (MMQ): the activations quantized per layer, H in FP32 and its group's quantized rows, the identity
    // row map, the group bounds, the group buffers of gathered experts
    void *Xq = nullptr, *Hq = nullptr;
    float* H = nullptr;
    int32_t *ids_identity = nullptr, *bounds_dev = nullptr;
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    float *Xscale = nullptr, *Hscale = nullptr, *grp_tail = nullptr;   // NVFP4: activation scales, expert tails
    float* row_sd = nullptr;          // NVFP4 in MMQ: each Dm row's s_down, applied by the combine as it reads the row
    std::vector<int32_t> bounds_host;
    std::unique_ptr<mmq::Context> mmq_ctx;
    std::vector<int32_t> ids_host, slot_host, src_host, cnt, off;
    // set_cpu_pool: a small chunk's CPU experts. The layer's MoE input (T x N, from `mixed`), the rows the pool writes
    // (Dm's tail, in Dm's row order), ones for row_sd (the CPU's rows carry s_down already), all pinned; and the
    // pool's per-token activations and jobs.
    float *cpu_x = nullptr, *cpu_rows = nullptr, *cpu_ones = nullptr;
    size_t cpu_x_n = 0, cpu_rows_n = 0, cpu_ones_n = 0;
    bool cpu_dead = false;   // a pinned buffer of the share could not be had (host memory is short): the GPU does it all
    std::vector<uint8_t> cpu_nact;
    std::vector<kernels::cpu::ActQ> cpu_actq;   // a Q2_0 layer's activations (the pool's Q2_0 kernels read ActQ)
    std::vector<kernels::cpu::ExpertJobMulti> cpu_jobs;
    // the measured share: running means of the CPU's ms per expert and the GPU's per streamed expert, and the share
    // they balance at (cpu_share_env)
    double cpu_c_ms = 0, cpu_g_ms = 0, cpu_share_now = 0.5;
    cudaEvent_t cpu_ev[2] = {};   // the GPU's expert work of the last CPU-sharing layer, read once the stream is done
    bool cpu_pend = false;
    double pend_cpu_ms = 0;
    int64_t pend_n_cpu = 0, pend_n_gpu = 0;
    // ... and whether sharing pays at all (auto): an eligible layer's wall time per non-resident expert, from before
    // its routing sync to its combine (CUDA events, two sets: a layer's start is recorded before the last one's is
    // read); two adjacent layers of the two arms give a ratio (with / without), and the median of the last
    // CPU_RATIOS decides.  Layers differ up to 8x in that cost, the same in every run, which is why adjacent layers
    // are compared rather than a running mean per arm.
    static constexpr int CPU_RATIOS = 5;
    int64_t cpu_layers = 0;   // the eligible layers so far (the arms' schedule)
    int64_t cpu_first_l = -1;   // the model's first eligible layer: 2x the others' cost in both arms, never read
    cudaEvent_t cpu_wall[4] = {};
    int pend_wall = 0, pend_set = 0, pend_l = 0;   // pend_wall 1: the last eligible layer ran without the share, 2: with
    int64_t pend_wall_n = 0;
    int last_arm = 0, last_l = 0;
    double last_w = 0;
    double cpu_ratio[CPU_RATIOS] = {};
    int cpu_nratio = 0;
    bool cpu_gate = false;   // the median ratio is below 1: the layers that share are the cheaper ones
    // The grouping tables in mapped pinned memory, [ids | slot | src] of T_max * K each, then the MMQ bounds: kernels
    // read and write them in place.  A cudaMemcpyAsync of them queues behind the expert blobs the copy stream already
    // holds (up to `ring` of them, ~70 us each), and the GPU idles meanwhile - measured 4.2 s of a 128K prompt's
    // 70 s at the default ring, 0.7 s with a 16-slot one.  STRATA_GROUP_COPY=1: the copies (the A/B arm).
    int32_t* grp_host = nullptr;
    int32_t* grp_dev = nullptr;          // its device alias
    size_t grp_n = 0, grp_tk = 0;        // int32s allocated; T_max * K (the offset of slot, and of src past it)
    uint16_t* dq_gu[DQ] = {};
    uint16_t* dq_d[DQ] = {};
    uint8_t* stage_dev[RING_MAX] = {};
    int ring = STAGE;                        // the slots of this layout's ring (ring_slots)
    // this layout's GU, H and Xq are the fused path's (moe_bufs' `fused`), so a layer that runs MMQ or the FP16 path may
    // only write stream_all_min() - 1 of its rows into them; fused_layout() keeps the two in step (#583, #954)
    bool fused_bufs = false;
    std::unique_ptr<Stager> stager;          // the unpinned experts' host copies (step 4)
    cudaEvent_t copied[RING_MAX] = {}, used[RING_MAX] = {};
    bool stage_live[RING_MAX] = {};
    // the event that releases each ring slot: its own `used`, or - when an MMQ group is gathered in one launch - the
    // `used` of the last slot gathered with it, recorded once for all of them (a later record only waits longer)
    int used_of[RING_MAX] = {};
    // PLE
    float* ple_emb = nullptr;
    std::vector<float> ple_pageable[2];      // the fallback when no more RAM can be pinned
    float* ple_emb_host[2] = {};             // pinned, double-buffered: the next chunk's rows are read while this
    cudaEvent_t ple_copied[2] = {};          // one runs; the event marks that buffer's upload done
    std::vector<uint32_t> ple_rows[2];
    float* ple_norm = nullptr;
    uint8_t* region = nullptr;               // the attention/MoE scratch region (idle while the PLE block runs)
    uint64_t region_bytes = 0;
    PrefillStats* stats = nullptr;
    // KV streaming: one layer's whole K/V, staged from the host copy per layer and chunk (identity layout)
    strata::kernels::KvHostPools stage;
    int32_t* ident_table = nullptr;
    cudaStream_t kv_copy = nullptr;
    cudaEvent_t kv_released = nullptr, kv_ready = nullptr;
    // layer split: the device, and the hand-off to the next stage (two pinned chunk buffers, used in turn)
    int device = -1;
    float* hand[2] = {};
    // STRATA_HANDOFF_PIPE: per hand-off buffer, "the chunk's work is done" and one "landed" event per piece
    int hand_pieces = 0;
    cudaEvent_t hand_done[2] = {};
    cudaEvent_t hand_ev[2][kHandPiecesMax] = {};
    std::unique_ptr<HandWorker> hand_worker;   // STRATA_HANDOFF_PIPE: the next stage's chunks' thread (lazily)
    // C-4: the chunk's token ids on the device, for one batched embedding gather
    int32_t* tok_dev = nullptr;
    std::vector<int32_t> tok_host;
    PinnedUpload tok_up;
    core::PinnedRing up_ring;                // #1996: every other upload of the prompt path whose host side is pageable
    // APR (prism.hadamard): per layer, this device's copies of the gate/up (n_embd) and down (n_ff) sign vectors
    // (nullptr: identity signs or not folded), resolved once in init so a prompt chunk reads a pointer
    std::vector<const float*> had_sx, had_sh;
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

Prefill::Prefill() : impl_(new Impl) {}
Prefill::~Prefill() {
    clear_taps();
    release();
}

void Prefill::clear_taps() {
    if (tap_dev_ != nullptr) {
        cudaFree(tap_dev_);
        tap_dev_ = nullptr;
    }
    tap_layers_.clear();
    on_taps = nullptr;
}

void Prefill::reset() {
    release();
    impl_.reset(new Impl);
    stats_ = PrefillStats{};
}

void Prefill::release() {
    if (!impl_) return;
    impl_->hand_worker.reset();   // idle here (run drains the chain before it returns); joins its thread
    if (impl_->cs) cudaStreamSynchronize(impl_->cs);
    if (impl_->copy) cudaStreamSynchronize(impl_->copy);
    if (impl_->kv_copy) cudaStreamSynchronize(impl_->kv_copy);
    for (int i = 0; i < RING_MAX; ++i) {
        if (impl_->copied[i]) cudaEventDestroy(impl_->copied[i]);
        if (impl_->used[i]) cudaEventDestroy(impl_->used[i]);
    }
    for (int b = 0; b < 2; ++b) {
        if (impl_->hand[b]) cudaFreeHost(impl_->hand[b]);
        if (impl_->hand_done[b]) cudaEventDestroy(impl_->hand_done[b]);
        for (cudaEvent_t& e : impl_->hand_ev[b])
            if (e) cudaEventDestroy(e);
        if (impl_->ple_copied[b]) cudaEventDestroy(impl_->ple_copied[b]);
        if (impl_->ple_emb_host[b] && impl_->ple_pageable[b].empty()) cudaFreeHost(impl_->ple_emb_host[b]);
    }
    if (impl_->copy) cudaStreamDestroy(impl_->copy);
    if (impl_->kv_copy) cudaStreamDestroy(impl_->kv_copy);
    if (impl_->kv_released) cudaEventDestroy(impl_->kv_released);
    if (impl_->kv_ready) cudaEventDestroy(impl_->kv_ready);
    if (impl_->grp_host) cudaFreeHost(impl_->grp_host);
    impl_->steps_up.release();
    impl_->tok_up.release();
    impl_->up_ring.release();
    for (float* p : {impl_->cpu_x, impl_->cpu_rows, impl_->cpu_ones})
        if (p) cudaFreeHost(p);
    for (cudaEvent_t e : impl_->cpu_ev)
        if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : impl_->cpu_wall)
        if (e) cudaEventDestroy(e);
    for (void* p : impl_->owned) cudaFree(p);
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
    std::vector<char> wmma_only;               // per layer: in the MMQ grouping for the WMMA tiles only (pf_wmma: IQ1_M)
    std::vector<char> fo;                      // per layer: no MMQ here but the native fused kernels cover its formats
                                               // (gfx11: UD-Q4_K_XL's Q4_K / Q5_K and Q5_1 / Q8_0 experts, STRATA_PF_FUSED=1)
    size_t gu_max = 0, d_max = 0;
    // STRATA_MMQ_DIRECT=1: per layer, whether resident experts are read in their cache slots (pointer-table MMQ),
    // and the bytes the down product reads past its matrix there (zeroed in the slot's guard before each group)
    std::vector<char> direct;
    std::vector<uint32_t> d_over;
    // STRATA_MMQ_GROUP_SORT=1 (without STRATA_MMQ_RESIDENT_SORT_NE): per MMQ layer, whether both products' 16-expert
    // launches run whole tiles on every visible GPU (mmq::whole_tiles), so its full groups may be taken in row-count order
    std::vector<char> group_sort;
};
// Volta (sm_70): MMQ runs on dp4a there (no int8 tensor cores).  pf_wmma(): the layers MMQ would take, and the IQ1_M
// ones the FP16 path would, run as one launch per expert group on FP16 tensor-core tiles instead
// (strata::kernels::gemm_iq_f16_grouped: each weight superblock dequantized once into shared memory by the FP16 path's
// own formulas, FP16 activations, FP32 sums).  OPT-IN: STRATA_PF_WMMA=1 (the activations are FP16 rather than MMQ's
// q8_1, so the answers differ from the default).  Only when EVERY visible CUDA card is compute capability 7.0 or
// newer: the tiles need tensor cores, and a Volta + Pascal rig (one process, one decision) keeps MMQ.
bool pf_wmma() {
#if defined(__HIPCC__)
    return false;
#else
    static const bool on = [] {
        const char* e = std::getenv("STRATA_PF_WMMA");
        if (e == nullptr || e[0] != '1') return false;
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return false;
        for (int d = 0; d < n; ++d) {
            int major = 0;
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, d) != cudaSuccess || major < 7) return false;
        }
        return true;
    }();
    return on;
#endif
}
const MmqPlan& mmq_plan() {
    static const MmqPlan plan = [] {
        MmqPlan p;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const char* env = std::getenv("STRATA_PREFILL_MMQ");
        const bool on = mmq::built() && (env == nullptr || std::atoi(env) != 0);
        const int64_t layers = lay.native ? (int64_t) lay.fmt.size() : lay.n_layers;
        p.layer.assign((size_t) std::max<int64_t>(layers, 0), 0);
        p.fo.assign(p.layer.size(), 0);
        p.wmma_only.assign(p.layer.size(), 0);
        p.fallback = !on || layers <= 0;
        for (int64_t l = 0; on && l < layers; ++l) {
            const int gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42, dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
            // #420: a tile on every GPU for these shapes (gate+up: 1280 rows, down: N rows), else the FP16 path
            if (!mmq::fits(gt, 1280) || !mmq::fits(dt, N)) {
                // pf_wmma: a format dq_dispatch takes (IQ1_M, an IQ4_NL down) rides the MMQ grouping, computed by the WMMA tiles only
                if (pf_wmma() && lay.native && (dt == 42 || dt == 20) && strata::kernels::iq_row_bytes(gt, N) > 0) {
                    p.layer[(size_t) l] = 1;
                    p.wmma_only[(size_t) l] = 1;
                    p.any = true;
                    p.gu_max = std::max(p.gu_max, mmq::matrix_bytes(gt, 1280, N));
                    p.d_max = std::max(p.d_max, mmq::matrix_bytes(dt, N, 640));
                    continue;
                }
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
        // STRATA_MMQ_DIRECT=1: a native layer's resident experts are read in place when the pointer-table MMQ covers
        // both products, gate/up reads nothing past its matrix, the down over-read fits the slot guard, and the blob
        // is exactly [gate | up | down] in MMQ's matrix sizes (the bytes a gather would copy)
        p.direct.assign(p.layer.size(), 0);
        p.d_over.assign(p.layer.size(), 0);
        if (on && lay.native && mmq::direct_mode() == 1) {
            int64_t nd = 0;
            for (int64_t l = 0; l < layers; ++l) {
                if (!p.layer[(size_t) l]) continue;
                const auto& f = lay.fmt[(size_t) l];
                const int gt = f.gu_type, dt = f.d_type;
                if (!mmq::tab_fits(gt, 1280) || !mmq::tab_fits(dt, N)) continue;
                if (mmq::overread_bytes(gt, N) != 0 || mmq::overread_bytes(dt, 640) > mmq::kSlotGuard) continue;
                if (f.up_off != mmq::matrix_bytes(gt, 1280, N) / 2 || f.down_off != 2 * f.up_off) continue;
                if (mmq::matrix_bytes(dt, N, 640) != f.bytes - f.down_off) continue;
                p.direct[(size_t) l] = 1;
                p.d_over[(size_t) l] = (uint32_t) mmq::overread_bytes(dt, 640);
                ++nd;
            }
            std::fprintf(stderr, "strata: prompt MMQ reads resident experts in place (STRATA_MMQ_DIRECT=1): %lld of %lld "
                                 "layers\n", (long long) nd, (long long) layers);
        }
        // the verdict depends only on the layer's formats and the visible GPUs: taken once here, not per chunk
        p.group_sort.assign(p.layer.size(), 0);
        if (on && detail::mmq_group_sort_requested() && !detail::mmq_resident_sort_requested()) {
            int64_t ns = 0;
            for (int64_t l = 0; l < layers; ++l) {
                if (!p.layer[(size_t) l]) continue;
                const int gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42, dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
                p.group_sort[(size_t) l] = mmq::whole_tiles(gt, 1280, MMQ_GROUP) && mmq::whole_tiles(dt, N, MMQ_GROUP);
                ns += p.group_sort[(size_t) l];
            }
            std::fprintf(stderr, "strata: prompt MMQ sorts the full expert groups (STRATA_MMQ_GROUP_SORT=1): %lld of %lld "
                                 "layers\n", (long long) ns, (long long) layers);
        }
        return p;
    }();
    return plan;
}
// the FP16 buffers (Xs, Hh): the FP16 path's, or pf_wmma's activations and down input
bool fp16_bufs(const MmqPlan& mp) { return mp.fallback || pf_wmma(); }
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
    return {std::max<size_t>(ts * K * 1280, (fused::group_bytes((int64_t) (T * K), (int) n_expert) + 3) / 4),
            std::max<size_t>(ts * K * 640, (fused::act_bytes((int64_t) (T * K), 640) + 3) / 4),
            std::max(mmq::q8_bytes((int64_t) (ts * K), N), fused::act_bytes((int64_t) T, N)),
            mmq::q8_bytes((int64_t) (ts * K), 640)};
}
uint64_t moe_set_bytes(size_t T, int64_t n_expert, bool fused) {
    const MmqPlan& mp = mmq_plan();
    const MoeBufs mb = moe_bufs(T, n_expert, fused);
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * n_expert, ok); a.take<float>(T * K, ok); a.take<int32_t>(T * K, ok); a.take<int32_t>(T * K, ok);
    a.take<int32_t>(T * K, ok);
    if (fp16_bufs(mp)) a.take<uint16_t>(T * K * N, ok);
    a.take<float>(mb.gu, ok);
    if (fp16_bufs(mp)) a.take<uint16_t>(T * K * 640, ok);
    a.take<float>(T * K * N, ok); a.take<float>(T * 640, ok);
    a.take<float>(T * 640, ok); a.take<uint16_t>(T * 640, ok); a.take<float>(T * N, ok); a.take<float>(T, ok);
    if (mp.any) {
        a.take<uint8_t>(mb.xq, ok);
        a.take<float>(mb.h, ok);
        a.take<uint8_t>(mb.hq, ok);
        a.take<float>(T * K, ok); a.take<float>(T * K, ok); a.take<float>(MMQ_GROUP * 4, ok);
        a.take<float>(T * K, ok);
        // APR: the rotated copy of `mixed` for the MMQ path (carve's m.Xr) - it must be counted here too, or the MoE
        // set outgrows the region `carve` sizes from this, and `init` fails with "do not fit" at any chunk
        if (strata::kernels::cpu::hadamard_spec().any()) a.take<float>(T * N, ok);
    }
    return a.used;
}
}

// The first BF16 and the first FP16 GEMM of a process each pay a one-time library / code-object load in hipBLAS (1.2 s and
// 1.0 s on an R9700, hipBLASLt 1.2.0: STRATA_COLD_TRACE), which the first prompt then carries (a 700-token prompt: 5.9 s,
// 3.5 s of it these).  One tiny product of each type, on a handle and stream of its own, moves that cost out of the first
// request; generate.cpp starts it on a thread before the expert arena loads, so it costs no start-up time, and
// Prefill::init waits for it.  The outputs go to scratch buffers of their own, so no later result changes.
// STRATA_PREFILL_WARM=0 turns it off.
void warm_gemm_libraries(int device) {
    static const bool off = [] { const char* e = std::getenv("STRATA_PREFILL_WARM"); return e != nullptr && e[0] == '0'; }();
    if (off) return;
#if defined(STRATA_USE_HIP)
    // #3204: not when the process sees cards of more than one architecture (gfx1151 + gfx1201: prompts -25%); see
    // warm_policy.hpp.  STRATA_PREFILL_WARM=1 asks for it anyway.
    static const bool arch_ok = [] {
        std::vector<std::string> archs;
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) { (void) cudaGetLastError(); return true; }
        for (int i = 0; i < n && i < 32; ++i) {
            cudaDeviceProp p{};
            if (cudaGetDeviceProperties(&p, i) == cudaSuccess) archs.emplace_back(p.gcnArchName);
            else (void) cudaGetLastError();
        }
        const bool wanted = warm_gemm_wanted(archs, std::getenv("STRATA_PREFILL_WARM"));
        if (!wanted)
            std::fprintf(stderr, "strata prefill: no hipBLAS warm-up: the cards are of more than one architecture "
                                 "(it slows prompts there, #3204); the first prompt loads the libraries (STRATA_PREFILL_WARM=1 asks for it)\n");
        return wanted;
    }();
    if (!arch_ok) return;
    static std::mutex mu;
    static unsigned done_mask = 0;   // per device: each card loads its own code objects
    if (device < 0 || device >= 32) return;
    std::lock_guard<std::mutex> lk(mu);
    if (done_mask & (1u << device)) return;
    done_mask |= 1u << device;
    int prev = 0;
    if (cudaGetDevice(&prev) != cudaSuccess) { (void) cudaGetLastError(); return; }
    if (prev != device && cudaSetDevice(device) != cudaSuccess) { (void) cudaGetLastError(); return; }
    const auto t0 = std::chrono::steady_clock::now();
    constexpr int64_t T = 64, Nn = 64, K = 320;
    uint16_t *x = nullptr, *w = nullptr;
    float* y = nullptr;
    cudaStream_t st = nullptr;
    bool ok = cudaStreamCreate(&st) == cudaSuccess && cudaMalloc((void**) &x, (size_t) T * K * 2) == cudaSuccess &&
              cudaMalloc((void**) &w, (size_t) Nn * K * 2) == cudaSuccess && cudaMalloc((void**) &y, (size_t) T * Nn * 4) == cudaSuccess;
    if (ok) {
        Gemm gm;
        std::string err;
        ok = gm.init((void*) st, T * K, err);
        if (ok) {
            cudaMemsetAsync(x, 0, (size_t) T * K * 2, st);
            cudaMemsetAsync(w, 0, (size_t) Nn * K * 2, st);
            gm.bf16(x, w, y, T, Nn, K);
            gm.f16(x, w, y, T, Nn, K);
            cudaStreamSynchronize(st);
        }
    }
    (void) cudaGetLastError();
    if (x) cudaFree(x);
    if (w) cudaFree(w);
    if (y) cudaFree(y);
    if (st) cudaStreamDestroy(st);
    if (prev != device) cudaSetDevice(prev);
    if (ok)
        std::fprintf(stderr, "strata prefill: warmed the BF16 and FP16 GEMM libraries on GPU %d in %.0f ms (STRATA_PREFILL_WARM=0 skips)\n",
                     device, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
#else
    (void) device;
#endif
}

bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                   core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes, bool borrow_spill) {
    Impl& m = *impl_;
#if defined(STRATA_GPU_METAL)
    (void) wt; (void) src; (void) cache; (void) host_res; (void) stream; (void) borrow;
    (void) borrow_bytes;
    m.T = chunk > 0 ? chunk : 1;
    return strata::metal_engine::window_ensure(g, ss, ss.qsa_states[ss.qsa_primary()].max_cells, err);
#endif
    m.wt = &wt; m.g = &g; m.ss = &ss; m.src = src; m.cache = cache; m.host_res = host_res;
    m.T = chunk; m.cs = (cudaStream_t) stream; m.stats = &stats_;
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K) {
        err = "prefill: geometry differs from the artifact's"; return false;
    }
    {   // a native pack's experts: this path's expert buffers and products (and so their Hadamard rotations) are
        // N wide in and 640 wide in the middle; the sign vectors are looked up at the layout's own widths
        const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
        for (size_t l = 0; lay.native && l < lay.fmt.size(); ++l)
            if (lay.fmt[l].n_embd != N || lay.fmt[l].n_ff != 640) {
                err = "prefill: layer " + std::to_string(l) + "'s experts are " + std::to_string(lay.fmt[l].n_embd) +
                      " x " + std::to_string(lay.fmt[l].n_ff) + ", not the artifact's 2560 x 640";
                return false;
            }
        // APR: the sign vectors on this device, once (hadamard_device_signs takes a mutex and, the first time per
        // device, cudaMallocs its copy): run_impl reads m.had_sx / m.had_sh
        m.had_sx.assign(lay.native ? lay.fmt.size() : 0, nullptr);
        m.had_sh.assign(m.had_sx.size(), nullptr);
        for (size_t l = 0; l < m.had_sx.size(); ++l) {
            if (lay.fmt[l].had_gu)
                m.had_sx[l] = strata::kernels::hadamard_device_signs(lay.fmt[l].gu_signs, (int) lay.fmt[l].n_embd);
            if (lay.fmt[l].had_d)
                m.had_sh[l] = strata::kernels::hadamard_device_signs(lay.fmt[l].d_signs, (int) lay.fmt[l].n_ff);
        }
    }
    cudaGetDevice(&m.device);
    if (stage_le_ < 0) stage_le_ = g.n_layers;
    if (stage_lb_ < 0 || stage_lb_ >= stage_le_ || stage_le_ > g.n_layers || (stage_le_ < g.n_layers) != (next_ != nullptr)) {
        err = "prefill: the stage's layer range is wrong";
        return false;
    }
    // the CPU share's staged-chunk limit, before the ring is sized below (a caller that sizes loans first armed it)
    arm_cpu_share(cpu_pool_ != nullptr);
    for (int b = 0; next_ != nullptr && b < 2; ++b)
        if (!m.hand[b] && cudaHostAlloc((void**) &m.hand[b], (size_t) chunk * D * 4, cudaHostAllocPortable) != cudaSuccess) {
            err = "prefill: the layer split's hand-off buffers";
            return false;
        }
    if (next_ != nullptr && handoff_pipe_pieces() > 0 && m.hand_pieces == 0) {   // events only: no device memory
        bool ok = true;
        auto make = [](cudaEvent_t& e) { return e != nullptr || cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess; };
        for (int b = 0; b < 2; ++b) {   // (a retry after a failed init keeps the events it made)
            ok = ok && make(m.hand_done[b]);
            for (int i = 0; ok && i < handoff_pipe_pieces(); ++i) ok = make(m.hand_ev[b][i]);
        }
        if (!ok) { err = "prefill: the layer split's hand-off events (STRATA_HANDOFF_PIPE)"; return false; }
        m.hand_pieces = handoff_pipe_pieces();
    }
    if (m.tok_dev == nullptr) {
        if (const cudaError_t e = cudaMalloc((void**) &m.tok_dev, (size_t) chunk * sizeof(int32_t)); e != cudaSuccess) {
            err = std::string("prefill: the token id buffer (") + cudaGetErrorString(e) + ")";
            return false;
        }
        m.owned.push_back(m.tok_dev);
        m.tok_host.resize((size_t) chunk);
        m.tok_up.init((size_t) chunk);
    }
    if (cudaStreamCreateWithFlags(&m.copy, cudaStreamNonBlocking) != cudaSuccess) { err = "prefill: copy stream"; return false; }
    const size_t T = (size_t) chunk;
    m.T_max = chunk;
    m.borrowed = borrow != nullptr;
    m.borrow_spill = m.borrowed && borrow_spill;
    bool ok = true;
    // one-time: events, the stager, the host buffers (for the largest chunk), the identity page table
    for (int i = 0; i < ring_cap(); ++i) {
        if (cudaEventCreateWithFlags(&m.copied[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
        if (cudaEventCreateWithFlags(&m.used[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
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
        // #1353: ... but a source answers transient() for every blob it copies out of the GGUF, including the ones whose
        // pages are already in the page cache (a Q8 pack on a big-RAM box with a warm cache).  Those copies are memcpy
        // from RAM, where 32 threads and a 128-deep ring ran 4-7x slower than the defaults.  So ask the OS: the SSD
        // profile only when fewer than 90% of a sample of the blobs' pages are resident (-1: it cannot tell, keep it).
        if (files && !stv) {
            const char* ev = std::getenv("STRATA_STAGER_SSD");   // 1 / 0: force the profile (A/B, support)
            if (ev != nullptr && ev[0] != '\0') files = ev[0] != '0';
            else if (const double warm = src->cached_share(256); warm >= 0.9) {
                files = false;
                std::fprintf(stderr, "prefill: the GGUF experts are %.0f%% in the page cache; the stager takes the RAM profile\n", warm * 100.0);
            }
        }
        const int threads = stv ? std::clamp(std::atoi(stv), 1, 32) : files ? 32 : std::max(2, std::min(4, hw / 4));
        if (files && std::getenv("STRATA_STAGER_RING") == nullptr) m.stager->kRing = 4 * threads;
        // pp-opt: with files, each thread claims runs of neighbouring blobs and reads them as one request (copy_blobs);
        // STRATA_STAGER_BATCH=1 is the A/B arm (one blob per request, as before)
        {
            const char* bv = std::getenv("STRATA_STAGER_BATCH");
            const int want = bv ? std::atoi(bv) : files ? 8 : 1;
            m.stager->batch = std::clamp(want, 1, Stager::kMaxBatch);
        }
        if (!m.stager->init((size_t) MAXBLOB(), threads)) ok = false;
        m.stager->batch = std::min(m.stager->batch, m.stager->kRing);
    }
    m.steps_host.resize(T * strata::kernels::kStepCount);
    m.steps_up.init(T * strata::kernels::kStepCount);
    m.ids_host.resize(T * K); m.slot_host.resize(T * K); m.src_host.resize(T * K); m.cnt.resize(m.g->n_expert); m.off.resize(m.g->n_expert + 1);
    {
        const size_t need = 3 * T * K + (size_t) (2 * (m.g->n_expert + m.g->n_expert / MMQ_GROUP + 2));
        const char* gc = std::getenv("STRATA_GROUP_COPY");
        if (m.grp_n < need && !(gc && gc[0] == '1')) {
            if (m.grp_host) cudaFreeHost(m.grp_host);
            m.grp_host = m.grp_dev = nullptr;
            m.grp_n = m.grp_tk = 0;
            void *h = nullptr, *d = nullptr;
            if (!force_pageable() &&
                cudaHostAlloc(&h, need * 4, cudaHostAllocMapped | (core::peer_portable() ? cudaHostAllocPortable : 0)) == cudaSuccess &&
                cudaHostGetDevicePointer(&d, h, 0) == cudaSuccess) {
                m.grp_host = (int32_t*) h;
                m.grp_dev = (int32_t*) d;
                m.grp_n = need;
                m.grp_tk = T * K;
            } else {                              // the copies, as before
                if (h) cudaFreeHost(h);
                cudaGetLastError();
            }
        }
    }
    for (int b = 0; b < 2; ++b) {
        if (!m.ple_emb_host[b] &&
            (force_pageable() || cudaHostAlloc((void**) &m.ple_emb_host[b], (size_t) T * N * 4, cudaHostAllocDefault) != cudaSuccess)) {
            cudaGetLastError();
            m.ple_pageable[b].resize(T * N);          // pageable: the upload is staged before it returns
            m.ple_emb_host[b] = m.ple_pageable[b].data();
        }
        if (!m.ple_copied[b] && cudaEventCreateWithFlags(&m.ple_copied[b], cudaEventDisableTiming) != cudaSuccess)
            ok = false;
        m.ple_rows[b].resize(T * strata::kernels::PLE_N_HEADS);
    }
    if (ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {   // KV streaming: the staging pool's identity page table
        const int64_t pages = ss.qsa_states[ss.qsa_primary()].n_pages;
        std::vector<int32_t> ident((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) ident[(size_t) i] = (int32_t) i;
        if (cudaMalloc((void**) &m.ident_table, ident.size() * 4) != cudaSuccess ||
            cudaMemcpy(m.ident_table, ident.data(), ident.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess)
            ok = false;
        else
            m.owned.push_back(m.ident_table);
    }
    if (!ok) { err = "prefill: host buffers or events for a chunk of " + std::to_string(chunk) + " tokens"; return false; }
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.spill = m.borrow_spill;
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
        static const bool f16_range = std::getenv("STRATA_F16_RANGE") != nullptr;   // what reaches FP16's range
        if (f16_range) { set_f16_range(true); gemm_set_f16_range(true); }
        // the hyper-connection's xn16 straight as the FP16 the tensor-core GEMM reads (its two readers then skip the
        // conversion pass; the same values): STRATA_XN_F16=0 keeps the BF16 image
        static const bool xn_env = [] { const char* e = std::getenv("STRATA_XN_F16"); return e == nullptr || e[0] != '0'; }();
        m.xn_f16 = xn_env && !m.f16_io && m.gemm.bf16_converts_x();
    }
    if (!carve(T, &o)) {
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        (void) cudaGetLastError();
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit (" +
              std::to_string(fb >> 20) + " of " + std::to_string(tb >> 20) + " MiB free at the failure; " +
              std::to_string(o.used >> 20) + " MiB taken, the failing buffer wanted " +
              std::to_string(o.failed_bytes >> 20) + " MiB)";
        return false;
    }
    { int wd = 0; if (cudaGetDevice(&wd) == cudaSuccess) warm_gemm_libraries(wd); else (void) cudaGetLastError(); }
    return true;
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
    m.mixed_h = o.take<uint16_t>(T * N, ok);
    // Embedding rows are dead after gr_broadcast on m.cs; subsequent half outputs use the same stream.
    m.bo = m.emb;
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
        m.Xs = fp16_bufs(mp) ? c.take<uint16_t>(T * K * N, ok) : nullptr;
        m.GU = c.take<float>(mb.gu, ok);
        m.Hh = fp16_bufs(mp) ? c.take<uint16_t>(T * K * 640, ok) : nullptr;
        m.Dm = c.take<float>(T * K * N, ok);
        m.sgate = c.take<float>(T * 640, ok); m.sup = c.take<float>(T * 640, ok); m.sh_h = c.take<uint16_t>(T * 640, ok);
        m.shared = c.take<float>(T * N, ok); m.sg = c.take<float>(T, ok);
        if (mp.any) {
            m.Xq = c.take<uint8_t>(mb.xq, ok);
            m.H = c.take<float>(mb.h, ok);
            m.Hq = c.take<uint8_t>(mb.hq, ok);
            m.Xscale = c.take<float>(T * K, ok); m.Hscale = c.take<float>(T * K, ok);
            m.grp_tail = c.take<float>(MMQ_GROUP * 4, ok);
            m.row_sd = c.take<float>(T * K, ok);
            // APR: the rotated expert input (only for Hadamard-folded packs)
            m.Xr = strata::kernels::cpu::hadamard_spec().any() ? c.take<float>(T * N, ok) : nullptr;
        }
        if (base == nullptr) ok = false;
    }
    for (int i = 0; i < DQ; ++i) { m.dq_gu[i] = o.take<uint16_t>(1280 * 2560, ok); m.dq_d[i] = o.take<uint16_t>(2560 * 640, ok); }
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
    const int nstage = stage_slots(T);   // the staged walk may use STAGE_GRP of them
    if (o.base == nullptr && m.ring > 0) {
        // OWNED buffers: the ring in ONE allocation.  384 separate 2.7 MiB cudaMallocs each round up to a 2 MiB page
        // (~1.3 MiB a slot, ~0.5 GiB in all) that no count ever saw.  A borrowed region keeps its per-slot layout (and
        // so its price, `bytes_needed`: the loans' slot counts do not move).
        uint8_t* ring_base = o.take<uint8_t>((size_t) nstage * (size_t) MAXBLOB(), ok);
        for (int i = 0; ok && i < nstage; ++i) {
            m.stage_dev[i] = ring_base + (size_t) i * (size_t) MAXBLOB();
            m.stage_live[i] = false;
            m.used_of[i] = i;
        }
    } else {
        for (int i = 0; i < nstage; ++i) {
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

bool Prefill::relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err) {
    Impl& m = *impl_;
    if (m.borrow_spill) {
        err = "prefill: a hybrid borrowed/owned layout cannot be changed between requests";
        return false;
    }
    if (!m.borrowed || borrow == nullptr || chunk <= 0 || chunk > m.T_max) {
        err = "prefill: relayout needs borrowed buffers and a chunk of at most " + std::to_string(m.T_max);
        return false;
    }
    if (cudaStreamSynchronize(m.cs) != cudaSuccess || cudaStreamSynchronize(m.copy) != cudaSuccess ||
        (m.kv_copy && cudaStreamSynchronize(m.kv_copy) != cudaSuccess)) {
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
        cudaMemGetInfo(&fb, &tb);
        (void) cudaGetLastError();
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit (" +
              std::to_string(fb >> 20) + " of " + std::to_string(tb >> 20) + " MiB free at the failure; " +
              std::to_string(o.used >> 20) + " MiB taken, the failing buffer wanted " +
              std::to_string(o.failed_bytes >> 20) + " MiB)";
        return false;
    }
    return true;
}

int64_t Prefill::chunk() const { return impl_->T; }

bool Prefill::draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                       std::string& err, const char** why) {
    Impl& m = *impl_;
#if defined(STRATA_GPU_METAL)
    (void) m; (void) mtp; (void) R_rows; (void) next_tokens; (void) n; (void) cell0; (void) err;
    if (why) *why = "the Metal window feeds the draft layer itself";
    return false;
#endif
    const core::OnDevice on_device(m.device);   // the stage's own GPU current (a layer split's drafter: the last stage)
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
    int64_t B = std::min<int64_t>(n - r0, (int64_t) (m.region_bytes / per_row) & ~(int64_t) 63);
    if (st.kv_mode == 2)   // a ring: one batch's cells must not share a slot (a batch can straddle one page more)
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
    if (timing) cudaStreamSynchronize(m.cs);
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
        // #1996: bh is on this stack and tk is refilled every pass: both go up through pinned slots
        if (q8 && m.up_ring.h2d(bnd, bh, sizeof bh, m.cs) != cudaSuccess) {
            err = "prefill: the draft bounds' upload failed";
            return false;
        }
        if (m.up_ring.h2d(tok, tk.data(), (size_t) nb * 4, m.cs) != cudaSuccess) {
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
        cudaMemcpyAsync(hn, R_rows + (size_t) b0 * HCN, (size_t) nb * HCN * 4, cudaMemcpyDeviceToDevice, m.cs);
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
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = std::string("prefill: the draft layer's K/V: ") + cudaGetErrorString(cudaGetLastError());
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

namespace {
// APR: the down projections' sign vectors on the current device (the peer's or the helper's), one per layer,
// resolved once at setup (hadamard_device_signs: a mutex and, the first time on that device, a cudaMalloc) into
// P.had_sh, which a prompt chunk reads directly.  The gate/up signs are not needed there: the primary sends its
// already rotated input (m.Xr).
void peer_had_signs(PeerPrefill& P) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    P.had_sh.assign(lay.native ? lay.fmt.size() : 0, nullptr);
    for (size_t l = 0; l < P.had_sh.size(); ++l)
        if (lay.fmt[l].had_d)
            P.had_sh[l] = strata::kernels::hadamard_device_signs(lay.fmt[l].d_signs, (int) lay.fmt[l].n_ff);
}
}  // namespace

bool Prefill::set_stage_helper(Prefill* helper, std::string& err) {
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
        ok = cudaEventCreateWithFlags(&pp->ev_in, cudaEventDisableTiming) == cudaSuccess;
    }
    int prev = 0;
    cudaGetDevice(&prev);
    pp->dev = h.device;
    cudaSetDevice(pp->dev);
    ok = ok && cudaStreamCreateWithFlags(&pp->s, cudaStreamNonBlocking) == cudaSuccess &&
         cudaStreamCreateWithFlags(&pp->s_out, cudaStreamNonBlocking) == cudaSuccess &&
         cudaStreamCreateWithFlags(&pp->s_cp, cudaStreamNonBlocking) == cudaSuccess &&
         cudaEventCreateWithFlags(&pp->ev_done, cudaEventDisableTiming) == cudaSuccess;
    for (int i = 0; ok && i < PeerPrefill::kGrpEv; ++i)
        ok = cudaEventCreateWithFlags(&pp->ev_grp[i], cudaEventDisableTiming) == cudaSuccess;
    for (int b = 0; ok && b < 2; ++b) ok = cudaEventCreateWithFlags(&pp->ev_dm[b], cudaEventDisableTiming) == cudaSuccess;
    // the ring's events; its slots are the helper stage's own ring, bound per prompt (bind_stage_helper)
    pp->pcopied.assign((size_t) kSplitHelpRing, nullptr);
    pp->pused.assign((size_t) kSplitHelpRing, nullptr);
    for (int i = 0; ok && i < kSplitHelpRing; ++i)
        ok = cudaEventCreateWithFlags(&pp->pcopied[(size_t) i], cudaEventDisableTiming) == cudaSuccess &&
             cudaEventCreateWithFlags(&pp->pused[(size_t) i], cudaEventDisableTiming) == cudaSuccess;
    if (ok) peer_had_signs(*pp);   // APR: on the helper's device
    cudaSetDevice(prev);
    if (!ok) { err = "prefill: the layer split's prompt help (streams and events)"; return false; }
    helper_ = helper;
    m.help_pp = std::move(pp);
    return true;
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
        auto grow = [](auto*& p, size_t& cap, size_t n) {
            if (cap >= n) return true;
            if (p) cudaFreeHost(p);
            p = nullptr;
            cap = 0;
            if (cudaHostAlloc((void**) &p, n * sizeof(*p), cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess) {
                cudaGetLastError();
                p = nullptr;
                return false;
            }
            cap = n;
            return true;
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

bool Prefill::set_peer(core::PeerExperts* peer, int64_t cap_rows, std::string& err) {
    Impl& m = *impl_;
    if (peer == nullptr || !peer->valid()) { m.pp.reset(); return true; }
    // Without P2P the activations and the result rows take the mapped host route of the layer split's helper
    // (STRATA_PF_PEER_HOST=0: refuse as before, the prompt path then stays on the primary)
    const bool host_route_ok = [] { const char* v = std::getenv("STRATA_PF_PEER_HOST"); return v == nullptr || std::atoi(v) != 0; }();
    if (!peer->p2p() && !host_route_ok) { err = "prefill peer: the two GPUs cannot access each other (no P2P)"; return false; }
    if (!mmq_plan().any) { err = "prefill peer: needs the MMQ prompt path"; return false; }
    {   // NVFP4: the peer's MMQ products have no per-expert scales (grp_tail, row_sd); its decode rows do
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (const auto& f : lay.fmt)
            if (lay.native && f.tail_off) { err = "prefill peer: not for NVFP4 experts yet"; return false; }
    }
    if (strata::kernels::cpu::hadamard_spec().any()) {   // APR: the peer's MMQ share does not rotate; declined
        std::fprintf(stderr, "strata: prefill peer: not used for a Hadamard-folded (APR) pack\n");
        m.pp.reset();
        return true;
    }
    auto pp = std::make_unique<PeerPrefill>();
    pp->peer = peer;
    pp->p2p = peer->p2p();
    pp->T_max = m.T_max;
    pp->cap_rows = std::max<int64_t>(1, std::min<int64_t>(cap_rows, m.T_max * K));
    const int64_t R = pp->cap_rows;
    const MmqPlan& mp = mmq_plan();
    if (cudaEventCreateWithFlags(&pp->ev_in, cudaEventDisableTiming) != cudaSuccess) { err = "prefill peer: event"; return false; }
    if (!pp->p2p) {   // the host route's copies on a stream of their own (STRATA_PF_PEER_XSTREAM=0: the compute stream)
        const char* xv = std::getenv("STRATA_PF_PEER_XSTREAM");
        if ((xv == nullptr || std::atoi(xv) != 0) &&
            (cudaStreamCreateWithFlags(&pp->x_stream, cudaStreamNonBlocking) != cudaSuccess ||
             cudaEventCreateWithFlags(&pp->ev_x, cudaEventDisableTiming) != cudaSuccess)) {
            err = "prefill peer: copy stream";
            return false;
        }
    }
    int prev = 0;
    cudaGetDevice(&prev);
    pp->dev = peer->device();
    cudaSetDevice(pp->dev);
    bool ok = cudaStreamCreateWithFlags(&pp->s, cudaStreamNonBlocking) == cudaSuccess &&
              cudaEventCreateWithFlags(&pp->ev_done, cudaEventDisableTiming) == cudaSuccess;
    {
        const char* v = std::getenv("STRATA_PF_PEER_OUT_PIPE");   // =0: one copy after the last group (the A/B)
        pp->out_pipe = (v == nullptr || std::atoi(v) != 0) || pp->compact;   // compact always pipes
    }
    if (ok && pp->out_pipe) {
        ok = cudaStreamCreateWithFlags(&pp->s_out, cudaStreamNonBlocking) == cudaSuccess;
        for (int i = 0; ok && i < PeerPrefill::kGrpEv; ++i)
            ok = cudaEventCreateWithFlags(&pp->ev_grp[i], cudaEventDisableTiming) == cudaSuccess;
    }
    auto take = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (!ok || cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; return nullptr; }
        pp->owned.push_back(p);
        return p;
    };
    pp->mixed = (float*) take((size_t) m.T_max * N * 4);
    {
        const char* v = std::getenv("STRATA_PF_PEER_COMPACT");
        pp->compact = (v == nullptr || std::atoi(v) != 0) || !pp->p2p;   // the layer-sized path copies over P2P only
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
        for (int b = 0; b < 2 && ok; ++b) ok = cudaEventCreateWithFlags(&pp->ev_dm[b], cudaEventDisableTiming) == cudaSuccess;
        const char* fs = std::getenv("STRATA_PF_PEER_STREAM");
        pp->ps_frac = fs ? std::atof(fs) : 0.35;   // measured: 0.25-0.5 all ~1950-1970 at 32K, 0.35 best
        if (pp->ps_frac > 0.0 && !mp.fallback && m.T_max >= stream_all_min()) {
            pp->RP = 48;
            if (const char* pr = std::getenv("STRATA_PF_PEER_RING"); pr != nullptr) pp->RP = std::atoi(pr);
            pp->pstage.assign((size_t) pp->RP, nullptr);
            pp->pcopied.assign((size_t) pp->RP, nullptr);
            pp->pused.assign((size_t) pp->RP, nullptr);
            pp->plive.assign((size_t) pp->RP, 0);
            ok = ok && cudaStreamCreateWithFlags(&pp->s_cp, cudaStreamNonBlocking) == cudaSuccess;
            for (int i = 0; ok && i < pp->RP; ++i) {
                pp->pstage[(size_t) i] = (uint8_t*) take((size_t) MAXBLOB());
                ok = ok && cudaEventCreateWithFlags(&pp->pcopied[(size_t) i], cudaEventDisableTiming) == cudaSuccess &&
                     cudaEventCreateWithFlags(&pp->pused[(size_t) i], cudaEventDisableTiming) == cudaSuccess;
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
    if (!pp->p2p && pp->compact) {
        const char* v = std::getenv("STRATA_PF_PEER_SUMS");
        pp->sums = v == nullptr || std::atoi(v) != 0;
    }
    if (pp->sums) {
        const char* v = std::getenv("STRATA_PF_PEER_F16");
        pp->f16 = (v == nullptr || std::atoi(v) != 0) && N % 8 == 0;
        if (pp->f16) pp->sum16 = (uint16_t*) take((size_t) m.T_max * N * 2);
        const char* gv = std::getenv("STRATA_PF_PEER_GATHER");
        pp->gather = gv == nullptr || std::atoi(gv) != 0;
        if (pp->gather) {   // tokens and starts (<= 2 rows + 1 per group) and rows: <= 3 x the rows + 2 per expert
            pp->adds_cap = 3 * (size_t) Rt + 2 * NE + 64;
            pp->adds = (int32_t*) take(pp->adds_cap * 4);
            pp->tcur.assign((size_t) m.T_max, 0);
        }
        pp->sum = (float*) take((size_t) m.T_max * N * 4);
        pp->wk = (float*) take((size_t) m.T_max * K * 4);
        pp->pair = (int32_t*) take((size_t) Rt * 4);
    }
    pp->src = (int32_t*) take((size_t) Rt * 4);
    pp->ident = (int32_t*) take((size_t) Rt * 4);
    pp->bounds = (int32_t*) take((size_t) (2 * (NE + NE / MMQ_GROUP + 2)) * 4);
    pp->grp_gu = (uint8_t*) take(MMQ_GROUP * mp.gu_max + MMQ_TAIL);
    pp->grp_d = (uint8_t*) take(MMQ_GROUP * mp.d_max + MMQ_TAIL);
    if (ok) {
        pp->ctx = std::make_unique<mmq::Context>();
        pp->run_ctx = pp->ctx.get();
        mmq::iota(pp->ident, pp->cap_rows, pp->s);
        peer_had_signs(*pp);   // APR: on the peer's device
        ok = cudaStreamSynchronize(pp->s) == cudaSuccess;
    }
    if (ok && !pp->p2p) {   // the host route's mapped buffers, sized for the largest chunk
        auto host = [&](auto*& ptr, size_t& cap, size_t n) {
            if (cudaHostAlloc((void**) &ptr, n * sizeof(*ptr), cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess) {
                cudaGetLastError();
                ptr = nullptr;
                ok = false;
                return;
            }
            cap = n;
        };
        size_t bounds_cap = 0, w_cap = 0, sum_cap = 0, pair_cap = 0;
        size_t x16_cap = 0, sum16_cap = 0;
        if (pp->f16) host(pp->host_x16, x16_cap, (size_t) m.T_max * N);
        else host(pp->host_x, pp->host_x_cap, (size_t) m.T_max * N);
        if (ok && pp->sums) {
            host(pp->host_w, w_cap, (size_t) m.T_max * K);
            if (ok && pp->f16) host(pp->host_sum16, sum16_cap, (size_t) m.T_max * N);
            else if (ok) host(pp->host_sum, sum_cap, (size_t) m.T_max * N);
            if (ok) host(pp->host_pair, pair_cap, (size_t) pp->cap_rows);
            size_t adds_cap = 0;
            if (ok && pp->gather) host(pp->host_adds, adds_cap, pp->adds_cap);
        } else if (ok) {
            host(pp->host_rows, pp->host_rows_cap, (size_t) pp->cap_rows * N);
        }
        if (ok) host(pp->host_src, pp->host_src_cap, (size_t) pp->cap_rows);
        if (ok) host(pp->host_bounds, bounds_cap, PeerPrefill::kHostBounds);
        if (!ok) {
            cudaSetDevice(prev);
            err = "prefill peer: the host route's pinned buffers (" +
                  std::to_string(((size_t) pp->cap_rows * N * 4 + (size_t) m.T_max * N * 4) >> 20) + " MiB) do not fit in RAM";
            return false;
        }
    }
    size_t fb = 0, tb = 0;
    cudaMemGetInfo(&fb, &tb);
    cudaSetDevice(prev);
    if (!ok) { err = "prefill peer: the peer's buffers do not fit (raise --peer-reserve-mib or lower --peer-prefill-rows)"; return false; }
    std::fprintf(stderr, "strata prefill: peer GPU %d computes its experts' rows of each prompt chunk (up to %lld rows per "
                         "layer%s%s); %zu MiB left free on it\n", pp->dev, (long long) pp->cap_rows,
                 pp->p2p ? "" : (pp->sums ? (pp->f16 ? ", no P2P: through mapped host memory in FP16, one weighted sum per token back"
                                                     : ", no P2P: through mapped host memory, one weighted sum per token back")
                                          : ", no P2P: through mapped host memory"),
                 pp->compact ? (pp->ps_frac > 0.0 ? (", compact group buffers, streams " + std::to_string((int) (pp->ps_frac * 100 + 0.5)) +
                                                    "% of the primary's streamed experts through a " + std::to_string(pp->RP) + "-slot ring").c_str()
                                                 : ", compact group buffers") : "", fb >> 20);
    // APR: this path rotates on the peer too (peer_had_signs), but no P2P pair has run it end to end with a
    // Hadamard-folded pack yet (the validated two-GPU runs had no P2P, so their prompts stayed on the first GPU)
    if (strata::kernels::cpu::hadamard_spec().any())
        std::fprintf(stderr, "strata prefill: warning: prompt processing on a P2P peer GPU is not validated end to end "
                             "with Hadamard-folded experts yet\n");
    m.pp = std::move(pp);
    return true;
}

void Prefill::set_pinned_share(double share) { g_pinned_share = share; }
void Prefill::set_ring_override(int slots) { g_ring_override = slots > 0 ? slots : 0; }
void Prefill::set_ring_budget(int slots, int64_t small_max) {
    g_ring_budget = slots > 0 ? slots : 0;
    g_ring_small_max = small_max > 0 ? small_max : 0;
}
double Prefill::pinned_share() { return g_pinned_share; }
int64_t Prefill::stream_all_min_tokens() { return stream_all_min(); }

// --prefill-pipe (opt-in; STRATA_PREFILL_PIPE): a layer split reads a prompt as a pipeline - stage s reads chunk c while
// stage s+1 reads chunk c-1. A stage-chunk costs its layers x (b + a C): a fixed part per chunk (b: the pass over the
// layer's experts - dequantized, or streamed over PCIe where they are not in VRAM) and a part per token (a). The prompt
// then takes about sum_i (b + a C_i) + (S - 1) max_i (b + a C_i) for chunks C_i over S stages: few large or uneven
// chunks keep the later stages waiting (a 10.7K prompt in a 7680 and a 3011 chunk over four GP100 dies: the first
// stage carried 24 of 36 s), many small ones pay b again and again. In units of a that is k r + n + (S - 1)(r + C) for
// k chunks of C tokens (the last one the rest) and r = b/a, so the rule tries every chunk count from the fewest the
// buffers allow, each chunk evened out on the 256-token grid and at least 512, and keeps the cheapest - when it beats
// today's split (chunks of `cap`, the last the rest) by 3%. r differs by an order of magnitude between rigs: ~700 tokens
// on four GP100 dies (every expert in VRAM, slow arithmetic), ~5000 on two RTX 3090 Ti (fast arithmetic, part of the
// experts streamed per chunk), and a wrong r costs (1024 on two RTX 4090: 22-36% slower). The engine neither guesses r
// nor learns it from the prompts it serves - that made a prompt's chunks, and so its output bits, depend on how busy
// the PC happened to be: setup's --calibrate measures it once (tools/calibrate.py) and writes `--prefill-pipe <r>`.
// `--prefill-pipe 1` needs no r: STRATA_PREFILL_EQUAL's rule - today's chunk count evened out, only when today's last
// chunk has stream_all_min() tokens or more.  A chunk that size streams every expert its stage does not hold, a shorter
// one only the experts it routes to: a tail of a few tokens is nearly free, and evening it out would add a full pass
// (two RTX 4090, a prompt 25 tokens past 8192: 25% slower).  --calibrate tries it like the other values.
namespace {
double g_pipe = -1.0;       // --prefill-pipe; < 0: not given (STRATA_PREFILL_PIPE, else 0)
double g_pipe_req = -1.0;   // the `pipe=` request key; < 0: none
double pipe_value() {
    if (g_pipe_req >= 0.0) return g_pipe_req;
    if (g_pipe >= 0.0) return g_pipe;
    static const double env = [] {
        const char* v = std::getenv("STRATA_PREFILL_PIPE");
        return v != nullptr && v[0] != '\0' ? std::max(0.0, std::atof(v)) : 0.0;
    }();
    return env;
}
}  // namespace
void Prefill::set_pipe(double value) { g_pipe = value; }
void Prefill::set_pipe_request(double value) { g_pipe_req = value; }
double Prefill::pipe_setting() { return pipe_value(); }
namespace {
// b/a in whole tokens: 0 for none (or NaN), at most 2^20 (llround of an infinity is undefined - `pipe=inf` parses)
int64_t ratio_tokens(double ratio) { return ratio > 0.0 ? (int64_t) std::llround(std::min(ratio, 1048576.0)) : 0; }
}  // namespace
int64_t Prefill::pipe_key(double value) { return !(value > 0.0) ? -1 : value <= 1.0 ? 0 : ratio_tokens(value); }
// In whole tokens (r rounded): integer sums and a 97/100 test, so a near-tie cannot fall one way in one build and the
// other way in another (the same chunks as the double version for every whole r: an exact 97% is not better).
int64_t Prefill::pipeline_chunk(int64_t n, int64_t cap, int stages, double ratio) {
    if (stages < 2 || n <= 0 || cap <= 0) return cap;
    const int64_t r = ratio_tokens(ratio), S1 = stages - 1;
    if (r == 0) {
        // `--prefill-pipe 1`: STRATA_PREFILL_EQUAL's rule (#693, generate.cpp equal_chunk) - today's chunk count, evened
        // out on the 256-token grid, only when today's last chunk streams every expert anyway (stream_all_min tokens or
        // more): evening a short tail out would add a full pass over the experts (two RTX 4090: 25% slower)
        const int64_t k = (n + cap - 1) / cap;
        const int64_t last = n - (k - 1) * cap;
        if (k <= 1 || last < stream_all_min()) return cap;
        const int64_t per = (n + k - 1) / k;
        return std::min(cap, ((per + 255) / 256) * 256);
    }
    auto cost = [&](int64_t c) {   // chunks of c tokens, the last the rest
        const int64_t k = (n + c - 1) / c;
        return k * r + n + S1 * (r + std::min(c, n));
    };
    const int64_t today = std::min(n, cap);
    const int64_t today_cost = cost(today);
    const int64_t k_min = (n + cap - 1) / cap;
    const int64_t k_max = ratio > 0.0 ? std::max<int64_t>(k_min, std::min<int64_t>(64, (n + 511) / 512)) : k_min;
    int64_t best = today;
    int64_t best_cost = today_cost;
    for (int64_t k = k_min; k <= k_max; ++k) {
        const int64_t c = std::min(cap, std::max<int64_t>(512, (((n + k - 1) / k + 255) / 256) * 256));
        const int64_t v = cost(c);
        if (v < best_cost) { best_cost = v; best = c; }
    }
    return (best < today && 100 * best_cost < 97 * today_cost) ? best : cap;
}
int Prefill::stage_count() const {
    int stages = 1;
    for (const Prefill* p = next_; p != nullptr; p = p->next_) ++stages;
    return stages;
}
int64_t Prefill::plan_chunk(int64_t n) const {
    const double pipe = pipe_value();
    const int stages = stage_count();
    if (pipe <= 0.0 || hand_in_ != nullptr || stages < 2) return 0;
    return pipeline_chunk(n, impl_->T, stages, pipe > 1.0 ? pipe : 0.0);
}
void Prefill::force_chunk(int64_t chunk, bool single, bool planned) {
    force_chunk_ = chunk;
    force_single_ = single;
    force_planned_ = planned;
}
void Prefill::log_plan(int64_t n, int64_t chunk, int stages) {
    const double pipe = pipe_value();
    std::fprintf(stderr, "strata prefill: %lld tokens in %lld-token chunks over %d stages (--prefill-pipe %s)\n",
                 (long long) n, (long long) chunk, stages,
                 pipe > 1.0 ? (std::to_string((long long) ratio_tokens(pipe)) + ": b/a").c_str()
                            : "1: today's chunk count, evened out");
}
int64_t Prefill::chain_experts_streamed() const {
    int64_t sum = 0;
    for (const Prefill* p = this; p != nullptr; p = p->next_) sum += p->stats_.experts_streamed;
    return sum;
}

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
    // the same allocation sequence as `init` (`carve`), counted - with the MoE layout `carve` picks for the same source
    const size_t T = (size_t) chunk;
    bool ok = true;
    Alloc o;
    o.count_only = true;
    if (owned_pages) o.granule = 2ull << 20;
    o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    o.take<uint8_t>(GEMM_WS, ok);
    auto f = [&](size_t n) { o.take<float>(n, ok); };
    // `carve`'s order, buffer for buffer: emb, R, xn, grs, xn16, lo, lo16, gated, inj, mixed, mixed_bf, mixed_h,
    // bo aliases emb. This counted `xn` unconditionally (carve takes it only under STRATA_GR_UNFUSED) and never counted
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
    f(T * D); f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok);
    if (!emb_reuse_account()) f(T * N);   // bo: aliases emb in carve; still counted unless STRATA_EMB_REUSE_ACCOUNT=1
    const bool f16_io = prompt_f16();   // the current device's mode (the stage's), as Prefill::init will decide it
    if (bf16x2_hc(f16_io)) { o.take<uint16_t>(T * D, ok); o.take<uint16_t>(T * LR, ok); }
    if (bf16x2(f16_io)) o.take<uint16_t>(T * N, ok);
    o.take<int32_t>(T * strata::kernels::kStepCount, ok);
    strata::kernels::QsaShapes s = core::qsa_shapes(g);
    const int64_t cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    const int64_t max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    o.take<uint8_t>((size_t) std::max({gdn_set_bytes(T), qsa_set_bytes(T, cap, max_blocks, 256, 32, s),
                                       moe_set_bytes(T, g.n_expert, fused_layout(T, src))}), ok);
    for (int i = 0; i < DQ; ++i) { o.take<uint16_t>(1280 * 2560, ok); o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        o.take<int32_t>(T * K, ok);
        o.take<int32_t>((size_t) (2 * (g.n_expert + g.n_expert / MMQ_GROUP + 2)), ok);
        o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL, ok);
    }
    if (owned_pages) {
        if (stage_slots(T) > 0) o.take<uint8_t>((size_t) stage_slots(T) * (size_t) MAXBLOB(), ok);   // one allocation
    } else {
        for (int i = 0; i < stage_slots(T); ++i) o.take<uint8_t>((size_t) MAXBLOB(), ok);
    }
    f(T * N);
    f((size_t) strata::kernels::NG_HC_DIM);
    strata::kernels::KvHostPools stage;
    take_stage(o, ss, s, stage, ok);
    return o.used + (8u << 20);   // alignment slack
}

uint64_t Prefill::bytes_needed_no_ring(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                        bool src) {
    return bytes_needed(g, ss, chunk, src) - (uint64_t) stage_slots((size_t) chunk) * (uint64_t) MAXBLOB();
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
void Prefill::set_cpu_pool(kernels::cpu::ExpertPool* pool) { cpu_pool_ = pool; }
void Prefill::arm_cpu_share(bool applies, bool by_default) {
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
    if (applies && by_default && cpu_share_explicit() == -2.0 && !g_share_default) {
        g_share_default = true;
        std::fprintf(stderr, "prefill: the CPU share is ON by default for prompt chunks below 1024 tokens (the idle CPU "
                             "takes some of the experts the GPU would stream; answers can differ slightly from 0.1.40.3, "
                             "mean KL ~0.004). STRATA_PREFILL_CPU_SHARE=0 turns it off.\n");
    }
#else
    (void) by_default;
#endif
    if (applies && cpu_share_on()) g_stream_min_share = cpu_share_max();
}

bool Prefill::ring_bytes_enabled() { return ring_bytes_on(); }
size_t Prefill::slot_guard_bytes() { return mmq::direct_slot_guard(); }

namespace {

const core::WeightRef* need(const core::LayerView& v, const char* suffix, std::string& err) {
    const core::WeightRef* r = v.get(suffix);
    if (!r) err = v.name(suffix) + " is missing";
    return r;
}
// STRATA_COLD_TRACE=1 (diagnostic): the wall time of the first 80 dense projections of the process, each synchronized, to show
// which first calls pay a one-time cost (library / code object loads).
struct ColdTrace {
    bool on; int n = 0; std::chrono::steady_clock::time_point t0;
    ColdTrace() { const char* e = std::getenv("STRATA_COLD_TRACE"); on = e != nullptr && e[0] == '1'; }
    bool begin() {
        if (!on || n >= 80) return false;
        cudaDeviceSynchronize();
        t0 = std::chrono::steady_clock::now();
        return true;
    }
    void end(const char* what, const std::string& name, int64_t T, int64_t N, int64_t K) {
        cudaDeviceSynchronize();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++n;
        std::fprintf(stderr, "strata cold-trace: #%d %s %s T=%lld N=%lld K=%lld %.2f ms\n", n, what, name.c_str(), (long long) T,
                     (long long) N, (long long) K, ms);
    }
};
ColdTrace g_cold;
bool native_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                 std::string& err, int64_t ldy = 0, int64_t ldx = 0) {
    if (!w->native_data) { err = "prefill: " + name + " has no native GGUF blocks (run with --native)"; return false; }
    const bool ct = g_cold.begin();
    gm.native(X, w->native_type, w->native_data, Y, T, w->ne1, w->ne0, ldy, 0.0f, ldx);
    if (ct) g_cold.end("native", name, T, w->ne1, w->ne0);
    return true;
}
bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
               std::string& err, int64_t ldy = 0, const uint16_t* X_lo = nullptr, int64_t ldx = 0,
               void* stream = nullptr, bool x_f16 = false) {
    if (w->kind != core::WeightKind::Bf16InF32 || !w->data) { err = "prefill: " + name + " is not a resident BF16 tensor"; return false; }
    const uint16_t* wb = (const uint16_t*) w->data;
    if (w->q8_scale != nullptr) {
        // STRATA_HC_REQ8: the GEMM takes bf16, so the int8 codes are expanded into one reused scratch on this stream
        static uint16_t* scratch = nullptr;
        static int64_t cap = 0;
        if (w->elements > cap) {
            if (scratch) cudaFree(scratch);
            if (cudaMalloc((void**) &scratch, (size_t) w->elements * 2) != cudaSuccess) {
                scratch = nullptr; cap = 0; err = "prefill: the STRATA_HC_REQ8 scratch"; return false;
            }
            cap = w->elements;
        }
        strata::kernels::hc_q8_to_bf16(w->data, w->q8_scale, scratch, (long long) w->elements, stream);
        wb = scratch;
    }
    const bool ct = g_cold.begin();
    gm.bf16(X, wb, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy, 0.0f, ldx, x_f16);
    if (ct) g_cold.end("bf16", name, T, w->ne1, w->ne0);
    if (X_lo) gm.bf16(X_lo, wb, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy, 1.0f);
    return true;
}

}  // namespace

namespace {
// STRATA_PREFILL_TIMING=1: the prompt path's GPU time by phase.  Events are recorded on the compute stream in order;
// the time between two consecutive marks is charged to the phase of the first, so a gap where the GPU waits (for the
// host's expert grouping, or for an expert's copy) lands on the phase that was waiting.  Events are reused: the marks
// are folded at every MoE layer's host sync, after which all of them have completed.
enum PfPhase { kPfStart, kPfHc, kPfGdn, kPfQsa, kPfQsaIdx, kPfQsaSel, kPfQsaAttn, kPfRouter, kPfHostGroup, kPfGather,
               kPfWaitCopy, kPfDequant, kPfGemmGU, kPfGemmD, kPfCombine, kPfPle, kPfKvStage, kPfGdnConv, kPfGdnRec, kPfGdnOut, kPfHcWrite,
               kPfCount };
const char* const kPfNames[kPfCount] = {"embed+steps", "hc read", "gdn", "qsa proj", "qsa indexer", "qsa select",
                                        "qsa attn", "router+shared", "host grouping", "gather", "wait copy", "dequant",
                                        "gemm gate/up", "gemm down", "combine", "ple", "kv stage", "gdn conv+gates",
                                        "gdn recurrence", "gdn out proj", "hc write"};
struct PfTimer {
    bool on = std::getenv("STRATA_PREFILL_TIMING") != nullptr;
    std::vector<cudaEvent_t> ev;
    std::vector<int> ph;
    size_t used = 0;
    double ms[kPfCount] = {};
    void mark(int phase, cudaStream_t s) {
        if (!on) return;
        if (used == ev.size()) {
            cudaEvent_t e = nullptr;
            cudaEventCreate(&e);
            ev.push_back(e);
            ph.push_back(0);
        }
        ph[used] = phase;
        cudaEventRecord(ev[used], s);
        ++used;
    }
    // every recorded mark has completed (the stream was synchronized): charge the gaps, keep the last mark
    void fold() {
        if (!on || used < 2) return;
        for (size_t i = 0; i + 1 < used; ++i) {
            float t = 0.0f;
            if (cudaEventElapsedTime(&t, ev[i], ev[i + 1]) == cudaSuccess) ms[ph[i]] += t;
        }
        std::swap(ev[0], ev[used - 1]);
        std::swap(ph[0], ph[used - 1]);
        used = 1;
    }
    ~PfTimer() {
        for (cudaEvent_t e : ev) cudaEventDestroy(e);
    }
};
// multi-GPU: the peer's own timeline (STRATA_PREFILL_TIMING): marks on the peer stream, folded with the primary's
enum PePhase { kPeIdle, kPeMoeIn, kPeMoeGemm, kPeMoeOut, kPeCount };
const char* const kPeNames[kPeCount] = {"idle", "moe in", "moe gemm", "moe out"};
struct PeTimer {
    bool on = std::getenv("STRATA_PREFILL_TIMING") != nullptr;
    int dev = -1;
    std::vector<cudaEvent_t> ev;
    std::vector<int> ph;
    size_t used = 0;
    double ms[kPeCount] = {};
    void mark(int phase, cudaStream_t s) {   // the peer device is current
        if (!on) return;
        if (used == ev.size()) {
            cudaEvent_t e = nullptr;
            cudaEventCreate(&e);
            ev.push_back(e);
            ph.push_back(0);
        }
        ph[used] = phase;
        cudaEventRecord(ev[used], s);
        ++used;
    }
    void fold() {   // every mark has completed
        if (!on || used < 2) return;
        for (size_t i = 0; i + 1 < used; ++i) {
            float t = 0.0f;
            if (cudaEventElapsedTime(&t, ev[i], ev[i + 1]) == cudaSuccess) ms[ph[i]] += t;
        }
        std::swap(ev[0], ev[used - 1]);
        std::swap(ph[0], ph[used - 1]);
        used = 1;
    }
    ~PeTimer() {
        if (dev < 0) return;
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(dev);
        for (cudaEvent_t e : ev) cudaEventDestroy(e);
        cudaSetDevice(prev);
    }
};
}  // namespace

bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    err.clear();
    // force_chunk is for this run only: taken before any early return can leave it to the next one
    const int64_t forced = hand_in_ == nullptr ? std::exchange(force_chunk_, 0) : 0;
    const bool forced_single = force_single_, forced_planned = force_planned_;
    Impl& m = *impl_;
    const core::OnDevice on_device(m.device);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    const auto t_start = Clock::now();
    const int64_t LB = stage_lb_, LE = stage_le_;
    // The direct successor's future lives on the Prefill object. Intermediate
    // stages therefore do not drain the complete remaining GPU chain here.
    double host_sync_ms = 0, host_chunk_ms = 0, host_setup_ms = 0;   // STRATA_PREFILL_TIMING: the host's share
    strata::kernels::QsaShapes s = core::qsa_shapes(g);
    // read once (STRATA_KV_PREFETCH=1/0: the next streamed layer's KV prefix is staged on its own stream)
    static const bool kv_prefetch_on = [] {
        const char* e = std::getenv("STRATA_KV_PREFETCH");
        return e ? std::atoi(e) != 0 : false;
    }();
    const bool kv_prefetch = kv_prefetch_on;
    if (kv_prefetch) {
        if ((!m.kv_copy && cudaStreamCreateWithFlags(&m.kv_copy, cudaStreamNonBlocking) != cudaSuccess) ||
            (!m.kv_released && cudaEventCreateWithFlags(&m.kv_released, cudaEventDisableTiming) != cudaSuccess) ||
            (!m.kv_ready && cudaEventCreateWithFlags(&m.kv_ready, cudaEventDisableTiming) != cudaSuccess)) {
            err = "prefill: KV prefetch stream/events";
            return false;
        }
    }
    // An early return must drain DMA before the caller refills the borrowed expert slots.
    struct KvDrain {
        cudaStream_t stream;
        ~KvDrain() { if (stream) cudaStreamSynchronize(stream); }
    } kv_drain{kv_prefetch ? m.kv_copy : nullptr};
    int64_t kv_prefetches = 0;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
    // layer split: a prompt of one chunk runs the stages one after the other, so the next stage's GPU idles while
    // this one reads; it streams and computes a share of this stage's experts then (set_stage_helper), in the
    // place of a --peer-device peer, and gives the stage back its own buffers when the run ends
    // the chunk this prompt is read in: the buffers' size, or with --prefill-pipe on a layer split the one that keeps
    // every stage busy (pipeline_chunk: 1 evens today's chunk count out, a number > 1 is the rig's b/a). Only the first
    // stage chooses: a later one is handed one chunk at a time. A run the engine forced (force_chunk: one part of a
    // segment it planned on its whole length) reads in that chunk and keeps the segment's single/planned flags.
    if (forced > m.T) {
        err = "prefill: a forced chunk of " + std::to_string((long long) forced) + " tokens is larger than the buffers' " +
              std::to_string((long long) m.T);
        return false;
    }
    const int64_t planned_C = forced > 0 ? 0 : plan_chunk(n);
    const int64_t C = forced > 0 ? forced : planned_C > 0 ? planned_C : m.T;
    if (planned_C > 0 && C < std::min(n, m.T)) log_plan(n, C, stage_count());
    const bool single_chunk = hand_in_ == nullptr ? (forced > 0 ? forced_single : n <= C) : single_chunk_;
    // a read the pipe chunked takes no CPU share: its chunks overlap on the stages by design, and which stage won the
    // pool would decide which experts the CPU computed (other bits) - timing, where the chunks are a function of n
    const bool pipe_chunked = hand_in_ == nullptr ? (forced > 0 ? forced_planned : planned_C > 0 && C < std::min(n, m.T))
                                                  : pipe_chunked_;
    const bool helped = single_chunk && bind_stage_helper(n);
    if (helped) std::swap(m.pp, m.help_pp);
    struct HelpScope {
        Impl& m;
        bool on;
        ~HelpScope() {
            if (!on) return;
            const core::OnDevice d(m.pp->dev);   // its work is done before the stage hands over; settle its streams
            cudaStreamSynchronize(m.pp->s_cp);
            cudaStreamSynchronize(m.pp->s_out);
            cudaStreamSynchronize(m.pp->s);
            std::swap(m.pp, m.help_pp);
        }
    } help_scope{m, helped};
    PfTimer pt;
    PeTimer pe;
    if (m.pp) pe.dev = m.pp->dev; else pe.on = false;
    const cudaStream_t cs = (cudaStream_t) m.cs;
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
    auto ple_gather = [&m, &ss, tokens, n, prev0, C](int64_t c0, int buf, std::string& e) -> bool {
        const int64_t T = std::min(C, n - c0);
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

    if (hand_in_ == nullptr) stopped_at_ = -1;
#if defined(STRATA_HAVE_DFLASH2)
    std::vector<int> tap_slot;
    if (!tap_layers_.empty() && stage_lb_ == 0 && (stage_le_ < 0 || stage_le_ == g.n_layers)) {
        if (tap_dev_ == nullptr &&
            cudaMalloc((void**) &tap_dev_, (size_t) tap_cap_ * tap_layers_.size() * (size_t) g.n_embd * 4) != cudaSuccess) {
            (void) cudaGetLastError();
            tap_dev_ = nullptr;
            err = "prefill: the DFlash2 tap staging does not fit in VRAM";
            return false;
        }
        tap_slot.assign((size_t) g.n_layers, -1);
        for (size_t i = 0; i < tap_layers_.size(); ++i) tap_slot[(size_t) tap_layers_[i]] = (int) i;
    }
#endif
    for (int64_t c0 = 0; c0 < n; c0 += C) {
        if (should_stop && should_stop()) { err = "cancelled"; return false; }
        // the first stage: end the run before this chunk (every chunk before it is handed on; the caller's drain
        // waits for the later stages to read them)
        if (c0 > 0 && hand_in_ == nullptr && stop_at_chunk && stop_at_chunk()) {
            stopped_at_ = c0;
            break;
        }
        if (std::getenv("STRATA_TRACE")) { std::fprintf(stderr, "strata trace: prompt chunk %lld of %lld\n", (long long) c0, (long long) n); std::fflush(stderr); }
        const int64_t T = std::min(C, n - c0), p0 = pos0 + c0;
        core::progress_at("reading the prompt (batched): preparing the chunk from token", p0);   // #251
#if defined(STRATA_HAVE_DFLASH2)
        // DFlash2 prompt taps: the rows of this chunk at positions >= tap_first_ (the drafter looks back one window)
        const int64_t tap_t0 = tap_dev_ != nullptr ? std::min<int64_t>(T, std::max<int64_t>(0, tap_first_ - p0)) : T;
        const int64_t tap_rows = T - tap_t0;
        if (tap_rows > tap_cap_) { err = "prefill: the DFlash2 tap staging is smaller than the chunk's tapped rows"; return false; }
#else
        const int64_t tap_t0 = 0, tap_rows = 0;
#endif
        ++stats_.chunks;
        pt.mark(kPfStart, cs);
        const auto tsetup = Clock::now();
        // ---- embeddings, broadcast to the four streams - or, in a later stage of a layer split, the rows the
        // previous stage handed on
        if (hand_in_ != nullptr && hand_ev_ != nullptr) {
            // STRATA_HANDOFF_PIPE: each piece's upload waits on the GPU for its download on the stage before.  The
            // stage before split the chunk it handed on (n rows); one chunk here is all of it, otherwise (a smaller
            // chunk here) every piece is waited for before the one copy.
            const auto* ev = static_cast<const cudaEvent_t*>(hand_ev_);
            const bool whole = c0 == 0 && T == n;
            bool ok = true;
            for (int i = 0; ok && i < hand_pieces_; ++i) {
                int64_t r0 = 0, nr = 0;
                handoff_piece(n, hand_pieces_, i, r0, nr);
                ok = cudaStreamWaitEvent(m.cs, ev[i], 0) == cudaSuccess;
                if (ok && whole && nr > 0)
                    ok = cudaMemcpyAsync(m.R + (size_t) r0 * D, hand_in_ + (size_t) r0 * D, (size_t) nr * D * 4,
                                         cudaMemcpyHostToDevice, m.cs) == cudaSuccess;
            }
            if (ok && !whole)
                ok = cudaMemcpyAsync(m.R, hand_in_ + (size_t) c0 * D, (size_t) T * D * 4, cudaMemcpyHostToDevice,
                                     m.cs) == cudaSuccess;
            if (!ok) {
                err = std::string("prefill: the layer split's hand-off upload failed (STRATA_HANDOFF_PIPE): ") +
                      cudaGetErrorString(cudaGetLastError());
                return false;
            }
        } else if (hand_in_ != nullptr) {
            if (cudaMemcpyAsync(m.R, hand_in_ + (size_t) c0 * D, (size_t) T * D * 4, cudaMemcpyHostToDevice, m.cs) !=
                cudaSuccess) {
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
            if (!m.tok_up.upload(m.tok_dev, m.tok_host.data(), (size_t) T, m.cs)) {
                err = "prefill: the token id upload failed";
                return false;
            }
            if (nemb) {
                nemb->gather_dev(m.tok_dev, T, m.emb, m.cs);
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
        for (int64_t t = 0; hand_in_ == nullptr && !batched && t < T;) {
            const float* row = embd_rows ? embd_rows[p0 + t] : nullptr;
            if (row) {
                // #1996: the picture's rows live in the caller's (pageable) memory.  A run of consecutive rows is
                // gathered into one pinned slot and goes up as one copy; the slot is guarded by its event.
                const int64_t row_bytes = N * 4, run_max = std::max<int64_t>(1, (4 << 20) / row_bytes);
                int64_t t1 = t + 1;
                while (t1 < T && t1 - t < run_max && embd_rows[p0 + t1]) ++t1;
                void* stage = m.up_ring.acquire((size_t) ((t1 - t) * row_bytes));
                cudaError_t e = cudaSuccess;
                if (stage != nullptr) {
                    for (int64_t r = t; r < t1; ++r)
                        std::memcpy((char*) stage + (size_t) ((r - t) * row_bytes), embd_rows[p0 + r], (size_t) row_bytes);
                    e = m.up_ring.submit(m.emb + t * N, stage, (size_t) ((t1 - t) * row_bytes), m.cs);
                } else {   // no pinned RAM: row by row from the caller's memory, each waited for
                    for (int64_t r = t; r < t1 && e == cudaSuccess; ++r)
                        e = core::PinnedRing::direct(m.emb + r * N, embd_rows[p0 + r], (size_t) row_bytes, m.cs);
                }
                if (e != cudaSuccess) {
                    err = "prefill: the image embedding upload failed";
                    return false;
                }
                t = t1;
            } else {
                if (!core::embed_row(*m.wt, g, tokens[c0 + t], m.emb + t * N, m.cs, err)) return false;
                ++t;
            }
        }
        if (hand_in_ == nullptr) gr_broadcast(m.emb, m.R, T, m.cs);
        // ---- the PLE rows of the whole chunk, one batched SSD request on a thread (see ple_gather).  A later chunk's
        // were read ahead during the previous chunk; the first chunk's are read beside layer 0 - they are needed from
        // layer 1 on, and gathering them here first left the GPU idle for the whole read (~0.4 s of a 32K prompt)
        if (ple_on && !ple_next.valid())
            ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c0, b = ple_buf] {
                strata::aux_cpus::pin_current_thread();
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
            // #1996: the pageable fallback (no RAM to pin) goes up through a pinned slot; the pinned buffers as before
            if ((m.ple_pageable[ple_buf].empty()
                     ? cudaMemcpyAsync(m.ple_emb, m.ple_emb_host[ple_buf], (size_t) T * N * 4, cudaMemcpyHostToDevice, m.cs)
                     : m.up_ring.h2d(m.ple_emb, m.ple_emb_host[ple_buf], (size_t) T * N * 4, m.cs)) !=
                    cudaSuccess ||
                cudaEventRecord(m.ple_copied[ple_buf], m.cs) != cudaSuccess) {
                err = std::string("prefill: the PLE rows' upload failed: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            if (c0 + C < n) {
                // the other buffer's upload (a chunk ago) is done before the SSD thread refills it
                if (cudaEventSynchronize(m.ple_copied[ple_buf ^ 1]) != cudaSuccess) {
                    err = std::string("prefill: the PLE rows' upload failed: ") + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c1 = c0 + C, b = ple_buf ^ 1] {
                    strata::aux_cpus::pin_current_thread();
                    return ple_gather(c1, b, ple_next_err);
                });
            }
            ple_buf ^= 1;
            stats_.ms_ple += ms_since(tp);
            return true;
        };
        for (int64_t t = 0; t < T; ++t) { prev[0] = prev[1]; prev[1] = (int32_t) tokens[c0 + t]; }
        // ---- the QSA step records of every position in the chunk
        for (int64_t t = 0; t < T; ++t) strata::kernels::qsa_step_fill(m.steps_host.data() + t * strata::kernels::kStepCount, p0 + t, s);
        if (!m.steps_up.upload(m.steps_dev, m.steps_host.data(), (size_t) T * strata::kernels::kStepCount, m.cs)) {
            err = "prefill: the QSA step records' upload failed";
            return false;
        }

        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < LB; ++l) (core::is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        int64_t kv_pending = -1;
        auto kv_prefetch_after = [&](int64_t layer, int64_t ordinal) -> bool {
            if (!kv_prefetch || p0 <= 0) return true;
            if (kv_pending >= 0) return true;   // one staging pool: the pending prefetch has not been consumed yet
            for (int64_t l = layer; l < LE; ++l) {
                if (!core::is_qsa_layer(g, l)) continue;
                const core::QsaState& state = ss.qsa_states[ordinal];
                if (state.kv_mode != 1) { ++ordinal; continue; }
                // Reuse one KV pool only after its last attention reader completes. The next layer's
                // [0, p0) host prefix is immutable throughout this chunk; its new rows are appended later.
                if (cudaEventRecord(m.kv_released, cs) != cudaSuccess ||
                    cudaStreamWaitEvent(m.kv_copy, m.kv_released, 0) != cudaSuccess) {
                    err = "prefill: releasing KV staging pool";
                    return false;
                }
                strata::kernels::kv_stage_from_host(pools_of(m.stage, m.ident_table), state.host,
                    core::qsa_kv_format(state), (p0 + s.page_size - 1) / s.page_size, s, m.kv_copy);
                if (cudaPeekAtLastError() != cudaSuccess ||
                    cudaEventRecord(m.kv_ready, m.kv_copy) != cudaSuccess) {
                    err = "prefill: enqueueing KV prefetch";
                    return false;
                }
                kv_pending = ordinal;
                ++kv_prefetches;
                break;
            }
            return true;
        };
        if (!kv_prefetch_after(LB, qsa_index)) return false;
        // step 3: this chunk's stream - every non-resident expert of every layer, layer by layer in id order (entry
        // k lands in ring slot k % ring); a copy is issued once the entry `ring` before it is consumed (its slot's
        // `used` event recorded), so the copy stream never waits on an event that is not queued yet
        const strata::kernels::cpu::ExpertLayout& lay0 = strata::kernels::cpu::expert_layout();
        const bool stream_all = m.ring > STAGE && T >= stream_all_min() && m.src != nullptr;
        // the CPU share: this chunk takes the pool if no other stage of a layer split has it now (released at the end
        // of the chunk, after the last layer's CPU thread is joined)
        struct PoolHold {
            bool held = false;
            ~PoolHold() { if (held) g_cpu_pool_busy.store(false, std::memory_order_release); }
        } pool_hold;
        if (cpu_pool_ != nullptr && cpu_share_on() && !stream_all && !pipe_chunked) {
            bool expect = false;
            pool_hold.held = g_cpu_pool_busy.compare_exchange_strong(expect, true, std::memory_order_acq_rel);
        }
        const bool ps_on = stream_all && m.pp && m.pp->ps_frac > 0.0;
        if (m.pp && !ps_on) m.pp->ps_flag.clear();
        // multi-GPU: the peer's ring - issue its copies up to `limit` / give entries back (the peer device is current)
        auto p_issue_until = [&](size_t limit) {
            PeerPrefill& P = *m.pp;
            limit = std::min(limit, P.pseq.size());
            while (P.p_issued < limit) {
                const PeerPrefill::PsEntry& en = P.pseq[P.p_issued];
                const size_t sl = P.p_issued % (size_t) P.RP;
                if (P.plive[sl]) cudaStreamWaitEvent(P.s_cp, P.pused[sl], 0);
                cudaMemcpyAsync(P.pstage[sl], en.blob, (size_t) strata::kernels::cpu::expert_layout().blob_bytes(en.l),
                                cudaMemcpyHostToDevice, P.s_cp);
                cudaEventRecord(P.pcopied[sl], P.s_cp);
                P.plive[sl] = 1;
                ++P.p_issued;
            }
        };
        auto p_release_to = [&](int64_t l, int32_t e_stop, cudaStream_t ps) {
            PeerPrefill& P = *m.pp;
            while (P.pk < P.pseq_start[(size_t) l + 1] && P.pseq[P.pk].e < e_stop) {
                cudaEventRecord(P.pused[P.pk % (size_t) P.RP], ps);
                ++P.pk;
                p_issue_until(P.pk + (size_t) P.RP);
            }
        };
        struct StreamEntry { int32_t l, e; const uint8_t* blob; int job; };
        std::vector<StreamEntry> seq;
        std::vector<size_t> seq_start;
        size_t issued = 0, consumed = 0;
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
                    if (m.src->transient(l, e)) {   // CS-T: copied by the source into the stager's buffer
                        job = (int) js.size();
                        js.push_back({nullptr, (size_t) lay0.blob_bytes(l), m.src, (int32_t) l, e});
                    } else {
                        b = m.src->blob_stable(l, e);
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
                        if (!m.src->pinned(l, e)) {
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
            limit = std::min(limit, seq.size());
            while (issued < limit) {
                const StreamEntry& en = seq[issued];
                const int sl = (int) (issued % (size_t) m.ring);
                const auto th = Clock::now();
                const size_t bytes = (size_t) lay0.blob_bytes(en.l);
                if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0);
                if (en.job < 0) {
                    cudaMemcpyAsync(m.stage_dev[sl], en.blob, bytes, cudaMemcpyHostToDevice, m.copy);
                    ++stats_.experts_dma;
                } else {
                    const uint8_t* hb = m.stager->wait(en.job);
                    cudaMemcpyAsync(m.stage_dev[sl], hb, bytes, cudaMemcpyHostToDevice, m.copy);
                    m.stager->issued_one(en.job, m.copy);
                }
                cudaEventRecord(m.copied[sl], m.copy);
                m.stage_live[sl] = true;
                stats_.ms_experts_host += ms_since(th);
                ++stats_.experts_streamed;
                ++issued;
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
                strata::platform::name_this_thread("strata-issue");
                strata::aux_cpus::pin_current_thread();
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
                    if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0);
                    if (en.job < 0) {
                        cudaMemcpyAsync(m.stage_dev[sl], en.blob, bytes, cudaMemcpyHostToDevice, m.copy);
                        ++iss_dma;
                    } else {
                        const uint8_t* hb = m.stager->wait(en.job);
                        cudaMemcpyAsync(m.stage_dev[sl], hb, bytes, cudaMemcpyHostToDevice, m.copy);
                        m.stager->issued_one(en.job, m.copy);
                    }
                    cudaEventRecord(m.copied[sl], m.copy);
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
        if (ps_on) {
            int pd = 0;
            cudaGetDevice(&pd);
            cudaSetDevice(m.pp->dev);
            p_issue_until((size_t) m.pp->RP);
            cudaSetDevice(pd);
        }
        host_setup_ms += ms_since(tsetup);
        bool normed = false;   // F-2: the previous half's write already normed R for this half (grs, xn16)
        // #579 #613 (opt-in diagnosis, STRATA_PF_STEP_SYNC=1): the compute and copy streams are waited for after each
        // step named below, a step that took over 250 ms is logged, and a stall's report names the step it is in.
        // Slower (a sync per step); the bytes are the same.
        static const bool step_sync = [] { const char* e = std::getenv("STRATA_PF_STEP_SYNC"); return e && e[0] == '1'; }();
        auto pf_step = [&](const char* what, int64_t layer) {
            if (!step_sync) return;
            core::progress_at(what, layer, p0);
            const auto ts = Clock::now();
            const cudaError_t a = cudaStreamSynchronize(m.cs), b = cudaStreamSynchronize(m.copy);
            const double ms = ms_since(ts);
            if (ms > 250.0 || a != cudaSuccess || b != cudaSuccess)
                std::fprintf(stderr, "strata pf-step: chunk from token %lld, layer %lld: %s took %.0f ms (%s / %s)\n",
                             (long long) p0, (long long) layer, what, ms, cudaGetErrorString(a), cudaGetErrorString(b));
        };
        for (int64_t l = LB; l < LE; ++l) {
            core::progress_beat();   // the serve watchdog: a prompt chunk of 8192 tokens is still moving
            if (l > LB) pf_step("reading the prompt (batched, step sync): the experts and the rest of layer", l - 1);
            core::progress_at("reading the prompt (batched): layer", l, p0);   // #251: a stall names layer and chunk
            const core::LayerView v(*m.wt, l);
            if (l == std::max<int64_t>(LB, 1)) {   // the PLE rows, read from layer 1 on (a wait for the SSD read of the first chunk lands on "ple")
                pt.mark(kPfPle, cs);
                if (!ple_land()) return false;
            }
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
                // STRATA_HC_Q8 without a bf16 copy: this path's GEMMs read bf16, so the matrices are dequantized per half
                core::WeightRef wd_q8, wu_q8;
                if (!wd->data && wd->hc_q8 && !wu->data && wu->hc_q8) {
                    wd_q8 = *wd; wu_q8 = *wu;
                    wd_q8.data = core::NativeDense::hc_bf16_view(*wd, m.cs, 0, 0);
                    wu_q8.data = core::NativeDense::hc_bf16_view(*wu, m.cs, 1, 0);
                    if (!wd_q8.data || !wu_q8.data) { err = "prefill: the Q8_0 hyper-connection projections could not be dequantized"; return false; }
                    wd = &wd_q8; wu = &wu_q8;
                }
                pt.mark(kPfHc, cs);
                // the xn16 token stride of this chunk (the writers of both halves and of the fused write-backs use it)
                const int64_t ldx = hc_pad() && T >= std::max<int64_t>(pf_switch_min_t(), 64) && !gr_unfused() &&
                                            !m.xn16_lo ? D + XN_PAD : D;
                const bool xf16 = m.xn_f16 && !m.xn16_lo && ldx == D;   // xn16 holds FP16 (its writers and readers below)
                if (gr_unfused()) gr_norm(m.R, (const float*) wn->data, EPS, m.xn, m.xn16, T, m.cs, m.xn16_lo, xf16);
                else if (!normed) gr_norm_rs(m.R, (const float*) wn->data, EPS, m.grs, m.xn16, T, m.cs, m.xn16_lo, ldx, xf16);
                normed = false;
                bool hcd = false;
                bool hdown = false;
                if (wd->q8_scale == nullptr && ldx != D && !pf_hcdown()) {   // not with STRATA_HC_REQ8's int8 rows   // STRATA_HCD_EXACT: the exact-order down kernel when hipBLASLt would take 1176 / 1177
                    hdown = wd->kind == core::WeightKind::Bf16InF32 && wd->data && wd->ne0 == D && wd->ne1 == LR &&
                            m.gemm.bf16_hcd_exact(m.xn16, ldx, (const uint16_t*) wd->data, m.lo, T, LR, D);
                } else if (wd->q8_scale == nullptr && ldx != D) {
                    if (wd->kind != core::WeightKind::Bf16InF32 || !wd->data || wd->ne0 != D || wd->ne1 != LR ||
                        wi->kind != core::WeightKind::Bf16InF32 || !wi->data || wi->ne0 != D || wi->ne1 != HC ||
                        !strata_pf_hcdown_bf16(m.xn16, ldx, (const uint16_t*) wd->data, (const uint16_t*) wi->data, LR,
                                               HC, m.lo, m.inj, T, D, m.cs)) {
                        err = "prefill: STRATA_PF_HCDOWN could not run the hyper-connection projections";
                        return false;
                    }
                    hcd = true;
                }
                if (!hcd && !hdown && !bf16_proj(m.gemm, wd, m.xn16, m.lo, T, sd, err, 0, m.xn16_lo, ldx != D ? ldx : 0, m.cs, xf16)) return false;
                gr_silu(m.lo, m.lo16, T, m.cs, m.lo16_lo);
                bool upmixed = false;
                if (wu->q8_scale == nullptr && hc_upmix() && T >= std::max(pf_switch_min_t(), hc_upmix_min_t()) && !gr_unfused() && !m.lo16_lo && !m.mixed_bf_lo &&
                    wu->kind == core::WeightKind::Bf16InF32 && wu->data && wu->ne0 == LR && wu->ne1 == D) {
                    static int checks = [] { const char* e = std::getenv("STRATA_HC_UPMIX_CHECK"); return e ? std::atoi(e) : 0; }();
                    if (checks > 0) {   // the default pair first, kept for the comparison
                        --checks;
                        if (!bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err)) return false;
                        gr_mix_r(m.R, m.grs, (const float*) wn->data, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h);
                        std::vector<float> ref((size_t) T * N), got((size_t) T * N);
                        cudaMemcpyAsync(ref.data(), m.mixed, ref.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        upmixed = gr_upmix(m.lo16, (const uint16_t*) wu->data, m.R, m.grs, (const float*) wn->data,
                                           m.mixed, m.mixed_bf, m.mixed_h, T, m.cs);
                        cudaMemcpyAsync(got.data(), m.mixed, got.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaStreamSynchronize(m.cs);
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
                if (!upmixed && !bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err, 0, m.lo16_lo, 0, m.cs)) return false;
                if (!hcd && !bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err, 0, m.xn16_lo, ldx != D ? ldx : 0, m.cs, xf16)) return false;
                if (upmixed) {
                } else if (gr_unfused()) {
                    gr_mix(m.xn, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h, m.mixed_bf_lo);
                } else {
                    gr_mix_r(m.R, m.grs, (const float*) wn->data, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h,
                             m.mixed_bf_lo);
                }

                if (wd_q8.data) {   // the HC GEMMs that read the scratch are enqueued: the next user waits for them
                    core::NativeDense::hc_bf16_release(m.cs, 0, 0);
                    core::NativeDense::hc_bf16_release(m.cs, 1, 0);
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
                            if (cudaStreamWaitEvent(cs, m.kv_ready, 0) != cudaSuccess) {
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
                        // the scores on tensor cores (3xTF32: FP32-level, not bitwise), below sm_80 the FP32 tiled kernel with STRATA_SELECT_SIMT=1;
                        // STRATA_SELECT_OLD=1: the warp kernel
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
                        cudaMemcpyAsync(ids.data(), m.sel_ids, ids.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaStreamSynchronize(m.cs);
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
                            (cudaMalloc((void**) &pooled16, (size_t) rows * s.idx_dim * 4) != cudaSuccess ||
                             cudaMalloc((void**) &dead16, (size_t) s.idx_dim * 4) != cudaSuccess ||
                             cudaMalloc((void**) &ids16, (size_t) (m.T * m.cap) * 4) != cudaSuccess)) {
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
                        cudaMemcpyAsync(a.data(), m.sel_ids, a.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaMemcpyAsync(b.data(), ids16, b.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaStreamSynchronize(m.cs);
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
                    // for tools/qsa_locality.py (how local the sparse attention's reads are: the KV-streaming question).
                    // STRATA_QSA_DUMP_LAYER=i: QSA layer i only; STRATA_QSA_DUMP_STEPS=1: each record then ends with the
                    // T * kStepCount steps too (tests/cuda/pa_bench.cpp replays such a record)
                    if (static const char* dump = std::getenv("STRATA_QSA_DUMP"); dump != nullptr) {
                        static const long long last = std::getenv("STRATA_QSA_DUMP_LAST")
                                                          ? std::atoll(std::getenv("STRATA_QSA_DUMP_LAST")) : 4096;
                        static const int only = std::getenv("STRATA_QSA_DUMP_LAYER") ? std::atoi(std::getenv("STRATA_QSA_DUMP_LAYER")) : -1;
                        if (p0 + T > pos0 + n - last && (only < 0 || only == (int) qsa_index)) {
                            std::vector<int32_t> h((size_t) (T * m.cap));
                            cudaMemcpyAsync(h.data(), m.sel_ids, h.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                            cudaStreamSynchronize(m.cs);
                            if (std::FILE* f = std::fopen(dump, "ab")) {
                                const int32_t hdr[4] = {(int32_t) qsa_index, (int32_t) p0, (int32_t) T, (int32_t) m.cap};
                                std::fwrite(hdr, 4, 4, f);
                                std::fwrite(h.data(), 4, h.size(), f);
                                if (std::getenv("STRATA_QSA_DUMP_STEPS")) {
                                    std::vector<int32_t> sv((size_t) (T * strata::kernels::kStepCount));
                                    cudaMemcpy(sv.data(), m.steps_dev, sv.size() * 4, cudaMemcpyDeviceToHost);
                                    std::fwrite(sv.data(), 4, sv.size(), f);
                                }
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
                    // RESEARCH HOOK (141-research, opt-in, never on by default): STRATA_DBG_FEAT=<file> dumps, per MoE layer and chunk,
                    // the routed ids + weights (and for the layers in STRATA_DBG_FEAT_LAYERS the router-input hidden state, bf16);
                    // STRATA_DBG_ROUTE_OVERRIDE=<file> replaces the routing of the layers listed in that file with the file's ids + weights.
                    {
                        static const char* feat_path = std::getenv("STRATA_DBG_FEAT");
                        static const char* ovr_path = std::getenv("STRATA_DBG_ROUTE_OVERRIDE");
                        if (feat_path != nullptr || ovr_path != nullptr) {
                            static std::FILE* ff = feat_path ? std::fopen(feat_path, "wb") : nullptr;
                            static std::vector<bool> featl = [] {
                                std::vector<bool> r(64, false);
                                const char* e = std::getenv("STRATA_DBG_FEAT_LAYERS");
                                std::string s = e ? e : "3,4,15,16,27,28,39,40";
                                size_t i = 0;
                                while (i < s.size()) { size_t j = s.find(',', i); if (j == std::string::npos) j = s.size(); int v = std::atoi(s.substr(i, j - i).c_str()); if (v >= 0 && v < 64) r[(size_t) v] = true; i = j + 1; }
                                return r;
                            }();
                            static int64_t posl[64] = {};
                            struct Ovr { std::vector<int32_t> ids; std::vector<float> w; };
                            static std::map<std::pair<int32_t, int32_t>, Ovr> ovr = [] {
                                std::map<std::pair<int32_t, int32_t>, Ovr> mp;
                                const char* p = std::getenv("STRATA_DBG_ROUTE_OVERRIDE");
                                std::FILE* f = p ? std::fopen(p, "rb") : nullptr;
                                if (f == nullptr) return mp;
                                int32_t h[3];
                                while (std::fread(h, 4, 3, f) == 3) {
                                    Ovr o; o.ids.resize((size_t) h[2] * 10); o.w.resize((size_t) h[2] * 10);
                                    if (std::fread(o.ids.data(), 4, o.ids.size(), f) != o.ids.size()) break;
                                    if (std::fread(o.w.data(), 4, o.w.size(), f) != o.w.size()) break;
                                    mp[{h[0], h[1]}] = std::move(o);
                                }
                                std::fclose(f);
                                std::fprintf(stderr, "route override: %zu records\n", mp.size());
                                return mp;
                            }();
                            const int32_t pos = (int32_t) posl[l & 63];
                            posl[l & 63] += T;
                            cudaStreamSynchronize((cudaStream_t) m.cs);
                            if (ff != nullptr) {
                                std::vector<int32_t> hid((size_t) T * K); std::vector<float> hw((size_t) T * K);
                                cudaMemcpy(hid.data(), m.ids, hid.size() * 4, cudaMemcpyDeviceToHost);
                                cudaMemcpy(hw.data(), m.w, hw.size() * 4, cudaMemcpyDeviceToHost);
                                std::vector<uint16_t> ids16(hid.size()), w16(hw.size());
                                for (size_t i = 0; i < hid.size(); ++i) { ids16[i] = (uint16_t) hid[i]; w16[i] = (uint16_t) (((*(const uint32_t*) &hw[i]) + 0x8000u) >> 16); }
                                const int32_t hd[4] = {(int32_t) l, pos, (int32_t) T, featl[(size_t) (l & 63)] ? 1 : 0};
                                std::fwrite(hd, 4, 4, ff);
                                std::fwrite(ids16.data(), 2, ids16.size(), ff);
                                std::fwrite(w16.data(), 2, w16.size(), ff);   // bf16 weights
                                if (hd[3]) {
                                    std::vector<uint16_t> hx((size_t) T * N);
                                    cudaMemcpy(hx.data(), m.mixed_bf, hx.size() * 2, cudaMemcpyDeviceToHost);
                                    std::fwrite(hx.data(), 2, hx.size(), ff);
                                }
                            }
                            auto it = ovr.find({(int32_t) l, pos});
                            if (it != ovr.end() && (int64_t) it->second.ids.size() == T * K) {
                                cudaMemcpy(m.ids, it->second.ids.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice);
                                cudaMemcpy(m.w, it->second.w.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice);
                            }
                        }
                    }
                    // the shared expert and its scalar gate
                    auto shared_expert = [&]() -> bool {
                        if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err)) return false;
                        if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err)) return false;
                        swiglu_pair(m.sgate, m.sup, m.sh_h, T, m.cs);
                        if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err)) return false;
                        if (wgi->kind != core::WeightKind::Bf16InF32) { err = "prefill: shared gate is not BF16"; return false; }
                        m.gemm.bf16(m.mixed_bf, (const uint16_t*) wgi->data, m.sg, T, 1, N);
                        if (m.mixed_bf_lo) m.gemm.bf16(m.mixed_bf_lo, (const uint16_t*) wgi->data, m.sg, T, 1, N, 0, 1.0f);
                        return true;
                    };
                    // Only the router needs to finish before readback. Queue the shared expert afterwards
                    // so it can overlap CPU grouping and the routed-only path's initial expert uploads.
                    const bool defer_shared = !stream_all && stream_ahead_enabled();
                    if (!defer_shared && !shared_expert()) return false;
                    // #136: STRATA_PF_FUSED=1 - the Q2_0 pack's experts on the fused int8 kernels (moe_fused.hpp),
                    // grouped on the GPU: no host sync.  Only where every expert's place is known before the routing -
                    // the streamed walk, in which every non-resident expert of the layer comes through the ring in id
                    // order - and where the MMQ buffers exist: they hold the fused path's own (the per-token int8
                    // activations in Xq, the int8 H in H, the grouping tables in GU).  Chunks below stream_all_min()
                    // keep MMQ; without the variable nothing here runs.  A native pack's layer takes the native kernels
                    // (moe_fused_iq.hpp) where they cover its two formats, else MMQ (or the FP16 path: IQ1_M).
                    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
                    // APR (prism.hadamard): folded gate/up take H (s * x), folded down H (s * h), on the FP16 and MMQ
                    // paths and on the multi-GPU prompt paths (the peer, the layer split's helper: MMQ); the fused
                    // kernels do not rotate (and have no trellis types), so a folded layer takes MMQ instead
                    const bool had_gu = lay.native && lay.fmt[(size_t) l].had_gu;
                    const bool had_d = lay.native && lay.fmt[(size_t) l].had_d;
                    const bool use_mmq = mmq_plan().any && mmq_plan().layer[(size_t) l];
                    const int had_block = lay.native ? lay.fmt[(size_t) l].had_block : 0;
                    const float* had_sx = had_gu ? m.had_sx[(size_t) l] : nullptr;   // resolved in init
                    const float* had_sh = had_d ? m.had_sh[(size_t) l] : nullptr;
                    const int mmq_gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42;
                    const int mmq_dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
                    // --peer-device: MMQ only, whether or not the peer took the prompt path (set_peer can decline), as
                    // fused_ring() sized the ring and the buffers for it
                    const bool no_peer = !core::peer_portable();
                    const bool fused_only = mmq_plan().any && mmq_plan().fo[(size_t) l];
                    const bool fused_nat = (use_mmq || fused_only) && stream_all && no_peer && lay.native && !had_gu && !had_d &&
                                           fused::native_supported(mmq_gt, mmq_dt);
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
                    // set_cpu_pool: this layer's CPU experts (rows [T * K - rows_cpu, T * K) of Dm), computed on a thread
                    // that reads on_cpu: declared first, so the future (which waits for the thread) goes first
                    std::vector<char> on_cpu;
                    std::vector<const uint8_t*> cpu_blob;   // the CPU experts' blobs, taken on this thread (the source is not thread-safe)
                    double cpu_ms = 0;                    // the thread's time (the measured share)
                    std::future<bool> cpu_fut;
                    int64_t rows_cpu = 0, n_cpu = 0, n_stream = 0;
                    bool cpu_arm = true;                  // auto: this layer may share (false: the arm without)
                    int cpu_set = -1;                     // auto: the wall events' set, -1 when not measured
                    bool cpu_cold = false;                // auto: one-time costs in the window (no reading)
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
                                cudaStreamWaitEvent(m.cs, m.copied[(k - 1) % (size_t) m.ring], 0);
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
                                cudaEventRecord(m.used[kk % (size_t) m.ring], m.cs);
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
                        else if (m.up_ring.d2h_sync(m.ids_host.data(), m.ids, (size_t) T * K * 4, m.cs) != cudaSuccess) {
                            err = "prefill: the routed ids' download failed";   // #1996
                            return false;
                        }
                        bool cpu_maybe = pool_hold.held && !stream_all && m.src != nullptr && !m.cpu_dead &&
                                         lay.native && !m.pp && !lay.fmt.empty();
                        // The share's two pinned buffers are small (~50 MB) but are had mid-request, when the host may have
                        // nothing left (0.1.41 report: tight RAM, chats failing).  A buffer that cannot be had ends the share
                        // for this engine - the GPU takes every expert, as with STRATA_PREFILL_CPU_SHARE=0 - instead of
                        // failing the request.
                        auto cpu_share_off = [&](const char* what) {
                            m.cpu_dead = true;
                            cpu_maybe = false;
                            std::fprintf(stderr, "prefill: the CPU share is off for this run: the host could not give %s "
                                                 "(RAM is short); the GPU streams every expert\n", what);
                        };
                        // auto shares only while the layers that share cost less per non-resident expert than their
                        // neighbours that do not.  #1282 (RX 7900 GRE + 5700X3D): every share lost, auto's 0.47 by
                        // 4.5%, the host's issue time per streamed expert rising 107 -> 200 us with the CPU busy -
                        // a cost g / (c + g) cannot see, as both sides are measured with the share on.  The first
                        // eligible layers alternate, without first, until CPU_MIN_RATIOS ratios are in; then the
                        // cheaper arm, and every CPU_PROBE-th layer the other one (a prime: the probes move across
                        // the layers from one chunk to the next), so both stay measured.
                        if (cpu_maybe && cpu_share_env() < 0.0) {
                            constexpr int64_t CPU_PROBE = 29;
                            constexpr int CPU_MIN_RATIOS = 3;
                            const int64_t i = m.cpu_layers++;
                            cpu_arm = m.cpu_nratio < CPU_MIN_RATIOS ? (i & 1) != 0 : (i % CPU_PROBE == 0) != m.cpu_gate;
                            if (m.cpu_first_l < 0) m.cpu_first_l = l;
                            cpu_cold = l == m.cpu_first_l;
                            if (m.cpu_wall[0] == nullptr)
                                for (cudaEvent_t& e : m.cpu_wall) cudaEventCreate(&e);
                            cpu_set = (int) (i & 1);
                            cudaEventRecord(m.cpu_wall[2 * cpu_set], m.cs);
                        }
                        if (cpu_maybe && cpu_arm) {
                            const size_t want = (size_t) T * N;
                            if (m.cpu_x_n < want) {
                                cpu_cold = true;
                                if (m.cpu_x) cudaFreeHost(m.cpu_x);
                                m.cpu_x = nullptr;
                                m.cpu_x_n = 0;
                                if (cudaHostAlloc((void**) &m.cpu_x, want * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
                                    (void) cudaGetLastError();
                                    m.cpu_x = nullptr;
                                    cpu_share_off("the activation buffer");
                                } else {
                                    m.cpu_x_n = want;
                                }
                            }
                            if (cpu_maybe) cudaMemcpyAsync(m.cpu_x, m.mixed, want * sizeof(float), cudaMemcpyDeviceToHost, m.cs);
                        }
                        // #579: a stall here is the GPU (this layer's attention and router, or the previous layer's
                        // work), not the host: the watchdog's report says so (only its text changes)
                        core::progress_at("reading the prompt (batched): waiting for the GPU (attention, router) at layer",
                                          l, p0);
                        cudaStreamSynchronize(m.cs);
                        core::progress_at("reading the prompt (batched): layer", l, p0);
                        if (on_routes) on_routes(l, ids_h, T * K);   // 149-prefetch: the RAM policy's prompt seed
                        if (cpu_maybe) {   // the rows buffer, once, at its largest (an expert takes at most MAXT tokens' rows)
                            const size_t cap = (size_t) std::min<int64_t>(T * K, (int64_t) strata::kernels::cpu::MAXT * m.g->n_expert) * N;
                            if (m.cpu_rows_n < cap) {
                                if (m.cpu_rows) cudaFreeHost(m.cpu_rows);
                                m.cpu_rows = nullptr;
                                m.cpu_rows_n = 0;
                                if (cudaHostAlloc((void**) &m.cpu_rows, cap * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
                                    (void) cudaGetLastError();
                                    m.cpu_rows = nullptr;
                                    cpu_share_off("the rows buffer");
                                } else {
                                    m.cpu_rows_n = cap;
                                }
                            }
                        }
                        if (m.cpu_pend) {   // the measured share: the last CPU-sharing layer's GPU time is final now
                            m.cpu_pend = false;
                            float g_ms = 0;
                            if (cudaEventElapsedTime(&g_ms, m.cpu_ev[0], m.cpu_ev[1]) == cudaSuccess) {
                                constexpr double a = 0.25;
                                const double c = m.pend_cpu_ms / (double) m.pend_n_cpu, g = g_ms / (double) m.pend_n_gpu;
                                m.cpu_c_ms = m.cpu_c_ms > 0 ? m.cpu_c_ms + a * (c - m.cpu_c_ms) : c;
                                m.cpu_g_ms = m.cpu_g_ms > 0 ? m.cpu_g_ms + a * (g - m.cpu_g_ms) : g;
                                m.cpu_share_now = std::clamp(m.cpu_g_ms / (m.cpu_c_ms + m.cpu_g_ms), 0.05, 0.9);
                            }
                        }
                        if (m.pend_wall) {   // the last eligible layer's wall, and its ratio to the one before it
                            float w_ms = 0;
                            if (cudaEventElapsedTime(&w_ms, m.cpu_wall[2 * m.pend_set], m.cpu_wall[2 * m.pend_set + 1]) == cudaSuccess) {
                                const double w = w_ms / (double) m.pend_wall_n;
                                if (m.last_arm != 0 && m.last_arm != m.pend_wall && m.pend_l > m.last_l) {   // same chunk
                                    std::rotate(m.cpu_ratio, m.cpu_ratio + 1, m.cpu_ratio + Impl::CPU_RATIOS);
                                    m.cpu_ratio[Impl::CPU_RATIOS - 1] = m.pend_wall == 2 ? w / m.last_w : m.last_w / w;
                                    m.cpu_nratio = std::min(m.cpu_nratio + 1, Impl::CPU_RATIOS);
                                    double v[Impl::CPU_RATIOS];
                                    std::copy(m.cpu_ratio + Impl::CPU_RATIOS - m.cpu_nratio, m.cpu_ratio + Impl::CPU_RATIOS, v);
                                    std::nth_element(v, v + m.cpu_nratio / 2, v + m.cpu_nratio);   // even: the upper one
                                    m.cpu_gate = v[m.cpu_nratio / 2] < 1.0;
                                }
                                // debug: STRATA_DBG_CPU_GATE=1 prints each reading and what the next layers do
                                if (static const bool dbg = std::getenv("STRATA_DBG_CPU_GATE") != nullptr; dbg)
                                    std::fprintf(stderr, "cpu gate: layer %d %s, %lld non-resident, %.2f ms (%.4f a expert) -> %s\n",
                                                 m.pend_l, m.pend_wall == 2 ? "shared" : "not shared", (long long) m.pend_wall_n,
                                                 w_ms, w, m.cpu_gate ? "share" : "do not share");
                                m.last_arm = m.pend_wall;
                                m.last_l = m.pend_l;
                                m.last_w = w;
                            }
                            m.pend_wall = 0;
                        }
                        pt.fold();
                        if (pe.on) {   // the peer's marks so far are done: the primary waited for its last rows
                            int pd = 0; cudaGetDevice(&pd); cudaSetDevice(pe.dev); cudaStreamSynchronize(m.pp->s); pe.fold(); cudaSetDevice(pd);
                        }
                        if (defer_shared) {
                            pt.mark(kPfRouter, cs);   // keep the existing router+shared timing attribution
                            if (!shared_expert()) return false;
                            pt.mark(kPfHostGroup, cs);
                        }
                        std::fill(m.cnt.begin(), m.cnt.end(), 0);
                        for (int64_t i = 0; i < T * K; ++i) {
                            const int32_t e = ids_h[(size_t) i];
                            if (e < 0 || e >= m.g->n_expert) { err = "prefill: routed id out of range"; return false; }
                            ++m.cnt[(size_t) e];
                        }
                        if (cpu_maybe) {
                            std::vector<std::pair<int32_t, int32_t>> cand;   // (tokens, expert)
                            int64_t nstream = 0;
                            for (int32_t e = 0; e < m.g->n_expert; ++e) {
                                const int32_t c = m.cnt[(size_t) e];
                                if (c == 0 || (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0))
                                    continue;
                                ++nstream;
                                // a blob the CPU reads from RAM: page-locked, or any other that is not assembled into a
                                // short-lived buffer (the arena's unpinned part, the mapped experts.bin's pages)
                                if (c <= strata::kernels::cpu::MAXT && (m.src->pinned(l, e) || !m.src->transient(l, e)))
                                    cand.emplace_back(c, e);
                            }
                            std::sort(cand.begin(), cand.end());
                            const double share = cpu_share_env() >= 0.0 ? cpu_share_env() : cpu_arm ? m.cpu_share_now : 0.0;
                            const size_t take = std::min(cand.size(), (size_t) std::llround(share * (double) nstream));
                            n_stream = nstream;
                            // The CPU thread touches no ExpertSource (its blob() counts reads without a lock, and its
                            // calls race with this thread's own blob_stable()/pinned() ones): the pointers are taken
                            // here, and an expert without a stable blob stays on the GPU.
                            for (size_t i = 0; i < cand.size() && n_cpu < (int64_t) take; ++i) {
                                const int32_t e = cand[i].second;
                                const uint8_t* b = m.src->blob_stable(l, e);
                                if (b == nullptr) continue;
                                if (on_cpu.empty()) {
                                    on_cpu.assign((size_t) m.g->n_expert, 0);
                                    cpu_blob.assign((size_t) m.g->n_expert, nullptr);
                                }
                                on_cpu[(size_t) e] = 1;
                                cpu_blob[(size_t) e] = b;
                                rows_cpu += cand[i].first;
                                ++n_cpu;
                            }
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
                                if ((on_peer.empty() || !on_peer[(size_t) e]) && (on_cpu.empty() || !on_cpu[(size_t) e])) {
                                    m.off[(size_t) e] = r;
                                    r += m.cnt[(size_t) e];
                                }
                            // the peer's rows in order_peer's order (peer-held, then peer-streamed): its groups need them contiguous
                            for (int kind = 1; kind <= 2 && !on_peer.empty(); ++kind)
                                for (int32_t e = 0; e < m.g->n_expert; ++e)
                                    if (on_peer[(size_t) e] == kind) { m.off[(size_t) e] = r; r += m.cnt[(size_t) e]; }
                            for (int32_t e = 0; e < m.g->n_expert && !on_cpu.empty(); ++e)   // the CPU's block, last
                                if (on_cpu[(size_t) e]) { m.off[(size_t) e] = r; r += m.cnt[(size_t) e]; }
                            m.off[(size_t) m.g->n_expert] = r;
                            rows_local -= rows_cpu;
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
                            if (m.cnt[(size_t) e] > 0 && (on_peer.empty() || on_peer[(size_t) e] != 2) &&
                                (on_cpu.empty() || !on_cpu[(size_t) e]))
                                (!on_peer.empty() && on_peer[(size_t) e] ? order_peer : order).push_back(e);
                        for (int32_t e = 0; e < m.g->n_expert && !on_peer.empty(); ++e)   // the peer-streamed ones last:
                            if (on_peer[(size_t) e] == 2) order_peer.push_back(e);        // their copies get the most time
                        // Aurora (STRATA_MMQ_RESIDENT_SORT_NE=1, opt-in): a layer whose experts are ALL resident and
                        // run through MMQ groups takes them in row-count order, so each 16-expert group's max_rows
                        // (its padded tile rows) is close to its experts' own; slot/src/off are rebuilt coherently
                        // before the upload, the routed ids, weights and the within-expert row order stay as they are.
                        // STRATA_MMQ_GROUP_SORT=1 (opt-in, exact): the same order for the full groups only, where each
                        // tile is one block's whole K (MmqPlan::group_sort); the partial group keeps its ID order and bits.
                        // STRATA_MMQ_RESIDENT_SORT_NE=1 takes precedence (it sorts the partial group too).
                        const bool sort_all = detail::mmq_resident_sort_requested();
                        const bool sort_full = !sort_all && use_mmq && mmq_plan().group_sort[(size_t) l] &&
                                               order.size() >= (size_t) MMQ_GROUP;
                        if (detail::mmq_resident_sort_eligible((sort_all || sort_full) && on_peer.empty(), use_mmq, lay.native,
                                m.cache != nullptr, m.host_res ? m.host_res + (size_t) l * m.g->n_expert : nullptr,
                                m.g->n_expert, stream_all, !stream_all || seq_start[(size_t) l] == seq_start[(size_t) l + 1]))
                            detail::mmq_resident_sort_rows(ids_h, T * K, (int32_t) K, m.cnt, m.off, order, slot_h, src_h,
                                                           detail::mmq_resident_sort_span(order.size(), MMQ_GROUP, sort_full));
                        if (grp_mapped) {
                            copy_i32(m.slot_dev, m.grp_dev + m.grp_tk, T * K, m.cs);
                            copy_i32(m.src_dev, m.grp_dev + 2 * m.grp_tk, T * K, m.cs);
                        } else {
                            if (m.up_ring.h2d(m.slot_dev, m.slot_host.data(), (size_t) T * K * 4, m.cs) != cudaSuccess ||   // #1996
                                m.up_ring.h2d(m.src_dev, m.src_host.data(), (size_t) T * K * 4, m.cs) != cudaSuccess) {
                                err = "prefill: the routing tables' upload failed";
                                return false;
                            }
                        }
                        // Diagnostics: STRATA_DUMP_MOE_INPUT=path, STRATA_DUMP_MOE_LAYER=l - the first chunk's MoE input
                        // of layer l ({T, N, K} int64, T x N floats, T x K expert ids) for testing products on real rows
                        static const char* dump_moe = std::getenv("STRATA_DUMP_MOE_INPUT");
                        static bool dumped_moe = false;
                        if (dump_moe && !dumped_moe && l == std::atoi(std::getenv("STRATA_DUMP_MOE_LAYER") ? std::getenv("STRATA_DUMP_MOE_LAYER") : "0")) {
                            dumped_moe = true;
                            std::vector<float> x((size_t) (T * N));
                            cudaMemcpyAsync(x.data(), m.mixed, x.size() * sizeof(float), cudaMemcpyDeviceToHost, m.cs);
                            cudaStreamSynchronize(m.cs);
                            if (std::FILE* f = std::fopen(dump_moe, "wb")) {
                                const int64_t hdr[3] = {(int64_t) T, (int64_t) N, (int64_t) K};
                                std::fwrite(hdr, sizeof hdr, 1, f);
                                std::fwrite(x.data(), sizeof(float), x.size(), f);
                                std::fwrite(ids_h, sizeof(int32_t), (size_t) (T * K), f);
                                std::fclose(f);
                            }
                        }
                        n_order = order.size();
                        // set_cpu_pool: the CPU's experts on a thread while this one streams and runs the GPU's; their
                        // unweighted rows land in Dm's row order, so the combine weights them as any other row
                        if (rows_cpu > 0) {
                            const size_t want = (size_t) rows_cpu * N;
                            if (m.cpu_rows_n < want) {
                                cpu_cold = true;
                                if (m.cpu_rows) cudaFreeHost(m.cpu_rows);
                                m.cpu_rows = nullptr;
                                m.cpu_rows_n = 0;
                                const size_t cap_rows = (size_t) rows_cpu * 2;   // headroom: few reallocations
                                if (cudaHostAlloc((void**) &m.cpu_rows, cap_rows * N * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
                                    err = "prefill: cannot allocate the CPU experts' rows";
                                    return false;
                                }
                                m.cpu_rows_n = cap_rows * N;
                            }
                            // this fork (NVFP4): row_sd 1 for the CPU's rows, which carry s_down already - sized on its
                            // own, as the rows buffer is now taken once at its largest above (0.1.42)
                            if (m.cpu_ones_n < (size_t) rows_cpu) {
                                if (m.cpu_ones) cudaFreeHost(m.cpu_ones);
                                m.cpu_ones = nullptr;
                                m.cpu_ones_n = 0;
                                const size_t n1 = std::max<size_t>((size_t) rows_cpu * 2, m.cpu_rows_n / N);
                                if (cudaHostAlloc((void**) &m.cpu_ones, n1 * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
                                    err = "prefill: cannot allocate the CPU experts' row scales";
                                    return false;
                                }
                                std::fill(m.cpu_ones, m.cpu_ones + n1, 1.0f);
                                m.cpu_ones_n = n1;
                            }
                            const int64_t r0c = T * K - rows_cpu;
                            const strata::kernels::cpu::NativeFmt& cf = lay.fmt[(size_t) l];
                            if (cpu_share_env() < 0.0) {
                                if (m.cpu_ev[0] == nullptr) { cudaEventCreate(&m.cpu_ev[0]); cudaEventCreate(&m.cpu_ev[1]); }
                                cudaEventRecord(m.cpu_ev[0], m.cs);
                            }
                            cpu_fut = std::async(std::launch::async, [&m, &on_cpu, &cpu_blob, &cf, &cpu_ms, src_h, l, T, r0c, this]() -> bool {
                                const auto t0 = std::chrono::steady_clock::now();
                                constexpr size_t AB = strata::kernels::cpu::kNativeActBytes;
                                // a Q2_0 layer (gate/up on the pool's Q2_0 kernels): ActQ activations, as decode's
                                const bool q2 = strata::kernels::cpu::q2_native_kernels(cf.gu_type);
                                if (q2 && m.cpu_actq.size() < (size_t) T) m.cpu_actq.resize((size_t) T);
                                if (!q2 && m.cpu_nact.size() < (size_t) T * AB) m.cpu_nact.resize((size_t) T * AB);
                                std::vector<char> need((size_t) T, 0);
                                for (int32_t e = 0; e < m.g->n_expert; ++e)   // the tokens the CPU's experts read
                                    if (on_cpu[(size_t) e])
                                        for (int32_t r = 0; r < m.cnt[(size_t) e]; ++r)
                                            need[(size_t) src_h[(size_t) m.off[(size_t) e] + (size_t) r]] = 1;
                                for (int64_t t = 0; t < T; ++t)
                                    if (need[(size_t) t] && q2)
                                        strata::kernels::cpu::act_quant_any(m.cpu_x + (size_t) t * N, (int) N, m.cpu_actq[(size_t) t]);
                                    else if (need[(size_t) t])
                                        strata::kernels::cpu::native_quant_act(cf, m.cpu_x + (size_t) t * N,
                                                                               m.cpu_nact.data() + (size_t) t * AB);
                                m.cpu_jobs.clear();
                                for (int32_t e = 0; e < m.g->n_expert; ++e) {
                                    if (!on_cpu[(size_t) e]) continue;
                                    strata::kernels::cpu::ExpertJobMulti j;
                                    j.blob = cpu_blob[(size_t) e];
                                    j.nt = m.cnt[(size_t) e];
                                    for (int r = 0; r < j.nt; ++r) {
                                        const int64_t p = (int64_t) m.off[(size_t) e] + r;
                                        if (q2) j.act[r] = &m.cpu_actq[(size_t) src_h[(size_t) p]];
                                        else j.nact[r] = m.cpu_nact.data() + (size_t) src_h[(size_t) p] * AB;
                                        j.out[r] = m.cpu_rows + (size_t) (p - r0c) * N;
                                    }
                                    m.cpu_jobs.push_back(j);
                                }
                                cpu_pool_->run_split_multi_native(cf, m.cpu_jobs.data(), (int) m.cpu_jobs.size());
                                cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                                return true;
                            });
                        }
                        // pf_wmma: this layer's groups on the FP16 tensor-core tiles (single GPU, native pack, Q2_0 / IQ4_NL down)
                        const bool wmma_l = use_mmq && pf_wmma() && !fused_l && on_peer.empty() && lay.native &&
                                            (mmq_dt == 42 || mmq_dt == 20) && !lay.fmt[(size_t) l].tail_off &&
                                            strata::kernels::iq_row_bytes(mmq_gt, N) > 0;
                        if (use_mmq && mmq_plan().wmma_only[(size_t) l] && !wmma_l) {
                            err = "prefill: an IQ1_M layer in the MMQ grouping needs the WMMA tiles, which this chunk cannot take";
                            return false;
                        }
                        const size_t mmq_gub = use_mmq ? mmq::matrix_bytes(mmq_gt, 1280, N) : 0;
                        const size_t mmq_db = use_mmq ? mmq::matrix_bytes(mmq_dt, N, 640) : 0;
                        pt.mark(kPfGather, cs);
                        // the routed experts' input: `mixed`, or (APR) its rotated copy, which the peer / the layer
                        // split's helper also take (one rotation for both cards; `mixed` itself feeds the shared expert)
                        const float* xsrc = m.mixed;
                        if (use_mmq) {
                            // step 2b: the layer's activations as q8_1 rows in expert order, straight from `mixed` (APR:
                            // from its rotated copy `Xr`): each token quantized once and written to its K rows
                            // (slot_dev, the inverse of src_dev).
                            // NVFP4 w4a4 activations carry a per-row scale (Xscale): that mode keeps the gather.
                            if (wmma_l) {   // FP16 rows (pf_wmma); APR: Hadamard-folded gate/up take H, rotated in fp16 storage as the FP16 path does
                                gather_rows16(m.mixed_h, m.src_dev, m.Xs, T * K, N, m.cs);
                                if (had_gu) strata::kernels::hadamard_rows_f16(m.Xs, T * K, N, had_block, had_sx, m.cs);
                            } else {
                                if (had_gu) {
                                    strata::kernels::hadamard_rows(m.mixed, m.Xr, T, N, had_block, had_sx, m.cs);
                                    xsrc = m.Xr;
                                }
                                if (mmq::fp4_activations(mmq_gt))
                                    mmq::quantize(xsrc, m.src_dev, m.Xq, mmq_gt, N, N, T * K, m.cs, m.Xscale);
                                else
                                    mmq::quantize_scatter(xsrc, m.slot_dev, m.src_dev, m.Xq, mmq_gt, N, N, T, (int) K, m.cs);
                            }
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
                                if (m.up_ring.h2d(m.bounds_dev, m.bounds_host.data(), m.bounds_host.size() * 4, m.cs) != cudaSuccess) {   // #1996
                                    err = "prefill: the group bounds' upload failed";
                                    return false;
                                }
                            }
                        } else {
                            gather_rows16(m.mixed_h, m.src_dev, m.Xs, T * K, N, m.cs);
                            // APR: Hadamard-folded gate/up take H (s * x); Xs is this layer's private gathered copy
                            // (rotated in fp16 storage, fp32 arithmetic: see the down projection's note below)
                            if (had_gu) strata::kernels::hadamard_rows_f16(m.Xs, T * K, N, had_block, had_sx, m.cs);
                        }
                        // multi-GPU: the peer's share, enqueued before the primary's own experts so both cards work at once
                        peer_now = use_mmq && !order_peer.empty();
                        if (peer_now) {
                            PeerPrefill& P = *m.pp;
                            // the host route's copies beside the compute stream: the primary's own experts start at once
                            // (mixed_h and w are rewritten only after this layer's combine, which waits for the peer)
                            cudaStream_t xs = m.cs;
                            if (P.x_stream != nullptr) {
                                cudaEventRecord(P.ev_x, m.cs);
                                cudaStreamWaitEvent(P.x_stream, P.ev_x, 0);
                                xs = P.x_stream;
                            }
                            if (!P.p2p && P.f16) cudaMemcpyAsync(P.host_x16, m.mixed_h, (size_t) T * N * 2, cudaMemcpyDeviceToHost, xs);
                            else if (!P.p2p) cudaMemcpyAsync(P.host_x, m.mixed, (size_t) T * N * 4, cudaMemcpyDeviceToHost, xs);
                            if (P.sums) {   // the routing weights, and each peer row's routed pair (t * K + k)
                                cudaMemcpyAsync(P.host_w, m.w, (size_t) T * K * 4, cudaMemcpyDeviceToHost, xs);
                                for (int64_t i = 0; i < T * K; ++i)
                                    if (const int32_t r = slot_h[(size_t) i]; r >= rows_local) P.host_pair[r - rows_local] = (int32_t) i;
                            }
                            cudaEventRecord(P.ev_in, xs);
                            int prevd = 0;
                            cudaGetDevice(&prevd);
                            cudaSetDevice(P.dev);
                            const cudaStream_t ps = P.s;
                            cudaStreamWaitEvent(ps, P.ev_in, 0);
                            if (P.out_pending) { cudaStreamWaitEvent(ps, P.ev_done, 0); P.out_pending = false; }
                            pe.mark(kPeMoeIn, ps);
                            if (P.p2p) cudaMemcpyPeerAsync(P.mixed, P.dev, m.mixed, prevd, (size_t) T * N * 4, ps);
                            else if (P.f16) f16_to_f32_wide(P.mixed, P.host_x16, T * N, ps);
                            else copy_f32_wide(P.mixed, P.host_x, T * N, ps);
                            P.back_at = rows_local;
                            P.back_rows = rows_peer;
                            if (P.p2p) {
                                // multi-GPU: the mapped table when on; src_host (a pageable vector) goes up from a pinned slot (#1996)
                                if ((grp_mapped ? cudaMemcpyAsync(P.src, src_h + rows_local, (size_t) rows_peer * 4,
                                                                  cudaMemcpyHostToDevice, ps)
                                                : P.up_ring.h2d(P.src, src_h + rows_local, (size_t) rows_peer * 4, ps)) !=
                                    cudaSuccess) {
                                    cudaSetDevice(prevd);
                                    err = "prefill: the peer's routing table upload failed";
                                    return false;
                                }
                            } else {
                                std::memcpy(P.host_src, src_h + rows_local, (size_t) rows_peer * 4);
                                copy_i32(P.src, P.host_src, rows_peer, ps);
                            }
                            const size_t n = order_peer.size();
                            if (P.sums) {
                                copy_f32_wide(P.wk, P.host_w, T * K, ps);
                                copy_i32(P.pair, P.host_pair, rows_peer, ps);
                                cudaMemsetAsync(P.sum, 0, (size_t) T * N * 4, ps);
                            }
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
                                    if (P.up_ring.h2d(P.bounds, P.bounds_host.data(), P.bounds_host.size() * 4, ps) != cudaSuccess) {   // #1996
                                        cudaSetDevice(prevd);
                                        err = "prefill: the peer's group bounds upload failed";
                                        return false;
                                    }
                                } else {
                                    std::memcpy(P.host_bounds, P.bounds_host.data(), P.bounds_host.size() * 4);
                                    copy_i32(P.bounds, P.host_bounds, (int64_t) P.bounds_host.size(), ps);
                                }
                                if (P.sums && P.gather) {   // each group's tokens, their row lists in the experts' order
                                    P.adds_at.resize(ng);
                                    P.ntok.resize(ng);
                                    size_t at = 0;
                                    for (size_t g2 = 0; g2 < ng; ++g2) {
                                        const int32_t r0 = P.bounds_host[P.groups[g2].first], r1 = P.bounds_host[P.groups[g2].second];
                                        P.touched.clear();
                                        for (int32_t r = r0; r < r1; ++r) {
                                            const int32_t t = P.host_pair[r] / (int32_t) K;
                                            if (P.tcur[(size_t) t]++ == 0) P.touched.push_back(t);
                                        }
                                        const size_t nt = P.touched.size();
                                        int32_t* a = P.host_adds + at;   // [tokens nt | starts nt + 1 | rows]
                                        a[nt] = 0;
                                        for (size_t i = 0; i < nt; ++i) {
                                            const int32_t t = P.touched[i];
                                            a[i] = t;
                                            a[nt + 1 + i] = a[nt + i] + P.tcur[(size_t) t];
                                            P.tcur[(size_t) t] = a[nt + i];   // the fill cursor
                                        }
                                        int32_t* list = a + 2 * nt + 1;
                                        for (int32_t r = r0; r < r1; ++r)
                                            list[P.tcur[(size_t) (P.host_pair[r] / (int32_t) K)]++] = r;
                                        for (int32_t t : P.touched) P.tcur[(size_t) t] = 0;
                                        P.adds_at[g2] = at;
                                        P.ntok[g2] = (int32_t) nt;
                                        at += 2 * nt + 1 + (size_t) (r1 - r0);
                                    }
                                    if (at > P.adds_cap) {
                                        err = "prefill: the peer's group tables overflow";
                                        cudaSetDevice(prevd);
                                        return false;
                                    }
                                    copy_i32(P.adds, P.host_adds, (int64_t) at, ps);
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
                                                cudaSetDevice(prevd);
                                                return false;
                                            }
                                            psl = (int) (P.pk % (size_t) P.RP);
                                            cudaStreamWaitEvent(ps, P.pcopied[(size_t) psl], 0);
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
                                            cudaEventRecord(P.pused[(size_t) psl], ps);
                                            ++P.pk;
                                            p_issue_until(P.pk + (size_t) P.RP);
                                        }
                                    }
                                    const int64_t r0 = P.bounds_host[j0], nr = P.bounds_host[j1] - r0;
                                    if (nr <= 0) continue;
                                    const int32_t* rel = P.bounds + n + 1 + g2 * (MMQ_GROUP + 1);
                                    cudaMemsetAsync(P.grp_gu + (size_t) ngx * mmq_gub, 0, MMQ_TAIL, ps);
                                    cudaMemsetAsync(P.grp_d + (size_t) ngx * mmq_db, 0, MMQ_TAIL, ps);
                                    mmq::quantize(P.mixed, P.src + r0, P.Xq_g, mmq_gt, N, N, nr, ps);
                                    mmq::Product gu;
                                    gu.w = P.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
                                    gu.n = ngx; gu.xq = P.Xq_g; gu.bounds = rel; gu.ids = P.ident;
                                    gu.total_rows = nr; gu.max_rows = maxr; gu.dst = P.GU_g; gu.ld_dst = 1280;
                                    P.run_ctx->run(gu, ps);
                                    mmq::swiglu(P.GU_g, P.H_g, nr, 640, !lay.native, ps);
                                    if (had_d)   // APR: folded down weights take H (s * h), on this card as on the primary
                                        strata::kernels::hadamard_rows(P.H_g, P.H_g, nr, 640, had_block, P.had_sh[(size_t) l], ps);
                                    mmq::quantize(P.H_g, nullptr, P.Hq_g, mmq_dt, 640, 640, nr, ps);
                                    const int b = (int) (g2 & 1);
                                    if (P.dm_live[b]) cudaStreamWaitEvent(ps, P.ev_dm[b], 0);   // its last rows have left
                                    mmq::Product dn;
                                    dn.w = P.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
                                    dn.n = ngx; dn.xq = P.Hq_g; dn.bounds = rel;
                                    dn.ids = P.ident; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = P.Dm_b[b];
                                    dn.ld_dst = N;
                                    P.run_ctx->run(dn, ps);
                                    if (P.sums && P.gather) {   // into the per-token sums, one launch for the group
                                        const int32_t* a = P.adds + P.adds_at[g2];
                                        const int64_t nt = P.ntok[g2];
                                        peer_gather_add(P.sum, P.Dm_b[b], r0, P.wk, P.pair, a, a + nt, a + 2 * nt + 1, nt, ps);
                                        continue;
                                    }
                                    if (P.sums) {   // into the per-token sums on this stream: Dm_b[b] is free after it
                                        for (size_t j = j0; j < j1; ++j)
                                            peer_scatter_add(P.sum, P.Dm_b[b] + (size_t) (P.bounds_host[j] - r0) * N, P.wk,
                                                             P.pair + P.bounds_host[j], P.bounds_host[j + 1] - P.bounds_host[j], ps);
                                        continue;
                                    }
                                    cudaEventRecord(P.ev_grp[g2 % PeerPrefill::kGrpEv], ps);
                                    cudaStreamWaitEvent(P.s_out, P.ev_grp[g2 % PeerPrefill::kGrpEv], 0);
                                    if (P.p2p)
                                        cudaMemcpyPeerAsync(m.Dm + (size_t) (rows_local + r0) * N, prevd, P.Dm_b[b], P.dev,
                                                            (size_t) nr * N * 4, P.s_out);
                                    else   // the primary reads them when it combines (below)
                                        cudaMemcpyAsync(P.host_rows + (size_t) r0 * N, P.Dm_b[b], (size_t) nr * N * 4,
                                                        cudaMemcpyDeviceToHost, P.s_out);
                                    cudaEventRecord(P.ev_dm[b], P.s_out);
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
                                    if (P.up_ring.h2d(P.bounds, P.bounds_host.data(), P.bounds_host.size() * 4, ps) != cudaSuccess) {   // #1996
                                        cudaSetDevice(prevd);
                                        err = "prefill: the peer's group bounds upload failed";
                                        return false;
                                    }
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
                                    cudaMemsetAsync(P.grp_gu + (size_t) ngx * mmq_gub, 0, MMQ_TAIL, ps);
                                    cudaMemsetAsync(P.grp_d + (size_t) ngx * mmq_db, 0, MMQ_TAIL, ps);
                                    mmq::Product gu;
                                    gu.w = P.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
                                    gu.n = ngx; gu.xq = P.Xq; gu.bounds = P.bounds + j0; gu.ids = P.ident;
                                    gu.total_rows = rows_peer; gu.max_rows = maxr; gu.dst = P.GU; gu.ld_dst = 1280;
                                    P.run_ctx->run(gu, ps);
                                    mmq::swiglu(P.GU + r0 * 1280, P.H + r0 * 640, nr, 640, !lay.native, ps);
                                    if (had_d)   // APR: as above
                                        strata::kernels::hadamard_rows(P.H + r0 * 640, P.H + r0 * 640, nr, 640, had_block,
                                                                       P.had_sh[(size_t) l], ps);
                                    mmq::quantize(P.H + r0 * 640, nullptr, P.Hq, mmq_dt, 640, 640, nr, ps);
                                    mmq::Product dn;
                                    dn.w = P.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
                                    dn.n = ngx; dn.xq = P.Hq; dn.bounds = P.bounds + n + 1 + g2 * (MMQ_GROUP + 1);
                                    dn.ids = P.ident; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = P.Dm + r0 * N;
                                    dn.ld_dst = N;
                                    P.run_ctx->run(dn, ps);
                                    if (P.out_pipe && nr > 0) {   // this group's rows go back while the next group computes
                                        cudaEventRecord(P.ev_grp[g2], ps);
                                        cudaStreamWaitEvent(P.s_out, P.ev_grp[g2], 0);
                                        cudaMemcpyPeerAsync(m.Dm + (size_t) (rows_local + r0) * N, prevd, P.Dm + (size_t) r0 * N,
                                                            P.dev, (size_t) nr * N * 4, P.s_out);
                                    }
                                }
                            }
                            pe.mark(kPeMoeOut, ps);
                            if (P.sums) {   // the sums go back: T x N
                                if (P.f16) {
                                    sums_to_f16(P.sum, P.sum16, T * N, ps);
                                    cudaMemcpyAsync(P.host_sum16, P.sum16, (size_t) T * N * 2, cudaMemcpyDeviceToHost, ps);
                                } else {
                                    cudaMemcpyAsync(P.host_sum, P.sum, (size_t) T * N * 4, cudaMemcpyDeviceToHost, ps);
                                }
                                cudaEventRecord(P.ev_done, ps);
                                P.out_pending = true;
                            } else if (P.out_pipe) {
                                cudaEventRecord(P.ev_done, P.s_out);
                                P.out_pending = true;
                            } else {
                                cudaMemcpyPeerAsync(m.Dm + (size_t) rows_local * N, prevd, P.Dm, P.dev, (size_t) rows_peer * N * 4, ps);
                                cudaEventRecord(P.ev_done, ps);
                            }
                            pe.mark(kPeIdle, ps);
                            cudaSetDevice(prevd);
                            ++P.layers;
                            P.experts += (int64_t) n;
                            P.rows += rows_peer;
                        }
                        if (ps_on) {   // multi-GPU: this layer's peer-ring entries the routing did not pick give their slots back
                            int pd = 0;
                            cudaGetDevice(&pd);
                            cudaSetDevice(m.pp->dev);
                            p_release_to(l, (int32_t) m.g->n_expert, m.pp->s);
                            cudaSetDevice(pd);
                        }
                        // STRATA_PREFILL_GROUP_GATHER=0: one gather, one wait and one record per expert.
                        static const bool group_env = [] {
                            const char* v = std::getenv("STRATA_PREFILL_GROUP_GATHER");
                            return v == nullptr || std::atoi(v) != 0;
                        }();
                        const bool group_gather = group_env && stream_all && use_mmq && lay.native &&
                                                  MMQ_GROUP <= mmq::kGatherGroupMax;
                        // The staged walk (a short prompt) groups too, resident and staged experts alike: its ring is
                        // then STAGE_GRP slots, two groups, and it stages no further ahead than the open group allows,
                        // so a group's staged slots stay unread-over until its flush (one wait on the last copy, one
                        // event releasing them all, as in the streamed walk).
                        const bool group_resident = group_env && !stream_all && use_mmq && lay.native &&
                                                    MMQ_GROUP <= mmq::kGatherGroupMax;
                        const bool gg_now = group_gather || group_resident;
                        const int stage_n = group_resident ? STAGE_GRP : STAGE;
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
                                if (m.src->pinned(l, e)) continue;
                                job_of[j] = (int) js.size();
                                if (m.src->transient(l, e)) {   // CS-T: copied by the source
                                    js.push_back({nullptr, (size_t) lay.blob_bytes(l), m.src, (int32_t) l, e});
                                    continue;
                                }
                                const uint8_t* b = m.src->blob_stable(l, e);
                                if (!b) { err = "prefill: expert source has no blob"; return false; }
                                js.push_back({b, (size_t) lay.blob_bytes(l)});
                            }
                            m.stager->start(std::move(js));
                        }
                        StagerDone stager_done{stream_all ? nullptr : m.stager.get()};
                        auto stage_one = [&](size_t j) -> bool {
                            const int32_t e = order[j];
                            const bool resident = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                            if (resident) return true;
                            const int sl = stage_next;
                            stage_next = (stage_next + 1) % stage_n;
                            const auto th = Clock::now();
                            const bool pinned = m.src->pinned(l, e);   // pinned: never transient
                            const uint8_t* b = pinned ? m.src->blob_stable(l, e) : nullptr;
                            if (pinned && !b) { err = "prefill: expert source has no blob"; return false; }
                            if (pinned) {
                                // DMA straight from the page-locked arena: the copy stream only waits for the slot
                                if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0);
                                cudaMemcpyAsync(m.stage_dev[sl], b, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
                                ++stats_.experts_dma;
                            } else {
                                // copied to a pinned buffer by the stager (waits only if it is behind), then DMA
                                const uint8_t* hb = m.stager->wait(job_of[j]);
                                if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0);
                                cudaMemcpyAsync(m.stage_dev[sl], hb, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
                                m.stager->issued_one(job_of[j], m.copy);
                            }
                            cudaEventRecord(m.copied[sl], m.copy);
                            m.stage_live[sl] = true;
                            stage_of[j] = sl;
                            stats_.ms_experts_host += ms_since(th);
                            ++stats_.experts_streamed;
                            return true;
                        };
                        // In the streamed walk an MMQ group is gathered in ONE launch, after ONE wait on its last
                        // streamed copy (the copy stream is in order), and its ring slots are released by ONE event.  A
                        // wait and a record per expert put ~10 us of GPU bubble each on the compute stream under WDDM
                        // (226 ms of a 32K prompt on an NVFP4 pack).  The same bytes into the same group slots.  A ring
                        // entry the routing skipped inside an open group first gathers what the group holds so far
                        // (`flush`), so no more than a group's entries are ever held back from the issuer.
                        // STRATA_PREFILL_GROUP_GATHER=0: one gather, one wait and one record per expert.
                        mmq::GatherGroup gg;
                        int gg_slots[MMQ_GROUP];
                        int gg_nslots = 0;   // ring slots gathered by the next flush
                        // STRATA_MMQ_DIRECT=1: a resident expert with guard room in its slot is not gathered - the
                        // group's pointer tables give MMQ each expert's own address (group position q), the others are
                        // gathered at compact positions c = 0..ns-1 of the group buffers, as before (same bytes)
                        const bool direct = use_mmq && lay.native && m.cache && mmq_plan().direct[(size_t) l] && !wmma_l;
                        mmq::WeightTab tgu, td;
                        mmq::GuardTab gd;
                        int ns = 0;
                        // `zero`: the MMQ tail zeroed after the gathered experts (the group's last flush); a direct
                        // group's tail is zeroed by prep_group
                        auto flush = [&](size_t zero) {
                            if (gg.n <= gg.first) return;
                            const auto& f = lay.fmt[(size_t) l];
                            if (gg_nslots > 0) {   // the copies land in order: the last one covers the others
                                pt.mark(kPfWaitCopy, cs);
                                cudaStreamWaitEvent(m.cs, m.copied[gg_slots[gg_nslots - 1]], 0);
                                pt.mark(kPfDequant, cs);
                            }
                            // NVFP4: each expert's scales go next to its group slot in the same launch
                            if (!mmq::gather_native_group(gg, f.up_off, mmq_gub / 2, f.down_off, mmq_db, m.grp_gu, mmq_gub,
                                                          m.grp_d, mmq_db, m.cs, f.tail_off,
                                                          f.tail_off ? m.grp_tail : nullptr, zero)) {
                                for (int i = gg.first; i < gg.n; ++i) {   // not 16-byte aligned: one at a time
                                    const uint8_t* b = gg.blob[i];
                                    mmq::gather_native(b, b + f.up_off, mmq_gub / 2, b + f.down_off, mmq_db,
                                                       m.grp_gu + i * mmq_gub, m.grp_d + i * mmq_db, m.cs,
                                                       f.tail_off ? b + f.tail_off : nullptr,
                                                       f.tail_off ? m.grp_tail + i * 4 : nullptr, i + 1 == gg.n ? zero : 0);
                                }
                            }
                            if (gg_nslots > 0) {
                                const int rel = gg_slots[gg_nslots - 1];
                                cudaEventRecord(m.used[rel], m.cs);
                                for (int i = 0; i < gg_nslots; ++i) m.used_of[gg_slots[i]] = rel;
                            }
                            gg_nslots = 0;
                            gg.first = gg.n;
                        };
                        // one expert's products from its blob on the device; `slot` (a ring slot, or -1 for a resident
                        // expert) is released once the blob is read; `cslot`: the expert-cache slot the blob is in (-1:
                        // a ring slot or the peer's), which STRATA_MMQ_DIRECT reads in place
                        auto compute = [&](size_t j, const uint8_t* blob_dev, int slot, int32_t cslot) -> bool {
                            const int32_t e = order[j];
                            pt.mark(kPfDequant, cs);
                            if (use_mmq) {
                                // gather the expert into its group slot (GGUF blocks, unchanged or converted)
                                const size_t q = j % MMQ_GROUP;
                                // the group's last expert also zeroes the MMQ tail after its slot (see MMQ_TAIL)
                                const bool last_of_group = q + 1 == MMQ_GROUP || j + 1 == order.size();
                                if (direct) {
                                    const auto& f = lay.fmt[(size_t) l];
                                    uint8_t* gp = cslot >= 0 && ((uintptr_t) blob_dev | (uintptr_t) f.down_off) % 16 == 0
                                                      ? m.cache->slot_tail(cslot, f.bytes, mmq_plan().d_over[(size_t) l])
                                                      : nullptr;
                                    if (gp) {   // read in its slot: the guard after the blob is zeroed at the group end
                                        tgu.p[q] = blob_dev;
                                        td.p[q] = blob_dev + f.down_off;
                                        gd.p[gd.n++] = gp;
                                        ++stats_.experts_direct;
                                    } else {
                                        const int c = ns++;
                                        tgu.p[q] = m.grp_gu + (size_t) c * mmq_gub;
                                        td.p[q] = m.grp_d + (size_t) c * mmq_db;
                                        if (gg_now) {
                                            gg.blob[c] = blob_dev;
                                            gg.n = c + 1;
                                            if (slot >= 0) gg_slots[gg_nslots++] = slot;
                                        } else {
                                            mmq::gather_native(blob_dev, blob_dev + f.up_off, mmq_gub / 2, blob_dev + f.down_off,
                                                               mmq_db, m.grp_gu + (size_t) c * mmq_gub,
                                                               m.grp_d + (size_t) c * mmq_db, m.cs);
                                        }
                                    }
                                    if (slot >= 0 && !gg_now) { cudaEventRecord(m.used[slot], m.cs); m.used_of[slot] = slot; }
                                    if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
                                    if (gg_now) {   // also when the last entry was direct (a no-op without gathers)
                                        flush(0);
                                        gg = mmq::GatherGroup{};
                                    }
                                } else {
                                if (gg_now) {
                                    gg.blob[q] = blob_dev;
                                    gg.n = (int) q + 1;
                                    if (slot >= 0) gg_slots[gg_nslots++] = slot;
                                    if (!last_of_group) return true;
                                    flush(MMQ_TAIL);
                                    gg = mmq::GatherGroup{};
                                } else if (lay.native) {
                                    const auto& f = lay.fmt[(size_t) l];
                                    // NVFP4: this expert's scales go next to its group slot in the same launch
                                    mmq::gather_native(blob_dev, blob_dev + f.up_off, mmq_gub / 2, blob_dev + f.down_off,
                                                       mmq_db, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs,
                                                       f.tail_off ? blob_dev + f.tail_off : nullptr,
                                                       f.tail_off ? m.grp_tail + q * 4 : nullptr, last_of_group ? MMQ_TAIL : 0);
                                } else {
                                    mmq::gather_strata_q2(blob_dev, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs);
                                    if (last_of_group) {
                                        cudaMemsetAsync(m.grp_gu + (size_t) (q + 1) * mmq_gub, 0, MMQ_TAIL, m.cs);
                                        cudaMemsetAsync(m.grp_d + (size_t) (q + 1) * mmq_db, 0, MMQ_TAIL, m.cs);
                                    }
                                }
                                if (slot >= 0 && !gg_now) { cudaEventRecord(m.used[slot], m.cs); m.used_of[slot] = slot; }
                                if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
                                }
                                // the group's products: gate/up, swiglu, the group's H to q8_1, down
                                const size_t j0 = j - q, g = j0 / MMQ_GROUP, n = order.size();
                                const int ngx = (int) (q + 1);
                                const int64_t r0 = m.bounds_host[j0], nr = m.bounds_host[j + 1] - r0;
                                int64_t maxr = 0;
                                for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
                                pt.mark(kPfGemmGU, cs);
                                if (wmma_l) {   // pf_wmma: gate/up, SwiGLU to FP16, down - the group in 2 tile launches
                                    const int32_t* dnb = m.bounds_dev + n + 1 + g * (MMQ_GROUP + 1);
                                    const bool ok1 = strata::kernels::gemm_iq_f16_grouped(
                                        mmq_gt, m.grp_gu, mmq_gub, 1280, (int) N, m.Xs, (int) N, m.bounds_dev + j0, ngx,
                                        (int) maxr, m.GU, 1280, m.cs);
                                    swiglu_split_f16(m.GU + r0 * 1280, m.Hh + r0 * 640, nr, 640, !lay.native, m.cs);
                                    if (had_d)   // APR: folded down weights take H (s * h), rotated in fp16 storage as the FP16 path does
                                        strata::kernels::hadamard_rows_f16(m.Hh + r0 * 640, nr, 640, had_block, had_sh, m.cs);
                                    pt.mark(kPfGemmD, cs);
                                    const bool ok2 = strata::kernels::gemm_iq_f16_grouped(
                                        mmq_dt, m.grp_d, mmq_db, (int) N, 640, m.Hh + r0 * 640, 640, dnb, ngx, (int) maxr,
                                        m.Dm + r0 * N, (int) N, m.cs);
                                    if (!ok1 || !ok2) { err = "prefill: the WMMA expert tiles could not launch"; return false; }
                                    return true;
                                }
                                if (direct) {
                                    // one launch: the direct experts' guards (their down over-read) and the zeroed tail
                                    // after the last gathered expert (see MMQ_TAIL)
                                    mmq::prep_group(gd, mmq_plan().d_over[(size_t) l],
                                                    ns ? m.grp_gu + (size_t) ns * mmq_gub : nullptr,
                                                    ns ? m.grp_d + (size_t) ns * mmq_db : nullptr, MMQ_TAIL, m.cs);
                                }
                                mmq::Product gu;
                                gu.w = m.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
                                gu.n = ngx; gu.xq = m.Xq; gu.bounds = m.bounds_dev + j0; gu.ids = m.ids_identity;
                                gu.total_rows = T * K; gu.max_rows = maxr; gu.dst = m.GU; gu.ld_dst = 1280;
                                gu.y_scale = mmq::fp4_activations(mmq_gt) ? m.Xscale : nullptr;
                                if (direct) gu.tab = &tgu;
                                m.mmq_ctx->run(gu, m.cs);
                                if (lay.native && lay.fmt[(size_t) l].tail_off)   // gate/up scales applied as read
                                    mmq::swiglu_scaled(m.GU + r0 * 1280, m.H + r0 * 640, nr, 640, m.bounds_dev + j0, ngx,
                                                       m.grp_tail, r0, m.cs);
                                else
                                    mmq::swiglu(m.GU + r0 * 1280, m.H + r0 * 640, nr, 640, !lay.native, m.cs);
                                if (had_d)
                                    strata::kernels::hadamard_rows(m.H + r0 * 640, m.H + r0 * 640, nr, 640, had_block, had_sh, m.cs);
                                pt.mark(kPfGemmD, cs);
                                mmq::quantize(m.H + r0 * 640, nullptr, m.Hq, mmq_dt, 640, 640, nr, m.cs, m.Hscale);
                                mmq::Product dn;
                                dn.w = m.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
                                dn.n = ngx; dn.xq = m.Hq; dn.bounds = m.bounds_dev + n + 1 + g * (MMQ_GROUP + 1);
                                dn.ids = m.ids_identity; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = m.Dm + r0 * N;
                                dn.ld_dst = N;
                                dn.y_scale = mmq::fp4_activations(mmq_dt) ? m.Hscale : nullptr;
                                if (direct) dn.tab = &td;
                                m.mmq_ctx->run(dn, m.cs);
                                if (lay.native && lay.fmt[(size_t) l].tail_off)   // s_down: applied by the combine
                                    mmq::down_row_scales(m.row_sd + r0, dn.bounds, ngx, m.grp_tail, nr, m.cs);
                                if (direct) {
                                    tgu = mmq::WeightTab{};
                                    td = mmq::WeightTab{};
                                    gd = mmq::GuardTab{};
                                    ns = 0;
                                }
                                return true;
                            }
                            const int q = (int) (j % DQ);
                            if (lay.native) {
                                // plan v0.3 P6: a native pack's layer, dequantized by llama.cpp's own formulas
                                const auto& f = lay.fmt[(size_t) l];
                                strata::kernels::iq_dequant_gu_f16(f.gu_type, blob_dev, blob_dev + f.up_off, f.n_ff, f.n_embd,
                                                                   m.dq_gu[q], m.cs,
                                                                   f.tail_off ? (const float*) (blob_dev + f.tail_off) : nullptr);
                                strata::kernels::iq_dequant_f16(f.d_type, blob_dev + f.down_off, f.n_embd * f.n_ff, m.dq_d[q], m.cs,
                                                                f.tail_off ? (const float*) (blob_dev + f.tail_off) + 2 : nullptr);
                            } else {
                                blob_dequant_f16(blob_dev, m.dq_gu[q], m.dq_d[q], m.cs);
                            }
                            if (slot >= 0) { cudaEventRecord(m.used[slot], m.cs); m.used_of[slot] = slot; }
                            const int64_t o0 = m.off[(size_t) e], ne = m.cnt[(size_t) e];
                            pt.mark(kPfGemmGU, cs);
                            m.gemm.f16(m.Xs + o0 * N, m.dq_gu[q], m.GU + o0 * 1280, ne, 1280, N);
                            swiglu_interleaved(m.GU + o0 * 1280, m.Hh + o0 * 640, ne, m.cs);
                            // APR, deliberately in place on the fp16 H: hadamard_rows_f16 reads fp16, runs the
                            // butterflies in fp32 and rounds once on the store, so against an fp32 rotation the only
                            // extra error is the one fp16 rounding of the SwiGLU output before it (~2^-11 relative per
                            // value; the GEMM reads fp16 either way).  An fp32 rotation would need a T*K*640 float
                            // buffer (~200 MiB at an 8k chunk) or a fused SwiGLU + rotation kernel, for the fallback
                            // of types and builds without MMQ (the MMQ path rotates fp32 H).  native_expert_parity
                            // bounds the fp16 rotation against the float one at relative L1 error < 2e-3.  The
                            // gate/up input (Xs above) is handled the same way.
                            if (had_d) strata::kernels::hadamard_rows_f16(m.Hh + o0 * 640, ne, 640, had_block, had_sh, m.cs);
                            pt.mark(kPfGemmD, cs);
                            m.gemm.f16(m.Hh + o0 * 640, m.dq_d[q], m.Dm + o0 * N, ne, N, 640);
                            return true;
                        };
                        if (!stream_all) {
                            size_t staged = 0;
                            size_t pending = 0;
                            const bool stream_ahead = stream_ahead_enabled();
                            const size_t lookahead = group_resident ? (size_t) (STAGE_GRP - MMQ_GROUP - 1) : STAGE - 1;
                            for (size_t j = 0; j < order.size(); ++j) {
                                // Resident experts occupy no staging slot. Keep STAGE actual transfers ahead,
                                // rather than STAGE positions in the mixed resident/streamed order.
                                while (staged < order.size() && (stream_ahead ? pending < STAGE : staged <= j + lookahead)) {
                                    if (!stage_one(staged)) return false;
                                    if (stage_of[staged] >= 0) ++pending;
                                    ++staged;
                                }
                                const int32_t e = order[j];
                                if (stage_of[j] < 0) {
                                    ++stats_.experts_resident;
                                    if (!compute(j, m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]), -1,
                                                 m.host_res[(size_t) l * m.g->n_expert + e])) return false;
                                } else {
                                    if (!group_resident) {   // grouped: the flush waits once, on the group's last copy
                                        pt.mark(kPfWaitCopy, cs);
                                        cudaStreamWaitEvent(m.cs, m.copied[stage_of[j]], 0);
                                    }
                                    if (!compute(j, m.stage_dev[stage_of[j]], stage_of[j], -1)) return false;
                                    --pending;   // compute recorded the slot's release event before any reuse
                                }
                            }
                        } else {
                            // the streamed walk: this layer's entries [k, kend) in id order; an entry the routing did not
                            // pick only gives its slot back
                            size_t k = seq_start[(size_t) l];
                            const size_t kend = seq_start[(size_t) l + 1];
                            auto release_to = [&](int32_t e_stop) {
                                while (k < kend && seq[k].e < e_stop) {
                                    if (gg_nslots > 0) flush(0);   // the open group's slots get their event first
                                    const int sl = (int) (k % (size_t) m.ring);
                                    cudaEventRecord(m.used[sl], m.cs);
                                    m.used_of[sl] = sl;
                                    consumed = ++k;
                                    give_back(consumed);
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
                                        cudaStreamWaitEvent(m.cs, m.copied[sl], 0);
                                    }
                                    if (!compute(j, m.stage_dev[sl], sl, -1)) return false;
                                    consumed = ++k;
                                    if (!group_gather || gg_nslots == 0) give_back(consumed);   // its group was gathered
                                } else {
                                    const bool r0 = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                                    const uint8_t* bp = r0 ? m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e])
                                                           : (m.pp && m.pp->peer ? m.pp->peer->slot_ptr(l, e) : nullptr);   // over the cap: P2P
                                    if (bp == nullptr) { err = "prefill: an expert is neither resident, streamed nor on the peer"; return false; }
                                    ++stats_.experts_resident;
                                    if (!compute(j, bp, -1, r0 ? m.host_res[(size_t) l * m.g->n_expert + e] : -1)) return false;
                                    if (group_gather && gg_nslots == 0) give_back(consumed);
                                }
                            }
                            release_to(m.g->n_expert);
                        }
                    }
                    pt.mark(kPfCombine, cs);
                    if (cpu_fut.valid()) {   // set_cpu_pool: the CPU's rows into Dm's tail, row_sd 1 (s_down is in them)
                        if (cpu_share_env() < 0.0) cudaEventRecord(m.cpu_ev[1], m.cs);   // after the GPU's expert work
                        if (!cpu_fut.get()) { err = "prefill: a CPU expert has no blob"; return false; }
                        if (cpu_share_env() < 0.0 && n_stream > n_cpu) {
                            m.cpu_pend = true;
                            m.pend_cpu_ms = cpu_ms;
                            m.pend_n_cpu = n_cpu;
                            m.pend_n_gpu = n_stream - n_cpu;
                        }
                        stats_.cpu_share = cpu_share_env() >= 0.0 ? cpu_share_env() : m.cpu_share_now;
                        const int64_t r0c = T * K - rows_cpu;
                        cudaMemcpyAsync(m.Dm + (size_t) r0c * N, m.cpu_rows, (size_t) rows_cpu * N * sizeof(float),
                                        cudaMemcpyHostToDevice, m.cs);
                        if (use_mmq && !fused_l && lay.native && lay.fmt[(size_t) l].tail_off)
                            cudaMemcpyAsync(m.row_sd + r0c, m.cpu_ones, (size_t) rows_cpu * sizeof(float),
                                            cudaMemcpyHostToDevice, m.cs);
                        stats_.experts_cpu += n_cpu;
                    }
                    // no reading from a layer the share could not change (nothing taken) or with one-time costs
                    if (cpu_set >= 0 && n_stream > 0 && (!cpu_arm || n_cpu > 0) && !cpu_cold) {
                        cudaEventRecord(m.cpu_wall[2 * cpu_set + 1], m.cs);
                        m.pend_wall = cpu_arm ? 2 : 1;
                        m.pend_set = cpu_set;
                        m.pend_l = (int) l;
                        m.pend_wall_n = n_stream;
                    }
                    if (peer_now) {   // multi-GPU: the peer's rows are in Dm (or, without P2P, in host memory)
                        cudaStreamWaitEvent(m.cs, m.pp->ev_done, 0);
                        if (!m.pp->p2p && !m.pp->sums)
                            copy_f32_wide(m.Dm + (size_t) m.pp->back_at * N, m.pp->host_rows, m.pp->back_rows * N, m.cs);
                    }
                    if (peer_now && m.pp->sums && m.pp->f16)
                        moe_combine_peer16(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.pp->host_sum16, m.pp->back_at, m.bo, T, m.cs);
                    else if (peer_now && m.pp->sums)
                        moe_combine_peer(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.pp->host_sum, m.pp->back_at, m.bo, T, m.cs);
                    else
                        moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs,
                                    use_mmq && !fused_l && lay.native && lay.fmt[(size_t) l].tail_off ? m.row_sd : nullptr);
                    // debug: STRATA_DBG_NAN=1 reports the first layer of a chunk whose MoE produced non-finite values
                    if (static const bool dbg = std::getenv("STRATA_DBG_NAN") != nullptr; dbg) {
                        cudaStreamSynchronize(m.cs);
                        auto bad = [&](const float* d, int64_t n) {
                            std::vector<float> h((size_t) n);
                            cudaMemcpy(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost);
                            int64_t c = 0;
                            for (float v : h) c += !std::isfinite(v);
                            return c;
                        };
                        // (fused: GU and H hold the grouping tables and int8 rows, not floats)
                        // (the CPU's rows, the last rows_cpu of GU and H, are never written by the GPU: not checked)
                        const int64_t gpu_rows = T * K - rows_cpu;
                        const int64_t bgu = fused_l ? 0 : bad(m.GU, gpu_rows * 1280), bdm = bad(m.Dm, T * K * N),
                                      bbo = bad(m.bo, T * N);
                        const int64_t bh = m.H && !fused_l ? bad(m.H, gpu_rows * 640) : -1;
                        static int64_t reported = -1;
                        if ((bgu || bdm || bbo || bh > 0) && reported != stats_.chunks) {
                            reported = stats_.chunks;
                            std::fprintf(stderr, "strata dbg: layer %lld (mmq %d, types %d/%d, %zu experts): non-finite GU %lld "
                                         "H %lld Dm %lld bo %lld of T %lld\n", (long long) l, (int) use_mmq, mmq_gt, mmq_dt,
                                         n_order, (long long) bgu, (long long) bh, (long long) bdm, (long long) bbo,
                                         (long long) T);
                        }
                    }
                }
                pt.mark(kPfHcWrite, cs);   // the write-back (and the next half's norm) on its own line; it used to be charged to the phase before
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
                        cudaMalloc(&Rc, (size_t) T * D * 4); cudaMalloc(&rc, (size_t) T * HC * 4);
                        cudaMalloc(&xc, (size_t) T * D * 2);
                        cudaMemcpyAsync(Rc, m.R, (size_t) T * D * 4, cudaMemcpyDeviceToDevice, m.cs);
                        gr_write(Rc, m.bo, m.inj, HC, T, m.cs);
                        strata::kernels::cvec_apply(Rc, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);
                        gr_norm_rs(Rc, (const float*) wnn->data, EPS, rc, xc, T, m.cs, nullptr, D, xf16);
                        gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, cv_dir + l * N, cv_s + l, cv_on,
                                              strata::kernels::cvec().mode, (const float*) wnn->data, EPS, m.grs,
                                              m.xn16, T, m.cs, m.xn16_lo, ldx, xf16);
                        std::vector<float> a((size_t) T * D), b((size_t) T * D), ra((size_t) T * HC), rb((size_t) T * HC);
                        std::vector<uint16_t> xa((size_t) T * D), xb((size_t) T * D);
                        cudaMemcpyAsync(a.data(), m.R, a.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaMemcpyAsync(b.data(), Rc, b.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaMemcpyAsync(ra.data(), m.grs, ra.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaMemcpyAsync(rb.data(), rc, rb.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                        cudaMemcpyAsync(xa.data(), m.xn16, xa.size() * 2, cudaMemcpyDeviceToHost, m.cs);
                        cudaMemcpyAsync(xb.data(), xc, xb.size() * 2, cudaMemcpyDeviceToHost, m.cs);
                        cudaStreamSynchronize(m.cs);
                        cudaFree(Rc); cudaFree(rc); cudaFree(xc);
                        std::fprintf(stderr, "strata: STRATA_CVEC_FUSE_CHECK layer %lld T %lld: R %s, rs %s, xn16 %s\n",
                                     (long long) l, (long long) T,
                                     std::memcmp(a.data(), b.data(), a.size() * 4) ? "DIFFERS" : "identical",
                                     std::memcmp(ra.data(), rb.data(), ra.size() * 4) ? "DIFFERS" : "identical",
                                     std::memcmp(xa.data(), xb.data(), xa.size() * 2) ? "DIFFERS" : "identical");
                    } else
                    gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, cv_dir + l * N, cv_s + l, cv_on,
                                          strata::kernels::cvec().mode, (const float*) wnn->data, EPS, m.grs, m.xn16,
                                          T, m.cs, m.xn16_lo, ldx, xf16);
                    normed = true;
                } else if (wnn) {
                    gr_write_norm_rs(m.R, m.bo, m.inj, HC, (const float*) wnn->data, EPS, m.grs, m.xn16, T, m.cs,
                                     m.xn16_lo, ldx, xf16);
                    normed = true;
                } else {
                    gr_write(m.R, m.bo, m.inj, HC, T, m.cs);
                }
                if (steered && !cv_fused)   // --control-vector-scaled
                    strata::kernels::cvec_apply(m.R, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);
#if defined(STRATA_HAVE_DFLASH2)
                if (half == 1 && tap_rows > 0 && tap_slot[(size_t) l] >= 0)   // DFlash2: this layer's completed output, mean of the streams
                    strata::kernels::df2_tap_rows(m.R + (size_t) tap_t0 * D, tap_dev_ + (size_t) tap_slot[(size_t) l] * N, (int64_t) tap_layers_.size() * N, N,
                                                  (int) HC, (int) tap_rows, m.cs);
#endif
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
            const int hp = m.hand_pieces;
            if (hp > 0) {
                // STRATA_HANDOFF_PIPE: the pieces' downloads stay queued; the host waits for the chunk's work only (the
                // state on_stage_chunk saves, the error check), the next stage's uploads wait on the pieces' events
                bool ok = cudaEventRecord(m.hand_done[hand_buf_], m.cs) == cudaSuccess;
                for (int i = 0; ok && i < hp; ++i) {
                    int64_t r0 = 0, nr = 0;
                    handoff_piece(T, hp, i, r0, nr);
                    if (nr > 0)
                        ok = cudaMemcpyAsync(h + (size_t) r0 * D, m.R + (size_t) r0 * D, (size_t) nr * D * 4,
                                             cudaMemcpyDeviceToHost, m.cs) == cudaSuccess;
                    ok = ok && cudaEventRecord(m.hand_ev[hand_buf_][i], m.cs) == cudaSuccess;
                }
                if (!ok || cudaEventSynchronize(m.hand_done[hand_buf_]) != cudaSuccess) {
                    err = std::string("prefill: the layer split's hand-off: ") + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
            } else if (cudaMemcpyAsync(h, m.R, (size_t) T * D * 4, cudaMemcpyDeviceToHost, m.cs) != cudaSuccess ||
                       cudaStreamSynchronize(m.cs) != cudaSuccess) {
                err = std::string("prefill: the layer split's hand-off: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            // Wait only for the DIRECT successor's previous chunk. That successor
            // may already have forwarded its older chunk to later GPUs.
            if (on_stage_chunk && !on_stage_chunk(p0 + T, err)) return false;
            if (next_run_.valid() && !next_run_.get()) { err = next_err_; return false; }
            next_err_.clear();
            next_->hand_in_ = h;
            next_->hand_ev_ = hp > 0 ? static_cast<const void*>(m.hand_ev[hand_buf_]) : nullptr;
            next_->hand_pieces_ = hp;
            next_->single_chunk_ = single_chunk;
            next_->pipe_chunked_ = pipe_chunked;
            if (handoff_pipe_worker()) {
                if (!m.hand_worker) m.hand_worker = std::make_unique<HandWorker>();
                next_run_ = m.hand_worker->submit(std::packaged_task<bool()>([this, tokens, c0, T, p0] {
                    const bool r = next_->run_impl(tokens + c0, T, p0, next_err_);
                    (void) cudaGetLastError();   // clear what a thread's end cleared (std::async)
                    return r;
                }));
            } else
            next_run_ = std::async(std::launch::async, [this, tokens, c0, T, p0] {
                strata::aux_cpus::pin_current_thread();
                return next_->run_impl(tokens + c0, T, p0, next_err_);
            });
            hand_buf_ ^= 1;
            continue;   // the last stage reports the chunk (on_chunk)
        }
        if (const char* dump = std::getenv("STRATA_PREFILL_DUMP_R")) {   // debug: the final residuals, every 64th
            cudaStreamSynchronize(m.cs);                                  // position (A/B quality of this path)
            if (std::FILE* f = std::fopen(dump, c0 == 0 ? "wb" : "ab")) {
                std::vector<float> row((size_t) D);
                for (int64_t t = (64 - p0 % 64) % 64; t < T; t += 64) {
                    cudaMemcpy(row.data(), m.R + t * D, (size_t) D * 4, cudaMemcpyDeviceToHost);
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
            cudaStreamSynchronize(m.cs);
            if (std::FILE* f = std::fopen(dump, "ab")) {
                constexpr int64_t kRows = 512;
                std::vector<float> rows((size_t) (kRows * D));
                std::vector<uint16_t> out((size_t) (kRows * D));
                for (int64_t t0 = 0; t0 < T; t0 += kRows) {
                    const int64_t nr = std::min<int64_t>(kRows, T - t0);
                    cudaMemcpy(rows.data(), m.R + t0 * D, (size_t) (nr * D) * 4, cudaMemcpyDeviceToHost);
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
#if defined(STRATA_HAVE_DFLASH2)
        if (tap_rows > 0 && on_taps) {
            if (cudaStreamSynchronize(m.cs) != cudaSuccess) { err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError()); return false; }
            if (!on_taps(tap_dev_, tap_rows, p0 + tap_t0, err)) return false;
        }
#endif
        if (on_chunk || on_stage_chunk) {
            const auto toc = Clock::now();
            if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
                err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
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
        cudaStreamSynchronize(m.cs);
        auto bad = [&](const float* d, int64_t n) {
            std::vector<float> h((size_t) n);
            cudaMemcpy(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost);
            int64_t c = 0;
            double mx = 0;
            for (float v : h) { c += !std::isfinite(v); if (std::isfinite(v)) mx = std::max(mx, (double) std::fabs(v)); }
            std::fprintf(stderr, " %lld non-finite (max |x| %.3g)", (long long) c, mx);
        };
        const int64_t last = (n - 1) % C;
        std::fprintf(stderr, "strata dbg: prompt end: last residual row");
        bad(m.R + last * D, D);
        if (ss.ple.ready()) { std::fprintf(stderr, "; PLE history"); bad(ss.ple.hist, (int64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM); }
        std::fprintf(stderr, "; GDN state 0");
        bad(ss.gdn_state, 64 * 1024);
        std::fprintf(stderr, "\n");
    }
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    // (PR #121) an expert copy that failed on the copy stream surfaces here, not in the next request
    if (const cudaError_t cst = cudaStreamSynchronize(m.copy); cst != cudaSuccess) {
        err = std::string("prefill: expert copy stream: ") + cudaGetErrorString(cst);
        return false;
    }
    if (kv_prefetch) {
        if (const cudaError_t cst = cudaStreamSynchronize(m.kv_copy); cst != cudaSuccess) {
            err = std::string("prefill: KV prefetch stream: ") + cudaGetErrorString(cst);
            return false;
        }
        std::fprintf(stderr, "strata KV prefetch: %lld layer prefixes, single existing staging pool\n",
                     (long long) kv_prefetches);
    }
    stats_.ms_total += ms_since(t_start);
    {
        static const bool f16_range = std::getenv("STRATA_F16_RANGE") != nullptr;
        float xm = 0.0f, wy[2] = {0.0f, 0.0f}; unsigned long long xo = 0, wyo[2] = {0, 0};
        if (f16_range && f16_range_read(xm, xo, true) && gemm_f16_range_read(wy, wyo, true))
            std::fprintf(stderr, "strata f16 range: %lld tokens: activation images max |x| %.1f (%llu beyond 65504 or NaN), "
                                 "BF16 weights max |w| %.3f (%llu), FP16 GEMM outputs max |y| %.1f (%llu not finite)\n",
                         (long long) n, xm, xo, wy[0], wyo[0], wy[1], wyo[1]);
    }
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
            int pd = 0; cudaGetDevice(&pd); cudaSetDevice(pe.dev); cudaStreamSynchronize(m.pp->s); pe.fold(); cudaSetDevice(pd);
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
        cudaStreamSynchronize(m.cs);
        std::vector<uint8_t> b((size_t) gdn_floats * 4);
        std::string line;
        char h[8];
        for (int64_t i = 0; i < ss.gdn_alloc; ++i) {
            cudaMemcpy(b.data(), ss.gdn_state + (size_t) i * gdn_floats, b.size(), cudaMemcpyDeviceToHost);
            uint64_t x = 1469598103934665603ull;
            for (uint8_t c : b) x = (x ^ c) * 1099511628211ull;
            std::snprintf(h, sizeof(h), "%04llx ", (unsigned long long) (x & 0xffff));
            line += h;
        }
        std::fprintf(stderr, "strata prefill: GDN_HASH %s\n", line.c_str());
    }
    return true;
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

bool Prefill::reserve_uploads(int64_t T) {
    bool ok = true;
    if (impl_ != nullptr && T > 0) {
        Impl& m = *impl_;
        const core::OnDevice on_device(m.device);   // (the ring is the device's that is current at its first use)
        // a text chunk's largest pageable uploads through the ring: the routing tables (and their ids down, T x K
        // each); the PLE rows when they could not be pinned (T x N).  A picture's rows are not covered.
        size_t bytes = (size_t) (T * K) * sizeof(int32_t);
        if (!m.ple_pageable[0].empty() || !m.ple_pageable[1].empty())
            bytes = std::max(bytes, (size_t) (T * N) * sizeof(float));
        ok = m.up_ring.reserve(bytes);
    }
    if (next_ != nullptr) ok = next_->reserve_uploads(T) && ok;
    return ok;
}

bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
#if defined(STRATA_GPU_METAL)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = strata::metal_engine::prompt_run(tokens, n, pos0, embd_rows, on_chunk, err);
        stats_.tokens += n;
        stats_.chunks += 1;
        stats_.ms_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return ok;
    }
#endif
    const bool body_ok = run_impl(tokens, n, pos0, err);

    std::string drain_err;
    const bool drain_ok = drain_pipeline(drain_err);

    if (!body_ok) return false;
    if (!drain_ok) {
        err = drain_err;
        return false;
    }
    // STRATA_MMQ_DIRECT=1 with STRATA_MMQ_DIRECT_LOG=1 (a check, not a default): each stage's totals so far - how many
    // resident experts its prompt MMQ read in their slots, of all its resident and streamed ones
    if (mmq::direct_mode() == 1) {
        static const bool log = [] { const char* v = std::getenv("STRATA_MMQ_DIRECT_LOG"); return v && std::atoi(v) != 0; }();
        for (const Prefill* p = this; log && p != nullptr; p = p->next_)
            std::fprintf(stderr, "strata prefill: layers %lld..%lld: %lld of %lld resident experts read in place, %lld "
                                 "streamed (STRATA_MMQ_DIRECT, totals)\n", (long long) p->stage_lb_,
                         (long long) p->stage_le_, (long long) p->stats_.experts_direct,
                         (long long) p->stats_.experts_resident, (long long) p->stats_.experts_streamed);
    }
    return true;
}

}  // namespace strata::prefill
