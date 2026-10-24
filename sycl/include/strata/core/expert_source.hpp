// include/strata/core/expert_source.hpp - P2.S3/P2.S5: the expert pool's adapter to the host loop.
//
// `session_loop` publishes, per layer, the NORMED activation `x_f`, the ten routed expert ids and their router
// weights, and expects `k x n_embd` floats back.  This is the piece that turns those four things into an
// `ExpertPool::run` call.  Nothing here is clever, and that is the point: the pool, the kernel and the loop are
// each already verified, so this file has exactly one job - get the CONTRACT between them right.
//
// THE CONTRACT, and each clause is a way to be wrong:
//
//   1. **`x_f` IS THE SAME PINNED BUFFER EVERY LAYER.**  Its ADDRESS never changes, so the `ActQ` cannot be
//      cached by pointer - a cache keyed on `x_f` would quantize layer 0 and reuse it for all 47 remaining
//      layers, which is a finite, plausible, completely wrong token.  There is no cache here at all: the
//      conversion is one 2560-element pass against a 0.3 ms/layer budget, and a correct answer is worth more
//      than the microseconds.
//   2. **THE POOL DOES NOT APPLY THE ROUTER WEIGHT.**  `moe_combine` (`src/core/layer.cpp:479`) sums
//      `w[i] * parts[i]` on the device.  `ExpertJob::weight` is a diagnostic field; setting it here would
//      apply the weight twice, which is invisible in a single layer and compounds over 48.
//   3. **THE KERNEL'S GEOMETRY IS FIXED BY THE ARTIFACT** (`H = 2560`, `FF = 640`, `BLOB = 1,382,400`).  A
//      different `n_embd` or a different expert width is REFUSED rather than mis-indexed, because the blob's
//      internal offsets are compile-time constants and reading a 640-wide expert as a 2560-wide one walks off
//      the end into the next expert's bytes without faulting.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/exchange_storage.hpp"
#include "strata/core/hit_hook.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <string>
#include <vector>

namespace strata::kernels::cpu {
struct ExpertLayout;
}

namespace strata::core {

class PeerExperts;   // multi-GPU: the second GPU's expert tier (peer_experts.hpp)
class RemoteExperts;
struct LoadStats;

namespace detail {

/// Sentinel used by the pure complement planner for a blob that remains in the mmap fallback.
inline constexpr uint64_t kNoCacheComplement = ~uint64_t{0};

/// Required cgroup-v2 usage counters for the conservative cache-reclaim allowance.
struct CgroupMemoryStat {
    uint64_t current = 0;
    uint64_t inactive_file = 0;
    uint64_t file_dirty = 0;
    uint64_t file_writeback = 0;
    bool valid = false;
};

/// Calculate additional bytes under a finite cgroup limit after reclaiming only clean inactive file cache.
/// Returns false when the required memory.stat counters were unavailable.
bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes);

/// #633: the RAM this process can get.  `available`: MemAvailable (Windows: the available physical memory), lowered
/// to the room under the tightest cgroup limit; `cgroup_limit`: that tightest limit itself (v2 memory.max of the group
/// and its ancestors, or v1 memory.limit_in_bytes), ~0 when there is none - what a container can never exceed.
struct HostMemory {
    uint64_t available = 0;
    uint64_t cgroup_limit = ~uint64_t{0};
    uint64_t commit = ~uint64_t{0};   ///< available Windows commit capacity; ~0 when not reported
};

/// Linux reads `meminfo`, `self_cgroup` and the cgroup tree under `cgroup_root` (the parameters are for tests; the
/// defaults are the real files).  cgroup v2 as before (an unreadable limit of a group that has one fails); cgroup v1's
/// memory controller (`<root>/memory/<path>`: memory.limit_in_bytes - memory.usage_in_bytes); no cgroup line at all is
/// MemAvailable alone.  False when the RAM cannot be determined.
bool host_available_memory(HostMemory& m, const std::string& meminfo = "/proc/meminfo",
                           const std::string& self_cgroup = "/proc/self/cgroup",
                           const std::string& cgroup_root = "/sys/fs/cgroup");

/// Bound a resident budget by RAM and commit capacity after headroom; leave 256 MiB more when clamping.
/// Pass UINT64_MAX for commit when the platform does not report it.
uint64_t clamp_resident_budget(uint64_t requested, uint64_t physical, uint64_t commit, uint64_t headroom);

/// Build compact offsets for experts absent from both the primary GPU cache and an optional second GPU tier.
/// Kept CPU-only so selection and byte accounting can be tested without initializing a GPU.
bool make_cache_complement_plan(
    int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs,
    std::vector<uint64_t>& offsets, uint64_t& bytes, std::string& err);

/// Resolve one blob through the compact copy when present, otherwise preserve its exact mapped-file fallback.
const uint8_t* cache_complement_blob_or_fallback(
    size_t index, const std::vector<uint64_t>& offsets, const uint8_t* complement_host,
    const uint8_t* mapped_fallback);

/// The resident RAM mode: which GPU-cache slots' experts are kept in RAM too.  The prompt path lends the cache's
/// LAST slots (from `lend_from` on; a short prompt lends only the last few), and a lent slot's expert is streamed
/// from RAM during the prompt and copied back into its slot after it.  `base_bytes` (every expert no slot holds)
/// must fit `budget`; slots are then added from the end down to `lend_from` while they still fit.  Returns the
/// first slot kept in RAM (`slot_bytes.size()` = none), or -1 when `base_bytes` alone exceeds `budget`.
int64_t choose_resident_keep_from(const std::vector<uint64_t>& slot_bytes, uint64_t base_bytes, uint64_t budget,
                                  int64_t lend_from);

/// The adaptive tier swapped `in` into a GPU slot and `out` out of it: `out` takes `in`'s place in the compact copy
/// (the caller copies out's bytes there).  False, and nothing changed, unless `in` is in the copy and `out` is not.
bool exchange_cache_complement(std::vector<uint64_t>& offsets, size_t in, size_t out);

}  // namespace detail



/// Where one routed expert's bytes come from.
///
/// Phase 2 has NO cache (`phase-2-correct-engine.md`: hit rate `h = 0`), so the only implementation is a
/// file-backed reader.  The interface exists anyway because Phase 3 replaces exactly this object with the VRAM
/// cache, and because a test can supply an in-memory source without a 34 GB artifact.
class ExpertSource {
public:
    /*
    DPCT1109: Virtual functions cannot be called in SYCL device code. You
    need to adjust the code.
    */
    virtual ~ExpertSource() = default;

    /// The expert-layout blob for `(layer, expert)`, or nullptr if it cannot be produced.
    ///
    /// The pointer only has to stay valid until the next `blob()` call: with `h = 0` every expert is computed
    /// immediately and nothing is retained.  A CACHING source must return pointers into the cache, not into a
    /// reused staging buffer - otherwise the pool would read the next expert's bytes while computing this one.
    virtual const uint8_t* blob(int64_t layer, int64_t expert) = 0;

    /// Blobs touched, for the driver to report.  A source that does not count returns 0.
    virtual int64_t reads() const { return 0; }

    /// Plan v0.3 P5: whether `blob(layer, expert)` lies in page-locked, CUDA-registered memory, so an
    /// asynchronous host-to-device copy can DMA it directly (no staging copy on the CPU).
    virtual bool pinned(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return false; }

    /// Called once before the first expert of a layer.  A source that reads from disk wants to start the read
    /// here so it overlaps the quantisation, and a prefetching source in Phase 3 wants the ids.
    virtual void begin_layer(int64_t layer, const int32_t* ids, int64_t k) { (void) layer; (void) ids; (void) k; }
    /// Plan v0.3 P6: the DEVICE address of a pinned, mapped blob (the GPU can read it over PCIe), or null.
    virtual const uint8_t* device_alias(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return nullptr; }
    /// A file-backed source: start reading this expert's pages now (it will be needed by the CPU); no-op elsewhere.
    virtual void prefetch(int64_t layer, int64_t expert) { (void) layer; (void) expert; }
    /// A file-backed source: this expert lives in VRAM, so its pages need not stay in RAM - hand them back to the
    /// kernel (a later read re-reads the file; no result depends on it).  Returns the bytes released; 0 elsewhere.
    virtual uint64_t release(int64_t layer, int64_t expert) { (void) layer; (void) expert; return 0; }
    /// Whether the verify window may give the GPU a PCIe share of this layer's misses at all (each expert is still
    /// checked with `pinned`).  The arena answers per layer through its expert 0; the resident RAM mode's compact
    /// copy has no expert 0 when the GPU cache holds it, so it answers for the whole copy.
    virtual bool pcie_layer(int64_t layer) const { return device_alias(layer, 0) != nullptr; }

    /// CS-T: whether `blob(layer, expert)` would be assembled into a short-lived buffer (a native pack read from its
    /// GGUF shards in place, where an expert's gate, up and down rows are three separate slices).  Such a pointer
    /// stays valid for the layer it was asked in and the next one or two; a consumer that keeps a blob longer (the
    /// prompt path's stager queues a whole chunk) copies it with `copy_blob` instead.
    virtual bool transient(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return false; }
    /// Disk sessions: the files this source read its experts from, as (role, path), resolved by the loader itself
    /// - the pack's experts.bin, or every GGUF tensor native_experts.txt named, per layer and role
    /// ("expert blk.L.ffn_up").  Filled by open(); empty before.
    const std::vector<std::pair<std::string, std::string>>& model_inputs() const { return inputs_; }
    /// The blob's bytes into `dst` (blob_bytes(layer) of them).  Safe from several threads for a source whose
    /// `transient` can be true.
    virtual bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst);
    /// Asks the OS for these pairs' bytes ahead of the `blob` calls that read them.  False (nothing asked) by default.
    virtual bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const {
        (void) pairs; (void) n; return false;
    }
    /// The `n` experts of `layer` the CPU is about to ask `blob` for, all at once: a source that reads a file may
    /// fetch them in parallel.  The bytes `blob` then returns are the same.  Default: nothing.
    virtual void prefetch(int64_t layer, const int64_t* experts, int64_t n) { (void) layer; (void) experts; (void) n; }
    /// CS-T: the experts of `layer` a predictor expects next - a source that reads a file may start reading their
    /// pages now, in the background.  Only warms: what `blob` returns is unchanged.  Default: nothing.
    virtual void warm(int64_t layer, const int64_t* experts, int64_t n) { (void) layer; (void) experts; (void) n; }
    /// Whether `warm` does anything (the predictor is not run otherwise).
    virtual bool warms() const { return false; }
protected:
    /// What model_inputs() returns; set by a source's open().
    void record_inputs(const std::string& pack_experts, const std::string& gguf,
                       const strata::kernels::cpu::ExpertLayout& lay, bool from_gguf);
    std::vector<std::pair<std::string, std::string>> inputs_;
};

/// CS-T, routing-aware prefetch of the file tier: when the CPU pool starts layer `l`, a worker thread applies layer
/// l+1's router (BF16, host copy) to layer l's MoE input - the residual stream changes little from one layer to the
/// next - takes each token's top `k` experts, drops the ones the GPU cache or the RAM copy holds, and asks the
/// source to `warm` the rest, so their pages are on the way while layer l computes.  A prediction only warms pages:
/// it never changes which experts are computed or how.
class RouterLookahead {
public:
    RouterLookahead() = default;
    ~RouterLookahead();
    RouterLookahead(const RouterLookahead&) = delete;
    RouterLookahead& operator=(const RouterLookahead&) = delete;
    /// `routers[l]`: layer l's ffn_gate_inp as BF16 bits, n_expert rows of n_embd.
    bool start(std::vector<std::vector<uint16_t>> routers, int64_t n_embd, int64_t n_expert, int k, ExpertSource* src,
               std::string& err);
    /// Layer `layer`'s MoE input for `n_tok` tokens (host floats): predict and warm layer + 1.  Never waits: a
    /// prediction still running for an earlier layer makes this one skip.
    void submit(int64_t layer, const float* x, int64_t n_tok, const int32_t* host_res);
    int64_t predicted() const { return predicted_.load(std::memory_order_relaxed); }
    int64_t skipped() const { return skipped_.load(std::memory_order_relaxed); }
    double busy_ms() const { return (double) busy_us_.load(std::memory_order_relaxed) / 1000.0; }

private:
    void run();
    std::vector<std::vector<uint16_t>> routers_;
    int64_t n_embd_ = 0, n_expert_ = 0;
    int k_ = 10;
    ExpertSource* src_ = nullptr;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool quit_ = false, pending_ = false, busy_ = false;
    int64_t layer_ = -1, n_tok_ = 0;
    const int32_t* host_res_ = nullptr;
    std::vector<float> x_;
    std::atomic<int64_t> predicted_{0}, skipped_{0};
    std::atomic<uint64_t> busy_us_{0};
};

/// Plan v0.3 P6: what the GPU computes in a verify window's layer, written by the pool (mapped host memory) right
/// after the ring and published before the CPU starts its own share.  Groups of entries that share a blob; the
/// blob is a VRAM slot or a pinned host blob read over PCIe.
struct GpuPlanSink {
    int32_t* counts = nullptr;             ///< [0] groups, [1] entries
    int32_t* start = nullptr;              ///< cap + 1
    int32_t* dst = nullptr;                ///< cap: the entry's row of parts (token * k + j)
    int32_t* tok = nullptr;                ///< cap: the entry's token
    unsigned long long* ptr = nullptr;     ///< cap: the group's blob, device address
    /// Plan v0.3 P6 (DMA): the PCIe share is a second list of groups - `ptr2[q]` is staging slot q, `start2` indexes
    /// the same dst/tok entries - computed by the GPU after `fetch` has copied their blobs into staging with the
    /// copy engine.  counts[0] = VRAM groups, counts[1] = all entries, counts[2] = PCIe groups.
    unsigned long long* ptr2 = nullptr;
    int32_t* start2 = nullptr;
    unsigned long long staging = 0;
    int64_t staging_cap = 0;
    int64_t cap = 0;
    void (*publish)(void* ctx) = nullptr;
    /// Starts the DMA copies of `n` host blobs (pinned) into staging slots 0..n-1 and signals the GPU when they land.
    void (*fetch)(void* ctx, const uint8_t* const* src, int n, size_t bytes) = nullptr;
    void* ctx = nullptr;
    /// Plan v0.3 P6: how a PCIe group reaches the GPU.  0 = the copy engine stages it (DMA, `fetch`); 1 = the grouped
    /// kernel reads the mapped arena directly; 2 = a copy kernel stages it inside the graph.  For 1 and 2 `ptr2`
    /// holds the arena's device alias.
    int pcie_mode = 0;
};

/// The adapter's own state.  One per session, reused every layer so the token path allocates nothing (P2.T10).
struct ExpertDispatch {
    strata::kernels::cpu::ExpertPool* pool = nullptr;
    ExpertSource* src = nullptr;
    RouterLookahead* lookahead = nullptr;   ///< CS-T: warms the next layer's predicted file-tier experts
    RemoteExperts* remote[3] = {}; ///< optional CUDA1..3 tiers for otherwise CPU-served rows
    int remote_count = 0;
    int64_t n_expert = strata::kernels::cpu::NE;

    /// Counters, for the driver to report rather than for control flow.
    int64_t layers = 0;
    int64_t experts = 0;
    int64_t missing = 0;

    // ================================ R4: THE VRAM TIER, MEASURED BEFORE IT IS USED ================================
    //
    // **THE CACHE IS CONSULTED AND FILLED HERE, AND NOTHING IS COMPUTED FROM IT YET.**  That is deliberate and
    // it is `R4-design-note.md` §7 step 2: the *dispatch* is measured on its own before any kernel is written,
    // because a hit rate measured after the kernel exists cannot say whether a disappointing result was the
    // policy, the split or the kernel.
    //
    // So every expert is still computed by the CPU, exactly as before, and the run is numerically identical
    // with the cache on or off.  What changes is that the engine now reports **h on its own routing, on a real
    // workload** - which is the number `R4.1` asks for and which until now existed only from offline traces.
    //
    // The policy is compulsory-miss: the first time `(layer, expert)` is routed, if a slot is free it is
    // admitted and filled from the arena.  No eviction, because eviction policy is the measured question
    // (`R4.1`'s LFU-decay vs LRU sweep) and a placeholder would set the hit rate everything is sized against.
    ExpertCache* cache = nullptr;
    int64_t cache_hits = 0;      ///< lookups already resident
    int64_t cache_admitted = 0;  ///< lookups that took a slot
    int64_t cache_refused = 0;   ///< lookups with no slot free (the cache is full)
    void* cache_stream = nullptr;
    const char* cache_fail = nullptr;

    // ================================ R4.2c: THE HITS GO TO THE GPU ================================
    //
    // **THE SPLIT IS BY ROUTER INDEX, AND THAT IS THE WHOLE TRICK.**  A layer routes ten experts; the resident
    // ones are computed on the GPU and the rest on the CPU, and both answers have to end up in `parts` at the
    // index the ROUTER gave them, because that is the order `moe_combine` weights against.  So the CPU's `out`
    // rows are zeroed for the hits, and each hit carries its routed index to the kernel as `dst`.
    //
    // Everything below is per-session state rather than per-call, so the token path allocates nothing (P2.T10).
    const uint8_t* cache_base = nullptr;   ///< the slot arena on the DEVICE
    int64_t cache_blob = 0;                ///< bytes per slot
    const uint64_t* cache_slot_off = nullptr;   ///< plan v0.3 P6: per-slot offsets when the slots differ in size
    void* hit_scratch = nullptr;           ///< `moe_hit_grouped_scratch_bytes(K, ...)`
    float* parts_out = nullptr;            ///< the graph's `parts` buffer, on the device
    /// Where the GPU's hits land, `K x n_embd`, DEVICE and separate from `parts_out` on purpose: see
    /// `HitPhase`.  Zeroed by the hit path each layer before the kernel writes it.
    float* hit_out = nullptr;
    int64_t parts_elems = 0;               ///< `K * n_embd`, the length of both buffers
    const float* mixed = nullptr;          ///< the layer's normed activation, for the hit kernel's Q8_0
    uint8_t* x_q8_0_hit = nullptr;         ///< `(n_embd/32) * 34` bytes, its own buffer
    /// **R4.2h: THE CPU's fp32 ACTIVATION SCALES, `(n_embd/32)` FLOATS.**  Without them the GPU's hits are
    /// computed with the `block_q8_0`'s fp16 `d` while the CPU's misses use the fp32 `ActQ::scale`
    /// (`cpu/expert.cpp:92`) - **4.761e-04 relative on 80 of 80 chunks**, measured with both real
    /// implementations linked in `bench/micro/act_quant_parity.cu`.  That disagreement is why turning the
    /// cache on changed the generated tokens.  Required whenever the hit path runs.
    float* x_q8_0_hit_scale = nullptr;
    int32_t* d_slot = nullptr;             ///< device, K entries
    int32_t* d_dst = nullptr;              ///< device, K entries
    std::vector<int32_t> h_slot, h_dst;    ///< host staging, sized at session setup
    /// **PER ROUTER INDEX, DECIDED IN `Launch` AND CONSUMED BY THE POOL.**  The two callbacks share it
    /// so the decision is made exactly once, on this layer's ids, and neither side can re-decide it.
    std::vector<uint8_t> is_hit;
    bool decided = false;
    /// Plan v0.3 P4 token graph: the STATIC residency table (`n_layers x n_expert`, slot or -1), the host's copy
    /// of what the device hit path reads.  When set, the pool leaves a resident expert's row at zero (the GPU
    /// computes it) without any `Launch` callback.
    const int32_t* host_res = nullptr;
    /// Plan v0.3 P4: split every expert by rows across the pool's threads (default on; A/B `--no-split-rows`).
    bool split_rows = true;

    // ================================ R4.2d: DID THE GPU ACTUALLY START? ================================
    //
    // **THE HIT PATH IS 0.91 ms SLOWER AND THE ONLY EXPLANATION LEFT IS THAT IT NEVER OVERLAPS.**  The kernel
    // is 159 GB/s against the CPU's 35, the CPU's half of the drain falls 6 ms, and none of it reaches the
    // token - which is what it would look like if the enqueued hit kernels did not BEGIN until the host next
    // entered the driver, i.e. after `pool()` returned.  That is the third sighting of this driver behaving
    // that way (rounds 195/287 on the doorbell, round 36 on the head and sampler) and it decides whether R4 can
    // pay at all, so it gets measured rather than argued.
    //
    // `hit_done` is recorded on the stream straight after the hit kernel.  `Combine` - which runs after the
    // pool - queries it: SUCCESS means the GPU finished while the CPU was working, NOT-READY means it had not.
    // Same shape as the doorbell's `rings_mid_graph`, and for the same reason.
    void* hit_done = nullptr;      ///< cudaEvent_t, created at session setup
    int64_t hit_ready = 0;         ///< layers where the hit work was DONE by the time the pool returned
    int64_t hit_late = 0;          ///< layers where it was not
    /// The candidate fix, as an A/B arm: enter the driver once right after the hit launch.  If submission is
    /// lazy, this starts the GPU work before the pool instead of after it.
    bool hit_poke = false;
    bool hit_cpu_order = false;    ///< experimental CPU-order GPU expert arithmetic; opt-in only
    int64_t n_hits = 0;                    ///< this layer's hits
    /// Set by `Launch` and consumed by `Combine`, so a `Combine` with no `Launch` in front of it cannot
    /// add a stale buffer into `parts`.
    bool hit_pending = false;
    const char* hit_fail = nullptr;

    /// Whether the hit path is wired up.  All of it or none of it: a half-configured hit path would compute
    /// some experts twice and others not at all, which is a wrong token rather than an error.
    bool hits_ready() const {
        return cache != nullptr && cache_base != nullptr && parts_out != nullptr && hit_out != nullptr &&
               mixed != nullptr &&
               hit_scratch != nullptr && x_q8_0_hit != nullptr && x_q8_0_hit_scale != nullptr && d_slot != nullptr && d_dst != nullptr;
    }

    std::vector<strata::kernels::cpu::ExpertJob> jobs;
    strata::kernels::cpu::ActQ act;
    /// Plan v0.3 P6 verify window: one quantized activation per token, the multi-token jobs, and the expert ->
    /// job map (reset after every layer).
    std::vector<strata::kernels::cpu::ActQ> act_multi;
    /// Plan v0.3 P6: a native pack's per-token activations (MAXT x kNativeActBytes).
    std::vector<uint8_t> nact_multi;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs_multi;
    std::vector<int16_t> job_of;
    /// Plan v0.3 P6: the verify window's GPU plan (VRAM hits + the PCIe share of the misses); `pcie_num`/256 of
    /// each layer's distinct missed experts (the last ones in routing order) are read by the GPU over PCIe.
    GpuPlanSink* plan = nullptr;
    int pcie_num = 0;
    int64_t pcie_experts = 0;      ///< distinct experts the GPU read over PCIe in verify windows
    /// #588: routed (token, expert) entries the GPU computed from outside its cache in verify windows: read over PCIe
    /// (--pcie-frac, kind 1) or on another GPU (kind 2).  In neither cache_hits nor cache_refused.
    int64_t offload_entries = 0;
    double ms_plan = 0, ms_actq = 0, ms_jobs = 0, ms_run = 0;   ///< verify-window dispatch sections
    /// Plan v0.3 P6: decayed routing counts per (layer, expert) during decode (sized by the caller; empty = off),
    /// which the driver uses to swap the most-routed missing experts into the VRAM tier between rounds.
    std::vector<float> usage;
    int64_t multi_misses = 0;      ///< distinct (layer, expert) pairs the CPU computed in verify windows
    int64_t multi_entries = 0;     ///< routed (token, expert) entries the CPU served in verify windows
    /// Multi-GPU: the second GPU's tier.  Its experts are computed there instead of on the CPU (kind 2).
    PeerExperts* peer = nullptr;
    int64_t peer_entries = 0;      ///< routed entries the peer served in verify windows
    /// Set when `dispatch` could not produce an answer.  The loop itself has no error channel, so this is
    /// where a source failure surfaces: the driver checks it after `session_loop` returns rather than the
    /// engine computing from a half-filled `parts`.
    ///
    /// **`failed` LATCHES AND `fail` IS ONLY THE MESSAGE.**  Clearing the message must not re-arm the adapter:
    /// a session that failed at layer 5 has already fed `moe_combine` whatever `parts` held, so every layer
    /// after it is built on a hole - and resuming into a plausible-looking token is the exact outcome this
    /// whole mechanism exists to prevent.  `failed` is what the short-circuit reads.
    bool failed = false;
    const char* fail = nullptr;
    int64_t fail_layer = -1;
    int64_t fail_expert = -1;
};

/// `strata::core::PoolFn`, exactly.  Silent on failure BY SIGNATURE - see `ExpertDispatch::fail`.
void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out);

/// Plan v0.3 P6: the pool for a verify window of `n_tok` tokens.  `x_f` is (n_tok, n_embd), `ids` (n_tok, k) and
/// `out` (n_tok * k, n_embd).  Each distinct missed expert is computed once for all the tokens routed to it;
/// resident experts' rows are zeroed (the GPU adds them).  Requires `host_res` (the token-graph residency).
void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out);

/// **THE HITS, LAUNCHED AFTER THE MISSES ARE STAGED AND BEFORE `post[l]`.**  Same shape as `PoolFn` and for the
/// same reason: `session_loop` owns the ORDER and this owns the work, so the loop needs to know nothing about
/// expert caches.  Returns immediately when there are no hits, which is every layer until the cache is warm.
///
/// It must run AFTER the host has copied the misses into `parts_dev` (it writes into the same buffer, on rows
/// the CPU zeroed) and BEFORE `post[l]` (which reads it).  Both are stream-ordered on the loop's own stream.
void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k);
/// The pool half of the same decision; see `ExpertDispatch::is_hit`.

/// **PHASE 2'S ONLY SOURCE: `experts.bin`, memory-mapped, no cache.**
///
/// `experts.bin` is 33,973,862,400 B and `BLOB` is 1,382,400, so it holds exactly `48 x 512 = 24,576` blobs and
/// the index is `layer * 512 + expert` with NO padding.  A mapping is therefore the whole implementation: the
/// blob pointer is base plus a multiply, and the page fault that follows is the read.
///
/// **AND THAT IS NOT AS SLOW AS IT SOUNDS, WHICH IS THE POINT.**  The expert set is 34 GB and this machine has
/// 64 GB of DDR5, so a warm OS page cache holds ALL of it - the second token onward is a DRAM read at the
/// measured 44.14 GB/s, not a disk read.  The first token's 663.6 MB comes off the disk and is the cold-path
/// number; `benches` should report the two separately rather than blending them.
///
/// IT IS NOT AN LRU, AN LFU OR A PREFETCHER.  Phase 3 replaces this object with those.  Phase 2 is hit rate
/// `h = 0` on purpose, so that the CPU path and the host round trip are exercised on every layer of every
/// token - which is the only way they get debugged before speed matters.
class FileExpertSource : public ExpertSource {
public:
    FileExpertSource() = default;
    ~FileExpertSource() override;
    FileExpertSource(const FileExpertSource&) = delete;
    FileExpertSource& operator=(const FileExpertSource&) = delete;

    /// Maps `<pack_dir>/experts.bin` and checks its size against the loaded expert layout.  Canonical packs use
    /// `n_layers * n_expert * BLOB`; native packs use their variable per-layer blob sizes and offsets.
    ///
    /// The size check is not a formality: a short file would fault at the END of a long sequence, and an
    /// over-long one means the pack is not the one the geometry came from.  Refuses with the two numbers.
    ///
    /// CS-T: a native pack WITHOUT experts.bin, after `set_gguf`, maps the model's GGUF shards instead (every file
    /// native_experts.txt names, after `check_experts_gguf`), and assembles a blob from its three role slices when
    /// it is asked for: the SSD tier, read in place through the OS file cache, with no 30-77 GB experts.bin copy.
    /// With experts.bin present nothing changes.
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err);
    /// CS-T: the --native shard (native_experts.txt names the other files beside it); see `open`.
    void set_gguf(const std::string& native) { gguf_ = native; }
    /// Whether the experts are read from the GGUF shards in place (no experts.bin).
    bool gguf_mode() const { return !role_ptr_.empty(); }
    /// Pin a compact host mirror of experts absent from a fully filled static GPU cache. The mmap remains open
    /// as a fallback for later cache reloads. This is opt-in because the complement may still be a large allocation.
    ///
    /// The resident RAM mode (`--resident-experts`, `--resident-cpu-experts`):
    ///   - `pin`: page-locked and mapped (`cudaHostAlloc`), so the prompt path copies it by DMA and the verify
    ///     window may read a share of the misses over PCIe; when the driver refuses, ordinary memory locked in the
    ///     working set instead.  `pin = false` is ordinary pageable memory (the ROCm arm: large pinned allocations
    ///     can fail there, and it is what the HIP measurements used).
    ///   - `lend_from_slot` >= 0: the GPU-cache slots from there to the end are the prompt path's lend region; their
    ///     experts are kept in RAM too, from the last slot down, as far as available memory minus headroom allows
    ///     (a lent slot's expert is streamed during the prompt and copied back after it).
    ///   - the rest (the experts no slot holds) must fit that budget, or nothing is allocated and this returns false.
    ///
    /// CS-T, `budget_bytes` > 0 (`--resident-budget-gib`): only as many of those experts as fit `budget_bytes`, taken
    /// in `rank` order (the expert profile: the hottest after the GPU cache's), are copied; the rest stay on the
    /// mapped files (the SSD tier).  No lend region then (a lent slot's expert is read from the files).
    /// Available memory is limited by both RAM and commit capacity on Windows.
    /// #467: `budget_bytes` = `kResidentWhatFits` sizes that path from available memory minus headroom and the #403
    /// margin - the soft --resident-experts mode's second try when the whole complement does not fit;
    /// false when not even one expert fits.  On Windows the mapped experts leave the working set before any reading.
    static constexpr uint64_t kResidentWhatFits = ~0ull;
    bool pin_cache_complement(
        const ExpertCache& cache, std::string& err, bool pin = true,
        const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs = {}, int64_t lend_from_slot = -1,
        uint64_t headroom_bytes = 8ull << 30, uint64_t budget_bytes = 0,
        const std::vector<std::pair<int32_t, int32_t>>* rank = nullptr);
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
    uint64_t pinned_bytes() const { return complement_pinned_ ? complement_pin_limit_ : 0; }
    uint64_t resident_bytes() const { return complement_bytes_; }
    bool complement_pinned() const { return complement_pinned_; }
    bool complement_ready() const { return complement_ready_; }
    uint64_t locked_bytes() const { return complement_locked_; }
    /// Lend-region slots whose experts the compact copy holds (the last ones of the cache).
    int64_t resident_lent_slots() const { return complement_lent_slots_; }

    // ---- the resident RAM mode and the adaptive tier.  A swap puts `in` (held here) into a GPU slot and evicts
    // `out` (held only by that slot).  Before the slot is overwritten the caller copies it back into an exchange
    // buffer and calls `stage_exchange`: `out` is then read from that buffer, and `in` still from here (the CPU
    // computes both until the swap lands).  Once the slot copy has landed, `commit_exchanges` moves `out` into
    // `in`'s place, so the copy keeps holding exactly the experts the GPU does not - with no read of the file.
    // STRATA_EXCHANGE_ROTATE=1 can instead transfer buffer ownership (uniform, fully pinned/mapped slots).
    /// Whether the compact copy holds `(layer, expert)`.
    bool has_resident(int64_t layer, int64_t expert) const;
    /// Host room for `n` evicted blobs (page-locked when possible).  Idempotent for the same or a smaller `n`.
    bool reserve_exchanges(int64_t n, std::string& err);
    int64_t exchange_capacity() const { return xstage_cap_; }
    /// The exchange buffers are page-locked (cudaHostAlloc): copies to and from them are asynchronous.
    bool exchange_pinned() const { return xstage_pinned_; }
    uint8_t* exchange_buffer(int64_t q) const;
    /// Requires `has_resident(layer, in)`, `!has_resident(layer, out)` and `exchange_buffer(q)` holding out's blob.
    bool stage_exchange(int64_t layer, int64_t in, int64_t out, int64_t q);
    /// --pipeline-windows: size the exchange table now, so a `stage_exchange` on the adaptive tier's thread never
    /// reallocates it under a concurrent `blob` (the pool reads it while windows are in flight).
    void prepare_overrides() { if (override_.empty()) override_.assign((size_t) blobs_, nullptr); }
    /// After the GPU copies of every staged swap have landed.  Returns how many exchanges were applied.
    int64_t commit_exchanges();
    /// `commit_exchanges` in two halves, for the asynchronous adaptive tier (--adapt-async): `commit_copies` moves
    /// every staged evicted blob into its `in`'s place in the copy (on another thread: safe once nothing computes `in`
    /// from RAM - it is resident on the GPU - while `out` is still read from its exchange buffer), then `commit_flip`
    /// (the caller's thread, between windows) points `out` there and drops the staging.  Returns how many were applied.
    void commit_copies();
    int64_t commit_flip();
    /// The compact copy's blob of `(layer, expert)`, or null; not counted as a read (any thread).
    const uint8_t* resident_blob(int64_t layer, int64_t expert) const;
    int64_t exchanges() const { return exchanges_; }
    bool exchange_rotation() const { return exchange_storage_.active(); }
    uint64_t rotated_exchanges() const { return exchange_storage_.exchanges(); }
    uint64_t avoided_exchange_copy_bytes() const { return exchange_storage_.avoided_bytes(); }
    /// With the compact copy ready: blobs read from the mapped file since (what the plain mmap mode may read from
    /// the SSD).  0 in a steady resident mode; lend-region experts that did not fit the RAM count here.
    int64_t file_reads() const { return file_reads_.load(std::memory_order_relaxed); }
    /// CS-T per-tier counters: blobs served from the RAM copy, and the bytes read from the mapped files (the blobs
    /// `file_reads` counts, plus the prompt path's copies of them).
    int64_t ram_reads() const { return ram_reads_.load(std::memory_order_relaxed); }
    uint64_t file_read_bytes() const { return file_read_bytes_.load(std::memory_order_relaxed); }
    /// Of those, the bytes `blob` read (decode windows, adaptive swaps; the rest are the prompt path's copies), and
    /// the time spent reading the files, summed over threads.
    uint64_t file_blob_bytes() const { return file_blob_bytes_.load(std::memory_order_relaxed); }
    double file_ms() const { return (double) file_us_.load(std::memory_order_relaxed) / 1000.0; }
    /// Threads `prefetch` reads the GGUF with (STRATA_FETCH_THREADS, default 8).
    void set_fetch_threads(int n) { fetch_threads_ = n < 1 ? 1 : n; }
    /// #286: read the experts straight from the drive (Windows: FILE_FLAG_NO_BUFFERING, overlapped; Linux: O_DIRECT and
    /// the kernel's asynchronous reads) instead of
    /// through the mapped files - the GGUF in place, or a pack's experts.bin - when the file cache could not keep them
    /// beside `ram_bytes` (the RAM budget), cached now or not (their mapped pages would land in the working set); see
    /// experts_unbuffered.  The mapped reads' page faults are one small request each, and the pages they bring in
    /// take the RAM the budget was sized for.  `why` says what decided.
    /// #577: `ram_bytes` counts up to every expert, and only the experts outside it are what the cache must keep.
    bool set_unbuffered(uint64_t ram_bytes, std::string& why);
    /// #577: the same decision once the RAM copy is built (pin_cache_complement), from the RAM it really holds and the
    /// expert bytes outside it; switches either way (startup only, nothing reading).  Returns whether unbuffered.
    bool recheck_unbuffered(std::string& why);
    bool unbuffered() const { return !direct_.empty(); }
    /// Every expert's bytes (n_layers x n_expert blobs).
    uint64_t expert_bytes() const;
    /// #286, unbuffered: assembles the blobs of these pairs ahead of the `blob` calls that will ask for them (the
    /// GPU cache's fill from the profile) - one batch of reads instead of one blob at a time.  At most 64 pairs.
    void prefetch_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n);
    /// Mapped reads: asks the OS for these pairs' blobs ahead (platform::advise_willneed).  False when unbuffered,
    /// when read-ahead is off, or before `open`.
    bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const override;

    const uint8_t* blob(int64_t layer, int64_t expert) override;
    /// Windows, opt-in with STRATA_FILE_RELEASE=1: trim the file mapping's full pages after their expert reaches
    /// VRAM. Never touches the resident complement or staging buffers. Disabled by default.
    uint64_t release(int64_t layer, int64_t expert) override;
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    bool pcie_layer(int64_t layer) const override;
    bool transient(int64_t layer, int64_t expert) const override;
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    /// The blob read from the model file(s) only, never from the RAM copy or an exchange buffer (STRATA_CHECK_SWAPS).
    bool copy_blob_from_file(int64_t layer, int64_t expert, uint8_t* dst) const {
        return base_ != nullptr && dst != nullptr && layer >= 0 && expert >= 0 && layer < n_layers_ && expert < n_expert_ &&
               copy_from_files(layer, expert, dst);
    }
    /// CS-T: advances the assembled blobs' age (see staged_blob).
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override;
    /// CS-T: the GGUF in place assembles the missed experts on `fetch_threads_` threads.
    void prefetch(int64_t layer, const int64_t* experts, int64_t n) override;
    /// CS-T: the GGUF in place asks the OS for the predicted experts' pages (PrefetchVirtualMemory on Windows,
    /// madvise(WILLNEED) elsewhere), skipping the RAM copy's.
    void warm(int64_t layer, const int64_t* experts, int64_t n) override;
    /// Not when the reads are unbuffered: the warmed pages would be read through the file cache, a second time.
    bool warms() const override { return !role_ptr_.empty() && direct_.empty(); }
    /// Of the blobs the file tier read for the decode, how many had been warmed for their layer beforehand.
    int64_t warmed_hits() const { return warm_hits_.load(std::memory_order_relaxed); }
    int64_t warmed() const { return warm_count_.load(std::memory_order_relaxed); }
    /// #286: blobs an unbuffered read could not deliver, read through the mapping instead (0 when all went direct).
    int64_t direct_fallbacks() const { return direct_fallbacks_.load(std::memory_order_relaxed); }

    /// Blobs touched, for the driver to report.  With `h = 0` this is `48 * k` per token and the number is only
    /// interesting once Phase 3 makes it not so.
    int64_t reads() const override { return reads_; }

private:
    const uint8_t* resident_blob(size_t index) const;
    const uint8_t* mapped_blob(int64_t layer, int64_t expert) const;
    /// The blob's bytes from the mapped file(s) - experts.bin, or the three GGUF role slices - into `dst`.
    bool copy_from_files(int64_t layer, int64_t expert, uint8_t* dst) const;
    bool open_gguf(std::string& err);
    const uint8_t* staged_blob(int64_t layer, int64_t expert);
    bool claim_stage(int64_t key, size_t& v, bool& fill);
    bool fill_stage(size_t v, int64_t layer, int64_t expert, uint8_t* dst);
    void publish_stage(size_t v, int64_t layer, bool ok, double us);
    struct Fill { size_t v; int64_t layer, e; uint8_t* dst; };
    /// The claimed buffers' blobs: one overlapped batch when unbuffered, else the fetch threads' mapped copies.
    void fill_many(const std::vector<Fill>& todo);
    /// #286: the blobs from the drive, unbuffered: every role's 4 KiB-aligned window is read at once (overlapped on
    /// Windows, io_submit on Linux) into this thread's aligned buffer, then copied into place.  False when a read fails.
    bool read_direct(const Fill* fills, size_t n) const;
    bool open_direct(std::string& why);
    std::vector<std::string> paths_;          ///< the mapped files, as maps_
    std::vector<void*> direct_;               ///< #286: per file, an unbuffered handle (Windows) or O_DIRECT fd (Linux)
    std::vector<int> role_file_;              ///< 3 x n_layers: index into maps_ / direct_
    /// blobs assembled in the stage buffers (`blob` hands those out): the GGUF in place, or any unbuffered source
    bool staged() const { return !role_ptr_.empty() || !direct_.empty(); }
    static constexpr uint64_t kNoComplement = detail::kNoCacheComplement;
    // ---- CS-T: the GGUF shards in place
    std::string gguf_;
    struct Map {
        const uint8_t* base = nullptr;
        uint64_t bytes = 0;
#if defined(_WIN32)
        void* file = nullptr;
        void* mapping = nullptr;
#else
        int fd = -1;
#endif
    };
    std::vector<Map> maps_;
    std::vector<const uint8_t*> role_ptr_;    ///< 3 x n_layers: gate / up / down of the layer's expert 0
    std::vector<uint64_t> role_bytes_;        ///< 3 x n_layers: bytes per expert of that role
    // the blobs assembled for `blob()`: a small pool of buffers, one per recent (layer, expert).  A buffer is
    // reused only once `kStageAge` layer changes have passed since its blob was last asked for, so a pointer holds
    // through the layer it was asked in and the next ones (the pool computes a layer's misses before the next).
    static constexpr uint64_t kStageAge = 3;
    std::mutex stage_mu_;
    std::vector<std::unique_ptr<uint8_t[]>> stage_buf_;
    std::vector<int64_t> stage_key_;
    std::vector<uint64_t> stage_epoch_, stage_used_;
    std::vector<char> stage_busy_;            ///< being filled (outside stage_mu_): never a victim
    std::condition_variable stage_cv_;
    int fetch_threads_ = 8;
    std::atomic<uint64_t> file_blob_bytes_{0}, file_us_{0};
    std::unique_ptr<std::atomic<uint32_t>[]> warm_stamp_;   ///< per (layer, expert): epoch_ + 1 when warmed
    std::atomic<int64_t> warm_hits_{0}, warm_count_{0};
    mutable std::atomic<int64_t> direct_fallbacks_{0};
    std::unordered_map<int64_t, size_t> stage_of_;
    uint64_t stage_blob_ = 0;
    uint64_t stage_seq_ = 0;
    uint64_t epoch_ = 0;
    int64_t last_layer_ = -1;
    bool stage_grew_ = false;
    std::atomic<int64_t> ram_reads_{0};
    std::atomic<uint64_t> file_read_bytes_{0};
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    uint64_t mapped_bytes_ = 0;
    std::vector<uint64_t> layer_offsets_, layer_blob_bytes_;
    void* complement_arena_ = nullptr;
    const uint8_t* complement_host_ = nullptr;
    const uint8_t* complement_device_ = nullptr;
    uint64_t complement_bytes_ = 0;
    std::vector<uint64_t> complement_offsets_;
    detail::ExchangeStorage exchange_storage_; // authoritative when active; original arenas still own memory
    bool complement_pinned_ = false;
    bool complement_partial_ = false;         ///< CS-T: only the first complement_pin_limit_ bytes are registered
    uint64_t complement_pin_limit_ = 0;
    uint64_t complement_lock_off_ = 0;        ///< the working-set lock covers [lock_off, lock_off + locked)
    bool complement_ready_ = false;
    uint64_t complement_locked_ = 0;          ///< bytes held in the working set (pin refused)
    int64_t complement_lent_slots_ = 0;
    std::vector<const uint8_t*> override_;    ///< staged exchanges: an evicted expert read from its exchange buffer
    struct Exchange { size_t in, out; int64_t q; uint64_t bytes; };
    std::vector<Exchange> staged_;
    uint8_t* xstage_ = nullptr;               ///< exchange buffers, `xstage_cap_ x xstage_blob_`
    bool xstage_pinned_ = false;
    int64_t xstage_cap_ = 0;
    uint64_t xstage_blob_ = 0;
    int64_t exchanges_ = 0;
    std::atomic<int64_t> file_reads_{0};
    int64_t reads_ = 0;
#if defined(_WIN32)
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// ================================ THE RESIDENT ARENA (R2.1) ================================
//
// **THE MMAP ABOVE IS THE REVIEW'S FINDING C1 AND IT IS WORTH 2.2x.**  Read the comment on `FileExpertSource`
// about the page cache holding all 34 GB: that is true of the expert file ALONE, and it is not what this engine
// does.  The PLE/n-gram shard is a 26.8 GB file that the engine also maps, so 34 GB of experts plus 26.8 GB of
// n-gram is 60.8 GB of mapped, file-backed pages on a 63 GB machine - and file-backed pages are exactly the ones
// the OS drops from the standby list when it wants memory.  The expert stream then re-faults from disk.
//
// Measured, this round: the pool runs at **~19 GB/s in the engine against 42.8 GB/s in `bench/micro/cpu_s2.cpp`
// on the same machine, reading the same 34 GB**.  The micro does `std::fread` into a heap arena and reads
// ordinary (anonymous, resident) memory; the engine reads `MapViewOfFile`.  That is the whole difference, and it
// is why this class exists.
//
// It reads `experts.bin` into a `PinnedArena` once at startup, so the expert stream comes from anonymous memory
// the OS has no cheaper reason to evict.  `PinnedArena` also tries `cudaHostRegister`, which the GPU needs for
// Phase 3's cache fills and the CPU/PCIe miss split - but registration is best-effort and reported, not assumed.
class ArenaExpertSource : public ExpertSource {
public:
    ArenaExpertSource() = default;
    ~ArenaExpertSource() override;
    ArenaExpertSource(const ArenaExpertSource&) = delete;
    ArenaExpertSource& operator=(const ArenaExpertSource&) = delete;

    /// Allocates and loads `<pack_dir>/experts.bin`.  Prints nothing; the caller reports `note()` and the load
    /// rate, because those are the two numbers that say whether the arena is the one that was asked for.
    bool open(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, int threads, std::string& err,
              uint64_t max_pinned_bytes = 0, const std::string& shared_arena_file = {});
    /// Plan v0.3 P6: a native pack without experts.bin takes its experts from the model's GGUF: `native` is the
    /// --native shard, and native_experts.txt names the other shards beside it (per layer, or per role in v4).
    void set_gguf(const std::string& native) { gguf_ = native; }
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    int64_t reads() const { return reads_; }
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    void prefetch(int64_t layer, int64_t expert) override;
    uint64_t release(int64_t layer, int64_t expert) override;

    /// What backing was obtained and why, for the startup print.  "The engine adapts to the machine it is on" is
    /// only true if the engine says what it got.
    const std::string& note() const { return note_; }
    /// #633: set by `open` when the RAM available is less than the arena needs (a recommendation; it still loads)
    const std::string& ram_warning() const { return ram_warning_; }
    double load_gib_per_second() const { return gib_per_s_; }
    // Loader fix: the load, split.  `load_seconds()` is the wall clock of the load loop; the other two are
    // sums over the reader threads (see LoadStats), so on their own they say how much of that wall was spent
    // waiting for the disk and how much in memcpy + FNV-1a.
    double load_seconds() const { return load_seconds_; }
    double load_read_seconds() const { return load_read_s_; }
    double load_copy_seconds() const { return load_copy_s_; }

private:
    void* arena_ = nullptr;          ///< the PinnedArena, owned
    void* map_ = nullptr;            ///< STRATA_ARENA_MMAP: the arena file, mapped read-only (not the PinnedArena)
    uint64_t map_bytes_ = 0;
    std::vector<const uint8_t*> dev_slice_;   ///< device alias of each registered slice (or of the whole range)
    uint64_t slice_bytes_ = 0;
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_expert_ = 0;
    int64_t reads_ = 0;
    std::string note_;
    std::string ram_warning_;
    double gib_per_s_ = 0.0;
    double load_seconds_ = 0.0;
    double load_read_s_ = 0.0;
    double load_copy_s_ = 0.0;
    uint64_t pinned_bytes_ = 0;
    std::string gguf_;
};

/// Plan v0.3 P6: checks native_experts.txt's GGUF spans against the files, before anything is read: each layer's
/// gate/up/down at its recorded (file, offset) must be that tensor (`blk.L.ffn_<role>_exps.weight`), of the
/// layout's type and dimensions, and inside the file.  `native` is the --native shard (see set_gguf).
bool check_experts_gguf(const std::string& native, const strata::kernels::cpu::ExpertLayout& lay, std::string& err);
/// Fills `dst` (lay.total bytes, the experts.bin layout) from the GGUF files, one role at a time.
/// `unbuffered`: each chunk read past the file cache (Windows); `ready`: layer l is written only once
/// *ready > l + 1 (an arena that is still being registered).
LoadStats load_experts_gguf(const std::string& native, uint8_t* dst, const strata::kernels::cpu::ExpertLayout& lay,
                            int threads, bool unbuffered = false, const std::atomic<int>* ready = nullptr);

}  // namespace strata::core
