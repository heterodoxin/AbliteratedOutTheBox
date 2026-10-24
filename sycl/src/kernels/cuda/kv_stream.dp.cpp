// src/kernels/cuda/kv_stream.cu - see include/strata/kernels/kv_stream.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace strata::kernels {
namespace {

void check(const char* what) {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}

constexpr int RT = 1024;   // the resolve block

// The per-block byte runs of the (up to four) pool arrays: block b of array i is bytes [b * len, (b + 1) * len).
struct Runs {
    const uint8_t* src[4];
    uint8_t* dst[4];
    int len[4];
    int n;
};

Runs runs_of(const QsaAttnPools& slots, const KvHostPools& host, int fmt, const QsaShapes& s) {
    const int rows = (int) (s.n_head_kv * s.page_size);
    Runs r{};
    if (fmt == kKvHybrid) {   // K8V4: int8 K codes, their scales, rotated q4_0 V
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        const int v = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.k_scale; r.dst[1] = (uint8_t*) slots.k_scale; r.len[1] = scales;
        r.src[2] = (const uint8_t*) host.v_q4;    r.dst[2] = (uint8_t*) slots.v_q4;    r.len[2] = v;
        r.n = 3;
    } else if (fmt == kKvQ4) {
        const int bytes = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q4; r.dst[0] = (uint8_t*) slots.k_q4; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_q4; r.dst[1] = (uint8_t*) slots.v_q4; r.len[1] = bytes;
        r.n = 2;
    } else if (fmt == kKvInt8) {
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.v_q;     r.dst[1] = (uint8_t*) slots.v_q;     r.len[1] = codes;
        r.src[2] = (const uint8_t*) host.k_scale; r.dst[2] = (uint8_t*) slots.k_scale; r.len[2] = scales;
        r.src[3] = (const uint8_t*) host.v_scale; r.dst[3] = (uint8_t*) slots.v_scale; r.len[3] = scales;
        r.n = 4;
    } else {
        const int bytes = rows * (int) s.head_dim * 2;
        r.src[0] = (const uint8_t*) host.k_pool; r.dst[0] = (uint8_t*) slots.k_pool; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_pool; r.dst[1] = (uint8_t*) slots.v_pool; r.len[1] = bytes;
        r.n = 2;
    }
    return r;
}

// Block-wide exclusive prefix sum of one int per thread (RT threads); `total` is the sum over the block.
inline int block_scan(int v, int *warp_sums, int &total) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31,
              w = item_ct1.get_local_id(2) >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        /*
        DPCT1108: '__shfl_up_sync' was migrated with the experimental feature
        masked sub_group function which may not be supported by all compilers or
        runtimes. You may need to adjust the code.
        */
        const int y = dpct::experimental::shift_sub_group_right(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[w] = x;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (w == 0) {
        int t = warp_sums[lane];
        for (int o = 1; o < 32; o <<= 1) {
            /*
            DPCT1108: '__shfl_up_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int y = dpct::experimental::shift_sub_group_right(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                t, o);
            if (lane >= o) t += y;
        }
        warp_sums[lane] = t;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    total = warp_sums[31];
    const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    return excl;
}

__dpct_inline__ void resolve_kernel(KvStreamMap m,
                                    const int32_t *__restrict__ ids,
                                    const int32_t *__restrict__ steps, int n_q,
                                    int cap, int page_size) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &s_nmiss = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_lookups = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_cut = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &warp_sums =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<int[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int epoch = m.ctl[0] + 1;
    if (item_ct1.get_local_id(2) == 0) { s_nmiss = 0; s_lookups = 0; }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // 1. hits take this epoch and their reference bit; a missing block is claimed exactly once (-1 -> -2)
    int lookups = 0;
    for (int q = 0; q < n_q; ++q) {
        const int width = steps[q * kStepCount + kStepWidth];
        const int32_t* qi = ids + (long long) q * cap;
        for (int i = item_ct1.get_local_id(2); i < width; i += RT) {
            const int b = qi[i] / page_size;
            if (i > 0 && qi[i - 1] / page_size == b) continue;   // ids are ascending: one lookup per block
            ++lookups;
            const int sl = m.page_table[b];
            if (sl >= 0) {
                m.slot_stamp[sl] = epoch;
                m.slot_ref[sl] = 1;
            } else if (sl == -1 &&
                       dpct::atomic_compare_exchange_strong<
                           sycl::access::address_space::generic_space>(
                           &m.page_table[b], -1, -2) == -1) {
                m.miss_block[dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(&s_nmiss, 1)] =
                    b;
            }
        }
    }
    if (lookups > 0)
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &s_lookups, lookups); // (#783, stuchapin909) most threads see none:
                                  // skip the shared atomic
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // 2. one victim per miss: a clock sweep from the hand. A slot this call uses (stamp == epoch) is never taken;
    //    a referenced one loses its bit as the hand passes it and is taken on the next pass.
    const int need = s_nmiss, n = (int) m.n_slots;
    int hand = m.ctl[1], got = 0;
    for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
        const int j = (int)(((long long)hand + item_ct1.get_local_id(2)) % n);
        const bool mine = m.slot_stamp[j] == epoch;
        const bool cand = !mine && (m.slot_block[j] < 0 || m.slot_ref[j] == 0);
        int total = 0;
        const int rank = block_scan(cand ? 1 : 0, warp_sums, total);
        const int want = need - got;
        if (item_ct1.get_local_id(2) == 0) s_cut = RT;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (cand && rank == want - 1) s_cut =
            item_ct1.get_local_id(2) +
            1; // the hand stops just past the last slot taken
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const int cut = s_cut;
        if (cand && rank < want) {
            m.miss_slot[got + rank] = j;
            m.slot_stamp[j] = epoch;   // taken: a sweep that wraps around must not take it twice
        } else if (item_ct1.get_local_id(2) < cut && !mine) {
            m.slot_ref[j] = 0;
        }
        got += total < want ? total : want;
        hand = (int) (((long long) hand + cut) % n);
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    // 3. re-point the table; the copy kernel fills the slots
    const int placed = got < need ? got : need;
    for (int k = item_ct1.get_local_id(2); k < need; k += RT) {
        const int b = m.miss_block[k];
        if (k >= placed) { m.page_table[b] = -1; continue; }   // overflow: never happens with a legal n_slots
        const int sl = m.miss_slot[k];
        const int old = m.slot_block[sl];
        if (old >= 0) m.page_table[old] = -1;
        m.slot_block[sl] = b;
        m.slot_stamp[sl] = epoch;
        m.slot_ref[sl] = 1;
        m.page_table[b] = sl;
    }
    if (item_ct1.get_local_id(2) == 0) {
        m.ctl[0] = epoch;
        m.ctl[1] = hand;
        m.ctl[2] = placed;
        if (placed < need) m.ctl[3] = 1;
        unsigned long long* c = reinterpret_cast<unsigned long long*>(m.ctl + 4);
        c[0] += (unsigned long long) placed;
        c[1] += (unsigned long long) s_lookups;
        c[2] += 1ull;
    }
}

// One block per missed block (grid-stride): copy its runs from the host copy into its slot, 16 B per thread.
__dpct_inline__ void copy_kernel(KvStreamMap m, Runs r) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int need = m.ctl[2];
    for (int k = item_ct1.get_group(2); k < need;
         k += item_ct1.get_group_range(2)) {
        const long long b = m.miss_block[k], sl = m.miss_slot[k];
        for (int a = 0; a < r.n; ++a) {
            const sycl::uint4 *src =
                reinterpret_cast<const sycl::uint4 *>(r.src[a] + b * r.len[a]);
            sycl::uint4 *dst =
                reinterpret_cast<sycl::uint4 *>(r.dst[a] + sl * r.len[a]);
#pragma unroll
            for (int i = item_ct1.get_local_id(2); i < r.len[a] / 16;
                 i += item_ct1.get_local_range(2)) dst[i] = src[i];
        }
    }
}

__dpct_inline__ void reset_kernel(KvStreamMap m) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i0 = (long long)item_ct1.get_group(2) *
                             item_ct1.get_local_range(2) +
                         item_ct1.get_local_id(2),
                    st = (long long)item_ct1.get_group_range(2) *
                         item_ct1.get_local_range(2);
#pragma unroll
    for (long long i = i0; i < m.n_blocks; i += st) m.page_table[i] = -1;
#pragma unroll
    for (long long i = i0; i < m.n_slots; i += st) {
        m.slot_block[i] = -1;
        m.slot_stamp[i] = -1;
        m.slot_ref[i] = 0;
    }
    if (i0 < kKvCtlInts) m.ctl[i0] = 0;
}

__dpct_inline__ void ring_kernel(int32_t *table, long long n_blocks,
                                 long long n_slots) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
#pragma unroll
    for (long long i =
             (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n_blocks; i += (long long)item_ct1.get_group_range(2) *
                            item_ct1.get_local_range(2))
        table[i] = (int32_t) (i % n_slots);
}

}  // namespace

uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvHybrid)
        return rows * (uint64_t) s.head_dim + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 +
               rows * kv_q4_bytes_per_head((int) s.head_dim);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                : rows * (uint64_t) s.head_dim * 2 * 2;
}

void kv_stream_reset(const KvStreamMap& m, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class reset_kernel_f02a82>>(
                sycl::nd_range<3>(sycl::range(1, 1, 128) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    reset_kernel(m);
                });
    }
    check("reset");
}

void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream) {
    if (n_q <= 0) return;
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0) {
        std::fprintf(stderr, "kv_stream: a block's scale run must be a multiple of 16 bytes\n");
        std::exit(1);
    }
    // one sweep step looks at RT consecutive slots `(hand + thread) % n_slots`; with fewer slots than RT two
    // threads see the same slot and may both take it for two different misses.  The engine never streams with fewer
    // than qsa_kv_resident_min() / page_size = 5,120 slots, so this is a guard, not a limit.
    if (m.n_slots < RT) {
        std::fprintf(stderr, "kv_stream: %lld slots is fewer than the resolve block (%d): the clock sweep would take a "
                             "slot twice\n", (long long) m.n_slots, RT);
        std::exit(1);
    }
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class resolve_kernel_148a81>>(
                sycl::nd_range<3>(sycl::range(1, 1, RT), sycl::range(1, 1, RT)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        resolve_kernel(m, ids, steps, (int)n_q, (int)cap,
                                       (int)s.page_size);
                    });
    }
    check("resolve");
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                auto runs_of_slots_host_fmt_s_ct1 =
                    runs_of(slots, host, fmt, s);

                cgh.parallel_for<dpct_kernel_name<class copy_kernel_4bb253>>(
                    sycl::nd_range<3>(sycl::range(1, 1, 96) *
                                          sycl::range(1, 1, 128),
                                      sycl::range(1, 1, 128)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        copy_kernel(m, runs_of_slots_host_fmt_s_ct1);
                    });
            });
    }
    check("copy");
}

void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class ring_kernel_f9df1d>>(
                sycl::nd_range<3>(sycl::range(1, 1, 64) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    ring_kernel(page_table, n_blocks, n_slots);
                });
    }
    check("ring table");
}

void kv_ring_restore(const QsaAttnPools &slots, const KvHostPools &host,
                     int fmt, int64_t b0, int64_t b1, int64_t n_slots,
                     const QsaShapes &s, void *stream) try {
    const Runs r = runs_of(slots, host, fmt, s);
    for (int64_t b = b0; b < b1;) {
        const int64_t sl = b % n_slots, run = std::min<int64_t>(b1 - b, n_slots - sl);   // up to the ring's end
        for (int a = 0; a < r.n; ++a)
            /*
            DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            if (DPCT_CHECK_ERROR(strata::q_of(stream)->memcpy(
                    r.dst[a] + sl * r.len[a], r.src[a] + b * r.len[a],
                    (size_t)(run * r.len[a]))) != 0)
                check("ring restore");
        b += run;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void kv_stage_from_host(const QsaAttnPools &stage, const KvHostPools &host,
                        int fmt, int64_t n_blocks, const QsaShapes &s,
                        void *stream) try {
    if (n_blocks <= 0) return;
    const Runs r = runs_of(stage, host, fmt, s);
    for (int a = 0; a < r.n; ++a)
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        if (DPCT_CHECK_ERROR(strata::q_of(stream)->memcpy(
                r.dst[a], r.src[a], (size_t)(n_blocks * r.len[a]))) != 0)
            check("stage");
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void kv_unstage_to_host(const QsaAttnPools &stage, const KvHostPools &host,
                        int fmt, int64_t b0, int64_t b1, const QsaShapes &s,
                        void *stream) try {
    if (b1 <= b0) return;
    const Runs r = runs_of(stage, host, fmt, s);   // src: the host copy, dst: the staging pool (identity layout both)
    for (int a = 0; a < r.n; ++a)
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        if (DPCT_CHECK_ERROR(strata::q_of(stream)->memcpy(
                (void *)(r.src[a] + b0 * r.len[a]), r.dst[a] + b0 * r.len[a],
                (size_t)((b1 - b0) * r.len[a]))) != 0)
            check("unstage");
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

KvStreamCounters kv_stream_counters(const KvStreamMap &m) try {
    int32_t c[kKvCtlInts] = {};
    KvStreamCounters r;
    if (m.ctl == nullptr ||
        DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memcpy(c, m.ctl, sizeof(c)).wait()) !=
            0) return r;
    const unsigned long long* u = reinterpret_cast<const unsigned long long*>(c + 4);
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];
    r.overflow = c[3] != 0;
    return r;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::kernels
