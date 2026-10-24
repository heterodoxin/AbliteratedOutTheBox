// include/strata/sycl_doorbell.hpp - the SYCL port: host<->device flag traffic that must bypass the GPU's caches.
//
// Strata's decode loop is a handshake through host-mapped memory: the GPU rings a sequence number the host polls,
// and a one-thread kernel spins on a flag the host writes. In CUDA those are `volatile` loads and stores, which
// nvcc turns into cache-bypassing accesses. A `volatile` in SYCL device code carries no such meaning on Intel
// GPUs: the spin read its first value from L3 forever (measured: the GPU at 100% and the host seeing no ring).
// Atomic loads and stores with system scope are the accesses that go to memory, so every side of the handshake
// goes through these two.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <string>
#include "strata/sycl_queue.hpp"

namespace strata {
// STRATA_SYCL_A770_FAST=1 (opt-in, experimental, PR #1710): the A770 router and 64-bit paired payload transfers.
inline bool a770_fast() {
    static const bool on = [] { const char* v = std::getenv("STRATA_SYCL_A770_FAST"); return v != nullptr && v[0] == '1'; }();
    return on;
}
using sys_atomic_u32 = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system>;

// Payload pairs use 64-bit transactions; flags remain 32-bit. Keep the same
// cache-bypass contract while transferring two floats per PCIe transaction.
template<class T> inline T sys_load_mapped(const T* p) {
    using read_props = decltype(sycl::ext::oneapi::experimental::properties(
        sycl::ext::intel::experimental::read_hint<sycl::ext::intel::experimental::cache_control<
            sycl::ext::intel::experimental::cache_mode::uncached,
            sycl::ext::oneapi::experimental::cache_level::L1,
            sycl::ext::oneapi::experimental::cache_level::L3>>));
    sycl::ext::oneapi::experimental::annotated_ptr<T, read_props> u(const_cast<T*>(p));
    return u[0];
}
inline sycl::float4 load_mapped_float4(const sycl::float4* p) {
    const uint64_t* words = reinterpret_cast<const uint64_t*>(p);
    const auto lo = sycl::bit_cast<sycl::float2>(sys_load_mapped(words));
    const auto hi = sycl::bit_cast<sycl::float2>(sys_load_mapped(words + 1));
    return sycl::float4(lo.x(), lo.y(), hi.x(), hi.y());
}

// A system-scope atomic load of host USM is still served from the GPU's cache on an Arc Pro B60 (NEO 26.31,
// oneAPI 2026.1.1): a device spin never sees the host's store and runs to the spin bound every time (bounded host/GPU
// ping-pong: 263 ms per round trip, 168 of 200 waits hit the bound). An explicit uncached L1+L3 read hint goes to
// memory every time: 2.4 us per round trip, 0 of 200. The acquire fence keeps the compiler from hoisting the load
// out of a spin loop. -DSTRATA_DOORBELL_ATOMIC_LOAD restores the atomic load.
#ifndef STRATA_DOORBELL_ATOMIC_LOAD
using doorbell_uncached_read = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::read_hint<sycl::ext::intel::experimental::cache_control<
        sycl::ext::intel::experimental::cache_mode::uncached,
        sycl::ext::oneapi::experimental::cache_level::L1, sycl::ext::oneapi::experimental::cache_level::L3>>));
inline uint32_t sys_load(const volatile uint32_t* p) {
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    sycl::ext::oneapi::experimental::annotated_ptr<uint32_t, doorbell_uncached_read> u(const_cast<uint32_t*>(p));
    return u[0];
}
#else
inline uint32_t sys_load(const volatile uint32_t* p) {
    return sys_atomic_u32(*const_cast<uint32_t*>(p)).load();
}
#endif
// The store side of the handshake (device -> host) needs the same hint. A system-scope atomic store, even followed by
// a release fence, is served from the GPU's cache on an Arc (xe) and reaches the host only when the kernel ends
// (measured on an A770: 482 ms into a 482 ms kernel), so a host that polls the ring mid-window never sees it; that is
// why the decode window had to wait for the whole graph (STRATA_VERIFY_NO_HOST) and every expert had to be
// device-resident. The uncached L1+L3 write hint reaches memory at once (44 ms, the launch latency).
// -DSTRATA_DOORBELL_ATOMIC_STORE restores the atomic store.
#ifndef STRATA_DOORBELL_ATOMIC_STORE
using doorbell_uncached_write = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::write_hint<sycl::ext::intel::experimental::cache_control<
        sycl::ext::intel::experimental::cache_mode::uncached,
        sycl::ext::oneapi::experimental::cache_level::L1, sycl::ext::oneapi::experimental::cache_level::L3>>));
inline void sys_store(volatile uint32_t* p, uint32_t v) {
    sycl::ext::oneapi::experimental::annotated_ptr<uint32_t, doorbell_uncached_write> u(const_cast<uint32_t*>(p));
    u[0] = v;
    // Ordering, not visibility: the value is already in memory. The release fence keeps a later payload store (the
    // experts' ids the host reads after the ring) after the ring, the same contract the atomic store had.
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}
// The doorbell's PAYLOAD (activations, expert ids, routing weights) is the other half of the handshake: a plain store
// to host USM reaches the host only when the kernel ends on an Arc (xe), so the host would spin on the ring and then
// read a stale payload. Same uncached L1+L3 hint as sys_store, for float/int32.
template<class T> inline void sys_store_mapped(T* p, T v) {
    sycl::ext::oneapi::experimental::annotated_ptr<T, doorbell_uncached_write> u(p);
    u[0] = v;
}
#else
inline void sys_store(volatile uint32_t* p, uint32_t v) {
    sys_atomic_u32(*const_cast<uint32_t*>(p)).store(v);
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}
template<class T> inline void sys_store_mapped(T* p, T v) {
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system>(*p).store(v);
}
#endif
// plain = true: an ordinary store. On the i915 driver (Arc A750) a plain store to host USM reaches the host in time
// and the uncached hint only slows the publish (decode -11% on an A750, 4K prompt, 5 rounds); on xe it must be hinted.
// STRATA_DOORBELL_PLAIN=0/1 overrides. The launchers read this on the host and pass it to the kernel.
template<class T> inline void sys_store_mapped(bool plain, T* p, T v) {
    if (plain) *p = v; else sys_store_mapped(p, v);
}
inline bool doorbell_plain_payload() {
    static const bool v = [] {
        if (const char* e = std::getenv("STRATA_DOORBELL_PLAIN")) return e[0] == '1';
        return intel_gpu_driver() == "i915";
    }();
    return v;
}

// Every device spin is bounded. An unbounded spin that never sees its flag is not a hang of one process: the
// xe driver times the queue out, resets the GT node by node (a window graph has 2,366 of them), and the card
// stays wedged until a reboot - measured twice. With a bound the failure is a wrong window instead, which the
// verifier's checks catch. ~2 M host-memory reads is a few seconds at PCIe latency.
// The bound is per device, chosen at run time (spin_max below): 20,000 reads on a card under the xe driver (the B-series
// on Linux), whose device-plan windows only spin in a failure and where a longer spin is what makes the driver reset the
// GT; 2,000,000 (a few seconds) elsewhere. A B-series card with no sysfs driver (Windows / OpenCL) gets 20,000 too (#1473,
// measured +46-67% with --mtp), found from the device's architecture or name. On an Arc A-series (i915) the CPU computes the experts the card does not hold and the GPU waits for it at
// every layer: 20,000 reads is a few tens of milliseconds, the first request after a start (cold pages, slow CPU
// layers) is slower than that, the GPU gave up, went on with the experts' outputs missing, and the answer was token 0
// ("!!!!!") or the engine crashed in the CPU pool on the garbage routing it read next.
// STRATA_SPIN_MAX=<reads> overrides both; a build's -DSTRATA_SYCL_SPIN_MAX=<reads> (CMake) fixes one bound for every
// device. The device functions take the bound as an argument: the launchers pass spin_max(queue).
// Without a sysfs driver name (Windows, OpenCL) a B-series card is recognised from the device itself (intel_gpu_gen):
// its window waits are the xe kind (expected to expire, so 20,000 reads), EXCEPT in a run that drafts without the MTP
// layer (--spec N, no --mtp): the suffix drafter's own waits have to complete there, and 20,000 reads left it empty-handed
// (0 drafts, 21.8 vs 68.8 tok/s, #1473). generate calls spin_drafts_need_long_bound(true) for such a run, before any
// launch. A Linux xe card keeps 20,000 in every mode (unchanged).
inline std::atomic<bool>& spin_long_drafts_flag() { static std::atomic<bool> f{false}; return f; }
inline void spin_drafts_need_long_bound(bool on) { spin_long_drafts_flag() = on; }
inline uint32_t spin_max(const sycl::queue& q) {
    static std::mutex mu;
    static std::unordered_map<sycl::device, uint32_t> cache;
    const sycl::device d = q.get_device();
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(d);
    if (it != cache.end()) return it->second;
    uint32_t v;
    if (const char* e = std::getenv("STRATA_SPIN_MAX"); e && std::atol(e) > 0) v = (uint32_t) std::atol(e);
#ifdef STRATA_SYCL_SPIN_MAX
    else v = STRATA_SYCL_SPIN_MAX;
#else
    else if (intel_gpu_driver() == "xe") v = 20000u;
    else if (intel_gpu_driver().empty() && intel_gpu_gen(d) == IntelGpuGen::Battlemage)
        v = spin_long_drafts_flag().load() ? 2000000u : 20000u;
    else v = 2000000u;
#endif
    cache.emplace(d, v);
    return v;
}
}  // namespace strata
