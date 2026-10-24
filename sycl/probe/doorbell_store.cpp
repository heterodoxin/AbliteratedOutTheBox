// doorbell_store: does the host see a device's store to host USM while the kernel is still running?  This is the
// ring publish of Strata's decode handshake (the GPU rings a sequence number the host polls, and the host answers a
// flag the GPU spins on).  A `volatile`/atomic store is served from the GPU's cache on an Arc (xe) and reaches the
// host only when the kernel ends; the uncached L1+L3 write hint (strata::sys_store, sycl_doorbell.hpp) reaches memory
// at once.  Build twice and compare:
//   icpx -fsycl -I sycl/include sycl/probe/doorbell_store.cpp -o /tmp/db        # uncached write (the fix)
//   icpx -fsycl -DSTRATA_DOORBELL_ATOMIC_STORE -I sycl/include ... -o /tmp/db_atomic   # old atomic store
// On an Arc A770 (xe) the first prints "SAW ... at ~0.2 s" with the kernel running to ~2 s; the second prints
// "NEVER SAW" and the value appears only at the kernel's end.  Exit 0 when the store was seen mid-kernel.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <chrono>
#include "strata/sycl_doorbell.hpp"

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    uint32_t* flag = (uint32_t*) sycl::malloc_host(sizeof(uint32_t), q);
    *flag = 0;
    const auto t0 = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            for (volatile long i = 0; i < 1000000; i++) {}      // ~0.2 s before the publish
            strata::sys_store(flag, 12345u);
            for (volatile long i = 0; i < 10000000; i++) {}     // ~2 s more running after the publish
        });
    });
    uint32_t v = 0;
    double seen_ms = -1.0;
    while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(6)) {
        v = *flag;
        if (v == 12345u) { seen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); break; }
    }
    q.wait();
    const double end_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
#ifdef STRATA_DOORBELL_ATOMIC_STORE
    const char* label = "ATOMIC_STORE ";
#else
    const char* label = "UNCACHED_WRITE";
#endif
    std::printf("%s: host %s flag=%u at %.1f ms; kernel ended at %.1f ms\n", label,
                seen_ms >= 0 ? "SAW" : "NEVER SAW", v, seen_ms, end_ms);
    return (seen_ms >= 0 && seen_ms < end_ms - 100.0) ? 0 : 3;   // seen well before the kernel ended
}
