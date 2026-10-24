// alias_scan: allocate device buffers of assorted sizes, keep them, and run strata::arena_alias_check on each: where does
// the xe page table alias pages (a write to one 2 MiB page showing at another)?  Prints each buffer's address mod 1 GiB.
//   alias_scan [n=40] [seed=1] [max_mib=6000]
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "strata/sycl_queue.hpp"

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 40;
    const unsigned seed = argc > 2 ? (unsigned) std::atoi(argv[2]) : 1;
    const size_t max_mib = argc > 3 ? (size_t) std::atoll(argv[3]) : 6000;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::mt19937_64 rng(seed);
    std::vector<void*> keep;
    size_t total = 0;
    int bad_bufs = 0;
    for (int i = 0; i < n; ++i) {
        size_t mib = (rng() % 4 == 0) ? 1 + rng() % 8 : 1 + rng() % max_mib;
        if (total + mib > 24000) break;
        const size_t bytes = mib << 20;
        void* p = sycl::malloc_device(bytes, q);
        if (!p) { std::printf("alloc %zu MiB failed\n", mib); break; }
        keep.push_back(p);
        total += mib;
        size_t first = 0;
        const uint64_t bad = strata::arena_alias_check(q, p, bytes, &first);
        std::printf("#%02d %6zu MiB at %p (mod 1GiB %5zu MiB, mod 2MiB %zu): %s", i, mib, p, (size_t) (((uintptr_t) p & ((1u << 30) - 1)) >> 20),
                    (size_t) ((uintptr_t) p & ((2u << 20) - 1)), bad ? "ALIAS" : "ok");
        if (bad) { ++bad_bufs; std::printf(" %llu blocks, first at +%zu MiB", (unsigned long long) bad, first >> 20); }
        std::printf("\n");
    }
    std::printf("%d of %zu buffers aliased\n", bad_bufs, keep.size());
    for (void* p : keep) sycl::free(p, q);
    return bad_bufs ? 3 : 0;
}
