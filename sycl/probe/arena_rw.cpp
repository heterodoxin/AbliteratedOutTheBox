// arena_rw: does a big device arena keep what is copied into it?  The engine's expert-cache fill (pinned host batches of
// 64 blobs, queue.memcpy into odd-sized slots, the arena zeroed by chunked kernels first) read back with 2 MiB pages of the
// arena holding wrong bytes on an Arc Pro B70 (xe).  This reproduces the shape without the engine.
//   arena_rw GIB [slot_bytes=1971200] [batch=64] [zero=1] [verify_after_fill=1]
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "strata/sycl_queue.hpp"

static inline uint8_t pat(uint64_t off) {
    uint64_t x = off * 0x9E3779B97F4A7C15ull;
    x ^= x >> 29;
    return (uint8_t) (x >> 11);
}

int main(int argc, char** argv) {
    const double gib = argc > 1 ? std::atof(argv[1]) : 4.0;
    const size_t slot = argc > 2 ? (size_t) std::atoll(argv[2]) : 1971200;
    const size_t batch = argc > 3 ? (size_t) std::atoll(argv[3]) : 64;
    const int zero = argc > 4 ? std::atoi(argv[4]) : 1;
    const size_t total = (size_t) (gib * 1073741824.0);
    const size_t n_slots = total / ((slot + 255) / 256 * 256);
    const size_t stride = (slot + 255) / 256 * 256;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    uint8_t* arena = (uint8_t*) sycl::malloc_device(n_slots * stride + (2 << 20), q);
    if (!arena) { std::printf("alloc failed\n"); return 1; }
    arena = (uint8_t*) (((uintptr_t) arena + (2 << 20) - 1) & ~(uintptr_t) ((2 << 20) - 1));   // as the engine's: 2 MiB aligned
    std::printf("arena %p, %zu slots of %zu (stride %zu)\n", (void*) arena, n_slots, slot, stride);
    if (zero) strata::big_fill_zero(q, arena, n_slots * stride);
    uint8_t* pin = sycl::malloc_host<uint8_t>(2 * batch * stride, q);
    sycl::event done[2];
    size_t k = 0;
    for (size_t b0 = 0; b0 < n_slots; b0 += batch, ++k) {
        const size_t n = std::min(batch, n_slots - b0);
        uint8_t* set = pin + (k % 2) * batch * stride;
        done[k % 2].wait();
        for (size_t j = 0; j < n; ++j)
            for (size_t x = 0; x < slot; ++x) set[j * stride + x] = pat((b0 + j) * stride + x);
        for (size_t j = 0; j < n; ++j) done[k % 2] = q.memcpy(arena + (b0 + j) * stride, set + j * stride, slot);
    }
    q.wait();
    std::vector<uint8_t> got(stride);
    size_t bad = 0;
    for (size_t i = 0; i < n_slots; ++i) {
        q.memcpy(got.data(), arena + i * stride, slot).wait();
        size_t nd = 0, fd = slot;
        for (size_t x = 0; x < slot; ++x)
            if (got[x] != pat(i * stride + x)) { ++nd; if (fd == slot) fd = x; }
        if (nd) {
            ++bad;
            if (bad <= 10)
                std::printf("slot %zu at %p (offset %zu MiB): %zu of %zu bytes differ, first at %zu\n", i, (void*) (arena + i * stride),
                            (i * stride) >> 20, nd, slot, fd);
        }
    }
    std::printf("%zu of %zu slots differ\n", bad, n_slots);
    return bad ? 3 : 0;
}
