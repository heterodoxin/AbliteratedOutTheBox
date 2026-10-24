// Test paired mapped transfers, odd tails, unaligned scalar fallback, and
// the checksum contracts of all three publish variants over repeated rings.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/elementwise.hpp"
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    auto& q = *strata::q_of(nullptr);
    using namespace strata::kernels;
    constexpr int capacity = 2560, k = 10, experts = 64;
    auto* hx = (float*)strata::host_malloc_polled((capacity + 4)*sizeof(float), q);
    auto* hi = (int32_t*)strata::host_malloc_polled(k*sizeof(int32_t), q);
    auto* hw = (float*)strata::host_malloc_polled(k*sizeof(float), q);
    auto* seq = (uint32_t*)strata::host_malloc_polled(64, q);
    auto* dx = sycl::malloc_device<float>(capacity, q);
    auto* di = sycl::malloc_device<int32_t>(k, q);
    auto* dw = sycl::malloc_device<float>(k, q);
    auto* dr = sycl::malloc_device<int32_t>(experts, q);
    std::vector<float> x(capacity), weights(k), copied(capacity);
    std::vector<int32_t> ids(k);
    for (int j = 0; j < capacity; ++j) x[j] = (j % 123 - 61)*0.25f;
    for (int j = 0; j < k; ++j) { ids[j] = j*3; weights[j] = (j + 1)*0.03125f; }
    q.memcpy(dx, x.data(), x.size()*sizeof(float));
    q.memcpy(di, ids.data(), k*sizeof(int32_t));
    q.memcpy(dw, weights.data(), k*sizeof(float));
    q.memset(dr, 0, experts*sizeof(int32_t));
    q.wait_and_throw();
    int failed = 0;
    for (int n : {1, 3, 17, 2559, 2560})
        for (int offset : {0, 1})
            for (int variant = 0; variant < 3; ++variant) {
                std::memset(seq, 0, 64);
                for (uint32_t ring = 1; ring <= 12; ++ring) {
                    std::memset(hx, 0x7f, (capacity + 4)*sizeof(float));
                    std::memset(hi, 0, k*sizeof(int32_t));
                    std::memset(hw, 0, k*sizeof(float));
                    if (variant == 0) doorbell_publish(dx, di, dw, n, k, hx+offset, hi, hw, seq, &q);
                    else if (variant == 1) doorbell_publish_res(dx, di, dr, experts, n, k, hx+offset, hi, seq, &q);
                    else doorbell_publish_value(dx, di, dw, n, k, hx+offset, hi, hw, seq, ring, &q);
                    q.wait_and_throw();
                    bool ok = seq[0] == ring && !std::memcmp(hx+offset, x.data(), n*sizeof(float)) &&
                        !std::memcmp(hi, ids.data(), k*sizeof(int32_t)) &&
                        doorbell_payload_ready(seq, hx+offset, n, hi, hw, k, ring);
                    if (variant != 1) ok = ok && !std::memcmp(hw, weights.data(), k*sizeof(float));
                    if (!ok) { ++failed; std::printf("FAIL n=%d offset=%d variant=%d ring=%u\n", n, offset, variant, ring); }
                }
            }
    std::memcpy(hx, x.data(), x.size()*sizeof(float));
    copy_from_mapped(dx, hx, capacity, &q);
    q.memcpy(copied.data(), dx, capacity*sizeof(float)).wait();
    if (std::memcmp(copied.data(), x.data(), capacity*sizeof(float))) ++failed;
    for (void* p : {static_cast<void*>(hx), static_cast<void*>(hi), static_cast<void*>(hw), static_cast<void*>(seq)})
        strata::host_free_polled(p, q);
    for (void* p : {static_cast<void*>(dx), static_cast<void*>(di), static_cast<void*>(dw), static_cast<void*>(dr)}) sycl::free(p, q);
    std::printf("paired payload and mapped read: %d failures\n", failed);
    return failed != 0;
}
