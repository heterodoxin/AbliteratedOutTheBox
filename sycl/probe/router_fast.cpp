// Compare all output bits against the existing router, including ties, NaN
// tails and the fast kernel's forced serial-sum fallback. Also time decode calls.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/router_top10.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int compare(int experts, int k, int rows) {
    auto& q = *strata::q_of(nullptr);
    std::mt19937 rng(17);
    std::normal_distribution<float> normal(0, 2);
    std::vector<float> logits((size_t)rows * experts);
    for (size_t i = 0; i < logits.size(); ++i) {
        const size_t r = i / experts;
        float x = normal(rng) * (r % 7 == 5 ? 40 : 1);
        if (r % 3 == 1) x = std::round(x * 4) / 4;
        if (r % 3 == 2) x = std::round(x * 64) / 64;
        if (r % 1000 == 999 && i % experts == 77) x = NAN;
        if (r % 1000 == 998) x = 1;
        logits[i] = x;
    }
    float* dl = sycl::malloc_device<float>(logits.size(), q);
    q.memcpy(dl, logits.data(), logits.size() * sizeof(float));
    const size_t n = (size_t)rows * k;
    float* dw[3]; int* di[3];
    std::vector<float> weights[3]; std::vector<int> ids[3];
    for (int v = 0; v < 3; ++v) {
        dw[v] = sycl::malloc_device<float>(n, q);
        di[v] = sycl::malloc_device<int>(n, q);
        q.memset(dw[v], 0x3c, n * sizeof(float));
        q.memset(di[v], 0x11, n * sizeof(int));
        if (!strata::kernels::router_top10_variant(dl, rows, experts, k, di[v], dw[v], &q, v)) return 2;
        weights[v].resize(n); ids[v].resize(n);
        q.memcpy(weights[v].data(), dw[v], n * sizeof(float));
        q.memcpy(ids[v].data(), di[v], n * sizeof(int));
        q.wait_and_throw();
    }
    int fails = 0;
    for (int v = 1; v < 3; ++v) {
        int bad = 0;
        for (int r = 0; r < rows; ++r)
            if (std::memcmp(ids[0].data() + (size_t)r*k, ids[v].data() + (size_t)r*k, k*sizeof(int)) ||
                std::memcmp(weights[0].data() + (size_t)r*k, weights[v].data() + (size_t)r*k, k*sizeof(float))) ++bad;
        std::printf("experts %d k %d rows %d variant %d: %d rows differ\n", experts, k, rows, v, bad);
        fails += bad != 0;
    }
    for (int v = 0; v < 2; ++v) {
        constexpr int reps = 300;
        auto start = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            strata::kernels::router_top10_variant(dl + (r % 64)*experts, 1, experts, k, di[v], dw[v], &q, v);
        q.wait_and_throw();
        auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-start).count()/reps;
        std::printf("variant %d single-token call: %.2f us (host submission + execution)\n", v, us);
    }
    for (int v = 0; v < 3; ++v) { sycl::free(dw[v], q); sycl::free(di[v], q); }
    sycl::free(dl, q);
    return fails;
}

int main() {
    int failures = compare(512, 10, 65536) + compare(256, 8, 8192) + compare(256, 10, 8192) + compare(512, 32, 1024);
    std::printf("FAILURES: %d\n", failures);
    return failures != 0;
}
