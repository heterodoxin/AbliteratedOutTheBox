// launch_cost: host time per submission on an in-order queue (empty-ish kernel), with and without queue profiling, and with
// an event per submission.  If the prompt path's ~6 submissions per expert cost this much each, the 25k experts of a 4K
// prompt are launch-bound.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

template <class F>
static double per_call_us(sycl::queue& q, int n, F&& f) {
    for (int i = 0; i < 100; ++i) f(i);
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) f(i);
    const double submit_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
    q.wait();
    const double total_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
    std::printf("    host submit %.2f us/call, until done %.2f us/call\n", submit_us, total_us);
    return submit_us;
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 20000;
    for (int prof = 0; prof < 2; ++prof) {
        sycl::queue q = prof ? sycl::queue{sycl::gpu_selector_v, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}}
                             : sycl::queue{sycl::gpu_selector_v, {sycl::property::queue::in_order()}};
        float* d = sycl::malloc_device<float>(1 << 20, q);
        std::printf("queue profiling %s\n", prof ? "on" : "off");
        std::printf("  tiny kernel, 1 work-group:\n");
        per_call_us(q, n, [&](int) { q.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> it) { d[it.get_global_id(0)] = 1.f; }); });
        std::printf("  kernel of 4096 work-groups (1M items):\n");
        per_call_us(q, n / 4, [&](int) { q.parallel_for(sycl::nd_range<1>(1 << 20, 256), [=](sycl::nd_item<1> it) { d[it.get_global_id(0)] += 1.f; }); });
        std::printf("  tiny kernel, event kept:\n");
        sycl::event e;
        per_call_us(q, n, [&](int) { e = q.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> it) { d[it.get_global_id(0)] = 1.f; }); });
        std::printf("  submit_barrier(event) + tiny kernel:\n");
        per_call_us(q, n, [&](int) {
            q.ext_oneapi_submit_barrier({e});
            e = q.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> it) { d[it.get_global_id(0)] = 1.f; });
        });
        sycl::free(d, q);
    }
    return 0;
}
