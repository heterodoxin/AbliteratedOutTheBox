// fp64_cost: what does FP64 (emulated on Arc Alchemist and Battlemage) cost in the engine's small reduction kernels?
// Same one-warp-per-row sum of squares + 1/sqrt as gdn_l2_norm / gdn_out_norm / the PLE rms norm, once with a double
// accumulator (the engine today) and once in float, plus a silu over n values the way silu_inplace does it.
//   fp64_cost [launches=4000]     (the A750 needs IGC_EnableDPEmulation=1 OverrideDefaultFP64Settings=1 for the double kernels)
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

template <class F>
static double us_per_launch(sycl::queue& q, int n, F&& f) {
    for (int i = 0; i < 50; ++i) f();
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) f();
    q.wait();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
}

template <class Acc>
static void l2norm(sycl::queue& q, float* x, int rows, int cols, float eps) {
    q.parallel_for(sycl::nd_range<1>(rows * 32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int row = it.get_group(0);
        float* p = x + (size_t) row * cols;
        auto sg = it.get_sub_group();
        Acc acc = 0;
        for (int i = it.get_local_id(0); i < cols; i += 32) acc += (Acc) p[i] * (Acc) p[i];
        for (int off = 16; off > 0; off >>= 1) acc += sycl::shift_group_left(sg, acc, off);
        acc = sycl::group_broadcast(sg, acc, 0);
        const float inv = (float) ((Acc) 1 / sycl::sqrt(acc + (Acc) eps));
        for (int i = it.get_local_id(0); i < cols; i += 32) p[i] *= inv;
    });
}

template <class Acc>
static void silu(sycl::queue& q, float* x, int n) {
    q.parallel_for(sycl::nd_range<1>((n + 255) / 256 * 256, 256), [=](sycl::nd_item<1> it) {
        const int i = it.get_global_id(0);
        if (i >= n) return;
        const Acc v = (Acc) x[i];
        x[i] = (float) (v / ((Acc) 1 + sycl::exp(-v)));
    });
}


// the sampler's tail: top_p cut and temperature draw over k kept logits, one warp (like sampled_tail_warp), 3 exps per element
template <class Acc>
static void sampler_tail(sycl::queue& q, const float* sel, int* out, int k, float top_p, float u) {
    q.parallel_for(sycl::nd_range<1>(32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int lane = it.get_local_id(0);
        Acc sum = 0;
        float mx = sel[0];
        for (int i = lane; i < k; i += 32) sum += sycl::exp((Acc) sel[i] - (Acc) mx);
        sum = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<Acc>());
        Acc cum = 0;
        int cut = k;
        if (lane == 0) {
            for (int i = 0; i < k; ++i) { cum += sycl::exp((Acc) sel[i] - (Acc) mx) / sum; if (cum >= (Acc) top_p) { cut = i + 1; break; } }
            Acc c2 = 0; int pick = cut - 1;
            for (int i = 0; i < cut; ++i) { c2 += sycl::exp((Acc) sel[i] - (Acc) mx) / sum; if ((Acc) u < c2) { pick = i; break; } }
            out[0] = pick;
        }
    });
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 4000;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("device %s, fp64 %s\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                q.get_device().has(sycl::aspect::fp64) ? "yes" : "NO (double kernels skipped)");
    const bool fp64 = q.get_device().has(sycl::aspect::fp64);
    float* x = sycl::malloc_device<float>(1 << 20, q);
    q.fill(x, 0.5f, 1 << 20).wait();
    struct Shape { const char* what; int rows, cols; };
    const Shape shapes[] = {{"gdn l2 norm (rows 48, cols 128)", 48, 128}, {"gdn out norm (rows 48, cols 128)", 48, 128},
                            {"ple rms (rows 8, cols 2560)", 8, 2560}, {"wide (rows 6, cols 2560)", 6, 2560}};
    for (const Shape& s : shapes) {
        const double f = us_per_launch(q, n, [&] { l2norm<float>(q, x, s.rows, s.cols, 1e-6f); });
        double d = -1;
        if (fp64) d = us_per_launch(q, n, [&] { l2norm<double>(q, x, s.rows, s.cols, 1e-6f); });
        std::printf("%-34s float %6.2f us   double %6.2f us\n", s.what, f, d);
    }
    for (int cnt : {2560, 10240, 61440}) {
        const double f = us_per_launch(q, n, [&] { silu<float>(q, x, cnt); });
        double d = -1;
        if (fp64) d = us_per_launch(q, n, [&] { silu<double>(q, x, cnt); });
        std::printf("silu over %-24d float %6.2f us   double %6.2f us\n", cnt, f, d);
    }
    {
        float* sel = sycl::malloc_device<float>(64, q);
        int* o = sycl::malloc_device<int>(1, q);
        std::vector<float> h(64);
        for (int i = 0; i < 64; ++i) h[i] = 10.f - 0.3f * i;
        q.memcpy(sel, h.data(), 64 * 4).wait();
        for (int k : {20, 64}) {
            const double f = us_per_launch(q, n, [&] { sampler_tail<float>(q, sel, o, k, 0.95f, 0.5f); });
            double d = -1;
            if (fp64) d = us_per_launch(q, n, [&] { sampler_tail<double>(q, sel, o, k, 0.95f, 0.5f); });
            std::printf("sampler tail, k=%-21d float %6.2f us   double %6.2f us\n", k, f, d);
        }
    }
    return 0;
}
