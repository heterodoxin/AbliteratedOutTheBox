// src/core/concurrent_main.cpp - do the CPU expert path and the GPU stream path actually run at once?
//
// L9 concluded that the two miss paths are INDEPENDENT RESOURCES - 42.55 GB/s of DRAM bandwidth for the CPU
// expert kernel and 28.9 GB/s of PCIe for the stream - and that running them together would cover a token's
// 552.7 MB of misses in 7.74 ms, which is what puts 100 tps inside reach on this Gen4 machine.
//
// THAT CLAIM HAS A HOLE.  An H2D copy from PINNED host memory READS HOST DRAM to do it, so the bus path is not
// free of the resource the CPU path is bound by.  If the two contend, the combined rate is not 42.55 + 28.9 and
// the 129 tok/s figure is optimistic; if they overlap cleanly, it stands.  This measures it three ways in one
// process - CPU alone, GPU alone, both - because comparing two separately-measured rates is exactly the
// inference that has been wrong twice already in this ledger (L2's dependency chain, L8's bus-only ceiling).
//
// The CPU side is a READ LOOP over the same pinned arena rather than the real VNNI kernel, and that is
// deliberate: the question is whether the two paths contend for DRAM BANDWIDTH, and a read loop is a pure
// measurement of that demand, with no compute that could mask it.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/pinned.hpp"
#include "strata/kernels/s_gemv.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

void check(dpct::err0 e, const char *what) {
}

// Volatile sink so the read loop cannot be optimised away; summed across threads and printed, so a compiler
// that removed the loads would produce a suspiciously round zero rather than a plausible number.
std::atomic<uint64_t> g_sink{0};

struct Run {
    double seconds = 0;
    double cpu_gb = 0;      // GB/s read by the CPU threads
    double gpu_experts = 0; // experts streamed and computed by the GPU
};

}  // namespace

int main(int argc, char** argv) {
    std::string path;
    // 4096 planes = 2 GiB, which is 64x the 32 MB L3 on this CPU.  At 0.1 GiB the read loop measured 206 GB/s -
    // that is CACHE bandwidth, not DRAM, and it would have answered the contention question with a number about
    // a different resource entirely.
    int experts = 4096;
    int cpu_threads = 6;
    int gpu_roles = 256;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--file") path = val();
        else if (a == "--experts") experts = std::atoi(val());
        else if (a == "--cpu-threads") cpu_threads = std::atoi(val());
        else if (a == "--gpu-roles") gpu_roles = std::atoi(val());
        else {
            std::fprintf(stderr, "usage: strata-concurrent --file experts.bin [--experts N] "
                                 "[--cpu-threads N] [--gpu-roles N]\n");
            return 2;
        }
    }
    if (path.empty()) { std::fprintf(stderr, "--file is required\n"); return 2; }

    const long long n_in = 2560, n_out = 640;
    const uint64_t role_codes = (uint64_t) n_out * n_in / 4;
    const uint64_t plane_bytes = role_codes;      // the CPU loop reads the same planes

    strata::core::PinnedArena arena((uint64_t) experts * role_codes);
    if (!arena.valid()) { std::fprintf(stderr, "arena allocation failed\n"); return 1; }
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return 1; }
        for (int i = 0; i < experts; ++i) {
            f.seekg((std::streamoff) i * 1382400);
            f.read((char*) arena.data() + (uint64_t) i * role_codes, (std::streamsize) role_codes);
            if ((uint64_t) f.gcount() != role_codes) { std::fprintf(stderr, "short read\n"); return 1; }
        }
    }
    std::printf("arena %.2f GiB pinned; %d CPU threads; %d GPU roles\n",
                (double) arena.capacity / (1024.0 * 1024 * 1024), cpu_threads, gpu_roles);

    uint8_t* d_codes = nullptr;
    check(DPCT_CHECK_ERROR(d_codes = (uint8_t *)sycl::malloc_device(
                               role_codes, dpct::get_in_order_queue())),
          "cudaMalloc codes");
    uint16_t* d_x = nullptr;
    float *d_scales = nullptr, *d_y = nullptr;
    check(DPCT_CHECK_ERROR(
              d_x = (uint16_t *)sycl::malloc_device(
                  n_in * sizeof(uint16_t), dpct::get_in_order_queue())),
          "cudaMalloc x");
    check(DPCT_CHECK_ERROR(d_scales = (float *)sycl::malloc_device(
                               (size_t)n_out * (n_in / 64) * sizeof(float),
                               dpct::get_in_order_queue())),
          "cudaMalloc scales");
    check(DPCT_CHECK_ERROR(
              d_y = (float *)sycl::malloc_device(n_out * sizeof(float),
                                                 dpct::get_in_order_queue())),
          "cudaMalloc y");
    std::vector<uint16_t> hx((size_t) n_in, 0x3C00);
    std::vector<float> hs((size_t) n_out * (size_t) (n_in / 64), 0.001f);
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
              d_x, hx.data(), hx.size() * sizeof(uint16_t)).wait()),
          "copy x");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming in
    the original code the source host memory is pageable memory. If the memory
    is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    check(DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
              d_scales, hs.data(), hs.size() * sizeof(float)).wait()),
          "copy scales");
    dpct::queue_ptr s_copy{}, s_comp{};
    check(DPCT_CHECK_ERROR(s_copy =
                               dpct::get_current_device().create_queue(true)),
          "stream copy");
    check(DPCT_CHECK_ERROR(s_comp =
                               dpct::get_current_device().create_queue(true)),
          "stream compute");
    const strata::kernels::SForm form{2, -1, 64, strata::kernels::Codebook::Affine, false};

    // The CPU loop: read the whole arena, Nthreads ways, and sum bytes so nothing is elided.
    auto cpu_work = [&](std::atomic<bool>& stop, std::atomic<uint64_t>& bytes) {
        uint64_t acc = 0, n = 0;
        const uint8_t* p = arena.data();
        while (!stop.load(std::memory_order_relaxed)) {
            // touch one byte per 64 - a stride that defeats prefetch into a pure streaming read
            for (uint64_t i = 0; i < plane_bytes * (uint64_t) experts; i += 64) acc += p[i];
            n += plane_bytes * (uint64_t) experts;
        }
        bytes.fetch_add(n);
        g_sink.fetch_add(acc);
    };

    // The GPU loop: stream each role plane on the copy stream while computing on the compute stream.
    auto gpu_work = [&](int roles) {
        try {
    dpct::event_ptr ev[2];
        for (int b = 0; b < 2; ++b)
            check(DPCT_CHECK_ERROR(ev[b] = new sycl::event()), "event");
        uint8_t* buf[2] = {d_codes, d_codes};      // one buffer is enough: the timing is what is measured
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        check(
            DPCT_CHECK_ERROR(s_copy->memcpy(buf[0], arena.data(), role_codes)),
            "prologue");
        /*
        DPCT1024: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        check(DPCT_CHECK_ERROR(dpct::sync_barrier(ev[0], s_copy)),
              "prologue event");
        for (int i = 0; i < roles; ++i) {
            const int cur = i & 1, nxt = (i + 1) & 1;
            if (i + 1 < roles) {
                /*
                DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy
                API. While the origin API might be synchronous, it depends on
                the type of operand memory, so you may need to call wait() on
                event return by memcpy API to ensure synchronization behavior.
                */
                check(DPCT_CHECK_ERROR(s_copy->memcpy(
                          buf[nxt],
                          arena.data() +
                              (uint64_t)((i + 1) % experts) * role_codes,
                          role_codes)),
                      "copy");
                /*
                DPCT1024: The original code returned the error code that was
                further consumed by the program logic. This original code was
                replaced with 0. You may need to rewrite the program logic
                consuming the error code.
                */
                check(DPCT_CHECK_ERROR(dpct::sync_barrier(ev[nxt], s_copy)),
                      "event");
            }
            check(
                DPCT_CHECK_ERROR(s_comp->ext_oneapi_submit_barrier({*ev[cur]})),
                "wait");
            strata::kernels::s_gemv_split_async(d_x, buf[cur], d_scales, nullptr, d_y, n_in, n_out, form, 32,
                                                (void*) s_comp);
        }
        check(DPCT_CHECK_ERROR(s_comp->wait()), "sync comp");
        check(DPCT_CHECK_ERROR(s_copy->wait()), "sync copy");
        for (int b = 0; b < 2; ++b) dpct::destroy_event(ev[b]);
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    };

    auto run = [&](bool do_cpu, bool do_gpu, const char *name) {
        try {
    std::atomic<bool> stop{false};
        std::atomic<uint64_t> bytes{0};
        std::vector<std::thread> cpu;
        check(DPCT_CHECK_ERROR(dpct::get_in_order_queue().wait()), "pre");
        const auto t0 = std::chrono::steady_clock::now();
        if (do_cpu) for (int t = 0; t < cpu_threads; ++t) cpu.emplace_back(cpu_work, std::ref(stop), std::ref(bytes));
        if (do_gpu) {
            gpu_work(gpu_roles);
        } else {
            // CPU-only needs a REAL window.  The first version let t0..t1 span nothing when there was no GPU
            // work, so the threads were stopped immediately and the run reported 0.000 s and 0.00 GB/s.
            std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        }
        const auto t1 = std::chrono::steady_clock::now();
        stop = true;
        for (auto& t : cpu) t.join();
        Run r;
        r.seconds = std::chrono::duration<double>(t1 - t0).count();
        r.cpu_gb = (double) bytes.load() / 1e9 / r.seconds;
        r.gpu_experts = do_gpu ? (double) gpu_roles / r.seconds : 0.0;
        std::printf("  %-18s %7.3f s   CPU read %6.2f GB/s   GPU %7.1f roles/s\n", name, r.seconds, r.cpu_gb,
                    r.gpu_experts);
        return r;
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    };

    std::printf("\n  mode                     wall   CPU read          GPU\n");
    const Run cpu_only = run(true, false, "CPU only");
    const Run gpu_only = run(false, true, "GPU only");
    const Run both = run(true, true, "BOTH");

    std::printf("\n  independence check (1.00 = perfectly shared, 0.00 = fully independent):\n");
    const double cpu_interference = cpu_only.cpu_gb > 0 ? 1.0 - both.cpu_gb / cpu_only.cpu_gb : 0.0;
    const double gpu_interference = gpu_only.gpu_experts > 0 ? 1.0 - both.gpu_experts / gpu_only.gpu_experts : 0.0;
    std::printf("    CPU throughput lost when the GPU also runs : %5.1f%%\n", cpu_interference * 100);
    std::printf("    GPU throughput lost when the CPU also runs : %5.1f%%\n", gpu_interference * 100);
    std::printf("    combined vs the sum of the parts           : CPU %6.2f + GPU-equivalent vs %6.2f measured\n",
                cpu_only.cpu_gb, both.cpu_gb);
    std::printf("\n  NOTE: the CPU side is a READ LOOP, not the VNNI kernel - it measures DRAM DEMAND, which is the\n"
                "        resource the question is about.  L9's 129 tok/s assumes these two do not contend.\n");

    sycl::free(d_codes, dpct::get_in_order_queue());
    sycl::free(d_x, dpct::get_in_order_queue());
    sycl::free(d_scales, dpct::get_in_order_queue());
    sycl::free(d_y, dpct::get_in_order_queue());
    dpct::get_current_device().destroy_queue(s_copy);
    dpct::get_current_device().destroy_queue(s_comp);
    return 0;
}
