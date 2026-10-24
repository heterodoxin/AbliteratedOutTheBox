// src/core/device.cu - P2.S1: the CUDA side of the runtime core.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/device.hpp"

#include <cstdio>
#include <cstring>

namespace strata::core {

namespace {

void check(dpct::err0 e, const char *what) {
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        throw CudaError(
            std::string(what) + ": " + dpct::get_error_string_dummy(e), (int)e);
    }
}

// A NaN pattern, not zero.  Zeros read from uninitialised memory are indistinguishable from real zeros in a
// dequantized weight or a masked attention score, which is exactly the kind of wrong-but-plausible value the
// Phase 1 harnesses kept catching.
__dpct_inline__ void poison_kernel(float *p, uint64_t n_floats) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint64_t i =
        (uint64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n_floats) p[i] = sycl::bit_cast<float>(0x7fc00000);
}

#if defined(STRATA_USE_HIP)
#if !defined(STRATA_HIP_ARCHS)
#error "STRATA_HIP_ARCHS (the compiled HIP architectures) is set by cmake/hip_backend.cmake"
#endif
// "gfx1201:sramecc-:xnack-" -> "gfx1201"
std::string base_arch(const char* gcn_arch_name) {
    std::string arch(gcn_arch_name);
    const size_t colon = arch.find(':');
    if (colon != std::string::npos) arch.resize(colon);
    return arch;
}

bool compiled_for(const std::string& arch) {
    const std::string list = STRATA_HIP_ARCHS;
    size_t a = 0;
    while (a <= list.size()) {
        size_t b = list.find(',', a);
        if (b == std::string::npos) b = list.size();
        if (!arch.empty() && list.compare(a, b - a, arch) == 0 && b - a == arch.size()) return true;
        a = b + 1;
    }
    return false;
}

std::string arch_problem(const cudaDeviceProp& p, int ordinal) {
    const std::string arch = base_arch(p.gcnArchName);
    const std::string card = "GPU " + std::to_string(ordinal) + " (" + p.name + ", " + arch + ")";
    if (!compiled_for(arch)) {
        return card + " is not an architecture this Strata engine was compiled for (" + STRATA_HIP_ARCHS +
               "); compile it for this card (./setup.sh --backend hip, or -DCMAKE_HIP_ARCHITECTURES=" + arch +
               ", docs/AMD_HIP.md) or choose another GPU with HIP_VISIBLE_DEVICES";
    }
    if (p.warpSize != 32) {
        return card + " runs wave" + std::to_string(p.warpSize) + "; Strata's HIP kernels need wave32";
    }
    return "";
}
#endif

}  // namespace

const char* compiled_gpu_archs() {
#if defined(STRATA_USE_HIP)
    return STRATA_HIP_ARCHS;
#else
    return "";
#endif
}

int device_count() try {
    int count = 0;
    if (DPCT_CHECK_ERROR(count = dpct::device_count()) !=
        0) { // HIP without a usable device reports an error, not 0
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        return 0;
    }
    return count < 0 ? 0 : count;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool device_summary(int ordinal, std::string &name, std::string &detail) try {
    dpct::device_info p{};
    if (ordinal < 0 || ordinal >= device_count() ||
        DPCT_CHECK_ERROR(dpct::get_device(ordinal).get_device_info(p)) != 0) {
        /*
        DPCT1026: The call to cudaGetLastError was removed because this
        functionality is redundant in SYCL.
        */
        return false;
    }
    char buf[160];
#if defined(STRATA_USE_HIP)
    std::snprintf(buf, sizeof(buf), "arch %s, %.1f GiB, wave%d", base_arch(p.gcnArchName).c_str(),
                  (double) p.totalGlobalMem / (1024.0 * 1024 * 1024), p.warpSize);
#else
    /*
    DPCT1005: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    std::snprintf(buf, sizeof(buf), "compute capability %d.%d, %.1f GiB",
                  p.get_major_version(), p.get_minor_version(),
                  (double)p.get_global_mem_size() / (1024.0 * 1024 * 1024));
#endif
    name = p.get_name();
    detail = buf;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::core

#if defined(STRATA_USE_HIP) && defined(_WIN32)
// The free VRAM figure on Windows HIP (include/strata/hip_compat/cuda_runtime.h maps cudaMemGetInfo here).
//
// hipMemGetInfo (ROCclr's PAL backend, Device::globalFreeMemory) is the card's size minus this process's own
// allocations: it asks Windows for this process's usage only, never for what the desktop and other programs hold.
// So on a card that also drives the desktop the engine counted ~930 MiB that was not there, the expert cache filled
// the card, and WDDM moved memory out to system RAM: decode at 30 tok/s instead of 41 (RX 6800, HIP SDK 7.2).
// Windows itself gives each process a video memory budget (DXGI QueryVideoMemoryInfo) that does account for the
// others.  Measured on that card: the budget sits 0.8 GiB below the card's size while this process is small and
// 1.8 GiB below once it holds 11 GiB; allocations past it still succeed, and are what Windows moves out.  The
// free figure here is hipMemGetInfo's lowered by what the budget withholds (the card's size minus the budget).
// Subtracting from HIP's own figure, rather than taking the budget minus DXGI's usage, keeps memory the HIP runtime
// has freed and holds in its cache counted as free, as hipMemGetInfo counts it.  dxgi.dll is loaded at run time, so
// nothing new is linked; when it or the card's adapter cannot be found, hipMemGetInfo's figure stands.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <atomic>
#include <cstdlib>
#include <mutex>

namespace strata::hip_compat {
namespace {
IDXGIAdapter3* budget_adapter(int device) {
    constexpr int kMaxDevices = 16;
    static std::mutex mu;
    static IDXGIAdapter3* adapters[kMaxDevices] = {};
    static bool tried[kMaxDevices] = {};
    if (device < 0 || device >= kMaxDevices) return nullptr;
    std::lock_guard<std::mutex> lock(mu);
    if (tried[device]) return adapters[device];
    tried[device] = true;
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, device) != hipSuccess) {
        (void) hipGetLastError();
        return nullptr;
    }
    LUID luid{};
    std::memcpy(&luid, p.luid, sizeof luid);
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    static HMODULE dxgi = LoadLibraryW(L"dxgi.dll");   // kept for the process' lifetime, like the adapters
    const auto create = dxgi ? (CreateFactory) (void*) GetProcAddress(dxgi, "CreateDXGIFactory1") : nullptr;
    IDXGIFactory4* factory = nullptr;
    if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory4), (void**) &factory))) return nullptr;
    IDXGIAdapter3* adapter = nullptr;
    if (FAILED(factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3), (void**) &adapter))) adapter = nullptr;
    factory->Release();
    adapters[device] = adapter;
    return adapter;
}
}  // namespace

// #873 / #654: on some Windows setups the first hipMemGetInfo of a process fails with hipErrorInvalidValue because the ROCm
// runtime has not finished initialising against the Adrenalin WDDM driver, and the first kernel launch after that
// crashes (0xC0000005 in amdhip64_7.dll).  A hipFree(0) before the first query initialises the context, and one
// failed query gets one retry after 250 ms.  Both happen once per process, before the DXGI budget below; any other
// error, and a failure of the retry, are returned as they are (a real failure is never retried).
static hipError_t first_mem_get_info(size_t* free_bytes, size_t* total_bytes) {
    static std::atomic<bool> warmed{false}, retried{false};
    if (!warmed.exchange(true)) (void) hipFree(nullptr);
    hipError_t e = hipMemGetInfo(free_bytes, total_bytes);
    if (e == hipErrorInvalidValue && !retried.exchange(true)) {
        (void) hipGetLastError();
        std::fprintf(stderr, "strata: hipMemGetInfo failed (%s); retrying once after 250 ms\n", hipGetErrorString(e));
        Sleep(250);
        e = hipMemGetInfo(free_bytes, total_bytes);
        std::fprintf(stderr, "strata: hipMemGetInfo retry: %s\n", e == hipSuccess ? "ok" : hipGetErrorString(e));
    }
    return e;
}

hipError_t mem_get_info(size_t* free_bytes, size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) return hipMemGetInfo(free_bytes, total_bytes);
    const hipError_t e = first_mem_get_info(free_bytes, total_bytes);
    static const bool off = [] {
        const char* v = std::getenv("STRATA_WDDM_BUDGET");
        return v != nullptr && std::atoi(v) == 0;
    }();
    if (e != hipSuccess || off || free_bytes == nullptr || total_bytes == nullptr) return e;
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
        (void) hipGetLastError();
        return e;
    }
    IDXGIAdapter3* adapter = budget_adapter(device);
    DXGI_QUERY_VIDEO_MEMORY_INFO local{};
    if (adapter == nullptr || FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) ||
        local.Budget == 0 || local.Budget >= *total_bytes)
        return e;
    const size_t withheld = *total_bytes - (size_t) local.Budget;   // the desktop's and other programs' share
    static std::atomic<bool> said{false};
    if (!said.exchange(true)) {
        std::fprintf(stderr, "strata: Windows budgets %llu of this card's %llu MiB for this process; free VRAM is "
                             "counted within that (STRATA_WDDM_BUDGET=0: off)\n",
                     (unsigned long long) (local.Budget >> 20), (unsigned long long) (*total_bytes >> 20));
    }
    *free_bytes = *free_bytes > withheld ? *free_bytes - withheld : 0;
    return e;
}
}  // namespace strata::hip_compat
#endif

namespace strata::core {

std::string gpu_arch_problem(int ordinal) {
#if defined(STRATA_USE_HIP)
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || ordinal < 0 || ordinal >= count) {
        cudaGetLastError();
        return "";
    }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) {
        cudaGetLastError();
        return "";
    }
    return arch_problem(p, ordinal);
#else
    (void) ordinal;
    return "";
#endif
}

std::string device_code_error() try {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    return "";   // gpu_arch_problem() checks the HIP architectures against STRATA_HIP_ARCHS, before this point
#else
    // every .cu of the engine is compiled for the same CMAKE_CUDA_ARCHITECTURES, so this kernel stands for all
    dpct::kernel_function_info a{};
    const dpct::err0 e = DPCT_CHECK_ERROR(
        dpct::get_kernel_function_info(&a, (const void *)poison_kernel));
    if (e == 0) return {};
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    return dpct::get_error_string_dummy(e);
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

DeviceInfo device_info(int ordinal) {
    int count = 0;
    check(DPCT_CHECK_ERROR(count = dpct::device_count()), "cudaGetDeviceCount");
    if (count == 0) {
#if defined(STRATA_USE_HIP)
        throw CudaError(std::string("no HIP device is present; this engine was compiled for ") + STRATA_HIP_ARCHS, -1);
#else
        throw CudaError("no CUDA device is present; Strata needs an NVIDIA GPU (RTX 20 series or newer)", -1);
#endif
    }
    if (ordinal < 0 || ordinal >= count) {
        throw CudaError("device ordinal " + std::to_string(ordinal) + " is out of range (have " +
                            std::to_string(count) + ")",
                        -1);
    }
    DeviceInfo d;
    d.ordinal = ordinal;
    /*
    DPCT1093: The "ordinal" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    check(DPCT_CHECK_ERROR(dpct::select_device(ordinal)), "cudaSetDevice");

    dpct::device_info p{};
    check(DPCT_CHECK_ERROR(dpct::get_device(ordinal).get_device_info(p)),
          "cudaGetDeviceProperties");
    d.name = p.get_name();
    /*
    DPCT1005: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    d.cc_major = p.get_major_version();
    /*
    DPCT1005: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    d.cc_minor = p.get_minor_version();
    d.multi_processor_count = p.get_max_compute_units();

    size_t free_b = 0, total_b = 0;
    /*
    DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    check(DPCT_CHECK_ERROR(
              dpct::get_current_device().get_memory_info(free_b, total_b)),
          "cudaMemGetInfo");
    d.free_bytes = free_b;
    d.total_bytes = total_b;

    /*
    DPCT1043: The version-related API is different in SYCL. An initial code
    was generated, but you need to adjust it.
    */
    check(DPCT_CHECK_ERROR(d.driver_version = dpct::get_major_version(
                               dpct::get_current_device())),
          "cudaDriverGetVersion");
    /*
    DPCT1043: The version-related API is different in SYCL. An initial code
    was generated, but you need to adjust it.
    */
    check(DPCT_CHECK_ERROR(d.runtime_version = dpct::get_major_version(
                               dpct::get_current_device())),
          "cudaRuntimeGetVersion");

    // The engine supports compute capability 7.5 and newer (Turing: the QSA scorer's tf32 mma has a portable
    // fp32-FMA fallback below sm_80, the tensor-core prompt kernels refuse and fall back).  Compiling for a
    // supported arch is enforced by CMake; RUNNING on an older card is caught here, because a binary can be carried
    // to a machine with an older card and would otherwise silently take whatever path the driver chose.  The HIP
    // backend checks the card against the architectures the binary was compiled for (and wave32).
#if defined(STRATA_HIP_GFX906)
    // AMD gfx906 (the STRATA_HIP_GFX906 compat build, not STRATA_USE_HIP): the build targets one gfx arch
    // (CMAKE_HIP_ARCHITECTURES, wave64); a card of another arch fails at the first kernel launch with
    // hipErrorInvalidDeviceFunction.  gcnArchName says which it is.
    {
        std::string arch(p.gcnArchName);
        if (const size_t colon = arch.find(':'); colon != std::string::npos) arch.resize(colon);
        d.arch = arch;
    }
    if (p.gcnArchName[0] != 0) d.name += std::string(" (") + p.gcnArchName + ")";
#elif defined(STRATA_USE_HIP)
    d.arch = base_arch(p.gcnArchName);
    if (const std::string why = arch_problem(p, ordinal); !why.empty()) throw CudaError(why, -1);
#else
    // #236: the experimental build (-DSTRATA_EXPERIMENTAL_SM60=ON: Pascal sm_60, Volta sm_70) runs on the cards it
    // was built for - refusing them below 7.5 there made the flag useless; the release engine keeps 7.5
#if defined(STRATA_EXPERIMENTAL_SM60)
    constexpr int kMinCc = 60;
    const char* const kNeed = "6.0 or newer (this is the experimental Pascal / Volta build)";
#else
    constexpr int kMinCc = 75;
    const char* const kNeed = "7.5 or newer (RTX 20 / 30 / 40 / 50 series)";
#endif
    if (d.cc_major * 10 + d.cc_minor < kMinCc) {
        throw CudaError("device " + d.name + " reports compute capability " + std::to_string(d.cc_major) +
                            "." + std::to_string(d.cc_minor) + "; Strata needs compute capability " + kNeed,
                        -1);
    }
#endif
    return d;
}

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison)
    : capacity_(bytes), ordinal_(ordinal), poison_(poison) {
    if (bytes == 0) throw CudaError("DeviceArena of 0 bytes", -1);
    /*
    DPCT1093: The "ordinal" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    check(DPCT_CHECK_ERROR(dpct::select_device(ordinal)), "cudaSetDevice");
    // One allocation for the whole region.  cudaMalloc of a large block is the thing that can fail late, so it
    // happens once, here, before anything depends on it.
    check(DPCT_CHECK_ERROR(base_ = (void *)strata::malloc_device_guarded(
                               (size_t)bytes, dpct::get_in_order_queue())),
          "cudaMalloc");
    if (poison_) {
        const int threads = 256;
        const uint64_t n = bytes / sizeof(float);
        const uint64_t blocks = (n + threads - 1) / threads;
        // gridDim.x is 32-bit, so a large region needs a loop.  12 GB of floats is 3e9 elements = 1.2e7
        // blocks, which fits, but the loop keeps it correct for any size rather than for today's sizes.
        const uint64_t max_blocks = 0x7FFFFFFFull;
        for (uint64_t b = 0; b < blocks; b += max_blocks) {
            const uint64_t chunk = (blocks - b < max_blocks) ? (blocks - b) : max_blocks;
            {
                auto exp_props = sycl::ext::oneapi::experimental::properties{
                    sycl::ext::oneapi::experimental::use_root_sync};

                dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                    auto float_base__b_threads_ct0 =
                        (float *)base_ + b * threads;
                    auto n_b_threads_ct1 = n - b * threads;

                    cgh.parallel_for<
                        dpct_kernel_name<class poison_kernel_58fc0a>>(
                        sycl::nd_range<3>(sycl::range(1, 1, (unsigned)chunk) *
                                              sycl::range(1, 1, threads),
                                          sycl::range(1, 1, threads)),
                        exp_props, [=](sycl::nd_item<3> item_ct1) {
                            poison_kernel(float_base__b_threads_ct0,
                                          n_b_threads_ct1);
                        });
                });
            }
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not
            use the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            check(0, "poison_kernel");
        }
        check(DPCT_CHECK_ERROR(
                  dpct::get_current_device().queues_wait_and_throw()),
              "poison sync");
    }
}

DeviceArena::~DeviceArena() {
    if (base_) sycl::free(
        base_,
        dpct::get_in_order_queue()); // best effort: a destructor must not throw
}

void* DeviceArena::alloc(uint64_t bytes, uint64_t align) {
    if (bytes == 0) return nullptr;
    if (align == 0 || (align & (align - 1)) != 0) {
        throw CudaError("DeviceArena::alloc alignment must be a power of two", -1);
    }
    const uint64_t start = (used_ + align - 1) & ~(align - 1);
    if (start + bytes > capacity_) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "DeviceArena out of memory: asked for %llu B at offset %llu (align %llu) in a %llu B "
                      "region - the plan from P1.S9 did not close",
                      (unsigned long long) bytes, (unsigned long long) start, (unsigned long long) align,
                      (unsigned long long) capacity_);
        throw CudaError(msg, -1);
    }
    used_ = start + bytes;
    return (char*) base_ + start;
}

}  // namespace strata::core
