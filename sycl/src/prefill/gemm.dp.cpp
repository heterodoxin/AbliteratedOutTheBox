// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include <dpct/blas_utils.hpp>

#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
#include "strata/prefill/moe_mmq.hpp"
#endif

#ifdef STRATA_USE_HIP
#include "wmma_gemm.h"
#endif

#include <algorithm>
#include <atomic>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// The HIP compatibility shim maps CUDA shuffle spellings to Strata helpers.
// hipBLASLt's public headers declare native HIP shuffle functions, so keep
// those declarations from being macro-expanded in this translation unit.
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#endif

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <cmath>

#include <dpct/lib_common_utils.hpp>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
#include "hipblaslt_tuning.hpp"
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <set>
#include <tuple>
#endif

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#endif

namespace strata::prefill {
namespace {
#if defined(__HIPCC__)
// Y's rows, written by an FP16-out GEMM as FP16 at the start of each FP32 row (ldc = 2 ldy halves), widened in place.
// Float c overwrites halves 2c and 2c+1, so a row is walked from its end in blocks: a block's halves are read into
// registers, the block syncs, then writes its floats - which only cover halves of blocks already read.
// STRATA_DBG_NAN: the FP16 outputs that are not finite (an FP32-accumulated sum past 65504 becomes inf in FP16).
__device__ unsigned long long g_f16_nonfinite = 0;
__global__ void widen_rows_f16(float* __restrict__ Y, int64_t n, int64_t ldy, int count_nonfinite) {
    float* y = Y + (int64_t) blockIdx.x * ldy;
    const __half* h = reinterpret_cast<const __half*>(y);
    const int64_t nb = (n + blockDim.x - 1) / blockDim.x;
    for (int64_t b = nb - 1; b >= 0; --b) {
        const int64_t c = b * blockDim.x + threadIdx.x;
        const float v = c < n ? __half2float(h[c]) : 0.0f;
        if (count_nonfinite && c < n && !isfinite(v)) atomicAdd(&g_f16_nonfinite, 1ull);
        __syncthreads();
        if (c < n) y[c] = v;
        __syncthreads();
    }
}
// BF16 weight rows -> FP16, saturated (bench/results/2026-10-04-rdna2-fp16-prompt: none leaves FP16's range)
__global__ void bf16_to_f16_rows(const uint16_t* __restrict__ s, __half* __restrict__ d, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float f = __uint_as_float((uint32_t) s[i] << 16);
        d[i] = __float2half(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f));   // as hf_sat: a NaN stays NaN
    }
}
#endif

void ck(int s, const char *what) {
    if (s != 0) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

// #247/#325: on Windows (seen on gfx1201), hipBLAS can return success with the correct BF16/FP16 product for some
// shapes (hc up once T >= 96, the router) and still leave hipErrorInvalidValue set, which the next kernel's error
// check turns into an exit. The multiply has finished, so that one stale error is cleared after a GEMM that succeeded;
// any other error still stops the engine. Windows only: on Linux a stale hipErrorInvalidValue is a real error from
// an earlier call and keeps being reported. A no-op everywhere else (CUDA compiles none of it).
#if defined(__HIPCC__) && defined(_WIN32)
void absorb_hipblas_sticky(const char* what) {
    const hipError_t sticky = hipGetLastError();
    if (sticky == hipSuccess || sticky == hipErrorInvalidValue) return;
    std::fprintf(stderr, "prefill gemm: %s left %s\n", what, hipGetErrorString(sticky));
    std::exit(1);
}
#define STRATA_ABSORB_HIPBLAS_STICKY(what) absorb_hipblas_sticky(what)
#else
#define STRATA_ABSORB_HIPBLAS_STICKY(what) ((void) 0)
#endif

// A setup call whose failure the engine survives (the handle keeps its defaults), as before #240 - but said.
void note(int s, const char *what) {
    if (s != 0) std::fprintf(
        stderr, "prefill gemm: %s: cuBLAS status %d (continuing)\n", what,
        (int)s);
}

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
struct HipLtCallKey {
    strata::prefill::hipblaslt::InputType type;
    int t;
    int n;
    int k;
    int ldy;
    uint32_t beta_bits;

    bool operator<(const HipLtCallKey& other) const {
        return std::tie(type, n, k, ldy, t, beta_bits) <
               std::tie(other.type, other.n, other.k, other.ldy, other.t, other.beta_bits);
    }
};

struct HipLtCachedAlgo {
    bool supported = false;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
};

struct HipLtState {
    hipblasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    strata::prefill::hipblaslt::TuningTable table;
    std::map<HipLtCallKey, HipLtCachedAlgo> cache;
    uint64_t lt_launches = 0;
    uint64_t fallbacks = 0;
    std::set<std::tuple<strata::prefill::hipblaslt::InputType, int, int, int, int>> fallback_shapes;

    ~HipLtState() {
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: hipBLASLt summary launches=%llu fallbacks=%llu unique_fallback_shapes=%zu\n",
                         (unsigned long long) lt_launches, (unsigned long long) fallbacks, fallback_shapes.size());
            for (const auto& shape : fallback_shapes) {
                const auto type = std::get<0>(shape);
                std::fprintf(stderr, "prefill gemm: fallback shape dtype=%s T=%d N=%d K=%d ldy=%d\n",
                             type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16",
                             std::get<1>(shape), std::get<2>(shape), std::get<3>(shape), std::get<4>(shape));
            }
        }
        if (handle) hipblasLtDestroy(handle);
    }
};

struct HipLtDescriptors {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;

    ~HipLtDescriptors() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }

    bool init(hipDataType type, int t, int n, int k, int ldy, int ldb = 0) {
        const hipblasOperation_t trans_a = HIPBLAS_OP_T;
        const hipblasOperation_t trans_b = HIPBLAS_OP_N;
        if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&a, type, k, n, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&b, type, k, t, ldb > k ? ldb : k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, ldy) != HIPBLAS_STATUS_SUCCESS) {
            return false;
        }
        return true;
    }
};

std::unique_ptr<HipLtState> create_hipblaslt_state(void* workspace, size_t workspace_bytes) {
    const char* path = std::getenv("STRATA_HIPBLASLT_TUNING");
    if (!path || !*path) return nullptr;

    auto state = std::make_unique<HipLtState>();
    state->workspace = workspace;
    state->workspace_bytes = workspace_bytes;
    if (hipblasLtCreate(&state->handle) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt handle creation failed; using hipBLASEx\n");
        return nullptr;
    }

    int version = 0;
    if (hipblasLtGetVersion(state->handle, &version) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt version query failed; using hipBLASEx\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; using hipBLASEx\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    const auto suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    std::string error;
    if (!state->table.load(path, arch, version, error)) {
        std::fprintf(stderr, "prefill gemm: %s; using hipBLASEx\n", error.c_str());
        return nullptr;
    }
    std::fprintf(stderr, "prefill gemm: hipBLASLt tuning enabled (%zu rows, %s, version %d)\n",
                 state->table.rows().size(), arch.c_str(), version);
    return state;
}

HipLtCachedAlgo resolve_hipblaslt_algo(HipLtState& state, strata::prefill::hipblaslt::InputType type, int t,
                                       int n, int k, int ldy, float beta) {
    uint32_t beta_bits = 0;
    static_assert(sizeof(beta_bits) == sizeof(beta));
    std::memcpy(&beta_bits, &beta, sizeof(beta));
    const HipLtCallKey key{type, t, n, k, ldy, beta_bits};
    const auto cached = state.cache.find(key);
    if (cached != state.cache.end()) return cached->second;

    HipLtCachedAlgo resolved;
    const bool verbose = std::getenv("STRATA_HIPBLASLT_VERBOSE") != nullptr;
    const auto* row = state.table.closest(type, n, k, ldy, t);
    if (!row) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; no calibration for dtype=%s T=%d N=%d K=%d ldy=%d\n",
                         type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, t, n, k, ldy)) {
        return state.cache.emplace(key, resolved).first->second;
    }

    std::vector<int> solution_ids{row->solution_id};
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    if (hipblaslt_ext::getAlgosFromIndex(state.handle, solution_ids, candidates) != HIPBLAS_STATUS_SUCCESS ||
        candidates.empty() || candidates.front().state != HIPBLAS_STATUS_SUCCESS ||
        hipblaslt_ext::getIndexFromAlgo(candidates.front().algo) != row->solution_id) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d unavailable for T=%d N=%d K=%d ldy=%d\n",
                         row->solution_id, t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    const float alpha = 1.0f;
    size_t required_workspace = 0;
    auto algo = candidates.front().algo;
    if (hipblaslt_ext::matmulIsAlgoSupported(state.handle, desc.op, &alpha, desc.a, desc.b, &beta, desc.c, desc.c,
                                             algo, required_workspace) != HIPBLAS_STATUS_SUCCESS) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d rejects actual T=%d N=%d K=%d ldy=%d beta=%.9g\n",
                         row->solution_id, t, n, k, ldy, beta);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    resolved.supported = true;
    resolved.algo = algo;
    resolved.workspace_bytes = required_workspace;
    if (verbose) {
        std::fprintf(stderr,
                     "prefill gemm: Lt solution=%d dtype=%s T=%d N=%d K=%d ldy=%d beta=%.9g workspace=%zu\n",
                     row->solution_id, type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n,
                     k, ldy, beta, required_workspace);
    }
    return state.cache.emplace(key, resolved).first->second;
}

// the hipBLASLt solution index the tuning table gives this call (-1: none / unsupported)
int hipblaslt_solution_for(void* opaque_state, strata::prefill::hipblaslt::InputType type, int64_t t, int64_t n, int64_t k,
                           int64_t ldy, float beta) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX || ldy < n) return -1;
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported || resolved.workspace_bytes > state->workspace_bytes) return -1;
    auto algo = resolved.algo;
    return hipblaslt_ext::getIndexFromAlgo(algo);
}

bool try_hipblaslt(void* opaque_state, strata::prefill::hipblaslt::InputType type, const uint16_t* x,
                   const uint16_t* w, float* y, int64_t t, int64_t n, int64_t k, int64_t ldy, float beta,
                   void* stream, int64_t ldx = 0) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX ||
        ldy < n) {
        return false;
    }
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        return false;
    }
    if (resolved.workspace_bytes > state->workspace_bytes) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution needs %zu workspace bytes, have %zu\n",
                         resolved.workspace_bytes, state->workspace_bytes);
        }
        return false;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, (int) t, (int) n, (int) k, (int) ldy, (int) ldx)) return false;
    const float alpha = 1.0f;
    const hipblasStatus_t status = hipblasLtMatmul(state->handle, desc.op, &alpha, w, desc.a, x, desc.b, &beta, y,
                                                   desc.c, y, desc.c, &resolved.algo, state->workspace,
                                                   state->workspace_bytes, (hipStream_t) stream);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ++state->lt_launches;
        return true;
    }

    std::fprintf(stderr, "prefill gemm: hipBLASLt launch failed with status %d\n", (int) status);
    if (beta != 0.0f) {
        std::fprintf(stderr, "prefill gemm: refusing a fallback after hipBLASLt failed with nonzero beta\n");
        std::exit(1);
    }
    auto* mutable_state = static_cast<HipLtState*>(opaque_state);
    uint32_t beta_bits = 0;
    std::memcpy(&beta_bits, &beta, sizeof(beta_bits));
    auto cached = mutable_state->cache.find(HipLtCallKey{type, (int) t, (int) n, (int) k, (int) ldy, beta_bits});
    if (cached != mutable_state->cache.end()) cached->second.supported = false;
    ++mutable_state->fallbacks;
    mutable_state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
    return false;
}
#endif

}  // namespace

bool prompt_f16() {
#if defined(__HIPCC__)
    constexpr int kMaxDev = 64;
    static std::atomic<int8_t> cached[kMaxDev] = {};   // 0 unknown, 1 off, 2 on
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxDev) { (void) hipGetLastError(); return false; }
    if (const int8_t c = cached[dev].load(std::memory_order_relaxed)) return c == 2;
    // Opt-in (#835): only STRATA_HIP_PROMPT_F16=1 turns it on.  It changes the prompt path's numbers (an FP16 rounding of
    // each 16-bit GEMM's output), so no card gets it unasked.  On gfx103x it is ~2x faster, so we say so once.
    const char* e = std::getenv("STRATA_HIP_PROMPT_F16");
    const bool on = e != nullptr && e[0] == '1';
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, dev) != hipSuccess) { (void) hipGetLastError(); p.gcnArchName[0] = 0; }
    if (on) std::fprintf(stderr, "strata prefill: HIP device %d%s%s - STRATA_HIP_PROMPT_F16=1: the prompt's 16-bit GEMMs run FP16 in and out "
                                 "(rocBLAS is tuned only for that on gfx103x; not bit-identical to the default)%s",
                         dev, p.gcnArchName[0] ? " " : "", p.gcnArchName, "\n");
    else if (e == nullptr && std::strncmp(p.gcnArchName, "gfx103", 6) == 0)
        std::fprintf(stderr, "strata prefill: HIP device %d %s - tip: STRATA_HIP_PROMPT_F16=1 reads long prompts about 2x faster here "
                             "(FP16 prompt GEMMs, rocBLAS is tuned only for FP16 on gfx103x; the numbers differ slightly, see docs/AMD_HIP.md)%s",
                     dev, p.gcnArchName, "\n");
    cached[dev].store(on ? 2 : 1, std::memory_order_relaxed);
    return on;
#else
    return false;
#endif
}

Gemm::~Gemm() {
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
    delete static_cast<strata::prefill::mmq::Context*>(mmq_ctx_);
    if (mmq_buf_) cudaFree(mmq_buf_);
#endif
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    delete static_cast<HipLtState*>(hipblaslt_state_);
#endif
    if (handle_) delete ((dpct::blas::descriptor_ptr)handle_);
    if (tc_w_) sycl::free(tc_w_, dpct::get_in_order_queue());
    if (tc_x_) sycl::free(tc_x_, dpct::get_in_order_queue());
    if (!external_) {
        if (scratch_) sycl::free(scratch_, dpct::get_in_order_queue());
        if (workspace_) sycl::free(workspace_, dpct::get_in_order_queue());
    }
}

bool Gemm::init_external(void *stream, uint16_t *scratch, int64_t scratch_elems,
                         void *workspace, size_t ws_bytes,
                         std::string &err) try {
    dpct::blas::descriptor_ptr h = nullptr;
    if (const int s = DPCT_CHECK_ERROR(h = new dpct::blas::descriptor());
        s != 0) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    note(DPCT_CHECK_ERROR(h->set_queue(strata::q_of(stream))),
         "cublasSetStream");
    workspace_ = workspace;
    /*
    DPCT1027: The call to cublasSetWorkspace was replaced with 0 because
    this functionality is redundant in SYCL.
    */
    note(0, "cublasSetWorkspace");
    note(DPCT_CHECK_ERROR(h->set_math_mode(dpct::blas::math_mode::mm_default)),
         "cublasSetMathMode");
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws_bytes).release();
#endif
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    /*
    DPCT1026: The call to cublasSetWorkspace was removed because this
    functionality is redundant in SYCL.
    */
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (hipblaslt_state_) {
        auto* state = static_cast<HipLtState*>(hipblaslt_state_);
        state->workspace = workspace_;
        state->workspace_bytes = ws_bytes;
    }
#endif
}

bool Gemm::init(void *stream, int64_t scratch_elems, std::string &err) try {
    // #240: every failure names the call and the real status, so "no VRAM" can be told from a broken install
    dpct::blas::descriptor_ptr h = nullptr;
    if (const int s = DPCT_CHECK_ERROR(h = new dpct::blas::descriptor());
        s != 0) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    note(DPCT_CHECK_ERROR(h->set_queue(strata::q_of(stream))),
         "cublasSetStream");
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (const dpct::err0 e =
            DPCT_CHECK_ERROR(workspace_ = (void *)sycl::malloc_device(
                                 ws, dpct::get_in_order_queue()));
        e != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        err = std::string("prefill gemm: workspace of 32 MiB: ") +
              dpct::get_error_string_dummy(e);
        return false;
    }
    /*
    DPCT1027: The call to cublasSetWorkspace was replaced with 0 because
    this functionality is redundant in SYCL.
    */
    note(0, "cublasSetWorkspace");
    note(DPCT_CHECK_ERROR(h->set_math_mode(dpct::blas::math_mode::mm_default)),
         "cublasSetMathMode");
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws).release();
#endif
    if (scratch_elems > 0) {
        /*
        DPCT1000: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (const dpct::err0 e = DPCT_CHECK_ERROR(
                scratch_ = (uint16_t *)sycl::malloc_device(
                    (size_t)scratch_elems * 2, dpct::get_in_order_queue()));
            e != 0) {
            /*
            DPCT1001: The statement could not be removed.
            */
            err = "prefill gemm: dequant scratch of " +
                  std::to_string(scratch_elems * 2 >> 20) + " MiB: " +
                  /*
                  DPCT1009: SYCL reports errors using exceptions and does
                  not use error codes. Please replace the
                  "get_error_string_dummy(...)" with a real error-handling
                  function.
                  */
                  dpct::get_error_string_dummy(e);
            return false;
        }
    }
    scratch_elems_ = scratch_elems;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#if !defined(__HIPCC__)
// ---- below sm_80: BF16 GEMMs without BF16 tensor cores (PR #655, #540, #395) ----------------------------------------
// Turing and Volta have no BF16 tensor cores: cublasGemmEx on CUDA_R_16BF inputs falls back to a SIMT fp32 kernel
// (magma_sgemmEx), a fifth of a 4K prompt on an RTX 2080 Ti (#655) and ~10% of a V100's prompt (#540).  BF16 -> FP16
// is exact for every value inside FP16's normal range (the 7-bit mantissa fits in 10 bits); weights and normalized
// activations sit there.  With the FP16 path, the weight is converted once per call and the activations in row
// slices, into the instance's own buffers, and the product runs as the FP16 GEMM (fp32 accumulate) on the tensor
// cores.  Finite values beyond FP16's range are clamped to +-65504 (#540) instead of becoming Inf.  A beta = 1
// product (STRATA_PREFILL_BF16X2's remainder, ~2^-9 of the original, deep in FP16's subnormal band) keeps cuBLAS.
//   Default: on for compute capability 7.0 - 7.4 (Volta: only the experimental STRATA_EXPERIMENTAL_SM60 build runs
//   there), OFF for 7.5 (RTX 20 in the ready-made engine: the sums round differently, so it is opt-in until it has
//   been gated).  STRATA_BF16_TC=1 / =0 turns it on / off on any 7.x card (=2, a test mode: on any card).
// Pascal (6.x) has no tensor cores and cuBLAS has no BF16 GEMM for it (#395, measured NOT_SUPPORTED on a P40): both
// operands are widened to fp32 by an exact shift and the product is cublasSgemm with the same fp32 accumulator.
namespace {
__dpct_inline__ void bf16_to_f16_kernel(const uint16_t *__restrict__ in,
                                        sycl::half *__restrict__ out,
                                        int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) {
        float f = sycl::bit_cast<float>((uint32_t)in[i] << 16);
        if (f > 65504.0f && !sycl::isinf(f)) f = 65504.0f;
        else if (f < -65504.0f && !sycl::isinf(f)) f = -65504.0f;
        out[i] = sycl::vec<float, 1>(f)
                     .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    }
}
void bf16_to_f16(const uint16_t *in, uint16_t *out, int64_t n,
                 dpct::queue_ptr st) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(st->get_device(), {sycl::aspect::fp16});

        st->parallel_for<dpct_kernel_name<class bf16_to_f16_kernel_a0031a>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)((n + 255) / 256)) *
                                  sycl::range(1, 1, 256),
                              sycl::range(1, 1, 256)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                bf16_to_f16_kernel(in, reinterpret_cast<sycl::half *>(out), n);
            });
    }
}
#if defined(STRATA_EXPERIMENTAL_SM60)   // Pascal runs only the experimental build
__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((uint32_t) in[i] << 16);
}
void bf16_to_f32(const uint16_t* in, float* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    bf16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, out, n);
}
#endif
// The current device's compute capability as 10 * major + minor, per device (a layer split can mix cards); 0: not
// known, read as "not an old card" so a failed query keeps the cuBLAS BF16 call.
int current_cc() try {
    static std::atomic<int> cc[64] = {};
    int dev = 0;
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 || dev < 0 ||
        dev >= 64) {
        ; return 0;
    }
    int v = cc[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int major = 0, minor = 0;
        if (DPCT_CHECK_ERROR(
                major = dpct::get_device(dev).get_major_version()) != 0 ||
            DPCT_CHECK_ERROR(
                minor = dpct::get_device(dev).get_minor_version()) != 0) {
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            return 0;
        }
        v = 10 * major + minor;
        cc[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
// 0: cuBLAS's BF16 GEMM, 1: through FP16 (tensor cores), 2: through FP32 (Pascal)
int bf16_path() {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_BF16_TC");
        return v != nullptr && v[0] != '\0' ? std::atoi(v) : -1;
    }();
    const int cc = current_cc();
    if (forced == 2 && cc > 0) return 1;   // a test mode: the FP16 path on any card (gemm_bf16_parity on sm_80+)
    if (cc <= 0 || cc >= 80) return 0;
    if (cc < 70) return 2;
    if (forced >= 0) return forced != 0 ? 1 : 0;
    return cc < 75 ? 1 : 0;
}
bool grow(uint16_t *&p, int64_t &have,
          int64_t want) try { // `have`, `want`: 2-byte elements
    if (have >= want) return true;
    if (p) sycl::free(p, dpct::get_in_order_queue());
    p = nullptr;
    have = 0;
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    if (DPCT_CHECK_ERROR(p = (uint16_t *)sycl::malloc_device(
                             (size_t)want * 2, dpct::get_in_order_queue())) !=
        0) {
        ; p = nullptr; return false;
    }
    have = want;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
constexpr int64_t kXSliceElems = 16ll << 20;   // 32 MiB of FP16 activations per slice (64 MiB as fp32)
}  // namespace
#endif
bool Gemm::bf16_hcd_exact(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K) {
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    static std::atomic<bool> told{false};
    if (!hipblaslt_state_ || N != 320 || K != 10240 || T < 1) {
        if (N == 320 && K == 10240 && !told.exchange(true))
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: no hipBLASLt tuning table is loaded, so the exact kernel has nothing to match: "
                                 "the hyper-connection down projection stays on hipBLAS\n");
        return false;
    }
    // The kernel copies hipBLASLt solution 1176 / 1177's summation order, which is only true of the library build the
    // table was calibrated with: take it only when the table makes hipBLASLt pick one of those two for this T.
    const int id = hipblaslt_solution_for(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, T, N, K, N, 0.0f);
    if (id != 1176 && id != 1177) {
        if (T >= 4096 && !told.exchange(true))   // (a short chunk may legitimately get another solution: not worth a line)
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: the tuning table gives the hyper-connection down projection (N 320, K 10240) "
                                 "solution %d at T %lld, not 1176 / 1177: hipBLASLt runs it (a changed table or library; "
                                 "tools/hip/gfx1151-hipblaslt-100401.txt is the one measured)\n",
                         id, (long long) T);
        return false;
    }
    return strata_pf_hcdown_exact_bf16(X, ldx, W, Y, T, N, K, stream_);
#else
    (void) X; (void) ldx; (void) W; (void) Y; (void) T; (void) N; (void) K;
    return false;
#endif
}

namespace {
// oneMKL's GEMM reads the last, partial tile of an operand from the wrong place when it lies more than 4 GiB into its
// allocation (Arc A770, oneMKL 2026.1: a BF16 B placed past 4 GiB of a 10.5 GiB allocation gave its last 21 columns
// wrong, and none below 4 GiB).  The prompt path borrows the top of the ~10 GiB expert cache for its buffers, so its
// GEMMs met this and the prompt came out non-finite / garbage.  An operand past 4 GiB is copied into a scratch buffer
// of the queue's own - a fresh allocation, so at offset 0 - and the GEMM runs on that.  Returns false, so the caller
// runs the direct GEMM, when neither operand needs staging (the common, correct case).
uint8_t* gemm_stage(size_t need, sycl::queue& q) {
    static thread_local uint8_t* p = nullptr;
    static thread_local size_t cap = 0;
    if (need > cap) {
        if (p != nullptr) {
            q.wait();   // the queue is in-order but asynchronous: a previous staged GEMM may still read p
            sycl::free(p, q);
        }
        p = (uint8_t*) sycl::malloc_device(need, q);
        cap = p != nullptr ? need : 0;
    }
    return p;
}
inline bool staged_gemm(sycl::queue& q, void* handle, const uint16_t* W, int lda, const uint16_t* X, int ldb,
                        dpct::library_data_t dt, int64_t N, int64_t T, int64_t K, float alpha, float beta,
                        float* Y, int64_t ldy) {
    // STRATA_SYCL_GEMM_NO_STAGE: the tests' negative control.  0.1.43: not on Battlemage either (the A770 oneMKL bug above did not
    // show on a B70: 4K and 49K prompts, operands at 18 GiB into the cache arena, first-token logits byte-identical with and
    // without staging; the 10 MB copy per expert was ~10% of a prompt).  STRATA_SYCL_GEMM_STAGE=1 stages there as before.
    static const bool no_stage = std::getenv("STRATA_SYCL_GEMM_NO_STAGE") != nullptr ||
        (strata::intel_gpu_gen(q.get_device()) == strata::IntelGpuGen::Battlemage && !(std::getenv("STRATA_SYCL_GEMM_STAGE") && std::getenv("STRATA_SYCL_GEMM_STAGE")[0] == '1'));
    constexpr size_t k4G = 4ull << 30;
    const size_t es = 2;
    const size_t w_bytes = (size_t) lda * (size_t) N * es;   // W stored K x N, col-major (lda = K)
    const size_t x_bytes = (size_t) ldb * (size_t) T * es;   // X stored ldb x T
    const bool far_w = strata::device_offset_end(W, w_bytes) > k4G;
    const bool far_x = strata::device_offset_end(X, x_bytes) > k4G;
    if (no_stage || (!far_w && !far_x)) return false;
    uint8_t* s = gemm_stage(w_bytes + x_bytes, q);
    if (s == nullptr) {
        std::fprintf(stderr, "strata: WARNING: a GEMM operand lies past 4 GiB and the %zu MiB staging buffer could not be "
                             "allocated; the GEMM runs directly and may be wrong on this card\n",
                     (w_bytes + x_bytes) >> 20);
        return false;
    }
    uint8_t* sw = s;
    uint8_t* sx = s + w_bytes;
    if (far_w) q.memcpy(sw, W, w_bytes);
    if (far_x) q.memcpy(sx, X, x_bytes);
    const uint16_t* Wu = far_w ? (const uint16_t*) sw : W;
    const uint16_t* Xu = far_x ? (const uint16_t*) sx : X;
    ck(DPCT_CHECK_ERROR(dpct::blas::gemm(
           (dpct::blas::descriptor_ptr) handle, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
           (int) N, (int) T, (int) K, &alpha, Wu, dt, lda, Xu, dt, ldb, &beta, Y,
           dpct::library_data_t::real_float, (int) ldy, dpct::compute_type::f32)),
       "cublasGemmEx (staged)");
    return true;
}
}  // namespace

// (#2091, CUDA Turing only) no FP16 activation image on this backend
bool Gemm::bf16_converts_x() const { return false; }

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta, int64_t ldx, bool /*x_f16*/) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (ldx <= K) ldx = 0;
#ifdef STRATA_USE_HIP
    // #313 (opt-in STRATA_WMMA_GEMM=1; STRATA_WMMA_BF16=0 excludes bf16): RDNA3 / RDNA3.5 WMMA dense GEMM, before
    // hipBLASLt; it returns false for shapes it does not take (and off gfx11) and this falls through
    static const bool wmma_on = [] { const char* v = std::getenv("STRATA_WMMA_GEMM"); return v && v[0] != 0 && v[0] != '0'; }();
    static const bool bf16_on = [] { const char* v = std::getenv("STRATA_WMMA_BF16"); return !v || (v[0] != 0 && v[0] != '0'); }();
    if (wmma_on && bf16_on && ldx == 0 && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_bf16(X, W, Y, T, N, K, ldy, beta, stream_)) {
        return;
    }
#endif
#if defined(__HIPCC__)
    if (f16_io_) {   // X is the FP16 image (set_act_f16); W goes through the dequantization scratch as FP16
        // the only accumulating BF16 product is the BF16X2 low part, which bf16x2_mode() turns off with this path
        if (beta != 0.0f) { std::fprintf(stderr, "prefill gemm: an accumulating BF16 GEMM on the FP16 prompt path\n"); std::exit(1); }
        const int64_t rows = std::min<int64_t>(N, scratch_elems_ / K);
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = std::min(rows, N - r0);
            bf16_to_f16_rows<<<1024, 256, 0, (cudaStream_t) stream_>>>(W + r0 * K, (__half*) scratch_, n * K);
            f16_inplace(X, scratch_, Y + r0, T, n, K, ldy);
        }
        return;
    }
#endif
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, X, W, Y, T, N, K, ldy,
                      beta, stream_, ldx)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt bf16");
        return;
    }
#endif
#if !defined(__HIPCC__)
    if (const int path = K > 0 ? bf16_path() : 0; path == 1 && N > 1 && beta == 0.0f) {
        // a single output row stays cuBLAS's GEMV, faster than the conversions; beta = 1: see above
        const int64_t x_rows = std::max<int64_t>(1, std::min<int64_t>(T, kXSliceElems / K));
        if (grow(tc_w_, tc_w_elems_, N * K) && grow(tc_x_, tc_x_elems_, x_rows * K)) {
            bf16_to_f16(W, tc_w_, N * K, strata::q_of(stream_));
            for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                const int64_t n = std::min<int64_t>(x_rows, T - t0);
                bf16_to_f16(X + t0 * K, tc_x_, n * K, strata::q_of(stream_));
                f16(tc_x_, tc_w_, Y + t0 * ldy, n, N, K, ldy, beta);
            }
            return;
        }
    }
#if defined(STRATA_EXPERIMENTAL_SM60)
    else if (path == 2) {
        // Pascal: fp32 copies (2 elements of the 2-byte buffers each).  Every tile is a disjoint block of Y, so each
        // gets the caller's beta.
        const int64_t x_rows = std::max<int64_t>(1, std::min<int64_t>(T, kXSliceElems / K));
        if (grow(tc_w_, tc_w_elems_, 2 * N * K) && grow(tc_x_, tc_x_elems_, 2 * x_rows * K)) {
            float* const wf = reinterpret_cast<float*>(tc_w_);
            float* const xf = reinterpret_cast<float*>(tc_x_);
            bf16_to_f32(W, wf, N * K, (cudaStream_t) stream_);
            for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                const int64_t n = std::min<int64_t>(x_rows, T - t0);
                bf16_to_f32(X + t0 * K, xf, n * K, (cudaStream_t) stream_);
                ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) n, (int) K, &alpha,
                               wf, (int) K, xf, (int) K, &beta, Y + t0 * ldy, (int) ldy),
                   "cublasSgemm (bf16 on Pascal)");
            }
            return;
        }
    }
#endif
#endif
    if (staged_gemm(*strata::q_of(stream_), handle_, W, (int) K, X, (int) (ldx ? ldx : K), dpct::library_data_t::real_bfloat16, N, T, K,
                    alpha, beta, Y, ldy))
        return;
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(DPCT_CHECK_ERROR(dpct::blas::gemm(
           (dpct::blas::descriptor_ptr)handle_, oneapi::mkl::transpose::trans,
           oneapi::mkl::transpose::nontrans, (int)N, (int)T, (int)K, &alpha, W,
           dpct::library_data_t::real_bfloat16, (int)K, X,
           dpct::library_data_t::real_bfloat16, (int)(ldx ? ldx : K), &beta, Y,
           dpct::library_data_t::real_float, (int)ldy,
           dpct::compute_type::f32)),
       "cublasGemmEx");
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx");
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
#ifdef STRATA_USE_HIP
    // #313 (opt-in STRATA_WMMA_GEMM=1): RDNA3 / RDNA3.5 WMMA dense GEMM, before hipBLASLt (falls through when false)
    static const bool wmma_on = [] { const char* v = std::getenv("STRATA_WMMA_GEMM"); return v && v[0] != 0 && v[0] != '0'; }();
    static const bool pf_on = [] { const char* v = std::getenv("STRATA_PF_GEMM"); return v && v[0] == '1'; }();
    static const int64_t pf_min_t = [] { const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T"); return v ? (int64_t) std::atoll(v) : (int64_t) 0; }();
    if (pf_on && T >= pf_min_t && strata_pf_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream_)) return;   // S23: opt-in
    if (wmma_on && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream_)) {
        return;
    }
#endif
#if defined(__HIPCC__)
    if (f16_io_ && beta == 0.0f) { f16_inplace(X, W, Y, T, N, K, ldy); return; }
#endif
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::f16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt f16");
        return;
    }
#endif
    if (staged_gemm(*strata::q_of(stream_), handle_, W, (int) K, X, (int) K, dpct::library_data_t::real_half, N, T, K, alpha, beta, Y, ldy))
        return;
    ck(DPCT_CHECK_ERROR(dpct::blas::gemm(
           (dpct::blas::descriptor_ptr)handle_, oneapi::mkl::transpose::trans,
           oneapi::mkl::transpose::nontrans, (int)N, (int)T, (int)K, &alpha, W,
           dpct::library_data_t::real_half, (int)K, X,
           dpct::library_data_t::real_half, (int)K, &beta, Y,
           dpct::library_data_t::real_float, (int)ldy,
           dpct::compute_type::f32)),
       "cublasGemmEx f16");
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx f16");
}

#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
bool Gemm::native_mmq(const uint16_t* X, int type, const void* W, float* Y, int64_t T, int64_t N, int64_t K,
                      int64_t ldy) {
    namespace mmq = strata::prefill::mmq;
    constexpr int64_t kRows = 1024, kMaxK = 8192;
    static const bool enabled = [] {
        const char* e = std::getenv("STRATA_DENSE_MMQ");
        return e && e[0] == '1';
    }();
    if (!enabled || mmq_failed_ || !mmq::built() || !mmq::fits(type, N)) return false;
    // K must be a multiple of 256: llama.cpp's MMQ loads the weights in 256-value K chunks, and the
    // chunk past a partial row reads past the row (the next row's bytes - or, for the last weight
    // row, bytes past the tensor, which no caller guarantees to be zeros). The MoE path is safe
    // because its gather buffers carry a zeroed tail; a dense tensor is passed as-is, so a K that
    // is not a full chunk multiple (e.g. the 640-value shared-expert down: 20 blocks of 32) stays on
    // the dequantize + cuBLAS path.
    if (K > kMaxK || K % 256 != 0) return false;
    int64_t R = scratch_elems_ / (2 * K);   // FP32 activations for a chunk fit in the FP16 dequant scratch
    if (R > kRows) R = kRows;
    if (R > T) R = T;
    if (R < 1) return false;
    if (ldy <= 0) ldy = N;

    const auto up = [](size_t v) { return (v + 255) & ~(size_t) 255; };
    const size_t ident_off = up(mmq::q8_bytes(kRows, kMaxK));
    const size_t bounds_off = ident_off + up((size_t) kRows * 4);
    if (!mmq_buf_) {
        if (cudaMalloc(&mmq_buf_, bounds_off + 256) != cudaSuccess) {
            cudaGetLastError();
            mmq_failed_ = true;
            return false;
        }
        mmq_ctx_ = new mmq::Context();
        mmq::iota((int32_t*) ((uint8_t*) mmq_buf_ + ident_off), kRows, stream_);
        std::fprintf(stderr, "prefill gemm: dense MMQ on (STRATA_DENSE_MMQ=1)\n");
    }
    void* xq = mmq_buf_;
    const int32_t* ident = (const int32_t*) ((uint8_t*) mmq_buf_ + ident_off);
    int32_t* bounds = (int32_t*) ((uint8_t*) mmq_buf_ + bounds_off);
    float* xf = (float*) scratch_;
    auto* ctx = static_cast<mmq::Context*>(mmq_ctx_);

    for (int64_t r0 = 0; r0 < T; r0 += R) {
        const int64_t rows = (T - r0 < R) ? T - r0 : R;
        mmq::f16_to_f32(X + r0 * K, xf, rows * K, stream_);
        mmq::quantize(xf, nullptr, xq, type, K, K, rows, stream_);
        mmq::set_bounds(bounds, (int32_t) rows, stream_);
        mmq::Product p;
        p.w = W;
        p.type = type;
        p.w_rows = N;
        p.w_cols = K;
        p.expert_bytes = mmq::matrix_bytes(type, N, K);
        p.n = 1;
        p.xq = xq;
        p.bounds = bounds;
        p.ids = ident;
        p.total_rows = rows;
        p.max_rows = rows;
        p.dst = Y + r0 * ldy;
        p.ld_dst = ldy;
        ctx->run(p, stream_);
    }
    return true;
}
#endif
void Gemm::f16_inplace(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
#if defined(__HIPCC__)
    const float one = 1.0f, zero = 0.0f;
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W, CUDA_R_16F,
                    (int) K, X, CUDA_R_16F, (int) K, &zero, Y, CUDA_R_16F, (int) (2 * ldy), CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16 out");
    static const bool dbg_nan = std::getenv("STRATA_DBG_NAN") != nullptr;
    widen_rows_f16<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(Y, N, ldy, dbg_nan ? 1 : 0);
    if (dbg_nan) {   // debug only: a sync per GEMM
        unsigned long long bad = 0, zero_count = 0;
        (void) hipStreamSynchronize((hipStream_t) stream_);
        if (hipMemcpyFromSymbol(&bad, HIP_SYMBOL(g_f16_nonfinite), sizeof bad) == hipSuccess && bad != 0) {
            std::fprintf(stderr, "strata prefill: STRATA_DBG_NAN: %llu non-finite FP16 GEMM outputs (T=%lld N=%lld K=%lld; FP16 ends at 65504)%s",
                         bad, (long long) T, (long long) N, (long long) K, "\n");
            (void) hipMemcpyToSymbol(HIP_SYMBOL(g_f16_nonfinite), &zero_count, sizeof zero_count);
        }
    }
#else
    (void) X; (void) W; (void) Y; (void) T; (void) N; (void) K; (void) ldy;
#endif
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta, int64_t ldx) {
#ifdef STRATA_USE_HIP
    // S23 (opt-in STRATA_PF_PAD=1, with STRATA_PF_GEMM=1 on chunks of STRATA_PF_SWITCH_MIN_T+ tokens): the weight
    // dequantized into the scratch with row stride K + 64 halves, so no 4 KB-multiple stride camps on the memory
    // channels (gemm_probe8: K 6144 19 -> 29 TFLOPS from W alone, 35 with X padded too).  Bitwise the same products.
    static const bool pad_on = [] { const char* v = std::getenv("STRATA_PF_PAD"); return v && v[0] == '1'; }();
    static const bool pf_on = [] { const char* v = std::getenv("STRATA_PF_GEMM"); return v && v[0] == '1'; }();
    static const int64_t pf_min_t = [] { const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T"); return v ? (int64_t) std::atoll(v) : (int64_t) 0; }();
    if (pad_on && pf_on && T >= pf_min_t && T >= 64 && N >= 512 && K % 32 == 0 && N * (K + 64) <= scratch_elems_) {
        const int64_t ldw = K + 64;
        if (strata::kernels::dequant_f16_ld(ggml_type, W_blocks, 0, N, K, ldw, scratch_, stream_) &&
            strata_pf_gemm_f16_ld(X, ldx > 0 ? ldx : K, scratch_, ldw, Y, T, N, K, ldy, beta, stream_)) {
            return;
        }
    }
#endif
    if (ldx > 0 && ldx != K) {
#ifdef STRATA_USE_HIP
        // X arrived padded but the weight cannot be (an i-quant without a strided dequant, or the scratch is too
        // small for N x (K + 64)): the weight unpadded, X read at its stride - the same products, the same bits
        const int64_t rows = scratch_elems_ / K;
        bool ok = rows > 0;
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; ok && r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            ok = strata_pf_gemm_f16_ld(X, ldx, scratch_, K, Y + r0, T, n, K, ldy, beta, stream_);
        }
        if (ok) return;
#endif
        std::fprintf(stderr, "prefill gemm: a padded X (ldx %lld, K %lld) needs STRATA_PF_PAD's path\n",
                     (long long) ldx, (long long) K);
        std::exit(1);
    }
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
    if (beta == 0.0f && (ldx == 0 || ldx == K) && native_mmq(X, ggml_type, W_blocks, Y, T, N, K, ldy)) return;
#endif
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
