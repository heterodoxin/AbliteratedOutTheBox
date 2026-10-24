// src/prefill/moe_fused_stub.cpp (SYCL port) - upstream's fused int8 prompt kernels (moe_fused.cu, moe_fused_iq.cu, #136)
// belong to its MMQ library, which this build does not have (the prompt path's MMQ plan is empty here, so prefill.cpp
// never takes the fused branch). These are the "not built" answers, so the prompt path links and keeps its own path.
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_iq.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::fused {

namespace {
[[noreturn]] void unreachable(const char* what) {
    std::fprintf(stderr, "prefill: %s called in a build without the fused kernels\n", what);
    std::abort();
}
}  // namespace

bool built() { return false; }
bool available() { return false; }
bool enabled() { return false; }
bool requested() { return false; }
size_t act_bytes(int64_t, int64_t) { return 0; }
size_t group_bytes(int64_t, int) { return 0; }
void quantize_act(const float*, int64_t, int64_t, void*, void*) { unreachable("fused::quantize_act"); }
void group(const int32_t*, int64_t, int, int, void*, int32_t*, int32_t*, void*) { unreachable("fused::group"); }
void experts(const Batch&, int, int64_t, const void*, const void*, const int32_t*, void*, float*, void*) {
    unreachable("fused::experts");
}
bool native_supported(int, int) { return false; }
void quantize_act_native(const float*, int64_t, int64_t, void*, void*) { unreachable("fused::quantize_act_native"); }
void experts_native(const Batch&, const NativeGeom&, int, int64_t, const void*, const void*, const int32_t*, void*, float*,
                    void*) {
    unreachable("fused::experts_native");
}

}  // namespace strata::prefill::fused
