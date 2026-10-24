// SYCL port: experts mirrored in pinned host memory (GgufExpertSource::mirror) in the device-built verify plan.
#pragma once
#include <cstdint>
namespace strata::kernels {
/// From now on (and in every graph captured after this call) the device-built plan (resident_plan) treats an expert
/// that is not in the VRAM cache but has a non-zero entry in `mirror_table` ([n_layers][n_expert] device-readable
/// addresses, 0 = none) as planned: the expert kernels read it from that address, over PCIe. `d_res` is the residency
/// table whose per-layer slices resident_plan receives, so the layer of a call is found from its pointer.
/// `entries` (#1980): the length of `d_res`, `mirror_table` and `usage` (n_layers x n_expert).  A call whose residency
/// slice lies outside `d_res` - a later layer-split stage's own table - gets no mirror and no counts (a plain miss): the
/// slice used to be taken as an offset from `d_res` anyway and read another allocation (xe page fault on the second card).
/// 0: unchecked, as before.
void resident_plan_set_mirror(const int32_t* d_res, const unsigned long long* mirror_table, int32_t* usage = nullptr,
                              long long entries = 0);
}
