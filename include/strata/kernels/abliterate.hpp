#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels {

// Configure before graph capture. All pointers stay fixed across requests.
bool abliterate_init(int layers, int width, int streams, std::string& err);
void abliterate_free();
// Chronological (band layers, tokens, width), averaged over residual streams.
bool abliterate_read(int take, std::vector<float>& output, std::string& err);
bool abliterate_covers(int64_t layer);
void abliterate_apply(float* residual, int64_t layer, int64_t tokens, int64_t stride,
                     const float* block_out, int64_t block_stride, const float* inject,
                     int64_t inject_stride, bool write, void* stream);
// Returns true for a handled ABL command; replies use stdout, like GEN.
bool abliterate_command(const std::string& line);

}  // namespace strata::kernels
