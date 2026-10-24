// src/core/vmm.cpp - SYCL port: no virtual memory management.
//
// Upstream's vmm.cpp reaches the CUDA driver's cuMem* functions through cudaGetDriverEntryPointByVersion; Level Zero
// has no equivalent through dpct.  Every entry point therefore reports "not available", the same answer a GPU
// without virtual memory management gives, so --vram-elastic and the elastic K/V (--kv-grow) refuse with their
// usual message and nothing else changes.
#include "strata/core/vmm.hpp"

namespace strata::core {

bool vmm_available() { return false; }
uint64_t vmm_granularity() { return 0; }
VmmChunk vmm_chunk_new() { return 0; }
void vmm_chunk_free(VmmChunk) {}

bool VmmRange::reserve(uint64_t) { return false; }
void VmmRange::release() {
    h_.clear();
    base_ = 0;
}
int64_t VmmRange::mapped_count() const { return 0; }
VmmChunk VmmRange::unmap(int64_t) { return 0; }
bool VmmRange::map_one(int64_t, VmmChunk) { return false; }
bool VmmRange::set_access(int64_t, int64_t) { return false; }
bool VmmRange::commit_run(int64_t, int64_t) { return false; }

}  // namespace strata::core
