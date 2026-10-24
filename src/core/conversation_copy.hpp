#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/progress.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <thread>
#include <vector>

namespace strata::core::conversation_detail {

/// Conversation snapshot / checkpoint copies go through a private non-blocking stream of the current device
/// instead of the legacy (null) stream.  A plain cudaMemcpy is ordered against every blocking stream of the
/// device, and while another thread captures a graph on one (a drafter or a stage worker), HIP refuses it with
/// "operation would make the legacy stream depend on a capturing blocking stream".  The caller's contract is
/// unchanged: the device was synchronized before the copy (the copy itself is complete on return).
/// private_stream: this thread's stream on the current device, created on first use.
inline cudaError_t private_stream(cudaStream_t& s) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    thread_local std::vector<cudaStream_t> streams;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= streams.size()) streams.resize((size_t) dev + 1, nullptr);   // never destroyed: process lifetime
    if (!streams[(size_t) dev]) {
        e = cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking);
        if (e != cudaSuccess) { streams[(size_t) dev] = nullptr; return e; }
    }
    s = streams[(size_t) dev];
    return cudaSuccess;
}

/// `queued`: the copy is only queued on the private stream, and the caller synchronizes the device (or orders a
/// later launch after it with order_after_queued) before anything reads the target or writes the source.
inline cudaError_t copy_nonblocking(void* dst, const void* src, size_t bytes, bool queued = false) {
    cudaStream_t s = nullptr;
    cudaError_t e = private_stream(s);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, s);
    if (e != cudaSuccess || queued) return e;
    e = cudaStreamSynchronize(s);
    core::progress_beat();   // #2049: a parked conversation is many of these; each one that lands is progress
    return e;
}

/// #2049 #2065 #1971: a host-to-host copy that tells the watchdog it is moving.  Parked state (the running
/// checkpoints, the K/V segments) sits in ordinary pageable memory for the whole session; when the OS has written
/// it to the pagefile, copying it back runs at the drive's page-in speed (measured in the reports: ~190 hard faults
/// a second, under 1 MB/s) and one memcpy of it stays silent past the 60 s limit, so the watchdog ended an engine
/// that was only slow.  One beat per 1 MiB slice keeps a copy that moves alive and still lets a real stall show.
/// STRATA_TEST_SLOW_COPY_US=N (tests only): every 1 MiB slice of a checkpoint copy takes N microseconds more, as when
/// the OS has paged the checkpoints out and each slice is read back from the drive.  Unset: no delay.
inline long test_slow_copy_us() {
    static const long v = [] { const char* e = std::getenv("STRATA_TEST_SLOW_COPY_US"); return e ? std::atol(e) : 0L; }();
    return v;
}

template<class T>
inline std::vector<T> copy_beating(const std::vector<T>& src) {
    std::vector<T> out;
    out.reserve(src.size());
    const size_t step = std::max<size_t>(1, ((size_t) 1 << 20) / sizeof(T));
    for (size_t at = 0; at < src.size(); at += step) {
        out.insert(out.end(), src.begin() + (std::ptrdiff_t) at,
                   src.begin() + (std::ptrdiff_t) std::min(src.size(), at + step));
        if (const long us = test_slow_copy_us(); us > 0) std::this_thread::sleep_for(std::chrono::microseconds(us));
        core::progress_beat();
    }
    return out;
}

inline ConversationCheckpoint copy_checkpoint_beating(const ConversationCheckpoint& c) {
    ConversationCheckpoint o;
    o.ids = copy_beating(c.ids);
    o.imgs = copy_beating(c.imgs);
    o.gdn = copy_beating(c.gdn);
    o.ple = copy_beating(c.ple);
    o.tails = copy_beating(c.tails);
    o.dead = copy_beating(c.dead);
    o.block_pos = copy_beating(c.block_pos);
    o.used = c.used;
    o.pinned = c.pinned;
    o.stage_parts.reserve(c.stage_parts.size());
    for (const auto& part : c.stage_parts) o.stage_parts.push_back(copy_checkpoint_beating(part));
    return o;
}

inline std::vector<ConversationCheckpoint> copy_checkpoints_beating(const std::vector<ConversationCheckpoint>& cs) {
    std::vector<ConversationCheckpoint> out;
    out.reserve(cs.size());
    for (const auto& c : cs) out.push_back(copy_checkpoint_beating(c));
    return out;
}

/// Work queued on the legacy stream after this call starts after every copy this thread queued on the current
/// device's private stream (an event the legacy stream waits for; no host wait).
inline cudaError_t order_after_queued() {
    cudaStream_t s = nullptr;
    cudaError_t e = private_stream(s);
    if (e != cudaSuccess) return e;
    int dev = 0;
    e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    thread_local std::vector<cudaEvent_t> events;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= events.size()) events.resize((size_t) dev + 1, nullptr);    // never destroyed: process lifetime
    if (!events[(size_t) dev]) {
        e = cudaEventCreateWithFlags(&events[(size_t) dev], cudaEventDisableTiming);
        if (e != cudaSuccess) { events[(size_t) dev] = nullptr; return e; }
    }
    e = cudaEventRecord(events[(size_t) dev], s);
    if (e != cudaSuccess) return e;
    return cudaStreamWaitEvent(nullptr, events[(size_t) dev], 0);
}

} // namespace strata::core::conversation_detail
