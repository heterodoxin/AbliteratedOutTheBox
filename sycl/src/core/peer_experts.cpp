// src/core/peer_experts.cpp (SYCL port) - upstream's --peer-device tier (a second GPU as an expert-cache tier, 0.1.36+)
// is CUDA-only for now: this build answers "no peer". open() refuses with a message, so --peer-device fails clearly,
// and every other entry point is the state a run without --peer-device has. Upstream's file: src/core/peer_experts.cpp.
#include "strata/core/peer_experts.hpp"

#include <atomic>

namespace strata::core {

namespace {
std::atomic<bool> g_portable{false};
}  // namespace

void set_peer_portable(bool on) { g_portable.store(on); }
bool peer_portable() { return g_portable.load(); }

PeerExperts::~PeerExperts() { close(); }

bool PeerExperts::open(int, const std::vector<std::pair<int32_t, int32_t>>&, const ExpertCache&, ExpertSource&, int64_t,
                       int64_t, int, int64_t, std::string& err) {
    err = "--peer-device: the second-GPU expert tier is not ported to SYCL yet";
    return false;
}

void PeerExperts::close() { device_ = -1; }

bool PeerExperts::launch(int64_t, const float*, const int32_t*, int64_t, int64_t, const int32_t*, std::string& err,
                         float*) {
    err = "--peer-device: not available in the SYCL build";
    return false;
}

bool PeerExperts::finish(float*, std::string& err) {
    err = "--peer-device: not available in the SYCL build";
    return false;
}

bool PeerExperts::adapt(const float*, const int32_t*, int, std::string&) { return true; }
void PeerExperts::apply_pending(bool) {}

}  // namespace strata::core
