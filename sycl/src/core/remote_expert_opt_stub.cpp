// --remote-expert-opt (Remote Expert Decode Optimization) needs a second CUDA device as an expert tier; the SYCL port
// has no such tier (the peer-expert and remote-expert helpers are single-device stubs). The class exists so the shared
// headers and generate.cpp build; init() refuses with a message, so the option reports itself unsupported.
#include "strata/core/remote_expert_opt.hpp"

namespace strata::core {

RemoteExpertOpt::~RemoteExpertOpt() = default;

void RemoteExpertOpt::attach(RemoteExperts&) {}

bool RemoteExpertOpt::init(std::string& err) {
    err = "--remote-expert-opt is not supported by the SYCL engine (no second-device expert tier)";
    return false;
}

bool RemoteExpertOpt::owns(int64_t, int32_t) const { return false; }

bool RemoteExpertOpt::adapt(const std::vector<float>&, const std::vector<int32_t>&,
                            const std::vector<std::pair<int32_t, int32_t>>&, int, ExpertSource&) {
    return false;
}

void RemoteExpertOpt::begin(const float*, int, int) {}

void RemoteExpertOpt::copy_rows(float*, const float*, int, int, const int32_t*, const int32_t*, void*) const {}

void RemoteExpertOpt::combine(float*, float*, int, int, const uint32_t*, uint32_t, void*) const {}

size_t RemoteExpertOpt::metadata_bytes() { return 0; }

void RemoteExpertOpt::prepare(const RemoteExperts&, void*) const {}

bool RemoteExpertOpt::reduce(RemoteExperts&, const void*, std::string&) { return true; }

void RemoteExpertOpt::accumulate(const RemoteExperts&) {}

}  // namespace strata::core
