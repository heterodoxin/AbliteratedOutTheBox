// src/core/copy_queue.cpp - see include/strata/copy_queue.hpp.
#include "strata/copy_queue.hpp"
#include "strata/sycl_queue.hpp"

#include <dpct/dpct.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#if defined(__has_include)
#if __has_include(<level_zero/ze_api.h>)
#include <level_zero/ze_api.h>
#define STRATA_HAVE_ZE_API 1
#endif
#endif

#include <atomic>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef STRATA_HAVE_ZE_API
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace strata {
namespace {
std::atomic<bool> g_active{false};

#ifdef STRATA_HAVE_ZE_API
struct ZeFns {
    decltype(&zeDeviceGetCommandQueueGroupProperties) get_groups = nullptr;
    decltype(&zeCommandListCreateImmediate) create_imm = nullptr;
};
bool load_ze(ZeFns& f) {
#if defined(_WIN32)
    HMODULE h = LoadLibraryA("ze_loader.dll");
    if (!h) return false;
    auto sym = [&](const char* n) { return (void*) GetProcAddress(h, n); };
#else
    void* h = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_NOLOAD);   // already loaded by the SYCL runtime
    if (!h) h = dlopen("libze_loader.so.1", RTLD_NOW);
    if (!h) return false;
    auto sym = [&](const char* n) { return dlsym(h, n); };
#endif
    f.get_groups = (decltype(f.get_groups)) sym("zeDeviceGetCommandQueueGroupProperties");
    f.create_imm = (decltype(f.create_imm)) sym("zeCommandListCreateImmediate");
    return f.get_groups && f.create_imm;
}
#endif
}  // namespace

bool copy_engine_active() { return g_active.load(); }

namespace {
// one copy-engine queue per device (a layer split calls this once per card): device -> queue; a device that has none maps to nullptr
struct DevQ { sycl::device dev; sycl::queue* q; };
std::vector<DevQ>& dev_queues() { static std::vector<DevQ> v; return v; }
std::mutex& dev_mu() { static std::mutex m; return m; }
}  // namespace
bool is_copy_engine_queue(const sycl::queue* q) {
    if (q == nullptr) return false;
    std::lock_guard<std::mutex> lk(dev_mu());
    for (const DevQ& d : dev_queues()) if (d.q == q) return true;
    return false;
}


sycl::queue* make_copy_engine_queue(const char* what) {
    static const bool off = [] { const char* v = std::getenv("STRATA_COPY_ENGINE"); return v && v[0] == '0'; }();
    if (off) return nullptr;
#ifndef STRATA_HAVE_ZE_API
    (void) what;
    return nullptr;
#else
    std::lock_guard<std::mutex> lk(dev_mu());
    sycl::device dev = dpct::get_current_device();
    for (const DevQ& d : dev_queues()) if (d.dev == dev) return d.q;   // asked before for this card (nullptr: it has none)
    dev_queues().push_back({dev, nullptr});
    const size_t slot = dev_queues().size() - 1;
    try {

        if (dev.get_backend() != sycl::backend::ext_oneapi_level_zero) return nullptr;
        // Battlemage only unless STRATA_COPY_ENGINE=1: on an Arc A750 (i915, JIT) the prompt path hung with it once the stager's
        // completion flag became a copy-engine memcpy (4K and 600-token prompts, 10+ minutes, GPU not hung), and the measured gain
        // there was 1.4% (a prompt there is compute bound).
        {
            const char* force = std::getenv("STRATA_COPY_ENGINE");
            if (!(force && force[0] == '1') && strata::intel_gpu_gen(dev) != strata::IntelGpuGen::Battlemage) return nullptr;
        }
        // The legacy Level Zero adapter already puts memcpy on the copy engine (and its kernel launches are slower); only
        // the v2 adapter needs this.  SYCL_UR_USE_LEVEL_ZERO_V2=0 selects the legacy one.
        const char* v2 = std::getenv("SYCL_UR_USE_LEVEL_ZERO_V2");
        if (v2 && v2[0] == '0') return nullptr;
        ZeFns zf;
        if (!load_ze(zf)) return nullptr;
        sycl::context ctx = dpct::get_default_context();
        auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
        auto zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
        uint32_t ng = 0;
        if (zf.get_groups(zdev, &ng, nullptr) != ZE_RESULT_SUCCESS || ng == 0) return nullptr;
        std::vector<ze_command_queue_group_properties_t> g(ng);
        for (auto& x : g) { std::memset(&x, 0, sizeof(x)); x.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES; }
        if (zf.get_groups(zdev, &ng, g.data()) != ZE_RESULT_SUCCESS) return nullptr;
        int ord = -1;
        for (uint32_t i = 0; i < ng; ++i)
            if ((g[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
                !(g[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) && g[i].numQueues > 0) { ord = (int) i; break; }
        if (ord < 0) return nullptr;
        ze_command_queue_desc_t d;
        std::memset(&d, 0, sizeof(d));
        d.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        d.ordinal = (uint32_t) ord;
        d.index = 0;
        d.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
        d.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
        d.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
        ze_command_list_handle_t cl = nullptr;
        if (zf.create_imm(zctx, zdev, &d, &cl) != ZE_RESULT_SUCCESS || cl == nullptr) return nullptr;
        sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> in(
            cl, dev, sycl::ext::oneapi::level_zero::ownership::keep,
            sycl::property_list{sycl::property::queue::in_order{}});
        auto* q = new sycl::queue(sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(in, ctx));
        dev_queues()[slot].q = q;
        g_active.store(true);
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            std::fprintf(stderr, "strata: copy engine: %s copies run on the GPU's copy engine (queue group %d), not the "
                                 "compute engine (STRATA_COPY_ENGINE=0 turns it off)\n", what, ord);
        return q;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata: copy engine queue not available (%s): ordinary queue\n", e.what());
        return nullptr;
    }
#endif
}

}  // namespace strata
