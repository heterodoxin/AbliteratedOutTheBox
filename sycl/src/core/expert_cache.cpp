#define DPCT_COMPAT_RT_VERSION 12080
// src/core/expert_cache.cpp - R4's slot storage and residency table.  Read the
// header first.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/expert_cache.hpp"

// #533's segmented cache uses CUDA's virtual memory management (cuMem*): not on HIP, neither the RDNA backend nor
// the gfx906 compat build (PR #638), which compiles this file as HIP without STRATA_USE_HIP
#define STRATA_EC_NO_VMM 1   // SYCL port: no driver virtual memory management on Level Zero (--vram-elastic is CUDA-only, #533)

#if !defined(STRATA_EC_NO_VMM)
   // #533: the virtual memory management types (the functions come through the
   // runtime's entry points)
#endif

#include <algorithm>
#include <atomic>
#include <set>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <utility>
#include <cstring>

namespace strata::core {

// Local (card) memory this process holds, from the DRM clients' fdinfo (i915: drm-total-local0, xe: drm-total-vram0).
// Level Zero's free-memory query reads the constant total on an Arc A750 (i915): the engine then sized its caches
// from the whole card and over-allocated (UR_RESULT_ERROR_OUT_OF_HOST_MEMORY at the first launch). Other processes'
// memory is not in this number, so it only ever LOWERS the figure the driver gave.  0 = not available.
static unsigned long long own_drm_local_bytes() {
#if defined(__linux__)
    unsigned long long total = 0;
    std::error_code ec;
    std::set<unsigned long long> seen;   // one DRM client may appear under several fds
    for (const auto& e : std::filesystem::directory_iterator("/proc/self/fdinfo", ec)) {
        std::FILE* f = std::fopen(e.path().string().c_str(), "r");
        if (f == nullptr) continue;
        char line[256];
        bool drm = false;
        unsigned long long client = 0, bytes = 0;
        while (std::fgets(line, sizeof line, f)) {
            unsigned long long v = 0;
            char unit[8] = {0};
            if (std::strncmp(line, "drm-driver:", 11) == 0) drm = true;
            else if (std::sscanf(line, "drm-client-id: %llu", &client) == 1) {}
            else if ((std::sscanf(line, "drm-total-local0: %llu %7s", &v, unit) >= 1 ||
                      std::sscanf(line, "drm-total-vram0: %llu %7s", &v, unit) >= 1) && v > 0) {
                const unsigned long long mul = unit[0] == 'K' ? 1024ull : unit[0] == 'M' ? 1048576ull
                                             : unit[0] == 'G' ? 1073741824ull : 1ull;
                bytes += v * mul;
            }
        }
        std::fclose(f);
        if (drm && bytes > 0 && seen.insert(client).second) total += bytes;
    }
    return total;
#else
    return 0;
#endif
}

// See the header.  NVIDIA's DGX Spark Porting Guide (section 5.5, "Memory reporting on UMA systems") recommends the same:
// not to rely on cudaMemGetInfo alone but to count the memory the OS can reclaim.  Swap is not counted here (unlike
// NVIDIA's reference snippet): an expert cache that pushes the system into swap would be far slower than a smaller one.
size_t device_free_bytes() try {
    size_t free_b = 0, total_b = 0;
    /*
    DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    dpct::get_current_device().get_memory_info(free_b, total_b);
    // An Arc on the xe driver whose Level Zero has no ext_intel_free_memory reports 0 free (Arc A770 16 GB, xe, #1545):
    // --expert-cache auto then sized its caches from 0 bytes.  0 with a known total means "the driver does not know", so on
    // Linux (xe is Linux-only) the card's total less what this process holds (the DRM fdinfo, as for the A750/i915 below)
    // is used instead.  Like that figure it does not see other processes' memory.  Elsewhere the 0 stands.
#if defined(__linux__)
    if (free_b == 0 && total_b > 0)
        if (const unsigned long long own = own_drm_local_bytes(); own < total_b) {
            static std::atomic<bool> said{false};
            if (!said.exchange(true))
                std::fprintf(stderr, "strata: the driver reports no free card memory (xe without ext_intel_free_memory); sizing from the DRM fdinfo (this process holds %.2f GiB)\n", (double) own / 1073741824.0);
            free_b = (size_t) (total_b - own);
        }
#endif
    // #1440: own_drm_local_bytes() sums every DRM fd of the process, i.e. both cards on a layer split, and would charge the
    // first card's cache to the second. With more than one GPU it runs only where the driver reports the whole card as
    // free (Arc A750/i915); a one-GPU run keeps the figure it always had.
    if (const unsigned long long own = (free_b + (16ull << 20) >= total_b || dpct::device_count() <= 1)
                                           ? own_drm_local_bytes() : 0ull;
        own > 0 && total_b > own && total_b - own < free_b) {
        static std::atomic<bool> said{false};
        if (free_b + (16ull << 20) >= total_b && !said.exchange(true))
            std::fprintf(stderr, "strata: the driver reports the whole card as free although this process holds %.2f GiB of it (Arc A750/i915 does this); sizing from the DRM fdinfo instead\n", (double) own / 1073741824.0);
        free_b = (size_t) (total_b - own);
    }
#if defined(__linux__) && !defined(STRATA_HIP_GFX906)   // (the gfx906 compat layer has no cudaDevAttrIntegrated; that GPU is discrete)
    // per device: a box can mix an integrated GPU (an APU) with a discrete one, and the answer is the CURRENT device's
    static std::atomic<int> uma_cache[64];   // 0 unknown, 1 integrated (unified memory), 2 discrete
    bool unified_memory = false;
    {
        int dev = 0, v = 0;
        if (DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) == 0 &&
            dev >= 0 && dev < 64) {
            int c = uma_cache[dev].load(std::memory_order_acquire);
            if (c == 0) {
                c = DPCT_CHECK_ERROR(
                        v = dpct::get_device(dev).get_integrated()) == 0 &&
                            v
                        ? 1
                        : 2;
                uma_cache[dev].store(c, std::memory_order_release);
            }
            unified_memory = c == 1;
        }
    }
    if (unified_memory) {
        if (FILE* m = std::fopen("/proc/meminfo", "r")) {
            char line[256];
            unsigned long long kb = 0;
            while (std::fgets(line, sizeof line, m))
                if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
            std::fclose(m);
            // STRATA_UMA_HEADROOM_GIB: a whole number of GiB, 0..1024; anything else keeps the default 6 (said once)
            static const long gib = [] {
                const char* h = std::getenv("STRATA_UMA_HEADROOM_GIB");
                if (h == nullptr) return 6L;
                char* end = nullptr;
                const long v = std::strtol(h, &end, 10);
                if (end != h && *end == '\0' && v >= 0 && v <= 1024) return v;
                std::fprintf(stderr, "strata: STRATA_UMA_HEADROOM_GIB=%s is not a whole number of GiB (0-1024): using 6\n", h);
                return 6L;
            }();
            const unsigned long long head = (unsigned long long) gib << 30;
            const unsigned long long avail = kb << 10;
            if (avail > head && avail - head > free_b) free_b = (size_t) (avail - head);
        }
    }
#endif
    return free_b;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "read_expert_profile: cannot open " + path;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " is too short to hold a header";
        return false;
    }
    if (std::memcmp(magic, "STRP", 4) != 0) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " does not start with STRP";
        return false;
    }
    const uint32_t version = hdr[0], nl = hdr[1], ne = hdr[2], want = hdr[3], n_ranked = hdr[4];
    if ((int64_t) nl != n_layers || (int64_t) ne != n_expert) {
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is %ux%u but this model is %lldx%lld - it is a profile for a "
                      "different artifact", path.c_str(), nl, ne, (long long) n_layers, (long long) n_expert);
        err = buf;
        return false;
    }
    if (n_ranked > want) {
        std::fclose(f);
        err = "read_expert_profile: the header claims more ranked pairs than slots";
        return false;
    }
    ranked.assign(n_ranked, {0, 0});
    std::vector<uint16_t> raw((size_t) n_ranked * 2);
    if (n_ranked > 0 && std::fread(raw.data(), 2, (size_t) n_ranked * 2, f) != (size_t) n_ranked * 2) {
        std::fclose(f);
        err = "read_expert_profile: the ranked list is truncated";
        return false;
    }
    std::fclose(f);
    for (uint32_t i = 0; i < n_ranked; ++i) {
        const int32_t l = (int32_t) raw[(size_t) i * 2], e = (int32_t) raw[(size_t) i * 2 + 1];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "read_expert_profile: pair %u is (layer %d, expert %d), out of range",
                          i, l, e);
            err = buf;
            return false;
        }
        ranked[(size_t) i] = {l, e};
    }
    slots = (int64_t) want;
    (void) version;   // a future format bumps it; the layout check above is what protects this reader today
    return true;
}

std::vector<std::pair<int32_t, int32_t>> rank_learned_profile(int64_t n_layers, int64_t n_expert,
                                                              const std::vector<uint8_t>& resident,
                                                              const std::vector<double>& heat,
                                                              const std::vector<std::pair<int32_t, int32_t>>& prior) {
    const size_t n = (size_t) (n_layers * n_expert);
    std::vector<int64_t> prior_rank(n, INT64_MAX);
    for (size_t r = 0; r < prior.size(); ++r) {
        const auto [l, e] = prior[r];
        if (l >= 0 && l < n_layers && e >= 0 && e < n_expert) {
            int64_t& pr = prior_rank[(size_t) (l * n_expert + e)];
            if (pr == INT64_MAX) pr = (int64_t) r;
        }
    }
    std::vector<int64_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = (int64_t) i;
    auto res = [&](int64_t i) { return (size_t) i < resident.size() && resident[(size_t) i] != 0; };
    auto ht = [&](int64_t i) { return (size_t) i < heat.size() ? heat[(size_t) i] : 0.0; };
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        if (res(a) != res(b)) return res(a);
        if (ht(a) != ht(b)) return ht(a) > ht(b);
        if (prior_rank[(size_t) a] != prior_rank[(size_t) b]) return prior_rank[(size_t) a] < prior_rank[(size_t) b];
        return a < b;
    });
    std::vector<std::pair<int32_t, int32_t>> ranked(n);
    for (size_t r = 0; r < n; ++r)
        ranked[r] = {(int32_t) (order[r] / n_expert), (int32_t) (order[r] % n_expert)};
    return ranked;
}

bool write_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                          const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    if (n_layers <= 0 || n_expert <= 0 || n_layers > 65535 || n_expert > 65535) {
        err = "write_expert_profile: the model's layout does not fit the format";
        return false;
    }
    std::vector<int32_t> table((size_t) (n_layers * n_expert), -1);
    std::vector<uint16_t> pairs;
    pairs.reserve(ranked.size() * 2);
    for (size_t r = 0; r < ranked.size(); ++r) {
        const auto [l, e] = ranked[r];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            err = "write_expert_profile: a ranked pair is out of range";
            return false;
        }
        table[(size_t) (l * n_expert + e)] = (int32_t) r;
        pairs.push_back((uint16_t) l);
        pairs.push_back((uint16_t) e);
    }
    const uint32_t hdr[5] = {1u, (uint32_t) n_layers, (uint32_t) n_expert, (uint32_t) ranked.size(),
                             (uint32_t) ranked.size()};
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        err = "write_expert_profile: cannot create " + tmp;
        return false;
    }
    // the format is little-endian (make_profile.py's "<"): so is every machine this engine runs on
    bool ok = std::fwrite("STRP", 1, 4, f) == 4 && std::fwrite(hdr, 4, 5, f) == 5 &&
              (pairs.empty() || std::fwrite(pairs.data(), 2, pairs.size(), f) == pairs.size()) &&
              std::fwrite(table.data(), 4, table.size(), f) == table.size();
    ok = (std::fclose(f) == 0) && ok;
    std::error_code ec;
    if (ok) std::filesystem::rename(tmp, path, ec);   // replaces an existing file (MoveFileEx / rename(2))
    if (!ok || ec) {
        std::filesystem::remove(tmp, ec);
        err = "write_expert_profile: cannot write " + path;
        return false;
    }
    return true;
}

ExpertCache::~ExpertCache() { close(); }

// ---- #533: the segmented arena (--vram-elastic).  The driver API's virtual memory functions, looked up through the
// runtime (no link against the driver library): one address range for the whole arena, backed by physical segments,
// and the tail's segments unmapped / mapped again later.  Nothing here runs unless a segment size was set.
#if !defined(STRATA_EC_NO_VMM)
namespace {
struct Vmm {
    int(CUDAAPI *device_get)(int *, int) = nullptr;
    int(CUDAAPI *attribute)(int *, CUdevice_attribute, int) = nullptr;
    int(CUDAAPI *granularity)(
        size_t *, const dpct::experimental::mem_prop *,
        sycl::ext::oneapi::experimental::granularity_mode) = nullptr;
    int(CUDAAPI *reserve)(dpct::device_ptr *, size_t, size_t, dpct::device_ptr,
                          unsigned long long) = nullptr;
    int(CUDAAPI *address_free)(dpct::device_ptr, size_t) = nullptr;
    int(CUDAAPI *create)(dpct::experimental::physical_mem_ptr *, size_t,
                         const dpct::experimental::mem_prop *,
                         unsigned long long) = nullptr;
    int(CUDAAPI *release)(dpct::experimental::physical_mem_ptr) = nullptr;
    int(CUDAAPI *map)(dpct::device_ptr, size_t, size_t,
                      dpct::experimental::physical_mem_ptr,
                      unsigned long long) = nullptr;
    int(CUDAAPI *unmap)(dpct::device_ptr, size_t) = nullptr;
    int(CUDAAPI *set_access)(dpct::device_ptr, size_t,
                             const dpct::experimental::mem_access_desc *,
                             size_t) = nullptr;
    bool ok = false;
};

template <class F> bool entry(const char *name, F &f) try {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
#if DPCT_COMPAT_RT_VERSION >= 12050
    /*
    DPCT1007: Migration of cudaGetDriverEntryPointByVersion is not
    supported.
    */
    const dpct::err0 e = cudaGetDriverEntryPointByVersion(
        name, &p, 12000, cudaEnableDefault, &q);
#else
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != 0 || q != cudaDriverEntryPointSuccess || p == nullptr) {
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        (void)0;
        return false;
    }
    f = reinterpret_cast<F>(p);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

const Vmm& vmm() {
    static const Vmm v = [] {
        Vmm x;
        x.ok = entry("cuDeviceGet", x.device_get) && entry("cuDeviceGetAttribute", x.attribute) &&
               entry("cuMemGetAllocationGranularity", x.granularity) && entry("cuMemAddressReserve", x.reserve) &&
               entry("cuMemAddressFree", x.address_free) && entry("cuMemCreate", x.create) &&
               entry("cuMemRelease", x.release) && entry("cuMemMap", x.map) && entry("cuMemUnmap", x.unmap) &&
               entry("cuMemSetAccess", x.set_access);
        return x;
    }();
    return v;
}

dpct::experimental::mem_prop device_prop(int dev) {
    dpct::experimental::mem_prop prop{};
    prop.type = 0;
    prop.location.type = 1;
    prop.location.id = dev;
    return prop;
}

// one segment: a physical allocation mapped at `va`, readable and writable by this device
bool map_segment(const Vmm &v, int dev, dpct::device_ptr va, size_t bytes,
                 unsigned long long &handle) {
    const dpct::experimental::mem_prop prop = device_prop(dev);
    dpct::experimental::physical_mem_ptr h = 0;
    if (v.create(&h, bytes, &prop, 0) != 0) return false;
    if (v.map(va, bytes, 0, h, 0) != 0) {
        v.release(h);
        return false;
    }
    dpct::experimental::mem_access_desc access{};
    access.location.type = 1;
    access.location.id = dev;
    access.flags =
        sycl::ext::oneapi::experimental::address_access_mode::read_write;
    if (v.set_access(va, bytes, &access, 1) != 0) {
        v.unmap(va, bytes);
        v.release(h);
        return false;
    }
    handle = (unsigned long long) h;
    return true;
}
}  // namespace
#endif

bool ExpertCache::open_segmented(uint64_t want, std::string &err) try {
#if defined(STRATA_EC_NO_VMM)
    (void) want;
    err = "ExpertCache: --vram-elastic (a segmented expert cache) is CUDA-only for now";
    return false;
#else
    const Vmm& v = vmm();
    int dev = 0, supported = 0;
    int cu = 0;
    if (!v.ok || DPCT_CHECK_ERROR(dev = dpct::get_current_device_id()) != 0 ||
        v.device_get(&cu, dev) != 0 ||
        v.attribute(&supported,
                    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                    cu) != 0 ||
        !supported) {
        err = "ExpertCache: --vram-elastic needs the driver's virtual memory management, which this GPU or driver "
              "does not offer";
        return false;
    }
    const dpct::experimental::mem_prop prop = device_prop(dev);
    size_t gran = 0;
    if (v.granularity(
            &gran, &prop,
            sycl::ext::oneapi::experimental::granularity_mode::recommended) !=
            0 ||
        gran == 0) {
        err = "ExpertCache: cannot read the driver's allocation granularity";
        return false;
    }
    const uint64_t g = (uint64_t) gran;
    const uint64_t total = (want + g - 1) / g * g;
    seg_ = (int64_t) (((uint64_t) seg_req_ + g - 1) / g * g);
    dpct::device_ptr va = 0;
    if (v.reserve(&va, (size_t)total, 0, 0, 0) != 0) {
        err = "ExpertCache: cannot reserve the address range of the segmented expert cache";
        return false;
    }
    base_ = reinterpret_cast<uint8_t*>(va);
    reserved_ = total;
    for (uint64_t at = 0; at < total; at += (uint64_t) seg_) {
        segs_.push_back(0);
        seg_size_.push_back((int64_t) std::min<uint64_t>((uint64_t) seg_, total - at));
    }
    for (size_t i = 0; i < segs_.size(); ++i) {
        if (!map_segment(v, dev,
                         va + (dpct::device_ptr)((uint64_t)i * (uint64_t)seg_),
                         (size_t)seg_size_[i], segs_[i])) {
            char buf[200];
            std::snprintf(buf, sizeof buf, "ExpertCache: cudaMalloc failed: segment %zu of %zu (%.2f GiB) of the "
                          "segmented cache could not be allocated", i + 1, segs_.size(),
                          (double) seg_size_[i] / 1073741824.0);
            err = buf;   // "cudaMalloc failed": the auto cache's smaller-retry path reads it as an allocation failure
            release_segmented();
            return false;
        }
        mapped_segs_ = (int64_t) i + 1;
    }
    // the reserved VA, so device_offset_end can place a borrowed prompt buffer or a resident expert inside it (the GEMM
    // stages an operand that lies more than 4 GiB in; oneMKL misreads such an operand's last partial tile)
    strata::device_alloc_register(base_, (size_t) reserved_);
    return true;
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void ExpertCache::release_segmented() {
#if !defined(STRATA_EC_NO_VMM)
    const Vmm& v = vmm();
    if (base_ != nullptr) dpct::get_current_device().queues_wait_and_throw();
    const dpct::device_ptr va = reinterpret_cast<dpct::device_ptr>(base_);
    for (size_t i = 0; i < segs_.size(); ++i)
        if (segs_[i] != 0) {
            v.unmap(va + (dpct::device_ptr)((uint64_t)i * (uint64_t)seg_),
                    (size_t)seg_size_[i]);
            v.release((dpct::experimental::physical_mem_ptr)segs_[i]);
        }
    if (base_ != nullptr && reserved_ > 0) v.address_free(va, (size_t) reserved_);
#endif
    segs_.clear();
    seg_size_.clear();
    mapped_segs_ = 0;
    reserved_ = 0;
    strata::device_alloc_unregister(base_);
    base_ = nullptr;
}

int64_t ExpertCache::mapped_bytes() const {
    if (segs_.empty()) return base_ != nullptr ? full_bytes() : 0;
    int64_t b = 0;
    for (int64_t i = 0; i < mapped_segs_; ++i) b += seg_size_[(size_t) i];
    return b;
}

int64_t ExpertCache::slots_within(int64_t bytes) const {
    if (bytes >= full_bytes()) return slots_;
    if (bytes <= 0) return 0;
    if (off_.empty()) return blob_ > 0 ? bytes / blob_ : 0;
    // off_[i + 1] is slot i's end: count the slots whose end is at most `bytes`
    return (int64_t) (std::upper_bound(off_.begin() + 1, off_.end(), (uint64_t) bytes) - (off_.begin() + 1));
}

bool ExpertCache::shrink(int64_t keep_bytes, std::string &err) try {
    if (segs_.empty()) {
        err = "the expert cache is not segmented (the engine needs --vram-elastic)";
        return false;
    }
#if defined(STRATA_EC_NO_VMM)
    (void) keep_bytes;
    return false;
#else
    const Vmm& v = vmm();
    int64_t keep = 0, at = 0;   // the segments [0, keep) hold the first keep_bytes
    while (keep < (int64_t) segs_.size() && at < keep_bytes) at += seg_size_[(size_t) keep++];
    if (keep >= mapped_segs_) return true;
    if (DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()) !=
        0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1010: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        err = std::string("the device failed before the cache shrank: ") +
              dpct::get_error_string_dummy(0);
        return false;
    }
    const dpct::device_ptr va = reinterpret_cast<dpct::device_ptr>(base_);
    for (int64_t i = mapped_segs_ - 1; i >= keep; --i) {
        const dpct::device_ptr p =
            va + (dpct::device_ptr)((uint64_t)i * (uint64_t)seg_);
        if (v.unmap(p, (size_t)seg_size_[(size_t)i]) != 0 ||
            v.release((dpct::experimental::physical_mem_ptr)segs_[(size_t)i]) !=
                0) {
            err = "the driver refused to release an expert-cache segment";
            live_slots_ = slots_within(mapped_bytes());
            return false;
        }
        segs_[(size_t) i] = 0;
        mapped_segs_ = i;
    }
    live_slots_ = slots_within(mapped_bytes());
    return true;
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool ExpertCache::grow(int64_t want_bytes, std::string& err) {
    if (segs_.empty()) {
        err = "the expert cache is not segmented (the engine needs --vram-elastic)";
        return false;
    }
#if defined(STRATA_EC_NO_VMM)
    (void) want_bytes;
    return false;
#else
    const Vmm& v = vmm();
    int dev = 0;
    dev = dpct::get_current_device_id();
    const dpct::device_ptr va = reinterpret_cast<dpct::device_ptr>(base_);
    int64_t at = mapped_bytes();
    while (mapped_segs_ < (int64_t) segs_.size() && at + seg_size_[(size_t) mapped_segs_] <= want_bytes) {
        const size_t i = (size_t) mapped_segs_;
        if (!map_segment(v, dev,
                         va + (dpct::device_ptr)((uint64_t)i * (uint64_t)seg_),
                         (size_t)seg_size_[i], segs_[i])) {
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not use
            the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            (void)0;
            err = "the driver has no VRAM for another expert-cache segment";
            live_slots_ = slots_within(mapped_bytes());
            return false;
        }
        at += seg_size_[i];
        ++mapped_segs_;
    }
    live_slots_ = slots_within(mapped_bytes());
    return true;
#endif
}

#if defined(STRATA_USE_HIP)
bool ExpertCache::ensure_blocking_staging(std::size_t bytes, std::string& err) {
    if (bytes <= blocking_staging_bytes_) return true;
    void* next = nullptr;
    const cudaError_t status = cudaHostAlloc(&next, bytes, cudaHostAllocDefault);
    if (status != cudaSuccess) {
        err = std::string("ExpertCache: HIP blocking staging allocation: ") + cudaGetErrorString(status);
        return false;
    }
    if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
    blocking_staging_ = static_cast<uint8_t*>(next);
    blocking_staging_bytes_ = bytes;
    return true;
}
#endif

namespace {
bool g_cache_vmm = false;   // set_vmm
}  // namespace

void ExpertCache::set_vmm(bool enabled) { g_cache_vmm = enabled; }

bool ExpertCache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                       std::string& err) {
    close();
    if (n_slots <= 0) {
        err = "ExpertCache: n_slots must be positive";
        return false;
    }
    if (n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0) {
        err = "ExpertCache: n_layers, n_expert and blob_bytes must all be positive";
        return false;
    }

    const uint64_t want = (uint64_t) n_slots * (uint64_t) blob_bytes;

    // ---- **THE ALLOCATION IS CHECKED AGAINST THE CARD, NOT AGAINST THE REQUEST.**
    //
    // `cudaMalloc` failing is the easy case. The one that matters is a machine where the weights already own
    // most of VRAM: the cache then takes what is left and `slots()` would report the number ASKED FOR while
    // `device_slot()` walks off the end. So the free-VRAM figure is read and compared BEFORE the allocation,
    // and the two numbers are named in the refusal.
    size_t free_b = 0, total_b = 0;
    /*
    DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    if (DPCT_CHECK_ERROR(
            dpct::get_current_device().get_memory_info(free_b, total_b)) == 0) {
        free_b = device_free_bytes();   // unified memory: what the OS can give back counts (see the header)
        if ((uint64_t) free_b < want) {
            char buf[320];
            std::snprintf(buf, sizeof buf,
                          "ExpertCache: %lld slots x %lld B = %.2f GiB, but only %.2f GiB of VRAM is free "
                          "(%.2f GiB of %.2f GiB total). Lower --expert-cache.",
                          (long long) n_slots, (long long) blob_bytes, (double) want / 1073741824.0,
                          (double) free_b / 1073741824.0, (double) (total_b - free_b) / 1073741824.0,
                          (double) total_b / 1073741824.0);
            err = buf;
            return false;
        }
    }

    if (seg_req_ > 0) {   // #533: --vram-elastic: physical segments behind one address range (zeroed below)
        if (!open_segmented(want, err)) return false;
    } else if (g_cache_vmm && vmm_available()) {
        // the elastic K/V: every chunk mapped now; the K/V may later take some of them (and give them back)
        auto r = std::make_unique<VmmRange>();
        if (!r->reserve(want) || !r->map_range(0, r->chunks(), [] { return (VmmChunk) 0; })) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "ExpertCache: mapping %.2f GiB of VRAM failed: out of memory",
                          (double) want / 1073741824.0);
            err = buf;
            return false;
        }
        base_ = r->base();
        vmm_ = std::move(r);
    } else if (DPCT_CHECK_ERROR(
                   base_ = (uint8_t *)strata::malloc_device_guarded(
                       (size_t)want, dpct::get_in_order_queue(), "expert cache arena")) != 0 ||
               base_ == nullptr) {   // Level Zero answers a too-large allocation with null, not an exception
        // (an Arc A750 caps one allocation at 4 GiB unless UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1)
        base_ = nullptr;
        char buf[256];
        std::snprintf(
            buf, sizeof buf, "ExpertCache: cudaMalloc(%.2f GiB) failed: %s",
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not
            use the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            (double)want / 1073741824.0, dpct::get_error_string_dummy(0));
        err = buf;
        return false;
    }
    // Zeroed so a slot read before it is filled is a DETERMINISTIC wrong answer rather than whatever the
    // allocator handed back.  A stale block of a previous process's memory would still sum to finite floats.
    if (DPCT_CHECK_ERROR(
            (strata::big_fill_zero(dpct::get_in_order_queue(), base_, (size_t)want), 0)) !=
        0) {
        err = "ExpertCache: cudaMemset of the slot arena failed";
        close();
        return false;
    }

    residency_.assign((size_t) (n_layers * n_expert), kNotResident);
    slots_ = n_slots;
    live_slots_ = n_slots;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    blob_ = blob_bytes;
#if defined(STRATA_USE_HIP)
    if (!ensure_blocking_staging((std::size_t) blob_, err)) {
        close();
        return false;
    }
#endif
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    // R4.2g: each layer starts at the bottom of its own range.  Built here rather than lazily so `admit`
    // stays allocation-free on the token path.
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    for (int64_t l = 0; l < n_layers; ++l) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(l, lo, hi);
        layer_next_[(size_t) l] = (int32_t) lo;
    }
    return true;
}

bool ExpertCache::open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert,
                             std::string& err) {
    if (slot_bytes.empty()) { err = "ExpertCache: no slots"; return false; }
    int64_t mx = 0;
    std::vector<uint64_t> off(slot_bytes.size() + 1, 0);
    for (size_t i = 0; i < slot_bytes.size(); ++i) {
        // 256-byte aligned slots, so every blob starts where the kernels' vector loads expect it
        off[i + 1] = off[i] + ((uint64_t) slot_bytes[i] + 255) / 256 * 256;
        mx = slot_bytes[i] > mx ? slot_bytes[i] : mx;
    }
    // one allocation of the summed size, through the uniform path's checks: n "slots" of 1 byte
    if (!open((int64_t) off.back(), n_layers, n_expert, 1, err)) return false;
    slots_ = (int64_t) slot_bytes.size();
    live_slots_ = slots_;
    blob_ = mx;
    off_ = std::move(off);
#if defined(STRATA_USE_HIP)
    if (!ensure_blocking_staging((std::size_t) blob_, err)) {
        close();
        return false;
    }
#endif
    // #369: each layer's cursor at the bottom of its own range, as open() seeds it - open() above ran on byte-sized
    // "slots", so its seeds are not slot indices
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    for (int64_t l = 0; l < n_layers; ++l) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(l, lo, hi);
        layer_next_[(size_t) l] = (int32_t) lo;
    }
    return true;
}

void ExpertCache::close() {
#if defined(STRATA_USE_HIP)
    if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
    blocking_staging_ = nullptr;
    blocking_staging_bytes_ = 0;
#endif
    off_.clear();
    if (!segs_.empty()) {
        release_segmented();
    } else if (vmm_) {
        vmm_.reset();   // unmaps and frees every chunk it still holds
        strata::device_alloc_unregister(base_);
        base_ = nullptr;
    } else if (base_ != nullptr) {
        strata::device_alloc_unregister(base_);
        sycl::free(base_, dpct::get_in_order_queue());
        base_ = nullptr;
    }
    residency_.clear();
    slots_ = 0;
    live_slots_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    blob_ = 0;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    layer_next_.clear();
}

/// R4.2g.  Layer `l` owns `[l*q, (l+1)*q)` with `q = slots_ / n_layers_`; the LAST layer takes whatever is
/// left over, so the ranges always cover `0..slots_` exactly and no slot is orphaned by the division.
void ExpertCache::layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const {
    lo = 0;
    hi = 0;
    if (n_layers_ <= 0 || slots_ <= 0 || layer < 0 || layer >= n_layers_) return;
    const int64_t q = slots_ / n_layers_;
    lo = layer * q;
    hi = (layer == n_layers_ - 1) ? slots_ : (layer + 1) * q;
}

int32_t ExpertCache::slot_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    return residency_[(size_t) (layer * n_expert_ + expert)];
}

int32_t ExpertCache::admit(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    const size_t at = (size_t) (layer * n_expert_ + expert);
    if (residency_[at] != kNotResident) return residency_[at];
    // R4.2g: THE PER-LAYER PATH.  Same "no eviction" rule, but the ceiling is this layer's own range rather
    // than one counter shared by all 48 - which is what confined the measured hit rate to 2.97%.
    if (per_layer_) {
        if (layer_next_.empty()) return kNotResident;
        int64_t lo = 0, hi = 0;
        layer_slot_range(layer, lo, hi);
        if ((int64_t) layer_next_[(size_t) layer] >= hi) return kNotResident;   // this layer's quota is full
        residency_[at] = layer_next_[(size_t) layer]++;
        ++admitted_;
        return residency_[at];
    }
    if (next_free_ >= slots_) return kNotResident;   // full: no eviction, deliberately - see the header
    residency_[at] = (int32_t) next_free_;
    return (int32_t) next_free_++;
}

uint8_t* ExpertCache::device_slot(int32_t slot) {
    if (slot < 0 || slot >= slots_) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

const uint8_t* ExpertCache::device_slot(int32_t slot) const {
    if (slot < 0 || slot >= slots_) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

bool ExpertCache::fill_slot(int32_t slot, const uint8_t *host_blob,
                            void *stream, std::string &err, int64_t bytes) try {
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot: slot " + std::to_string(slot) + " is outside 0.." +
              std::to_string(slots_ - 1);
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot: the host blob is null";
        return false;
    }
    /*
    DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    const dpct::err0 e =
        DPCT_CHECK_ERROR(strata::q_of(stream)->memcpy(dst, host_blob, n));
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
        err = std::string("ExpertCache::fill_slot: ") +
              dpct::get_error_string_dummy(e);
        return false;
    }
    ++fills_;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool ExpertCache::fill_slot_blocking(int32_t slot, const uint8_t *host_blob,
                                     std::string &err, int64_t bytes) try {
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot_blocking: slot outside the arena";
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot_blocking: the host blob is null";
        return false;
    }
#if defined(STRATA_USE_HIP)
    // Bound HIP's pageable-source staging to one expert instead of repeatedly
    // registering regions of the mmap. The blocking copy completes before reuse.
    if (!blocking_staging_ || n > blocking_staging_bytes_) {
        err = "ExpertCache::fill_slot_blocking: HIP staging buffer is too small";
        return false;
    }
    std::memcpy(blocking_staging_, host_blob, n);
    const cudaError_t e = cudaMemcpy(dst, blocking_staging_, n, cudaMemcpyHostToDevice);
#else
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    const dpct::err0 e =
        DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(dst, host_blob, n).wait());
#endif
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
        err = std::string("ExpertCache::fill_slot_blocking: ") +
              dpct::get_error_string_dummy(e);
        return false;
    }
    ++fills_;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool ExpertCache::fill_slot_queued(int32_t slot, const uint8_t *host_blob,
                                   std::string &err, int64_t bytes) try {
    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr || host_blob == nullptr) {
        err = dst == nullptr ? "ExpertCache::fill_slot_queued: slot outside the arena"
                             : "ExpertCache::fill_slot_queued: the host blob is null";
        return false;
    }
    /*
    DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    const dpct::err0 e =
        DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(dst, host_blob, n).wait());
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
        err = std::string("ExpertCache::fill_slot_queued: ") +
              dpct::get_error_string_dummy(e);
        return false;
    }
    ++fills_;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool ExpertCache::sync_queued(std::string &err) try {
    const dpct::err0 e = DPCT_CHECK_ERROR(dpct::get_in_order_queue().wait());
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
        err = std::string("ExpertCache::sync_queued: ") +
              dpct::get_error_string_dummy(e);
        return false;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool ExpertCache::verify_slot(int32_t slot, const uint8_t *host_blob,
                              std::string &err, int64_t bytes) try {
    const int64_t nb = bytes > 0 && bytes <= blob_ ? bytes : blob_;
    const uint8_t* src = device_slot(slot);
    if (src == nullptr) {
        err = "ExpertCache::verify_slot: slot outside the arena";
        return false;
    }
    // `cudaMemcpy` and not `cudaMemcpyAsync`: this is a startup check, and a check that can be read before it
    // has happened is not a check.  It also synchronises the fills queued before it, which is what makes the
    // comparison meaningful.
    std::vector<uint8_t> got((size_t) nb);
    const dpct::err0 e = DPCT_CHECK_ERROR(
        dpct::get_in_order_queue().memcpy(got.data(), src, (size_t)nb).wait());
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
        err = std::string("ExpertCache::verify_slot: ") +
              dpct::get_error_string_dummy(e);
        return false;
    }
    if (std::memcmp(got.data(), host_blob, (size_t) nb) != 0) {
        size_t first = 0;
        while (first < (size_t) nb && got[first] == host_blob[first]) ++first;
        char buf[384];
        std::snprintf(buf, sizeof buf,
                      "ExpertCache::verify_slot: slot %d differs from the arena at byte %llu (of %lld): device %02x%02x%02x%02x, arena %02x%02x%02x%02x",
                      (int) slot, (unsigned long long) first, (long long) blob_, first < (size_t) nb ? got[first] : 0,
                      first + 1 < (size_t) nb ? got[first + 1] : 0, first + 2 < (size_t) nb ? got[first + 2] : 0,
                      first + 3 < (size_t) nb ? got[first + 3] : 0, first < (size_t) nb ? host_blob[first] : 0,
                      first + 1 < (size_t) nb ? host_blob[first + 1] : 0, first + 2 < (size_t) nb ? host_blob[first + 2] : 0,
                      first + 3 < (size_t) nb ? host_blob[first + 3] : 0);
        err = buf;
        return false;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::core
