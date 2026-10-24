// src/core/pinned.cu - P2.S1: the pinned host arena and the parallel expert load.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/pinned.hpp"
#include "strata/platform/memory.hpp"

#include <atomic>
#include <climits>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>

// Loader fix: `fseek`/`ftell` are 32-bit on Windows by default (and the pack is 42.9 GB), and the 64-bit
// spelling is not the same on the two platforms the engine builds for.
#ifdef _WIN32
#define STRATA_FSEEK64(f, o) _fseeki64((f), (long long) (o), SEEK_SET)
#else
#define STRATA_FSEEK64(f, o) fseeko((f), (off_t) (o), SEEK_SET)
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <linux/mman.h>
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)
#endif
#endif

namespace strata::core {

namespace {

constexpr uint64_t kSharedArenaHeaderBytes = 4096;
constexpr char kSharedArenaMagic[16] = "STRATA-ARENA-V1";

struct SharedArenaHeader {
    char magic[16];
    uint32_t version;
    uint32_t header_bytes;
    uint64_t arena_bytes;
    uint64_t pack_hash;
    uint64_t reserved[4];
};
static_assert(sizeof(SharedArenaHeader) <= kSharedArenaHeaderBytes);

// A 2 MB-aligned reservation.  Large pages first, then the largest alignment the OS will give us for free.
// A non-empty shared_file instead maps one file whose first 4 KiB identify the pack and whose remaining bytes
// are the resident arena.  This shared-file layout is intended for tmpfs (/dev/shm); hugetlbfs would need
// hugepage-aligned file size and arena offset rather than the 4 KiB header layout used here.
void* reserve(uint64_t bytes, PageBacking& got, std::string& note, const std::string& shared_file,
              uint64_t shared_pack_hash, void*& mapping_base, uint64_t& mapping_bytes) {
    mapping_base = nullptr;
    mapping_bytes = 0;
#ifdef _WIN32
    if (!shared_file.empty()) {
        note = "shared-file arena backing is not implemented on Windows";
        return nullptr;
    }
    // MEM_LARGE_PAGES needs SeLockMemoryPrivilege.  Having it assigned to the account is not enough: the
    // PROCESS must enable it in its own token (AdjustTokenPrivileges) before VirtualAlloc, or the call fails.
    // An account without the assignment, or a failure to enable, leaves the process as it was: VirtualAlloc
    // then refuses and the 4 KB fallback below runs - that is the EXPECTED outcome on a desktop.
    {
        HANDLE tok = nullptr;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
            TOKEN_PRIVILEGES tp{};
            tp.PrivilegeCount = 1;
            if (LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) {
                tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
                if (!AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr) && GetLastError() != ERROR_NOT_ALL_ASSIGNED)
                    (void) 0;   // nothing actionable: the large-page attempt below reports the outcome
            }
            CloseHandle(tok);
        }
    }
    // MEM_LARGE_PAGES needs SeLockMemoryPrivilege; a normal account does not have it and VirtualAlloc then
    // fails with ERROR_PRIVILEGE_NOT_HELD.  That is the EXPECTED outcome on a desktop, not an error.
    SIZE_T large = GetLargePageMinimum();
    // A/B switch: STRATA_NO_LARGEPAGES=1 skips the large-page attempt, same run, same boot.
    if (large > 0 && std::getenv("STRATA_NO_LARGEPAGES") == nullptr) {
        // MEM_LARGE_PAGES requires the allocation size to be an exact multiple of the large page size -
        // anything else is ERROR_INVALID_PARAMETER (87), which reads like a privilege problem but is not.
        // Round up: the slack is under 2 MB and the tail stays unused.
        const SIZE_T lbytes = (SIZE_T) (((SIZE_T) bytes + large - 1) / large * large);
        void* p = VirtualAlloc(nullptr, lbytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                               PAGE_READWRITE);
        if (p) {
            got = PageBacking::LargePages;
            note = "large pages (" + std::to_string((unsigned long long) large) + " B)";
            return p;
        }
        // 1450 (ERROR_NO_SYSTEM_RESOURCES) is the large-page pool saying no, 87 is a size that is not a
        // multiple of the minimum, 1314 is the privilege: without the byte count the three read as one bug.
        note = "large pages refused for " + std::to_string((unsigned long long) lbytes) + " B (GetLargePageMinimum=" +
               std::to_string((unsigned long long) large) + ", VirtualAlloc error " +
               std::to_string((unsigned long long) GetLastError()) + "); using 4 KB pages";
    } else if (std::getenv("STRATA_NO_LARGEPAGES") != nullptr) {
        note = "large pages skipped (STRATA_NO_LARGEPAGES); using 4 KB pages";
    } else {
        note = "this system has no large-page minimum; using 4 KB pages";
    }
    void* p = VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    got = PageBacking::NormalPages;
    return p;
#else
    if (!shared_file.empty()) {
        if (shared_pack_hash == 0) {
            note = "shared arena requires a nonzero pack hash";
            return nullptr;
        }
        if (bytes > UINT64_MAX - kSharedArenaHeaderBytes) {
            note = "shared arena size overflows its header";
            return nullptr;
        }
        const uint64_t file_bytes = kSharedArenaHeaderBytes + bytes;
        const int fd = open(shared_file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd < 0) {
            note = "cannot open shared arena " + shared_file + ": " + std::strerror(errno);
            return nullptr;
        }
        struct stat st{};
        if (fstat(fd, &st) != 0) {
            const int e = errno;
            close(fd);
            note = "cannot stat shared arena " + shared_file + ": " + std::strerror(e);
            return nullptr;
        }

        const bool fresh = st.st_size == 0;
        if (fresh) {
            if (ftruncate(fd, (off_t) file_bytes) != 0) {
                const int e = errno;
                close(fd);
                note = "cannot size shared arena " + shared_file + ": " + std::strerror(e);
                return nullptr;
            }
            SharedArenaHeader hdr{};
            std::memcpy(hdr.magic, kSharedArenaMagic, sizeof hdr.magic);
            hdr.version = 1;
            hdr.header_bytes = (uint32_t) kSharedArenaHeaderBytes;
            hdr.arena_bytes = bytes;
            hdr.pack_hash = shared_pack_hash;
            const ssize_t written = pwrite(fd, &hdr, sizeof hdr, 0);
            if (written != (ssize_t) sizeof hdr) {
                const int e = errno;
                close(fd);
                note = "cannot write shared arena header " + shared_file + ": " +
                       (written < 0 ? std::string(std::strerror(e)) : std::string("short write"));
                return nullptr;
            }
        } else {
            if (st.st_size < 0 || (uint64_t) st.st_size != file_bytes) {
                close(fd);
                note = "shared arena " + shared_file + " is " +
                       std::to_string((unsigned long long) st.st_size) + " B, expected " +
                       std::to_string((unsigned long long) file_bytes) + " B including its header";
                return nullptr;
            }
            SharedArenaHeader hdr{};
            const ssize_t got_header = pread(fd, &hdr, sizeof hdr, 0);
            if (got_header != (ssize_t) sizeof hdr ||
                std::memcmp(hdr.magic, kSharedArenaMagic, sizeof hdr.magic) != 0 ||
                hdr.version != 1 || hdr.header_bytes != kSharedArenaHeaderBytes || hdr.arena_bytes != bytes) {
                close(fd);
                note = "shared arena " + shared_file + " has an incompatible or missing header";
                return nullptr;
            }
            if (hdr.pack_hash != shared_pack_hash) {
                close(fd);
                char b[256];
                std::snprintf(b, sizeof b,
                              "shared arena %s was written for pack hash %016llx, expected %016llx",
                              shared_file.c_str(), (unsigned long long) hdr.pack_hash,
                              (unsigned long long) shared_pack_hash);
                note = b;
                return nullptr;
            }
        }

        void* map = mmap(nullptr, (size_t) file_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        const int e = errno;
        close(fd);
        if (map == MAP_FAILED) {
            note = "cannot map shared arena " + shared_file + ": " + std::strerror(e);
            return nullptr;
        }
        mapping_base = map;
        mapping_bytes = file_bytes;
        got = PageBacking::NormalPages;
        note = "MAP_SHARED file " + shared_file + " (pack hash checked)";
        return (uint8_t*) map + kSharedArenaHeaderBytes;
    }

    // STRATA_NO_LARGEPAGES=1 is the same-run A/B switch the Windows branch documents; honor it
    // here too, so the large-page path can be compared without changing the pool or rebooting.
    if (std::getenv("STRATA_NO_LARGEPAGES") != nullptr) {
        note = "large pages skipped (STRATA_NO_LARGEPAGES); using 4 KB pages";
    } else {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (p != MAP_FAILED) {
            got = PageBacking::LargePages;
            note = "hugetlb 2 MB pages";
            return p;
        }
        // MAP_HUGETLB is all-or-nothing: a pool smaller than the mapping fails exactly like an absent
        // one, and the old message guessed "no hugetlb pool configured" either way. Name the shortfall:
        // how many 2 MiB pages the mapping needs against what vm.nr_hugepages actually holds.
        const unsigned long long need = ((unsigned long long) bytes + (1ull << 21) - 1) / (1ull << 21);
        unsigned long long pool = 0;
        bool have_pool = false;
        if (std::FILE* f = std::fopen("/proc/sys/vm/nr_hugepages", "r")) {
            have_pool = std::fscanf(f, "%llu", &pool) == 1;
            std::fclose(f);
        }
        note = "MAP_HUGETLB unavailable (needed " + std::to_string(need) + " 2 MiB pages, vm.nr_hugepages=" +
               (have_pool ? std::to_string(pool) : std::string("?")) + "); using 4 KB pages";
    }
    // No hugetlb mapping: a 2 MiB-aligned anonymous mapping with MADV_HUGEPAGE, so transparent huge pages back
    // it where THP is "madvise" (the common distro default). The CPU expert pool streams whole experts out of
    // this arena; with 4 KB pages every 3 MB expert costs ~750 TLB misses.
    constexpr uint64_t kAlign = 2ull << 20;
    const uint64_t padded = bytes + kAlign;
    void* raw = mmap(nullptr, padded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    got = PageBacking::NormalPages;
    if (raw == MAP_FAILED) return nullptr;
    const uintptr_t start = (uintptr_t) raw;
    const uintptr_t aligned = (start + kAlign - 1) & ~(uintptr_t) (kAlign - 1);
    if (aligned > start) munmap(raw, aligned - start);
    const uintptr_t end = aligned + bytes, raw_end = start + padded;
    if (raw_end > end) munmap((void*) end, raw_end - end);
    void* p = (void*) aligned;
    // #771: with THP "always" the kernel already backs this mapping with huge pages where it can; the extra
    // MADV_HUGEPAGE only adds direct reclaim and compaction on every fault (defrag=madvise), which on a fragmented
    // machine stretched a 25 s start to 432 s.  So it is not asked for there.  STRATA_NO_ARENA_THP=1 skips it
    // anywhere (the request alone; STRATA_NO_LARGEPAGES also skips the hugetlb try).
    std::string thp_mode;
    if (std::FILE* f = std::fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r")) {
        char line[128] = {0};
        if (std::fgets(line, sizeof line, f) != nullptr) {
            const char* open = std::strchr(line, '[');
            const char* close = open ? std::strchr(open, ']') : nullptr;
            if (open && close) thp_mode.assign(open + 1, close);
        }
        std::fclose(f);
    }
    const bool skip_thp_request = std::getenv("STRATA_NO_ARENA_THP") != nullptr || thp_mode == "always";
    if (std::getenv("STRATA_NO_LARGEPAGES") == nullptr && skip_thp_request) {
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += std::getenv("STRATA_NO_ARENA_THP") != nullptr
                    ? "; transparent huge pages skipped (STRATA_NO_ARENA_THP)"
                    : "; transparent huge pages are on for every mapping (THP always): MADV_HUGEPAGE not requested";
    } else if (std::getenv("STRATA_NO_LARGEPAGES") == nullptr && madvise(p, bytes, MADV_HUGEPAGE) == 0) {
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += "; transparent huge pages requested (MADV_HUGEPAGE)";
    }
    return p;
#endif
}

void release(void* p, uint64_t bytes) {
    if (!p) return;
#ifdef _WIN32
    (void) bytes;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, bytes);
#endif
}

}  // namespace

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

namespace {
/*
DPCT1010: SYCL uses exceptions to report errors and does not use the error
codes. The cudaGetLastError function call was replaced with 0. You need to
rewrite this code.
*/
bool clear_error() {(void)0; return true; }
}  // namespace

int arena_pin_cap_gib() {
    const char* e = std::getenv("STRATA_ARENA_PIN_GIB");
    if (e == nullptr || *e == '\0') return -1;
    if (std::string(e) == "auto") return -2;   // #243: the Windows shared-memory budget sets the sliced pin's cap
    const int v = std::atoi(e);
    return v < 0 ? -1 : v;
}

namespace {
#ifdef _WIN32
// #243: how much of the arena the sliced registration may pin on Windows when the whole arena was refused.
// Page-locked memory the GPU maps is charged to its shared (non-local) WDDM segment; pinned slice by slice until
// the driver refused one (28 GiB of a 63 GB PC), that segment was left full and every later cudaMalloc failed
// "out of memory".  4 GiB of the budget stay free for what the engine allocates after the arena; without the DXGI
// numbers, RAM/2 - 8 GiB (the budget is about half the RAM).  False: no limit could be worked out.
bool sliced_pin_limit(uint64_t& limit, std::string& why) {
    constexpr uint64_t GiB = 1ull << 30;
    char buf[256];
    int dev = 0;
    cudaDeviceProp p{};
    uint64_t budget = 0, usage = 0;
    std::string err = "no CUDA device properties";
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
        strata::platform::gpu_shared_memory_budget(p.luid, budget, usage, err)) {
        limit = budget > usage + 4 * GiB ? budget - usage - 4 * GiB : 0;
        std::snprintf(buf, sizeof buf, "the GPU's shared-memory budget %.1f GiB - %.1f GiB in use - 4 GiB",
                      (double) budget / GiB, (double) usage / GiB);
        why = buf;
        return true;
    }
    (void) cudaGetLastError();
    const uint64_t ram = strata::platform::total_physical_memory();
    if (ram == 0) return false;
    limit = ram / 2 > 8 * GiB ? ram / 2 - 8 * GiB : 0;
    std::snprintf(buf, sizeof buf, "%s: RAM/2 - 8 GiB of %.1f GiB", err.c_str(), (double) ram / GiB);
    why = buf;
    return true;
}
#endif
}  // namespace

namespace {
std::vector<uint64_t> uniform_bounds(uint64_t bytes, uint64_t slice) {
    std::vector<uint64_t> b;
    if (slice == 0) return b;
    for (uint64_t off = 0; off + slice <= bytes; off += slice) b.push_back(off);
    if (!b.empty()) b.push_back(b.back() + slice);
    return b;
}
}  // namespace

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice) : PinnedArena(bytes, uniform_bounds(bytes, slice)) {
    if (slice_bytes) slice_bytes = slice;   // sliced registration: record the uniform size
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t> &bounds,
                         uint64_t max_pinned_bytes,
                         const std::string &shared_file,
                         uint64_t shared_pack_hash) try
    : capacity(bytes) {
    if (bytes == 0) return;
    base = reserve(bytes, backing, note, shared_file, shared_pack_hash, mapping_base, mapping_bytes);
    if (base != nullptr && mapping_base == nullptr) {
        mapping_base = base;
        mapping_bytes = bytes;
    }

    // Register with CUDA BEFORE any page is touched: cudaHostRegister pins what is resident now, and a region
    // that has already been faulted in page by page is far more expensive to register and may fail outright.
    if (base) {
        // #243: STRATA_ARENA_PIN_GIB=N caps the registration from the start where the caller set no cap
        const int env_gib = arena_pin_cap_gib();
        uint64_t cap = max_pinned_bytes;
        std::string cap_why = "by the engine (multi-GPU under WDDM, or remote experts)";
        if (cap == 0 && env_gib > 0) {
            cap = (uint64_t) env_gib << 30;
            cap_why = "by STRATA_ARENA_PIN_GIB";
        }
        const bool capped = cap > 0 && cap < bytes && bounds.size() >= 2;
        const dpct::err0 e =
            capped ? 0 :
                   /*
                   DPCT1027: The call to cudaHostRegister was replaced with
                   0 because SYCL currently does not support registering of
                   existing host memory for use by device. Use USM to allocate
                   memory for use by host and device.
                   */
                0;
        if (!capped && e == 0) {
            note = "cudaHostRegister PORTABLE ok; " + note;
            registered_bytes = bytes;
        } else if (bounds.size() >= 2 && (capped || clear_error())) {
            // Plan v0.3 P5: the whole range is refused, so pin it slice by slice from the start.  The rest stays
            // resident through the working-set lock below.  (P6: slices may differ in size, one per layer.)
            slice_bytes = 1;   // sliced; the uniform constructor records the size
            bool limited = capped;
            uint64_t limit = cap;
            std::string limit_why;
#ifdef _WIN32
            // #243 (opt-in, STRATA_ARENA_PIN_GIB=auto): not up to the driver's refusal but below the shared-memory
            // budget, for a PC where the full sliced pin leaves WDDM refusing later allocations.  Not the default: a
            // 64 GB PC pins 30 GiB past that budget without trouble, and capping it at 26 cost ~20% prompt speed.
            if (!capped && env_gib == -2) limited = sliced_pin_limit(limit, limit_why);
#endif
            for (size_t i = 0; i + 1 < bounds.size(); ++i) {
                const uint64_t off = bounds[i], n = bounds[i + 1] - bounds[i];
                if (limited && (off > limit || n > limit - off)) break;
                /*
                DPCT1027: The call to cudaHostRegister was replaced with 0
                because SYCL currently does not support registering of existing
                host memory for use by device. Use USM to allocate memory for
                use by host and device.
                */
                if (0 != 0) {
                    /*
                    DPCT1010: SYCL uses exceptions to report errors and
                    does not use the error codes. The cudaGetLastError function
                    call was replaced with 0. You need to rewrite this code.
                    */
                    (void)0;
                    break;
                }
                slice_starts.push_back(off);
                registered_bytes = off + n;
                ++registered_slices;
            }
            char gib[32];
            std::snprintf(gib, sizeof gib, "%.1f", (double) limit / (double) (1ull << 30));
            note =
                (capped
                     ? "cudaHostRegister limited to " +
                           std::to_string(cap >> 30) + " GiB " + cap_why + "; "
                     :
                     /*
                     DPCT1009: SYCL reports errors using exceptions and
                     does not use error codes. Please replace the
                     "get_error_string_dummy(...)" with a real error-handling
                     function.
                     */
                     "cudaHostRegister of the whole arena FAILED (" +
                         std::string(dpct::get_error_string_dummy(e)) + "); " +
                         (limited ? "slices capped at " + std::string(gib) +
                                        " GiB (" + limit_why +
                                        "; STRATA_ARENA_PIN_GIB=auto; N sets a "
                                        "cap; #243); "
                                  : std::string())) +
                std::to_string(registered_slices) + " slices pinned (" +
                std::to_string(registered_bytes >> 30) + " GiB); " + note;
            if (registered_bytes < bytes) {
                const char* env = std::getenv("STRATA_ARENA_LOCK");
                if (backing == PageBacking::LargePages) {   // #779: large pages cannot be paged out: nothing to lock
                    note = "large pages are resident without a lock; " + note;
                } else if (env == nullptr || std::string(env) != "0") {
                    const strata::platform::LockResult lr =
                        strata::platform::lock_resident((uint8_t*) base + registered_bytes, bytes - registered_bytes);
                    locked_bytes = lr.locked_bytes;
                    note = lr.note + "; " + note;
                }
            }
        } else {
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            note = std::string("cudaHostRegister FAILED (") +
                   dpct::get_error_string_dummy(e) +
                   ") - the arena is NOT pinned, so copies will be slow; " +
                   note;
            // **CONSUME THE ERROR, OR IT LIES ABOUT SOMETHING ELSE LATER.**
            //
            // `cudaGetLastError()` returns the last error and CLEARS it; until something reads it, the error
            // state is sticky.  This failure is caught and handled right here - the arena is simply not pinned -
            // but leaving it set meant the next `cudaGetLastError()` in the engine, which is `gr_read`'s launch
            // check, reported "out of memory" for kernels that allocate nothing.  That cost a round: the arena
            // was written off as not fitting the machine when in fact the only thing wrong was a stale error
            // from this line.
            //
            // It is the same trap `gr.cu` warns about for ASYNC faults, in the other direction: a synchronous
            // failure is sticky too, and it lies about where it happened just as convincingly.
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not
            use the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            (void)0;
            // Plan v0.3 P0.1: keep it RESIDENT instead. Unpinned, Windows trims the arena under memory pressure
            // and the CPU pool's rate then depends on the OS; locking it through the working set needs no
            // special privilege. STRATA_ARENA_LOCK=0 is the A/B arm.
            const char* env = std::getenv("STRATA_ARENA_LOCK");
            if (backing == PageBacking::LargePages) {       // #779: large pages cannot be paged out: nothing to lock
                note = "large pages are resident without a lock; " + note;
            } else if (env == nullptr || std::string(env) != "0") {
                const strata::platform::LockResult lr = strata::platform::lock_resident(base, bytes);
                locked_bytes = lr.locked_bytes;
                note = lr.note + "; " + note;
            } else {
                note = "arena lock disabled (STRATA_ARENA_LOCK=0); " + note;
            }
        }
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, Deferred)
    : capacity(bytes), bounds_(bounds) {
    if (bytes == 0 || bounds.size() < 2) return;
    base = reserve(bytes, backing, note, std::string{}, 0, mapping_base, mapping_bytes);
    if (base != nullptr && mapping_base == nullptr) {
        mapping_base = base;
        mapping_bytes = bytes;
    }
    slice_bytes = 1;   // sliced: one registration per layer
}

void PinnedArena::register_slices(std::atomic<int> &ready) try {
    const int n = (int) bounds_.size() - 1;
    int i = 0;
    for (; base != nullptr && i < n; ++i) {
        const uint64_t off = bounds_[(size_t) i], len = bounds_[(size_t) i + 1] - off;
        /*
        DPCT1027: The call to cudaHostRegister was replaced with 0 because
        SYCL currently does not support registering of existing host memory for
        use by device. Use USM to allocate memory for use by host and device.
        */
        if (0 != 0) {
            /*
            DPCT1010: SYCL uses exceptions to report errors and does not use
            the error codes. The cudaGetLastError function call was replaced
            with 0. You need to rewrite this code.
            */
            (void)0;
            break;
        }
        slice_starts.push_back(off);
        registered_bytes = off + len;
        ++registered_slices;
        ready.store(i + 1, std::memory_order_release);
    }
    note = "cudaHostRegister per layer, pipelined with the load: " + std::to_string(registered_slices) + " of " +
           std::to_string(n) + " slices pinned; " + note;
    if (i < n && base != nullptr) {   // the rest stays resident through the working-set lock, as the sliced fallback does
        const char* env = std::getenv("STRATA_ARENA_LOCK");
        if (backing == PageBacking::LargePages) {           // #779: large pages cannot be paged out: nothing to lock
            note = "large pages are resident without a lock; " + note;
        } else if (env == nullptr || std::string(env) != "0") {
            const strata::platform::LockResult lr =
                strata::platform::lock_resident((uint8_t*) base + registered_bytes, capacity - registered_bytes);
            locked_bytes = lr.locked_bytes;
            note = lr.note + "; " + note;
        }
    }
    ready.store(INT_MAX, std::memory_order_release);   // every slice done (or given up on)
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

PinnedArena::~PinnedArena() {
    if (base) {
        if (locked_bytes) strata::platform::unlock_resident((uint8_t*) base + (slice_bytes ? registered_bytes : 0), locked_bytes);
        if (slice_bytes) {
            /*
            DPCT1027: The call to cudaHostUnregister was replaced with 0
            because SYCL currently does not support registering of existing host
            memory for use by device. Use USM to allocate memory for use by host
            and device.
            */
            for (uint64_t off : slice_starts) 0;
        } else {
            /*
            DPCT1026: The call to cudaHostUnregister was removed because
            SYCL currently does not support registering of existing host memory
            for use by device. Use USM to allocate memory for use by host and
            device.
            */
        }
        release(mapping_base ? mapping_base : base, mapping_bytes ? mapping_bytes : capacity);
        base = nullptr;
        mapping_base = nullptr;
        mapping_bytes = 0;
    }
}

LoadStats load_experts(const std::string& path, uint8_t* dst, uint64_t blob_bytes, uint64_t blobs_per_layer,
                       uint64_t layers, int threads, uint64_t chunk) {
    std::vector<uint64_t> off((size_t) layers), n((size_t) layers, blobs_per_layer * blob_bytes);
    for (uint64_t L = 0; L < layers; ++L) off[(size_t) L] = L * blobs_per_layer * blob_bytes;
    return load_experts_ranges(path, dst, off, n, threads, chunk);
}

LoadStats load_experts_direct(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready) {
    LoadStats st;
    st.ok = false;
#ifdef _WIN32
    constexpr uint64_t kAlign = 4096;
    if (((uintptr_t) dst % kAlign) != 0 || chunk == 0 || chunk % kAlign != 0) return st;
    struct Piece { uint64_t off, n; int layer; };
    std::vector<Piece> pieces;
    uint64_t bytes = 0;
    for (size_t L = 0; L < layer_off.size(); ++L) {
        if (layer_off[L] % kAlign != 0 || layer_bytes[L] % kAlign != 0) return st;
        for (uint64_t p = 0; p < layer_bytes[L]; p += chunk)
            pieces.push_back({layer_off[L] + p, std::min<uint64_t>(chunk, layer_bytes[L] - p), (int) L});
        bytes += layer_bytes[L];
    }
    if (threads < 1) threads = 1;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<size_t> next{0};
    std::mutex err_mu;
    std::string err;
    const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath((size_t) (std::max)(wide, 1), L'\0');
    if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
    auto worker = [&]() {
        HANDLE h = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            std::lock_guard<std::mutex> g(err_mu);
            if (err.empty())
                err = "cannot open " + path + " unbuffered (error " + std::to_string((unsigned long long) GetLastError()) + ")";
            next = pieces.size();
            return;
        }
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= pieces.size()) break;
            if (ready != nullptr)      // the slice must be registered before its pages are touched
                while (ready->load(std::memory_order_acquire) <= pieces[i].layer + 1) std::this_thread::yield();
            OVERLAPPED ov{};
            ov.Offset = (DWORD) pieces[i].off;
            ov.OffsetHigh = (DWORD) (pieces[i].off >> 32);
            DWORD got = 0;
            if (!ReadFile(h, dst + pieces[i].off, (DWORD) pieces[i].n, &got, &ov) || got != pieces[i].n) {
                std::lock_guard<std::mutex> g(err_mu);
                if (err.empty())
                    err = "short unbuffered read at offset " + std::to_string(pieces[i].off) + ": got " + std::to_string(got) +
                          " of " + std::to_string(pieces[i].n) + " B (error " +
                          std::to_string((unsigned long long) GetLastError()) + ")";
                next = pieces.size();
                break;
            }
        }
        CloseHandle(h);
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    st.layers = layer_off.size();
    if (!err.empty()) {
        st.seconds = -1.0;
        st.error = err;
        return st;
    }
    st.ok = true;
    st.bytes = bytes;
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
#else
    (void) path; (void) dst; (void) layer_off; (void) layer_bytes; (void) threads; (void) chunk; (void) ready;
#endif
    return st;
}

bool experts_unbuffered(const std::vector<std::string>& files, uint64_t arena_bytes, std::string& why,
                        bool cache_counts, uint64_t read_bytes) {
    const char* env = std::getenv("STRATA_UNBUFFERED_LOAD");
    if (env != nullptr && env[0] != '\0') {
        why = std::string("STRATA_UNBUFFERED_LOAD=") + env;
        return env[0] != '0';
    }
#ifdef _WIN32
    LARGE_INTEGER freq{}, a{}, b{};
    QueryPerformanceFrequency(&freq);
    constexpr DWORD kRead = 64 << 10;
    constexpr int kSamples = 16;
    std::vector<uint8_t> buf(kRead);
    uint64_t total_bytes = 0;
    int fast = 0, n = 0;
    uint64_t seed = (uint64_t) GetTickCount64() * 6364136223846793005ull + 1442695040888963407ull;
    for (const std::string& f : files) {
        const int wide = MultiByteToWideChar(CP_UTF8, 0, f.c_str(), -1, nullptr, 0);
        std::vector<wchar_t> w((size_t) (std::max)(wide, 1), L'\0');
        if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, f.c_str(), -1, w.data(), wide);
        HANDLE h = CreateFileW(w.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS,
                               nullptr);
        if (h == INVALID_HANDLE_VALUE) continue;
        LARGE_INTEGER size{};
        GetFileSizeEx(h, &size);
        total_bytes += (uint64_t) size.QuadPart;
        // random offsets: a probe must not find the blocks an earlier probe put into the cache
        for (int i = 0; i < kSamples && (uint64_t) size.QuadPart > 2ull * kRead; ++i) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            const uint64_t off = ((seed >> 17) % ((uint64_t) size.QuadPart - kRead)) / kRead * kRead;
            OVERLAPPED ov{};
            ov.Offset = (DWORD) off;
            ov.OffsetHigh = (DWORD) (off >> 32);
            DWORD got = 0;
            QueryPerformanceCounter(&a);
            const BOOL ok = ReadFile(h, buf.data(), kRead, &got, &ov);
            QueryPerformanceCounter(&b);
            if (!ok) continue;
            ++n;
            fast += (double) (b.QuadPart - a.QuadPart) * 1e6 / (double) freq.QuadPart < 30.0;
        }
        CloseHandle(h);
    }
    const bool cached = n > 0 && fast * 4 >= n * 3;
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    GlobalMemoryStatusEx(&ms);
    const uint64_t avail = ms.ullAvailPhys;
    // what the cache could keep beside the arena (~4 GiB for everything else)
    const uint64_t need = read_bytes == kAllFileBytes ? total_bytes : read_bytes;
    const bool keepable = strata::platform::file_cache_keeps(avail, arena_bytes, need);
    char msg[256];
    if (read_bytes == kAllFileBytes)
        std::snprintf(msg, sizeof msg, "%d of %d probe reads from the file cache; %.1f GiB available, %.1f GiB of files",
                      fast, n, (double) avail / (1ull << 30), (double) total_bytes / (1ull << 30));
    else
        std::snprintf(msg, sizeof msg, "%.1f GiB available, %.1f GiB of it still to be taken by the RAM copy, %.1f GiB "
                      "of experts read from the files: the file cache %s keep them",
                      (double) avail / (1ull << 30), (double) arena_bytes / (1ull << 30), (double) need / (1ull << 30),
                      keepable ? "can" : "cannot");
    why = msg;
    return (!cached || !cache_counts) && !keepable;
#else
    (void) files; (void) arena_bytes; (void) cache_counts; (void) read_bytes;
    why = "buffered (not Windows)";
    return false;
#endif
}

LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready) {
    LoadStats st;
    const uint64_t layers = (uint64_t) layer_off.size();
    st.layers = layers;
    st.bytes = 0;
    for (uint64_t b : layer_bytes) st.bytes += b;
    if (threads < 1) threads = 1;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint64_t> layer_hash((size_t) layers, 1469598103934665603ull);
    std::atomic<uint64_t> next_layer{0};
    std::atomic<uint64_t> read_ns_sum{0};   // summed over the threads: see LoadStats::read_seconds
    std::atomic<uint64_t> copy_ns_sum{0};
    std::mutex err_mu;
    std::string err;

    auto worker = [&]() {
        std::vector<uint8_t> buf((size_t) chunk);
        // One handle per thread, seeked once per layer: a shared handle would need a lock around the seek and
        // would serialise the very thing the threads are here to parallelise.  `fread` on a `FILE*` rather than
        // `std::ifstream`: see the header - MSVC's `basic_filebuf::xsgetn` splits any request larger than
        // `_INTERNAL_BUFSIZ - 1` into 4095-byte freads, which turned one 8 MiB chunk into ~2048 4 KiB reads.
        // `fread` sees a request bigger than the stream buffer and passes it to `_read()`/
        // `ReadFile()` unchanged, so the chunk size reaches the disk.  Buffered, not `FILE_FLAG_NO_BUFFERING`:
        // the cache should still hold what it can.
        FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            std::lock_guard<std::mutex> g(err_mu);
            err = "cannot open " + path;
            return;
        }
        // A `FILE*` has no destructor that closes it, and this function has early returns below (open, seek and
        // short-read failures), so the guard is what keeps the closing correct on every path.
        struct Closer {
            FILE* f;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } closer{f};
        uint64_t read_ns = 0, copy_ns = 0;
        for (;;) {
            const uint64_t L = next_layer.fetch_add(1);
            if (L >= layers) break;
            if (ready != nullptr)      // the slice must be registered before its pages are touched
                while (ready->load(std::memory_order_acquire) <= (int) L + 1) std::this_thread::yield();
            const uint64_t off = layer_off[(size_t) L];
            uint64_t remaining = layer_bytes[(size_t) L];
            uint64_t pos = 0;
            uint64_t h = 1469598103934665603ull;
            // 64-bit seek: the pack is 42.9 GB, so the 32-bit `fseek` would wrap past 4 GiB
            if (STRATA_FSEEK64(f, off) != 0) {
                std::lock_guard<std::mutex> g(err_mu);
                err = "seek to " + std::to_string(off) + " B failed in layer " + std::to_string(L);
                return;
            }
            while (remaining > 0) {
                const uint64_t n = remaining < chunk ? remaining : chunk;
                const auto t_read = std::chrono::steady_clock::now();
                const size_t got = std::fread(buf.data(), 1, (size_t) n, f);
                read_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_read).count();
                // A short read is EOF or an I/O error, never a silent zero fill: say WHERE and HOW SHORT, and
                // keep going no further - the caller turns this into a refused load, not a wrong answer.
                if (got != (size_t) n) {
                    std::lock_guard<std::mutex> g(err_mu);
                    err = "short read in layer " + std::to_string(L) + ": got " + std::to_string(got) + " of "
                          + std::to_string(n) + " B at offset " + std::to_string(off + pos)
                          + (std::ferror(f) != 0 ? " (ferror set)" : "");
                    return;
                }
                const auto t_copy = std::chrono::steady_clock::now();
                std::memcpy(dst + off + pos, buf.data(), (size_t) n);
                h = fnv1a64(buf.data(), n, h);
                copy_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_copy).count();
                pos += n;
                remaining -= n;
            }
            layer_hash[(size_t) L] = h;
        }
        read_ns_sum.fetch_add(read_ns);
        copy_ns_sum.fetch_add(copy_ns);
    };

    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();

    if (!err.empty()) {
        std::fprintf(stderr, "load_experts: %s\n", err.c_str());
        st.seconds = -1.0;
        st.ok = false;
        st.error = err;
        return st;
    }
    st.read_seconds = (double) read_ns_sum.load() / 1e9;
    st.copy_seconds = (double) copy_ns_sum.load() / 1e9;
    st.layer_checksums = std::move(layer_hash);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

StreamStats stream_bandwidth(const uint8_t *src, uint64_t bytes, uint64_t chunk,
                             int iters) try {
    StreamStats st;
    st.bytes = bytes * (uint64_t) iters;
    st.chunk = chunk;
    uint8_t* dst = nullptr;
    dpct::queue_ptr s{};
    if (DPCT_CHECK_ERROR(dst = (uint8_t *)sycl::malloc_device(
                             (size_t)chunk, dpct::get_in_order_queue())) != 0) {
        std::fprintf(stderr, "stream_bandwidth: cudaMalloc failed for %llu B\n", (unsigned long long) chunk);
        st.seconds = -1.0;
        return st;
    }
    s = dpct::get_current_device().create_queue(true);

    // one untimed pass so the first transfer's page-fault and setup cost is not in the measurement
    for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        s->memcpy(dst, src + off, (size_t)chunk);
    }
    s->wait();

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
            /*
            DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy
            API. While the origin API might be synchronous, it depends on the
            type of operand memory, so you may need to call wait() on event
            return by memcpy API to ensure synchronization behavior.
            */
            s->memcpy(dst, src + off, (size_t)chunk);
        }
    }
    s->wait();
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    dpct::get_current_device().destroy_queue(s);
    sycl::free(dst, dpct::get_in_order_queue());
    return st;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::core
