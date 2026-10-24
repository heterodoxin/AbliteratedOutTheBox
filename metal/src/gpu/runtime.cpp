// metal/src/gpu/runtime.cpp - include/strata/gpu/runtime.hpp on Metal (metal-cpp).  Skeleton of MAC-PLAN Phase 1.
//
// Memory model (measured on the M5 Max, phaseA-tests/mac-design.md):
//   - every allocation is one MTLBuffer, storageModeShared, hazard tracking untracked (the engine orders its own work
//     on streams, as on CUDA); the "device pointer" handed to the host code is the buffer's gpuAddress, which differs
//     from contents() on Apple GPUs, so host code that wrongly dereferences a device pointer faults instead of
//     silently working.  host_view() is the sanctioned CPU view.
//   - all buffers sit in one MTLResidencySet attached to every queue, so kernels may take raw gpuAddress pointers.
// Streams: a command queue + the open command buffer + the open compute encoder + a shared-event timeline.  Work is
// committed at event records, synchronizations, host functions and every kFlushEvery dispatches.
// Graphs: begin_capture switches the stream to recording; dispatch/copy/memset/event ops are kept as data and
// re-encoded by graph_launch (encoding measured at ~0.17 us per dispatch on the M5 Max).
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <execinfo.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "strata/gpu/metal.hpp"
#include "strata/gpu/runtime.hpp"

#include <sys/sysctl.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::gpu {

namespace {

constexpr int kFlushEvery = 4096;   // dispatches per command buffer before an automatic commit

struct Alloc {
    MTL::Buffer* buf = nullptr;
    uint64_t gpu = 0;
    uint8_t* cpu = nullptr;
    size_t bytes = 0;
    bool host = false;   // from host_alloc (the host code holds the CPU address)
    bool purged = false; // purge_device_allocations: its memory given back (any later copy is an error)
};

struct Op {
    enum Kind { Dispatch, Copy, Fill, Record, Wait } kind = Dispatch;
    MTL::ComputePipelineState* pipe = nullptr;
    uint8_t args[256];
    size_t arg_bytes = 0;
    std::vector<uint8_t> big_args;
    metal::Grid groups, threads;
    size_t tg_bytes = 0;
    void* dst = nullptr;
    const void* src = nullptr;
    size_t bytes = 0;
    int value = 0;
    Event ev = nullptr;
};

}  // namespace

struct EventImpl {
    MTL::SharedEvent* ev = nullptr;   // the recording stream's timeline
    uint64_t value = 0;               // 0: never recorded (complete, as a fresh CUDA event)
};

struct GraphImpl {
    std::vector<Op> ops;
};
struct GraphExecImpl {
    std::vector<Op> ops;
};

struct StreamImpl {
    MTL::CommandQueue* q = nullptr;
    MTL::CommandBuffer* cb = nullptr;
    MTL::ComputeCommandEncoder* enc = nullptr;
    MTL::SharedEvent* tl = nullptr;
    uint64_t submitted = 0;   // the timeline value the last committed command buffer signals
    int pending = 0;          // dispatches in the open command buffer
    GraphImpl* rec = nullptr;
};

namespace {

struct Runtime {
    MTL::Device* dev = nullptr;
    MTL::ResidencySet* rs = nullptr;
    std::mutex mu;
    std::map<uint64_t, Alloc> by_gpu;   // gpuAddress -> allocation
    std::map<uintptr_t, uint64_t> by_cpu;   // contents() -> gpuAddress
    StreamImpl* legacy = nullptr;
    std::map<std::string, MTL::ComputePipelineState*> pipes;
    std::map<std::string, MTL::Library*> libs;
    metal::Stats stats;
};

Runtime& rt() {
    static Runtime* r = [] {
        auto* x = new Runtime;
        x->dev = MTL::CreateSystemDefaultDevice();
        NS::Error* e = nullptr;
        MTL::ResidencySetDescriptor* d = MTL::ResidencySetDescriptor::alloc()->init();
        x->rs = x->dev->newResidencySet(d, &e);
        d->release();
        return x;
    }();
    return *r;
}

thread_local Error t_last = kSuccess;
Error fail(Error e) {
    if (e != kSuccess && e != kNotReady) t_last = e;
    return e;
}

StreamImpl* make_stream() {
    auto* s = new StreamImpl;
    s->q = rt().dev->newCommandQueue();
    s->q->addResidencySet(rt().rs);
    s->tl = rt().dev->newSharedEvent();
    s->tl->setSignaledValue(0);
    return s;
}

StreamImpl* S(Stream s) {
    if (s) return s;
    Runtime& r = rt();
    std::lock_guard<std::mutex> lk(r.mu);
    if (!r.legacy) r.legacy = make_stream();
    return r.legacy;
}

/// The allocation that holds GPU address `p` (or CPU address `p` of a host allocation), with the offset into it.
const Alloc* find(const void* p, size_t* off) {
    Runtime& r = rt();
    std::lock_guard<std::mutex> lk(r.mu);
    const uint64_t a = (uint64_t) (uintptr_t) p;
    auto it = r.by_gpu.upper_bound(a);
    if (it != r.by_gpu.begin()) {
        --it;
        if (a < it->first + it->second.bytes) { *off = a - it->first; return &it->second; }
    }
    auto ic = r.by_cpu.upper_bound((uintptr_t) p);
    if (ic != r.by_cpu.begin()) {
        --ic;
        const Alloc& al = r.by_gpu[ic->second];
        if ((uintptr_t) p < ic->first + al.bytes) { *off = (uintptr_t) p - ic->first; return &al; }
    }
    return nullptr;
}

MTL::CommandBuffer* open_cb(StreamImpl* s) {
    if (!s->cb) {
        s->cb = s->q->commandBufferWithUnretainedReferences();
        s->cb->retain();
        ++rt().stats.command_buffers;
    }
    return s->cb;
}
void end_encoder(StreamImpl* s) {
    if (s->enc) { s->enc->endEncoding(); s->enc = nullptr; }
}
MTL::ComputeCommandEncoder* encoder(StreamImpl* s) {
    if (!s->enc) s->enc = open_cb(s)->computeCommandEncoder();   // serial dispatch: each dispatch sees the last
    return s->enc;
}
void flush(StreamImpl* s) {
    if (!s->cb) return;
    end_encoder(s);
    s->cb->encodeSignalEvent(s->tl, ++s->submitted);
    s->cb->commit();
    s->cb->release();
    s->cb = nullptr;
    s->pending = 0;
}
Error wait_value(MTL::SharedEvent* ev, uint64_t v) {
    while (ev->signaledValue() < v) std::this_thread::yield();
    ++rt().stats.host_syncs;
    return kSuccess;
}

void check_purged(const Alloc* a, const char* what) {
    if (a && a->purged) {
        std::fprintf(stderr, "strata metal runtime: a %s touches a device allocation of %.1f MB that was given back "
                             "(STRATA_METAL_KEEP_HOST_WEIGHTS=1 keeps it)\n", what, a->bytes / 1e6);
        void* fr[48];
        backtrace_symbols_fd(fr, backtrace(fr, 48), 2);   // which host code still reads it
        std::abort();
    }
}

Error encode_copy(StreamImpl* s, void* dst, const void* src, size_t n) {
    size_t od = 0, os = 0;
    const Alloc* ad = find(dst, &od);
    const Alloc* as = find(src, &os);
    check_purged(ad, "copy");
    check_purged(as, "copy");
    if (ad && as) {   // buffer to buffer: a blit, in stream order
        end_encoder(s);
        MTL::BlitCommandEncoder* b = open_cb(s)->blitCommandEncoder();
        b->copyFromBuffer(as->buf, os, ad->buf, od, n);
        b->endEncoding();
        ++rt().stats.blits;
        return kSuccess;
    }
    // ordinary host memory on one side: the stream's earlier work must be complete first (the conservative CUDA
    // pageable-memory semantics: the call returns when the copy is done)
    flush(s);
    wait_value(s->tl, s->submitted);
    uint8_t* d = ad ? ad->cpu + od : (uint8_t*) dst;
    const uint8_t* sp = as ? as->cpu + os : (const uint8_t*) src;
    std::memcpy(d, sp, n);
    return kSuccess;
}

Error encode_fill(StreamImpl* s, void* p, int value, size_t n) {
    size_t off = 0;
    const Alloc* a = find(p, &off);
    if (!a) return fail(kErrorInvalidValue);
    check_purged(a, "fill");
    end_encoder(s);
    MTL::BlitCommandEncoder* b = open_cb(s)->blitCommandEncoder();
    b->fillBuffer(a->buf, NS::Range::Make(off, n), (uint8_t) value);
    b->endEncoding();
    ++rt().stats.blits;
    return kSuccess;
}

Error encode_dispatch(StreamImpl* s, const Op& o) {
    MTL::ComputeCommandEncoder* e = encoder(s);
    e->setComputePipelineState(o.pipe);
    if (o.big_args.empty()) e->setBytes(o.args, o.arg_bytes, 0);
    else e->setBytes(o.big_args.data(), o.big_args.size(), 0);
    if (o.tg_bytes) e->setThreadgroupMemoryLength(o.tg_bytes, 0);
    e->dispatchThreadgroups(MTL::Size(o.groups.x, o.groups.y, o.groups.z), MTL::Size(o.threads.x, o.threads.y, o.threads.z));
    ++rt().stats.dispatches;
    if (++s->pending >= kFlushEvery) flush(s);
    return kSuccess;
}

Error record_event(EventImpl* e, StreamImpl* s) {
    flush(s);   // CUDA submits on launch: a recorded event must become complete without a further call
    e->ev = s->tl;
    e->value = s->submitted;
    return kSuccess;
}
Error wait_event(StreamImpl* s, EventImpl* e) {
    if (!e || e->value == 0) return kSuccess;
    end_encoder(s);
    open_cb(s)->encodeWait(e->ev, e->value);
    return kSuccess;
}

Error run_op(StreamImpl* s, const Op& o) {
    switch (o.kind) {
        case Op::Dispatch: return encode_dispatch(s, o);
        case Op::Copy: return encode_copy(s, o.dst, o.src, o.bytes);
        case Op::Fill: return encode_fill(s, o.dst, o.value, o.bytes);
        case Op::Record: return record_event(o.ev, s);
        case Op::Wait: return wait_event(s, o.ev);
    }
    return kSuccess;
}

// STRATA_METAL_ALLOC_LOG=1: every allocation of 16 MB or more, with the running total (memory budgets on small Macs)
void alloc_log(const char* what, size_t n) {
    static const bool on = std::getenv("STRATA_METAL_ALLOC_LOG") != nullptr;
    static std::atomic<uint64_t> tot{0};
    const uint64_t t = tot += n;
    if (on && n >= (16u << 20)) std::fprintf(stderr, "strata metal runtime: %s %.1f MB (total %.2f GB)\n", what, n / 1e6, t / 1e9);
}

Error new_alloc(void** p, size_t bytes, bool host) {
    Runtime& r = rt();
    const size_t n = bytes ? bytes : 1;
    MTL::Buffer* b = r.dev->newBuffer(n, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked);
    if (!b) return fail(kErrorMemoryAllocation);
    Alloc a{b, b->gpuAddress(), (uint8_t*) b->contents(), n, host};
    alloc_log(host ? "host alloc" : "device alloc", n);
    {
        std::lock_guard<std::mutex> lk(r.mu);
        r.by_gpu[a.gpu] = a;
        r.by_cpu[(uintptr_t) a.cpu] = a.gpu;
        r.rs->addAllocation(b);
        r.rs->commit();
    }
    *p = host ? (void*) a.cpu : (void*) (uintptr_t) a.gpu;
    return kSuccess;
}

Error free_alloc(void* p) {
    if (!p) return kSuccess;
    Runtime& r = rt();
    size_t off = 0;
    const Alloc* a = find(p, &off);
    if (!a || off != 0) return fail(kErrorInvalidValue);
    device_synchronize();   // as cudaFree: the GPU is done with it
    std::lock_guard<std::mutex> lk(r.mu);
    MTL::Buffer* b = a->buf;
    const uint64_t g = a->gpu;
    r.by_cpu.erase((uintptr_t) a->cpu);
    r.by_gpu.erase(g);
    r.rs->removeAllocation(b);
    r.rs->commit();
    b->release();
    return kSuccess;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------- runtime.hpp
void* host_view(const void* p) {
    size_t off = 0;
    const Alloc* a = find(p, &off);
    return a ? a->cpu + off : nullptr;
}

// The Metal decode window reads the model itself; the device copies of the weights the shared host code uploads for
// the CUDA-shaped kernels (the pack's dense pool, the native head, ...) are never read on Metal.  On a small Mac their
// memory is worth more as expert arena: every device (not host) allocation of min_bytes or more is made purgeable-empty
// and dropped from the residency set.  Returns the bytes given back.
uint64_t purge_device_allocations(size_t min_bytes) {
    Runtime& r = rt();
    device_synchronize();
    std::lock_guard<std::mutex> lk(r.mu);
    uint64_t n = 0;
    for (auto& [gpu, a] : r.by_gpu) {
        if (a.host || a.purged || a.bytes < min_bytes) continue;
        r.rs->removeAllocation(a.buf);
        a.buf->setPurgeableState(MTL::PurgeableStateEmpty);
        a.purged = true;
        n += a.bytes;
    }
    r.rs->commit();
    return n;
}

Error get_last_error() { const Error e = t_last; t_last = kSuccess; return e; }
Error peek_last_error() { return t_last; }
const char* error_string(Error e) {
    switch (e) {
        case kSuccess: return "no error";
        case kNotReady: return "device not ready";
        case kErrorInvalidValue: return "invalid argument";
        case kErrorMemoryAllocation: return "out of memory";
        case kErrorUnsupported: return "operation not supported on Metal";
        case kErrorLaunchFailure: return "command buffer failed";
    }
    return "unknown Metal runtime error";
}

Error get_device(int* dev) { *dev = 0; return kSuccess; }
Error set_device(int dev) { return dev == 0 ? kSuccess : fail(kErrorInvalidValue); }
Error device_count(int* n) { *n = rt().dev ? 1 : 0; return kSuccess; }
Error device_synchronize() {
    Runtime& r = rt();
    StreamImpl* s = S(nullptr);
    flush(s);
    return wait_value(s->tl, s->submitted);
    (void) r;
}
Error mem_get_info(size_t* free_b, size_t* total_b) {
    // the GPU's budget is the working set the OS grants this process; what is free is that minus what is allocated
    MTL::Device* d = rt().dev;
    const uint64_t budget = d->recommendedMaxWorkingSetSize();
    const uint64_t used = d->currentAllocatedSize();
    *total_b = (size_t) budget;
    *free_b = (size_t) (used < budget ? budget - used : 0);
    return kSuccess;
}
Error device_integrated(int* v, int) { *v = 1; return kSuccess; }
Error device_sm_count(int* v, int) { *v = 0; return fail(kErrorUnsupported); }   // Apple reports no core count
Error device_clock_khz(int* v, int) { *v = 0; return fail(kErrorUnsupported); }
Error runtime_version(int* v) { *v = 0; return kSuccess; }
Error device_can_access_peer(int* v, int, int) { *v = 0; return kSuccess; }
Error device_enable_peer_access(int, unsigned) { return fail(kErrorUnsupported); }
Error get_device_properties(DeviceProp* p, int dev) {
    if (dev != 0) return fail(kErrorInvalidValue);
    *p = DeviceProp{};
    MTL::Device* d = rt().dev;
    std::snprintf(p->name, sizeof p->name, "%s", d->name()->utf8String());
    std::snprintf(p->gcnArchName, sizeof p->gcnArchName, "%s", d->supportsFamily(MTL::GPUFamilyApple10) ? "apple10" : "apple9-or-older");
    p->warpSize = 32;   // the SIMD-group width of every Apple GPU
    p->totalGlobalMem = (size_t) d->recommendedMaxWorkingSetSize();
    return kSuccess;
}

Error malloc_device_raw(void** p, size_t bytes) { return new_alloc(p, bytes, false); }
Error free_device(void* p) { return free_alloc(p); }
Error host_alloc_raw(void** p, size_t bytes, unsigned) { return new_alloc(p, bytes, true); }
Error free_host(void* p) {
    {
        Runtime& r = rt();
        std::lock_guard<std::mutex> lk(r.mu);
        auto it = r.by_cpu.find((uintptr_t) p);
        if (it == r.by_cpu.end()) return fail(kErrorInvalidValue);
    }
    return free_alloc(p);
}
Error host_register(void* p, size_t bytes, unsigned) {
    // existing host memory (an mmapped file, an arena) as a NoCopy buffer: page-aligned start and length required
    const uintptr_t pg = 16384;
    if (((uintptr_t) p % pg) != 0) return fail(kErrorInvalidValue);
    Runtime& r = rt();
    const size_t n = (bytes + pg - 1) / pg * pg;
    MTL::Buffer* b = r.dev->newBuffer(p, n, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked, nullptr);
    if (!b) return fail(kErrorMemoryAllocation);
    Alloc a{b, b->gpuAddress(), (uint8_t*) b->contents(), n, true};
    alloc_log("host register", n);
    std::lock_guard<std::mutex> lk(r.mu);
    r.by_gpu[a.gpu] = a;
    r.by_cpu[(uintptr_t) a.cpu] = a.gpu;
    r.rs->addAllocation(b);
    r.rs->commit();
    return kSuccess;
}
Error host_unregister(void* p) { return free_alloc(p); }
Error host_device_pointer_raw(void** dev, void* host) {
    size_t off = 0;
    const Alloc* a = find(host, &off);
    if (!a) return fail(kErrorInvalidValue);
    *dev = (void*) (uintptr_t) (a->gpu + off);
    return kSuccess;
}
Error memset_device(void* p, int value, size_t bytes) {
    StreamImpl* s = S(nullptr);
    const Error e = encode_fill(s, p, value, bytes);
    flush(s);
    wait_value(s->tl, s->submitted);
    return e;
}
Error memset_async(void* p, int value, size_t bytes, Stream st) {
    StreamImpl* s = S(st);
    if (s->rec) {
        Op o;
        o.kind = Op::Fill; o.dst = p; o.value = value; o.bytes = bytes;
        s->rec->ops.push_back(std::move(o));
        return kSuccess;
    }
    return encode_fill(s, p, value, bytes);
}

Error memcpy_sync(void* dst, const void* src, size_t bytes, CopyKind) {
    StreamImpl* s = S(nullptr);
    const Error e = encode_copy(s, dst, src, bytes);
    flush(s);
    wait_value(s->tl, s->submitted);
    return e;
}
Error memcpy_2d_async(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, CopyKind k, Stream st) {
    for (size_t r = 0; r < height; ++r) {   // skeleton: one copy per row (rare in the engine)
        const Error e = memcpy_async((uint8_t*) dst + r * dpitch, (const uint8_t*) src + r * spitch, width, k, st);
        if (e != kSuccess) return e;
    }
    return kSuccess;
}
Error memcpy_peer_async(void* dst, int dst_dev, const void* src, int src_dev, size_t bytes, Stream st) {
    if (dst_dev != 0 || src_dev != 0) return fail(kErrorInvalidValue);   // one GPU
    return memcpy_async(dst, src, bytes, kDeviceToDevice, st);
}
Error memcpy_async(void* dst, const void* src, size_t bytes, CopyKind, Stream st) {
    StreamImpl* s = S(st);
    if (s->rec) {
        Op o;
        o.kind = Op::Copy; o.dst = dst; o.src = src; o.bytes = bytes;
        s->rec->ops.push_back(std::move(o));
        return kSuccess;
    }
    return encode_copy(s, dst, src, bytes);
}

Error stream_create(Stream* s, unsigned) { *s = make_stream(); return kSuccess; }
Error stream_destroy(Stream s) {
    if (!s) return kSuccess;
    flush(s);
    wait_value(s->tl, s->submitted);
    s->tl->release();
    s->q->release();
    delete s;
    return kSuccess;
}
Error stream_synchronize(Stream st) {
    StreamImpl* s = S(st);
    flush(s);
    return wait_value(s->tl, s->submitted);
}
Error stream_query(Stream st) {
    StreamImpl* s = S(st);
    flush(s);
    return s->tl->signaledValue() >= s->submitted ? kSuccess : kNotReady;
}
Error stream_wait_event(Stream st, Event e, unsigned) {
    StreamImpl* s = S(st);
    if (s->rec) {
        Op o;
        o.kind = Op::Wait; o.ev = e;
        s->rec->ops.push_back(std::move(o));
        return kSuccess;
    }
    return wait_event(s, e);
}
Error launch_host_func(Stream st, HostFn fn, void* arg) {
    // skeleton: in stream order by completing the stream first (the engine's uses are rare; Phase 2 replaces this by a
    // completion handler + a shared-event wait so the host thread does not block)
    StreamImpl* s = S(st);
    if (s->rec) return fail(kErrorUnsupported);
    flush(s);
    wait_value(s->tl, s->submitted);
    fn(arg);
    return kSuccess;
}

Error event_create(Event* e, unsigned) { *e = new EventImpl; return kSuccess; }
Error event_destroy(Event e) { delete e; return kSuccess; }
Error event_record(Event e, Stream st) {
    StreamImpl* s = S(st);
    if (s->rec) {
        Op o;
        o.kind = Op::Record; o.ev = e;
        s->rec->ops.push_back(std::move(o));
        return kSuccess;
    }
    return record_event(e, s);
}
Error event_query(Event e) { return !e->ev || e->ev->signaledValue() >= e->value ? kSuccess : kNotReady; }
Error event_synchronize(Event e) { return e->ev ? wait_value(e->ev, e->value) : kSuccess; }
Error event_elapsed_ms(float* ms, Event, Event) { *ms = 0.0f; return fail(kErrorUnsupported); }

Error begin_capture(Stream st) {
    StreamImpl* s = S(st);
    if (s->rec) return fail(kErrorInvalidValue);
    flush(s);
    s->rec = new GraphImpl;
    return kSuccess;
}
Error end_capture(Stream st, Graph* g) {
    StreamImpl* s = S(st);
    if (!s->rec) { *g = nullptr; return fail(kErrorInvalidValue); }
    *g = s->rec;
    s->rec = nullptr;
    return kSuccess;
}
Error graph_node_count(Graph g, size_t* n) { *n = g ? g->ops.size() : 0; return kSuccess; }
Error graph_instantiate(GraphExec* x, Graph g) {
    *x = new GraphExecImpl{g->ops};
    return kSuccess;
}
Error graph_launch(GraphExec x, Stream st) {
    StreamImpl* s = S(st);
    if (s->rec) {   // a graph launched into a capture becomes part of it (CUDA: a child graph)
        s->rec->ops.insert(s->rec->ops.end(), x->ops.begin(), x->ops.end());
        return kSuccess;
    }
    for (const Op& o : x->ops) {
        const Error e = run_op(s, o);
        if (e != kSuccess) return e;
    }
    return kSuccess;
}
Error graph_upload(GraphExec, Stream) { return kSuccess; }
Error graph_destroy(Graph g) { delete g; return kSuccess; }
Error graph_exec_destroy(GraphExec x) { delete x; return kSuccess; }

// ---------------------------------------------------------------------------------------------------- metal.hpp
namespace metal {

MTL::Device* device() { return rt().dev; }

MTL::ComputePipelineState* pipeline(const char* source_name, const char* source, const char* function, std::string& err) {
    Runtime& r = rt();
    const std::string key = std::string(source_name) + "::" + function;
    std::lock_guard<std::mutex> lk(r.mu);
    if (auto it = r.pipes.find(key); it != r.pipes.end()) return it->second;
    MTL::Library*& lib = r.libs[source_name];
    NS::Error* e = nullptr;
    if (!lib) {
        MTL::CompileOptions* co = MTL::CompileOptions::alloc()->init();
        co->setMathMode(MTL::MathModeSafe);   // no fast-math reassociation: parity with the CUDA reference first
        lib = r.dev->newLibrary(NS::String::string(source, NS::UTF8StringEncoding), co, &e);
        co->release();
        if (!lib) {
            err = std::string("MSL compile of ") + source_name + ": " + (e ? e->localizedDescription()->utf8String() : "?");
            return nullptr;
        }
    }
    MTL::Function* f = lib->newFunction(NS::String::string(function, NS::UTF8StringEncoding));
    if (!f) { err = std::string("no MSL function ") + function; return nullptr; }
    MTL::ComputePipelineState* p = r.dev->newComputePipelineState(f, &e);
    f->release();
    if (!p) { err = std::string("pipeline ") + function + ": " + (e ? e->localizedDescription()->utf8String() : "?"); return nullptr; }
    r.pipes[key] = p;
    return p;
}

Error dispatch(Stream st, MTL::ComputePipelineState* p, const void* args, size_t arg_bytes, Grid groups, Grid threads,
               size_t tg_bytes) {
    if (arg_bytes > 4096) return fail(kErrorInvalidValue);   // setBytes limit
    {   // Metal skips a dispatch above the pipeline's maxTotalThreadsPerThreadgroup without any error (#2323): stop instead
        const uint64_t want = (uint64_t) threads.x * threads.y * threads.z;
        if (p && want > (uint64_t) p->maxTotalThreadsPerThreadgroup()) {
            Runtime& r = rt();
            std::string name = "?";
            {
                std::lock_guard<std::mutex> lk(r.mu);
                for (const auto& kv : r.pipes)
                    if (kv.second == p) { name = kv.first; break; }
            }
            std::fprintf(stderr,
                         "strata metal: the kernel %s can run at most %lu threads per threadgroup on this GPU (%s), but the "
                         "launch asks for %lu.\nMetal skips such a dispatch without any error, so every answer would be "
                         "wrong; stopping here instead.  Please report this (the GPU, macOS version and this line) at "
                         "https://github.com/Niko1221/Strata/issues .  (#2323)\n",
                         name.c_str(), (unsigned long) p->maxTotalThreadsPerThreadgroup(),
                         r.dev->name()->utf8String(), (unsigned long) want);
            std::fflush(stderr);
            std::exit(6);
        }
    }
    StreamImpl* s = S(st);
    Op o;
    o.kind = Op::Dispatch;
    o.pipe = p;
    if (arg_bytes <= sizeof o.args) { std::memcpy(o.args, args, arg_bytes); o.arg_bytes = arg_bytes; }
    else o.big_args.assign((const uint8_t*) args, (const uint8_t*) args + arg_bytes);
    o.groups = groups;
    o.threads = threads;
    o.tg_bytes = tg_bytes;
    if (s->rec) { s->rec->ops.push_back(std::move(o)); return kSuccess; }
    return encode_dispatch(s, o);
}

Stats stats() { return rt().stats; }

}  // namespace metal
}  // namespace strata::gpu
