// metal/src/mtl_rt.mm - see mtl_rt.h.  Built with ARC (-fobjc-arc).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mtl_rt.h"
#include "resource_path.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace mtl {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "mtl: cannot read %s\n", path.c_str());
        std::exit(2);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

struct Context::Impl {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    std::vector<id<MTLLibrary>> libs;
    id<MTLSharedEvent> ev = nil;   // run_graph_seg's handshake timeline
    uint64_t ev_v = 0;
};

static inline id<MTLBuffer> B(const Buffer& b) { return (__bridge id<MTLBuffer>)b.handle; }
// the MTLBuffer and offset that bind byte `off` of b (a chunked buffer: the chunk the offset selects)
static inline id<MTLBuffer> BO(const Buffer& b, size_t off, size_t& o) {
    if (!b.chunks) { o = off; return B(b); }
    const size_t k = std::min(off / b.stride, b.chunks->size() - 1);
    o = off - k * b.stride;
    return (__bridge id<MTLBuffer>)(*b.chunks)[k];
}
static inline id<MTLComputePipelineState> P(const Pipeline* p) { return (__bridge id<MTLComputePipelineState>)p->handle; }
static std::string g_device_name = "unknown GPU";

Context::Context() : impl_(new Impl) {
    impl_->dev = MTLCreateSystemDefaultDevice();
    if (!impl_->dev) {
        std::fprintf(stderr, "mtl: no Metal device\n");
        std::exit(2);
    }
    impl_->q = [impl_->dev newCommandQueue];
    // env, then <exe dir>/metal/ (a component package), then the compiled-in source-tree path
    ggml_common_path = strata_metal::resource("STRATA_GGML_COMMON", "ggml-common.h", STRATA_GGML_COMMON_DEFAULT);
    kernel_dir = strata_metal::resource("STRATA_METAL_KERNELS", "kernels", STRATA_METAL_KERNELS_DEFAULT);
}

Context::~Context() { delete impl_; }

std::string Context::device_name() const { return [[impl_->dev name] UTF8String]; }
bool Context::has_family_apple(int n) const {
    return [impl_->dev supportsFamily:(MTLGPUFamily)(MTLGPUFamilyApple1 + (n - 1))];
}
bool Context::supports_metal4() const {
    if (@available(macOS 26.0, *)) return [impl_->dev supportsFamily:MTLGPUFamilyMetal4];
    return false;
}
uint64_t Context::recommended_working_set() const { return [impl_->dev recommendedMaxWorkingSetSize]; }
uint64_t Context::max_buffer_length() const { return [impl_->dev maxBufferLength]; }

int Context::compile(const std::vector<std::string>& files, const std::string& prelude, bool fast_math) {
    std::string src = prelude + "\n#include <metal_stdlib>\nusing namespace metal;\n";
    if (!ggml_common_path.empty() && prelude.find("NO_GGML_COMMON") == std::string::npos) {
        src += "#define GGML_COMMON_DECL_METAL\n#define GGML_COMMON_IMPL_METAL\n";
        src += read_file(ggml_common_path);
        src += "\n";
    }
    for (const auto& f : files) {
        std::string p = (f.size() && f[0] == '/') ? f : kernel_dir + "/" + f;
        src += "#line 1 \"" + f + "\"\n" + read_file(p) + "\n";
    }
    MTLCompileOptions* o = [MTLCompileOptions new];
    // Safe math by default: no reassociation, so every token of a window is bitwise what a one-token call gives
    // (the kernels write their contractions as explicit fma).  Measured free on the M5 Max.
    fast_math = false;
    if (const char* m = std::getenv("STRATA_METAL_MATH")) fast_math = std::string(m) == "fast";
    if (@available(macOS 15.0, *)) {
        o.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
    } else {
        o.fastMathEnabled = fast_math;
    }
    NSError* err = nil;
    id<MTLLibrary> lib = [impl_->dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:o error:&err];
    if (!lib) {
        std::fprintf(stderr, "mtl: compile failed (%s, macOS %s; %s):\n%s\n", [[impl_->dev name] UTF8String],
                     [[[NSProcessInfo processInfo] operatingSystemVersionString] UTF8String],
                     files.empty() ? "?" : files.back().c_str(), [[err localizedDescription] UTF8String]);
        std::exit(3);
    }
    impl_->libs.push_back(lib);
    return (int)impl_->libs.size() - 1;
}

Pipeline Context::pipeline(int lib, const std::string& fn, const Constants& c) {
    NSError* err = nil;
    id<MTLFunction> f;
    NSString* name = [NSString stringWithUTF8String:fn.c_str()];
    if (c.vals.empty()) {
        f = [impl_->libs[lib] newFunctionWithName:name];
        // a function with (optional) function constants needs a specialization even when none is set
        if (f && f.functionConstantsDictionary.count > 0)
            f = [impl_->libs[lib] newFunctionWithName:name constantValues:[MTLFunctionConstantValues new] error:&err];
    } else {
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        for (const auto& v : c.vals) {
            int32_t iv = (int32_t)v.v;
            bool bv = v.v != 0;
            uint32_t uv = v.v;
            switch (v.type) {
                case 0: [cv setConstantValue:&iv type:MTLDataTypeInt atIndex:v.index]; break;
                case 1: [cv setConstantValue:&bv type:MTLDataTypeBool atIndex:v.index]; break;
                default: [cv setConstantValue:&uv type:MTLDataTypeUInt atIndex:v.index]; break;
            }
        }
        f = [impl_->libs[lib] newFunctionWithName:name constantValues:cv error:&err];
    }
    if (!f) {
        std::fprintf(stderr, "mtl: no function %s %s\n", fn.c_str(), err ? [[err localizedDescription] UTF8String] : "");
        std::exit(3);
    }
    MTLComputePipelineDescriptor* d = [MTLComputePipelineDescriptor new];
    d.computeFunction = f;
    d.supportIndirectCommandBuffers = YES;
    id<MTLComputePipelineState> ps = [impl_->dev newComputePipelineStateWithDescriptor:d
                                                                               options:MTLPipelineOptionNone
                                                                            reflection:nil
                                                                                 error:&err];
    if (!ps) {
        std::fprintf(stderr, "mtl: pipeline %s: %s\n", fn.c_str(), [[err localizedDescription] UTF8String]);
        std::exit(3);
    }
    Pipeline p;
    p.handle = (__bridge_retained void*)ps;
    p.max_threads = (int)ps.maxTotalThreadsPerThreadgroup;
    if (const char* cap = std::getenv("STRATA_METAL_MAX_THREADS_CAP")) {   // test hook, see require_fit
        const int lim = std::atoi(cap);
        if (lim > 0 && lim < p.max_threads) p.max_threads = lim;
    }
    p.simd_width = (int)ps.threadExecutionWidth;
    p.static_tg_mem = (int)ps.staticThreadgroupMemoryLength;
    p.name = fn;
    if (!c.vals.empty()) {
        p.name += "{";
        for (size_t i = 0; i < c.vals.size(); ++i)
            p.name += (i ? "," : "") + std::to_string(c.vals[i].index) + "=" + std::to_string((int)c.vals[i].v);
        p.name += "}";
    }
    g_device_name = [[impl_->dev name] UTF8String];
    return p;
}

void require_fit(const Pipeline& p, uint32_t threads, const char* where) {
    if (threads == 0 || (int64_t)threads <= (int64_t)p.max_threads) return;
    std::fprintf(stderr,
                 "strata metal: the kernel %s can run at most %d threads per threadgroup on this GPU (%s), but the %s asks "
                 "for %u.\nMetal skips such a dispatch without any error, so every answer would be wrong; stopping here "
                 "instead.  Please report this (the GPU, macOS version and this line) at "
                 "https://github.com/Niko1221/Strata/issues .  (#2323)\n",
                 p.name.empty() ? "?" : p.name.c_str(), p.max_threads, g_device_name.c_str(), where, threads);
    std::fflush(stderr);
    std::exit(6);
}

Buffer Context::alloc(size_t bytes) {
    id<MTLBuffer> b = [impl_->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) {
        std::fprintf(stderr, "mtl: alloc %zu failed\n", bytes);
        std::exit(4);
    }
    std::memset([b contents], 0, bytes);
    allocated_ += bytes;
    Buffer r;
    r.handle = (__bridge_retained void*)b;
    r.ptr = [b contents];
    r.bytes = bytes;
    return r;
}

Buffer Context::wrap(void* p, size_t bytes, size_t max_span) {
    const size_t pg = 16384, maxlen = [impl_->dev maxBufferLength] / pg * pg;
    if (bytes <= maxlen || std::getenv("STRATA_METAL_WRAP_SPLIT") && std::string(std::getenv("STRATA_METAL_WRAP_SPLIT")) == "0")
        return wrap(p, bytes);
    const size_t span = (max_span + pg - 1) / pg * pg;
    if (span >= maxlen / 2) {
        std::fprintf(stderr, "mtl: wrap %zu: a tensor of %zu bytes does not fit the chunks\n", bytes, max_span);
        std::exit(4);
    }
    Buffer r;
    r.chunks = std::make_shared<std::vector<void*>>();
    r.stride = maxlen - span;
    for (size_t base = 0; base < bytes; base += r.stride) {
        const size_t len = std::min(maxlen, (bytes - base + pg - 1) / pg * pg);
        id<MTLBuffer> b = [impl_->dev newBufferWithBytesNoCopy:(uint8_t*) p + base length:len
                                                        options:MTLResourceStorageModeShared deallocator:nil];
        if (!b) {
            std::fprintf(stderr, "mtl: wrap chunk %zu+%zu failed\n", base, len);
            std::exit(4);
        }
        r.chunks->push_back((__bridge_retained void*)b);
        if (base + len >= bytes) break;
    }
    allocated_ += bytes;
    r.handle = (*r.chunks)[0];
    r.ptr = p;
    r.bytes = bytes;
    std::fprintf(stderr, "mtl: wrapped %.2f GB as %zu overlapping buffers (maxBufferLength %.2f GB, overlap %.2f GB)\n",
                 bytes / 1e9, r.chunks->size(), maxlen / 1e9, span / 1e9);
    return r;
}

Buffer Context::wrap(void* p, size_t bytes) {
    id<MTLBuffer> b = [impl_->dev newBufferWithBytesNoCopy:p length:bytes options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) {
        std::fprintf(stderr, "mtl: wrap %zu failed\n", bytes);
        std::exit(4);
    }
    allocated_ += bytes;
    Buffer r;
    r.handle = (__bridge_retained void*)b;
    r.ptr = p;
    r.bytes = bytes;
    return r;
}

void Context::free(Buffer& b) {
    if (b.chunks) {
        for (void* h : *b.chunks) { id<MTLBuffer> x = (__bridge_transfer id<MTLBuffer>)h; x = nil; }
        b = Buffer{};
        return;
    }
    if (b.handle) {
        id<MTLBuffer> x = (__bridge_transfer id<MTLBuffer>)b.handle;
        x = nil;
    }
    b = Buffer{};
}

static void encode(id<MTLComputeCommandEncoder> enc, const Dispatch& d) {
    require_fit(d, "dispatch");
    [enc setComputePipelineState:P(d.p)];
    for (size_t i = 0; i < d.args.size(); ++i) {
        const Arg& a = d.args[i];
        if (a.buf) {
            size_t o;
            id<MTLBuffer> mb = BO(*a.buf, a.off, o);
            [enc setBuffer:mb offset:o atIndex:i];
        }
        else
            [enc setBytes:a.bytes.data() length:a.bytes.size() atIndex:i];
    }
    if (d.tg_mem) [enc setThreadgroupMemoryLength:(d.tg_mem + 15) / 16 * 16 atIndex:0];
    if (d.indirect)
        [enc dispatchThreadgroupsWithIndirectBuffer:B(*d.indirect)
                               indirectBufferOffset:d.ind_off
                              threadsPerThreadgroup:MTLSizeMake(d.tg[0], d.tg[1], d.tg[2])];
    else
        [enc dispatchThreadgroups:MTLSizeMake(d.grid[0], d.grid[1], d.grid[2])
            threadsPerThreadgroup:MTLSizeMake(d.tg[0], d.tg[1], d.tg[2])];
}

double Context::run(const std::vector<Dispatch>& ds, int reps, bool concurrent, bool barrier_between) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [impl_->q commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc =
            [cb computeCommandEncoderWithDispatchType:concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
        for (int r = 0; r < reps; ++r)
            for (const auto& d : ds) {
                encode(enc, d);
                if (barrier_between) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "mtl: command buffer failed: %s\n", [[cb.error localizedDescription] UTF8String]);
            std::exit(5);
        }
        return cb.GPUEndTime - cb.GPUStartTime;
    }
}

double Context::run_graph(const std::vector<Dispatch>& ds) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [impl_->q commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        for (size_t i = 0; i < ds.size(); ++i) {
            if (i > 0 && ds[i].barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode(enc, ds[i]);
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "mtl: command buffer failed: %s\n", [[cb.error localizedDescription] UTF8String]);
            std::exit(5);
        }
        return cb.GPUEndTime - cb.GPUStartTime;
    }
}

double Context::run_graph_seg(const std::vector<Dispatch>& ds, const std::vector<int>& cuts, void (*at_cut)(void*, int),
                              void* user) {
    @autoreleasepool {
        if (impl_->ev == nil) impl_->ev = [impl_->dev newSharedEvent];
        id<MTLSharedEvent> ev = impl_->ev;
        const uint64_t base = impl_->ev_v;
        id<MTLCommandBuffer> cb = [impl_->q commandBufferWithUnretainedReferences];
        size_t i = 0;
        for (size_t s = 0; s <= cuts.size(); ++s) {
            const size_t end = s < cuts.size() ? (size_t) cuts[s] : ds.size();
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            const size_t first = i;
            for (; i < end; ++i) {
                if (i > first && ds[i].barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode(enc, ds[i]);
            }
            [enc endEncoding];
            if (s < cuts.size()) {
                [cb encodeSignalEvent:ev value:base + 2 * s + 1];
                [cb encodeWaitForEvent:ev value:base + 2 * s + 2];
            }
        }
        [cb commit];
        for (size_t s = 0; s < cuts.size(); ++s) {
            const uint64_t want = base + 2 * s + 1;
            while (ev.signaledValue < want) {
                if (cb.status >= MTLCommandBufferStatusCompleted) break;   // an error: the wait below reports it
            }
            at_cut(user, (int) s);
            ev.signaledValue = base + 2 * s + 2;
        }
        impl_->ev_v = base + 2 * cuts.size() + 2;
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "mtl: command buffer failed: %s\n", [[cb.error localizedDescription] UTF8String]);
            std::exit(5);
        }
        return cb.GPUEndTime - cb.GPUStartTime;
    }
}

double Context::run_wall(const std::vector<Dispatch>& ds, int reps, double* gpu_seconds) {
    double gpu = 0;
    const double t0 = now_s();
    for (int r = 0; r < reps; ++r) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [impl_->q commandBufferWithUnretainedReferences];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (const auto& d : ds) encode(enc, d);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            gpu += cb.GPUEndTime - cb.GPUStartTime;
        }
    }
    if (gpu_seconds) *gpu_seconds = gpu;
    return now_s() - t0;
}

void Context::warm(const std::vector<Dispatch>& ds, double seconds) {
    const double t0 = now_s();
    while (now_s() - t0 < seconds) run(ds);
}

double Context::run_icb(const std::vector<Dispatch>& ds, int reps, bool barrier_between) {
    @autoreleasepool {
        size_t maxb = 1;
        for (const auto& d : ds) maxb = std::max(maxb, d.args.size());
        MTLIndirectCommandBufferDescriptor* desc = [MTLIndirectCommandBufferDescriptor new];
        desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        desc.inheritBuffers = NO;
        desc.inheritPipelineState = NO;
        desc.maxKernelBufferBindCount = maxb;
        id<MTLIndirectCommandBuffer> icb = [impl_->dev newIndirectCommandBufferWithDescriptor:desc
                                                                              maxCommandCount:ds.size()
                                                                                      options:0];
        NSMutableSet* used = [NSMutableSet new];
        for (size_t i = 0; i < ds.size(); ++i) {
            const Dispatch& d = ds[i];
            require_fit(d, "indirect-command dispatch");
            id<MTLIndirectComputeCommand> c = [icb indirectComputeCommandAtIndex:i];
            [c setComputePipelineState:P(d.p)];
            for (size_t a = 0; a < d.args.size(); ++a) {
                if (!d.args[a].buf) {
                    std::fprintf(stderr, "mtl: run_icb needs buffer arguments only\n");
                    std::exit(5);
                }
                size_t o;
                id<MTLBuffer> mb = BO(*d.args[a].buf, d.args[a].off, o);
                [c setKernelBuffer:mb offset:o atIndex:a];
                [used addObject:mb];
            }
            if (d.tg_mem) [c setThreadgroupMemoryLength:(d.tg_mem + 15) / 16 * 16 atIndex:0];
            [c concurrentDispatchThreadgroups:MTLSizeMake(d.grid[0], d.grid[1], d.grid[2])
                        threadsPerThreadgroup:MTLSizeMake(d.tg[0], d.tg[1], d.tg[2])];
            if (barrier_between) [c setBarrier];
        }
        id<MTLCommandBuffer> cb = [impl_->q commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        for (id<MTLBuffer> b in used) [enc useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        for (int r = 0; r < reps; ++r) {
            [enc executeCommandsInBuffer:icb withRange:NSMakeRange(0, ds.size())];
            if (barrier_between) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            std::fprintf(stderr, "mtl: icb command buffer failed: %s\n", [[cb.error localizedDescription] UTF8String]);
            std::exit(5);
        }
        return cb.GPUEndTime - cb.GPUStartTime;
    }
}

}  // namespace mtl
