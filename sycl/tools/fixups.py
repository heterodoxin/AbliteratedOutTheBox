#!/usr/bin/env python3
"""Hand fixes on top of the SYCLomatic output (sycl/migrate.sh). Idempotent: re-run after a re-migration.

Each entry is a thing dpct 2025.3 got wrong or could not do, with the reason. Upstream files are never touched.
"""
import re, sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
changed = 0

def edit(rel, fn):
    global changed
    p = root / rel
    if not p.exists():
        print("  (absent)", rel); return
    s = p.read_text(); t = fn(s)
    if t != s:
        p.write_text(t); changed += 1; print("  fixed", rel)

def sub(pattern, repl, flags=0):
    return lambda s: re.sub(pattern, repl, s, flags=flags)

def all_sources():
    return [p.relative_to(root) for p in (root / "src").rglob("*.cpp")]

# 1. dpct's helper headers predate the 2026.1 compiler: the non-uniform group API was renamed.
edit("include/dpct/util.hpp", lambda s: s.replace("experimental::get_tangle_group(", "experimental::entangle(")
     .replace("experimental::get_fixed_size_group<", "experimental::chunked_partition<"))

for rel in all_sources():
    rel = str(rel)
    # 2. `(dpct::queue_ptr)stream->memcpy(...)`: the cast binds to the member call result, not the pointer.
    edit(rel, sub(r"\(dpct::queue_ptr\)\s*(\w+)\s*->", r"((dpct::queue_ptr)\1)->"))
    # 3. cudaStreamSynchronize(x) came out as `(cudaStream_t)x->wait()`; a null stream is the default queue.
    edit(rel, sub(r"\(cudaStream_t\)0->wait\(\)", "dpct::get_in_order_queue().wait()"))
    edit(rel, sub(r"\(cudaStream_t\)(\w+)->wait\(\)", r"((dpct::queue_ptr)\1)->wait()"))
    edit(rel, sub(r"DPCT_CHECK_ERROR\((main_cs|token_stream)->wait\(\)\)", r"DPCT_CHECK_ERROR(((dpct::queue_ptr)\1)->wait())"))
    # a null stream: the default in-order queue
    edit(rel, lambda s: s.replace("(nullptr)->ext_oneapi_graph(", "dpct::get_in_order_queue().ext_oneapi_graph("))
    # cudaGraphUpload has no SYCL counterpart (an executable command_graph is ready when finalized).
    edit(rel, sub(r"const dpct::err0 ue = cudaGraphUpload\(\w+(\[\w+\])?, \w+\);", "const dpct::err0 ue = 0;   // no cudaGraphUpload on SYCL: a finalized command_graph is already resident"))
    edit(rel, sub(r"^(\s*)cudaGraphUpload\(\w+, \w+\);", r"\1// no cudaGraphUpload on SYCL: a finalized command_graph is already resident", re.M))
    # __fadd_rn / __float_as_uint spellings
    for _ in range(3):   # nested __fadd_rn(__fadd_rn(a, b), c)
        # the second operand parenthesised: `__fadd_rn(sum, c ? x : y)` must not become `(sum + c ? x : y)`
        # (2026-10-01: it did, in the QSA indexer's pooled sum and the QSA block scores - the sum was replaced, not added)
        edit(rel, sub(r"__fadd_rn\((\([^()]*\)|[^,()]+),\s*([^()]+)\)", r"(\1 + (\2))"))
    edit(rel, lambda s: s.replace("__float_as_uint(", "sycl::bit_cast<uint32_t>("))
    # 4. volatile casts on kernel arguments (the kernels take plain pointers).
    edit(rel, lambda s: s.replace("(const volatile sycl::float4 *)", "(const sycl::float4 *)"))
    # 5. CUDA math intrinsics with no SYCL spelling.
    edit(rel, lambda s: s.replace("__isnanf(", "sycl::isnan(").replace("__isinff(", "sycl::isinf("))
    # 6. __nanosleep inside a doorbell spin: spin without the nap.
    edit(rel, sub(r"__nanosleep\(\d+\);", "/* spin (no __nanosleep on SYCL) */;"))
    # 7. `__fadd_rn(a, b ? c : d)` lost its parentheses.
    edit(rel, sub(r"= (\w+) \+ (\w+) \? (\w+\[\w+\]) : 0\.0f;", r"= \1 + (\2 ? \3 : 0.0f);"))
    # 8. cudaStreamQuery(s): dpct writes `DPCT_CHECK_ERROR(s->ext_oneapi_empty())`, which is 0 whenever the call does not
    #    throw, so the ring waits' `q != 1` guard (1 = still running) was always true and any wait over 2 ms "never rang".
    edit(rel, sub(r"(const dpct::err0 q =)\s*DPCT_CHECK_ERROR\(\(+(\w+)\)*->ext_oneapi_empty\(\)\)+;",
                  r"\1 \2->ext_oneapi_empty() ? 0 : 1;"))

# 2b. every `(dpct::queue_ptr) stream` cast goes through strata::q_of(), which maps CUDA's null stream to the
#     default in-order queue instead of dereferencing a null sycl::queue* (include/strata/sycl_queue.hpp).
def q_of(s):
    t = re.sub(r"\(\(sycl::queue \*\)\(\(dpct::queue_ptr\)\s*(\w+)\)\)", r"strata::q_of(\1)", s)
    t = re.sub(r"(?<!\w)\(\(dpct::queue_ptr\)\s*(\w+)\)", r"strata::q_of(\1)", t)   # not a call's own paren
    t = re.sub(r"\(dpct::queue_ptr\)\s*((?:\w+(?:->|\.))*\w+)\b", r"strata::q_of(\1)", t)
    t = re.sub(r"static_cast<dpct::queue_ptr>\((\w+)\)", r"strata::q_of(\1)", t)
    if t != s and '#include "strata/sycl_queue.hpp"' not in t:
        t = t.replace("#include <dpct/dpct.hpp>\n", '#include <dpct/dpct.hpp>\n#include "strata/sycl_queue.hpp"\n', 1)
    return t
for rel in all_sources():
    edit(str(rel), q_of)

# 7b. `__ldg((const float*) p)` lost its cast: dpct rewrote it as `*p`, reading one byte of the scale.
edit("src/kernels/cuda/s_gemv.dp.cpp", lambda s: s.replace("const float d = *blk;", "const float d = *(const float*) blk;")
     .replace("Q8K ? *xb", "Q8K ? *(const float*) xb"))

# 7c. the doorbell: `volatile` device loads/stores of host-mapped flags do not bypass the GPU caches on Intel
#     (the spin never saw the host's write). System-scope atomic load/store do (include/strata/sycl_doorbell.hpp).
def doorbell(s):
    if "sycl_doorbell.hpp" not in s:
        s = s.replace('#include "strata/sycl_queue.hpp"\n', '#include "strata/sycl_queue.hpp"\n#include "strata/sycl_doorbell.hpp"\n', 1)
    s = s.replace("    *seq = *seq + 1u;", "    strata::sys_store(seq, strata::sys_load(seq) + 1u);")
    s = s.replace("    const uint32_t want = *seq;\n", "    const uint32_t want = strata::sys_load(seq);\n")
    s = s.replace("    while (*flag != want) /* spin (no __nanosleep on SYCL) */;", "    while (strata::sys_load(flag) != want) /* spin (no __nanosleep on SYCL) */;")
    s = s.replace("    while (*flag < value) /* spin (no __nanosleep on SYCL) */;", "    while (strata::sys_load(flag) < value) /* spin (no __nanosleep on SYCL) */;")
    s = s.replace("    if (*skip == value) return;", "    if (strata::sys_load(skip) == value) return;")
    s = s.replace("    *skip = ring;\n}", "    strata::sys_store(skip, ring);\n}")
    s = s.replace("        *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;", "        strata::sys_store(seq, strata::sys_load(seq) + 1u);")
    # bounded spins (see kSpinMax in sycl_doorbell.hpp)
    s = s.replace("    while (strata::sys_load(flag) != want) /* spin (no __nanosleep on SYCL) */;",
                  "    for (uint32_t spin = 0; spin < strata::kSpinMax && strata::sys_load(flag) != want; ++spin) {}")
    s = s.replace("    while (strata::sys_load(flag) < value) /* spin (no __nanosleep on SYCL) */;",
                  "    for (uint32_t spin = 0; spin < strata::kSpinMax && strata::sys_load(flag) < value; ++spin) {}")
    # upstream 0.1.31 spells the waits `while (*flag ...) strata_spin_pause();` and the ring with 4-space indent:
    # same treatment (system-scope loads/stores, bounded: an unbounded orphaned spin wedges the B70's GT)
    s = s.replace("    *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;", "    strata::sys_store(seq, strata::sys_load(seq) + 1u);")
    s = s.replace("    while (*flag != want) strata_spin_pause();",
                  "    for (uint32_t spin = 0; spin < strata::kSpinMax && strata::sys_load(flag) != want; ++spin) strata_spin_pause();")
    s = s.replace("    while (*flag < value) strata_spin_pause();",
                  "    for (uint32_t spin = 0; spin < strata::kSpinMax && strata::sys_load(flag) < value; ++spin) strata_spin_pause();")
    return s
# 0.1.31: dp4a.hpp's spin pause is __nanosleep, which SYCL lacks: the bounded spin (kSpinMax) is the backoff
edit("include/strata/kernels/dp4a.hpp", lambda s: s.replace("    __nanosleep(100);\n",
     "    // SYCL port: no __nanosleep; the doorbell waits are bounded by strata::kSpinMax instead\n"))
# 0.1.31/0.1.32: fused_gr's per-block shared-memory query stays CUDA (dpct leaves the attribute untranslated)
edit("src/kernels/cuda/fused_gr.dp.cpp", lambda s: s.replace(
    "        cudaDeviceGetAttribute(&per_block, cudaDevAttrMaxSharedMemoryPerBlock, dev);\n",
    "        per_block = (int) dpct::get_device(dev).get_local_mem_size();   // SYCL: the work-group local memory\n"))
# upstream PR #413 / 0.1.36+: the key-head DeltaNet kernel's cp.async helpers keep CUDA's address conversion in the
# non-NVPTX branch; the SYCL build takes the plain-copy form (upstream's own pre-sm_80 path)
def gdn_plain_copies(s):
    s = s.replace("#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800\n#define STRATA_GDN_CP_ASYNC 0",
                  "#if 1   // SYCL port: plain copies (no cp.async)\n#define STRATA_GDN_CP_ASYNC 0")
    return s.replace("*reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);",
                     "*reinterpret_cast<sycl::float4*>(smem) = *reinterpret_cast<const sycl::float4*>(gmem);")
edit("src/prefill/kernels.dp.cpp", gdn_plain_copies)
# 0.1.38: upstream's sm_90 thread-block-cluster greedy sampler (cudaLaunchKernelEx with a cluster dimension) has no
# SYCL counterpart: the HIP branch's "not available" answer, so the caller takes the plain greedy kernel
edit("src/kernels/cuda/sampler.dp.cpp", lambda s: s.replace(
    "                           int *out, void *stream) try {\n#if defined(__HIPCC__)\n",
    "                           int *out, void *stream) try {\n#if 1   // SYCL port: no thread-block clusters (sm_90): the caller takes the plain greedy kernel\n"))
edit("src/kernels/cuda/qsa_select.dp.cpp", lambda s: s.replace(
    "                            void *stream) try {\n#if defined(__HIPCC__)\n",
    "                            void *stream) try {\n#if 1   // SYCL port: no thread-block clusters (sm_90): the caller takes the plain top-k\n"))
# 0.1.38: dpct could not deduce fused_gr's templated kernel names (the staged read and the MAX_T down kernels)
def gr_kernel_names(s):
    pat = re.compile(r",\s*dpct_placeholder /\*Fix the type mannually\*/>>\((.{0,900}?)\b(gr_down_staged_kernel|gr_down_multi_kernel<(\d+)>)\(", re.S)
    return pat.sub(lambda m: (">>(" if m.group(2) == "gr_down_staged_kernel" else f", dpct_kernel_scalar<{m.group(3)}>>>(")
                   + m.group(1) + m.group(2) + "(", s)
edit("src/kernels/cuda/fused_gr.dp.cpp", gr_kernel_names)
# 7c'. #1397: the bound of those spins is chosen per device at run time (strata::spin_max(queue), sycl_doorbell.hpp), not the
#      build's constant kSpinMax: each waiting kernel takes it as a `uint32_t spin_max` argument and its launcher passes
#      strata::spin_max(*strata::q_of(stream)). Idempotent: a file that already has the argument is left alone.
def spin_bound_at_run_time(s):
    for kernel, launcher in (("doorbell_wait_kernel", "doorbell_wait"), ("wait_flag_ge_or_kernel", "wait_flag_ge_or"),
                             ("wait_flag_ge_kernel", "wait_flag_ge")):
        sig = re.compile(r"(__dpct_inline__ void " + kernel + r"\([^)]*?)\)(\s*\{)")
        m = sig.search(s)
        if m and "spin_max" not in m.group(1):
            s = s[:m.start()] + m.group(1) + ", uint32_t spin_max)" + m.group(2) + s[m.end():]
        call = re.compile(r"(\b" + kernel + r"\((?:[^()]|\([^()]*\))*?)\);")
        s = call.sub(lambda c: c.group(0) if "spin_max" in c.group(1) else c.group(1) + ", spin_max);", s)
        head = re.compile(r"^(void " + launcher + r"\([^)]*\)(?: try)? \{\n)", re.M)
        h = head.search(s)
        if h and "strata::spin_max(" not in s[h.end():h.end() + 200]:
            at = h.end()
            guard = re.match(r"    if \([^\n]*\) return;\n", s[at:])   # a null-argument guard stays first
            if guard:
                at += guard.end()
            s = s[:at] + "    const uint32_t spin_max = strata::spin_max(*strata::q_of(stream));\n" + s[at:]
    return s.replace("strata::kSpinMax", "spin_max")
edit("src/kernels/cuda/elementwise.dp.cpp", lambda s: spin_bound_at_run_time(doorbell(s)))
edit("src/kernels/cuda/verify_kernels.dp.cpp", lambda s: spin_bound_at_run_time(doorbell(s)))

# 7d. cudaMemcpy / cudaMemset are synchronous; dpct emitted `get_in_order_queue().memcpy(...)` with no wait where
#     the source was pageable ("call wait() if needed"). The streaming expert source hands out ring-buffer blobs,
#     so an unwaited fill copies from a buffer that has already been reused (measured: non-deterministic experts).
#     Every default-queue memcpy/memset that is not already waited on gets its .wait() back.
def wait_default_queue_copies(s):
    out = []; i = 0; n = 0
    key = "dpct::get_in_order_queue()."
    while True:
        j = s.find(key, i)
        if j < 0: out.append(s[i:]); break
        k = j + len(key)
        if not (s.startswith("memcpy(", k) or s.startswith("memset(", k)):
            out.append(s[i:k]); i = k; continue
        p0 = s.find("(", k); depth = 0; q = p0
        while q < len(s):
            if s[q] == "(": depth += 1
            elif s[q] == ")":
                depth -= 1
                if depth == 0: break
            q += 1
        tail = s[q + 1:q + 8]
        if tail.startswith(".wait()"):
            out.append(s[i:q + 1]); i = q + 1; continue
        out.append(s[i:q + 1]); out.append(".wait()"); i = q + 1; n += 1
    return "".join(out)
for rel in all_sources():
    edit(str(rel), wait_default_queue_copies)

# 7e. dpct::dp4a is emulation (eight integer ops); sycl::ext::oneapi::dot_acc is the DP4A instruction on Xe.
def native_dp4a(s):
    if "dpct::dp4a(" not in s: return s
    s = s.replace("dpct::dp4a(", "strata::dp4a(")
    if "sycl_math.hpp" not in s:
        s = s.replace("#include <dpct/dpct.hpp>\n", '#include <dpct/dpct.hpp>\n#include "strata/sycl_math.hpp"\n', 1)
    return s
for rel in all_sources():
    edit(str(rel), native_dp4a)

# 7f. dpct's static global_memory/constant_memory objects destruct after the default queue at exit and segfault
#     in queue::get_device(). Heap-allocate them and never delete: the process is exiting anyway.
edit("src/kernels/cuda/native_mmvq.dp.cpp", lambda s: s.replace(
    "inline dpct::global_memory<int8_t, 1>\n    iq4nl_values(sycl::range(16), {",
    "inline dpct::global_memory<int8_t, 1>& iq4nl_values = *new dpct::global_memory<int8_t, 1>(sycl::range(16), {"))
edit("src/kernels/cuda/s2_gemv_fast.dp.cpp", lambda s: s.replace(
    "inline dpct::constant_memory<float, 2> c_codes(256, 4);",
    "inline dpct::constant_memory<float, 2>& c_codes = *new dpct::constant_memory<float, 2>(256, 4);   // never freed: exit-order safe"))

# 8. ggml-common.h has a SYCL declaration mode (sycl::half instead of cuda_fp16.h).
edit("src/kernels/cuda/iq_kernels.dp.cpp", lambda s: s.replace("#define GGML_COMMON_DECL_CUDA", "#define GGML_COMMON_DECL_SYCL")
     .replace("#define GGML_COMMON_IMPL_CUDA", "#define GGML_COMMON_IMPL_SYCL"))

# 8b. with the SYCL declaration mode the ggml tables are plain arrays, not dpct::global_memory objects.
edit("src/kernels/cuda/iq_kernels.dp.cpp", sub(r"\b(\w+)\.get_ptr\(\)", r"\1"))

edit("src/kernels/cuda/iq_kernels.dp.cpp", sub(r"^\s*\w+\.init\([^;]*\);\n", "", re.M))
#     ...and the kernels receive them as const pointers (dpct declared the parameters non-const).
TABLES = r"(iq2xxs_grid|iq2xs_grid|iq2s_grid|iq3xxs_grid|iq3s_grid|iq1s_grid_gpu|kmask_iq2xs|ksigns_iq2xs|ksigns64|kvalues_iq4nl)"
edit("src/kernels/cuda/iq_kernels.dp.cpp", sub(r"(?<!const )\b(uint64_t|uint32_t|uint8_t|int8_t) \*" + TABLES + r"\b", r"const \1 *\2"))
# 8c. one nested __fadd_rn the generic rewrite above does not reach
edit("src/kernels/cuda/native_qsa_score.dp.cpp", lambda s: s.replace(
    "float sum=__fadd_rn(((h[0] + h[1]) + h[2]),h[3]);", "float sum=(((h[0] + h[1]) + h[2]) + h[3]);"))

# 8d. dpct threaded every ggml table through kernel parameters, one name per template regardless of which table
#     that quant type needs. With plain `static const` arrays (usable from device code, as ggml-sycl does) the
#     device functions read the globals directly: drop the parameters, the arguments and the capture copies.
def untangle_tables(s):
    s = re.sub(r"^\s*auto \w+_ptr_ct\d+ = \w+;\n", "", s, flags=re.M)
    s = re.sub(r",\s*const (uint64_t|uint32_t|uint8_t|int8_t) \*" + TABLES + r"\b", "", s)
    s = re.sub(r"\(\s*const (uint64_t|uint32_t|uint8_t|int8_t) \*" + TABLES + r"\s*\)", "()", s)
    s = re.sub(r",\s*" + TABLES + r"(_ptr_ct\d+)?\b(?![\[\w(])", "", s)
    s = re.sub(r"\(\s*" + TABLES + r"(_ptr_ct\d+)?\s*\)(?![\[\w])", "()", s)
    return s
edit("src/kernels/cuda/iq_kernels.dp.cpp", untangle_tables)
#     the one helper that took the table under another name keeps its parameter; the callers name the table
edit("src/kernels/cuda/iq_kernels.dp.cpp", lambda s: s.replace("get_int_from_table_16(aux_q4);", "get_int_from_table_16(aux_q4, kvalues_iq4nl);"))

# 9. a ternary on the stream argument of cudaMemcpyAsync was migrated around the wrong operand.
edit("src/program/generate.cpp", lambda s: s.replace(
    "gs ? gs->adapt_stream\n                           : adapt_stream->memcpy(",
    "(gs ? gs->adapt_stream : adapt_stream)->memcpy("))

# 10. %globaltimer: there is no device-side wall clock in SPIR-V; the verify-window stage profiler reads zeros.
edit("src/kernels/cuda/verify_kernels.dp.cpp", lambda s: s.replace(
    'asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));',
    "t = 0;   // SYCL: no %globaltimer equivalent; the stage profiler is inert on this backend"))

# 11. The three tensor-core kernels (inline PTX: mma.sync, ldmatrix, cp.async). dpct cannot migrate assembly.
#     dpct.hpp defines DPCT_COMPATIBILITY_TEMP as 900, which selects the sm_80 path; select the other one, and
#     make the launchers that have no in-kernel fallback refuse the device (their callers then use the older
#     kernels, exactly as on a pre-Ampere card). The XMX joint_matrix versions are phase-4 follow-up work.
OFF = "#if 0   // SYCL: inline PTX (mma/ldmatrix/cp.async) - the XMX port is pending; see tools/fixups.py"
# 0.1.31: a Turing m16n8k8 branch (`#elif !STRATA_PA_SM80`) with its own inline PTX; dpct's DPCT_COMPATIBILITY_TEMP
# selects it on SYCL. Unreachable there (qsa_prompt_attn_batch refuses the device), so it compiles to nothing.
edit("src/kernels/cuda/qsa_prompt_attn.dp.cpp", lambda s: s.replace(
    '#elif !STRATA_PA_SM80\n    asm volatile("mma.sync.aligned.m16n8k8',
    '#elif 0   // SYCL: Turing\'s m16n8k8 PTX (unreachable: the launcher refuses this device)\n    asm volatile("mma.sync.aligned.m16n8k8'))
ON = "#if 1   // SYCL: the scalar path (the PTX one above is not ported yet)"
for rel in ["src/kernels/cuda/qsa_prompt_attn.dp.cpp", "src/kernels/cuda/qsa_select.dp.cpp",
            "src/kernels/cuda/native_qsa_score.dp.cpp"]:
    edit(rel, lambda s: s.replace("#if !defined(DPCT_COMPATIBILITY_TEMP) || DPCT_COMPATIBILITY_TEMP >= 800", OFF)
         .replace("#elif !defined(DPCT_COMPATIBILITY_TEMP) || DPCT_COMPATIBILITY_TEMP >= 800", OFF.replace("#if 0", "#elif 0"))   # after upstream's __HIPCC__ guard (0.1.27)
         .replace("#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800", ON)
         .replace("    __trap();", "    /* unreachable on SYCL: the launcher refuses this device */"))
for rel in ["src/kernels/cuda/qsa_prompt_attn.dp.cpp", "src/kernels/cuda/qsa_select.dp.cpp"]:
    edit(rel, lambda s: s.replace("        if (cc_major[dev] < 8) return false;",
         "        (void) cc_major[dev];\n        return false;   // SYCL: the tensor-core kernel is not ported yet; the caller takes the older kernel"))

# 12. IQ4_XS: dpct turned the __constant__ table into a kernel argument and gave that trait a 3-argument load(),
#     which the shared multi-column kernel cannot call. A constexpr copy of the table restores the 2-argument form.
def mmvq(s):
    if "kIq4nlTable" in s: return s   # already applied (the table is the marker)
    m = re.search(r"iq4nl_values[^;{]*\(sycl::range\(16\), \{([^}]*)\}", s, re.S)   # also after 8c's heap allocation
    vals = " ".join(m.group(1).split())
    table = "static constexpr int8_t kIq4nlTable[16] = {%s};\n" % vals
    s = s.replace("__dpct_inline__ sycl::int2 iq4_table_lookup(", table + "__dpct_inline__ sycl::int2 iq4_table_lookup(", 1)
    s = s.replace("    static float apply(const W &r, const Q81Block *__restrict__ x, int k,\n                       int8_t *iq4nl_values) {",
                  "    static float apply(const W &r, const Q81Block *__restrict__ x, int k) { return apply(r, x, k, const_cast<int8_t*>(kIq4nlTable)); }\n"
                  "    static float apply(const W &r, const Q81Block *__restrict__ x, int k,\n                       int8_t *iq4nl_values) {", 1)
    s = s.replace("    static W load(const Block* __restrict__ w, int iqs, int8_t *iq4nl_values) {",
                  "    static W load(const Block* __restrict__ w, int iqs) { return load(w, iqs, const_cast<int8_t*>(kIq4nlTable)); }\n"
                  "    static W load(const Block* __restrict__ w, int iqs, int8_t *iq4nl_values) {", 1)
    return s
edit("src/kernels/cuda/native_mmvq.dp.cpp", mmvq)

# 13. sycl::free takes void*; the parity harness frees const device pointers.
edit("src/kernels/qsa_prompt_attn_parity.cpp", sub(r"sycl::free\(([\w.]+),", r"sycl::free((void *)\1,"))

# 14. verify.cpp: graph introspection (kernel names, a debug print) has no SYCL API; the node count stays.
def verify(s):
    a = s.find("            if (ty == sycl::ext::oneapi::experimental::node_type::kernel) {")
    b = s.find("            } else if (ty == sycl::ext::oneapi::experimental::node_type::memcpy)")
    if a > 0 and b > a:
        s = s[:a] + "            if (ty == sycl::ext::oneapi::experimental::node_type::kernel) {\n                name = \"kernel\";   // SYCL: no kernel-name introspection on graph nodes\n" + s[b:]
    # cudaLaunchHostFunc(stream, fn, &fs): the flag set lives in a ring on the verifier, so a pointer to it is safe
    s = s.replace("""  v->copy_->submit([&](sycl::handler &cgh) {
    cgh.host_task([=]() {
      [](void *p) {
        FlagSet *s = (FlagSet *)p; raise_flag(s->flag, s->value);
      }(&fs);
    });
  });""", """  FlagSet* fsp = &fs;
  v->copy_->submit([&](sycl::handler &cgh) {
    cgh.host_task([=]() { raise_flag(fsp->flag, fsp->value); });
  });""")
    return s
edit("src/core/verify.cpp", verify)
# 15. cudaInitDevice(spin-wait scheduling, mapped host memory): device flags with no SYCL equivalent.
edit("src/core/remote_experts.cpp", lambda s: s.replace(
    "    if (!(spin && spin[0] == '0'))\n        cudaInitDevice(device, cudaDeviceScheduleSpin | cudaDeviceMapHost, 0);",
    "    (void) spin;   // SYCL: no cudaInitDevice scheduling flags; the runtime picks its own wait policy"))
# 16. graph.hpp: the event's timestamp is written from a const launch(); dpct added the member without `mutable`.
edit("include/strata/core/graph.hpp", lambda s: s.replace(
    "    std::chrono::time_point<std::chrono::steady_clock> done__ct1;", "    mutable std::chrono::time_point<std::chrono::steady_clock> done__ct1;"))
print("files changed:", changed)

# 0.1.29: a stream cast dpct leaves as cudaStream_t (sampler's capture check): the port's queue lookup.
for rel in all_sources():
    edit(str(rel), sub(r"\(\(cudaStream_t\)(\w+)\)->", r"strata::q_of(\1)->"))
# 0.1.29: sycl::free takes void*; NativeEmbed frees a const device pointer.
edit("src/core/native_head.cpp", lambda s: s.replace("sycl::free(dev_, dpct::get_in_order_queue())", "sycl::free((void *) dev_, dpct::get_in_order_queue())"))

# 2026-10-01: cudaMemcpy / cudaMemset on the legacy default stream wait for every earlier launch on the device's
# blocking streams; dpct's `get_in_order_queue().memcpy(...)` does not wait for work on another queue. The parity
# tests launch on their own queue (`s`, `cs`) and copy results back on the default one, so a slow kernel's output
# could be read before it finished (iq_multi_parity's IQ2_XS reference, qsa_parity's sequential side). In the tests,
# every default-queue copy/fill first drains the device's queues - cudaMemcpy's semantics.
for p in sorted((root / "src" / "kernels").glob("*_parity.cpp")):
    edit(str(p.relative_to(root)), sub(r"(?<!queues_wait_and_throw\(\), )dpct::get_in_order_queue\(\)(\s*)\.(memcpy|memset)\(",
                                       r"(dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())\1.\2("))

# 0.1.39 (#423, tmking01): generate.cpp's stage_room (the expert cache of every later stage of a layer split) lost its
# cudaMemGetInfo in the migration - the DPCT1106 note stayed, the call did not - so every later GPU read 0 bytes free.
edit("src/program/generate.cpp", sub(
    r"(    const strata::core::OnDevice on\(dev\);\n        size_t fb = 0, tb = 0;\n)(?!        dpct::get_current_device)",
    r"\1        dpct::get_current_device().get_memory_info(fb, tb);   // #423 (tmking01): dpct dropped cudaMemGetInfo here\n"))

# A770: only subgroup row zero reaches the native router's synchronization.
# CUDA's early-exit block pattern must not become a divergent SYCL work-group barrier.
edit("src/kernels/cuda/native_router.dp.cpp", lambda s: s.replace(
    "    item_ct1.barrier(sycl::access::fence_space::local_space);",
    "    sycl::group_barrier(item_ct1.get_sub_group());  // only subgroup row zero participates"))

# MtpDrafter: CUDA's `cudaStream_t cs_ = nullptr` came out as the device's default in-order queue. A drafter that never
# loads (no --mtp) kept it, and its destructor waited on and destroyed that queue: exit 139 after the output.
edit("include/strata/core/mtp.hpp", lambda s: s.replace(
    "    dpct::queue_ptr cs_ = &dpct::get_in_order_queue();\n",
    "    dpct::queue_ptr cs_ = nullptr;   // load() creates it; not the device's queue, which ~MtpDrafter would destroy\n"))

# `strata --version` must print before any GPU is touched (the engine smoke test of `strata update` runs it, and a user outside the
# `render` group has no usable device). Two things in dpct's helper headers looked up the SYCL devices before main:
# 1. blas::descriptor's `static inline queue_ptr _saved_queue_ptr = &get_default_queue();` (an inline variable): the default is
#    looked up in get_saved_queue() instead.
def lazy_saved_queue(s):
    s = s.replace("  static inline ::dpct::cs::queue_ptr _saved_queue_ptr =\n      &::dpct::cs::get_default_queue();",
                  "  static inline ::dpct::cs::queue_ptr _saved_queue_ptr = nullptr;")
    return s.replace("  static inline sycl::queue &get_saved_queue() noexcept {\n    return *_saved_queue_ptr;",
                     "  static inline sycl::queue &get_saved_queue() noexcept {\n"
                     "    if (!_saved_queue_ptr) _saved_queue_ptr = &::dpct::cs::get_default_queue();\n"
                     "    return *_saved_queue_ptr;")
edit("include/dpct/blas_utils.hpp", lazy_saved_queue)

# 2. dpct::device_memory (global_memory / constant_memory) sized its per-device pointer table, and so built mem_mgr and dev_mgr, in its
#    constructor. The engine's namespace-scope tables (iq4nl_values, c_codes) are heap objects built before main. Both now happen at the
#    first get_ptr_impl(); the destruction order the original comment protected does not arise for objects that are never freed.
def lazy_device_memory(s):
    s = s.replace("        _host_ptr(nullptr), _device_ptrs(get_ptrs_size(), nullptr) {\n", "        _host_ptr(nullptr) {\n")
    s = s.replace("    // Make sure that singleton class mem_mgr and dev_mgr will destruct later\n    // than this.\n"
                  "    detail::mem_mgr::instance();\n    dev_mgr::instance();\n  }\n",
                  "    // Strata: the pointer table, mem_mgr and dev_mgr are made at the first get_ptr_impl() (sycl/tools/fixups.py)\n  }\n")
    s = s.replace("  std::vector<value_t *> _device_ptrs;\n  sycl::queue _q;\n};",      # a default-built sycl::queue picks a device
                  "  std::vector<value_t *> _device_ptrs;\n  std::optional<sycl::queue> _q;   // Strata: a default-built sycl::queue picks a device (sycl/tools/fixups.py)\n};")
    return s.replace("  value_t *get_ptr_impl(sycl::queue &q) {\n#ifdef DPCT_USM_LEVEL_NONE\n",
                     "  value_t *get_ptr_impl(sycl::queue &q) {\n"
                     "    if (_device_ptrs.empty()) {\n      detail::mem_mgr::instance();\n      dev_mgr::instance();\n"
                     "      _device_ptrs.assign(get_ptrs_size(), nullptr);\n    }\n#ifdef DPCT_USM_LEVEL_NONE\n")
edit("include/dpct/detail/memory_detail.hpp", lazy_device_memory)

# 3. the migrated generate.cpp has no `--version` (src/program/generate.cpp answers it before any GPU is touched): the engine printed its
#    usage and exited 2. Same lines as the CUDA tree, at the top of main().
def version_flag(s):
    if '"--version"' in s:
        return s
    return s.replace("    std::setvbuf(stdout, nullptr, _IONBF, 0);\n#if (defined(STRATA_USE_HIP)",
                     "    std::setvbuf(stdout, nullptr, _IONBF, 0);\n"
                     "    // `strata --version` / `-V` (also `strata generate --version`): the version and nothing else, before any GPU is touched, as the CUDA\n"
                     "    // and HIP engines have answered it. `strata update` runs it as the engine's smoke test, and the Intel package build runs it with\n"
                     "    // an empty environment.\n"
                     "    for (int i = 1; i < argc && i < 3; ++i) {\n"
                     "        if (std::strcmp(argv[i], \"--version\") == 0 || std::strcmp(argv[i], \"-V\") == 0) {\n"
                     "            std::printf(\"strata %s\\n\", STRATA_VERSION);\n"
                     "            return 0;\n"
                     "        }\n"
                     "    }\n"
                     "#if (defined(STRATA_USE_HIP)", 1)
edit("src/program/generate.cpp", version_flag)
