# Strata on Intel GPUs - the SYCL build of the engine (Phase 4 of the Intel port, docs/INTEL.md).
#
# This is a separate project on purpose. The top-level build describes the CUDA engine; every source it lists is
# taken from ../src unless this directory carries a migrated copy: `src/X.cu` -> `sycl/src/X.dp.cpp`,
# `src/X.cpp` -> `sycl/src/X.cpp` (only host files that called the CUDA runtime were migrated). Headers resolve
# the same way through the include order. Upstream files are never edited, so the port is the content of this
# directory plus the tool that produced it (../../sycl/migrate.sh).
#
#   cmake -S sycl -B build-sycl -G Ninja -DCMAKE_CXX_COMPILER=icpx && cmake --build build-sycl
#
# Device code is JIT-compiled from SPIR-V unless STRATA_SYCL_AOT names a device (bmg-g31 for the Arc Pro B70).
cmake_minimum_required(VERSION 3.24)
# the engine's version is the project's (../CMakeLists.txt): the log line and BUILD.json say 0.1.40.2, not a number of the port's own
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt _strata_proj REGEX "^project\\(strata VERSION [0-9.]+")
string(REGEX MATCH "VERSION ([0-9.]+)" _m "${_strata_proj}")
if(CMAKE_MATCH_1)
  set(STRATA_SYCL_VERSION ${CMAKE_MATCH_1})
else()
  set(STRATA_SYCL_VERSION 0.1.40.2)
endif()
project(strata_sycl VERSION ${STRATA_SYCL_VERSION} LANGUAGES C CXX)
set(STRATA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
add_compile_definitions(STRATA_VERSION="${PROJECT_VERSION}")

set(STRATA_SYCL_AOT "" CACHE STRING "AOT device for the GPU code, e.g. bmg-g31 (empty: SPIR-V, JIT at run time)")
# The device spins on host flags are bounded (include/strata/sycl_doorbell.hpp). The bound is chosen per device at run
# time (20,000 reads on a B-series card, 2,000,000 elsewhere; STRATA_SPIN_MAX overrides). STRATA_SYCL_SPIN_MAX fixes one
# bound for every device in this build.
set(STRATA_SYCL_SPIN_MAX "" CACHE STRING "fixed bound of the device's spins on host flags, in reads (empty: chosen per device at run time)")
if(STRATA_SYCL_SPIN_MAX)
  add_compile_definitions(STRATA_SYCL_SPIN_MAX=${STRATA_SYCL_SPIN_MAX}u)
endif()
set(_sycl_flags -fsycl)
if(STRATA_SYCL_AOT MATCHES "^(bmg|lnl)")
  add_compile_definitions(STRATA_XMX_SG16=1)   # the grouped XMX kernels on sub-group 16, joint_matrix 8x16x16 (Xe2)
elseif(STRATA_SYCL_AOT MATCHES "^pvc")
  add_compile_definitions(STRATA_NO_DG2_XMX=1)   # the grouped XMX kernels (sub-group 8, 8x8x16) do not compile for Xe2 / Xe-HPC
endif()
# The fused int8 DPAS expert kernels (src/prefill/esimd_moe.dp.cpp, ESIMD, s8 8x16x32).  AUTO: built for the JIT build and for an AOT build of
# Xe2 (bmg, lnl); stubs for an AOT build of any other device.  OFF: stubs everywhere, available() is false and the prompt's experts take the
# grouped XMX path instead (slower prompts on Battlemage, same answers).  #3219: IGC 2.36.3 (Compute Runtime 26.22.38646.4, in an old
# strata-sycl-dev image) dies with an internal error on 77 of these kernels for bmg-g31; a newer IGC builds them.  ON keeps them on a device AUTO
# would stub.
set(STRATA_SYCL_I8_ESIMD "AUTO" CACHE STRING "the fused int8 ESIMD expert kernels: AUTO, ON or OFF (OFF: the XMX path, for a device compiler that cannot build them)")
set_property(CACHE STRATA_SYCL_I8_ESIMD PROPERTY STRINGS AUTO ON OFF)
if(NOT STRATA_SYCL_I8_ESIMD MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "STRATA_SYCL_I8_ESIMD must be AUTO, ON or OFF (got '${STRATA_SYCL_I8_ESIMD}')")
endif()
set(_i8_esimd ON)
if(STRATA_SYCL_I8_ESIMD STREQUAL "OFF")
  set(_i8_esimd OFF)
elseif(STRATA_SYCL_I8_ESIMD STREQUAL "AUTO" AND STRATA_SYCL_AOT AND NOT STRATA_SYCL_AOT MATCHES "^(bmg|lnl)")
  set(_i8_esimd OFF)   # 151-b70: DPAS s8 8x16x32 is Xe2-only: no AOT build for Alchemist / Xe-HPC
endif()
if(NOT _i8_esimd)
  add_compile_definitions(STRATA_I8_NO_ESIMD=1)
  message(STATUS "Strata SYCL: the fused int8 ESIMD expert kernels are off (STRATA_SYCL_I8_ESIMD=${STRATA_SYCL_I8_ESIMD}): stubs, the XMX path serves the prompt's experts")
endif()
if(STRATA_SYCL_AOT)
  list(APPEND _sycl_flags -fsycl-targets=spir64_gen)
  # the divide/sqrt rounding option (below) must reach ocloc here: an AOT build has no spir64 backend to take it
  set(_sycl_link -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen
      "-device ${STRATA_SYCL_AOT} -options -cl-fp32-correctly-rounded-divide-sqrt")
else()
  set(_sycl_link -fsycl)
endif()
# The Intel GPU's fp32 divide is not correctly rounded by default (OpenCL allows 2.5 ulp). Strata's quantizers
# are byte-exact against ggml through `amax / 127`, so the device compiler is told to round divides and sqrt
# properly; measured: quantize_act_parity goes from 303k mismatches to byte-exact. (Same as
# SYCL_PROGRAM_COMPILE_OPTIONS=-cl-fp32-correctly-rounded-divide-sqrt at run time.) The AOT build passes it to
# ocloc above; without that the 303k mismatches came back on 2026-09-30, unnoticed since the AOT switch.
if(NOT STRATA_SYCL_AOT)
  # The JIT build is the Alchemist (Arc A-series) build. There IGC turns a kernel's pointer arguments into surface
  # accesses with a 32-bit offset from the allocation's base, so a kernel misreads bytes more than 4 GiB into an
  # allocation while copies of the same bytes are right. Measured on an Arc A770 (Coder IQ1_M, 4.46 GiB expert cache):
  # every expert in the cache's last 0.46 GiB dequantized to NaN in the prompt path, and every prompt answered "!!!!"
  # (token 0). This option keeps the kernels on 64-bit (stateless) addresses; with it the same prompts are right.
  # (One backend argument for both: CMake drops a repeated -Xsycl-target-backend=spir64.)
  option(STRATA_SYCL_JIT_STATELESS "JIT build: keep kernels on 64-bit addresses (-ze-opt-greater-than-4GB-buffer-required)" ON)
  if(STRATA_SYCL_JIT_STATELESS)
    list(APPEND _sycl_link "-Xsycl-target-backend=spir64"
         "-cl-fp32-correctly-rounded-divide-sqrt -ze-opt-greater-than-4GB-buffer-required")
  else()
    list(APPEND _sycl_link "-Xsycl-target-backend=spir64" "-cl-fp32-correctly-rounded-divide-sqrt")
  endif()
endif()
# Every translation unit gets -fsycl: the migrated host files construct queues and USM, not only the kernels.
# icpx defaults to a fast floating-point model (reciprocal approximations, reassociation); the kernels are
# compared byte-for-byte against ggml's, which nvcc built value-safe, so ask for the same here.
# Every kernel was written for 32-lane warps (lane = tid & 31, warp-synchronous reductions). dpct pins the
# sub-group size only where it saw a shuffle; the rest would get Xe2's default of 16 and race. Force 32 everywhere.
list(APPEND _sycl_flags -fsycl-default-sub-group-size=32)
# One device module per kernel: the first launch of a kernel then loads only its own module. With the default
# split the first launch in the prompt path (the embedding gather) paid 245 ms for the image it shared (2026-09-30).
list(APPEND _sycl_link -fsycl-device-code-split=per_kernel)
list(APPEND _sycl_flags -fsycl-device-code-split=per_kernel)
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${_sycl_flags}>" -fp-model=precise -Wno-unused-parameter
                    -Wno-unused-variable -Wno-deprecated-declarations)
add_link_options(${_sycl_link})

# --- source resolution: the migrated copy when there is one, the original otherwise -------------------------------
function(strata_resolve out)
  set(r "")
  foreach(f IN LISTS ARGN)
    if(f MATCHES "\\.cu$")
      string(REGEX REPLACE "\\.cu$" ".dp.cpp" m "${f}")
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${m})
        list(APPEND r ${CMAKE_CURRENT_SOURCE_DIR}/${m})
      else()
        message(FATAL_ERROR "no migrated copy of ${f} (expected ${m})")
      endif()
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${f})
      list(APPEND r ${CMAKE_CURRENT_SOURCE_DIR}/${f})
    elseif(EXISTS ${STRATA_ROOT}/${f})
      list(APPEND r ${STRATA_ROOT}/${f})
    else()
      message(FATAL_ERROR "no such source: ${f}")
    endif()
  endforeach()
  set(${out} ${r} PARENT_SCOPE)
endfunction()

add_library(strata_includes INTERFACE)
target_include_directories(strata_includes INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include      # migrated headers + dpct/ helpers, first
  ${STRATA_ROOT}/include)
target_compile_options(strata_includes INTERFACE -Wall -Wextra)

# --- CPU-side pieces, identical to the top-level build ------------------------------------------------------------
strata_resolve(_spec src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp
                    src/spec/draft_source.cpp)
add_library(strata_spec STATIC ${_spec})
target_link_libraries(strata_spec PUBLIC strata_includes)

strata_resolve(_cpu src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp src/kernels/cpu/expert_layout.cpp
                    src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/hadamard.cpp)   # hadamard.cpp: expert_layout.cpp reads hadamard.txt
add_library(strata_kernels_cpu STATIC ${_cpu})
set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS
  "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
# AVX-VNNI copies of the AVX-2 expert kernels (q2_avx2.cpp, iq_avx2.cpp), taken at run time where the CPU has them
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -mf16c")
check_cxx_source_compiles([=[
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avxvnni")))
#endif
static __m256i dots(__m256i a, __m256i b) { return _mm256_dpbusd_avx_epi32(_mm256_dpwssd_avx_epi32(a, a, b), a, b); }
int main() {
    volatile int x = 1;
    return _mm256_extract_epi32(dots(_mm256_set1_epi32(x), _mm256_set1_epi32(x)), 0) == 0;
}
]=] STRATA_HAVE_AVXVNNI_INTRINSICS)
unset(CMAKE_REQUIRED_FLAGS)
if(STRATA_HAVE_AVXVNNI_INTRINSICS)
  set(_strata_avxvnni STRATA_AVXVNNI=1)
else()
  set(_strata_avxvnni STRATA_AVXVNNI=0)
endif()
set_property(SOURCE ${STRATA_ROOT}/src/kernels/cpu/q2_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC strata_includes Threads::Threads)

option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
# Q6_K gate/up in the grouped expert kernels (UD-Q6_K_XL, layer 2 of UD-Q5_K_XL): on by default, as in the CUDA / HIP
# build (MODEL-UPDATE Phase 1)
option(STRATA_Q6K_EXPERTS "Build the Q6_K gate/up grouped expert kernels (UD-Q6_K_XL, UD-Q5_K_XL)" ON)
if(STRATA_Q6K_EXPERTS)
  add_compile_definitions(STRATA_Q6K_EXPERTS=1)
endif()
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(STRATA_NATIVE_EXPERTS)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_SYCL OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  if(STRATA_GGML_DIR)
    add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(strata_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
    FetchContent_GetProperties(strata_llamacpp)
    if(NOT strata_llamacpp_POPULATED)
      FetchContent_Populate(strata_llamacpp)
    endif()
    add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  endif()
  strata_resolve(_cpu_native src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/iq_avx2.cpp
                             src/kernels/cpu/kq_avx2.cpp src/kernels/cpu/kq_avx1.cpp   # 0.1.31: the Q4_K/Q5_K/Q5_1/Q8_0 expert rows
                             src/kernels/cpu/q2_avx1.cpp src/kernels/cpu/iq_avx1.cpp   # 0.1.42: AVX1 (no AVX2) Q2_0 / i-quant rows
                             src/kernels/cpu/nvfp4_avx512.cpp src/kernels/cpu/nvfp4_avx2.cpp   # NVFP4 expert rows (native_expert.cpp calls them)
                             src/kernels/cpu/tq_cpu.cpp src/kernels/cpu/tq_avx2.cpp src/kernels/cpu/tq_avx512.cpp)   # Gyro trellis CPU rows (native_expert.cpp calls them; the SYCL GPU path does not run these types)
  target_sources(strata_kernels_cpu PRIVATE ${_cpu_native})
  target_include_directories(strata_kernels_cpu PRIVATE ${STRATA_ROOT}/third_party/ggml)
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/iq_avx512.cpp ${STRATA_ROOT}/src/kernels/cpu/nvfp4_avx512.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/tq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/tq_avx512.cpp PROPERTIES COMPILE_OPTIONS
    "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni;-mavx512vbmi;-mavx2;-mfma;-mf16c")
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/q2_avx1.cpp ${STRATA_ROOT}/src/kernels/cpu/iq_avx1.cpp PROPERTIES
    COMPILE_OPTIONS "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
  set_property(SOURCE ${STRATA_ROOT}/src/kernels/cpu/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_strata_avxvnni})
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
endif()

# --- the GPU engine ------------------------------------------------------------------------------------------------
strata_resolve(_core src/core/device.cu src/core/pinned.cu src/platform/memory.cpp src/core/graph.cpp
                     src/core/weights.cpp src/core/layout.cpp)
add_library(strata_core STATIC ${_core})
target_link_libraries(strata_core PUBLIC strata_includes)
# zeMemAllocHost for the host memory a kernel polls (include/strata/sycl_queue.hpp, host_malloc_polled): only when libze-dev's header is there
find_library(STRATA_ZE_LOADER ze_loader)
if(STRATA_ZE_LOADER)
  target_link_libraries(strata_core PUBLIC ${STRATA_ZE_LOADER})
endif()

set(_kernel_cu dequant_s2 s2_gemv s_gemv s2_gemv_quads s2_gemv_fast router_top10 quantize_act rope sampler s2_gemv_q8
    s2_expert_grouped shared_expert gr gdn qsa elementwise bf16_gemv native_mmvq native_bf16 native_gr_norm native_moe
    native_gdn native_gr_postops native_router native_gdn_preprocess native_qsa native_rope native_ple_postops
    native_flash_attn native_qsa_indexer native_qsa_score ple kv_q8 kv_stream kv_q4 fused_gr cvec qsa_decode_attn
    qsa_prompt_attn fused_gdn dequant_bf16 qsa_select verify_kernels iq_kernels)
set(_kernel_files "")
foreach(k IN LISTS _kernel_cu)
  list(APPEND _kernel_files src/kernels/cuda/${k}.cu)
endforeach()
strata_resolve(_kernels ${_kernel_files} src/kernels/ngram.cpp src/ngram/ple_reader.cpp src/platform/direct_file.cpp)
list(APPEND _kernels ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp)   # port-only: the XMX prompt attention
add_library(strata_kernels STATIC ${_kernels})
target_include_directories(strata_kernels PRIVATE ${STRATA_ROOT}/third_party/ggml)
target_link_libraries(strata_kernels PUBLIC strata_core)

strata_resolve(_engine src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp src/core/remote_experts.cpp
                       src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp
                       src/core/mtp.cpp src/core/conversation_snapshot.cpp src/core/conversation_state.cpp
                       src/core/conversation_memory.cpp   # 0.1.31: the conversation snapshots (#216)
                       src/core/arch_defaults.cpp src/core/vmm.cpp   # vmm.cpp is the port's stub: no driver VMM on Level Zero
                       src/core/conversation_file.cpp
                       src/core/peer_experts.cpp src/core/remote_expert_opt_stub.cpp)         # 0.1.36+: --peer-device (a stub in the port: no second GPU tier yet)
add_library(strata_engine STATIC ${_engine} ${CMAKE_CURRENT_SOURCE_DIR}/src/core/gguf_expert_source.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/core/copy_queue.cpp)   # 0.1.43: a queue on the copy engine
target_include_directories(strata_engine PRIVATE ${STRATA_ROOT}/src/core)   # conversation_checked.hpp (#include "...")
target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu)

# cuBLAS -> oneMKL through dpct::blas (needs the MKL SYCL BLAS domain)
strata_resolve(_prefill src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/prefill.cpp)
add_library(strata_prefill STATIC ${_prefill}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/moe_fused_stub.cpp   # upstream's fused int8 kernels: not built here
            ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/xmx_moe.dp.cpp         # A770 port: grouped XMX expert GEMM
            ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/esimd_moe.dp.cpp)       # 151-b70: fused int8 DPAS expert products (ESIMD, Battlemage)
target_compile_options(strata_prefill PUBLIC -qmkl=sequential)
target_link_options(strata_prefill PUBLIC -qmkl=sequential)
target_link_libraries(strata_prefill PUBLIC strata_engine)

strata_resolve(_main src/program/generate.cpp)
add_executable(strata ${_main})
target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec)

# --- the kernel parity self-tests (each runs on the card and compares against a CPU reference) ---------------------
option(STRATA_SYCL_PARITY "Build the kernel parity tests" ON)
if(STRATA_SYCL_PARITY)
  enable_testing()
  foreach(p IN ITEMS dequant_s2 s2_gemv shared_expert gr gdn s2_gemv_q8 sampler rope quantize_act router_top10
                     s_gemv elementwise bf16_gemv s_gemv_q8k qsa kv_q8 kv_stream kv_q4 cvec)
    strata_resolve(_src src/kernels/${p}_parity.cpp)
    add_executable(${p}_parity ${_src})
    target_include_directories(${p}_parity PRIVATE ${STRATA_ROOT}/src/kernels)   # the oracle .inc fixtures
    target_link_libraries(${p}_parity PRIVATE strata_kernels)
    add_test(NAME ${p}_parity COMMAND ${p}_parity --selftest)
  endforeach()
  target_link_libraries(gdn_parity PRIVATE strata_prefill)  # prefill recurrence variant tests
  add_test(NAME prefill_gdn_parity COMMAND gdn_parity --selftest --prefill-only)
  # #1795: iq_parity and ple_parity are registered as the top-level CMakeLists.txt does, not by the loop above
  # (iq_parity takes a fixture directory, not --selftest; ple_parity reads real artifacts from the engine root).
  foreach(p IN ITEMS iq ple)
    strata_resolve(_src src/kernels/${p}_parity.cpp)
    add_executable(${p}_parity ${_src})
    target_include_directories(${p}_parity PRIVATE ${STRATA_ROOT}/src/kernels)
    target_link_libraries(${p}_parity PRIVATE strata_kernels)
  endforeach()
  # iq: tools/iq_fixture.py writes the fixtures into the build tree (needs python3 with numpy; exit 3 = skip)
  set(_iq_python "")
  set(_iq_candidates "")
  if(IQ_FIXTURE_PYTHON)
    list(APPEND _iq_candidates ${IQ_FIXTURE_PYTHON})
  else()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
      list(APPEND _iq_candidates ${Python3_EXECUTABLE})
    endif()
    list(APPEND _iq_candidates "${STRATA_ROOT}/.venv/bin/python" "${STRATA_ROOT}/.venv/Scripts/python.exe" /usr/bin/python3 python3)
  endif()
  foreach(_cand IN LISTS _iq_candidates)
    if(_cand STREQUAL "")
      continue()
    endif()
    execute_process(COMMAND ${_cand} -c "import numpy" OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE _iq_probe)
    if(_iq_probe EQUAL 0)
      set(_iq_python "${_cand}")
      break()
    endif()
  endforeach()
  if(NOT _iq_python STREQUAL "")
    add_test(NAME iq_parity_fixtures COMMAND ${_iq_python} ${STRATA_ROOT}/tools/iq_fixture.py
                                             --out ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
    set_tests_properties(iq_parity_fixtures PROPERTIES FIXTURES_SETUP iq_fixture_data SKIP_RETURN_CODE 3)
    add_test(NAME iq_parity COMMAND iq_parity ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
    set_tests_properties(iq_parity PROPERTIES FIXTURES_REQUIRED iq_fixture_data SKIP_RETURN_CODE 3)
  else()
    message(STATUS "STRATA SYCL tests: no python3 with numpy found - iq_parity and its fixtures are not registered")
  endif()
  # ple: real inputs (the Q2_0 shard-2 GGUF, the dense pack, the ggml capture); a missing one is a hard error INSIDE
  # the test, as in the top-level build.  The working directory is the engine root, so the relative defaults resolve
  # there.  Point it at your files with -DSTRATA_PLE_GGUF=<shard 2 .gguf> and -DSTRATA_PLE_FIXTURE_DIR=<dir>.
  set(STRATA_PLE_GGUF "" CACHE FILEPATH "Q2_0 shard 2 GGUF that ple_parity reads (default: ../../Q2_0/... from the engine root)")
  set(STRATA_PLE_FIXTURE_DIR "${STRATA_ROOT}/bench/micro" CACHE PATH
      "Directory containing required ple_in.bin and ple_out.bin parity fixtures")
  set(_ple_args --selftest --in "${STRATA_PLE_FIXTURE_DIR}/ple_in.bin" --out "${STRATA_PLE_FIXTURE_DIR}/ple_out.bin")
  if(STRATA_PLE_GGUF)
    list(APPEND _ple_args --gguf "${STRATA_PLE_GGUF}")
  endif()
  add_test(NAME ple_parity COMMAND ple_parity ${_ple_args} WORKING_DIRECTORY ${STRATA_ROOT})
  # 0.1.31: the IQ kernels that decode a weight part once for every column/entry, bitwise vs the per-column ones
  # (no --selftest flag: it runs its checks by default), and the rewritten per-hit/grouped expert kernels vs the
  # previous ones, bitwise
  strata_resolve(_src src/kernels/iq_multi_parity.cpp)
  add_executable(iq_multi_parity ${_src})
  target_link_libraries(iq_multi_parity PRIVATE strata_kernels)
  add_test(NAME iq_multi_parity COMMAND iq_multi_parity)
  strata_resolve(_src src/kernels/s2_expert_grouped_parity.cpp)
  add_executable(s2_expert_grouped_parity ${_src})
  target_link_libraries(s2_expert_grouped_parity PRIVATE strata_kernels)
  add_test(NAME s2_expert_grouped_parity COMMAND s2_expert_grouped_parity --selftest)
  # upstream PR #363: native_expert_grouped's group stride and fused SwiGLU + q8_1, bitwise vs the v1 launches
  strata_resolve(_src src/kernels/native_grouped_parity.cpp)
  add_executable(native_grouped_parity ${_src})
  target_link_libraries(native_grouped_parity PRIVATE strata_kernels)
  add_test(NAME native_grouped_parity COMMAND native_grouped_parity)
  # 0.1.40 (#783): the batched verify-window kernels against the ones they replace, bitwise (synthetic, no model)
  foreach(p IN ITEMS verify native_multi kv_steps resident_plan gr_multi)
    strata_resolve(_src src/kernels/${p}_parity.cpp)
    add_executable(${p}_parity ${_src})
    target_include_directories(${p}_parity PRIVATE ${STRATA_ROOT}/src/kernels)
    target_link_libraries(${p}_parity PRIVATE strata_kernels)
    add_test(NAME ${p}_parity COMMAND ${p}_parity)
  endforeach()
  # not built here: qsa_topk_parity / qsa_topk_active_parity (the sm_90 thread-block-cluster QSA top-k has no SYCL
  # counterpart: the callers take the plain kernels), route_window_parity and fused_gr_bench (no migrated copy yet)
  # (upstream PR #413's gdn_rec_parity is an SM-holding NVIDIA bench - PTX timers, __nanosleep - not built here; the
  # key-head kernel is checked through the engine: output hashes with STRATA_GDN_KEYHEAD=1 and without)
  # the conversation snapshots (#216) against the engine, as upstream's STRATA_BUILD_CONVERSATION_TESTS
  strata_resolve(_src src/core/conversation_snapshot_test.cpp)
  add_executable(conversation_snapshot_test ${_src})
  target_link_libraries(conversation_snapshot_test PRIVATE strata_engine)
  add_test(NAME conversation_snapshot_test COMMAND conversation_snapshot_test)
  set_tests_properties(conversation_snapshot_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
  # the prompt-attention parity takes a context length, not --selftest, and needs the tensor-core kernel (not
  # ported yet: it reports "refused the pools" here). Built, not registered.
  strata_resolve(_src src/kernels/qsa_prompt_attn_parity.cpp)
  add_executable(qsa_prompt_attn_parity ${_src})
  target_link_libraries(qsa_prompt_attn_parity PRIVATE strata_kernels)
  # the hybrid K8V4 KV test (upstream 0.1.27) passes its appends, gathers and attention here and fails only its
  # last step, the same tensor-core prompt kernel. Built, not registered, until that kernel is ported.
  strata_resolve(_src src/kernels/kv_hybrid_parity.cpp)
  add_executable(kv_hybrid_parity ${_src})
  target_include_directories(kv_hybrid_parity PRIVATE ${STRATA_ROOT}/src/kernels)
  target_link_libraries(kv_hybrid_parity PRIVATE strata_kernels)
  add_executable(dequant_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/dequant_bench.cpp)   # prompt-path expert dequant timing + output hash
  target_link_libraries(dequant_bench PRIVATE strata_kernels)
  add_executable(mmvq_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_bench.cpp)   # dense mmvq timing
  target_link_libraries(mmvq_bench PRIVATE strata_kernels)
  add_executable(dense_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/dense_bench.cpp)   # the decode window's dense GEMVs, graph, hash
  target_link_libraries(dense_bench PRIVATE strata_kernels)
  add_executable(mmvq_sg_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mmvq_sg_bench.cpp)   # SIMD16 vs 32 decode mmvq
  target_link_libraries(mmvq_sg_bench PRIVATE strata_kernels)
  add_executable(q6k_align_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/q6k_align_bench.cpp)   # Q6_K block alignment
  target_link_libraries(q6k_align_bench PRIVATE strata_kernels)
  add_executable(gr_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/gr_bench.cpp)   # fused GR read timing
  target_link_libraries(gr_bench PRIVATE strata_kernels)
  add_executable(sycl_router_fast ${CMAKE_CURRENT_SOURCE_DIR}/probe/router_fast.cpp)
  target_link_libraries(sycl_router_fast PRIVATE strata_kernels)
  add_test(NAME sycl_router_fast COMMAND sycl_router_fast)
  set_tests_properties(sycl_router_fast PROPERTIES ENVIRONMENT "STRATA_SYCL_A770_FAST=1")
  add_executable(sycl_payload_pairs ${CMAKE_CURRENT_SOURCE_DIR}/probe/payload_pairs.cpp)
  target_link_libraries(sycl_payload_pairs PRIVATE strata_kernels)
  add_test(NAME sycl_payload_pairs COMMAND sycl_payload_pairs)
  set_tests_properties(sycl_payload_pairs PROPERTIES ENVIRONMENT "STRATA_SYCL_A770_FAST=1;STRATA_DOORBELL_CHECK=1")   # opt-in paths; publish_res copies x only with the check
  add_executable(xmx_gemm_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/xmx_gemm_bench.cpp)   # XMX quantized GEMM vs dequant+oneMKL
  target_link_libraries(xmx_gemm_bench PRIVATE strata_kernels strata_prefill)   # -qmkl for the oneMKL reference
  add_executable(xmx_group_parity ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/xmx_group_parity.cpp)   # A770: fused grouped XMX
  target_link_libraries(xmx_group_parity PRIVATE strata_kernels strata_prefill)
  add_executable(esimd_moe_parity ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/esimd_moe_parity.cpp)   # 151-b70: fused int8 expert products vs dequant+oneMKL and FP64
  target_link_libraries(esimd_moe_parity PRIVATE strata_kernels strata_prefill)
  add_test(NAME esimd_moe_parity COMMAND esimd_moe_parity 60 8 2)   # 151-b70: skips (exit 0) on a device without the s8 8x16x32 DPAS
  add_executable(esimd_moe_mode_test ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/esimd_moe_mode_test.cpp)   # 151-b70: the default choice of the expert chain (no GPU)
  target_include_directories(esimd_moe_mode_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME esimd_moe_mode_test COMMAND esimd_moe_mode_test)
  if(_i8_esimd)   # DPAS s8 8x16x32 (Xe2): not for an Alchemist / Xe-HPC AOT build, nor with STRATA_SYCL_I8_ESIMD=OFF
    add_executable(esimd_dpas_probe ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/esimd_dpas_probe.cpp)   # ESIMD DPAS peak (registers only)
    add_executable(esimd_i8_bench ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/esimd_i8_bench.cpp)   # ESIMD int8 DPAS GEMM on pre-decoded weights
  endif()
  # one native expert three ways on real GGUF rows (ggml reference, CPU path, GPU native_expert_grouped):
  #   native_expert_parity <shard1.gguf> [layer ...]
  if(STRATA_NATIVE_EXPERTS)
    add_executable(native_expert_parity ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE ${STRATA_ROOT}/third_party/ggml)
    target_link_libraries(native_expert_parity PRIVATE strata_kernels strata_kernels_cpu)
    # Gyro (APR) trellis experts on real GGUF tensors: tq_sycl_parity <pack> <model.gguf> [layer ...]
    add_executable(tq_sycl_parity ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/tq_sycl_parity.cpp)
    target_include_directories(tq_sycl_parity PRIVATE ${STRATA_ROOT}/third_party/ggml)
    target_compile_options(tq_sycl_parity PRIVATE -ffp-contract=off)
    target_link_libraries(tq_sycl_parity PRIVATE strata_engine strata_kernels strata_kernels_cpu)
    # random trellis codes (every bit pattern is valid): the GPU kernels against a host replay of their own arithmetic, bit for bit,
    # the prompt-path dequantizers against the fp16 codebook, and the Hadamard rotations against the CPU's
    add_test(NAME tq_sycl_parity_synthetic COMMAND tq_sycl_parity --synthetic 145/146 144/144 144/146 145/144)
    add_test(NAME tq_sycl_parity_synthetic_hadamard COMMAND tq_sycl_parity --synthetic --hadamard 145/146 144/144)
    # MODEL-UPDATE: the CPU pool's expert rows per format against the RAM bandwidth (as the top-level build's)
    add_executable(kq_cpu_bench ${STRATA_ROOT}/src/kernels/cpu/kq_cpu_bench.cpp)
    target_link_libraries(kq_cpu_bench PRIVATE strata_kernels_cpu)
  endif()
endif()
