# metal/ - Strata on Apple Silicon (Metal), MAC-PLAN.md.  Its own CMake project like sycl/, macOS only, never part of
# the CUDA / HIP / SYCL builds.
#
# Two parts:
#   * the kernel tree (mac-kern): the runtime face src/mtl_rt.*, the MSL kernels kernels/*.metal (compiled at run time),
#     CPU references src/ref_*.cpp and the parity / bench programs tests/*.cpp;
#   * the engine (mac-design): the shared host sources from ../src, UNCHANGED - include/metal_compat goes first on the
#     include path and gives them the CUDA runtime names on Metal, as include/strata/hip_compat does on HIP - over
#     the Metal device runtime src/gpu/runtime.cpp (metal-cpp) and the kernel library src/engine/*.
#
#   cmake -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release -DMETAL_CPP_DIR=... [-DSTRATA_GGML_DIR=...]
#   ninja -C build-metal
cmake_minimum_required(VERSION 3.24)
project(strata_metal LANGUAGES C CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "metal/ builds on macOS only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_OBJCXX_FLAGS_RELEASE "-O2")

get_filename_component(STRATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(GGML_COMMON "${STRATA_ROOT}/third_party/ggml/ggml-common.h")

# ---- the kernel tree (mac-kern) ------------------------------------------------------------------------------------
add_library(mtl_rt STATIC src/mtl_rt.mm src/ref_quant.cpp src/gguf_lite.cpp)
target_include_directories(mtl_rt PUBLIC src ${STRATA_ROOT}/third_party/ggml ${STRATA_ROOT}/include)
target_compile_options(mtl_rt PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_compile_definitions(mtl_rt PRIVATE
  STRATA_GGML_COMMON_DEFAULT="${GGML_COMMON}"
  STRATA_METAL_KERNELS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/kernels")
target_link_libraries(mtl_rt PUBLIC "-framework Metal" "-framework Foundation")

file(GLOB STRATA_METAL_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
list(REMOVE_ITEM STRATA_METAL_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_hello.cpp)
foreach(src ${STRATA_METAL_TESTS})
  get_filename_component(t ${src} NAME_WE)
  add_executable(${t} ${src})
  target_link_libraries(${t} PRIVATE mtl_rt)
endforeach()

# ---- the device runtime and the engine (mac-design) ---------------------------------------------------------------
set(METAL_CPP_DIR "" CACHE PATH "metal-cpp headers (https://developer.apple.com/metal/cpp/), e.g. metal-cpp_26")
if(NOT EXISTS "${METAL_CPP_DIR}/Metal/Metal.hpp")
  message(STATUS "strata metal: METAL_CPP_DIR is not set: the device runtime and the engine are not built")
  return()
endif()

add_library(strata_gpu_metal STATIC src/gpu/runtime.cpp)
target_include_directories(strata_gpu_metal BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/metal_compat
                           ${CMAKE_CURRENT_SOURCE_DIR}/include ${STRATA_ROOT}/include)
target_include_directories(strata_gpu_metal SYSTEM PUBLIC ${METAL_CPP_DIR})
target_compile_definitions(strata_gpu_metal PUBLIC STRATA_GPU_METAL=1)
target_link_libraries(strata_gpu_metal PUBLIC "-framework Metal" "-framework Foundation" "-framework QuartzCore")

add_library(strata_core_metal STATIC ${STRATA_ROOT}/src/core/graph.cpp)
target_link_libraries(strata_core_metal PUBLIC strata_gpu_metal)
add_executable(runtime_hello tests/runtime_hello.cpp)
target_link_libraries(runtime_hello PRIVATE strata_core_metal)
enable_testing()
add_test(NAME runtime_hello COMMAND runtime_hello)

# The engine: ggml (the native expert formats' type traits and ggml-cpu) from a llama.cpp checkout at the commit the
# PC build pins (CMakeLists.txt: 3cf03257).
option(STRATA_METAL_ENGINE "Build the engine (strata) for Metal" ON)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout at the pinned commit")
if(NOT STRATA_METAL_ENGINE)
  return()
endif()
if(NOT EXISTS "${STRATA_GGML_DIR}/ggml/CMakeLists.txt")
  message(FATAL_ERROR "STRATA_GGML_DIR='${STRATA_GGML_DIR}' is not a llama.cpp checkout")
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)

set(S ${STRATA_ROOT}/src)
set(STRATA_METAL_HOST_SOURCES
  ${S}/core/device.cu ${S}/core/pinned.cu ${S}/platform/memory.cpp ${S}/core/graph.cpp ${S}/core/weights.cpp ${S}/core/layout.cpp
  ${S}/core/layer.cpp ${S}/core/session.cpp ${S}/core/expert_source.cpp ${S}/core/remote_experts.cpp ${S}/core/expert_cache.cpp
  ${S}/core/native_head.cpp ${S}/core/native_dense.cpp ${S}/core/verify.cpp ${S}/core/mtp.cpp
  ${S}/core/conversation_snapshot.cpp ${S}/core/conversation_state.cpp ${S}/core/conversation_memory.cpp
  ${S}/core/peer_experts.cpp ${S}/core/arch_defaults.cpp ${S}/core/vmm.cpp ${S}/core/conversation_file.cpp
  ${S}/kernels/ngram.cpp ${S}/ngram/ple_reader.cpp ${S}/platform/direct_file.cpp
  ${S}/prefill/prefill.cpp
  ${S}/spec/suffix_drafter.cpp ${S}/spec/controller.cpp ${S}/spec/draft_policy.cpp ${S}/spec/draft_source.cpp
  ${S}/kernels/cpu/pool.cpp ${S}/kernels/cpu/expert_layout.cpp ${S}/kernels/cpu/native_expert.cpp
  ${S}/kernels/cpu/hadamard.cpp ${S}/kernels/cpu/tq_cpu.cpp)   # Gyro: the Hadamard rotation, the scalar trellis rows
set_source_files_properties(${S}/core/device.cu ${S}/core/pinned.cu PROPERTIES LANGUAGE CXX)

# The kernel library: Metal implementations (src/engine/*.cpp) and, for every entry point not ported yet, a stub
# that stops the engine with the entry point's name (src/engine/stubs.cpp, generated by tools/gen_stubs.py).
file(GLOB STRATA_METAL_ENGINE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/*.cpp)

add_library(strata_metal_host STATIC ${STRATA_METAL_HOST_SOURCES})
target_include_directories(strata_metal_host PUBLIC ${STRATA_ROOT}/include ${S}/core ${STRATA_ROOT}/third_party/ggml
                           ${STRATA_GGML_DIR}/ggml/include ${STRATA_GGML_DIR}/ggml/src ${STRATA_GGML_DIR}/ggml/src/ggml-cpu)
target_compile_definitions(strata_metal_host PUBLIC STRATA_NATIVE_EXPERTS=1 STRATA_VERSION="0.1.48-metal")
target_compile_options(strata_metal_host PRIVATE -Wno-unused-parameter -Wno-deprecated-declarations)
target_link_libraries(strata_metal_host PUBLIC strata_gpu_metal ggml-cpu ggml-base)

add_library(strata_metal_kernels STATIC ${STRATA_METAL_ENGINE_SOURCES})
target_include_directories(strata_metal_kernels PRIVATE src ${STRATA_ROOT}/third_party/ggml)
target_compile_definitions(strata_metal_kernels PRIVATE
  STRATA_GGML_COMMON_DEFAULT="${GGML_COMMON}" STRATA_METAL_KERNELS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
  STRATA_METAL_ENGINE_KERNELS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/src/engine/msl"
  STRATA_METAL_DATA_DEFAULT="${STRATA_ROOT}/data")
target_link_libraries(strata_metal_kernels PUBLIC strata_metal_host mtl_rt)

add_executable(strata ${S}/program/generate.cpp)
target_link_libraries(strata PRIVATE strata_metal_kernels strata_metal_host strata_metal_kernels)
