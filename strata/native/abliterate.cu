#include "strata/kernels/abliterate.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace strata::kernels {
namespace {
constexpr int threads = 256;
constexpr int history = 25;
int layers = 0, width = 0, streams = 0, first = 0, last = -1, edit_first = 0;
int* mode = nullptr;  // 0: off, 1: capture, 2: edit
int* counts = nullptr;
float *directions = nullptr, *references = nullptr, *states = nullptr;
std::vector<bool> ready;

bool checked(cudaError_t result, std::string& err) {
    if (result == cudaSuccess) return true;
    err = cudaGetErrorString(result);
    return false;
}

bool set_mode(int value, std::string& err) {
    return checked(cudaDeviceSynchronize(), err) &&
           checked(cudaMemcpy(mode, &value, sizeof(value), cudaMemcpyHostToDevice), err);
}

__device__ float reduce(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int step = threads / 2; step; step /= 2) {
        if (threadIdx.x < step) scratch[threadIdx.x] += scratch[threadIdx.x + step];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

// One block per residual stream and token. The pending GR write uses cvec's arithmetic.
__global__ void project(float* R, int n, int hc, int64_t stride, const float* bo, int64_t bo_stride,
                        const float* inj, int64_t inj_stride, bool write, const int* flag,
                        const float* dirs, const float* refs, bool editable) {
    const int c = blockIdx.x;
    const int64_t t = blockIdx.y;
    float* r = R + t * stride + c * n;
    if (write) {
        const float weight = 2.0f * (1.0f / (1.0f + __expf(-inj[t * inj_stride + c] / float(hc))));
        for (int j = threadIdx.x; j < n; j += threads) r[j] = fmaf(bo[t * bo_stride + j], weight, r[j]);
        __syncthreads();
    }
    if (*flag != 2 || !editable) return;
    __shared__ float scratch[threads];
    for (int direction = 0; direction < 2; ++direction) {
        const float* v = dirs + direction * n;
        float dot = 0;
        for (int j = threadIdx.x; j < n; j += threads) dot += r[j] * v[j];
        const float excess = fmaxf(reduce(dot, scratch) - refs[direction], 0.0f);
        for (int j = threadIdx.x; j < n; j += threads) r[j] -= excess * v[j];
        __syncthreads();
    }
}

// Capture the mean over hyper-connection streams, retaining the last 25 token states per layer.
__global__ void capture(const float* R, int n, int hc, int64_t stride, int64_t T,
                        const int* flag, const int* count, float* output) {
    if (*flag != 1) return;
    const int64_t t = T - min(T, int64_t(history)) + blockIdx.y;
    const int slot = int((int64_t(*count) + t) % history);
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int c = 0; c < hc; ++c) sum += R[t * stride + c * n + j];
        output[slot * n + j] = sum / float(hc);
    }
}

__global__ void advance(int* count, int64_t T, const int* flag) {
    if (*flag == 1) {
        const int64_t total = int64_t(*count) + T;
        *count = total < history ? int(total) : history + int(total % history);
    }
}

bool finite_number(std::istringstream& input, float& value) {
    return bool(input >> value) && std::isfinite(value);
}
}  // namespace

void abliterate_free() {
    cudaDeviceSynchronize();
    cudaFree(mode); cudaFree(counts); cudaFree(directions); cudaFree(references); cudaFree(states);
    mode = nullptr; counts = nullptr; directions = nullptr; references = nullptr; states = nullptr;
    layers = width = streams = 0; last = -1; ready.clear();
}

bool abliterate_init(int L, int N, int HC, std::string& err) {
    if (L < 3 || L > 256 || N < 1 || N > 32768 || HC < 1 || HC > 64) {
        err = "unsupported abliteration geometry";
        return false;
    }
    if (mode) { err = "abliteration already initialized"; return false; }
    layers = L; width = N; streams = HC;
    first = L / 3; last = 2 * L / 3; edit_first = 3 * L / 7;
    ready.assign(L, false);
    const size_t vectors = size_t(L) * 2 * N;
    const size_t samples = size_t(L) * history * N;
    if (!checked(cudaMalloc(&mode, sizeof(int)), err) ||
        !checked(cudaMalloc(&counts, size_t(L) * sizeof(int)), err) ||
        !checked(cudaMalloc(&directions, vectors * sizeof(float)), err) ||
        !checked(cudaMalloc(&references, size_t(L) * 2 * sizeof(float)), err) ||
        !checked(cudaMalloc(&states, samples * sizeof(float)), err) ||
        !checked(cudaMemset(mode, 0, sizeof(int)), err) ||
        !checked(cudaMemset(counts, 0, size_t(L) * sizeof(int)), err) ||
        !checked(cudaMemset(directions, 0, vectors * sizeof(float)), err) ||
        !checked(cudaMemset(references, 0, size_t(L) * 2 * sizeof(float)), err) ||
        !checked(cudaMemset(states, 0, samples * sizeof(float)), err)) {
        abliterate_free();
        return false;
    }
    return true;
}

bool abliterate_covers(int64_t layer) { return mode && layer >= first && layer <= last; }

bool abliterate_read(int take, std::vector<float>& output, std::string& err) {
    if (!mode || take < 1 || take > history) { err = "READ expects 1..25 states"; return false; }
    std::vector<int> cursor(size_t(layers), 0);
    std::vector<float> host(size_t(layers) * history * width);
    if (!checked(cudaDeviceSynchronize(), err) ||
        !checked(cudaMemcpy(cursor.data(), counts, cursor.size() * sizeof(int), cudaMemcpyDeviceToHost), err) ||
        !checked(cudaMemcpy(host.data(), states, host.size() * sizeof(float), cudaMemcpyDeviceToHost), err)) return false;
    output.clear();
    output.reserve(size_t(last - first + 1) * take * width);
    for (int l = first; l <= last; ++l) {
        if (cursor[size_t(l)] < take) { err = "missing layer-output capture at layer " + std::to_string(l); return false; }
        for (int t = 0; t < take; ++t) {
            const int slot = (cursor[size_t(l)] + history - take + t) % history;
            const float* begin = host.data() + (size_t(l) * history + slot) * width;
            for (int j = 0; j < width; ++j) {
                if (!std::isfinite(begin[j])) { err = "non-finite captured activation"; return false; }
                output.push_back(begin[j]);
            }
        }
    }
    return true;
}

void abliterate_apply(float* R, int64_t layer, int64_t T, int64_t stride,
                     const float* bo, int64_t bo_stride, const float* inj, int64_t inj_stride,
                     bool write, void* stream) {
    if (!abliterate_covers(layer) || T < 1) return;
    if (!R || stride < int64_t(width) * streams || T > 65535 || (write && (!bo || !inj)))
        throw std::runtime_error("abliteration: invalid residual layout");
    const auto cs = (cudaStream_t) stream;
    project<<<dim3(unsigned(streams), unsigned(T)), threads, 0, cs>>>(R, width, streams, stride,
        bo, bo_stride, inj, inj_stride, write, mode, directions + layer * 2 * width,
        references + layer * 2, layer >= edit_first);
    capture<<<dim3(unsigned((width + threads - 1) / threads), unsigned(std::min<int64_t>(T, history))), threads, 0, cs>>>(
        R, width, streams, stride, T, mode, counts + layer, states + layer * history * width);
    advance<<<1, 1, 0, cs>>>(counts + layer, T, mode);
    if (cudaPeekAtLastError() != cudaSuccess) throw std::runtime_error("abliteration: kernel launch failed");
}

bool abliterate_command(const std::string& line) {
    if (line.rfind("ABL ", 0) != 0) return false;
    std::string err;
    try {
        if (!mode) throw std::runtime_error("start the engine with --abliterate");
        std::istringstream input(line.substr(4));
        std::string cmd, extra;
        input >> cmd;
        if (cmd == "INFO") {
            if (input >> extra) throw std::runtime_error("unexpected argument");
            std::printf("ABL INFO %d %d %d %d\n", layers, width, streams, history);
        } else if (cmd == "OFF" || cmd == "CAPTURE" || cmd == "ON") {
            if (input >> extra) throw std::runtime_error("unexpected argument");
            if (cmd == "ON") {
                for (int l = edit_first; l <= last; ++l)
                    if (!ready[size_t(l)]) throw std::runtime_error("missing direction at layer " + std::to_string(l));
            }
            if (!set_mode(cmd == "ON" ? 2 : cmd == "CAPTURE" ? 1 : 0, err)) throw std::runtime_error(err);
            if (cmd == "CAPTURE") {
                std::fill(ready.begin(), ready.end(), false);
                if (!checked(cudaMemset(counts, 0, size_t(layers) * sizeof(int)), err) ||
                    !checked(cudaMemset(states, 0, size_t(layers) * history * width * sizeof(float)), err))
                    throw std::runtime_error(err);
            }
            std::printf("ABL OK\n");
        } else if (cmd == "SET") {
            int l = -1;
            float refs[2];
            if (!(input >> l) || l < edit_first || l > last ||
                !finite_number(input, refs[0]) || !finite_number(input, refs[1]))
                throw std::runtime_error("invalid layer or references");
            std::vector<float> dirs(size_t(width) * 2);
            for (float& x : dirs) if (!finite_number(input, x)) throw std::runtime_error("invalid direction");
            if (input >> extra) throw std::runtime_error("unexpected direction values");
            for (int k = 0; k < 2; ++k) {
                double norm = 0;
                for (int j = 0; j < width; ++j) norm += double(dirs[size_t(k) * width + j]) * dirs[size_t(k) * width + j];
                if (norm != 0 && std::abs(norm - 1.0) > 0.01) throw std::runtime_error("direction must be unit length or zero");
            }
            if (!set_mode(0, err) ||
                !checked(cudaMemcpy(directions + l * 2 * width, dirs.data(), dirs.size() * sizeof(float), cudaMemcpyHostToDevice), err) ||
                !checked(cudaMemcpy(references + l * 2, refs, sizeof(refs), cudaMemcpyHostToDevice), err))
                throw std::runtime_error(err);
            ready[size_t(l)] = true;
            std::printf("ABL OK\n");
        } else if (cmd == "READ") {
            int take = 0;
            if (!(input >> take) || take < 1 || take > history || (input >> extra)) throw std::runtime_error("READ expects 1..25 states");
            std::vector<float> host;
            if (!abliterate_read(take, host, err)) throw std::runtime_error(err);
            for (int l = first; l <= last; ++l) {
                std::printf("ABL ROW %d", l);
                for (int t = 0; t < take; ++t) {
                    for (int j = 0; j < width; ++j) std::printf(" %.9g", host[(size_t(l - first) * take + t) * width + j]);
                }
                std::printf("\n");
            }
            std::printf("ABL END\n");
        } else throw std::runtime_error("unknown ABL command");
    } catch (const std::exception& e) {
        if (mode) { std::string ignored; set_mode(0, ignored); }
        std::printf("ERR abliterate: %s\n", e.what());
    }
    std::fflush(stdout);
    return true;
}

}  // namespace strata::kernels
