#include "strata/kernels/abliterate.hpp"
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

namespace k = strata::kernels;

static void check(bool ok, const char* name) {
    std::printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) std::exit(1);
}

static void gpu(cudaError_t status) { check(status == cudaSuccess, "GPU operation"); }

int main() {
    constexpr int L = 12, N = 257, HC = 3, T = 30, D = N * HC;
    std::string err;
    check(!k::abliterate_init(0, N, HC, err), "invalid geometry rejected");
    check(k::abliterate_init(L, N, HC, err), "init");
    check(k::abliterate_covers(4) && k::abliterate_covers(8) && !k::abliterate_covers(3), "middle band");
    float* R = nullptr;
    gpu(cudaMalloc(&R, size_t(T) * D * sizeof(float)));
    std::vector<float> input(size_t(T) * D), got(input.size());
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 29) - 14) / 7.0f;
    auto upload = [&] { gpu(cudaMemcpy(R, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice)); };
    auto download = [&] { gpu(cudaMemcpy(got.data(), R, got.size() * sizeof(float), cudaMemcpyDeviceToHost)); };
    auto apply = [&](int layer, int tokens) { k::abliterate_apply(R, layer, tokens, D, nullptr, 0, nullptr, 0, false, nullptr); };

    upload();
    apply(5, T);
    download();
    check(std::memcmp(input.data(), got.data(), got.size() * sizeof(float)) == 0, "disabled is bit-identical");

    float *bo = nullptr, *inj = nullptr;
    gpu(cudaMalloc(&bo, size_t(T) * N * sizeof(float)));
    gpu(cudaMalloc(&inj, size_t(T) * HC * sizeof(float)));
    std::vector<float> block(size_t(T) * N, 0.25f), inject(size_t(T) * HC, 0.0f);
    gpu(cudaMemcpy(bo, block.data(), block.size() * sizeof(float), cudaMemcpyHostToDevice));
    gpu(cudaMemcpy(inj, inject.data(), inject.size() * sizeof(float), cudaMemcpyHostToDevice));
    k::abliterate_apply(R, 5, T, D, bo, N, inj, HC, true, nullptr);
    download();
    bool pending_ok = true;
    for (size_t i = 0; i < input.size(); ++i) pending_ok = pending_ok && got[i] == std::fma(0.25f, 1.0f, input[i]);
    check(pending_ok, "disabled hook materializes the pending GR write exactly once");
    gpu(cudaFree(bo)); gpu(cudaFree(inj));
    upload();

    k::abliterate_command("ABL CAPTURE");
    for (int layer = 4; layer <= 8; ++layer) apply(layer, T);
    download();
    check(std::memcmp(input.data(), got.data(), got.size() * sizeof(float)) == 0, "capture preserves residuals");
    std::vector<float> states;
    check(k::abliterate_read(25, states, err), "capture read");
    bool ring_ok = states.size() == size_t(5 * 25 * N);
    for (int l = 0; l < 5; ++l) for (int t = 0; t < 25; ++t) for (int j = 0; j < N; ++j) {
        float expected = 0;
        for (int c = 0; c < HC; ++c) expected += input[size_t(t + 5) * D + c * N + j] / HC;
        ring_ok = ring_ok && std::abs(states[(size_t(l) * 25 + t) * N + j] - expected) < 1e-6;
    }
    check(ring_ok, "ring wrap and hyper-connection mean");
    for (int l = 4; l <= 8; ++l) apply(l, 1);
    check(k::abliterate_read(1, states, err), "capture after wrap");
    check(std::abs(states[0] - (input[0] + input[N] + input[2 * N]) / HC) < 1e-6, "chronological ring cursor");

    k::abliterate_command("ABL CAPTURE");
    check(!k::abliterate_read(1, states, err), "missing captures rejected");
    for (int layer = 5; layer <= 8; ++layer) {
        std::ostringstream command;
        command << "ABL SET " << layer << " 0.25 -0.5";
        for (int j = 0; j < N; ++j) command << ' ' << (j == 0 ? 1.0f : 0.0f);
        for (int j = 0; j < N; ++j) command << ' ' << (j == 0 ? 0.6f : j == 1 ? 0.8f : 0.0f);
        k::abliterate_command(command.str());
    }
    k::abliterate_command("ABL ON");
    upload();
    apply(5, T);
    download();
    bool projection_ok = true;
    for (int t = 0; t < T; ++t) for (int c = 0; c < HC; ++c) {
        const size_t off = size_t(t) * D + c * N;
        const double x = double(input[off]) - std::max(double(input[off]) - 0.25, 0.0);
        const double excess = std::max(x * 0.6 + double(input[off + 1]) * 0.8 + 0.5, 0.0);
        projection_ok = projection_ok && std::abs(got[off] - (x - 0.6 * excess)) < 2e-6 &&
                                        std::abs(got[off + 1] - (input[off + 1] - 0.8 * excess)) < 2e-6;
        for (int j = 2; j < N; ++j) projection_ok = projection_ok && got[off + j] == input[off + j];
    }
    check(projection_ok, "sequential clipped projections match double reference");
    upload(); apply(4, T); download();
    check(std::memcmp(input.data(), got.data(), got.size() * sizeof(float)) == 0, "gate-only layers are not edited");

    // Mode is read on the device when a graph replays, rather than baked in during capture.
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    gpu(cudaStreamCreate(&stream));
    gpu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    k::abliterate_apply(R, 5, T, D, nullptr, 0, nullptr, 0, false, stream);
    gpu(cudaStreamEndCapture(stream, &graph));
    gpu(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    k::abliterate_command("ABL OFF");
    upload(); gpu(cudaGraphLaunch(exec, stream)); gpu(cudaStreamSynchronize(stream)); download();
    check(std::memcmp(input.data(), got.data(), got.size() * sizeof(float)) == 0, "graph replay reads disabled mode");
    k::abliterate_command("ABL ON");
    gpu(cudaGraphLaunch(exec, stream)); gpu(cudaStreamSynchronize(stream)); download();
    check(std::memcmp(input.data(), got.data(), got.size() * sizeof(float)) != 0, "same graph reads enabled mode");
    gpu(cudaGraphExecDestroy(exec)); gpu(cudaGraphDestroy(graph)); gpu(cudaStreamDestroy(stream));
    gpu(cudaFree(R));
    k::abliterate_free();
    std::puts("All abliteration parity checks passed.");
}
