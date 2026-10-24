// sycl/src/kernels/tq_sycl_parity.cpp - the Gyro trellis experts (TQ2_T 144, TQK6 145, TQK7 146, Hadamard-folded) on the
// SYCL kernels, against real tensors of a Gyro GGUF.
//
//   tq_sycl_parity <pack-dir> <model.gguf> [layer ...]
//   tq_sycl_parity --synthetic [--hadamard] GU/DOWN ...      (random trellis codes and scales, e.g. 145/146 144/144)
//   tq_sycl_parity --bench GU/DOWN [groups [entries-per-group [iterations]]]   (native_expert_grouped on distinct experts in VRAM)
//
// For one expert of each layer (the gate/up/down rows read from the GGUF, the Hadamard signs and block from the pack):
//   (a) float reference: fp16-codebook weights, x and h rotated in double, SwiGLU in double.
//   (b) the CPU engine path (native_quant_act, native_gu_rows, native_quant_h, native_down_rows).
//   (c) the GPU path (hadamard_rows, quantize_q8_1_rows, native_expert_grouped) with 3 entries in one group.
// Checks: (c) against (a) and (b) within the bound every expert type gets (relative L1 < 3e-2); the GPU Hadamard rotation
// against the CPU's; the gate/up rows and the down rows of (c) BIT FOR BIT against a host emulation of the kernels'
// integer and float operations (the int8 codebook, the q8_1 blocks and the h the device produced are read back, the
// lane-strided sums and the xor tree are replayed); the prompt-path dequantizers (iq_dequant_f32 / _f16) against the
// fp16 codebook, bit for bit.  Exit 0 only if all hold.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "ggml-common-tq.h"
#include "../../../src/kernels/cpu/tq_states.hpp"
#include "ggml.h"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define __device__
#include "../../../src/kernels/cuda/tq_lut_i8.cuh"
#undef __device__

namespace cpu = strata::kernels::cpu;
namespace {
constexpr int NT = 3;
constexpr int64_t H = 2560, FF = 640;
constexpr int LN = 8;   // STRATA_EXPERT_LANES

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}
float h2f(uint16_t b) { return cpu::tq_detail::tq_half_to_float(b); }
int kbits(int t) { return t == 144 ? 8 : t == 145 ? 6 : 7; }
size_t bsize(int t) { return 2 + 4 * (size_t) kbits(t); }

// the 32 trellis states of block b (qs after the fp16 scale)
void states(int t, const uint8_t* blk, uint32_t* st) {
    if (t == 144) cpu::tq_detail::tq_states<8>(blk + 2, st);
    else if (t == 145) cpu::tq_detail::tq_states<6>(blk + 2, st);
    else cpu::tq_detail::tq_states<7>(blk + 2, st);
}
// float dequantization of one row of n values with the fp16 codebook
void dq_row(int t, const uint8_t* row, int64_t n, float* y) {
    for (int64_t b = 0; b < n / 128; ++b) {
        const uint8_t* blk = row + b * bsize(t);
        const float d = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
        uint32_t st[32];
        states(t, blk, st);
        for (int s = 0; s < 32; ++s) {
            const uint32_t x = (st[s] & TQ_STATE_MASK) * 0x9e3779b1u;
            const uint32_t pa = x >> 21, pb = (x >> 10) & 2047u;
            y[b * 128 + 4 * s + 0] = d * h2f(tq_lut_f16[2 * pa]);
            y[b * 128 + 4 * s + 1] = d * h2f(tq_lut_f16[2 * pa + 1]);
            y[b * 128 + 4 * s + 2] = d * h2f(tq_lut_f16[2 * pb]);
            y[b * 128 + 4 * s + 3] = d * h2f(tq_lut_f16[2 * pb + 1]);
        }
    }
}
// host replay of the kernels' dot of one row against one q8_1 row (36-byte blocks): lane `sub` of LN accumulates calls
// sub, sub+LN, ... in order, then the xor tree (o = LN/2 .. 1)
float emu_dot(int t, const uint8_t* row, const uint8_t* q8, int64_t n) {
    const int nb = (int) (n / 128);
    float s[LN] = {};
    for (int sub = 0; sub < LN; ++sub)
        for (int k = sub; k < nb * 8; k += LN) {
            const int kbx = k / 8, lane = k % 8;
            const uint8_t* blk = row + (size_t) kbx * bsize(t);
            uint32_t st[32];
            states(t, blk, st);
            int sumi = 0;
            const uint8_t* b8 = q8 + (size_t) (kbx * 4 + lane / 2) * 36;
            for (int i = 0; i < 4; ++i) {
                const uint32_t x = (st[4 * lane + i] & TQ_STATE_MASK) * 0x9e3779b1u;
                const uint16_t p0 = tq_lut_i8[x >> 21], p1 = tq_lut_i8[(x >> 10) & 2047u];
                const int8_t w[4] = {(int8_t) (p0 & 0xff), (int8_t) (p0 >> 8), (int8_t) (p1 & 0xff), (int8_t) (p1 >> 8)};
                const int8_t* a = (const int8_t*) (b8 + 4 + 16 * (lane & 1) + 4 * i);
                for (int j = 0; j < 4; ++j) sumi += (int) w[j] * (int) a[j];
            }
            const uint16_t dsb = (uint16_t) (b8[0] | (b8[1] << 8));
            const float d = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
            volatile float tmp = h2f(dsb) * TQ_LUT_I8_SCALE;
            volatile float v = d * tmp;
            v = v * (float) sumi;
            volatile float acc = s[sub];
            acc = acc + v;
            s[sub] = acc;
        }
    for (int o = LN / 2; o > 0; o >>= 1) {
        float nx[LN];
        for (int l = 0; l < LN; ++l) nx[l] = s[l] + s[l ^ o];
        std::memcpy(s, nx, sizeof s);
    }
    return s[0];
}

template <class T> T* dmalloc(size_t n) { return (T*) sycl::malloc_device((n ? n : 1) * sizeof(T), dpct::get_in_order_queue()); }
template <class T> void to_dev(T* d, const void* h, size_t n) { dpct::get_in_order_queue().memcpy(d, h, n).wait(); }
template <class T> void from_dev(void* h, const T* d, size_t n) { dpct::get_in_order_queue().memcpy(h, d, n).wait(); }

int check_layer(const cpu::NativeFmt& f, const std::vector<uint8_t>& blob, int seed, const std::string& label) {
    int failures = 0;
    sycl::queue* s = &dpct::get_in_order_queue();
    // (a) float reference (rows dequantized with the fp16 codebook)
    std::vector<float> G((size_t) FF * H), U((size_t) FF * H), D((size_t) H * FF);
    for (int64_t r = 0; r < FF; ++r) {
        dq_row(f.gu_type, blob.data() + r * f.gu_row, H, G.data() + r * H);
        dq_row(f.gu_type, blob.data() + f.up_off + r * f.gu_row, H, U.data() + r * H);
    }
    for (int64_t r = 0; r < H; ++r) dq_row(f.d_type, blob.data() + f.down_off + r * f.d_row, FF, D.data() + r * FF);
    std::mt19937 rng(11 + seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NT * H);
    for (auto& v : x) v = nd(rng);
    std::vector<float> xr(x);
    if (f.had_gu) for (int k = 0; k < NT; ++k) cpu::hadamard_rotate(xr.data() + k * H, (int) H, f.had_block, f.gu_signs);
    std::vector<float> ref((size_t) NT * H), got_c((size_t) NT * H), got_g((size_t) NT * H);
    for (int k = 0; k < NT; ++k) {
        std::vector<float> h(FF);
        for (int64_t r = 0; r < FF; ++r) {
            double g = 0, u = 0;
            for (int64_t i = 0; i < H; ++i) { g += (double) G[r * H + i] * xr[k * H + i]; u += (double) U[r * H + i] * xr[k * H + i]; }
            h[r] = (float) ((g / (1.0 + std::exp(-g))) * u);
        }
        if (f.had_d) cpu::hadamard_rotate(h.data(), (int) FF, f.had_block, f.d_signs);
        for (int64_t r = 0; r < H; ++r) {
            double o = 0;
            for (int64_t i = 0; i < FF; ++i) o += (double) D[r * FF + i] * h[i];
            ref[k * H + r] = (float) o;
        }
    }
    // (b) the CPU engine path
    {
        std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(cpu::kNativeActBytes));
        std::vector<std::vector<uint8_t>> hq(NT, std::vector<uint8_t>(cpu::kNativeHBytes));
        std::vector<std::vector<float>> ff(NT, std::vector<float>(FF));
        const void* a[NT];
        float* ffp[NT];
        const void* hp[NT];
        float* op[NT];
        for (int k = 0; k < NT; ++k) {
            cpu::native_quant_act(f, x.data() + k * H, act[k].data());
            a[k] = act[k].data();
            ffp[k] = ff[k].data();
        }
        cpu::native_gu_rows(f, blob.data(), a, NT, ffp, 0, (int) FF);
        for (int k = 0; k < NT; ++k) {
            cpu::native_quant_h(f, ff[k].data(), hq[k].data());
            hp[k] = hq[k].data();
            op[k] = got_c.data() + k * H;
        }
        cpu::native_down_rows(f, blob.data(), hp, NT, op, 0, (int) H);
    }
    // (c) the GPU: one group holding the NT entries
    auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, H, FF);
    if (f.had_d) { L.had_block = f.had_block; L.d_signs = strata::kernels::hadamard_device_signs(f.d_signs, (int) FF); }
    uint8_t* dblob = dmalloc<uint8_t>(blob.size());
    float* dx = dmalloc<float>(x.size());
    uint8_t* dxq = dmalloc<uint8_t>((size_t) NT * H / 32 * 36);
    const size_t scr_bytes = strata::kernels::native_expert_scratch_bytes(NT, FF);
    uint8_t* dscr = dmalloc<uint8_t>(scr_bytes);
    float* dout = dmalloc<float>((size_t) NT * H);
    unsigned long long* dptr = dmalloc<unsigned long long>(1);
    int32_t *dstart = dmalloc<int32_t>(2), *dn = dmalloc<int32_t>(1), *ddst = dmalloc<int32_t>(NT), *dtok = dmalloc<int32_t>(NT);
    to_dev(dblob, blob.data(), blob.size());
    to_dev(dx, x.data(), x.size() * 4);
    const unsigned long long p = (unsigned long long) dblob;
    const int32_t st2[2] = {0, NT}, one = 1, idx[NT] = {0, 1, 2};
    to_dev(dptr, &p, 8);
    to_dev(dstart, st2, 8);
    to_dev(dn, &one, 4);
    to_dev(ddst, idx, NT * 4);
    to_dev(dtok, idx, NT * 4);
    if (f.had_gu) {
        strata::kernels::hadamard_rows(dx, dx, NT, H, f.had_block, strata::kernels::hadamard_device_signs(f.gu_signs, (int) H), s);
        std::vector<float> gx(x.size());
        s->wait();
        from_dev(gx.data(), dx, gx.size() * 4);
        const bool same = std::memcmp(gx.data(), xr.data(), gx.size() * 4) == 0;
        std::printf("          gpu hadamard_rows vs cpu hadamard_rotate: %s (rel %.2e)\n", same ? "bitwise equal" : "differ", rel(gx, xr));
        if (!(rel(gx, xr) < 1e-6)) ++failures;
        std::vector<uint16_t> xh(x.size()), xh_out(x.size());
        for (size_t i = 0; i < x.size(); ++i) xh[i] = ggml_fp32_to_fp16(x[i]);
        uint16_t* dxh = dmalloc<uint16_t>(xh.size());
        to_dev(dxh, xh.data(), xh.size() * 2);
        strata::kernels::hadamard_rows_f16(dxh, NT, H, f.had_block, strata::kernels::hadamard_device_signs(f.gu_signs, (int) H), s);
        s->wait();
        from_dev(xh_out.data(), dxh, xh.size() * 2);
        sycl::free(dxh, *s);
        std::vector<float> xf(x.size());
        for (size_t i = 0; i < xf.size(); ++i) xf[i] = ggml_fp16_to_fp32(xh_out[i]);
        std::printf("          gpu hadamard_rows_f16 vs cpu float rotation: rel %.2e\n", rel(xf, xr));
        if (!(rel(xf, xr) < 2e-3)) ++failures;
    }
    strata::kernels::quantize_q8_1_rows(dx, NT, H, dxq, s);
    strata::kernels::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, 1, NT, dxq, dscr, dout, s);
    s->wait();
    from_dev(got_g.data(), dout, got_g.size() * 4);
    {   // bit-for-bit replay of the lane kernels
        std::vector<uint8_t> hxq((size_t) NT * H / 32 * 36);
        from_dev(hxq.data(), dxq, hxq.size());
        const size_t fb = (size_t) NT * FF * 4, fa = (fb + 255) & ~(size_t) 255;
        std::vector<float> gate((size_t) NT * FF), up((size_t) NT * FF), hh((size_t) NT * FF);
        from_dev(gate.data(), (const float*) dscr, gate.size() * 4);
        from_dev(up.data(), (const float*) (dscr + fa), up.size() * 4);
        from_dev(hh.data(), (const float*) (dscr + 2 * fa), hh.size() * 4);
        std::vector<uint8_t> hhq((size_t) NT * FF / 32 * 36);
        from_dev(hhq.data(), dscr + 3 * fa, hhq.size());
        size_t gu_diff = 0, dn_diff = 0;
        for (int k = 0; k < NT; ++k) {
            for (int64_t r = 0; r < FF; ++r) {
                const float eg = emu_dot(f.gu_type, blob.data() + r * f.gu_row, hxq.data() + (size_t) k * H / 32 * 36, H);
                const float eu = emu_dot(f.gu_type, blob.data() + f.up_off + r * f.gu_row, hxq.data() + (size_t) k * H / 32 * 36, H);
                if (std::memcmp(&eg, &gate[k * FF + r], 4) != 0 && gu_diff < 4)
                    std::printf("          [debug] gate k %d r %lld: device %g host replay %g (d of block 0: %g)\n", k, (long long) r,
                                gate[k * FF + r], eg, h2f((uint16_t) (blob[r * f.gu_row] | (blob[r * f.gu_row + 1] << 8))));
                gu_diff += std::memcmp(&eg, &gate[k * FF + r], 4) != 0;
                gu_diff += std::memcmp(&eu, &up[k * FF + r], 4) != 0;
            }
            for (int64_t r = 0; r < H; ++r) {
                const float ed = emu_dot(f.d_type, blob.data() + f.down_off + r * f.d_row, hhq.data() + (size_t) k * FF / 32 * 36, FF);
                dn_diff += std::memcmp(&ed, &got_g[k * H + r], 4) != 0;
            }
        }
        std::printf("          kernel replay on the host (int8 codebook, q8_1, lane sums): gate/up %zu of %lld, down %zu of %lld values differ in any bit\n",
                    gu_diff, (long long) (2 * NT * FF), dn_diff, (long long) (NT * H));
        if (gu_diff || dn_diff) ++failures;
        (void) hh;
    }
    {   // (d) the GPU dequantizers (prompt path) against the fp16-codebook floats
        float* dq = dmalloc<float>((size_t) FF * H);
        uint16_t* dh = dmalloc<uint16_t>((size_t) FF * H);
        std::vector<float> got_dq((size_t) FF * H);
        std::vector<uint16_t> got_h((size_t) FF * H);
        size_t differ = 0, h_differ = 0;
        const struct { const uint8_t* src; int type; const std::vector<float>* want; } mats[3] = {
            {dblob, f.gu_type, &G}, {dblob + f.up_off, f.gu_type, &U}, {dblob + f.down_off, f.d_type, &D}};
        for (const auto& m : mats) {
            strata::kernels::iq_dequant_f32(m.type, m.src, FF * H, dq, s);
            strata::kernels::iq_dequant_f16(m.type, m.src, FF * H, dh, s);
            s->wait();
            from_dev(got_dq.data(), dq, got_dq.size() * 4);
            from_dev(got_h.data(), dh, got_h.size() * 2);
            for (size_t i = 0; i < got_dq.size(); ++i) {
                differ += std::memcmp(&got_dq[i], &(*m.want)[i], 4) != 0;
                h_differ += got_h[i] != ggml_fp32_to_fp16(got_dq[i]);
            }
        }
        sycl::free(dq, *s); sycl::free(dh, *s);
        std::printf("          GPU dequant vs fp16-codebook floats: %zu of %lld values differ in any bit; fp16 of it: %zu differ\n",
                    differ, (long long) (3 * FF * H), h_differ);
        if (differ || h_differ) ++failures;
    }
    sycl::free(dblob, *s); sycl::free(dx, *s); sycl::free(dxq, *s); sycl::free(dscr, *s); sycl::free(dout, *s);
    sycl::free(dptr, *s); sycl::free(dstart, *s); sycl::free(dn, *s); sycl::free(ddst, *s); sycl::free(dtok, *s);
    const double ec = rel(got_c, ref), eg = rel(got_g, ref), ecg = rel(got_c, got_g);
    const bool ok = ec < 3e-2 && eg < 3e-2 && std::isfinite(ec) && std::isfinite(eg);
    std::printf("%-9s types %d/%d blob %8zu  cpu rel %.2e  gpu rel %.2e  cpu-gpu %.2e  %s\n", label.c_str(), f.gu_type, f.d_type,
                f.bytes, ec, eg, ecg, ok ? "ok" : "FAIL");
    return failures + (ok ? 0 : 1);
}
}  // namespace

// random trellis bytes (every bit pattern is a valid trellis), scales in a sane fp16 range
std::vector<uint8_t> synthetic_blob(const cpu::NativeFmt& f, int seed) {
    std::mt19937 rng(1234 + seed);
    std::vector<uint8_t> blob(f.bytes);
    for (auto& b : blob) b = (uint8_t) rng();
    auto fix = [&](uint8_t* base, int type, size_t row_bytes, int64_t rows) {
        for (int64_t r = 0; r < rows; ++r)
            for (size_t b = 0; b < row_bytes / bsize(type); ++b) {
                uint8_t* blk = base + r * row_bytes + b * bsize(type);
                const float d = 0.002f + 0.03f * ((rng() & 0xffff) / 65535.0f);
                const uint16_t h = ggml_fp32_to_fp16(d);
                blk[0] = (uint8_t) (h & 0xff);
                blk[1] = (uint8_t) (h >> 8);
            }
    };
    if (f.gu_type >= 144 && f.d_type >= 144) {   // (the bench also times other formats on random bytes)
        fix(blob.data(), f.gu_type, f.gu_row, FF);
        fix(blob.data() + f.up_off, f.gu_type, f.gu_row, FF);
        fix(blob.data() + f.down_off, f.d_type, f.d_row, H);
    }
    return blob;
}
int type_of(const std::string& s) { return std::atoi(s.c_str()); }

// the grouped expert kernels on `groups` distinct experts held in VRAM (more bytes than the card's L2), `epg` entries each:
// microseconds per call and per expert, and the weight bytes read per second
int bench(int gu, int dn, int groups, int epg, int iters, bool had, int phase) {
    strata::kernels::native_expert_set_mode(7, phase);   // 0: all, 1: gate/up + SwiGLU + quantize, 2: down only
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu, dn, H, FF, f, err) || !strata::kernels::native_expert_supported(gu, dn, H, FF)) {
        std::printf("%d/%d: not a pair the GPU expert kernels take\n", gu, dn);
        return 1;
    }
    static std::vector<float> sx((size_t) H), sh((size_t) FF);
    std::mt19937 srng(5);
    for (auto& v : sx) v = (srng() & 1) ? 1.f : -1.f;
    for (auto& v : sh) v = (srng() & 1) ? 1.f : -1.f;
    if (had) { f.had_block = 128; f.had_gu = f.had_d = true; f.gu_signs = sx.data(); f.d_signs = sh.data(); }
    sycl::queue* s = &dpct::get_in_order_queue();
    const int ent = groups * epg, ntok = 4;
    uint8_t* dblob = dmalloc<uint8_t>((size_t) groups * f.bytes);
    for (int g = 0; g < groups; ++g) {
        const std::vector<uint8_t> b = synthetic_blob(f, g);
        to_dev(dblob + (size_t) g * f.bytes, b.data(), b.size());
    }
    std::vector<float> x((size_t) ntok * H);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    float* dx = dmalloc<float>(x.size());
    to_dev(dx, x.data(), x.size() * 4);
    uint8_t* dxq = dmalloc<uint8_t>((size_t) ntok * H / 32 * 36);
    uint8_t* dscr = dmalloc<uint8_t>(strata::kernels::native_expert_scratch_bytes(ent, FF));
    float* dout = dmalloc<float>((size_t) ent * H);
    std::vector<unsigned long long> ptr(groups);
    std::vector<int32_t> st(groups + 1), dst(ent), tok(ent);
    for (int g = 0; g < groups; ++g) { ptr[g] = (unsigned long long) (dblob + (size_t) g * f.bytes); st[g] = g * epg; }
    st[groups] = ent;
    for (int e = 0; e < ent; ++e) { dst[e] = e; tok[e] = e % ntok; }
    unsigned long long* dptr = dmalloc<unsigned long long>(groups);
    int32_t *dstart = dmalloc<int32_t>(groups + 1), *dn_ = dmalloc<int32_t>(1), *ddst = dmalloc<int32_t>(ent), *dtok = dmalloc<int32_t>(ent);
    to_dev(dptr, ptr.data(), groups * 8);
    to_dev(dstart, st.data(), (groups + 1) * 4);
    to_dev(dn_, &groups, 4);
    to_dev(ddst, dst.data(), ent * 4);
    to_dev(dtok, tok.data(), ent * 4);
    auto L = strata::kernels::native_expert_layout(gu, dn, H, FF);
    if (had) { L.had_block = f.had_block; L.d_signs = strata::kernels::hadamard_device_signs(f.d_signs, (int) FF); }
    strata::kernels::quantize_q8_1_rows(dx, ntok, H, dxq, s);
    for (int i = 0; i < 3; ++i) strata::kernels::native_expert_grouped(L, dptr, dstart, dn_, ddst, dtok, groups, ent, dxq, dscr, dout, s);
    s->wait();
    std::vector<double> us;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        strata::kernels::native_expert_grouped(L, dptr, dstart, dn_, ddst, dtok, groups, ent, dxq, dscr, dout, s);
        s->wait();
        us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(us.begin(), us.end());
    const double med = us[us.size() / 2];
    std::printf("bench %d/%d%s: %d groups x %d entries, %zu B per expert: median %.0f us per call = %.1f us per expert, %.1f GB/s of weights (min %.0f us)\n",
                gu, dn, had ? " hadamard" : "", groups, epg, f.bytes, med, med / groups, (double) groups * f.bytes / (med * 1e-6) / 1e9, us.front());
    sycl::free(dblob, *s); sycl::free(dx, *s); sycl::free(dxq, *s); sycl::free(dscr, *s); sycl::free(dout, *s);
    sycl::free(dptr, *s); sycl::free(dstart, *s); sycl::free(dn_, *s); sycl::free(ddst, *s); sycl::free(dtok, *s);
    return 0;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc >= 3 && std::string(argv[1]) == "--bench") {
        const std::string arg = argv[2];
        const size_t slash = arg.find('/');
        const bool had = std::getenv("TQ_BENCH_HADAMARD") != nullptr;
        return bench(type_of(arg.substr(0, slash)), type_of(arg.substr(slash + 1)), argc > 3 ? std::atoi(argv[3]) : 64,
                     argc > 4 ? std::atoi(argv[4]) : 1, argc > 5 ? std::atoi(argv[5]) : 50, had, argc > 6 ? std::atoi(argv[6]) : 0);
    }
    if (argc >= 2 && std::string(argv[1]) == "--synthetic") {
        int failures = 0, i = 2;
        const bool had = argc > 2 && std::string(argv[2]) == "--hadamard";
        if (had) ++i;
        static std::vector<float> sx((size_t) H), sh((size_t) FF);
        std::mt19937 srng(5);
        for (auto& v : sx) v = (srng() & 1) ? 1.f : -1.f;
        for (auto& v : sh) v = (srng() & 1) ? 1.f : -1.f;
        for (; i < argc; ++i) {
            const std::string arg = argv[i];
            const size_t slash = arg.find('/');
            const int gu = type_of(arg.substr(0, slash)), dn = type_of(arg.substr(slash + 1));
            cpu::NativeFmt f;
            std::string err;
            if (slash == std::string::npos || !cpu::native_fmt(gu, dn, H, FF, f, err) ||
                !strata::kernels::native_expert_supported(gu, dn, H, FF)) {
                std::printf("%s: %s\n", arg.c_str(), err.empty() ? "not a pair the GPU expert kernels take" : err.c_str());
                ++failures;
                continue;
            }
            if (had) {
                f.had_block = 128;
                f.had_gu = f.had_d = true;
                f.gu_signs = sx.data();
                f.d_signs = sh.data();
            }
            failures += check_layer(f, synthetic_blob(f, i), i, had ? "hadamard" : "synthetic");
        }
        std::printf("tq_sycl_parity: %d failures\n", failures);
        return failures ? 1 : 0;
    }
    if (argc < 3) { std::fprintf(stderr, "usage: tq_sycl_parity <pack-dir> <model.gguf> [layer ...]\n"); return 2; }
    std::string err;
    if (!cpu::expert_layout_load(argv[1], 48, 512, err)) { std::printf("expert_layout_load: %s\n", err.c_str()); return 1; }
    const auto& lay = cpu::expert_layout();
    const strata::GgufModel model(strata::gguf_split_paths(argv[2]));
    std::vector<int> layers;
    for (int i = 3; i < argc; ++i) layers.push_back(std::atoi(argv[i]));
    if (layers.empty()) layers = {0, 1, 2, 3, 20, 47};
    int failures = 0;
    for (int l : layers) {
        const strata::TensorInfo* t[3] = {};
        const uint8_t* data[3] = {};
        const char* roles[3] = {"gate", "up", "down"};
        for (int r = 0; r < 3; ++r) {
            size_t at = 0;
            t[r] = model.find("blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight", &at);
            if (t[r]) data[r] = model.shard(at).tensor_data(*t[r]);
        }
        if (!t[0] || !t[1] || !t[2]) { std::printf("layer %d: no expert tensors\n", l); ++failures; continue; }
        cpu::NativeFmt f = lay.fmt[(size_t) l];
        for (int E : {7, 100}) {
            std::vector<uint8_t> blob(f.bytes);
            std::memcpy(blob.data(), data[0] + (size_t) E * f.up_off, f.up_off);
            std::memcpy(blob.data() + f.up_off, data[1] + (size_t) E * f.up_off, f.up_off);
            std::memcpy(blob.data() + f.down_off, data[2] + (size_t) E * (f.bytes - f.down_off), f.bytes - f.down_off);
            failures += check_layer(f, blob, l * 1000 + E, "layer " + std::to_string(l) + " e" + std::to_string(E));
        }
    }
    std::printf("tq_sycl_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
