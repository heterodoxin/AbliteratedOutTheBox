// 151-b70: fused int8 DPAS expert products (ESIMD, Battlemage) - see include/strata/prefill/esimd_moe.hpp and
// docs/INTEL_INT8_EXPERTS.md (the behaviour note).
//
//  * Tile.  One hardware thread owns 16 output rows (a DPAS N tile; for gate/up the 16 virtual rows alternate gate/up) and up to
//    MT x 8 activation rows (MT = 2, 4 or 8, picked per group from the experts' mean row count).  It walks K in steps of 32:
//    decode the 16 weight rows' 32 values into the B operand (registers only), then for each of its row tiles one s8 DPAS 8x16x32
//    into a zeroed int32 tile, converted to float and scaled by (weight scale of the row and 32-block) x (activation scale of the
//    row and 32-block) into a float accumulator.  There is no shared memory and no barrier.
//  * Decode once per tile: the B operand is decoded once per thread and applied to all its MT row tiles.  An expert with more rows
//    than 8 x MT is split over row-block threads, each decoding again.  A thread's accumulators are MT tiles of 512 bytes in an
//    8 KB register file (256 GRF of 32 bytes): MT 12, or two output tiles per thread, spilled (3-15 KB) and ran 2-5x slower.
//  * Scales.  Activations: q8_1 as llama.cpp (d = amax / 127, q = round half away from zero), one float per row and 32 of K.
//    Weights: IQ3_S d (1 + 2 s), IQ3_XXS d (2 s + 1) / 4, IQ2_S d (2 s + 1) / 8 per 16 (two DPAS per 32 with the other half
//    zeroed), IQ4_NL d per 32, Q2_0 d per 64; the int8 values are the codebook entries with their signs, never rounded.  The int32
//    dot product of a 32-block is exact and exactly convertible to float.  The float accumulation order is fixed (K ascending).
//  * Edge cases.  Rows are padded to tiles of 8 with zeros (stores are masked to the real rows); a group of up to 16 experts per
//    launch; K a multiple of 256 (IQ3_S / IQ3_XXS / IQ2_S) / 64 (Q2_0) / 32 (IQ4_NL); output rows a multiple of 16.  Blob reads
//    never go past a matrix end.
//  * Fallback.  available() is false without the s8 8x16x32 DPAS (a non-Battlemage device) and the caller keeps the dequantize +
//    oneMKL path; gate_up()/down() return false for a type / shape they do not handle.
//  * ESIMD on this compiler (oneAPI 2026.1, bmg-g31): a dynamic index into the kernel-argument arrays, and `/` or `%` by a run-time
//    value, gave wrong values / garbage pointers (the first weight read faulted the GPU); the kernels use constant-index reads
//    (pick_expert) and 2-D grids.  gather<T, N, VS> with VS > 1 is not available on Battlemage.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <string>
#include <vector>
#include <unordered_map>
#include "strata/sycl_queue.hpp"
#include "strata/prefill/esimd_moe.hpp"

#ifdef STRATA_I8_NO_ESIMD
// an AOT build for a device without the s8 8x16x32 DPAS (Alchemist, Xe-HPC): no ESIMD kernels, available() is false
namespace strata::prefill::i8 {
void make_layout(Layout& lo, const int32_t* cnt, int n) {
    lo.n = n;
    int r = 0, t = 0;
    for (int i = 0; i < n; ++i) { lo.cnt[i] = cnt[i]; lo.row0[i] = r; lo.tile0[i] = t; r += cnt[i]; t += (cnt[i] + 7) / 8; }
    lo.row0[n] = r;
    lo.tile0[n] = t;
}
bool available(void*) { return false; }
bool battlemage(void*) { return false; }
bool any_battlemage() { return false; }
size_t xq_bytes(const Layout& lo, int K) { return (size_t) lo.tile0[lo.n] * 8 * K; }
size_t dx_bytes(const Layout& lo, int K) { return (size_t) lo.tile0[lo.n] * (K / 32) * 8 * sizeof(float); }
bool workspace(const Layout&, int, int, Workspace&, void*) { return false; }
void quantize_rows(const uint16_t*, const Layout&, int, int8_t*, float*, void*) {}
bool gate_up(int, const uint8_t* const*, size_t, int, const Layout&, int, const int8_t*, const float*, float*, void*) { return false; }
void swiglu_quantize(const float*, const Layout&, int, int8_t*, float*, void*) {}
bool down(int, const uint8_t* const*, size_t, int, const Layout&, int, const int8_t*, const float*, float*, void*) { return false; }
}  // namespace strata::prefill::i8
#else
namespace strata::prefill::i8 {
namespace {
#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };
#include "iq_tables.inc"
#undef GGML_TABLE_BEGIN
#undef GGML_TABLE_END

using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using half = sycl::half;

// ggml types handled
constexpr int T_IQ3XXS = 18, T_IQ4NL = 20, T_IQ3S = 21, T_IQ2S = 22, T_Q2_0 = 42;
template <int F> constexpr int blk_bytes() { return F == T_IQ3S ? 110 : F == T_IQ3XXS ? 98 : F == T_IQ2S ? 82 : F == T_IQ4NL ? 18 : 18; }
constexpr int block_bytes_of(int f) { return f == T_IQ3S ? 110 : f == T_IQ3XXS ? 98 : f == T_IQ2S ? 82 : 18; }
constexpr int block_vals_of(int f) { return (f == T_IQ3S || f == T_IQ3XXS || f == T_IQ2S) ? 256 : f == T_Q2_0 ? 64 : 32; }

// One device table buffer (uint32 units): the codebooks the decoders gather from.
constexpr int O_IQ3S = 0;        // iq3s_grid[512]
constexpr int O_IQ3XXS = 512;    // iq3xxs_grid[256]
constexpr int O_IQ2S = 768;      // iq2s_grid[1024] as 2048 dwords (low dword of an entry first)
constexpr int O_KSIGNS = 2816;   // ksigns_iq2xs[128], one byte in the low byte of each dword
constexpr int O_IQ4 = 2944;      // [256]: byte b -> kvalues_iq4nl[b & 15] in byte 0, kvalues_iq4nl[b >> 4] in byte 1
constexpr int O_Q2 = 3200;       // [256]: byte b -> its four 2-bit codes minus 1, as four int8 (Q2_0 values)
constexpr int N_TABLE = 3456;

uint32_t* tables(sycl::queue& q) {
    static std::mutex mu;
    static std::unordered_map<sycl::device, uint32_t*> per_dev;   // a layer split has one prompt path per card
    std::lock_guard<std::mutex> lk(mu);
    uint32_t*& t = per_dev[q.get_device()];
    if (!t) {
        std::vector<uint32_t> h(N_TABLE, 0);
        for (int i = 0; i < 512; ++i) h[O_IQ3S + i] = iq3s_grid[i];
        for (int i = 0; i < 256; ++i) h[O_IQ3XXS + i] = iq3xxs_grid[i];
        for (int i = 0; i < 1024; ++i) { h[O_IQ2S + 2 * i] = (uint32_t) iq2s_grid[i]; h[O_IQ2S + 2 * i + 1] = (uint32_t) (iq2s_grid[i] >> 32); }
        for (int i = 0; i < 128; ++i) h[O_KSIGNS + i] = ksigns_iq2xs[i];
        for (int b = 0; b < 256; ++b) {
            h[O_IQ4 + b] = (uint32_t) (uint8_t) kvalues_iq4nl[b & 15] | ((uint32_t) (uint8_t) kvalues_iq4nl[b >> 4] << 8);
            uint32_t q2 = 0;
            for (int k = 0; k < 4; ++k) q2 |= (uint32_t) (uint8_t) (int8_t) (((b >> (2 * k)) & 3) - 1) << (8 * k);
            h[O_Q2 + b] = q2;
        }
        uint32_t* d = sycl::malloc_device<uint32_t>(N_TABLE, q);
        q.memcpy(d, h.data(), N_TABLE * 4);
        q.wait();
        t = d;
    }
    return t;
}

// unaligned 32-bit read at a 2-byte aligned byte offset (per lane) through two dword gathers
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> ld32(const uint8_t* base, simd<uint32_t, 16> off) {
    simd<uint32_t, 16> a = off & ~3u;
    simd<uint32_t, 16> lo = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(base), a);
    simd<uint32_t, 16> hi = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(base), a + 4u);
    simd<uint32_t, 16> sh = (off & 2u) << 3;   // 0 or 16
    return (lo >> sh) | ((hi << 16) << (16u - sh));
}
// the same with the second dword clamped to byte offset `lim` (the last dword of the matrix): the down matrix ends the expert blob, and a block
// whose last field is not dword aligned would otherwise read up to 3 bytes past it
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> ld32c(const uint8_t* base, simd<uint32_t, 16> off, uint32_t lim) {
    simd<uint32_t, 16> a = off & ~3u;
    simd<uint32_t, 16> a2 = a + 4u;
    a2.merge(simd<uint32_t, 16>(lim), a2 > lim);
    simd<uint32_t, 16> lo = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(base), a);
    simd<uint32_t, 16> hi = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(base), a2);
    simd<uint32_t, 16> sh = (off & 2u) << 3;
    return (lo >> sh) | ((hi << 16) << (16u - sh));
}
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> ld16(const uint8_t* base, simd<uint32_t, 16> off) {
    simd<uint32_t, 16> a = off & ~3u;
    simd<uint32_t, 16> lo = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(base), a);
    return (lo >> ((off & 2u) << 3)) & 0xffffu;
}
SYCL_ESIMD_FUNCTION inline simd<float, 16> half_of(simd<uint32_t, 16> h16) {
    simd<uint16_t, 16> lo = h16.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    simd<half, 16> hh = lo.template bit_cast_view<half>();
    simd<float, 16> f = hh;
    return f;
}
SYCL_ESIMD_FUNCTION inline simd<int, 128> dpas_s8(simd<uint32_t, 128> b, simd<int, 64> a) {
    simd<int, 128> bi = b.template bit_cast_view<int>();
    return xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), bi, a);
}
// sign nibble -> four negation bytes: (g ^ mask) + m negates the bytes whose sign bit is set (g is never 0xFF per byte)
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> apply_signs(simd<uint32_t, 16> g, simd<uint32_t, 16> nib) {
    simd<uint32_t, 16> m = ((nib * 0x00204081u) & 0x01010101u);
    simd<uint32_t, 16> mask = m * 255u;
    return (g ^ mask) + m;
}

struct GArgs {
    const uint8_t* blob[kMaxGroup];
    uint32_t sec_off;                 // gate/up: byte offset of the up rows; down: byte offset of the down rows
    int n;
    int row0[kMaxGroup + 1];
    int tile0[kMaxGroup + 1];
    int rb0[kMaxGroup + 1];           // first row block of expert i
    const int8_t* Xq;
    const float* dx;
    float* Y;
    const uint32_t* tab;
    int ntiles;                       // output tiles of 16
    int nkb;                          // K / 32
    int rowb;                         // bytes per weight row
    int ldy;
    uint32_t lim;                     // byte offset of the last dword of the matrix the clamped reads (ld32c) stay inside
    int dbg;                          // bring-up switches (STRATA_I8_DBG): 1 skip the decode, 2 skip the activation loads, 4 skip the stores, 8 skip the K loop
};

// The expert a work item belongs to, with constant-index reads of the kernel argument arrays.
struct Pick { int e, row0, cnt, tile0, rb0; const uint8_t* blob; };
SYCL_ESIMD_FUNCTION inline Pick pick_expert(const GArgs& a, int rbg) {
    Pick p{0, a.row0[0], a.row0[1] - a.row0[0], a.tile0[0], a.rb0[0], a.blob[0]};
#pragma unroll
    for (int i = 1; i < kMaxGroup; ++i)
        if (i < a.n && rbg >= a.rb0[i]) p = Pick{i, a.row0[i], a.row0[i + 1] - a.row0[i], a.tile0[i], a.rb0[i], a.blob[i]};
    return p;
}

int fill_args_mt(GArgs& a, const Layout& lo, int MT) {
    int rb = 0;
    for (int i = 0; i < lo.n; ++i) {
        a.row0[i] = lo.row0[i];
        a.tile0[i] = lo.tile0[i];
        a.rb0[i] = rb;
        const int ntl = (lo.cnt[i] + 7) / 8;
        rb += (ntl + MT - 1) / MT;
    }
    a.row0[lo.n] = lo.row0[lo.n];
    a.tile0[lo.n] = lo.tile0[lo.n];
    a.rb0[lo.n] = rb;
    a.n = lo.n;
    return rb;
}

template <int F, int MT, int G> struct TagGu;
template <int F, int MT, int G> struct TagDn;

// ------------------------------------------------------------------------------------------------ gate/up
template <int F, int MT, int G>
void launch_gu(sycl::queue& q, const GArgs a, int nrowblocks) {
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<G>};
    constexpr int NS = (F == T_IQ2S) ? 2 : 1;   // IQ2_S: two scales per 32 values -> two DPAS with the other half zeroed
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagGu<F, MT, G>>(sycl::range<2>((size_t) nrowblocks, (size_t) a.ntiles), props, [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int rbg = (int) id[0], nt = (int) id[1];
            const Pick pk = pick_expert(a, rbg);
            const int cnt = pk.cnt;
            const int ntl = (cnt + 7) >> 3;
            const int tl0 = (rbg - pk.rb0) * MT;
            const int nvalid = (ntl - tl0) < MT ? (ntl - tl0) : MT;
            const int tbase = pk.tile0 + tl0;
            const uint8_t* We = pk.blob;
            const uint32_t* T = a.tab;
            simd<uint32_t, 16> lane(0, 1);
            simd<uint32_t, 16> ro = (lane & 1u) * a.sec_off + ((lane >> 1) + (uint32_t) (8 * nt)) * (uint32_t) a.rowb;
            simd<float, 128 * MT> acc = 0.f;
            const int* Xi = reinterpret_cast<const int*>(a.Xq);
            for (int sb = 0; sb < ((a.dbg & 8) ? 0 : (a.nkb >> 3)); ++sb) {
                const uint32_t P = (uint32_t) sb * blk_bytes<F>();
                const simd<float, 16> d = half_of(ld16(We, ro + P));
                simd<uint32_t, 16> w0 = 0, w1 = 0, w2 = 0, w3 = 0;   // sb-level fields
                if constexpr (F == T_IQ3S) {
                    w0 = ld32(We, ro + (P + 106));   // scales
                    w1 = ld32(We, ro + (P + 66));    // qh 0..3
                    w2 = ld32(We, ro + (P + 70));    // qh 4..7
                } else if constexpr (F == T_IQ2S) {
                    w0 = ld32(We, ro + (P + 74));    // scales 0..3
                    w1 = ld32(We, ro + (P + 78));    // scales 4..7
                    w2 = ld32(We, ro + (P + 66));    // qh 0..3
                    w3 = ld32(We, ro + (P + 70));    // qh 4..7
                }
#pragma unroll 1
                for (int ib = 0; ib < 8; ++ib) {
                    const int kb = sb * 8 + ib;
                    simd<uint32_t, 128> fr0 = 0x01010101u, fr1 = 0u;
                    simd<float, 16> sw0 = 1.f, sw1 = 1.f;
                    if (!(a.dbg & 1)) {
                        if constexpr (F == T_IQ3S) {
                            simd<uint32_t, 16> q0 = ld32(We, ro + (P + 2 + 8 * ib));
                            simd<uint32_t, 16> q1 = ld32(We, ro + (P + 6 + 8 * ib));
                            simd<uint32_t, 16> sg = ld32(We, ro + (P + 74 + 4 * ib));
                            simd<uint32_t, 16> qhv = (((ib < 4) ? w1 : w2) >> (8 * (ib & 3))) & 0xffu;
                            simd<uint32_t, 16> scn = ((w0 >> (8 * (ib >> 1))) >> (4 * (ib & 1))) & 0xfu;
                            sw0 = d * simd<float, 16>(simd<int, 16>(scn * 2u + 1u));
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                simd<uint32_t, 16> qsw = r < 4 ? q0 : q1;
                                simd<uint32_t, 16> idx = ((qsw >> (8 * (r & 3))) & 0xffu) | (((qhv >> r) & 1u) << 8);
                                simd<uint32_t, 16> g = gather<uint32_t, 16>(T + O_IQ3S, idx << 2);
                                simd<uint32_t, 16> nib = ((sg >> (8 * (r >> 1))) >> (4 * (r & 1))) & 0xfu;
                                fr0.template select<16, 1>(16 * r) = apply_signs(g, nib);
                            }
                        } else if constexpr (F == T_IQ3XXS) {
                            simd<uint32_t, 16> q0 = ld32(We, ro + (P + 2 + 8 * ib));
                            simd<uint32_t, 16> q1 = ld32(We, ro + (P + 6 + 8 * ib));
                            simd<uint32_t, 16> aux = ld32(We, ro + (P + 66 + 4 * ib));
                            sw0 = d * simd<float, 16>(simd<int, 16>((aux >> 28) * 2u + 1u)) * 0.25f;
#pragma unroll
                            for (int il = 0; il < 4; ++il) {
                                simd<uint32_t, 16> sbyte = gather<uint32_t, 16>(T + O_KSIGNS, ((aux >> (7 * il)) & 127u) << 2);
#pragma unroll
                                for (int hh = 0; hh < 2; ++hh) {
                                    const int r = 2 * il + hh;
                                    simd<uint32_t, 16> qsw = r < 4 ? q0 : q1;
                                    simd<uint32_t, 16> idx = (qsw >> (8 * (r & 3))) & 0xffu;
                                    simd<uint32_t, 16> g = gather<uint32_t, 16>(T + O_IQ3XXS, idx << 2);
                                    fr0.template select<16, 1>(16 * r) = apply_signs(g, (sbyte >> (4 * hh)) & 0xfu);
                                }
                            }
                        } else if constexpr (F == T_IQ2S) {
                            simd<uint32_t, 16> idw = ld32(We, ro + (P + 2 + 4 * ib));
                            simd<uint32_t, 16> sgw = ld32(We, ro + (P + 34 + 4 * ib));
                            simd<uint32_t, 16> qhb = (((ib < 4) ? w2 : w3) >> (8 * (ib & 3))) & 0xffu;
                            simd<uint32_t, 16> scb = (((ib < 4) ? w0 : w1) >> (8 * (ib & 3))) & 0xffu;
                            sw0 = d * simd<float, 16>(simd<int, 16>((scb & 0xfu) * 2u + 1u)) * 0.125f;
                            sw1 = d * simd<float, 16>(simd<int, 16>((scb >> 4) * 2u + 1u)) * 0.125f;
#pragma unroll
                            for (int il = 0; il < 4; ++il) {
                                simd<uint32_t, 16> idx = ((idw >> (8 * il)) & 0xffu) | ((qhb << (8 - 2 * il)) & 0x300u);
                                simd<uint32_t, 16> gl = gather<uint32_t, 16>(T + O_IQ2S, (idx << 3));
                                simd<uint32_t, 16> gh = gather<uint32_t, 16>(T + O_IQ2S, (idx << 3) + 4u);
                                simd<uint32_t, 16> sbyte = (sgw >> (8 * il)) & 0xffu;
                                simd<uint32_t, 16> vl = apply_signs(gl, sbyte & 0xfu);
                                simd<uint32_t, 16> vh = apply_signs(gh, sbyte >> 4);
                                if (il < 2) {
                                    fr0.template select<16, 1>(16 * (2 * il)) = vl;
                                    fr0.template select<16, 1>(16 * (2 * il + 1)) = vh;
                                } else {
                                    fr1.template select<16, 1>(16 * (2 * il)) = vl;
                                    fr1.template select<16, 1>(16 * (2 * il + 1)) = vh;
                                }
                            }
                            // fr0 holds values 0..15 (dwords 0..3, the rest zero), fr1 values 16..31 (dwords 4..7)
                            fr0.template select<64, 1>(64) = 0u;
                            fr1.template select<64, 1>(0) = 0u;
                        }
                    }
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        if (i < nvalid) {
                            simd<int, 64> av = 0x01010101;
                            simd<float, 8> dxv = 1.f;
                            if (!(a.dbg & 2)) {
                                av = block_load<int, 64>(Xi + ((size_t) (tbase + i) * a.nkb + kb) * 64);
                                dxv = block_load<float, 8>(a.dx + ((size_t) (tbase + i) * a.nkb + kb) * 8);
                            }
                            simd<float, 128> tf = dpas_s8(fr0, av);
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const float dxr = dxv[r];
                                acc.template select<16, 1>(128 * i + 16 * r) += tf.template select<16, 1>(16 * r) * (sw0 * dxr);
                            }
                            if constexpr (NS == 2) {
                                simd<float, 128> tg = dpas_s8(fr1, av);
#pragma unroll
                                for (int r = 0; r < 8; ++r) {
                                    const float dxr = dxv[r];
                                    acc.template select<16, 1>(128 * i + 16 * r) += tg.template select<16, 1>(16 * r) * (sw1 * dxr);
                                }
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i < nvalid) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int rl = (tl0 + i) * 8 + r;
                        if (rl < cnt && !(a.dbg & 4))
                            block_store<float, 16>(a.Y + ((size_t) (pk.row0 + rl)) * a.ldy + 16 * nt, acc.template select<16, 1>(128 * i + 16 * r));
                    }
                }
            }
        });
    });
}

// ------------------------------------------------------------------------------------------------ down
template <int F, int MT, int G>
void launch_dn(sycl::queue& q, const GArgs a, int nrowblocks) {
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<G>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagDn<F, MT, G>>(sycl::range<2>((size_t) nrowblocks, (size_t) a.ntiles), props, [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int rbg = (int) id[0], nt = (int) id[1];
            const Pick pk = pick_expert(a, rbg);
            const int cnt = pk.cnt;
            const int ntl = (cnt + 7) >> 3;
            const int tl0 = (rbg - pk.rb0) * MT;
            const int nvalid = (ntl - tl0) < MT ? (ntl - tl0) : MT;
            const int tbase = pk.tile0 + tl0;
            const uint8_t* We = pk.blob;
            const uint32_t* T = a.tab;
            simd<uint32_t, 16> lane(0, 1);
            simd<uint32_t, 16> ro = (lane + (uint32_t) (16 * nt)) * (uint32_t) a.rowb + a.sec_off;
            simd<float, 128 * MT> acc = 0.f;
            const int* Xi = reinterpret_cast<const int*>(a.Xq);
            for (int kb = 0; kb < ((a.dbg & 8) ? 0 : a.nkb); ++kb) {
                simd<float, 16> sw0 = 1.f;
                simd<uint32_t, 128> fr0 = 0x01010101u;
                if (!(a.dbg & 1)) {
                    if constexpr (F == T_IQ4NL) {
                        const uint32_t P = (uint32_t) kb * 18;
                        sw0 = half_of(ld16(We, ro + P));
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            simd<uint32_t, 16> qd = ld32c(We, ro + (P + 2 + 4 * j), a.lim);
                            // four byte-pair lookups (lo value in byte 0, hi value in byte 1), repacked into the two dwords
                            simd<uint32_t, 16> t0 = gather<uint32_t, 16>(T + O_IQ4, (qd & 0xffu) << 2);
                            simd<uint32_t, 16> t1 = gather<uint32_t, 16>(T + O_IQ4, ((qd >> 8) & 0xffu) << 2);
                            simd<uint32_t, 16> t2 = gather<uint32_t, 16>(T + O_IQ4, ((qd >> 16) & 0xffu) << 2);
                            simd<uint32_t, 16> t3 = gather<uint32_t, 16>(T + O_IQ4, (qd >> 24) << 2);
                            fr0.template select<16, 1>(16 * j) = (t0 & 0xffu) | ((t1 & 0xffu) << 8) | ((t2 & 0xffu) << 16) | (t3 << 24);
                            fr0.template select<16, 1>(16 * (j + 4)) =
                                ((t0 >> 8) & 0xffu) | (t1 & 0xff00u) | ((t2 & 0xff00u) << 8) | ((t3 & 0xff00u) << 16);
                        }
                    } else {   // T_Q2_0: 64 values per block (2 + 16 bytes), two 32-blocks per block, one table lookup per byte
                        const uint32_t P = (uint32_t) (kb >> 1) * 18;
                        const int hb = kb & 1;
                        sw0 = half_of(ld16(We, ro + P));
                        simd<uint32_t, 16> q0 = ld32c(We, ro + (P + 2 + 8 * hb), a.lim);
                        simd<uint32_t, 16> q1 = ld32c(We, ro + (P + 6 + 8 * hb), a.lim);
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            simd<uint32_t, 16> qb = ((r < 4 ? q0 : q1) >> (8 * (r & 3))) & 0xffu;
                            fr0.template select<16, 1>(16 * r) = gather<uint32_t, 16>(T + O_Q2, qb << 2);
                        }
                    }
                }
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    if (i < nvalid) {
                        simd<int, 64> av = 0x01010101;
                        simd<float, 8> dxv = 1.f;
                        if (!(a.dbg & 2)) {
                            av = block_load<int, 64>(Xi + ((size_t) (tbase + i) * a.nkb + kb) * 64);
                            dxv = block_load<float, 8>(a.dx + ((size_t) (tbase + i) * a.nkb + kb) * 8);
                        }
                        simd<float, 128> tf = dpas_s8(fr0, av);
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const float dxr = dxv[r];
                            acc.template select<16, 1>(128 * i + 16 * r) += tf.template select<16, 1>(16 * r) * (sw0 * dxr);
                        }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i < nvalid) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int rl = (tl0 + i) * 8 + r;
                        if (rl < cnt && !(a.dbg & 4))
                            block_store<float, 16>(a.Y + ((size_t) (pk.row0 + rl)) * a.ldy + 16 * nt, acc.template select<16, 1>(128 * i + 16 * r));
                    }
                }
            }
        });
    });
}

// ------------------------------------------------------------------------------------------------ work-group versions
// A work group is WGT = 8 threads side by side on 8 consecutive output tiles (128 outputs) of the same row block.  The
// activation tiles of the row block are read from global memory ONCE per group, in chunks of 4 steps of 32 (128 of K), into local
// memory (double buffered, one barrier per chunk) and every thread reads them from there, instead of every thread reading all of
// them from L2 (80 threads per expert re-read the same ~200 KB: the activation loads were 8 of 19 us of gate/up).  The codebook
// the decoders gather from is copied into local memory by the group once (its gathers then cost ~nothing).
constexpr int WGT = 8;
constexpr auto kAl16 = sycl::ext::oneapi::experimental::properties{sycl::ext::oneapi::experimental::alignment<16>};

// bytes of the tables a format gathers from (local memory), and the segments copied from the global table buffer
template <int F> constexpr int tab_bytes() {
    return F == T_IQ3S ? 2048 : F == T_IQ3XXS ? 1536 : F == T_IQ2S ? 8192 : 1024;   // IQ3XXS: grid 1024 + ksigns 512
}

template <int F, int MT>
struct WgLayout {
    static constexpr int TB = tab_bytes<F>();
    static constexpr int ASZ = MT * 1024;   // activation tiles of one stage: MT tiles x 4 steps x 256 bytes
    static constexpr int DSZ = MT * 128;    // their scales: MT tiles x 4 steps x 32 bytes
    static constexpr int STAGE = ASZ + DSZ;
    static constexpr int TOTAL = TB + 2 * STAGE;
    static constexpr int SPLIT = 8 / MT;    // threads per tile in the staging load
    static constexpr int PART = 1024 / SPLIT;   // bytes of one tile's activations a thread stages
};

// Staging of one chunk's activation tiles: global -> registers (LOAD_CHUNK) -> local memory stage SS (STORE_CHUNK).  Macros, not
// lambdas: a lambda capturing the staging registers by reference put them in private memory.
#define LOAD_CHUNK(CC)                                                                                                         \
    if (s_tile < nvalid) {                                                                                                     \
        const int* src_ = Xi + ((size_t) (tbase + s_tile) * a.nkb + (CC) * 4) * 64 + s_part * (L::PART / 4);                  \
        _Pragma("unroll")                                                                                                      \
        for (int k = 0; k < L::PART / 256; ++k) stg.template select<64, 1>(64 * k) = block_load<int, 64>(src_ + 64 * k, kAl16); \
        if (s_part == 0) stg_dx = block_load<float, 32>(a.dx + ((size_t) (tbase + s_tile) * a.nkb + (CC) * 4) * 8, kAl16);    \
    }
#define STORE_CHUNK(SS)                                                                                                        \
    if (s_tile < nvalid) {                                                                                                     \
        const uint32_t base_ = L::TB + (SS) * L::STAGE;                                                                        \
        _Pragma("unroll")                                                                                                      \
        for (int k = 0; k < L::PART / 256; ++k)                                                                                \
            slm_block_store<int, 64>(base_ + s_tile * 1024 + s_part * L::PART + 256 * k, stg.template select<64, 1>(64 * k), kAl16); \
        if (s_part == 0) slm_block_store<float, 32>(base_ + L::ASZ + s_tile * 128, stg_dx, kAl16);                             \
    }

template <int F, int MT, int G> struct TagGuW;
template <int F, int MT, int G> struct TagDnW;

template <int F, int MT, int G>
void launch_gu_wg(sycl::queue& q, const GArgs a, int nrowblocks) {
    using L = WgLayout<F, MT>;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<G>};
    constexpr int NS = (F == T_IQ2S) ? 2 : 1;
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagGuW<F, MT, G>>(sycl::nd_range<2>(sycl::range<2>((size_t) nrowblocks, (size_t) a.ntiles), sycl::range<2>(1, WGT)), props,
                                         [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            slm_init<L::TOTAL>();
            const int rbg = (int) it.get_group(0), lt = (int) it.get_local_id(1), nt = (int) it.get_global_id(1);
            const Pick pk = pick_expert(a, rbg);
            const int cnt = pk.cnt;
            const int ntl = (cnt + 7) >> 3;
            const int tl0 = (rbg - pk.rb0) * MT;
            const int nvalid = (ntl - tl0) < MT ? (ntl - tl0) : MT;
            const int tbase = pk.tile0 + tl0;
            const uint8_t* We = pk.blob;
            const uint32_t* T = a.tab;
            const int* Xi = reinterpret_cast<const int*>(a.Xq);
            // the codebook into local memory: thread lt copies its eighth of each segment
            if constexpr (F == T_IQ3S) {
#pragma unroll
                for (int k = 0; k < tab_bytes<F>() / WGT / 64; ++k)
                    slm_block_store<uint32_t, 16>(lt * (tab_bytes<F>() / WGT) + 64 * k, block_load<uint32_t, 16>(T + O_IQ3S + (lt * (tab_bytes<F>() / WGT) + 64 * k) / 4, kAl16), kAl16);
            } else if constexpr (F == T_IQ3XXS) {
#pragma unroll
                for (int k = 0; k < 1024 / WGT / 64; ++k)
                    slm_block_store<uint32_t, 16>(lt * (1024 / WGT) + 64 * k, block_load<uint32_t, 16>(T + O_IQ3XXS + (lt * (1024 / WGT) + 64 * k) / 4, kAl16), kAl16);
                slm_block_store<uint32_t, 16>(1024 + lt * (512 / WGT), block_load<uint32_t, 16>(T + O_KSIGNS + lt * (512 / WGT) / 4, kAl16), kAl16);
            } else {   // IQ2_S
#pragma unroll
                for (int k = 0; k < tab_bytes<F>() / WGT / 64; ++k)
                    slm_block_store<uint32_t, 16>(lt * (tab_bytes<F>() / WGT) + 64 * k, block_load<uint32_t, 16>(T + O_IQ2S + (lt * (tab_bytes<F>() / WGT) + 64 * k) / 4, kAl16), kAl16);
            }
            // staging: this thread's piece of the chunk c (tile, part) into registers, then into stage s
            const int s_tile = lt / L::SPLIT, s_part = lt % L::SPLIT;
            simd<int, L::PART / 4> stg;
            simd<float, 32> stg_dx;
            simd<uint32_t, 16> lane(0, 1);
            simd<uint32_t, 16> ro = (lane & 1u) * a.sec_off + ((lane >> 1) + (uint32_t) (8 * nt)) * (uint32_t) a.rowb;
            simd<float, 128 * MT> acc = 0.f;
            const int nchunks = (a.dbg & 8) ? 0 : (a.nkb >> 2);
            if (nchunks > 0) { LOAD_CHUNK(0); STORE_CHUNK(0); }
            barrier();   // the codebook and chunk 0 are in local memory
            simd<float, 16> d = 0.f;
            simd<uint32_t, 16> w0 = 0, w1 = 0, w2 = 0, w3 = 0;
            for (int c = 0; c < nchunks; ++c) {
                const int sb = c >> 1;
                const int stage_off = L::TB + (c & 1) * L::STAGE;
                if (c + 1 < nchunks) { LOAD_CHUNK(c + 1); }
                if ((c & 1) == 0) {   // new superblock: its scale and the fields shared by its 8 steps
                    const uint32_t P = (uint32_t) sb * blk_bytes<F>();
                    d = half_of(ld16(We, ro + P));
                    if constexpr (F == T_IQ3S) {
                        w0 = ld32(We, ro + (P + 106)); w1 = ld32(We, ro + (P + 66)); w2 = ld32(We, ro + (P + 70));
                    } else if constexpr (F == T_IQ2S) {
                        w0 = ld32(We, ro + (P + 74)); w1 = ld32(We, ro + (P + 78)); w2 = ld32(We, ro + (P + 66)); w3 = ld32(We, ro + (P + 70));
                    }
                }
                const uint32_t P = (uint32_t) sb * blk_bytes<F>();
#pragma unroll 1
                for (int ib4 = 0; ib4 < 4; ++ib4) {
                    const int ib = ((c & 1) << 2) + ib4;
                    simd<uint32_t, 128> fr0 = 0x01010101u, fr1 = 0u;
                    simd<float, 16> sw0 = 1.f, sw1 = 1.f;
                    if (!(a.dbg & 1)) {
                        if constexpr (F == T_IQ3S) {
                            simd<uint32_t, 16> q0 = ld32(We, ro + (P + 2 + 8 * ib));
                            simd<uint32_t, 16> q1 = ld32(We, ro + (P + 6 + 8 * ib));
                            simd<uint32_t, 16> sg = ld32(We, ro + (P + 74 + 4 * ib));
                            simd<uint32_t, 16> qhv = (((ib < 4) ? w1 : w2) >> (8 * (ib & 3))) & 0xffu;
                            simd<uint32_t, 16> scn = ((w0 >> (8 * (ib >> 1))) >> (4 * (ib & 1))) & 0xfu;
                            sw0 = d * simd<float, 16>(simd<int, 16>(scn * 2u + 1u));
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                simd<uint32_t, 16> qsw = r < 4 ? q0 : q1;
                                simd<uint32_t, 16> idx = ((qsw >> (8 * (r & 3))) & 0xffu) | (((qhv >> r) & 1u) << 8);
                                simd<uint32_t, 16> g = slm_gather<uint32_t, 16>(idx << 2);
                                simd<uint32_t, 16> nib = ((sg >> (8 * (r >> 1))) >> (4 * (r & 1))) & 0xfu;
                                fr0.template select<16, 1>(16 * r) = apply_signs(g, nib);
                            }
                        } else if constexpr (F == T_IQ3XXS) {
                            simd<uint32_t, 16> q0 = ld32(We, ro + (P + 2 + 8 * ib));
                            simd<uint32_t, 16> q1 = ld32(We, ro + (P + 6 + 8 * ib));
                            simd<uint32_t, 16> aux = ld32(We, ro + (P + 66 + 4 * ib));
                            sw0 = d * simd<float, 16>(simd<int, 16>((aux >> 28) * 2u + 1u)) * 0.25f;
#pragma unroll
                            for (int il = 0; il < 4; ++il) {
                                simd<uint32_t, 16> sbyte = slm_gather<uint32_t, 16>(1024u + (((aux >> (7 * il)) & 127u) << 2));
#pragma unroll
                                for (int hh = 0; hh < 2; ++hh) {
                                    const int r = 2 * il + hh;
                                    simd<uint32_t, 16> qsw = r < 4 ? q0 : q1;
                                    simd<uint32_t, 16> idx = (qsw >> (8 * (r & 3))) & 0xffu;
                                    simd<uint32_t, 16> g = slm_gather<uint32_t, 16>(idx << 2);
                                    fr0.template select<16, 1>(16 * r) = apply_signs(g, (sbyte >> (4 * hh)) & 0xfu);
                                }
                            }
                        } else if constexpr (F == T_IQ2S) {
                            simd<uint32_t, 16> idw = ld32(We, ro + (P + 2 + 4 * ib));
                            simd<uint32_t, 16> sgw = ld32(We, ro + (P + 34 + 4 * ib));
                            simd<uint32_t, 16> qhb = (((ib < 4) ? w2 : w3) >> (8 * (ib & 3))) & 0xffu;
                            simd<uint32_t, 16> scb = (((ib < 4) ? w0 : w1) >> (8 * (ib & 3))) & 0xffu;
                            sw0 = d * simd<float, 16>(simd<int, 16>((scb & 0xfu) * 2u + 1u)) * 0.125f;
                            sw1 = d * simd<float, 16>(simd<int, 16>((scb >> 4) * 2u + 1u)) * 0.125f;
#pragma unroll
                            for (int il = 0; il < 4; ++il) {
                                simd<uint32_t, 16> idx = ((idw >> (8 * il)) & 0xffu) | ((qhb << (8 - 2 * il)) & 0x300u);
                                simd<uint32_t, 16> gl = slm_gather<uint32_t, 16>(idx << 3);
                                simd<uint32_t, 16> gh = slm_gather<uint32_t, 16>((idx << 3) + 4u);
                                simd<uint32_t, 16> sbyte = (sgw >> (8 * il)) & 0xffu;
                                simd<uint32_t, 16> vl = apply_signs(gl, sbyte & 0xfu);
                                simd<uint32_t, 16> vh = apply_signs(gh, sbyte >> 4);
                                if (il < 2) {
                                    fr0.template select<16, 1>(16 * (2 * il)) = vl;
                                    fr0.template select<16, 1>(16 * (2 * il + 1)) = vh;
                                } else {
                                    fr1.template select<16, 1>(16 * (2 * il)) = vl;
                                    fr1.template select<16, 1>(16 * (2 * il + 1)) = vh;
                                }
                            }
                            fr0.template select<64, 1>(64) = 0u;
                            fr1.template select<64, 1>(0) = 0u;
                        }
                    }
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        if (i < nvalid) {
                            simd<int, 64> av = 0x01010101;
                            simd<float, 8> dxv = 1.f;
                            if (!(a.dbg & 2)) {
                                av = slm_block_load<int, 64>(stage_off + i * 1024 + ib4 * 256, kAl16);
                                dxv = slm_block_load<float, 8>(stage_off + L::ASZ + i * 128 + ib4 * 32, kAl16);
                            }
                            simd<float, 128> tf = dpas_s8(fr0, av);
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const float dxr = dxv[r];
                                acc.template select<16, 1>(128 * i + 16 * r) += tf.template select<16, 1>(16 * r) * (sw0 * dxr);
                            }
                            if constexpr (NS == 2) {
                                simd<float, 128> tg = dpas_s8(fr1, av);
#pragma unroll
                                for (int r = 0; r < 8; ++r) {
                                    const float dxr = dxv[r];
                                    acc.template select<16, 1>(128 * i + 16 * r) += tg.template select<16, 1>(16 * r) * (sw1 * dxr);
                                }
                            }
                        }
                    }
                }
                if (c + 1 < nchunks) { STORE_CHUNK((c + 1) & 1); }
                barrier();
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i < nvalid) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int rl = (tl0 + i) * 8 + r;
                        if (rl < cnt && !(a.dbg & 4))
                            block_store<float, 16>(a.Y + ((size_t) (pk.row0 + rl)) * a.ldy + 16 * nt, acc.template select<16, 1>(128 * i + 16 * r));
                    }
                }
            }
        });
    });
}

template <int F, int MT, int G>
void launch_dn_wg(sycl::queue& q, const GArgs a, int nrowblocks) {
    using L = WgLayout<F, MT>;
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<G>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagDnW<F, MT, G>>(sycl::nd_range<2>(sycl::range<2>((size_t) nrowblocks, (size_t) a.ntiles), sycl::range<2>(1, WGT)), props,
                                         [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            slm_init<L::TOTAL>();
            const int rbg = (int) it.get_group(0), lt = (int) it.get_local_id(1), nt = (int) it.get_global_id(1);
            const Pick pk = pick_expert(a, rbg);
            const int cnt = pk.cnt;
            const int ntl = (cnt + 7) >> 3;
            const int tl0 = (rbg - pk.rb0) * MT;
            const int nvalid = (ntl - tl0) < MT ? (ntl - tl0) : MT;
            const int tbase = pk.tile0 + tl0;
            const uint8_t* We = pk.blob;
            const uint32_t* T = a.tab;
            const int* Xi = reinterpret_cast<const int*>(a.Xq);
            constexpr uint32_t TOFF = (F == T_IQ4NL) ? O_IQ4 : O_Q2;
#pragma unroll
            for (int k = 0; k < tab_bytes<F>() / WGT / 64; ++k)
                slm_block_store<uint32_t, 16>(lt * (tab_bytes<F>() / WGT) + 64 * k, block_load<uint32_t, 16>(T + TOFF + (lt * (tab_bytes<F>() / WGT) + 64 * k) / 4, kAl16), kAl16);
            const int s_tile = lt / L::SPLIT, s_part = lt % L::SPLIT;
            simd<int, L::PART / 4> stg;
            simd<float, 32> stg_dx;
            simd<uint32_t, 16> lane(0, 1);
            simd<uint32_t, 16> ro = (lane + (uint32_t) (16 * nt)) * (uint32_t) a.rowb + a.sec_off;
            simd<float, 128 * MT> acc = 0.f;
            const int nchunks = (a.dbg & 8) ? 0 : (a.nkb >> 2);
            if (nchunks > 0) { LOAD_CHUNK(0); STORE_CHUNK(0); }
            barrier();
            for (int c = 0; c < nchunks; ++c) {
                const int stage_off = L::TB + (c & 1) * L::STAGE;
                if (c + 1 < nchunks) { LOAD_CHUNK(c + 1); }
#pragma unroll 1
                for (int ib4 = 0; ib4 < 4; ++ib4) {
                    const int kb = c * 4 + ib4;
                    simd<float, 16> sw0 = 1.f;
                    simd<uint32_t, 128> fr0 = 0x01010101u;
                    if (!(a.dbg & 1)) {
                        if constexpr (F == T_IQ4NL) {
                            const uint32_t P = (uint32_t) kb * 18;
                            sw0 = half_of(ld16(We, ro + P));
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                simd<uint32_t, 16> qd = ld32c(We, ro + (P + 2 + 4 * j), a.lim);
                                simd<uint32_t, 16> t0 = slm_gather<uint32_t, 16>((qd & 0xffu) << 2);
                                simd<uint32_t, 16> t1 = slm_gather<uint32_t, 16>(((qd >> 8) & 0xffu) << 2);
                                simd<uint32_t, 16> t2 = slm_gather<uint32_t, 16>(((qd >> 16) & 0xffu) << 2);
                                simd<uint32_t, 16> t3 = slm_gather<uint32_t, 16>((qd >> 24) << 2);
                                fr0.template select<16, 1>(16 * j) = (t0 & 0xffu) | ((t1 & 0xffu) << 8) | ((t2 & 0xffu) << 16) | (t3 << 24);
                                fr0.template select<16, 1>(16 * (j + 4)) =
                                    ((t0 >> 8) & 0xffu) | (t1 & 0xff00u) | ((t2 & 0xff00u) << 8) | ((t3 & 0xff00u) << 16);
                            }
                        } else {   // T_Q2_0
                            const uint32_t P = (uint32_t) (kb >> 1) * 18;
                            const int hb = kb & 1;
                            sw0 = half_of(ld16(We, ro + P));
                            simd<uint32_t, 16> q0 = ld32c(We, ro + (P + 2 + 8 * hb), a.lim);
                            simd<uint32_t, 16> q1 = ld32c(We, ro + (P + 6 + 8 * hb), a.lim);
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                simd<uint32_t, 16> qb = ((r < 4 ? q0 : q1) >> (8 * (r & 3))) & 0xffu;
                                fr0.template select<16, 1>(16 * r) = slm_gather<uint32_t, 16>(qb << 2);
                            }
                        }
                    }
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        if (i < nvalid) {
                            simd<int, 64> av = 0x01010101;
                            simd<float, 8> dxv = 1.f;
                            if (!(a.dbg & 2)) {
                                av = slm_block_load<int, 64>(stage_off + i * 1024 + ib4 * 256, kAl16);
                                dxv = slm_block_load<float, 8>(stage_off + L::ASZ + i * 128 + ib4 * 32, kAl16);
                            }
                            simd<float, 128> tf = dpas_s8(fr0, av);
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const float dxr = dxv[r];
                                acc.template select<16, 1>(128 * i + 16 * r) += tf.template select<16, 1>(16 * r) * (sw0 * dxr);
                            }
                        }
                    }
                }
                if (c + 1 < nchunks) { STORE_CHUNK((c + 1) & 1); }
                barrier();
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i < nvalid) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int rl = (tl0 + i) * 8 + r;
                        if (rl < cnt && !(a.dbg & 4))
                            block_store<float, 16>(a.Y + ((size_t) (pk.row0 + rl)) * a.ldy + 16 * nt, acc.template select<16, 1>(128 * i + 16 * r));
                    }
                }
            }
        });
    });
}

// MT from the group's mean number of 8-row tiles per expert.
int pick_mt(const Layout& lo) {
    const double mean = lo.n ? (double) lo.tile0[lo.n] / lo.n : 0;
    if (mean <= 2.0) return 2;
    if (mean <= 4.0) return 4;
    return 8;
}
int dbg_flags() {
    static const int v = [] { const char* s = std::getenv("STRATA_I8_DBG"); return s ? std::atoi(s) : 0; }();
    return v;
}
int mt_for(const Layout& lo) {
    static const int mo = [] { const char* s = std::getenv("STRATA_I8_MT"); return s ? std::atoi(s) : 0; }();
    return mo ? mo : pick_mt(lo);
}

// register mode per tile height: STRATA_I8_GRF=128 (more threads per EU, spills at MT 8) or 256; default 256
int wg_for() {
    static const int v = [] { const char* s = std::getenv("STRATA_I8_WG"); return s ? std::atoi(s) : 1; }();
    return v;
}
int grf_for(int mt) {
    static const int g = [] { const char* s = std::getenv("STRATA_I8_GRF"); return s ? std::atoi(s) : 256; }();
    (void) mt;
    return g == 128 ? 128 : 256;
}

template <class F>
bool dispatch_mt(int mt, F&& f) {
    switch (mt) {
        case 2: f(std::integral_constant<int, 2>{}); return true;
        case 4: f(std::integral_constant<int, 4>{}); return true;
        case 8: f(std::integral_constant<int, 8>{}); return true;
        default: return false;
    }
}

struct QArgs {
    int n;
    int row0[kMaxGroup + 1];
    int tile0[kMaxGroup + 1];
    int nkb;
    int ld;   // row length in elements of the source
};
class QuantTag;
class SwigluTag;

}  // namespace

void make_layout(Layout& lo, const int32_t* cnt, int n) {
    lo.n = n;
    int r = 0, t = 0;
    for (int i = 0; i < n; ++i) {
        lo.cnt[i] = cnt[i];
        lo.row0[i] = r;
        lo.tile0[i] = t;
        r += cnt[i];
        t += (cnt[i] + 7) / 8;
    }
    lo.row0[n] = r;
    lo.tile0[n] = t;
}

namespace {
bool bmg_arch(const sycl::device& d) {
    namespace ex = sycl::ext::oneapi::experimental;
    if (!d.is_gpu()) return false;
    const auto a = d.get_info<ex::info::device::architecture>();
    return a == ex::architecture::intel_gpu_bmg_g21 || a == ex::architecture::intel_gpu_bmg_g31;
}
}  // namespace

bool battlemage(void* stream) {
    sycl::queue& q = *strata::q_of(stream);
    return bmg_arch(q.get_device());
}

bool any_battlemage() {
    static const bool v = [] {
        try {
            for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
                if (bmg_arch(d)) return true;
        } catch (...) {}
        return false;
    }();
    return v;
}

bool available(void* stream) {
    sycl::queue& q = *strata::q_of(stream);
    static std::mutex mu;
    static std::unordered_map<sycl::device, bool> per_dev;
    std::lock_guard<std::mutex> lk(mu);
    const sycl::device d = q.get_device();
    if (auto it = per_dev.find(d); it != per_dev.end()) return it->second;
    const bool ok = [&] {
        const char* v = std::getenv("STRATA_I8_FORCE");
        if (v && v[0] == '0') return false;
        if (!d.has(sycl::aspect::ext_intel_esimd)) return false;
        namespace mx = sycl::ext::oneapi::experimental::matrix;
        bool s8 = false;
        for (const auto& c : d.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>())
            if (c.atype == mx::matrix_type::sint8 && c.btype == mx::matrix_type::sint8 && c.nsize == 16 && c.ksize == 32) s8 = true;
        return s8;
    }();
    per_dev[d] = ok;
    return ok;
}

size_t xq_bytes(const Layout& lo, int K) { return (size_t) lo.tile0[lo.n] * 8 * K; }
size_t dx_bytes(const Layout& lo, int K) { return (size_t) lo.tile0[lo.n] * (K / 32) * 8 * sizeof(float); }

bool workspace(const Layout& lo, int K, int n_ff, Workspace& ws, void* stream) {
    sycl::queue& q = *strata::q_of(stream);
    static std::mutex mu;
    struct St { Workspace cur; size_t cap_x = 0, cap_h = 0; };
    static std::unordered_map<sycl::device, St> per_dev;
    std::lock_guard<std::mutex> lk(mu);
    St& st = per_dev[q.get_device()];
    Workspace& cur = st.cur;
    size_t& cap_x = st.cap_x;
    size_t& cap_h = st.cap_h;
    const size_t nx = xq_bytes(lo, K), nh = xq_bytes(lo, n_ff);
    if (nx > cap_x || nh > cap_h) {
        q.wait();
        if (cur.xq) { sycl::free(cur.xq, q); sycl::free(cur.dx, q); sycl::free(cur.hq, q); sycl::free(cur.dh, q); }
        cap_x = std::max(nx, cap_x + cap_x / 4);
        cap_h = std::max(nh, cap_h + cap_h / 4);
        cur.xq = sycl::malloc_device<int8_t>(cap_x, q);
        cur.dx = sycl::malloc_device<float>(cap_x / 32 / 8 * 8 + 64, q);   // rows * K/32 floats = bytes / 32
        cur.hq = sycl::malloc_device<int8_t>(cap_h, q);
        cur.dh = sycl::malloc_device<float>(cap_h / 32 + 64, q);
        if (!cur.xq || !cur.dx || !cur.hq || !cur.dh) { cap_x = cap_h = 0; cur = Workspace{}; ws = Workspace{}; return false; }
    }
    ws = cur;
    return true;
}

void quantize_rows(const uint16_t* X, const Layout& lo, int K, int8_t* Xq, float* dx, void* stream) {
    if (lo.tile0[lo.n] == 0) return;
    sycl::queue& q = *strata::q_of(stream);
    QArgs a{};
    a.n = lo.n;
    for (int i = 0; i <= lo.n; ++i) { a.row0[i] = lo.row0[i]; a.tile0[i] = lo.tile0[i]; }
    a.nkb = K / 32;
    a.ld = K;
    const size_t nthreads = (size_t) lo.tile0[lo.n] * a.nkb;
    const half* Xh = reinterpret_cast<const half*>(X);
    uint32_t* Xd = reinterpret_cast<uint32_t*>(Xq);
    q.submit([&](sycl::handler& h) {
        h.parallel_for<QuantTag>(sycl::range<2>((size_t) lo.tile0[lo.n], (size_t) a.nkb), [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int T = (int) id[0], kb = (int) id[1];
            // the expert of tile T, by constant-index reads of the argument arrays (see pick_expert)
            int t0e = a.tile0[0], r0e = a.row0[0], r1e = a.row0[1];
#pragma unroll
            for (int i = 1; i < kMaxGroup; ++i)
                if (i < a.n && T >= a.tile0[i]) { t0e = a.tile0[i]; r0e = a.row0[i]; r1e = a.row0[i + 1]; }
            const int cnt = r1e - r0e;
            simd<float, 8> dxv = 0.f;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int rl = (T - t0e) * 8 + r;
                simd<uint32_t, 8> qd = 0u;
                if (rl < cnt) {
                    simd<half, 32> xh = block_load<half, 32>(Xh + ((size_t) (r0e + rl)) * a.ld + kb * 32);
                    simd<float, 32> x = xh;
                    float amax = hmax<float>(abs(x));
                    const float dd = amax / 127.f;
                    const float id127 = amax > 0.f ? 127.f / amax : 0.f;
                    simd<float, 32> y = x * id127;
                    simd<float, 32> adj = 0.5f;
                    adj.merge(simd<float, 32>(-0.5f), y < 0.f);
                    simd<int, 32> qi = convert<int>(y + adj);
                    simd<int8_t, 32> q8 = convert<int8_t>(qi);
                    qd = q8.template bit_cast_view<uint32_t>();
                    dxv[r] = dd;
                }
                block_store<uint32_t, 8>(Xd + ((size_t) T * a.nkb + kb) * 64 + r * 8, qd);
            }
            block_store<float, 8>(dx + ((size_t) T * a.nkb + kb) * 8, dxv);
        });
    });
}

void swiglu_quantize(const float* GU, const Layout& lo, int n_ff, int8_t* Hq, float* dh, void* stream) {
    if (lo.tile0[lo.n] == 0) return;
    sycl::queue& q = *strata::q_of(stream);
    QArgs a{};
    a.n = lo.n;
    for (int i = 0; i <= lo.n; ++i) { a.row0[i] = lo.row0[i]; a.tile0[i] = lo.tile0[i]; }
    a.nkb = n_ff / 32;
    a.ld = 2 * n_ff;
    const size_t nthreads = (size_t) lo.tile0[lo.n] * a.nkb;
    uint32_t* Hd = reinterpret_cast<uint32_t*>(Hq);
    q.submit([&](sycl::handler& h) {
        h.parallel_for<SwigluTag>(sycl::range<2>((size_t) lo.tile0[lo.n], (size_t) a.nkb), [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int T = (int) id[0], kb = (int) id[1];
            // the expert of tile T, by constant-index reads of the argument arrays (see pick_expert)
            int t0e = a.tile0[0], r0e = a.row0[0], r1e = a.row0[1];
#pragma unroll
            for (int i = 1; i < kMaxGroup; ++i)
                if (i < a.n && T >= a.tile0[i]) { t0e = a.tile0[i]; r0e = a.row0[i]; r1e = a.row0[i + 1]; }
            const int cnt = r1e - r0e;
            simd<float, 8> dxv = 0.f;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int rl = (T - t0e) * 8 + r;
                simd<uint32_t, 8> qd = 0u;
                if (rl < cnt) {
                    simd<float, 64> v = block_load<float, 64>(GU + ((size_t) (r0e + rl)) * a.ld + kb * 64);
                    simd<float, 32> g = v.template select<32, 2>(0);
                    simd<float, 32> u = v.template select<32, 2>(1);
                    simd<float, 32> x = g * u / (1.f + exp(-g));
                    float amax = hmax<float>(abs(x));
                    const float dd = amax / 127.f;
                    const float id127 = amax > 0.f ? 127.f / amax : 0.f;
                    simd<float, 32> y = x * id127;
                    simd<float, 32> adj = 0.5f;
                    adj.merge(simd<float, 32>(-0.5f), y < 0.f);
                    simd<int, 32> qi = convert<int>(y + adj);
                    simd<int8_t, 32> q8 = convert<int8_t>(qi);
                    qd = q8.template bit_cast_view<uint32_t>();
                    dxv[r] = dd;
                }
                block_store<uint32_t, 8>(Hd + ((size_t) T * a.nkb + kb) * 64 + r * 8, qd);
            }
            block_store<float, 8>(dh + ((size_t) T * a.nkb + kb) * 8, dxv);
        });
    });
}

bool gate_up(int gu_type, const uint8_t* const* blob, size_t up_off, int n_ff, const Layout& lo, int K, const int8_t* Xq,
             const float* dx, float* GU, void* stream) {
    if (lo.n == 0 || lo.tile0[lo.n] == 0) return true;
    if ((gu_type != T_IQ3S && gu_type != T_IQ3XXS && gu_type != T_IQ2S) || K % 256 != 0 || (2 * n_ff) % 16 != 0 ||
        (up_off & 3) != 0 || lo.n > kMaxGroup)
        return false;
    sycl::queue& q = *strata::q_of(stream);
    GArgs a{};
    for (int i = 0; i < lo.n; ++i) a.blob[i] = blob[i];
    a.sec_off = (uint32_t) up_off;
    a.Xq = Xq; a.dx = dx; a.Y = GU;
    a.tab = tables(q);
    a.ntiles = 2 * n_ff / 16;
    a.nkb = K / 32;
    a.rowb = (K / 256) * block_bytes_of(gu_type);
    a.ldy = 2 * n_ff;
    a.dbg = dbg_flags();
    const int mt = mt_for(lo);
    const int rb = fill_args_mt(a, lo, mt);
    return dispatch_mt(mt, [&](auto mtc) {
        constexpr int MT = decltype(mtc)::value;
        if (wg_for() && gu_type != T_IQ2S && a.nkb % 4 == 0 && a.ntiles % WGT == 0) {   // IQ2_S: its 8 KB codebook copied into local memory by every group costs more than it saves (35 vs 16 us)
            if (gu_type == T_IQ3S) launch_gu_wg<T_IQ3S, MT, 256>(q, a, rb);
            else if (gu_type == T_IQ3XXS) launch_gu_wg<T_IQ3XXS, MT, 256>(q, a, rb);
            else launch_gu_wg<T_IQ2S, MT, 256>(q, a, rb);
        } else         if (grf_for(MT) == 128) {
            if (gu_type == T_IQ3S) launch_gu<T_IQ3S, MT, 128>(q, a, rb);
            else if (gu_type == T_IQ3XXS) launch_gu<T_IQ3XXS, MT, 128>(q, a, rb);
            else launch_gu<T_IQ2S, MT, 128>(q, a, rb);
        } else {
            if (gu_type == T_IQ3S) launch_gu<T_IQ3S, MT, 256>(q, a, rb);
            else if (gu_type == T_IQ3XXS) launch_gu<T_IQ3XXS, MT, 256>(q, a, rb);
            else launch_gu<T_IQ2S, MT, 256>(q, a, rb);
        }
    });
}

bool down(int d_type, const uint8_t* const* blob, size_t down_off, int n_embd, const Layout& lo, int n_ff, const int8_t* Hq,
          const float* dh, float* Dm, void* stream) {
    if (lo.n == 0 || lo.tile0[lo.n] == 0) return true;
    if ((d_type != T_IQ4NL && d_type != T_Q2_0) || n_ff % block_vals_of(d_type) != 0 || n_embd % 16 != 0 || (down_off & 3) != 0 ||
        lo.n > kMaxGroup)
        return false;
    sycl::queue& q = *strata::q_of(stream);
    GArgs a{};
    for (int i = 0; i < lo.n; ++i) a.blob[i] = blob[i];
    a.sec_off = (uint32_t) down_off;
    a.Xq = Hq; a.dx = dh; a.Y = Dm;
    a.tab = tables(q);
    a.ntiles = n_embd / 16;
    a.nkb = n_ff / 32;
    a.rowb = (n_ff / block_vals_of(d_type)) * block_bytes_of(d_type);
    a.lim = (uint32_t) (down_off + (size_t) n_embd * (size_t) a.rowb) - 4u;
    a.ldy = n_embd;
    a.dbg = dbg_flags();
    const int mt = mt_for(lo);
    const int rb = fill_args_mt(a, lo, mt);
    return dispatch_mt(mt, [&](auto mtc) {
        constexpr int MT = decltype(mtc)::value;
        if (wg_for() && a.nkb % 4 == 0 && a.ntiles % WGT == 0) {
            if (d_type == T_IQ4NL) launch_dn_wg<T_IQ4NL, MT, 256>(q, a, rb);
            else launch_dn_wg<T_Q2_0, MT, 256>(q, a, rb);
        } else         if (grf_for(MT) == 128) {
            if (d_type == T_IQ4NL) launch_dn<T_IQ4NL, MT, 128>(q, a, rb);
            else launch_dn<T_Q2_0, MT, 128>(q, a, rb);
        } else {
            if (d_type == T_IQ4NL) launch_dn<T_IQ4NL, MT, 256>(q, a, rb);
            else launch_dn<T_Q2_0, MT, 256>(q, a, rb);
        }
    });
}

}  // namespace strata::prefill::i8
#endif  // STRATA_I8_NO_ESIMD
