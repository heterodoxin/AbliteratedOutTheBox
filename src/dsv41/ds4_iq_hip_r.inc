// Strata. Part of ds4_iq_hip.inc (included from it): the weights straight into the matrix cores' registers.
// One lane of the B operand of a 16x16x16 step holds 8 consecutive k of one weight row: exactly one 8-value group of a 32-value
// sub-block of an i-quant block (the grid entry, its signs).  grp_dec decodes that group alone: 8 int8 codes (two words) and the
// scale (weight = scale * code), so a wave needs no shared memory for the weights and no barrier: every wave decodes its own
// columns once and uses them for every row of the tile.  kb: the 32-value sub-block of the row, p: its group (0..3).
// The decode is in two halves so that the loads of a k-step can be issued one k-step ahead of its use: grp_load fetches the
// group raw bytes into registers (Raw), grp_dec turns them into codes and scale (the codebook lookups, the signs).
__device__ __forceinline__ float hf(uint32_t h) { return __half2float(__ushort_as_half((uint16_t) h)); }
#include "ds4_iq_hip_s.inc"

constexpr int kKC = 64;   // K a step: two sub-blocks
// BM rows (MT tiles of 16) x BN = NW 16 NTW weight rows per block; NW waves, each NTW column tiles of 16 and every row tile
template <int T, bool GU, int BM, int NW, int NTW>
__global__ void __launch_bounds__(NW * 32) set_kernel_r(const SetArgs a, int dbg) {
    constexpr int NM = GU ? 2 : 1, MT = BM / 16, BN = NW * 16 * NTW, NTH = NW * 32;
    static_assert(BM % 16 == 0, "row tiles");
    __shared__ uint32_t tab[Ty<T>::TAB > 0 ? Ty<T>::TAB : 1];
    __shared__ int srow[BM];
    __shared__ __align__(16) _Float16 xs[2][BM][kStH];
    const int e = blockIdx.z;
    const int lo = a.bounds[e] - a.row0, hi_r = a.bounds[e + 1] - a.row0;
    const int r0 = lo + blockIdx.x * BM;
    if (r0 >= hi_r || hi_r - lo <= a.skip_le) return;   // (uniform)
    load_tab<T>(tab);
    const int nr = min(BM, hi_r - r0);
    const int c0 = blockIdx.y * BN;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int K = GU ? a.n_embd : a.n_ff;
    const size_t rb = row_bytes_t<T>(K);
    const int tid = (int) threadIdx.x, warp = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    for (int i = tid; i < BM; i += NTH) srow[i] = GU ? a.idx[r0 + min(i, nr - 1)] : r0 + min(i, nr - 1);
    __syncthreads();
    const int nst = K / kKC;
    // the weight rows of this lane column tiles
    const uint8_t* wrow[NM][NTW];
#pragma unroll
    for (int m = 0; m < NM; ++m)
#pragma unroll
        for (int j = 0; j < NTW; ++j)
            wrow[m][j] = blob + a.tr[GU ? m : 2] + (size_t) ((c0 + warp * (16 * NTW) + j * 16 + l16)) * rb;

    constexpr int NX = GU ? (BM * 16 + NTH - 1) / NTH : (BM * 8 + NTH - 1) / NTH;
    struct XR {
        float4 f[GU ? NX : 1];
        uint4 u[GU ? 1 : NX];
    };
    auto fetchx = [&](int s, XR& R) {
#pragma unroll
        for (int xi = 0; xi < NX; ++xi) {
            const int c = tid + xi * NTH;
            if constexpr (GU) {
                if (c < BM * 16) {
                    const int row = c >> 4, ch = c & 15;
                    R.f[xi] = *(const float4*) (a.xf + (size_t) srow[row] * a.ldx + (size_t) s * kKC + ch * 4);
                }
            } else {
                if (c < BM * 8) {
                    const int row = c >> 3, ch = c & 7;
                    R.u[xi] = *(const uint4*) (a.xh + (size_t) srow[row] * a.ldx + (size_t) s * kKC + ch * 8);
                }
            }
        }
    };
    auto storex = [&](const XR& R, int b) {
#pragma unroll
        for (int xi = 0; xi < NX; ++xi) {
            const int c = tid + xi * NTH;
            if constexpr (GU) {
                if (c < BM * 16) {
                    const int row = c >> 4, ch = c & 15;
                    const float4 v = R.f[xi];
                    const hv2 p = {(_Float16) v.x, (_Float16) v.y}, q = {(_Float16) v.z, (_Float16) v.w};
                    *(uint2*) &xs[b][row][ch * 4] = make_uint2(as_u32(p), as_u32(q));
                }
            } else {
                if (c < BM * 8) {
                    const int row = c >> 3, ch = c & 7;
                    *(uint4*) &xs[b][row][ch * 8] = R.u[xi];
                }
            }
        }
    };

    fv8 acc[NM][MT][NTW];
#pragma unroll
    for (int m = 0; m < NM; ++m)
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NTW; ++j) acc[m][i][j] = fv8{0, 0, 0, 0, 0, 0, 0, 0};

    {
        XR R0;
        fetchx(0, R0);
        storex(R0, 0);
    }
    // the weights of a step (64 values) are loaded one step ahead: two sets of registers, the step loop unrolled by two
    SRaw sr0[NM][NTW], sr1[NM][NTW];
    auto load_stage = [&](SRaw (&dst)[NM][NTW], int s) {
#pragma unroll
        for (int m = 0; m < NM; ++m)
#pragma unroll
            for (int j = 0; j < NTW; ++j) stage_load<T>(wrow[m][j], s, hi, dst[m][j]);
    };
    auto do_stage = [&](int s, SRaw (&cur)[NM][NTW], SRaw (&nxt)[NM][NTW]) {
        const int b = s & 1;
        XR R;
        if (s + 1 < nst) {
            fetchx(s + 1, R);
            load_stage(nxt, s + 1);
        }
        hv8 bf[NM][NTW];
        auto step = [&](auto tt) {
            constexpr int t = decltype(tt)::value;
            constexpr int sub = t >> 1, ks = t & 1;
#pragma unroll
            for (int m = 0; m < NM; ++m)
#pragma unroll
                for (int j = 0; j < NTW; ++j) {
                    int c0_, c1_;
                    float sc;
                    step_dec<T, t>(cur[m][j], s, hi, tab, c0_, c1_, sc);
                    const hv2 h = {(_Float16) sc, (_Float16) sc};
                    const uint2 u0 = codes4(c0_, h), u1 = codes4(c1_, h);
                    bf[m][j] = __builtin_bit_cast(hv8, make_uint4(u0.x, u0.y, u1.x, u1.y));
                }
            const int kcol = sub * 32 + ks * 16 + hi * 8;
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i * 16 < nr) {   // (a tile past the rows: skipped, uniform)
                    const hv8 af = *(const hv8*) &xs[b][i * 16 + l16][kcol];
#pragma unroll
                    for (int m = 0; m < NM; ++m)
#pragma unroll
                        for (int j = 0; j < NTW; ++j) acc[m][i][j] = wmma16(af, bf[m][j], acc[m][i][j]);
                }
            }
        };
        step(std::integral_constant<int, 0>{});
        step(std::integral_constant<int, 1>{});
        step(std::integral_constant<int, 2>{});
        step(std::integral_constant<int, 3>{});
        if (s + 1 < nst) storex(R, b ^ 1);
        __syncthreads();
    };
    load_stage(sr0, 0);
    __syncthreads();
    for (int s = 0; s < nst; s += 2) {   // (nst is a multiple of 4)
        do_stage(s, sr0, sr1);
        do_stage(s + 1, sr1, sr0);
    }
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int row = i * 16 + 8 * hi + q;
            if (row >= nr) continue;
            const size_t lr = (size_t) (r0 + row);
#pragma unroll
            for (int j = 0; j < NTW; ++j) {
                const int col = c0 + warp * (16 * NTW) + j * 16 + l16;
                if (GU) a.h[lr * a.ldo + col] = __half_as_ushort(__float2half_rn(swiglu(acc[0][i][j][q], acc[1][i][j][q], a.limit)));
                else a.y[lr * a.ldo + col] = acc[0][i][j][q];
            }
        }
}

template <int T, bool GU, int BM, int NW, int NTW>
inline bool runr(const SetArgs& a, cudaStream_t s) {
    const int N = GU ? a.n_ff : a.n_embd;
    constexpr int BN = NW * 16 * NTW;
    if (N % BN != 0) return false;
    const dim3 grid((unsigned) ((a.max_rows + BM - 1) / BM), (unsigned) (N / BN), (unsigned) a.n_exp);
    static const int dbg = std::getenv("HIPK_DBG") ? std::atoi(std::getenv("HIPK_DBG")) : 0;
    set_kernel_r<T, GU, BM, NW, NTW><<<grid, NW * 32, 0, s>>>(a, dbg);
    return cudaPeekAtLastError() == cudaSuccess;
}
