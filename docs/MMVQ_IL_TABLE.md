# The interleaved small-batch matvec (STRATA_MMVQ_IL): how to measure a table for your card

The decode and verify windows run their dense projections (2 to 8 tokens at a time) through `native_mmvq_il`, which reads an
interleaved copy of the q8_1 activations. Every choice it makes gives the same bits as the plain kernels
(`STRATA_MMVQ_IL=0` turns the whole path off); only the speed differs. How many matrix rows one warp takes
(1, 2 or 4, or 0 = use the plain kernel) is a table in `src/kernels/cuda/native_mmvq.cu`, indexed by the
quantisation type (IQ4_XS, Q4_K, Q5_K, Q6_K, Q3_K), the token count (2-4) and the matrix's row count (five classes).
Windows of 5 to 8 tokens have a second table (`kIlRows58_<cc>`, 1 or 2 rows a warp); only sm_120 has one so far (measured on
the RTX PRO 5000: e.g. the Q5_K head at 5 tokens 899 -> 491 us), every other card keeps the plain kernels there.
`STRATA_MMVQ_IL_WIDE=0` turns the 5-8 token path off. `--emit-table` prints both tables.

Which table a card gets:

| card | table |
|---|---|
| sm_86 (Ampere consumer) and sm_120 cards with fewer than 80 SMs (RTX 5070 and smaller) | `kIlRowsShared`, measured on an RTX 3060 and an RTX 5070 together (a cell is non-zero only if the worse of the two cards still gains 3%) |
| sm_120 cards with 80 or more SMs (RTX PRO 5000 / 6000 Blackwell, RTX 5080 / 5090) | `kIlRows_120`, measured on an RTX PRO 5000 Blackwell (110 SMs); `STRATA_MMVQ_IL_120=0` gives them the shared table |
| sm_89 (Ada) and every other sm_80+ card | `kIlRowsShared` too. Nobody has measured sm_89 yet (we have no such card). |
| sm_75 and older | the path is off; the plain kernels run |
| a card listed in `kIlArch` | its own table |

## Measuring sm_89 (or any other card)

1. Build with tests: `cmake -S . -B build -DSTRATA_BUILD_TESTS=ON` and `cmake --build build --target mmvq_il_parity`.
2. Close everything else on the GPU, then run `build/mmvq_il_parity --bench --emit-table` (a few minutes).
   It first checks bit-for-bit parity of every shape and every rows choice (it must end with `mmvq_il_parity: OK`),
   prints a `BENCH` line per shape and token count, and last a ready-to-paste block, one line per type.
3. Post the whole output on the issue. A maintainer pastes the block as `kIlRows_89[]` and adds
   `{89, kIlRows_89, sizeof(kIlRows_89) / sizeof(IlRows)}` to `kIlArch`.
4. To try a table without rebuilding, set `STRATA_MMVQ_IL_ROWS="type:ncols:r0,r1,r2,r3,r4;..."`, for example
   `STRATA_MMVQ_IL_ROWS="23:3:0,0,2,4,4;12:2:0,1,1,1,1"` (the five numbers are the row classes below 2048, 4096, 8192,
   12288 and above; each is 0, 1, 2 or 4). Then A/B the decode: same flags, `STRATA_MMVQ_IL_ROWS` set or unset, 10
   interleaved pairs, medians.

The emit rule: per cell, the fastest rows count if it takes at least 3% off the plain kernel's time, else 0; shapes of one
class that disagree give 0; a class with no measured shape copies its neighbour below.
