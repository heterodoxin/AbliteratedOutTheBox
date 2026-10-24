# Unsloth UD-Q6_K_XL

**In setup since MODEL-UPDATE Phase 2** (`START-HERE.bat --setup --family unsloth --model UD-Q6_K_XL`, engine 0.1.44 or
newer: every expert in RAM from ~118 GB, else it asks and keeps a RAM budget; quality, speed and images in
[QUANTS.md](QUANTS.md); UD-Q5_K_XL and Q8_0 likewise). The manual workflow below still works. This is Unsloth's 6-bit
file of the same model the other packs use:
[unsloth/Qwen3.8-Flash-Next-GGUF, `UD-Q6_K_XL`, revision `38bb39e`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q6_K_XL).
Build it for engine 0.1.38 or newer (earlier engines read `per_layer_token_embd.weight` in IQ4_NL, Q5_0 or FP8 only,
below). UD-Q4_K_XL's manual workflow asks nothing of setup, and neither does this one.

> **Status (MODEL-UPDATE Phase 1).** The Q6_K gate/up kernels are part of every build now (they were the opt-in
> `-DSTRATA_Q6K_EXPERTS=ON`): measured to leave other models' expert-cache slots unchanged ([QUANTS.md](QUANTS.md)). The
> MMQ prompt kernels for Q6_K come with the K-quant library and `STRATA_MMQ_KQ=1`; without it the prompt takes the FP16
> path. The file runs on the standard engine (AMD R9700: the quality yardstick in [QUANTS.md](QUANTS.md)).

## What is in the file

Six shards, 169,165,382,688 bytes (169.2 GB) together ([sizes and SHA-256 from the HF tree API](https://huggingface.co/api/models/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q6_K_XL);
shards 1, 2 and 6 were also checked bit for bit against their Hugging Face LFS hashes after download). Shard 1 holds
only metadata; shard 2 only the output head (`output.weight`, `output_hc_*`, all Q8_0 but `output_hc_norm`, F32);
shard 3 is exactly the PLE table, nothing else.

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf` | 10,946,624 | `8bdc6bad55b6699f46f60708829932ff4018d7ac9bd309c388534fa31753bdb7` |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf` | 682,434,912 | `494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e` |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf` | 54,400,261,312 | `34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20` |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf` | 49,814,108,512 | `0ea4b599880a5a52fcd9c188ba1443f646c69c96571904ddbd6a79246a01a997` |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf` | 49,419,317,344 | `9948e81ae8368144b135a7403d8c0f2cf9a6c2f9f692313865855b2832e696c9` |
| `Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf` | 14,838,313,984 | `e4ae255234f42b18012f6d6eec5bf615b57479b7e0a0233c47436a321ffeec3f` |

Formats, read from the shard headers with `tools/gguf_reader.py` (all six shards verified: shard 4 = blocks 0-20,
520 tensors, shard 5 = blocks 20-41, 530 tensors, and per layer the types above):

- **Routed experts:** `ffn_gate_exps` and `ffn_up_exps` are Q6_K (seen in every layer of shard 6: blocks 41-47);
  `ffn_down_exps` is Q8_0 (per layer 891.3 MB at 640×2560×512, against Q6_K's 688.1 MB at 2560×640×512 for gate or
  up). Gate and up are Q6_K in 47 of 48 layers and Q8_0 in layer 2; down is Q8_0 in every layer (the pack's
  `native_experts.txt`: 47 rows of ggml types 14/8, and layer 2 is 8/8). Q6_K gate/up is native: the
  engine multiplies them directly on the GPU in every path check below, and the CPU has the ggml block layout (210
  bytes per 256 elements); the per-role layout in `native_experts.txt` v4 handles shards whose boundaries cut
  through a layer like UD-Q4_K_XL's layer 11 case does.
- **Everything else:** attention and shared-expert projections, SSM, hyper-connections, the PLE key/value, the token
  embedding and the output head are Q8_0, or F32/BF16 where the engine reads floats — the same pattern
  [UD-Q4_K_XL](UNSLOTH_Q4.md) handles with `tools/iq_pack.py --compat-bf16`.
- **The PLE table** (`per_layer_token_embd.weight`, [160, 320001536]): **Q8_0**, its own shard 3. A row is
  5 blocks of 32 elements at 34 bytes = 170 bytes, and the whole table is 54.4 GB, against the 28.8 GB IQ4_NL of
  UD-Q4_K_XL. Engine 0.1.38 reads Q8_0 rows natively: `PleTable` accepts the type, the row reader is already
  parameterized by bytes-per-row (its selftest runs 90, 110 and 170 byte rows), and no GPU kernel changes —
  `gather` outputs floats. (UD-Q4_K_XL's table is IQ4_NL; the FP8 table of [tools/ple_fp8_pack.py](../tools/ple_fp8_pack.py)
  needs the original checkpoint, which this file is not a substitute for. Converting this Q8_0 table to FP8 would
  trade table precision away for 6% smaller rows; Q8_0 read directly has none that conversion's error.)

Q6_K gate/up costs 210 bytes per 256 elements, against Q4_K's 144, so the routed experts' share of the file grows
about 45% per element read; expected decode speed accordingly lower than UD-Q4_K_XL's on the same hardware. The
table row gather grows 170/90 = 1.89x over IQ4_NL.

## The pack (by hand)

Same workflow as [UD-Q4_K_XL](UNSLOTH_Q4.md#the-pack-by-hand), from the repository root:

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf /path/to/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf \
  --out packs/ud-q6_k_xl --compat-bf16
```

**`--compat-bf16` is required** for the same reason as there (the 195 small Q8_0 projections the engine reads as
BF16). Do not add `--experts-bin`; the experts stay in the GGUF files. Built here end to end on this file
(`packs/ud-q6_k_xl`, the packer's own tests cover the shard and conversion logic via
`.venv/bin/python -m unittest discover -s tools -p test_iq_pack.py`): 1079 tensors indexed, 303 served natively, 460
tensors converted (264 exact, 196 rounded, max abs error 0.0144 - the same profile as UD-Q4_K_XL's), 195
compat-BF16 tensors (1.20 GiB), arena 1.38 GiB, and `native_experts.txt` v4 with the per-role shard column on
layers 20 and 41.

## The server

Like UD-Q4_K_XL's [config](UNSLOTH_Q4.md#the-server), with the Q6 paths and names. `--ple-gguf` is not needed: the
engine finds shard 3 by the table's name. On a machine whose RAM holds the ~110 GB of routed experts plus the rest
(499 GB total, ~325 GB free, measured on the machine this was written on), there is no RAM budget and no
`--expert-cache` gymnastics: drop `--resident-budget-gib` and `--expert-cache-device1` from the Q4 config. The
54.4 GB table is mmap'd (the default) — do not switch it to `--ple-io ram` unless the RAM really is there to lock
it. `mtp/rt` is the base model's draft layer as always.

```json
{
  "exe": "build/strata",
  "args": [
    "--pack", "packs/ud-q6_k_xl",
    "--native", "/path/to/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf",
    "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "8192"
  ],
  "cwd": ".",
  "tokenizer": "packs/ud-q6_k_xl/tokenizer",
  "model_name": "qwen3.8-flash-next-ud-q6_k_xl",
  "log": "strata-ud-q6_k_xl.log",
  "host": "127.0.0.1",
  "port": 8080
}
```

On 64 GB machines treat the Q4 numbers below as the starting point: the RAM budget matters the same way, and
everything it leaves out comes from the SSD, faster now per expert (Q6_K) and per table row (170 bytes).

## What the engine did not have, and has now

UD-Q4_K_XL's expert formats (gate/up Q4_K, down Q5_1) were in every path; Q6_K gate/up was not. This branch adds
it (the PLE Q8_0 reader is the other half):

- **Decode windows:** the grouped expert kernels (`native_expert_grouped`) take gate/up Q6_K — a pinned copy of
  llama.cpp's `vec_dot_q6_K_q8_1` behind `Fmt<14>`, and the dequantizer (`dequantize_q6_K` folded onto the kernel's
  32 threads). Per-token MMVQ (`native_mmvq`) already had Q6_K.
- **The prompt path:** llama.cpp's MMQ prompt kernels instantiate Q6_K (the K-quant library, `STRATA_MMQ_KQ=1`; it builds
  `template-instances/mmq-instance-q6_k.cu` alongside q4_k/q5_k/q5_1; `moe_mmq` accepts and dispatches it).
- **The gates:** `native_expert_supported` took Q6_K/Q8_0 pairs (gate/up Q8_0 as in layer 2 was already in the
  lists) — the engine previously refused the whole pack with "no GPU kernels for".

Tests: `native_grouped_parity` loops Q6_K/Q8_0 pairs; `prefill_mmq_kquant_test` runs Q6_K gate/up products at
1280×2560 against a double-precision reference; `native_expert_parity --synthetic q6_K/q8_0` checks GPU and CPU
against ggml's dequantized weights. All three are built (CUDA, `86;120`). They have not been run on a GPU: the
card was serving the file below.

## Status: what is verified and what is not

Verified here (test clone, Linux, NVIDIA GPU, source build):

- The six files' sizes and SHA-256 (HF tree API; shards 1, 2, 6 additionally against the downloaded blobs).
- The tensor inventory of shards 1, 2 and 6, read from the headers (`tools/gguf_reader.py`); shards 4 and 5's
  tensor counts and per-layer expert types, from the pack's `conversions.json` and `native_experts.txt` v4.
- Q8_0 PLE rows: `ple_reader_test --selftest` runs 90, 110 and 170-byte rows (all OK); `ple_q8_parity` on a
  synthetic Q8_0 table (`tools/ple_q8_fixture.py`) matches ggml's reference dequantizer bit for bit through both
  Mmap and Direct (`max_abs 0.000e+00`). Both still pass on this branch with the Q6_K expert work in.
- The pack on the finished file (numbers above), and the engine binary builds (CUDA `86;120`, MMQ K-quants on).
- Q6_K expert source support: decode-side MMVQ was already there; the grouped verify-window kernels, the MMQ
  prompt kernels and the `native_expert_supported` gate were not, and are in this branch (section above).

The server has run this file. On 2026-10-04, Linux, an RTX PRO 4500 Blackwell (compute capability 12.0) plus a
second GPU, a Threadripper PRO 5975WX (AVX2, no AVX-512), 30 expert-pool workers and 499 GB of RAM, engine 0.1.39
loaded 101.75 GiB of experts (8.43 GiB/s) and served one review. The cold read was 14,915 tokens at 3,176 tok/s.
Fourteen turns generated 5,799 tokens in 74.1 s (78.3 tok/s) and finished at 44,160 tokens of context; drafts were
accepted 3,485 of 4,406 (79.1%). The same machine on engine 0.1.38, before this branch was rebased onto 0.1.39,
read 249,193 tokens cold at 2,802 tok/s and then generated 1,079 tokens at 69.0 tok/s. Quality against llama.cpp
on this file has not been measured, so the page stays experimental. The three Q6_K parity programs above are the
remaining GPU check.