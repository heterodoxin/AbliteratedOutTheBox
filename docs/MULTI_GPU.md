# Strata on two or more GPUs (layer split)

One model can run across several NVIDIA cards in one PC. The layers are split into contiguous ranges, one per GPU:
the first card runs layers 0 to K-1, the next card runs K onward, and so on; the last card also runs the output head
and the draft (MTP) layer. Each card keeps an expert cache for **its own layers only**, so two cards hold about twice
the experts one card holds - for the Coder model on a 16 GB + 24 GB pair, nearly all of them, which is where the
speed comes from (decode then barely touches the CPU pool).

This is pipeline (layer) parallelism, not tensor parallelism: a token crosses from one card to the next once per
verify window (a few hundred KB through pinned RAM), not twice per layer. No NVLink or peer-to-peer access is
needed; cards on x4 or x1 slots work, and the PCIe share of each card is probed on its own link. A `--pcie-frac` you give
is every card's share and skips those probes; there is no per-card setting yet.

**Two models on two cards (the app).** A layer split puts one model on several cards. The other way to use two cards is two
models, one per card, each answering on its own. In the app, turn on **Run multiple models** on the Models page: the card
list then says which card a loaded model uses ("In use by ..."), starts that card as Off for the next model, and
explains a failed start on a card that is taken. Measured on a PC with Radeon AI PRO R9700 cards (two of four, Linux): Q2_0 on
one card and IQ3_XXS on another both loaded (33 s and 59 s with the files cached, 114 s the first time), both answered chats
(about 60 tokens/s each), the Monitor listed both, and unloading one left the other answering. This was run with the AMD build only;
the same app code drives NVIDIA cards, but we have not run two models on two NVIDIA cards.

**One make of card per engine.** An engine drives one make of card (NVIDIA, AMD or Intel). The app, `strata launch` and setup
offer only the cards the engine can drive, never split a model across two makes, say in one plain line when the other make has
the bigger card, and say that two models at once on two makes is not supported.

## Using it

**Nothing to type.** `START-HERE.bat` (Linux: `./setup.sh`) lists your NVIDIA cards and says for each one whether
Strata can use it:

```
  Your NVIDIA GPUs:
    GPU 0: NVIDIA GeForce RTX 5080, 16 GB VRAM - can be used
    GPU 1: NVIDIA GeForce GTX 1080 Ti, 11 GB VRAM - not supported - older than the RTX 20 series (compute capability 6.1; Strata needs 7.5 or newer)
    GPU 2: NVIDIA GeForce RTX 3090, 24 GB VRAM - can be used
  ...
  1) RTX 3090 + RAM: fits: needs 44 GB of the 70 GB available   (recommended)
  2) RTX 3090 + RTX 5080 + RAM (layer split): fits: needs 44 GB of the 84 GB available
       prompts are read about 20% faster; the cards share the layers (docs/MULTI_GPU.md)
  3) RTX 5080 + RAM: fits: needs 44 GB of the 66 GB available
Which GPUs? [1]:
```

## Main, Helper, Off

Every card has one of three roles:

| Role | What it does |
|---|---|
| **Main** | Runs the **last** stage of the split: the output head, the draft layer and the check of its guesses. A prompt chunk waits on it. Exactly one card is Main. |
| **Helper** | Holds a share of the layers and of the expert cache, so more of the model fits on graphics cards instead of being read from the drive. Prompts are read about 20% faster. |
| **Off** | Strata does not touch the card (a game, the screen, a card that is too small). |

The Main card is the **last** card in the engine's order, not the first. The first card runs the first layers and, by default,
the picture reader. (Older versions of this page said "the first is the main one"; that has not been true since 0.1.41.1.)

**Recommended roles** (the same rule in the app, `strata launch` and setup; `strata_control/gpu_roles.py`): the faster card
is Main (speed is multiprocessors x max clock, the engine's own measure; cards within 5% are the same card, and the one with
more memory wins, then the lower number; two identical cards: the highest number is Main, so the engine order stays 0, 1 as it always was; a card with clearly more memory beats a faster one that is at least half as fast).
Every other card starts as Helper, except a card with **8 GB or less** when a bigger card is there (Off; with only 8 GB cards, say two GTX 1080, the second one is a Helper), built-in graphics beside a card with memory of
its own (Off) and a card the engine cannot run (Off). Every role is one click (the app), one key (`strata launch`: M, H, O;
R goes back to the recommendation) or one line (below).

Only a few things are warned about, in one short line each: built-in graphics as Main, a card the engine cannot run, too
little memory for the chosen setup (part of the model would be read from the drive). **Every card Off** is allowed. The CUDA,
HIP and SYCL engines cannot start without a card, so the server then says so in one line and stops; an engine that can run on
the processor (the Mac engine) says "Performance is degraded on CPU only." and goes on.

**Setting roles**

```
strata launch --gpu-roles 1=main,0=helper       cards you do not name are Off
./setup.sh --gpu-roles 1=main,0=helper          (START-HERE.bat --gpu-roles ... on Windows); saved
strata launch                                     the Cards step: M Main, H Helper, O Off, R recommended, D details
```

or in the run settings (`strata-*.json`), which is what the app writes:

```json
"gpu_roles": {"1": "main", "0": "helper", "2": "off"},
"gpu": [0, 1],
"gpu_order": "as_given"
```

`gpu` is the same choice for older engines and the Monitor: the Helpers (slowest first), then the Main card, with the Off
cards left out. The server reads `gpu_roles`; when `gpu` was edited by hand and no longer matches, `gpu` wins and the log says
so (a settings file that names its cards keeps them). Nothing is rewritten on upgrade: a file with only `"gpu": [1, 0]` shows
as Main = the last card, Helper = the first, the rest Off. The recommendation itself is not saved, so a card you add later is
recommended again; a card you add after choosing roles starts Off. Docker: `GPU_ROLES=1=main,0=helper` is passed as `--gpu-roles`.

**One rule for which cards count** (`strata_control/gpu_fit.py` keeps the memory arithmetic): not an integrated chip beside
a real card, at least 4 GB of graphics memory to be listed, 8 GB or less starts Off beside a bigger card. The arithmetic is the app's `fitOn`;
each extra card of a split costs an estimated 2 GB of its own buffers.

**Choosing the cards without roles** (at setup or at any start):

```
--gpus 0,2                 these cards together, as nvidia-smi numbers them; Strata puts the faster card last. Remembered.
--gpus all                 every card that can share the model
--gpu 0                    one card (at a start: for that start only)
--layer-split auto         (default) or the first layer of each later card, e.g. 18 or 16,32
```

**Not supported** (setup says so and names the cards that can be used instead):
- a card older than the RTX 20 series (compute capability below 7.5: GTX 10 and older), unless you name it: Pascal
  and Volta cards (Tesla P100 / P40, GTX 10, V100) are admitted when you choose them with `--gpus` (or `--gpu`,
  `--cuda 12`) and then run the experimental CUDA 12 engine, see [OLDER_GPUS.md](OLDER_GPUS.md). Two Tesla P40s
  (`"gpu": [0, 1]` in the config) ran the layer split in a community benchmark: IQ2_XS decode 19.6 tok/s on one card,
  34.8 on both (#1028, experimental, one report);
- a card with less than 8 GB of VRAM, together with others (each card holds a copy of the dense weights and its
  own prompt buffers) - unless you name it with `--gpus`: then setup says the risk and asks (`--yes` with the named
  cards goes ahead);
- Intel GPUs, and a mix of NVIDIA and AMD cards. (AMD cards share a model among themselves: `./setup.sh --backend
  hip --gpus 1,0`, see [AMD_HIP.md](AMD_HIP.md).)

Or edit an existing config (`strata-*.json`), then restart:

```json
"gpu": [0, 2],
"layer_split": "auto"
```

`"layer_split"` is `"auto"` (placed by each card's free VRAM) or the **first layer of each later card**: one rising
number per card after the first, not a count of layers per card. With 4 cards and a 48-layer model, `"24,36,42"`
(or `[24, 36, 42]`) puts layers 0-23 on the first card, 24-35 on the second, 36-41 on the third and 42-47 on the last.
The server checks it before the start and says what is wrong (0.1.39, #644).

**Card order with `"auto"`** (NVIDIA, #1352): the faster card (multiprocessors x max clock) goes **last** - the last stage runs the head, the draft layer and the verify, and a prompt chunk waits on it (a 4070 Ti SUPER + 5060 Ti read a 6K prompt in 50 s one way round and 15 s the other). Cards whose scores are within 5% of each other (two of the same model) keep your order; a manual split keeps it too. `"gpu_order": "as_given"` keeps the order you wrote under `"auto"` as well. The engine log line `layer split: card order ...` says when it changed.

**An unequal pair under `"auto"` (#2644).** The search prices a decode window, and it can give almost every layer to the card it
rates faster. A card with one or two layers holds a handful of experts and clamps the prompt chunk of the whole pipeline
(#1616), so the search then asks for at least 3 layers per card - which on a 4060 Ti 16 GB + 3060 12 GB (48 layers, IQ3_S)
was K=3: the 3060 ran 3 layers, 12 GB idle, 452 tok/s of prefill against 802 at 28 layers on the 3060 (the reporter's sweep of
`"layer_split"` 12, 18, 24, 28, 32: 582, 693, 762, 802, 803 tok/s; the 3060's 360 GB/s of memory bandwidth against the 4060 Ti's
288 counts for more than its SMs). On two cards the engine now first asks for a placement in which each card owns at least its
share of the layers by its room for experts, and the log line says which rule chose the split. Measured on 2x RTX A4000 made
unequal (stage 0 priced at 0.57 of the speed with 3.7 GiB less room, `STRATA_SPLIT_SPEED_SCALE=0.57` and `--vram-reserve-mib 4400`; 14.5K-token
prompt, IQ3_XXS, 5 interleaved pairs): the old rule chose K=3 (3 layers on stage 0), the new one K=20; prompt reading 945 -> 1,441 tok/s (+52%,
5/5 pairs), decode 50.2 -> 65.9 tok/s (+31%, 5/5). Equal cards and three or more GPUs are unchanged.
On x16 + x4 rigs the link-miss scaling (#2007, from a 4x gap) can also move `auto` off the balanced placement (2x RTX 3090, x16 + x4:
K=31 with miss cost x4.20, 114-119 tok/s decode against 130-136 at K=27, #2418): `STRATA_SPLIT_UNIFORM=1` (opt-in, with `"gpu_order": "as_given"`)
prices every link the same, as before 0.1.41.
Prefill that is flat in the prompt length (no gain from 4K to 16K) is the sign of an idle card; **on an unequal pair sweep
`"layer_split"` once** - a 4K-only benchmark cannot see it.

**Skip the split when the first card holds everything** (opt-in, 0.1.31): `"split_skip_if_fits": true` in the config
(engine flag `--split-skip-if-fits`, with `--layer-split auto`) runs on the first card alone when it holds every
profiled expert plus the context's KV, the draft layer and the reserve, and says so in the log; otherwise the split
stays. On an R9700 32 GB + RX 9070 XT the R9700 holds all of the Coder's experts: with the flag 4K prompts read at
1,776 tok/s instead of 1,244 (split) and decode runs at ~60 tok/s instead of ~51 (16K prompts ~5% slower than split).

**Short prompts on a split (0.1.32, #340).** 0.1.30 gave each card's prompt path a loan from its own expert cache,
refilled after every request; on cards that hold nearly all their experts that cost short prompts up to a third of
their speed. 0.1.32 refills all cards at once, uses a smaller streaming ring on a split, and lets a card with free
VRAM keep its own prompt buffers - the same output as 0.1.31, measured on an R9700 + RX 9070 XT: 2K prompts 993 ->
1,265 tok/s, 16K 1,852 -> 1,950, decode unchanged. `STRATA_SPLIT_OWN=1` (opt-in) gives every card its own buffers:
2K 1,450 and 16K 2,227 tok/s there, but a full card then keeps a different set of experts resident, so the output
differs from the default's (stable and coherent); `STRATA_SPLIT_OWN=auto` does that only where the buffers are at
most 12% of each card's VRAM.

**The idle card can help one-chunk prompts (opt-in, `STRATA_PREFILL_HELP=1`).** A prompt that fits one chunk runs the stages one after the other, so while
one card reads its layers the other idles. With it on, each stage hands a share of its streamed experts to the idle card: it
streams them over its own PCIe link into its own (lent) prompt buffers, computes their rows on the MMQ path and sends
them back - `--peer-device`'s peer streaming, without P2P (the activations and the rows go through mapped host memory,
read by copy kernels, so they do not queue behind the expert blobs on either card's copy engine). The share falls with
the prompt (0.41 of the streamed experts at 1.5K tokens, 0.32 at 3K) and is off from ~3.3K tokens, where the stages
overlap anyway and no share paid. Measured on 2x RTX 3090 (UD-Q4_K_XL, no P2P), prompt tok/s without / with: 1.1K 496 /
723, 1.5K 674 / 833, 2K 878 / 1,128, 2.5K 1,017 / 1,296, 3K 1,260 / 1,440; 4K and 8K unchanged, decode unchanged. It
costs no VRAM (the idle card's own prompt buffers) and ~110 KB of mapped host memory per token of the largest chunk it
helped (~360 MB at 3.3K tokens). The rows it computes round like a different MMQ grouping, so the output is not
bit-identical to the default's (it is repeatable: same prompt, same output), which is why it is **opt-in**:
`STRATA_PREFILL_HELP=1` turns it on. Native packs on the MMQ prompt path only (not with the fused prompt kernels,
`STRATA_PF_FUSED=1`, nor with `--peer-device`). With it on, `STRATA_PREFILL_HELP_FRAC=f` fixes the share.
On two Radeon AI PRO R9700 (each limited to 16 GB with `--vram-reserve-mib 17084`, Q2_0 pack, layer split auto, 5 interleaved pairs,
noise arm within 0.2%), `STRATA_PREFILL_HELP=1` gave prompt tok/s without -> with: 700 tokens 638 -> 672 (x1.053, 5/5), 1.5K 839 -> 809
(x0.964), 3K 1,105 -> 929 (x0.840), 6K 1,284 -> 1,211 (x0.943): the PCIe 4.0 x16 links of this box (28.4 GB/s each) make the share lose
from 1.5K tokens up, so it stays off (`STRATA_PREFILL_HELP_FRAC` was not tuned for it).

The engine flags behind it: `--layer-split K1[,K2..]|auto` and `--split-device D1[,D2..]` (the later stages'
devices; default the next visible ones). `--layer-split K --split-device 0` runs both stages on one card sharing
everything - the bit-exact check of the hand-off, not a speed mode.

**The draft layer's prompt K/V in batches on a split (opt-in, `STRATA_SPLIT_MTP_BATCH=1`).** With `--mtp` the draft
layer sits on the last card, and each prompt chunk ends by filling that layer's K/V for the chunk's rows. On one card the
prompt path does it in batches (`Prefill::draft_kv`, a few large matrix products per chunk); on a split the drafter's own
pass always did it, one graph launch and four device copies per group of 8 rows (1,024 groups per 8,192-token chunk).
With the variable set, the last card's prompt path batches it as it does on one card. Restart-only; default off, so the
default start is unchanged. It needs a `--native` pack (the GGUF-form token table) and a paged or ring (`--kv-resident`)
draft K/V (`STRATA_MTP_BATCH_RING=0` sends a ring back to the old pass); otherwise it says why once and keeps the old pass:
`strata serve: draft layer prompt K/V batched on CUDA<d> ...` or `strata serve: STRATA_SPLIT_MTP_BATCH=1 declined, the
drafter's own pass runs: <reason>`. The draft layer's K/V come out of Q8_1 x Q8_0 MMQ instead of its own mmvq, so the
drafts, and how many are accepted, can move; the target's tokens are decided by the verify window.
`tools/split_mtp_batch_parity.py` compares the greedy texts of the two arms. Measured on one machine (2x RTX 3080 20 GB
at 220 W, Xeon E5-2696 v4, UD-Q4_K_XL, `--layer-split 23`, `--batch-mtp`, two slots, `--prefill auto:16384`), two full
restarts per arm in the order off, on, off, on, the same binary in every arm: pooled medians of the prompt read were 1,996
tok/s without and 2,129 with at 25K tokens (+6.7%), 2,448 and 2,553 at 51K (+4.3%), 2,846 and 2,940 at 104K (+3.3%); all 8
on/off ratios of medians are above 1 (1.03 to 1.08), and the two off restarts differ by at most 2% at 25K and 0.3% at the
other sizes. The host's "after each chunk" time per request fell from 0.9-1.3 s to 0.1-0.25 s (one restart per arm). Greedy
output (`--adapt-every 0 --suffix-draft 0`, five prompts of 256 tokens and one 25.5K-token prompt) was identical in 6 of 6
prompts in all five comparisons, the two runs inside one process (A/A) included, and the drafts accepted were 826 of 1,210
in every run, so no difference in acceptance was visible. Decode is not shown to be unchanged: the medians with the variable
set were about 3% lower (pooled solo 82.6 -> 80.2 tok/s, two streams 93.2 -> 90.8), and the same configuration restarted
twice moved up to 4.7%, so two restarts per arm can neither confirm nor exclude it; the change touches only the prompt path.

**Each card loads only its own layers' dense weights** (0.1.39, PR #639) with explicit split points (`--layer-split
27`, not `auto`): every card used to keep a full copy (~3.4 GB for the Coder) though its stage reads only its own
layers, and the VRAM it frees goes to that card's expert cache (2x MI50 16 GB, Coder: 8,819 -> 10,626 experts in
VRAM, decode 39.2 -> 41.7 tok/s). Opt-in for now, on AMD and NVIDIA alike: `STRATA_STAGE_TRIM=1` (please report how
it goes).
A card holding more experts can change which experts run on the GPU, so the output can differ slightly from a run
without it.

**The resident RAM mode works on a split** (`--resident-experts`, the low-RAM mode). The RAM copy of the experts
leaves out the ones every card's cache holds, not only the first card's, and when the rest does not fit whole it keeps
the hottest by the expert profile over all the layers. An adaptive swap copies the evicted expert back into RAM from
the card that owns its layer. Before, `--resident-experts` with a split ran as `--mmap-experts`, and setup recommended
one card in the low-RAM mode; with an engine that has it (`RESIDENT_SPLIT_ENGINE` in setup.py), setup keeps the cards
together and the experts no card holds in RAM. The same copy, sized by a RAM budget (`--resident-budget-gib`, the
Unsloth UD-Q4_K_XL and UD-IQ4_XS configs), is kept on a split as well ([UNSLOTH_Q4.md](UNSLOTH_Q4.md)). Swift 1.5 IQ3_XXS at
160K (q4_0 KV, `--prefill 4096`, `--spec 4` with the stock draft layer), RTX 4060 Ti (layers 0-19) + RTX 5080 (20-47),
i9-14900KF, 32 GB of RAM, Windows 11, four greedy prompts at a time, decode tok/s:

| | first four prompts | after three more rounds |
|---|---|---|
| 5080 alone, `--resident-experts` | 25 | 29 |
| split, `--mmap-experts` (what `--resident-experts` became on a split) | 32 | 64 |
| split, `--resident-experts` (22 GiB of experts locked in RAM) | 71 | 69 |

The split with `--mmap-experts` catches up once the OS file cache holds the experts, on a PC with nothing else
running; the resident copy is there from the first request and stays locked when other programs need the RAM. With
`--pcie-frac 0 --adapt-every 0` the split's greedy output is the same with either mode.

**The prompt path's loan on a split and its streamed ring (`STRATA_SPLIT_RING`).** Under a split the prompt path borrows
the tail slots of each card's expert cache for its buffers (they are given back after the prompt), and the resident RAM
copy keeps those experts too, so a borrowed expert is not read back from the model file during a prompt. How many slots
are borrowed depends on the prompt path's streamed ring: a split uses 96 ring slots when at least 75% of the (layer,
expert) pairs sit in some card's cache, and `STRATA_SPLIT_RING=N` sets N slots (`0` goes back to the pinned-share rule).
The ring used to be chosen after the RAM copy's regions were sized, so the regions were sized for a different ring than
the prompt path then used: with `STRATA_SPLIT_RING=384` it borrowed 5,396 slots and the copy kept 4,662, and the
other 734 were read from the model file in every chunk. The ring is now chosen first (`Prefill::set_ring_override`
says it must be set before the buffers are counted). The start-up line `strata serve: lend sizing: the prompt path
borrows N slots (ring R slots at chunk C), K of them keep their experts in RAM too` shows both numbers, and a
`WARNING` follows when K is below N. Without a split, or without `--resident-experts`, nothing changes. With a split
the order of the two steps changed for every ring choice, the 96-slot rule included: its regions used to be sized for
the larger default ring, so there the copy probably kept more slots than were borrowed (not measured). The 5,396 /
4,662 figures were measured on the test machine (2x RTX 3080 20 GB at 220 W, Xeon E5-2696 v4, UD-Q4_K_XL,
`--layer-split 23`, resident RAM mode, `--prefill auto:16384`, 2 slots), one restart per arm, with `STRATA_SPLIT_RING=384`
and `STRATA_PREFILL_RING` unset. Before the change a 104K-token prompt made 413 blob reads and read 10.3 GB of experts
from the model file per request; after it, 0, with all 5,396 borrowed slots kept in RAM, for 2.15 GiB more pinned RAM
(13.61 -> 15.76 GiB). 25K and 51K prompts read nothing from the file in either case. With the override unset and
`STRATA_PREFILL_RING=384` the lend lines are identical before and after (5,396 / 5,396, 15.76 GiB). No speed gain is
shown: the medians of three 104K reads were 2,803 tok/s before and 2,907 after, but the spread before is 2,540-2,897
(its first read faulted 4.2 GB in from NVMe) and there is no A/A restart. Here the page cache absorbed the 10 GB per
request; with less spare page cache it would be disk traffic. Not measured: both ring variables unset.

**A separate VRAM reserve for the later cards:** `--vram-reserve-later-mib N` (default: `--vram-reserve-mib`'s value).
The card that drives the monitors needs more headroom than one that drives none; with the display on the last card,
`--vram-reserve-mib 300 --vram-reserve-later-mib 1800` gives the first card's cache that VRAM.

**auto** tries every placement (all of them for two, three or, since 0.1.40, four cards; proportional to the free VRAM beyond that) and
keeps the one whose caches would hold the most of the expert profile, hottest pairs weighted most; ties go to the
placement that leaves the fullest card the most room. The startup log prints the choice:

```
strata generate: layer split auto: K=19 - the caches hold 11767 of 12288 profiled pairs (fullest device 100%)
strata serve: layer split: layers 0-18 (CUDA0), 19-47 (CUDA1), one hand-off per window
```

## What each card holds

- **every card**: a copy of the dense weights (~3.4 GB for the Coder), its own session state (the KV cache of the full
  context), its verify window and its prompt-path buffers, and an expert cache for its layers filled from the profile;
- **the last card**: also the output head and the draft layer (~0.8 GB);
- **host RAM**: the expert arena once, shared by all cards (the CPU pool computes whatever no card holds).

Prompts are read in chunks that flow through the cards in turn; while a later card reads chunk c, the first card
already reads chunk c+1. (0.1.44, AMD gfx12: the auto chunk on a split stays at 8192, not the single-card 16384: a 16K prompt is then two chunks
that overlap instead of one, a 32K one four instead of two. Same tokens; 2x / 3x / 4x R9700 prompt tok/s at 16K +36% / +45% / +53%,
32K +18% / +21% / +29%, 64K +4% / +12% / +17%, 8K unchanged. `STRATA_SPLIT_CHUNK16K=1` keeps 16384.) Conversation checkpoints save and restore every card's state; the adaptive expert swaps copy
into the card that owns the layer.

## Overlapped hand-off (`STRATA_SPLIT_OVERLAP=1`, opt-in)

By default each verify window runs the stages one after another: the host syncs stage 1, then stages and launches
stage 2's graph, and so on. `STRATA_SPLIT_OVERLAP=1` (stages on different cards only) gives the hand-off a ready
flag: the earlier stage's GPU writes the hand-off with volatile stores and raises the flag (`handoff_publish`), the
next stage's graph is staged and launched on a helper thread while the earlier stage is still running and waits for
the flag on its GPU (`wait_flag_ge`), and the host goes straight from serving one stage's layers to the next, syncing
the earlier stage only once the chain is done. It takes one stream sync and one graph launch per window off the
critical path. Batch windows (several conversations) keep the serial order. It is off beside `--pipeline-windows`, whose loop overlaps the stages its own way.

Measured on 2x RX 7900 XTX (gfx1100, ROCm, Linux), Ryzen 9 9900X, 52 GB RAM, Swift 1.5 IQ3_XXS, 128K context, 8-bit
KV, layers 0-25 / 26-47; 512-token greedy answers, median of 8 runs:

| | serial (default) | `STRATA_SPLIT_OVERLAP=1` | one card |
|---|---|---|---|
| decode, code prompt | 53 tok/s | 70.1 tok/s | ~70 tok/s |
| decode, prose prompt | 45 tok/s | 55.9 tok/s | ~68 tok/s |

Greedy output is byte-identical with the flag at 0 and 1; the quiz set scored the same (47/51). On NVIDIA it
gives nothing: 2x RTX A4000 (IQ3_XXS, 5 interleaved pairs, 0.2.1 engine), decode story +0.3 %, code +2.1 %, facts -0.3 % against
+2.8 % of noise, prompts +0.1 %, with the same tokens. A sync and a graph launch cost less there. The second card then reads about 94 %
busy in `nvidia-smi` (a kernel waiting for the flag), against 58 % without the flag, with no more tokens per second.

The reading stage's wait for the flag is bounded on the GPU: `STRATA_SPLIT_WAIT_MS` (default 30000, 0 = no bound).
If the flag does not arrive in time, the wait marks an error word and returns, the window's graph completes, and the
host fails that window with an error naming the setting (the server then restarts the engine) instead of using a
stale hand-off. Normally the host's own per-layer wait (20 s, #267) ends a stalled window first and releases every
GPU wait, the hand-off's included. Test hook: `STRATA_TEST_HANDOFF_DROP=N` withholds the flag in the first stage's
N-th window, as if the publish were lost.
## Prompt chunks sized for the pipeline (`--prefill-pipe`, opt-in; setup's `--calibrate` sets it)

`--prefill auto` takes the largest chunk the buffers allow (up to 8192), which is what one card wants: a chunk costs a
fixed pass over its layers' experts (b: dequantized, or streamed over PCIe where they are not in VRAM) plus a part per
token (a), so few large chunks are cheapest. A layer split reads a prompt as a pipeline - stage s reads chunk c while
stage s+1 reads chunk c-1 - and then takes about sum_i (b + a C_i) + (S - 1) max_i (b + a C_i) for chunks C_i over S
stages: large or uneven chunks keep the later cards waiting (a 10.7K-token prompt in a 7680 and a 3011 chunk over
four stages: the first stage carried 24 of the 36 s), small ones pay b again and again.

`--prefill-pipe R` reads each prompt in the chunk count with the least pipeline time for b/a = R tokens, when that
beats today's split by 3% or more: every count from the fewest the buffers allow is tried, each chunk evened out on the
256-token grid (at least 512, at most the buffers' chunk; keep `--prefill auto` so long prompts may use large chunks).
b/a differs by an order of magnitude between rigs - about 700 tokens on four GP100 dies (every expert in VRAM, slow
arithmetic), about 5000 on two RTX 3090 Ti (part of the experts streamed per chunk, fast arithmetic) - and a wrong one
costs: a fixed 1024 read 19-24% slower on the two RTX 3090 Ti and 22-36% slower on two RTX 4090 (reported in the PR by
MistyMoonR). So it is neither guessed nor learned from the prompts the engine serves (that made a prompt's chunks, and
so its output bits, depend on how busy the PC happened to be): like the PCIe share and the draft floor, it is a
setting of the PC that setup's `--calibrate` measures once (`tools/calibrate.py`: on a layer split, the prompt read
speed of a 2K- and an 8K-token prompt - the same text every time, after a header of its own so nothing is reused -
with the chunks off, evened out and b/a 768, 1536, 3072 and 6144; the best is kept only when it reads faster than off
in each of three paired rounds and by more than 3% in the median) and writes into the config. Measured so: four GP100
dies keep b/a 768 (355-356 against 219-220 tok/s in every round, +62%); two RTX 3090 Ti keep it off (nothing read
faster than off, b/a 768 20% slower). A first version that read a different stretch of text per value and confirmed in
two rounds picked 1536 on the RTX 3090 Ti pair from one fast read - where the experts are streamed, the read speed
depends on the text. `--prefill-pipe 1` needs no
b/a: it is `STRATA_PREFILL_EQUAL=1`'s rule (#693) - today's chunk count, evened out on the 256-token grid, and only
when today's last chunk has 1024 tokens or more (3072 with `STRATA_PREFILL_CPU_SHARE` set). A chunk that size streams
every expert its card does not hold, a shorter one only the experts it routes to, so a prompt just past the chunk size
reads as one full chunk and a nearly free tail, and evening that out would add a second full pass over the experts:
the first version evened every split out, and on two RTX 4090 MistyMoonR's calibration read 1,601 tok/s with it
against 2,129 off. 0 (the default) is today's split;
`STRATA_PREFILL_PIPE=<R>` is the same as the flag where the config cannot carry it. The chunk geometry changes the
rounding, which is why it is opt-in.

A prompt segment (the prompt between two of its cuts: the system-prompt root, the last turn, a `pin=`) is planned once,
on its whole length and R, and every run of it reads in that chunk. So its chunks, and the loan its buffers are laid
out for, are the same whether it is read in one run, in pieces between the windows of decoding `--batch` slots, or goes
on after a `BYIELD` or from a periodic checkpoint taken inside it (the slot and the checkpoint keep the plan; the read
goes on in it when it continues the same segment, else it plans again and says so in the log). The output bits then
match as far as the experts the cards hold do (`--adapt-swaps 0` for byte-identical repeats, as without the pipe).
Before this, a read beside decoding slots re-planned every piece, and one that went on after a `BYIELD` re-planned what
was left. A chunk the pipe planned takes no `STRATA_PREFILL_CPU_SHARE`: the stages read
such chunks at the same time, and which stage got the CPU pool first would decide which experts the CPU computed. The
DONE line ends with `chunk=` (the first chunk of the request's longest batched segment) and `experts_streamed=` (the
expert blobs its prompt copied to the cards), and the INFO line carries `prefill_cap`, `prefill_stages`,
`stream_all_min` and `prefill_pipe`. One line in the log says what a plan chose:

    strata prefill: 10058 tokens in 1280-token chunks over 4 stages (--prefill-pipe 706: b/a)

Four GP100 dies (PH402 SKU 200, `--layer-split 12,24,36`, Flash-Next IQ3_XXS, 64K context), prompt tok/s, fresh
prompts, two interleaved rounds (each within 0.2%), with b/a 1024 (the same chunks as 700-890, measured on this rig):

| Prompt tokens | `--prefill auto` | `--prefill 2048` | `--prefill auto` + `--prefill-pipe 1024` (chunk) |
| ---: | ---: | ---: | ---: |
| 2,013 | 180.9 | 181.4 | 228.0 (1024) |
| 3,865 | 207.6 | 276.5 | 312.4 (1024) |
| 10,041 | 261.1 | 433.0 | 432.6 (2048) |
| 28,998 | 475.7 | 554.1 | 559.7 (3328) |

Two RTX 3090 Ti (CUDA 12.9 build for sm_86, two stages, Swift 1.5 IQ3_XXS, 1M context, 13,371 of 24,576 experts in
VRAM), prompt tok/s, fresh prompts, two starts per arm, each reading the four prompts twice (the cold first pass after
a card was freed is left out); b/a 5836 and 6432 (timed on the rig), against 1024:

| Prompt tokens | `--prefill auto` | + b/a 5836 / 6432 (chunk) | for comparison: b/a 1024 |
| ---: | ---: | ---: | ---: |
| 2,035 | 694-815 | 698-816 (today's) | 560-562 |
| 3,887 | 1,127-1,207 | 1,115-1,227 (today's) | 880-885 |
| 10,063 | 1,437-1,553 | 1,654-1,681 (2 x 5120) | 1,109-1,113 |
| 29,019 | 2,532-2,603 | 2,566-2,620 (today's) | 2,064-2,089 |

## Limits (for now)

- **Works across cards** (bench/results/2026-09-29-layer-split-limits):
  - images (`--vision`): each card keeps its own image-position table;
  - control vectors and the experimental speed projection: each card holds the vector's tables, switched on and
    off per request on all of them;
  - KV streaming (`--kv-resident`): each card streams the KV of its own session;
  - mid-prompt checkpoints (`--prompt-cache-every`): each card saves its part of a checkpoint when it has read that
    chunk;
  - the older helper-GPU caches (`--expert-cache-remote`, docs/SECOND_GPU.md): they take the visible GPUs no stage
    runs on, and hold only experts no stage's cache holds. On the test rig, a 2080 Ti helper made decoding slower,
    as it did without a split: its per-layer round trip costs more than the CPU pool needs for those experts.
- `--mmap-experts` needs a canonical pack (`experts.bin`), with or without a split; a native (IQ) pack says so at
  start.
- The prompt path has its own buffers on every card (1.5 GB each at the default 2048-token chunk; `--prefill 1024`
  halves that) instead of borrowing cache slots as one card does. An explicit `--expert-cache` on the first card is
  capped to leave room for them.
- Under WDDM (Windows, and WSL2) only 8 GiB of the expert arena is pinned (more, mapped into two GPU contexts,
  leaves WDDM refusing allocations); the rest streams through the pinned staging ring. A Linux driver has no such
  limit, so there the whole arena is pinned (since 0.1.31; the cap cost a 4090 + 3060 split two thirds of its
  prompt speed, #253). `STRATA_ARENA_PIN_GIB=N` pins at most N GiB, `0` the whole arena, on any OS.
- Every card needs compute capability 7.5 (RTX 20 or newer). The pre-sm_80 QSA scorer path is fp32 FMAs, so a
  Turing card runs the same kernels instead of the tensor-core prompt attention.

## Measured

The Coder on an RTX 5080 + RTX 3090 (Ryzen 9 9950X3D), 32K context; details in
`bench/results/2026-09-29-layer-split/`:

| | Prompt 16K / 28K tok/s | Decode story / code tok/s |
|---|---|---|
| 5080 alone | 1,726-2,017 / 1,970 | 83-87 / 88-105 |
| 5080 + 3090, best split (K=26) | 2,039 / 2,357 | 84 / 110 |
| 5080 + 3090, auto (K=22) | 2,037 / 2,073 | 80 / 109 |

- **Prompts gain the most** (+18-20%): each card reads its own layers of the chunk while the other reads the next.
- **Decode is on par with the faster card alone**, and ahead on code. Once both caches hold nearly every routed
  expert, the per-layer GPU time decides.
- **Correctness:** one GPU is byte-identical to 0.1.20, and the hand-off itself is bit-exact.

**Which cards and in what order:**
- Put the fastest card first; auto gives it as many layers as its cache allows.
- Leave out a much slower card when two already hold the model. An RTX 2080 Ti as a third card made the 5080 +
  3090 pair slower (68 / 90 tok/s decode): every extra card costs its own round per window.
- More cards pay off when the model's routed experts do not fit the faster ones.

## One conversation with both cards busy (`--pipeline-windows`, default on two or more AMD gfx12 stages)

**Default on AMD RDNA4 (gfx1200 / gfx1201).** With `--serve`, the draft layer and a layer split of **two or more stages**,
`--pipeline-windows 2` is on unless you say otherwise (the log says "is on by default for a layer split of N stages"; two
stages since 0.1.44, three or more since 0.1.43). It is no quality trade: the tokens are the serial loop's up to CPU rounding
(below). `--pipeline-windows 0` or `STRATA_PIPELINE_AUTO=0` turns it off; `--pipeline-windows 2` asks for it on any split.
The same options that switch it off when you ask for it (`--batch`, `--peer-device`, the helper caches, no draft layer,
`--adapt-async 1`) leave it off without a line when it is only the default. NVIDIA builds keep it opt-in.

**Where the draft layer sits (0.1.44).** The draft layer's chain used to run on the last card, beside the last stage's window.
On 2x / 4x R9700 the two streams then slowed each other, in most engine starts and for the whole life of that engine: the
last stage's window took 27 ms instead of 13 ms while a chain ran (14.9 ms in the lucky starts, which also decoded 25% faster;
the 0.1.43 notes' "128.9 does not reproduce" was this). With the pipeline on, the engine now loads the draft layer on the
**first** card, whose window has finished when the chain starts: the chain's source rows are copied across once per chain and the
draft head's rows once at load. Same tokens; `STRATA_MTP_STAGE0=0` keeps the old place. Exact-flags A/B, 5 pairs, decode tok/s
(serial loop -> now): 2 cards code 103 -> 125 (+21%), 4K prompt +8%, prose equal; 3 cards code +19%, 4K +12%, prose +2%; 4 cards
code +22%, prose +4%, 4K +15%.

With a split the cards take turns on a verify window: the first card runs its layers and hands off, then waits while
the last card runs the rest, the head and the draft. `--pipeline-windows 2` lets the first card start the **next**
window while the last card still verifies this one. The next window is a guess: that this window is accepted whole
and that its bonus token is the one the draft layer predicts (the draft layer is run on through the drafts of the
window in flight). When the guess holds, half of the next window is already done; when it does not, the first card
puts its state back (a copy of its recurrent state taken while the window ran) and the next window is built from the
real tokens. A guessed window is only started when the draft layer's estimate says it is likely to be kept. Windows
copied from earlier context (`--suffix-draft`) are guessed past as well, which is where edits that copy text gain
most. `--pipeline-windows 1` overlaps only the short prompt reads that go through the verify windows
(`--short-read`).

Two or more cards, `--serve`. With three or more stages every stage but the last is a front stage: the guessed
window follows the verified one through them one card behind (each front stage keeps its own pair of recurrent-state
copies and puts its state back on a wrong guess), and only the last stage waits for the verdict. `--pipeline-windows 1`
and the asynchronous tier stay two-stage. In the config:

```
"args": [ ..., "--pipeline-windows", "2" ],
"layer_split": "20"
```

- **Cost**: a second verify window on each card, 160 MiB more kept out of each card's expert cache, plus two copies
  of the first card's recurrent state (about 3 MiB per GDN layer it runs) on the first card with `2` (on every front
  card with three or more stages; only the first card's room is kept out of its cache, the others allocate theirs after).
- **Same text**: the last card only ever runs windows that are verified, and every window row computes what it
  would in any other window, so the tokens are the serial loop's. With `STRATA_IQ_MT_MIN=1 --pcie-frac 0
  --adapt-every 0` the greedy output is identical bit for bit to the serial loop's with the same expert caches. The pipeline keeps
  its VRAM out of the caches, so against a run without the flag a few experts move from a card to the CPU, which
  rounds them differently, and a near-tie can flip (a serial run given the same caches through `--vram-reserve-mib`
  matches it exactly).
- **Off, with one line in the log saying why**, with `--batch` slots, `--peer-device`, the helper caches
  (`--expert-cache-device1..3`, which `--remote-expert-opt` builds on), a split onto one GPU (`--split-device 0`), or no
  draft layer; on three or more stages also `--pipeline-windows 1` and `--adapt-async 1`. A request with repetition penalties (`penalty_last_n`) or coupled
  draft sampling decodes serially.
- **With the resident RAM mode's asynchronous swaps** (`--adapt-async 1`, [DETAILS.md](DETAILS.md)) a round's steps
  advance between the verified windows. Each card's copies are queued by the decode loop itself while that card has
  no window in flight, after every window that may still read what they overwrite has finished; the moves into RAM
  wait the same way.
- **Measured** (Swift 1.5 IQ3_XXS, 160K context, q4_0 KV, the stock draft layer, RTX 4060 Ti (layers 0-19) +
  RTX 5080 (20-47), i9-14900KF, 32 GB of RAM with the resident RAM mode on the split (#848); greedy, 500 tokens, two
  interleaved pairs of three rounds, decode tok/s): Python code 90.4 -> 103.1, C code 76.6 -> 81.6, English prose
  63.0 -> 71.1, Italian prose 41.6 -> 46.3, a copy-heavy edit (a 5 KB file back with a rename) 90.5 -> 117.9; mean
  72.4 -> 84.0 (+16%). It pays when the windows are GPU-bound: with the experts read through the OS file cache
  (`--mmap-experts` on that 32 GB PC) the file reads dominate and it measured no faster.
- **Measured on four stages** (Flash-Next GSQ-RCO IQ3_S, 262K context, int8 KV, layer split 12,24,36 on 4x RX 7900 XT,
  every expert in VRAM; greedy, reasoning off, 256 tokens, repository-text prompts): decode 59.4 -> 67.6 tok/s at 4K
  (median of 3) and 51.8 -> 65.6 at 32K. A guessed window that holds costs ~15 ms against ~43 ms for a fresh one; about
  a third of the guesses held. The greedy text was identical to the serial loop's on all four prompts (`STRATA_IQ_MT_MIN=1
  --pcie-frac 0 --adapt-every 0`), and with every guess forced wrong (`STRATA_PIPELINE_FORCE_MISS=1`).
- **HIP**: the draft layer's per-row `gr_read` used the last stage's session scratch (`ss.block.gr`), which that stage's
  verifier also uses; a chain launched at a verdict and the next window's last stage then ran on the card at once and
  the window's head read a clobbered mix (wrong tokens after a guess that held). The draft layer now has its own scratch.

The `STRATA_PIPELINE_*` tuning and test variables (THETA, FORCE_MISS, SWITCH, LOG, TRACE, SPEC_DEPTH and the like) are read only with
`STRATA_PIPELINE_DEBUG=1`. `--pipeline-windows` and `--adapt-async 1` combine: the engine turns the asynchronous tier
off beside `--pipeline-windows 2` only when `STRATA_PIPELINE_ADAPT_ASYNC=0` is set. What switches either one off is
printed once at start ("is off: ..."). `--remote-expert-opt` does something only with a helper cache
(`--expert-cache-device1..3`); on a plain layer split it is inert, and setup no longer writes it there (#1447).

Measured on 2x RTX 3090 (sm_86, 250 W limits; GPU0 PCIe 4.0 x16, GPU1 x4, no NVLink; Ryzen 7 9800X3D, 32 GB RAM),
Qwen3.8-Flash-Next GSQ-RCO IQ3_S, `--resident-experts`, KV int8, `--spec 4 --mtp`, `"layer_split": "29"` for the
`--pipeline-windows` rows (without `--remote-expert-opt`); the others are setup's config (`auto`). Decode is the mean of
runs 2-6 of 1,500-token coding replies at temperature 0.6, prefill one cold 19.9K-token prompt; one run per arm unless
noted (reported by adambenhassen, #1447; not repeated on our boxes):

| Arm | Decode tok/s | Prefill tok/s |
|---|---:|---:|
| setup defaults (two runs) | 137.0 / 136.9 | 1798 |
| `--adapt-async 1` (two runs) | 145.2 / 143.7 | 1800 / 1797 |
| `--adapt-async 1`, no `--remote-expert-opt` | 145.0 | 1807 |
| `STRATA_ADAPT_LAG=2` | 142.8 | 1800 |
| `STRATA_EXCHANGE_ROTATE=1` | 139.6 | 1800 |
| `STRATA_PF_FUSED=1` | 137.2 | 1913 |
| `--pipeline-windows 2` | 136.7 | 2097 |
| `STRATA_SPEC_COUPLED=1` | 137.6 | 1804 |
| `STRATA_SPEC_PROB=1` | 137.2 | 1788 |
| `STRATA_SPEC_COUPLED=1` + `STRATA_SPEC_GUMBEL=1` | 136.5 | 1805 |
| async + lag 2 + rotate + pf_fused | 145.0 | 1829 |
| pw2 + lag 2 + rotate + pf_fused | 135.9 | 2296 |
| pw2 + async + lag 2 + rotate + pf_fused | 142.0 | 2302 |

The last row keeps most of the asynchronous tier's decode gain and the pipeline's +28% prefill; no stalls in any arm.
These are opt-in settings on one rig, not defaults.

## Several conversations at once

With a layer split, `--batch N --batch-groups G --trim-stage-weights` decodes several conversations together and
pipelines them through the cards: see [BATCHING.md](BATCHING.md).
