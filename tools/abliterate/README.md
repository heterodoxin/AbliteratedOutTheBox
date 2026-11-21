# abliterate

`llama-abliterate` attempts to reduce refusals and evasive non-compliance without modifying model weights. It reads three directions from the current prompt and applies them through `llama_adapter_abliterate` at the shared `build_cvec` layer-output hook. It needs no fit, sidecar, corpus calibration, or second model. The correction is zero when the gate stays off, and an unfired prompt is prefilled exactly as the base model would prefill it; benign prompts can still trigger the gate.

Removing the refusal direction alone often leaves the model evading rather than complying (off topic, joking, hedged, or degenerate), which is what issue #1 reported on an MoE model. Three things in the current version target that:

- The comply and refuse openings that define the refusal direction are written as twins in the same formal register ("The procedure is as follows." against "The procedure is not something I will share."). Upbeat comply openings against solemn refusals also measure tone, and clamping that direction pushed a whimsy-prone Gemma MoE into jokes.
- The decision to refuse is read out at the chat-template suffix after the user text, so each suffix position is shifted toward comply from its own unedited projection, while the user text only gets the plain clamp. The last suffix step, which closes the template and starts the answer, also projects strongly onto the refusal direction, so the shift direction is made orthogonal to that step; otherwise the shift partly reopened the template, which on Gemma came out as the model reopening its empty thinking channel and reasoning back into a refusal. If the shift ends the turn within eight look-ahead tokens, it is retried at half strength down to the plain clamp, which keeps architectures that cannot absorb it, such as the hybrid qwen3_5 architecture, at the previous behavior.
- The edit releases after the first few generated tokens. Clamping the whole answer kept the model joking on the Gemma MoE; once the opening has committed, the unedited model continues the answer itself.
- The first word is chosen among the edited model's top few candidates by the same refusal readout the gate uses, taking the candidate whose next state reads least like refusal. On the Gemma MoE this removed most garbled or disclaimer-first openings; on the hybrid qwen3_5 architecture it helps less, because one committed word is often not enough there.

A delivery direction is also clamped up for the first few generated tokens, a repetition penalty applies to the edited generation, and an edited answer skips template protocol tokens such as a reopened thinking channel. All of these act only on a fired prompt.

The default uses disposable anchor prefills: five comply, five refuse, four clean, four disclaimer, four delivery, and four deflection openings. A clean prompt prefill and up to 24 greedy look-ahead tokens supply the gate decision. On a fired prompt the user text is prefilled once more to read the suffix baselines, the shifted prompt is prefilled and checked with eight look-ahead tokens, the accepted prompt is prefilled again, and each extra first-word candidate adds one prefill; a weaker retry adds one prefill and look-ahead per step. The answer receives no forced opening. Both prompt processing and decoding have additional cost.

Decoding is greedy, and the chat template is rendered with thinking disabled. Base-model comparisons must match those settings. Refusal removal, delivery, coherence, and benign preservation need direct response checks on the intended model and prompts. Some answers still open with a short disclaimer before the content, and very low bit quantizations of large models remain the hardest case.

``` bash
> ./build/bin/llama-abliterate -m qwen3-8b.gguf -ngl 99 -n 256 -p "..."
[abliterate] gate fired: editing this response
[abliterate] suffix shift 1.00
```

The second line reports the suffix strength the look-ahead check accepted; a lower value than `--suffix-strength` means the shift was backed off for this prompt.

Flags accept finite numbers:

- `--ff-threshold <f>` (default 0.70) sets the gate threshold; lower values allow more edits.
- `--edit-strength <f>` (default 2.0) places the refusal reference this many comply-opening population standard deviations below the comply projection mean.
- `--suffix-strength <f>` (default 1.0) shifts each chat-template suffix position this many comply-to-refuse gaps toward comply; zero uses the plain clamp there.
- `--clamp-window <n>` (default 8) is how many generated tokens the edit stays on before it releases; zero keeps it on for the whole answer.
- `--first-candidates <n>` (default 4) is how many top first-word candidates are scored by the refusal readout on a fired prompt; 1 keeps plain greedy decoding for the first word.
- `--disc-strength <f>` (default 0.4) places the disclaimer reference this fraction of the clean-to-disclaimer gap below the clean projection mean; any negative value disables the disclaimer edit and skips its eight anchor prefills.
- `--deliver-strength <f>` (default 1.0) scales the delivery clamp; higher pushes harder into delivery but can degrade or empty the output on sensitive quantizations; zero disables it and skips its eight anchor prefills.
- `--deliver-window <n>` (default 8) is how many early generated tokens the delivery clamp is applied to before it releases.
- `--rep-penalty <f>` (default 1.15) is the repetition penalty on the edited generation; 1 disables it.

Stronger edits can cause repetition or incoherent text. The remaining model, context, device, and prompt flags use the standard `common` arguments. Sampling flags do not change this tool's greedy decoding.
