# abliterate

`llama-abliterate` attempts to reduce refusals and evasive non-compliance without modifying model weights. It reads three directions from the current prompt and applies them through `llama_adapter_abliterate` at the shared `build_cvec` layer-output hook. It needs no fit, sidecar, corpus calibration, or second model. The correction is zero when the gate stays off; benign prompts can still trigger the gate.

Removing the refusal direction alone often leaves the model evading rather than complying (off topic, hedged, or degenerate), which is what issue #1 reported on an MoE model. To counter that, when the gate fires the tool also clamps a delivery direction up for the first few generated tokens, pushing the response from evasion into delivery, and applies a repetition penalty to keep the result coherent. The delivery clamp and the penalty are active only on a fired prompt, so benign output stays identical to the base model.

The default uses disposable anchor prefills: five comply, eight refuse, four clean, four disclaimer, four delivery, and four deflection openings. A clean prompt prefill and up to 24 greedy look-ahead tokens supply the gate decision. A final prompt prefill starts generation, with edits enabled only if the gate fired. The answer receives no forced opening. Both prompt processing and decoding have additional cost.

Decoding is greedy, and the chat template is rendered with thinking disabled. Base-model comparisons must match those settings. Refusal removal, delivery, coherence, and benign preservation need direct response checks on the intended model and prompts. Very low bit quantizations of large models are more sensitive to the delivery clamp and may show residual refusal text; q6 or q8 give the cleanest delivery.

``` bash
> ./build/bin/llama-abliterate -m qwen3-8b.gguf -ngl 99 -n 256 -p "..."
[abliterate] gate fired: editing this response
```

Flags accept finite numbers:

- `--ff-threshold <f>` (default 0.70) sets the gate threshold; lower values allow more edits.
- `--edit-strength <f>` (default 2.0) places the refusal reference this many comply-opening population standard deviations below the comply projection mean.
- `--disc-strength <f>` (default 0.4) places the disclaimer reference this fraction of the clean-to-disclaimer gap below the clean projection mean; any negative value disables the disclaimer edit and skips its eight anchor prefills.
- `--deliver-strength <f>` (default 1.0) scales the delivery clamp; higher pushes harder into delivery but can degrade or empty the output on sensitive quantizations; zero disables it and skips its eight anchor prefills.
- `--deliver-window <n>` (default 8) is how many early generated tokens the delivery clamp is applied to before it releases.
- `--rep-penalty <f>` (default 1.15) is the repetition penalty on the edited generation; 1 disables it.

Stronger edits can cause repetition or incoherent text. The remaining model, context, device, and prompt flags use the standard `common` arguments. Sampling flags do not change this tool's greedy decoding.
