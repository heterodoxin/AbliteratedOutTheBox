# abliterate

`llama-abliterate` runs a model so it answers flagged harmful prompts and delivers the content instead of
refusing, while leaving every other prompt untouched. It needs no fit and no sidecar: for each prompt it
reads two directions from the prompt itself and removes them at the `build_cvec` layer-output site through
the `llama_adapter_abliterate` adapter, scaled by a runtime gate scalar that is exactly zero on prompts the
gate leaves alone.

Each prompt is handled in two passes. A clean pass forces the assistant to start several comply vs refuse
openings (the refusal direction) and clean vs disclaimered openings (the disclaimer direction), then a
look-ahead over the model's own next tokens decides whether this is a prompt it would refuse. If it fires, a
second pass re-runs the prompt with both directions removed (refusal down to `--edit-strength` comply-opening
standard deviations below the comply level, disclaimer by `--disc-strength`) and generates. Decoding is greedy.

``` bash
> ./build/bin/llama-abliterate -m qwen3-8b.gguf -ngl 99 -n 256 -p "..."
[abliterate] gate fired: editing this response
```

Flags: `--ff-threshold <f>` (default 0.70) trades harmful recall against benign false positives;
`--edit-strength <f>` (default 2.0) sets how hard the refusal direction is removed, in comply-opening standard
deviations past the comply level, a scale-free bound that behaves the same across architectures;
`--disc-strength <f>` (default 0.4) sets how hard the safety-disclaimer direction is removed (higher removes
more disclaimers but can degenerate long outputs; negative disables). The remaining flags (`-m`, `-ngl`,
`-n`, `-p`, and the usual device flags) are the standard `common` arguments.
