# Optional checkpoint near the prompt tail

`--prompt-cache-tail` saves at most one extra mid-prompt checkpoint near the end
of a request, at a boundary the batched prefill already reaches. It is off by
default and applies to single-GPU serve with positive `--prompt-cache` and
`--prompt-cache-every` values.

A branch can share most of a prompt but diverge before its final cached state.
If the periodic checkpoint before that branch is far away, the engine has to
read the gap again. The extra checkpoint can reduce that gap. It uses the
existing bounded checkpoint cache; it does not add another unbounded history.

The tail is the first chunk boundary within one chunk of the prompt's end, where the chunk is the size that chunk
really had (`--prefill auto` chunks differ from `--prefill N`). The tail checkpoint is evicted first (only one stays
alive, the root is never dropped), so it never pushes out a periodic checkpoint when `--prompt-cache` is full.

The option does not split prefill chunks or alter the periodic checkpoint
schedule. It selects the first completed batched chunk within one configured
chunk of the prompt's end, excluding the short final suffix controlled by
`--short-read`. If a periodic checkpoint already falls there, the same state is
saved once. Reread diagnostics and layer splits keep their existing behavior.

For example, with a 6144-token chunk and a 16384-token checkpoint interval,
the first periodic checkpoint is at 18432 and the next would be at 36864.
A branch sharing 32769 tokens cannot use the latter. A near-tail checkpoint
at 30720 can reuse more of that prefix,
without changing how the original prompt was computed. The benefit depends
on the request and the branch position; it is not a universal speed increase.

This is a separate conservative alternative to the Strata-2080Ti Daily's
checkpoint-grid patch. The grid changes chunk boundaries to reach exact
checkpoint intervals and failed its local paired numerical gate. This option
does not claim that the original grid patch has been fixed or validated.
