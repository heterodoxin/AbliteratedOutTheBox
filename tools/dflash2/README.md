# DFlash2 acceptance experiment (research tooling, not shipped)

Measures how many tokens per round the incoai DFlash2 drafter for Qwen3.8-Flash-Next would get on OUR quantized target, offline, from the hidden states
the engine dumps. Results and method: phaseA-tests/notes-dflash2.md. All scripts keep the paths of the run (C:\dflash2-work on the owner PC, C:\strata-test\dflash2
on the test box): edit them before reuse.

Engine side (both off unless the variable is set):

- `STRATA_TAP_FILE=<file>` (+ `STRATA_TAP_LAYERS`, default `5,15,25,35,45`): `Verifier::record_window` copies each verified row's completed layer output (the mean of
  the 4 hyper-connection streams, the quantity the drafter was trained on) and `Verifier::run` appends `pos, token, taps` records.
- `STRATA_SPEC_T_CYCLE=1,2,...,8` (CLI bench only): the verify window size cycles; the report line `window cost (ms/round)` gives the cost of each width.

Scripts:

- `mk_prompts.py` builds chat-template prompt ids with the server's own tokenizer and template.
- `orch.py <prompt> <max_new> [gen,genf,genfm,tap]` drives `run_one.ps1` on the test box: free generation (ids, MTP statistics), the same text with MTP only, and the
  teacher-forced tap dump with oracle drafts (`--spec 8 --spec-oracle --spec-follow`, a 2-token prompt: a native pack never starts the speculative loop from a 1-token prompt).
- `dflash2_ref.py` is a plain-torch port of z-lab's `DFlash2DraftModel` (checked against it by `test_port.py`), with the target's embedding and head dequantized from the GGUF.
- `run_eval.py` (walk, a few block sizes), `run_all2.py` (every position, resumable: per-anchor correctness and the selector's confidences), `analyze2.py` and
  `final_tables.py` (policies, speed estimates from the measured window costs).
