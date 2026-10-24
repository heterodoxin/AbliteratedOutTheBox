# Local GGUF import — 14 October 2026

## User flow

Models → Version → **Use a local model…** opens a local file browser. It starts
in the configured model directory, offers Home / Up a folder, and lets users
paste a full folder or file path. It lists folders and GGUF files; no upload,
copy, remote request, or weight download is involved.

Users choose the GGUF file. For standard split GGUFs, choosing any shard resolves
the first shard and checks the full set in its directory. The review shows the
name, declared architecture, quantization, total size, context limit, and file
locations. **Use this model** saves a reference and selects it; **Load model**
is still an explicit separate action. Normal mode retains one-model semantics.

Local entries appear in the Version dropdown and under **Local models** in the
family switcher. Names can be edited during review. They do not inherit a catalog
family's identity, accuracy claims, measured speeds, or optimization approval.
Generic tuning uses their own GGUF geometry. New local selections begin at up
to 32K, limited by the file's declared context and the app's supported presets.

## Host integration

- `server/gguf.mjs` reads bounded GGUF v2/v3 little-endian metadata and tensor
  directories. It skips tokenizer array contents and tensor weights, checks
  metadata/tensor bounds and expected tensor payload extents, and rejects
  truncated files, unsupported encodings, duplicate keys, and invalid geometry.
- `server/local-models.mjs` validates text architectures, tokenizer presence,
  context, layers and attention geometry; validates split numbering/counts and
  aggregate tensor counts; and persists references atomically in
  `<dataDir>/local-models.json`. IDs come from the canonical first-file path.
- Supported import metadata: qwen2, qwen2moe, qwen3, qwen3moe, qwen3next, qwen35,
  qwen35moe, llama, gemma, gemma2, gemma3 and phi3. This is the importer's scope,
  not a claim about the installed engine build. Engine support is checked at load.
- Browse: `GET /api/local-models/browse?directory=…`; inspect:
  `POST /api/local-models/inspect {path}`; register:
  `POST /api/local-models {path,name?}`. Registration reinspects the file rather
  than trusting renderer metadata, and cannot replace an active reference whose
  file signature changed.
- The model catalog and normal loader resolve imported IDs to the exact local
  first-file path. llama.cpp handles the remaining standard shards. File size,
  modification time and change time guard stale references; a moved/changed file
  requires reconnecting through **Use a local model…**. Canonical paths dedupe.
- Default model-directory changes preserve these explicit local references.
- Removing a local model removes its reference only. Original weights and other
  folder files stay intact. Every loaded instance must be unloaded first.

The file declares its architecture; metadata does not prove model provenance or
that it is an authentic fine-tune of a named catalog model. Tensor geometry checks
are structural, not a cryptographic integrity check. Hybrid/recurrent working
memory remains approximate; actual allocation and adapter compatibility need the
native engine. LoRA adapters, projectors, image models, unsupported architectures,
big-endian GGUFs, and nonstandard shard filenames get a clear error.

## Sources

- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGUF metadata and quantization constants](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py)
- [llama.cpp split-file behavior](https://github.com/ggml-org/llama.cpp/blob/master/tools/gguf-split/gguf-split.cpp)

## Verification

Focused parser/service/API tests cover single/split files, missing and mismatched
shards, truncated tensor payloads, unknown architectures, invalid context,
malformed metadata bounds, folder filtering, persistence, path deduplication,
missing-file detection, explicit loading, active-reference protection, native
path resolution, directory changes and preserving files on removal.

Browser flow uses a disposable simulated host and tiny GGUF fixtures. No real
inference weights were imported into the live library or executed. The live
host remains on its original one-model setup. Evidence:
`docs/evidence/local-model-import-2026-10-14/`.
