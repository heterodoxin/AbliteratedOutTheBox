# Persona 05 — local AI tinkerer, independent round 1

## Perspective and goals

I spend much of my day changing local inference configurations. I understand quantization, context, prompt batching and cache formats. I want to compare exact variants quickly, understand which configuration produced a number, keep an experiment reversible, and reach technical parameters without introductory prose. I still need plain labels, understandable failures and trustworthy evidence; familiarity with local AI does not make an ambiguous state acceptable.

Review only. No application/source changes, model runs, downloads, servers or browser launches. I did not read other persona reports. Production and native integration remain deferred.

## Evidence and current friction

Read `docs/engine-models-onboarding-2026-10-05.md`, relevant focus/density/disclosure/keyboard guidance in `docs/interaction-guidelines.md`, `Tuning.tsx`, `Models.tsx`, `Setup.tsx`, `EngineExperience.tsx`, `ModelCatalog.tsx` and `experience.ts`. Visually inspected `01-engine-light.png`, `07-models-light.png`, `09-onboarding-profile.png` and `08-variant-comparison.png` in `docs/evidence/instrument-redesign-2026-10-05/`. Screenshots and source take precedence over historical claims.

Observed:

- Engine has a two-line promotional heading, introductory reassurance, two repeated gesture captions, long dial hints, a three-stage duration path, memory prose and a savebar explanation. The exact inputs and state labels are useful; the surrounding prose competes with them.
- Models repeats one exact variant across Balanced and Performance, with full identity, facts, purpose, tradeoff, repeated-winner sentence and disclosure in each card. The Agentic empty card occupies the same large footprint.
- Comparison repeats the complete performance panel three times, including its heading, memory legend and identical unified-memory explanation. Exact family/version names and quantizations are visible and must survive a reduction.
- Expert is accessible from the mode switch; Raw is a second switch within Expert. Parameter rows already retain technical keys, scope and individual Revert controls. Raw's editable draft is described as “Saved configuration only,” which is unclear beside the shared draft behavior.
- Profile onboarding repeats the selected profile's purpose inside its recommendation after the same purpose appeared in the profile choice. Tune already hides adjustment controls initially. Prepare and Ready behavior is supported by source; their current appearance was not visually inspected.
- `experience.ts` accepts speed evidence only for matching hardware/backend/configuration/workload/cold-prompt condition and valid ranges. Demo curves are separate. Missing data remains unavailable. Memory is an approximation with baseline runtime and no numeric batch-workspace or partial-offload prediction. Recommendation confidence is provisional unless the required evidence/review conditions hold.

Hypotheses, not measured findings: shorter cards would improve scan speed; remembering Expert mode would save repeat navigation; an aligned comparison table would make quantization tradeoffs easier to inspect. This review does not establish timing, comprehension or native performance gains.

## Proposed changes

Priority: P1 = must address in the next design revision; P2 = useful subsequent improvement. Every proposal keeps exact variant identities and distinguishes illustrative, estimated, measured and unavailable evidence.

### 1. Engine Beginner: remove the second introduction — P1

**Current:** “TUNE YOUR ENGINE,” “A little adjustment. A clearer tradeoff.” and “Explore freely. Changes stay in your draft.” sit above the dials; top-right says “Draft retained while you navigate.”

**Replacement/interaction:** Drop the marketing heading and eyebrow. Keep the screen title, exact loaded variant subtitle, mode switch and a single status: “Unsaved draft” / “Saved for next load” / “Currently applied” / “Saved defaults · Demo.” Show “Draft retained” when relevant beside status rather than as another paragraph. Keep the full “Save for next load,” “Apply now” and “Discard changes” labels.

**Disclosure:** A keyboard-accessible “Configuration status” disclosure explains draft retention and separates saved settings from the loaded model's applied configuration. Show the active operation (“Saving…” / “Applying…”) in the same status region without prematurely claiming it is applied.

**Benefit/risk:** Controls move into the initial scan; less reassurance repeats the same fact. Risk: “Unsaved draft” could sound temporary, so retention remains directly accessible and discard remains explicit.

### 2. Engine Beginner dials: terse hints, precise reset — P2

**Current:** Each dial repeats “Drag up to increase · arrow keys to adjust”; context and batch have multi-sentence explanatory hints. Batch has “Reset” plus “Automatic · recommended 2,048.”

**Replacement/interaction:** Keep exact numeric entry, detents, saved marker and keyboard controls. Use “Context · tokens” with “More context uses more cache memory.” Use “Prompt batch · tokens” with “Larger batches may need more memory.” Offer one “Dial help” affordance for gestures. Batch reset should say “Use Automatic” and show “Automatic (2,048)” when active; context reset can say “Recommended.”

**Disclosure:** “Dial help” explains keyboard/direct-entry behavior. Per-control “Details” contains the longer explanation, supported values and the fact that bigger batches are not always faster.

**Benefit/risk:** Retains precision while reducing repeated instruction. Risk: shared help may be less discoverable; retain accessible descriptions and never move an input error into disclosure.

### 3. Engine Expert and Raw: compress guidance, retain technical scope — P1

**Current:** Expert displays an adapter integration paragraph, full `p.effect` on every row, “recommended …,” scope, technical key and Revert. Raw says “Configuration JSON,” “Edit values, validate, then save,” and “Saved configuration only. Extra JSON keys remain preserved…”

**Replacement/interaction:** Keep Parameters / Raw config, technical keys, scope, recommendation values, search, Changed only and Revert. Collapse row effects behind a row “Details” control; leave validation messages inline. Use a compact “Adapter: llama.cpp” disclosure. Raw title becomes “Configuration draft”; caption “Edits share the Engine draft.” Keep Import/Export/Reload saved; label export “Export draft” if that matches its actual payload. Remember the chosen mode/view after navigation without changing values.

**Disclosure:** Adapter details hold native-integration context. Raw “Schema and scope” says extra keys are preserved and unsupported keys are not applied. Revert continues to restore saved values; recommended reset remains a distinct action.

**Benefit/risk:** Makes Expert a working surface and clarifies Raw's state. Risk: removing effect text makes unfamiliar parameters harder; row disclosures must be keyboard accessible and stable, with scope always visible.

### 4. Engine preview: two rates, memory and evidence first — P2

**Current:** “EXPECTED EXPERIENCE,” “See the tradeoff,” source pill, two speeds, Read prompt / Start answering / Generate path, memory block, delta sentence and “Estimate details.”

**Replacement/interaction:** Title “Performance preview”; retain source badge, workload, reading/answer rates with tok/s, memory requirement/fit and “Saved → Draft: +0.09 GB estimated.” Move the three-stage durations behind “Timing details,” with unknown time explicitly shown there. Keep source labels attached to metrics even if the panel is reused elsewhere.

**Disclosure:** Rename “Estimate details” to “Evidence & configuration.” Include full model/version/quantization identity, hardware, backend, context, batch, cache, placement, CPU threads, workload, cache condition, source/date/attribution and available evidence identity. Preserve the approximation exclusions. Do not imply driver/build identity exists when the adapter does not supply it.

**Benefit/risk:** Main surface shows decision facts without repeating derived values. Risk: rates alone can hide task duration; timing is one deliberate disclosure away, and no total-time or first-token claim is introduced.

### 5. Models recommendation cards: remove duplicated narrative — P1

**Current:** Cards include profile purpose, model family/version, friendly name, quantization, hero rate, other rate, memory/download, fit/context, tradeoff sentence, duplicate-winner sentence and “Why this choice?”

**Replacement/interaction:** Keep profile name, confidence/source badge, complete family/version plus quantization, both labeled rate ranges, memory/fit, download size and explicit action. Remove the friendly alias when it adds no distinguishing information. Use one line of profile purpose in the section's accessible help. If two profiles choose the same exact variant, show one card with “Balanced · Performance” profile chips and one action; preserve separate rationale entries. Agentic becomes a compact “No approved fit at 32K” notice with “Review context” action, rather than an empty full-height card.

**Disclosure:** “Why this choice?” contains purpose/tradeoff, profile-specific ranking rule, approval/review status and speed provenance. Demo choice stays visible on every result; no fitting Agentic candidate must never be relabeled as an approved alternative.

**Benefit/risk:** Substantially reduces text and duplicate downloads/actions. Risk: merging winners can blur distinct priorities; both profile labels and separate rationales must remain inspectable.

### 6. Model library: quick exact-variant expansion — P2

**Current:** Family → Version → preferred quantization, “Show all … quantizations,” repeated family purpose/capabilities and the catalog footer warning “Smaller files do not guarantee faster or better answers.” Discover is the initial tab.

**Replacement/interaction:** Keep the grouping and exact quantization labels. Add “Expand quantizations” to the library toolbar for this session, and remember Installed/Discover choice. Condense family subtitle to capability chips. Show “N variants · 32K context” plus an accessible configuration disclosure. Keep compare checkboxes and explicit incompatible-variant toggle.

**Disclosure:** “Catalog details” holds the file-size caveat and full comparison configuration. Variant details hold review/capability evidence. Installed variants remain available even when their current setup does not fit.

**Benefit/risk:** Experts can compare quantizations without repeatedly opening version controls. Risk: expansion creates too many rows; retain paging and reversible collapse, and avoid forcing all families open.

### 7. Comparison: aligned facts, shared assumptions — P2

**Current:** Three full preview panels repeat “EXPECTED EXPERIENCE / Performance,” timing path, source badge, memory legend and pool explanation.

**Replacement/interaction:** Use aligned desktop rows: exact variant, reading tok/s, answer tok/s, estimated working memory, fit/headroom or over-budget amount, download and action. One shared configuration bar shows workload, context, batch, cache, placement and threads. Keep source/confidence for each variant. Preserve stacked labeled rows on narrow screens and the three-variant limit.

**Disclosure:** Per-variant “Evidence” shows provenance and configuration identity; shared “Memory method” explains pool/baseline/exclusions. Optional “Timing” adds derived prompt/output duration rows with unknown scheduling/reasoning shown explicitly.

**Benefit/risk:** Direct comparison occupies fewer lines and separates actual differences from common assumptions. Risk: shared assumptions can imply all evidence matches; per-variant unavailable/mismatch states and source labels remain visible.

### 8. Onboarding Computer: keep detection, shorten verdict — P2

**Current:** Hardware verdict headline/detail, capacity illustration, available snapshot, Processor/Accelerator/Memory/Available storage, “Computer details,” retry/browse controls.

**Replacement/interaction:** Keep this step and precise processor/accelerator/memory/storage facts. Title “Your computer”; one status line “Hardware detected” or “Hardware unavailable · fit unknown.” Keep reported available memory explicitly marked “snapshot,” never relabel it as the model budget. Keep “Check again” and “Browse with fit unknown.”

**Disclosure:** Computer details holds fit caveats and detailed hardware checks; actionable detection failures stay outside it.

**Benefit/risk:** Less narrative before useful hardware facts. Risk: broad status obscures limitations; accelerator support failures and fit unknown must remain visible.

### 9. Onboarding Profile & model: selection is a draft choice — P2

**Current:** “What matters to you?”, three purpose paragraphs, repeated selected-profile purpose/tradeoff in the recommendation, “Compare models,” “Use this model.”

**Replacement/interaction:** Title “Choose a model priority.” Short purposes: “Tools & long prompts,” “Everyday use,” “Faster output.” Reuse the compact recommendation from proposal 5 with exact identity. Footer “Continue with this model” plus “Compare exact variants”; small status “Selection only · loads in Prepare.” Keep changing profile reversible and capability approval meaningful.

**Disclosure:** Profile help and “Why this choice?” contain original detail and provisional/demo rationale.

**Benefit/risk:** Makes the consequence of Continue clear and removes repeated explanations. Risk: “Tools & long prompts” can sound like capability approval; selected candidate's approval status remains explicit.

### 10. Onboarding Tune: keep optional adjustment, show full setup — P2

**Current:** “A good place to start,” exact model/quantization, context/batch/setup summary, MemoryBudget, “Adjust controls,” download/load action.

**Replacement/interaction:** Title “Review setup.” Keep collapsed adjustments, exact identity, context and batch; mark “Automatic (2,048)” distinctly from manual 2,048. Add a compact cache/placement/threads line because those are also saved in preparation. Keep “Download and load” / “Load model” / “Simulate download and load” explicit. Surface retained-setting errors alongside the relevant control; never silently replace values after a variant change.

**Disclosure:** “Execution details” shows all saved values and scope; expandable adjustment retains current dials and preview evidence.

**Benefit/risk:** Expert sees what will actually be saved without opening every control. Risk: added facts increase density; use one concise settings line instead of another paragraph.

### 11. Onboarding Prepare: literal stage title and scoped recovery — P2

**Current:** “Bringing your model online,” stage list, progress, “Preparation details,” Cancel download and Retry loading/download; details say interrupted downloads restart in this host.

**Replacement/interaction:** Title “Preparing [exact model]”; retain quantization/files, completed/current stages, progress and cancel/retry. A load failure reads “Loading failed · files retained,” with “Retry loading.” Keep check naming literal: “Check downloaded file size” must not become “Verified” or “Integrity verified.” During download show “Cancel download”; before cancel, make restart behavior accessible beside that action.

**Disclosure:** Preparation details contains host restart limitation, retained settings/files and operation error detail; the actionable error remains inline.

**Benefit/risk:** Recovery is clear with fewer ambient sentences. Risk: shortening verification wording can overclaim security/integrity; literal stage names are required.

### 12. Onboarding Ready: confirm loaded configuration, then chat — P2

**Current:** “Your next idea starts here,” model-ready pill, family, quantization/profile/context, full MemoryBudget, “Preview speed test · Demo,” “Start chatting.”

**Replacement/interaction:** Title “Model ready” only after successful loading; preserve “Demo model ready” where appropriate. Show exact family/version/quantization and loaded execution summary including batch/cache/placement/threads. Keep “Start chatting” primary; move full memory breakdown behind “Loaded setup.” Put “Preview speed test · Demo” in an optional Tests disclosure with the visible Demo label retained.

**Disclosure:** Loaded setup includes memory/evidence details; Tests opens the existing explicitly illustrative flow. Ready must not claim saved/draft values are applied unless they correspond to the successful load.

**Benefit/risk:** Clear endpoint without another large repeated panel. Risk: hiding the test reduces discovery, but it is optional and produces no measurement; error/readiness gates remain visible.

## Vote

**Three must-do:** 1 (remove Engine's second introduction while preserving state), 3 (make Expert/Raw concise and unambiguous), 5 (compact recommendation cards and consolidate identical winners). These address the text complaint directly without sacrificing precision or evidence.

**Two opposed proposals:**

1. Oppose removing quantization/version/backend/configuration details or replacing exact identities with friendly aliases alone. “Qwen3” is insufficient for selection, evidence matching or a reversible experiment. Identity should remain in the main row/card; longer provenance can disclose.
2. Oppose one “Optimize”/“Save” action that silently applies or restarts a model, or a profile selection that changes sampling. Preserve draft, saved and applied state, next-request versus next-load scope, explicit Apply now, per-field revert and recommendation selection as separate operations.

No requested change depends on invented benchmark numbers or deferred native integration. The vote is a design judgment; the scan/comprehension benefits remain hypotheses until users try the revised surfaces.
