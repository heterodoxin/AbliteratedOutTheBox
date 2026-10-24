# Persona 07 — Casual research and ordinary work

Independent round 1, 5 October 2026. Review only; no other persona reports were consulted.

## Goals and evidence

I want to summarize documents, draft ordinary work, and explore questions without learning hardware tuning. I want a useful starting choice and a clear path back to my work. I need to know what is actually established about a model; speed is not evidence that its research answers are accurate or reliable.

Evidence inspected: current `Tuning.tsx`, `Models.tsx`, `Setup.tsx`, `EngineExperience.tsx`, `ModelCatalog.tsx`, and `experience.ts`; implementation report and interaction guidelines; captures `01-engine-light.png`, `07-models-light.png`, `09-onboarding-profile.png`, and `08-variant-comparison.png`. Current source supplies details absent from the captures, including all five setup steps and Expert mode. This is a simulated persona review, not a usability study or a benchmark. Likely confusion and expected benefits below are hypotheses.

Observed: Beginner presents two equally prominent dials, multiple explanatory paragraphs, throughput, three timing stages, and a memory breakdown. Model cards repeat profile purpose and tradeoff text, and two cards repeat the same exact choice. Comparison repeats an entire performance panel three times. Onboarding repeats the selected profile purpose inside its recommendation. These are visible density problems; task failure or abandonment has not been measured.

Useful existing behavior to preserve: Balanced is the new-setup default; optional tuning is disclosed in setup; exact variants remain accessible; fit and evidence have explicit unknown states; selection does not silently switch the running model; preparation preserves completed files and errors; readiness gates entry into Chat. No recommendation here depends on native hookup or invented quality measurements. Keep current body size and allow wrapping rather than shrinking text.

## Concrete proposals

### 1. Engine Beginner: make context the main decision — P0, must-do

**Observed control/copy:** `Context` and `Prompt batch` are peer dials, with `Exact value`, drag instructions, and long hints beneath each. Batch also has `Automatic · recommended 2,048`.

**Change:** Keep context as the visible dial, labelled `Context · text the model can consider`, with the shorter hint `More context holds longer documents and uses more memory.` Put batch under `Advanced controls`, showing `Prompt batch: Automatic (2,048)` as its collapsed summary. Retain the explicit automatic toggle, batch dial, exact numeric inputs, recommendation reset, saved marker, and keyboard behavior inside the disclosure. Preserve a visible `Custom batch: 1,024` summary when applicable.

**Details destination:** `Context details` holds token definition, supported limits, and numeric entry; `Advanced controls → Prompt batch` holds `Larger batches use more working memory and may not be faster.`

**Benefit/risk:** Likely makes the document-related setting recognizable and removes an unnecessary first decision. Risk: custom batch changes become easier to overlook; visible custom summary and shared draft state mitigate it.

### 2. Engine expected experience: lead with fit, keep speed evidence explicit — P0, must-do

**Observed copy:** `EXPECTED EXPERIENCE`, `See the tradeoff.`, two large throughput figures, `Read prompt / Start answering / Generate`, memory legend, shared-memory explanation, and saved-to-draft sentence.

**Change:** Title `Estimated fit and speed`. First show `Comfortable fit · estimated`, `1.9 GB required`, and `Draft uses +0.09 GB estimated memory` when changed. Keep separate reading and answer ranges with their units and `Demo estimate`, `Measured`, or unavailable status directly adjacent, at normal metric size. Replace repeated source text with one scoped source label only when both values truly share it. Move the timing path and memory segment legend behind `Details`; if timing is expanded, retain `Start answering: Not estimated` and `Durations exclude reasoning and scheduling.`

**Details destination:** Existing `Estimate details`, expanded to include the timing path, allocation explanation, workload/configuration identity, and estimation limitations. Keep workload selector visible.

**Benefit/risk:** Likely helps answer “will this fit?” without making demo throughput the screen's hero. Risk: hidden timing limitations could imply total task time; the expanded timing path must retain its qualifier, and the collapsed surface must never label speed as total task duration.

### 3. Engine Expert: shorten repeated explanations, preserve applicability — P1

**Observed copy/control:** Each parameter row shows label, technical key, full effect, applicability, recommendation, input, and `Revert`; the view also has a build-adapter paragraph.

**Change:** Keep parameter label, technical key, input, recommendation, `Next model load` / `Next request` / `Preserved only`, and `Revert` visible. Put full `p.effect` text behind per-row `Details`. Replace the build paragraph with `Adapter: llama.cpp · Details`.

**Details destination:** Row Details contains effect/support information; adapter Details retains native integration dependency and the distinction between preserved keys and supported options. Raw config retains its scope explanation and import/export controls.

**Benefit/risk:** Likely makes occasional troubleshooting scan-friendly. Risk: users may miss consequences; applicability and validation errors must stay visible. Do not collapse warnings merely because they are lengthy.

### 4. Models recommendations: one explanation and one compact identity — P0, must-do

**Observed copy:** Profile purpose appears before the model; the card repeats `A practical starting point for everyday work.` or the performance/tool tradeoff later. Exact family, name, quantization, speed, fit, memory, download, duplicate-winner sentence, and disclosure all occupy separate blocks.

**Change:** Keep profile title, one short purpose (`Everyday work`, `Tools and long prompts`, `Answer speed; capability tradeoffs`), full exact variant identity, `Comfortable fit · estimated · 32K context`, memory, download, compact reading/answer ranges with source, and one action. Remove the second purpose paragraph for Balanced. For Performance/Agentic keep one visible limitation: `Capabilities need evaluation` / `Tool reliability needs evaluation`. Use `Same variant as Balanced · one download` for repeated winners instead of repeating a generic sentence. Keep unavailable Agentic state with `No approved fit yet` and `Try Balanced or shorter context`.

**Details destination:** Rename `Why this choice?` to `Choice details`; retain full selection explanation, review evidence, speed provenance, quantization tradeoff, and demo approval caveat there.

**Benefit/risk:** Likely reduces reading while preserving the purchase/download decision. Risk: `Everyday work` could sound like verified research suitability; Details must explain the catalog-based starting choice, and the card must not acquire an accuracy or quality badge without evidence.

### 5. Models library: keep exact hierarchy; reduce repeated row metrics — P1

**Observed control/copy:** Family → Version → Quantization, reversible `Show all … quantizations`, filters, and rows carrying download, working memory, paired speeds, source and fit.

**Change:** Keep this hierarchy and exact search. Default variant row shows exact quantization, state, `Fit · estimated/unknown`, download size, Compare, and action. Add `Variant details` for working-memory breakdown and speed ranges/source. Shorten `Show all 4 quantizations` to `All 4 variants`; keep the exact names on expansion. Preserve a clear filter for incompatible variants and show its active state.

**Details destination:** Per-variant disclosure within its version, with the complete exact label, capability evidence and configuration-specific estimates. Keep `Exceeds memory budget` and `Context not supported` visible rather than hiding them there.

**Benefit/risk:** Likely makes browsing manageable without removing specialist choice. Risk: extra disclosure slows metric-based browsing; comparison remains the direct route for that task.

### 6. Comparison: aligned decision rows instead of three full panels — P1

**Observed copy/control:** `Compare exact variants` repeats `EXPECTED EXPERIENCE`, `Performance`, all timing stages and memory allocation legends in three columns.

**Change:** Keep exact model/version/quantization headers and actions. Show aligned rows for Fit, Required memory, Download, Reading speed, Answer speed, and Evidence status. Shared header: `Same workload and settings · speed does not assess answer quality.` Preserve separate unavailable values and over-budget numbers. Use a single workload selector and `Comparison details` rather than three repeated panel headings.

**Details destination:** Shared settings/limitations disclosure plus per-variant provenance and memory allocation disclosures. On narrow screens use identical labelled rows per variant; preserve all three choices and removal controls.

**Benefit/risk:** Likely makes real differences easier to see. Risk: common Evidence status could mask different sources, so retain a source cell per variant and never use a shared demo label for mixed sources.

### 7. Setup 1 Computer: keep the check; make its conclusion concise — P1

**Observed copy/control:** Capacity illustration plus processor, accelerator, memory, storage, checks disclosure, `Check again` and `Find my models`.

**Keep/change:** Keep the check, memory/storage available values with `current snapshot`, retry, and `Browse with fit unknown`. Change the primary detected-state action to `Continue to models`. Keep one verdict sentence; move processor/accelerator rows into `Computer details`, retaining the compact memory capacity display and available storage in the initial view.

**Details destination:** Existing Computer details includes all detected specifications, checks, and `Fit is approximate; other apps and configuration can change it.`

**Benefit/risk:** Likely avoids making me assess CPU and accelerator specs. Risk: hiding detector mistakes; keep device identity in the details summary when useful and retain easy retry.

### 8. Setup 2 Profile & model: one selected recommendation, fewer repeated choices — P1

**Observed copy/control:** Three large profile cards then a large recommendation restating the selected profile and purpose; `Compare models` and `Use this model`.

**Keep/change:** Keep Balanced default and all three accessible choices. Replace large profile cards with a labelled choice group `Priority` using `Balanced · everyday work`, `Agentic · tools`, `Performance · answer speed`. Show the selected profile's single full purpose sentence directly below. The recommendation uses proposal 4 without repeating that purpose. Keep `Compare models` and `Use this model`, including unavailable-candidate blocking.

**Details destination:** `Profile details` explains tradeoffs and provisional/demo evidence; `Choice details` retains exact model selection reasoning.

**Benefit/risk:** Likely shortens a screen that asks for one decision. Risk: novice users may not understand Agentic; visible `tools` plus its selected explanation must preserve the long-prompt meaning, not rename it to “Research.”

### 9. Setup 3 Tune: keep optional controls and strengthen the default route — P1

**Observed copy/control:** `A good place to start.`, context/batch/setup summary, full MemoryBudget, `Adjust controls`, then download/load action.

**Keep/change:** Keep optional adjustments and retained-value validation. Title `Your starting settings`. Show `32K context · Automatic batch` and `Comfortable fit · estimated · 1.9 GB required` by default; for custom values show their exact values. Change `Adjust controls` to `Adjust settings`. Keep the explicit `Download and load` / `Load model` / `Simulate download and load` action, with download size next to it.

**Details destination:** `Settings details` holds memory segments and configuration identity; Adjust settings exposes the shared accessible controls and preview. Blocking context/memory messages remain inline and actionable.

**Benefit/risk:** Likely lets me proceed with a reviewed starting point. Risk: concealing limitations behind a cheerful default; estimated/unknown/too-large fit wording must stay prominent.

### 10. Setup 4 Prepare: keep honest stages and local recovery — P1

**Observed copy/control:** `Bringing your model online.`, file identity/size, stage list, Bar label plus a repeated stage/percentage sentence, Preparation details, cancellation/retry.

**Keep/change:** Title `Preparing your model` (online could suggest a network requirement after download). Keep all actual stages, exact variant, download size, cancellation, progress, and specific retry action. Show the active stage once, then `42% · about 3 min left` when host values exist; transfer rate moves to Details. Keep error text near its retry control once instead of repeating it in both disclosure and the trailing global error for the same failure.

**Details destination:** Existing Preparation details retains transfer rate, verification scope, and restart behavior. At Cancel download, show `Partial downloads restart from the beginning` before committing the cancellation when applicable.

**Benefit/risk:** Likely gives a clearer sense of waiting and recovery. Risk: over-simplifying “verification”; keep `Check downloaded file size` rather than claiming integrity verification the host does not perform.

### 11. Setup 5 Ready: make starting work the obvious finish — P1

**Observed copy/control:** `Your next idea starts here.`, ready pill, exact model/profile/context, full MemoryBudget, `Preview speed test · Demo`, `Start chatting`.

**Keep/change:** Title `Ready to chat` only when load succeeded; keep the not-ready route. Keep ready/demo status, exact variant, profile/context summary, and Start chatting. Replace the full memory panel with the short estimated fit line. Put the demo preview in `Setup details` as `View speed-test demo`.

**Details destination:** Setup details holds memory explanation and optional demo entry, with `No benchmark runs` immediately attached. Entry into Chat continues to require loading; preview never becomes a gate.

**Benefit/risk:** Likely ends setup cleanly. Risk: some users want testing before entry; disclosure preserves it without competing with the finish action.

### 12. Shared evidence language: make unavailable scope understandable — P1

**Observed copy:** `No comparable evidence`, `Unavailable`, and `No comparable speed evidence for this configuration.` are correct but technical; the current demo fixture can present large numeric ranges.

**Change:** In speed-only contexts use `Speed not established for this setup` and in demos retain `Demo estimate · illustrative`. Add one shared Details sentence: `These figures describe speed and memory, not research accuracy or answer quality.` Do not replace absent evidence with ratings or inferred reliability. Keep actual review text separate from speed evidence.

**Details destination:** Engine Estimate details, recommendation Choice details, and comparison evidence disclosure use the same scope explanation with exact source, date, workload and configuration identity retained.

**Benefit/risk:** Likely prevents a casual researcher reading speed as a quality recommendation. Risk: boilerplate grows again; use one scoped sentence per surface and keep the source/fit qualifiers next to the applicable facts.

## Ballot for the next round

Must-do: **1** (Beginner hierarchy), **2** (fit/evidence hierarchy), **4** (compact recommendation cards). These address the visible density complaint without requiring new evidence or native work.

Oppose **removing exact quantizations/versions and comparison access** to make the app “simple.” A casual default can coexist with exact access; variant identity matters for download and reproduction.

Oppose **a “Best for research” or reliability/accuracy score inferred from Balanced rank, fit, speed, model size, or demo estimates**. The inspected implementation supplies no verified evidence for that claim. Use a qualified starting choice and allow later real capability evidence to justify a narrower claim.

No implementation, browser/server launch, download, or model run was performed. The recommended outcomes remain usability hypotheses to validate after the review process.
