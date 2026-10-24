# Engine Models and Onboarding implementation

Implemented on 5 October 2026 in the existing React application. Engine, Models and the five onboarding steps now share model recommendations, performance previews and tuning controls. This report records the implementation and its verification, and supplements the [consolidated design report](design-improvements-2026-10-05.md).

The current host remains a UI test adapter. Speed previews and simulated operations are visibly labelled Demo. Real benchmark execution, production Strata configuration parity and native engine hookup remain deferred, as requested. No real model was downloaded or run for this work.

## What changed

### Engine

- Kept the existing tuning route and renamed the visible destination Engine, including sidebar, command palette, Settings and inspector links.
- Added context and prompt batch dials with SVG detents, a saved-position marker, pointer dragging, click positioning, arrow keys, Home/End, direct numeric entry and recommendation reset.
- Kept values stable while dragging; the pointer tracks movement and the value settles on release. Unsupported Expert values remain visible without an invalid slider range.
- Added Automatic batch selection and an explanation when the adapter cannot apply batch settings.
- Kept CPU allocation, GPU placement and cache settings under More controls. Sampling has its own disclosure; choosing a recommendation profile does not alter temperature.
- Added a shared expected-experience panel with reading and answer ranges, sample workload, memory segments and saved-to-draft memory differences. Unified memory is one pool; discrete graphics and system budgets remain separate.
- Preserved the existing shared Beginner, Expert and Raw draft, validation, import/export and per-field reverts. Added a compact Expert performance disclosure.
- Kept Save for next load primary and Apply now explicit. Draft, saved and currently applied state remain distinct. Debounced validation rejects stale responses.
- Added a cancellable Preview test · Demo flow with warm-up, reading, answering and results states. It performs no benchmark and writes no measured results.

### Models

- Added the running model summary and hardware details above three profile recommendations.
- Recommendations identify the exact model and quantization, both speeds where available, memory, fit, download size, tradeoff and evidence explanation.
- Agentic selection gates on approved tool/agent capability before comparing reading throughput. Performance compares answer throughput. Balanced uses the catalog's reviewed-order field and comfortable memory fit.
- Missing evidence produces provisional choices. Missing hardware and no fitting approved candidate have explicit states. The same variant can win multiple profiles without creating duplicate downloads.
- Hardware/evidence updates request a recommendation refresh instead of moving cards beneath the pointer. Profile browsing never switches the running model.
- Added Family → Version → Quantization grouping, reversible expansion, Installed/Discover, search, capability/quantization/fit filters and an explicit incompatible-variant option.
- Added comparison of up to three exact variants, aligned on desktop and stacked on narrow screens, with a shared workload selector. A fourth selection is disabled until one is removed.
- Kept download and model-operation progress in one activity section. Model switching reviews the selected configuration and active-request consequences before applying it.
- Limited initial family rendering to 30 groups with Show more. Variant detail rows mount only when expanded.

### Onboarding

1. **Computer:** actual detected processor, accelerator, memory and storage; capacity illustration; uncertainty/details disclosure; retry and browsing with fit unknown.
2. **Profile and model:** keyboard-accessible Agentic/Balanced/Performance choices, Balanced for a new setup, one exact recommendation and the shared comparison picker.
3. **Tune:** recommended setup summary and optional shared dials. Retained adjustments are validated after a model change; invalid values are explained.
4. **Prepare:** download, supported verification, load and readiness stages reflect host state. Errors, cancellation and retry remain local to the operation. Installed files remain completed while loading.
5. **Ready:** selected model/profile/context, Start chatting and optional Demo test. Entry requires successful loading; benchmarking never blocks Chat.

Profile, context, batch and Automatic selection persist with setup progress. Finish later and browser reload resume the saved step. Preparation preserves unrelated engine and sampling settings. Backend validation now checks the chosen candidate variant rather than only the previously loaded model.

## Shared data and evidence rules

The common provider is `src/lib/experience.ts`; the controls and preview live in `src/components/EngineExperience.tsx`. `src/components/ModelCatalog.tsx` supplies recommendations, catalog browsing, comparison and selection to Models, Engine and onboarding.

| Data | Behavior |
|---|---|
| Sample workload | Short 1K, Document 4K, Long agent context 16K input tokens, each with 256 output tokens. Document is the initial sample. Options exceeding context plus output allowance are unavailable. |
| Speed evidence | Ranges carry source, workload, hardware/backend/configuration identity, cold-prompt cache condition, attribution and date. Only compatible evidence supplies real numeric speeds. |
| Demo speed | Separate illustrative fixtures. Always labelled Demo; not a measurement or hardware prediction. Never stored as benchmark evidence. |
| Missing speed | Not estimated. No zero, accuracy score, temperature prediction or invented first-token latency. |
| Duration | Reading and output durations derive from their own rates. Reasoning and scheduling remain unknown; the display does not claim total task time. |
| Memory | Catalog weights plus context/cache and a baseline runtime allowance. Estimates explain the execution budget. Partial GPU offload and batch workspace are not numerically predicted. |
| Capability review | The real test catalog has no newly invented agent approval. Demo choices are isolated; production approval and benchmark curation still need actual review. |
| Preferences | Recommendation profile is separate from sampling priority. Existing configurations, unknown raw keys and unsaved drafts remain preserved. |

The same inputs use the same provider across all screens. Browsing defaults may differ from a user's retained custom tuning; each preview exposes its exact configuration. A recommendation applies to that displayed configuration, not merely the model's filename.

## Design and interaction

Retained supplied Strata identity, GT Walsheim, Anron symbols, green accent and existing token system. Light remains the default. Reading surfaces, numbers and model rows are opaque. Floating controls use the existing restrained material tokens.

The implementation uses React, SVG and CSS without new dependencies, WebGL or continuous animation loops. Existing 100/160/240/320/460ms timing tokens remain in use. Narrow layouts stack controls, previews, recommendations and comparisons. Reduced motion disables dial travel; reduced transparency removes translucent materials; increased contrast remains supported.

## Verification

Used a disposable demo host and state directory on localhost. The user's native engine configuration and actual model files were not used for lifecycle testing.

| Check | Result and scope |
|---|---|
| TypeScript | `npm run typecheck` passed on final source. |
| UI source audit | `npm run audit` passed: valid tokens/classes/icons, kit in sync, five timing tokens, 11 approved material surfaces and no native selects. |
| Production build | `npm run build` passed: 64 modules; JS 419.02 kB / 128.27 kB gzip; CSS 99.24 kB / 18.26 kB gzip. Bundle size is not a RAM or responsiveness measurement. |
| Automated tests | `node --test tests/*.test.mjs`: 14 passed. Includes six new estimator/recommendation/catalog tests and extended setup persistence assertions. The final follow-up changes only adjusted presentation and memory-impact wording; typecheck/audit/build were rerun. |
| Evidence logic | Tested unknown hardware, invalid drafts, unsupported workload/context, unified memory, unknown partial offload, capability gating, incompatible evidence, provisional choices, no fit and one winner across profiles. |
| Large catalog | Pure grouping test covers 3,000 variants. UI mounts detail only on expansion and pages families. A 3,000-row browser performance benchmark was not performed. |
| Engine interaction | Dragged the context pointer; exercised arrows, Home/End and direct entry; confirmed shared Raw values; validated/saved/reloaded settings; checked unsupported workload selection and completed Demo test. |
| Models interaction | Clicked filters, incompatible variants, family/version expansion and reversible quantization expansion. Compared three variants; verified fourth-choice limit and the wide comparison dialog. |
| Onboarding interaction | Walked all five steps through simulated successful loading and Chat. Changed profile/context/batch, used Finish later, resumed, reloaded and confirmed retained step/choices. |
| Recovery | Host contract tests cover failed lifecycle outcomes and cancellation; setup resume was exercised in the browser. Exhaustive browser fault injection and native download-resume validation remain outstanding. |
| Responsive layout | Engine checked at 390, 768, 1024 and 1512px; document width matched viewport. Models and onboarding were also inspected at 390px, with no document overflow. Desktop comparison inspected at 1512px. |
| Preferences | Light/Dark inspected. Reduced motion, Opaque and Increased contrast selected through Settings; rendered dial transition duration was 0s. Defaults restored after the check. |
| Contrast sample | Ten Engine text pairings passed in each theme: minimum 4.81:1 Light, 5.73:1 Dark. This samples text, not every control boundary or interaction state. |
| Console | No warnings or errors in the inspected final preview log. |

The repository's external-browser `contrast`, `measure`, `luminance`, `flow`, `flows`, `motion` and `motion:budget` launchers were not run. Browser interactions used the supported browser API; the contrast result reused the repository's read-only probe body with added Engine selectors.

## Saved evidence

Screenshots show development verification states; values labelled Demo must not be read as machine benchmarks.

| Evidence | View |
|---|---|
| [Engine Light](evidence/instrument-redesign-2026-10-05/01-engine-light.png) | Dials and shared expected experience. |
| [Engine Dark](evidence/instrument-redesign-2026-10-05/05-engine-dark.png) | Dark reading surfaces and controls. |
| [Engine mobile](evidence/instrument-redesign-2026-10-05/04-engine-mobile-light.png) | Stacked tuning at 390px. |
| [Accessibility preferences](evidence/instrument-redesign-2026-10-05/06-engine-accessible.png) | Reduced motion, opaque surfaces, increased contrast. |
| [Models](evidence/instrument-redesign-2026-10-05/07-models-light.png) | Exact recommendation names, no fitting Agentic candidate and shared winner. |
| [Variant comparison](evidence/instrument-redesign-2026-10-05/08-variant-comparison.png) | Three exact variants and visible memory constraints. |
| [Onboarding profile](evidence/instrument-redesign-2026-10-05/09-onboarding-profile.png) | Profile selection and one recommendation. |
| [Onboarding mobile](evidence/instrument-redesign-2026-10-05/10-onboarding-mobile.png) | Stacked choices and compact step progress. |
| [Computer](evidence/instrument-redesign-2026-10-05/02-onboarding-computer.png), [Ready](evidence/instrument-redesign-2026-10-05/03-onboarding-ready.png) | Earlier captures from the completed walkthrough, before final spacing refinement. |
| [Browser checks](evidence/instrument-redesign-2026-10-05/browser-checks.json) | Raw contrast samples, responsive widths and accessibility style observation. |

## Remaining integration and verification

1. Connect actual Strata capabilities and supported parameter ranges. The current adapter catalog is not a complete native Strata configuration contract.
2. Curate approved agent/tool candidates and Balanced ordering with reviewed capability evidence. Collect comparable throughput data; do not turn the demo curves into a production recommendation formula.
3. Implement native, user-initiated, cancellable benchmarks. Before persisting production evidence, add engine build/driver identity and explicit measurement validity/invalidation rules to the existing evidence identity.
4. Calibrate memory estimates, including real batch workspace, GPU placement and cache allocation. The current memory strip is an explained approximation, not an out-of-memory guarantee.
5. Verify true 200% browser zoom, full screen-reader operation, forced colors and all focus/disabled/hover contrast states. These are not implied by sampled contrast and keyboard checks.
6. Measure drag, scrolling and streaming together on representative hardware. No FPS, CPU/RAM, compositor or native throughput performance claim has been made.

The requested UI redesign is implemented. These remaining items concern production evidence, native integration and broader platform/performance acceptance.
