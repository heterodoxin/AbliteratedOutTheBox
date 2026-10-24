# Persona review: implementation and verification

Implemented on 2026-10-05 against [the decision report](decision-report.md). The original votes and dissent remain unchanged. All 18 majority decisions are applied; the two unresolved decisions use the report's explicitly provisional choices: keep Agentic / Balanced / Performance, and show a modest pair of speed readings.

## What changed

| Decisions | Implementation |
|---|---|
| B01, B20 | Removed repeated introductions and teaching. Shortened workload labels to Short text, Document text and Long context. Kept token counts and context constraints. |
| B02 | Beginner opens with Context. Prompt batch has an Automatic/Custom summary and an Adjust disclosure; custom values, exact input, keyboard operation and reset remain available. One shared gesture hint. |
| B03, B18 | Compact Reading/Answer pair with one speed-source label, memory requirement and headroom. Timing, allocation, configuration and provenance move into Details. Unknown speeds stay unknown. |
| B04 | Save for next load is primary; Apply now remains secondary; Discard appears for a dirty draft. A keyboard-accessible More actions menu holds defaults, validation and Demo preview. |
| B05 | Expert retains parameters, recommendations, applicability, errors, per-field revert and shared Raw editing. General descriptions are optional. Raw is labelled Configuration draft. |
| B06, B09 | Merge recommendations only when variant **and full execution configuration** match. Multiple profile chips share one action. No-fit and missing-hardware profiles use compact, actionable rows. |
| B07 | Exact family/version and quantization stay visible. Source repository, filename and Copy file source are available in recommendation, variant and comparison details. |
| B08, B13 | Provisional: keep both compact rates and existing profile names. These choices are not presented as majority decisions or measured usability wins. |
| B10 | Installed is the initial library when files exist; deliberate Installed/Discover choice persists. Removed the redundant Models picker. Kept Engine/setup picker, filters, search and reversible expansion. |
| B11 | Comparison uses aligned facts and shared workload/context controls. Each variant has optional details. Mobile comparison stacks labelled rows. |
| B12, B14 | Computer → Model → Prepare → Ready. Model includes an explicit review before loading/downloading, with optional tuning. v1 progress migrates once without losing choices or completed work. |
| B15 | One preparation stage and progress presentation. Error recovery reports retained files and retries the appropriate operation. An installed model with a load error retries loading, without waiting for another download. |
| B16 | Ready requires a successfully loaded model and retains the Demo label. Setup detail and preview testing remain optional. |
| B17 | Shared lifecycle policy checks active/queued work, connection, missing status and in-progress model operations before load/unload/restart or related configuration writes. Save-only remains available. Global scope appears near affected actions. |
| B19 | A concise actual capability-review status replaces generic tradeoff prose. No new capability claim was added. |

The supplied brand, typefaces, icons, design tokens, materials and motion scale remain in use. Cards wrap long identities. The Engine save bar becomes part of normal flow on narrow or short viewports so enlarged content remains reachable.

## Observable reduction in default content

Read from rendered `innerText`, with disclosures closed. These are whitespace-delimited copy counts, not a reading-time or human usability study. Models used the same 16 GB demo fixture, installed Starter variant, no loaded model, 32K context, Document text workload and Discover library. Desktop layout was 1512 × 982. Engine used the default 32K / 2048 setup. Setup compares the previous Model + Tune steps with their combined review.

| Surface | Before | After |
|---|---:|---:|
| Engine visible surface copy | 221 words | 100 words |
| Models visible surface copy | 257 words | 135 words |
| Computer setup copy | 93 words | 67 words |
| Model selection + tuning review | 216 words across two steps | 103 words in one step |
| Shared-winner recommendation cards | 3 | 1, plus a compact no-fit row |
| Model library top from viewport origin | 1126 px | 828 px |
| Core setup steps | 5 | 4 |

The library begins about 299 px earlier in this fixture. No text-size reduction was used. Primary setup navigation reduces from four user actions to three; Prepare still advances only after successful loading. Actual task success, comprehension and speed of use require human testing.

## Automated verification

Final source checks passed:

- `npm run typecheck`
- `npm run audit` — tokens, kit consistency, motion scale, materials, icons and classes clean.
- `node --test tests/*.test.mjs` — **18 passed, 0 failed**.
- `npm run build` — 67 modules; production JS 420.65 kB / 128.53 kB gzip; CSS 105.65 kB / 19.22 kB gzip.

Relevant added regression coverage checks migration from every old step; exact setup grouping including cache, placement and thread differences; shared lifecycle handling; and actual store actions making **zero network calls** when a lifecycle write is blocked. Save-only still issues its configuration PATCH. Existing demo-host recovery, configuration, serving cancellation, request ordering and streaming tests also pass.

## Clicked browser verification

Performed through CUA in an isolated local demo host, with disposable storage and an API-edge fixture proxy. No real model was downloaded or run. The proxy in `tests/ui-fixture-host.mjs` is a test helper, not an application dependency.

| Flow / state | Result |
|---|---|
| Context keyboard change and exact batch entry | 32K → 16K and custom 1024 reflected in draft feedback; Expert and Raw showed the same saved values. |
| Raw unknown configuration | An unknown fixture key survived editing and saving. |
| More actions keyboard | Initial focus on Recommended defaults; ArrowDown reached Validate draft; Escape closed the menu and returned focus to More actions. |
| File source | Expanded details, copied source, and verified the expected filename in the copied text. |
| Installed/Discover | Initial installed default and deliberate Discover preference verified across navigation. |
| Library and comparison | Expanded/collapsed quantizations, selected three variants, fourth selection disabled; aligned desktop and stacked mobile comparison; Escape returned focus to Compare variants. |
| Large catalog | 1200 variants grouped into 12 collapsed families with zero expanded variant rows initially. Exact search narrowed matches; long names wrapped at 390 px. |
| First setup | Explicit review, simulated download/load, Ready, then Chat. Profile selection alone did not download. |
| Cancellation | Cancelled preparation retained choices and offered the appropriate retry; retry completed. |
| Legacy setup and invalid retained values | Old Tune step mapped to Model. Unsupported 128K remained visible, opened adjustment controls and disabled Load. Corrected 8K/custom batch survived Finish later and resume. |
| Failed loading | HTTP-success/error-state fixture produced one recovery message, retained files and Retry loading. Switching fixture to normal reached Ready without another download. |
| Setup → Engine | Custom batch 1024 remained Custom in Engine after setup applied it. |
| Active and queued requests | One active/two queued fixture blocked unloading and load confirmation; loaded model remained. The store regression additionally checked that associated settings writes were blocked. |
| Unknown request status | Missing counters produced an unavailable explanation, not idle zeroes. |
| Unknown hardware | All profiles showed Fit not checked and Check computer, without claiming nothing fits. |
| Missing speed evidence | Not estimated / Unavailable / Provisional remained visible; no measured performance was fabricated. |
| Responsive layout | Engine, Models and setup checked at 1512, 1024, 768 and 390 px. Document and relevant containers had no horizontal overflow in all 12 combinations. |
| Browser zoom | Chrome's native zoom control confirmed **200%**. Models library, Engine actions and setup's final Load model action remained reachable. Zoom restored to 100%; temporary tab closed. |
| Appearance/accessibility | Light and Dark inspected. Reduced motion produced `0s` dial-pointer transitions; Opaque produced `backdrop-filter: none`; increased contrast applied to all three screens. Test preferences restored afterwards. |

The existing rendered contrast probe was run through CUA's read-only evaluator for Light, standard Dark and increased-contrast Dark on Engine. Every sampled pairing passed its 4.5:1 requirement. This is a sampled contrast check, not a complete accessibility certification. The final browser console sample contained no warnings or errors.

## Evidence

- [Engine](../evidence/persona-simplification-2026-10-05/01-engine-desktop.png)
- [Models](../evidence/persona-simplification-2026-10-05/02-models-desktop.png)
- [Setup review](../evidence/persona-simplification-2026-10-05/05-setup-review.png)
- [Desktop comparison](../evidence/persona-simplification-2026-10-05/03-comparison-desktop.png) / [mobile comparison](../evidence/persona-simplification-2026-10-05/04-comparison-mobile.png)
- [Final load failure](../evidence/persona-simplification-2026-10-05/20-setup-load-failure.png) / [successful retry](../evidence/persona-simplification-2026-10-05/07-setup-ready.png)
- [Unknown hardware](../evidence/persona-simplification-2026-10-05/14-unknown-hardware.png) / [long names](../evidence/persona-simplification-2026-10-05/15-long-model-name-mobile.png)
- [Dark accessibility preferences](../evidence/persona-simplification-2026-10-05/17-engine-dark-accessibility.png)
- [Measurements and responsive checks](../evidence/persona-simplification-2026-10-05/verification-data.json)
- [Evidence directory notes](../evidence/persona-simplification-2026-10-05/README.md)

## Practical limits

Native Strata integration and real inference benchmarking remain deferred as requested. Demo speeds cannot establish recommendation quality or resource savings. Quantitative drag/scroll responsiveness during streaming, battery impact and full luminance/motion performance measurements were not performed; no performance improvement is claimed. Accessibility verification here covers the touched controls, sampled contrast and listed layouts, not a full screen-reader audit.

The reusable workflow is installed separately at `/Users/nikoveit/.codex/skills/ui-persona-review/`, with independent reviews, peer challenges, validated voting and reporting. Application implementation remains an explicitly authorized follow-up rather than part of its default review behavior.
