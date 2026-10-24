# Strata audit fixes — implementation and review

Executed by Codex on 7 October 2026. No Claude process or Claude subagent was used.

## Scope and setup

- Package extracted to `audit-fixes/`; bundled agent definition copied unchanged to `.claude/agents/strata-fixer.md`.
- Original 420 audit PNGs remain in `docs/evidence/audit-2026-10-07/`.
- All 19 tasks were handled in dependency order. The required scope was **15 high + 82 medium findings**. The 80 optional low findings were not scheduled for a separate polish pass; some disappeared as a consequence of rebuilding the affected surfaces.
- Every required finding has a disposition in [FINDING-STATUS.md](FINDING-STATUS.md), with a machine-readable copy in [finding-status.json](finding-status.json).
- This app and its parent folder have no Git repository. No branches/commits were invented. Original source and preview state were saved in `baseline-source.tar.gz` and `baseline-preview-state.json`. [changed-files.txt](changed-files.txt) records the source comparison.
- Design-kit files under `src/styles/kit/` were not changed.

## Phase 1 — shared fixes

### T01 · One model identity

Loaded-model identity now uses formatModelName. Removed the chat model catalog and multi-model profile flow; Model and Setup share a version ladder.

- Required findings: F001, F011, F059, F060, F077, F090, F123, F150.
- Main files: `src/lib/format.ts; src/app/Sidebar.tsx; src/components/ComposerControls.tsx; src/screens/{Models,Setup,Tuning,Monitor}.tsx; server/catalog.mjs`.
- Dispositions and qualifications: see the T01 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T02 · Overlay layering

Opaque dialogs and palettes, explicit overlay layers, dropdowns above their parent dialog, dark scrims and stable scroll gutters.

- Required findings: F140, F149, F155, F158, F169.
- Main files: `src/components/{ui,Popover,FloatingMenu}.tsx; src/styles/app.css; src/screens/Chat.tsx`.
- Dispositions and qualifications: see the T02 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T03 · Control states

Enabled primary buttons, neutral disabled controls, bright on toggles, brand checkboxes, a compact growing composer, and labelled task actions/detail chrome.

- Required findings: F025, F035, F094, F126, F133, F142, F170, F176.
- Main files: `src/styles/app.css; src/screens/{Chat,Tasks}.tsx`.
- Dispositions and qualifications: see the T03 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T04 · Errors

Shared ErrorNotice sanitizes Fixture prefixes and supports cause, recovery text and retry. Applied to composer/website, models, skills, tasks, setup, Serving and Cloudflare. Request states have semantic colors.

- Required findings: F021, F053, F098, F102, F111, F127, F131, F134, F153.
- Main files: `src/components/{ui,CloudflareDialog,ComposerControls}.tsx; src/lib/format.ts; src/screens/{Models,Skills,Tasks,Serving,Setup,Chat}.tsx`.
- Dispositions and qualifications: see the T04 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T05 · Live numbers

Hide idle chat statistics; show live estimated speed/tokens until reported usage arrives. Context comes from engine config. Clamp and flag impossible memory readings without asserting unobserved swapping.

- Required findings: F003, F006, F023, F121.
- Main files: `src/lib/{chat-metrics,hardware-summary,store}.ts; src/components/TelemetryPopover.tsx; src/screens/Monitor.tsx; server/monitor.mjs`.
- Dispositions and qualifications: see the T05 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T06 · Response hover layout

Response actions use a horizontal row below content and a portaled floating menu, so opening More never reflows or clips the response.

- Required findings: F034.
- Main files: `src/screens/Chat.tsx; src/styles/app.css`.
- Dispositions and qualifications: see the T06 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T07 · Unbuilt controls

Feature flags hide unfinished access modes, thinking levels and session-log export. A single Preview build badge identifies simulated mode; provenance remains where otherwise misleading.

- Required findings: F051.
- Main files: `src/lib/features.ts; src/components/{ComposerControls,ChatToolbar,layout,ui}.tsx; src/components/TelemetryPopover.tsx`.
- Dispositions and qualifications: see the T07 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T08 · Primary navigation

Engine and singular Model are primary navigation destinations. More contains Serving and Settings.

- Required findings: F159.
- Main files: `src/app/Sidebar.tsx; src/lib/commands.ts`.
- Dispositions and qualifications: see the T08 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

## Phase 2 — Model and Engine

### T09 · Model page

Single-model hero, seven version rows, detail action, installed list and shared Setup selection. Fit uses detected memory and context, with only the highest-quality fast fit recommended. Repeated profile warnings and compare trays were removed.

- Required findings: F061, F064, F070, F071, F072, F073, F074, F076, F152.
- Main files: `src/screens/{Models,Setup}.tsx; src/components/VersionLadder.tsx; src/lib/model-versions.ts; server/{catalog,models}.mjs`.
- Dispositions and qualifications: see the T09 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T10 · Engine v3

Simple workloads/headroom and capability-aware context slider; animated auto-tune recommendation; compact grouped Expert parameters with unsupported inputs hidden/disabled; underline sub-tabs; inline JSON line/column errors; dirty-only action bar.

- Required findings: F078, F079, F080, F081, F082, F084, F087, F092, F093, F096, F097.
- Main files: `src/screens/Tuning.tsx; src/components/{AutoTune,EngineExperience}.tsx; src/lib/config.ts; src/styles/app.css`.
- Dispositions and qualifications: see the T10 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

## Phase 3 — remaining screen fixes

### T11 · Chat

Zero-agent menus hidden; title truncation signalled; explicit template selection; approval summaries; failed-agent log; simplified slash commands; branch-aware empty state; inline emphasis; file actions; text attachment support; demos moved to Settings About.

- Required findings: F002, F004, F007, F012, F013, F015, F017, F018, F026, F028, F030, F032, F039, F041, F043, F052, F055, F056, F057, F058.
- Main files: `src/screens/Chat.tsx; src/components/{ChatToolbar,SubagentRow,ArtifactCard,ComposerControls,Markdown}.tsx; src/lib/{store,commands}.ts`.
- Dispositions and qualifications: see the T11 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T12 · Time-skip

Stable Version A/B labels, readable amber status text, primary Open version B and secondary Compare at completion. The selected example result opens in the preview.

- Required findings: F045, F047, F048.
- Main files: `src/components/TimeSkipPanel.tsx; src/screens/Chat.tsx; src/styles/app.css`.
- Dispositions and qualifications: see the T12 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T13 · Serving

Queued cancellation, generated key dialog with copy only for a value, explicit local-network status/action and URL copy, and Connect an app in the empty state.

- Required findings: F103, F108, F117, F118, F119, F132.
- Main files: `src/screens/Serving.tsx; src/components/CloudflareDialog.tsx`.
- Dispositions and qualifications: see the T13 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T14 · Monitor

Shared-memory labels explain pool overlap and idle states are clear. Memory validation is shared with T05.

- Required findings: F122.
- Main files: `src/screens/Monitor.tsx; src/lib/hardware-summary.ts`.
- Dispositions and qualifications: see the T14 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T15 · Skills and Tasks

Sans-serif instruction fields; scheduling state distinguishes scheduled and paused tasks; skills scope is explicit. Shared errors, toggles and dropdown layering cover related findings.

- Required findings: F136, F141.
- Main files: `src/screens/{Skills,Tasks}.tsx; src/components/TaskEditor.tsx; src/styles/app.css`.
- Dispositions and qualifications: see the T15 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T16 · Settings and Setup

Settings index follows the visible/search-result section. Setup context controls are inline, and its opaque bounded dialog retains a sticky footer.

- Required findings: F144, F151.
- Main files: `src/screens/{Settings,Setup}.tsx; src/styles/app.css`.
- Dispositions and qualifications: see the T16 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T17 · Navigation dialogs

Rename uses a stacked full-width selected input; deletion offers Archive instead; shared overlay/gutter fixes prevent page movement.

- Required findings: F156, F157.
- Main files: `src/app/Sidebar.tsx; src/styles/app.css`.
- Dispositions and qualifications: see the T17 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T18 · Preview panel

Active tabs scroll within their strip with edge fades. Narrow windows use an overlay; wide windows retain enough chat width. Images/PDF share Fit/100% controls and PDF uses icon paging.

- Required findings: F162, F163, F164.
- Main files: `src/components/{ArtifactPanel,PreviewControls,PdfPreview}.tsx; src/screens/Chat.tsx; src/styles/app.css`.
- Dispositions and qualifications: see the T18 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

### T19 · Dark theme

Shared field-error style is visible in dark mode; opaque overlays and dark scrims preserve depth. Primary/toggle changes pass contrast and surface-luminance checks.

- Required findings: F173.
- Main files: `src/styles/app.css; src/components/ComposerControls.tsx`.
- Dispositions and qualifications: see the T19 entries in [FINDING-STATUS.md](FINDING-STATUS.md).

## Verification

| Check | Result | Evidence |
|---|---|---|
| Baseline unit suite | 34 passed | Existing suite before changes |
| Final unit suite | 39 passed; 0 failed, skipped or cancelled | [tests.log](tests.log) |
| Production build | Passed | [build.log](build.log) |
| TypeScript, UI audit, WCAG contrast, reading measure, dark-surface luminance | Passed | [check-final.log](check-final.log) |
| Motion/reduced motion | Passed | [motion.log](motion.log) |
| Quantified motion budget | **Not measured**: harness self-check could not detect its deliberately expensive control effect | [motion-budget.log](motion-budget.log) |

The package command `node --test tests/` is not accepted as a directory entry point by the installed Node version. Used `node --test tests/*.test.mjs` instead. Existing tests were retained. Fixture IDs and stale UI selectors were updated to the new single-model catalog and current accessible control labels; five logic tests were added (identity, memory, model fit, usage, JSON diagnostics).

Interactive checks covered Repeat above the task modal; dark/light enabled and disabled controls; settings search; failed-agent logs; website validation; active-tab closing and scrolling; PDF/image controls; Time-skip completion and opening its selected result; JSON error placement; and reply menu layout. A reply rectangle stayed exactly 864 × 298.8px before/after opening More at the wide viewport. Expert now shows five parameter rows and the next section heading at 1280×720.

The screenshot set is a **targeted regression review**, not a replacement capture of all 210 old states. Retired model pickers/profile/compare surfaces cannot be recaptured. An isolated read-only synthetic fixture endpoint was blocked by the browser (`ERR_BLOCKED_BY_CLIENT`); it was stopped without attempting a bypass. Synthetic hardware-unavailable, model-load/prepare failure, service-outage, queued and active-sharing states are source/logic reviewed but not all visually recaptured. No API key or network exposure was enabled during review.

## Product qualifications

- The seven Qwen3.8 versions are **design-preview data** from the supplied reference, explicitly marked Preview build. They are not claims of available downloadable model files. Real-engine mode retains the original verified model catalog and real download sources; this is why a literal source grep still finds Qwen2.5 in `server/real-model-catalog.mjs`.
- Model fit uses detected hardware. The current 16GB computer cannot run the preview Balanced weights within the estimated budget; the UI correctly warns even though the simulated preview marks that version Running.
- Numerical per-version benchmark speeds are not invented. Model says Not measured; Engine labels its demo estimates. Auto-tune animates checking/recommending supported settings and asks to review/save; it does not claim to have executed a benchmark.
- Primary colors were adjusted to satisfy measured text contrast and the existing dark-surface budget. Literal white on #10b981 in light mode would fail text contrast.
- The Serving guide header stays icon-only because the user explicitly requested that earlier. Its empty state now includes a labelled Connect an app action.
- Text/code attachment is supported in the visible prompt. Image/video/GIF previews remain available; binary model input is outside the current text inference API.
- Time-skip is still the scripted two-version prototype. Open version B opens the actual selected example summary; it does not imply a generated OMP artifact.
- Setup/preview popovers are suppressed when setup is open. HMR edits to the central store require a browser reload during development; a normal fresh page load initializes correctly.

## Evidence and follow-up

See [the screenshot gallery](../docs/evidence/audit-fixes-2026-10-07/index.html) and [its coverage notes](../docs/evidence/audit-fixes-2026-10-07/README.md). Optional low-priority work remains listed in the supplied task files and finding ledger. Examples include richer raw-config syntax highlighting, advanced storage actions, and minor copy/spacing polish.
