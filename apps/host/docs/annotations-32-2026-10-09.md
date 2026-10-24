# 32 annotation review — 9 October 2026

Completed in `/Users/nikoveit/Documents/Strata-Design/backups/strata`.

App: http://localhost:5160 · Host API: http://127.0.0.1:5161

## Checklist

| Comment | Result |
| --- | --- |
| 1 | Chat model rows show name – quant – GB on one line. |
| 2 | All seven model API states have explicit labels and actions; transitions disable repeat actions, progress is shown, and errors retry download or loading according to whether files are installed. |
| 3 | Removed the shared-model heading from the chat model menu. |
| 4 | Running version sorts first in the chat menu. |
| 5 | Removed Manage model versions from the chat menu footer. |
| 6 | Removed the example-agents footer from the subagent popover. |
| 7 | Removed the Design example badge from chat responses. |
| 8 | Time-skip card arrow points right toward the side view. |
| 9 | Empty chats show a usage dashboard with Overview / Models, All / 30d / 7d, six statistics, activity heatmap, stacked model usage chart, token breakdown and percentages. Generic heading contains no personal name. |
| 10 | Every skill uses the same glyph from the existing icon set. |
| 11 | GPU inventory lists all detected devices. Temperature and utilization are matched per device, with unavailable sensor readings shown explicitly. |
| 12 | Physical processors are listed separately. Aggregate core counts remain available to Engine tuning. The local M4 now correctly reports 10 physical cores and 10 logical threads. |
| 13 | Reset to recommended and Run speed test are in the Engine header after the mode button. |
| 14 | Removed the specified working-memory warning from the visible validation list. Validation still runs. |
| 15 | Removed the specified fast-memory warning from the visible validation list. Validation still runs. |
| 16 | Conversation context is linear from 4K to the model maximum, in exact 1K increments. The host validates, saves and translates every supported integer K value. |
| 17 | Removed the performance Details disclosure. |
| 18 | Removed the Speeds · Demo estimate caption. Provenance remains available to assistive technology and on hover. |
| 19 | Removed the Sample workload selector from the performance preview. |
| 20 | Removed Shared by all chats from the model hero. |
| 21 | Removed Ready from the model hero. |
| 22 | Removed Version context selection; the Model page uses the saved Engine context. |
| 23 | On this computer rows show quant and GB without Smallest / Balanced names. |
| 24 | Removed View running version. |
| 25 | Removed View version. |
| 26 | The precision explanation sits inline beside Choose a version. |
| 27 | Model hero includes a family selector populated from the actual catalog, with versions filtered by the selected family. |
| 28 | Removed Not set beside API key. |
| 29 | API key label remains on one line. |
| 30 | Same badge removal as comment 7. |
| 31 | Removed the Mode heading and corrected option heights. All five choices fit without scrolling at the current desktop viewport. |
| 32 | User message hover/focus shows relative sent time, copy and Rewind to here. Rewind preserves the original chat and opens a branch with the earlier prompt ready to edit. It is disabled during generation. |

## State mapping

| API state | Picker presentation |
| --- | --- |
| available | Download |
| downloading | Downloading with percentage and progress |
| installed | Downloaded; Switch version |
| loading | Loading; repeat action disabled |
| loaded | Running when it is the active model, otherwise Downloaded |
| unloading | Unloading; repeat action disabled |
| error | Retry; installed files retry loading, otherwise retry download |

## Data and hardware limits

- Usage derives from saved browser history. Message and token counts deduplicate copied branch turns. Showcase and Time-skip example turns are excluded. Token totals include only reported engine usage; demo, estimated and unavailable usage do not become measured totals.
- The current build runs in simulated mode and its catalog contains one model family. No extra production families were invented. A test covers multiple families in a fixture.
- NVIDIA sensors use stable GPU UUIDs. Linux DRM sensors are read from each device's own sysfs path. Processor topology uses macOS physical/logical/package counts, Linux socket/core topology, and Windows processor instances.
- This Mac does not expose GPU temperature/utilization through these probes, so its card shows Unavailable. Two-processor and multiple-GPU cases were verified with fixtures, including distinct temperatures and a missing sensor.
- Hardware inventory does not assume that different GPUs' VRAM can be pooled by the inference adapter.

Primary probe references: [NVIDIA SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html), [Linux CPU topology](https://man7.org/linux/man-pages/man1/lscpu.1.html), [Windows processors](https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-processor), [Linux AMD GPU sensors](https://docs.kernel.org/gpu/amdgpu/thermal.html).

## Verification

- `npm run typecheck` — passed.
- `npm run build` — passed; 117 modules.
- `node --test tests/annotations-32.test.mjs tests/dictation.test.mjs tests/agent-updates.test.mjs tests/tasks-and-sharing.test.mjs` — 20 passed.
- Live API accepted 33K; rejected 33.5K and 1025K. Engine launch argument tests cover 4K through 1024K, including non-power-of-two values.
- Browser: context ArrowRight moved 32K → 33K; test draft discarded, saved context remains 32K.
- Browser: Mode menu client height and scroll height both 297px. Model menu client height and scroll height both 306px; all model rows 42px.
- Browser: available version opens Download; installed version opens Switch; running version remains first.
- Browser: dashboard tabs and periods, model family menu, removed labels, hardware cards, API key layout, skill glyphs, message hover and Time-skip arrow checked.
- Rewind test branch `s_l1uci3r3idc9` was archived. The original `give me timeskip` transcript and its `Show me multi` draft were preserved.

## Screenshots

Evidence is in `docs/evidence/annotations-32-2026-10-09/`:

- `usage-overview.jpg`, `usage-models.jpg`
- `model-picker.jpg`, `mode-menu.jpg`
- `models.jpg`, `installed-versions.jpg`
- `engine.jpg`, `monitor.jpg`, `skills.jpg`, `serving.jpg`
- `subagents.jpg`, `message-hover-timeskip.jpg`, `rewind-branch.jpg`
