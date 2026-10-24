from pathlib import Path
from collections import Counter, defaultdict
import json, html
from PIL import Image, ImageDraw
r=Path('audit-fixes'); findings=json.loads((r/'findings.json').read_text())
summary={
'T01':('One model identity', 'Loaded-model identity now uses formatModelName. Removed the chat model catalog and multi-model profile flow; Model and Setup share a version ladder.', 'src/lib/format.ts; src/app/Sidebar.tsx; src/components/ComposerControls.tsx; src/screens/{Models,Setup,Tuning,Monitor}.tsx; server/catalog.mjs'),
'T02':('Overlay layering', 'Opaque dialogs and palettes, explicit overlay layers, dropdowns above their parent dialog, dark scrims and stable scroll gutters.', 'src/components/{ui,Popover,FloatingMenu}.tsx; src/styles/app.css; src/screens/Chat.tsx'),
'T03':('Control states', 'Enabled primary buttons, neutral disabled controls, bright on toggles, brand checkboxes, a compact growing composer, and labelled task actions/detail chrome.', 'src/styles/app.css; src/screens/{Chat,Tasks}.tsx'),
'T04':('Errors', 'Shared ErrorNotice sanitizes Fixture prefixes and supports cause, recovery text and retry. Applied to composer/website, models, skills, tasks, setup, Serving and Cloudflare. Request states have semantic colors.', 'src/components/{ui,CloudflareDialog,ComposerControls}.tsx; src/lib/format.ts; src/screens/{Models,Skills,Tasks,Serving,Setup,Chat}.tsx'),
'T05':('Live numbers', 'Hide idle chat statistics; show live estimated speed/tokens until reported usage arrives. Context comes from engine config. Clamp and flag impossible memory readings without asserting unobserved swapping.', 'src/lib/{chat-metrics,hardware-summary,store}.ts; src/components/TelemetryPopover.tsx; src/screens/Monitor.tsx; server/monitor.mjs'),
'T06':('Response hover layout', 'Response actions use a horizontal row below content and a portaled floating menu, so opening More never reflows or clips the response.', 'src/screens/Chat.tsx; src/styles/app.css'),
'T07':('Unbuilt controls', 'Feature flags hide unfinished access modes, thinking levels and session-log export. A single Preview build badge identifies simulated mode; provenance remains where otherwise misleading.', 'src/lib/features.ts; src/components/{ComposerControls,ChatToolbar,layout,ui}.tsx; src/components/TelemetryPopover.tsx'),
'T08':('Primary navigation', 'Engine and singular Model are primary navigation destinations. More contains Serving and Settings.', 'src/app/Sidebar.tsx; src/lib/commands.ts'),
'T09':('Model page', 'Single-model hero, seven version rows, detail action, installed list and shared Setup selection. Fit uses detected memory and context, with only the highest-quality fast fit recommended. Repeated profile warnings and compare trays were removed.', 'src/screens/{Models,Setup}.tsx; src/components/VersionLadder.tsx; src/lib/model-versions.ts; server/{catalog,models}.mjs'),
'T10':('Engine v3', 'Simple workloads/headroom and capability-aware context slider; animated auto-tune recommendation; compact grouped Expert parameters with unsupported inputs hidden/disabled; underline sub-tabs; inline JSON line/column errors; dirty-only action bar.', 'src/screens/Tuning.tsx; src/components/{AutoTune,EngineExperience}.tsx; src/lib/config.ts; src/styles/app.css'),
'T11':('Chat', 'Zero-agent menus hidden; title truncation signalled; explicit template selection; approval summaries; failed-agent log; simplified slash commands; branch-aware empty state; inline emphasis; file actions; text attachment support; demos moved to Settings About.', 'src/screens/Chat.tsx; src/components/{ChatToolbar,SubagentRow,ArtifactCard,ComposerControls,Markdown}.tsx; src/lib/{store,commands}.ts'),
'T12':('Time-skip', 'Stable Version A/B labels, readable amber status text, primary Open version B and secondary Compare at completion. The selected example result opens in the preview.', 'src/components/TimeSkipPanel.tsx; src/screens/Chat.tsx; src/styles/app.css'),
'T13':('Serving', 'Queued cancellation, generated key dialog with copy only for a value, explicit local-network status/action and URL copy, and Connect an app in the empty state.', 'src/screens/Serving.tsx; src/components/CloudflareDialog.tsx'),
'T14':('Monitor', 'Shared-memory labels explain pool overlap and idle states are clear. Memory validation is shared with T05.', 'src/screens/Monitor.tsx; src/lib/hardware-summary.ts'),
'T15':('Skills and Tasks', 'Sans-serif instruction fields; scheduling state distinguishes scheduled and paused tasks; skills scope is explicit. Shared errors, toggles and dropdown layering cover related findings.', 'src/screens/{Skills,Tasks}.tsx; src/components/TaskEditor.tsx; src/styles/app.css'),
'T16':('Settings and Setup', 'Settings index follows the visible/search-result section. Setup context controls are inline, and its opaque bounded dialog retains a sticky footer.', 'src/screens/{Settings,Setup}.tsx; src/styles/app.css'),
'T17':('Navigation dialogs', 'Rename uses a stacked full-width selected input; deletion offers Archive instead; shared overlay/gutter fixes prevent page movement.', 'src/app/Sidebar.tsx; src/styles/app.css'),
'T18':('Preview panel', 'Active tabs scroll within their strip with edge fades. Narrow windows use an overlay; wide windows retain enough chat width. Images/PDF share Fit/100% controls and PDF uses icon paging.', 'src/components/{ArtifactPanel,PreviewControls,PdfPreview}.tsx; src/screens/Chat.tsx; src/styles/app.css'),
'T19':('Dark theme', 'Shared field-error style is visible in dark mode; opaque overlays and dark scrims preserve depth. Primary/toggle changes pass contrast and surface-luminance checks.', 'src/styles/app.css; src/components/ComposerControls.tsx')}
# IDs are stable, one-based positions in the supplied findings.json.
notes={
1:'Formatter reads the actual loaded record; simulated preview uses the design family, real-engine mode retains its verified catalog.',
11:'Closed by removal: chat version chip links to Model; no multi-model picker remains.',
12:'Closed by removal of the model-picker surface; running status is a non-action status on Model.',
15:'Closed by hiding session-log export until it is implemented.',
17:'Removed duplicate /clear; /new is the single new-chat command.',
25:'Adapted colors to measured contrast: light #10b981 uses dark ink; dark uses #087654 with white ink to meet both text contrast and the app surface-luminance budget.',
26:'View log exposes the recorded failure. No unsupported backend retry is implied.',
35:'Empty textarea is one line and grows to 180px. Three stats chips remain one compact row directly below it, preserving the prior requested separate popovers.',
51:'Unbuilt controls hidden. Preview badge is shared; estimated/simulated per-value labels remain where removing them would misrepresent data.',
52:'Attach text file inserts inspected text/code into the visible prompt. Binary media remains preview-only because the inference endpoint accepts text. Showcase links moved to Settings About.',
55:'Closed by hiding unbuilt tool-access choices behind a disabled feature flag (T07).',
59:'New demo replies use the loaded identity. Existing saved historical messages are preserved.',
70:'One consolidated memory warning and Use shorter context action replace repeated profile dead ends; budgets derive from hardware, not sample 2.1 GB numbers.',
71:'Closed by removing the multi-select compare tray and profile filters.',
74:'Plain switch sentence plus a disclosure of saved Engine context. No unmeasured five-second loading promise.',
76:'One retry action in the switch dialog and shared sanitized error. Synthetic failed-load visual capture was unavailable; source path reviewed.',
79:'Range is filtered against model capabilities, including 1M for the design preview.',
90:'Header uses the actual loaded record; a real adapter may correctly report the original Qwen2.5 test model.',
97:'Removed the duplicated unsupported Server access editor; Expert links to Serving.',
102:'Shared sanitizer removes Fixture prefix. Synthetic upstream-error visual capture was unavailable.',
108:'Key is generated in the editor draft on open; copy appears only for non-empty text. No key was saved during this review.',
111:'Shared ErrorNotice and one retry path; synthetic connector-error visual capture was unavailable.',
119:'Shared local-network state has an amber dot, explicit status and Stop action. Public/local sharing was not enabled for screenshots.',
121:'Used memory is clamped to total with a warning when the source reports an overflow. The UI does not claim swapping without a signal proving it.',
127:'Retry ErrorNotice sits under the page header. Synthetic service-outage visual capture was unavailable.',
131:'Central userError sanitization is used by ErrorNotice, including shared ActionError; source review plus unit coverage.',
132:'Added empty-state Connect an app. Header remains icon-only with an accessible label, honoring the user’s earlier explicit annotation.',
134:'Last run failed opens a sanitized ErrorNotice. Synthetic failed-run visual capture was unavailable.',
141:'Scheduled creation shows next run or Scheduled while the scheduler computes it, never Paused solely because nextRun is initially empty.',
150:'Setup shares the version ladder and fit recommendation; when no version fits fast, it does not falsely tag a slow version Recommended.',
152:'Bounded opaque setup with scrollable body and persistent header/footer verified at 1280×720.',
153:'Preparation error heading and ErrorNotice replace raw fixture text; synthetic prepare-failure visual capture was unavailable.',
163:'Preview switches to overlay below 1120px available main width; wide-screen chat was measured at 644.6px at a 1654px viewport.',
170:'Dark primary is #087654 with white ink: verified contrast and luminance requirements both pass.',
173:'Shared field ErrorNotice + aria-invalid and red field border, captured in both themes.'}
removed={11,12,15,55,71,97}
counters=defaultdict(int); ledger=[]
for i,f in enumerate(findings,1):
 key=(f['task'],'optional' if f['sev']=='L' else 'required'); counters[key]+=1
 row={'id':f'F{i:03}', 'task_finding':f'{f["task"]} {key[1]} #{counters[key]}',**f}
 row['status']='optional-not-scheduled' if f['sev']=='L' else 'closed-by-removal' if i in removed else 'implemented'
 row['resolution']='Optional low-priority polish was not scheduled in this first pass. Shared/rebuilt surfaces may also resolve this incidentally; no separate closure is claimed.' if f['sev']=='L' else notes.get(i,summary[f['task']][1])
 ledger.append(row)
(r/'finding-status.json').write_text(json.dumps(ledger,indent=2)+'\n')
lines=['# Finding dispositions — 7 October 2026','','Stable `F001`–`F177` IDs follow the original `findings.json` order. The task reference also gives the number within its required or optional table.','', '| ID | Task finding | Priority | State | Disposition | Resolution |','|---|---|---|---|---|---|']
for x in ledger:lines.append('| '+' | '.join(str(x[k]).replace('|',' / ').replace('\n',' ') for k in ['id','task_finding','sev','screen','status','resolution'])+' |')
(r/'FINDING-STATUS.md').write_text('\n'.join(lines)+'\n')
progress=['# Strata audit fixes — implementation and review','','Executed by Codex on 7 October 2026. No Claude process or Claude subagent was used.','', '## Scope and setup','', '- Package extracted to `audit-fixes/`; bundled agent definition copied unchanged to `.claude/agents/strata-fixer.md`.', '- Original 420 audit PNGs remain in `docs/evidence/audit-2026-10-07/`.', '- All 19 tasks were handled in dependency order. The required scope was **15 high + 82 medium findings**. The 80 optional low findings were not scheduled for a separate polish pass; some disappeared as a consequence of rebuilding the affected surfaces.', '- Every required finding has a disposition in [FINDING-STATUS.md](FINDING-STATUS.md), with a machine-readable copy in [finding-status.json](finding-status.json).', '- This app and its parent folder have no Git repository. No branches/commits were invented. Original source and preview state were saved in `baseline-source.tar.gz` and `baseline-preview-state.json`. [changed-files.txt](changed-files.txt) records the source comparison.', '- Design-kit files under `src/styles/kit/` were not changed.', '', '## Phase 1 — shared fixes','']
for task,(title,description,files) in summary.items():
 if task=='T09':progress+=['## Phase 2 — Model and Engine','']
 if task=='T11':progress+=['## Phase 3 — remaining screen fixes','']
 ids=[x['id'] for x in ledger if x['task']==task and x['sev']!='L']
 progress +=[f'### {task} · {title}', '',description,'',f'- Required findings: {", ".join(ids)}.',f'- Main files: `{files}`.',f'- Dispositions and qualifications: see the {task} entries in [FINDING-STATUS.md](FINDING-STATUS.md).','']
progress +=['## Verification','', '| Check | Result | Evidence |','|---|---|---|', '| Baseline unit suite | 34 passed | Existing suite before changes |', '| Final unit suite | 39 passed; 0 failed, skipped or cancelled | [tests.log](tests.log) |', '| Production build | Passed | [build.log](build.log) |', '| TypeScript, UI audit, WCAG contrast, reading measure, dark-surface luminance | Passed | [check-final.log](check-final.log) |', '| Motion/reduced motion | Passed | [motion.log](motion.log) |', '| Quantified motion budget | **Not measured**: harness self-check could not detect its deliberately expensive control effect | [motion-budget.log](motion-budget.log) |', '', 'The package command `node --test tests/` is not accepted as a directory entry point by the installed Node version. Used `node --test tests/*.test.mjs` instead. Existing tests were retained. Fixture IDs and stale UI selectors were updated to the new single-model catalog and current accessible control labels; five logic tests were added (identity, memory, model fit, usage, JSON diagnostics).', '', 'Interactive checks covered Repeat above the task modal; dark/light enabled and disabled controls; settings search; failed-agent logs; website validation; active-tab closing and scrolling; PDF/image controls; Time-skip completion and opening its selected result; JSON error placement; and reply menu layout. A reply rectangle stayed exactly 864 × 298.8px before/after opening More at the wide viewport. Expert now shows five parameter rows and the next section heading at 1280×720.', '', 'The screenshot set is a **targeted regression review**, not a replacement capture of all 210 old states. Retired model pickers/profile/compare surfaces cannot be recaptured. An isolated read-only synthetic fixture endpoint was blocked by the browser (`ERR_BLOCKED_BY_CLIENT`); it was stopped without attempting a bypass. Synthetic hardware-unavailable, model-load/prepare failure, service-outage, queued and active-sharing states are source/logic reviewed but not all visually recaptured. No API key or network exposure was enabled during review.', '', '## Product qualifications','', '- The seven Qwen3.8 versions are **design-preview data** from the supplied reference, explicitly marked Preview build. They are not claims of available downloadable model files. Real-engine mode retains the original verified model catalog and real download sources; this is why a literal source grep still finds Qwen2.5 in `server/real-model-catalog.mjs`.', '- Model fit uses detected hardware. The current 16GB computer cannot run the preview Balanced weights within the estimated budget; the UI correctly warns even though the simulated preview marks that version Running.', '- Numerical per-version benchmark speeds are not invented. Model says Not measured; Engine labels its demo estimates. Auto-tune animates checking/recommending supported settings and asks to review/save; it does not claim to have executed a benchmark.', '- Primary colors were adjusted to satisfy measured text contrast and the existing dark-surface budget. Literal white on #10b981 in light mode would fail text contrast.', '- The Serving guide header stays icon-only because the user explicitly requested that earlier. Its empty state now includes a labelled Connect an app action.', '- Text/code attachment is supported in the visible prompt. Image/video/GIF previews remain available; binary model input is outside the current text inference API.', '- Time-skip is still the scripted two-version prototype. Open version B opens the actual selected example summary; it does not imply a generated OMP artifact.', '- Setup/preview popovers are suppressed when setup is open. HMR edits to the central store require a browser reload during development; a normal fresh page load initializes correctly.', '', '## Evidence and follow-up','', 'See [the screenshot gallery](../docs/evidence/audit-fixes-2026-10-07/index.html) and [its coverage notes](../docs/evidence/audit-fixes-2026-10-07/README.md). Optional low-priority work remains listed in the supplied task files and finding ledger. Examples include richer raw-config syntax highlighting, advanced storage actions, and minor copy/spacing polish.','']
(r/'PROGRESS.md').write_text('\n'.join(progress))
# Preserve raw captures and publish PNG files with descriptive names plus a browsable index.
p=Path('docs/evidence/audit-fixes-2026-10-07'); captures=sorted(x for x in p.glob('*.jpg') if not x.name.startswith('contact-sheet-'))
manifest=[]
for f in captures:
 im=Image.open(f); dest=f.with_suffix('.png');im.save(dest)
 manifest.append({'file':dest.name,'width':im.width,'height':im.height,'source':f.name})
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
style='body{margin:0;background:#f4f5f7;color:#182322;font:15px system-ui}header{padding:32px}h1{margin:0 0 12px}p{max-width:880px;line-height:1.6}main{padding:0 24px 40px;display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}figure{margin:0;padding:12px;background:white;border:1px solid #d6dede;border-radius:12px}img{width:100%;height:auto;border-radius:6px}figcaption{padding-top:10px}small{display:block;color:#567;font-size:12px}a{color:inherit}'
items='\n'.join(f'<figure><a href="{x["file"]}"><img loading="lazy" src="{x["file"]}" alt="{html.escape(x["file"][:-4])}"></a><figcaption>{html.escape(x["file"][:-4])}<small>{x["width"]} × {x["height"]}</small></figcaption></figure>' for x in manifest)
(p/'index.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Strata audit fixes — evidence</title><style>{style}</style><header><h1>Strata · audit fixes</h1><p>{len(manifest)} screenshots · 7 October 2026. A targeted light/dark regression review of the remediation work. Click a screenshot for its full size. The original audit remains in the neighboring audit-2026-10-07 directory.</p><p>Some captures were collected during implementation; final Model, Engine, Setup, empty-chat and reply-menu states were refreshed after fixes. This is not a complete synthetic-state matrix. See README.md for limitations.</p></header><main>{items}</main></html>')
(p/'README.md').write_text(f'''# Audit-fix visual evidence

Captured through Codex browser control against the existing local preview at http://localhost:5199 on 7 October 2026.

- **{len(manifest)} screenshots**; [open the gallery](index.html).
- Screenshots use 1280×720 for the compact-layout gate and 1654×977 for the wide split view. Exact dimensions are in [manifest.json](manifest.json).
- PNGs are published copies of the browser's JPEG captures. Raw JPEG files are retained; conversion does not add detail.
- `contact-sheet-*.jpg` are thumbnail review aids, not individual screen evidence.
- Original evidence remains unchanged in `../audit-2026-10-07/` (420 PNGs, 210 paired states).

## Coverage

Model overview/context; Engine beginner/Expert/unsupported/raw/invalid/auto-tune; Setup computer/version choice; chat empty/generation/completion/actions/templates/add/validation/context/subagents; preview website/code/image/PDF and tab closing; Time-skip building/completion/result; Serving overview/guide/Cloudflare/local-sharing dialog; Monitor; Skills; Tasks creation/repeat/weekly/list/detail; Settings overview/search.

Main captures were visually reviewed using contact sheets, with full-size inspection of final Engine, menu, preview and setup states. Accidental transition/incorrect-route captures found in review were replaced. Some earlier screenshots predate the final dark primary-token adjustment; use the final Engine/Model/Setup/empty-chat screenshots as the final control-color reference.

## Limits

This is a targeted regression set, not all 210 original states recaptured. Model profiles, comparison and model-picker states were removed. No real key/network-sharing changes were performed. Browser access to an isolated read-only fixture server was blocked, so synthetic unavailable-hardware, failed-load/prepare, service-outage and queued/shared-active variants were not all visually recaptured. Their changes have source and relevant logic-test coverage. Motion-budget performance was not measurable by the existing harness; no FPS/performance claim is made.

See [implementation progress](../../../audit-fixes/PROGRESS.md) and [finding dispositions](../../../audit-fixes/FINDING-STATUS.md).
''')
# Refresh thumbnail review sheets after the final replacements.
for start in range(0,len(captures),20):
 sheet=Image.new('RGB',(1440,1150),'#333333');d=ImageDraw.Draw(sheet)
 for i,f in enumerate(captures[start:start+20]):
  im=Image.open(f);im.thumbnail((350,197));x=(i%4)*360;y=(i//4)*230;sheet.paste(im,(x,y));d.text((x+3,y+199),f.stem,fill='white')
 sheet.save(p/f'contact-sheet-{start//20+1}.jpg')
print(f'{len(ledger)} dispositions, {len(manifest)} screenshots, progress + gallery written')
