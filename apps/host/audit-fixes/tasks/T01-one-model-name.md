# T01 · One model, one name

**Phase 1** · 11 findings (6 high, 2 medium, 3 low) · Depends on: nothing, can start now

## Goal
The loaded model is shown under six different names, and the UI still offers model pickers. Create one source of truth and one formatter, and remove every model picker. The only remaining choice is the version (quantisation), which moves to the Model page in T09.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/lib/format.ts` (add formatModelName)
- `src/lib/store.ts / src/lib/types.ts` (single loaded-model record)
- `src/components/ComposerControls.tsx`
- `src/components/ChatToolbar.tsx`
- `src/components/ModelCatalog.tsx`
- `src/app/Sidebar.tsx`
- `src/screens/Chat.tsx`
- `src/screens/Tuning.tsx`
- `src/screens/Monitor.tsx`
- `src/screens/Serving.tsx`
- `src/screens/Setup.tsx`
- `src/components/TaskEditor.tsx`
- `server/catalog.mjs, server/models.mjs` (names returned by the host)

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | High | `chat-empty` | Composer says "Qwen2.5 0.5B", sidebar says "Qwen3.8-Flash-Next UD-Q4_K_M" | one formatter from store.loaded |
| 2 | High | `chat-model-picker` | "Choose a model" dialog + "Browse all models" — product serves one model; shows Qwen2.5-0.5B-Instruct | remove; replace with read-only version chip → Model page |
| 3 | High | `chat-response-complete` | Reply says "Running on Starter · 0.5B Q4_K_M" — a fifth name for the model | same single formatter |
| 4 | High | `models-*` | Page is a multi-model library (Recommended profiles, Model library, Discover, families, Compare) for a product that serves one model; running model shows "Qwen2.5-0.5B-Instruct" | replace with single-model version ladder (Model.dc.html) |
| 5 | High | `engine-*` | Header "Qwen2.5-0.5B-Instruct · Q4_K_M" + "Change model" button | version chip linking to Model page |
| 6 | Medium | `engine-adapter-details` | Sidebar says "No model loaded" while the Engine header still names Qwen2.5-0.5B | header reflects store.loaded; empty state when none |
| 7 | Medium | `monitor-*` | Header subtitle shows "Starter · 0.5B Q4_K_M"; sidebar card switches between "Qwen3.8-Flash-Next UD-Q4_K_M" and "Qwen2.5-0.5B-Instruct Q4_K_M" depending on state | single formatter |
| 8 | High | `setup-model` | Step 2 is a multi-model flow: profile cards (Agentic/Balanced/Performance), "Your choice Qwen2.5 0.5B", "Change model" opening the full library modal | single model: "Pick a version" with the recommended one pre-selected (reuse Model page ladder) |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `models-file-details` | Mono file source wraps mid-token ("0.5B-\nInstruct-GGUF") | ellipsis + copy button, or break at "/" |
| 2 | Low | `serving-requests-all` | Model ID "qwen2.5-0.5b-q4km" (yet another model name) | from single formatter / real served id |
| 3 | Low | `tasks-new-menu` | "Create with Qwen2.5" names a model in a menu label | "Create with AI" / "Describe it" |

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] Every place that shows the model uses formatModelName() and renders "Qwen3.8-Flash-Next · <Version> (<QUANT>)", e.g. "Qwen3.8-Flash-Next · Balanced (Q4_K_M)".
- [ ] `grep -rniE "qwen2\.5|0\.5b|starter ·" src server` returns nothing outside test fixtures.
- [ ] No "Choose a model", "Change model" or "Browse all models" control remains anywhere.
- [ ] Add a unit test for formatModelName in tests/.
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `chat-empty-light.png`
- `chat-model-picker-light.png`
- `chat-response-complete-light.png`
- `models-file-details-light.png`
- `engine-adapter-details-light.png`
- `serving-requests-all-light.png`
- `tasks-new-menu-light.png`
- `setup-model-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t01-one-model-name` with message `T01: One model, one name`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
