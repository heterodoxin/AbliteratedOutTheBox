# T11 · Chat: remaining findings

**Phase 3** · 38 findings (0 high, 20 medium, 18 low) · Depends on: T01, T05, T06, T07

## Goal
Fix the remaining Chat findings after the shared fixes have landed. Skip any finding that a phase 1 task already resolved.

## Files to change
These are the likely files. Read them before editing, and change others only if the fix needs it.
- `src/screens/Chat.tsx`
- `src/components/ChatToolbar.tsx`
- `src/components/ComposerControls.tsx`
- `src/components/ArtifactCard.tsx`
- `src/components/ChatMedia.tsx`
- `src/components/SubagentRow.tsx`
- `src/components/Markdown.tsx`

## Findings to fix (high and medium)
| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Medium | `chat-empty` | "0 subagents ▾" dropdown with zero items | hide at 0 |
| 2 | Medium | `chat-empty` | Disabled Send = pale green on pale green, reads as selected | neutral disabled style |
| 3 | Medium | `chat-generating` | Header title clipped mid-word ("…short paragrap") with no ellipsis | text-overflow: ellipsis + title attr |
| 4 | Medium | `chat-model-picker` | "Running" rendered as a disabled outlined button | status chip (green dot + text), not a button |
| 5 | Medium | `chat-template` | Template cards carry a ▾ chevron, implying expand, but the action is "pick" | remove chevron; whole card selectable, "Use template" on hover |
| 6 | Medium | `chat-session-log` | Dialog for an unbuilt feature ("will be connected in a future build"), only "Done" | hide the entry until it works |
| 7 | Medium | `chat-slash-commands` | /new and /clear have identical descriptions | merge, or /clear = "Clear this conversation" |
| 8 | Medium | `chat-slash-commands` | /time-skip "Try the two-stream UI demo · add a prompt" — internal jargon | "Try two versions and keep the better one" |
| 9 | Medium | `chat-subagents-statuses` | Failed subagent ("Preview server did not respond") offers no action | Retry / View log on failed rows |
| 10 | Medium | `chat-context-length` | Context limit shows 128K; model supports up to 1M (Engine setting) | read limit from engine config |
| 11 | Medium | `chat-approval-collapsed` | "▶ Read project notes · done" uses native ▶ and bold H3 weight — louder than the approval card it sits above | small grey row with chevron icon, like tool calls elsewhere |
| 12 | Medium | `chat-approval-approved` | Approved change collapses to "· done"; denied says "· denied" — asymmetric, and "done" hides that a write happened | "Approved" (green check) / "Denied" (grey x) |
| 13 | Medium | `chat-artifact-cards` | File cards: subtitle "Website · Open in sidebar" reads like a link but isn't; chevron opens a menu — primary click unclear | whole card opens in sidebar; ••• menu for the rest |
| 14 | Medium | `chat-message-actions` | Hover "branch" icon button sits on top of the user bubble and clips its text ("…a conversat") | place actions below the bubble with the other icons |
| 15 | Medium | `chat-edit-branch` | A new branch shows the empty-state hero + suggestions while the composer holds the edited message | hide empty state in branches; show "Editing from: …" note above composer |
| 16 | Medium | `chat-add-menu` | + menu mixes demo entries ("Try preview examples", "Try chat showcase") with real actions; no way to attach a file to the message | Attach file / Open website; move demos to Help |
| 17 | Medium | `chat-tool-access` | Read only / Workspace write / Full access with no description; Full access not marked as risky | one-line description each; amber for Full access |
| 18 | Medium | `chat-no-conversations` | Two different empty states: this one (icon, small heading, "Ask a question, work with text…", extra New chat button) vs chat-empty (big heading, suggestions, different subline) | one empty state; drop the redundant New chat button (composer is already live) |
| 19 | Medium | `chat-response-complete` | Markdown not rendered: "*where*" shows literal asterisks | fix inline emphasis parsing |
| 20 | Medium | `chat-response-complete` | After a reply, stats show "60 tokens/s · — total tokens · 1%" — total missing | populate total; label 1% ("1% of context") |

## Optional polish (low)
Do these only if the task is otherwise done and the budget allows.

| # | Severity | State | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | Low | `chat-empty` | 3 suggestion cards in 2-col grid leave an orphan | 3-col or 1 row of 3 |
| 2 | Low | `chat-generating` | Two chats both titled "Chat showcase · code, to…" in list | dedupe titles / show distinguishing snippet |
| 3 | Low | `chat-generating` | Unexplained green dot on "Build a calm website…" list item | legend/tooltip ("Unread" / "Running task") |
| 4 | Low | `chat-template` | Templates duplicate but differ from empty-state suggestions (two sets of starters) | single set of starters |
| 5 | Low | `chat-slash-commands` | /setup "choose model", /model "Open Models" — plural/multi-model language | "/model Open Model" |
| 6 | Low | `chat-slash-commands` | Send turns active with just "/" typed | keep disabled while the input is only a command prefix |
| 7 | Low | `chat-open-website-invalid` | "Open preview" same pale-green style whether valid or not | disabled style until valid |
| 8 | Low | `chat-showcase-code` | Text "More" under assistant message next to icon actions | ••• icon button |
| 9 | Low | `chat-session-statistics` | "LLM time" jargon; "Turns / tool calls 1 / 2" slash-pair | "Thinking & writing"; separate rows |
| 10 | Low | `chat-context-length` | Footnote about compaction "not connected in this build" | remove until built |
| 11 | Low | `chat-approval-collapsed` | Expand chevron floats mid-row, detached from title and from actions | put chevron at far right or next to title |
| 12 | Low | `chat-approval-approved` | Section heading "A change waiting for you" stays after it's resolved | update heading or treat as static demo copy |
| 13 | Low | `chat-gif-paused` | With preview open, chat column narrows and the composer wraps the model chip onto its own row; visible scrollbar between panes | collapse composer controls into icons at narrow widths |
| 14 | Low | `chat-artifact-cards` | Blockquote "A calmer place…" is indented with no quote bar, reads as misalignment | 3px left rule + muted text |
| 15 | Low | `chat-artifact-actions` | Menu says "Show in file explorer" while copy above says "Finder" | OS-aware label |
| 16 | Low | `chat-message-actions` | "‹ Latest" pill doesn't say where you are | "Branch 1 of 2 · Back to latest" |
| 17 | Low | `chat-edit-branch` | Title gets "(branch)" appended in parentheses | branch icon chip after title |
| 18 | Low | `chat-thinking-menu` | Auto/Low/Medium/High with no hint of trade-off | one-line hint each ("Faster" … "Most careful") |

## How to check
Run these and fix anything that fails:
- `npm run typecheck`
- `node --test tests/` (all existing tests stay green)
- `npm run check` (typecheck, UI audit, contrast, measure, luminance)
- `npm run build`

Task-specific checks:
- [ ] Each finding above is resolved in the running app, in light and dark mode.
- [ ] Each finding above is resolved, or marked "already fixed by Txx" or "not reproducible" with a one-line reason.

Screens to re-capture and compare with the audit screenshots in `docs/evidence/audit-2026-10-07/`:
- `chat-empty-light.png`
- `chat-generating-light.png`
- `chat-model-picker-light.png`
- `chat-template-light.png`
- `chat-session-log-light.png`
- `chat-slash-commands-light.png`
- `chat-open-website-invalid-light.png`
- `chat-showcase-code-light.png`
- `chat-subagents-statuses-light.png`
- `chat-session-statistics-light.png`
- `chat-context-length-light.png`
- `chat-approval-collapsed-light.png`
- `chat-approval-approved-light.png`
- `chat-gif-paused-light.png`
- `chat-artifact-cards-light.png`
- `chat-artifact-actions-light.png`
- `chat-message-actions-light.png`
- `chat-edit-branch-light.png`
- `chat-add-menu-light.png`
- `chat-thinking-menu-light.png`
- `chat-tool-access-light.png`
- `chat-no-conversations-light.png`
- `chat-response-complete-light.png`

## Done when
- All checks pass.
- Commit on branch `fix/t11-chat` with message `T11: Chat: remaining findings`.
- Append a short entry to `audit-fixes/PROGRESS.md`: what changed, which findings were fixed, skipped or not reproducible, and anything the reviewer should look at.
