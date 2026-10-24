# Browser annotations 1–9

Implemented in the restored Strata source served at `http://localhost:5160`.

1. Workspace chat timestamps share a fixed, right-aligned column.
2. The chat header displays the loaded model family without its quantization.
3. The header counts working subagents. Its dropdown displays names, descriptions, statuses, tokens and live elapsed times. Queued and finished agents are excluded from the running count. No Standard mode label is displayed.
4. Task creation and editing use Instructions and Schedule steps. Both fit within the existing 560px modal; long instructions scroll inside the text field.
5. The Add menu contains Add files or photos, Add folder and Slash commands. Cmd/Ctrl+Shift+A opens the file picker.
6. The composer places the microphone after Add, followed by Auto/Manual/Accept edits/Plan/Bypass permissions modes.
7. The right composer control displays Off/Low/Medium/High/Max reasoning effort. Its menu identifies the loaded model family.
8. Skills, Monitor, Engine, Model and Settings use the chat header's surface, bottom border and height.
9. The area surrounding the composer uses the page background and has no glass backdrop.

## Verification

- `npm run typecheck` and `npm run build` passed.
- `node --test tests/agent-updates.test.mjs tests/tasks-and-sharing.test.mjs`: six tests passed.
- Browser checks confirmed timestamp right edges at x=207, all six headers at 64px with matching background and border, and both task steps with equal scroll/client heights (400px).
- Checked preferences across a reload, all requested menu choices, slash commands, microphone preview, file and folder pickers, split-view composer fit, light/dark themes, and advancing subagent durations.
- No browser console errors or warnings were observed.
- Screenshots are in `docs/evidence/annotations-followup-2026-10-09/`.

## Current engine boundaries

- The subsequent [19-annotation review](annotations-19-2026-10-09.md) connected dictation to browser speech recognition. It requires microphone permission and browser service support.
- Mode and reasoning selections persist per chat as frontend preferences. Backend permission and inference policies are not changed by these controls.
- Subagent data can arrive as `agent` SSE events or NDJSON chat frames with `sessionId` and an `agent` object. Agent fields are merged by `id`; supported statuses are working, queued, completed, failed and cancelled. Tokens and durationMs are nonnegative numbers; optional updatedAt is a timestamp in milliseconds. The current demo engine does not orchestrate real subagents.
- In demo mode, `/subagents` creates a clearly labeled example chat to inspect the dropdown.
- Text files and supported folder contents are added to the prompt. Photos and PDFs open local previews; multimodal inference is not implemented.
