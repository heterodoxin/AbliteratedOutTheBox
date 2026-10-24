# Agent bridge (Strata chats as DeepSeek Harness sessions)

The Harness is a separate, pinned component (`DSH 0.2.1-alpha.3`, commit `57190202`, MIT). The host finds it through
`STRATA_HARNESS_DIR` (or `<STRATA_COMPONENTS_DIR>/harness`), starts one `dsh --profile web` process on a loopback port,
and talks to its Gateway. Nothing here downloads or builds anything.

| File | Job |
| --- | --- |
| `harness-runtime.mjs` | locate the component, start/stop the process, scrub its environment, clean up a leftover process from a dead host |
| `gateway-client.mjs` | HTTP RPC plus the `remote.mux` WebSocket (`$events`, `session/follow`) |
| `profile.mjs` | generate the Harness profile: one route per ready local instance and per saved API connection; credential names only |
| `task-run.mjs` | a scheduled task as the agent |
| `events.mjs` | pure projection of Harness session events into the chat's NDJSON frames |
| `children.mjs` | pure projection of one helper agent's events into its row and its approval card |
| `preset.mjs` | the agent preset (which tools the chat has) |
| `plugins/strata-gate.mjs` | the Manual-mode approval gate (a Harness plugin, loaded by path) |
| `plugins/strata-context.mjs`, `context-budget.mjs`, `context-fit.mjs` | measured context pressure, summaries that fit the window, the check before a message goes out |
| `agent-bridge.mjs` | the chat-to-session map, turns, approvals, Stop, status |

## Frames sent on `/api/chat/stream` for an agent turn

`queued`, `start` (`agent:true`), `reasoning`, `delta`, `agent-tool` (new: upserts a tool card by call id with
`running | proposed | done | denied | error`), `done` (usage, `finishReason`, `cancelled`), `error`, and `notice`
(why a chat is not using the agent). The chat UI renders them with its existing components.

## Routes

- `GET /api/agent/status`: component, routes, active turns, pending approvals (no tokens, no keys)
- `POST /api/chat/approval {sessionId, toolId, decision: approve|deny}`
- `POST /api/chat/stop {sessionId}`: cancels that chat's Harness session and answers once the turn has ended
- `GET /api/agent/pending?chatId=`

## Behaviour worth knowing

- **Stop is per chat.** `session/cancel` is only an admission; the bridge waits for the durable `turn/end`. Stop before the
  prompt reached the Harness cancels the start. The process is never killed to implement Stop.
- **A closed tab is not Stop.** The turn keeps running; an approval it needs stays pending until someone answers.
- **Resume.** The chat id maps to a Harness session id (`<data>/agent/sessions.json`). Reopening prompts that session; the
  Harness resumes it from its own record, so nothing is replayed. (`session/create` with a started id returns a new
  session, so it is never used for resume.)
- **Models.** Each chat sends its choice; the bridge calls `session/selectModel` for that route before the prompt. A route
  that is gone is a typed `model-unavailable` error, never another model. Keys travel as environment variables named by
  `apiKeyEnv`; the agent's shell does not inherit them, and logs are redacted.
- **No answerer, no hang.** The agent's own questions (`ask_user_question`) are declared unanswerable until the app has a
  surface for them.
- **Fallback.** Without the component the host says so (`/api/agent/status`, a `notice` frame) and the chat runs on the
  direct path with the model the user picked.

## Modes (the Mode menu)

The chat's Mode menu is real Harness behaviour. Each turn the page sends the chat's mode (`mode` in the `/api/chat/stream` body); before the prompt reaches the Harness the bridge puts that chat's session into the matching permission preset with the Harness's own `/permission` command (`commands/execute`), reads the `permissions` projection back, and refuses to start the turn (typed error `mode-unavailable`) if it does not match. The preset table is ours (profile entry `permission`).

| Menu | Preset | Sandbox | Approval | What it means |
| --- | --- | --- | --- | --- |
| Manual (default) | `strata-manual` | open (`danger-full-access`) | `ask` | The Strata gate (`plugins/strata-gate.mjs`) asks BEFORE any change runs: one Approve/Deny card, nothing has happened when it appears, no blocked attempt, no sandbox or PowerShell error. Reads and read-only commands run without a question; anything the gate cannot prove read-only asks. |
| Plan | `strata-plan` | `read-only` | `never` | Nothing can change. A request is rejected without asking anyone, so no Approve card is shown; the card ends "Not allowed in this mode". |
| Bypass | `strata-bypass` | `danger-full-access` | `never` | Full access without prompts, as before. |

- Manual relies on the gate, so a Manual turn does not start unless the bridge has confirmed the gate is loaded (`/strata-gate` command): without it Manual would be an open sandbox that nobody asks about.
- The gate listens to `tools/pre-execute` (the seam the Harness's own Auto-review layer uses) and returns an `ask` decision; the Harness's approval service then asks, runs the tool only after `allowed-once`, and records the answer. Read-only tools and a short allowlist of look-only shell commands (Get-ChildItem, git status, ...) run unasked; MCP and unknown tools ask.
- Web search and web pages (`web_search`, `web_fetch`) ask once per chat in Manual, not once per call. The gate asks through the approval service itself (`webDecision`: "The assistant wants to search the web for: ... Search words go to <service>." / "... open the web page ..."), remembers a yes for the chat (a helper agent shares its chat's answer, and two searches started together share one card), and on a no gives the model `WEB_REFUSAL` ("The person did not allow web access ... Go on without it and say so."). Plan and Bypass are unchanged. Settings -> Tools has a switch, "Let the assistant search the web without asking" (off by default, `webWithoutAsking` in `tools-settings.mjs`, passed to the gate as `config.web.skipAsk`), that skips the question in Manual. A scheduled task has nobody at the card, so its web call is answered no unless that switch is on. Tests: `tests/web-ask.test.mjs`.
- The mode is read once per turn; an unknown or missing value is Manual.

## Tools

`preset.mjs` is the agent preset: the Harness `standard` preset minus the scheduler (Tasks is the only one), the agent's own question tool (no place to answer) and the Harness's plan mode. `tests/agent-tools.test.mjs` compares the list with the pinned Harness's own file, so a Harness update that adds or renames a plugin fails a test instead of silently dropping a tool.

### Settings -> Tools

What the person chose is saved by `tools-settings.mjs` (inside the host's settings file) and shown by `src/components/SettingsTools.tsx` (`GET/PUT /api/agent/tools`, `POST/PATCH/DELETE /api/agent/tools/custom[/id]`). A change reaches the agent from the next message: the profile carries a `toolsSignature`, and the bridge restarts the Harness (after the running turns end) when it differs.

- Tool groups (`tools-catalog.mjs`): each is one switch in plain words and is one or more preset plugins (`withoutDisabled`); `fetch` and `search` are the two halves of the Harness's `tool-web` (both off drops it). Goals off: `/goal` answers that goals are turned off.
- Web search needs no account. The default is DuckDuckGo's HTML result page (`../web-search-ddg.mjs`), with Bing's plain result page (`../web-search-bing.mjs`) as the second source. Manners, because DuckDuckGo refuses a program that asks fast (a real box: two searches in 1.2 s answered, the third got HTTP 202 and every DuckDuckGo page refused for many minutes): a browser User-Agent and the page's own request shape (Referer, Origin, Sec-Fetch-*), one request at a time per source, 2.5 to 4 s apart (random), at most 8 a minute, a 30 minute session cache per question, a cooldown after a refusal that doubles (1, 2, 4 ... 10 minutes, `Retry-After` honoured) during which that source gets no request, no lite-page follow-up after a refusal, one retry for a timeout or a 5xx, a 15 s limit. When every source refuses, the tool answers one plain sentence (`LIMITED_SENTENCE`: "Web search is limited right now. Try again in a minute, or add a search key (Exa, Perplexity or DeepSeek) in Settings -> Tools -> Web search.") plus an instruction to the model not to retry; the turn guard in `agent-bridge.mjs` (see Turn guard) stops a model that retries anyway. **The results come from DuckDuckGo or Bing**; they are not ranked or filtered here. The person may switch to Exa, Perplexity or DeepSeek with their own key (`../web-search-keyed.mjs`; DeepSeek uses the Harness's own provider): the key is saved in the settings file, reaches the agent only as the environment variable `STRATA_SEARCH_KEY`, never the profile file and never the screen (it only learns `keySet`). A keyed service without its key falls back to DuckDuckGo. The profile pins `web.searchProvider` to the chosen id so two usable backends cannot make the choice ambiguous.
- The browser: Playwright's MCP server (`@playwright/mcp`, staged next to the Harness by `apps/desktop/scripts/stage-harness.mjs`) on the person's own Chrome, Edge or Chromium (`browser-probe.mjs`), `--isolated` so it never sees their profile. Tools appear as `mcp__browser__browser_*`. In Manual, opening a page, clicking, typing, uploading, running a script in the page and closing tabs ask (plain reasons: "Open the web page ...", "Click on the page: ..."); looking (snapshot, screenshot, console, network, waiting, listing tabs) does not. Plan refuses every connected tool. The screen says plainly when no browser, or no tool server, is found.
- The person's own tools: `instructions` (the tool returns their text; look-only, never asks), `command` (runs their command with the assistant's text in the `STRATA_INPUT` environment variable, never spliced into the command line; asks in Manual, refused in Plan) and `server` (a web address or a program, added as `mcp__custom-<name>__*`; asks in Manual). Instruction and command tools are served by `custom-tools-mcp.mjs` (stdio MCP) from a JSON file the settings module rewrites on every change, so a switch in Settings is seen without a restart.
- `strata-gate.mjs` implements the Manual/Plan rules above (`planDecision`, `browserDecision`, the look-only instruction server).

## Helper agents (subagents)

- A Harness `subagent/catalog` event on the chat's session creates a row; `children.mjs` folds the helper's `session/follow` (address `{kind:'subagent', parentSessionId, childSessionId}`) into `{type:'agent', agent:{id,name,description,status,durationMs,tokens,updatedAt}}`, the shape the Subagents popover already reads. The delegation tools have no card.
- A helper's request for approval is the same card (state `proposed`, id `<helper session>:<call>`), answered with the same `/api/chat/approval`; a chat can only answer its own helpers.
- The run (the chat's one NDJSON response) stays open until the parent is idle, every helper has finished and reported back, and no armed goal is waiting; `done` is sent then.
- Stop: pauses an active goal, interrupts every helper (`subagents/interruptByParent`; `session/cancel` on a helper session is refused by the Harness), cancels the parent, waits; a report that would wake the parent afterwards is cancelled too.
- The number of helpers at once equals the engine's request slots (`maxActiveSubagents` in the profile); the Harness refuses more with a message the model reads.

## /goal

`/goal` typed in a chat goes to `bridge.goalCommand`: nothing = show the goal, `clear`, `pause`, `resume`, `edit <text>`, `complete`, anything else = a new goal (one unfinished goal at a time). Create/resume use the Harness goal Remotes (`goals/create|get|resume|pause|edit|clear|complete`); the Harness's goal driver runs the rounds and they stream into the chat ("Working on the goal…", with no step counter); the run ends when the goal stops (complete, paused, blocked, round limit - the default cap is the Harness's 256, a run with `maxRounds` is capped) and says how in plain words. After a restart an active goal stays disarmed until `/goal resume`.

## Thinking and the output budget

- A local route is reasoning-capable (`reasoning: true`) with `compat.thinkingFormat: 'chat-template'`: the request carries `chat_template_kwargs {enable_thinking, reasoning_effort}`, which the Strata server reads (a top-level `enable_thinking` is ignored by it, which is why pi-ai's `qwen` format left the model thinking). The chat's Reasoning menu travels as `thinking` in the stream body; the bridge sends it with `session/selectModel` (`reasoningEffort`) whenever the model or the level changed; a level a route does not have is dropped (never an error). API routes keep the provider default (the menu is disabled for them in the UI).
- `localOutputBudget(context)` = context minus 4096 (it was context/4 capped at 8192, which cut big answers off at exactly 8192 tokens); pi-ai sends the window minus the prompt. App-loaded models get `fit_max_tokens: true` so the engine clamps instead of answering 400. Sampling is not set here: the Harness sends none, so the server's own defaults apply.

## Tasks

`task-run.mjs` runs a scheduled task as the agent: a Harness session of its own (`task:<id>`, dropped after every run), the task's mode (`plan` default, `manual`, `bypass`), and a result built by `../task-report.mjs`. Nobody is at the screen: an approval request is answered "no" at once and reported ("Nothing was changed"). `server/tasks.mjs` stays the only scheduler; it sends a `task` event when a run ends (the page shows a toast).

## Plain words

Failures the person sees are plain sentences with a next step (`AgentUnavailable.message`); the technical text is in `.details` / the `details` field of error and notice frames, and in `/api/agent/status` `detail`. A failed tool's card result is a plain sentence followed by "Details: <original text>".

## Tests

`tests/agent-events.test.mjs` (pure), `tests/agent-tools-settings.test.mjs` (Settings -> Tools, web search, the browser and the person's own tools; the Harness-backed ones need STRATA_HARNESS_DIR and the browser one also Chrome or Edge), `tests/agent-bridge.test.mjs`, `tests/agent-modes.test.mjs` and `tests/agent-host.test.mjs` (need a built Harness:
`STRATA_HARNESS_DIR=<checkout>`, otherwise they skip and say so). The model in those tests is `tests/helpers/scripted-worker.mjs`.
`scripts/agent-flow.mjs` drives the real app in Chrome and takes screenshots.

## Helper agents on a named model (`plugins/strata-models.mjs`, `model-routes.mjs`)

The Harness's `subagent` tool takes an exact `provider` route key and model id, and the app had its model choice switched off, so every helper ran on the chat's model. The profile now enables the Harness's `subagent-model-selection-settings` with every route the app can reach (local instances and saved API models), and the `strata-models` plugin wraps the two tool definitions the Harness registers for each chat (`subagent`, `list_subagent_models`) in place:

- the model is found from its id, `provider/id`, the name in the model picker, `connection name`, or "local" (any letters). A close guess is not started: the tool answers "Did you mean ...?" and nothing runs until the call is made again with that model. Several fits are listed; nothing fitting lists the models that can be used and says they can be added in Models.
- the tool's own text says when to use it ("ask a subagent", "ask model X"), not to look for the model with shell commands, and not to wait with sleep. The system prompt (`preset.mjs`) says the same, so it reaches every model, local or hosted.
- `list_subagent_models` lists the models by the names the person knows.

If a Harness update renames those tools the wrapping does nothing and `tests/agent-subagent-models.test.mjs` (real Harness) fails.

## Reasoning for API models

A model on OpenRouter whose model list names `reasoning` in `supported_parameters` gets `reasoning: true` and the efforts `{off: none, low, medium, high, max: xhigh}` and the `openrouter` thinking format, so the Reasoning menu works and the choice reaches OpenRouter as `reasoning: { effort }`. Other API models keep the provider's own default (the menu stays greyed).

## Pictures

Pictures are on by default for every model that can read them: an API model reads them unless the person switched it off (Settings > Models > "Read pictures with this model") or its server says it cannot; a local model reads them when it was loaded with Read pictures (on by default). `src/lib/pictures.ts` is the one answer used by the composer and by Send; the agent route declares image input from the same flag.

## Turn guard (`turn-guard.mjs`)

A turn that keeps failing the same way is stopped by the app, never left to the model. Seen on a search-heavy `/plan`: every web search was refused, the model searched again and rewrote its plan after each one until the context reached 100%, then the turn ended with no answer and no message.

- **Counted, from what the turn shows:** web searches that failed in a row (3), searches in one turn (10 in a plan turn, 25 otherwise), a plan turn that starts its numbered list from `1.` again (4 times), and a context of 93% or more (the Harness's own compaction did not keep up).
- **Stopped the way Stop stops** (`session/cancel`, helpers first) but ended as a normal turn (`done`, not cancelled, no error card), so the plan the model had written is still proposed. The plan card carries the reason (`plan.caution`); any other turn ends with one `delta` frame that has `own: true` and `key` (the English sentence, translated by the page; `GUARD_NOTES` are in the interface catalog in all eight languages). Each sentence says what happened and gives one next step.
- **Compaction:** after a guard stop (and after a turn the model server refused because the context was full) the bridge asks the Harness for `/compact` on that session, so the next message has room. Best effort and logged.
- **Never silent:** a turn that ends (`completed` or `max-tokens`) with no answer text gets "I ran out of room in this chat..." when the last request used 85% of the window or more, otherwise "I stopped without writing an answer..." (not when tool cards were shown, not after Stop, not after an error, which says why by itself). Goal runs have their own ending lines and are not guarded.
- Tests: `tests/agent-turn-guard.test.mjs` (counting, always; end to end with the real Harness, the scripted model's `searchloop` and `bloat` scripts and `searchMock: 'limited'` on the bridge, which runs the real search code on a pretend web that refuses, no network).

## Context: measured pressure, summaries that fit, the check before a message goes out (2026-10-22)

Found by the owner's hands-on test of the installed 0.2.1 (Gyro-S, 64K): a 118 KB log paste was 66,500 tokens to the engine and about 29,500 to the Harness (4 characters a token; logs and code are about 2); the engine refused it with a 400, the refusal stayed in the history, and then neither the next message nor `/compact` could go through, because the summary request carried the same history and was refused for its size too. `/compact` itself was never a command of the app: it went to the model as a question and the model answered "nothing to compact".

- **`plugins/strata-context.mjs`** (inserted by the profile, like the gate) wraps `tokenMeter.measure` and `llm.stream` in place, once each. Pressure = the last prompt size the model server reported for an answer, plus what changed since priced at the density that measurement showed (`context-budget.mjs`: `pressureNow`, `densityOf`); never lower than the Harness's own price. Node prices are scaled the same way, so the retained tail is in real tokens. A summary request (`purpose: 'compaction'`) that would not fit the window is asked for in pieces: a message bigger than a piece is cut for the request only, each piece builds on the summary before it, a piece the server refuses is halved and tried again. `/strata-context` (a Harness command, like `/strata-gate`) answers `{tokens, window}` for a session so the host can tell the person true figures.
- **`context-fit.mjs`**: before a message is sent the model server counts it (`POST /v1/messages/count_tokens`, its own tokenizer). `decideFit`: send; summarise first and then send (the message fits an otherwise empty chat); or never fits. The last one is not sent: the error frame has `code: 'CONTEXT_MESSAGE_TOO_LARGE'` and the page offers "Attach as a file" (`oversize.mjs`, `/api/chat/oversize`: the text is saved in the chat's folder, the assistant reads it in parts) or "Send a shortened preview". An engine that cannot count lets the message through; the Harness's own compact-and-retry is the net.
- **`/compact`** is handled by the host (`compactCommand`): the Harness's `/compact`, then a plain line with the measured figures ("The chat was 90% full and is now 38% full"), or "nothing older to summarize" when the newest message is what fills the chat.
- **Visible:** `compaction/start` shows "Summarizing earlier messages..." in the status line; `compaction/end` adds one plain line to the reply. The projector also sends a `context` frame after every model request, so the meter follows the running turn.
- **A refusal that is left** (`CONTEXT_WINDOW_EXCEEDED`) is a plain sentence with the engine's text under Details and the buttons "Summarize and continue" and "Start a new chat". "Try sending the message again" is for other errors only.
- Tests: `tests/agent-context.test.mjs` (arithmetic; the real Harness against the scripted worker with `window`, `charsPerToken` and `noCount`), `tests/oversize.test.mjs`, `tests/context-ui.test.mjs`. The run against a real engine is `scripts/context-e2e.mjs`.
