# Strata UI improvements

**Date:** 5 October 2026  
**Status:** UI and supplied-brand implementation completed for the current test host; latest findings and evidence in section 24. Full native-engine hookup is deferred by the user; platform/accessibility/performance verification limits remain explicit.  
**Audience:** Product designer and engineer implementing the existing Strata interface  
**App:** `design-iteration/strata`  
**Design source:** `design-iteration/ui`  
**Brand source:** `design-iteration/Strata-Logo-DesignFiles` — supplied Brand Guidelines v1.0, October 2026

## Summary

Make Strata understandable on first use, quiet during everyday work and complete for experts. Keep its identity: GT Walsheim, Anron icons, green accent, existing spacing and radius tokens. Improve hierarchy and interactions using Apple's Liquid Glass principles, with restrained materials around controls and opaque surfaces for reading.

The implementation pass first corrected misleading success states, raw configuration saving, reset scope and setup readiness. It then simplified navigation and each screen, while retaining technical details through local disclosures. Production-engine dependencies are listed in sections 21 and 23. Section 24 records the later UI-focused completion pass and the user’s decision to defer full native-engine hookup.

Use progressive disclosure throughout the app. Everyone gets the same simple starting view, with technical details close to the operation they describe. Engine tuning alone retains an explicit **Beginner / Expert** pair.

This document is the consolidated audit and implementation record for all existing screens and shared components. Sections 1–19 preserve the original specification; section 20 records the initial browser findings before fixes. The subsequent authorized implementation pass changed the React interface and the local design-host contracts needed to make its controls truthful. Section 21 records the 120 findings and their implementation history. Section 22 adds 12 brand requirements, source inventory and acceptance criteria. **Section 24 is the latest status update** for affected findings and the brand requirements; it supersedes earlier pending statements. Section 23 reconciles the prototype with the actual Strata repository and records a more precise implementation handoff. There are **132 tracked items: 120 existing findings plus 12 brand additions**, not 132 unresolved defects. The section 23 observations explain existing dependencies and add no new issue IDs. No real model was downloaded or loaded.

## Contents

- [1. Evidence, priorities and boundaries](#1-evidence-priorities-and-boundaries)
- [2. Design decisions](#2-design-decisions)
- [3. App shell and navigation](#3-app-shell-and-navigation)
- [4. Launch and connection](#4-launch-and-connection)
- [5. First-run setup](#5-first-run-setup)
- [6. Chat](#6-chat)
- [7. Response settings drawer](#7-response-settings-drawer)
- [8. Conversation inspector and telemetry popovers](#8-conversation-inspector-and-telemetry-popovers)
- [9. Models](#9-models)
- [10. Engine tuning](#10-engine-tuning)
- [11. Monitor](#11-monitor)
- [12. Serving](#12-serving)
- [13. Workspace](#13-workspace)
- [14. Skills](#14-skills)
- [15. Settings](#15-settings)
- [16. Shared component inventory](#16-shared-component-inventory)
- [17. Functional dependencies and interfaces](#17-functional-dependencies-and-interfaces)
- [18. Delivery order and verification](#18-delivery-order-and-verification)
- [19. Coverage and completion checklist](#19-coverage-and-completion-checklist)
- [20. Initial browser walkthrough and additional findings](#20-initial-browser-walkthrough-and-additional-findings)
- [21. Implementation status and verification](#21-implementation-status-and-verification)
- [22. Logo assets and implementation requirements](#22-logo-assets-and-implementation-requirements)
- [23. Source reconciliation and implementation handoff](#23-source-reconciliation-and-implementation-handoff)
- [24. UI-focused completion, brand integration and new verification](#24-ui-focused-completion-brand-integration-and-new-verification)

## 1. Evidence, priorities and boundaries

### How to read the issues

Each issue has an ID, priority, classification, required behavior, dependency and acceptance check. The original issue tables describe the **audit baseline**, not the current application. Source line references are historical. Section 21 gives those issues their current disposition; it supersedes the original Pending status. New BRAND issues have their own status in section 22. A design proposal is not evidence that the original implementation was broken.

| Classification | Meaning |
|---|---|
| **Defect** | The inspected implementation contradicts its label, loses expected behavior or presents misleading information. |
| **Design** | A deliberate improvement to hierarchy, wording, interaction or presentation. |
| **Dependency** | The intended interface needs a capability or data contract beyond the current frontend. |
| **Keep / existing** | Useful behavior is already implemented or recorded as shipped; preserve it and verify it during related changes. |

| Priority | Meaning |
|---|---|
| **P0** | Truth, data preservation, application scope, readiness or an interaction that advertises behavior it does not deliver. |
| **P1** | Everyday clarity, discoverability, keyboard access and understandable control of existing capabilities. |
| **P2** | Visual refinement, responsive polish, motion and screenshot presentation. |

Accessibility defects are fixed with their affected component. P2 does not mean an inaccessible screen can ship until a later polish pass.

### Evidence sources

The inspection covers all routes in `src/lib/router.ts`, the nine screen components, app shell components, shared primitives, charts, state types and relevant host behavior. It reconciles these with [the existing per-screen plans](plans/README.md), [their execution checklist](plans/INDEX-CHECKLIST.md), [interaction guidelines](interaction-guidelines.md), [design-system notes](design-system-notes.md) and [the UI kit](../../ui/README.md).

Existing screenshots and prior verification notes provide historical visual context. The initial audit used source inspection; the subsequent browser walkthrough supplies current interaction and screenshot evidence for the specific observations in section 20. Contrast compliance, animation timing and performance were not measured in the initial walkthrough. The later rendered-text checks and their limitations are recorded in section 21.

Some older documentation understates or overstates implementation. For example, expert grouping and file previews already exist, while a completed-looking setup stage is not proof that the engine loaded. Follow the current handlers and returned state when those disagree with older prose.

The brand addendum reviews the supplied eight-page [logo guidelines](../../Strata-Logo-DesignFiles/Strata-Logo.pdf), all 37 deliverable files, the existing after screenshots and responsive/theme evidence, and the current branding/static-file source. [The asset inventory](evidence/brand-audit-2026-10-05/asset-inventory.json) records dimensions, hashes and confirmed export/integration issues. All 37 inventory SHA-256 hashes were rechecked against the unchanged masters for this follow-up. This is a documentation and asset-readiness review; no logo repair or application integration is included in the report deliverable.

### Boundaries and defaults

- The initial phase was documentation only. The subsequent authorized improvement pass changed application code and the local design host, with browser evidence under `docs/evidence/ui-audit-2026-10-05/`. The UI kit and older per-screen plans are preserved.
- Preserve the React/TypeScript web app and current routes. A desktop shell, native rewrite and engine replacement are separate decisions.
- Do not add new product modules such as Time-Skip, a marketplace, account system or cloud sync in this pass.
- Keep Light as the default; add System as an available appearance choice without resetting existing users' preferences.
- Keep **More** for configuration destinations. Preserve access to Skills and Workspace.
- Keep Balanced and Recommended tuning defaults. Extra controls must not appear until their capability and effect are known.
- Prefer the host's existing contracts. The interface changes in section 17 describe the minimum missing capability, not an instruction to rebuild the backend.

### Existing work to preserve

These are not new implementation requests. The acceptance column is a regression check for future UI changes, not a claim that those checks were run during this report.

| Area | Keep / existing | Regression acceptance |
|---|---|---|
| Chat | Retry, edit, branch, continue, unread marks, jump-to-latest and conversation scroll restoration exist. | Preserve their availability while correcting target/scope behavior in CHAT-04 and CHAT-05. |
| Long conversations | `content-visibility`, pinned scrolling and focus-mode turn navigation exist. | Long-history reading does not jump when output arrives or conversations change. |
| Models | Fit labels, recommendation emphasis, estimated download cost and folded oversized variants exist. | Retain these cues while correcting wording, shared action handling and disclosure behavior. |
| Tuning | Beginner draft prediction, in-place issues, expert groups, changed counts and default markers exist. | Preserve them; complete the summary and unified-draft behavior rather than rebuilding the screen from scratch. |
| Monitor | Time windows, memory/temperature verdicts, zero-based speed range, sparkline hover and idle summary exist. | Keep readable units and thresholds while fixing provenance and unequal-throughput presentation. |
| Serving | URL copy, curl example, filtering, sorting and state glyphs exist. | They remain functional after connection guidance is moved out of the main view. |
| Workspace | Real file previews, glyphs, read markers and arrow-key tree traversal exist. | Keep tree navigation and readable file contents. |
| Skills | Source groups, instructions, triggers, usage counts and workspace scope exist. The folder action calls the host. | Keep functional folder import; replace ambiguous scope wording. |
| Settings | Section index, search, appearance reset and persisted telemetry preference exist. | Keep index/search usefulness; distinguish persistence from actual collection capability. |
| Shell | Connection banner, sidebar pin/archive/context menu and fuzzy command search exist. | Keep recovery and conversation actions discoverable. |
| Overlays | Palette, shortcuts, setup and sampling use focus management. Toasts have a cap, hover pause and a live region. | Preserve working focus return and announcements; complete missing surfaces. |
| Semantics and preferences | Screen/section headings and OS preference handling already have implementations. | Keep one semantic screen heading and test manual/system preference combinations. |

## 2. Design decisions

### Identity, typography and geometry

- Use local GT Walsheim fonts and the existing Anron-style icon sprite for interface controls. Use the supplied three-bar Strata artwork for brand identity; it is separate from the functional icon system. Section 22 replaces the old layers-glyph logo treatment.
- Retain the kit's 4-point spacing scale and 8 / 14 / 20 / 24 surface radii. Single-line controls use the established capsule treatment; multiline editors use a finite surface radius so their first and last lines remain visible (BROWSER-03). Padded nested corners remain concentric.
- Use weight 400 for body text, 500 for labels, 700 for functional headings/actions and 900 for large statistics or non-logo display text. Preserve the supplied lowercase wordmark artwork; do not recreate it with a weight-900 text label. Do not make every screen title a display headline.
- Use the existing green accent for primary actions and generation; blue for reading/information, amber for queue/warnings and red for errors. State also needs text or an icon.
- Keep one semantic `h1` per screen. A decorative compact title used for a scroll transition stays hidden from assistive technology.
- Keep conversation prose at a maximum comfortable measure of 68 characters. Code, tables and scan-oriented layouts can use the full content width.

### Liquid Glass adaptation

Apple describes Liquid Glass as a functional layer for navigation and controls. Apply that principle to Strata's floating header, composer and small overlay controls; keep cards, tables, charts, code and conversation text opaque. Use restrained regular material treatment, adequate foreground contrast and opaque accessibility fallbacks. [Apple: Materials](https://developer.apple.com/design/human-interface-guidelines/materials)

Use consistent alignment, clear content/control separation and adaptable split views. Extra depth must help users understand layers; it must not compete with the work. [Apple: Layout](https://developer.apple.com/design/human-interface-guidelines/layout)

The React/CSS implementation approximates these principles. It is not the native optical Liquid Glass material provided by Apple platform components. Retain the current portability decision against custom refraction and lensing. Native adoption would be a separate platform project. [Apple: Adopting Liquid Glass](https://developer.apple.com/documentation/TechnologyOverviews/adopting-liquid-glass)

### Responsive behavior

| Available window/content width | Required behavior |
|---|---|
| Window >= 1280px | Allow the expanded 268px sidebar; respect the user's collapsed preference. |
| Window 768–1279px | Use the compact 64px rail. The expansion control opens the labelled sidebar as an overlay. |
| Window < 768px | Remove the permanent rail; expose a labelled menu control and a dismissible sidebar sheet. |
| Main content area >= 1040px | An opened conversation inspector may sit inline without narrowing prose below its reading layout. |
| Main content area < 1040px | Inspector opens as a sheet with focus handling and close control. |
| Form/list columns cannot fit | Stack them; do not compress labels into unreadable fragments. Table and code overflow remain local. |

Width decisions are based on the actual content container for inspectors and form columns, not only on the viewport. Keep the mobile menu reachable in boot, disconnected and no-model states.

### Motion budget

| Interaction | Existing token | Use |
|---|---|---|
| Hover, press, focus-state color | 100ms micro | Immediate response without moving surrounding layout. |
| Menus, local control changes | 160ms control | Short opacity/transform transition. |
| Drawers, sheets, disclosure | 240ms local | Interruptible open/close; no waiting for animation before interaction. |
| Screen/panel change | 320ms surface | Use only when it improves continuity; skip route snapshot transitions during streaming. |
| Initial coordinated transition | 460ms choreographed | Available in the token system, but not used to stagger cards across ordinary pages. |

Use the existing easing family. Springs are for direct manipulation. Keep one meaningful moving element per region; no decorative idle loops, parallax, animated backgrounds or per-token entrance animations. A user-triggered sheet remains usable during generation, without a multi-element reveal sequence.

Reduced motion must govern programmatic scrolling and chart transitions as well as CSS. Reduced transparency keeps identical geometry with an opaque fill. Remove the full-window blur behind dialogs; use a dimmed backdrop. Freeze displayed data in Snapshot view, rather than animating values during a screenshot.

## 3. App shell and navigation

**Source:** `src/app/Sidebar.tsx`, `App.tsx`, `CommandPalette.tsx`, `Shortcuts.tsx`; `src/components/layout.tsx`; `src/lib/router.ts`.

**Default hierarchy:** New chat → Search chats → active workspace and conversations → Skills / Monitor / Serving → More. Workspace files are reached through **View files**; Models, Engine tuning and Settings stay in More. All routes remain searchable through the palette.

| ID / priority / kind | Current issue | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| NAV-01 / P1 / Design | Search is hidden behind an icon; workspace groups do not make file access obvious. | Show Search chats when expanded, name the active folder and add View files. Keep pinned and recent groups and the existing Skills destination. | Existing workspace/session state. | A new user can locate a chat and open the active folder without using the palette. |
| NAV-02 / P0 / Defect | `Sidebar.tsx:100` caps grouped session results at 20, including searches. | Search every retained conversation, including archived chats; apply the initial recent-list limit only to unfiltered browsing. | Conversation store. | An older matching conversation beyond the initial 20 results is found and opened. |
| NAV-03 / P1 / Design | Hover pin/delete controls, right-click menu and native rename prompt create competing action patterns. | Use one ellipsis menu with Rename, Pin/Unpin, Archive and Delete; use app dialogs for text entry and consequential actions. Keep right-click as an alternate entry. | UI-11; command registry. | Mouse, touch and keyboard can perform all four actions; cancelling preserves the chat. |
| NAV-04 / P1 / Design | The collapsed tree exposes numerous indistinguishable conversation icons. | Keep a few stable destination icons; open recent chats in a labelled popover. Put theme choice only in Settings. | UI-11; responsive shell. | Every rail control has a name; recent-chat titles remain readable; no duplicate theme action remains. |
| NAV-05 / P0 / Defect | Engine status logic is repeated and can describe an unloaded engine as ready. | Derive one status: No model loaded, Loading, Ready, Reading, Thinking, Generating or Error. Present it once per shell region; details remain optional. | CAP and OPS contracts. | Every phase, including thinking and no-model, has the same meaning in shell, chat and focus bar. |
| NAV-06 / P2 / Design | Header emphasis and repeated cards/pills compete with primary content. | One title, optional explanatory sentence and one main action. Use separators/plain groups for secondary information. Apply any scroll mask to the backdrop/content edge, not labels or controls. | Existing layout/token system. | Titles and actions remain legible during scroll; no screen repeats its loaded-model/status fact in several adjacent badges. |
| NAV-07 / P1 / Design | Rails, inspectors and hardcoded form columns do not share a single responsive policy. | Apply the width behavior in section 2; preserve route and draft when switching layout. | UI-11; responsive containers. | 1512/1024/768/390px windows have reachable navigation and no page-wide horizontal overflow. |
| CMD-01 / P0 / Defect | `Shortcuts.tsx:36` advertises Home as Cmd/Ctrl+1, while `App.tsx:72` maps 1 to Chat. New conversation is also advertised without a matching app handler. | One registry supplies keyboard handlers, palette labels and help. Keep mappings 1 Chat, 2 Monitor, 3 Serving, 4 Models, 5 Engine tuning, 6 Workspace, 7 Skills, 8 Settings. Implement New chat; remove any unsupported binding. | CMD contract. | Every displayed binding opens its named destination/action; there is no nonexistent Home shortcut. |
| CMD-02 / P1 / Design | An empty command query presents a long action/session inventory. | Start with common actions and recent conversations; searching exposes every supported destination and action. Use platform-appropriate key labels and accessible active-result announcements. | Existing fuzzy search; CMD contract. | Full search coverage remains while the initial palette is short and navigable by arrows/Enter/Escape. |
| CMD-03 / P0 / Defect | Restart/unload shortcuts and palette actions bypass consistent interruption handling. | All entry points use the same lifecycle action. If busy, offer Wait until idle or an explicit interruption action naming the affected work. | OPS contract. | A restart, unload or switch initiated from any entry point shows the same effect and error result. |

## 4. Launch and connection

**Source:** `src/app/App.tsx` (`BootSkeleton`, `ConnectionBar`, `FocusBar`), `src/lib/store.ts` boot/connection state and `src/lib/router.ts` fallback handling.

**Layout:** Existing shell → destination-shaped loading placeholder → chat or setup. Retain the existing route fallback to Chat; no additional splash or Home screen is needed.

| ID / priority / kind | Current issue | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| BOOT-01 / P1 / Design | The same card skeleton appears regardless of destination. | Use a conversation-shaped placeholder for Chat, rows for lists/forms and metric blocks for Monitor. Show loading only while waiting for actual data. | Current route and boot state. | Opening each saved route produces a recognizably relevant placeholder and no artificial delay. |
| BOOT-02 / P1 / Design | Connection copy discusses the host process and empty metrics. | Say “Strata isn’t connected to its engine.” Offer Retry and expandable address/error details. Keep old data visibly stale rather than replacing it with reassuring zeroes. | OPS; connection state. | Connection loss explains what happened and provides a recovery action without requiring technical knowledge. |
| BOOT-03 / P0 / Dependency | Page reload recovery can discard unsent component-local drafts. | Reconnect in place; preserve draft, selected chat and reading position. Retry host data/event connection instead of making full reload the normal action. | Draft ownership in the store; connection retry. | Disconnect and reconnect while typing returns to the same draft and chat. |
| BOOT-04 / P0 / Defect | Model simulation and demo inference are not uniformly identified. | Label design/demo operation where it affects the user. Show Ready only when the appropriate real or visibly simulated operation reports readiness. | CAP; model library mode and inference origin. | Simulated setup/model activity cannot be mistaken for actual installed weights or live inference. |

## 5. First-run setup

**Source:** `src/screens/Setup.tsx`; setup endpoints/state in `server/index.mjs`; `server/engine-install.mjs`, `models.mjs`, `supervisor.mjs`.

**Five steps:** Check computer → Choose model → Basics → Download and load → Ready. Keep the existing modal, named step rail, back navigation and Finish later escape. The main action changes with the actual operation.

| ID / priority / kind | Current issue | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| SETUP-01 / P1 / Design | Hardware scanning waits 1100ms for animation (`Setup.tsx:84`), and details compete with the verdict. | Lead with the detected compatibility verdict, then GPU/RAM/CPU/storage details. Remove the delay; show pending only during a real check. Use “computer,” including on macOS. | Existing hardware probe. | A cached check appears immediately; a real check has honest pending/error/recheck states. |
| SETUP-02 / P1 / Design | Step 2 presents every variant at equal depth. | Lead with one recommended option, memory and download requirements. Put alternatives behind Compare variants. Explain tradeoffs without promising measured speed. | Model recommendation and fit data; MODEL-03. | Recommended choice is obvious; all alternatives and their constraints remain accessible. |
| SETUP-03 / P0 / Defect | Step 3's memory choice is local state (`Setup.tsx:54`); `startInstall` saves beginner values without that choice. | Keep context/image defaults prominent. Move memory policy under More options and save it through the supported engine parameter, or suppress the control if unsupported. Retain choices when resuming. | CFG, CAP and SETUP contracts. | Every offered choice survives re-entry and changes the configuration actually used to load. |
| SETUP-04 / P0 / Defect | Environment and engine rows are hardcoded done (`Setup.tsx:372–374`); installed files trigger Ready before model load. | Populate stages from operation state. Installed files mean download complete; load/warm-up is separate. Only show Ready after the reported model load succeeds. Omit stages this host does not perform. | SETUP; OPS; engine installer/supervisor status. | Engine installation and model-load failure cannot produce a success check or ready page. |
| SETUP-05 / P0 / Defect | Finish later calls `completeSetup()` (`Setup.tsx:181`), marking unfinished setup complete. | Dismiss setup for this session while retaining incomplete state, selected variant, settings and step. Settings offers Resume setup. Mark complete only after successful readiness. | SETUP contract. | Leaving step 2/3 and restarting/resuming restores unfinished progress. |
| SETUP-06 / P0 / Dependency | Copy promises resumability without distinguishing actual download behavior. | State whether a failed stage retries from scratch or resumes retained bytes. Keep logs collapsed except on request/error. Retry the failed stage without clearing successful work. | Actual installer/download behavior. | A failed download has accurate progress/retry text; no completed stage restarts unnecessarily. |
| SETUP-07 / P1 / Design | The ready page claims loading/warm-up/API work before its final button performs load. | Ready summarizes the confirmed model/context/accelerator. Main action is Open chat; API details remain secondary and reflect listening state. | SETUP-04; CAP. | Open chat does not initiate an undisclosed first load; users see any remaining load before Ready. |

**Stage presentation:** completed stage → check and outcome; active stage → progress if measurable, otherwise activity label; pending stage → neutral label; failed stage → reason, Retry and details. Do not assign overall 100% while loading is still pending. Unknown duration stays unknown.

## 6. Chat

**Source:** `src/screens/Chat.tsx` (`ChatScreen`, composer, `ChatIntro`, message actions, `ToolCard`, Markdown), conversation helpers in `src/lib/store.ts`.

**Layout:** Quiet header → readable conversation → compact composer. Keep the same shell for empty, active and completed chats.

| ID / priority / kind | Current issue | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| CHAT-01 / P1 / Design | Intro leads with a large repeated model identity and developer-heavy examples; input says “Do anything.” | Heading: “What would you like to do?” Three examples: general question, summarize a document and build a small thing. Placeholder: “Message Strata…”. No-model view offers Load recommended model. | Model recommendation; existing chat creation. | A first-time user can begin without choosing a technical category; the empty state explains model readiness. |
| CHAT-02 / P0 / Defect | Attachment/image icon buttons have no action (`Chat.tsx:644` and adjacent controls). | Remove unavailable actions. Show a working attachment menu only for supported input types; gate images by model/backend capabilities. | CAP; real attachment submission. | Every visible attachment action selects and sends supported input or explains an unavailable capability. |
| CHAT-03 / P1 / Design | Assistant replies, completed tools and per-message statistics create excessive containment. | Open assistant prose; contain code and structured results. Collapse completed tools to one-line outcomes. Latest answer shows Copy and Retry; older replies and secondary actions use overflow. Keep stats optional. | UI-10; Markdown; existing message/tool data. | A long answer reads continuously; detailed evidence and actions are still reachable by keyboard/touch. |
| CHAT-04 / P0 / Defect | Every assistant retry calls `retryLast()` (`Chat.tsx:844`), regardless of the selected reply. | Retry the selected reply's user request in a branch when later turns exist. Latest retry may replace the latest attempt while retaining recoverability. | Conversation branch/history helpers. | Retrying reply 2 in a six-turn chat uses reply 2's request and preserves subsequent original turns. |
| CHAT-05 / P0 / Defect | Earlier edits use `rewindTo` and truncate the current transcript. | Editing an earlier request creates and opens a branch with the prompt in the composer; the original chat remains intact. Name the branch relationship in optional details. | Existing branch helper; draft ownership. | Editing, cancelling and sending do not delete the original conversation. |
| CHAT-06 / P0 / Defect | Composer Enter can approve a pending tool; approval labels are generic. | Approval buttons state the action: Allow file change / Allow command, with target/scope and expandable evidence. Sending a message must not approve a tool. Denial is an explicit action. | Existing tool approval model; UI-11. | Enter in the composer never grants file/command permission; Approve/Decline are accessible in the tool panel. |
| CHAT-07 / P1 / Design | Footer chips mix conversation and engine totals; the model/status also appear elsewhere. | Keep one compact activity/context line. Detailed statistics open in the inspector/popover and state their scope. Reasoning remains collapsed; use actual duration when known, otherwise “Thinking details.” | DATA; UI-12. | A past chat never shows another request's context or invented thinking time. |
| CHAT-08 / P0 / Defect | `/compact` reports fabricated freed tokens; `/export` reports success without producing an artifact. | Remove compact until implemented. Export the selected chat to a real local Markdown or JSON download, retaining speaker order and code. Report completion only after artifact creation. | Existing session content; client download. | Export contains the selected chat; no compact success or freed-token figure appears without a real operation. |
| CHAT-09 / P1 / Design | Some focus navigation uses unconditional smooth scrolling; reading dimming can reduce old-message contrast. | Preserve follow-reader, scroll restoration and jump-to-latest behavior. Use instant programmatic scrolling under reduced motion. Hide peripheral chrome in focus mode while keeping all prose at readable contrast. | Existing scroll/focus behavior; preference resolution. | Streaming never pulls a reader forward; reduced-motion turn navigation has no smooth scroll; older text remains readable. |
| CHAT-10 / P1 / Design | Markdown supports a small subset, with line-by-line paragraphs and no full headings/links/tables; code copy depends on a language label. | Render headings, links, semantic lists and usable tables. Keep code copy for all fenced blocks. Handle incomplete streaming Markdown without moving the reader or running model-authored HTML/scripts. | UI-15; safe renderer behavior. | Common AI responses and unfinished code fences render safely and remain usable at narrow widths. |

**Important states:** no session; no model; model loading; ready draft; reading prompt; thinking; generating; completed; stopped partial answer; request failure with retry; tool approval/running/error; historical chat while another request runs; branched conversation; disconnected with draft intact.

## 7. Response settings drawer

**Source:** `src/app/SamplingDrawer.tsx`; sampling parameter patch/reset behavior in `src/lib/store.ts` and `server/engine.mjs`.

**Layout:** Response settings → one randomness control → Advanced sampling → scoped reset. The heading and accessible composer action use the same name.

| ID / priority / kind | Current issue | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| RESP-01 / P1 / Design | “Sampling” and detailed controls assume technical knowledge. | Rename to Response settings. Show Randomness with an explanation; expand Advanced sampling for temperature/top-p/top-k/repetition/seed. Show explicit values and valid units/ranges. | Existing sampling parameters; UI-06. | A beginner can change randomness without reading five technical fields; experts retain every sampling control. |
| RESP-02 / P0 / Defect | The surface implies conversation scope but modifies engine defaults. | State “Applies to all requests using these defaults.” Stage changes and save together; do not imply per-chat scope until requests carry overrides. | CFG; existing engine patch. | The displayed scope matches Chat and API behavior; cancelling staged edits changes nothing. |
| RESP-03 / P0 / Defect | Reset posts `/api/engine/reset` and resets unrelated configuration. | Patch only sampling keys to their recommended values. Use “Reset sampling” as the action and preserve context/memory/server settings. | CFG; parameter defaults. | Change context/server settings, reset sampling, and verify those settings are unchanged. |
| RESP-04 / P0 / Defect | Show thinking only updates component-local state; values can remain stale across openings. | Wire Show thinking to a persisted presentation preference, separate from reasoning effort. Refresh controls on open/model change without overwriting an unsaved draft. | PREF; current engine config. | Preference changes affect actual reasoning disclosure and survive reload; reopened controls match effective configuration. |

## 8. Conversation inspector and telemetry popovers

**Source:** `src/app/ContextPanel.tsx`, `src/components/TelemetryPopover.tsx`, `src/components/charts.tsx` and session/monitor state.

**Default sections:** conversation context and relevant references open; conversation statistics, tools and model details collapsed. Remove the duplicated hardware panel and link to Monitor.

| ID / priority / kind | Current issue | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| CTX-01 / P1 / Design | Five sections open by default, with redundant gauge/bar/percent and hardware information. | Use one context measure with amount/capacity. Show references when present; collapse secondary sections and remember disclosure choices. Keep hardware in Monitor. | Existing collapse behavior; PREF. | First opening is short and understandable; expanding details is reversible and remembered. |
| CTX-02 / P0 / Defect | Selected-chat context comes from global monitor state; missing data falls back to assumed 32K and zero usage. | Use conversation-associated data where known; otherwise display Not available. Clearly label engine totals as engine totals and show historical model identity only if recorded. | DATA contract. | Opening an old chat during an API request cannot attribute that request's context/model to the old chat. |
| CTX-03 / P0 / Defect | A hardcoded TOOLS catalog promises permissions and behaviors independently of the actual harness. | List available tools and granted scopes from capability state. Keep recorded tool-use counts separate from availability. No unverified “always asks” or sandbox guarantee. | CAP; harness tool state. | A disabled/unavailable tool is not advertised as usable; counts refer only to the selected chat. |
| CTX-04 / P1 / Design | Popovers and inspector do not share a full keyboard/presentation policy. | One telemetry popover at a time; keyboard open, meaningful name, Escape/click-away dismissal and focus return. Use an inline inspector or modal sheet according to content width. | UI-11, UI-12 and NAV-07. | Focus never lands behind a modal inspector; opening another popover closes the first. |

## 9. Models

**Source:** `src/screens/Models.tsx`, fit/recommendation logic, `src/lib/store.ts` model actions and `server/models.mjs`.

**Layout:** Running model → Installed variants → Available variants → storage details. Default metadata is model name, variant, estimated memory requirement and disk/download size. Technical architecture belongs in Details.

| ID / priority / kind | Current issue | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| MODEL-01 / P1 / Design | Current, loaded and selected states are easy to confuse; technical metadata competes with action. | Use Running / Installed / Available consistently. Show one running summary and concise rows; put layers, experts and architecture in Details. | Model library state. | Users can tell which variant is executing and which only occupies disk. |
| MODEL-02 / P0 / Defect | Recommendation text claims smaller variants “trade quality for nothing” (`Models.tsx:256`). | Explain that smaller variants generally use fewer resources, with quality/speed depending on the variant. State the recommendation's actual fit basis; avoid unmeasured performance promises. | Existing catalog/recommendation. | Fit guidance offers a meaningful reason without denying smaller variants' resource advantages. |
| MODEL-03 / P1 / Design | Memory-class fit can be read as a guarantee of current capacity. | Label estimates and base fit on reported accelerator/system headroom plus runtime/context overhead. Distinguish compatible from comfortable with other apps open. | Hardware/footprint data; CAP. | A tight fit is visibly different from a comfortable fit; unavailable headroom is not presented as certainty. |
| MODEL-04 / P0 / Defect | Showing all sets `hidden=[]`; the disclosure then disappears (`Models.tsx:31`). | Compute hidden count independently of expanded state. Show a persistent Show more / Hide extra variants action. | Existing list state. | Expand, select/view an extra variant and collapse again without reloading. |
| MODEL-05 / P1 / Design | Long download buttons and duplicate progress make operations noisy. | Button says Download; size and estimated duration are adjacent metadata. Use one progress location per operation with cancel/failure/retry. Estimates change to reported ETA once available. | Operation state; UI-13. | One variant has one progress indicator; retry reflects actual retained-file behavior. |
| MODEL-06 / P0 / Defect | Oversized-load confirmation is applied on one row but bypassed from other entry points; native confirmation is inconsistent. | Share fit/interruption handling across row, running summary, setup and palette. Explain loaded-model removal consequences with an app dialog. | OPS; UI-11. | The same variant requires the same handling wherever loaded or removed. |
| MODEL-07 / P0 / Defect | Model-storage proportion is based on model bytes plus free bytes, rather than actual disk usage. | Label Used by models and Free on drive separately unless total drive use is known. Put model path/manage-storage action in optional detail. | Existing disk values; actual disk totals if provided. | Storage labels never imply a total-disk percentage from incomplete values. |
| MODEL-08 / P0 / Defect | The host's design mode simulates installs/loads (`models.mjs:23,92`); frontend types omit the mode. | Consume mode and show “Demo model library — downloads and loads are simulated” on affected views. Real mode uses confirmed filesystem/process state. | CAP; existing library payload. | A simulated install cannot appear to have created real model files. |

## 10. Engine tuning

**Source:** `src/screens/Tuning.tsx`; `src/lib/store.ts` save/validate/export helpers; `server/engine.mjs`, `catalog.mjs`, `supervisor.mjs`.

### Beginner

**Layout:** Response preferences → Computer resources → Expected effect → persistent save bar. Group Priority and Context together; Images and Free GPU when idle form the capability/resource group. Keep Balanced and Recommended defaults. This grouping is explicitly authorized by the current plan even though an older iteration deferred it.

| ID / priority / kind | Current issue | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| TUNE-01 / P1 / Design | Four flat controls and “Best quality” imply a guaranteed answer improvement. | Use the two groups above. Describe priority as a settings tradeoff. Explain context as text the model can consider; show memory consequence at the field. | Existing beginner configuration; CFG. | Each choice describes its resource/behavior effect without guaranteeing quality. |
| TUNE-02 / P0 / Defect | Field impact uses draft prediction, but the summary reads `engine.predicted` (`Tuning.tsx:267–273`). | Fields and summary use the same validated draft prediction; show saved → proposed values only when changed. Mark estimates as estimates. | Existing draft validation; CFG. | Changing context changes both field impact and summary before saving. |
| TUNE-03 / P0 / Dependency | Image support is offered without a complete capability check. | Enable the choice only when the backend/model supports it; otherwise explain Text only or missing vision support. | CAP. | An unsupported image choice cannot be saved as a working feature. |
| TUNE-04 / P0 / Design | Save always restarts; no clear saved-versus-applied state or discard exists. | One draft, changed count, Discard changes, primary Save for next load and secondary Apply now. Name restart/active-request effects before applying. Disable invalid changes with a visible reason. | CFG; OPS. | Save preserves running behavior until load; Apply now confirms the result; Discard restores the saved values. |

### Expert — Parameters

Keep all existing groups: Context & memory, Performance, Sampling, Server and Advanced. All groups remain accessible in Expert. Collapsing a group reduces visual noise without removing options.

| ID / priority / kind | Current issue | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| EXP-01 / P1 / Design | Rows show labels/effects but not complete technical identity or application timing. | Show label and actual key, unit, value, recommendation and next-load/live scope. Add Changed only; preserve search, group filters, changed counts and per-field revert. | CFG parameter metadata. | Searching technical keys finds their controls; changed filter contains every differing value. |
| EXP-02 / P0 / Defect | `patchExpert` saves each edit immediately; raw/Beginner edits have different ownership. | Stage all views in one draft. Inputs keep intermediate text until valid; validation applies to the complete candidate configuration. Saving is deliberate. | CFG. | Typing a multi-digit value or clearing an input does not save partial/zero values; changing views preserves the same draft. |
| EXP-03 / P0 / Dependency | The prototype catalog is a subset/wrapper, not verified coverage of the production Strata engine. | Reconcile supported config keys, arguments and environment overrides with the connected engine's version/capabilities. Keep unsupported or unknown raw values preserved and identified; never claim every native option is mapped without checking. | CAP; production adapter; CFG. | A supported engine option remains accessible in raw form even if no visual field exists. |

### Expert — Raw config, import and export

| ID / priority / kind | Current issue | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| RAW-01 / P0 / Defect | Textarea updates `raw`, but Save only restarts; Validate checks saved configuration. | Raw and Parameters represent the same draft. Parse/validate before save, locate syntax errors and preserve the editor text after failure. Do not restart on an invalid document. | CFG; complete-draft validation. | A valid raw edit changes saved configuration; invalid JSON leaves running/saved config intact. |
| RAW-02 / P0 / Defect | “Export / import” performs export only and can replace the raw editor with a different export shape. | Separate Export config and Import config. Import previews changed values in the unified draft and requires Save; export serializes the declared configuration format without replacing unsaved text. | CFG export/import mapping. | Export → import retains effective values; neither action silently discards an existing draft. |
| RAW-03 / P0 / Dependency | Visual schema does not represent every possible engine option. | Preserve recognized and unrecognized valid config entries through raw/visual switches and round trips. Do not strip expert `args` or `env` because no form control exists. | Production config adapter; CFG. | A raw-only setting survives a visual edit and export/import cycle. |
| RAW-04 / P1 / Design | Reload from engine can erase edits without clear scope. | Name source “Reload saved config.” If dirty, offer Keep editing / Discard and reload. Reset-to-recommended stages its declared scope and explains it before saving. | CFG; UI-11. | Reload/reset never discards unsaved content without a specific choice. |

**Production reference:** verify actual config, arguments, environment overrides and runtime controls against the connected engine and [Strata configuration documentation](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md). The local design host's llama-server wrapper, catalog and ports are not proof of production parity.

## 11. Monitor

**Source:** `src/screens/Monitor.tsx`, `src/components/charts.tsx`, frontend monitor types, `server/monitor.mjs` and supervisor metrics.

**Default overview:** Current activity → answer speed / prompt-reading speed → memory / temperature health → recent request summary. Hardware details and engine totals expand locally; do not add a competing navigation dashboard.

| ID / priority / kind | Current issue | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| MON-01 / P0 / Defect | A raw implementation comment is inside JSX text (`Monitor.tsx:237`), and prompt progress renders for all busy phases. | Remove visible implementation prose. Show token/percent progress only while reading; loading/thinking/generating have their own honest status. | Current phase state. | No code comment appears; generation never says it is still processing a completed prompt. |
| MON-02 / P1 / Design | Phase/status and totals repeat; idle speeds can look broken. | One phase heading, concise queue information and relevant active value. While idle show retained latest/recorded results, clearly labelled. Put engine totals in disclosure; remove arbitrary uptime percentage. | DATA; recorded request data. | Idle is informative without pretending historic values are live; elapsed uptime is not a fill-to-one-hour gauge. |
| MON-03 / P2 / Design | A shared prompt/output y-range can flatten answer-speed variation. | Give each phenomenon a labelled zero-based scale and shared time window. State differing ranges; do not imply direct shape comparison. Hover/keyboard summaries use timestamps from samples. | UI-10; existing `MonitorSample.t`. | A 40 tok/s output series remains readable next to a 1500 tok/s prompt series; time labels are correct for irregular sampling. |
| MON-04 / P0 / Defect | Telemetry source metadata exists but is not typed/rendered; memory is labelled engine while later code synthesizes values. | Show Measured / Estimated / Demo / Unavailable per metric or coherent group. Correct source assignment at origin before labelling measured. Do not treat estimated thermal/load data as hardware evidence. | DATA; provenance correction in host. | A real throughput value beside estimated temperature has distinct labels; no fabricated value is marked measured. |
| MON-05 / P1 / Design | Hardware and resource cards have equal weight and shared-memory pools can be read as separate. | Overview emphasizes answer/prompt speed, memory and temperature health. Expand GPU/CPU load, disk, expert cache, power/bus only when reported. Label unified memory without double-counting capacity. | Hardware capability data; DATA. | Missing power/PCIe/CPU readings show unavailable or are omitted; unified memory is not presented as two independent limits. |
| MON-06 / P1 / Design | Many colored verdicts compete for attention. | Use one highest-priority actionable warning with a reason and next step; retain neutral capacity/context readings. Base thresholds on real reported limits. | DATA; supported model actions. | A warning identifies an action such as reducing context/closing apps; estimated temperatures cannot trigger a measured throttling claim. |
| MON-07 / P2 / Design | Live values and navigation make consistent screenshots difficult. | Snapshot view freezes one displayed snapshot, adds model/time/source labels and hides navigation/private paths. Arrange the default overview for 1440×900; retain provenance and warning text. Exit returns to current live data. | Existing snapshots; DATA; UI layout. | Values stay still, no prompt/chat/path is exposed, and the screenshot fits the target frame at default scale. |

**Snapshot boundary:** this is a presentation mode for an OS screenshot, not a new image export dependency. Never freeze or stop inference to freeze the displayed dashboard. If data changes while frozen, show a compact “Snapshot from [time]” label.

**States to cover:** no model, loading, idle before first request, reading, thinking, generating, queue present, engine failure, estimated/demo readings, unavailable sensor, connection lost/stale data, high resource pressure and Snapshot view.

## 12. Serving

**Source:** `src/screens/Serving.tsx`, `server/serving.mjs`, request logging/endpoints in `server/index.mjs`, serving state/helpers.

**Layout:** Compact connection summary → active/queued requests → recent requests. Base URL, model ID and access status stay visible; explanations belong in Connection guide.

| ID / priority / kind | Current issue | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| SERVE-01 / P1 / Design | Large connection cards push requests below the fold; Connection guide only emits a toast (`Serving.tsx:183`). | Use a compact summary with Copy actions. Open a real guide with OpenAI / Anthropic tabs, endpoint examples and supported extra endpoints. Requests remain visible in the default screen. | CAP; UI-11; current endpoint data. | A user can copy connection values and observe active requests without scrolling through documentation. |
| SERVE-02 / P0 / Defect | API/server state can be confused with model readiness. | State API listening, Paused or Disconnected separately from Ready/No model/Loading. Explain whether Pause stops admission or execution according to host behavior. | CAP; OPS; actual pause policy. | A listening API with no loaded model is not called Ready; pausing explains the fate of active and queued requests. |
| SERVE-03 / P0 / Defect | Selected request is a copied object (`Serving.tsx:12`); historic queue position uses current waiting count (`:406`). | Store selected ID and derive the current entry from incoming request state. Show only recorded history; omit unrecorded queue/TTFT figures. | DATA; existing SSE request state. | Open queued request progresses to generating/done in the drawer; historical requests do not inherit today's queue position. |
| SERVE-04 / P1 / Defect | Clickable rows are not fully keyboard accessible; `aria-sort` is placed on sort buttons. | Rows expose a keyboard details action; place sorting state on column headers. Preserve filters/counts/sorting and use the shared details drawer. | UI-09; UI-11. | Keyboard user filters, sorts, opens and closes a request; focus returns to its row/action. |
| SERVE-05 / P0 / Defect | API-key input is unmasked; text suggests generation that does not exist; success feedback can precede patch completion. | Mask entry with Reveal; separate Set/Change/Remove. Offer generation only if implemented. Show completion from the confirmed host state; copy only a key available in this UI session. | OPS; existing access patch. | Failed key updates retain the draft and show an error; an unrecoverable stored key is not offered as copyable. |
| SERVE-06 / P0 / Defect | Curl examples omit authentication when a key is configured. | Templates include required authentication without leaking secrets in logs/screenshots. The guide explains model/URL fields; label examples as templates, not exact replay. | Current access policy/endpoints. | A configured-key connection example is usable when the user supplies the key; it does not imply the original prompt was recorded. |
| SERVE-07 / P0 / Design | Pause, cancel and access toggles need consistent pending/failure feedback. | Keep the owning control pending until confirmation; preserve previous state on failure and show retry there. Lifecycle consequences remain visible. | OPS; actual host cancel/access semantics. | Rapid duplicate clicks do not submit conflicting actions; failed cancellation does not show a completed cancellation. |

**Guide tabs:** use proper tab/panel behavior. OpenAI Chat Completions is shown only if supported; Responses is an extra supported endpoint, not an assumption based on the production repo. Anthropic uses its advertised endpoint. Port and base URL come from this host, never copied as constants from another backend.

## 13. Workspace

**Source:** `src/screens/Workspace.tsx`, workspace file endpoints, `loadWorkspaceFiles`, chat draft/reference state and touched-path tracking.

**Layout:** Folder breadcrumb → searchable tree → selected-file preview. The page is a way to understand and reference files, not an editor redesign.

| ID / priority / kind | Current issue | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| FILE-01 / P1 / Design | Absolute paths repeat and split columns become cramped. | Lead with folder name/breadcrumb; put full path behind details/copy. Keep metadata compact. Narrow preview opens as a sheet; code/text gets Wrap and local horizontal scrolling. | NAV-07; UI-11; existing preview data. | Long paths and source lines remain accessible without page-wide overflow at 390px. |
| FILE-02 / P0 / Defect | Rescan only displays a toast (`Workspace.tsx:251`). | Call the existing refresh action, show pending/result/error and preserve selection if the file still exists. | `loadWorkspaceFiles`; OPS. | A changed file list actually refreshes; no rescan success appears after a failed request. |
| FILE-03 / P0 / Defect | Discuss creates a session and reports Reference added without attaching/prefilling the file (`:352–354`). | “Ask about this file” transfers a real supported reference, or prefills an honest request naming its workspace-relative path. Open that chat; do not claim content was attached unless it was. | Draft/ref ownership; current chat tools. | The new composer contains the selected reference/path and the original folder scope is preserved. |
| FILE-04 / P1 / Defect | Folder selection also toggles expansion; filtering can remove the only tabbable row. | Separate chevron expand from row select. Preserve arrow/Home/End behavior and one valid focusable tree row after filtering. | Existing tree implementation. | Filter to a different file after selecting one; Tab still enters the visible tree and arrows work. |
| FILE-05 / P0 / Defect | A failed read is converted into `binary:true` (`Workspace.tsx:100`). | Distinguish loading, read error with Retry, binary file, oversized preview and empty file. Keep known file metadata during error. | Existing file response/error. | A permission/network failure is never presented as an unsupported binary file. |
| FILE-06 / P0 / Defect | Read markers use a path-keyed map; scope copy promises a sandbox independently of verified enforcement. | Key activity by workspace and path, with session association when reported. Explain actual supported access/grants without unverified containment promises. | DATA; CAP; existing tool/harness behavior. | Same relative filename in another folder is not marked read; text matches actual access controls. |

## 14. Skills

**Source:** `src/screens/Skills.tsx`, skill types/helpers, skill folder/import endpoints in the host.

**Layout:** Effective skills in this workspace → concise rows → optional instructions/details. Preserve existing source grouping and real Add from folder behavior.

| ID / priority / kind | Current issue | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| SKILL-01 / P1 / Design | Off / Here / Always hides whether the choice means enabled, available or automatic execution. | One Enabled toggle; scope in details: This workspace / All workspaces. Explain “Available when relevant; enabling does not run it on every request.” | Existing enable/scope state; UI-05. | Scope can be understood without a tooltip; disabling is one direct action. |
| SKILL-02 / P0 / Defect | Here can be selected with no active workspace, and counts include skills outside the effective scope. | Disable workspace scope until selected; explain why. Header counts only enabled/effective skills for the current workspace, with a clearly labelled all-workspaces view if needed. | Workspace and skill scope state. | No workspace-specific selection becomes global via a null workspace; counts match the visible/effective set. |
| SKILL-03 / P1 / Design | Raw instruction blocks, source badges and usage metadata compete with purpose. | Row shows name, one-line purpose and Enabled. Expansion shows readable instructions, triggers, source and usage; Raw Markdown is optional. Group heading carries source once. | Existing skill body/triggers; UI-15. | Scanning identifies each skill's purpose; full original instructions remain inspectable. |
| SKILL-04 / P1 / Design | Long explanation and folder-path instructions make onboarding technical. | One sentence plus How skills work disclosure. Empty state leads with Add from folder; show host/import error at that action. Avoid unsupported context-cost promises. | Existing folder import; OPS. | Adding a skill from an empty library produces a visible row or a recoverable error. |

## 15. Settings

**Source:** `src/screens/Settings.tsx`, persisted preferences in `src/lib/store.ts`, appearance/accessibility CSS and storage/host capability data.

Preserve the current index and section order: General, Reading & focus, Engine, Storage, About, Privacy. Consolidate duplicate content inside those sections. The current plan authorizes grouping and simplification; it does not require a new settings navigation hierarchy.

| ID / priority / kind | Current issue | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| PREF-01 / P0 / Defect | Startup, update and history switches are component-local (`Settings.tsx:102–104`) and do not implement their promises. | General contains only supported/persisted behavior. Remove unsupported controls; if capability is later added, implement its actual effect before exposing the control. | PREF; CAP. | Reloading never silently resets a offered preference; changing it changes the stated behavior. |
| PREF-02 / P1 / Design | Appearance/reading choices are dense and lack a local preview. | Group appearance, conversation text size, interface scale, density, motion, transparency and contrast. Show a small representative preview. Add System appearance; retain Light default and saved choices. | PREF; token system. | Light/Dark/System and large-size choices produce the preview and real app behavior; system changes update only System users. |
| PREF-03 / P0 / Defect | Engine lifecycle controls overlap tuning and can reset broad configuration from Settings. | Engine contains links to Engine tuning and Models with a short current-state summary. Keep scoped reset/apply inside the owning screen. | NAV; CFG. | No generic settings reset silently clears tuned engine values. |
| PREF-04 / P0 / Defect | Storage says sessions are in the data folder while chats are browser-persisted; Clear caches only emits success (`:370`). | Describe actual model/config/chat storage. Offer working open/manage/clear actions only. Clearly distinguish browser conversations from server request logs and model files. | Storage facts; CAP; OPS. | Clearing a declared storage category affects only that category; unsupported cache clearing is absent. |
| PREF-05 / P0 / Defect | Telemetry preference persists, but collection is not implemented; privacy wording blurs logs and saved chats. | Say what is actually collected/stored. If nothing is sent, show “No usage data collected.” Do not expose a collection toggle until consent controls real collection. | PREF; actual collection/logging behavior. | Privacy text agrees with network/storage behavior; toggling a cosmetic preference is not described as consent. |
| PREF-06 / P1 / Design | About and setup actions are duplicated; implementation credits distract from product facts. | One About section with app/engine version, support and Resume/Run setup action. Put diagnostics in disclosure; paths stay in Storage. | Current build/version/setup state. | Version and setup entry appear once; diagnostics are copyable and avoid unrequested secret/prompt exposure. |
| PREF-07 / P0 / Defect | Reset Strata is enabled but says not wired (`:486`); cache reset gives false success. | Remove unsupported broad reset. For any working reset, name scope and show review before clearing local data. Preserve appearance-only reset with its existing confirmation. | OPS; implemented reset endpoints. | No control reports wiping/resetting data without doing so; appearance reset leaves conversations/model configuration intact. |
| PREF-08 / P1 / Defect | Search uses current DOM field text and selected option content, missing hidden choices/aliases. | Use descriptors for labels, descriptions, aliases and all options. Result navigates/highlights its field; show clear no-results feedback. Preserve index. | PREF descriptor list. | Searching “dark” while Light is selected finds Appearance; searching an alternate option finds the relevant field. |
| PREF-09 / P1 / Design | Focus can be enabled on Settings, obscuring configuration chrome. | Focus mode applies to conversation reading. An entry in Settings opens the active chat before enabling it, or explains that no conversation is available. | Existing focus/navigation state. | Settings never becomes a stripped, difficult-to-exit focus layout. |

## 16. Shared component inventory

**Source:** `src/components/ui.tsx`, `layout.tsx`, `charts.tsx`, `TelemetryPopover.tsx`, `src/lib/focus.ts`, relevant `src/styles/app.css` and copied kit styles.

This inventory covers primitives as well as composed app surfaces. Keep the UI kit as the canonical system. When future implementation changes tokens or kit recipes, reconcile the source kit and generated app copy so a sync does not erase the intended result. Do not undo existing contrast corrections simply to match stale kit values.

| ID / priority / disposition | Component(s) | Required behavior and acceptance | Dependency |
|---|---|---|---|
| UI-01 / P1 / Improve | `Btn`, `IconBtn` | Action verbs, accessible names, pending state and duplicate-submit prevention. A disabled reason is available in nearby text, not only on an unfocusable control. No enabled inert button remains. | OPS; handler/capability audit. |
| UI-02 / P0 / Keep | `Icon`, sprite | Preserve Anron geometry/currentColor and below-16px simplification. Decorative icons stay hidden from assistive technology; controls carry names. No unrelated icon family is introduced. | Existing kit. |
| UI-03 / P1 / Improve | `Panel`, `PanelHead`, `Screen`, `SectionHead` | Preserve semantic heading hierarchy. Reduce nesting and repeated labels; use plain sections when a border carries no meaning. Header text/actions never fade with a backdrop mask. | NAV-06; kit layout. |
| UI-04 / P1 / Fix | `Field`, field hints/impacts/errors | Connect control label and descriptions by IDs. Announce field errors and units; visible labels remain clickable for applicable inputs. Screen reader knows label, hint and invalid state. | Form ID support. |
| UI-05 / P1 / Fix | `Seg`, tuning/protocol tabs | Separate actual tab/panel controls from radio-style segmented values/filters. Add roving focus, arrows/Home/End and correct selection announcements. One tab stop per group. | Shared patterns; no visual redesign required. |
| UI-06 / P1 / Improve | `Select`, sliders, toggles | Preserve portal/typeahead/arrows; supply `aria-activedescendant` and scroll the active option into view. Numeric controls expose units and valid limits; toggles state scope. Keyboard and pointer choose the same value. | Existing parameter metadata; Field. |
| UI-07 / P1 / Improve | `Pill`, badges, state glyphs | Use one status vocabulary. Text/glyph communicates state without color. Do not repeat the same badge alongside its equivalent heading. | NAV-05; CAP. |
| UI-08 / P1 / Fix | `Bar`, `Steps` | Use task progress only for measured completion; omit `aria-valuenow` for indeterminate work. Capacity is a labelled meter/value, not fake task progress. Step completion reflects actual stage state. | SETUP; DATA. |
| UI-09 / P1 / Improve | Tables, list rows, file tree | Keep local overflow and readable columns; keyboard details action; header sorting semantics. Correct tree focus after filtering. No hundreds-of-rows Tab walk. | Serving/Workspace state. |
| UI-10 / P1 / Improve | `Metric`, `MetricGrid`, `Spark`, `Gauge` | Allow plain grouped metrics; grid minimums must fit narrow containers. Expose value/unit/window/source and accessible summaries. Use sample timestamps, not assumed one-second spacing. Unknown != zero; uptime != capacity. | DATA; MON-03. |
| UI-11 / P1 / Fix | `useSurfaceFocus`, dialogs, setup modal, sampling/request drawers, sidebar overlay, menus | Preserve working traps on existing modals; add missing request/sidebar sheets. Escape closes the top layer, focus returns, closed surfaces are inert and background controls are unavailable during modal interaction. Menus use menu navigation, not a modal Tab trap. | Shared focus/overlay coordination. |
| UI-12 / P1 / Improve | Telemetry popovers, slash menu, conversation overflow | One active popover/menu per context. Keyboard entry, active item announcement, dismissal and return focus. No nested glass sampling. | UI-11; command registry. |
| UI-13 / P0 / Fix | `ActionError`, operation labels, toasts | Success follows confirmed operation state; persistent field/action error carries retry. Preserve toast cap/hover pause/live region. Use close glyph, not rotated chevron. Routine Copy can use local acknowledgement. | OPS; action scope. |
| UI-14 / P1 / Improve | `Empty`, skeletons, connection/stale states | Distinguish genuine empty, filtered empty, loading and failure. Include one relevant next step; retries do not erase data. Never leave an indefinite skeleton after a known error. | BOOT; per-screen error state. |
| UI-15 / P1 / Improve | Markdown, code, tool previews, skill instructions | Readable headings/links/lists/tables, local code overflow and copy without language tag. Treat model/file markup as data; no active HTML execution. Streaming partial Markdown is stable. | Safe renderer; CHAT-10. |
| UI-16 / P2 / Improve | Materials, scroll-edge treatment, scrim, focus bar, composer | Glass stays in floating controls; content and sheets opaque; scrim dimmed with no full-screen blur. No nested glass. Reduced-transparency geometry is identical. | Existing material tokens/audits. |
| UI-17 / P2 / Improve | Motion, chart transitions, tree branches, route transitions | Keep duration/easing scale and interruptibility; remove decorative staging. Reduced motion includes JS scroll and chart animation; route snapshots remain skipped during streaming. | Resolved motion preference. |
| UI-18 / P1 / Keep + refine | Focus ring, contrast, high-contrast behavior, density/text/UI scaling | Preserve measured contrast corrections and semantic headings. Text requires 4.5:1 (3:1 for qualifying large text), relevant controls/focus indicators 3:1. Large scale/zoom never removes actions; focus mode does not dim prose below target. | Existing contrast/measure/luminance harness. |

### Overlay dismissal rules

- Escape dismisses the topmost select/menu/popover/dialog first, then focus mode. A global handler must not close multiple unrelated layers or execute an engine action.
- Approval/decline remains explicit in the tool panel. Generic composer Enter or dismissing another overlay does not approve or deny a tool.
- Click outside dismisses menus/popovers and non-destructive sheets. Dirty editors show Keep editing / Discard before losing draft content.
- Modal surfaces trap focus and make the background inert. Nonmodal popovers and menus use their appropriate keyboard pattern and return focus on close.
- A closing animation must not block a new action, leave a hidden focused control or delay restored focus until a guessed timeout.

## 17. Functional dependencies and interfaces

These are requirements to record for implementation. No engine/API work is performed by creating this document. Reuse current endpoints where adequate; do not invent a second parallel settings store or replace external public APIs for a visual change.

| Dependency | Minimum required contract/behavior | Existing seam | Acceptance |
|---|---|---|---|
| **CFG — configuration** | One canonical full draft shared by Beginner, Parameters and Raw. Distinguish draft, saved and applied values. Validate the candidate before mutation; return field/key-associated issues and prediction. Preserve valid raw-only entries. Save and apply are separate intents. | `/api/engine` patch, `/validate`, `/export`, restart; current export uses `strata.engine/1` and differs from the editor's array shape. | Equivalent visual/raw edits have equivalent saved output; invalid drafts do not mutate or restart; saved settings remain visibly unapplied until confirmed. |
| **CAP — capabilities** | Declare real/simulated model mode, inference origin, supported modalities/tools/endpoints and engine parameter coverage/application scope. Reflect actual backend/version, including unavailable capabilities. | Library already returns `mode`; hardware exposes accelerator/unified memory; monitor returns engine state. Frontend types omit some host fields. | Disabled/hidden features match capabilities; prototype llama-server options are not presented as verified native Strata parity. |
| **DATA — telemetry and scope** | Consume existing telemetry provenance; correct origin labelling. Add association/time data only where needed to distinguish conversation/request/engine, stale samples and actual history. Keep missing readings absent. | Monitor's `telemetry`, engine metrics/source, sample `t`, recorded requests and Session fields. | Historic chat never adopts unrelated global request data; memory/thermals from synthesized code remain estimated/demo. |
| **OPS — operation results** | Pending/error/outcome is owned by its action. HTTP 200 does not imply engine running or desired state achieved. Deduplicate conflicting lifecycle actions and apply shared busy/interruption behavior. | Existing helpers, action errors, busy flags and supervisor status/error. | Failure leaves prior state/draft recoverable and never emits a success toast; all entry points agree. |
| **PREF — preferences** | Persist supported presentation choices, effective system overrides and inspector disclosure. Descriptors supply searchable labels/aliases/options. Unsupported startup/update/collection behavior is not exposed as functioning. | Existing preference/local-storage layer and Settings index/search. | Reload/OS change resolves preferences as labelled; resetting appearance leaves engine/chat state intact. |
| **CMD — commands** | One registry supplies action identity, availability, keyboard binding, labels, search aliases and lifecycle handler. Platform key display follows handler behavior. | App keyboard handlers, Shortcuts list, palette and slash command list. | No advertised shortcut/action diverges from its implementation; unsupported commands are omitted. |
| **SETUP — setup progress** | Store unfinished/completed/dismissed states distinctly, chosen variant/options and confirmed stage outcomes. Complete only after loaded readiness. Surface retry/resume capabilities accurately. | `/api/setup`, `/step`, `/complete`, model progress and installer/supervisor state. | Finish later preserves resume data; installation or load failure cannot mark setup complete. |

### Compatibility requirements

- Preserve existing URLs and route fallback. Do not introduce Home or change the numbered navigation order while centralizing commands.
- Preserve existing conversation records and useful preference values. New association metadata is optional for historical records; label absent information rather than inventing it.
- Import must adapt the declared engine export format to the visual draft without losing config fields. Unrecognized raw keys remain visible and preserved, with support status explained.
- Keep scope differences explicit: response defaults can affect API requests; showing thinking is a presentation preference; reasoning effort is a generation setting.
- Use reported endpoints/ports for the connected host. Production Strata's configuration and native API are references, not constants to copy into this prototype.
- Keep operation success distinct from simulated success. Design mode remains useful, but its presentation must say what is being simulated.

## 18. Delivery order and verification

### Implementation sequence

| Stage | Work | Exit condition |
|---|---|---|
| **A — P0 truth and state** | CFG/SETUP/OPS/CAP/DATA foundations; installation-response crash (BROWSER-01); raw save, reset scope and draft preservation across navigation (BROWSER-02); setup resume/readiness; correct reported endpoint (BROWSER-06); false success messages; message targeting/data preservation; simulated/provenance labels; shortcut mismatch. | Advertised actions work with their declared scope and source; failures retain recoverable data and cannot blank the shell. |
| **B — P1 everyday clarity** | Navigation, chat/inspector, models, beginner tuning, Serving, Workspace, Skills and Settings simplification; field/overlay/table keyboard work; multiline-editor clipping, semantic select activation, inaccessible narrow inspector and obscured Settings anchors (BROWSER-03–05, BROWSER-07). | A beginner can set up, chat, change a variant and connect another app without expert controls; expert functionality remains reachable at supported widths. |
| **C — P2 refinement** | Restrained materials, responsive arrangements, chart/scroll motion, Snapshot view and measured performance refinement. | Both themes and accessibility fallbacks pass visual/interaction checks; performance claims have measurements or an explicit unverified status. |

Within each stage, implement common contracts/components before dependent screen changes. Accessibility defects are part of each screen's exit condition. Keep changes reviewable by subsystem; the UI should not require a simultaneous engine rewrite to become more honest.

### End-to-end scenarios

| Scenario | Checks and expected result |
|---|---|
| **Setup success** | Hardware verdict → recommended variant → basics → actual download/load → Ready → Open chat. Every stage reflects confirmed state. |
| **Setup interruption** | Leave at model/basics/install, reload/resume; preserve selected variant/settings and completed work. Failed stage offers accurate retry/resume. |
| **Lifecycle** | Load/switch/unload/restart from Models, palette and shortcuts, idle and busy. Same consequences, pending state and confirmed result everywhere. |
| **Conversation preservation** | Edit/retry an old turn; original chat remains intact. Branch opens correctly. Draft and reading position survive reconnect. |
| **Tools and attachments** | Enter sends only; approval names action/scope; unavailable attachments are absent. Tool errors retain evidence and a useful next action. |
| **Configuration** | Equivalent visual/raw edit, invalid JSON/numeric input, import/export round trip, raw-only key, save-for-next-load and Apply now. Leave and return through sidebar, palette and shortcuts; unsaved values are retained or discarded only by explicit choice. Values and effects agree. |
| **Reset scope** | Set nondefault context/server values, then Reset sampling. Only sampling changes. Appearance reset and config reset also obey their stated scope. |
| **Monitoring** | No model, loading, reading, thinking, generation, idle, queue, failure, stale connection and pressure. Source labels match each value; unavailable != zero. |
| **Snapshot** | Freeze overview during a request; inference continues. Screenshot is 1440×900, time/model/source labelled and free of prompts/local paths. Exit returns to live values. |
| **Serving** | Copy connection values, authenticated example, protocol tabs, filter/sort, evolving request detail, cancel/pause failure. History uses recorded facts. |
| **Workspace** | Filter after selection, arrow navigation, rescan changed files, file-read error, binary/large preview, Ask about file, same filename in another workspace. |
| **Skills** | Add from folder, enable/disable, workspace/global scope, no selected workspace and effective counts. Instructions/triggers remain inspectable. |
| **Settings** | Search hidden option/alias, persistence, System appearance, reset scope, truthful privacy/storage, one About/setup entry. |

### Accessibility and visual matrix

- Walk every screen, actual tab and overlay by keyboard: Tab/Shift+Tab, arrows, Home/End, Enter, Space and Escape as appropriate. Check focus entry, return and visible indicator.
- Inspect Light and Dark at 1512px and 1024px; inspect narrow navigation/forms at 768px and 390px. Exercise the inspector-width transition on each side of 1040px main-content width.
- Inspect large interface scaling, 200% zoom and conversation text sizes 10/16/22px. No action disappears, prose stays readable and only tables/code use horizontal overflow.
- Check system and manual reduced motion/transparency, increased contrast and System appearance. Programmatic scroll/chart transitions must honor resolved motion preference.
- Inspect common content: long model/folder/chat names, many sessions, empty and filtered lists, large numbers, long error messages, code without a language label, tables and streaming partial Markdown.
- Check screen-reader names/descriptions/state for tabs, segmented choices, fields, progress, charts, request rows, trees, modal surfaces and toasts. Avoid announcing every token or chart hover value as urgent.

### Existing implementation checks

These commands are future implementation gates. They were **not run for this documentation task**, and some require a running design host/UI. Run them against a clearly labelled test/simulated environment; do not download or load a model merely to verify typography.

| Check | Purpose |
|---|---|
| `npm run typecheck` | Frontend contracts and component typing. |
| `npm run check` | Existing type, UI audit, contrast, reading-measure and luminance checks. |
| `npm run flow -- tour` / `npm run flow -- tour-dark` | Existing route tour in both themes; update relevant flows when controls intentionally change. |
| `npm run flows -- <relevant names>` | Targeted scenario flows for the changed subsystem; use only names present in the harness. |
| `npm run motion` | Existing motion/interruptibility verification. |
| `npm run motion:budget` | Performance check whose documented NOT MEASURED outcome must remain unverified, not be converted into a pass. |

Broaden checks only for a concrete remaining risk. Record real-engine/model integration separately from interface verification. No new test should merely assert a renamed label while missing the actual behavior.

### Performance acceptance

- No decorative recurring work on idle pages. No new canvas/refraction system, animation library or heavy screenshot dependency is required.
- Streaming preserves responsive composer/navigation and reader position. Avoid per-token entrance animation and unnecessary full-dashboard updates.
- Hidden/nonactive visualizations do not perform independent animation loops. Snapshot freezes presentation only.
- Measure scroll/streaming and glass cost on representative hardware after implementation. Record device, runtime, active data/model conditions and method; no FPS/resource claim is considered verified from a source review or an insensitive harness.
- If a measurement cannot detect a known-expensive case, report **Not measured** and leave the gate unresolved rather than publishing a success figure.

## 19. Coverage and completion checklist

### Route and subview coverage

| Current route/surface | Reviewed parts | Disposition / issue coverage |
|---|---|---|
| `#/chat` | Empty/model-ready states, header, intro, messages, composer, attachments, tools/approvals, reasoning, stats, actions, slash menu, Markdown, focus reading | Keep working history/scroll features; CHAT-01–10, RESP-01–04, CTX-01–04. |
| `#/models` | Running summary, installed/available rows, fit, alternatives, download/load/remove, operation progress and storage | MODEL-01–08; shared OPS and CAP. |
| `#/tuning` Beginner | Four existing choices, capability/resource groups, impact, validation and save bar | TUNE-01–04; CFG. |
| `#/tuning` Expert / Parameters | Five groups, search/filter, value inputs, defaults, changed counts and reset | EXP-01–03; CFG/CAP. |
| `#/tuning` Expert / Raw config | Editor, parse/validation, reload, save/apply, import/export and unknown-key retention | RAW-01–04; CFG. |
| `#/monitor` | Phase, prompt processing, speeds, hardware/context/resources, totals, warnings, snapshot and all reading origins | MON-01–07; DATA. |
| `#/serving` | Connection/access summary, protocol guide, API key, queue/list/filter/sort, pause/cancel and request drawer | SERVE-01–07; OPS. |
| `#/workspace` | Breadcrumb/path, tree/filter/keyboard, file preview, rescan, reference transfer and activity/scope | FILE-01–06; DATA/CAP. |
| `#/skills` | Groups, purpose, body/triggers/source/usage, enable/scope, folder import and empty/help | SKILL-01–04. |
| `#/settings` | General, Reading & focus, Engine, Storage, About, Privacy, index/search/reset/setup and preference persistence | PREF-01–09. |
| `#/setup` / setup modal | Hardware, model, basics, download/load, ready, step rail, Back/Finish later/Open chat and interruption/failure | SETUP-01–07; SETUP/OPS. |
| Launch/connection | Boot placeholder, route fallback, disconnected/stale/retry, retained draft and demo label | BOOT-01–04. |
| Sidebar/navigation | Expanded/compact/mobile, workspace/session search/groups/actions, fixed destinations, engine state and More | NAV-01–07. |
| Palette/shortcut help | Empty-query recommendations, full search, binding consistency, platform labels and lifecycle actions | CMD-01–03. |
| Shared primitives/surfaces | Icons/buttons/fields/tabs/selects/sliders/toggles/status/progress/steps, panels/lists/tables, metrics/charts, overlays/toasts/empties, Markdown/materials/motion/accessibility | UI-01–18; section 16 inventories the complete component groups. |
| Brand identity and distribution | Expanded/compact/mobile branding, setup/About/Snapshot, light/dark exports, favicon/touch/manifest icons, asset delivery and conditional native packages | BRAND-01–12; section 22 specifies exact assets, pending work and release checks. |

### Important state coverage

Every relevant screen must account for loading, ready, genuine empty, filtered empty, unavailable capability, pending mutation, failure with recovery and disconnected/stale data. Only show states that the operation actually supports. Chat adds stopped/partial/tool/branch states; model management adds installed/loading/unloading/download failure; setup adds incomplete/resume; Monitor adds provenance/snapshot; Serving adds queued/live/historical detail.

### Document completion

- [x] Every current route, setup step and tuning subview has a reviewed disposition.
- [x] Shared components, shell surfaces and important states are inventoried.
- [x] Remaining improvements have IDs, classifications, priorities, dependencies and acceptance checks.
- [x] Existing completed work is distinguished from new implementation requests.
- [x] Brand tokens, progressive disclosure, Beginner/Expert tuning and responsive/motion defaults are specified.
- [x] Source-confirmed behavior is distinguished from design proposals, capability gaps and unmeasured runtime claims.
- [x] Functional interface dependencies, implementation order and verification scenarios are recorded.

**UI implementation status:** see sections 21 and 24. The original specification is retained for traceability. Completed changes are recorded as implemented; production-engine dependencies and unperformed verification are not presented as completed work. Section 22 retains the logo specification; section 24 records its web implementation and remaining platform checks.

## 20. Initial browser walkthrough and additional findings

### Environment and scope

**Audit date:** 5 October 2026, before implementation. Opened the existing design in the Codex in-app browser and used actual navigation, buttons, inputs, menus and keyboard shortcuts. This supplements the source audit; it is not a completed implementation verification pass.

- Web preview: `http://127.0.0.1:5211`; isolated host: `http://127.0.0.1:8793`.
- Host ran with `STRATA_REAL_ENGINE=0` and a separate temporary `STRATA_HOME`. Model installation/loading and inference in this walkthrough were simulated; hardware detection remained read-only.
- The preview's separate origin and temporary host state isolated test conversations and configuration from the usual Strata state.
- Reviewed desktop at 1512×982, chat/sidebar at 1024×768, and chat/tuning at 390×844. Initial setup was also inspected in the browser's approximately 671px-wide natural viewport.
- Inspected chat in Light and Dark, then restored Light. This visual inspection is not a contrast-compliance result.
- Did not execute real-model operations, engine restart, permanent deletion, credential changes or network-access changes. Did not claim those controls passed. Reduced preferences, 200% zoom, every numeric input and complete screen-reader behavior remain future checks.

### Click-through coverage

| Surface | Actions actually exercised | Outcome / disposition |
|---|---|---|
| Launch and setup | Opened fresh preview; Continue through hardware, model and basics; opened memory-mode select; Download & install; reload after failure; resumed setup and opened chat. | Visited all five steps. Installation temporarily blanked the entire app (BROWSER-01). False loaded-ready claims remain SETUP issues. |
| Chat | Clicked a starter example, sent it, observed the demo response, switched conversations, opened/closed context and Sampling, dismissed Sampling with Escape, opened slash commands and ran `/workspace`. | Core demo chat and these navigation actions worked. Context and footer remain noisy; narrow context activation fails (BROWSER-05). |
| Monitor | Opened from sidebar and with Cmd+2; inspected activity, speed, hardware, context and totals after a request. | Printed implementation comment and missing provenance confirmed. No real sensor/throughput accuracy claim. |
| Serving | Opened from sidebar; switched OpenAI/Anthropic format; clicked Connection guide; opened a request; pressed Escape; closed with Done; filtered Errors and inspected filtered empty state. | Guide remains a toast. Request drawer ignores Escape. Filtered empty state provides Clear filters. |
| Models | Opened through More; expanded the additional variants; inspected loaded, fit and storage summaries; clicked Tune current model. | Variant expansion cannot be collapsed. Simulated operation disclosure remains incomplete. |
| Tuning | Changed Beginner context to 128K; switched Expert/Parameters/Raw; searched parameters; entered invalid JSON; clicked Validate; navigated away and returned. | Invalid JSON reported valid (RAW-01). Unsaved Beginner/raw values disappeared (BROWSER-02). Multiline editor clips text (BROWSER-03). |
| Skills | Opened Skills; expanded instructions; searched a nonexistent skill; cleared the search. | Instruction disclosure and filtered empty state worked. Scope clarity and effective counts remain SKILL issues. |
| Workspace | Opened from palette and `/workspace`; filtered README files; selected a real preview; clicked Rescan and Discuss in a new chat. | Preview/filter worked. Rescan only showed feedback; file handoff opened an empty composer with a false reference-added toast (FILE-02). |
| Settings | Clicked General, Reading & focus, Engine, Storage, About and Privacy index entries; searched `dark` and cleared it. | All sections reached. Search omitted Appearance; sticky index obscures some target headings/actions (BROWSER-07). |
| Shared navigation/overlays | Opened More, command palette and shortcut help; used Cmd+K, Cmd+/, Cmd+2 and Escape; collapsed/expanded sidebar; tested semantic, coordinate and keyboard select activation. | Palette/help/Sampling Escape worked. Cmd+2 opens Monitor despite help advertising Chat. Semantic option activation did not change Select values (BROWSER-04). |

### Additional issues

These seven IDs add behavior or concrete failure modes not previously specified. Related original IDs remain the owning screen requirements; do not create a second implementation for a shared cause. Narrow header/sidebar clipping itself is a runtime confirmation of NAV-07, rather than an eighth new issue.

| ID / priority / kind | Observed gap and reproduction | Required behavior | Depends on | Acceptance |
|---|---|---|---|---|
| BROWSER-01 / P0 / Defect | Fresh setup → Download & install blanked the entire application. Browser console reported `Cannot read properties of undefined (reading 'find')` in Sidebar. Reload recovered; the simulated install had continued. | Return an awaited, valid library payload; reject malformed responses before replacing app state; contain unexpected screen errors with a recoverable error surface. Preserve install/setup progress. | Model install endpoint, client response guard, OPS; SETUP-04/05. | First and repeated installation never blank the shell. A malformed/rejected response leaves valid library state and shows Retry. Reload resumes the actual operation without pretending an engine loaded. |
| BROWSER-02 / P0 / Defect | Set Beginner context to 128K and edit Raw config, then navigate to another screen and return. Both unsaved values reset without a warning or explicit discard. | Keep the unified configuration draft outside the route component. Navigation preserves edits; if an operation must replace them, offer Keep editing / Discard changes. Distinguish saved and applied values. | CFG; TUNE-04, EXP-02, RAW-04; shared navigation. | Beginner, Parameters and Raw edits survive route changes and browser back/forward during the session. Any supported close/reload policy is explicit; invalid raw text also survives navigation. |
| BROWSER-03 / P1 / Defect | Raw config applies the capsule-shaped `st-input` geometry to a roughly 380px-high textarea. The top-left arc clips the first line: `{ invalid json` is only partially visible. | Use a 14px or 20px finite radius, readable padding and a dedicated multiline-editor treatment. Keep capsule geometry for single-line controls. Give the editor an accessible label. | Shared input/editor styles; UI-02; RAW-01. | Opening, editing and scrolling raw text leaves the full first/last line and selection visible at desktop/narrow widths and increased scale. No syntax information is hidden by corners. |
| BROWSER-04 / P1 / Defect | Activating an exposed Select option through the semantic click path did not select it in setup/tuning. Keyboard End/Enter and a coordinate pointer click did work. This is an activation-path gap, not a claim that ordinary mouse selection is universally broken. | Support option activation through click/accessibility activation as well as pointer and keyboard paths, using one selection routine. Preserve focus and active-option announcements. | `Select` primitive; UI-05. | Pointer, Enter/Space as appropriate, and accessibility click choose the same option once, close the menu and announce the result. Verify with assistive technology; that part was not run in this audit. |
| BROWSER-05 / P1 / Defect | At 390px, with the sidebar collapsed, clicking Conversation context hides the context panel but still reserves its 340px grid track. The transcript/composer collapse to a narrow sliver; the inspector never becomes visible. | Below the inline-inspector threshold, open a usable sheet and reserve no conversation track. Synchronize toggle state, visibility and layout; retain a reachable close control. | Responsive split view; NAV-07, CTX-04, UI-11. | At 390/768/1024px and on both sides of the content-width threshold, opening/closing context never collapses the conversation. Inspector content is visible, keyboard reachable and dismissible; focus returns correctly. |
| BROWSER-06 / P0 / Defect | Setup Ready displayed `127.0.0.1:8791` while this host actually listened on 8793. Serving/Settings correctly used 8793; Expert separately showed internal server port 8080 without clarifying the distinction. | Read the public endpoint from the connected host everywhere. If an internal inference port is editable, label its purpose separately from the public API port. | Host endpoint/capability data; SETUP-05, SERVE-01/02; CFG metadata. | Run with a nondefault host port: setup, Settings, Serving and copied examples agree. Expert identifies any internal port and does not present it as the public connection address. |
| BROWSER-07 / P1 / Defect | Clicking Settings section links scrolls the section header underneath the sticky index. In Engine, its title and Open Engine tuning action were obscured; the heading's measured top was about 74px while the sticky controls covered that area. Reading/General/About had similar header loss. | Apply the resolved sticky-header/index offset to section scrolling. Keep the target heading and section action visible, and keep keyboard focus predictable. | Settings index/section layout; PREF-08, NAV-06; reading preferences. | Clicking or keyboard-activating every section shows its title and main action below sticky controls at supported widths/scales. Near the page end, clamping must still leave the chosen section understandable. |

### Source follow-up for the new failures

The browser observations prompted a focused source check; these are confirmed implementation mechanisms, not changes made during the audit:

- **BROWSER-01:** `server/models.mjs:123` declares `install` async, but `server/index.mjs:529` passes `models.install(id)` directly to JSON serialization without awaiting it. The client at `src/lib/store.ts:554` stores that response as the library; `src/app/Sidebar.tsx:58` then calls `.find` on absent `variants`. This explains the blank screen and independently continuing installation. Fix both the response contract and frontend recovery.
- **BROWSER-02:** `src/screens/Tuning.tsx:50–51` owns Beginner/raw drafts in component-local state, so unmounting the route drops them. Integrate draft ownership into CFG rather than adding isolated warnings to individual buttons.
- **BROWSER-03:** Raw config uses `textarea.st-input` with `minHeight: 380` in `Tuning.tsx`; it inherits the single-line capsule geometry. This requires an explicit multiline exception in the shared component styling.
- **BROWSER-04:** `src/components/ui.tsx:526` selects options only in `onMouseDown`; there is no option `onClick` handler. The visible audit result distinguishes semantic activation from native pointer activation.
- **BROWSER-05:** `src/styles/app.css:2201` reserves 340px when `data-context=true`; the narrower media rule changes the base grid and hides `.contextpanel`, but does not override the more specific open-state selector. Hiding the panel does not remove its reserved track.
- **BROWSER-06:** `src/screens/Setup.tsx:455` hardcodes 8791. Production and local design-host port contracts must still be distinguished as specified in section 17.
- **BROWSER-07:** `src/screens/Settings.tsx:118` uses `scrollIntoView({ block: 'start' })`; the observed target had no scroll margin. Use the actual sticky stack height rather than another fixed offset that fails when controls wrap.

### Existing findings confirmed through interaction

These are confirmations, not additional issue counts:

| Existing issue | Browser evidence |
|---|---|
| SETUP readiness/progress; NAV-05 | Setup claimed the engine was loaded before Start/open-chat, while the sidebar still showed installed rather than loaded. Later, tuning showed Engine idle while the sidebar said Ready for the next prompt. |
| RAW-01 | Entered `{ invalid json` and clicked Validate; success toast said “Configuration is valid” and “Everything fits this machine.” No parsing error appeared. |
| TUNE-02; EXP-02/RAW-01 | Beginner context changed to 128K while the summary retained saved memory values. Raw config still showed the saved Recommended/32K values. |
| MODEL-04; MODEL wording/disclosure | Show more expanded extra variants without a Hide action. The misleading smaller-variant claim remained visible; simulated installation did not gain a clear product label. |
| MON-01; MON-04 | Monitor printed the JSX implementation comment. Demo/synthetic hardware values were presented without per-group source disclosure. |
| SERVE-01; SERVE-03/04; UI-11 | Connection guide emitted a toast. Completed request detail showed current queue information. Escape did not close the request drawer, whereas Done did. |
| FILE-02 | Rescan produced only its toast; source confirms no rescan handler. Discuss in a new chat opened an empty composer and falsely reported Reference added. |
| PREF-08 | Searching `dark` hid Appearance even though Dark is a valid option; the extra About card remained as unrelated residual content. |
| CMD-01 | Shortcut help advertised Cmd+2 → Chat; pressing it opened Monitor. Help also retained the nonexistent Home mapping. |
| NAV-01; NAV-07 | Sidebar folder only expanded conversations. Workspace was reachable through palette/slash commands. At 390px the expanded sidebar consumed 268px; collapsed tuning still clipped headers/actions. At 1024px expanded navigation continued consuming content width. |

### Screenshot evidence

Screenshots are raw browser captures from this isolated preview. They document observed layouts, not proposed designs or verified production telemetry.

- [Raw editor geometry](evidence/ui-audit-2026-10-05/raw-editor.jpg) — capsule corners clip the raw input at desktop width.
- [Narrow tuning with expanded sidebar](evidence/ui-audit-2026-10-05/narrow-tuning.jpg) — navigation consumes most of the 390px viewport and main actions leave the frame.
- [Narrow chat after opening context](evidence/ui-audit-2026-10-05/narrow-chat-inspector.jpg) — hidden inspector still consumes the conversation track.
- [Settings section anchor](evidence/ui-audit-2026-10-05/settings-anchor.jpg) — Engine title/action are obscured by the sticky navigation after section activation.

**Initial implementation recommendation:** BROWSER-01 was the first priority, followed by the shared draft and endpoint contracts. BROWSER-03–05/07 were functional and accessibility requirements. These baseline findings are now resolved as recorded in section 21.

## 21. Implementation status and verification

### Delivered pass

**Date:** 5 October 2026. The interface was opened, clicked through, changed, and rechecked in the browser. This pass implements the screen hierarchy and responsive behavior, corrects the confirmed source/browser defects, and records production capability gaps. It preserves React, GT Walsheim, Anron icons, Strata green, the kit files, and the current routes. No dependency or animation framework was added.

The web materials approximate Apple's hierarchy and material principles: floating controls use restrained blur, reading/form surfaces stay opaque, sheets use a simple dimmed backdrop, and motion/transparency preferences have fallbacks. This is not native platform Liquid Glass.

The local design host now distinguishes simulated operations from real inference. Demo mode cannot discover and use an unrelated running inference service, and its restart action cannot accidentally launch a real engine. Preview state remains in a separate temporary home. Browser test conversations are local to the preview origin.

### Current disposition ledger

The ranges below include every original issue ID. Their required behavior, priority, dependency and acceptance criteria remain in the original issue tables. **Implemented** describes code delivered in this pass, not a claim that every acceptance scenario was executed. Exceptions and remaining verification are listed immediately after the ledger.

| Issue coverage | Current disposition | Delivered result / implementation owner |
|---|---|---|
| NAV-01–04 | Implemented | Expanded search, named workspace/file action, pinned/recent groups, app conversation dialogs, overflow actions, compact recent-chat disclosure and archived-chat discovery. `Sidebar.tsx`. |
| NAV-05–07 | Implemented | Consistent readiness vocabulary, quieter headings, expanded/compact/overlay navigation, responsive forms and inspector. `App.tsx`, store, responsive helpers and app CSS. |
| CMD-01–03 | Implemented; consolidation completed in section 24 | One descriptor registry now supplies keyboard bindings/help, palette actions and slash aliases, including dynamic installed-model actions and availability. Common/recent results remain short; lifecycle actions use the same store guards. |
| BOOT-01–04 | Implemented | Route-shaped loading, plain connection copy/details, retained drafts and conversations, response-contract guard, recoverable error boundary and explicit demo state. Stale SSE detection and reconnect refresh added. |
| SETUP-01–03 | Implemented with unavailable controls suppressed | No artificial scan delay; one recommendation; comparison disclosure; persisted choices. Unsupported memory/image options do not imply a working capability. |
| SETUP-04–07 | Implemented | Download, load and readiness are distinct; completion requires confirmed load; Finish later preserves unfinished progress; retry wording describes this host's restart-from-beginning behavior; Open chat follows readiness. |
| CHAT-01–03 | Implemented | Simple first-use prompt/examples, honest no-model action, removed inert attachments, open assistant prose, collapsed tools and optional response statistics. |
| CHAT-04–05 | Implemented | Retry targets the chosen request and creates a preserved branch. Edit prefills a separate branch; the original transcript survives. Retry clears the automatically sent prompt from the new composer. |
| CHAT-06–07 | Implemented UI; harness dependency | Composer Enter never approves tools. Approvals identify actions; demo execution is labelled. Conversation details are scoped and engine estimates identify their scope. Real tool execution/permission contracts remain unavailable. |
| CHAT-08 | Implemented; export delivery unverified | Fabricated compaction removed. Export builds ordered Markdown and initiates a Blob download without a false success toast. The in-app browser did not return a download event; normal-browser file delivery still needs confirmation. |
| CHAT-09–10 | Implemented | Reduced-motion scrolling, readable older replies, reader-follow behavior preserved, safe headings/links/lists/tables/fences and copy without a language label. Incomplete list-marker streaming freeze fixed. |
| RESP-01–04 | Implemented | Response settings, simple Randomness plus Advanced, explicit All requests scope, staged Save/Discard, sampling-only reset and refreshed values. Unsupported Show thinking switch removed. |
| CTX-01–04 | Implemented UI; data/capability dependency | Short inspector, remembered disclosures, scoped recorded statistics, Not available for missing conversation data, no duplicate hardware, responsive modal sheet and focus handling. No invented tool catalog/permissions. |
| MODEL-01–02 | Implemented | Running / Installed / Available, concise metadata, technical Details and truthful quantization wording. |
| MODEL-03 | Partially implemented; estimate dependency | Fit is explicitly estimated and shared across entry points. Current recommendation uses memory class, not measured free headroom plus a complete native-engine footprint. |
| MODEL-04–08 | Implemented | Reversible variant comparison, one operation progress location, shared fit/busy guards, confirmed load/unload/install outcomes, accurate storage labels and visible simulated library. |
| TUNE-01–04 | Implemented UI; capability dependency | Response/resource groups, draft impact, unavailable image control, full retained draft, validation, discard, Save for next load and real-engine Apply now explanation. |
| EXP-01–02 | Implemented | Five searchable groups, keys/units/effects/application timing, Changed only, per-field revert and staged edits shared with Raw. |
| EXP-03 | Integration dependency | The inspected host is a llama-server adapter. Its catalog is not full native Strata **JSON run-config** parity. Unmapped parameters are labelled Preserved only instead of pretending to apply. Production setup writes `strata-<model>.json`; see section 23. |
| RAW-01–04 | Implemented preservation; adapter dependency | Raw contents are parsed, validated and saved; separate import/export, explicit reload/discard and unknown-key retention. Retention does not promise native engine interpretation. |
| MON-01–02 | Implemented | Removed rendered implementation comment; distinct phases, reading-only progress, useful idle results, fewer repeated totals and no arbitrary uptime bar. |
| MON-03 | Implemented | Separate zero-based speed scales, common window, actual sample timestamps for horizontal position/readout, keyboard sample inspection and accessible units. |
| MON-04–06 | Implemented UI/source correction; sensor dependency | Demo/Estimated/Measured/Unavailable labels, no unavailable-throughput zero chart, absent real sensors shown unavailable, shared-memory labels and one actionable estimated-memory warning. Exact native telemetry remains a dependency. |
| MON-07 | Implemented | Frozen model/time/source-labelled snapshot, hidden navigation/paths/request content, 1440×900 layout and keyboard exit with focus return. |
| SERVE-01–04 | Implemented | Compact connection facts, actual protocol guide, tab/panel keyboard relationships, listening versus readiness, selected request derived by ID, recorded history, accessible rows/sort headers and shared drawer. |
| SERVE-05 | Implemented in current host; human credential workflow not tested | Masked Set/Change, reveal of the typed draft, explicit Remove confirmation, network-bound removal guard and confirmed outcomes. Only disposable test keys were changed by contract tests; the browser check opened an empty editor and verified its input type. |
| SERVE-06–07 | Implemented in current host; native verification deferred | Auth placeholders, labelled templates, pause semantics and actual cancellation through an abortable serial queue. Queue lifetime is explicitly in memory. Completion events cannot be overwritten by stale cancel responses. Native-engine cancellation remains unverified. |
| FILE-01–06 | Implemented UI; harness dependency | Folder-first path disclosure, narrow preview sheet, wrap/local code overflow, actual rescan, real preview text passed to an unsent chat draft, separate folder expansion, valid filtered tree focus and distinct read errors. Workspace markers are scoped; unverified sandbox claims removed. |
| SKILL-01–04 | Implemented for the current test adapter; native harness deferred | Enabled/scope controls, effective counts, readable/raw instructions and folder import retained. Enabled trigger matches in the correct workspace now supply instructions to the test adapter. Demo replies do not consume skills; usage counts mean inclusion, not successful execution. Native tools/harness remain separate. |
| PREF-01–09 | Implemented | Only functioning preferences/actions; Light/Dark/System, preview, size/density/accessibility choices, tuning/model links, honest storage/privacy, one About/setup entry, descriptor search, scoped appearance reset and focus confined to chat. |
| UI-01–04 | Implemented / retained | Named/pending buttons, Anron sprite retained, semantic quieter headings and shared field label/hint IDs. Form errors remain local; full assistive-technology invalid-state announcements need further verification. |
| UI-05–09 | Implemented | Radio-style choices; real protocol tabs/panel; option click/keyboard/active announcements; scoped status; named progress with indeterminate semantics; truthful setup steps; accessible request/tree rows and sort headers. |
| UI-10 | Implemented active surfaces; sensors unavailable | Narrow-safe metrics, values/units/window/source, actual timed chart points and keyboard readouts. Unused legacy gauge/popover internals retained; active inspector no longer uses their duplicated global metrics. |
| UI-11–12 | Implemented sheets/dialogs; menu refinement partial | Focus trapping, ancestor background inertness, top-surface Escape, click-away scrims and focus return for active modal surfaces. Overflow uses native disclosures/buttons; a single coordinated arrow-navigation menu system remains optional follow-up work. |
| UI-13–15 | Implemented | Confirmed outcomes, persistent action errors, genuine/filtered/disconnected empty states, safe Markdown/code and readable skill instructions. Existing toast cap/pause/live region retained. |
| UI-16–17 | Implemented; performance unmeasured | Restrained token-gated floating blur, opaque content/sheets, dim scrims, no nested blur in inspected screens, motion tokens and reduced-motion JavaScript scroll. No frame-rate/resource claim. |
| UI-18 | Retained and refined; verification partial | Focus/contrast/scale behavior preserved; opaque/high-contrast enlarged controls and both themes checked. Full screen reader, 200% browser zoom and all interactive-state contrast remain unverified. |
| BROWSER-01–07 | Fixed | Awaited install response, response validation/error boundary, retained full configuration draft, finite editor radius, semantic Select click, narrow inspector sheet, public/internal endpoint distinction and visible Settings anchors. |

### Additional findings during implementation

| ID / priority / kind | Observed problem | Fix / dependency | Acceptance and current evidence |
|---|---|---|---|
| BROWSER-08 / P0 / Defect | A response stream ending temporarily at `1. ` or `- ` entered a Markdown list branch whose inner loop did not advance. The browser froze/crashed; reopening the stored partial answer reproduced it. | Outer list detection now requires non-whitespace item content. Incomplete markers render as ordinary text. CHAT-10 / UI-15. | Every prefix of a mixed Markdown answer and all incomplete list markers render without looping in the SSR regression. Fresh browser streams completed after the fix. **Fixed.** |
| BROWSER-09 / P1 / Defect | Setup's five labelled segments clipped in the phone viewport. | Show a readable current-step sentence and five compact numbered segments at narrow widths; labels remain accessible. UI-08 / SETUP. | All five steps remain represented at 390px; Ready's step rail and action are visible in the saved phone capture. **Fixed.** |
| BROWSER-10 / P1 / Defect | Exiting Snapshot lost keyboard focus because the trigger was unmounted with the replaced screen. | Restore focus to the remounted Snapshot button after closing. MON-07 / UI-11. | Browser Tab stays within Snapshot; Escape closes it and focus returns to Snapshot view. **Fixed.** |
| BROWSER-11 / P0 / Defect | Reloading a stored interrupted stream retained its streaming flag, hid Retry, and an error hid the partial answer. | Normalize interrupted saved messages to a recoverable state; show preserved partial content alongside its error and enable Retry. CHAT-09/10. | Reopening the originally frozen conversation shows the interrupted notice and Copy/Ask again while preserving the partial text. **Fixed.** |

Two existing requirements also gained concrete corrections in the final pass: Retry originally left its already-sent request in the new composer (CHAT-04), and a stalled SSE socket could retain old request data after host restart (BOOT-02/03). Retry now clears only its automatic draft; the event stream detects silence and reconnect refreshes host facts in place.

### Verification record

All browser interactions used the in-app browser: actual clicks, inputs, key presses and read-only rendered-DOM inspection. Shell checks did not operate the browser. The preview used `STRATA_REAL_ENGINE=0` throughout; no real weights or inference service was used.

| Evidence | Check actually performed | Result / limits |
|---|---|---|
| E1 — Configuration | Browser changed Beginner/Expert/Raw, navigated away and reloaded, entered invalid JSON, validated and saved a raw-only key plus extra metadata. | Draft retained; invalid JSON blocked; real saved output contained custom value 42 and extra metadata. Raw/visual editors share one canonical draft. |
| E2 — Response settings | Keyboard changed Randomness from 0.60 to 0.05; host remained 0.60 before Save, then became 0.05. Reset staged 0.60 while host stayed 0.05; saving reset changed only sampling. | Batch 2048, custom key 42, quality preference and extra metadata were preserved. This verifies staged application and scoped reset. |
| E3 — Chat/workspace | Sent demo prompts; retried and edited older turns; inspected original/branch histories; filtered/select a real workspace file; Ask about this file. | Correct request targeted, original preserved, edit draft prefilled, retry composer empty, file preview content appeared in an unsent chat draft. Fresh partial Markdown stream completed. |
| E4 — Setup/models | Clicked all five setup steps, compared variants, loaded an existing demo install, opened chat, expanded/collapsed model alternatives and unloaded/reloaded. | Ready followed load; no-model/installed/loading/running remained distinct. Backend regression additionally covers interrupted setup and cancellation. Real downloads/load failures not executed. |
| E5 — Shared navigation | Sidebar/More, palette search/common actions, shortcuts, phone overlay outside click/Escape, inspector and response sheet. | Reachable navigation, matching shortcut destinations, named controls and dismissible surfaces. Full OS screen-reader testing not performed. |
| E6 — Serving | Opened recorded request details, inspected recorded tokens/duration, keyboard Tab/Escape and focus return; switched guide with ArrowRight. | No fabricated queue position. Request drawer returned focus; protocol tab selected Anthropic and labelled its panel. The update-by-ID implementation was inspected; a live request transition was not caught in the drawer. |
| E7 — Responsive matrix | Eight main routes at 390, 768, 1024 and 1512px; eight routes at 390px with 125% interface scale, reduced motion, opaque materials and increased contrast. | 32 standard-width and 8 enlarged-control checks had no page-wide horizontal overflow. These are reflow/geometry checks, not true 200% browser zoom. |
| E8 — Themes/materials | Read-only visible leaf-text contrast inspection on eight routes in Light and Dark; inspected large dark surfaces and nested backdrop filters. | 429 visible text checks per theme, no failures under 4.5:1 normal / 3:1 qualifying large text in those samples. No nested blur; no excessive large bright dark-theme surface. Hidden, disabled, hovered and all focus states were not exhaustively measured. |
| E9 — Snapshot/charts | At 1440×900 entered Snapshot, inspected frozen data twice, captured it, used keyboard dismissal; used Home to inspect chart's first timestamped sample. | Snapshot fits without page scroll, model/time/source are present, paths/chat/request content absent, values frozen, focus restored. Chart reports value/unit/sample age. |
| E10 — Static/build | `npm run typecheck`, `npm run audit`, `npm run build`. | Passed. Kit remains in sync; token/class/icon/material audits clean. Build size is recorded below as an artifact size, not a runtime-resource benchmark. |
| E11 — Reliability regression | `node --test tests/reliability.test.mjs`. | 2 tests passed: incomplete Markdown prefixes and disposable demo-host setup/config/lifecycle outcomes. No model downloads/browser automation. |
| E12 — Restart/recovery | Kept an unsent draft in a selected chat, stopped the isolated host, observed the offline banner, restarted the same host and let the stream reconnect. Reopened the earlier interrupted answer. | Offline detected; automatic refresh removed the banner while retaining the selected chat and exact draft. Partial answer remained visible with an interrupted notice and Copy/Ask again. Draft was cleared after the test. |

The original `npm run check`, flow/contrast/measure/luminance/motion scripts launch a separate browser harness. They were not run as written; the rendered checks above were performed through the supported browser tool instead. This is not an equivalent claim that every legacy harness gate passed.

Final checks passed after the transport and interrupted-answer corrections: typecheck, UI audit, production build, both reliability tests (about 14 seconds) and syntax checks for every server module. The build contains 59 modules: 375.39 kB JavaScript / 114.70 kB gzip and 82.92 kB CSS / 15.36 kB gzip. These are artifact sizes. Streaming responsiveness, idle CPU/RAM, native glass cost and motion frames were **not benchmarked**.

### Evidence artifacts

- [Responsive matrix](evidence/ui-audit-2026-10-05/responsive-verification.json) and [enlarged controls](evidence/ui-audit-2026-10-05/large-interface-verification.json).
- [Light rendered checks](evidence/ui-audit-2026-10-05/light-visual-verification.json) and [Dark rendered checks](evidence/ui-audit-2026-10-05/dark-visual-verification.json).
- [Monitor snapshot](evidence/ui-audit-2026-10-05/after-monitor-snapshot.jpg), [desktop Monitor](evidence/ui-audit-2026-10-05/after-monitor-desktop.jpg), [mobile chat](evidence/ui-audit-2026-10-05/after-chat-mobile.jpg) and [mobile setup](evidence/ui-audit-2026-10-05/after-setup-mobile.jpg).
- [Desktop chat](evidence/ui-audit-2026-10-05/after-chat-desktop.jpg). Temporary viewport overrides were reset after responsive testing; the reviewed preview was left open.
- [Changed files compared with the inspected baseline](evidence/ui-audit-2026-10-05/changed-files.json). The comparison covers the 119 baseline files; 30 of those changed and none are UI-kit files. New helpers, regression tests, this report and evidence are additional files.

The original screenshots in section 20 are **before** evidence. They do not represent unresolved defects in the current build. New captures are **after** evidence from simulated operation, not native telemetry benchmarks.

### Remaining work and exact boundaries

| Priority | Remaining requirement | Dependency / acceptance before claiming completion |
|---|---|---|
| P0 | Full native Strata expert configuration parity | Map the actual production `strata-<model>.json` schema, its `args`/`env`/server fields, capability/version and application timing. Prototype `strata.engine/1` keys must not be described as applied until an adapter translates and verifies them against the native engine. EXP-03 / RAW-03; section 23. |
| P0 | Real harness tools, images and skill application | Report available tools/modalities/grants and execute approved actions/inject effective skills through the real harness. Existing UI suppresses or labels unavailable/demo behavior. CHAT-02/06, CTX-03, TUNE-03, SKILL. |
| P0 | Precise request/conversation telemetry | Provide actual token accounting, prompt progress, context ownership, latency and sensor provenance. Request-log token/rate estimates and model-footprint estimates are labelled; unavailable sensors are absent. CTX-02 / MON-04 / DATA. |
| P0 | Production privacy wording when native Strata is connected | The prototype's request log is metadata-only. Native Strata's optional API monitor retains request and response content in memory. Settings/Serving copy must report the connected backend's actual collection and retention, with authentication and a clear opt-in boundary. PREF / SERVE / CAP; section 23. |
| P1 | Capacity recommendation accuracy | Include measured free headroom and native runtime/context footprint, while retaining estimate labels and shared fit guards. MODEL-03. |
| Completed in section 24 | Key-management and current-host cancellation | Remove/reveal and cancellation are implemented with disposable contract/browser evidence. Queues are deliberately in memory, with their lifetime stated. Native credential/cancellation paths remain deferred. |
| Completed / optional refinement | Command consolidation | Supported palette/slash actions now use the shared registry. Existing native disclosures and the sidebar’s keyboard menu remain; replacing every disclosure with a coordinated menu system is optional. CMD / UI-11/12. |
| Verification | Real-engine lifecycle and credential paths | Exercise confirmed success/failure on a disposable real-engine configuration; perform credential changes through the human user. Current pass did not run them. |
| Verification | Export delivery | Confirm the created Markdown Blob downloads in a normal supported browser/desktop shell. The in-app browser exposed no completed download event; no false success is shown. |
| Verification | Accessibility and performance | Test screen reader, all focus/hover/control contrast, OS preference changes, true 200% zoom, irregular-time telemetry and measured stream/scroll/glass resource cost. Current static and browser sample checks do not certify these. |

**Completion of this UI pass:** all 116 original issue IDs are covered by the current ledger, with four additional browser findings (120 total). Every route, setup step, tuning view and component group has a current disposition; confirmed browser defects are fixed, evidence is saved, and remaining integration/refinement/verification is named. Future work should use this ledger rather than treating all baseline issues as still pending or all engine capabilities as implemented. The 12 brand additions below are a subsequent specification; this completion statement does not mark them implemented.

## 22. Logo assets and implementation requirements

### Review scope and current status

**Added:** 5 October 2026 as a report-only addendum. **Latest implementation status: see section 24.** This section preserves the source audit and acceptance requirements. At the time of this review, only the document and inventory changed; the later authorized UI pass integrated derived runtime assets while preserving all supplied masters.

The [Brand Guidelines PDF](../../Strata-Logo-DesignFiles/Strata-Logo.pdf) is the authority for logo geometry, colors, clear space and permitted treatments. The UI kit remains the authority for interface typography, controls, spacing and semantic colors. Placement sizes below are proposed application defaults, within the brand limits. The PDF's reference to one particular model explains the identity; it does not establish which models the current engine supports.

The package contains **37 deliverables: 11 SVGs, 23 PNGs, one eight-page PDF, one HTML head snippet and one web manifest**. Two `.DS_Store` files are excluded. Exact source paths, sizes, dimensions and SHA-256 hashes are recorded in [asset-inventory.json](evidence/brand-audit-2026-10-05/asset-inventory.json).

| Evidence | Confirmed observation | Consequence |
|---|---|---|
| ASSET-E1 — current `index.html` and `public/` | The document requests `/favicon.svg`; that file is absent from the application's public source directory. | Adding the supplied folder beside the app does not bundle it. A runtime export/copy step is required. |
| ASSET-E2 — supplied `web/favicon.svg` | The root has `fill="none"`. Its three rectangles only have `i`/`a` classes; no styles or explicit fills define their colors. | This standalone SVG has no visible artwork. The automatic Light/Dark behavior illustrated on PDF page 7 is not implemented in the export. |
| ASSET-E3 — both supplied lockup SVGs | The wordmark is live `<text>` using `StrataWord`, `GT Walsheim Pro`, then system fallbacks. Neither SVG includes the font or outlined lettering. | Appearance depends on font availability. The app's own webfont declaration is not a portable font source for an external SVG image. Preserve the approved lettering through a self-contained export. |
| ASSET-E4 — shell source and saved after screenshots | Desktop uses the Anron layers glyph in a green tile, capitalized typed `Strata` and `LOCAL AI`. Mobile and Snapshot also use typed branding. | The supplied three-bar logo and lowercase wordmark are not yet integrated. Existing after screenshots establish the old treatment, not approval of the new one. |
| ASSET-E5 — `server/index.mjs` | The static MIME map supports SVG/PNG but lacks `.webmanifest`. Missing static files fall back to `index.html`. | Add the manifest content type when shipping it; verify asset body/type as well as HTTP status. A successful response can contain HTML instead of an icon. |
| ASSET-E6 — all 11 SVGs | Each includes C2PA metadata. | Preserve the supplied masters and record derived exports. Do not silently strip provenance or claim an altered file retains an unchanged validated signature. |
| ASSET-E7 — native Strata web host | Production Strata separately serves `serve/web/index.html` and `serve/web/monitor.html`; its `/web/` asset handler currently allows CSS, JS and SVG, while this design prototype uses Vite `public/` and a different Node static host. | Shipping assets in the prototype does not put them into the production browser UI. Choose the deployment owner, then add the required native paths and MIME types if that UI is the target. See section 23. |

These observations come from source inspection. Browser rendering of newly integrated logos, native icon installation and HTTP delivery are still unverified. SVG images have restrictions on external resources, so font/style independence must be checked in the actual image context. [MDN: SVG as an image](https://developer.mozilla.org/en-US/docs/Web/SVG/Guides/SVG_as_an_image)

### Brand rules to preserve

- **Artwork:** three horizontal rounded bars on a 48×48 grid. Outer bars are 26×7 units; the middle bar is 38×7; radius is 3.5 and the vertical gap is 4.5. Preserve the supplied proportions and optical placement. Never substitute the generic `layers` control icon for the brand.
- **Wordmark:** use the supplied lowercase `strata` lettering and its spacing. Ordinary prose, accessible names and the product/window name remain `Strata`.
- **Light background:** outer bars/wordmark `#221E1F`; middle bar `#10B981`. **Dark background:** outer bars/wordmark `#EEF1F3`; middle bar `#34D399`. Reference backgrounds are Paper `#F9FAFB`, Night `#0E1113`; the dark app-icon tile is `#161A1D`.
- **Color scope:** these are artwork colors. Keep existing tested UI tokens for buttons, links and status. A brand-color update must not globally retint controls or undo the contrast work in UI-18. Do not recolor the mark to signal Reading, Error, Loading or Ready; keep the status indicator separate.
- **Clear space:** page 4 defines `x` as one bar's height, 7 source units. Leave at least `2x` around the visible artwork. At a 48px mark this is 14px; scale proportionally. Transparent space already inside the asset counts, but measure from the visible artwork to surrounding text, controls and edges.
- **Minimums:** the mark must not render below 16px; the full lockup must not render below 20px high. Those are limits, not default touch-target sizes. Prefer a 32px mark in compact navigation and a 28px-high lockup in expanded navigation. Keep intrinsic aspect ratios: 1:1 for the mark and 182:48 for the lockup.
- **Materials:** the logo stays flat and crisp. Glass may belong to its navigation container; it must not blur, distort, glow, cast a new artwork shadow or animate the bars. Prefer a stable opaque backing when surrounding content would impair visibility.
- **Restricted variants:** the guide reserves monochrome black/white marks for single-color print and embossing. Keep them in the master package; do not substitute them for normal Light/Dark interface artwork. Rounded app-icon tiles are separate assets, not wrappers to add around every in-app mark.

### Asset selection and destination map

Source paths below are relative to `design-iteration/Strata-Logo-DesignFiles`. Proposed runtime paths are relative to the Strata app. Only selected, validated runtime assets should enter `public/` and the build; preserve the complete source package outside it.

| Source assets | Intended role | Proposed delivery / required preparation |
|---|---|---|
| `logo/strata-mark-light.svg`, `logo/strata-mark-dark.svg` | Compact rail, narrow brand treatment, small setup identifier | `public/brand/strata-mark-{light,dark}.svg`; preserve geometry and explicit fills. |
| `logo/strata-lockup-light.svg`, `logo/strata-lockup-dark.svg` | Expanded sidebar, mobile header where it fits, About and Monitor Snapshot | `public/brand/strata-lockup-{light,dark}.svg` after resolving the live-font dependency. Outline the approved lettering in a derived export and compare it with the supplied reference. Do not retype it with a convenient system font. |
| `logo/png/strata-lockup-{light,dark}-4x.png` — 728×192 | Deterministic interim wordmark or raster consumer | Approved temporary alternative if the SVG lettering cannot yet be made self-contained; serve only if used. Reserve the same display dimensions and verify both backgrounds. Replace with validated vector exports for general scaling. |
| `logo/png/strata-mark-{light,dark}-512.png` | Raster-only documentation/export consumer | Retain as supplied. The React shell should normally use SVG; do not load 512px raster marks where a small vector is available. |
| `logo/strata-mark-mono-{black,white}.svg` | Single-color print/embossing | Reference masters only for this web-app pass. |
| `web/favicon.svg` | Browser tab / bookmark icon | Repair as a self-contained, theme-aware SVG, then deliver at `public/favicon.svg`. It must work independently of app CSS and JavaScript. |
| `web/favicon-{16,32,48}.png`, `web/favicon-dark-{16,32,48}.png` | Raster favicon fallbacks | Use matching `public/` filenames and explicit sizes for the chosen fallback entries. Define/test light and dark fallback selection; the supplied snippet currently references only the light 32px file. An ICO is optional for a demonstrated target need, not a required new asset. |
| `web/apple-touch-icon.png` — 180×180 | Add-to-home-screen touch icon | `public/apple-touch-icon.png`; retain the supplied opaque dark tile. Do not add another rounded mask. |
| `web/icon-192.png`, `web/icon-512.png` | Manifest icons | Same filenames in `public/`, linked from the reviewed manifest. Their opaque square backgrounds suit system masking; verify safe areas before declaring maskable support. |
| `web/head-snippet.html`, `web/site.webmanifest` | Integration reference / install metadata | Merge the relevant head entries into `index.html`; do not publish the snippet as a second page. Place a reviewed manifest at `public/site.webmanifest` only with deliberate launch/scope/display behavior. |
| `app-icon/app-icon-{light,dark}.svg` and corresponding 128/256/512/1024 PNGs | Rounded desktop/documentation illustrations; platform packaging inputs where appropriate | Keep in the source package unless consumed by a selected platform or document. Dark is the guide's default dock/taskbar/store treatment. These are not 16px toolbar glyphs. |
| `app-icon/app-icon-{light,dark}-square.svg`, `app-icon/png/app-icon-{light,dark}-square-1024.png` | Unmasked platform/store artwork | Inputs for future platform packaging. Keep square; the target OS/build tool supplies its required mask. |
| `Strata-Logo.pdf` | Brand reference | Keep linked from this report; do not ship it in the initial app bundle. |

For the current root-served app, the supplied root URLs are an appropriate starting point. If deployment under a path prefix is introduced, update HTML, manifest and asset URLs together. Manifest-relative icon paths resolve against the manifest URL. The existing manifest is a starting template: missing optional members do not make it invalid, but identity, launch URL, scope and display behavior must be chosen before presenting an install experience. Its icons currently default to `purpose: any`; only add `maskable` after mask testing. [MDN: Manifest icons](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Manifest/Reference/icons)

### Placement by surface

Keep branding concentrated at the shell and the few surfaces that can stand alone. Use the shared artwork component instead of separate hand-built wordmarks.

| Surface | Required treatment | Preserve / regression check |
|---|---|---|
| Expanded sidebar, every route | Replace the green layers tile plus typed wordmark/caption with one 28px-high lockup, approximately 106px wide. Preserve the existing Chat link and accessible action name. | No duplicate `Strata` or `LOCAL AI` beside it. Keep workspace, status and navigation below it; remove obsolete brand-only CSS. NAV-01/05/06. |
| Compact 64px rail | Use the 32px mark inside an approximately 48px interaction area, with the brand's clear space maintained. | A visible keyboard focus ring must fit without clipping; `Strata — open chat` identifies the destination. The action must preserve unsent drafts. NAV-04/07. |
| Mobile top bar and sidebar overlay | Replace the typed header branding with a 24px-high lockup where space allows. Use the mark when enlarged controls leave insufficient room. | Keep Open navigation and New chat as distinct controls. Overlay navigation uses the same assets; do not show two exposed brand links when the background is inert. Test the 390px layout and larger interface size. |
| Launch / reconnect / recovery | Retain the shell's logo. Use a small static mark only if a recovery surface is displayed without that shell. | No extra splash stage, loading delay or animated logo. Loading and connection status remain separate readable text. BOOT-01–04. |
| Setup: computer, model, basics, download/load, ready | One small mark beside the existing setup heading, shared across all five steps. Use a noninteractive decorative image when the heading already names Strata. | Preserve the semantic heading, steps, demo notice, Back/Finish later and state retention. Do not add a second introduction card or make the logo silently exit setup. SETUP-01–07 / BROWSER-09. |
| Chat: empty, streaming, completed, tools and composer | Shell branding is sufficient. | Preserve the quiet introduction and 68-character prose measure. No new logo avatar on every message, watermark behind answers or animated mark in the composer. CHAT / UI-15–17. |
| Models, tuning Beginner/Expert/Raw, Serving and connection guide, Workspace, Skills | Use the shell logo; retain functional titles and Anron icons within content. | Do not put an app-icon tile or brand lockup on every card, tab or empty state. Existing capabilities and scopes stay readable. |
| Monitor live overview | Use shell branding only. | Current activity and metrics remain the focus. MON-01–06. |
| Monitor Snapshot | Add one compact lockup beside the `Monitor` heading. Keep a single semantic screen heading and an accessible Strata identity. | Preserve model, frozen timestamp, source labels, 1440×900 fit, hidden paths/content, Escape and focus return. A decorative image must not create a duplicate spoken `Strata`. MON-07 / BROWSER-10. |
| Settings | Appearance preview should reflect the selected logo variant; About gets one modest lockup beside version information. | Other settings sections need no extra branding. Keep Light default, persistent Light/Dark/System and the existing section search. PREF-02/06/08. |
| Palette, dialogs, drawers, menus and toasts | Keep existing task/action labels and functional icons. | No repeated decorative branding. Preserve focus management, action hierarchy and compact dimensions. UI-01/11–13. |
| Window/tab, bookmark and installed shortcut | Product name `Strata` plus the appropriate favicon/platform icon. | A model switch must not rename the product or replace its logo with a model badge. Installation artwork is distinct from the theme-responsive in-app lockup. |

### New implementation ledger

The table below is the original specification. At its creation these requirements were pending. **Current dispositions are in section 24**; BRAND-11 remains conditional on choosing a native distribution target. File references identify integration points, not work completed by this review. P0 engine/reliability dependencies in section 21 retain precedence; the brand pass is primarily P1 identity and delivery work.

| ID / priority / kind | Issue and required behavior | Owner / dependencies | Acceptance |
|---|---|---|---|
| **BRAND-01 / P1 / Design** | Assets exist outside the application with no deployment contract. Maintain the supplied package as masters, select runtime exports from the map, and record any repair/conversion against its source hash. Exclude reference-only files and `.DS_Store` from the app build. | Asset preparation; new shared `Brand` component and `public/brand/`; asset inventory. | Every shipped brand file has a named source/role. Masters remain unchanged; derived files have a reproducible preparation record. Existing Anron control icons remain intact. |
| **BRAND-02 / P1 / Defect** | The supplied SVG favicon is invisible. Add explicit light fills and self-contained dark-mode styling for the same three rectangles. Use the browser's color-scheme context for this standalone icon. | `web/favicon.svg` derived runtime export; BRAND-01/08. | The actual favicon URL displays three bars in light and dark browser chrome at 16/32/48px. Opening it standalone shows artwork without app styles. Raster fallback selection is verified. |
| **BRAND-03 / P1 / Defect** | The SVG wordmark depends on unembedded live text and can substitute lettering or alter its bounds. Produce self-contained outlined lettering from the approved design, retaining geometry/colors. Use the provided PNGs as an interim choice if needed. | Both lockup exports; exact GT Walsheim lettering and approved PDF/PNG reference; BRAND-01. | Light and dark exports match the reference with local fonts unavailable, have no clipping, and retain their appearance as external images and in screenshots. Font download timing does not change layout. |
| **BRAND-04 / P1 / Design** | Old shell branding differs from the supplied identity. Implement one shared mark/lockup component with explicit dimensions and decorative/named modes, then replace desktop, rail and mobile instances. Remove old brand-only green-tile/18px-SVG and weight-900 wordmark rules. | `src/app/Sidebar.tsx`, `src/app/App.tsx`, `src/styles/app.css`; BRAND-01/03/05/06/07. | All shell layouts show the same approved artwork. Chat navigation and draft retention work, no duplicate caption remains, and no global `Icon`/SVG rule resizes or recolors the brand. |
| **BRAND-05 / P1 / Design** | App theme and browser theme can disagree. Select in-app artwork from the resolved `html[data-theme]` already set by `applyPrefs`, including OS changes while System is selected. Keep favicon adaptation independent. | Shared Brand component; `src/lib/store.ts` preference contract; existing Light default and PREF-02. | Manual Light on a dark OS and manual Dark on a light OS both use the correct in-app artwork. System responds to OS changes. Initial/restored appearance does not visibly swap to an illegible variant. |
| **BRAND-06 / P1 / Design** | Current icon sizing rules would shrink or box the artwork incorrectly. Preserve aspect ratio, minimum dimensions and proportional clear space; reserve layout space before assets load. Collapse lockup to mark when needed instead of squeezing it. | Shared Brand sizing; sidebar/mobile/overlay CSS; NAV-07 and UI-18. | At 390/768/1024/1512px, enlarged interface settings and 200% browser zoom, the logo stays readable with no stretch, clip, lost focus outline or navigation overlap. Hit targets remain usable independently of artwork size. |
| **BRAND-07 / P1 / Design** | Image replacement can duplicate accessible names or introduce decorative motion. Give a linked brand one action name and hide its decorative image; name a standalone meaningful logo once. Provide a text fallback on load failure. Keep bars static and colors independent of engine state. | Shared Brand component; UI-01/16–18; existing reduced-motion/transparency/contrast settings. | Keyboard/screen-reader users encounter one named link where actionable and no extra focus target for decoration. Focus remains visible, fallback keeps the destination usable, and opaque/reduced-motion/forced-color contexts remain understandable. |
| **BRAND-08 / P1 / Defect** | Current HTML points at a missing public favicon; merely linking the supplied snippet leaves deployment gaps. Publish chosen files, merge head entries and prevent missing asset requests from appearing successful through the HTML fallback. Add `.webmanifest` MIME support if shipping it. | `index.html`, `public/`, `server/index.mjs` and build output; BRAND-01/02/03. | Development and built-host URLs return the intended SVG/PNG/manifest body and content type. Missing brand assets yield a real missing-resource response, while existing app route fallback still works. No new remote asset/font dependency. |
| **BRAND-09 / P1 / Dependency** | Supplied manifest/head entries are templates. Decide stable install identity, start URL, scope and standalone behavior; retain square opaque touch/manifest artwork. Define browser theme-color behavior for the app's manual/System preferences. Keep static manifest launch colors explicit. | `public/site.webmanifest`, head metadata and target browser support; BRAND-08. | Touch and manifest icons load; installed launch reaches the intended local UI/connection state. Mask previews retain the mark before any `maskable` declaration. Do not claim offline inference or full PWA behavior merely because a manifest exists. |
| **BRAND-10 / P2 / Design** | Standalone Snapshot/About/setup surfaces need a consistent identity without adding noise across every screen. Apply the placement matrix with shared artwork and no new logo panels. | Setup header in `App.tsx`, `Monitor.tsx`, `Settings.tsx`; BRAND-03–07; MON-07/PREF/SETUP. | All five setup steps retain actions and readiness semantics. Snapshot includes branding plus model/time/source and still fits 1440×900 without exposing private data. About and appearance preview use the same resolved theme. Other content surfaces remain uncluttered. |
| **BRAND-11 / P2 / Dependency** | PNG/SVG exports are not a complete native icon package. Once a target shell/OS is chosen, derive its required icon resources from the supplied square artwork and verify installed appearance. | Future desktop/platform packaging; no native wrapper is added by this plan. | The chosen target has its required build resources and appearance variants; installed dock/taskbar/window/store icons are inspected for crop, margins and double masking. Mark this conditional, not completed or a blocker for the current web app. |
| **BRAND-12 / P1 / Dependency** | Existing verification predates the supplied-logo integration. Add a focused brand regression pass and save new evidence without overwriting the old 120-finding record. | BRAND-01–10; section 21 baseline evidence and existing audits. | Brand screenshots and URL/type checks are saved; source masters still match inventory hashes. Existing UI checks remain green. Report exact checks performed and any unsupported browser/native cases; do not reuse historical evidence as proof of new artwork. |

### Native icon handoff

The guide recommends dark rounded icons for desktop/docs and square artwork for system masking. Apply those recommendations according to the actual target's packaging rules. Current Apple guidance uses unmasked square source artwork; the system applies the final mask. For an Apple Icon Composer workflow, prepare appropriate artwork layers and appearance variants and deliver its `.icon` output through Xcode. Preserve the recognizable three-bar silhouette and avoid baking a second corner mask or custom glass/shadow into the artwork. A CSS-glass web header does not produce a native Liquid Glass app icon. [Apple: App icons](https://developer.apple.com/design/human-interface-guidelines/app-icons), [Apple: Creating your app icon using Icon Composer](https://developer.apple.com/documentation/xcode/creating-your-app-icon-using-icon-composer)

No `.icon`, `.icns` or `.ico` package is supplied. Generate only the resources required by the shell/platform eventually selected; do not introduce a desktop framework just to consume these assets. Retain the supplied dark default and light artwork for appearance variants where supported. An installed OS icon does not necessarily change when the user changes Strata's in-app theme.

### Implementation sequence and release evidence

1. **Prepare assets:** preserve/hash the masters, repair the blank favicon, make the lockup self-contained and choose the runtime subset. Compare source and prepared artwork before replacing the existing identity. BRAND-01–03.
2. **Integrate the shell:** add shared mark/lockup behavior, resolved-theme selection, reserved dimensions, clear space, accessible naming and failure fallback. Replace expanded, rail and mobile instances together. BRAND-04–07.
3. **Complete standalone surfaces:** integrate setup/About/appearance preview and branded Snapshot according to the placement matrix; preserve existing focus and state behavior. BRAND-10.
4. **Connect distribution metadata:** publish icons, merge the head entries, handle MIME/missing files and review manifest launch behavior. Native packaging stays a separate conditional step. BRAND-08/09/11.
5. **Record new evidence:** verify the following matrix and update only the corresponding BRAND statuses. BRAND-12.

| Check | Required evidence / pass condition |
|---|---|
| Asset integrity | Compare all master hashes with the inventory. Inspect repaired SVGs for explicit colors and self-contained lettering. Preserve metadata/source relationships in the export record. |
| File delivery | Check favicon, both mark/lockup themes, selected fallback PNGs, touch icon and manifest from Vite and the built local host. Check response content type and body, not status alone; verify missing-asset behavior. |
| Theme matrix | Light/Dark/System × light/dark OS; also reduced transparency and increased contrast. Inspect browser-tab identity independently of page identity. Confirm stored preference at cold load and live System changes. |
| Layout and interaction | At 390/768/1024/1512px, check expanded/rail/mobile/overlay identity, 125% interface scale and true 200% browser zoom. Click the brand link, retain an unsent draft, and inspect keyboard focus and accessible names. |
| Quiet content | Visit all eight main routes plus all five setup steps. Check headings, buttons and content still have room; no repeated logos or new decorative loading stage. |
| Export/standalone surface | Capture branded Monitor Snapshot at 1440×900 in both themes. Keep frozen time, model and data-source labels, no paths or conversations, and working Escape/focus return. Check artwork after font/asset caching is cleared. |
| Small and installed icons | Inspect SVG and raster favicons at 16/32/48px; touch/manifest previews and supported installation flow. Add platform dock/taskbar/store captures only if BRAND-11 is in scope. |
| Regression gates | Run existing typecheck, UI audit and build after code integration. Recheck affected focus/contrast/reflow behavior. Update old `nav__mark` assumptions narrowly if an audit refers to the obsolete brand structure; retain the Anron control-icon checks. |
| Resource behavior | Inspect shipped asset sizes and network requests. Load only the active in-app variant where practical; reserve dimensions, use local cached assets and no new animation/runtime dependency. Do not claim measured CPU/RAM or frame-rate improvements without measurements. |

**Completion of the original addendum:** all 12 requirements and their source evidence were recorded. Section 24 now records the web integration and new asset-specific checks. Preserve the 120-item history and do not interpret the web-brand pass as native-engine or desktop-packaging completion.

## 23. Source reconciliation and implementation handoff

### What the current evidence proves

This section records the earlier **report-and-findings-only review**, before the authorized implementation in section 24. The earlier UI pass in section 21 remains the application baseline; the logo masters and current app artwork remain as described in section 22. The current source, saved browser evidence and supplied assets were cross-checked without loading a model. The 37 source-asset hashes in the [brand inventory](evidence/brand-audit-2026-10-05/asset-inventory.json) still match. After restoring the pre-logo application state, `npm run typecheck`, `npm run audit` and `npm run build` pass; this is a baseline integrity check, not proof that BRAND-01–12 were implemented. The current build contains 59 modules, consistent with the section 21 UI pass.

| Evidence class | What can be stated | What remains open |
|---|---|---|
| Source and supplied artwork | Current handlers, config formats, static routes, source asset geometry/hashes and missing public favicon are inspectable. | Runtime behavior can diverge from source under a different host or engine version. |
| Saved UI browser evidence | Section 21's simulated host was clicked through; 32 standard-width route checks and eight enlarged-control checks show no page-wide overflow. Both theme samples passed the recorded text-contrast inspection. | These captures precede logo integration and do not prove brand appearance, real-engine inference, native packaging, true browser zoom or screen-reader behavior. |
| Native Strata repository | The actual server and setup sources identify their public status/metrics APIs, model-run config, optional images/MCP and API-monitor retention. | No native engine, GPU, model, authenticated endpoint, installed app or hardware sensor was run in this documentation follow-up. |

The count remains **109 specification IDs + seven initial browser IDs + four implementation browser IDs = 120**, plus **12 BRAND IDs = 132**. Their status at this historical review was pending/conditional; section 24 records the subsequent implementation. A source clarification below does not silently turn an implemented UI item into a verified native-engine feature. Section 21 is the disposition ledger for the first 120; section 22 is the pending ledger for the additional 12.

### Production Strata and the design prototype are different applications today

| Area | Design prototype in `design-iteration/strata` | Actual repository in `Strata` | Implementation consequence |
|---|---|---|---|
| Browser UI and host | Vite/React interface in `src/`, Node host in `server/`, and static build in `dist/`. Design mode is simulated; its real-mode supervisor starts `llama-server`. | Python server in [`serve/server.py`](../../../Strata/serve/server.py) serves its own `serve/web/index.html` and `monitor.html`; setup and model execution use the native Strata engine. | Choose whether the React UI replaces, is packaged into, or remains separate from the native web UI. Copying assets into Vite `public/` alone cannot update production Strata. Do not rename the prototype's llama.cpp adapter to native Strata. |
| Configuration | The editable document is [`strata.engine/1`](../src/lib/config.ts), with `beginner` and dotted `expert` keys. The Node host maps only a subset to llama-server arguments and labels other values **Preserved only**. | [`setup.py`](../../../Strata/setup.py) writes `strata-<model>.json`; [`serve/server.py`](../../../Strata/serve/server.py) reads that JSON as `--config`. It contains an engine executable, argument list, paths, model name and server fields, with an optional `env` map. [INSTALL: storage](../../../Strata/docs/INSTALL.md#where-things-are-stored) describes the location. | There is no production `.env` file to mirror. Define a versioned translation and reverse import for each supported native field. Show saved versus currently applied values and restart timing; preserve unknown native keys without pretending they work in the prototype. EXP-03 / RAW-03 / CFG. |
| Models and lifecycle | The prototype catalog lists GGUF downloads and a llama-server supervisor. Its **real** mode means real llama.cpp inference, not the native Strata setup flow. | Native setup chooses Strata-supported families/variants, prepares model packs and writes a per-model run config. The Python server reports loaded state separately from API availability. | Do not carry prototype model IDs, fit estimates, download stages or internal ports over as production facts. Use the selected native config and live capability/status responses. MODEL-03 / SETUP / OPS. |
| Capabilities | The UI suppresses or labels tools/images/skills it cannot execute through its current host. | The native server supports image input only when a `vision` config is present; MCP tools are available when configured and the request opts in with `strata_mcp: true`. [`/health`, `/v1/models` and `/mcp`](../../../Strata/serve/server.py) expose parts of this state. | A production adapter can enable features conditionally. Do not turn on attachments or tool approvals solely because the backend has an endpoint; define file transfer, permission and skill-injection contracts first. CAP / CHAT / SKILL. |
| Monitoring | Prototype labels unavailable/simulated readings and uses its own request store. | Native [`/metrics`](../../../Strata/serve/server.py) returns engine/live/request/totals/hardware/history data, including optional prompt-read progress; `/status` and `/v1/status` supply further status. API-monitor records are a separate optional facility. | Map each visible metric to the native field and timestamp. Preserve nullable readings and origin labels, and distinguish the current request from the selected old conversation. Do not infer a conversation's context from the engine's latest request. DATA / MON / CTX. |
| Request privacy | Prototype Settings says its host log retains metadata without prompt text. That statement describes the inspected Node host. | Native `api_monitor` is **off by default**. If enabled, its last 100 in-memory records include request JSON and output/reasoning text; list responses omit content but a request-detail lookup can return it. [`begin_request` and `/api/requests`](../../../Strata/serve/server.py) show this behavior. | Make Privacy and Serving copy conditional on the connected backend and monitor mode before native integration. Do not promise “metadata only” for a native content log. Protect request-detail access and avoid exposing prompts in Snapshot. PREF / SERVE / CAP. |
| Static brand delivery | Vite `public/` publishes the current design app's root assets; its Node static host uses an HTML fallback for missing files and has no `.webmanifest` MIME entry. | The Python server has a separate `/web/` asset allowlist for CSS, JS and SVG, and serves its own HTML documents. It does not automatically consume Vite `public/` or PNG/manifest assets. | BRAND-08 needs a target-specific delivery plan. If the React UI is embedded in production Strata, package its built files and MIME/static routes there; if the old UI remains, integrate identity in that HTML/CSS too. Verify the actual URL body and type on each chosen host. |

The native sources above are evidence of **available seams**, not an assertion that this design app already consumes them. In particular, a working `/metrics` endpoint does not settle conversation attribution, and configured MCP tools do not themselves implement this prototype's approval workflow.

### Decisions to make before an engineer implements the logo

These decisions are deliberately explicit so the assets have one source of truth and no implementation guesses become product policy.

1. **Target surface:** identify the first shipping browser UI: the current React prototype, the native server's existing web UI, or a packaged React replacement. Specify its actual base URL and static-file owner. If both UIs ship, use one reviewed set of derived brand exports in both bundles and check each separately.
2. **Derived artwork:** keep all 37 supplied deliverables untouched. Repair the transparent favicon and replace live SVG text in the two lockups with approved self-contained outlines or use the supplied reference PNGs temporarily. Record the source hash, transformation, output hash and reviewer approval for each derivative. The C2PA claim in a changed source SVG must not be represented as a still-valid signature on its derivative.
3. **Theme ownership:** use resolved **in-app** Light/Dark/System for the sidebar/setup/About/Snapshot artwork. Browser-tab SVG favicon and raster fallbacks follow the browser/OS color scheme separately. Decide and test how the tab `theme-color` reacts to a manual in-app theme that differs from the OS. Keep the default Light preference recorded in section 2.
4. **Installation scope:** review manifest ID, start URL, scope, display and icon purposes against the actual served path. The supplied manifest has no explicit ID/start URL/scope and advertises standalone; it is a template, not evidence of an installable, offline or native app. Keep touch/manifest icons square and opaque; do not mark them maskable until a mask preview passes.
5. **Native package:** leave BRAND-11 conditional until the desktop shell and OS targets exist. The supplied square/rounded SVGs and PNGs are inputs, not `.icon`, `.icns` or `.ico` outputs. Icon Composer, dock/taskbar and store checks belong to that packaging decision, not to the React logo pass.

### Acceptance sequence for the next implementation pass

| Gate | Reviewable outcome | Evidence to save |
|---|---|---|
| A — Asset preparation | Chosen mark/lockup/favicon exports are self-contained, retain the approved geometry/colors, and have traceable source/output hashes. Original files are unchanged. | Generated-file list and hash record; rendered 16/32/48px favicon and Light/Dark lockup comparisons to supplied references. |
| B — Shell and standalone surfaces | One shared logo treatment replaces the old desktop/rail/mobile branding; setup, About and Snapshot use the placement matrix without repeating logos in message/content cards. | 390/768/1024/1512px captures, five setup steps, both themes and a 1440×900 Snapshot in each theme. |
| C — Behavior and access | Brand navigation keeps unsent draft; focus and named link are intact; Manual/System theme changes, reduced motion/transparency, increased contrast and failure fallback remain legible. | Keyboard and accessible-name inspection, cold-load/manual-theme matrix, asset-failure case and true 200% browser-zoom result. Record unsupported cases as unverified. |
| D — Delivery | Every referenced SVG/PNG/manifest returns image/manifest content from the **chosen shipping host**, and a missing asset produces a genuine missing-resource result while app routes still work. | Dev/built/native-host URL, status, content type and body checks as applicable; install/launch capture only where the target supports it. |
| E — Regression | UI screens and controls retain the section 21 behavior. Any native adapter separately validates config, lifecycle, capability, telemetry and privacy contracts before declaring native parity. | Typecheck, UI audit, build and affected flows; then real-engine results on supported hardware. No benchmark claim without measured CPU/RAM/latency. |

**Boundary of that report-only review:** the report, 120-finding reconciliation, 12-item logo specification, inventory and native/prototype differences were documented. Product implementation had not yet occurred. See section 24 for the later UI implementation; native configuration, harness and telemetry integration remain deferred.


## 24. UI-focused completion, brand integration and new verification

### Scope confirmed by the user

**5 October 2026, later implementation pass.** The user clarified: “We're mostly doing UI/experience and not full hookup yet. Current model is just for testing.” This pass therefore implements and verifies the React experience and the small design-host contracts its controls depend on. A Windows/Linux native Strata server is **not a completion gate for this UI pass**.

The native repository was not changed. No native adapter, production configuration conversion, native hardware telemetry or real model installation was added. Browser testing used an isolated temporary `STRATA_HOME` and workspace on `127.0.0.1:8794`, with `STRATA_REAL_ENGINE=0`. Model loading/downloads and answers were visibly **Demo**. API fixtures used a local disposable server; they did not call a model provider.

The count remains **132 tracked requirements**, including 120 earlier findings and 12 brand additions. The concrete failures below are follow-ups to existing requirements, not a new count of unresolved defects. “Implemented” means the specified code exists in this test-host scope; it does not certify unperformed platform tests.

### Further findings corrected during implementation

| Existing requirement | Evidence / problem | Delivered change | Verification |
|---|---|---|---|
| CMD-01–03 / CMD | Palette and chat slash actions had separate lists despite the shared shortcut registry. | `src/lib/commands.ts` supplies action labels, aliases, availability, bindings and handlers to the palette, slash menu and shortcut help. Installed-model actions use the existing load guard. Unsupported/irrelevant actions are omitted; shortcuts report unavailable actions. | Opened slash commands, exported through `/export`, switched theme through the palette, opened shortcut help from the palette, and navigated from the compact More menu. Help lists Chat as the first destination. Export delivery limitation remains below. |
| SETUP-05–07 / BOOT | Restarting an already completed setup retained `complete: true`; a page reload then restarted the wizard. Separately, the setup overlay could mount before persisted setup arrived and keep its initial step-one state. This was reproduced in the browser, not inferred from an API-only test. | Saving an in-progress step clears completion. The overlay hydrates saved step/choices once when host data arrives. “Finish later” retains the incomplete step; advancing focuses the new step heading. Persistence/finish failures remain visible beside the wizard. | Completed → restarted → Basics → reload → Basics; Finish later → `/setup` → Basics; advanced to Load/Ready using the keyboard. Automated host test also covers restarting a completed wizard. |
| SERVE-03/07 / OPS | Cancel previously changed a log row without aborting inference. A browser test then found that a slower cancel HTTP response could replace a newer completion event and leave “Cancelling…” on screen. | Abort controllers reach active execution and remove queued tasks before dispatch. A serial queue releases the next request on completion or cancellation. Serving state has an ordered revision, preventing stale mutation responses from replacing newer events. The UI shows **Cancelled**, recorded duration and the cancellation explanation; focus moves to the result when its action disappears. | Contract tests cancel active and queued requests; a queued cancellation leaves the active task running. Browser opened a queued request by ID, cancelled it and observed the same sheet reach its final state. A focused regression reproduces the out-of-order response problem. |
| SERVE-05 | Typed-key reveal and Remove were missing. | An initially masked draft has a **Show typed key** control. Remove has a concrete explanation and confirmation; removal is unavailable on a network-bound host and the server enforces that rule. Successful state is returned by the host. | Browser opened an empty editor and checked password → text input behavior; it did not enter or change a human credential. Disposable contract test verified unauthenticated 401, authenticated 200, and key removal on loopback. Native credential handling remains deferred. |
| CTX-02 / MON-04 / DATA | Streaming chunks were treated as output tokens when the server did not report usage. Cache reuse was estimated from the preceding prompt. Older conversations could inherit unrelated global context. | Usage comes from the model server’s reported counts. Missing counts remain unavailable; demo counts remain labelled. Reasoning output can establish first-token latency. Cache reuse is counted only when reported. Messages retain request/model/usage metadata; the inspector shows the selected conversation’s last recorded request size and distinguishes it from live cache occupancy. | Fixture splits an SSE line, ends a usage frame without a newline, and emits several words in one chunk: result is reported usage, not chunk count. A second fixture omits usage and returns null counts/rate. Browser verified demo provenance in Chat, the inspector and Serving. |
| SKILL-01–04 / CAP | Skill use was counted without passing the instructions to inference. A disabled workspace-scoped skill could still match. | Selection requires enabled state, matching workspace scope, matching trigger text and a nonempty body. Selected instructions are included with the current test adapter’s system prompt; client system instructions are preserved. Demo output does not claim skill use. UI copy explains inclusion versus execution/quality. | Contract checks global/scoped/disabled selection and transmitted system instructions. Browser inspected readable instructions, raw disclosure and changed Code review scope to the disposable workspace. Real local-model behavior and native harness tools are not asserted. |
| MODEL-03 | “Runs well” implied a performance guarantee from RAM-class matching. | Fit labels explicitly say **estimate**. Details explain that matching uses total installed memory, not measured free headroom or a measured runtime footprint. | Opened running/available variants and toggled the larger-variant comparison both ways. Native capacity prediction is still deferred. |
| MON-07 | Snapshot data was frozen, but some labels/readiness still depended on live library/hardware values. | Capture includes the model label, demo/source mode, loaded state and shared-memory flag as well as metric values and time. | Light/Dark snapshots at 1440×900, with a 900px snapshot scroll height, one heading, source labels and no local paths/request content. Escape returns to the remounted Snapshot button. External backend changes during Snapshot were source-reviewed; that concurrent mutation scenario was not exercised. |
| UI-11–12 / CMD | Opening Shortcuts from the command palette left its visible dialog inert and inaccessible: the closing palette restored an obsolete sibling `inert` value over React’s newly opened dialog. | The shared focus manager now recomputes background inertness from the active surface stack, respects current `data-open` state, and restores focus only to a reachable target. | Reproduced the blank accessibility tree; after the fix, Search shortcuts receives focus, Shift+Tab remains inside, Escape restores Snapshot view, and palette → setup works with the step heading focused. |
| UI-03/04/11 | The chat title used `role="button"` on an `h1`, removing heading semantics; its edit field had no explicit name. Setup step changes left focus on a removed button. | The title is an `h1` containing a normal rename button; the edit field is named Conversation title. Setup focuses its active step heading, and cancellation retains a reachable result. | Browser accessibility tree reports the chat’s level-one heading and nested button, named controls and the focused Basics/Load/Ready headings. This is an accessibility-tree/keyboard check, not a full screen-reader certification. |

These changes extend the existing first-pass fixes for branching/retry, configuration drafts, sampling reset, setup readiness, file previews, settings persistence, responsive inspectors, unavailable data, readable Markdown and shared dialogs. Their original evidence remains in section 21.

### Brand implementation disposition

| Requirement | Current status | Delivered behavior and remaining boundary |
|---|---|---|
| BRAND-01 | Implemented | `scripts/prepare-brand-assets.py` prepares 15 selected runtime assets. `public/brand/source-manifest.json` records sources, hashes, output hashes and conversion methods. All 37 supplied masters still match the original inventory. Anron controls and UI color tokens are retained. |
| BRAND-02 | Implemented; browser-chrome matrix partial | Repaired favicon has explicit light fills and its own dark color-scheme rule. Six 16/32/48px raster fallbacks are linked with theme media queries. Standalone SVG and all raster files visibly render in the asset review. Actual fallback selection across Safari/Chrome/platform bookmark chrome was not tested. |
| BRAND-03 | Implemented | Both lockups use outlined GT Walsheim Pro Bold glyphs with tracking/kerning; no live SVG text, remote font or system-font substitution. Original bar geometry and light/dark fills are preserved. Compared both exports beside supplied PNGs in the browser; saved the comparison. This is a visual comparison, not a pixel-identity certification. |
| BRAND-04 | Implemented | Shared `Brand` component replaces the old green tile/typed lockup in expanded, compact and mobile navigation. Old brand-only rules are removed. Brand navigation retains the existing Chat destination. |
| BRAND-05 | Implemented; OS-change check partial | In-app marks resolve Light/Dark/System preferences consistently with the app. The saved appearance and theme-color are applied before initial paint. Favicon appearance remains independent of manual app theme. Manual Light/Dark were exercised; changing the operating system’s appearance while System is selected remains unverified. |
| BRAND-06 | Implemented; true browser zoom unverified | Intrinsic aspect ratio and reserved dimensions; 28px expanded lockup, 32px rail mark and 24px mobile lockup. Checked 1512/1024/768/390px and 125% interface scale. At 390px with enlarged controls there was no document overflow and artwork retained its dimensions. Interface scaling is **not** a substitute for the remaining true 200% browser-zoom check. |
| BRAND-07 | Implemented; assistive/platform checks partial | One accessible name for meaningful artwork or the enclosing link; decorative artwork is hidden. Static bars, no engine-state recoloring, visible text fallback on image failure and forced-color fallback. AX names and keyboard navigation checked. Deliberate image failure, actual forced-colors mode and a screen-reader session were not exercised. |
| BRAND-08 | Implemented | Head links, touch icon, manifest, `.webmanifest` MIME support and real 404s for missing assets. All 15 runtime files matched prepared hashes when fetched from the built host; missing SVG/PNG requests returned 404 and an app route returned HTML. No remote asset dependency. |
| BRAND-09 | Web metadata implemented; installed-launch check deferred | Stable root-local identity `/`, start URL `/#/chat`, scope `/`, standalone display and explicit static dark launch colors. Theme-color follows the resolved app preference after load. Icons remain `purpose: any`; no maskable/offline/PWA-completeness claim. An installed shortcut/standalone browser launch was not tested. |
| BRAND-10 | Implemented | Setup heading mark, About/appearance preview lockups and a compact Monitor Snapshot lockup. Branding is absent from repeated message/card/tool surfaces. All five setup stages were traversed; narrow Ready and both Snapshot themes were captured. |
| BRAND-11 | Conditional / outside current scope | No desktop wrapper or target OS icon package was selected. `.icns`, `.ico`, Icon Composer, dock/taskbar/store and installed-platform masking checks remain a later packaging task. |
| BRAND-12 | Completed for the current web evidence scope | New screenshots, source hash audit, asset URL/type checks and build/audit results are saved separately from the older 120-finding evidence. Unsupported browser/native/accessibility cases remain listed rather than marked passed. |

**Preparation:** run `python3 scripts/prepare-brand-assets.py` from the app when intentionally regenerating derivatives. It needs fontTools at preparation time; exported artwork adds no Python/font-processing dependency to the running app. Original C2PA metadata is retained in the masters; derived SVGs record their source without pretending to preserve original provenance signatures.

### Verification performed in this pass

| Check | Result / scope |
|---|---|
| TypeScript | `npm run typecheck` passed on the final UI source. |
| UI source audit | `npm run audit` passed: kit synchronized, tokens resolved, five motion tokens, 11 approved material surfaces, icon/class references valid, no native selects. |
| Production build | `npm run build` passed: 61 modules; JS **380.88 kB / 116.36 kB gzip**; CSS **83.43 kB / 15.55 kB gzip**; HTML **2.17 kB / 0.78 kB gzip**. These are bundle sizes, not RAM, GPU or responsiveness measurements. |
| Regressions | `node --test tests/*.test.mjs`: **8 tests passed**. Covers incomplete streaming Markdown, setup/config/lifecycle truth, serial-queue cancellation, enabled/scoped skill selection, reported stream usage with split/final frames, unavailable usage, active/queued cancellation plus disposable auth, and stale cancel-response ordering. |
| Browser interaction | Clicked/navigated Chat, Models, Beginner/Expert/Raw tuning, Monitor/Snapshot, Serving/details/guide, Workspace/preview, Skills/details/scope, Settings and all setup steps. Exercised palette/slash/shortcut help, command Enter, tab ArrowRight, menu focus, sheet Escape/return, raw draft validate/save and setup resume. The final keyboard pass exposed a visible-but-inert dialog when opening Shortcuts from the palette. That shared overlay regression was corrected and the transition, focus entry, Shift+Tab trap, Escape return and palette-to-setup navigation were retested. DOM visibility alone was not treated as an accessibility pass. |
| Responsive and preferences | Checked 1512, 1024, 768 and 390px layouts; Monitor Snapshot at 1440×900; manual Light/Dark; 125% interface controls, reduced motion, opaque surfaces and increased contrast. Mobile About had document width 390px for a 390px viewport and loaded 24/28px lockups. |
| Rendered contrast sample | Reused the repository’s **read-only probe body** through the supported browser API on completed Chat. Ten sampled text pairings passed in each theme: Light minimum **4.81:1**, Dark minimum **5.73:1**. This does not cover all focus/hover/disabled/control-boundary states or every screen. |
| Dark surface sample | Repository luminance probe body on completed Chat found **no large bright surfaces above 0.15 luminance and no nested materials**. This is one rendered screen, not a measured compositor/resource benchmark. |
| Reading measure sample | At 1512px and 16px reading size, the existing `56ch` recipe rendered a 546.55px paragraph width. That recipe targets the established approximate 68-character reading measure. The complete text-size/focus matrix was not rerun. |
| Console | No warning/error entries in the inspected final preview log. |
| Assets | 37 master hashes unchanged; 15 runtime assets returned expected bytes/types. Missing SVG and PNG return 404; application-route fallback remains 200 HTML. Both outlined lockups and small raster icons were visually inspected. |
| Export | `/export` invoked the existing Markdown download path and cleared the command. The in-app browser again did not report a completed download event within its wait. File delivery remains unverified; no success notification claims it succeeded. |

The full `contrast`, `measure`, `luminance`, `flow`, `flows`, `motion` and `motion:budget` launch scripts were not run as external browser drivers. Supported browser interactions and the explicitly noted read-only probes provide the evidence above. Native engine tests, frame-rate/resource benchmarks and assistive-technology acceptance are not implied by green build/tests.

### Evidence index

New evidence lives in [`docs/evidence/ui-completion-2026-10-05/`](evidence/ui-completion-2026-10-05/). Earlier audit screenshots and the original brand inventory remain intact.

| File | Evidence |
|---|---|
| [01-desktop-chat-light.png](evidence/ui-completion-2026-10-05/01-desktop-chat-light.png) | Expanded shell and quiet Chat, supplied Light lockup. |
| [02-chat-context-dark.png](evidence/ui-completion-2026-10-05/02-chat-context-dark.png) | Dark identity and conversation-scoped demo statistics. |
| [03-monitor-snapshot-dark-1440.png](evidence/ui-completion-2026-10-05/03-monitor-snapshot-dark-1440.png), [04-monitor-snapshot-light-1440.png](evidence/ui-completion-2026-10-05/04-monitor-snapshot-light-1440.png) | Both snapshot themes at the target size. |
| [05-compact-monitor-1024.png](evidence/ui-completion-2026-10-05/05-compact-monitor-1024.png) | Compact mark and Monitor layout. |
| [06-raw-config-1024.png](evidence/ui-completion-2026-10-05/06-raw-config-1024.png) | Validated/saved shared raw draft. |
| [07-workspace-sheet-768.png](evidence/ui-completion-2026-10-05/07-workspace-sheet-768.png) | File preview as a sheet, with a disposable local file. |
| [08-mobile-chat-390.png](evidence/ui-completion-2026-10-05/08-mobile-chat-390.png) | Mobile lockup and actual file text in an unsent draft. |
| [09-mobile-about-dark-large.png](evidence/ui-completion-2026-10-05/09-mobile-about-dark-large.png) | Dark About, enlarged interface, opaque surfaces and increased contrast. |
| [10-serving-cancelled-mobile.png](evidence/ui-completion-2026-10-05/10-serving-cancelled-mobile.png) | Same request sheet reaches Cancelled with unavailable token usage retained. |
| [11-logo-reference-comparison.png](evidence/ui-completion-2026-10-05/11-logo-reference-comparison.png) | Supplied PNGs beside outlined Light/Dark lockups; 16/32/48px raster icons and standalone favicon. The temporary comparison page is not shipped. |
| [12-setup-ready-mobile.png](evidence/ui-completion-2026-10-05/12-setup-ready-mobile.png) | Narrow setup steps, decorative mark, explicit demo readiness and Open chat. |
| [13-final-monitor-snapshot-dark.png](evidence/ui-completion-2026-10-05/13-final-monitor-snapshot-dark.png) | Final-build snapshot, frozen values/time/source, no paths or chat content. |
| [asset-verification.json](evidence/ui-completion-2026-10-05/asset-verification.json) | Individual source/hash/URL/type checks and manifest identity. |
| [rendered-probes.json](evidence/ui-completion-2026-10-05/rendered-probes.json), [reading-measure.json](evidence/ui-completion-2026-10-05/reading-measure.json) | Raw sampled contrast, luminance/material and reading-width results with their scopes. |

### Remaining boundaries

1. **Deferred by scope:** actual native Strata configuration parity, Windows/Linux runtime lifecycle, native tool/image grants, production privacy/logging behavior, actual model capacity/headroom and precise native sensors. Existing unavailable/preserved/demo distinctions remain visible. The current test adapter is not presented as production Strata.
2. **Platform verification still needed:** normal-browser export delivery, installed manifest launch, browser favicon selection/fallbacks, OS appearance changes, forced-colors mode, deliberate image failure, true 200% browser zoom and native icon packaging after a shell is chosen.
3. **Quality verification still needed:** full screen-reader coverage, every interactive contrast state, the full reading-size/focus matrix and representative hardware measurements of streaming, scrolling, blur and memory. Current visual checks are not an accessibility or performance certification.
4. **Optional refinement:** a single coordinated menu system for all existing disclosure overflows. Current sidebar menus have keyboard handling; native disclosures remain usable and labelled.

**Completion of this scoped pass:** the current UI, its remaining command/request/setup fixes and the supplied web identity are implemented with saved evidence. Native hookup remains intentionally deferred under the user’s clarification. No real model was downloaded or run, no human credential was changed, and no supplied logo master was altered.

## 25. Engine Models and Onboarding redesign implemented

The subsequent approved redesign is implemented: shared context/batch dials and performance feedback, three recommendation profiles, a grouped quantization library with comparison, and persistent five-step onboarding. See the [implementation and verification report](engine-models-onboarding-2026-10-05.md) for the delivered behavior, 14 passing tests, final build results, browser evidence and remaining production-engine boundaries. This supersedes earlier layout proposals for these three screens; the existing findings and historical evidence remain intact.

## 26. Persona review of text and information density

Eight GPT-6.1 Sol Medium subagents completed independent reviews, cross-review debate and a 20-item ballot. The [decision report](persona-review-2026-10-05/decision-report.md) records 160 votes, 18 majority decisions, 14 unanimous decisions and two unresolved choices, with every original review, debate and vote linked. These are simulated persona judgments, not a human usability study. The majority recommends context-first Beginner tuning, one card per unique recommended setup, aligned comparison rows and optional tuning within onboarding. This follow-up changes documentation only; its proposed simplifications are not yet implemented.
