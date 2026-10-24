# Design-system notes

How the app maps onto the Strata UI kit (`../ui`), plus the places where the kit
and the app still disagree. Measurements here are computed, not eyeballed —
contrast ratios use the WCAG 2.1 relative-luminance formula, with translucent
tints composited over `--st-surface` first.

## 1. How the app consumes the kit

`src/styles/kit/{tokens,components}.css` are synced verbatim from `../ui` by
`npm run sync:kit` (the only rewrite is making the `@font-face` URLs absolute).
`src/styles/app.css` sits on top and contains, in order: extensions to the kit's
classes, app shell layout, utilities, app-only components, and the first-run
takeover. There are no gradients in it and no literal colours — every value is
either a kit token or a layout constant.

```
npm run sync:kit     # re-import tokens/components/fonts/icons from ../ui
```

Every component the kit defines is in use:

| Kit class | Where the app uses it |
| --- | --- |
| `.st-header` | screen headers (`Screen`, Chat) |
| `.st-card` / `__head` / `__body` | every panel — 55 call sites |
| `.st-btn` (+ `--primary/--secondary/--danger/--ghost/--icon`) | all actions |
| `.st-pill` / `.st-badge` | status lines vs. table cells and tool scopes |
| `.st-progress` | context, VRAM, RAM, temperature, downloads |
| `.st-metric` / `__label` / `__value` / `__unit` / `__spark` | every number tile |
| `.st-tabs` / `.st-tab` | Beginner/Expert, All parameters/Raw config |
| `.st-table` | request log, expert parameter table |
| `.st-input` / `.st-select` / `.st-toggle` / `.st-slider` / `.st-field` / `.st-label` | forms |
| `.st-collapse` | model thinking |
| `.st-code` | fenced code blocks |
| `.st-gauge` / `__track` / `__fill` | GPU load, context fill |
| `.st-drawer` | sampling, request detail |
| `.st-composer` | chat composer |
| `.st-msg` / `.st-bubble` | conversation |
| `.st-toast` / `.st-toasts` | notifications |
| `.st-icon` | all 32 Anron duotone icons |

Two things the app adds that the kit does not define, both built only from kit
tokens: `.st-btn--sm` / `--lg` / `--block` / `--active` / `--ghost` (the kit ships
one button size) and `.st-progress--sm` / `--lg` / `--indeterminate`.

## 2. Findings in the kit itself

### 2.1 Light-mode primary colour fails contrast — `blocker` — **corrected in the app layer**

`--st-accent-ink` on `--st-accent` is the pairing behind **every primary button
and the user's own chat bubble**.

| Theme | Pair | Idle | Hover | AA (4.5:1) |
| --- | --- | --- | --- | --- |
| light, kit as shipped | `#ffffff` on `#10b981` | **2.54** | 2.97 | fails |
| dark, kit as shipped | `#052e20` on `#34d399` | 7.70 | 8.50 | passes |

Dark mode is fine. Light mode — the theme the kit treats as default — is not, and
the label is 14px/700, so the 4.5:1 threshold applies rather than the large-text 3:1.

**What the app does now.** `src/styles/app.css` overrides two tokens in light mode
and nothing else, adopting the relationship the kit's own dark palette already
uses (dark ink on the accent fill):

```css
:root:not([data-theme='dark']) {
  --st-accent-ink: #05261b;      /* idle  6.37:1 */
  --st-accent-hover: #12c68e;    /* hover 7.31:1 */
}
```

The hover had to move too: with a dark ink, darkening the fill on hover drops the
ratio to 3.71:1, so the fill lifts instead. Blast radius is three surfaces
(primary buttons, user bubble, brand mark) plus one hover rule; `--st-accent`
itself is untouched, so progress bars, gauges, toggles, sparklines and pills look
exactly as before. Delete that block to return to the kit verbatim.

**The alternative — option B.** Deepen the accent and keep the white label:

```css
:root:not([data-theme='dark']) {
  --st-accent: #0a7a55;          /* idle  5.35:1 */
  --st-accent-hover: #086a4a;    /* hover 6.62:1 */
}
```

Both were rendered on real screens for comparison —
`strata/.shots/cmp-accent-options.png`. A keeps the bright green that the rest of
the app is built on and changes the label; B keeps white labels and deepens every
green surface in the app. A is what is currently in place, because it changes the
fewest things; B is two lines away.

### 2.2 `--st-accent-text` on `--st-accent-tint` is marginally under AA — `minor`

`#0c8a60` on `#ecfdf5` = **4.14:1**, used by accent pills and badges at 12–13px.
Every other tint pairing clears AA comfortably:

| Pairing | light | dark |
| --- | --- | --- |
| info-text on info-tint | 4.99 | 7.74 |
| warn-text on warn-tint | 5.51 | 8.87 |
| danger-text on danger-tint | 6.07 | 6.70 |
| accent-text on accent-tint | **4.14** | 8.02 |
| ink-muted on surface | 5.02 | 5.90 |
| ink-muted on surface-2 | 4.69 | 5.41 |

Lowering `--st-accent-text` from `#0c8a60` to about `#0a7a55` clears 4.5:1.

### 2.3 The icon set has no file or folder glyph — `minor`

The 32 Anron icons cover hardware, engine, chat and status, but nothing for a
document or a directory. The workspace tree currently gives folders the `layers`
icon and leaves files without one, which reads acceptably but is a workaround.
Two additions to `ui/icons/` (`file.svg`, `folder.svg`, same 24-grid / 1.5px
round / 12% duotone recipe) would let the tree say what it means.

## 3. DSH surfaces and where they live

The kit supplies the surface; DSH supplies the shape of the product:

| DSH behaviour | In this app |
| --- | --- |
| sessions in the sidebar | sidebar → Recent, with rename and delete |
| status line for the engine | sidebar engine chip + header pill |
| ⌘K command surface | `CommandPalette` (net-new; DSH has only `/` and `Mod+/`) |
| slash commands | composer `/` menu |
| tool approval takeover | `ToolCard`, Enter approves / Esc denies |
| per-conversation sampling | `SamplingDrawer`, bound to engine expert params |
| right-hand rail | `ContextPanel` — context window, tools in reach, engine |
| file browsing | `Workspace`, a tree with a detail pane |
| keyboard shortcuts dialog | `Shortcuts`, every listed binding wired |

## 3b. Motion, fluidity and focus

The interaction layer added on top of the kit — see
[`interaction-guidelines.md`](interaction-guidelines.md) for the rules themselves.

- **One motion scale**, fusing the kit's single pair (`--st-ease` / `--st-dur`,
  left untouched so the kit's own classes keep their timing) with DSH's .1/.2/.3s
  vocabulary: five duration tokens and five easings. Twenty-three bespoke
  keyframes collapsed onto two shared recipes (`st-enter`, `st-exit`) whose
  direction comes from custom properties.
- **Reduced motion substitutes opacity for movement** instead of the previous
  blanket `0.01ms`, which had also removed state feedback. The distance
  properties are zeroed on the elements themselves — a value set on `:root` is
  only inherited, so a rule declaring its own `--st-from-x` would have won over
  it. There is a manual override in Settings for people who want stillness
  without changing an OS setting.
- **Interruptible overlays**: drawers, palettes and dialogs stay mounted and are
  driven by `[data-open]` transitions, so closing mid-open reverses rather than
  restarting. `visibility` flips discretely, which keeps a closed overlay out of
  the tab order.
- **Capabilities verified before use, not assumed**: `corner-shape:
  superellipse()`, `interpolate-size`, `startViewTransition`, `scroll-timeline`,
  `timeline-scope`, `@starting-style`, `::details-content` and
  `transition-behavior: allow-discrete` were each confirmed present in the app's
  own browser. Every one is behind `@supports` or a feature check, so the layout
  is identical without it.
- **Focus mode** (`⌘⇧F`) withdraws the chrome to an ambient bar, dims earlier
  turns, and narrows the column. Escape unwinds one layer at a time.

## 3c. Materials and the dark accent

- **Materials**: three blur tiers over six surfaces, each token-gated so reduced
  transparency and the Settings override both collapse them to flat with
  identical geometry (verified). Specular 1px inset edges, not gradients.
- **The dark accent is a material now, not a fill.** A solid `#34d399` button was
  luminance 0.496 against a page at 0.005 — a 9.9:1 jump, making the brightest
  object in the UI the one you are invited to click. Large accent surfaces are
  now a 26% tint (luminance 0.055, 1.90:1 against the page) with an accent label
  at 5.73:1. Thin strokes keep the bright accent. `npm run luminance` guards it.
- **Elevation is rationed**: shadows only on surfaces that genuinely overlap
  content. Cards keep the hairline alone; a card on a flat background does not
  float. Audit-enforced with an allowlist.
- **Separators align with the content** rather than running edge to edge.
- **The header lifts by fading a hairline in**, not by animating `box-shadow`,
  which repainted the strip on every scroll frame. The subtitle retires past 64px
  of scroll; both ride the existing scroll timeline.

## 3d. Liquid Glass geometry

- **Capsules for controls, the kit's scale for surfaces.** Seven control classes
  take `--st-r-pill`; cards, drawers, modals and toasts keep 8/14/20/24. The kit
  itself is untouched and still sync-checked, so this is an app-layer override —
  all departures now live in one `6. system overrides` section at the end of
  `app.css`, because an override of an app-layer class has to come after it.
- **Concentric corners** via `--st-r-nested` (outer − gap, floored at zero).
- **Press compresses** (`scale(0.97)`), and the header title collapses on scroll
  by cross-fading two stacked copies rather than animating `font-size`, which
  would reflow every frame.
- **Edge fade** under the header is a `mask-image`, not a gradient background.
- **Glass-on-glass was a real defect and is fixed.** `.slashmenu` sat inside the
  material on `.composer`, so it sampled an already-blurred backdrop. It now
  renders as a sibling in `.slashmenu-host`, which repeats the composer's padding
  so the menu keeps its column. Verified: no material has a material ancestor,
  and `npm run luminance` fails if one appears.
- **Lensing was prototyped, measured and declined** — it executes fine and costs
  1.03× at p95, but changes **zero pixels** over the composer because the backdrop
  is uniform. Full reasoning and numbers in
  [`interaction-guidelines.md` §10](interaction-guidelines.md).

## 3e. The focused shell

- **The primary button is a material in both themes now.** Light was a solid
  `#10b981` carrying a colour-matched green glow — paint, not glass. It is a 20 %
  tint with an accent hairline, a specular top edge and a *neutral* shadow;
  `#086a4a` on that fill measures **5.48:1**. The glow is deleted in both themes.
- **The toggle was missed by that same fix** — a pure-white knob with a hard
  shadow sitting on the loud solid accent. It now shares `--st-accent-surface`
  with the buttons, carries a hairline, and presses with a knob scale.
- **The accent border was bound to the white specular token**, which is an
  invisible border on a near-white page. Border and specular highlight are now
  separate tokens. This is why `npm run contrast` asserts *definition* rather
  than fill-vs-page in light mode: a tinted control on white gets its edge from
  the hairline, and the check composites that border (79/255 against the page)
  instead of comparing a raw declared value, which would let a 2 %-alpha border
  pass.
- **Eight icons authored** in the kit's duotone recipe — a `0.12`-opacity
  `currentColor` fill plus a `1.5`-stroke path — because the kit had none of:
  panel, search, sliders, folder, folder-plus, file, ellipsis, x. Sprite is now
  40 symbols, and the audit fails if `<Icon name>` references a missing one.
- **Navigation restructured** to the harness shape: collapse row under the logo,
  New session, Skills, a Workspaces tree with search / collapse-all / add-folder,
  and a footer of Monitor, Serving and **… More** (Models, Engine tuning,
  Settings). Home is gone; Chat is the landing screen.
- **Workspaces are real, not decorative.** The host gained a persisted registry
  (`data/workspaces.json`), `GET/POST/DELETE /api/workspaces`, `?root=` on the
  workspace listing, and `POST /api/workspaces/pick` which borrows the *system*
  folder chooser via `osascript`. The seed root cannot be removed and a stale id
  falls back to it, so a deleted folder cannot break a saved session.
- **First run is a modal over the live shell.** The wizard was a
  `position: fixed; inset: 0` takeover, which contributes no height when nested —
  it had to become static inside the surface before the modal would render. Its
  stacking context also meant the scrim's `z-index: 60` covered a surface at
  `z-index: 1`; the surface now outranks it.
- **Session telemetry** under the composer: turns, steps, tok/s, tokens and
  memory used, in tabular figures so the numbers do not jitter.

## 4. Verification

The evidence for this document is the running app, not the source:

- `strata/.shots/design-system-index.png` — every screen in both themes, side by
  side, captured at 1512×1110. Light is the default. (Split for reading as
  `index-1.png` / `index-2.png`.)
- Overlay surfaces: `120-palette`, `121-palette-filtered`, `122-shortcuts`,
  `100-context-panel`, `96-sampling-changed`, `80-slash-menu`, `70-tool-approval`.
- `npm run typecheck` and `npm run build` both clean.
- Every tour run reports **no console errors**.

Two checks are scripted rather than eyeballed, so they can be re-run after any
change: an off-scale audit over `app.css` (radii, type sizes, shadows, literal
colours, gradients) and a contrast probe that composites translucent tints and
measures each interactive state separately. Both are described in §2 and §4.

## 5. Detail pass

A zoomed review of every screen (2× crops, not whole-page glances) turned up four
things that only show at real size:

| Symptom | Cause | Fix |
| --- | --- | --- |
| Metric rows lost their rhythm — one card's value wrapped to two lines and pushed its caption out of line with its neighbours | the kit's `.st-metric__value` is the 32px *stat* size, used for names like "Apple M4 (8-core GPU)" | `Metric compact` renders text values at `--st-fs-lg`, so names stay on one line and captions align |
| The composer's context meter was invisible — just a hairline on white | `.st-progress` uses `--st-surface-2` for its track, which disappears on the white composer | the composer swaps the track to `--st-line`, the token meant for visible rules |
| The engine chip repeated itself: "Loaded and ready · engine idle" | title and subtitle both described load state | title carries the model, subtitle carries readiness — the engine process itself belongs on Monitor |
| Two values sat off the type scale (12.5px) | hand-written sizes | moved to the kit's 13px mono step |
| Modal search fields drew a focus ring around an invisible box | the global `:focus-visible` outline, on a borderless autofocused input | `.st-input--bare`: the field loses its border and marks focus with a rule underneath |
| "Keyboard shortcuts" was listed twice — once for `⌘/`, once for `?` | the shortcut model assumed one binding per action | a shortcut now carries a list of *bindings*, rendered in one row as `⌘ /  or  ?` |

Also fixed earlier in the same pass: the setup wizard showed two competing primary
CTAs on the basics step ("Continue" and "Download & install" both advanced), now a
single action per step.


## Settings panels and API models — 10 October 2026

Settings now has four independent panels: Appearance (General and Reading & focus), Models, Storage, and About (Privacy and About). The settings search is removed. Arrow keys and Home/End move between tabs; an unfinished model connection stays intact when switching panels. The dialog still opens over the current page.

Models supports OpenAI, Anthropic, DeepSeek and OpenRouter presets, plus custom servers over HTTP/HTTPS. Connections can use OpenAI Chat Completions, OpenAI Responses or Anthropic Messages. Fetch available models retrieves the API catalog; users choose which models appear in chat, or add IDs and display names manually. Custom servers can be addressed by another computer’s network address. The API base URL includes the server’s API prefix, without a completion endpoint suffix.

Saved connections appear in the chat picker alongside local versions. Selecting an API model routes text conversations to that connection without requiring a local model to be loaded. Conversation history, text attachments already included in the conversation, and enabled skill instructions are sent to the selected API. API models use provider defaults for sampling and reasoning. Local Engine tuning and local tool execution remain separate. The app reports API token usage only when supplied by the provider and does not substitute local engine context limits for remote models.

Credentials are written atomically to `model-providers.json` in the host data directory with owner-only file permissions (0600). This is a local credentials file, not encrypted keychain storage. The browser receives only `apiKeySet`; editing a connection never reveals the saved key. An unchanged blank key retains the current key, while changing the API destination requires re-entering it. Redirects are rejected. Raw provider error responses are excluded from UI errors to avoid exposing credentials or request content.

Protocol references: [OpenAI streaming](https://developers.openai.com/api/docs/guides/streaming-responses), [Chat Completions events](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events), [Anthropic streaming](https://platform.claude.com/docs/en/build-with-claude/streaming), [DeepSeek API](https://api-docs.deepseek.com/), and [OpenRouter API](https://openrouter.ai/docs/quickstart).

Verification uses an isolated local mock API and host, with fake keys and test conversations. It covers discovery, pagination, saved key retention/redaction, all three stream formats, history, cancellation and incomplete responses. UI checks cover adding a provider, manual custom models, picker integration, text chat without a loaded local model, keyboard tabs, light/dark themes and phone widths. Screenshots are under `docs/evidence/settings-models-2026-10-10`.


## Downloaded model picker and compact model header — 10 October 2026

The chat picker groups downloaded local versions by model family. A family row shows its running version, or its smallest downloaded version when idle; the adjacent arrow reveals other downloaded versions. Available catalog entries and unfinished downloads stay in the Model library. Installed files remain visible through loading, unloading and recoverable load errors. Existing switch confirmations and engine lifecycle checks still apply. Saved API connections appear in the same picker under their provider names. Expanding a group scrolls within the available viewport height.

The Model hero uses the authored family icon without its surrounding tile. Its title and current quantization/file size share a line where space permits, with responsive wrapping on phones. The redundant family eyebrow and running-status row are removed. Opening the family selector with a pointer leaves its border transparent; keyboard focus remains visible.

Verification: 10 focused tests passed, plus TypeScript and production build. Manual UI checks covered actual downloaded Qwen versions, keyboard expansion and cancelled switch confirmation, multiple model families and API selection using a read-only fixture, pointer versus keyboard focus, and 320-pixel layouts. Screenshots are under `docs/evidence/model-picker-2026-10-10`; files named `fixture` use disposable model/provider data.
