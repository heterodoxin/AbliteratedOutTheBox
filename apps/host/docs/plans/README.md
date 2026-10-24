# Strata · per-screen improvement plans

One plan per page and per scenario, written against the two rulebooks the app is
built on — [interaction-guidelines.md](../interaction-guidelines.md) (Strata's own
rules) and Apple's Liquid Glass HIG — and against what the code actually does
today, not what it is meant to do.

Every plan follows the same shape: **what it is today** (grounded in the file and
in measured numbers), **where it falls short**, **the plan**, **motion**, and
**how the result is verified**. Claims about the present are checkable with the
scripts in `scripts/`; claims about the future are written so they can be checked
the same way once built.

## Screens

| # | Page | Route | Plan |
|---|---|---|---|
| 01 | Chat | `#/chat` | [01-chat.md](01-chat.md) |
| 02 | Models | `#/models` | [02-models.md](02-models.md) |
| 03 | Engine tuning | `#/tuning` | [03-engine-tuning.md](03-engine-tuning.md) |
| 04 | Monitor | `#/monitor` | [04-monitor.md](04-monitor.md) |
| 05 | Serving | `#/serving` | [05-serving.md](05-serving.md) |
| 06 | Workspace | `#/workspace` | [06-workspace.md](06-workspace.md) |
| 07 | Skills | `#/skills` | [07-skills.md](07-skills.md) |
| 08 | Settings | `#/settings` | [08-settings.md](08-settings.md) |
| 09 | First run | modal | [09-first-run.md](09-first-run.md) |

## Scenarios

| # | Scenario | Plan |
|---|---|---|
| 10 | Sidebar and navigation | [10-sidebar.md](10-sidebar.md) |
| 11 | Command palette | [11-command-palette.md](11-command-palette.md) |
| 12 | Context rail | [12-context-rail.md](12-context-rail.md) |
| 13 | Overlays — drawers, menus, selects, popovers | [13-overlays.md](13-overlays.md) |
| 14 | Empty, loading and error states | [14-states.md](14-states.md) |
| 15 | Motion and fluidity | [15-motion.md](15-motion.md) |
| 16 | Accessibility and long sessions | [16-accessibility.md](16-accessibility.md) |

## The shared vocabulary these plans assume

Fixed by the design system, so plans do not re-litigate them:

- **Light is the reference theme.** Every change is designed light-first and
  verified in dark, never the reverse.
- **Capsules are for controls; the 8/14/20/24 scale is for surfaces.** Audit-enforced.
- **Glass is only for the floating control layer.** Glass never samples glass;
  `npm run luminance` fails the build if a material ends up nested in another.
- **One motion scale** — 100/160/240/320/460 ms, one standard curve, springs only
  for direct manipulation. No layout property is transitioned directly; grid
  tracks and opacity carry the movement.
- **Contrast is measured on rendered pairings**, compositing alpha, in every
  interactive state.
- **Reading measure is capped** at 68 characters in the conversation, verified at
  10, 16 and 22 px.

## Verification harness these plans refer to

```
npm run check       typecheck + audit + contrast + measure + luminance
npm run audit       off-scale values, materials, elevation, capsules, icons, native selects
npm run contrast    rendered pairings, composited, per state, both themes
npm run measure     characters per line at every content size
npm run luminance   dark-mode brightness budget + material layering
npm run motion      interruptibility, frame-sampled contact sheets
npm run flow <name> scripted screen flows; `tour` and `tour-dark` cover every route
```
