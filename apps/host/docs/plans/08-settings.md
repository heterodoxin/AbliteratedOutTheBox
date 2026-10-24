# 08 · Settings

**Route** `#/settings` · **File** `src/screens/Settings.tsx` (333 lines)

The catch-all. It carries the most `Field`s (37), the most `Select`s (6) and seven
`SectionHead`s, which makes it the densest screen by control count and the one
most likely to become a junk drawer.

## What it is today

Sections, in order: **General** · **Reading & focus** · **Engine** · **Storage** ·
**Privacy** · **About**.

- *General* — appearance, on-startup, updates, history.
- *Reading & focus* — text size slider, density, motion, translucency, focus mode.
- *Engine* — defaults and a link to tuning.
- *Storage* — where models and data live.
- *Privacy* — the local-first statement.
- *About* — the machine summary inherited from the retired Home screen.

## Where it falls short

1. ~~No navigation within the screen.~~ **Fixed.** A sticky index of all six
   sections sits under the header. Verified: jumping to *About* scrolls the
   section to 74 px and the index marks it current; all six ids resolve.
2. **Sections are ordered by history, not by use.** General first is convention;
   Reading & focus is the one people actually return to, and Privacy is read
   once, ever.
3. **Privacy is a statement, not a setting.** A whole section for prose that
   belongs as a line under About.
4. **A gap I introduced: About says "Run setup again" and nothing else.** The
   machine summary landed there last round but the section has no version, no
   build, no paths — the things an About section is for.
5. **Reading & focus holds five controls of three different kinds** — a slider, a
   select, a toggle — with no visual grouping.
6. **No reset.** Nothing restores defaults, individually or wholesale.
7. **Search does not exist** on the screen with 37 fields, which is the screen
   that most needs it.

## The plan

### Substance
1. ~~A section index in the left margin~~ (or a segmented jump row),
   highlight-on-scroll. **Shipped** — a sticky index that tracks the section you
   are reading and now carries the search field. The annotation was missing from
   this list long after the work landed, so it read as unbuilt.
2. **Reorder by use** — still open. The order is unchanged (*General* first,
   *Privacy* last); the index makes it navigable but not better ordered, and
   *Privacy* is still a whole section for prose that belongs under About.
3. **Flesh out About** — **partly done, and the rest deliberately not.**
   *Engine* joined the machine summary, naming the engine build and backend
   (`0.1.0 · metal · idle`); the version line was already there.

   The folder paths and their Open buttons are **not** duplicated here. They live
   in *Storage*, beside the actions that use them, and copying them into About
   would put one fact in two places — the failure plan 12's own rule warns
   against. Config file paths are omitted for the same reason the audience is
   described as non-technical: they are a power-user detail with no action
   attached.
4. **Group Reading & focus into two labelled blocks** — *Reading* (text size,
   density) and *Motion & materials* (motion, translucency, focus mode tell the
   same story about how the app moves).

### Fluidity
5. **Settings search** that filters across all sections, reusing the command
   palette's matching, so `translucency` finds the control without knowing which
   section it lives in.
6. **Per-section and global "Reset to defaults"**, with the modified fields
   marked so what would change is visible before it changes.

## Motion

| Interaction | Spec |
|---|---|
| Index jump | Instant scroll, no smooth-scroll animation — a settings jump should be immediate |
| Section highlight on scroll | Opacity and weight only, driven by a scroll timeline; no layout |
| Reset | Fields animate to their new values with a `--st-t-control` colour settle; no movement |
| Search filtering | Sections fade out, `--st-t-control` |

## Verification

- Assert every section is reachable **in one action** from the top of the screen.
- Assert theme and text size still survive a reload — the prefs regression that
  would hurt most.
- Extend `flow settings`: index jump, search, change text size, reset.

## About / Update — 12 October 2026

The UI now shows the app and engine versions reported by the host, a Check for
updates entry and a Roll back entry. Both open a compact component picker.
The current design host has no version installer or previous-installation
history, so install/restore actions remain disabled with a concrete explanation.
No release was checked, downloaded, installed or rolled back by this UI change.

Integration needs a host-owned version service, not renderer filesystem writes:

- Read installed app/engine builds, compatible releases, checked-at state and
  saved previous installations. An unavailable catalog is different from Up to
  date. Resolve engine builds through Strata's compatibility manifest.
- Check updates without changing the running model. Review the chosen build and
  release notes before installation; report download/progress/errors from the
  host. Wait for current requests or explicitly stop them before restarting.
- Verify artifacts, stage a replacement installation and retain the previous
  healthy build plus a compatible configuration snapshot. Activate only after
  startup/health checks succeed; recover the old build on failure.
- Rollback selects a real retained installation and validates schema compatibility.
  Keep model weights and chats. Apply app and engine version changes together
  when their compatibility requires it, and flag saved tuning for review after
  an engine change.
- The frontend needs status/check/install/rollback operations and progress events,
  with capability flags so external engines or this web preview do not advertise
  installation operations they cannot carry out.

### Subsequent review: checks stay inline

The follow-up browser comment removes the update modal. Check for updates now
calls `GET /api/updates/check` and changes the Update card in place: Checking,
Update available, Up to date, No release yet, Check unavailable or Check failed.
The endpoint reads the official Strata repository's stable releases and compares
semantic versions; drafts/prereleases are ignored. Results include `checkedAt`,
separate `strata`/`engine` states, `installedVersion`, optional `latestVersion` and
`releaseUrl`, and `canInstall: false` for this unmanaged host.

An available Strata release adds **Get update**, a link to its verified GitHub
release page. It does not install software. The real browser check exercised this
state on 12 October. Failed requests remain retryable and do not leave a stale
Up to date verdict on screen. Engine checks require a tested compatibility feed
and an actual installed build ID; the existing binary downloader alone cannot
establish that verdict. Rollback still requires retained installations.


## Model directory — 14 October 2026

Storage now shows the active model directory and provides **Change directory**
and **Check folder**. Enter the full path of an existing writable folder; the
preference survives a host restart. New downloads use that folder. Switching
keeps the previous folder’s files, rescans supported weights, and restores the
previous library when users switch back. Conversations, setup completion, API
keys, and tuning stay in the app data directory.

If models are loaded, the action reads **Unload and change directory**. Active
requests, downloads, loading, or another engine operation block switching. Normal
mode still loads one model with automatic placement; concurrency stays in Expert.

The real chat library recognizes catalog filenames in `<directory>/<variant-id>/`
or directly inside the chosen directory, with exact file size and a GGUF header.
This is catalog discovery, not a general custom-model importer or cryptographic
verification of chat weights. Existing image weights are recognized in
`<directory>/image/<quant>/` or directly inside the folder and are checked against
the pinned image SHA-256 before reuse. **Check folder** discovers files copied
in after selecting a directory. Unknown architectures and arbitrary recursive
folder layouts still need a separate import flow.

The design host scopes simulated installs by directory; it neither downloads nor
creates pretend weight files. Deleting a version removes its known GGUF file and
preserves unrelated files in the chosen folder. Image model placement and native
multi-engine execution remain pending integrations.
