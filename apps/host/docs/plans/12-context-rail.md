# 12 · Context rail

**File** `src/app/ContextPanel.tsx` (172 lines) · **Opened from** the Chat header

The right-hand rail answering "what is this turn working with?". It was rebuilt
last round into an animated grid track and tightened so it no longer clips.

## What it is today

- **A grid track** that animates `0 → 340 px` on `--st-t-surface`, with the panel
  staying mounted and `inert` while hidden.
- **Five cards** — Context window (gauge + bar), This conversation (model,
  messages, tokens, tool calls), Tools in reach (four tools with scope badges),
  Engine (speeds, uptime, links), and Hardware.
- **Content scrolls** — 1083 px of content in 959 px of rail.
- **A "Hide" button** in the header.

## Where it falls short

1. ~~The rail still scrolls.~~ **Fixed.** Sections collapse and animate on a grid
   track, and the restructure means the rail **no longer scrolls at all** with
   everything open (`scrollHeight === clientHeight` at 1110 px height). Verified
   mid-collapse: 162.9 px → 9.6 px → 0 px.
2. **"Hide" is a verb in a header**, where a close affordance is expected. It
   also competes with the transcript's own content for attention.
3. **The rail duplicates the telemetry popovers.** Context window, tokens and
   tool calls all appear in both, at different precisions.
4. ~~Tools in reach is a fixed list of four.~~ **Fixed.** Rows show how many
   times each tool has actually been used in this conversation, unused ones
   recede to a quieter weight, and the footer counts the session's calls. A list
   that never changes is documentation; this is context.
5. **Hardware appears in the rail and on Monitor**, computed independently.
6. **No section is collapsible**, so five cards must all be scrolled past to
   reach the fifth.
7. **The rail has no empty state.** With no conversation it shows zeros rather
   than saying what it would show.

## The plan

### Substance — make it about *this* conversation
1. ~~Tools in reach reflects reality.~~ **Done.** Verified: after one read, *Read
   file* shows `Used 1×` and the other three keep their catalogue descriptions at
   a muted weight.
2. ~~Remove the duplication with the telemetry popovers.~~ **Diagnosed and
   partly fixed.** The rule — *agree exactly, or appear in only one* — immediately
   found a disagreement: the rail printed the context maximum as **32K** and the
   popover as **33k**, the same number through two formatters (÷1024 and ÷1000).
   Both now use the token formatter and read **32K**. The remaining overlap
   (message counts and tool calls appear in both) is deliberate: the rail's
   *This conversation* is a narrative summary and the popover is a counter
   readout, and they now agree.
3. ~~Collapsible sections.~~ **Done.** Everything open by default — collapsing is
   for getting past a section, not for hiding information on first sight.
4. **An honest empty state** — what the rail will show once there is a
   conversation, rather than a set of zeros.

### Hierarchy
5. ~~Invert the card weight.~~ **Done.** Five bordered, elevated cards competed
   with the transcript they describe. The sections drop the card chrome entirely
   — no fill, no shadow, no border — and are separated by a hairline, with the
   panel itself keeping one surface so the rail reads as a single object rather
   than a stack of fragments. Verified: section background transparent, shadow
   `none`, panel opaque.

   **This exposed a duplication I had introduced.** Giving the sections heads (an
   earlier round) left the metric inside still carrying its own label, so the
   panel literally read *"Context window / Context window / 6%"*. Removed, and
   the flow now asserts the phrase appears exactly once.
6. **Replace "Hide" with a close icon** in the header corner, with the panel
   toggle in the Chat header remaining the primary control.

### Fluidity
7. **Sections expand and collapse on a grid track**, matching the workspace trees.
8. **The gauge and bars animate to their value on open** rather than snapping, so
   the rail reads as a live instrument rather than a static report.

## Motion

| Interaction | Spec |
|---|---|
| Rail open/close | grid track 0 ↔ 340 px, `--st-t-surface` — built, interruptible |
| Section expand | grid track `0fr → 1fr`, `--st-t-local` |
| Gauge on open | `stroke-dasharray` to value, `--st-t-choreo`, once |
| Value updating live | No transition on the number; the bar it drives eases `--st-t-local` |

## Verification

- Assert the rail content **fits without scrolling at 1110 px height** once
  sections are collapsible.
- Assert every figure shown in both the rail and a telemetry popover is
  **identical**, or appears in only one.
- Assert the rail is `inert` when hidden — already correct; keep under test.
