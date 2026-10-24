# 09 · First run

**Modal over the shell** · **File** `src/screens/Setup.tsx` (425 lines)

The first five minutes decide whether the app is trusted. The audience is
explicitly people who find computers intimidating, so every step has to answer
"what is happening and is it safe?" without being asked.

## What it is today

- **A five-step wizard** — Check PC · Model · Basics · Install · Ready — as a
  modal over the live shell, with direction-aware step transitions.
- **Step 1** — a hardware scan with an animated scan line and a settling checklist.
- **Step 2** — model choice with `MetricGrid` specs.
- **Step 3** — three `Select`s (context length, images, memory mode).
- **Step 4** — install progress with staged marks.
- **Step 5** — a ready state with a mark-pop animation.
- **A step rail** at the top showing 1–5 with the current step underlined.

## Where it falls short

1. ~~The step rail is a progress bar, not a map.~~ **Half wrong.** The rail
   already renders every step *name* — `1·Check PC 2·Model 3·Basics 4·Install
   5·Ready` — so what is coming was always visible. What was genuinely missing is
   the duration, which is now stated up front.
2. ~~No time estimate anywhere.~~ **Fixed.** The header states `5 steps · about
   N minutes · nothing leaves this computer`, computed from the selected model's
   download size, and the install row switches to the host's own ETA once a
   download is running.
3. ~~Step 3 asks questions the user cannot answer.~~ **Fixed.** Each field now
   carries the consequence of its current value on the row itself — `Comfortable
   · about 1.9 GB of 12 GB`, `Text only — nothing extra loaded`, `Strata decides,
   and adjusts as it goes` — using the same `/api/engine/validate` preview the
   Engine tuning screen uses, so the two screens cannot give different answers to
   the same question.
4. ~~The exit does not explain itself.~~ **Fixed.** It is now *Finish later* with
   a tooltip naming where to resume from. The plan over-claimed that no exit
   existed — one did, it was simply named for leaving rather than pausing.
5. **The scan is a spectacle with no substance.** It animates convincingly, then
   reports four facts. The wait should be doing something useful — it is the
   natural moment to reveal the plan.
6. **No confirmation that anything is local.** The reassurance is present in the
   copy but not as a visible, verifiable fact at the moment of the download.
7. **Step 5 ends flat.** The most satisfying moment in the product is a check
   mark and a button.

## The plan

### Substance — remove anxiety at each step
1. ~~Open with the shape of the task.~~ **Done.** Verified rendering as
   `5 steps · about 2 minutes · nothing leaves this computer` for the default
   model. The estimate follows the formula `max(2, 1 + downloadGb × 1.2)` —
   2 min for the 0.5 GB default, 13–16 min for the 9.6–12.9 GB options — but
   **the model switch itself is unverified in the UI**, because setup is already
   complete on this machine and the wizard renders its completed path.
2. **The step rail shows all step names with icons**, and the current one
   expanded, so the map is legible rather than a set of tick marks.
3. ~~Real time estimates on download and install.~~ **Done**, and the step-3
   fields now state their consequences.
4. **Step 1's scan reveals its result progressively** and ends on a summary that
   says plainly what the machine can and cannot do — the same fit language the
   Models plan introduces, so the vocabulary is learned once.
5. **"Finish later" is explicit** and says what will and will not work until
   setup completes. Today the modal can be dismissed without telling you the
   consequence.

### Fluidity
6. **Step 5 becomes a hand-off, not a dead end**: the ready state names the model,
   states that it is loaded, and offers the first prompt directly in the modal —
   or closes into Chat with the composer focused.
7. **The completion mark settles** with the existing `st-markpop`, then the modal
   fades rather than cutting.

## Motion

The wizard is the one place the choreographed tier is appropriate — it is seen
once, and it sets the tone.

| Interaction | Spec |
|---|---|
| Step forward/back | Direction-aware slide, `--st-t-choreo`, already built |
| Scan line | Loops only while scanning; stops entirely when done |
| Check marks settling | `st-checkin` staggered 60 ms — built |
| Completion | `st-markpop` spring, then a `--st-t-surface` fade of the modal |
| Reduced motion | Every step becomes a cross-fade; the scan becomes a static progress line |

## Verification

- Assert the modal can be dismissed and **reopened with state intact**.
- Assert every step renders a name in the rail, not just a number.
- Assert the download step shows a time estimate.
- `flow setup` already exists; extend it to dismiss mid-way and resume.
