# Persona review and UI simplification decisions

**Implementation update — 2026-10-05:** the adopted decisions and two provisional coordinator choices have now been implemented. See [implementation, verification and screenshots](implementation-verification.md). The review and original votes below are preserved as the historical decision record.

Eight subagents reviewed Engine, Models and onboarding, each using **GPT-6.1 Sol at Medium effort**. They first reviewed independently, then read each other's reports, challenged specific proposals and voted on a common ballot.

The main recommendation is to **remove repeated explanations and duplicate objects, keep useful facts compact, and reveal mechanisms on request**. The review supports a context-first Engine, one recommendation card per unique model setup, a compact comparison table and optional tuning during onboarding.

**Result:** 160 valid votes across 20 decisions; 18 decisions have a majority, including 14 unanimous decisions. Two remain unresolved. No application code changed during this review. These are simulated persona judgments, not results from eight human participants.

## Review scope and method

Each reviewer inspected current source and actual saved captures for Engine, Models, comparison and onboarding. Their individual reports identify which additional states were assessed from source rather than screenshots. Expert/Raw, the library, all five existing setup steps, mobile implications and important failure/unknown states were considered. No browser, server or local inference engine was launched for this review.

The eight first-round reports contain 12 proposals each: **96 overlapping proposals, not 96 distinct defects**. The coordinator consolidated these into [20 ballot decisions](ballot.md). Reviewers supplied individual reasons, peer challenges, concessions and changed or held positions. All eight votes carry equal weight; at least five votes are required for a majority. No deciding vote was added by the coordinator.

Saved screenshots and current source take precedence over historical design notes. Supplied identity, GT Walsheim, Anron, readable text sizes, opaque content surfaces and restrained control motion remain the baseline. This review focuses on information hierarchy; it does not propose another visual identity or more animation.

## The main votes

| Change | Vote | Disposition |
|---|---|---|
| Remove slogans, duplicate purposes and repeated instruction text | **8–0** | Do |
| Engine Beginner shows context first; batch moves to Automatic/Custom summary with Adjust | **7–1** | Do; gamer prefers both dials |
| Keep compact reading and answer rates visible in Engine | **6–2** | Do; newcomer and office prefer memory-first with speeds disclosed |
| One card per unique recommended model setup | **6–2** | Do; Mac and newcomer prefer a profile selector with one result |
| Exact model/version/quantization remains readable | **8–0** | Keep |
| Replace nested comparison panels with aligned facts | **8–0** | Do |
| Four core onboarding steps, with tuning optional before Prepare | **6–2** | Do; teacher and office prefer a separate Review setup step |
| Short capability-review status replaces generic tradeoff prose | **8–0** | Do using actual metadata, without new capability claims |
| Put both rates on ordinary model cards | **4 both / 1 primary only / 3 speeds in Details** | No majority |
| Rename Agentic / Balanced / Performance | **4 keep / 4 rename** | No majority |

The most frequent top-three priorities were compact comparison (**5 reviewers**), unique recommendation cards (**4**), and removing repeated copy / clearer download review (**3 each**). These priority mentions are separate from approval votes.

## What the debate changed

- **Tinkerer and multi-chat operator conceded batch disclosure.** Both initially preferred two visible dials. They accepted a context-first Beginner view once Automatic/Custom state, exact values and immediate Expert access were protected.
- **Gamer conceded optional tuning in onboarding.** The download/configuration review matters more than requiring a separate Tune stop. The teacher and office reviewers still favor that separate checkpoint.
- **Mac conceded visible Discard and recommended Expert values.** Those support recovery and comparison; they should not disappear into a generic menu.
- **Several reviewers changed from duplicate profile cards to unique model cards.** The Mac and newcomer retained their preference for one selected result. The majority wanted simultaneous visibility of genuinely different winners and unavailable profiles.
- **Office changed to memory-first Engine feedback; research did not.** Research distinguished choosing a model from adjusting it: compact rates help while tuning, while ordinary recommendation cards can prioritize fit. This disagreement remains visible in the vote.
- **Teacher corrected two proposed simplifications.** Quantization is not a file format, and selecting a model does not immediately begin its download. Friendlier wording must remain accurate.
- **Multi-chat operator narrowed the queue proposal.** Relevant activity and consequences belong near lifecycle actions; idle zero counters and a new scheduling dashboard would add noise.

These are recorded positions and concessions, not an invented consensus narrative. See the individual debate files below.

## Engine changes

### Default surface

Keep the title, exact loaded variant, Beginner/Expert choice and one concise configuration state. Remove the extra TUNE YOUR ENGINE introduction, slogan and draft reassurance paragraph.

Show the context dial with one short hint, for example **“More text uses more memory.”** Batch becomes **“Prompt batch · Automatic”** or **“Prompt batch · Custom: 1,024”**, followed by an accessible Adjust disclosure. Opening it reveals the existing batch dial, numeric entry, reset and explanation. A custom value must never silently revert to Automatic. Exact entry and keyboard access remain available without pointer precision or hover-only help.

The preview shows a modest pair of **Reading / Answer** ranges, one correctly scoped speed-source label, and a compact memory requirement/headroom/fit summary. Keep the workload selector. The three-phase timing path, memory-component legend, shared-pool explanation and full provenance move into Details. Show a changed-memory line only when relevant, such as **“Draft: +0.09 GB estimated.”** No total task-time claim replaces the removed timing diagram.

### Actions and Expert

- Primary **Save for next load**, secondary **Apply now…**, and visible **Discard changes** when dirty.
- More actions contains recommended reset, manual validation and Demo preview. Save still validates automatically.
- Sampling changes keep their actual next-request scope beside the edited controls; removing the footer paragraph must not imply every saved setting waits for a model reload.
- Expert retains technical key, value, recommendation, applicability, preserved-only status, errors and per-field revert. General descriptions collapse; **Show descriptions** restores them together.
- Raw becomes **Configuration draft**. Adapter and schema explanations remain accessible through Details. Imported or unrecognized keys remain preserved, without implying support.

**Acceptance:** every control has a readable label; no smaller typography is used to fit content; draft/saved/applied states are distinguishable; normal Engine use does not require reading an introduction; batch and precise entry remain one deliberate interaction away.

## Models changes

### Recommendation structure

Render one compact card for each **unique exact variant plus execution configuration**. If Balanced and Performance choose the same setup, that card gets both profile chips and one action. Do not merge the same model file when the recommended context/cache/placement or other execution settings differ.

Keep a profile without a candidate as a compact actionable row: **“No approved Agentic model fits this setup · Review context.”** It should not occupy a tall empty card. All three priorities remain identifiable; no profile selection silently switches the loaded model.

Recommended card order:

1. Profile chip or chips.
2. Exact model/version and quantization; omit redundant nicknames.
3. Estimated working memory/fit and download size, clearly distinguished.
4. A compact speed row if retained after resolving B08.
5. One short actual review status replacing generic prose, such as **“Capability review pending.”** Scope it to the metadata available.
6. One state-aware action and **Why this model?**

Move long purpose, selection rationale, timing breakdown, full configuration and provenance into Why this model. Keep relevant limitations, unsupported states and unknown fit visible. Review status is not a claim that a model can edit XLSX, safely handle contracts, produce reliable research or keep data private.

### Library and comparison

Default to Installed when models exist, otherwise Discover; remember a deliberate tab choice. Remove the second Choose a model button from Models because the library already supplies that task. Keep the shared picker in Engine and onboarding.

Retain Family → Version → Quantization, exact search, filters, compatible defaults, optional incompatible variants and reversible expansion. Prefer **“All variants”** or **“All quantizations”** over “file formats.” Capability explanations remain accessible; model names cannot manufacture capability evidence.

Comparison becomes aligned rows for exact variant, fit, memory/headroom, download, reading, answer and evidence. Share the workload/configuration controls. Per-variant Details contains timing, allocation and provenance. Unknown values remain labelled and over-budget warnings stay beside the action. On mobile, stack the same labelled rows without requiring horizontal swiping.

**Acceptance:** the existing shared-winner fixture produces one model card and one compact Agentic no-fit state; the library is substantially closer to the recommendations; exact variants cannot be confused; no repeated full preview panels remain in comparison. Card height grows naturally for long names and large text rather than clipping or shrinking them.

## Onboarding changes

The majority supports **Computer → Model → Prepare → Ready**, with a mandatory compact download/configuration review at the end of Model and optional **Adjust settings** inside that review. This changes navigation count, not the required commitment checks.

| Existing step | Proposed disposition |
|---|---|
| Computer | Keep actual verdict, memory/storage and uncertainty. Move full processor/accelerator checks into Computer details. Retry is prominent on failure. |
| Profile and model | Keep Balanced as the new-user default, all three priorities and one selected recommendation. Remove repeated selected-profile purpose. Compare remains optional. |
| Tune | Fold its summary into the explicit pre-download review. Show exact choice, estimated fit, actual download amount and Custom status. Adjust settings expands the controls; invalid retained values open them with a recovery action. |
| Prepare | One actual stage label and progress visual. Keep accurate verification scope, specific retry, retained-files status and cancellation consequences beside the relevant action. Detailed output is disclosed. |
| Ready | Truthful loaded/Demo status, exact model, short profile/context summary and Start chatting. Memory breakdown and Demo test move to Setup details. |

The teacher and office dissent is substantive: a separately named Review setup step provides orientation and a clear commitment boundary. The four-step implementation must preserve both. Do not collapse model selection and downloading into one surprising click.

**Acceptance:** selecting a profile or variant never starts a download; the final load/download action follows an explicit review; interrupted old five-step progress maps to the equivalent new state without losing choices; files retained after a failed load stay retained; readiness still requires successful loading. No new promise of resumable bytes, background setup or concurrent inference is introduced.

## Operational finding to fix with the simplification

The parallel-session reviewer identified inconsistent visible lifecycle checks. Source confirms `ModelAction` considers running plus queued requests, while `lifecycleAvailable` in `src/lib/store.ts` checks streaming/running work and lifecycle operations. The Tuning confirmation does not use the same visible queue blocker. This observation does **not** establish that the backend loses or interrupts queued work.

Use one lifecycle policy and compact **“Global · all chats”** scope near reload/switch/unload. Known active/queued work appears when relevant, with a Serving link; unavailable status must not become zero. Review blockers before a consequential operation or related settings write. Preserve the different effects of saving sampling defaults and restarting the engine. No scheduler or “14 simultaneous agents” control is proposed.

## Unresolved choices and coordinator recommendation

These recommendations are the coordinator's judgment, **not additional persona votes**.

| Split | Recommendation for the next iteration | What to test |
|---|---|---|
| Card speeds: 4 both / 1 primary / 3 in Details | Start with one modest two-rate row, without hero numerals. This preserves the existing performance-focused brief while the unanimous cuts remove much more prose and duplication. Keep fit/download/review above it. Retain a fit-first alternative for testing. | Can users distinguish reading speed, answer speed and task quality? Does the row help model choice or distract from fit/download? |
| Names: 4 keep / 4 rename | Keep Agentic / Balanced / Performance for continuity, with short plain descriptors shown once. Do not rename a stable concept based on a tied simulated vote. | Can actual newcomers explain the priorities and select a sensible model? Compare with Tools and agents / Everyday work / Faster answers. |

Do not present either option as established by usability evidence. A real participant walkthrough can settle these more usefully than manufacturing another AI consensus.

## What to keep out of this revision

- No smaller text, unexplained icon-only controls or hover-only essential help.
- No deletion of Expert/Raw, exact variant identity, comparison or recovery actions.
- No generic green “Works great” badge replacing unknown fit, Demo or provisional evidence.
- No quality/reliability score derived from token speed, memory fit or model size.
- No hidden restart, automatic model switch or profile-driven sampling change.
- No new queue dashboard, automatic drain scheduler or unsupported parallel-capacity claim.
- No extra warning paragraphs for hypothetical uses. Keep relevant short facts and actual review status; put supporting explanation in Details.

## Implementation order and checks

1. **Shared copy and data hierarchy:** remove duplicate introductions/purposes, consolidate source labels correctly, shorten sample names to text workloads, preserve effect timing and review scope.
2. **Models and comparison:** unique setup identity, compact empty states/cards, aligned comparison, Installed default. Comparison received the most top-three mentions.
3. **Engine:** context-first layout, compact preview, simplified action bar, descriptions toggle in Expert and clear draft labels.
4. **Onboarding:** reuse the compact components, fold optional Tune into review, migrate saved progress and verify recovery.
5. **Lifecycle consistency:** implement shared checks alongside affected action changes, before relying on the simplified copy to communicate consequences.

Verify keyboard/direct entry, Light/Dark, reduced motion/transparency, increased contrast, long model names, unknown evidence, distinct configurations sharing a file, no fit, shared winners, 390/768/1024/1512px and 200% zoom. Walk through first successful setup, interrupted setup, failed loading and cancellation. Test that a busy or unknown engine state receives consistent action handling. Measure visible copy and reading/navigation burden after implementation; this review claims no measured reduction or usability gain.

## Complete ballot results

| ID | Decision | Votes | Result |
|---|---|---|---|
| B01 | Remove repeated introductions and teaching | YES 8 | Adopt |
| B02 | Beginner dial hierarchy | CONTEXT 7; TWO 1 | Context first |
| B03 | Engine feedback | RATES 6; MEMORY 2 | Keep compact rates |
| B04 | Simplify action bar | YES 8 | Adopt |
| B05 | Compact Expert with optional descriptions | YES 8 | Adopt |
| B06 | Recommendation presentation | UNIQUE 6; SELECTOR 2; THREE 0 | Unique setups |
| B07 | Keep exact variant identity visible | YES 8 | Keep |
| B08 | Ordinary card metrics | BOTH 4; PRIMARY 1; FIT 3 | Unresolved |
| B09 | Compact no-fit and duplicate-winner states | YES 8 | Adopt |
| B10 | Library entry and duplicate picker | YES 8 | Adopt |
| B11 | Aligned comparison | YES 8 | Adopt |
| B12 | Onboarding structure | OPTIONAL 6; FIVE 2 | Optional Tune |
| B13 | Profile naming | KEEP 4; RENAME 4 | Unresolved |
| B14 | Compact pre-download setup review | YES 8 | Adopt |
| B15 | Concise progress with truthful recovery | YES 8 | Adopt |
| B16 | Quiet Ready screen | YES 8 | Adopt |
| B17 | Consistent global lifecycle consequences | YES 8 | Adopt |
| B18 | Consolidate correctly scoped evidence labels | YES 8 | Adopt |
| B19 | Visible actual capability-review status | VISIBLE 8 | Adopt |
| B20 | Workloads clearly labelled as text samples | YES 8 | Adopt |

## Individual evidence and votes

| Reviewer | Independent report | Peer debate | Raw vote |
|---|---|---|---|
| 1 PC gamer | [Review](01-gamer.md) | [Debate](01-gamer-debate.md) | [JSON](01-gamer-vote.json) |
| 2 Typical Mac user | [Review](02-mac.md) | [Debate](02-mac-debate.md) | [JSON](02-mac-vote.json) |
| 3 Older teacher exploring local AI | [Review](03-teacher.md) | [Debate](03-teacher-debate.md) | [JSON](03-teacher-vote.json) |
| 4 Newcomer who heard about it on TikTok | [Review](04-newcomer.md) | [Debate](04-newcomer-debate.md) | [JSON](04-newcomer-vote.json) |
| 5 Local AI tinkerer | [Review](05-tinkerer.md) | [Debate](05-tinkerer-debate.md) | [JSON](05-tinkerer-vote.json) |
| 6 Operator of many agentic chats | [Review](06-parallel.md) | [Debate](06-parallel-debate.md) | [JSON](06-parallel-vote.json) |
| 7 Casual research and work user | [Review](07-research.md) | [Debate](07-research-debate.md) | [JSON](07-research-vote.json) |
| 8 Regular workplace document user | [Review](08-office.md) | [Debate](08-office-debate.md) | [JSON](08-office-vote.json) |

[Common ballot](ballot.md) · [Validated tally](vote-tally.json) · [Implementation being reviewed](../engine-models-onboarding-2026-10-05.md) · [Screenshot evidence](../evidence/instrument-redesign-2026-10-05/)

Only documentation was created during the original review. The subsequent authorized implementation is recorded in [implementation and verification](implementation-verification.md).
