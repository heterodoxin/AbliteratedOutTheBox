# Shared debate and decision ballot

This is round two of eight simulated persona reviews. All reviewers use GPT-6.1 Sol with Medium effort. These are design judgments from AI reviewers, not observations from eight human participants. The current task changes review documents only.

## Debate rules

Read the eight independent reports in this directory before voting. Compare the actual arguments, not demographic stereotypes. Explicitly challenge at least two other reviewers by report/proposal reference, concede at least one useful opposing point, and identify any position you changed after reading them. It is acceptable to keep a position; do not manufacture consensus or a change of mind.

Vote on every item exactly as written. Each vote has one allowed choice and a short reason. You may record an amendment, but a vote against the current wording remains against it. No abstentions or conditional yes votes: choose the best stated option and describe concerns separately. Ties or lack of a majority remain unresolved; the coordinator will not invent a deciding persona vote. Decisions remain recommendations pending real user testing.

Universal constraints: retain current font sizes/contrast/accessibility, supplied identity and motion rules; preserve Expert/Raw controls and exact variant access; keep actionable errors, unknowns and capability limitations visible when they affect a decision; never invent benchmark, quality, privacy, document-tool or concurrency capabilities. Native integration remains deferred.

## Proposals

| ID | Decision | Allowed choices |
|---|---|---|
| B01 | Remove decorative introductory copy and repeated teaching text: Engine slogan/eyebrow/reassurance, Models section slogans, repeated selected-profile purpose and duplicate gesture captions. Retain one short useful control hint and a concise draft/saved/applied state. | YES / NO |
| B02 | Engine Beginner primary controls. All choices retain shared draft, full Expert access, keyboard and exact-entry alternatives, and a visibly identified custom batch. | TWO = keep both dials visible; CONTEXT = context dial visible, batch defaults to a labelled Automatic/Custom summary with Adjust disclosure. |
| B03 | Engine default feedback. Neither option includes the full timing path, memory taxonomy or explanatory paragraphs by default; those remain in Details. Source, fit blockers, memory requirement/headroom and workload meaning remain explicit. | RATES = both labelled reading/answer ranges plus compact memory visible; MEMORY = memory/fit visible, speed ranges and sample controls under Speed details. |
| B04 | Engine actions: primary Save for next load, secondary Apply now when available, Discard when changed; put defaults reset, manual validation and demo preview in a labelled More menu. Save validates; restart remains explicit with scope and interruption review. | YES / NO |
| B05 | Expert rows retain label/key/value, recommended value, applicability/scope, preserved-only state, errors and per-field revert. Collapse general effect paragraphs by default; offer Show descriptions for those who want them. Raw is labelled Configuration draft; adapter narrative moves to Details. | YES / NO |
| B06 | Recommendation presentation on Models. Every option keeps all three priorities accessible and does not change the loaded model on selection. No-fit is explicit, with a recovery action. | THREE = three compact profile cards; SELECTOR = profile selector with one recommendation for the selected priority; UNIQUE = one compact card per unique exact winner with profile chips and a compact no-fit row. |
| B07 | Keep exact model/version and quantization visibly readable on recommendation cards, variant rows, comparison and selection confirmation; remove redundant nicknames. Long identity may wrap. Full filename/source remains copyable in Details. | YES / NO |
| B08 | Normal recommendation card metrics, alongside visible fit, working memory and download size. Comparisons still show both rates. Mixed evidence labels remain locally scoped. | BOTH = both rates in a compact row, priority metric emphasized; PRIMARY = only the profile's primary rate, secondary in Details; FIT = fit/download first, all speed information in Details. |
| B09 | Avoid full-height empty recommendations and duplicate downloads/actions. Show no approved fit in a compact actionable state. When winners match, identify that exact match once with profile chips or a short shared-winner note appropriate to B06. | YES / NO |
| B10 | Library defaults to Installed when files exist and Discover otherwise, remembers deliberate tab choice, and removes the redundant Choose a model button on the Models page. Keep family/version/quantization grouping, search, filters and exact comparison. | YES / NO |
| B11 | Replace three nested performance panels in comparison with aligned rows: exact identity, fit, memory/headroom, download, reading, answer and per-variant evidence. Share workload/configuration controls; disclose timing, memory method and provenance. Stack equivalent labelled rows on mobile. | YES / NO |
| B12 | Onboarding structure. Both retain hardware detection, profile/model selection, an explicit download/configuration review, optional tuning, preparation, successful-readiness gate and resume behavior. | FIVE = keep all five named steps; OPTIONAL = four core steps, with Tune as an optional expandable review before Prepare. |
| B13 | Profile naming. Descriptions stay short and capability warnings remain relevant and visible; these labels never constitute a claim of verified tool quality. | KEEP = Agentic / Balanced / Performance with short descriptors Tools and long prompts / Everyday work / Faster answers; RENAME = Tools and agents / Everyday work / Faster answers, canonical technical profile names in Details. |
| B14 | Compact setup steps 1–3: one detection verdict and memory/storage facts, other specifications in Computer details; remove repeated profile purpose; summarize selected exact variant, estimated fit and actual download amount before commitment; optional controls remain collapsed until requested or invalid. | YES / NO |
| B15 | Prepare shows actual stage once, one progress visual and stage-specific retry/cancel. Keep partial-download restart consequences and actionable error beside the affected action. Put raw progress/log detail in disclosure; no fake verification/ETA/background continuation. | YES / NO |
| B16 | Ready shows loaded or Demo state, exact selected variant, brief profile/context summary and Start chatting. Move repeated memory breakdown and optional Demo test into Setup details. Failed readiness still offers Return to loading. | YES / NO |
| B17 | Engine and model lifecycle actions show concise global scope. Show active/queued counts when relevant and known, otherwise unknown; use consistent checks before reload/switch/unload and link to Serving for blockers. Avoid idle zero-count clutter and any promise of parallel inference. | YES / NO |
| B18 | Consolidate duplicate evidence wording into one visibly scoped source badge for a metric group only when all numbers share that source. Keep Estimated memory separate and mixed/unavailable/Provisional states local. Detailed provenance stays accessible. Never hide all Demo/evidence labels. | YES / NO |
| B19 | Capability review on ordinary recommendation cards. Both options keep no-approved-tool-fit and relevant known limitations visible, and never infer spreadsheet/translation/research/contract quality from speed. | VISIBLE = one short review-status line replaces generic tradeoff prose, such as Capability review pending, using actual metadata; DETAILS = review status/rationale in Why this model except decision-blocking gaps, with existing Provisional label visible. |
| B20 | Sample workloads use Short text · 1K / Document text · 4K / Long context · 16K, keeping exact token amounts, 256-output allowance and invalid-sample handling. Help explains these are text performance samples, not file-format or document-task capability promises. | YES / NO |

## Submission

Create only your own `NN-name-vote.json` and `NN-name-debate.md` in this directory. Do not change source, other reviewers' files or the ballot.

JSON format:

```json
{
  "reviewer": "01",
  "persona": "PC gamer",
  "model": "gpt-6.1-sol",
  "effort": "medium",
  "votes": {
    "B01": { "choice": "YES", "reason": "One brief reason." }
  },
  "top_three": ["B01", "B02", "B11"],
  "amendments": ["Any non-voting wording refinement or evidence limitation."]
}
```

Include B01 through B20. The debate Markdown records two or more specific challenges, at least one concession and any changed/held position. Finish your turn after saving and summarizing; the coordinator will count the actual submitted votes.
