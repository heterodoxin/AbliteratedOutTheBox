/**
 * Plans (the /plan command) and goal points, as the host sees them. Pure helpers plus a small store; the bridge drives them.
 *
 * A plan is a short numbered list the assistant wrote after looking around (the chat was in Plan mode: nothing could change). The
 * person approves it, edits it, rejects it, or starts it as a goal whose steps are the plan's steps. A /goal whose text already has
 * numbered points ("1. ... 2. ... 3. ...") is run the same way, so a point is a step and each one is reported.
 *
 * Plan shape (also what the page renders, notes-split-agent6.md):
 *   { id, chatId, title, request, source: 'plan'|'goal', state, steps: [{ id, text, detail, state, summary }], createdAt, updatedAt, goalStarted }
 *   plan.state:  proposed | approved | rejected | running | paused | done | stopped
 *   step.state:  pending | running | done | failed | skipped
 */

import { randomUUID } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';

export const MAX_STEPS = 12;
const MAX_TEXT = 140;
const MAX_DETAIL = 220;

const clean = s => String(s ?? '').replace(/\*\*|__|`/g, '').replace(/\s+/g, ' ').trim();
const clip = (s, n) => (s.length > n ? s.slice(0, n - 1).trimEnd() + '…' : s);

/** "Write the file — one file, inline styles" into { text, detail }. */
function splitStep(raw) {
  const s = clean(raw).replace(/^(?:step\s*\d+\s*[:.\-]\s*)/i, '');
  const at = s.search(/\s[—–]\s|\s-\s|:\s/);
  const head = at > 8 ? s.slice(0, at) : s;
  const rest = at > 8 ? s.slice(at).replace(/^\s*[—–:\-]\s*/, '') : '';
  return { text: clip(head.replace(/[.;:]+$/, ''), MAX_TEXT), detail: clip(rest, MAX_DETAIL) };
}

/** The instruction a plan turn gets in front of the person's request. */
export function planPrompt(task) {
  return [
    'Make a plan for the request below. Do not change anything: you may only look (read files, list folders, search).',
    'When you have looked enough, end your reply with the plan as a numbered list: one short line per step, in the order they should be done.',
    'Put a short heading line right before the list, like "## Pagoda page" (two to five words that name the plan).',
    'You may add the reason after a dash. Use 3 to 8 steps. Write nothing after the list.',
    '',
    `Request: ${task}`,
  ].join('\n');
}

/**
 * The plan in a reply: its last run of numbered lines (at least two). Lines that follow a step and are indented or bulleted are that
 * step's detail. Returns null when the reply holds no list, so the page shows the reply as it is and never a made-up plan.
 */
export function parsePlan(text, request = '') {
  const lines = String(text ?? '').replace(/\r/g, '').split('\n');
  const item = /^\s{0,3}(?:step\s+)?(\d{1,2})[.):]\s+(\S.*)$/i;
  const runs = [];
  let run = null;
  lines.forEach((line, i) => {
    const m = item.exec(line);
    if (m) {
      const n = Number(m[1]);
      if (run && n === run.items.length + 1) { run.items.push({ raw: m[2], extra: [] }); run.end = i; return; }
      if (n === 1) { run = { start: i, end: i, items: [{ raw: m[2], extra: [] }] }; runs.push(run); return; }
      run = null; return;
    }
    if (run && /^\s+\S/.test(line)) { run.items.at(-1).extra.push(clean(line.replace(/^\s*[-*•]\s*/, ''))); run.end = i; return; }
    if (run && /^\s*[-*•]\s+\S/.test(line)) { run.items.at(-1).extra.push(clean(line.replace(/^\s*[-*•]\s*/, ''))); run.end = i; return; }
    if (run && line.trim() === '') return;
    run = null;
  });
  const best = runs.filter(r => r.items.length >= 2).at(-1);
  if (!best) return null;
  const steps = best.items.slice(0, MAX_STEPS).map((it, k) => {
    const { text: t, detail } = splitStep(it.raw);
    const extra = clip(it.extra.join(' '), MAX_DETAIL);
    return { id: `s${k + 1}`, text: t, detail: detail || extra, state: 'pending', summary: null };
  }).filter(s => s.text);
  if (steps.length < 2) return null;
  let title = '';
  for (let i = best.start - 1; i >= 0 && i >= best.start - 6; i--) {
    const h = /^\s*#{1,6}\s+(.+?)\s*$/.exec(lines[i]);
    if (h) { title = clean(h[1]); break; }
  }
  return { title: clip(title || clean(request) || 'Plan', 80), steps };
}

/** Numbered points inside a goal's own words ("1. a 2. b" on lines or in one line); at least two, counting up from 1. */
export function parsePoints(objective) {
  const text = String(objective ?? '').replace(/\r/g, '');
  const found = [];
  const re = /(?:^|\s)(\d{1,2})[.)]\s+/g;
  let m; const marks = [];
  while ((m = re.exec(text))) marks.push({ n: Number(m[1]), at: m.index, after: m.index + m[0].length });
  let want = 1; let start = -1;
  const seq = [];
  for (const k of marks) {
    if (k.n === want) { if (want === 1) start = seq.length; seq.push(k); want++; }
    else if (k.n === 1) { seq.length = 0; seq.push(k); want = 2; }
  }
  if (seq.length < 2) return null;
  seq.forEach((k, i) => found.push(clean(text.slice(k.after, i + 1 < seq.length ? seq[i + 1].at : undefined))));
  const aim = clean(text.slice(0, seq[0].at)).replace(/[:\-–—]+$/, '').trim();
  const steps = found.slice(0, MAX_STEPS).map((t, k) => {
    const { text: s, detail } = splitStep(t);
    return { id: `s${k + 1}`, text: s, detail, state: 'pending', summary: null };
  }).filter(s => s.text);
  if (steps.length < 2) return null;
  return { title: clip(aim || steps[0].text, 80), steps };
}

/** A step that is about looking at the result in a browser or window: it must never hang or fail a whole run on a machine with no screen. */
export const isViewStep = text => /\b(open|view|launch|preview|look at|check|confirm|verify)\b[^\n]{0,60}\b(browser|in chrome|in edge|in firefox|window|on screen)\b/i.test(String(text ?? ''));

/** The goal the Harness runs for step `index` (0-based): one step, with the aim and the steps around it named. */
export function stepObjective(plan, index) {
  const n = plan.steps.length; const s = plan.steps[index];
  const done = plan.steps.slice(0, index).map((x, i) => `${i + 1}. ${x.text}${x.summary ? ' (' + String(x.summary).slice(0, 100) + ')' : ''}`).join('; ');
  return [
    `Overall aim: ${plan.title}.`,
    `You are on step ${index + 1} of ${n}: ${s.text}${s.detail ? ` (${s.detail})` : ''}.`,
    done ? `Already done: ${done}.` : '',
    plan.briefing ? String(plan.briefing) : '',
    isViewStep(s.text) ? 'This step is about seeing the result. To show a local image, SVG, HTML page or PDF, use the present tool (a browser cannot open file:// addresses). Never wait for a program to close. If a person is at this computer and you start a browser, start it so the command returns at once (Windows: Start-Process; macOS: open; Linux: xdg-open) and say only that you started it, not that the page rendered. If there is no screen, check the page without a visible window: use the headless browser tool if you have one, otherwise read the file and its script for mistakes, and say in one plain sentence what you could and could not check. Then finish the step.' : '',
    'Do only this step, check that it worked, and then finish the goal.',
    index + 1 < n ? 'The next steps are handled separately: do not start them.' : '',
  ].filter(Boolean).join(' ');
}

const WORD = { started: 'started', done: 'done', failed: "didn't finish", stopped: 'stopped', skipped: 'skipped' };
/** One short friendly line for a step: "Write the roof — done" (no step counter: the owner removed it). */
export function stepLine(index, total, text, state) {
  return `${clip(clean(text), 90)} — ${WORD[state] ?? state}`;
}

/** The short summary of a step from the words it closed with: the first sentence, plain. */
export function firstSentence(text, max = 140) {
  const s = clean(String(text ?? '').replace(/^#+\s.*$/gm, '').replace(/^[-*]\s+/gm, ''));
  if (!s) return null;
  const m = /^(.{12,}?[.!?])(?:\s|$)/.exec(s);
  return clip(m ? m[1] : s, max);
}

/**
 * The short summary of a finished step: what it changed (from the tool cards: "Wrote pagoda.html, Changed style.css"), else the first
 * sentence of its closing words when that reads as a result. A sentence about what is NOT done, or about later steps, is never a summary.
 */
export function stepSummary(cards, closing) {
  const done = [];
  for (const c of cards ?? []) {
    const verb = /^(Wrote|Changed|Created|Edited|Deleted|Removed|Moved|Renamed)\b/.exec(String(c?.verb ?? ''))?.[1];
    const target = String(c?.target ?? '').split(/[\\/]/).filter(Boolean).at(-1);
    if (!verb || !target) continue;
    const line = verb + ' ' + clip(target, 40);
    if (!done.includes(line)) done.push(line);
  }
  if (done.length) return clip(done.slice(0, 3).join(', ') + (done.length > 3 ? ' and more' : ''), 140);
  const s = firstSentence(closing);
  return s && !/not done|not yet|remaining|handled separately|next steps?|later steps?|still (?:need|to)|todo/i.test(s) ? s : null;
}

/** The words a person sees for the plan, as one message the model can carry out (the plain Approve path). */
export function planAsPrompt(plan) {
  return ['Carry out this plan, step by step, and tell me briefly when each step is done.', '', plan.title ? `Plan: ${plan.title}` : '', ...plan.steps.map((s, i) => `${i + 1}. ${s.text}${s.detail ? ` (${s.detail})` : ''}`)].filter((l, i) => l !== '' || i === 1).join('\n');
}

/** A step list from the person's edit: texts only, validated and capped. Throws a plain sentence for an unusable list. */
export function editedSteps(input, previous = []) {
  if (!Array.isArray(input)) throw Object.assign(new Error('A plan needs a list of steps.'), { code: 'plan-steps' });
  const steps = input.map((x, k) => {
    const text = clip(clean(typeof x === 'string' ? x : x?.text), MAX_TEXT);
    const detail = clip(clean(typeof x === 'string' ? '' : x?.detail), MAX_DETAIL);
    const kept = previous.find(p => p.id === x?.id);
    return text ? { id: kept?.id ?? `s${k + 1}`, text, detail, state: 'pending', summary: null } : null;
  }).filter(Boolean).slice(0, MAX_STEPS);
  if (!steps.length) throw Object.assign(new Error('A plan needs at least one step.'), { code: 'plan-steps' });
  // ids must be unique and in order of the list
  steps.forEach((s, i) => { s.id = `s${i + 1}`; });
  return steps;
}

/** The latest plan of each chat, kept in memory and in one small file so a restart does not lose a proposal. */
export class PlanStore {
  constructor(file = null) {
    this.file = file; this.plans = new Map();
    if (file && existsSync(file)) {
      try { for (const p of JSON.parse(readFileSync(file, 'utf8'))) if (p?.chatId) this.plans.set(p.chatId, p); } catch { /* a damaged file starts empty */ }
    }
  }
  #save() {
    if (!this.file) return;
    try {
      mkdirSync(dirname(this.file), { recursive: true });
      const tmp = this.file + '.tmp';
      writeFileSync(tmp, JSON.stringify([...this.plans.values()]));
      renameSync(tmp, this.file);
    } catch { /* the plan still lives in memory */ }
  }
  get(chatId) { return this.plans.get(chatId) ?? null; }
  create(chatId, { title, steps, request = '', source = 'plan', state = 'proposed', caution = null }) {
    const now = new Date().toISOString();
    const plan = { id: randomUUID(), chatId, title, request, source, state, steps, createdAt: now, updatedAt: now, goalStarted: false, ...(caution ? { caution } : {}) };
    this.plans.set(chatId, plan); this.#save();
    return plan;
  }
  update(chatId, fn) {
    const plan = this.plans.get(chatId);
    if (!plan) return null;
    fn(plan); plan.updatedAt = new Date().toISOString();
    this.#save();
    return plan;
  }
  clear(chatId) { this.plans.delete(chatId); this.#save(); }
}

/** What the page gets: a copy with no host-only fields. */
export const planView = plan => (plan ? JSON.parse(JSON.stringify(plan)) : null);

// ---- /research ------------------------------------------------------------------------------------------------------------------

/** The first research prompt: find and read pages, take notes. What the assistant writes here is not shown; the answer is written in a second step. */
export function researchPrompt(topic) {
  return [
    `Research task: ${topic}`,
    '',
    'Do these steps in order. Do not skip a step.',
    '1. Call web_search to find pages about the task.',
    '2. Call web_fetch to open at least two of the pages you found, and read them. Do not rely on the search results alone.',
    '3. When you have read them, reply with short notes: the facts you found, each with the web address of the page it came from.',
  ].join('\n');
}

/** Sent when the assistant answered without opening pages. */
export const RESEARCH_NUDGE = 'You have not opened enough pages yet. Call web_fetch now on at least two of the pages from your search, read them, then reply with short notes and the web address of each page.';

/** The second research prompt: write the answer in the fixed shape. */
export function researchWritePrompt(topic) {
  return [
    `Now write the final answer to the research task: ${topic}`,
    'Use only what you read on the pages. Do not call any tool. Start your reply with the line "## Summary" and use exactly these three headings:',
    '## Summary',
    'Three to five sentences, in plain words. If the sources disagree or something could not be found, say so.',
    '## Key findings',
    '- One finding per line, each ending with its source number in square brackets, like [1].',
    '## Sources',
    '1. Page title - web address',
    '2. Page title - web address',
    'List only pages you opened.',
  ].join('\n');
}

/** The numbered sources at the end of a research answer: [{ n, title, url }]. Empty when there are none. */
export function parseSources(text) {
  const lines = String(text ?? '').replace(/\r/g, '').split('\n');
  const at = lines.map(l => /^\s{0,3}#{1,6}\s*sources\b/i.test(l) || /^\s*\*{0,2}sources\*{0,2}:?\s*$/i.test(l)).lastIndexOf(true);
  if (at < 0) return linkSources(text);
  const out = [];
  for (const line of lines.slice(at + 1)) {
    if (/^\s{0,3}#{1,6}\s/.test(line)) break;
    const url = /(https?:\/\/[^\s)>\]"']+)/.exec(line)?.[1]?.replace(/[.,;]+$/, '');
    if (!url) continue;
    const md = /\[([^\]]+)\]\(https?:\/\//.exec(line);
    const before = line.slice(0, line.indexOf(url)).replace(/^\s*(?:\d+[.)]|[-*•])\s*/, '').replace(/[\s\-–—:(<[]+$/, '').replace(/\*\*|`/g, '').trim();
    const title = clean(md?.[1] ?? before) || (() => { try { return new URL(url).hostname; } catch { return url; } })();
    if (!out.some(s => s.url === url)) out.push({ n: out.length + 1, title: clip(title, 120), url });
  }
  return out.slice(0, 20);
}

/** No Sources section: the pages the answer links to, in the order they appear, so the page can still list where it came from. */
function linkSources(text) {
  const out = [];
  const re = /\[([^\]]{1,160})\]\((https?:\/\/[^\s)]+)\)|(https?:\/\/[^\s)>\]"']+)/g;
  let m;
  while ((m = re.exec(String(text ?? ''))) && out.length < 8) {
    const url = (m[2] ?? m[3]).replace(/[.,;]+$/, '');
    if (out.some(x => x.url === url)) continue;
    let host = url; try { host = new URL(url).hostname; } catch { /* keep the address */ }
    out.push({ n: out.length + 1, title: clip(clean(m[1] ?? '') || host, 120), url });
  }
  return out;
}
