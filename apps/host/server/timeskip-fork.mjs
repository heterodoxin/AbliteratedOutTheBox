/**
 * Time-skip = /plan + /goal + a smart subagent fork at the step that matters most (notes-split-agent8.md, item 11).
 *
 *   1. The request is dissected into a plan with steps (the same read-only plan turn as /plan), in a COPY of the project folder (A).
 *   2. One short question picks the lever: the single step where stopping, checking the results so far and improving or diverting
 *      matters most for the final result (with its reason; a fixed rule when the answer is unusable).
 *   3. Agent A runs the plan as a goal, step by step, all the way: RESULT A.
 *   4. When A has finished the step before the lever, the host copies A to B and a read-only helper writes an improvement plan from the
 *      plan, A's results so far and the lever's reason. Agent B gets that plan (and .timeskip/context.md as its attachment) and finishes
 *      its own improved path in B: RESULT B. With one model slot B starts when A is done and the person is told so.
 *   5. A and B are compared side by side; the person keeps one (Keep A / Keep B), with a rollback copy.
 * The person's own folder is never touched until Keep.
 */

import { randomUUID } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { diffRun, runImagePath, isNoisePath } from './timeskip.mjs';
import { isViewStep } from './agent/plan.mjs';

/** The note both agents carry on every step: the work is in a copy, unattended, on whatever machine this is. */
export const UNATTENDED = 'This work runs in a copy of the folder with nobody watching. Never open a visible window and never wait for a program to close. To check a web page use the headless browser tool if you have one; otherwise read the file and say plainly what you could not check. A check that cannot be done here is noted in one sentence and the step is finished.';

const SKIP = new Set(['.git', '.strata', '.timeskip', 'node_modules', '__pycache__']);
const TERMINAL = new Set(['completed', 'failed', 'cancelled', 'interrupted']);

/** Copy a project folder for a trial, without dependency/VCS folders. */
export function copyProject(from, to) {
  mkdirSync(to, { recursive: true });
  cpSync(from, to, { recursive: true, force: true, filter: src => !SKIP.has(src.split(sep).pop()) || resolve(src) === resolve(from) });
}

/** The question that picks the lever. */
export function leverPrompt(request, steps) {
  return [
    'Below is a plan for a request. Find the ONE biggest lever in it: the single step where stopping, checking the results so far and improving or changing direction would matter most for the quality of the final result.',
    'Rules: never step 1 (nothing exists yet to check); prefer the step right after the first real result exists and before the expensive or hard-to-undo part; choose exactly one step.',
    '',
    `Request: ${request}`,
    'Plan:',
    ...steps.map((s, i) => `${i + 1}. ${s.text}${s.detail ? ` (${s.detail})` : ''}`),
    '',
    'Answer with only this JSON on one line, no other words: {"step": <number>, "why": "<one short plain sentence: why checking here matters most>"}',
  ].join('\n');
}

/** The model's answer as { step, why }, or null. The step is 1-based and must be inside the plan and not the first. */
export function parseLever(text, total) {
  const m = /\{[^{}]*"step"[^{}]*\}/s.exec(String(text ?? ''));
  if (!m) return null;
  try {
    const j = JSON.parse(m[0]);
    const step = Number(j.step);
    if (!Number.isInteger(step) || step < 2 || step > total) return null;
    const why = String(j.why ?? '').replace(/\s+/g, ' ').trim().slice(0, 220);
    return { step, why: why || 'This is where the results so far are worth checking.' };
  } catch { return null; }
}

/** A fixed rule for the lever when the model's answer cannot be used: the step after the first step that makes something. */
export function defaultLever(steps) {
  const n = steps.length;
  const making = steps.findIndex((s, i) => i >= 1 && /\b(write|create|build|draw|implement|generate|make|design|code|add)\b/i.test(s.text));
  let step = making >= 0 ? Math.min(n, making + 2) : Math.ceil(n / 2) + 1;
  step = Math.max(2, Math.min(n, step));
  return { step, why: 'Chosen by a default rule (the answer from the model could not be used): the first results exist after the step before it.', defaulted: true };
}

/** The improvement plan request for B: what is done, what the lever is, and what to write. */
export function improvementPrompt({ request, steps, lever, doneSummaries }) {
  return [
    'You are improving work that is already in progress in this folder. Read .timeskip/context.md and the files first (you may only look, not change anything).',
    `Original request: ${request}`,
    'Original plan:',
    ...steps.map((s, i) => `${i + 1}. ${s.text}`),
    `Steps 1 to ${lever.step - 1} are done in this folder${doneSummaries.length ? ': ' + doneSummaries.join(' ') : '.'}`,
    `Step ${lever.step} ("${steps[lever.step - 1]?.text}") is the checkpoint, chosen because: ${lever.why}`,
    '',
    'Write an IMPROVEMENT PLAN: the numbered steps that should now finish the work, improving or changing direction wherever the results so far show a better way. 2 to 8 short steps. Put a short heading line before the list.',
  ].join('\n');
}

const stepObjectiveBriefing = ctx => ctx;

export function createForkTimeskip({ agent, dir, notify = () => {}, now = Date.now, stepTimeoutMs = 15 * 60_000 }) {
  mkdirSync(dir, { recursive: true });
  const runs = new Map();     // id -> internal run

  const pub = r => ({
    id: r.id, kind: 'fork', phase: r.phase, workspace: r.workspace, prompt: r.prompt, startedAt: r.startedAt / 1000, finishedAt: r.finishedAt ? r.finishedAt / 1000 : null,
    attempts: [], checkpoints: [], selectedCheckpoint: null, decision: null, needsUserReview: r.phase === 'completed' && !!r.b?.root,
    error: r.error, plan: r.plan, lever: r.lever, a: r.a, b: r.b, labels: { original: 'A', updated: 'B' }, activity: r.activity,
  });
  const saveFile = r => join(dir, r.id, 'run.json');
  const persist = r => { try { mkdirSync(join(dir, r.id), { recursive: true }); const tmp = saveFile(r) + '.tmp'; writeFileSync(tmp, JSON.stringify(pub(r))); renameSync(tmp, saveFile(r)); } catch { /* memory only */ } };
  let lastSent = new Map();
  const announce = r => {
    persist(r);
    const view = { id: r.id, phase: r.phase, activity: r.activity, plan: r.plan, lever: r.lever, a: r.a, b: r.b };
    const key = JSON.stringify(view);
    if (lastSent.get(r.id) !== key) { lastSent.set(r.id, key); notify('timeskip', JSON.parse(key)); }
  };
  const say = (r, code, text, args = {}) => { r.activity = { code, text, args }; announce(r); };
  // The page's older phase names keep working (queued/checkpointing/generating/checking/completed); the new fields say the rest.
  const setPhase = (r, phase) => { r.phase = phase; announce(r); };

  async function ask(r, chatId, line, ws, choice, thinking, mode = 'plan') {
    let text = '';
    await agent.runTurn({ chatId, message: line, workspacePath: ws, choice, mode, thinking, onFrame: f => { if (f.type === 'delta') text += f.text; } });
    return text;
  }

  /**
   * One plan step in its own session: tried twice if it ends early; stopped when it takes longer than `stepTimeoutMs` (a small model that keeps
   * debugging one step must not hold the whole version for an hour). A step that cannot finish (too long, or a look-in-a-browser step on a machine
   * with no screen) is left as it is with a plain note and the next step goes on; any other step that fails ends the version.
   * Returns 'done' | 'skipped' | 'failed' | 'stopped'.
   */
  async function runOneStep(r, { tag, chat, planId, plan, at, root, choice, thinking, onFrame }) {
    let finished = false, timedOut = false;
    for (let attempt = 0; attempt < 2 && !finished && !timedOut && !r.stopping; attempt++) {
      if (attempt) { agent.reopenPlan(chat); say(r, tag + '-retry', `${tag.toUpperCase()}, trying step ${at + 1} again…`, { step: at + 1 }); }
      agent.resetSession(chat);
      const turn = agent.planGoal({ chatId: chat, planId, once: true, workspacePath: root, mode: 'bypass', choice, thinking, onFrame });
      let timer;
      const raced = await Promise.race([turn.then(() => 'ended'), new Promise(res => { timer = setTimeout(() => res('timeout'), stepTimeoutMs); timer.unref?.(); })]);
      clearTimeout(timer);
      if (raced === 'timeout') { timedOut = true; try { await agent.stop(chat, { waitMs: 15_000 }); } catch { /* ending anyway */ } await turn.catch(() => {}); }
      if (r.stopping) return 'stopped';
      finished = agent.planState(chat)?.steps[at]?.state === 'done';
    }
    if (finished) return 'done';
    if (timedOut || isViewStep(plan.steps[at].text)) {
      agent.skipStep(chat, at, timedOut ? 'This step took too long, so it was left as it is.' : 'Could not look at the page here; the files were checked instead.');
      say(r, tag + '-skip', timedOut ? `${tag.toUpperCase()}, step ${at + 1} took too long and was left as it is.` : `${tag.toUpperCase()}, step ${at + 1} (viewing the page) was skipped on this machine.`, { step: at + 1 });
      return 'skipped';
    }
    return 'failed';
  }

  async function runFlow(r, { choice, thinking, slots }) {
    const aChat = `ts:${r.id}:A`, bChat = `ts:${r.id}:B`, leverChat = `ts:${r.id}:lever`;
    r.chats = [aChat, bChat, leverChat];
    const stopped = () => r.stopping;
    try {
      // 1. plan
      setPhase(r, 'checkpointing'); say(r, 'planning', 'Planning the work…');
      copyProject(r.workspace, r.a.root);
      await agent.planCommand({ chatId: aChat, line: r.prompt, workspacePath: r.a.root, choice, thinking, mode: 'plan', onFrame: f => { if (f.type === 'status' && f.text) say(r, 'planning', f.text, f.args ?? {}); } });
      if (stopped()) return;
      const plan = agent.planState(aChat);
      if (!plan || plan.steps.length < 2) throw new Error('The assistant could not split this request into steps. Describe the steps you want, or use /goal instead.');
      agent.setPlanBriefing(aChat, UNATTENDED);
      agent.planControl({ chatId: aChat, planId: plan.id, action: 'approve' });
      r.plan = { id: plan.id, title: plan.title, steps: plan.steps.map(s => ({ id: s.id, text: s.text, detail: s.detail })) };
      announce(r);
      // 2. the lever
      say(r, 'choosing-lever', 'Choosing the step that matters most…');
      let lever = null;
      if (plan.steps.length >= 3) {
        try { const said = await ask(r, leverChat, leverPrompt(r.prompt, plan.steps), r.a.root, choice, 'Low'); if (process.env.TS_DEBUG) console.log('lever said:', JSON.stringify(said).slice(0, 300)); lever = parseLever(said, plan.steps.length); } catch (e) { lever = null; if (process.env.TS_DEBUG) console.log('lever failed', e.message); }
        if (stopped()) return;
        lever ??= defaultLever(plan.steps);
      }
      r.lever = lever ? { step: lever.step, total: plan.steps.length, text: plan.steps[lever.step - 1].text, why: lever.why, defaulted: !!lever.defaulted } : null;
      if (!r.lever) { r.b = { state: 'none', note: 'This plan is too short to be worth forking: only A runs.' }; }
      announce(r);

      // 3. A runs all the way; B forks when A has finished the step before the lever
      let bDone = null;
      const doneSummaries = [];
      const startB = async () => {
        r.b.state = 'planning'; r.b.waiting = false; say(r, 'b-planning', 'B: planning the improved path…');
        await agent.planCommand({ chatId: bChat, line: improvementPrompt({ request: r.prompt, steps: plan.steps, lever, doneSummaries }), workspacePath: r.b.root, choice, thinking, mode: 'plan', onFrame: () => {} });
        if (stopped()) return;
        const bp = agent.planState(bChat);
        if (!bp || bp.steps.length < 2) { r.b.state = 'failed'; r.b.note = 'The improvement plan could not be written, so only A is shown.'; announce(r); return; }
        agent.setPlanBriefing(bChat, UNATTENDED + ' This work continues from a first attempt that is already in this folder; its notes are in .timeskip/context.md (read them first). Improve on it where it is weak instead of starting over.');
        agent.planControl({ chatId: bChat, planId: bp.id, action: 'approve' });
        r.b.plan = bp.steps.map(s => ({ id: s.id, text: s.text }));
        r.b.state = 'running'; r.b.total = bp.steps.length; announce(r);
        for (let at = 0; at < bp.steps.length && !stopped(); at++) {
          const res = await runOneStep(r, { tag: 'b', chat: bChat, planId: bp.id, plan: bp, at, root: r.b.root, choice, thinking,
            onFrame: f => {
              if (f.type === 'error') r.b.note = String(f.message ?? '').slice(0, 240);
              if (f.type === 'goal' && f.goal?.blocked) r.b.note = String(f.goal.blocked).slice(0, 240);
              if (f.type === 'plan-step') { r.b.step = f.index; r.b.total = f.total; r.b.text = f.text; r.b.line = f.line; say(r, 'b-step', `B, ${f.line}`, { step: f.index, steps: f.total }); }
            } });
          if (res === 'stopped') return;
          if (res === 'failed') break;
        }
        const end = agent.planState(bChat);
        r.b.state = end?.steps.every(x => ['done', 'skipped'].includes(x.state)) ? 'done' : 'failed';
        if (r.b.state === 'done') r.b.note = end.steps.some(x => x.state === 'skipped') ? 'A step that could not finish was left as it is.' : null;
        if (r.b.state === 'failed') r.b.note = stopped() ? 'Stopped.' : 'B did not finish, so only A is shown.';
        announce(r);
      };
      let forked = false;
      const forkNow = () => {
        if (forked || !r.lever) return;
        forked = true;
        copyProject(r.a.root, r.b.root);
        mkdirSync(join(r.b.root, '.timeskip'), { recursive: true });
        writeFileSync(join(r.b.root, '.timeskip', 'context.md'), [
          `# Work so far (copied from the first attempt, A)`, '', `Request: ${r.prompt}`, '', 'Plan:', ...plan.steps.map((s, i) => `${i + 1}. ${s.text}`), '',
          `Steps 1 to ${lever.step - 1} are done in this folder.`, ...r.a.summaries.map(s => `- ${s}`), '', `Checkpoint step ${lever.step}: ${plan.steps[lever.step - 1].text}`, `Why this step: ${lever.why}`, ''].join('\n'));
        r.b = { ...r.b, root: r.b.root, state: 'waiting', waiting: slots < 2, step: 0, total: 0 };
        doneSummaries.push(...r.a.summaries);
        announce(r);
        if (slots >= 2) bDone = startB();
      };
      r.b ??= {};
      if (r.lever) { r.b = { state: 'pending', root: join(dir, r.id, 'B'), waiting: slots < 2 }; }
      r.a.state = 'running'; r.a.total = plan.steps.length; setPhase(r, 'generating'); announce(r);
      // Each step is its own turn in its own agent session (the folder carries the work from step to step): a small model window never fills up
      // across a long plan. The fork happens between two steps, so B gets exactly the files of the finished steps.
      for (let at = 0; at < plan.steps.length && !stopped(); at++) {
        const res = await runOneStep(r, { tag: 'a', chat: aChat, planId: plan.id, plan, at, root: r.a.root, choice, thinking,
          onFrame: f => {
            if (f.type === 'error') r.a.note = String(f.message ?? '').slice(0, 240);
            if (f.type === 'goal' && f.goal) r.a.goal = { phase: f.goal.phase, blocked: f.goal.blocked ?? null, round: f.goal.round };
            if (f.type === 'plan-step') {
              r.a.step = f.index; r.a.total = f.total; r.a.text = f.text; r.a.line = f.line;
              if (f.state === 'done') r.a.summaries.push(`${f.index}. ${f.text}${f.summary ? ': ' + f.summary : ''}`);
              say(r, 'a-step', `A, ${f.line}`, { step: f.index, steps: f.total });
            }
          } });
        if (res === 'stopped') return;
        if (res === 'failed') break;
        if (r.lever && at + 1 === r.lever.step - 1) { try { forkNow(); } catch (e) { r.b = { ...r.b, state: 'failed', note: 'B could not be prepared, so only A is shown.' }; } }
      }
      if (stopped()) return;
      const aEnd = agent.planState(aChat);
      r.a.state = aEnd?.steps.every(x => ['done', 'skipped'].includes(x.state)) ? 'done' : 'failed';
      announce(r);
      if (r.a.state !== 'done') throw new Error('A did not finish the plan' + (r.a.note ? ' (' + r.a.note + ')' : r.a.goal?.blocked ? ' (' + r.a.goal.blocked + ')' : '') + ', so there is nothing to compare. Your files were not changed.');
      if (r.lever && !forked) { forkNow(); }
      if (r.lever && slots < 2 && r.b.state === 'waiting') { say(r, 'b-waiting', 'A is done. B starts now (one model slot: B waited for A).'); bDone = startB(); }
      if (bDone) await bDone.catch(e => { r.b.state = 'failed'; r.b.note = 'B stopped: ' + String(e.message ?? e).slice(0, 160); });
      if (stopped()) return;
      setPhase(r, 'checking'); say(r, 'comparing', 'Comparing A and B…');
      r.finishedAt = now();
      r.activity = r.b?.state === 'done' ? { code: 'ready', text: 'Both versions are ready. Compare A and B, then keep the one you like.', args: {} } : { code: 'ready', text: 'The result is ready.' + (r.b?.note ? ' ' + r.b.note : ''), args: {} };
      setPhase(r, 'completed');
    } catch (e) {
      if (stopped()) return;
      r.error = String(e.message ?? e).slice(0, 400); r.finishedAt = now(); r.activity = null; setPhase(r, 'failed');
    } finally {
      for (const c of r.chats) { try { agent.forget?.(c); } catch { /* best effort */ } }
    }
  }

  return {
    /** Start a run; resolves at once with the first snapshot. */
    start({ prompt, workspace, chatId = null, choice, thinking = 'Low', slots = 1 }) {
      const id = randomUUID().replaceAll('-', '');
      const r = { id, chatId, prompt: String(prompt ?? '').trim().slice(0, 4000), workspace: resolve(workspace), phase: 'queued', startedAt: now(), finishedAt: null, error: null, plan: null, lever: null, b: null, activity: { code: 'planning', text: 'Planning the work…', args: {} }, stopping: false,
        a: { state: 'pending', root: join(dir, id, 'A'), step: 0, total: 0, summaries: [] }, chats: [] };
      runs.set(id, r);
      announce(r);
      r.done = runFlow(r, { choice, thinking, slots });
      return pub(r);
    },
    has: id => runs.has(String(id)),
    status: id => { const r = runs.get(String(id)); return r ? pub(r) : null; },
    list: () => [...runs.values()].map(pub),
    byChat: chatId => [...runs.values()].find(r => r.chatId === chatId && !TERMINAL.has(r.phase))?.id ?? null,
    async stop(id) {
      const r = runs.get(String(id)); if (!r) return false;
      if (TERMINAL.has(r.phase)) return false;
      r.stopping = true;
      await Promise.allSettled((r.chats ?? []).map(c => agent.stop(c, { waitMs: 8000 })));
      r.finishedAt = now(); r.activity = null; r.phase = 'cancelled'; if (r.a) r.a.state = r.a.state === 'done' ? 'done' : 'stopped'; if (r.b?.state && r.b.state !== 'done') r.b.state = 'stopped';
      announce(r);
      try { await r.done; } catch { /* ended */ }
      try { rmSync(join(dir, r.id), { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ }
      return true;
    },
    /** A against B (or the original against A when there was no fork). */
    files(id) {
      const r = runs.get(String(id)); if (!r || !TERMINAL.has(r.phase) || r.phase === 'cancelled') return null;
      return diffRun(recordOf(r));
    },
    image(id, rel, version) { const r = runs.get(String(id)); return r ? imageOf(r, rel, version) : null; },
    /** Keep A or B: copy what differs into the person's folder, keeping the replaced files for "put the original back". */
    apply({ id, which = 'b', restore = false }) {
      const r = runs.get(String(id));
      if (!r) throw new Error('No such Time-skip run.');
      const back = join(dir, r.id, 'rollback');
      if (restore) {
        if (!existsSync(back)) throw new Error('There is nothing to put back.');
        cpSync(back, r.workspace, { recursive: true, force: true });
        for (const rel of readIndex(back).created) { try { rmSync(join(r.workspace, rel), { force: true }); } catch { /* gone */ } }
        rmSync(back, { recursive: true, force: true });
        return { restored: true };
      }
      if (r.phase !== 'completed') throw new Error('This run has not finished.');
      const from = which === 'a' ? r.a.root : (r.b?.root && r.b.state === 'done' ? r.b.root : null);
      if (!from) throw new Error('B is not available: only A can be kept.');
      if (existsSync(back)) throw new Error('A result is already in your files. Put the original back first.');
      mkdirSync(back, { recursive: true });
      const created = []; let copied = 0;
      for (const rel of walk(from)) {
        if (isNoisePath(rel)) continue;
        const src = join(from, rel), dst = join(r.workspace, rel);
        if (existsSync(dst) && Buffer.compare(readFileSync(src), readFileSync(dst)) === 0) continue;
        if (existsSync(dst)) { mkdirSync(dirname(join(back, rel)), { recursive: true }); cpSync(dst, join(back, rel)); } else created.push(rel);
        mkdirSync(dirname(dst), { recursive: true }); cpSync(src, dst); copied += 1;
      }
      writeFileSync(join(back, '.index.json'), JSON.stringify({ created }));
      return { applied: which, files: copied };
    },
  };

  function recordOf(r) {
    if (r.b?.root && r.b.state === 'done') return { checkpoints: [{ id: 'cp-a', label: 'baseline', path: r.a.root }, { id: 'cp-b', label: 'b', path: r.b.root }], selected_checkpoint: 'cp-b', attempts: [] };
    return { checkpoints: [{ id: 'cp-orig', label: 'baseline', path: r.workspace }], selected_checkpoint: 'cp-orig', attempts: [{ index: 1, path: r.a.root }] };
  }
  function imageOf(r, rel, version) { return runImagePath(recordOf(r), rel, version); }
}

function* walkGen(root, base = root) {
  let entries; try { entries = readdirSync(root, { withFileTypes: true }); } catch { return; }
  for (const e of entries) {
    const full = join(root, e.name);
    if (e.isDirectory()) { if (!SKIP.has(e.name)) yield* walkGen(full, base); }
    else if (e.isFile()) yield relative(base, full).split(sep).join('/');
  }
}
const walk = root => [...walkGen(root)];
function readIndex(back) { try { return JSON.parse(readFileSync(join(back, '.index.json'), 'utf8')); } catch { return { created: [] }; } }
void stepObjectiveBriefing; void statSync;
