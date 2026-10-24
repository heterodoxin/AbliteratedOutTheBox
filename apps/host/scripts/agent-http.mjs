/**
 * Drive the host over HTTP (no browser) against a real model and print what the page would receive.
 *
 *   STRATA_URL=http://127.0.0.1:8899 node scripts/agent3-http.mjs <command> [args]
 *
 *   connect <baseUrl> <modelId>        save an API connection to a model server (once per STRATA_HOME)
 *   stream "<message>" [mode]          one chat turn; prints every frame compactly with its time (status, goal, tools, deltas counted)
 *   goal "<objective>"                 /goal run, then the goal bar calls: state, pause, edit, clear, resume
 *   timeskip "<request>" [stopAfterS]  a Time-skip job; polls status (phase + activity); with stopAfterS, presses Stop that many seconds in
 *
 * Environment: STRATA_URL (required), CHAT (chat id, default a fresh one), WORKSPACE_ID.
 */
const BASE = process.env.STRATA_URL;
if (!BASE) { console.error('STRATA_URL is required'); process.exit(2); }
const [command, ...rest] = process.argv.slice(2);
const t0 = Date.now();
const stamp = () => `${String(Math.round((Date.now() - t0) / 1000)).padStart(4)}s`;
const j = async (path, body, method = body ? 'POST' : 'GET') => {
  const res = await fetch(BASE + path, { method, headers: { 'content-type': 'application/json' }, ...(body ? { body: JSON.stringify(body) } : {}) });
  const text = await res.text();
  let data; try { data = JSON.parse(text); } catch { data = { raw: text }; }
  if (!res.ok) throw new Error(`${path} -> ${res.status} ${data.error ?? text.slice(0, 200)}`);
  return data;
};
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function model() {
  const providers = await j('/api/model-providers');
  const c = providers.connections.at(-1);
  if (!c) throw new Error('no API connection saved: run "connect" first');
  return { connectionId: c.id, modelId: c.models[0].id };
}

/** One chat stream; returns the frames. `print` shows a compact line per frame, deltas and reasoning as counters. */
async function stream(body, { quiet = false, path = '/api/chat/stream', onFrame = null } = {}) {
  const res = await fetch(BASE + path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`stream -> ${res.status} ${await res.text()}`);
  const frames = []; let buffer = '', text = 0, reasoning = 0, lastLine = '';
  const flushCounters = () => { if (text || reasoning) { console.log(`${stamp()}  ... ${reasoning} reasoning chars, ${text} answer chars`); text = 0; reasoning = 0; } };
  const reader = res.body.getReader(); const dec = new TextDecoder();
  for (;;) {
    const { value, done } = await reader.read();
    if (value) {
      buffer += dec.decode(value, { stream: true });
      const lines = buffer.split('\n'); buffer = lines.pop() ?? '';
      for (const line of lines.filter(Boolean)) {
        const f = JSON.parse(line); f._at = Date.now(); frames.push(f);
        try { onFrame?.(f); } catch (e) { console.log('onFrame error', e.message); }
        if (f.type === 'delta') { text += (f.text ?? '').length; continue; }
        if (f.type === 'reasoning') { reasoning += (f.text ?? '').length; continue; }
        if (f.type === 'stats') continue;
        flushCounters();
        let out = f.type;
        if (f.type === 'status') out = `status[${f.seq}] ${f.phase}/${f.code}: ${f.text}${f.tokens !== undefined ? ` (tokens ${f.tokens})` : ''} +${f.elapsedMs}ms`;
        else if (f.type === 'goal') out = `goal ${JSON.stringify(f.goal)}`;
        else if (f.type === 'agent-tool') {
          out = `tool ${f.tool.state} ${f.tool.verb} ${f.tool.target ?? ''}${f.tool.state === 'proposed' ? `  [asks: ${f.tool.effect}]` : ''}`;
          // ANSWER=approve|deny answers every card that asks (the card is the real Approve/Deny card of Manual mode)
          if (f.tool.state === 'proposed' && process.env.ANSWER) void j('/api/chat/approval', { sessionId: body.sessionId, toolId: f.tool.id, decision: process.env.APPROVE_MATCH && new RegExp(process.env.APPROVE_MATCH, 'i').test(`${f.tool.target} ${f.tool.effect}`) ? 'approve' : process.env.ANSWER }).then(() => console.log(`${stamp()}  -> ${process.env.ANSWER} ${f.tool.id}`), e => console.log(`${stamp()}  answer failed: ${e.message}`));
        }
        else if (f.type === 'plan') out = `PLAN ${f.event} ${f.plan.state} "${f.plan.title}" steps=${f.plan.steps.map(x => x.state[0] + ':' + x.text.slice(0, 40)).join(' | ')}`;
        else if (f.type === 'plan-step') out = `PLAN-STEP ${f.line}${f.summary ? ' | summary: ' + f.summary : ''}`;
        else if (f.type === 'queued-message') out = `QUEUED-MESSAGE ${f.state} "${f.message.text.slice(0, 50)}" deliverAt=${f.message.deliverAt}${f.afterStep ? ' afterStep=' + f.afterStep : ''} queue=${f.queue.length}`;
        else if (f.type === 'queued-release') out = `QUEUED-RELEASE ${f.messages.map(m => m.text.slice(0, 40)).join(' | ')}`;
        else if (f.type === 'research') out = `RESEARCH sources=${f.sources.map(x => x.n + ':' + x.url).join(' ')}`;
        else if (f.type === 'done') out = `done ${f.finishReason} tokensOut=${f.tokensOut}`;
        else if (f.type === 'error') out = `ERROR ${f.message}`;
        if (!quiet && out !== lastLine) console.log(`${stamp()}  ${out}`);
        lastLine = out;
      }
    }
    if (done) break;
  }
  flushCounters();
  if (process.env.SHOWTEXT && !quiet) console.log(['---- answer text ----', frames.filter(f => f.type === 'delta').map(f => f.text).join(''), '---- end ----'].join(String.fromCharCode(10)));
  return frames;
}

if (command === 'connect') {
  const [baseUrl, modelId] = rest;
  await j('/api/model-providers', { kind: 'custom', name: 'Cloud model', baseUrl, protocol: 'openai-chat', apiKey: '', models: [{ id: modelId, name: 'Test model', ...(process.env.CTX ? { contextMax: Number(process.env.CTX) } : {}), ...(process.env.MAXOUT ? { maxOutput: Number(process.env.MAXOUT) } : {}) }] });
  console.log('saved');
} else if (command === 'stream') {
  const [message, mode = 'bypass'] = rest;
  const chat = process.env.CHAT ?? `c-${Date.now()}`;
  await stream({ sessionId: chat, message, mode, remoteModel: await model(), workspaceId: process.env.WORKSPACE_ID, ...(process.env.PLAIN ? { agent: false } : {}), ...(process.env.THINK ? { thinking: process.env.THINK } : {}) });
} else if (command === 'skill') {
  // create a skill through the Skills page route: skill "Name" "description" "instructions"
  const [name, description, instructions] = rest;
  const out = await j('/api/skills', { name, description, body: instructions });
  console.log('saved', out.id, '| skills now:', out.skills.map(s => `${s.id}(${s.source})`).join(', '));
} else if (command === 'goal') {
  const [objective] = rest;
  const chat = process.env.CHAT ?? `g-${Date.now()}`;
  const common = { sessionId: chat, mode: 'bypass', remoteModel: await model() };
  console.log(`${stamp()}  -- /goal run`);
  const frames = await stream({ ...common, message: `/goal ${objective}` });
  const last = frames.filter(f => f.type === 'goal').at(-1);
  console.log(`${stamp()}  -- goal bar: state`, JSON.stringify((await j(`/api/goal?chatId=${chat}`)).goal));
  console.log(`${stamp()}  -- pause`, JSON.stringify(await j('/api/goal/pause', { chatId: chat })));
  console.log(`${stamp()}  -- edit`, JSON.stringify(await j('/api/goal/edit', { chatId: chat, text: `${objective} (edited)` })));
  console.log(`${stamp()}  -- resume (stream)`);
  await stream({ ...common, goalAction: 'resume' });
  console.log(`${stamp()}  -- clear`, JSON.stringify(await j('/api/goal/clear', { chatId: chat })));
  console.log(`${stamp()}  -- state after clear`, JSON.stringify(await j(`/api/goal?chatId=${chat}`)));
  void last;
} else if (command === 'goalbar') {
  // Pause from the goal bar while a round runs, then resume through the chat stream.
  const [objective, pauseAfter = '6'] = rest;
  const chat = process.env.CHAT ?? `gb-${Date.now()}`;
  const common = { sessionId: chat, mode: 'bypass', remoteModel: await model() };
  const running = stream({ ...common, message: `/goal ${objective}` });
  await sleep(Number(pauseAfter) * 1000);
  console.log(`${stamp()}  -- pause from the bar`, JSON.stringify(await j('/api/goal/pause', { chatId: chat })));
  await running;
  console.log(`${stamp()}  -- state after the run`, JSON.stringify((await j(`/api/goal?chatId=${chat}`)).goal));
  console.log(`${stamp()}  -- resume (stream, no message)`);
  await stream({ ...common, goalAction: 'resume' });
  console.log(`${stamp()}  -- state after resume`, JSON.stringify((await j(`/api/goal?chatId=${chat}`)).goal));
  console.log(`${stamp()}  -- remove`, JSON.stringify(await j('/api/goal/clear', { chatId: chat })));
} else if (command === 'timeskip') {
  const [request, stopAfter] = rest;
  const chat = process.env.CHAT ?? `t-${Date.now()}`;
  const started = await j('/api/timeskip/start', { prompt: request, sessionId: chat, attempts: 1, workspaceId: process.env.WORKSPACE_ID, remoteModel: await model() });
  const id = started.run.id;
  console.log(`${stamp()}  started ${id} activity=${JSON.stringify(started.run.activity)}`);
  let last = '', stopped = false;
  for (;;) {
    const { run } = await j(`/api/timeskip/status?id=${id}`);
    const line = `${run.phase}${run.activity ? ` | ${run.activity.text}${run.activity.args?.step && typeof run.activity.args.step === 'string' ? ` | assistant: ${run.activity.args.step}` : ''}` : ''}${run.lever ? ` | lever: step ${run.lever.step}/${run.lever.total} ${run.lever.defaulted ? '(default) ' : ''}- ${run.lever.why}` : ''}${run.a?.line ? ` | A: ${run.a.state} ${run.a.line.slice(0, 50)}` : ''}${run.b?.state ? ` | B: ${run.b.state} ${(run.b.line ?? '').slice(0, 50)}` : ''}`;
    if (line !== last) { console.log(`${stamp()}  ${line}`); last = line; }
    if (stopAfter && !stopped && (Date.now() - t0) / 1000 >= Number(stopAfter)) {
      console.log(`${stamp()}  -- pressing Stop (the chat Stop route)`);
      console.log(`${stamp()}  `, JSON.stringify(await j('/api/chat/stop', { sessionId: chat })));
      stopped = true;
    }
    if (['completed', 'cancelled', 'failed', 'interrupted'].includes(run.phase)) {
      console.log(`${stamp()}  END ${run.phase}${run.error ? ` error=${run.error}` : ''} needsReview=${run.needsUserReview}`);
      try { const files = await j(`/api/timeskip/files?id=${id}`); console.log(`${stamp()}  files: ${files.files.map(f => `${f.status} ${f.path}`).join(', ') || '(none)'}`); } catch (e) { console.log(`${stamp()}  files: ${e.message}`); }
      break;
    }
    await sleep(4000);
  }
} else if (command === 'plan') {
  // /plan <task>, then (unless STOP_AT_PLAN) Start as goal with an optional queued message: plan "<task>" ["<queued message>"]
  const [task, queued] = rest;
  const chat = process.env.CHAT ?? `pl-${Date.now()}`;
  const common = { sessionId: chat, mode: process.env.MODE ?? 'manual', remoteModel: await model(), workspaceId: process.env.WORKSPACE_ID };
  console.log(`${stamp()}  -- /plan`);
  const frames = await stream({ ...common, message: `/plan ${task}`});
  const plan = frames.find(f => f.type === 'plan')?.plan;
  if (!plan) { console.log('NO PLAN FRAME (the reply stays plain text)'); process.exit(1); }
  console.log(JSON.stringify(plan.steps.map(x => x.text)));
  console.log(`${stamp()}  -- GET /api/plan`, (await j(`/api/plan?sessionId=${chat}`)).plan?.state);
  if (process.env.EDIT_LAST) { const steps = plan.steps.map(x => x.text); steps.push(process.env.EDIT_LAST); console.log(`${stamp()}  -- edit`, (await j('/api/plan/edit', { sessionId: chat, planId: plan.id, steps })).plan.steps.length, 'steps'); }
  if (process.env.STOP_AT_PLAN) { console.log(`${stamp()}  -- ${process.env.STOP_AT_PLAN}`, JSON.stringify((await j('/api/plan/' + process.env.STOP_AT_PLAN, { sessionId: chat, planId: plan.id })).plan.state)); process.exit(0); }
  console.log(`${stamp()}  -- start as goal`);
  let sent = false;
  const run = stream({ ...common, planId: plan.id }, { path: '/api/plan/start-goal', onFrame: f => {
    if (queued && !sent && f.type === 'plan-step' && f.state === 'running') { sent = true; setTimeout(async () => { try { console.log(`${stamp()}  -- queue`, JSON.stringify((await j('/api/chat/queue', { sessionId: chat, message: queued })).message)); } catch (e) { console.log('queue failed', e.message); } }, Number(process.env.QUEUE_AFTER ?? 8) * 1000); }
  } });
  await run;
  console.log(`${stamp()}  -- final plan`, JSON.stringify((await j(`/api/plan?sessionId=${chat}`)).plan?.steps.map(x => x.state + ':' + x.text.slice(0, 30))));
} else if (command === 'modeswitch') {
  // Manual run that needs several approvals; after the first card the chat is switched to Bypass: no further card may ask.
  const [message] = rest;
  const chat = process.env.CHAT ?? `ms-${Date.now()}`;
  const common = { sessionId: chat, mode: 'manual', remoteModel: await model(), workspaceId: process.env.WORKSPACE_ID };
  let asked = 0, switched = false;
  const frames = await stream({ ...common, message }, { onFrame: async f => {
    if (f.type === 'agent-tool' && f.tool.state === 'proposed') {
      asked++;
      if (!switched) { switched = true; console.log(`${stamp()}  -- switching to Bypass`, JSON.stringify(await j('/api/chat/mode', { sessionId: chat, mode: 'bypass' }))); }
    }
  } });
  const done = frames.filter(f => f.type === 'agent-tool' && f.tool.state === 'done').length;
  console.log(`${stamp()}  RESULT cards that asked=${asked} (expected 1), tools done=${done}`);
} else if (command === 'long') {
  // Fill the context with long pastes so the Harness has to compact, then ask about the first one: long <rounds> <chars>
  const [rounds = '6', chars = '16000'] = rest;
  const chat = process.env.CHAT ?? `lg-${Date.now()}`;
  const common = { sessionId: chat, mode: 'bypass', remoteModel: await model(), workspaceId: process.env.WORKSPACE_ID };
  const filler = n => Array.from({ length: Math.ceil(Number(chars) / 60) }, (_, i) => `Entry ${n}-${i}: the quick brown fox number ${i * 7 + n} jumps over the lazy dog.`).join('\n');
  let frames = await stream({ ...common, message: 'Remember this code word for later: PAGODA-7731. Reply with just OK.' }, { quiet: true });
  console.log(`${stamp()}  round 0 done, ctx=${frames.at(-1).contextTokens ?? "?"}`);
  for (let n = 1; n <= Number(rounds); n++) {
    frames = await stream({ ...common, message: `Here is log chunk ${n}. Reply with just the word OK.

${filler(n)}`}, { quiet: true });
    const d = frames.at(-1);
    console.log(`${stamp()}  round ${n} ${d.type} ${d.finishReason ?? d.message ?? ""} tokensIn=${d.tokensIn} ctx=${d.contextTokens ?? d.context ?? "?"}`);
  }
  frames = await stream({ ...common, message: 'What was the code word I asked you to remember at the very start? Answer with the code word only.' }, { quiet: true });
  console.log(`${stamp()}  ANSWER: ${frames.filter(f => f.type === "delta").map(f => f.text).join("").trim().slice(0, 120)}`);
} else if (command === 'effort') {
  // The same question at each reasoning level (and optional thinking budget): what really changes is the thinking the model did.
  const [question = 'What is 17 times 23, and is the result divisible by 3? Explain briefly.', budget = '0'] = rest;
  const rows = [];
  for (const level of (process.env.LEVELS ? process.env.LEVELS.split(',') : ['Off', 'Low', 'Medium', 'High', 'Max'])) {
    const t = Date.now();
    const frames = await stream({ sessionId: `ef-${level}-${Date.now()}`, message: question, mode: 'bypass', remoteModel: await model(), thinking: level, thinkingBudget: Number(budget), agent: process.env.PLAIN ? false : undefined }, { quiet: true });
    const reasoning = frames.filter(x => x.type === 'reasoning').map(x => x.text ?? '').join('').length;
    const answer = frames.filter(x => x.type === 'delta').map(x => x.text ?? '').join('');
    const done = frames.at(-1);
    rows.push({ level, budget: Number(budget), reasoningChars: reasoning, answerChars: answer.length, tokensOut: done.tokensOut, seconds: Math.round((Date.now() - t) / 1000), answer: answer.replace(/\s+/g, ' ').slice(0, 90) });
    console.log(JSON.stringify(rows.at(-1)));
  }
} else if (command === 'sampling') {
  // Set Expert sampling through the engine settings route (as the Response settings dialog does), then ask for a random word a few times.
  const [temp = '0', seed = '1', n = '3'] = rest;
  const before = await j('/api/engine');
  const set = async values => j('/api/engine', { expert: Object.entries(values).map(([id, value]) => ({ id, value })) }, 'PATCH');
  await set({ 'sample.temp': Number(temp), 'sample.seed': Number(seed) });
  const words = [];
  for (let i = 0; i < Number(n); i++) {
    const frames = await stream({ sessionId: `sm-${Date.now()}-${i}`, message: 'Reply with one random everyday English word and nothing else.', mode: 'bypass', remoteModel: await model(), thinking: 'Off' }, { quiet: true });
    words.push(frames.filter(x => x.type === 'delta').map(x => x.text ?? '').join('').trim().slice(0, 30));
  }
  console.log(JSON.stringify({ temp: Number(temp), seed: Number(seed), words }));
  await set({ 'sample.temp': 0.6, 'sample.seed': -1 });
} else {
  console.error('commands: connect | stream | goal | timeskip | plan | modeswitch | long | effort | sampling'); process.exit(2);
}
