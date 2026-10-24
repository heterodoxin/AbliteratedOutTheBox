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
async function stream(body, { quiet = false } = {}) {
  const res = await fetch(BASE + '/api/chat/stream', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
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
        if (f.type === 'delta') { text += (f.text ?? '').length; continue; }
        if (f.type === 'reasoning') { reasoning += (f.text ?? '').length; continue; }
        if (f.type === 'stats') continue;
        flushCounters();
        let out = f.type;
        if (f.type === 'status') out = `status[${f.seq}] ${f.phase}/${f.code}: ${f.text}${f.tokens !== undefined ? ` (tokens ${f.tokens})` : ''} +${f.elapsedMs}ms`;
        else if (f.type === 'goal') out = `goal ${JSON.stringify(f.goal)}`;
        else if (f.type === 'agent-tool') out = `tool ${f.tool.state} ${f.tool.verb} ${f.tool.target ?? ''}`;
        else if (f.type === 'done') out = `done ${f.finishReason} tokensOut=${f.tokensOut}`;
        else if (f.type === 'error') out = `ERROR ${f.message}`;
        if (!quiet && out !== lastLine) console.log(`${stamp()}  ${out}`);
        lastLine = out;
      }
    }
    if (done) break;
  }
  flushCounters();
  return frames;
}

if (command === 'connect') {
  const [baseUrl, modelId] = rest;
  await j('/api/model-providers', { kind: 'custom', name: 'Cloud model', baseUrl, protocol: 'openai-chat', apiKey: '', models: [{ id: modelId, name: 'Test model' }] });
  console.log('saved');
} else if (command === 'stream') {
  const [message, mode = 'bypass'] = rest;
  const chat = process.env.CHAT ?? `c-${Date.now()}`;
  await stream({ sessionId: chat, message, mode, remoteModel: await model(), workspaceId: process.env.WORKSPACE_ID, ...(process.env.PLAIN ? { agent: false } : {}), ...(process.env.THINK ? { thinking: process.env.THINK } : {}) });
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
    const line = `${run.phase}${run.activity ? ` | ${run.activity.text}${run.activity.args?.step ? ` | assistant: ${run.activity.args.step}` : ''}` : ''}`;
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
} else {
  console.error('commands: connect | stream | goal | timeskip'); process.exit(2);
}
