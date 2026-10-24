/**
 * A scripted OpenAI-compatible chat-completions worker, for the agent-bridge tests.
 *
 * It stands in for the Strata API worker so that the Harness agent loop, the tool executor, the
 * approval gate and the host bridge are all real while the model is scripted. A scripted model proves
 * the integration path, not model quality; the real-model run is a separate step.
 *
 * Script selection is by a directive in the last user message (`SCRIPT:<name> <args>`), never by
 * request count: the Harness makes requests a script author does not plan for (a session title, a
 * re-ask after a tool result), and a counter would shift every later reply.
 *
 *   SCRIPT:hello                    reasoning then a short text reply
 *   SCRIPT:slow [n]                 n (default 400) text chunks, 40 ms apart: a window for Stop
 *   SCRIPT:tool <file> <text>       one shell call that appends <text> to <file>; it asks the sandbox for
 *                                   wider access, so the approval gate is exercised
 *   SCRIPT:plain <file> <text>      the same call without the escalation (no approval needed)
 *   SCRIPT:echo <text>              replies with <text>
 *   SCRIPT:ask                      one ask_user_question call (the agent asking the person something)
 *   SCRIPT:envdump <file>           one shell call that writes the shell's environment to <file>
 *   SCRIPT:retrywrite <file> <text> the `write` tool, like a careful model: when the sandbox denies it and offers
 *                                   an escalation, retry once with `sandbox_permissions: workspace-write`
 *   SCRIPT:retryshell <file> <text> the same with the shell tool (an in-workspace append)
 *   SCRIPT:read <file>              the `read` tool on <file>
 *   SCRIPT:spawn <n> <script...>    n `subagent` calls in one reply; each child's prompt is SCRIPT:<script...>. When a child's
 *                                   completion notice wakes the parent, it answers with a short text quoting the notice.
 *   SCRIPT:goalstep                 one goal round: get_goal, then update_goal complete, then a short closing text
 *   SCRIPT:planreply                a reply ending with a numbered plan of three steps (each step's text carries SCRIPT:goalstep)
 *   SCRIPT:call <tool> <json>       one call to any tool with the given JSON arguments
 *   SCRIPT:multi <json>             {"calls":[[tool,args],...],"say":"final text"}: all the calls in one reply, then the text once their results are in
 *   SCRIPT:searchloop               a model that never stops: every reply rewrites a numbered plan and calls web_search again with a new query
 *   SCRIPT:bloat <prompt tokens>    a reply that is only thinking (no answer text), with usage reporting that many prompt tokens (a context at 100%)
 *
 * A request whose last message is a tool result is answered with a text that quotes the result.
 */
import { createServer } from 'node:http';
import { createHash } from 'node:crypto';

const delay = ms => new Promise(r => setTimeout(r, ms));

export async function startScriptedWorker({ port = 0, chunkDelayMs = 40, model = 'scripted-model', serial = false, window = 0, charsPerToken = 0, noCount = false } = {}) {
  // `window` + `charsPerToken` make it behave like the engine: every request's prompt is counted (its size over charsPerToken), the usage it
  // reports is that count, and a prompt that leaves no room to answer is refused with the engine's own 400 ("requests are never truncated").
  // `serial` stands in for an engine with one request slot: the next request waits until the one before it has finished.
  let tail = Promise.resolve();
  const turn = () => { let release; const mine = new Promise(r => { release = r; }); const wait = tail; tail = tail.then(() => mine); return wait.then(() => release); };
  const requests = [];
  const counts = [];
  let aborted = 0;
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, 'http://x');
    const json = (code, body) => { res.writeHead(code, { 'content-type': 'application/json' }); res.end(JSON.stringify(body)); };
    if (req.method === 'GET' && url.pathname === '/health') return json(200, { ok: true });
    if (req.method === 'GET' && url.pathname === '/v1/models') return json(200, { object: 'list', data: [{ id: model, object: 'model' }] });
    if (req.method === 'GET' && url.pathname === '/__requests') return json(200, { requests, aborted });
    // The engine's own token count of a message (Anthropic style): characters over the density, plus the template's few tokens.
    if (req.method === 'POST' && url.pathname.endsWith('/messages/count_tokens')) {
      if (noCount) return json(404, { error: 'no count here' });   // an older engine: the message cannot be counted before it is sent
      let text = '';
      for await (const c of req) text += c;
      const sent = JSON.parse(text);
      const chars = (sent.messages ?? []).reduce((n, m) => n + (typeof m.content === 'string' ? m.content.length : JSON.stringify(m.content).length), 0);
      counts.push(chars);
      return json(200, { input_tokens: Math.ceil(chars / (charsPerToken || 4)) + 12 });
    }
    if (req.method !== 'POST' || !url.pathname.endsWith('/chat/completions')) return json(404, { error: 'no route' });
    const release = serial ? await turn() : () => {};
    res.on('close', release);
    let raw = '';
    for await (const c of req) raw += c;
    const body = JSON.parse(raw);
    const messages = body.messages ?? [];
    const last = messages[messages.length - 1] ?? {};
    const text = m => typeof m?.content === 'string' ? m.content : Array.isArray(m?.content) ? m.content.map(p => p.text ?? '').join(' ') : '';
    // The Harness appends runtime-context messages after the prompt, so the newest user message is not
    // the prompt: take the newest one that carries a directive.
    const directive = [...messages].reverse().filter(m => m.role === 'user')
      .map(m => /SCRIPT:(\w+)([^\n]*)/.exec(text(m))).find(Boolean);
    requests.push({
      at: Date.now(), model: body.model, stream: body.stream === true,
      toolNames: (body.tools ?? []).map(t => t.function?.name),
      // everything the client sent besides the conversation (sampling, limits, stream options)
      params: Object.fromEntries(Object.entries(body).filter(([k]) => !['messages', 'tools', 'model'].includes(k))),
      lastRole: last.role, directive: directive?.[1] ?? null, messageCount: messages.length,
      authorization: req.headers.authorization ? 'present' : 'absent',
      authSha: req.headers.authorization ? createHash('sha256').update(req.headers.authorization).digest('hex') : null,
      promptChars: raw.length, raw,
    });
    const counted = charsPerToken > 0 ? Math.ceil(raw.length / charsPerToken) : 0;
    if (window > 0 && counted > 0 && counted > window - 64) {
      requests.at(-1).refused = counted;
      return json(400, { error: { type: 'invalid_request_error', message: `request exceeds the context window: prompt (${counted} tokens) leaves no room to answer in the context (${window}); requests are never truncated` } });
    }
    const shellName = () => (body.tools ?? []).find(t => ['bash', 'pwsh'].includes(t.function?.name))?.function?.name ?? 'bash';
    const shellCommand = (file, value) => (shellName() === 'pwsh' ? `Add-Content -Path '${file}' -Value '${value}'` : `echo ${value} >> ${file}`);
    let closed = false;
    res.on('close', () => { if (!res.writableEnded) { closed = true; aborted++; } });

    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
    const id = `chatcmpl-${requests.length}`;
    const send = (delta, finish = null, extra = {}) => res.write(`data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created: 0, model: body.model, choices: [{ index: 0, delta, finish_reason: finish }], ...extra })}\n\n`);
    let promptTokens = counted || 100 + messages.length;
    const finish = (reason, tokens = 10) => {
      send({}, reason);
      res.write(`data: ${JSON.stringify({ id, object: 'chat.completion.chunk', created: 0, model: body.model, choices: [], usage: { prompt_tokens: promptTokens, completion_tokens: tokens, total_tokens: promptTokens + tokens } })}\n\n`);
      res.write('data: [DONE]\n\n'); res.end();
    };
    const toolCall = async (name, call) => {
      send({ role: 'assistant', content: '', reasoning_content: `I will use ${name}.` });
      await delay(chunkDelayMs);
      const argsJson = JSON.stringify(call);
      send({ tool_calls: [{ index: 0, id: `call_${requests.length}`, type: 'function', function: { name, arguments: '' } }] });
      for (let i = 0; i < argsJson.length; i += 17) { send({ tool_calls: [{ index: 0, function: { arguments: argsJson.slice(i, i + 17) } }] }); await delay(5); }
      return finish('tool_calls', 30);
    };
    const say = async (parts, tokens) => {
      send({ role: 'assistant', content: '' });
      for (const p of parts) { if (closed) return; send({ content: p }); await delay(chunkDelayMs); }
      if (!closed) finish('stop', tokens ?? parts.length);
    };

    // A model that read "[sandbox: escalation available ...]" retries the exact call once with the narrowest wider mode.
    // Only this prompt's own tool results count: a chat that has run tools before carries older ones.
    const promptAt = messages.findLastIndex(m => m.role === 'user' && /SCRIPT:/.test(text(m)));
    const toolMessages = messages.slice(promptAt + 1).filter(m => m.role === 'tool');
    if (last.role === 'tool' && ['retrywrite', 'retryshell'].includes(directive?.[1]) && toolMessages.length === 1 && /escalation available/.test(text(last))) {
      const [file, ...rest] = (directive[2] ?? '').trim().split(/\s+/);
      const value = rest.join(' ') || 'run';
      return toolCall(directive[1] === 'retrywrite' ? 'write' : shellName(), directive[1] === 'retrywrite'
        ? { file_path: file, content: `${value}
`, sandbox_permissions: 'workspace-write', justification: 'The user asked for this file to be written.' }
        : { command: shellCommand(file, value), description: `Append one line to ${file}`, sandbox_permissions: 'workspace-write', justification: 'The user asked for this file to be written.' });
    }
    // Time-skip's two own questions (they quote the plan, whose steps carry directives, so they are recognised by their words first).
    {
      const t = messages.filter(m => m.role === 'user').map(text).join(' ');
      const answered = messages.some(m => m.role === 'assistant');
      if (/biggest lever/.test(t) && !answered && last.role !== 'tool') return say(['{"step": 2, "why": "The layout decided in step 1 shapes everything after it."}']);
      if (/IMPROVEMENT PLAN/.test(t) && !answered && last.role !== 'tool') return say(['## Better path\n\n', '1. Improve the first attempt SCRIPT:goalstep\n', '2. Polish and check SCRIPT:goalstep\n']);
    }
    if (directive?.[1] === 'searchloop') {
      const n = requests.filter(r => r.directive === 'searchloop').length;
      send({ role: 'assistant', content: `Plan, version ${n}:

## Search plan

1. Search for the topic
2. Read the best result
3. Write the summary

` });
      const argsJson = JSON.stringify({ queries: [`topic search number ${n}`] });
      send({ tool_calls: [{ index: 0, id: `call_${requests.length}`, type: 'function', function: { name: 'web_search', arguments: argsJson } }] });
      return finish('tool_calls', 60);
    }
    if (directive?.[1] === 'bloat') {
      promptTokens = parseInt((directive[2] ?? '').trim(), 10) || 100;
      send({ role: 'assistant', content: '', reasoning_content: 'Thinking about it for a long time and never getting to the answer.' });
      return finish('stop', 20);
    }
    if (directive?.[1] === 'goalstep') {
      // A goal round: read the goal, mark it complete, then say so.
      if (!toolMessages.length) return toolCall('get_goal', {});
      if (toolMessages.length === 1 && last.role === 'tool') {
        const t = text(last); const id = /"id"\s*:\s*"([^"]+)"/.exec(t)?.[1]; const revision = Number(/"revision"\s*:\s*(\d+)/.exec(t)?.[1] ?? 1);
        if (id) return toolCall('update_goal', { goal_id: id, revision, action: 'complete' });
      }
      return say(['Step finished. The result is checked.']);
    }
    if (directive?.[1] === 'planslow') return say(['## Slow plan\n\n', '1. First thing SCRIPT:goalstep\n', '2. A step that never finishes SCRIPT:slow 4000\n', '3. Third thing SCRIPT:goalstep\n']);
    if (directive?.[1] === 'planreply') return say(['I looked around the folder first.\n\n## The pagoda plan\n\n', '1. First thing SCRIPT:goalstep\n', '2. Second thing SCRIPT:goalstep\n', '3. Third thing SCRIPT:goalstep\n']);
    if (directive?.[1] === 'multi' && last.role === 'tool') {
      let cfg = {};
      try { cfg = JSON.parse((directive[2] ?? '').trim()); } catch { /* a test bug */ }
      return say([String(cfg.say ?? 'Done.')]);
    }
    if (last.role === 'tool') {
      const result = text(last).replace(/\s+/g, ' ').slice(0, 120);
      return say(['Tool finished. ', `Result: ${result}`]);
    }
    const [name, args] = [directive?.[1], (directive?.[2] ?? '').trim()];
    // The Harness's compaction request (its summarizer prompt asks for the person's evolving goals): answer with a short summary.
    if (messages.some(m => /evolving goals/.test(text(m)))) { requests.at(-1).summary = true; return say(['Summary of the earlier conversation: the person asked for echoes of a word and got them.']); }
    // The Harness's session-title request carries the same user text but offers no tools.
    if (!(body.tools ?? []).length && messages.some(m => /concise title/i.test(text(m)))) return say(['Scripted title']);
    if (name === 'slow') {
      const n = parseInt(args, 10) || 400;
      return say(Array.from({ length: n }, (_, i) => `w${i} `));
    }
    if (name === 'hang') { send({ role: 'assistant', content: '' }); return; }   // headers sent, then silence: a model server that went quiet
    if (name === 'echo') return say([args]);
    if (name === 'retrywrite' || name === 'retryshell') {
      const [file, ...rest] = args.split(/\s+/);
      const value = rest.join(' ') || 'run';
      return name === 'retrywrite'
        ? toolCall('write', { file_path: file, content: `${value}
` })
        : toolCall(shellName(), { command: shellCommand(file, value), description: `Append one line to ${file}` });
    }
    if (name === 'read') return toolCall('read', { file_path: args });
    if (name === 'call') {
      // This prompt's one call already ran (the Harness may add a context message after its result): answer, never call twice.
      if (toolMessages.length) return say(['Tool finished. ', `Result: ${text(toolMessages[toolMessages.length - 1]).replace(/\s+/g, ' ').slice(0, 120)}`]);
      const at = args.indexOf(' ');
      let callArgs = {};
      try { callArgs = JSON.parse(at < 0 ? '{}' : args.slice(at + 1).replace(/\s+#\d+.*$/, '')); } catch { /* an unparsable script argument is a test bug; send no arguments */ }
      return toolCall(at < 0 ? args : args.slice(0, at), callArgs);
    }
    if (name === 'multi') {
      let cfg = {};
      try { cfg = JSON.parse(args.trim()); } catch { /* a test bug: no calls */ }
      if (toolMessages.length) return say([String(cfg.say ?? 'Done.')]);
      send({ role: 'assistant', content: '', reasoning_content: 'I will use the tools.' });
      await delay(chunkDelayMs);
      (cfg.calls ?? []).forEach(([tool, callArgs], i) => send({ tool_calls: [{ index: i, id: `call_${requests.length}_${i}`, type: 'function', function: { name: tool, arguments: JSON.stringify(callArgs ?? {}) } }] }));
      return finish('tool_calls', 20);
    }
    if (name === 'spawn') {
      // A settlement notice (or any later user-role message) after this prompt already got its tool round: just answer.
      if (messages.slice(promptAt + 1).some(m => m.role === 'assistant') && last.role === 'user') {
        return say(['Noted: ', text(last).replace(/\s+/g, ' ').slice(0, 90)]);
      }
      const [count, ...script] = args.split(/\s+/);
      const n = Math.max(1, Number(count) || 1);
      send({ role: 'assistant', content: '', reasoning_content: `I will start ${n} helper agents.` });
      await delay(chunkDelayMs);
      for (let i = 0; i < n; i++) {
        const argsJson = JSON.stringify({ description: `Helper ${i + 1}`, prompt: `SCRIPT:${script.join(' ') || 'echo done'} #${i + 1}` });
        send({ tool_calls: [{ index: i, id: `call_${requests.length}_${i}`, type: 'function', function: { name: 'subagent', arguments: '' } }] });
        for (let k = 0; k < argsJson.length; k += 23) { send({ tool_calls: [{ index: i, function: { arguments: argsJson.slice(k, k + 23) } }] }); await delay(2); }
      }
      return finish('tool_calls', 40 * n);
    }
    if (name === 'ask') {
      send({ role: 'assistant', content: '' });
      send({ tool_calls: [{ index: 0, id: `call_${requests.length}`, type: 'function', function: { name: 'ask_user_question', arguments: JSON.stringify({ questions: [{ id: 'q1', question: 'Which colour?', options: [{ label: 'Red' }, { label: 'Blue' }] }] }) } }] });
      return finish('tool_calls', 20);
    }
    if (name === 'tool' || name === 'plain' || name === 'envdump') {
      const [file, ...rest] = args.split(/\s+/);
      const shell = (body.tools ?? []).find(t => ['bash', 'pwsh'].includes(t.function?.name))?.function?.name ?? 'bash';
      const value = rest.join(' ') || 'run';
      let command = shell === 'pwsh' ? `Add-Content -Path '${file}' -Value '${value}'` : `echo ${value} >> ${file}`;
      if (name === 'envdump') command = shell === 'pwsh' ? `Get-ChildItem Env: | ForEach-Object { $_.Name + '=' + $_.Value } | Set-Content -Path '${file}'` : `env > ${file}`;
      const call = { command, description: `Append one line to ${file}` };
      if (name === 'tool') {
        call.sandbox_permissions = 'danger-full-access';
        call.justification = 'The write must not be refused by the file sandbox.';
      }
      send({ role: 'assistant', content: '', reasoning_content: 'I will write the file with the shell.' });
      await delay(chunkDelayMs);
      const argsJson = JSON.stringify(call);
      send({ tool_calls: [{ index: 0, id: `call_${requests.length}`, type: 'function', function: { name: shell, arguments: '' } }] });
      // Fragment the arguments the way real servers do.
      for (let i = 0; i < argsJson.length; i += 17) {
        send({ tool_calls: [{ index: 0, function: { arguments: argsJson.slice(i, i + 17) } }] });
        await delay(5);
      }
      return finish('tool_calls', 30);
    }
    // hello, and the default for anything unscripted (including the Harness's own title request)
    send({ role: 'assistant', content: '', reasoning_content: 'Considering a greeting.' });
    await delay(chunkDelayMs);
    return say(['Hello ', 'from ', 'the ', 'scripted ', 'worker.']);
  });
  await new Promise(r => server.listen(port, '127.0.0.1', r));
  return {
    port: server.address().port,
    url: `http://127.0.0.1:${server.address().port}/v1`,
    requests, counts, get aborted() { return aborted; },
    close: () => new Promise(r => { server.closeAllConnections?.(); server.close(r); }),
  };
}

if (import.meta.url === `file://${process.argv[1].replace(/\\/g, '/')}` || process.argv[1]?.endsWith('scripted-worker.mjs')) {
  const w = await startScriptedWorker({ port: Number(process.env.PORT ?? 0) });
  console.log(`scripted worker on ${w.url}`);
}
