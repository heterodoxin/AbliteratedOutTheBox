#!/usr/bin/env node
/**
 * One revision attempt for Time-skip, performed by the app's own assistant.
 *
 * Strata Control runs this as its `--revise-command`, with the candidate folder (an isolated copy of the user's work) as the working
 * directory, the request in STRATA_TIMESKIP_PROMPT and the attempt number in STRATA_TIMESKIP_ATTEMPT. It asks the host (which owns the
 * model and the Harness) for one agent turn inside that folder and returns when the turn ends. Exit 0 means "an attempt was made", not
 * "it helped": whether it helped is decided by the checks and, failing those, by the person.
 *
 * The host answers with a STREAM of one-line JSON messages (`beat`, `activity`, then `result` or `error`) and sends the first one at once.
 * A turn on a slow computer can take many minutes with nothing to say; an answer that sends no byte for five minutes is abandoned by
 * Node's fetch, which is how a Time-skip used to end with "The assistant could not be reached".
 */
const base = process.env.STRATA_HOST_URL;
const secret = process.env.STRATA_INTERNAL_SECRET;
const prompt = process.env.STRATA_TIMESKIP_PROMPT ?? '';
const attempt = Number(process.env.STRATA_TIMESKIP_ATTEMPT ?? 1);

if (!base || !secret || !prompt) {
  console.error('The revision step needs the Strata app to be running.');
  process.exit(2);
}

try {
  const res = await fetch(`${base}/api/internal/timeskip-revise`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-strata-internal': secret },
    body: JSON.stringify({ cwd: process.cwd(), prompt, attempt }),
  });
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    console.error(body.error ?? `The assistant could not make the change (HTTP ${res.status}).`);
    process.exit(1);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result = null, failure = null;
  const take = line => {
    if (!line.trim()) return;
    let message;
    try { message = JSON.parse(line); } catch { return; }
    if (message.type === 'result') result = String(message.text ?? '').trim();
    else if (message.type === 'error') failure = String(message.message ?? 'The assistant could not make the change.');
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (value) { buffer += decoder.decode(value, { stream: true }); const lines = buffer.split('\n'); buffer = lines.pop() ?? ''; lines.forEach(take); }
    if (done) break;
  }
  take(buffer);
  if (failure) { console.error(failure); process.exit(1); }
  if (result === null) { console.error('The assistant stopped before it finished.'); process.exit(1); }
  console.log(result);
} catch (e) {
  console.error(`The assistant could not be reached: ${e.message}`);
  process.exit(1);
}
