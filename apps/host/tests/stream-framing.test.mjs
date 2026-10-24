/**
 * Stream framing: partial UTF-8 and split SSE lines must not corrupt output (checklist A02).
 *
 * This is the failure mode that looks like a model problem and is actually a decoder problem. A
 * multi-byte character split across two TCP reads becomes U+FFFD if the decoder is reset per chunk,
 * and an SSE line split across two reads is dropped or merged if the buffer is not carried. Both
 * produce answers with mojibake or missing text in them, which is very hard to attribute later.
 *
 * The fixture therefore splits at the **byte** level, not at convenient character boundaries, so the
 * test cannot accidentally pass because the writer happened to split somewhere safe.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function listen(server) {
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  return server.address().port;
}

/** An SSE data line carrying one delta. */
function line(text) {
  return `data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\n`;
}

/** Split a UTF-8 string into byte chunks of `n` bytes, so multi-byte characters straddle them. */
function byteChunks(text, n) {
  const bytes = Buffer.from(text, 'utf8');
  const out = [];
  for (let i = 0; i < bytes.length; i += n) out.push(bytes.subarray(i, i + n));
  return out;
}

async function loadCompletion() {
  const home = await mkdtemp(join(tmpdir(), 'strata-framing-'));
  process.env.STRATA_HOME = home;
  process.env.STRATA_MODELS_DIR = join(home, 'models');
  const { proxyCompletion } = await import('../server/inference.mjs');
  return { proxyCompletion, home };
}

test('a multi-byte character split across reads survives intact', async () => {
  const { proxyCompletion, home } = await loadCompletion();
  // Every one of these is multi-byte in UTF-8, and several are outside the BMP (4 bytes), so a
  // chunk boundary can land inside any of them.
  const wanted = 'héllo — 世界 🌍 日本語 ✅';
  const payload = line(wanted) + 'data: [DONE]\n\n';
  const fixture = http.createServer((req, res) => {
    req.resume();
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    // Three bytes at a time: every emoji and every CJK character is straddled.
    for (const chunk of byteChunks(payload, 3)) res.write(chunk);
    res.end();
  });
  const port = await listen(fixture);
  try {
    const frames = [];
    await proxyCompletion({
      baseUrl: `http://127.0.0.1:${port}`, message: 'x', onFrame: (f) => frames.push(f),
      signal: new AbortController().signal,
    });
    const text = frames.filter((f) => f.type === 'delta').map((f) => f.text).join('');
    assert.equal(text, wanted, 'the streamed text must be byte-identical to what was sent');
    assert.ok(!text.includes('\uFFFD'), 'no replacement character may appear');
  } finally {
    await new Promise((r) => fixture.close(r));
    await rm(home, { recursive: true, force: true });
  }
});

test('an SSE line split across reads is reassembled, not dropped', async () => {
  const { proxyCompletion, home } = await loadCompletion();
  const fixture = http.createServer((req, res) => {
    req.resume();
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    // Split the JSON mid-object, mid-escape and mid-line, with pauses so the reads really separate.
    const body = line('alpha') + line('beta') + 'data: [DONE]\n\n';
    const chunks = byteChunks(body, 7);
    (async () => {
      for (const c of chunks) { res.write(c); await wait(2); }
      res.end();
    })();
  });
  const port = await listen(fixture);
  try {
    const frames = [];
    await proxyCompletion({
      baseUrl: `http://127.0.0.1:${port}`, message: 'x', onFrame: (f) => frames.push(f),
      signal: new AbortController().signal,
    });
    const text = frames.filter((f) => f.type === 'delta').map((f) => f.text).join('');
    assert.equal(text, 'alphabeta', 'both deltas must survive the split');
  } finally {
    await new Promise((r) => fixture.close(r));
    await rm(home, { recursive: true, force: true });
  }
});

test('a stream that ends without a trailing newline still yields its last delta', async () => {
  const { proxyCompletion, home } = await loadCompletion();
  const fixture = http.createServer((req, res) => {
    req.resume();
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    // No trailing blank line: the reader must flush the remaining buffer when the stream ends.
    res.end(`data: ${JSON.stringify({ choices: [{ delta: { content: 'tail' } }] })}`);
  });
  const port = await listen(fixture);
  try {
    const frames = [];
    await proxyCompletion({
      baseUrl: `http://127.0.0.1:${port}`, message: 'x', onFrame: (f) => frames.push(f),
      signal: new AbortController().signal,
    });
    const text = frames.filter((f) => f.type === 'delta').map((f) => f.text).join('');
    assert.equal(text, 'tail');
  } finally {
    await new Promise((r) => fixture.close(r));
    await rm(home, { recursive: true, force: true });
  }
});

test('a malformed frame is skipped without losing the frames around it', async () => {
  const { proxyCompletion, home } = await loadCompletion();
  const fixture = http.createServer((req, res) => {
    req.resume();
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    // A truncated JSON object in the middle. The contract is that it is skipped, not that it
    // aborts the turn: a provider may emit a partial line and the answer is still usable.
    res.end(line('before') + 'data: {"choices":[{"delta":{"content":"trunc\n\n' + line('after') + 'data: [DONE]\n\n');
  });
  const port = await listen(fixture);
  try {
    const frames = [];
    await proxyCompletion({
      baseUrl: `http://127.0.0.1:${port}`, message: 'x', onFrame: (f) => frames.push(f),
      signal: new AbortController().signal,
    });
    const text = frames.filter((f) => f.type === 'delta').map((f) => f.text).join('');
    assert.equal(text, 'beforeafter', 'the good frames on both sides must survive');
  } finally {
    await new Promise((r) => fixture.close(r));
    await rm(home, { recursive: true, force: true });
  }
});

test('an error frame is surfaced rather than silently ending the turn', async () => {
  const { proxyCompletion, home } = await loadCompletion();
  const fixture = http.createServer((req, res) => {
    req.resume();
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.end(`data: ${JSON.stringify({ error: { message: 'the engine refused this request' } })}\n\n`);
  });
  const port = await listen(fixture);
  try {
    await assert.rejects(
      () => proxyCompletion({
        baseUrl: `http://127.0.0.1:${port}`, message: 'x', onFrame: () => {},
        signal: new AbortController().signal,
      }),
      /refused this request/,
      'an error frame must reach the caller with the provider\'s own words',
    );
  } finally {
    await new Promise((r) => fixture.close(r));
    await rm(home, { recursive: true, force: true });
  }
});
