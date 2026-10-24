/**
 * A slow model for tests: forwards a model server and holds every streamed chunk back, so a fast engine answers like a slow computer.
 *   TARGET=http://127.0.0.1:18797 PORT=18900 DELAY_MS=200 node scripts/slow-proxy.mjs
 * Only streamed (text/event-stream) answers are slowed; everything else passes straight through.
 */
import http from 'node:http';
import { Transform } from 'node:stream';

const target = new URL(process.env.TARGET ?? 'http://127.0.0.1:18797');
const port = Number(process.env.PORT ?? 18900);
const delay = Number(process.env.DELAY_MS ?? 200);

http.createServer((req, res) => {
  const up = http.request({ host: target.hostname, port: target.port, path: req.url, method: req.method, headers: { ...req.headers, host: target.host } }, answer => {
    res.writeHead(answer.statusCode ?? 502, answer.headers);
    if (String(answer.headers['content-type'] ?? '').includes('text/event-stream')) {
      answer.pipe(new Transform({ transform(chunk, _enc, done) { setTimeout(() => done(null, chunk), delay); } })).pipe(res);
    } else answer.pipe(res);
  });
  up.on('error', e => { if (!res.headersSent) res.writeHead(502); res.end(String(e.message)); });
  req.pipe(up);
  res.on('close', () => up.destroy());
}).listen(port, '127.0.0.1', () => console.log(`slow model on ${port} (+${delay} ms per streamed chunk) -> ${target.origin}`));
