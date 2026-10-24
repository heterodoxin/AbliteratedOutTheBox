/** An isolated local API for connection and streaming verification. No external requests. */
import http from 'node:http';
import { pathToFileURL } from 'node:url';
export async function createMockModelApi(port = 0) {
  const requests = [];
  const server = http.createServer(async (req, res) => {
    const chunks = []; for await (const chunk of req) chunks.push(chunk);
    const body = chunks.length ? JSON.parse(Buffer.concat(chunks)) : null;
    requests.push({ path: req.url, headers: req.headers, body });
    if (req.url === '/redirect/models') { res.writeHead(302, { location: '/v1/models' }); res.end(); return; }
    if (req.url.startsWith('/bad/')) { res.writeHead(401); res.end('secret-echo: test-key'); return; }
    if (req.url.includes('/models')) {
      res.setHeader('content-type', 'application/json');
      const paginated = req.url.includes('limit=100');
      res.end(JSON.stringify({ data: req.url.includes('after_id=') ? [{ id: 'second-model', display_name: 'Second model' }] : [{ id: 'mock-chat', name: 'Test chat model', context_length: 32768 }, { id: 'mock-reasoning', name: 'Test reasoning model', context_length: 65536 }], ...(paginated ? { has_more: !req.url.includes('after_id='), last_id: 'mock-reasoning' } : {}) })); return;
    }
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    const send = value => res.write(`data: ${JSON.stringify(value)}\r\n\r\n`);
    if (body.model === 'truncated') { send({ choices: [{ index: 0, delta: { content: 'Partial' } }] }); res.end(); return; }
    if (body.model === 'stream-error') { send({ error: { message: 'secret test-key' } }); res.end(); return; }
    if (body.model === 'slow') { send({ choices: [{ index: 0, delta: { content: 'Partial' } }] }); const timer = setInterval(() => res.write(': ping\n\n'), 500); res.on('close', () => clearInterval(timer)); return; }
    if (req.url.endsWith('/messages')) {
      send({ type: 'message_start', message: { usage: { input_tokens: 14, output_tokens: 0 } } });
      send({ type: 'content_block_delta', delta: { type: 'thinking_delta', thinking: 'Thinking.' } });
      send({ type: 'content_block_delta', delta: { type: 'text_delta', text: 'Hello from the test API.' } });
      send({ type: 'message_delta', usage: { output_tokens: 7 } }); send({ type: 'message_stop' });
    } else if (req.url.endsWith('/responses')) {
      send({ type: 'response.reasoning_summary_text.delta', delta: 'Thinking.' });
      send({ type: 'response.output_text.delta', delta: 'Hello from the test API.' });
      send({ type: 'response.completed', response: { usage: { input_tokens: 14, output_tokens: 7 } } });
    } else {
      // Divide a UTF-8 character and SSE event across network writes.
      const bytes = Buffer.from('data: ' + JSON.stringify({ choices: [{ index: 0, delta: { content: 'Hello ✓ from the test API.' } }] }) + '\r\n\r\n');
      const cut = bytes.indexOf(Buffer.from('✓')) + 1;
      res.write(bytes.subarray(0, cut)); res.write(bytes.subarray(cut));
      send({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 14, completion_tokens: 7 } });
      res.write('data: [DONE]\r\n\r\n');
    }
    res.end();
  });
  await new Promise(resolve => server.listen(port, '127.0.0.1', resolve));
  return { server, requests, baseUrl: `http://127.0.0.1:${server.address().port}/v1`, close: () => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }) };
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const mock = await createMockModelApi(Number(process.env.MOCK_PORT ?? 5181));
  console.log(`Mock model API: ${mock.baseUrl}`);
}
