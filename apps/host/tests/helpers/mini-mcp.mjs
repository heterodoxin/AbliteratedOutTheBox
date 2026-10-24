// A minimal MCP server over stdio (newline-delimited JSON-RPC), for the tool-server test. One tool, `shout`: it upper-cases
// its text and reports whether the secret from the environment arrived, WITHOUT printing the secret.
import { createInterface } from 'node:readline';

const send = message => process.stdout.write(`${JSON.stringify({ jsonrpc: '2.0', ...message })}\n`);
createInterface({ input: process.stdin }).on('line', line => {
  let m; try { m = JSON.parse(line); } catch { return; }
  if (m.method === 'initialize') {
    send({ id: m.id, result: { protocolVersion: m.params?.protocolVersion ?? '2025-06-18', capabilities: { tools: { listChanged: false } }, serverInfo: { name: 'mini', version: '1.0.0' } } });
  } else if (m.method === 'tools/list') {
    send({ id: m.id, result: { tools: [{ name: 'shout', description: 'Upper-case the text.', inputSchema: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'] } }] } });
  } else if (m.method === 'tools/call') {
    const text = String(m.params?.arguments?.text ?? '');
    send({ id: m.id, result: { content: [{ type: 'text', text: `${text.toUpperCase()}|token:${process.env.DEMO_TOKEN ? 'set' : 'missing'}` }] } });
  } else if (m.id !== undefined && m.method) {
    send({ id: m.id, result: {} });
  }
});
