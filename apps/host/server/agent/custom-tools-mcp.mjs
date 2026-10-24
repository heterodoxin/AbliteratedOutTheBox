#!/usr/bin/env node
/**
 * The app's own tool server for the tools a person writes in Settings -> Tools.
 *
 *   node custom-tools-mcp.mjs instructions <tools.json>   one tool per instruction tool: calling it returns the person's text
 *   node custom-tools-mcp.mjs commands     <tools.json>   one tool per command tool: calling it runs the person's command
 *
 * It speaks the Model Context Protocol over stdio (newline-delimited JSON-RPC), the transport the Harness's MCP client uses for a program
 * it starts. The file is re-read on every request, so a change in Settings is seen without restarting anything. The assistant's text for
 * a command tool reaches the command in the STRATA_INPUT environment variable (never spliced into the command line, so it cannot add
 * commands of its own). A command runs in the shell that suits it (PowerShell for a Windows command that reads as PowerShell, cmd.exe for
 * the rest, sh elsewhere), in its own folder or the one the assistant names, gets 2 minutes unless its tool allows more (up to an hour) and
 * 20,000 characters of output (custom-command-run.mjs).
 */
import { readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { runCommand } from './custom-command-run.mjs';

const [mode, file] = process.argv.slice(2);
if (!['instructions', 'commands'].includes(mode) || !file) { console.error('usage: custom-tools-mcp.mjs instructions|commands <tools.json>'); process.exit(2); }

const load = () => { try { return JSON.parse(readFileSync(file, 'utf8'))[mode] ?? {}; } catch { return {}; } };

const schema = mode === 'commands'
  ? { type: 'object', properties: {
    input: { type: 'string', description: 'Text to give the command (optional).' },
    cwd: { type: 'string', description: 'The folder to run the command in (optional). Give the working folder of this chat when the command uses relative paths.' },
  }, additionalProperties: false }
  : { type: 'object', properties: {}, additionalProperties: false };

function listTools() {
  return Object.entries(load()).map(([name, t]) => ({ name, description: String(t.description ?? ''), inputSchema: schema, annotations: mode === 'instructions' ? { readOnlyHint: true } : {} }));
}

async function call(name, args) {
  const t = load()[name];
  if (!t) return { isError: true, content: [{ type: 'text', text: 'That tool is no longer available.' }] };
  if (mode === 'instructions') return { content: [{ type: 'text', text: String(t.text ?? '') }] };
  const r = await runCommand({ command: String(t.command ?? ''), input: args?.input, callCwd: args?.cwd, cwd: t.cwd, shell: t.shell, timeoutSec: t.timeoutSec });
  return { isError: !r.ok, content: [{ type: 'text', text: r.text }] };
}

const send = message => process.stdout.write(JSON.stringify({ jsonrpc: '2.0', ...message }) + '\n');

async function handle(line) {
  let msg;
  try { msg = JSON.parse(line); } catch { return; }
  const { id, method, params } = msg;
  if (id === undefined) return;                                  // notifications need no answer
  try {
    if (method === 'initialize') {
      return send({ id, result: { protocolVersion: params?.protocolVersion ?? '2024-11-05', capabilities: { tools: { listChanged: false } }, serverInfo: { name: `strata-${mode}`, version: '1.0.0' } } });
    }
    if (method === 'ping') return send({ id, result: {} });
    if (method === 'tools/list') return send({ id, result: { tools: listTools() } });
    if (method === 'tools/call') return send({ id, result: await call(String(params?.name ?? ''), params?.arguments ?? {}) });
    return send({ id, error: { code: -32601, message: `Method not found: ${method}` } });
  } catch (e) {
    return send({ id, error: { code: -32603, message: String(e?.message ?? e) } });
  }
}

createInterface({ input: process.stdin }).on('line', line => { void handle(line); }).on('close', () => process.exit(0));
