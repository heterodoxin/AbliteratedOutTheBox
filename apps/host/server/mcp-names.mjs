// The one rule for the name of a tool server (MCP), shared by the Settings editor (src/lib/mcp-names.ts keeps a copy; tests/mcp-rows.test.mjs
// checks they agree), the save check (native-settings.mjs) and the agent profile (agent/profile.mjs).
//
// The model sees every tool as `mcp__<server>__<tool>`, and an OpenAI-style API takes at most 64 characters for that name, so the server's
// part is limited to 32 (T6, tools and images review 2026-10-22). Before, the editor and the save check accepted 64 and the profile quietly
// dropped everything over 32 (or with a dot): the model never saw the tool and nothing said why.
export const MCP_NAME_MAX = 32;
export const MCP_NAME_RE = /^[A-Za-z0-9_-]{1,32}$/;
export const validMcpName = name => typeof name === 'string' && MCP_NAME_RE.test(name);
export const MCP_NAME_HELP = 'Use a simple server name of up to 32 characters: letters, numbers, hyphens or underscores.';
