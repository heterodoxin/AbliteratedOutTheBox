/** The one rule for the name of a tool server. A copy of server/mcp-names.mjs (tests/mcp-rows.test.mjs checks that they agree). */
export const MCP_NAME_MAX = 32;
export const MCP_NAME_RE = /^[A-Za-z0-9_-]{1,32}$/;
export const validMcpName = (name: string) => MCP_NAME_RE.test(name);
export const MCP_NAME_HELP = 'Use a simple server name of up to 32 characters: letters, numbers, hyphens or underscores.';
