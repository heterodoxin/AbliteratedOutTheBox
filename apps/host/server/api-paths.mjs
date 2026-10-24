/**
 * Anthropic clients add `/v1/messages` to the base URL themselves, so the base URL they are given is the server root (no `/v1`).
 * A base URL copied from an OpenAI-style setup ends in `/v1`, and the client then posts `/v1/v1/messages`. That one path means the
 * same as the real one; nothing else is aliased (an OpenAI client never adds `/v1` on its own).
 */
const ALIASES = new Map([['/v1/v1/messages', '/v1/messages'], ['/v1/v1/messages/count_tokens', '/v1/messages/count_tokens']]);

export function apiPath(path) {
  const clean = path.length > 1 ? path.replace(/\/+$/, '') : path;
  return ALIASES.get(clean) ?? path;
}

