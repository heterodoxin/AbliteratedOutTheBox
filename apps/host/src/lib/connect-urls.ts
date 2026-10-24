/**
 * The address an app is given depends on who builds the final request URL.
 *
 * OpenAI-style clients (Chat Completions, Responses) are given the base that ends in /v1 and add /chat/completions or /responses.
 * Anthropic-style clients (Claude Code, the Anthropic SDKs) add /v1/messages themselves, so they are given the server root. With
 * the /v1 base they post /v1/v1/messages.
 */
export type ConnectProtocol = 'openai' | 'responses' | 'anthropic';

/** The server root: the base address without its trailing /v1. */
export function serverRoot(baseUrl: string): string {
  return String(baseUrl ?? '').trim().replace(/\/+$/, '').replace(/\/v1$/i, '');
}

/** The address to type into an app that speaks this protocol. */
export function appBaseUrl(baseUrl: string, protocol: ConnectProtocol): string {
  return protocol === 'anthropic' ? serverRoot(baseUrl) : String(baseUrl ?? '').trim().replace(/\/+$/, '');
}
