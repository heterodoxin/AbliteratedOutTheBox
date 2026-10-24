/**
 * Registers the app's web search with the Harness's web seam (`ctx.web`), in place of the Harness's own backends that all need a service key.
 *
 *   config.provider   'duckduckgo' (default, no key), 'exa' or 'perplexity' (the person's own key, read from the environment variable
 *                     named by config.keyEnv, which the host sets; the key is never in the profile file).
 *
 * The profile pins `web.searchProvider` to the id registered here, so a second usable backend cannot make the choice ambiguous.
 * Plain ESM, loaded by absolute path like strata-gate.mjs; its only imports are the app's own search modules next to the host.
 */
import { createDuckDuckGoSearch } from '../../web-search-ddg.mjs';
import { createExaSearch, createPerplexitySearch } from '../../web-search-keyed.mjs';

export const name = 'strata-web-search';
export const inject = ['web'];

export function apply(ctx, config = {}) {
  const keyOf = () => (config.keyEnv ? process.env[config.keyEnv] ?? '' : '');
  // Tests only (the bridge option `searchMock`): the same search code on a pretend web, so nothing leaves the machine.
  const pretend = config.mock === 'offline' ? { fetch: async () => { throw new TypeError('fetch failed'); } } : config.mock === 'limited' ? { fetch: async url => new Response('<html>please verify you are human: captcha</html>', { status: /duckduckgo/.test(String(url)) ? 202 : 429 }) } : null;
  const provider = pretend ? createDuckDuckGoSearch(pretend)
    : config.provider === 'exa' ? createExaSearch({ key: keyOf })
    : config.provider === 'perplexity' ? createPerplexitySearch({ key: keyOf })
    : createDuckDuckGoSearch();
  ctx.web.registerSearchProvider(provider);
}
