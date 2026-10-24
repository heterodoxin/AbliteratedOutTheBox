import { tr } from './i18n';

/**
 * The sentence under the request table, in plain words: how many requests, the average output speed, the speed of reading prompts and how much
 * was reused ("In 18 requests your average output was 47.4 tok/s and prompt reading 1,332 tok/s, reusing 91K tokens."). Parts the engine has
 * not reported are left out of the sentence rather than shown as dashes.
 */
export function requestSummary(input: { requests: number; output?: number | null; reading?: number | null; reused?: number | null }, format: { number: (n: number, digits?: number) => string; tokens: (n: number) => string }): string {
  const has = (n: number | null | undefined): n is number => typeof n === 'number' && Number.isFinite(n) && n > 0;
  const requests = format.tokens(input.requests);
  if (has(input.output) && has(input.reading) && has(input.reused)) return tr('In {0} requests your average output was {1} tok/s and prompt reading {2} tok/s, reusing {3} tokens.', [requests, format.number(input.output, 1), format.number(input.reading), format.tokens(input.reused)]);
  if (has(input.output) && has(input.reading)) return tr('In {0} requests your average output was {1} tok/s and prompt reading {2} tok/s.', [requests, format.number(input.output, 1), format.number(input.reading)]);
  if (has(input.output) && has(input.reused)) return tr('In {0} requests your average output was {1} tok/s, reusing {2} tokens.', [requests, format.number(input.output, 1), format.tokens(input.reused)]);
  if (has(input.output)) return tr('In {0} requests your average output was {1} tok/s.', [requests, format.number(input.output, 1)]);
  return tr('{0} requests so far.', [requests]);
}
