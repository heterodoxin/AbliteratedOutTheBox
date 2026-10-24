/**
 * Subsequence matching for the command palette.
 *
 * Prefix matching was the wrong tool: `mtx` should find *Monitor*, and `stg`
 * should find *Settings*, because that is what every other launcher does and
 * what a keyboard user's fingers already assume.
 *
 * Returns the matched character indices so the caller can highlight them — a
 * fuzzy result you cannot explain is worse than no result, because the user
 * cannot tell whether the thing they wanted was ranked low or is missing.
 */

export interface FuzzyResult {
  /** Character indices in the haystack that the query matched. */
  indices: number[];
  /** Higher is better. Consecutive matches and word starts score highest. */
  score: number;
}

const WORD_START = /[\s\-_/.]/;

export function fuzzyMatch(query: string, text: string): FuzzyResult | null {
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  if (!q) return { indices: [], score: 0 };

  let qi = 0;
  let score = 0;
  let lastMatch = -2;
  const indices: number[] = [];

  for (let ti = 0; ti < t.length && qi < q.length; ti += 1) {
    if (t[ti] !== q[qi]) continue;

    indices.push(ti);
    // Consecutive characters are what make a match feel intentional.
    if (ti === lastMatch + 1) score += 6;
    // Starting a word is the next strongest signal.
    if (ti === 0 || WORD_START.test(t[ti - 1])) score += 4;
    // Earlier matches beat later ones, so the ranking follows the reading order.
    score += Math.max(0, 3 - Math.floor(ti / 8));

    lastMatch = ti;
    qi += 1;
  }

  if (qi < q.length) return null;

  // A short haystack containing the match is more likely to be the target than
  // a long one that merely contains the same letters.
  score += Math.max(0, 12 - Math.floor(t.length / 4));
  return { indices, score };
}

/**
 * Ranks a list, dropping non-matches and sorting best first.
 *
 * `text` returns the strings to try, best first. A label match wins outright;
 * aliases are tried after it, so "tune" finds *Engine tuning* without the label
 * having to be renamed to something worse. Only a label match yields highlight
 * indices — highlighting a word the user cannot see would be confusing.
 */
export function rank<T>(query: string, items: T[], text: (item: T) => string[]): { item: T; indices: number[] }[] {
  if (!query.trim()) return items.map((item) => ({ item, indices: [] }));
  const q = query.trim();
  const out: { item: T; indices: number[]; score: number }[] = [];
  for (const item of items) {
    const candidates = text(item);
    for (let i = 0; i < candidates.length; i += 1) {
      const hit = fuzzyMatch(q, candidates[i]);
      if (!hit) continue;
      // A label hit always outranks an alias hit, and later aliases rank lower.
      const penalty = i * 40;
      out.push({ item, indices: i === 0 ? hit.indices : [], score: hit.score - penalty });
      break;
    }
  }
  return out.sort((a, b) => b.score - a.score).map(({ item, indices }) => ({ item, indices }));
}

/**
 * Splits a query into its scope prefix and the rest.
 *
 * `>` scopes to commands, `#` to conversations, `@` to workspaces — three
 * prefixes covering everything the palette can reach, so the user does not have
 * to read a mixed list to find the kind of thing they already know they want.
 */
export type Scope = 'all' | 'command' | 'session' | 'workspace';

export function parseScope(raw: string): { scope: Scope; term: string } {
  const c = raw[0];
  if (c === '>') return { scope: 'command', term: raw.slice(1) };
  if (c === '#') return { scope: 'session', term: raw.slice(1) };
  if (c === '@') return { scope: 'workspace', term: raw.slice(1) };
  return { scope: 'all', term: raw };
}

/** Renders a label with its matched characters emphasised. */
export function highlight(label: string, indices: number[]): { text: string; hit: boolean }[] {
  if (!indices.length) return [{ text: label, hit: false }];
  const set = new Set(indices);
  const parts: { text: string; hit: boolean }[] = [];
  for (let i = 0; i < label.length; i += 1) {
    const hit = set.has(i);
    const last = parts[parts.length - 1];
    if (last && last.hit === hit) last.text += label[i];
    else parts.push({ text: label[i], hit });
  }
  return parts;
}
