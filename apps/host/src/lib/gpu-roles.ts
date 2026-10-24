/**
 * Main / Helper / Off for every graphics card. The same rules as strata_control/gpu_roles.py (the Terminal, setup and the server); both run
 * tests/data/gpu_roles_vectors.json.
 *
 *  - Main:   the card that runs the LAST stage of the layer split (the output head, the draft layer and the check of its guesses). Exactly one.
 *  - Helper: holds a share of the model, so more of it fits on graphics cards instead of being read from the drive.
 *  - Off:    Strata does not touch it.
 *
 * The engine takes the cards in stage order (`gpu: [helper, ..., main]`, `gpu_order: "as_given"`), so Main is the LAST one.
 */
export type Role = 'main' | 'helper' | 'off';
export const ROLES: Role[] = ['main', 'helper', 'off'];
export type Roles = Record<number, Role>;

export interface RoleCard {
  index: number;
  name: string;
  vram_gb: number;
  /** SMs x clock (or the AMD equivalent): only the order between cards matters. */
  speed?: number;
  usable?: boolean;
  problem?: string | null;
  vendor?: string;
  uma?: boolean;
  integrated?: boolean;
  unified_gb?: number;
  /** PCI bus id when known: used to notice a renumbering. */
  bus?: string;
}

export const OFF_AT_OR_BELOW_GB = 8;
const SAME_SPEED = 1.05;
const MEMORY_GUARD_SPEED = 0.5;
const MIN_CARD_GB = 4;
const SPLIT_CARD_OVERHEAD_GB = 2;
export const CPU_ONLY_LINE = 'Performance is degraded on CPU only.';
export const NO_CARD_LINE = 'This engine needs a graphics card to start. Turn one on.';

const mem = (c: RoleCard) => Number(c.vram_gb) || 0;
const speed = (c: RoleCard) => (typeof c.speed === 'number' && c.speed > 0 ? c.speed : 0);
const shares = (c: RoleCard) => !!(c.integrated || c.uma || c.unified_gb || c.vendor === 'apple');
const canRun = (c: RoleCard) => c.usable !== false;
const rnd = (n: number) => Math.round(n);

export function shortName(c: RoleCard): string {
  let n = String(c.name || `GPU ${c.index + 1}`);
  for (const junk of ['NVIDIA GeForce ', 'NVIDIA ', 'GeForce ', 'AMD Radeon ', 'AMD ', 'Intel(R) ', 'Intel ']) if (n.startsWith(junk)) { n = n.slice(junk.length); break; }
  return n.trim();
}
/** The number the screens show: GPU 1 is card 0. */
export const gpuLabel = (c: RoleCard) => `GPU ${c.index + 1}`;

/** The cards shown as rows: every card the engine's make can name, in card order. */
export function rows<T extends RoleCard>(cards: T[] | null | undefined, vendor?: string | null): T[] {
  const list = cards ?? [];
  const make = vendor || pickVendor(list);
  const mine = make ? list.filter(c => !c.vendor || c.vendor === make) : list;
  return [...mine].sort((a, b) => a.index - b.index);
}
function pickVendor(cards: RoleCard[]): string | null {
  const makes = new Set(cards.filter(c => canRun(c) && c.vendor).map(c => c.vendor as string));
  if (makes.size <= 1) return [...makes][0] ?? null;
  return makes.has('nvidia') ? 'nvidia' : cards[0]?.vendor ?? null;
}

function eligible(cs: RoleCard[]): RoleCard[] {
  const real = cs.filter(c => canRun(c) && !shares(c) && mem(c) >= MIN_CARD_GB);
  return real.length ? real : cs.filter(canRun);
}

export type MainWhy = 'only' | 'faster' | 'same' | 'memory' | 'none';

export function pickMain(cs: RoleCard[]): { main: RoleCard | null; why: MainWhy } {
  const pool = eligible(cs);
  if (!pool.length) return { main: null, why: 'none' };
  if (pool.length === 1) return { main: pool[0], why: 'only' };
  const top = Math.max(...pool.map(speed));
  const group = top > 0 ? pool.filter(c => speed(c) * SAME_SPEED >= top) : [...pool];
  const byMem = (a: RoleCard, b: RoleCard) => rnd(mem(b)) - rnd(mem(a)) || a.index - b.index;
  let main = [...group].sort(byMem)[0];
  let why: MainWhy = group.length === 1 ? 'faster' : 'same';
  if (group.length > 1 && new Set(group.map(c => `${c.name}|${rnd(mem(c))}`)).size === 1) main = [...group].sort((a, b) => b.index - a.index)[0]; // identical cards: the HIGHEST number is Main
  if (top <= 0 && new Set(pool.map(c => rnd(mem(c)))).size > 1) why = 'memory';
  const bigger = pool.filter(c => rnd(mem(c)) > rnd(mem(main)) && (top <= 0 || speed(c) >= MEMORY_GUARD_SPEED * top));
  if (bigger.length) { main = [...bigger].sort(byMem)[0]; why = 'memory'; }
  return { main, why };
}

export type OffWhy = 'cannot_run' | 'igpu' | 'small' | 'helper';
export interface Recommendation { roles: Roles; main: number | null; why: MainWhy; reasons: Record<number, MainWhy | OffWhy> }

/** The best card is Main; every other card starts as Helper, EXCEPT a card with 8 GB or less when a bigger card is there (or it has under 4 GB), built-in graphics beside a real card, and a card the engine cannot run: those start Off. */
export function recommend(cards: RoleCard[] | null | undefined, vendor?: string | null): Recommendation {
  const cs = rows(cards, vendor);
  const { main, why } = pickMain(cs);
  const roles: Roles = {}, reasons: Recommendation['reasons'] = {};
  // An 8 GB card is a poor Helper beside a bigger card, but a perfectly good one when 8 GB is all there is: two GTX 1080s are a working
  // 16 GB pool (#3655). A card with 8 GB or less starts Off only when a bigger usable card is there, or when it is too small to help.
  const hasBigger = cs.some(c => canRun(c) && !shares(c) && rnd(mem(c)) > OFF_AT_OR_BELOW_GB);
  for (const c of cs) {
    const i = c.index;
    if (main && c === main) { roles[i] = 'main'; reasons[i] = why; }
    else if (!canRun(c)) { roles[i] = 'off'; reasons[i] = 'cannot_run'; }
    else if (shares(c) && main) { roles[i] = 'off'; reasons[i] = 'igpu'; }
    else if (rnd(mem(c)) <= OFF_AT_OR_BELOW_GB && (hasBigger || mem(c) < MIN_CARD_GB)) { roles[i] = 'off'; reasons[i] = 'small'; }
    else { roles[i] = 'helper'; reasons[i] = 'helper'; }
  }
  return { roles, main: main ? main.index : null, why, reasons };
}

const on = (roles: Roles) => Object.keys(roles).map(Number).filter(i => roles[i] !== 'off');
const byNumber = (roles: Roles) => Object.keys(roles).map(Number).sort((a, b) => a - b);

/** Exactly one Main among the cards that are on; a lone card that is on is Main. */
export function normalize(roles: Roles, cards: RoleCard[]): Roles {
  const cs = new Map(cards.map(c => [c.index, c]));
  const out: Roles = {};
  for (const i of byNumber(roles)) if (cs.has(i)) out[i] = ROLES.includes(roles[i]) ? roles[i] : 'off';
  const onList = on(out), mains = onList.filter(i => out[i] === 'main');
  if (onList.length && !mains.length) {
    const best = [...onList].sort((a, b) => speed(cs.get(b)!) - speed(cs.get(a)!) || rnd(mem(cs.get(b)!)) - rnd(mem(cs.get(a)!)) || a - b)[0];
    out[best] = 'main';
  }
  for (const extra of mains.slice(1)) out[extra] = 'helper';
  if (onList.length === 1) out[onList[0]] = 'main';
  return out;
}

export interface RoleChange { roles: Roles; note: string | null }

/** One click: the new roles and a code for the quiet line (old_main_helper:<i>, promoted:<i>, only_main). */
export function setRole(roles: Roles, cards: RoleCard[], idx: number, role: Role): RoleChange {
  const cs = new Map(cards.map(c => [c.index, c]));
  const out: Roles = { ...roles };
  let note: string | null = null;
  if (!cs.has(idx) || !ROLES.includes(role)) return { roles: out, note };
  const ids = byNumber(out);
  const fastest = (list: number[]) => [...list].sort((a, b) => speed(cs.get(b)!) - speed(cs.get(a)!) || a - b);
  if (role === 'main') {
    const old = ids.find(i => i !== idx && out[i] === 'main');
    out[idx] = 'main';
    if (old !== undefined) { out[old] = 'helper'; note = `old_main_helper:${old}`; }
  } else if (role === 'helper') {
    if (out[idx] === 'main') {
      const others = fastest(ids.filter(i => i !== idx && out[i] === 'helper'));
      if (!others.length) return { roles: out, note: 'only_main' };
      out[idx] = 'helper'; out[others[0]] = 'main'; note = `promoted:${others[0]}`;
    } else out[idx] = 'helper';
  } else {
    const wasMain = out[idx] === 'main';
    out[idx] = 'off';
    if (wasMain) {
      const next = fastest(ids.filter(i => out[i] === 'helper'));
      if (next.length) { out[next[0]] = 'main'; note = `promoted:${next[0]}`; }
    }
  }
  const onList = on(out);
  if (onList.length === 1) out[onList[0]] = 'main';
  else if (onList.length && !onList.some(i => out[i] === 'main')) out[onList[0]] = 'main';
  return { roles: out, note };
}

/** The engine's card list: Helpers (slowest first, ties by number), then the Main card LAST; Off cards left out. */
export function order(roles: Roles, cards: RoleCard[]): number[] {
  const cs = new Map(cards.map(c => [c.index, c]));
  const r = normalize(roles, cards);
  const helpers = byNumber(r).filter(i => r[i] === 'helper').sort((a, b) => speed(cs.get(a)!) - speed(cs.get(b)!) || a - b);
  return [...helpers, ...byNumber(r).filter(i => r[i] === 'main')];
}

export interface RoleMirror { gpu_roles: Record<string, Role>; gpu: number | number[]; gpu_order?: 'as_given'; gpu_none?: true; gpu_cards: Record<string, string> }
/** What a writer stores: "gpu_roles" plus the legacy mirror older engines and the Monitor read. Nothing on: "gpu": [] and "gpu_none": true. */
export function mirror(roles: Roles, cards: RoleCard[]): RoleMirror {
  const r = normalize(roles, cards), lst = order(r, cards);
  const out: RoleMirror = { gpu_roles: Object.fromEntries(byNumber(r).map(i => [String(i), r[i]])), gpu: [], gpu_cards: {} };
  if (lst.length) { out.gpu = lst.length === 1 ? lst[0] : lst; out.gpu_order = 'as_given'; } else out.gpu_none = true;
  for (const c of cards) if (c.index in r) out.gpu_cards[String(c.index)] = [shortName(c), c.bus ?? ''].filter(Boolean).join('|');
  return out;
}

/** Roles of an old "gpu" setting, for display: the LAST card of the engine's order is Main, earlier ones Helpers, unlisted cards Off. */
export function fromLegacy(gpu: unknown, cards: RoleCard[], gpuOrder?: string | null): Roles {
  const cs = new Map(cards.map(c => [c.index, c]));
  let items: number[] = Array.isArray(gpu) ? gpu.map(Number) : gpu === null || gpu === undefined || gpu === '' ? [] : String(gpu).split(',').filter(x => x.trim() !== '').map(Number);
  items = items.filter(i => cs.has(i));
  if (!items.length) return { ...recommend(cards).roles };
  if (items.length > 1 && gpuOrder !== 'as_given') {
    const sp = items.map(i => speed(cs.get(i)!));
    if (sp.every(x => x > 0) && Math.max(...sp) >= SAME_SPEED * Math.min(...sp)) items = [...items].sort((a, b) => speed(cs.get(a)!) - speed(cs.get(b)!));
  }
  const roles: Roles = {};
  for (const c of cards) roles[c.index] = 'off';
  for (const i of items.slice(0, -1)) roles[i] = 'helper';
  roles[items[items.length - 1]] = 'main';
  return roles;
}

export const sameRoles = (a: Roles, b: Roles) => { const ka = byNumber(a), kb = byNumber(b); return ka.length === kb.length && ka.every((k, n) => k === kb[n] && a[k] === b[k]); };

export interface RoleWarning { code: 'igpu_main' | 'cannot_run' | 'low_memory' | 'cpu_only'; card: number | null; text: string; fix: { card: number; role: Role } | null }

/** One short line each, only what a person could not know. Never blocking. */
export function warnings(roles: Roles, cards: RoleCard[], opts: { spillGb?: number; engineRunsCpuOnly?: boolean } = {}): RoleWarning[] {
  const cs = new Map(cards.map(c => [c.index, c]));
  const out: RoleWarning[] = [];
  for (const i of byNumber(roles)) {
    const c = cs.get(i), role = roles[i];
    if (!c || role === 'off') continue;
    if (!canRun(c)) out.push({ code: 'cannot_run', card: i, text: `${shortName(c)} cannot run Strata.`, fix: { card: i, role: 'off' } });
    else if (role === 'main' && shares(c) && cards.some(x => !shares(x) && canRun(x))) {
      const real = cards.filter(x => !shares(x) && canRun(x)).sort((a, b) => speed(b) - speed(a) || mem(b) - mem(a) || a.index - b.index)[0];
      out.push({ code: 'igpu_main', card: i, text: `${shortName(c)} is built-in graphics; it is slow as Main.`, fix: { card: real.index, role: 'main' } });
    }
  }
  if (!on(roles).length) out.push({ code: 'cpu_only', card: null, text: opts.engineRunsCpuOnly ? CPU_ONLY_LINE : NO_CARD_LINE, fix: null });
  else if ((opts.spillGb ?? 0) >= 1) out.push({ code: 'low_memory', card: null, text: `Not enough memory for this setup: about ${Math.round(opts.spillGb!)} GB is read from the drive.`, fix: null });
  return out;
}

/** The graphics memory the cards that are on give for the model (an extra card costs its own dense layers); 0 when all are off. */
export function vramOn(roles: Roles, cards: RoleCard[]): number {
  const cs = new Map(cards.map(c => [c.index, c]));
  const onCards = on(roles).map(i => cs.get(i)).filter((c): c is RoleCard => !!c && !shares(c));
  return Math.max(0, onCards.reduce((n, c) => n + mem(c), 0) - SPLIT_CARD_OVERHEAD_GB * Math.max(0, onCards.length - 1));
}

export function noteText(code: string | null, cards: RoleCard[]): string {
  if (!code) return '';
  const [kind, arg] = code.split(':');
  const c = arg !== undefined ? cards.find(x => x.index === Number(arg)) : undefined;
  const name = c ? shortName(c) : 'That card';
  if (kind === 'old_main_helper') return `${name} is now a Helper.`;
  if (kind === 'promoted') return `${name} is now Main.`;
  if (kind === 'only_main') return 'The only card that is on has to be Main.';
  return '';
}

/** The one plain line above the cards. */
export function sentence(cards: RoleCard[] | null | undefined, vendor?: string | null): string {
  const cs = rows(cards, vendor), rec = recommend(cards, vendor);
  if (rec.main === null) return 'No graphics card that Strata can use was found.';
  const byi = new Map(cs.map(c => [c.index, c]));
  const m = byi.get(rec.main)!;
  if (cs.length === 1) return `Strata will use your ${shortName(m)}.`;
  const why = { same: 'the cards are alike', memory: 'it has more memory', faster: 'it is faster', only: 'it is the only card', none: '' }[rec.why];
  let s = `${shortName(m)} is Main: ${why}.`;
  const helpers = byNumber(rec.roles).filter(i => rec.roles[i] === 'helper').map(i => shortName(byi.get(i)!));
  if (helpers.length) s += ` ${helpers.join(' and ')} ${helpers.length > 1 ? 'help' : 'helps'}.`;
  for (const i of byNumber(rec.roles)) if (rec.roles[i] === 'off') {
    const code = rec.reasons[i], c = byi.get(i)!;
    s += ` ${shortName(c)} starts Off: ${code === 'cannot_run' ? 'Strata cannot run it.' : code === 'igpu' ? 'built-in graphics.' : `${OFF_AT_OR_BELOW_GB} GB or less.`}`;
  }
  return s;
}

/** `Main: GPU 2 RTX 4070 Ti. Helper: GPU 1 RTX 3060.` */
export function describe(roles: Roles, cards: RoleCard[]): string {
  const cs = new Map(cards.map(c => [c.index, c]));
  const parts: string[] = [];
  for (const [role, word] of [['main', 'Main'], ['helper', 'Helper'], ['off', 'Off']] as const) {
    const names = byNumber(roles).filter(i => roles[i] === role).map(i => (cs.has(i) ? `GPU ${i + 1} ${shortName(cs.get(i)!)}` : `GPU ${i + 1}`));
    if (names.length) parts.push(`${word}: ${names.join(', ')}.`);
  }
  return parts.join(' ');
}

/** A saved "gpu_roles" object -> roles; null when it is not one. */
export function parseSaved(value: unknown): Roles | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const out: Roles = {};
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    if (!/^\d+$/.test(k.trim()) || !ROLES.includes(String(v).toLowerCase() as Role)) return null;
    out[Number(k)] = String(v).toLowerCase() as Role;
  }
  if (!Object.keys(out).length || Object.values(out).filter(r => r === 'main').length > 1) return null;
  return out;
}
