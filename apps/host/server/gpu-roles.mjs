/**
 * The host's copy of the roles rules it needs to WRITE a model's settings (src/lib/gpu-roles.ts has the screen's, strata_control/gpu_roles.py the
 * Terminal's and the server's; tests/data/gpu_roles_vectors.json is run against all three).
 * Roles are { [card number]: 'main' | 'helper' | 'off' }. The engine takes the cards in stage order: Helpers, then the Main card LAST.
 */
export const ROLES = ['main', 'helper', 'off'];
const mem = c => Number(c.vram_gb) || 0;
const speed = c => (typeof c.speed === 'number' && c.speed > 0 ? c.speed : 0);
const nums = roles => Object.keys(roles).map(Number).sort((a, b) => a - b);
const onList = roles => nums(roles).filter(i => roles[i] !== 'off');
const shortName = c => { let n = String(c.name || `GPU ${c.index + 1}`); for (const j of ['NVIDIA GeForce ', 'NVIDIA ', 'GeForce ', 'AMD Radeon ', 'AMD ', 'Intel(R) ', 'Intel ']) if (n.startsWith(j)) { n = n.slice(j.length); break; } return n.trim(); };

export function parseSaved(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const out = {};
  for (const [k, v] of Object.entries(value)) {
    if (!/^\d+$/.test(k.trim()) || !ROLES.includes(String(v).toLowerCase())) return null;
    out[Number(k)] = String(v).toLowerCase();
  }
  if (!Object.keys(out).length || Object.values(out).filter(r => r === 'main').length > 1) return null;
  return out;
}

export function normalize(roles, cards) {
  const cs = new Map(cards.map(c => [c.index, c]));
  const out = {};
  for (const i of nums(roles)) if (cs.has(i)) out[i] = ROLES.includes(roles[i]) ? roles[i] : 'off';
  const list = onList(out), mains = list.filter(i => out[i] === 'main');
  if (list.length && !mains.length) {
    const best = [...list].sort((a, b) => speed(cs.get(b)) - speed(cs.get(a)) || Math.round(mem(cs.get(b))) - Math.round(mem(cs.get(a))) || a - b)[0];
    out[best] = 'main';
  }
  for (const extra of mains.slice(1)) out[extra] = 'helper';
  if (list.length === 1) out[list[0]] = 'main';
  return out;
}

export function order(roles, cards) {
  const cs = new Map(cards.map(c => [c.index, c]));
  const r = normalize(roles, cards);
  const helpers = nums(r).filter(i => r[i] === 'helper').sort((a, b) => speed(cs.get(a)) - speed(cs.get(b)) || a - b);
  return [...helpers, ...nums(r).filter(i => r[i] === 'main')];
}

/** "gpu_roles" plus the legacy mirror ("gpu" Main last, "gpu_order": "as_given"); nothing on: "gpu": [] and "gpu_none": true. */
export function mirror(roles, cards) {
  const r = normalize(roles, cards), lst = order(r, cards);
  const out = { gpu_roles: Object.fromEntries(nums(r).map(i => [String(i), r[i]])), gpu: [], gpu_cards: {} };
  if (lst.length) { out.gpu = lst.length === 1 ? lst[0] : lst; out.gpu_order = 'as_given'; } else out.gpu_none = true;
  for (const c of cards) if (c.index in r) out.gpu_cards[String(c.index)] = [shortName(c), c.bus ?? ''].filter(Boolean).join('|');
  return out;
}
