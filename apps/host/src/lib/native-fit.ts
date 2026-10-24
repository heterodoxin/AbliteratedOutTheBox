/**
 * What a model version asks of this computer, from the catalog's own figures and the hardware report.
 *
 * The numbers are RAM + VRAM together WITHOUT the n-gram table (it stays on the SSD and is read when needed). The experts are held
 * where there is room: first in the graphics memory beyond about 5 GB (the dense weights, buffers and a 32K cache), then in RAM beyond
 * about 10 GB left for the system; whatever is left is read from the SSD. The same rules setup uses to choose its low-RAM mode
 * (setup.py: low_ram_gpu_gb, LOW_RAM_HEADROOM_GB), so the page and the engine agree.
 */
export interface FitInput { ram_needed_gb?: number; experts_gb?: number | null; memory_gb?: number; mem_total?: boolean }
export interface FitHardware { ram_gb?: number | null; vram_gb?: number | null }
export interface Fit {
  /** RAM + VRAM the version needs (the n-gram table not counted). */
  memoryGb: number;
  /** About what it takes of the graphics memory / of RAM on this PC. */
  vramUseGb: number;
  ramUseGb: number;
  /** About how much is read from the SSD while answering; 0 when it all fits in memory. */
  spillGb: number;
  known: boolean;
}

const DENSE_GB = 5, SYSTEM_ROOM_GB = 10, DENSE_EXTRA_GB = 1;
/** The draft layer (MTP) takes about 1 GB of graphics memory (786 MiB of files + the draft head's ~218 MiB); it comes out of the expert cache while MTP is on (setup.py MTP_VRAM_GB). */
export const MTP_VRAM_GB = 1;
const round1 = (n: number) => Math.round(n * 10) / 10;

export function memoryNeeded(v: FitInput): number {
  if (typeof v.memory_gb === 'number' && Number.isFinite(v.memory_gb)) return v.memory_gb;
  if (v.mem_total && typeof v.ram_needed_gb === 'number') return v.ram_needed_gb;
  return round1((v.experts_gb ?? v.ram_needed_gb ?? 0) + DENSE_EXTRA_GB);
}

export function fitOn(v: FitInput, hw: FitHardware | null | undefined, mtp = true): Fit {
  const memoryGb = memoryNeeded(v);
  const ram = Number(hw?.ram_gb), vram = Number(hw?.vram_gb) || 0;
  if (!Number.isFinite(ram) || ram <= 0) return { memoryGb, vramUseGb: 0, ramUseGb: 0, spillGb: 0, known: false };
  const experts = Math.max(0, memoryNeeded(v) - DENSE_EXTRA_GB);
  const onGpu = Math.max(0, Math.min(experts, vram - DENSE_GB - (mtp && vram > 0 ? MTP_VRAM_GB : 0)));
  const ramRoom = Math.max(0, ram - SYSTEM_ROOM_GB);
  const spillGb = Math.max(0, experts - onGpu - ramRoom);
  const inRam = Math.max(0, experts - onGpu - spillGb);
  const vramUseGb = vram > 0 ? Math.min(vram, DENSE_GB + (mtp ? MTP_VRAM_GB : 0) + onGpu) : 0;
  return { memoryGb, vramUseGb: round1(vramUseGb), ramUseGb: round1(inRam + DENSE_EXTRA_GB * (vram > 0 ? 0 : 1)), spillGb: round1(spillGb), known: true };
}

/** The family groups of the version list, in the order shown. `family` is the engine's family id. */
export const VERSION_GROUPS: { id: string; title: string; families: string[] }[] = [
  { id: 'swift', title: 'Swift models', families: ['swift'] },
  { id: 'gsq', title: 'GSQ-RCO models', families: ['qwen', 'coder'] },
  { id: 'nvfp4', title: 'NVFP4 models', families: ['nvfp4'] },
  { id: 'gyro', title: 'Gyro models', families: ['gyro'] },
  { id: 'unsloth', title: 'Unsloth models', families: ['unsloth'] },
];

/** The fixed "Recommended" picks, in order: engine family + model name + the short note shown beside the name. */
export const RECOMMENDED: { family: string; model: string; note: string }[] = [
  { family: 'gyro', model: 'Gyro-S', note: 'the smallest' },
  { family: 'swift', model: 'IQ2_XS', note: 'good balance' },
  { family: 'qwen', model: 'IQ3_S', note: 'best quality for its size' },
  { family: 'unsloth', model: 'UD-Q4_K_XL', note: 'more quality' },
];

/** Is a group of the version list open? What the person chose wins; until they touch it, only the group holding the selected version is open
 *  (the running version's group used to stay open whatever was clicked). */
export const groupOpen = (opened: Record<string, boolean>, id: string, containsSelected: boolean): boolean => opened[id] ?? containsSelected;

/** What the engine's own pick for this PC is (catalog.recommendation: the memory tiers of setup.py and tools/strata_mcp.py, the same answer `strata launch` shows). */
export interface PcPick { family?: string | null; model?: string | null; why?: string }
export const PICK_NOTE = 'the pick for this PC';
export const OTHER_NOTE = 'the other choice';

/**
 * The recommended set, found in a version list (a version the catalog does not offer is skipped). The four fixed cards are only a display list;
 * the pick comes from the engine's own recommendation for this PC (RAM + VRAM tiers: 40 GB Swift 1.5 IQ2_XS, 48 GB IQ3_XXS, 64 GB and up IQ3_S), so
 * the page and `strata launch` agree. The pick comes first and is marked; when the engine names Gyro-M as the other choice (40 GB tier, NVIDIA and AMD only,
 * which the engine decides) that card follows it. Without a pick the list is the fixed four.
 */
export function recommendedVersions<V extends { sourceFamily: string; model: string }>(versions: V[], pick?: PcPick | null): { r: typeof RECOMMENDED[number]; v: V }[] {
  type Row = { r: typeof RECOMMENDED[number]; v: V };
  const find = (family: string, model: string) => versions.find(v => v.sourceFamily === family && v.model === model);
  const rows: Row[] = RECOMMENDED.map(r => ({ r, v: find(r.family, r.model) })).filter((x): x is Row => !!x.v);
  if (!rows.length || !pick?.family || !pick.model) return rows;
  const chosen = find(pick.family, pick.model);
  if (!chosen) return rows;
  const rest = rows.filter(x => x.v !== chosen);
  const first: Row = { r: { family: pick.family, model: pick.model, note: PICK_NOTE }, v: chosen };
  const out: Row[] = [first];
  const gyroM = /Gyro-M/.test(pick.why ?? '') ? find('gyro', 'Gyro-M') : undefined;
  if (gyroM) out.push({ r: { family: 'gyro', model: 'Gyro-M', note: OTHER_NOTE }, v: gyroM });
  return [...out, ...rest.filter(x => x.v !== gyroM)];
}

/** A card as the installer's hardware report lists it. */
export interface RunCard { index: number; name: string; vram_gb: number; usable?: boolean; vendor?: string; uma?: boolean; integrated?: boolean; unified_gb?: number }

/**
 * The cards a model can be placed on: the ones this engine can drive (strata_control/gpu_fit.py `usable_cards` and `for_vendor`, same rule).
 * A card the engine has no code for, a card of the other make (an engine drives one make; the report numbers each make from 0, so
 * two cards could share a number) and a chip that shares the system memory while the computer has a card of its own are left out.
 */
export function runCards<T extends RunCard>(gpus: T[] | null | undefined, engineVendor?: string | null): T[] {
  const sharesMemory = (g: RunCard) => !!(g.integrated || g.uma || g.unified_gb);
  const mine = (gpus ?? []).filter(g => g.usable !== false && (!engineVendor || !g.vendor || g.vendor === engineVendor));
  const hasReal = mine.some(g => !sharesMemory(g) && g.vram_gb >= 4);
  return mine.filter(g => !(hasReal && sharesMemory(g)));
}
