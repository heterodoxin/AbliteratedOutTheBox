/** What a saved tuning changes in the engine's start-up flags (GET /api/native/models/<id>/settings -> tuned.changed), as the Expert mode fields show it. */
export interface TunedChange { flag: string; was: string | null; now: string | null }
export interface TunedFields { kv?: string; lowRam?: 'auto' | 'resident' | 'mmap'; vramReserveMib?: number }

export function tunedFields(changed: TunedChange[] | undefined | null): TunedFields {
  const out: TunedFields = {};
  for (const c of changed ?? []) {
    if (c.flag === '--kv' && c.now) out.kv = c.now;
    else if (c.flag === '--resident-experts') out.lowRam = c.now !== null ? 'resident' : c.was !== null ? 'auto' : out.lowRam;
    else if (c.flag === '--mmap-experts') out.lowRam = c.now !== null ? 'mmap' : c.was !== null ? 'auto' : out.lowRam;
    else if (c.flag === '--vram-reserve-mib' && c.now !== null && Number.isFinite(Number(c.now))) out.vramReserveMib = Number(c.now);
  }
  return out;
}

/** "--spec 6 (was 4)" for the list of what the tuning changed. */
export function describeChange(c: TunedChange): { name: string; now: string; was: string | null } {
  return { name: c.flag.replace(/^--/, '').replace(/-/g, ' '), now: c.now ?? '', was: c.was };
}
