/**
 * Does the installed engine read a model's files? The catalog (tools/strata_mcp.py, from setup.py's MODELS) names the first engine
 * that does (`engine_min`, e.g. Gyro: 0.1.45.1, because its expert files are native_experts v5 and the released 0.1.45 stops at v4).
 * An older engine would start the download and then fail on the files, so the app says "needs a newer engine" first.
 */
export function parseEngineVersion(value) {
  const m = /^\s*v?(\d+(?:\.\d+){0,3})/.exec(String(value ?? ''));
  return m ? m[1].split('.').map(Number) : null;
}

/** True when `installed` is known and older than `minimum`. An engine built from this checkout ('local') or one whose version cannot be read is never called too old. */
export function engineTooOld(installed, minimum, source = null) {
  if (!minimum || source === 'local') return false;
  const have = parseEngineVersion(installed), need = parseEngineVersion(minimum);
  if (!have || !need) return false;
  for (let i = 0; i < 4; i++) {
    const a = have[i] ?? 0, b = need[i] ?? 0;
    if (a !== b) return a < b;
  }
  return false;
}

/** The catalog with `needsNewerEngine` on every size the installed engine cannot read. */
export function markEngine(catalog, engine) {
  if (!catalog || !Array.isArray(catalog.families)) return catalog;
  const version = engine?.version ?? null, source = engine?.source ?? null;
  return { ...catalog, families: catalog.families.map(f => ({ ...f, sizes: (f.sizes ?? []).map(s => ({ ...s, needsNewerEngine: engineTooOld(version, s.engine_min, source) })) })) };
}

/** The plain sentence for a refused install or load, or null when the engine is new enough. */
export function engineRefusal(catalog, modelId, engine) {
  const size = catalog?.families?.flatMap(f => f.sizes ?? []).find(s => s.id === modelId);
  if (!size?.needsNewerEngine) return null;
  return `This model needs a newer engine: ${size.model} needs engine ${size.engine_min} or newer, and this one is ${engine?.version}. Update the engine in Settings, then try again.`;
}
