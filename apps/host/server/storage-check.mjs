/**
 * What changing (or only checking) the model folder does to the models that are loaded.
 *
 * "Check folder" reads the SAME folder again, so nothing that is loaded may be touched: it used to unload every optimized model
 * and the status went to "No model loaded" (QA 2026-10-18, O01). Only an actual change of folder unloads them, and only when
 * the person agreed (the screen says so first and offers "Unload and change directory").
 */
export function nativeUnloadPlan({ changed, unload, instances }) {
  const running = Array.isArray(instances) ? instances : [];
  if (!running.length || !changed) return { refuse: null, ids: [] };
  if (unload !== true) return { refuse: 'Unload native models before changing the model folder.', ids: [] };
  return { refuse: null, ids: running.map(i => i.id) };
}
