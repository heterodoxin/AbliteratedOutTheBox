/** v1 had Computer / Model / Tune / Prepare / Ready. v2 folds Tune into Model. */
export function migrateSetup(setup = {}) {
  if (setup.flowVersion === 2) return setup;
  const oldStep = Math.max(1, Math.min(5, Number(setup.step) || 1));
  return { ...setup, flowVersion: 2, step: [0, 1, 2, 2, 3, 4][oldStep] };
}
