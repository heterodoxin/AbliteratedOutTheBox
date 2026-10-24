/**
 * A short name for a downloaded optimized model: the family and the quantization ("Swift IQ2_XS", "Gyro-S", "IQ3_S"), made from the id the
 * installation gives it (the family tag in front of the lower-cased quantization). The long name stays available as a tooltip.
 */
const FAMILIES: [string, string][] = [['swift-', 'Swift'], ['coder-', 'Coder'], ['unsloth-', 'Unsloth'], ['gyro-gyro-', 'Gyro'], ['gyro-', 'Gyro'], ['nvfp4-', 'NVFP4'], ['qwen36-', 'Qwen3.6']];

/**
 * The three NVFP4 versions, named by who made the weights (the plain model names NVFP4 / NVFP4-NVIDIA / NVFP4-QAD looked like the same thing):
 *  - NVFP4: Strata's own GPTQ conversion of Qwen's weights, the best quality of the three;
 *  - NVFP4-NVIDIA: NVIDIA's own NVFP4 checkpoint (plain rounding), converted for Strata, the same size and speed;
 *  - NVFP4-QAD: NVIDIA's FP4 checkpoint that was trained for FP4 (quantization-aware distillation), slightly faster on short prompts.
 * `note` is the one line that says how they differ.
 */
export const NVFP4_VERSIONS: Record<string, { name: string; note: string }> = {
  'NVFP4': { name: 'NVFP4 (Strata’s version)', note: 'Strata’s own conversion. The best quality of the NVFP4 versions.' },
  'NVFP4-NVIDIA': { name: 'NVFP4 (NVIDIA’s version)', note: 'NVIDIA’s own NVFP4 checkpoint, converted for Strata. Same size and speed.' },
  'NVFP4-QAD': { name: 'NVFP4 (NVIDIA, trained for FP4)', note: 'NVIDIA’s FP4 checkpoint, trained for FP4. Slightly faster on short prompts.' },
};
export const nvfp4Version = (model: string) => NVFP4_VERSIONS[String(model ?? '').toUpperCase()];

export function nativeShortName(id: string): string {
  const lower = String(id ?? '').toLowerCase();
  for (const [prefix, label] of FAMILIES) {
    if (!lower.startsWith(prefix)) continue;
    const rest = lower.slice(prefix.length).toUpperCase();
    if (!rest) return label;
    if (label === 'NVFP4') return nvfp4Version(rest)?.name ?? `${label} ${rest}`;
    return label === 'Gyro' ? `Gyro-${rest.replace(/^GYRO-/, '')}` : `${label} ${rest}`;
  }
  return lower.toUpperCase() || String(id);
}
