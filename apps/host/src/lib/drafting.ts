/**
 * The Drafting choices the Models screen offers for a model. DFlash2 is for Qwen3.8-Flash-Next only (the qwen and gyro families) AND only while
 * the engine serves it: strata_control/drafting.py ENGINE_SERVES_DFLASH2 (Control reports it as `capabilities.dflash2`). Until then the setting
 * is Auto / MTP / Off; DFlash2 appears by itself when the engine can run it. The Terminal's --drafting flag has it either way, gated with a note.
 */
export const DRAFTING_FAMILIES = ['qwen', 'gyro'];

export function draftingChoices(family: string | null | undefined, dflash2Served: boolean | null | undefined): string[] {
  return DRAFTING_FAMILIES.includes(String(family)) && dflash2Served === true ? ['auto', 'mtp', 'dflash2', 'off'] : ['auto', 'mtp', 'off'];
}
