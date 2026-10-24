import type { TuneResult } from './native-tune';

/**
 * One layout for every way a tune can end: a tone (the round tile's colour and icon), a headline, one plain sentence, up to three facts, and
 * the buttons that make sense. The host says which outcome it was (`code`, `reason_code`, `values`, see strata_control/autotune.py); older
 * answers without a code are read from `outcome` and `verdict`. All words go through the catalog in the component, not here.
 */
export type OutcomeCode = 'faster' | 'already_fastest' | 'busy_pc' | 'failed' | 'stopped' | 'still_fast' | 'slower' | 'unsure';
export type Tone = 'ok' | 'warn' | 'bad' | 'neutral';

export interface Outcome {
  code: OutcomeCode;
  tone: Tone;
  icon: 'check' | 'clock' | 'x' | 'stop';
  /** the reason within the outcome (busy_pc: noisy | drift | not_tried; failed: start_failed | no_memory | other) */
  reason: string;
  gainPct: number | null;
  baseTokS: number | null;
  tunedTokS: number | null;
  noisePct: number | null;
  minutes: number | null;
  presetName: string | null;
  /** the person can try again straight away (nothing is running) */
  retry: boolean;
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

export function readOutcome(result: TuneResult | null | undefined, jobState: 'done' | 'error' | 'cancelled' | 'running' | undefined): Outcome {
  const v = result?.values ?? {};
  const facts = {
    gainPct: num(v.gain_pct) ?? num(result?.gain_pct),
    baseTokS: num(v.base_tok_s) ?? num(result?.base?.decode_tok_s),
    tunedTokS: num(v.tuned_tok_s) ?? num(result?.chosen?.decode_tok_s),
    noisePct: num(v.noise_pct), minutes: num(v.minutes),
    presetName: v.preset_name ?? result?.preset?.name ?? null,
  };
  const make = (code: OutcomeCode, tone: Tone, icon: Outcome['icon'], reason = '', retry = true): Outcome => ({ code, tone, icon, reason, retry, ...facts });
  if (jobState === 'cancelled') return make('stopped', 'neutral', 'stop');
  if (jobState === 'error' && !result) return make('failed', 'bad', 'x', 'other');
  const code = result?.code;
  if (code === 'faster') return make('faster', 'ok', 'check');
  if (code === 'already_fastest') return make('already_fastest', 'ok', 'check');
  if (code === 'busy_pc') return make('busy_pc', 'warn', 'clock', result?.reason_code ?? 'noisy');
  if (code === 'failed') return make('failed', 'bad', 'x', result?.reason_code ?? 'other');
  if (code === 'stopped') return make('stopped', 'neutral', 'stop');
  // answers from a check of a saved tuning
  if (result?.verdict === 'keep') return make('still_fast', 'ok', 'check');
  if (result?.verdict === 'retune') return make('slower', 'warn', 'clock');
  if (result?.verdict === 'inconclusive') return make('unsure', 'warn', 'clock');
  // answers without a code (an older Control)
  switch (result?.outcome) {
    case 'improved': return make('faster', 'ok', 'check');
    case 'default_confirmed': return make('already_fastest', 'ok', 'check');
    case 'inconclusive': return make('busy_pc', 'warn', 'clock', 'noisy');
    case 'cancelled': return make('stopped', 'neutral', 'stop');
    case 'base_failed': return make('failed', 'bad', 'x', 'other');
    default: return make(jobState === 'error' ? 'failed' : 'unsure', jobState === 'error' ? 'bad' : 'warn', jobState === 'error' ? 'x' : 'clock', jobState === 'error' ? 'other' : '');
  }
}

export const tokS = (n: number | null): string => (n == null ? '—' : `${(Math.round(n * 10) / 10).toFixed(1)}`);
