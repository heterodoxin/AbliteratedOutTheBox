import { api } from './client';

/** A saved tuning, as Strata Control reports it (the same store the Terminal uses). */
export interface TunePreset {
  id: string;
  name: string;
  active: boolean;
  created?: string;
  answers?: { class: 'exact' | 'placement' | 'nonrepeatable' | 'quality'; fields: string[]; default_repeatable?: boolean };
  settings?: { args: Record<string, unknown>; env: Record<string, unknown>; per_request: Record<string, unknown> };
  gain_pct?: number | null;
  decode_tok_s?: number | null;
  baseline_decode_tok_s?: number | null;
  recheck?: { last_ok?: string | null; last_result?: { verdict?: string; reason?: string } | null } | null;
  slot?: string;
}

export interface TuneStatus {
  ok: boolean;
  slot: string;
  suggested_name: string;
  active: TunePreset | null;
  /** fresh, soft (a short check is offered) or hard (the settings are not applied; tune again). */
  state: 'fresh' | 'soft' | 'hard' | null;
  reasons: string[];
  presets: TunePreset[];
  can_undo: boolean;
  /** the tuning was switched off for these choices (it is kept: last_active can be switched on again) */
  switched_off?: boolean;
  last_active?: TunePreset | null;
}

export interface TuneProgress { stage: string; text: string | null; minutesLeft: number | null; runsDone: number; runsPlanned: number; elapsedS: number; step: string; /** each measurement so far, as decode tokens per second */ speeds?: { label: string; tok_s: number }[] }

export interface TuneResult {
  summary: string;
  outcome?: 'improved' | 'default_confirmed' | 'inconclusive' | 'base_failed' | 'cancelled';
  message?: string;
  gain_pct?: number;
  seconds?: number;
  pairs?: number;
  faster_pairs?: number;
  base?: { decode_tok_s?: number | null };
  chosen?: { decode_tok_s?: number | null };
  answers?: { class: 'exact' | 'placement' | 'nonrepeatable' | 'quality'; fields: string[]; default_repeatable?: boolean };
  preset?: { id: string; name: string } | null;
  notes?: string[];
  /** what the screen shows: faster | already_fastest | busy_pc | failed | stopped */
  code?: 'faster' | 'already_fastest' | 'busy_pc' | 'failed' | 'stopped';
  /** busy_pc: noisy | drift | not_tried; failed: start_failed | no_memory */
  reason_code?: string;
  values?: { gain_pct?: number; gain_decode_pct?: number; base_tok_s?: number | null; tuned_tok_s?: number | null; noise_pct?: number | null; drift_pct?: number | null; minutes?: number; pairs?: number; faster_pairs?: number; preset_name?: string | null };
  verdict?: 'keep' | 'retune' | 'inconclusive';
  reason?: string;
}

const q = (o: Record<string, string | undefined>) => Object.entries(o).filter(([, v]) => v).map(([k, v]) => `${k}=${encodeURIComponent(v as string)}`).join('&');

export const tuneStatus = (model: string, resourceProfile: string, vision: string) =>
  api.get<TuneStatus>(`/api/native/tune/status?${q({ model, resourceProfile, vision })}`);
export const tunePresetAction = (id: string, action: 'apply' | 'rename' | 'delete' | 'undo', body: Record<string, unknown> = {}) =>
  api.post(`/api/native/tune/presets/${id}/${action}`, body);
export const tuneOff = (model: string, resourceProfile: string, vision: string) =>
  api.post('/api/native/tune/off', { model, resourceProfile, vision });
/** What a start will really run: the saved tuning's values on top of the configuration (Expert mode shows these). */
export const tuneEffective = (model: string, resourceProfile: string, vision: string) =>
  api.get<{ applied: boolean; preset: { id: string; name: string } | null; args: string[]; env: Record<string, string>; changed: { flag: string; was: string | null; now: string | null }[]; why_not?: string | null }>(`/api/native/tune/effective?${q({ model, resourceProfile, vision })}`);
export const tuneUndo = (model: string, resourceProfile: string, vision: string) =>
  api.post('/api/native/tune/undo', { model, resourceProfile, vision });

/** The four answer classes, in the words the app uses (the Terminal prints the same sentences). */
export const ANSWER_NOTICE: Record<string, string> = {
  exact: 'Answers are identical to the standard setup.',
  placement: 'Answers may differ slightly from the standard setup.',
  nonrepeatable: 'Answers are not repeatable run to run.',
  quality: 'Includes a quality setting that changes the numbers.',
};

/** Where the standard setup is itself not repeatable (RTX 3060 in low-RAM mode, Arc A750), an exact setting cannot promise identical answers. */
export const answerNotice = (cls?: string, defaultRepeatable?: boolean) =>
  cls === 'exact' && defaultRepeatable === false ? 'Answers can differ from run to run on this PC even with the standard setup; this setting does not add to that.' : ANSWER_NOTICE[cls ?? ''] ?? '';
