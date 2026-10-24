import type { ConfigDocument } from './config';
import type { EngineConfig, Hardware, ImageModelChoice } from './types';

export type ComputerUse = 'shared' | 'full';
export interface RunProfile {
  id: string;
  name: string;
  variantId: string;
  hardware: string;
  engineVersion: string;
  config: ConfigDocument;
  imageModel?: ImageModelChoice;
  updatedAt: string;
}
export interface RunProfiles {
  schema: 1;
  profiles: RunProfile[];
  defaults: Record<string, string>;
}
export const RUN_PROFILES_KEY = 'strata.run-profiles.v1';
export const emptyProfiles = (): RunProfiles => ({ schema: 1, profiles: [], defaults: {} });

export function computerIdentity(hardware: Hardware) {
  return JSON.stringify({
    host: hardware.host.name, platform: hardware.host.platform, arch: hardware.host.arch,
    cpu: hardware.cpu.model, cores: hardware.cpu.cores, ram: hardware.memory.totalGb,
    processors: hardware.processors?.map(p => [p.socket, p.model, p.cores]),
    gpus: hardware.gpus.map(g => [g.id ?? g.index, g.name, g.vramGb, g.backend]),
    backend: hardware.accelerator.backend,
  });
}
export function defaultProfileKey(hardware: string, variantId: string) { return `${hardware}\n${variantId}`; }
export function configIdentity(config: ConfigDocument): string {
  const canonical = (v: unknown): unknown => Array.isArray(v) ? v.map(canonical) : v && typeof v === 'object'
    ? Object.fromEntries(Object.entries(v).sort(([a], [b]) => a.localeCompare(b)).map(([k, item]) => [k, canonical(item)])) : v;
  return JSON.stringify(canonical(config));
}
export function readProfiles(text: string | null): RunProfiles {
  if (!text) return emptyProfiles();
  const parsed = JSON.parse(text);
  if (parsed?.schema !== 1 || !Array.isArray(parsed.profiles) || !parsed.defaults || typeof parsed.defaults !== 'object' || Array.isArray(parsed.defaults)) throw new Error('Saved run profiles could not be read.');
  const profiles = parsed.profiles.filter((p: RunProfile) => p && typeof p.id === 'string' && typeof p.name === 'string' && typeof p.variantId === 'string' && typeof p.hardware === 'string' && typeof p.engineVersion === 'string' && p.config?.schema === 'strata.engine/1' && p.config.beginner && p.config.expert && !Array.isArray(p.config.expert)).map((p: RunProfile) => ({ ...p, imageModel: ['off','q4','q5','q6','q8'].includes(p.imageModel ?? '') ? p.imageModel : 'off' }));
  const defaults = Object.fromEntries(Object.entries(parsed.defaults).filter(([, id]) => typeof id === 'string' && profiles.some((p: RunProfile) => p.id === id))) as Record<string, string>;
  return { schema: 1, profiles, defaults };
}
export function saveProfile(store: RunProfiles, profile: RunProfile, makeDefault: boolean): RunProfiles {
  const snapshot = structuredClone(profile);
  const next = { ...store, profiles: [...store.profiles.filter(p => p.id !== profile.id), snapshot], defaults: { ...store.defaults } };
  const key = defaultProfileKey(profile.hardware, profile.variantId);
  if (makeDefault) next.defaults[key] = profile.id;
  else if (next.defaults[key] === profile.id) delete next.defaults[key];
  return next;
}
export function resourceConfig(config: ConfigDocument, use: ComputerUse, hardware: Hardware): ConfigDocument {
  const cores = Math.max(1, hardware.cpu.cores);
  return {
    ...config, headroom: use,
    expert: { ...config.expert, 'perf.threads': use === 'full' ? cores : Math.max(1, cores - Math.max(2, Math.ceil(cores * .2))) },
  };
}
export function recommendedConfig(engine: EngineConfig, hardware: Hardware, context = 'recommended', use: ComputerUse = 'shared'): ConfigDocument {
  return resourceConfig({
    schema: 'strata.engine/1', batchAutomatic: true,
    beginner: { priority: 'balanced', contextLength: context as ConfigDocument['beginner']['contextLength'], readImages: false, gpuIdle: 'never' },
    expert: { ...Object.fromEntries(engine.expert.map(p => [p.id, p.recommended])), 'ctx.length': context === 'recommended' ? '128K' : context.toUpperCase() },
  }, use, hardware);
}
