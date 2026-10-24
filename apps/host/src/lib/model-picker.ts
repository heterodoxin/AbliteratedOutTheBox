import type { ModelLibrary, ModelVariant } from './types';

export function pickerVersionState(version: ModelVariant, loaded: string | null) {
  const busy = ['downloading', 'loading', 'unloading'].includes(version.state);
  const installed = ['installed', 'loaded', 'loading', 'unloading'].includes(version.state) || !!version.installedAt;
  const running = version.id === loaded && !['loading', 'unloading', 'error'].includes(version.state);
  const labels = { available: 'Download', downloading: `Downloading ${Math.round(version.progress?.pct ?? 0)}%`, installed: 'Downloaded', loading: 'Loading…', loaded: 'Downloaded', unloading: 'Unloading…', error: 'Retry' };
  return { busy, installed, running, label: running ? 'Running' : labels[version.state], action: version.state === 'error' ? installed ? 'Retry loading' : 'Retry download' : installed ? 'Switch version' : 'Download version' };
}
export function pickerVersions(library: ModelLibrary, versions: ModelVariant[]) {
  return [...versions].sort((a, b) => Number(b.id === library.loaded) - Number(a.id === library.loaded) || a.diskGb - b.diskGb);
}

/** Keep downloadable catalog entries in the library; the chat picker uses files already on this PC. */
export function pickerGroups(library: ModelLibrary) {
  const groups = new Map<string, ModelVariant[]>();
  for (const version of pickerVersions(library, library.variants)) {
    if (!pickerVersionState(version, library.loaded).installed) continue;
    const versions = groups.get(version.family) ?? [];
    versions.push(version);
    groups.set(version.family, versions);
  }
  return Array.from(groups, ([family, versions]) => ({ family, versions }));
}
