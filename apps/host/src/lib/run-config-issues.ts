/** The checks of a hand-edited run config (server/run-config-check.mjs) as sentences in the person's language. */
import { tr } from './i18n';

export interface RunConfigIssue { level: 'error' | 'warning'; code: string; key: string | null; line: number | null; params: Record<string, string | number>; message: string }
export interface SettingsFileInfo { model: string; path: string; name: string; ok: boolean; issues: RunConfigIssue[]; note: string }

/** One issue as a sentence. Every sentence is a fixed string so every language has it; names and numbers are slots. */
export function issueText(issue: RunConfigIssue): string {
  const p = issue.params;
  const text = (() => {
    switch (issue.code) {
      case 'json': return p.line ? tr('The settings file is not valid JSON; look near line {0}. Check for a missing comma between two entries, a comma after the last entry, or a missing quote.', [p.line]) : tr('The settings file is not valid JSON. Check for a missing comma between two entries, a comma after the last entry, or a missing quote.');
      case 'not-object': return tr('The settings file must hold one object, { … }, with the settings inside.');
      case 'unknown-key-near': return tr('"{0}" is not a setting Strata reads, so it is ignored. Did you mean "{1}"?', [p.key, p.near]);
      case 'unknown-key': return tr('"{0}" is not a setting Strata reads, so it is ignored.', [p.key]);
      case 'args-type': return tr('"args" must be a list of quoted text entries, for example ["--ple-io", "mmap"]. Numbers need quotes too.');
      case 'args-managed': return tr('Strata manages the network address, port and API key itself, so --host, --port and --api-key in "args" are not honoured.');
      case 'env-type': return tr('"env" must be an object of STRATA_ names with quoted text values.');
      case 'env-value': return tr('"env" entry {0} must be quoted text, for example "1".', [p.name]);
      case 'sampling-type': return tr('"sampling" must be an object such as { "temperature": 0.7 }.');
      case 'bool': return tr('"{0}" must be true or false, not {1}.', [p.key, p.value ?? '']);
      case 'number-range': return tr('"{0}" must be a number from {1} to {2}, not {3}.', [p.key, p.min, p.max, p.value]);
      case 'whole-range': return tr('"{0}" must be a whole number from {1} to {2}, not {3}.', [p.key, p.min, p.max, p.value]);
      case 'choice': return tr('"{0}" must be one of {1}, not {2}.', [p.key, p.choices, p.value]);
      case 'text': return tr('"{0}" must be short text, or a list of short names.', [p.key]);
      case 'layer-split': return tr('"layer_split" points must increase from 1 to 47, for example "16,32", or be "auto".');
      case 'gpu': return tr('"gpu" must be a graphics card number such as 0, or a list such as [0, 1].');
      default: return issue.message;
    }
  })();
  return issue.line ? tr('Line {0}: {1}', [issue.line, text]) : text;
}
