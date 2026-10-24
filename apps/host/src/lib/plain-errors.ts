/**
 * Plain words for failures. The host and Control speak in codes and log lines (`server_exited`, exit codes, "component-missing",
 * a Python message); a person needs what happened and what to do next. Every message the UI shows through ErrorNotice or a toast
 * passes through here: a known failure becomes a sentence and a next step, and the original text stays one click away under
 * "Details" for experts. Strings are fixed so every language has them.
 */
import { tr } from './i18n';

export interface PlainFailure { message: string; next?: string; detail?: string; kind: string }

interface Rule { kind: string; match: RegExp; message: string | ((m: RegExpMatchArray) => string); next?: string; detail?: (m: RegExpMatchArray, raw: string) => string | undefined }

const MEMORY = 'The model stopped while starting, most likely because the PC ran low on memory.';
const MEMORY_NEXT = 'Close other programs and try again, or choose “Leave 20% for my work”. In Expert mode you can also turn on the low-RAM option, so part of the model is read from the SSD.';

/** Order matters: the first rule that matches wins. */
const RULES: Rule[] = [
  // The host names the files it could not find (#3478). The sentence with the names is a template in every language: tr() fills it.
  { kind: 'model-files-missing', match: /Some model files are missing: (.+?)\. Use Download/i, message: m => `These model files are missing: ${m[1]}.`, next: 'Press Download & load again to fetch them. Files that are already there are kept. If the same files are listed again, open Details and share them when you ask for help.' },
  { kind: 'tunnel-own-running', match: /Strata’s tunnel for this address is already running \(PID/i, message: 'Strata’s tunnel for this address is already running.', next: 'Stop it first, or use it as it is, then connect again here.' },
  { kind: 'gpu-shared', match: /GPU \d+ is already used by .*, so this model did not fit there/i, message: 'Another model on the same graphics card left no room for this one.', next: 'Load it on a different GPU, or unload the other model first.' },
  { kind: 'gpu-no-code', match: /none of this computer.s graphics cards has code in this Strata engine|not an architecture this Strata engine was compiled for/i, message: 'This Strata engine has no code for the graphics card it tried to use.', next: 'Pick another card under Run on, or open Details to see which cards were found and what the engine is built for.' },
  { kind: 'engine-too-old', match: /needs a newer engine: .* needs engine/i, message: 'This model needs a newer engine.', next: 'Update the engine (Settings, Engine, Update engine), then try again.' },
  { kind: 'settings-file', match: /settings file for this model has a mistake/i, message: 'Your settings file has a mistake, so the model was not started.', next: 'Open the settings file under Expert settings, fix the line shown under Details, then load the model again.' },
  { kind: 'connector-missing', match: /connector \(cloudflared\) is not installed/i, message: 'Strata could not find Cloudflare’s connector (cloudflared) on this computer.', next: 'Install cloudflared (with your package manager, or from Cloudflare’s website), then press Retry setup. Strata finds it on your PATH and in the usual places.' },
  { kind: 'connector-old', match: /cloudflared at .* is version .* too old/i, message: 'The Cloudflare connector on this computer is too old.', next: 'Update cloudflared to a newer version, then press Retry setup.' },
  { kind: 'connector-broken', match: /did not run as cloudflared/i, message: 'The file found as cloudflared did not run.', next: 'Reinstall cloudflared, then press Retry setup.' },
  { kind: 'picture-reading', match: /vision files|enabling picture reading/i, message: 'Picture reading isn’t available for this model.', next: 'Turn off picture reading and try again.' },
  { kind: 'access', match: /PermissionError|access is denied|can.t open its settings file/i, message: 'Strata can’t open its settings file.', next: 'Run Strata once as yourself (not as administrator) or press Repair.' },
  // A start that died: the exit code or the engine's own last line says why. Say "low on memory" only when that evidence is there
  // (2026-10-17: a missing DLL, 0xC0000135, was reported as "ran low on memory" and told the tester to free RAM).
  { kind: 'dll-missing', match: /0xC0000135|3221225781|-1073741515|STATUS_DLL_NOT_FOUND|error while loading shared libraries|cannot open shared object file|Library not loaded/i, message: 'The engine could not start because one of its program files is missing on this computer.', next: 'Reinstall Strata (the installer includes it) or press Repair. Closing other programs will not help.' },
  { kind: 'dll-bad', match: /0xC000007B|3221225595|-1073741701|0xC0000142|3221225794|0xC000012F|bad image/i, message: 'The engine could not start because one of its program files does not fit this computer or is damaged.', next: 'Reinstall Strata or press Repair. If it keeps happening, update your graphics driver.' },
  { kind: 'cpu-unsupported', match: /0xC000001D|3221225501|illegal instruction|SIGILL/i, message: 'The engine needs a newer processor feature than this computer has.', next: 'Open Details and share them when you ask for help.' },
  { kind: 'crash', match: /0xC0000005|3221225477|0xC0000409|3221226505|segmentation fault|SIGSEGV|exit code 139/i, message: 'The engine crashed while starting.', next: 'Restart the PC and try again. If it keeps happening, update your graphics driver and open Details when you ask for help.' },
  { kind: 'blocked', match: /0xC0000022|3221225506|\bEACCES\b|\bEPERM\b|operation not permitted/i, message: 'Windows or your security software stopped the engine from starting.', next: 'Allow Strata in your security software, then try again.' },
  { kind: 'file-missing', match: /\bENOENT\b|no such file or directory|cannot find the (?:file|path)|0x80070002|exit code 127|file not found/i, message: 'Strata could not find a file it needs to start the engine.', next: 'Reinstall Strata or press Repair.' },
  { kind: 'memory', match: /out of memory|memory guard|\boom\b|\bkilled\b|exit code 137|cannot allocate memory|not enough memory|bad_alloc|pinned-memory-exhausted|mapped staging|cudaErrorMemoryAllocation|insufficient memory/i, message: MEMORY, next: MEMORY_NEXT },
  { kind: 'previous-session', match: /lost host child is still present|retry after it exits/i, message: 'The last Strata session is still closing down.', next: 'Wait a moment and try again. If it keeps happening, restart the computer.' },
  { kind: 'start-failed', match: /server_exited|engine stopped|exited before|died before/i, message: 'The model stopped while starting.', next: 'Try again. If it keeps happening, open Details and share them when you ask for help.' },
  { kind: 'slow-load', match: /loading took more than/i, message: 'Loading is taking too long.', next: 'The PC may be short on memory. Close other programs and try again.' },
  { kind: 'engine-missing', match: /no engine for this pc|engine component|never downloads an engine|engine .*not installed|no strata engine/i, message: 'Strata’s engine for this computer isn’t installed yet.', next: 'It is added when you download a model. Download your model again to add it.' },
  { kind: 'network', match: /cannot reach|could not reach|can't reach|timed out|econn|enotfound|getaddrinfo|urlopen|connection (?:reset|refused|aborted)|network is unreachable/i, message: 'The download couldn’t reach the internet.', next: 'Check your connection, then press Resume download.' },
  { kind: 'integrity', match: /sha-?256|checksum|does not match|did not match|hash mismatch|safety check/i, message: 'A downloaded file didn’t pass Strata’s safety check.', next: 'Press Resume download and Strata will fetch it again.' },
  { kind: 'assistant-missing', match: /agent component|deepseek harness|component-missing|harness .*not (?:installed|found)/i, message: 'The coding assistant isn’t installed on this computer yet.', next: 'Chat works without it.' },
  { kind: 'control', match: /strata control (?:could not start|did not answer|stopped)|control did not answer/i, message: 'Strata couldn’t read its own components.', next: 'Close and reopen Strata. If this keeps happening, reinstall the app.' },
  { kind: 'technical', match: /traceback|\bexit(?:ed)?(?: code)? \d+|errno|ERR_[A-Z_]+|\bat \S+:\d+|\[X\]|\bENOENT\b|\bEPERM\b|\bEACCES\b/i, message: 'Something went wrong.', next: 'Try again. If it keeps happening, open Details and share them when you ask for help.' },
];

/** The message as a person should read it, its next step, and the original text for the Details disclosure. */
export function plainFailure(raw: string | null | undefined): PlainFailure {
  const text = String(raw ?? '').replace(/^\s*Fixture:\s*/i, '').trim();
  for (const rule of RULES) {
    const m = text.match(rule.match);
    if (!m) continue;
    return { kind: rule.kind, message: typeof rule.message === 'function' ? rule.message(m) : rule.message, next: rule.next, detail: rule.detail ? rule.detail(m, text) : text };
  }
  return { kind: 'plain', message: text.replace(/^./, c => c.toUpperCase()) };
}

/** Translated copy of a failure, ready to show. */
export function plainFailureText(raw: string | null | undefined) {
  const f = plainFailure(raw);
  return { ...f, message: tr(f.message), next: f.next ? tr(f.next) : undefined };
}
