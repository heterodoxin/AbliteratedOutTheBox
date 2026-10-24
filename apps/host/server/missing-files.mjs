/**
 * Which files of a model are missing, in words a person can act on (#3478). Control's model list says which required parts of the run
 * config are absent (`missing`: part + file name), and the installer's status says which GGUF files and shards it could not find
 * (`missing_files`, full paths). Together they give the names; the app shows them instead of only "some files are missing".
 */
import { basename } from 'node:path';

const PART_WORDS = { engine: 'the engine program', 'tokenizer/vocab.json': 'the tokenizer', 'tokenizer/merges.txt': 'the tokenizer', 'tokenizer/token_type.json': 'the tokenizer' };

/** Short readable names, no duplicates, most useful first. */
export function missingNames(controlParts, installerFiles) {
  const names = [];
  const add = text => { const t = String(text ?? '').trim(); if (t && !names.includes(t)) names.push(t); };
  for (const file of Array.isArray(installerFiles) ? installerFiles : []) add(basename(String(file).split(String.fromCharCode(92)).join('/')));
  for (const item of Array.isArray(controlParts) ? controlParts : []) {
    add(PART_WORDS[item?.part] ?? item?.file ?? item?.part);
  }
  return names;
}

/** The error a load or a tune raises for a model that is not whole. Names are in the sentence (at most four) and all of them in `details`. */
export function missingFilesError(model, names) {
  const list = names ?? [];
  const shown = list.slice(0, 4).join(', ') + (list.length > 4 ? ` and ${list.length - 4} more` : '');
  const error = new Error(list.length
    ? `Some model files are missing: ${shown}. Use Download / repair to restore them.`
    : 'Some model files are missing. Use Download / repair to restore them.');
  error.details = [list.length ? `Missing for ${model}: ${list.join(', ')}` : `Missing for ${model}: Strata could not tell which files`];
  return error;
}
