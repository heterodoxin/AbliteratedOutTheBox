/** Locate catalog weights in a Strata folder or directly inside an existing GGUF folder. */
import { closeSync, existsSync, openSync, readSync, realpathSync, statSync } from 'node:fs';
import { isAbsolute, join, relative } from 'node:path';

export function matchesGGUF(file, bytes, directory) {
  let fd;
  try {
    if (!existsSync(file)) return false;
    const path = relative(realpathSync(directory), realpathSync(file));
    if (path.startsWith('..') || isAbsolute(path)) return false;
    const stat = statSync(file);
    if (!stat.isFile() || stat.size !== bytes) return false;
    fd = openSync(file, 'r');
    const magic = Buffer.alloc(4); readSync(fd, magic, 0, 4, 0);
    return magic.toString() === 'GGUF';
  } catch { return false; }
  finally { if (fd !== undefined) closeSync(fd); }
}

export function locateModelFile(directory, model) {
  const nested = join(directory, model.id, model.source.file);
  if (matchesGGUF(nested, model.source.bytes, directory)) return nested;
  const flat = join(directory, model.source.file);
  // The design catalog deliberately has ambiguous filenames across model families.
  if (!model.designOnly && matchesGGUF(flat, model.source.bytes, directory)) return flat;
  return null;
}
