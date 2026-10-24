/**
 * A message too large for the model's window, made small enough to send (agent/context-fit.mjs decides that it is too large).
 * The person picks how, on the error card:
 *   file     the whole text is saved in the chat's folder and the message names the file; the assistant reads it in parts
 *   preview  the start and the end of the text, cut to what the window can hold
 * Nothing is sent from here: the page sends the returned message as the person's next message.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { countTokens, oversizedReplacement } from './agent/context-fit.mjs';

const stamp = d => `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}-${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}${String(d.getSeconds()).padStart(2, '0')}`;

/**
 * @param {{mode:'file'|'preview', text:string, root:string|null, instance?:{port:number, key?:string, apiModelId?:string, context?:number, metrics?:any}|null, now?:Date, fetchImpl?:typeof fetch}} input
 * @returns {Promise<{message:string, mode:string, tokens:number|null, kept:number, path?:string}>}
 */
export async function prepareOversized({ mode, text, root, instance = null, now = new Date(), fetchImpl = fetch }) {
  const window = instance?.metrics?.engine?.max_context ?? instance?.context ?? 32768;
  const tokens = instance ? await countTokens({ baseUrl: `http://127.0.0.1:${instance.port}/v1`, key: instance.key ?? '', model: instance.apiModelId ?? '', text, fetchImpl }) : null;
  const counted = tokens ?? Math.ceil(text.length / 3);   // not countable: three characters a token is on the safe side for code and logs
  if (mode === 'file') {
    if (!root) throw new Error('This chat has no folder yet. Choose one first.');
    mkdirSync(root, { recursive: true });
    const path = join(root, `pasted-message-${stamp(now)}.txt`);
    writeFileSync(path, text, { encoding: 'utf8', flag: 'wx' });
    return { mode, tokens, path, ...oversizedReplacement({ text, mode, tokens: counted, window, path }) };
  }
  return { mode: 'preview', tokens, ...oversizedReplacement({ text, mode: 'preview', tokens: counted, window }) };
}
