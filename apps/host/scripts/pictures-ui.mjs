// Needs: a throwaway host (STRATA_UI_BASE, default http://127.0.0.1:18791) and tests/helpers/scripted-worker.mjs listening on 18899 as the model (see ui-drive.mjs ensureMock).
// Pictures in the real app page: paste, "+" and drag-and-drop attach a picture; the OFF switch (per API model) refuses it with a plain message.
import { deflateSync } from 'node:zlib';
import { writeFileSync, mkdirSync } from 'node:fs';
import { ensureMock, open, api } from './ui-drive.mjs';

import { tmpdir } from 'node:os';
import { join } from 'node:path';
const OUT = process.env.PICS_OUT ?? join(tmpdir(), 'pics-ui').split(String.fromCharCode(92)).join('/');
mkdirSync(OUT, { recursive: true });
const crc = (() => { const t = new Int32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c; }); return b => { let c = -1; for (const x of b) c = t[(c ^ x) & 255] ^ (c >>> 8); return (c ^ -1) >>> 0; }; })();
function png(w, h, rgb) {
  const chunk = (type, data) => { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const body = Buffer.concat([Buffer.from(type), data]); const c = Buffer.alloc(4); c.writeUInt32BE(crc(body)); return Buffer.concat([len, body, c]); };
  const raw = Buffer.alloc((w * 3 + 1) * h); for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const o = y * (w * 3 + 1) + 1 + x * 3; raw[o] = rgb[0]; raw[o + 1] = rgb[1]; raw[o + 2] = rgb[2]; }
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}
const file = `${OUT}/blue.png`; writeFileSync(file, png(64, 64, [30, 80, 220]));
const b64 = png(48, 48, [220, 40, 40]).toString('base64');
let failed = false;
const ok = (name, cond, detail = '') => { if (!cond) failed = true; console.log(`${cond ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`); };

const connection = await ensureMock();
const { browser, page } = await open();
await page.locator('.chat-model-picker').first().click();
await page.locator('.model-picker-api').first().click();
await page.waitForTimeout(500);
const photos = () => page.locator('.composer .chat-photos img').count();
const textarea = page.locator('.composer textarea');

// 1. paste from the clipboard (a screenshot arrives as image.png)
await textarea.focus();
await page.evaluate(b64 => {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const data = new DataTransfer(); data.items.add(new File([bytes], 'image.png', { type: 'image/png' }));
  document.querySelector('.composer textarea').dispatchEvent(new ClipboardEvent('paste', { clipboardData: data, bubbles: true, cancelable: true }));
}, b64);
await page.waitForTimeout(700);
ok('paste: the picture is attached', await photos() === 1, `${await photos()} picture(s)`);
ok('paste: it has its own name', /Pasted picture/.test(await page.locator('.composer .chat-photos span').first().innerText()));
await page.screenshot({ path: `${OUT}/1-paste.png` });

// 2. the "+" menu
const chooser = page.waitForEvent('filechooser');
await page.locator('.composer-add').click();
await page.getByText('Add files or photos').click();
(await chooser).setFiles(file);
await page.waitForTimeout(700);
ok('+: the picture is attached', await photos() === 2, `${await photos()} picture(s)`);

// 3. drag and drop onto the conversation
await page.evaluate(b64 => {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const data = new DataTransfer(); data.items.add(new File([bytes], 'dropped.png', { type: 'image/png' }));
  const target = document.querySelector('.chatcol');
  for (const type of ['dragenter', 'dragover', 'drop']) target.dispatchEvent(new DragEvent(type, { dataTransfer: data, bubbles: true, cancelable: true }));
}, b64);
await page.waitForTimeout(700);
ok('drop: the picture is attached', await photos() === 3, `${await photos()} picture(s)`);
await page.screenshot({ path: `${OUT}/2-three.png` });

// 4. send: the model's request carries the pictures
await textarea.fill('what is in the pictures SCRIPT:echo seen');
await page.getByRole('button', { name: 'Send' }).click();
await page.waitForTimeout(6000);
const seen = (await (await fetch('http://127.0.0.1:18899/__requests')).json()).requests;
const withImages = seen.filter(r => (r.raw.match(/"type":"image_url"/g) ?? []).length || (r.raw.match(/"image_url"/g) ?? []).length);
ok('send: the model received the pictures as image input', withImages.length >= 1, `${withImages.length} request(s) with image input`);
await page.screenshot({ path: `${OUT}/3-sent.png` });

// 5. the OFF switch in Settings > Models
await page.getByText('More', { exact: true }).first().click();
await page.getByText('Settings', { exact: true }).first().click();
await page.locator('#settings-tab-Models').click();
await page.locator('#settings-panel-Models').getByRole('button', { name: /Edit/ }).first().click();
const box = page.locator('#settings-panel-Models .provider-selected input[type=checkbox]').first();
await box.scrollIntoViewIfNeeded();
ok('off switch: on by default', await box.isChecked());
await page.screenshot({ path: `${OUT}/4-settings-on.png` });
await box.uncheck();
await page.locator('#settings-panel-Models').getByRole('button', { name: 'Apply' }).first().click();
await page.waitForTimeout(1000);
const saved = (await api('/api/model-providers')).body.connections.find(c => c.id === connection.id).models[0];
ok('off switch: saved as off', saved.vision === false && saved.reads === false, JSON.stringify({ vision: saved.vision, reads: saved.reads }));
await page.keyboard.press('Escape');
await page.waitForTimeout(500);
await page.locator('.composer-add').click().catch(() => {});
await page.keyboard.press('Escape');
await page.evaluate(b64 => {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const data = new DataTransfer(); data.items.add(new File([bytes], 'image.png', { type: 'image/png' }));
  document.querySelector('.composer textarea').dispatchEvent(new ClipboardEvent('paste', { clipboardData: data, bubbles: true, cancelable: true }));
}, b64);
await page.waitForTimeout(700);
const message = await page.locator('.composer .verror, .composer [role=alert]').first().innerText().catch(() => '');
ok('off: a pasted picture is refused with a plain message and a next step', /turned off/.test(message) && /Models/.test(message), message.replace(/\s+/g, ' ').slice(0, 160));
ok('off: nothing was attached', await photos() === 0);
await page.screenshot({ path: `${OUT}/5-off.png` });
await browser.close();
process.exit(failed ? 1 : 0);
