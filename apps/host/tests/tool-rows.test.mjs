/** Item 3: the tool rows in a chat are compact lines with no divider between them (they used to inherit the settings disclosure's top border and 18px padding). */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const css = await readFile(new URL('../src/styles/app.css', import.meta.url), 'utf8');
function lastRule(selector) {
  const at = css.lastIndexOf(`\n${selector} {`);
  if (at < 0) return '';
  return css.slice(at, css.indexOf('}', at));
}

test('a tool row has no top border and little padding', () => {
  const rule = lastRule('.tool-summary');
  assert.match(rule, /border-top:\s*0/);
  const pad = /padding-block:\s*(\d+)px/.exec(rule);
  assert.ok(pad && Number(pad[1]) <= 4, `padding-block ${pad?.[1]}`);
});

test('the chat renders tool rows as a disclosure that the compact rule overrides', async () => {
  const chat = await readFile(new URL('../src/screens/Chat.tsx', import.meta.url), 'utf8');
  assert.match(chat, /className="disclosure tool-summary"/);
});
