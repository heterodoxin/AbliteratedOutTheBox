/** Local overrides leave the original bundled/workspace instructions intact. */
export function skillEdit(body) {
  const edit = {};
  for (const [key, max, required] of [['name', 120, true], ['description', 500, false], ['body', 64000, true]]) {
    if (body[key] === undefined) continue;
    if (typeof body[key] !== 'string' || body[key].length > max || (required && !body[key].trim())) throw new Error(`Invalid ${key}: ${required ? 'enter text, ' : ''}maximum ${max} characters.`);
    edit[key] = key === 'body' ? body[key] : body[key].trim();
  }
  if (body.triggers !== undefined) {
    if (!Array.isArray(body.triggers) || body.triggers.length > 32 || body.triggers.some(t => typeof t !== 'string' || t.length > 160)) throw new Error('Use up to 32 trigger phrases, at most 160 characters each.');
    edit.triggers = [...new Set(body.triggers.map(t => t.trim()).filter(Boolean))];
  }
  return edit;
}
