/**
 * A number typed into a settings field (QA report F05: "Max output tokens" took -1 below its minimum of 0 and saved it).
 * The field keeps what the person typed, says in plain words what is wrong, and the save stays off until it is right.
 */
export type NumberCheck = { ok: true; value: number } | { ok: false; problem: 'empty' | 'range' };

/** A whole number between `min` and `max` (both included). Spaces around it are ignored; "1e3", "0.5" and "-" are not whole numbers. */
export function checkWholeNumber(raw: string, min: number, max: number): NumberCheck {
  const text = String(raw ?? '').trim();
  if (!text) return { ok: false, problem: 'empty' };
  if (!/^[+-]?\d+$/.test(text)) return { ok: false, problem: 'range' };
  const value = Number(text);
  if (!Number.isSafeInteger(value) || value < min || value > max) return { ok: false, problem: 'range' };
  return { ok: true, value };
}

/** The sentence under a field that does not hold a good number. `format` writes a number the way the person's language does. */
export function numberProblem(check: NumberCheck, min: number, max: number, format: (n: number) => string): { text: string; args: string[] } | null {
  if (check.ok) return null;
  return check.problem === 'empty' ? { text: 'Enter a number.', args: [] } : { text: 'Enter a whole number from {0} to {1}.', args: [format(min), format(max)] };
}
