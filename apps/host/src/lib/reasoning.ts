/** The newest line of the model's reasoning, trimmed for one compact row (while it is still reasoning, the row shows what it is thinking about). */
export function lastReasoningLine(text: string | undefined, max = 110): string {
  const lines = String(text ?? '').split(/\r?\n/).map(l => l.replace(/^[\s>*#\-•\d.]+/, '').replace(/[*_`]/g, '').trim()).filter(Boolean);
  const line = lines.at(-1) ?? '';
  return line.length > max ? `${line.slice(0, max - 1)}…` : line;
}

/** "38s", "1m 04s": how long the model reasoned. null when it is too short to be worth saying. */
export function thoughtDuration(ms: number | undefined): string | null {
  if (typeof ms !== 'number' || !Number.isFinite(ms) || ms < 1000) return null;
  const total = Math.round(ms / 1000);
  return total < 60 ? `${total}s` : `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, '0')}s`;
}

/** Is a scrolling box at its bottom (within a few pixels)? Reasoning sticks to the bottom only while the person has not scrolled up. */
export function atBottom(el: { scrollHeight: number; scrollTop: number; clientHeight: number }, slack = 24): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= slack;
}
