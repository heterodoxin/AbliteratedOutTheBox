/** Numbers that move: the pure parts of a short count-up / count-down between two readings. */
export const easeOutCubic = (t: number) => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);

/** The value between `from` and `to` after `elapsed` of `duration` milliseconds. */
export function tweenValue(from: number, to: number, elapsed: number, duration: number): number {
  if (!Number.isFinite(from) || !Number.isFinite(to) || duration <= 0) return to;
  return from + (to - from) * easeOutCubic(elapsed / duration);
}
