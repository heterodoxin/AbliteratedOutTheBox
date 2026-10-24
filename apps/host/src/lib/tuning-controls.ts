/** Pointer geometry is independent of pixel size and does not wrap at the gap. */
export function dialAngle(x: number, y: number) {
  return Math.atan2(x, -y) * 180 / Math.PI;
}
export function unwrapAngle(next: number, previous: number) {
  let delta = next - previous;
  while (delta > 180) delta -= 360;
  while (delta < -180) delta += 360;
  return previous + delta;
}
export function dialIndex(angle: number, count: number) {
  return (Math.max(-135, Math.min(135, angle)) + 135) / 270 * Math.max(0, count - 1);
}
export function closestStop(value: number, stops: number[]) {
  return stops.reduce((best, stop) => Math.abs(stop - value) < Math.abs(best - value) ? stop : best, stops[0]);
}
