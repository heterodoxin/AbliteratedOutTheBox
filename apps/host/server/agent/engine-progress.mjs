/**
 * How far a local engine has got, as a short text that changes whenever it makes progress. The agent's stall watchdog
 * compares it with the last one it saw: while a long prompt is read the Harness hears nothing for minutes, but the
 * engine's /metrics (polled every 2 s) shows `live.prompt_read` moving (#3623). Null when the engine is idle or says nothing.
 */
export function engineProgressMark(instance) {
  const live = instance?.metrics?.live;
  if (!live || live.state === 'idle' || live.state === 'unloaded') return null;
  const slots = Array.isArray(live.slots) ? live.slots.map(s => `${s?.state}:${s?.prompt_read ?? ''}:${s?.generated ?? ''}`).join(',') : '';
  return `${live.state}|${live.prompt_read ?? ''}|${live.generated ?? ''}|${slots}`;
}
