/** One conservative policy for operations affecting the shared loaded model. */
export function lifecyclePolicy(input: {
  connected: boolean;
  streaming: boolean;
  stats?: { running?: number; waiting?: number } | null;
  busy: Record<string, boolean | undefined>;
}) {
  const { running, waiting } = input.stats ?? {};
  if (!input.connected || !Number.isFinite(running) || !Number.isFinite(waiting))
    return { allowed: false, reason: 'Request status unavailable. Reconnect before changing the loaded model.' };
  if (input.streaming || running! > 0 || waiting! > 0)
    return { allowed: false, reason: `${running! > 0 ? running : input.streaming ? 'A' : 0} active · ${waiting} queued. Finish or cancel requests first.` };
  if (Object.entries(input.busy).some(([key, value]) => value && (key === 'restart' || key === 'unload' || key === 'modelDirectory' || key.startsWith('load:'))))
    return { allowed: false, reason: 'A model operation is in progress.' };
  return { allowed: true, reason: '' };
}
