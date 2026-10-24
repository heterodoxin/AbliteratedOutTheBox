import type { Session } from './types';
export type UsagePeriod = 'all' | '30d' | '7d';
export const dayKey = (date: Date) => `${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,'0')}-${String(date.getDate()).padStart(2,'0')}`;
const validCount = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n) && n >= 0;

/** Saved transcript counts; token totals include only usage reported by an engine. */
export function usageDashboard(sessions: Session[], period: UsagePeriod, now = new Date()) {
  const first = new Date(now); first.setHours(0,0,0,0);
  if (period !== 'all') first.setDate(first.getDate() - (period === '7d' ? 6 : 29));
  const cutoff = period === 'all' ? 0 : first.getTime();
  const days = new Map<string, { messages:number; models:Record<string,number> }>();
  const models = new Map<string, { id:string; input:number; output:number; requests:number }>();
  const seen = new Set<string>(); const hours = Array<number>(24).fill(0);
  let messages = 0, sessionCount = 0, input = 0, output = 0, unreported = 0;
  for (const session of sessions) {
    if (session.showcase) continue;
    let active = false;
    for (const message of session.messages) {
      if (message.role === 'system' || message.example || message.timeSkipId || message.streaming || !message.content.trim()) continue;
      const date = new Date(message.at), time = date.getTime();
      if (!Number.isFinite(time) || time < cutoff || time > now.getTime()) continue;
      active = true;
      // Branches retain the source timestamp/content but receive fresh message IDs.
      const identity = message.requestId ? `${message.requestId}:${message.role}` : `${message.at}:${message.role}:${message.content}`;
      if (seen.has(identity)) continue;
      seen.add(identity); messages++; hours[date.getHours()]++;
      const key = dayKey(date), day = days.get(key) ?? {messages:0,models:{}}; day.messages++; days.set(key,day);
      if (message.role !== 'assistant') continue;
      if (message.demo || message.usageSource !== 'reported' || !validCount(message.inputTokens) || !validCount(message.tokens)) { unreported++; continue; }
      const id = message.model || 'unknown', model = models.get(id) ?? {id,input:0,output:0,requests:0};
      model.input += message.inputTokens; model.output += message.tokens; model.requests++; models.set(id,model);
      input += message.inputTokens; output += message.tokens;
      day.models[id] = (day.models[id] ?? 0) + message.inputTokens + message.tokens;
    }
    if (active) sessionCount++;
  }
  const ranked = [...models.values()].sort((a,b)=>(b.input+b.output)-(a.input+a.output));
  const peak = Math.max(...hours);
  return { sessions:sessionCount, messages, input, output, total:input+output, activeDays:days.size, peakHour:peak ? hours.indexOf(peak) : null, favorite:ranked[0]?.id ?? null, days, models:ranked, unreported };
}
export function calendarDays(start: Date, end: Date) {
  const days: string[] = [], date = new Date(start); date.setHours(0,0,0,0);
  while (date <= end) { days.push(dayKey(date)); date.setDate(date.getDate()+1); }
  return days;
}
