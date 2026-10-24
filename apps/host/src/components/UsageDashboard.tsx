import {localeTag} from '../lib/i18n';
import { tr } from '../lib/i18n';
import { useMemo, useState } from 'react';
import { Btn, Icon } from './ui';
import { Brand } from './Brand';
import { useNative } from '../lib/native';
import { go, useStore } from '../lib/store';
import { tokens } from '../lib/format';
import { calendarDays, usageDashboard, type UsagePeriod } from '../lib/usage-dashboard';
const colors = ['#4d82d8', '#6998e6', '#8aaeea', '#a6c1ee', '#c4d7f4', '#dce7f8'];
export function UsageDashboard({ loaded }: {
    loaded: boolean;
}) {
    const native = useNative(), remoteModel = useStore(s => s.remoteModel);
    const canChat = loaded || !!remoteModel || !!native?.instances.some(i => i.id === native.defaultId && i.status === 'ready');
    const sessions = useStore(s => s.sessions), library = useStore(s => s.library);
    const [tab, setTab] = useState<'overview' | 'models'>('overview'), [period, setPeriod] = useState<UsagePeriod>('all'), [expanded, setExpanded] = useState(false);
    const now = new Date();
    const usage = useMemo(() => usageDashboard(sessions, period), [sessions, period]);
    const label = (id: string | null) => library?.variants.find(v => v.id === id)?.family ?? id ?? '—';
    const start = new Date(now);
    start.setHours(0, 0, 0, 0);
    const earliest = [...usage.days.keys()].sort()[0];
    const earliestDate = earliest ? new Date(`${earliest}T00:00:00`) : null;
    const span = period === '7d' ? 7 : period === '30d' ? 30 : earliestDate ? (Date.UTC(start.getFullYear(), start.getMonth(), start.getDate()) - Date.UTC(earliestDate.getFullYear(), earliestDate.getMonth(), earliestDate.getDate())) / 86400000 + 1 : 91;
    start.setDate(start.getDate() - span + 1);
    const chartDays = calendarDays(start, now);
    const heatStart = new Date(start);
    if (span > 365)
        heatStart.setDate(heatStart.getDate() + (span - 365));
    if (span < 182)
        heatStart.setDate(heatStart.getDate() - (182 - span));
    heatStart.setDate(heatStart.getDate() - heatStart.getDay());
    const heatDays = calendarDays(heatStart, now);
    while (heatDays.length % 7)
        heatDays.push('');
    const maxMessages = Math.max(1, ...[...usage.days.values()].map(d => d.messages));
    // Up to 90 columns. Longer ranges are grouped into consecutive date bins.
    const binSize = Math.max(1, Math.ceil(chartDays.length / 90));
    const bins = Array.from({ length: Math.ceil(chartDays.length / binSize) }, (_, i) => {
        const dates = chartDays.slice(i * binSize, (i + 1) * binSize), amounts: Record<string, number> = {};
        for (const date of dates)
            for (const [id, n] of Object.entries(usage.days.get(date)?.models ?? {}))
                amounts[id] = (amounts[id] ?? 0) + n;
        return { dates, amounts, total: Object.values(amounts).reduce((n, v) => n + v, 0) };
    });
    const maxTokens = Math.max(1, ...bins.map(b => b.total));
    const dateLabel = (key: string) => new Date(`${key}T00:00:00`).toLocaleDateString(localeTag(), { month: 'short', day: 'numeric' });
    const peak = usage.peakHour === null ? '—' : new Date(2026, 0, 1, usage.peakHour).toLocaleTimeString(localeTag(), { hour: 'numeric', minute: '2-digit' });
    const firstChat = !sessions.some(s => !s.showcase && s.messages.some(m => m.role === 'user' && !m.example && !m.timeSkipId));
    if (firstChat && canChat)
        return <section className="first-chat-start" aria-label={tr("Start your first chat")}><span className="first-chat-mark"><Brand variant="mark" height={38} decorative/><span className="first-chat-status"><Icon name="check" size={11}/></span></span><h2>{tr("Hello, I\u2019m Strata.")}</h2><p>{tr("Say hello, or jump straight in.")}</p></section>;
    return <section className="usage-home" aria-label={tr("Usage dashboard")}>
    <h2>{tr("What do we start with?")}</h2>
    <div className="usage-card">
      <div className="usage-card__toolbar"><div role="tablist" aria-label={tr("Usage view")}>{tr((['overview', 'models'] as const).map(v => <button key={v} role="tab" id={`usage-tab-${v}`} aria-controls="usage-content" aria-selected={tab === v} tabIndex={tab === v ? 0 : -1} onKeyDown={e => {
                if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) {
                    e.preventDefault();
                    const next = e.key === 'Home' ? 'overview' : e.key === 'End' ? 'models' : tab === 'overview' ? 'models' : 'overview';
                    setTab(next);
                    document.getElementById(`usage-tab-${next}`)?.focus();
                }
            }} onClick={() => setTab(v)}>{tr(v === 'overview' ? 'Overview' : 'Models')}</button>))}</div><div role="group" aria-label={tr("Usage period")}>{tr((['all', '30d', '7d'] as const).map(v => <button key={v} aria-pressed={period === v} onClick={() => setPeriod(v)}>{tr(v === 'all' ? 'All' : v)}</button>))}</div></div>
      <div id="usage-content" role="tabpanel" aria-labelledby={`usage-tab-${tab}`}>
        {tr(tab === 'overview' ? <><dl className="usage-stats">{tr([{ label: 'Sessions', value: usage.sessions.toLocaleString(localeTag()) }, { label: 'Messages', value: usage.messages.toLocaleString(localeTag()) }, { label: 'Total tokens', value: tokens(usage.total) }, { label: 'Active days', value: usage.activeDays.toLocaleString(localeTag()) }, { label: 'Peak hour', value: peak }, { label: 'Favorite model', value: label(usage.favorite) }].map(stat => <div key={stat.label}><dt>{tr(stat.label)}</dt><dd title={tr(stat.value)}>{tr(stat.value)}</dd></div>))}</dl>
          <div className="usage-heatmap" style={{ gridTemplateColumns: `repeat(${heatDays.length / 7},minmax(0,1fr))` }} role="img" aria-label={tr("Daily message activity. {0} active days in this period.", [usage.activeDays])}>{tr(heatDays.map((date, i) => { const n = usage.days.get(date)?.messages ?? 0; return <span key={date || `pad-${i}`} aria-hidden="true" data-level={!n ? 0 : Math.min(4, Math.ceil(n / maxMessages * 4))} data-empty={!date || undefined} title={tr(date ? date < chartDays[0] && period !== 'all' ? `${dateLabel(date)}: outside selected period` : `${dateLabel(date)}: ${n} messages` : undefined)}/>; }))}</div>
          <div className="usage-heatmap__legend"><span>{tr(dateLabel(heatDays[0]))} – {tr(dateLabel(chartDays.at(-1)!))}</span><span>{tr("Less ")}<i data-level="0"/><i data-level="1"/><i data-level="2"/><i data-level="3"/><i data-level="4"/>{tr(" More")}</span></div>
        </> : <>
          <svg className="usage-model-chart" viewBox="0 0 700 210" role="img" aria-label={tr("Reported token usage by model over time. {0} tokens.", [usage.total.toLocaleString()])}>
            {tr((usage.total ? [0, .5, 1] : [0]).map(f => <g key={f}><line x1="48" y1={175 - f * 145} x2="686" y2={175 - f * 145}/><text x="38" y={179 - f * 145} textAnchor="end">{tr(tokens(Math.round(maxTokens * f)))}</text></g>))}
            {tr(bins.map((bin, i) => { let y = 175; const width = 638 / bins.length; return <g key={bin.dates[0]}><title>{tr(dateLabel(bin.dates[0]))}{tr(bin.dates.length > 1 ? ` – ${dateLabel(bin.dates.at(-1)!)}` : '')}: {tr(bin.total.toLocaleString(localeTag()))}{tr(" tokens")}</title>{tr(usage.models.map((model, m) => { const h = (bin.amounts[model.id] ?? 0) / maxTokens * 145; y -= h; return h ? <rect key={model.id} x={48 + i * width + 1} y={y} width={Math.max(1, width - 2)} height={h} fill={colors[m % colors.length]}/> : null; }))}</g>; }))}
            {tr([0, Math.floor((bins.length - 1) / 2), bins.length - 1].filter((n, i, a) => a.indexOf(n) === i).map(i => <text key={i} x={48 + (i + .5) * 638 / bins.length} y="200" textAnchor={i === 0 ? 'start' : i === bins.length - 1 ? 'end' : 'middle'}>{tr(dateLabel(bins[i].dates[0]))}</text>))}
          </svg>
          {tr(usage.models.length ? <div className="usage-model-list">{tr(usage.models.slice(0, expanded ? undefined : 6).map((model, i) => <div key={model.id}><span className="usage-model-name"><i style={{ background: colors[i % colors.length] }}/>{tr(label(model.id))}<small>{tr(library?.variants.find(v => v.id === model.id)?.quant)}</small></span><span className="muted">{tr(tokens(model.input))}{tr(" in \u00B7 ")}{tr(tokens(model.output))}{tr(" out")}</span><strong>{tr(((model.input + model.output) / Math.max(1, usage.total) * 100).toFixed(1))}%</strong></div>))}{tr(usage.models.length > 6 ? <button className="usage-show-more" onClick={() => setExpanded(v => !v)}>{tr(expanded ? 'Show less' : `Show ${usage.models.length - 6} more`)}</button> : null)}</div> : <p className="usage-empty muted">{tr("No reported model usage in this period.")}</p>)}
        </>)}
      </div>
    </div>
    {tr(!canChat ? <Btn onClick={() => go('models')}>{tr("Choose a model")}</Btn> : null)}
  </section>;
}
