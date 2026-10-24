import { tr } from '../lib/i18n';
import { useEffect, useRef, useState } from 'react';
import { Btn, ErrorNotice, Dialog, Icon, IconBtn } from './ui';
import { api } from '../lib/client';
import { setCloudflare, useStore } from '../lib/store';
import type { TunnelState } from '../lib/types';
export function CloudflareDialog({ open, onClose, onCreateKey }: {
    open: boolean;
    onClose: () => void;
    onCreateKey: () => void;
}) {
    const serving = useStore(s => s.serving);
    const [mode, setMode] = useState<'quick' | 'domain'>('quick'), [step, setStep] = useState(0);
    const [hostname, setHostname] = useState(''), [token, setToken] = useState(''), [pending, setPending] = useState(false), [error, setError] = useState('');
    const [detected, setDetected] = useState<TunnelState | null>(null), [retry, setRetry] = useState(0);
    const heading = useRef<HTMLHeadingElement>(null);
    const tunnel = serving?.tunnel ?? detected;
    const active = !!tunnel && ['starting', 'connected', 'stopping'].includes(tunnel.status);
    const steps = mode === 'quick' ? ['Address', 'Connector', 'API key', 'Connect'] : ['Address', 'Connector', 'API key', 'Domain', 'Connect'];
    const last = steps.length - 1;
    const installed = !!tunnel?.installed, keyReady = !!serving?.access.apiKeySet, other = tunnel?.external ?? null;
    const canContinue = step === 1 ? installed : step === 2 ? true : step === 3 && mode === 'domain' ? !!hostname.trim() && !!token.trim() : true;
    async function inspect() { setPending(true); setError(''); try {
        setDetected(await api.get<TunnelState>('/api/serving/tunnel'));
    }
    catch (e) {
        setError((e as Error).message);
    }
    finally {
        setPending(false);
    } }
    useEffect(() => { if (open)
        void inspect();
    else {
        setToken('');
        setError('');
    } }, [open]);
    useEffect(() => {
        if (!open || step !== 1 || installed)
            return;
        let alive = true;
        setPending(true);
        setError('');
        void setCloudflare({ action: 'prepare' }).then(() => { if (alive) {
            setPending(false);
            setStep(2);
        } }).catch(e => { if (alive) {
            setError((e as Error).message);
            setPending(false);
        } });
        return () => { alive = false; };
    }, [open, step, retry]);
    useEffect(() => { if (open && step > 0)
        heading.current?.focus({ preventScroll: true }); }, [step, open]);
    function dismiss() { setStep(0); onClose(); }
    function next() { setError(''); setStep(s => s === 0 && installed ? 2 : Math.min(last, s + 1)); }
    function back() { setError(''); if (!step)
        dismiss();
    else
        setStep(s => s === 2 && installed ? 0 : s - 1); }
    async function connect() { setPending(true); setError(''); try {
        await setCloudflare({ action: 'start', mode, hostname, token });
        setToken('');
    }
    catch (e) {
        setError((e as Error).message);
    }
    finally {
        setPending(false);
    } }
    async function stop() { setPending(true); setError(''); try {
        await setCloudflare({ action: 'stop' });
        setStep(0);
    }
    catch (e) {
        setError((e as Error).message);
    }
    finally {
        setPending(false);
    } }
    const titles = ['Choose your address', installed ? 'Connector ready' : 'Preparing your connector', 'Protect your connection', ...(mode === 'domain' ? ['Set up your domain'] : []), 'Ready to connect'];
    return <Dialog open={open} title={tr("Connect with Cloudflare")} onClose={dismiss} actions={<>
    {tr(active ? <><Btn onClick={dismiss}>{tr("Done")}</Btn><Btn disabled={pending || tunnel?.status === 'stopping'} onClick={() => void stop()}>{tr(tunnel?.status === 'stopping' ? 'Stopping…' : 'Stop tunnel')}</Btn></> : <><Btn disabled={pending} onClick={back}>{tr(step ? 'Back' : 'Cancel')}</Btn>{tr(step === last ? <Btn variant="primary" disabled={pending || !installed || (mode === 'domain' && (!hostname.trim() || !token.trim()))} onClick={() => void connect()}>{tr(pending ? 'Connecting…' : (error || tunnel?.error) ? 'Try again' : mode === 'quick' ? 'Create temporary URL' : 'Connect domain')}</Btn> : <Btn variant="primary" disabled={pending || !canContinue} onClick={next}>{tr("Continue")}<Icon name="chevron" className="wizard-next" size={14}/></Btn>)}</>)}
  </>}>
    <div className="tunnel-wizard">
      {tr(!active ? <ol className="tunnel-progress" aria-label={tr("Connection progress")}>{tr(steps.map((label, i) => <li key={label} data-current={step === i || undefined} data-complete={step > i || undefined}><button disabled={i >= step || pending} aria-current={step === i ? 'step' : undefined} onClick={() => { setError(''); setStep(i); }}><span>{tr(step > i ? <Icon name="check" size={13}/> : i + 1)}</span><small>{tr(label)}</small></button></li>))}</ol> : null)}
      {tr(active ? <section className="tunnel-stage tunnel-stage--status" key={tunnel?.status} aria-live="polite"><span className="tunnel-stage__symbol"><Icon name={tunnel?.status === 'connected' ? 'check' : 'clock'} size={26}/></span><h3 className="h3">{tr(tunnel?.status === 'connected' ? 'You’re connected' : tunnel?.status === 'stopping' ? 'Stopping the tunnel…' : 'Connecting to Cloudflare…')}</h3>{tr(tunnel?.url ? <><p className="small muted">{tr("Use this address and your Strata API key in your other app.")}</p><div className="install-command"><code>{tunnel.url}</code><IconBtn name="copy" title={tr("Copy tunnel URL")} onClick={() => void navigator.clipboard.writeText(tunnel.url!)}/></div><p className="xs muted">{tr(tunnel.mode === 'quick' ? 'Temporary address · turn off streaming in your client.' : `Your Cloudflare route must point to ${tunnel.originUrl}.`)}</p></> : <p className="small muted">{tr("This usually takes a few seconds. You can close this window while it connects.")}</p>)}</section> : <section className="tunnel-stage" key={step} aria-labelledby="tunnel-step-title">
        <h3 ref={heading} id="tunnel-step-title" className="h3" tabIndex={-1}>{tr(titles[step])}</h3>
        {tr(step === 0 ? <><p className="small muted">{tr("Give your local API an HTTPS address that another app can reach.")}</p><div className="tunnel-choices" role="radiogroup" aria-label={tr("Cloudflare connection type")}>{tr([{ value: 'quick' as const, title: 'Free temporary URL', detail: 'No account or domain needed. Good for a quick connection.', icon: 'bolt' as const }, { value: 'domain' as const, title: 'My own domain', detail: 'A stable address on a domain you manage with Cloudflare.', icon: 'link' as const }].map(choice => <button key={choice.value} role="radio" aria-checked={mode === choice.value} tabIndex={mode === choice.value ? 0 : -1} onKeyDown={e => { if (['ArrowRight', 'ArrowLeft', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
            e.preventDefault();
            setMode(mode === 'quick' ? 'domain' : 'quick');
            (e.currentTarget.parentElement?.children[mode === 'quick' ? 1 : 0] as HTMLElement)?.focus();
        } }} onClick={() => setMode(choice.value)}><Icon name={choice.icon} size={20}/><span><strong>{tr(choice.title)}</strong><small>{tr(choice.detail)}</small></span><span className="tunnel-choice-dot">{tr(mode === choice.value ? <Icon name="check" size={13}/> : null)}</span></button>))}</div></> : null)}
        {tr(step === 1 ? <div className="tunnel-automatic" data-ready={installed || undefined} aria-live="polite" aria-busy={pending}><span className="tunnel-stage__symbol"><Icon name={installed ? 'check' : error ? 'warning' : 'refresh'} size={28}/></span><p className="small muted">{tr(installed ? 'Your Cloudflare connector is installed and ready.' : error ? 'Automatic setup needs another try.' : 'Strata downloads and sets up the Cloudflare connector for you.')}</p>{tr(installed ? <>{tr(tunnel?.connector ? <p className="xs muted">{tr("Using {0}", [tunnel.connector.summary])}</p> : null)}{tr(other ? <p className="xs muted" role="status">{tr(other.others.length === 1 && !other.own.length ? "cloudflared is already running for another tunnel (PID {0})." : other.own.length ? "Strata’s tunnel for this address is already running (PID {0})." : "cloudflared is already running for other tunnels (PIDs {0}).", [(other.own.length ? other.own : other.others).join(", ")])}</p> : null)}</> : error ? <Btn onClick={() => setRetry(v => v + 1)}>{tr("Retry setup")}</Btn> : <span className="small muted" role="status">{tr("Setting up\u2026")}</span>)}</div> : null)}
        {tr(step === 2 ? <><p className="small muted">{tr(keyReady ? "Only apps with your Strata API key can use this connection." : "Without an API key, anyone who finds the link can use your model.")}</p>{tr(keyReady ? <div className="tunnel-check"><Icon name="shield"/><div><strong>{tr("API key ready")}</strong><span>{tr("Use your existing key in the app you connect.")}</span></div></div> : <div className="tunnel-key-step"><Icon name="shield" size={30}/><p className="small">{tr("Create a key to protect your API. You\u2019ll return here when it\u2019s ready. You can also continue without one, but anyone who finds the link can then use your model.")}</p><Btn onClick={onCreateKey}>{tr("Create an API key (recommended)")}</Btn></div>)}</> : null)}
        {tr(mode === 'domain' && step === 3 ? <><p className="small muted">{tr("Create a tunnel in your ")}<a href="https://dash.cloudflare.com/" target="_blank" rel="noreferrer">{tr("Cloudflare dashboard \u2197")}</a>{tr(". Copy the token from its connector setup.")}</p><label className="input-field"><span>{tr("Tunnel token")}</span><input type="password" className="st-input" autoComplete="off" spellCheck={false} value={token} onChange={e => setToken(e.target.value)} placeholder={tr("Paste only the token after --token")}/></label><label className="input-field"><span>{tr("Your hostname")}</span><input className="st-input" autoComplete="off" spellCheck={false} value={hostname} onChange={e => setHostname(e.target.value)} placeholder={tr("ai.example.com")}/></label><div><p className="small muted">{tr("Add a published application route for this hostname with this service URL:")}</p><div className="install-command"><code>{tunnel?.originUrl ?? 'Loading…'}</code><IconBtn name="copy" title={tr("Copy tunnel service URL")} disabled={!tunnel?.originUrl} onClick={() => void navigator.clipboard.writeText(tunnel!.originUrl)}/></div></div><p className="xs muted">{tr("The token is used for this connection only and is not saved.")}</p></> : null)}
        {tr(step === last ? <><p className="small muted">{tr(mode === 'quick' ? 'Cloudflare will create your address when you connect.' : 'Start the connector for the domain you configured.')}</p><dl className="tunnel-review"><div><dt>{tr("Address")}</dt><dd>{tr(mode === 'quick' ? 'Free temporary URL' : hostname)}</dd></div><div><dt>{tr("Connector")}</dt><dd><Icon name="check" size={14}/>{tr(tunnel?.connector?.version ? "Installed · cloudflared {0}" : "Installed", tunnel?.connector?.version ? [tunnel.connector.version] : undefined)}</dd></div><div><dt>{tr("Access")}</dt><dd><Icon name="shield" size={14}/>{tr(keyReady ? "API key required" : "No API key: anyone with the link can use it")}</dd></div></dl><p className="small muted">{tr(mode === 'quick' ? 'The address changes each time. Turn off streaming in your client; replies arrive all at once.' : 'Use this hostname and your Strata API key in your client. Streaming is supported.')}</p><p className="xs muted">{tr("Only the inference API is shared. Your computer and Strata must stay on.")}</p></> : null)}
      </section>)}
      {tr(error || tunnel?.error ? <ErrorNotice message={error || tunnel?.error || 'Connection failed'} next={tr("Check your internet connection, then retry.")}/> : null)}
    </div>
  </Dialog>;
}
