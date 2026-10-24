import { tr } from '../lib/i18n';
import { useState } from 'react';
import { Btn, Dialog, ErrorNotice, IconBtn, Select } from './ui';
import { MCP_NAME_HELP, validMcpName } from '../lib/mcp-names';
export function NativeMcpEditor({ text, onChange }: {
    text: string;
    onChange: (text: string) => void;
}) {
    const [editing, setEditing] = useState<string | null>(null), [name, setName] = useState(''), [transport, setTransport] = useState('command'), [address, setAddress] = useState(''), [args, setArgs] = useState(''), [advanced, setAdvanced] = useState('{}'), [error, setError] = useState('');
    let servers: Record<string, any> = {}, parseError = '';
    try {
        const parsed = JSON.parse(text);
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed) || Object.values(parsed).some(s => !s || typeof s !== 'object' || Array.isArray(s)))
            throw new Error('Use an object with one entry per tool server.');
        servers = parsed;
    }
    catch (e) {
        parseError = e instanceof Error ? e.message : 'Fix the tool configuration JSON.';
    }
    function open(id: string) { const server = servers[id] ?? {}; setEditing(id); setName(id); setTransport(server.url ? 'url' : 'command'); setAddress(server.url ?? server.command ?? ''); setArgs((server.args ?? []).join('\n')); setAdvanced(JSON.stringify(Object.fromEntries(Object.entries(server).filter(([k]) => !['url', 'command', 'args', 'disabled'].includes(k))), null, 2)); setError(''); }
    function save() {
        try {
            if (!validMcpName(name))
                throw new Error(MCP_NAME_HELP);
            if (!address.trim())
                throw new Error('Enter the server URL or program.');
            if (name !== editing && servers[name])
                throw new Error('That server name is already used.');
            const extra = JSON.parse(advanced);
            const next = { ...servers };
            if (editing)
                delete next[editing];
            next[name] = { ...extra, ...(transport === 'url' ? { url: address.trim() } : { command: address.trim(), args: args.split('\n').filter(Boolean) }), disabled: servers[editing ?? '']?.disabled ?? false };
            onChange(JSON.stringify(next, null, 2));
            setEditing(null);
        }
        catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }
    return <><div className="native-mcp-list">{tr(Object.entries(servers).map(([id, server]) => <div key={id}><span><strong>{id}</strong><small>{server.url ?? server.command} · {tr(server.disabled ? 'Disabled' : 'Enabled')}</small></span><button className="st-toggle" role="switch" aria-label={tr("Enable MCP server {0}", [id])} aria-checked={!server.disabled} onClick={() => onChange(JSON.stringify({ ...servers, [id]: { ...server, disabled: !server.disabled } }, null, 2))}/><IconBtn name="pencil" title={tr("Edit MCP server {0}", [id])} onClick={() => open(id)}/><IconBtn name="trash" title={tr("Remove MCP server {0}", [id])} onClick={() => { const next = { ...servers }; delete next[id]; onChange(JSON.stringify(next, null, 2)); }}/></div>))}</div><Btn disabled={!!parseError} size="sm" icon="plus" onClick={() => open('')}>{tr("Add tool server")}</Btn><details className="native-mcp-json"><summary>{tr("Advanced JSON")}</summary><textarea className="st-input native-json" aria-label={tr("MCP server configuration")} value={text} onChange={e => onChange(e.target.value)} spellCheck={false}/></details>{tr(parseError ? <ErrorNotice message={parseError}/> : null)}
 <Dialog open={editing !== null} title={tr(editing ? 'Edit tool server' : 'Add tool server')} compact onClose={() => setEditing(null)} actions={<><Btn onClick={() => setEditing(null)}>{tr("Cancel")}</Btn><Btn variant="primary" onClick={save}>{tr("Add to settings")}</Btn></>}><div className="stack"><label className="input-field"><span>{tr("Name")}</span><input className="st-input" aria-label={tr("MCP server name")} value={name} onChange={e => setName(e.target.value)} placeholder={tr("files or search")}/></label><Select ariaLabel={tr("MCP transport")} value={transport} options={[{ value: 'command', label: 'Local program' }, { value: 'url', label: 'HTTP server' }]} onChange={setTransport}/><label className="input-field"><span>{tr(transport === 'url' ? 'Server URL' : 'Program')}</span><input className="st-input" aria-label={tr(transport === 'url' ? 'MCP server URL' : 'MCP program')} value={address} onChange={e => setAddress(e.target.value)} placeholder={tr(transport === 'url' ? 'http://127.0.0.1:3000/mcp' : 'npx, uvx or python')}/></label>{tr(transport === 'command' ? <label className="input-field"><span>{tr("Arguments \u00B7 one per line")}</span><textarea className="st-input" aria-label={tr("MCP program arguments")} rows={4} value={args} onChange={e => setArgs(e.target.value)} placeholder={tr('-y\n@modelcontextprotocol/server-filesystem\n/your/folder')}/></label> : null)}<details><summary>{tr("Credentials & environment")}</summary><textarea className="st-input native-json" aria-label={tr("MCP advanced connection settings")} value={advanced} onChange={e => setAdvanced(e.target.value)} spellCheck={false}/><p className="xs muted">{tr("Use env, headers and cwd here. Existing credentials are masked and retained when you keep the mask.")}</p></details>{tr(error ? <ErrorNotice message={error}/> : null)}</div></Dialog></>;
}
