from pathlib import Path
root=Path(__file__).resolve().parent.parent
def edit(file,old,new):
 p=root/file;s=p.read_text();assert old in s,(file,old[:100]);p.write_text(s.replace(old,new))
def append(file,text):
 p=root/file;p.write_text(p.read_text()+text)
append('src/lib/format.ts','''
/** Identity comes from the host; never substitute a different real model. */
export function formatModelName(model: {family:string;quant:string;name?:string}|null|undefined): string {
  if (!model) return 'No model loaded';
  const version = ({Q2_K:'Smallest',Q3_K_M:'Compact',Q4_K_M:'Balanced',Q5_K_M:'High quality',Q6_K:'Very high',Q8_0:'Near-original',BF16:'Original'} as Record<string,string>)[model.quant] ?? model.name;
  return `${model.family} · ${version ? `${version} (${model.quant})` : model.quant}`;
}
export function userError(message: string): string {
  return message.replace(/^\\s*Fixture:\\s*/i, '').replace(/^./, c => c.toUpperCase());
}
''')
edit('src/components/ui.tsx',"import { cx } from '../lib/format';","import { cx, userError } from '../lib/format';")
append('src/components/ui.tsx','''
/** Shared, announced error with a cause and a concrete recovery action. */
export function ErrorNotice({message, next, action, onRetry}: {message:string;next?:string;action?:string;onRetry?:()=>void}) {
  return <div className="error-notice" role="alert"><Icon name="error" size={18}/><div><strong>{userError(message)}</strong>{next?<p>{next}</p>:null}</div>{onRetry?<Btn size="sm" onClick={onRetry}>{action??'Try again'}</Btn>:null}</div>;
}
export function PreviewBadge(){const demo=useStore(s=>s.library?.mode==='simulated');return demo?<span className="preview-badge">Preview build</span>:null;}
''')
edit('src/components/ui.tsx','style={{ left: placed.left, top: placed.top, width: placed.width }}',"style={{ left: placed.left, top: placed.top, width: placed.width, zIndex: triggerRef.current?.closest('[role=\"dialog\"]') ? 'var(--layer-dialog-popover)' : 'var(--layer-popover)' }}")
edit('src/components/Popover.tsx','style={{...position,width,',"style={{...position,zIndex:anchor?.closest('[role=\"dialog\"]')?'var(--layer-dialog-popover)':'var(--layer-popover)',width,")
edit('src/components/FloatingMenu.tsx','style={{ ...placed, width:',"style={{ ...placed, zIndex:anchor?.closest('[role=\"dialog\"]')?'var(--layer-dialog-popover)':'var(--layer-popover)', width:")
edit('src/components/layout.tsx',"import type { ReactNode } from 'react';","import type { ReactNode } from 'react';\nimport { PreviewBadge } from './ui';")
edit('src/components/layout.tsx','{title}</h1>','{title} <PreviewBadge/></h1>')
edit('src/app/Sidebar.tsx',"import { cx, relTime }", "import { cx, relTime, formatModelName }")
p=root/'src/app/Sidebar.tsx';s=p.read_text();start=s.index('  const modelName=');end=s.index('  const phase',start);s=s[:start]+"  const modelName=formatModelName(current);\n  const modelQuant=current ? 'Shared by all chats' : 'View model versions';\n"+s[end:];s=s.replace('<NavLink route="monitor"', '<NavLink route="skills" label="Skills" icon="thinking" current={location.route} />\n        <NavLink route="monitor"');s=s.replace('<NavLink route="serving" label="Serving" icon="link" current={location.route} meta={serving?.stats.running ? String(serving.stats.running) : undefined} />','<NavLink route="tuning" label="Engine" icon="gauge" current={location.route} /><NavLink route="models" label="Model" icon="layers" current={location.route} />');s=s.replace('More — Skills, Models, Engine and Settings','More — Serving and Settings').replace("['skills', 'models', 'tuning', 'settings']","['serving', 'settings']");s=s.replace("{ route: 'skills', label: 'Skills', icon: 'thinking' }, { route: 'models', label: 'Models', icon: 'layers' }, { route: 'tuning', label: 'Engine', icon: 'gauge' },", "{ route: 'serving', label: 'Serving', icon: 'link' },");s=s.replace('<label className="strong" htmlFor="chat-name">Chat name</label><input className="st-input" id="chat-name" value={name} onChange={e => setName(e.target.value)} />','<label className="input-field" htmlFor="chat-name"><span>Chat name</span><input className="st-input" id="chat-name" value={name} onFocus={e=>e.currentTarget.select()} onChange={e => setName(e.target.value)} /></label>');s=s.replace('<Btn variant="danger" onClick={() => { if (deleteId) deleteSession(deleteId);','<Btn onClick={() => { if (deleteId) toggleArchive(deleteId); setDeleteId(null); }}>Archive instead</Btn><Btn variant="danger" onClick={() => { if (deleteId) deleteSession(deleteId);');p.write_text(s)
edit('src/lib/commands.ts',"label: 'Models'", "label: 'Model'")
edit('src/lib/commands.ts',"slash: ['new', 'clear']", "slash: ['new']")
edit('src/lib/commands.ts',"Check computer, choose model and load", "Check computer, pick a version and load")
# Flags retain unfinished controls without presenting them as working features.
(root/'src/lib/features.ts').write_text("export const features = { toolAccess: false, thinkingLevels: false, sessionLog: false, chatAttachments: false, demosInComposer: false } as const;\n")
edit('src/components/ChatToolbar.tsx',"import { Btn, Dialog, Icon, IconBtn }", "import { Btn, Dialog, Icon, IconBtn, PreviewBadge }")
edit('src/components/ChatToolbar.tsx',"import { SubagentRow }", "import { features } from '../lib/features';\nimport { SubagentRow }")
edit('src/components/ChatToolbar.tsx','</h1><button ref={agentsRef}', '</h1><PreviewBadge/>{!!session?.agents?.length&&<button ref={agentsRef}')
edit('src/components/ChatToolbar.tsx','size={13}/></button><span className="spacer"/>', 'size={13}/></button>}<span className="spacer"/>')
edit('src/components/ChatToolbar.tsx','</button><button role="menuitem" className="menu__item" onClick={()=>{setMenu(false);setLog(true);}}>', '</button>{features.sessionLog&&<button role="menuitem" className="menu__item" onClick={()=>{setMenu(false);setLog(true);}}>')
edit('src/components/ChatToolbar.tsx','Download Session Log</button></FloatingMenu>','Download session log</button>}</FloatingMenu>')
edit('src/components/ChatToolbar.tsx','<Icon name="chevron" size={16}/></button>)}</div></Dialog>', '<span className="xs muted">Use template</span></button>)}</div></Dialog>')
append('src/styles/app.css','''
/* Audit fixes: shared states and overlay ownership. */
:root { --layer-base: 0; --layer-sticky: 10; --layer-sidebar: 20; --layer-popover: 100; --layer-dialog: 160; --layer-dialog-popover: 180; --layer-toast: 200; }
html, body { overflow: hidden; }
.main__body, .dialog-body { scrollbar-gutter: stable; }
.dialog-layer { z-index: var(--layer-dialog); }
.dialog-scrim, .scrim { background: rgb(0 0 0 / .36); }
[data-theme='dark'] .dialog-scrim, [data-theme='dark'] .scrim { background: rgb(0 0 0 / .55); }
.dialog-surface, .palette, .setup__panel { opacity: 1; background: var(--st-surface); animation: none; }
.dialog-head, .dialog-actions { flex-shrink: 0; }
.select__pop { z-index: var(--layer-dialog-popover); background: var(--st-surface); }
.menu, .anchored-popover { z-index: var(--layer-popover); background: var(--st-surface); }
.st-btn--primary, .nav-new.st-btn--primary { background: #10b981; color: #0b1f17; border-color: #10b981; }
.st-btn--primary:hover:not(:disabled), .nav-new.st-btn--primary:hover:not(:disabled) { background: #22ce95; color: #0b1f17; border-color: #22ce95; }
.st-btn:disabled { background: var(--st-surface-2); border-color: var(--st-line); color: var(--st-ink-muted); opacity: .65; box-shadow: none; }
.st-toggle[aria-checked='true'] { background: #10b981; border-color: #10b981; }
.st-toggle[aria-checked='true']::after { background: #fff; }
input[type='checkbox'], input[type='radio'] { accent-color: #10b981; }
textarea { font-family: inherit; }
.error-notice { display:flex; gap:10px; align-items:flex-start; padding:12px 14px; margin:12px 0; border:1px solid var(--st-danger-text); border-radius:12px; background:var(--st-surface); color:var(--st-danger-text); font-size:13px; }
.error-notice > svg { flex-shrink:0; margin-top:2px; }
.error-notice > div { flex:1; min-width:0; }
.error-notice p { margin:4px 0 0; color:var(--st-ink-soft); }
.st-input[aria-invalid='true'] { border-color:var(--st-danger-text); }
.preview-badge { display:inline-flex; flex-shrink:0; align-items:center; border:1px solid var(--st-line); border-radius:6px; color:var(--st-ink-muted); padding:3px 6px; font-size:10px; font-weight:500; white-space:nowrap; }
.st-msg { min-width:0; }
.st-msg__meta { display:flex; flex-direction:row; flex-wrap:nowrap; min-width:0; width:100%; gap:8px; }
.st-msg__actions { display:flex; flex-wrap:nowrap; opacity:0; }
.st-msg:hover .st-msg__actions, .st-msg:focus-within .st-msg__actions { opacity:1; }
.message-overflow { position:relative; }
.message-overflow > div { position:absolute; top:100%; bottom:auto; left:0; min-width:240px; width:max-content; max-width:calc(100vw - 40px); background:var(--st-surface); border:1px solid var(--st-line); border-radius:12px; padding:10px; z-index:5; display:flex; flex-direction:column; align-items:stretch; box-shadow:var(--st-sh-pop); }
.st-msg--user .message-overflow > div { left:auto; right:0; }
.response-stats { white-space:nowrap; }
.composer .st-composer textarea { min-height:44px; height:44px; max-height:180px; padding:12px 16px; }
.composer { padding-top:10px; padding-bottom:10px; }
.composer-model { max-width:270px; }
.composer-model span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.chat-toolbar__conversation h1 { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.tool-summary { font-size:13px; color:var(--st-ink-soft); }
.tool-summary summary { display:flex; align-items:center; gap:6px; font-weight:500; list-style:none; }
.tool-summary summary::-webkit-details-marker { display:none; }
.tool-summary[open] summary > svg:first-child { transform:rotate(180deg); }
@media (hover:none) { .st-msg__actions { opacity:1; } }
''')
