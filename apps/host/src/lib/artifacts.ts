export interface PreviewArtifact { id: string; kind: 'website' | 'markdown' | 'pdf' | 'image' | 'code' | 'table'; title: string; url?: string; content?: string; demo?: boolean; }
export const PREVIEW_EXAMPLES: PreviewArtifact[] = [
  { id:'sample-website', kind:'website', title:'Fieldnotes — website', url:'/examples/fieldnotes.html', demo:true },
  { id:'sample-markdown', kind:'markdown', title:'project-brief.md', url:'/examples/project-brief.md', demo:true },
  { id:'sample-image', kind:'image', title:'fieldnotes-cover.svg', url:'/examples/fieldnotes-cover.svg', demo:true },
  { id:'sample-code', kind:'code', title:'note-card.ts', url:'/examples/note-card.ts', demo:true },
  { id:'sample-table', kind:'table', title:'release-plan.csv', url:'/examples/release-plan.csv', demo:true },
  { id:'sample-pdf', kind:'pdf', title:'project-brief.pdf', url:'/examples/project-brief.pdf', demo:true },
];
export function safePreviewUrl(value: string) {
  try { const u = new URL(value, location.origin); return ['http:','https:','blob:'].includes(u.protocol) ? u.href : null; }
  catch { return null; }
}
export function artifactFromUrl(url: string): PreviewArtifact | null {
  const safe = safePreviewUrl(url); if (!safe) return null;
  const path = new URL(safe).pathname;
  const kind = /\.pdf$/i.test(path) ? 'pdf' : /\.(png|jpe?g|webp|gif|svg)$/i.test(path) ? 'image' : /\.csv$/i.test(path)?'table':/\.(tsx?|jsx?|mts|cts|mjs|cjs|pyw?|json|css|rs|go|sh|bash|zsh|ya?ml|java|c|h|cpp|cc|cxx|hpp|cs|sql)$/i.test(path)?'code': /\.(md|mdx|markdown|txt)$/i.test(path) ? 'markdown' : 'website';
  let filename = path.split('/').at(-1) || 'Document';
  try { filename = decodeURIComponent(filename); } catch { /* Keep malformed URL escapes readable. */ }
  return { id:safe, kind, url:safe, title:kind === 'website' ? new URL(safe).hostname : filename };
}

export const artifactLabel=(kind:PreviewArtifact['kind'])=>({website:'Website',markdown:'Markdown',pdf:'PDF',image:'Image',code:'Code',table:'Table'}[kind]);

export function artifactFilename(a:PreviewArtifact) {
  if(a.url&&!a.url.startsWith('blob:')){try{const u=new URL(a.url,location.origin);const file=decodeURIComponent(u.pathname.split('/').filter(Boolean).at(-1)??'');return /\.[a-z0-9]{1,8}$/i.test(file)?file:a.kind==='website'?u.hostname:a.title;}catch{/* Use the supplied title. */}}
  return a.title;
}
export const artifactIcon=(a:PreviewArtifact)=>a.kind==='website'?'link' as const:a.kind==='image'?'image' as const:'file' as const;
