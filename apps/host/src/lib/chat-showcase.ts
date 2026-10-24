import type { ChatMessage, Session } from './types';
import { PREVIEW_EXAMPLES } from './artifacts';
export function showcaseMessages(at:string):ChatMessage[] {
  const base={at,role:'assistant' as const,example:true};
  return [
    {id:'showcase-prompt',at,role:'user',content:'Show me how code, approvals, images, video and GIFs look in a conversation.'},
    {...base,id:'showcase-code',content:'## A small idea, in a few different forms\nThis chat is a design playground. Scroll through code, tool approvals and media, then open a file to try the integrated preview tabs. The agents and statistics are example data.\n\n### Code\nA small, typed building block for Fieldnotes.\n\n```typescript\ntype Note = { id: string; title: string; pinned: boolean };\n\nexport function pinNote(notes: Note[], id: string): Note[] {\n  return notes.map(note =>\n    note.id === id ? { ...note, pinned: !note.pinned } : note\n  );\n}\n```',tokens:1250,tps:299,durationMs:4181,ttftMs:420},
    {...base,id:'showcase-approval',content:'### A change waiting for you\nThis example approval is interactive. Approving or denying updates the card here; it does not write a file or run a command.',tools:[{id:'showcase-read',name:'Read project notes',args:{file:'project-brief.md'},preview:'Read project-brief.md',effect:'Read the project brief.',scope:'read',state:'done',result:'Found the brief, release goals and design principles.',durationMs:240},{id:'showcase-write',name:'Update note-card.ts',args:{file:'note-card.ts'},preview:'+ export function pinNote(notes: Note[], id: string) {\n+   return notes.map(note => note.id === id\n+     ? { ...note, pinned: !note.pinned } : note);\n+ }',effect:'Add pinning to the note card. Example only — no files will be changed.',scope:'write',state:'proposed'}]},
    {...base,id:'showcase-image',content:'### Image\nA cover illustration can live directly in the conversation. Click it to inspect it in the sidebar.',media:[{kind:'image',src:'/examples/fieldnotes-cover.svg',title:'fieldnotes-cover.svg',caption:'Fieldnotes · cover illustration'}]},
    {...base,id:'showcase-video',content:'### Video\nA short motion study with native playback, scrubbing and full-screen controls.',media:[{kind:'video',src:'/examples/fieldnotes-motion.mp4',poster:'/examples/fieldnotes-motion-poster.png',title:'fieldnotes-motion.mp4',caption:'A little room for your next idea · 4 seconds · silent'}]},
    {...base,id:'showcase-gif',content:'### Animated GIF\nThe same idea as a lightweight loop. Pause it below if you want a still.',media:[{kind:'gif',src:'/examples/fieldnotes-ideas.gif',poster:'/examples/fieldnotes-motion-poster.png',title:'fieldnotes-ideas.gif',caption:'A stack of ideas, gently in motion.'}]},
    {...base,id:'showcase-table',content:'### Tables, quotes and next steps\n| Piece | Format | Status |\n| --- | --- | --- |\n| Cover | SVG image | Ready |\n| Motion study | MP4 + GIF | Ready |\n| Project brief | PDF + Markdown | Ready |\n\n> A calmer place for what comes next.\n\n1. Open a preview from a file card.\n2. Switch between the tabs in the top bar.\n3. Try full screen, the subagent menu and each statistics popover.\n\n### Files to explore\nOpen in the sidebar, show the local file in Finder, or use your default application.',artifacts:PREVIEW_EXAMPLES,tokens:1140,inputTokens:25740,contextMax:128000,tps:299,durationMs:3813,ttftMs:390,usageSource:'demo'}
  ];
}

export const SHOWCASE_AGENTS: NonNullable<Session['agents']> = [
  {id:'example-design',name:'Design review',description:'Layout and preview interactions reviewed.',status:'completed',durationMs:399000,tokens:10500000},
  {id:'example-code',name:'Component review',description:'Checking note-card interaction states.',status:'working',durationMs:82000,tokens:24800},
  {id:'example-preview',name:'Preview check',description:'Preview server did not respond.',status:'failed',durationMs:21000,tokens:3400},
];
