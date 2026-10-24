import type { BeginnerConfig, EngineConfig, ExpertParam } from './types';

/** Portable saved configuration. Metadata stays separate from editable values. */
export interface ConfigDocument {
  schema: 'strata.engine/1';
  beginner: BeginnerConfig;
  expert: Record<string, string | number | boolean>;
  [key: string]: unknown;
}

export function configDocument(engine: EngineConfig): ConfigDocument {
  return {
    ...(engine.extra ?? {}),
    schema: 'strata.engine/1',
    beginner: { ...engine.beginner },
    expert: { ...Object.fromEntries(engine.expert.map(p => [p.id, p.value])), 'ctx.length': engine.beginner.contextLength === 'recommended' ? '128K' : engine.beginner.contextLength.toUpperCase() },
  };
}

export function readConfig(text: string): ConfigDocument {
  let value: Record<string, unknown>;
  try { value = JSON.parse(text); }
  catch (error) {
    const offset = jsonErrorOffset(text);
    const before = text.slice(0, offset);
    const line = before.split('\n').length, column = offset - before.lastIndexOf('\n');
    throw new Error(`Line ${line}, column ${column}: ${error instanceof Error ? error.message : 'Invalid JSON'}`);
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Configuration must be a JSON object.');
  if (!value.beginner || typeof value.beginner !== 'object' || Array.isArray(value.beginner)) throw new Error('Add a beginner settings object.');
  if (!value.expert || typeof value.expert !== 'object') throw new Error('Add an expert settings object.');
  // Accept the original array export when opening older saved files.
  const expert = Array.isArray(value.expert)
    ? Object.fromEntries(value.expert.map(p => [p.id, p.value]))
    : value.expert;
  for (const [key, v] of Object.entries(expert)) {
    if (!['string', 'number', 'boolean'].includes(typeof v)) throw new Error(`${key}: use a text, number or true/false value.`);
  }
  const { model: _model, exportedAt: _time, ...rest } = value;
  return { ...rest, schema: 'strata.engine/1', beginner: value.beginner as BeginnerConfig, expert } as ConfigDocument;
}

export function configPatch(doc: ConfigDocument) {
  const { schema: _schema, beginner, expert, ...extra } = doc;
  return { beginner, expert: Object.entries(expert).map(([id, value]) => ({ id, value })), extra, replaceExpert: true };
}

export function draftParams(engine: EngineConfig, doc: ConfigDocument): ExpertParam[] {
  const known = new Set(engine.expert.map(p => p.id));
  return [
    ...engine.expert.map(p => ({ ...p, value: doc.expert[p.id] ?? p.value })),
    ...Object.entries(doc.expert).filter(([id]) => !known.has(id)).map(([id, value]) => ({
      id, value, recommended: value, label: id, group: 'Additional settings',
      effect: 'Preserved; no verified engine mapping in this build.',
      type: (typeof value === 'boolean' ? 'toggle' : typeof value === 'number' ? 'number' : 'text') as ExpertParam['type'],
    })),
  ];
}

/** Locate the first grammar error consistently across browser JSON.parse messages. */
export function jsonErrorOffset(text:string):number {
  let i=0;
  const whitespace=()=>{while(/[\t\r\n ]/.test(text[i]??'!'))i++;};
  const fail=()=>{throw i;};
  const string=()=>{
    if(text[i++]!=='"') {i--;fail();}
    while(i<text.length){
      const c=text[i++];
      if(c==='"')return;
      if(c.charCodeAt(0)<32){i--;fail();}
      if(c==='\\'){
        const escape=text[i++];
        if(escape==='u'){for(let j=0;j<4;j++)if(!/[a-fA-F0-9]/.test(text[i++]??'!')){i--;fail();}}
        else if(!escape||!['"','\\','/','b','f','n','r','t'].includes(escape)){i--;fail();}
      }
    }
    fail();
  };
  const value=()=>{
    whitespace();const c=text[i];
    if(c==='"'){string();return;}
    if(c==='{'||c==='['){
      const object=c==='{',end=object?'}':']';i++;whitespace();if(text[i]===end){i++;return;}
      while(i<text.length){
        if(object){string();whitespace();if(text[i++]!==':'){i--;fail();}}
        value();whitespace();if(text[i]===end){i++;return;}
        if(text[i++]!==','){i--;fail();}whitespace();
      }fail();
    }
    const match=text.slice(i).match(/^(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/);
    if(!match)fail();else i+=match[0].length;
  };
  try{value();whitespace();if(i!==text.length)fail();}catch{return Math.min(i,text.length);}
  return i;
}
