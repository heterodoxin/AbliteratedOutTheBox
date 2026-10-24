/** The engine options a person set by hand in a model's run config (strata_control/user_args.py is the same rule in Python).
 *  "user_args" holds them; "args_written" is the whole "args" as setup or the app last wrote it, so an option in "args" that is not in
 *  that copy was added or changed by hand since. Every load and every write puts the person's options over the generated ones. */
export const USER_KEY='user_args',WRITTEN_KEY='args_written';
/** Options the app or setup choose from their own answers (context, KV type, pictures, draft layer, layer split): never taken from "args" as hand edits. */
export const CHOICE_FLAGS=new Set(['--max-context','--kv','--vision','--mtp','--spec-min-p','--batch-mtp','--layer-split']);

export function groups(args){
 const out=[];
 for(const raw of Array.isArray(args)?args:[]){const a=String(raw);if(a.startsWith('--')||!out.length)out.push([a]);else out[out.length-1].push(a);}
 return out;
}
export const flat=gs=>gs.flat();
export const flagOf=g=>g?.[0]?.startsWith('--')?g[0]:'';
/** `args` with every option the person set replaced by theirs (appended after the generated ones). */
export function merge(args,user){
 const ug=groups(user),mine=new Set(ug.map(flagOf).filter(Boolean));
 return flat([...groups(args).filter(g=>!mine.has(flagOf(g))),...ug]);
}
function byFlag(gs){const m=new Map();for(const g of gs){const f=flagOf(g);if(!m.has(f))m.set(f,[]);m.get(f).push(g);}return m;}
const same=(a,b)=>JSON.stringify(a??null)===JSON.stringify(b??null);
/** The person's options of a config: its user_args, with the options of "args" that differ from "args_written" on top. */
export function capture(cfg,skip=CHOICE_FLAGS){
 const mine=byFlag(groups(cfg?.[USER_KEY]));
 if(Array.isArray(cfg?.[WRITTEN_KEY])&&Array.isArray(cfg?.args)){
  const cur=byFlag(groups(cfg.args)),was=byFlag(groups(cfg[WRITTEN_KEY]));
  for(const [flag,gs] of cur)if(flag&&!skip.has(flag)&&!same(was.get(flag),gs))mine.set(flag,gs);
  for(const flag of [...mine.keys()])if(was.has(flag)&&!cur.has(flag))mine.delete(flag);
 }
 return flat([...mine.values()].flat());
}
export function setFlag(user,flag,value){
 const rest=groups(user).filter(g=>flagOf(g)!==flag);
 if(value!==null&&value!==undefined)rest.push([flag,String(value)]);
 return flat(rest);
}
export function flagValue(args,flag){
 const list=(Array.isArray(args)?args:[]).map(String),at=list.indexOf(flag);
 return at>=0&&at+1<list.length?list[at+1]:null;
}
/** `config` with the person's options merged into "args", recorded in "user_args", and "args_written" refreshed (only when the config has one). */
export function stamped(config,user){
 const next={...config};
 if(user.length){next.args=merge(next.args,user);next[USER_KEY]=user;}else delete next[USER_KEY];
 if(WRITTEN_KEY in config)next[WRITTEN_KEY]=[...(next.args??[])];
 return next;
}
