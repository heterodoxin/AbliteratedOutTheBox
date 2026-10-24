import type { MonitorState, Session } from './types';

export function chatMetrics(session:Session|null, monitor:MonitorState|null, configuredContext?:number, now:number=Date.now()) {
  const messages=session?.messages??[];
  const replies=messages.filter(m=>m.role==='assistant'&&!m.timeSkipId);
  const latest=[...replies].reverse().find(m=>m.inputTokens!==undefined);
  const remote=replies.at(-1)?.remoteModel===true;
  const working=replies.some(m=>m.streaming);
  // The Monitor's reading belongs to this chat when it names it, and also while this chat is the one working: a goal or an agent runs under its own
  // session ids, so matching on the chat id alone left the numbers frozen during a goal.
  const active=!remote&&!!session&&!!monitor&&(monitor.context?.sessionId===session.id||(working&&['reading','thinking','generating'].includes(monitor.phase)));
  // One source: what the context held after the last reply (the engine's own count for that request); the monitor's live figure only while a reply streams.
  const finished=latest?(latest.contextUsed??latest.inputTokens!+(latest.tokens??0)):null;
  const liveUsed=active&&Number.isFinite(monitor!.context.usedTokens)&&monitor!.context.usedTokens>0?monitor!.context.usedTokens:null;
  const live=replies.some(m=>m.streaming)||(active&&['reading','thinking','generating'].includes(monitor!.phase));
  // While a turn runs, its own figure comes first: the host reports what the context holds after each model request of the turn. The engine's live
  // figure belongs to the request in flight (a small one between two tools read as 1%), and `finished` to an earlier reply.
  const turnUsed=replies.find(m=>m.streaming&&Number.isFinite(m.contextUsed)&&m.contextUsed!>0)?.contextUsed??null;
  const used=live?(turnUsed??liveUsed??finished):(finished??liveUsed);
  const max=remote?(replies.at(-1)?.contextMax||null):configuredContext??(active?monitor!.context.maxTokens:latest?.contextMax??null);
  const streaming=replies.find(m=>m.streaming);
  // Answer speed: everything the model generated over its decode time. While a reply streams it is that reply's; after it, the LAST reply's own
  // (never an older reply's number under the words "last reply").
  const lastReply=streaming??replies.at(-1);
  const speed=lastReply?.tps??null,promptSpeed=lastReply?.promptTps??null;
  const ttfts=replies.flatMap(m=>m.ttftMs===undefined?[]:[m.ttftMs]);
  const hasUsage=session?.statsAvailable===true;
  const llmTotal=replies.reduce((n,m)=>n+(m.durationMs??(m.streaming?Math.max(0,now-(Date.parse(m.at)||now)):0)),0),thinkTotal=replies.reduce((n,m)=>n+(m.reasoningMs??(m.streaming&&m.reasoningStartedAt?Math.max(0,now-m.reasoningStartedAt):0)),0);
  return {streaming:!!streaming,writeMs:Math.max(0,llmTotal-thinkTotal),estimated:streaming?.usageSource==='estimated',hasReplies:replies.length>0,example:!!session?.showcase||replies.some(m=>m.usageSource==='demo'),
    speed:active&&['generating','thinking'].includes(monitor?.phase??'')&&monitor!.outputTps>0?monitor!.outputTps:speed,promptSpeed,
    total:session?.statsAvailable===false?null:hasUsage?session.tokensIn+session.tokensOut+(streaming?.tokens??0)+(streaming?.inputTokens??0):replies.every(m=>m.inputTokens!==undefined&&m.tokens!==undefined)?replies.reduce((n,m)=>n+m.inputTokens!+m.tokens!,0):null,
    input:hasUsage?session.tokensIn:null,output:hasUsage?session.tokensOut:null,
    llmMs:replies.reduce((n,m)=>n+(m.durationMs??(m.streaming?Math.max(0,now-(Date.parse(m.at)||now)):0)),0),
    thinkMs:replies.reduce((n,m)=>n+(m.reasoningMs??(m.streaming&&m.reasoningStartedAt?Math.max(0,now-m.reasoningStartedAt):0)),0),
    toolMs:replies.reduce((n,m)=>n+(m.tools??[]).reduce((v,t)=>v+(t.durationMs??0),0),0),
    ttft:ttfts.length?ttfts.reduce((n,v)=>n+v,0)/ttfts.length:null,
    used,max,pct:used!==null&&max?Math.min(100,100*used/max):null,
    turns:messages.filter(m=>m.role==='user').length,
    steps:messages.reduce((n,m)=>n+(m.tools?.length??0),0)};
}
export const compactTokens=(n:number|null)=>n===null?'—':new Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(n);
export function elapsed(ms:number){if(ms<1000)return `${Math.round(ms)} ms`;const sec=Math.round(ms/1000);return sec<60?`${sec}s`:`${Math.floor(sec/60)}m ${sec%60}s`;}
