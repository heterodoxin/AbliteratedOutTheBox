/** A browser recognizer adapter. It only emits text; it never submits a message. */
export type DictationState = 'starting'|'listening'|'speaking'|'finishing'|'finished'|'error';
export interface RecognitionResult { isFinal: boolean; [index:number]: {transcript:string}; }
export interface Recognizer {
  continuous: boolean; interimResults: boolean; lang: string;
  onstart: (()=>void)|null; onspeechstart: (()=>void)|null; onspeechend: (()=>void)|null;
  onresult: ((event:{results:ArrayLike<RecognitionResult>})=>void)|null;
  onerror: ((event:{error:string})=>void)|null; onend: (()=>void)|null;
  start: ()=>void; stop: ()=>void; abort: ()=>void;
}
export type RecognizerConstructor = new()=>Recognizer;
type Clock = {set:(fn:()=>void,ms:number)=>unknown;clear:(id:unknown)=>void};
export const DICTATION_SILENCE_MS = 3500;
export const dictationError = (error:string) => ({
  'not-allowed':'Microphone access was denied. Allow it in your browser, then try again.',
  'service-not-allowed':'Speech recognition is unavailable in this browser. Try a browser that supports dictation.',
  'audio-capture':'No microphone is available. Check your microphone connection.',
  'network':'The browser’s speech service could not connect. Check your connection or try another browser.',
  'no-speech':'No speech was detected. Try speaking closer to the microphone.',
  'language-not-supported':'Your browser’s speech service does not support this language.',
}[error] ?? 'Dictation could not start. Try again or use your computer’s built-in dictation.');

export function createDictation(recognizer:Recognizer, options:{lang:string;onState:(state:DictationState,error?:string)=>void;onText:(text:string)=>void}, clock:Clock={set:(fn,ms)=>setTimeout(fn,ms),clear:id=>clearTimeout(id as ReturnType<typeof setTimeout>)}) {
  let active=false, failed=false, finishing=false, disposed=false;
  let silence:unknown=null, finishTimer:unknown=null;
  const clearSilence=()=>{if(silence!==null)clock.clear(silence);silence=null;};
  const clearTimers=()=>{clearSilence();if(finishTimer!==null)clock.clear(finishTimer);finishTimer=null;};
  function finished(){if(disposed||!active)return;active=false;clearTimers();if(!failed)options.onState('finished');}
  function fail(error:string){if(disposed||!active)return;failed=true;clearTimers();options.onState('error',dictationError(error));try{recognizer.abort();}catch{/* Already ended. */}active=false;}
  function stop(){
    if(!active||finishing||disposed)return;finishing=true;clearSilence();options.onState('finishing');
    finishTimer=clock.set(()=>{finished();try{recognizer.abort();}catch{/* Already ended. */}},2000);
    try{recognizer.stop();}catch{finished();try{recognizer.abort();}catch{/* Already ended. */}}
  }
  const armSilence=(ms=DICTATION_SILENCE_MS)=>{clearSilence();if(active&&!finishing)silence=clock.set(stop,ms);};
  recognizer.continuous=true;recognizer.interimResults=true;recognizer.lang=options.lang;
  recognizer.onstart=()=>{if(!active||disposed)return;options.onState('listening');armSilence(10000);};
  recognizer.onspeechstart=()=>{if(!active||finishing||disposed)return;clearSilence();options.onState('speaking');};
  recognizer.onspeechend=()=>{if(!active||finishing||disposed)return;options.onState('listening');armSilence();};
  recognizer.onresult=event=>{
    if(!active||disposed||failed)return;
    // Results contain the whole recognition session; interim entries may be replaced.
    const text=Array.from(event.results,r=>r[0]?.transcript?.trim()??'').filter(Boolean).join(' ');
    options.onText(text);if(!finishing){options.onState('speaking');armSilence();}
  };
  recognizer.onerror=event=>{if(event.error==='aborted'&&finishing){finished();return;}fail(event.error);};
  recognizer.onend=finished;
  return {
    start(){if(disposed||active)return;active=true;failed=false;finishing=false;options.onState('starting');armSilence(10000);try{recognizer.start();}catch{fail('start-failed');}},
    stop,
    dispose(){disposed=true;active=false;clearTimers();recognizer.onstart=recognizer.onspeechstart=recognizer.onspeechend=recognizer.onend=null;recognizer.onresult=recognizer.onerror=null;try{recognizer.abort();}catch{/* Already ended. */}},
  };
}
