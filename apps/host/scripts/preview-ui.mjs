#!/usr/bin/env node
/** Portable UI review: isolated demo host + native-model fixture. No real inference or weights. */
import {spawn} from 'node:child_process';
import {mkdtempSync,existsSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {resolve,join} from 'node:path';
const root=resolve(import.meta.dirname,'..');
const home=mkdtempSync(join(tmpdir(),'strata-ui-preview-'));
const hostPort=Number(process.env.STRATA_PREVIEW_HOST_PORT??8792);
const webPort=Number(process.env.STRATA_PREVIEW_PORT??5272);
const children=[];let stopping=false;
function stop(code=0){if(stopping)return;stopping=true;for(const child of children)child.kill('SIGTERM');setTimeout(()=>process.exit(code),300).unref();}
function run(args,extra={}){const child=spawn(process.execPath,args,{cwd:root,env:{...process.env,STRATA_HOME:home,STRATA_MODELS_DIR:join(home,'models'),STRATA_WORKSPACE:home,STRATA_REAL_ENGINE:'0',STRATA_NATIVE_REPO:join(home,'no-native-installation'),STRATA_PORT:String(hostPort),...extra},stdio:'inherit'});children.push(child);child.on('error',error=>{console.error(error.message);stop(1);});child.on('exit',code=>{if(!stopping)stop(code??1);});return child;}
process.on('SIGINT',()=>stop());process.on('SIGTERM',()=>stop());
if(!existsSync(join(root,'dist/index.html'))){console.error('Run npm run build before starting the UI preview.');process.exit(1);}
run(['server/index.mjs']);
for(let attempt=0;attempt<100;attempt++){
 if(stopping)break;
 try{const response=await fetch(`http://127.0.0.1:${hostPort}/api/health`);if(response.ok)break;}catch{}
 if(attempt===99){console.error('The isolated preview host did not start. Check the port and console output.');stop(1);}
 await new Promise(resolve=>setTimeout(resolve,150));
}
if(!stopping){
 run(['tests/fixtures/onboarding-ui-host.mjs'],{STRATA_UI_PREVIEW:'1',STRATA_ONBOARDING_UPSTREAM:`http://127.0.0.1:${hostPort}`,STRATA_ONBOARDING_PORT:String(webPort),STRATA_ONBOARDING_START:'ready',STRATA_ONBOARDING_SCENARIO:'telemetry',STRATA_ONBOARDING_PREVIEW:'1'});
 console.log(`\nUI PREVIEW ONLY — hardware, models and telemetry are simulated.\nOpen http://localhost:${webPort}/#/models\nOnboarding: http://localhost:${webPort}/#/setup\nTemporary preview data: ${home}\nCtrl+C stops both preview processes.\n`);
}
