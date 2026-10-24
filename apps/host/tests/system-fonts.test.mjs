import assert from 'node:assert/strict';
import {test} from 'node:test';
import {systemFonts} from '../server/fonts.mjs';
test('macOS font families are collected across files and deduplicated',async()=>{
 const result=await systemFonts({platform:'darwin',run:async(cmd,args)=>{assert.equal(cmd,'/usr/sbin/system_profiler');assert(args.includes('SPFontsDataType'));return{stdout:JSON.stringify({SPFontsDataType:[{typefaces:[{family:'Arial'},{family:'Arial'},{family:'Helvetica'}]}]})};}});
 assert.deepEqual(result,{fonts:['Arial','Helvetica'],available:true});
});
test('Windows supports one or many installed font families',async()=>{
 assert.deepEqual(await systemFonts({platform:'win32',run:async()=>({stdout:'["Segoe UI","Arial"]'})}),{fonts:['Arial','Segoe UI'],available:true});
 assert.deepEqual(await systemFonts({platform:'win32',run:async()=>({stdout:'"Arial"'})}),{fonts:['Arial'],available:true});
});
test('Linux fontconfig aliases are normalized without listing font file paths',async()=>{
 const result=await systemFonts({platform:'linux',run:async(cmd,args)=>{assert.equal(cmd,'fc-list');assert.deepEqual(args,['-f','%{family}\n']);return{stdout:'Noto Sans,Noto Sans UI\nDejaVu Sans\nNoto Sans\n'};}});
 assert.deepEqual(result,{fonts:['DejaVu Sans','Noto Sans','Noto Sans UI'],available:true});
});
test('font discovery failure keeps manual installed-font selection usable',async()=>{
 assert.deepEqual(await systemFonts({platform:'linux',run:async()=>{throw new Error('fontconfig unavailable');}}),{fonts:[],available:false});
});
