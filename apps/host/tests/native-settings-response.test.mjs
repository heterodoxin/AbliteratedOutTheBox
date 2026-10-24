import test from 'node:test';
import assert from 'node:assert/strict';
import {build} from 'esbuild';
const compiled=await build({entryPoints:['src/lib/native-settings-response.ts'],bundle:true,write:false,platform:'node',format:'esm'});
const {nativeSettingsResponse,nativeSettingsValues}=await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
test('settings distinguish a host error body from a valid configuration',()=>{
 assert.throws(()=>nativeSettingsResponse({error:'This model is not installed. Download and prepare it first.'}),/not installed/);
 for(const value of [null,[],{}, {values:null},{values:[]},{values:{},mcp:[]}])assert.throws(()=>nativeSettingsResponse(value));
 assert.deepEqual(nativeSettingsResponse({values:{sampling:{temperature:0.7}}}),{values:{sampling:{temperature:0.7}},mcp:{}});
 assert.deepEqual(nativeSettingsResponse({values:{},mcp:{files:{command:'node'}}}).mcp,{files:{command:'node'}});
});
test('switching from raw JSON rejects scalar or array configurations before rendering fields',()=>{
 for(const value of [null,[],true,5,'text'])assert.throws(()=>nativeSettingsValues(value));
 assert.deepEqual(nativeSettingsValues({}),{});
});
