import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {transform} from 'esbuild';
const {code}=await transform(await readFile(new URL('../src/lib/onboarding-download.ts',import.meta.url),'utf8'),{loader:'ts',format:'esm'});
const {downloadCounter,onboardingError,modelFolderLabel}=await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
test('download remaining reflects the current shard rather than the whole model',()=>{
 assert.deepEqual(downloadCounter(66.4,{file:'shard-1',total_gb:22.1,done_gb:8.4,percent:38},true),{value:13.700000000000001,label:'GB left in this file',key:'shard-1'});
 assert.ok(downloadCounter(66.4,{file:'shard-1',total_gb:22.1,done_gb:15.9},true).value<6.3);
 assert.equal(downloadCounter(66.4,{file:'shard-2',total_gb:22.1,done_gb:0},true).key,'shard-2');
 assert.equal(downloadCounter(66.4,{remaining_gb:0},true).value,0);
});
test('unknown download totals never invent a countdown',()=>{
 assert.equal(downloadCounter(66.4,{percent:50},true).value,undefined);
 assert.equal(downloadCounter(66.4,undefined,true).value,undefined);
 assert.equal(downloadCounter(66.4,undefined,false).value,66.4);
 assert.equal(downloadCounter(66.4,{total_gb:10,percent:150},true).value,0);
});
test('storage errors expose needed/free space without paths or command flags',()=>{
 for(const location of ['/Users/name/Documents/Private','D:\\private\\weights']){
  const result=onboardingError(`Not enough free disk space for Q2_0: about 75 GB needed at ${location}, 18 GB free; choose data_dir (e.g. D:\\Strata-data)`);
  assert.equal(result.storage,true);assert.equal(result.message,'This model needs about 75 GB. You have 18 GB free.');
  assert.equal(JSON.stringify(result).includes(location),false);assert.equal(JSON.stringify(result).includes('data_dir'),false);
 }
 assert.equal(onboardingError('ENOSPC: write failed at /secret').message,'There isn’t enough free space to download this model.');
 assert.equal(onboardingError('Network unavailable').message,'Network unavailable');
});
test('default folder labels are readable across platform separators',()=>{
 assert.equal(modelFolderLabel('C:\\Users\\Me\\Documents\\Strata\\Models'),'Documents / Strata / Models');
 assert.equal(modelFolderLabel('/home/me/Documents/Strata/Models'),'Documents / Strata / Models');
 assert.equal(modelFolderLabel('/mnt/external/models'),'/mnt/external/models');
});
