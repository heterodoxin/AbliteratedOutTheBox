import { fileURLToPath } from 'node:url';
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

test('model deletion requires unloading, deletes files, and preserves installation on a file error', t => {
  const directory=mkdtempSync(join(tmpdir(),'strata-storage-test-'));
  t.after(()=>rmSync(directory,{recursive:true,force:true}));
  const result=spawnSync(process.execPath,['--input-type=module','-e',`
    import assert from 'node:assert/strict';
    import fs from 'node:fs';
    import {syncBuiltinESMExports} from 'node:module';
    import {join} from 'node:path';
    const store=await import('./server/store.mjs');
    const models=await import('./server/models.mjs');
    const {VARIANTS}=await import('./server/catalog.mjs');
    const id=VARIANTS[0].id, folder=join(store.MODELS_DIR,id);
    const file=join(folder,VARIANTS[0].source.file);
    fs.mkdirSync(folder,{recursive:true});fs.writeFileSync(file,'GGUFdisposable fixture');fs.writeFileSync(join(folder,'unrelated.txt'),'keep me');
    store.update({installed:{[id]:{installedAt:new Date().toISOString(),simulated:true}},loaded:id,current:id});
    models.init({send(){}});
    assert.throws(()=>models.remove(id),/Unload/);
    assert.ok(fs.existsSync(folder));assert.ok(store.get().installed[id]);
    store.update({loaded:null});
    const original=fs.rmSync;fs.rmSync=()=>{throw new Error('Permission denied fixture');};syncBuiltinESMExports();
    assert.throws(()=>models.remove(id),/Permission denied/);
    assert.ok(store.get().installed[id]);assert.ok(fs.existsSync(folder));
    fs.rmSync=original;syncBuiltinESMExports();
    const library=models.remove(id);
    assert.equal(fs.existsSync(file),false);assert.equal(fs.readFileSync(join(folder,'unrelated.txt'),'utf8'),'keep me');assert.equal(store.get().installed[id],undefined);
    assert.equal(library.variants.find(v=>v.id===id).state,'available');
  `],{cwd:fileURLToPath(new URL('..',import.meta.url)),env:{...process.env,STRATA_HOME:directory,STRATA_MODELS_DIR:join(directory,'models'),STRATA_REAL_ENGINE:'0'},encoding:'utf8'});
  assert.equal(result.status,0,result.stderr||result.stdout);
});
