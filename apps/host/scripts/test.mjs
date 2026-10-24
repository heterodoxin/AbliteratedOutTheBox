#!/usr/bin/env node
/** Enumerate test files explicitly so npm test works without shell glob expansion. */
import {readdirSync} from 'node:fs';
import {spawn} from 'node:child_process';
import {resolve} from 'node:path';
const root=resolve(import.meta.dirname,'..');
const files=readdirSync(resolve(root,'tests')).filter(name=>name.endsWith('.test.mjs')).sort().map(name=>resolve(root,'tests',name));
const child=spawn(process.execPath,['--test',...files],{cwd:root,stdio:'inherit'});
child.on('error',error=>{console.error(error.message);process.exitCode=1;});
child.on('exit',(code,signal)=>{process.exitCode=code??(signal?1:0);});
