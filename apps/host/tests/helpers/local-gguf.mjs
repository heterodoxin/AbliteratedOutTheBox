/** Tiny, structurally valid fixtures. They are never real inference weights. */
import { writeFileSync } from 'node:fs';
const u32=n=>{const b=Buffer.alloc(4);b.writeUInt32LE(n);return b;};
const u64=n=>{const b=Buffer.alloc(8);b.writeBigUInt64LE(BigInt(n));return b;};
const string=s=>Buffer.concat([u64(Buffer.byteLength(s)),Buffer.from(s)]);
export function ggufFixture(file,{arch='qwen3next',name='OrcaRouter fixture',context=32768,splitNo,splitCount,tensorName='test.weight',tensorCount=1,version=3,extra=[],truncate=0}={}) {
 const entries=[['general.architecture',8,arch],['general.name',8,name],['general.type',8,'model'],['general.file_type',4,15],[`${arch}.context_length`,4,context],[`${arch}.block_count`,4,48],[`${arch}.embedding_length`,4,1024],[`${arch}.attention.head_count`,4,16],[`${arch}.attention.head_count_kv`,4,2],[`${arch}.expert_count`,4,128],['tokenizer.ggml.model',8,'gpt2'],['tokenizer.ggml.tokens',9,['token']],['tokenizer.chat_template',8,'fixture only; no instructions'],...extra];
 if(splitCount!==undefined)entries.push(['split.count',2,splitCount],['split.no',2,splitNo],['split.tensors.count',5,splitCount]);
 const scalar=(type,value)=>type===8?string(value):type===9?Buffer.concat([u32(8),u64(value.length),...value.map(string)]):type===2?(()=>{const b=Buffer.alloc(2);b.writeUInt16LE(value);return b;})():type===5?(()=>{const b=Buffer.alloc(4);b.writeInt32LE(value);return b;})():u32(value);
 const header=Buffer.concat([Buffer.from('GGUF'),u32(version),u64(tensorCount),u64(entries.length),...entries.map(([key,type,value])=>Buffer.concat([string(key),u32(type),scalar(type,value)])),...(tensorCount?[Buffer.concat([string(tensorName),u32(1),u64(16),u32(0),u64(0)])]:[])]);
 const bytes=Buffer.concat([header,Buffer.alloc((32-header.length%32)%32),Buffer.alloc(tensorCount?64:0)]);
 writeFileSync(file,truncate?bytes.subarray(0,bytes.length-truncate):bytes);return bytes;
}
