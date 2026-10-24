/** Bounded metadata/tensor-directory reader. Tensor weights are never read into memory. */
import { closeSync, fstatSync, openSync, readSync } from 'node:fs';
const MAX_HEADER = 96 * 1024 ** 2, MAX_COUNT = 2_000_000;
const widths = { 0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8 };
// ggml block width / bytes. See gguf-py/gguf/constants.py, GGML_QUANT_SIZES.
const tensorSizes = {0:[1,4],1:[1,2],2:[32,18],3:[32,20],6:[32,22],7:[32,24],8:[32,34],9:[32,36],10:[256,84],11:[256,110],12:[256,144],13:[256,176],14:[256,210],15:[256,292],16:[256,66],17:[256,74],18:[256,98],19:[256,50],20:[32,18],21:[256,110],22:[256,82],23:[256,136],24:[1,1],25:[1,2],26:[1,4],27:[1,8],28:[1,8],29:[256,56],30:[1,2],34:[256,54],35:[256,66],39:[32,17],40:[64,36],41:[128,18],42:[64,18]};
export function readGGUF(file) {
 const fd=openSync(file,'r');
 try {
  const stat=fstatSync(fd);if(!stat.isFile())throw new Error('Choose a GGUF file.');
  let p=0,cacheStart=-1,cacheBytes=0;const cache=Buffer.alloc(65536);
  const bounded=n=>{if(!Number.isSafeInteger(n)||n<0||p+n>stat.size||p+n>MAX_HEADER)throw new Error('The GGUF header is truncated or exceeds the supported metadata size.');};
  function bytes(n){bounded(n);if(n>cache.length){const b=Buffer.alloc(n);if(readSync(fd,b,0,n,p)!==n)throw new Error('Could not read GGUF metadata.');p+=n;return b;}if(p<cacheStart||p+n>cacheStart+cacheBytes){cacheStart=p;cacheBytes=readSync(fd,cache,0,Math.min(cache.length,stat.size-p),p);}const b=cache.subarray(p-cacheStart,p-cacheStart+n);p+=n;return b;}
  function u64(){const n=bytes(8).readBigUInt64LE();if(n>BigInt(Number.MAX_SAFE_INTEGER))throw new Error('GGUF metadata contains an unsupported integer.');return Number(n);}
  const u32=()=>bytes(4).readUInt32LE();
  function string(keep=true,limit=8*1024**2){const n=u64();if(n>limit)throw new Error('GGUF metadata contains an oversized string.');bounded(n);if(!keep){p+=n;return null;}return bytes(n).toString('utf8');}
  function value(type,keep,depth=0){
   if(type===8)return string(keep);
   if(type===9){if(depth>4)throw new Error('GGUF metadata arrays are too deeply nested.');const element=u32(),count=u64();if(count>MAX_COUNT)throw new Error('GGUF metadata array is too large.');if(widths[element]){bounded(count*widths[element]);p+=count*widths[element];}else for(let i=0;i<count;i++)value(element,false,depth+1);return {count};}
   const w=widths[type];if(!w)throw new Error('GGUF metadata uses an unsupported value type.');const b=bytes(w);if(!keep)return null;
   if(type===0||type===7)return b.readUInt8();if(type===1)return b.readInt8();if(type===2)return b.readUInt16LE();if(type===3)return b.readInt16LE();if(type===4)return b.readUInt32LE();if(type===5)return b.readInt32LE();if(type===6)return b.readFloatLE();if(type===10){const n=b.readBigUInt64LE();if(n>BigInt(Number.MAX_SAFE_INTEGER))throw new Error('GGUF integer is too large.');return Number(n);}if(type===11){const n=b.readBigInt64LE();if(n>BigInt(Number.MAX_SAFE_INTEGER)||n<BigInt(Number.MIN_SAFE_INTEGER))throw new Error('GGUF integer is too large.');return Number(n);}return b.readDoubleLE();
  }
  if(bytes(4).toString()!=='GGUF')throw new Error('This is not a GGUF model file.');
  const version=u32();if(![2,3].includes(version))throw new Error('This GGUF version or byte order is not supported.');
  const tensorCount=u64(),count=u64();if(tensorCount>MAX_COUNT||count>100_000)throw new Error('GGUF metadata contains too many entries.');
  const metadata=Object.create(null),keys=new Set();
  for(let i=0;i<count;i++){
   const key=string(true,65535);if(keys.has(key))throw new Error('GGUF metadata contains a duplicate key.');keys.add(key);
   const keep=/^(general\.(architecture|name|type|file_type|alignment|size_label)|split\.(no|count|tensors.count)|tokenizer\.(ggml.model|ggml.tokens|chat_template))$/.test(key)||/\.(context_length|block_count|embedding_length|expert_count|attention\.(head_count|head_count_kv|key_length|value_length))$/.test(key);
   const item=value(u32(),keep);if(keep)metadata[key]=key==='tokenizer.chat_template'?!!item:item;
  }
  const alignment=metadata['general.alignment']??32;if(!Number.isInteger(alignment)||alignment<1||alignment>4096||(alignment&(alignment-1)))throw new Error('GGUF tensor alignment is invalid.');
  let tensorEnd=0,parameters=0;const names=new Set();
  for(let i=0;i<tensorCount;i++){
   const name=string(true,1024);if(names.has(name))throw new Error('GGUF contains duplicate tensor names.');names.add(name);
   const dims=u32();if(dims<1||dims>4)throw new Error('GGUF tensor dimensions are invalid.');let elements=1;for(let d=0;d<dims;d++){const size=u64();if(size<1||!Number.isSafeInteger(elements*size))throw new Error('GGUF tensor dimensions are invalid.');elements*=size;}
   const type=u32(),offset=u64(),block=tensorSizes[type];if(!block)throw new Error('This GGUF tensor format is not supported by the importer yet.');if(offset%alignment||elements%block[0])throw new Error('GGUF tensor layout is invalid.');tensorEnd=Math.max(tensorEnd,offset+elements/block[0]*block[1]);parameters+=elements;
  }
  const dataOffset=Math.ceil(p/alignment)*alignment;
  if(tensorEnd+dataOffset>stat.size)throw new Error('The GGUF file is incomplete. Download all of its weights before adding it.');
  return {metadata,version,tensorCount,parameters,size:stat.size,mtimeMs:stat.mtimeMs,ctimeMs:stat.ctimeMs};
 } finally {closeSync(fd);}
}
