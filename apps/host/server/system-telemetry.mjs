/** Host counters. CPU sockets are measured only when their logical CPU IDs are known. */
import os from 'node:os';
import { readFile, readdir } from 'node:fs/promises';
import { availableMemoryGb } from './hardware.mjs';
import { probe } from './util.mjs';

export function cpuUtilization(previous, current, ids) {
  if (!previous || !ids.length) return null;
  let total=0,idle=0;
  for (const id of ids) {
    const before=previous[id]?.times,after=current[id]?.times;
    if(!before||!after)return null;
    const elapsed=Object.values(after).reduce((n,v)=>n+v,0)-Object.values(before).reduce((n,v)=>n+v,0);
    if(elapsed<0||after.idle<before.idle)return null;
    total+=elapsed; idle+=after.idle-before.idle;
  }
  return total>0?Math.round(Math.max(0,Math.min(100,100*(1-idle/total)))):null;
}

export function diskRate(previous, current, time) {
  const elapsed=previous?(time-previous.time)/1000:0;
  if(elapsed<=0||!current||!previous?.counters)return {readMbs:null,writeMbs:null};
  const read=current.read-previous.counters.read,write=current.write-previous.counters.write;
  return {readMbs:read>=0?read/1024**2/elapsed:null,writeMbs:write>=0?write/1024**2/elapsed:null};
}

export function parseLinuxDiskCounters(raw, devices) {
  let read=0,write=0,found=false;
  for(const line of raw.trim().split('\n')) {
    const fields=line.trim().split(/\s+/);
    if(!devices.includes(fields[2])||fields.length<14)continue;
    const reads=Number(fields[5]),writes=Number(fields[9]);
    if(!Number.isFinite(reads)||!Number.isFinite(writes))continue;
    read+=reads*512; write+=writes*512; found=true;
  }
  return found?{read,write}:null;
}

export function parseMacDiskCounters(raw) {
  const reads=[...raw.matchAll(/"Bytes \(Read\)"\s*=\s*(\d+)/g)],writes=[...raw.matchAll(/"Bytes \(Write\)"\s*=\s*(\d+)/g)];
  return reads.length&&writes.length?{read:reads.reduce((n,m)=>n+Number(m[1]),0),write:writes.reduce((n,m)=>n+Number(m[1]),0)}:null;
}

async function disks() {
  try {
    if(process.platform==='linux') {
      const [raw,devices]=await Promise.all([readFile('/proc/diskstats','utf8'),readdir('/sys/block')]);
      return parseLinuxDiskCounters(raw,devices.filter(d=>!/^(loop|ram|dm-|md|zram)/.test(d)));
    }
    if(process.platform==='darwin')return parseMacDiskCounters(await probe('ioreg',['-r','-c','IOBlockStorageDriver','-l'],1500));
    if(process.platform==='win32') {
      const raw=await probe('powershell.exe',['-NoProfile','-NonInteractive','-Command',"Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk -Filter \"Name='_Total'\" | Select-Object DiskReadBytesPersec,DiskWriteBytesPersec | ConvertTo-Json -Compress"],2000);
      const value=JSON.parse(raw);
      if(value.DiskReadBytesPersec!=null&&value.DiskWriteBytesPersec!=null)return {read:Number(value.DiskReadBytesPersec),write:Number(value.DiskWriteBytesPersec)};
    }
  } catch { /* An unsupported counter stays unavailable. */ }
  return null;
}

export function createSystemSampler() {
  let previousCpu=null,previousDisk=null;
  return async processors => {
    const current=os.cpus(),time=Date.now();
    const cpuLoad=cpuUtilization(previousCpu,current,current.map((_,i)=>i));
    const cpus=processors.map(p=>({id:p.id,utilizationPct:cpuUtilization(previousCpu,current,p.logicalIds??(processors.length===1?current.map((_,i)=>i):[]))}));
    previousCpu=current;
    const [available,counters]=await Promise.all([availableMemoryGb(process.platform),disks()]);
    const rate=diskRate(previousDisk,counters,time);
    previousDisk={time,counters};
    const total=os.totalmem()/1024**3,free=available??os.freemem()/1024**3;
    return {cpuLoad,cpus,ramUsedGb:Math.max(0,Math.min(total,total-free)),ramTotalGb:total,diskReadGbs:rate.readMbs===null?null:rate.readMbs/1024,diskWriteMbs:rate.writeMbs};
  };
}
