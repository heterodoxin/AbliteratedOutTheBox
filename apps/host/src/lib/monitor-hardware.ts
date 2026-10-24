import type { GpuInfo, GpuReading } from './types';
export const hasReading=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v);
export const gpuId=(gpu:GpuInfo)=>gpu.id??`${gpu.vendor}:${gpu.index}`;

/** A total, mean, or maximum must cover every selected device. Partial totals are misleading. */
export function gpuMetric(devices:GpuInfo[],readings:GpuReading[]|undefined,key:keyof GpuReading,mode:'sum'|'mean'|'max'='sum'):number|null {
  const values=devices.map(device=>readings?.find(reading=>reading.id===gpuId(device))?.[key]);
  if(!values.length||!values.every(hasReading))return null;
  if(key==='memoryUsedGb'&&mode==='sum') {
    const shared=values.filter((_,i)=>devices[i].unified),dedicated=values.filter((_,i)=>!devices[i].unified);
    return dedicated.reduce((n,v)=>n+v,0)+(shared.length?Math.max(...shared):0);
  }
  return mode==='max'?Math.max(...values):values.reduce((n,v)=>n+v,0)/(mode==='mean'?values.length:1);
}

export function gpuCapacity(devices:GpuInfo[]) {
  const shared=devices.filter(g=>g.unified).map(g=>g.vramGb);
  return devices.filter(g=>!g.unified).reduce((n,g)=>n+g.vramGb,0)+(shared.length?Math.max(...shared):0);
}
