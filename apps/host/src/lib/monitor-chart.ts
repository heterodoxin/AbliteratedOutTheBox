/** The original dashboard's 100×32 spark: y=30−value/top×26, 12% area, 1.6px stroke. */
export function referencePaths(data:(number|null|undefined)[],maximum?:number) {
  const top=Math.max(maximum??0,...data.filter((v):v is number=>typeof v==='number'&&Number.isFinite(v)),1e-9);
  const points=data.map((value,i)=>typeof value==='number'&&Number.isFinite(value)?[data.length>1?i/(data.length-1)*100:0,30-value/top*26] as const:null);
  const segments:(readonly [number,number])[][]=[];
  for(const point of points) {
    if(!point){if(segments.at(-1)?.length)segments.push([]);continue;}
    if(!segments.length)segments.push([]);
    segments.at(-1)!.push(point);
  }
  const paths=segments.filter(segment=>segment.length>1).map(segment=>{
    const line=segment.map(([x,y],i)=>`${i?'L':'M'}${x.toFixed(2)},${y.toFixed(2)}`).join('');
    return {line,area:`${line}L${segment.at(-1)![0].toFixed(2)},32L${segment[0][0].toFixed(2)},32Z`};
  });
  return {points,paths,top};
}
