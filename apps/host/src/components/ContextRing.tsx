/** The foreground arc grows with the same percentage shown beside it. */
export function ContextRing({percent}:{percent:number|null}) {
  const value=percent===null?0:Math.max(0,Math.min(100,percent));
  return <svg className="context-ring" viewBox="0 0 20 20" aria-hidden="true" data-unknown={percent===null||undefined} data-level={value>=90?'high':value>=75?'medium':'normal'}>
    <circle className="context-ring__track" cx="10" cy="10" r="7.5"/>
    <circle className="context-ring__fill" cx="10" cy="10" r="7.5" pathLength="100" strokeDasharray={`${value} 100`} transform="rotate(-90 10 10)"/>
  </svg>;
}
