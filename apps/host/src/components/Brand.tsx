import { tr } from '../lib/i18n';
/** The approved three-bar mark and outlined wordmark, sized as artwork. */
import { useEffect, useState, type CSSProperties } from 'react';
import { useMedia } from '../lib/responsive';
import { useStore } from '../lib/store';
export function Brand({ variant, height, decorative = false, className = '', }: {
    variant: 'mark' | 'lockup';
    height: number;
    decorative?: boolean;
    className?: string;
}) {
    const preference = useStore(s => s.theme);
    const systemDark = useMedia('(prefers-color-scheme: dark)');
    const theme = preference === 'system' ? (systemDark ? 'dark' : 'light') : preference;
    const src = `/brand/strata-${variant}-${theme}.svg`;
    const [failed, setFailed] = useState(false);
    useEffect(() => setFailed(false), [src]);
    const width = variant === 'mark' ? height : height * 182 / 48;
    const style = { width, height } satisfies CSSProperties;
    return <span className={`brand brand--${variant} ${className}`} style={style} role={decorative ? undefined : 'img'} aria-label={tr(decorative ? undefined : 'Strata')} aria-hidden={decorative || undefined}>
    {tr(!failed ? <img className="brand__img" src={src} width={width} height={height} alt={tr("")} onError={() => setFailed(true)}/> : null)}
    <span className="brand__fallback" aria-hidden="true">{tr(variant === 'mark' ? 'S' : 'Strata')}</span>
  </span>;
}
