import { useEffect, useRef, useState } from 'react';
import { tweenValue } from '../lib/tween';

const reduced = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

/** The number a reading is moving towards, shown as it travels there (a short count-up or count-down, 600 ms). Tabular figures keep the width steady. */
export function useTweened(target: number | null, duration = 600): number | null {
  const [shown, setShown] = useState<number | null>(target);
  const from = useRef<number | null>(target);
  const frame = useRef(0);
  useEffect(() => {
    if (target === null || from.current === null || reduced() || target === from.current) { from.current = target; setShown(target); return; }
    const start = performance.now(), origin = from.current;
    cancelAnimationFrame(frame.current);
    const step = (now: number) => {
      const value = tweenValue(origin, target, now - start, duration);
      from.current = value; setShown(value);
      if (now - start < duration) frame.current = requestAnimationFrame(step); else { from.current = target; setShown(target); }
    };
    frame.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame.current);
  }, [target, duration]);
  return shown;
}

export function Num({ value, format, className }: { value: number | null; format: (n: number) => string; className?: string }) {
  const shown = useTweened(value);
  return <span className={`num ${className ?? ''}`.trim()} style={{ fontVariantNumeric: 'tabular-nums' }}>{shown === null ? '—' : format(shown)}</span>;
}
