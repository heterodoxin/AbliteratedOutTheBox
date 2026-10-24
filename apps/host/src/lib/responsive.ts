import { useEffect, useState } from 'react';

export function useMedia(query: string) {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const media = window.matchMedia(query);
    const update = () => setMatches(media.matches);
    update();
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, [query]);
  return matches;
}

export function scrollBehavior(): ScrollBehavior {
  const override = document.documentElement.dataset.motion;
  return override === 'reduced' || (override !== 'full' && window.matchMedia('(prefers-reduced-motion: reduce)').matches) ? 'instant' : 'smooth';
}

/** The tallest the message box may grow: 30% of the window, between 44 and 180 pixels (the rest of the window keeps the toolbar and Send in view). */
export function composerCap(windowHeight: number): number {
  return Math.max(44, Math.min(180, Math.floor(windowHeight * 0.3)));
}
