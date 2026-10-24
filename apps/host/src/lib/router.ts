/** Hash router — no dependency, survives static hosting and file:// shells. */
import { useEffect, useState } from 'react';

export const ROUTES = [
  'setup',
  'chat',
  'models',
  'tuning',
  'serving',
  'monitor',
  'tasks',
  'skills',
  'settings',
] as const;

export type Route = (typeof ROUTES)[number];

export interface Location {
  route: Route;
  param: string | null;
  query: URLSearchParams;
}

export function parseLocation(hash: string): Location {
  const raw = hash.replace(/^#\/?/, '');
  const [pathPart, queryPart] = raw.split('?');
  const segments = pathPart.split('/').filter(Boolean);
  const first = segments[0] as Route | undefined;
  // Chat is the landing screen; an unknown or retired route (a stale #/launch
  // bookmark) falls back to it rather than rendering nothing.
  const route: Route = first && (ROUTES as readonly string[]).includes(first) ? first : 'chat';
  const query = new URLSearchParams(queryPart ?? '');
  if (route === 'settings') query.set('settings', '1');
  return {
    route: route === 'settings' ? 'chat' : route,
    param: segments[1] ?? null,
    query,
  };
}

export function useLocation(): Location {
  const [loc, setLoc] = useState<Location>(() => {
    const initial = parseLocation(window.location.hash);
    normalise(initial);
    return initial;
  });
  useEffect(() => {
    const onHash = () => {
      const next = parseLocation(window.location.hash);
      normalise(next);
      setLoc(next);
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  return loc;
}

/**
 * Rewrite a retired or unknown hash to the screen actually being shown, so a
 * stale `#/launch` bookmark does not leave a dead URL in the address bar.
 */
function normalise(loc: Location): void {
  const raw = window.location.hash.replace(/^#\/?/, '');
  const first = raw.split('?')[0].split('/').filter(Boolean)[0];
  if (first === 'settings') {
    window.history.replaceState(null, '', locationHref(loc));
  } else if (first && !(ROUTES as readonly string[]).includes(first)) {
    window.history.replaceState(null, '', `#/${loc.route}`);
  }
}

/**
 * Navigate, morphing between screens when the browser can.
 *
 * A view transition needs the DOM update to happen inside its callback, and
 * React commits asynchronously — so we wait for the commit (never for a frame: see afterRouteUpdate). It is
 * skipped while a response is streaming (the transcript is being appended to on
 * every token, and snapshotting it mid-stream would fight the SSE updates) and
 * when the user has asked for reduced motion.
 */
export function navigate(route: Route, param?: string, query?: Record<string, string>) {
  const next = route === 'settings' ? settingsHref(window.location.hash) : locationHref({route, param: param ?? null, query: new URLSearchParams(query)});
  navigateHash(next, route !== 'settings');
}

export function closeSettings() {
  const location = parseLocation(window.location.hash);
  location.query.delete('settings');
  navigateHash(locationHref(location), false);
}

/**
 * WebKitGTK (the Linux app window; Safari does not exist on Linux) crashes its own UI process with a null pointer in
 * AcceleratedBackingStore::update when a page calls the View Transitions API (seen on Ubuntu 24.04 and 26.04, WebKitGTK
 * 2.52, under Xvfb and on a real NVIDIA/Wayland desktop alike; one startViewTransition call is enough). The route
 * cross-fade is only a nicety, so on that engine a route changes at once.
 */
export function isWebKitGtk(ua: string = navigator.userAgent): boolean {
  return /AppleWebKit/.test(ua) && /Linux|X11/.test(ua) && !/Chrome|Chromium|CriOS|Edg|OPR|Android/.test(ua);
}

/**
 * Resolve once the new route has been drawn into the page, or after a short deadline, whichever comes first.
 *
 * The update callback of a view transition must not wait for requestAnimationFrame: while the callback is pending the browser holds
 * rendering (Chromium and Firefox both do), so a frame callback never runs, and every click waits for the browser's own guard,
 * about 4 s in Chromium and the WebView2 window, about 5 s in Firefox (#3673). A DOM change after the hash flipped means React has
 * committed the new screen; the deadline covers a route that changes nothing visible.
 */
export function afterRouteUpdate(done: () => void, timeoutMs = 150): void {
  let settled = false;
  let observer: MutationObserver | undefined;
  const finish = () => {
    if (settled) return;
    settled = true;
    observer?.disconnect();
    done();
  };
  try {
    if (typeof MutationObserver === 'function' && typeof document !== 'undefined' && document.body) {
      observer = new MutationObserver(() => setTimeout(finish, 0));
      observer.observe(document.body, { childList: true, subtree: true });
    }
  } catch {
    /* the deadline alone is enough */
  }
  setTimeout(finish, timeoutMs);
}

function navigateHash(next: string, morph = true) {
  if (window.location.hash === next) return;

  const root = document.documentElement;
  const mayMorph =
    morph &&
    typeof document.startViewTransition === 'function' &&
    !isWebKitGtk() &&
    root.dataset.motion !== 'reduced' &&
    !window.matchMedia('(prefers-reduced-motion: reduce)').matches &&
    root.dataset.streaming !== 'true';

  if (!mayMorph) {
    window.location.hash = next;
    return;
  }

  const transition = document.startViewTransition(
    () =>
      new Promise<void>((resolve) => {
        window.location.hash = next;
        afterRouteUpdate(resolve);
      }),
  );
  // A transition is *expected* to be skipped when one is already running (rapid
  // ⌘1…⌘9) or the document is hidden. That is not an error, and the rejected
  // promises would otherwise surface as unhandled rejections in the console.
  transition.ready.catch(() => {});
  transition.updateCallbackDone.catch(() => {});
  transition.finished.catch(() => {});
}


export function href(route: Route, param?: string): string {
  return route === 'settings' ? settingsHref(window.location.hash) : `#/${route}${param ? `/${param}` : ''}`;
}

function locationHref(location: Location) {
  const query = location.query.toString();
  return `#/${location.route}${location.param ? `/${location.param}` : ''}${query ? `?${query}` : ''}`;
}

/** Keep the working page mounted while Settings is open, including its draft. */
export function settingsHref(hash: string) {
  const location = parseLocation(hash);
  location.query.set('settings', '1');
  return locationHref(location);
}
