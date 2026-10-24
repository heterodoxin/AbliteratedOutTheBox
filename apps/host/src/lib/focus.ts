/**
 * Focus management for surfaces that reveal themselves.
 *
 * Overlays in this app stay mounted and animate open, which means their
 * `visibility` flips through a discrete transition. A `focus()` call in the same
 * commit as the open is therefore dropped — the element is still
 * `visibility: hidden` when the effect runs — which left the command palette
 * open with focus on a sidebar button *behind* it.
 *
 * So focus is applied once the surface actually reports visible, Tab is kept
 * inside while it is open, and focus returns to whatever opened it on close.
 */
import { useEffect, useRef, type RefObject } from 'react';

const FOCUSABLE = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  'summary',
  '[tabindex]:not([tabindex="-1"])',
].join(',');
const activeSurfaces: { identity: symbol; root: HTMLElement }[] = [];
export function isTopSurface(root: HTMLElement | null) { return activeSurfaces.at(-1)?.root === root; }
const managedInert = new Map<HTMLElement, boolean>();

/** Recompute from the top surface instead of restoring stale sibling snapshots.
 * React may open one mounted dialog in the same commit that closes another.
 * Its current data-open value takes precedence over the previous inert value.
 */
function syncBackgroundInert() {
  for (const [node, previous] of managedInert) {
    node.inert = node.matches('[role="dialog"][data-open]')
      ? node.dataset.open !== 'true'
      : previous;
  }
  managedInert.clear();
  const top = activeSurfaces.at(-1)?.root;
  if (!top) return;
  let branch = top;
  while (branch.parentElement) {
    // A lower surface may previously have made this branch inert.
    branch.inert = false;
    for (const node of Array.from(branch.parentElement.children)) {
      if (node instanceof HTMLElement && node !== branch && !node.matches('.scrim,.dialog-scrim,.nav-scrim')) {
        managedInert.set(node, node.inert);
        node.inert = true;
      }
    }
    branch = branch.parentElement;
    if (branch === document.body) break;
  }
}

function focusables(root: HTMLElement): HTMLElement[] {
  return [...root.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(
    (el) => !el.closest('[inert]') && getComputedStyle(el).visibility === 'visible' && (el.offsetParent !== null || el === document.activeElement),
  );
}

/**
 * @param open    whether the surface is showing
 * @param ref     the surface element
 * @param initial optional element to focus instead of the first focusable one
 */
export function useSurfaceFocus(
  open: boolean,
  ref: RefObject<HTMLElement | null>,
  initial?: RefObject<HTMLElement | null>,
) {
  const token = useRef(Symbol('surface'));
  const returnTo = useRef<HTMLElement | null>(null);
  const wasOpen = useRef(false);

  useEffect(() => {
    if (!open || !ref.current) return;
    const identity = token.current;
    returnTo.current = document.activeElement as HTMLElement | null;
    wasOpen.current = true;
    activeSurfaces.push({ identity, root: ref.current });
    syncBackgroundInert();
    return () => {
      const at = activeSurfaces.findIndex(surface => surface.identity === identity);
      if (at >= 0) activeSurfaces.splice(at, 1);
      syncBackgroundInert();
    };
  }, [open, ref]);

  // Focus once the surface is genuinely visible, not merely mounted.
  useEffect(() => {
    if (!open) return undefined;
    let frame = 0;
    let tries = 0;
    const attempt = () => {
      const el = ref.current;
      if (!el) return;
      if (getComputedStyle(el).visibility !== 'visible' && tries < 30) {
        tries += 1;
        frame = requestAnimationFrame(attempt);
        return;
      }
      const target = initial?.current ?? el.querySelector<HTMLElement>('[data-initial-focus]') ?? focusables(el)[0] ?? el;
      target?.focus();
    };
    frame = requestAnimationFrame(attempt);
    // A page that is not being painted (a window that is covered or in the background) holds animation frames back, and then focus would stay
    // on whatever opened the surface, behind it (QA F03). A timer is not held back, so it moves focus in if the frame did not.
    const fallback = setTimeout(() => { const el = ref.current; if (el && !el.contains(document.activeElement) && isTopSurface(el)) attempt(); }, 80);
    return () => { cancelAnimationFrame(frame); clearTimeout(fallback); };
  }, [open, ref, initial]);

  // Keep Tab inside while open.
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e: KeyboardEvent) => {
      if (activeSurfaces.at(-1)?.identity !== token.current) return;
      if (e.key !== 'Tab') return;
      const el = ref.current;
      if (!el) return;
      const items = focusables(el);
      if (!items.length) {
        e.preventDefault();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement as HTMLElement | null;
      const outside = !active || !el.contains(active);
      if (e.shiftKey && (active === first || outside)) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (active === last || outside)) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', onKey, true);
    return () => document.removeEventListener('keydown', onKey, true);
  }, [open, ref]);

  // Hand focus back on close — but not on first mount, when nothing was taken.
  useEffect(() => {
    if (open || !wasOpen.current) return;
    wasOpen.current = false;
    const target = returnTo.current;
    const top = activeSurfaces.at(-1)?.root;
    if (target && document.contains(target) && !target.closest('[inert]') && (!top || top.contains(target))) target.focus();
  }, [open]);
}
