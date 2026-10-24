import { rawUrlOf } from './deliverables';

/**
 * Which picture addresses a model's Markdown may put in an <img> (tools and images review O3, 2026-10-22).
 *
 * An answer can carry `![chart](...)`. It is shown only when the picture is already on this computer or inside the message:
 *   - a data: address of a PNG, JPEG, WebP or GIF (never SVG: an SVG can carry script and styles);
 *   - the host's own /api/files/raw address (a file inside the chat's folder, served by the host and by nobody else);
 *   - a path without a scheme, which is read as a file of the chat's folder through that same address.
 * Nothing is fetched from the internet by default (an address in the answer would tell its owner that the answer was read, and
 * which IP read it): an http(s) picture stays a link the person can open. file:, javascript:, blob: and every other scheme show as text.
 */
export type ImagePlan =
  | { kind: 'img'; src: string }
  | { kind: 'link'; href: string }
  | { kind: 'text' };

const DATA_IMAGE = /^data:image\/(?:png|jpe?g|webp|gif);base64,[A-Za-z0-9+/]+={0,2}$/i;
const SCHEME = /^[a-z][a-z0-9+.-]*:/i;

export function imagePlan(src: string, sessionId?: string | null): ImagePlan {
  const value = src.trim();
  if (!value) return { kind: 'text' };
  if (DATA_IMAGE.test(value)) return { kind: 'img', src: value };
  if (/^https?:\/\//i.test(value)) return { kind: 'link', href: value };
  if (value.startsWith('/api/files/raw?')) return { kind: 'img', src: value };
  // a drive letter (C:\x.png, C:/x.png) is a path, not a scheme
  if (SCHEME.test(value) && !/^[a-z]:[\\/]/i.test(value)) return { kind: 'text' };
  if (value.startsWith('//') || value.startsWith('\\\\')) return { kind: 'text' };      // another computer's share
  if (!sessionId) return { kind: 'text' };
  return { kind: 'img', src: rawUrlOf(sessionId, value) };
}
