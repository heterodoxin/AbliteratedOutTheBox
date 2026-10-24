/**
 * Which saved skills an answer used, for the line under the answer ("Used 2 skills: Weekly report, Brand voice").
 *
 * The agent loads a skill with its `skill` tool (the name is the folder name the app wrote, see skill-files.mjs). Each skill that was loaded
 * successfully during the turn is reported once, in a `skills-used` frame that carries the whole list so far (the page replaces its copy), with the
 * name the person gave it and its one-line description from the skill's own file. A skill the app does not know by file (a folder the agent
 * found elsewhere) is named from its folder name. Pure apart from reading `<dir>/<name>/SKILL.md`.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseSkillFile } from '../skill-files.mjs';

/** "weekly-report" -> "Weekly report" */
export const skillLabel = slug => {
  const text = String(slug ?? '').replace(/[-_]+/g, ' ').trim();
  return text ? text[0].toUpperCase() + text.slice(1) : '';
};

/** A function from the skill's folder name to `{ name, description }`, read from the app's skills folder. */
export function skillInfoFrom(dir) {
  return slug => {
    const safe = /^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$/.test(String(slug)) ? String(slug) : null;
    let parsed = null;
    if (safe && dir) { try { parsed = parseSkillFile(readFileSync(join(dir, safe, 'SKILL.md'), 'utf8')); } catch { /* not a file of ours */ } }
    const name = (parsed?.title || skillLabel(slug)).slice(0, 80);
    // the app writes "<name>. <description>" into the file so the agent's catalog names the skill: the line shows only the description
    let description = String(parsed?.description ?? '').replace(/\s+/g, ' ').trim();
    if (description.toLowerCase().startsWith(`${name.toLowerCase()}. `)) description = description.slice(name.length + 2);
    return { name, description: description.slice(0, 200) };
  };
}

export class SkillsUsed {
  constructor({ emit, info = slug => ({ name: skillLabel(slug), description: '' }) }) { this.emit = emit; this.info = info; this.used = new Map(); }
  /** A tool card frame (every state change). Counts a `skill` call once, when it has worked. */
  card(tool) {
    if (!tool || tool.name !== 'skill' || tool.state !== 'done') return;
    const slug = String(tool.args?.name ?? tool.args?.skill ?? '').trim();
    if (!slug || this.used.has(slug)) return;
    this.used.set(slug, this.info(slug));
    this.emit({ type: 'skills-used', skills: [...this.used.values()] });
  }
}
