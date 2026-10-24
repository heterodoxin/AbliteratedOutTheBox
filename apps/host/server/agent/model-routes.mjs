/**
 * Which model did the person (or the assistant) mean when a helper agent is started with a model?
 *
 * The Harness's `subagent` tool wants an exact `provider` route key and an exact model id. People say "ask DeepSeek", "use the local
 * model", "ask deepseek/deepseek-v3.2" or a display name in other letters; a route key such as `api-1b2c3d4e-5f` is something no
 * model can guess. This module turns what was asked into one route, or into a plain answer that says what to do next.
 *
 * Pure and dependency free: the profile writes the route table, the `strata-models` Harness plugin and the tests call it.
 *
 * A route is `{ provider, model, name, connection, local }`:
 *   provider    the route key in the Harness profile (what the Harness needs)
 *   model       the model id the server knows
 *   name        the name the person sees in the model picker
 *   connection  the connection's name ("OpenRouter") or "this computer" for a local model
 *   local       true for a model running in the Strata engine
 */

/** The words people use for the model that runs in Strata on this computer. */
const LOCAL_WORDS = new Set(['local', 'localmodel', 'localstrata', 'thelocalmodel', 'mylocalmodel', 'strata', 'stratalocal', 'stratamodel', 'localstratamodel', 'strataslocalmodel', 'default', 'thismodel', 'samemodel', 'themodel', 'thislocalmodel']);

/** Letters and digits only, lower case: "DeepSeek: V3.2" and "deepseek v3.2" meet. */
export const squash = value => String(value ?? '').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');
const words = value => String(value ?? '').toLowerCase().split(/[^\p{L}\p{N}.]+/u).filter(Boolean);

/** Edit distance, capped: enough to tell "deepseek-v3.2" from "deepseek-v3.1" and "depseek" from "deepseek". */
function distance(a, b, cap = 4) {
  if (Math.abs(a.length - b.length) > cap) return cap + 1;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    let best = row[0];
    for (let j = 1; j <= b.length; j++) {
      row[j] = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      best = Math.min(best, row[j]);
    }
    if (best > cap) return cap + 1;
    prev = row;
  }
  return prev[b.length];
}

/** Every spelling by which this route can be asked for, as squashed strings (an exact spelling, not a guess). */
function exactKeys(route) {
  const keys = new Set();
  const add = value => { const s = squash(value); if (s) keys.add(s); };
  add(route.model); add(route.name);
  add(`${route.provider}/${route.model}`);
  add(`${route.connection}/${route.model}`); add(`${route.connection} ${route.name}`); add(`${route.connection}: ${route.name}`); add(`${route.name} ${route.connection}`);
  add(`${route.name} (${route.connection})`);
  return keys;
}

/** The loose pieces a guess can match: the id, its last part after a slash, the name and the connection. */
function looseForms(route) {
  const forms = [route.model, String(route.model).split('/').pop(), route.name, `${route.connection} ${route.name}`];
  return forms.map(squash).filter(Boolean);
}

/** One line for a model in a list the assistant or the person reads. */
export function routeLine(route) {
  const where = route.local ? 'running on this computer' : route.connection;
  return `${route.name} (${where}): provider "${route.provider}", model "${route.model}"`;
}

/** The whole list as text, kept short enough for a small model's context. */
export function routeList(routes, { max = 30 } = {}) {
  const lines = routes.slice(0, max).map(r => `- ${routeLine(r)}`);
  if (routes.length > max) lines.push(`- ...and ${routes.length - max} more (ask for the name of one of them, or the part of its id you know)`);
  return lines.join('\n');
}

/**
 * Resolve one request.
 * @param {Array<{provider:string,model:string,name:string,connection:string,local:boolean}>} routes
 * @param {{provider?:string, model?:string}} request what the caller wrote (either or both)
 * @returns {{status:'inherit'}|{status:'exact',route:object,how:string}|{status:'confirm',route:object,others:object[]}|{status:'ambiguous',options:object[]}|{status:'none'}}
 *   inherit: nothing was asked (the helper uses the chat's model); exact: one model; confirm: one close guess to check with the person;
 *   ambiguous: several models fit; none: nothing fits.
 */
export function resolveModel(routes, request = {}) {
  const provider = typeof request.provider === 'string' ? request.provider.trim() : '';
  const model = typeof request.model === 'string' ? request.model.trim() : '';
  if (!provider && !model) return { status: 'inherit' };
  // 1. The exact route the Harness knows.
  if (provider && model) {
    const hit = routes.find(r => r.provider === provider && r.model === model);
    if (hit) return { status: 'exact', route: hit, how: 'route' };
  }
  const texts = [model, provider, provider && model ? `${provider}/${model}` : '', provider && model ? `${provider} ${model}` : ''].filter(Boolean);
  const squashed = [...new Set(texts.map(squash).filter(Boolean))];
  // 2. A spelling of one route: its id, "provider/id", its name or "connection name", in any letters.
  const exact = routes.filter(r => { const keys = exactKeys(r); return squashed.some(s => keys.has(s)); });
  // A name that fits several routes of one model id on different connections is not decided by the id alone: prefer the one the other half names.
  if (exact.length > 1 && provider && model) {
    const both = exact.filter(r => { const c = squash(r.connection); return squash(provider) === c || squash(provider) === squash(r.provider); });
    if (both.length === 1) return { status: 'exact', route: both[0], how: 'spelling' };
  }
  if (exact.length === 1) return { status: 'exact', route: exact[0], how: 'spelling' };
  if (exact.length > 1) return { status: 'ambiguous', options: exact };
  // 3. The local model by its everyday words.
  const locals = routes.filter(r => r.local);
  if (squashed.some(s => LOCAL_WORDS.has(s)) && locals.length) {
    return locals.length === 1 ? { status: 'exact', route: locals[0], how: 'local' } : { status: 'ambiguous', options: locals };
  }
  // 4. A close guess: part of an id or a name, a small typo.
  const scored = [];
  for (const route of routes) {
    const forms = looseForms(route);
    let best = Infinity;
    for (const s of squashed) {
      for (const form of forms) {
        if (s.length >= 3 && form === s) best = Math.min(best, 0);
        else if (s.length >= 4 && (form.includes(s) || s.includes(form) && form.length >= 4)) best = Math.min(best, 1);
        else if (s.length >= 5) best = Math.min(best, distance(s, form, 2) <= 2 ? 2 : Infinity);
      }
      // every word the caller wrote appears in the id or the name ("flash deepseek")
      const wanted = words(texts.join(' '));
      const haystack = squash(`${route.name} ${route.model} ${route.connection}`);
      if (wanted.length && wanted.every(w => w.length >= 2 && haystack.includes(squash(w)))) best = Math.min(best, 1.5);
    }
    if (best < Infinity) scored.push({ route, score: best });
  }
  scored.sort((a, b) => a.score - b.score || a.route.name.length - b.route.name.length);
  if (!scored.length) return { status: 'none' };
  const top = scored[0];
  const close = scored.filter(s => s.score === top.score);
  if (close.length === 1) return { status: 'confirm', route: top.route, others: scored.slice(1, 4).map(s => s.route) };
  return { status: 'ambiguous', options: scored.slice(0, 6).map(s => s.route) };
}

/** What the person or assistant typed, for a message ("deepseek", "openrouter/deepseek"). */
export const asked = request => [request.provider, request.model].filter(v => typeof v === 'string' && v.trim()).join(' / ');

/**
 * The text the `subagent` tool answers with when the model could not be settled. `null` when the call can go ahead (the caller then
 * uses the `exact` route or inherits). Plain words, with the next step; never a route key as the only thing said.
 */
export function modelProblem(routes, request) {
  const result = resolveModel(routes, request);
  const what = asked(request);
  if (result.status === 'inherit' || result.status === 'exact') return null;
  if (!routes.length) return 'No other model is set up. Nothing was started. Add a model in Models, or start the helper without naming a model.';
  if (result.status === 'confirm') {
    const r = result.route;
    return `No model is called "${what}". Nothing was started. Did you mean ${r.name} (${r.local ? 'running on this computer' : r.connection})? Ask the person "Did you mean ${r.name}?" and, when they agree, call subagent again with provider "${r.provider}" and model "${r.model}".`;
  }
  if (result.status === 'ambiguous') {
    return `More than one model fits "${what}". Nothing was started. Ask the person which one they mean, then call subagent again with that model's provider and model:\n${routeList(result.options, { max: 8 })}`;
  }
  return `No model is called "${what}". Nothing was started. These are the models that can be used:\n${routeList(routes)}\nChoose one of them, or tell the person it is not set up (they can add it in Models).`;
}
