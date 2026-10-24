/**
 * Helper agents on a model the person names: the Harness `subagent` tool with a forgiving model choice.
 *
 * The pinned Harness's `subagent` tool takes an exact `provider` route key and model id, and its discovery tool lists them as
 * `api-1b2c3d4e-5f/deepseek/deepseek-v3.2`. Nobody (and no small local model) can guess that from "ask DeepSeek". This plugin keeps the
 * Harness's tool and its checks and changes three things at the seam where the tool is called:
 *
 *   - the choice: the model id, `provider/id`, the name in the model picker, or "the local model" (any letters) is turned into the route
 *     the Harness needs; a close guess is not started but answered with "Did you mean ...?" for the person to confirm; nothing that fits
 *     is answered with the plain list of models that can be used (server/agent/model-routes.mjs);
 *   - the words the assistant reads: the tool says when to use it ("ask a subagent", "ask model X") and that the model must never be looked for
 *     with shell commands or in the environment;
 *   - the discovery tool lists the models by the names the person knows.
 *
 * It rewrites no Harness file: the two tool definitions the Harness registers for each chat are wrapped in place (their `execute` and their
 * text), once each. A Harness update that renames them leaves the tools as they were and the test that runs the real Harness fails.
 *
 * Plain ESM, loaded by absolute path like strata-gate.mjs. `config.routes` is the table the profile wrote (model-routes.mjs documents a route).
 */
import { modelProblem, resolveModel, routeList } from '../model-routes.mjs';

export const name = 'strata-models';
export const inject = ['tools'];

/** What the assistant is told in the tool's own text, so every model (a small local one and a hosted one alike) knows when and how. */
export const SUBAGENT_GUIDANCE = ' Use this tool whenever the person asks you to ask a subagent, a helper or another model ("ask a subagent", "ask DeepSeek", "use the local model", "ask model X"): start it with this tool and do not look for the model yourself with shell commands, in the environment or on this computer. To use a particular model, give `model` as its name or id (for example "DeepSeek V3.2" or "deepseek/deepseek-v3.2", or "local" for the model running on this computer); `provider` is optional. `list_subagent_models` shows what can be used. After starting a helper, do not wait with sleep or commands: end your turn and its answer arrives as a new message.';
const MODEL_WORDS = 'The model to use for the helper: its name as the person knows it ("DeepSeek V3.2"), its id ("deepseek/deepseek-v3.2") or "local" for the model running on this computer. Omit it to use this chat\'s model.';
const PROVIDER_WORDS = 'Optional: the connection the model belongs to ("OpenRouter"). Usually the model alone is enough.';

const wrapped = new WeakSet();

/** The two wrappers' behaviour, apart from the Harness objects (the tests call these directly). */
export function subagentArgs(routes, args) {
  const result = resolveModel(routes, args ?? {});
  if (result.status !== 'exact') return { ok: result.status === 'inherit', args };
  const { provider, model } = result.route;
  return { ok: true, args: { ...args, provider, model } };
}

export function apply(ctx, config = {}) {
  const routes = Array.isArray(config.routes) ? config.routes : [];

  const wrapSubagent = def => {
    if (wrapped.has(def)) return;
    wrapped.add(def);
    const original = def.execute;
    def.description = `${def.description}${SUBAGENT_GUIDANCE}`;
    const props = def.parameters?.properties;
    if (props?.model) props.model = { ...props.model, description: MODEL_WORDS };
    if (props?.provider) props.provider = { ...props.provider, description: PROVIDER_WORDS };
    def.execute = function execute(args, exec) {
      const request = args && typeof args === 'object' ? args : {};
      const problem = modelProblem(routes, request);
      if (problem) throw new Error(problem);
      const settled = subagentArgs(routes, request);
      return original.call(this, settled.args, exec);
    };
  };

  const wrapList = def => {
    if (wrapped.has(def)) return;
    wrapped.add(def);
    def.description = 'List the models a helper agent can be started on, by the names the person knows. Call it with no arguments, then give `model` (a name or id from the list, or "local") to the subagent tool.';
    def.execute = function execute() {
      return routes.length
        ? `Models a helper can use:\n${routeList(routes)}\nStart one with the subagent tool: model = its name or id.`
        : 'No other model is set up. A helper agent uses this chat\'s own model.';
    };
  };

  const wrapAll = () => {
    let agents;
    try { agents = ctx.get('agents')?.list?.() ?? []; } catch { return; }
    for (const agent of agents) {
      try {
        const sub = ctx.tools.get('subagent', agent);
        if (sub) wrapSubagent(sub);
        const list = ctx.tools.get('list_subagent_models', agent);
        if (list) wrapList(list);
      } catch { /* a tool that is not there yet is wrapped at the next change */ }
    }
  };
  ctx.on('tools/change', wrapAll);
  wrapAll();
}
