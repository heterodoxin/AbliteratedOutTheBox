/**
 * A scheduled task as the agent: a Harness session of its own, under the mode the task was given, tools included.
 *
 * Nobody is at the screen when it runs, so a request for approval is answered "no" at once and reported in the result: nothing is
 * changed without a person saying yes. (Plan changes nothing; Manual therefore stops at the first change and says what it was; Bypass
 * is full access and asks nothing.) Every run starts a fresh session, so yesterday's run does not fill today's context.
 */
import { taskReport } from '../task-report.mjs';

/**
 * @param {{agent: object, choice: () => ({instanceId:string}|{connectionId:string, modelId:string}|null), workspace: string}} deps `choice` names the model the task runs on
 * @returns {(task: object, signal?: AbortSignal) => Promise<{text:string, attention:string|null, demo:boolean}|null>} null = no local model is ready (the caller falls back)
 */
export function createAgentTaskRunner({ agent, choice, workspace }) {
  return async function runAgentTask(task, signal) {
    const model = choice();
    if (!model) return null;
    const chatId = `task:${task.id}`;
    const cards = new Map();
    let text = '', failure = null;
    const files = new Map();
    const stop = () => void agent.stop(chatId).catch(() => {});
    signal?.addEventListener('abort', stop, { once: true });
    try {
      await agent.runTurn({
        chatId, message: task.prompt, workspacePath: workspace, choice: model, mode: task.mode ?? 'plan', history: [],
        onFrame: frame => {
          if (frame.type === 'delta') text += frame.text ?? '';
          else if (frame.type === 'error') failure = frame.message;
          else if (frame.type === 'deliverable' && frame.abs && files.size < 30) files.set(frame.abs, { name: frame.name ?? frame.path, path: frame.abs });
          else if (frame.type === 'agent-tool') {
            cards.set(frame.tool.id, { ...(cards.get(frame.tool.id) ?? {}), ...frame.tool });
            if (frame.tool.state === 'proposed') void agent.approve({ chatId, callId: frame.tool.id, decision: 'deny' }).catch(() => {});
          }
        },
      });
    } finally { signal?.removeEventListener('abort', stop); agent.forget(chatId); }
    signal?.throwIfAborted();
    if (failure) throw new Error(failure);
    return { ...taskReport({ text, cards: [...cards.values()] }), files: [...files.values()] };
  };
}
