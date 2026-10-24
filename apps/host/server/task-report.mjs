/**
 * What a task that ran as the agent shows in Tasks ("Last result"): the agent's own answer, then a short plain list of what it
 * did, then - if it hit something that needed a person's yes - exactly what that was. Pure, so it is tested without an agent.
 */
const clip = (value, max) => {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
};

/** One card as a line a person can read. */
export function cardLine(card) {
  const what = clip(card.effect || card.preview || card.name, 140);
  const detail = card.preview && card.effect && card.preview !== card.effect ? ` (${clip(card.preview, 100)})` : '';
  if (card.state === 'denied') return `- Could not do this by itself: ${what}${detail}. Nothing was changed.`;
  if (card.state === 'error') return `- Tried: ${what}${detail}. It did not work.`;
  return `- ${what}${detail}`;
}

/**
 * @param {{text:string, cards:Array}} run
 * @returns {{text:string, attention:string|null, demo:boolean}}
 */
export function taskReport({ text, cards }) {
  const done = cards.filter(c => c.state !== 'proposed');
  const refused = cards.filter(c => c.state === 'denied');
  const lines = done.map(cardLine);
  const answer = String(text ?? '').trim();
  let out = answer || (cards.length ? 'The task ran.' : 'The task ran and had nothing to report.');
  if (lines.length) out += `\n\n---\n\n**What it did**\n\n${lines.join('\n')}`;
  const attention = refused.length
    ? `This task wanted to change something and needed a yes, which nobody could give while it ran by itself: ${refused.map(c => clip(c.effect || c.name, 100)).join('; ')}. Nothing was changed. Edit the task and let it make changes if you want it to.`
    : null;
  if (attention) out += `\n\n> ${attention}`;
  return { text: out, attention, demo: false };
}

/**
 * One plain line for a board card: the first sentence of the answer, with the markdown taken off and the "What it did" list and the
 * "needed a yes" note left out. '' when there is nothing to say.
 */
export function lastLine(text, max = 160) {
  let body = String(text ?? '').split(/\n\n---\n\n/)[0].replace(/\n\n> [\s\S]*$/, '');
  body = body.replace(/```[\s\S]*?```/g, ' ').replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/^\s*(#{1,6}|[-*+]|\d+[.)]|>)\s+/gm, '').replace(/[*_`~]/g, '');
  body = body.replace(/\s+/g, ' ').trim();
  if (!body) return '';
  const sentence = /^.*?[.!?](?=\s|$)/.exec(body)?.[0] ?? body;
  // A very short first sentence ("Done.") says nothing: take the next one too.
  const two = /^.*?[.!?](?=\s|$)\s*.*?[.!?](?=\s|$)/.exec(body)?.[0] ?? body;
  const pick = sentence.length < 24 && body.length > sentence.length ? two : sentence;
  return clip(pick, max);
}
