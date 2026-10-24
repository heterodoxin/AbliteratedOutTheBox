/** A chat sent while no model can answer: the engine's own wording ("Choose a loaded model ID.") said nothing to a person. */
export const plainNoModel = message => /not ready/i.test(message) ? 'Your model is still loading. You can chat in a moment.'
  : /choose a loaded model/i.test(message) ? 'No model is loaded. Load one in Models first.' : message;
