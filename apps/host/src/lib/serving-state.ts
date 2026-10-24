/** A slower mutation response must not replace a newer request-completion event. */
export function newestServingState<T extends { revision?: number }>(current: T | null, incoming: T): T {
  return current && current.revision !== undefined && incoming.revision !== undefined && current.revision > incoming.revision
    ? current
    : incoming;
}
