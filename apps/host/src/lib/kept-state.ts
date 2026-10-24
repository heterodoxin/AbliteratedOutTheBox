import { useCallback, useEffect, useState, type Dispatch, type SetStateAction } from 'react';

/**
 * A piece of screen state that survives leaving the screen. The Models page keeps its choices (version, context, computer use, picture
 * reading ...) in useState, so going to Monitor and back reset them and a person lost track of what they were downloading. Kept values
 * live in memory for the whole session (not on disk: a new start begins from the installed model again).
 */
const kept = new Map<string, unknown>();

export function useKept<T>(key: string, initial: T): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => (kept.has(key) ? kept.get(key) as T : initial));
  useEffect(() => { kept.set(key, value); }, [key, value]);
  const set = useCallback<Dispatch<SetStateAction<T>>>(next => setValue(old => {
    const result = typeof next === 'function' ? (next as (o: T) => T)(old) : next;
    kept.set(key, result);
    return result;
  }), [key]);
  return [value, set];
}

export const keptValue = <T,>(key: string): T | undefined => kept.get(key) as T | undefined;
/** Forget everything kept (tests, and a changed installation). */
export const forgetKept = () => kept.clear();
