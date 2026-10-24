/** The few React hooks the pure-logic modules import; the tests never render, so they do nothing. */
export const useSyncExternalStore = (subscribe, get) => get();
export const useEffect = () => {};
export const useState = value => [value, () => {}];
export const useRef = value => ({ current: value });
