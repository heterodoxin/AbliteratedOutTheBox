/** One active inference request; queued work is cancellable and never survives host exit. */
export class RequestQueue {
  #pending = [];
  #active = false;
  constructor(onChange = () => {}) { this.onChange = onChange; }
  get waiting() { return this.#pending.length; }
  run(signal, task) {
    return new Promise((resolve, reject) => {
      const entry = { signal, task, resolve, reject, abort: null };
      entry.abort = () => {
        const index = this.#pending.indexOf(entry);
        if (index < 0) return;
        this.#pending.splice(index, 1);
        reject(signal.reason ?? new DOMException('Request cancelled', 'AbortError'));
        this.onChange(this.waiting);
      };
      if (signal.aborted) { reject(signal.reason); return; }
      signal.addEventListener('abort', entry.abort, { once: true });
      this.#pending.push(entry);
      this.#drain();
    });
  }
  async #drain() {
    if (this.#active) { this.onChange(this.waiting); return; }
    const entry = this.#pending.shift();
    if (!entry) { this.onChange(0); return; }
    this.#active = true;
    entry.signal.removeEventListener('abort', entry.abort);
    this.onChange(this.waiting);
    try { entry.signal.throwIfAborted(); entry.resolve(await entry.task()); }
    catch (error) { entry.reject(error); }
    finally { this.#active = false; this.#drain(); }
  }
}
