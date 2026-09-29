// events.js - tiny synchronous EventEmitter (on / off / once / emit).
// No DOM, no three.js. Listener errors are caught and reported via console.warn so that one
// broken listener cannot break the game loop.

export class EventEmitter {
  constructor() {
    this._listeners = new Map();
  }

  /** Subscribe. Returns an unsubscribe function. */
  on(type, fn) {
    if (typeof fn !== 'function') throw new TypeError('listener must be a function');
    let list = this._listeners.get(type);
    if (!list) {
      list = [];
      this._listeners.set(type, list);
    }
    list.push(fn);
    return () => this.off(type, fn);
  }

  /** Subscribe for a single emission. Returns an unsubscribe function. */
  once(type, fn) {
    const wrapper = (...args) => {
      this.off(type, wrapper);
      fn(...args);
    };
    wrapper._original = fn;
    return this.on(type, wrapper);
  }

  /** Unsubscribe one listener, or all listeners of a type when `fn` is omitted. */
  off(type, fn) {
    const list = this._listeners.get(type);
    if (!list) return;
    if (!fn) {
      this._listeners.delete(type);
      return;
    }
    const i = list.findIndex((l) => l === fn || l._original === fn);
    if (i >= 0) list.splice(i, 1);
    if (list.length === 0) this._listeners.delete(type);
  }

  /**
   * Emit synchronously with up to two arguments (fixed arity: no rest-array allocation, this
   * runs in the game loop). Listeners may unsubscribe themselves during emit. Returns the
   * number of listeners called.
   */
  emit(type, a, b) {
    const list = this._listeners.get(type);
    if (!list || list.length === 0) return 0;
    let called = 0;
    for (let i = 0; i < list.length; i++) {
      const fn = list[i];
      try {
        fn(a, b);
      } catch (err) {
        if (typeof console !== 'undefined' && console.warn) {
          console.warn(`[events] listener for '${String(type)}' threw:`, err);
        }
      }
      called++;
      // the listener removed itself (or an earlier one): do not skip the element that moved up
      if (list[i] !== fn) i--;
      if (list.length === 0) break;
    }
    return called;
  }

  listenerCount(type) {
    const list = this._listeners.get(type);
    return list ? list.length : 0;
  }

  removeAllListeners() {
    this._listeners.clear();
  }
}
