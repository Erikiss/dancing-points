// clock.js - BeatClock: the master timeline of a play/record session. Time `t` is seconds since
// beat 0 (negative during the count-in). Uses AudioContext.currentTime when available (so audio
// scheduling and scoring share one clock), otherwise performance.now(); an injectable
// `timeSource` and a `speed` factor make it fully deterministic for tests. No allocations in
// `now()` (returns a reused object). No DOM / three.js.

const defaultTimeSource = () => {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now() / 1000;
  }
  return Date.now() / 1000;
};

export class BeatClock {
  /**
   * @param {object} [opts]
   * @param {AudioContext} [opts.audioContext] use its currentTime as time source
   * @param {function(): number} [opts.timeSource] seconds; overrides audioContext (tests)
   * @param {number} [opts.speed=1] time speed factor
   */
  constructor(opts = {}) {
    this.audioContext = opts.audioContext || null;
    this._timeSource = opts.timeSource || null;
    this.speed = opts.speed > 0 ? opts.speed : 1;
    this.bpm = 120;
    this.beatsPerBar = 4;
    this.countInBeats = 4;
    this.beatDuration = 0.5;
    this.running = false;
    this.paused = false;
    this._startSource = 0;     // source time at which t == -countIn
    this._pauseSource = 0;
    this._pausedAccum = 0;     // total paused source time
    this._state = { t: 0, beat: 0, bar: 0, sourceTime: 0, countIn: 0, running: false, paused: false };
  }

  /** Current source time in seconds (audio clock, injected source or performance.now). */
  sourceNow() {
    if (this._timeSource) return this._timeSource();
    if (this.audioContext && typeof this.audioContext.currentTime === 'number') return this.audioContext.currentTime;
    return defaultTimeSource();
  }

  /**
   * Start the timeline: t == -countInBeats * 60 / bpm right now; beat 0 happens after the
   * count-in. @param {object} [opts] { audioContext, timeSource, speed, beatsPerBar, startAt }
   * `startAt` is a source time at which the count-in starts (default: now).
   */
  start(bpm, countInBeats = 4, opts = {}) {
    if (!(bpm > 0)) throw new RangeError('bpm must be > 0');
    if (opts.audioContext) this.audioContext = opts.audioContext;
    if (opts.timeSource) this._timeSource = opts.timeSource;
    if (opts.speed > 0) this.speed = opts.speed;
    if (opts.beatsPerBar > 0) this.beatsPerBar = opts.beatsPerBar;
    this.bpm = bpm;
    this.countInBeats = Math.max(0, countInBeats | 0);
    this.beatDuration = 60 / bpm;
    this._startSource = typeof opts.startAt === 'number' ? opts.startAt : this.sourceNow();
    this._pausedAccum = 0;
    this.running = true;
    this.paused = false;
    return this;
  }

  get countInSeconds() {
    return this.countInBeats * this.beatDuration;
  }

  /** Source time that corresponds to timeline time t (inverse of now()); valid while running. */
  timeToSource(t) {
    return this._startSource + this._pausedAccum + (t + this.countInSeconds) / this.speed;
  }

  /** Source time of beat b (b may be negative during count-in). */
  beatToSource(beat) {
    return this.timeToSource(this.beatToTime(beat));
  }

  sourceToTime(source) {
    return (source - this._startSource - this._pausedAccum) * this.speed - this.countInSeconds;
  }

  beatToTime(beat) {
    return beat * this.beatDuration;
  }

  timeToBeat(t) {
    return t / this.beatDuration;
  }

  /**
   * Current timeline state (reused object): t (s since beat 0), beat (float), bar (int, floor),
   * sourceTime, countIn (float beats remaining in count-in, 0 when past).
   */
  now() {
    const s = this._state;
    let source;
    if (!this.running) {
      s.t = -this.countInSeconds;
    } else {
      source = this.paused ? this._pauseSource : this.sourceNow();
      s.t = this.sourceToTime(source);
      s.sourceTime = source;
    }
    s.beat = s.t / this.beatDuration;
    s.bar = Math.floor(s.beat / this.beatsPerBar);
    s.countIn = s.beat < 0 ? -s.beat : 0;
    s.running = this.running;
    s.paused = this.paused;
    return s;
  }

  pause() {
    if (!this.running || this.paused) return;
    this._pauseSource = this.sourceNow();
    this.paused = true;
  }

  resume() {
    if (!this.running || !this.paused) return;
    this._pausedAccum += this.sourceNow() - this._pauseSource;
    this.paused = false;
  }

  stop() {
    this.running = false;
    this.paused = false;
  }

  /** Jump the timeline so that `t` is now (tests / seeking). */
  seek(t) {
    if (!this.running) return;
    const source = this.paused ? this._pauseSource : this.sourceNow();
    this._startSource = source - this._pausedAccum - (t + this.countInSeconds) / this.speed;
  }
}
