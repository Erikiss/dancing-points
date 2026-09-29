// audio.js - AudioEngine: WebAudio drum-pattern synth ('hiphop', 'house', 'metronome'), count-in
// clicks, optional operator-supplied track (decodeAudioData) with offset, hit-grade SFX and a
// master gain. Events are scheduled on the BeatClock's audio timeline with a 100 ms lookahead
// (`update()` once per frame). Every method is a no-op when there is no AudioContext, so the
// engine can be constructed and driven headless (Node / tests). No three.js, no DOM.

import { FLOW } from '../config.js';

export const PATTERNS = Object.freeze(['hiphop', 'house', 'metronome']);

// instrument bits
const KICK = 1, SNARE = 2, HAT = 4, OPENHAT = 8, CLICK = 16, ACCENT = 32, CLAP = 64;

/** Instrument bitmask for a 16th-note step (0..15) within a bar, per pattern. */
export function patternStep(name, step) {
  const s = ((step % 16) + 16) % 16;
  switch (name) {
    case 'house': {
      let m = 0;
      if (s % 4 === 0) m |= KICK;                 // four on the floor
      if (s % 4 === 2) m |= OPENHAT;              // off-beat open hat
      else if (s % 2 === 0) m |= HAT;
      if (s === 4 || s === 12) m |= CLAP;         // clap on 2 and 4
      return m;
    }
    case 'metronome': {
      if (s === 0) return CLICK | ACCENT;
      if (s % 4 === 0) return CLICK;
      return 0;
    }
    case 'hiphop':
    default: {
      let m = 0;
      if (s === 0 || s === 8 || s === 10) m |= KICK;  // 1, 3 and the "and" of 3 (boom-bap feel)
      if (s === 4 || s === 12) m |= SNARE;             // 2 and 4
      if (s % 2 === 0) m |= HAT;                       // 8ths
      if (s === 7 || s === 15) m |= HAT;               // swing-ish ghost hats
      return m;
    }
  }
}

export class AudioEngine {
  /**
   * @param {object} [opts]
   * @param {AudioContext} [opts.audioContext]  null -> everything is a no-op
   * @param {function} [opts.fetchImpl]         for loadTrack (default globalThis.fetch)
   * @param {number} [opts.lookahead]           seconds (default FLOW.audioLookaheadSec)
   */
  constructor(opts = {}) {
    this.ctx = opts.audioContext || null;
    this.fetchImpl = opts.fetchImpl || globalThis.fetch;
    this.lookahead = opts.lookahead > 0 ? opts.lookahead : FLOW.audioLookaheadSec;
    this.master = null;
    this.synthGain = null;
    this.sfxGain = null;
    this.trackGain = null;
    this._noise = null;
    this.track = null;           // { buffer, url, offsetSec, gain }
    this._trackSource = null;
    this.clock = null;
    this.pattern = 'hiphop';
    this.countInBeats = 4;
    this.beatsPerBar = 4;
    this.durationBeats = 0;
    this._nextStep = 0;          // next 16th-note step to schedule (relative to beat 0, may be negative)
    this._endStep = 0;
    this.playing = false;
    this.scheduledEvents = 0;
    if (this.ctx) this._buildGraph();
  }

  get available() {
    return this.ctx !== null;
  }

  _buildGraph() {
    const ctx = this.ctx;
    this.master = ctx.createGain();
    this.master.gain.value = 0.8;
    this.master.connect(ctx.destination);
    this.synthGain = ctx.createGain();
    this.synthGain.gain.value = 0.7;
    this.synthGain.connect(this.master);
    this.sfxGain = ctx.createGain();
    this.sfxGain.gain.value = 0.5;
    this.sfxGain.connect(this.master);
    this.trackGain = ctx.createGain();
    this.trackGain.gain.value = 0.8;
    this.trackGain.connect(this.master);
    // 1 s of white noise for snare / hats
    const len = Math.max(1, Math.floor(ctx.sampleRate));
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0);
    let seed = 12345;
    for (let i = 0; i < len; i++) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      d[i] = (seed / 4294967296) * 2 - 1;
    }
    this._noise = buf;
  }

  /** Resume a suspended context (must be called from a user gesture in browsers). */
  async resume() {
    if (!this.ctx) return false;
    try {
      if (this.ctx.state === 'suspended' && typeof this.ctx.resume === 'function') await this.ctx.resume();
      return this.ctx.state === 'running';
    } catch (e) {
      return false;
    }
  }

  setMasterGain(g) {
    if (!this.master) return;
    this.master.gain.value = Math.max(0, Math.min(1.5, g));
  }

  getMasterGain() {
    return this.master ? this.master.gain.value : 0;
  }

  /** Select the synth pattern ('hiphop' | 'house' | 'metronome' | null = silent). */
  setPattern(name) {
    this.pattern = name && PATTERNS.includes(name) ? name : (name === null ? null : 'hiphop');
  }

  /**
   * Load an operator-supplied track. Resolves true on success, false otherwise (the synth
   * pattern keeps playing). Never throws.
   */
  async loadTrack(url, opts = {}) {
    this.track = null;
    if (!this.ctx || !url) return false;
    try {
      if (typeof this.fetchImpl !== 'function') return false;
      const res = await this.fetchImpl(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.arrayBuffer();
      const buffer = await new Promise((resolve, reject) => {
        const p = this.ctx.decodeAudioData(data, resolve, reject);
        if (p && typeof p.then === 'function') p.then(resolve, reject);
      });
      this.track = {
        buffer,
        url,
        offsetSec: typeof opts.offsetSec === 'number' ? opts.offsetSec : 0,
        gain: typeof opts.gain === 'number' ? opts.gain : 0.8,
      };
      return true;
    } catch (e) {
      if (typeof console !== 'undefined') console.warn('[audio] track load failed', url, e);
      this.track = null;
      return false;
    }
  }

  /**
   * Bind to a started BeatClock and begin scheduling from the count-in.
   * @param {import('./clock.js').BeatClock} clock
   * @param {object} [opts] { synth, countInBeats, beatsPerBar, durationBeats, useTrack = true }
   */
  start(clock, opts = {}) {
    this.stop();
    this.clock = clock;
    if (opts.synth !== undefined) this.setPattern(opts.synth);
    this.countInBeats = opts.countInBeats !== undefined ? opts.countInBeats : clock.countInBeats;
    this.beatsPerBar = opts.beatsPerBar > 0 ? opts.beatsPerBar : clock.beatsPerBar;
    this.durationBeats = opts.durationBeats > 0 ? opts.durationBeats : 0;
    this._nextStep = -this.countInBeats * 4;
    this._endStep = this.durationBeats > 0 ? Math.ceil(this.durationBeats * 4) : Infinity;
    this.playing = true;
    this.scheduledEvents = 0;
    if (!this.ctx) return;
    if (this.track && opts.useTrack !== false) this._startTrack();
    this.update();
  }

  _startTrack() {
    const ctx = this.ctx, tr = this.track;
    if (!ctx || !tr || !this.clock) return;
    try {
      const src = ctx.createBufferSource();
      src.buffer = tr.buffer;
      this.trackGain.gain.value = tr.gain;
      src.connect(this.trackGain);
      const beat0 = this.clock.beatToSource(0);
      const now = ctx.currentTime;
      let offset = tr.offsetSec;
      let when = beat0;
      if (when < now) { offset += (now - when); when = now; }
      if (offset < 0) { when += -offset; offset = 0; }
      src.start(when, offset);
      this._trackSource = src;
    } catch (e) {
      if (typeof console !== 'undefined') console.warn('[audio] track start failed', e);
    }
  }

  /**
   * Pause/resume support for a loaded track (the synth pattern follows the clock by itself:
   * `update()` schedules nothing while `clock.paused`). Call `pauseTrack()` after
   * `clock.pause()` and `resumeTrack()` after `clock.resume()`; the track restarts at the
   * clock's current position.
   */
  pauseTrack() {
    if (!this._trackSource) return false;
    try { this._trackSource.stop(); } catch (e) { /* already stopped */ }
    try { this._trackSource.disconnect(); } catch (e) { /* ignore */ }
    this._trackSource = null;
    return true;
  }

  resumeTrack() {
    if (!this.playing || !this.track || !this.clock || this._trackSource) return false;
    this._startTrack();
    return !!this._trackSource;
  }

  /** Schedule all events inside the lookahead window. Call once per frame. */
  update() {
    if (!this.ctx || !this.playing || !this.clock || !this.clock.running || this.clock.paused) return;
    const ctx = this.ctx;
    const horizon = ctx.currentTime + this.lookahead;
    let guard = 0;
    while (this._nextStep < this._endStep && guard++ < 64) {
      const beat = this._nextStep / 4;
      const time = this.clock.beatToSource(beat);
      if (time > horizon) break;
      if (time >= ctx.currentTime - 0.02) this._scheduleStep(this._nextStep, time);
      this._nextStep++;
    }
    if (this._nextStep >= this._endStep && this.playing) {
      // pattern finished; keep `playing` so a late SFX still works, nothing more to schedule
    }
  }

  _scheduleStep(step, time) {
    const beat = step / 4;
    if (beat < 0) {
      // count-in: click on every beat, accent on the first
      if (step % 4 === 0) {
        this._click(time, step === -this.countInBeats * 4);
        this.scheduledEvents++;
      }
      return;
    }
    if (!this.pattern || this.track) {
      // custom track: no synth pattern (metronome still audible for 'metronome')
      if (this.pattern !== 'metronome') return;
    }
    const inBar = ((step % (this.beatsPerBar * 4)) + this.beatsPerBar * 4) % (this.beatsPerBar * 4);
    const m = patternStep(this.pattern, inBar % 16);
    if (m & KICK) this._kick(time);
    if (m & SNARE) this._snare(time);
    if (m & CLAP) this._clap(time);
    if (m & HAT) this._hat(time, false);
    if (m & OPENHAT) this._hat(time, true);
    if (m & CLICK) this._click(time, (m & ACCENT) !== 0);
    if (m) this.scheduledEvents++;
  }

  // ---- voices --------------------------------------------------------------------------------

  _kick(t) {
    const ctx = this.ctx;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(160, t);
    osc.frequency.exponentialRampToValueAtTime(45, t + 0.12);
    g.gain.setValueAtTime(1.0, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.28);
    osc.connect(g); g.connect(this.synthGain);
    osc.start(t); osc.stop(t + 0.3);
  }

  _snare(t) {
    const ctx = this.ctx;
    const n = ctx.createBufferSource();
    n.buffer = this._noise;
    const f = ctx.createBiquadFilter();
    f.type = 'bandpass'; f.frequency.value = 1800; f.Q.value = 0.8;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.7, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.18);
    n.connect(f); f.connect(g); g.connect(this.synthGain);
    n.start(t); n.stop(t + 0.2);
    const osc = ctx.createOscillator();
    const g2 = ctx.createGain();
    osc.type = 'triangle'; osc.frequency.setValueAtTime(190, t);
    g2.gain.setValueAtTime(0.5, t);
    g2.gain.exponentialRampToValueAtTime(0.001, t + 0.1);
    osc.connect(g2); g2.connect(this.synthGain);
    osc.start(t); osc.stop(t + 0.12);
  }

  _clap(t) {
    const ctx = this.ctx;
    for (let i = 0; i < 3; i++) {
      const n = ctx.createBufferSource();
      n.buffer = this._noise;
      const f = ctx.createBiquadFilter();
      f.type = 'bandpass'; f.frequency.value = 1200; f.Q.value = 1.2;
      const g = ctx.createGain();
      const t0 = t + i * 0.012;
      g.gain.setValueAtTime(0.5, t0);
      g.gain.exponentialRampToValueAtTime(0.001, t0 + (i === 2 ? 0.15 : 0.03));
      n.connect(f); f.connect(g); g.connect(this.synthGain);
      n.start(t0); n.stop(t0 + 0.16);
    }
  }

  _hat(t, open) {
    const ctx = this.ctx;
    const n = ctx.createBufferSource();
    n.buffer = this._noise;
    const f = ctx.createBiquadFilter();
    f.type = 'highpass'; f.frequency.value = 7000;
    const g = ctx.createGain();
    const dur = open ? 0.18 : 0.045;
    g.gain.setValueAtTime(open ? 0.28 : 0.22, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    n.connect(f); f.connect(g); g.connect(this.synthGain);
    n.start(t); n.stop(t + dur + 0.01);
  }

  _click(t, accent) {
    const ctx = this.ctx;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = 'square';
    osc.frequency.setValueAtTime(accent ? 1600 : 1100, t);
    g.gain.setValueAtTime(accent ? 0.5 : 0.35, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.03);
    osc.connect(g); g.connect(this.synthGain);
    osc.start(t); osc.stop(t + 0.035);
  }

  /** Immediate count-in / UI click (outside the scheduled timeline). */
  playClick(accent = false) {
    if (!this.ctx) return;
    this._click(this.ctx.currentTime, accent);
  }

  /** Feedback sound for a beat grade: 'perfekt' | 'gut' | 'ok' | 'daneben'. */
  playSfx(grade) {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const t = ctx.currentTime;
    const tone = (freq, at, dur, gain, type = 'sine') => {
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, at);
      g.gain.setValueAtTime(0.0001, at);
      g.gain.exponentialRampToValueAtTime(gain, at + 0.008);
      g.gain.exponentialRampToValueAtTime(0.0001, at + dur);
      osc.connect(g); g.connect(this.sfxGain);
      osc.start(at); osc.stop(at + dur + 0.01);
    };
    switch (grade) {
      case 'perfekt':
        tone(1046.5, t, 0.12, 0.5);
        tone(1318.5, t + 0.07, 0.14, 0.5);
        tone(1568.0, t + 0.14, 0.2, 0.45);
        break;
      case 'gut':
        tone(880, t, 0.12, 0.45);
        tone(1108.7, t + 0.08, 0.16, 0.4);
        break;
      case 'ok':
        tone(660, t, 0.14, 0.35, 'triangle');
        break;
      case 'daneben':
      default: {
        const n = ctx.createBufferSource();
        n.buffer = this._noise;
        const f = ctx.createBiquadFilter();
        f.type = 'lowpass'; f.frequency.value = 400;
        const g = ctx.createGain();
        g.gain.setValueAtTime(0.4, t);
        g.gain.exponentialRampToValueAtTime(0.001, t + 0.12);
        n.connect(f); f.connect(g); g.connect(this.sfxGain);
        n.start(t); n.stop(t + 0.14);
        break;
      }
    }
  }

  /** Stop the track and scheduling (already scheduled synth notes ring out, < 0.3 s). */
  stop() {
    this.playing = false;
    this.clock = null;
    if (this._trackSource) {
      try { this._trackSource.stop(); } catch (e) { /* already stopped */ }
      try { this._trackSource.disconnect(); } catch (e) { /* ignore */ }
      this._trackSource = null;
    }
  }

  dispose() {
    this.stop();
    if (this.master) {
      try { this.master.disconnect(); } catch (e) { /* ignore */ }
    }
  }
}
