// session.js - PlaySession: ties BeatClock + Choreo + input backend + Calibration + Scorer
// (+ AudioEngine) together for one run. `update(now)` handles the count-in, converts the latest
// input sample to the stage frame, scores at <= 30 Hz, records the player's stage-frame 3-point
// run (for the ghost duo) and emits events: 'start', 'countdown', 'beat', 'move', 'grade',
// 'tick', 'finished' (Result), 'abort'. Per-frame state for renderers/HUD is exposed as
// properties (reused objects, no allocation in update). No three.js, no DOM.

import { EventEmitter } from '../util/events.js';
import { FLOW } from '../config.js';
import { createSample } from '../xr/input.js';
import { createChoreoSample } from './choreo.js';

export class PlaySession extends EventEmitter {
  /**
   * @param {object} deps
   * @param {import('./choreo.js').Choreo} deps.choreo
   * @param {import('./clock.js').BeatClock} deps.clock
   * @param {import('./scoring.js').Scorer} deps.scorer
   * @param {object} deps.input           XRInput or EmulatedInput (W-frame samples)
   * @param {import('../xr/calibration.js').Calibration} deps.calibration
   * @param {import('./audio.js').AudioEngine} [deps.audio]
   * @param {string} [deps.mode='solo']   'solo' | 'ghost' | 'online' | 'record'
   * @param {string} [deps.player='']
   * @param {number} [deps.scoreHz=30]
   * @param {boolean} [deps.recordRun=true]
   * @param {boolean} [deps.driveInput]   call input.update(now) in update(); default: input.emulated
   */
  constructor(deps) {
    super();
    if (!deps || !deps.choreo || !deps.clock || !deps.scorer || !deps.input || !deps.calibration) {
      throw new TypeError('PlaySession needs choreo, clock, scorer, input and calibration');
    }
    this.choreo = deps.choreo;
    this.clock = deps.clock;
    this.scorer = deps.scorer;
    this.input = deps.input;
    this.calibration = deps.calibration;
    this.audio = deps.audio || null;
    this.mode = deps.mode || 'solo';
    this.player = deps.player || '';
    this.scoreHz = deps.scoreHz > 0 ? deps.scoreHz : 30;
    this.recordRun = deps.recordRun !== false;
    this.driveInput = deps.driveInput !== undefined ? !!deps.driveInput : this.input.emulated === true;

    this.state = 'idle';          // 'idle' | 'countdown' | 'playing' | 'finished' | 'aborted'
    this.t = -Infinity;
    this.beat = 0;
    this.bar = 0;
    this.frameScore = 0;
    this.combo = 0;
    this.score = 0;               // running estimate 0..100
    this.moveIndex = -1;
    this.currentMove = null;
    this.lastBeatEvent = null;
    this.result = null;
    this.playerS = createSample();                  // latest player sample in S
    this.refSample = createChoreoSample();          // reference at t (unmirrored, for the teacher)
    this._tickEvent = { t: 0, frameScore: 0, delta: 0 };  // reused: listeners must not retain it
    this._tickInterval = 1 / this.scoreHz;
    this._nextTick = 0;
    this._lastBeatInt = NaN;
    this._countdownBeat = NaN;
    this.ticks = 0;
    this.frames = 0;
    this.startedAt = null;

    // run recording (stage frame, 30 Hz)
    this.run = null;
    if (this.recordRun) {
      const fps = FLOW.runRecordHz;
      const n = Math.round(this.choreo.duration * fps) + 1;
      this.run = {
        choreoId: this.choreo.id,
        fps,
        frameCount: n,
        head: new Float32Array(n * 7),
        left: new Float32Array(n * 3),
        right: new Float32Array(n * 3),
        filled: new Uint8Array(n),
        lastIndex: -1,
      };
    }

    this.scorer.onBeat = (ev) => this._onBeatGraded(ev);
  }

  /**
   * Start the count-in now. opts: { speed } (clock speed factor, tests), { startAt } (clock
   * source time at which the count-in begins, e.g. the synchronised online-duo start).
   */
  start(opts = {}) {
    const c = this.choreo;
    this.scorer.reset();
    this.result = null;
    this.state = 'countdown';
    this.t = -c.countInBeats * c.beatDuration;
    this._nextTick = 0;
    this._lastBeatInt = NaN;
    this._countdownBeat = NaN;
    this.ticks = 0;
    this.frames = 0;
    this.moveIndex = -1;
    this.currentMove = null;
    this.frameScore = 0;
    this.combo = 0;
    this.score = 0;
    if (this.run) { this.run.filled.fill(0); this.run.lastIndex = -1; }
    this.startedAt = new Date().toISOString();
    this.clock.start(c.bpm, c.countInBeats, { speed: opts.speed, beatsPerBar: c.beatsPerBar, startAt: opts.startAt });
    if (this.audio) {
      this.audio.start(this.clock, {
        synth: c.audio.synth,
        countInBeats: c.countInBeats,
        beatsPerBar: c.beatsPerBar,
        durationBeats: c.durationBeats,
      });
    }
    this.emit('start', { choreoId: c.id, mode: this.mode });
    if (c.countInBeats === 0) this.state = 'playing';
    return this;
  }

  /**
   * Advance the session. `now` = wall time in ms (performance.now(); passed to the emulated
   * input when `driveInput`). Returns the state string.
   */
  update(now) {
    if (this.state !== 'countdown' && this.state !== 'playing') return this.state;
    this.frames++;
    if (typeof now !== 'number') now = typeof performance !== 'undefined' ? performance.now() : Date.now();

    const clk = this.clock.now();
    const t = clk.t;
    this.t = t;
    this.beat = clk.beat;
    this.bar = clk.bar;
    if (this.audio) this.audio.update();

    // input -> stage frame
    if (this.driveInput && typeof this.input.update === 'function') this.input.update(now);
    const sW = this.input.sample;
    if (sW) this.calibration.toStage(sW, this.choreo.referenceHeight, this.playerS);

    // integer beat transitions (count-in and play)
    const beatInt = Math.floor(clk.beat);
    if (beatInt !== this._lastBeatInt) {
      this._lastBeatInt = beatInt;
      if (beatInt < 0) {
        this.emit('countdown', { beatsLeft: -beatInt, beat: beatInt });
      }
      if (beatInt >= 0 && beatInt < this.choreo.durationBeats) {
        this.emit('beat', { beat: beatInt, bar: clk.bar, beatInBar: ((beatInt % this.choreo.beatsPerBar) + this.choreo.beatsPerBar) % this.choreo.beatsPerBar });
      }
    }

    if (t < 0) {
      this.state = 'countdown';
      this.choreo.sampleAt(0, this.refSample);
      return this.state;
    }
    if (this.state === 'countdown') this.state = 'playing';

    // reference for the teacher avatar
    this.choreo.sampleAt(t, this.refSample);

    // move transitions
    const mi = this.choreo.moveIndexAt(t);
    if (mi !== this.moveIndex && mi >= 0) {
      const prev = this.currentMove;
      this.moveIndex = mi;
      this.currentMove = this.choreo.moves[mi];
      this.emit('move', { index: mi, move: this.currentMove, prev });
    }

    // scoring at <= scoreHz
    if (t >= this._nextTick) {
      const s = this.scorer.scoreTick(this.playerS, t);
      if (s >= 0) {
        this.frameScore = s;
        this.ticks++;
        const te = this._tickEvent;
        te.t = t; te.frameScore = s; te.delta = this.scorer.lastDelta;
        this.emit('tick', te);
      }
      this._recordRunFrame(t);
      this._nextTick += this._tickInterval;
      if (t - this._nextTick > this._tickInterval) this._nextTick = t + this._tickInterval; // fell behind: resync
    }

    if (t >= this.choreo.duration) this._finish();
    return this.state;
  }

  _recordRunFrame(t) {
    const run = this.run;
    if (!run) return;
    let idx = Math.round(t * run.fps);
    if (idx < 0) idx = 0;
    if (idx >= run.frameCount) idx = run.frameCount - 1;
    const s = this.playerS;
    const o7 = idx * 7, o3 = idx * 3;
    run.head[o7] = s.head.p[0]; run.head[o7 + 1] = s.head.p[1]; run.head[o7 + 2] = s.head.p[2];
    run.head[o7 + 3] = s.head.q[0]; run.head[o7 + 4] = s.head.q[1]; run.head[o7 + 5] = s.head.q[2]; run.head[o7 + 6] = s.head.q[3];
    run.left[o3] = s.left.p[0]; run.left[o3 + 1] = s.left.p[1]; run.left[o3 + 2] = s.left.p[2];
    run.right[o3] = s.right.p[0]; run.right[o3 + 1] = s.right.p[1]; run.right[o3 + 2] = s.right.p[2];
    run.filled[idx] = 1;
    if (idx > run.lastIndex) run.lastIndex = idx;
  }

  _onBeatGraded(ev) {
    this.lastBeatEvent = ev;
    this.combo = ev.combo;
    this.score = this.scorer.currentScore();
    if (this.audio) this.audio.playSfx(ev.grade);
    this.emit('grade', ev);
  }

  _finish() {
    if (this.state === 'finished') return;
    this.result = this.scorer.finalize({ mode: this.mode, player: this.player, playedAt: this.startedAt });
    this.score = this.result.score;
    this.combo = this.scorer.combo;
    this.state = 'finished';
    if (this.audio) this.audio.stop();
    this.clock.stop();
    this._fillRunGaps();
    this.emit('finished', this.result);
  }

  _fillRunGaps() {
    const run = this.run;
    if (!run || run.lastIndex < 0) return;
    let last = -1;
    for (let i = 0; i < run.frameCount; i++) {
      if (run.filled[i] === 1) { last = i; continue; }
      const src = last >= 0 ? last : firstFilled(run);
      if (src < 0) break;
      for (let k = 0; k < 7; k++) run.head[i * 7 + k] = run.head[src * 7 + k];
      for (let k = 0; k < 3; k++) {
        run.left[i * 3 + k] = run.left[src * 3 + k];
        run.right[i * 3 + k] = run.right[src * 3 + k];
      }
      run.filled[i] = 2;
    }
  }

  /** Abort the run (menu / quit). Emits 'abort'. */
  abort() {
    if (this.state === 'finished' || this.state === 'aborted' || this.state === 'idle') return;
    this.state = 'aborted';
    if (this.audio) this.audio.stop();
    this.clock.stop();
    this.emit('abort', { t: this.t });
  }

  pause() {
    this.clock.pause();
  }

  resume() {
    this.clock.resume();
  }

  get paused() {
    return this.clock.paused;
  }

  /** Beats left in the count-in (ceil), 0 when playing. */
  get countdownBeatsLeft() {
    return this.t < 0 ? Math.ceil(-this.t / this.choreo.beatDuration) : 0;
  }

  /** Progress 0..1 of the playable part. */
  get progress() {
    if (this.t <= 0) return 0;
    const p = this.t / this.choreo.duration;
    return p > 1 ? 1 : p;
  }

  /**
   * The recorded stage-frame run as plain JSON (for localStorage['dp.runs'] / POST /api/runs):
   * { choreoId, fps, frameCount, head: number[][7], left: number[][3], right: number[][3],
   *   score, playedAt, player }
   */
  runToJSON() {
    const run = this.run;
    if (!run || run.lastIndex < 0) return null;
    const n = run.frameCount;
    const r4 = (x) => Math.round(x * 10000) / 10000;
    const head = new Array(n), left = new Array(n), right = new Array(n);
    for (let i = 0; i < n; i++) {
      const o7 = i * 7, o3 = i * 3;
      head[i] = [r4(run.head[o7]), r4(run.head[o7 + 1]), r4(run.head[o7 + 2]), r4(run.head[o7 + 3]), r4(run.head[o7 + 4]), r4(run.head[o7 + 5]), r4(run.head[o7 + 6])];
      left[i] = [r4(run.left[o3]), r4(run.left[o3 + 1]), r4(run.left[o3 + 2])];
      right[i] = [r4(run.right[o3]), r4(run.right[o3 + 1]), r4(run.right[o3 + 2])];
    }
    return {
      format: 'dancing-points-run/1',
      choreoId: run.choreoId,
      fps: run.fps,
      frameCount: n,
      head,
      left,
      right,
      score: this.result ? this.result.score : null,
      playedAt: this.startedAt,
      player: this.player,
      mode: this.mode,
    };
  }

  dispose() {
    this.removeAllListeners();
    if (this.scorer) this.scorer.onBeat = null;
  }
}

function firstFilled(run) {
  for (let i = 0; i < run.frameCount; i++) if (run.filled[i] === 1) return i;
  return -1;
}
