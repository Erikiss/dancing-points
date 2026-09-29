// avatar-driver.js - NeuralAvatarDriver: main-thread glue between the render loop and the
// inference Web Worker (src/net/worker.js).
//
// Responsibility: resample the stage-frame 3-point input to the 30 Hz network tick, convert it to
// an unscaled DP frame, post it to the worker, receive poses, interpolate between the last two
// poses for smooth 72 Hz rendering, watch inference time (performance guard: slow -> 15 Hz,
// too slow -> neural avatar disabled with a German notice), expose telemetry export.
//
// Must not: import three.js or onnxruntime, touch `window`/`document` at import time (Node
// unit tests inject a fake Worker class through `WorkerClass`), allocate per render frame
// (the per-tick 13-float frame message and the per-pose message object are the only steady-state
// allocations; pose buffers travel back and forth as transferables).
//
// Output frame: `getPose(out, tMs)` and `latest.positionsWorldS` are joint positions in the
// UNSCALED stage frame S (real-world metres, calibration origin/yaw applied). To render them at
// the player's physical place use `calibration.stageToWorld(out, p, calibration.h0)` (scale 1);
// multiply by `driver.scale` (= sample.k) to compare with a choreography at reference height.

import { FRAME_DIM, N_JOINTS, sampleSToFrameDP } from './pipeline.js';

export const POSE_FLOATS = N_JOINTS * 3;            // 102
export const POSE_BUFFER_BYTES = (POSE_FLOATS + 3 + 4) * 4; // positions + rootS + contacts

/** German player-facing notices (the HUD shows `status.notice`). */
export const NOTICES = Object.freeze({
  loading: 'Neural-Avatar wird geladen …',
  ready: 'Neural-Avatar bereit',
  slow: 'Neural-Avatar: reduzierte Rate (15 Hz)',
  disabledSlow: 'Neural-Avatar deaktiviert (zu langsam)',
  error: 'Neural-Avatar nicht verfügbar',
});

const DEFAULT_GUARD = Object.freeze({
  windowTicks: 60,   // average inferenceMs over this many inferences
  slowMs: 25,        // > slowMs   -> infer every 2nd frame (15 Hz)
  disableMs: 60,     // > disableMs -> disable the neural avatar
  minTicks: 10,      // evaluate the guard only after this many inferences
  recoverRatio: 0.7, // back to 30 Hz when the average drops below slowMs * recoverRatio
});

function defaultNow() {
  return (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
}

function defaultThreads() {
  if (typeof crossOriginIsolated === 'undefined' || !crossOriginIsolated) return 1;
  const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 2;
  return Math.max(1, Math.min(4, cores - 1));
}

function resolveUrl(rel) {
  try {
    return new URL(rel, import.meta.url).href;
  } catch (e) {
    return rel;
  }
}

export class NeuralAvatarDriver {
  /**
   * @param {object} o
   *   workerUrl   URL of src/net/worker.js (default: next to this module)
   *   modelsUrl   directory URL with meta.json/skeleton.json/init_pose.json + int8 models (default ../../models/<style>/)
   *   style       model style directory (default 'free'); set 'leader'|'follower' (default 'leader')
   *   tickHz      network tick rate (30)
   *   rootCorrection / eulerRatio   post-processing parameters (forwarded to the worker)
   *   telemetry / telemetryTicks    record network I/O in the worker
   *   guard       {windowTicks, slowMs, disableMs, minTicks, recoverRatio}
   *   onPose(latest), onStatus({state, notice, ...}), onError(err)
   *   WorkerClass (tests), now() -> ms (tests), ortUrl / wasmDir (override the vendored ORT location)
   */
  constructor(o = {}) {
    this.style = o.style || 'free';
    this.set = o.set || 'leader';
    this.workerUrl = o.workerUrl || resolveUrl('./worker.js');
    this.modelsUrl = o.modelsUrl || resolveUrl('../../models/' + this.style + '/');
    this.ortUrl = o.ortUrl || null;
    this.wasmDir = o.wasmDir || null;
    this.tickHz = o.tickHz || 30;
    this.tickSec = 1 / this.tickHz;
    this.rootCorrection = o.rootCorrection !== undefined ? o.rootCorrection : 0.35;
    this.eulerRatio = o.eulerRatio !== undefined ? o.eulerRatio : 0.5;
    this.telemetry = !!o.telemetry;
    this.telemetryTicks = o.telemetryTicks || 2000;
    // WASM threads need SharedArrayBuffer = a cross-origin isolated page (COOP/COEP headers); without
    // it onnxruntime falls back to 1 thread anyway. Leave one core for the render thread / XR runtime.
    this.numThreads = o.numThreads || defaultThreads();
    this.guard = Object.assign({}, DEFAULT_GUARD, o.guard || {});
    this.onPose = o.onPose || null;
    this.onStatus = o.onStatus || null;
    this.onError = o.onError || null;
    this.WorkerClass = o.WorkerClass || (typeof Worker !== 'undefined' ? Worker : null);
    this.now = o.now || defaultNow;

    this.worker = null;
    this.state = 'idle'; // idle | loading | ready | running | slow | disabled | error | disposed
    this.notice = '';
    this.info = null;    // 'ready' message payload
    this.inferEvery = 1;

    // input resampling
    this.frame = new Float32Array(FRAME_DIM);     // posted frame (DP, unscaled)
    this.frameA = new Float64Array(FRAME_DIM);    // previous sample as DP frame
    this.frameB = new Float64Array(FRAME_DIM);    // latest sample as DP frame
    this.tA = -1;
    this.tB = -1;
    this.nextTickT = -1;
    this.seq = 0;
    this.scale = 1;

    // poses: double buffer for interpolation
    this.posePrev = new Float32Array(POSE_FLOATS);
    this.poseLast = new Float32Array(POSE_FLOATS);
    this.rootPrev = new Float32Array(3);
    this.rootLast = new Float32Array(3);
    this.poseOut = new Float32Array(POSE_FLOATS);
    this.rootOut = new Float32Array(3);
    this.tPrev = 0;   // arrival time (ms) of posePrev
    this.tLast = 0;   // arrival time (ms) of poseLast
    this.poseIntervalMs = 1000 * this.tickSec; // smoothed pose arrival interval (interpolation window)
    this.poseCount = 0;
    this.latest = {   // last received pose (reused object)
      tick: -1, t: 0, seq: 0, positionsWorldS: this.poseLast, rootS: this.rootLast,
      contacts: new Float32Array(4), rootYawS: 0, inferenceMs: 0, mappingMs: 0, trackingMs: 0, arrivedMs: 0,
    };

    // guard statistics
    this.msRing = new Float32Array(Math.max(1, this.guard.windowTicks));
    this.msCount = 0;
    this.msIndex = 0;
    this.inferenceMs = 0;
    this.stats = { frames: 0, poses: 0, dropped: 0, meanMs: 0, maxMs: 0, minMs: Infinity, disabledAt: 0 };

    this._pendingTelemetry = [];
    this._readyResolve = null;
    this._readyReject = null;
    this._onMessage = (ev) => this._handle(ev.data);
    this._onError = (ev) => this._fail('worker error: ' + (ev && ev.message ? ev.message : String(ev)), 'worker');
  }

  /** Create the worker, load ORT and the models. Resolves with the worker's ready info. */
  init() {
    if (this.state !== 'idle') return Promise.reject(new Error('NeuralAvatarDriver.init(): state ' + this.state));
    if (!this.WorkerClass) return Promise.reject(new Error('Web Workers not available'));
    this._setState('loading', NOTICES.loading);
    return new Promise((resolve, reject) => {
      this._readyResolve = resolve;
      this._readyReject = reject;
      try {
        this.worker = new this.WorkerClass(this.workerUrl, { type: 'module', name: 'dp-net' });
      } catch (e) {
        this._fail('cannot start worker: ' + e.message, 'init');
        return;
      }
      this.worker.onmessage = this._onMessage;
      this.worker.onerror = this._onError;
      this.worker.postMessage({
        type: 'init',
        modelsUrl: this.modelsUrl,
        ortUrl: this.ortUrl,
        wasmDir: this.wasmDir,
        set: this.set,
        rootCorrection: this.rootCorrection,
        eulerRatio: this.eulerRatio,
        inferEvery: this.inferEvery,
        telemetry: this.telemetry,
        telemetryTicks: this.telemetryTicks,
        numThreads: this.numThreads,
        simd: true,
      });
    });
  }

  /** Start feeding frames (after init resolved). */
  start() {
    if (this.state !== 'ready' && this.state !== 'running' && this.state !== 'slow') return false;
    this.tA = this.tB = -1;
    this.nextTickT = -1;
    this.poseIntervalMs = 1000 * this.tickSec * this.inferEvery;
    if (this.state === 'ready') this._setState('running', '');
    return true;
  }

  /** Stop feeding frames and forget the history (the worker keeps its models). */
  stop() {
    if (this.state === 'running' || this.state === 'slow') this._setState('ready', '');
    this.tA = this.tB = -1;
    this.nextTickT = -1;
    this.poseCount = 0;
    if (this.worker) this._post({ type: 'reset' });
  }

  /** Whether frames are currently accepted. */
  get active() {
    return this.state === 'running' || this.state === 'slow';
  }

  /**
   * Feed one stage-frame sample (src/xr/input.js shape, at reference height with sample.k) from
   * the render loop. Frames are linearly resampled onto the 30 Hz grid (sample.t in seconds).
   */
  update(sampleS, tSec) {
    if (!this.active) return false;
    const t = tSec !== undefined ? tSec : sampleS.t;
    const k = sampleS.k > 0 ? sampleS.k : 1;
    this.scale = k;
    // shift B -> A, new sample -> B
    if (this.tB >= 0) {
      this.frameA.set(this.frameB);
      this.tA = this.tB;
    }
    sampleSToFrameDP(this.frameB, 0, sampleS, k);
    this.tB = t;
    if (this.tA < 0) {
      // first sample: start the grid here
      this.nextTickT = t;
    }
    let posted = 0;
    while (this.nextTickT <= this.tB + 1e-9 && posted < 4) {
      const span = this.tB - this.tA;
      let alpha = (this.tA >= 0 && span > 1e-9) ? (this.nextTickT - this.tA) / span : 1;
      if (alpha < 0) alpha = 0; else if (alpha > 1) alpha = 1;
      for (let i = 0; i < FRAME_DIM; i++) this.frame[i] = this.frameA[i] + (this.frameB[i] - this.frameA[i]) * alpha;
      this._normalizeQuat();
      this._postFrame(this.nextTickT);
      this.nextTickT += this.tickSec;
      posted++;
    }
    if (this.nextTickT < this.tB - 4 * this.tickSec) this.nextTickT = this.tB; // fell far behind: resync
    return posted > 0;
  }

  /** Feed a ready-made DP frame (13 values) at time t (tests, tools). */
  pushFrameDP(frame, tSec) {
    if (!this.active) return false;
    for (let i = 0; i < FRAME_DIM; i++) this.frame[i] = frame[i];
    this._postFrame(tSec);
    return true;
  }

  _normalizeQuat() {
    const f = this.frame;
    const n = Math.hypot(f[3], f[4], f[5], f[6]);
    if (n > 1e-9 && Math.abs(n - 1) > 1e-6) {
      f[3] /= n; f[4] /= n; f[5] /= n; f[6] /= n;
    }
  }

  _postFrame(t) {
    this.seq++;
    this.stats.frames++;
    this._post({ type: 'frame', seq: this.seq, t, frame: this.frame });
  }

  /**
   * Interpolated joint positions (unscaled S, Float32Array(102)) for render time tMs (default now).
   * Returns null while no pose is available. `out` defaults to an internal reused array.
   */
  getPose(out, tMs) {
    if (this.poseCount === 0 || !this.active) return null;
    const dst = out || this.poseOut;
    if (this.poseCount === 1) {
      dst.set(this.poseLast);
      this.rootOut.set(this.rootLast);
      return dst;
    }
    const now = tMs !== undefined ? tMs : this.now();
    let a = (now - this.tLast) / this.poseIntervalMs;
    if (a < 0) a = 0; else if (a > 1) a = 1;
    for (let i = 0; i < POSE_FLOATS; i++) dst[i] = this.posePrev[i] + (this.poseLast[i] - this.posePrev[i]) * a;
    for (let i = 0; i < 3; i++) this.rootOut[i] = this.rootPrev[i] + (this.rootLast[i] - this.rootPrev[i]) * a;
    return dst;
  }

  /** Interpolated root position (unscaled S) of the last getPose() call. */
  get rootS() {
    return this.rootOut;
  }

  /** Change post-processing parameters at runtime. */
  setConfig(cfg) {
    if (cfg.rootCorrection !== undefined) this.rootCorrection = cfg.rootCorrection;
    if (cfg.eulerRatio !== undefined) this.eulerRatio = cfg.eulerRatio;
    if (cfg.telemetry !== undefined) this.telemetry = !!cfg.telemetry;
    this._post({ type: 'config', rootCorrection: this.rootCorrection, eulerRatio: this.eulerRatio, telemetry: this.telemetry, inferEvery: this.inferEvery });
  }

  /** Re-initialise the avatar at the next frame's root. */
  reset() {
    this.poseCount = 0;
    this._post({ type: 'reset' });
  }

  /** Ask the worker for its telemetry ring (inputs/outputs of the last N ticks). */
  exportTelemetry() {
    if (!this.worker || this.state === 'disposed') return Promise.reject(new Error('no worker'));
    return new Promise((resolve, reject) => {
      this._pendingTelemetry.push({ resolve, reject });
      this._post({ type: 'telemetry' });
    });
  }

  /** Browser only: download the telemetry as a JSON file. Resolves with the file name. */
  async downloadTelemetry(filename) {
    const data = await this.exportTelemetry();
    if (typeof document === 'undefined' || typeof Blob === 'undefined' || typeof URL === 'undefined') return null;
    const name = filename || ('dp-telemetry-' + new Date().toISOString().replace(/[:.]/g, '-') + '.json');
    const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    return name;
  }

  /** Terminate the worker. */
  dispose() {
    if (this.worker) {
      try { this.worker.postMessage({ type: 'dispose' }); } catch (e) { /* ignore */ }
      this.worker.terminate();
      this.worker = null;
    }
    for (const p of this._pendingTelemetry) p.reject(new Error('disposed'));
    this._pendingTelemetry.length = 0;
    this._setState('disposed', '');
  }

  // -------------------------------------------------------------------------------------------

  _post(msg) {
    if (!this.worker) return;
    try {
      this.worker.postMessage(msg);
    } catch (e) {
      this._fail('postMessage failed: ' + e.message, 'post');
    }
  }

  _setState(state, notice) {
    this.state = state;
    this.notice = notice;
    if (this.onStatus) {
      try {
        this.onStatus({ state, notice, inferenceMs: this.inferenceMs, inferEvery: this.inferEvery, driver: this });
      } catch (e) {
        console.warn('NeuralAvatarDriver onStatus listener failed', e);
      }
    }
  }

  _fail(message, stage) {
    console.warn('NeuralAvatarDriver: ' + message);
    const err = new Error(message);
    err.stage = stage;
    if (this._readyReject) {
      this._readyReject(err);
      this._readyReject = this._readyResolve = null;
    }
    this._setState('error', NOTICES.error);
    if (this.onError) this.onError(err);
  }

  _handle(msg) {
    if (!msg || typeof msg !== 'object') return;
    switch (msg.type) {
      case 'ready':
        this.info = msg;
        this._setState('ready', NOTICES.ready);
        if (this._readyResolve) {
          this._readyResolve(msg);
          this._readyResolve = this._readyReject = null;
        }
        break;
      case 'pose':
        this._handlePose(msg);
        break;
      case 'progress':
        if (this.onStatus) this.onStatus({ state: this.state, notice: NOTICES.loading, progress: msg, driver: this });
        break;
      case 'telemetry': {
        const waiters = this._pendingTelemetry.splice(0);
        for (const w of waiters) w.resolve(msg.data);
        break;
      }
      case 'error':
        this._fail(msg.message || 'worker error', msg.stage || 'worker');
        break;
      case 'log':
        console.log('[dp-net worker]', msg.message);
        break;
      default:
        break;
    }
  }

  _handlePose(msg) {
    const buffer = msg.buffer;
    if (!this.active || !buffer) {
      if (buffer && this.worker) this._recycle(buffer);
      return;
    }
    const positions = new Float32Array(buffer, 0, POSE_FLOATS);
    const root = new Float32Array(buffer, POSE_FLOATS * 4, 3);
    const contacts = new Float32Array(buffer, (POSE_FLOATS + 3) * 4, 4);
    // rotate the double buffer
    this.posePrev.set(this.poseLast);
    this.rootPrev.set(this.rootLast);
    this.tPrev = this.tLast;
    this.poseLast.set(positions);
    this.rootLast.set(root);
    this.latest.contacts.set(contacts);
    this.tLast = this.now();
    this.poseCount++;
    if (this.poseCount >= 2) {
      // EMA of the arrival gaps, clamped to [1 tick, 4 ticks]: adapts to 15 Hz mode / slow devices
      // without letting a single hiccup stall the interpolation.
      const nominal = 1000 * this.tickSec;
      let gap = this.tLast - this.tPrev;
      if (gap < nominal) gap = nominal; else if (gap > 4 * nominal) gap = 4 * nominal;
      this.poseIntervalMs = 0.8 * this.poseIntervalMs + 0.2 * gap;
    }
    this._recycle(buffer);
    const l = this.latest;
    l.tick = msg.tick; l.t = msg.t; l.seq = msg.seq; l.rootYawS = msg.rootYawS;
    l.inferenceMs = msg.inferenceMs; l.mappingMs = msg.mappingMs; l.trackingMs = msg.trackingMs; l.arrivedMs = this.tLast;
    this.stats.poses++;
    this.stats.dropped = msg.dropped || 0;
    this._guard(msg.inferenceMs);
    if (this.onPose && this.active) this.onPose(l);
  }

  _recycle(buffer) {
    if (!this.worker) return;
    try {
      this.worker.postMessage({ type: 'recycle', buffer }, [buffer]);
    } catch (e) {
      // structured clone fallback (fake workers / no transfer support): nothing to do
    }
  }

  _guard(ms) {
    if (!(ms >= 0)) return;
    this.msRing[this.msIndex] = ms;
    this.msIndex = (this.msIndex + 1) % this.msRing.length;
    if (this.msCount < this.msRing.length) this.msCount++;
    let sum = 0;
    for (let i = 0; i < this.msCount; i++) sum += this.msRing[i];
    const mean = sum / this.msCount;
    this.inferenceMs = mean;
    this.stats.meanMs = mean;
    if (ms > this.stats.maxMs) this.stats.maxMs = ms;
    if (ms < this.stats.minMs) this.stats.minMs = ms;
    if (this.msCount < this.guard.minTicks) return;
    if (mean > this.guard.disableMs) {
      this.stats.disabledAt = mean;
      this._disable();
    } else if (mean > this.guard.slowMs && this.inferEvery === 1) {
      this.inferEvery = 2;
      this._post({ type: 'config', inferEvery: 2 });
      this._setState('slow', NOTICES.slow);
    } else if (this.inferEvery === 2 && mean < this.guard.slowMs * this.guard.recoverRatio && this.state === 'slow') {
      this.inferEvery = 1;
      this._post({ type: 'config', inferEvery: 1 });
      this._setState('running', '');
    }
  }

  _disable() {
    if (this.worker) {
      try { this.worker.postMessage({ type: 'dispose' }); } catch (e) { /* ignore */ }
      this.worker.terminate();
      this.worker = null;
    }
    this.poseCount = 0;
    this._setState('disabled', NOTICES.disabledSlow);
  }
}
