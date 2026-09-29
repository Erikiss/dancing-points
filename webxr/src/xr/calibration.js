// calibration.js - Calibration: W (WebXR local-floor world) <-> S (stage frame) transform with
// height normalisation (DESIGN.md section 3). Captures the standing head height h0, the facing
// yaw and the floor origin; `toStage` translates, rotates the facing direction onto -Z and scales
// positions by k = referenceHeight / h0 (quaternions are rotated, not scaled). Persistence goes
// through an injectable storage object (localStorage-like: getItem/setItem/removeItem).
// No three.js, no DOM.

import { STAGE, STORAGE_KEYS } from '../config.js';
import { yawFromQuat, quatFromYaw, quatMul, wrapAngle, matrixFromYawPosScale } from '../util/math.js';
import { createSample } from './input.js';

export const CALIBRATION_VERSION = 1;
export const MIN_HEAD_HEIGHT = 0.8;   // metres; below this the sample is not a standing player
export const MAX_HEAD_HEIGHT = 2.4;

export class Calibration {
  /**
   * @param {object} [opts]
   * @param {object} [opts.storage]  localStorage-like object (null = no persistence)
   * @param {string} [opts.key]      storage key (default STORAGE_KEYS.calibration)
   * @param {number} [opts.referenceHeight] default reference height (default STAGE.referenceHeight)
   */
  constructor(opts = {}) {
    this.storage = opts.storage || null;
    this.key = opts.key || STORAGE_KEYS.calibration;
    this.referenceHeight = opts.referenceHeight > 0 ? opts.referenceHeight : STAGE.referenceHeight;
    this.data = null;               // { h0, yaw, origin:[x,0,z], capturedAt, version }
    this._qYawInv = [0, 0, 0, 1];   // rotation W -> S (about Y by -yaw)
    this._qYaw = [0, 0, 0, 1];      // rotation S -> W
    this._cos = 1;
    this._sin = 0;
    this._acc = null;               // accumulation state for averaged capture
    this._stageSample = createSample();
  }

  get valid() {
    return this.data !== null;
  }

  get h0() {
    return this.data ? this.data.h0 : this.referenceHeight;
  }

  get yaw() {
    return this.data ? this.data.yaw : 0;
  }

  get origin() {
    return this.data ? this.data.origin : ZERO3;
  }

  /** Scale factor k = referenceHeight / h0 for the given (or default) reference height. */
  scale(referenceHeight = this.referenceHeight) {
    return referenceHeight / this.h0;
  }

  /** Install calibration data (from capture or storage). Returns false if invalid. */
  set(data) {
    if (!data || !(data.h0 >= MIN_HEAD_HEIGHT) || !(data.h0 <= MAX_HEAD_HEIGHT) || !Number.isFinite(data.yaw)) return false;
    const origin = Array.isArray(data.origin) && data.origin.length === 3 ? data.origin : [0, 0, 0];
    if (!Number.isFinite(origin[0]) || !Number.isFinite(origin[2])) return false;
    this.data = {
      h0: data.h0,
      yaw: wrapAngle(data.yaw),
      origin: [origin[0], 0, origin[2]],
      capturedAt: data.capturedAt || new Date().toISOString(),
      version: CALIBRATION_VERSION,
    };
    this._updateRotation();
    return true;
  }

  _updateRotation() {
    const yaw = this.data ? this.data.yaw : 0;
    quatFromYaw(this._qYaw, yaw);
    quatFromYaw(this._qYawInv, -yaw);
    this._cos = Math.cos(yaw);
    this._sin = Math.sin(yaw);
  }

  /**
   * Capture from a single W-frame sample: h0 = head height, yaw = head forward projected on the
   * ground, origin = head position projected on the floor. Returns the data or null when the
   * sample is unusable (no valid head pose / implausible height).
   */
  capture(sample, opts = {}) {
    const head = sample && sample.head;
    if (!head || head.valid === false) return null;
    const h0 = head.p[1];
    if (!(h0 >= MIN_HEAD_HEIGHT) || !(h0 <= MAX_HEAD_HEIGHT)) return null;
    const fallbackYaw = typeof opts.fallbackYaw === 'number' ? opts.fallbackYaw : this.yaw;
    const yaw = yawFromQuat(head.q, fallbackYaw);
    const ok = this.set({ h0, yaw, origin: [head.p[0], 0, head.p[2]], capturedAt: new Date().toISOString() });
    return ok ? this.data : null;
  }

  /** Begin an averaged capture (e.g. while the trigger is held). */
  beginCapture() {
    this._acc = { n: 0, h: 0, x: 0, z: 0, c: 0, s: 0 };
  }

  /** Add a sample to the averaged capture. Returns the number of accepted samples. */
  addSample(sample) {
    if (!this._acc) this.beginCapture();
    const head = sample && sample.head;
    if (!head || head.valid === false) return this._acc.n;
    const h = head.p[1];
    if (!(h >= MIN_HEAD_HEIGHT) || !(h <= MAX_HEAD_HEIGHT)) return this._acc.n;
    const yaw = yawFromQuat(head.q, NaN);
    if (!Number.isFinite(yaw)) return this._acc.n;
    const a = this._acc;
    a.n++; a.h += h; a.x += head.p[0]; a.z += head.p[2]; a.c += Math.cos(yaw); a.s += Math.sin(yaw);
    return a.n;
  }

  /** Finish the averaged capture; installs and returns the data (null if nothing accepted). */
  endCapture() {
    const a = this._acc;
    this._acc = null;
    if (!a || a.n === 0) return null;
    const yaw = Math.atan2(a.s, a.c);
    const ok = this.set({ h0: a.h / a.n, yaw, origin: [a.x / a.n, 0, a.z / a.n], capturedAt: new Date().toISOString() });
    return ok ? this.data : null;
  }

  // ---- point / quaternion transforms -------------------------------------------------------

  /** W -> S position (translate, rotate by -yaw, scale by k). `out` may alias `pW`. */
  worldToStage(out, pW, referenceHeight = this.referenceHeight) {
    const k = referenceHeight / this.h0;
    const o = this.origin;
    const x = pW[0] - o[0], y = pW[1], z = pW[2] - o[2];
    // rotate by -yaw: x' = x cos - z sin ; z' = x sin + z cos
    const c = this._cos, s = this._sin;
    out[0] = (x * c - z * s) * k;
    out[1] = y * k;
    out[2] = (x * s + z * c) * k;
    return out;
  }

  /** S -> W position (unscale, rotate by +yaw, translate). `out` may alias `pS`. */
  stageToWorld(out, pS, referenceHeight = this.referenceHeight) {
    const invK = this.h0 / referenceHeight;
    const o = this.origin;
    const x = pS[0] * invK, y = pS[1] * invK, z = pS[2] * invK;
    const c = this._cos, s = this._sin;
    out[0] = x * c + z * s + o[0];
    out[1] = y;
    out[2] = -x * s + z * c + o[2];
    return out;
  }

  /** W -> S orientation. */
  worldToStageQuat(out, qW) {
    return quatMul(out, this._qYawInv, qW);
  }

  /** S -> W orientation. */
  stageToWorldQuat(out, qS) {
    return quatMul(out, this._qYaw, qS);
  }

  /**
   * Column-major 4x4 (three.js `Matrix4.fromArray` compatible) mapping S -> W including the
   * 1/k scale: useful to parent the whole stage (teacher, ghost, floor marker) under one object.
   */
  stageToWorldMatrix(out16, referenceHeight = this.referenceHeight) {
    return matrixFromYawPosScale(out16, this.yaw, this.origin, this.h0 / referenceHeight);
  }

  /**
   * Convert a W-frame input sample into a stage-frame sample (positions scaled by k, quaternions
   * rotated). Writes into `out` (default: internal reusable sample). Sets `out.k`.
   */
  toStage(sample, referenceHeight = this.referenceHeight, out = this._stageSample) {
    out.t = sample.t;
    out.k = referenceHeight / this.h0;
    this._pointToStage(out.head, sample.head, referenceHeight);
    this._pointToStage(out.left, sample.left, referenceHeight);
    this._pointToStage(out.right, sample.right, referenceHeight);
    return out;
  }

  /** Inverse of toStage (used to place a reference/teacher sample in the world). */
  fromStage(sampleS, referenceHeight = this.referenceHeight, out) {
    if (!out) out = createSample();
    out.t = sampleS.t;
    out.k = 1;
    this._pointToWorld(out.head, sampleS.head, referenceHeight);
    this._pointToWorld(out.left, sampleS.left, referenceHeight);
    this._pointToWorld(out.right, sampleS.right, referenceHeight);
    return out;
  }

  _pointToStage(o, p, referenceHeight) {
    this.worldToStage(o.p, p.p, referenceHeight);
    if (p.q) this.worldToStageQuat(o.q, p.q);
    else { o.q[0] = 0; o.q[1] = 0; o.q[2] = 0; o.q[3] = 1; }
    o.valid = p.valid !== false;
  }

  _pointToWorld(o, p, referenceHeight) {
    this.stageToWorld(o.p, p.p, referenceHeight);
    if (p.q) this.stageToWorldQuat(o.q, p.q);
    else { o.q[0] = 0; o.q[1] = 0; o.q[2] = 0; o.q[3] = 1; }
    o.valid = p.valid !== false;
  }

  // ---- persistence -------------------------------------------------------------------------

  save() {
    if (!this.storage || !this.data) return false;
    try {
      this.storage.setItem(this.key, JSON.stringify(this.data));
      return true;
    } catch (e) {
      if (typeof console !== 'undefined') console.warn('[calibration] save failed', e);
      return false;
    }
  }

  load() {
    if (!this.storage) return false;
    try {
      const raw = this.storage.getItem(this.key);
      if (!raw) return false;
      const data = JSON.parse(raw);
      if (!data || data.version !== CALIBRATION_VERSION) return false;
      return this.set(data);
    } catch (e) {
      if (typeof console !== 'undefined') console.warn('[calibration] load failed', e);
      return false;
    }
  }

  clear() {
    this.data = null;
    this._updateRotation();
    if (this.storage) {
      try { this.storage.removeItem(this.key); } catch (e) { /* ignore */ }
    }
  }

  toJSON() {
    return this.data;
  }
}

const ZERO3 = Object.freeze([0, 0, 0]);
