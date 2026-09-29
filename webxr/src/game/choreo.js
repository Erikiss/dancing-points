// choreo.js - choreography format `dancing-points-choreo/1` (DESIGN.md section 5):
// loading, validation, typed-array storage, interpolated sampling, move lookup, optional
// full-body frames (base64 float32). Pure JS, importable in Node, no three.js. `sampleAt` and
// `fullBodyAt` write into reusable output objects and never allocate.

import { slerp, lerp3At } from '../util/math.js';

export const CHOREO_FORMAT = 'dancing-points-choreo/1';
export const ID_PATTERN = /^[a-z0-9-]+$/;
export const SYNTH_PATTERNS = Object.freeze(['hiphop', 'house', 'metronome']);

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

function isVec(v, n) {
  if (!Array.isArray(v) || v.length !== n) return false;
  for (let i = 0; i < n; i++) if (!isNum(v[i])) return false;
  return true;
}

/** Expected number of frames for a choreography header (index 0 = beat 0, inclusive end). */
export function expectedFrameCount(c) {
  return Math.round((c.durationBeats * 60 / c.bpm) * c.fps) + 1;
}

/**
 * Validate a choreography object. Returns `{ ok, errors }`. Tolerant to missing optional fields
 * (artist, difficulty, audio, fullBody, meta, move hints); strict about lengths, NaNs and move
 * contiguity.
 */
export function validateChoreo(c) {
  const errors = [];
  const err = (m) => { errors.push(m); };
  if (!c || typeof c !== 'object') return { ok: false, errors: ['not an object'] };

  if (c.format !== CHOREO_FORMAT) err(`format must be "${CHOREO_FORMAT}"`);
  if (typeof c.id !== 'string' || !ID_PATTERN.test(c.id)) err('id must match [a-z0-9-]+');
  if (typeof c.title !== 'string' || c.title.length === 0) err('title missing');
  if (c.artist !== undefined && c.artist !== null && typeof c.artist !== 'string') err('artist must be a string');
  if (!isNum(c.bpm) || c.bpm <= 0 || c.bpm > 400) err('bpm must be a number in (0, 400]');
  if (!isNum(c.beatsPerBar) || c.beatsPerBar < 1 || c.beatsPerBar !== Math.floor(c.beatsPerBar)) err('beatsPerBar must be a positive integer');
  if (!isNum(c.countInBeats) || c.countInBeats < 0 || c.countInBeats !== Math.floor(c.countInBeats)) err('countInBeats must be a non-negative integer');
  if (!isNum(c.durationBeats) || c.durationBeats <= 0) err('durationBeats must be > 0');
  if (!isNum(c.fps) || c.fps <= 0) err('fps must be > 0');
  if (!isNum(c.referenceHeight) || c.referenceHeight <= 0.5 || c.referenceHeight > 3) err('referenceHeight must be in (0.5, 3]');
  if (c.mirror !== undefined && typeof c.mirror !== 'boolean') err('mirror must be a boolean');
  if (c.difficulty !== undefined && (!isNum(c.difficulty) || c.difficulty < 1 || c.difficulty > 5)) err('difficulty must be 1..5');

  if (c.audio !== undefined && c.audio !== null) {
    if (typeof c.audio !== 'object') err('audio must be an object');
    else {
      if (c.audio.url !== undefined && c.audio.url !== null && typeof c.audio.url !== 'string') err('audio.url must be a string or null');
      if (c.audio.synth !== undefined && c.audio.synth !== null && !SYNTH_PATTERNS.includes(c.audio.synth)) err(`audio.synth must be one of ${SYNTH_PATTERNS.join(', ')}`);
      if (c.audio.offsetSec !== undefined && !isNum(c.audio.offsetSec)) err('audio.offsetSec must be a number');
      if (c.audio.gain !== undefined && (!isNum(c.audio.gain) || c.audio.gain < 0)) err('audio.gain must be >= 0');
    }
  }

  // moves: contiguous, ordered, covering [0, durationBeats]
  if (!Array.isArray(c.moves) || c.moves.length === 0) err('moves must be a non-empty array');
  else if (isNum(c.durationBeats)) {
    let expectedStart = 0;
    for (let i = 0; i < c.moves.length; i++) {
      const m = c.moves[i];
      if (!m || typeof m !== 'object') { err(`moves[${i}] not an object`); break; }
      if (typeof m.name !== 'string' || m.name.length === 0) err(`moves[${i}].name missing`);
      if (!isNum(m.startBeat) || !isNum(m.endBeat)) { err(`moves[${i}] start/endBeat must be numbers`); break; }
      if (m.endBeat <= m.startBeat) err(`moves[${i}] endBeat must be > startBeat`);
      if (Math.abs(m.startBeat - expectedStart) > 1e-9) err(`moves[${i}].startBeat must be ${expectedStart} (contiguous)`);
      if (m.hint !== undefined && m.hint !== null && typeof m.hint !== 'string') err(`moves[${i}].hint must be a string`);
      expectedStart = m.endBeat;
    }
    if (errors.length === 0 && Math.abs(expectedStart - c.durationBeats) > 1e-9) err(`moves must end at durationBeats (${c.durationBeats}), got ${expectedStart}`);
  }

  // frames
  if (!c.frames || typeof c.frames !== 'object') err('frames missing');
  else if (isNum(c.bpm) && isNum(c.durationBeats) && isNum(c.fps) && c.bpm > 0 && c.fps > 0) {
    const n = expectedFrameCount(c);
    const check = (key, width) => {
      const arr = c.frames[key];
      if (!Array.isArray(arr)) { err(`frames.${key} missing`); return; }
      if (arr.length !== n && arr.length !== n - 1) { err(`frames.${key} has ${arr.length} entries, expected ${n}`); return; }
      for (let i = 0; i < arr.length; i++) {
        if (!isVec(arr[i], width)) { err(`frames.${key}[${i}] must be ${width} finite numbers`); return; }
      }
    };
    check('head', 7);
    check('left', 3);
    check('right', 3);
    if (Array.isArray(c.frames.head) && Array.isArray(c.frames.left) && Array.isArray(c.frames.right)) {
      if (c.frames.head.length !== c.frames.left.length || c.frames.head.length !== c.frames.right.length) {
        err('frames.head/left/right must have the same length');
      }
    }
  }

  // fullBody (optional)
  if (c.fullBody !== undefined && c.fullBody !== null) {
    const fb = c.fullBody;
    if (typeof fb !== 'object') err('fullBody must be an object');
    else {
      if (!Array.isArray(fb.joints) || fb.joints.length === 0) err('fullBody.joints missing');
      if (!Array.isArray(fb.parents) || (Array.isArray(fb.joints) && fb.parents.length !== fb.joints.length)) err('fullBody.parents must match joints');
      if (!isNum(fb.fps) || fb.fps <= 0) err('fullBody.fps must be > 0');
      if (fb.encoding !== 'base64-float32') err('fullBody.encoding must be "base64-float32"');
      if (typeof fb.positions !== 'string') err('fullBody.positions must be a base64 string');
      else if (Array.isArray(fb.joints) && fb.joints.length > 0) {
        const bytes = base64ByteLength(fb.positions);
        const stride = fb.joints.length * 3 * 4;
        if (bytes % stride !== 0) err(`fullBody.positions byte length ${bytes} is not a multiple of joints*3*4 (${stride})`);
      }
    }
  }

  if (c.meta !== undefined && c.meta !== null && typeof c.meta !== 'object') err('meta must be an object');

  return { ok: errors.length === 0, errors };
}

// ---------------------------------------------------------------------------------------------
// base64 <-> Float32Array (works in Node and browsers)

function base64ByteLength(s) {
  const clean = s.replace(/[^A-Za-z0-9+/=]/g, '');
  let pad = 0;
  if (clean.endsWith('==')) pad = 2;
  else if (clean.endsWith('=')) pad = 1;
  return Math.floor(clean.length * 3 / 4) - pad;
}

export function decodeBase64Float32(s) {
  let bytes;
  if (typeof globalThis.Buffer === 'function' && typeof globalThis.Buffer.from === 'function') {
    const buf = globalThis.Buffer.from(s, 'base64');
    bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  } else {
    const bin = globalThis.atob(s);
    bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  }
  const usable = bytes.byteLength - (bytes.byteLength % 4);
  const copy = new Uint8Array(usable);
  copy.set(bytes.subarray(0, usable));
  return new Float32Array(copy.buffer);
}

export function encodeBase64Float32(f32) {
  const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  if (typeof globalThis.Buffer === 'function' && typeof globalThis.Buffer.from === 'function') {
    return globalThis.Buffer.from(bytes).toString('base64');
  }
  let bin = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return globalThis.btoa(bin);
}

// ---------------------------------------------------------------------------------------------

/** Allocate a reusable sample object in the shape returned by `Choreo.sampleAt`. */
export function createChoreoSample() {
  return {
    t: 0,
    head: { p: [0, 0, 0], q: [0, 0, 0, 1] },
    left: { p: [0, 0, 0] },
    right: { p: [0, 0, 0] },
  };
}

/**
 * Mirror a choreo sample (x -> -x, left/right swapped, head yaw mirrored). `out` must not alias
 * `s`. Used for `mirror: true` choreographies (scoring + display).
 */
export function mirrorChoreoSample(out, s) {
  out.t = s.t;
  out.head.p[0] = -s.head.p[0]; out.head.p[1] = s.head.p[1]; out.head.p[2] = s.head.p[2];
  // mirror rotation through the YZ plane: (x, y, z, w) -> (x, -y, -z, w)
  out.head.q[0] = s.head.q[0]; out.head.q[1] = -s.head.q[1]; out.head.q[2] = -s.head.q[2]; out.head.q[3] = s.head.q[3];
  out.left.p[0] = -s.right.p[0]; out.left.p[1] = s.right.p[1]; out.left.p[2] = s.right.p[2];
  out.right.p[0] = -s.left.p[0]; out.right.p[1] = s.left.p[1]; out.right.p[2] = s.left.p[2];
  return out;
}

export class Choreo {
  /** @param {object} json a validated (or validatable) choreography object */
  constructor(json) {
    const v = validateChoreo(json);
    if (!v.ok) {
      const e = new Error(`invalid choreography: ${v.errors.join('; ')}`);
      e.errors = v.errors;
      throw e;
    }
    this.json = json;
    this.format = json.format;
    this.id = json.id;
    this.title = json.title;
    this.artist = json.artist || '';
    this.bpm = json.bpm;
    this.beatsPerBar = json.beatsPerBar;
    this.countInBeats = json.countInBeats;
    this.durationBeats = json.durationBeats;
    this.fps = json.fps;
    this.referenceHeight = json.referenceHeight;
    this.mirror = json.mirror === true;
    this.difficulty = json.difficulty !== undefined ? json.difficulty : 1;
    this.audio = {
      url: null,
      synth: 'hiphop',
      offsetSec: 0,
      gain: 0.8,
      ...(json.audio || {}),
    };
    this.moves = json.moves.map((m, i) => ({
      index: i,
      name: m.name,
      startBeat: m.startBeat,
      endBeat: m.endBeat,
      hint: m.hint || '',
    }));
    this.meta = json.meta || {};

    this.beatDuration = 60 / this.bpm;
    this.duration = this.durationBeats * this.beatDuration;   // seconds
    this.frameDt = 1 / this.fps;

    const n = json.frames.head.length;
    this.frameCount = n;
    this.headP = new Float32Array(n * 3);
    this.headQ = new Float32Array(n * 4);
    this.leftP = new Float32Array(n * 3);
    this.rightP = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const h = json.frames.head[i];
      this.headP[i * 3] = h[0]; this.headP[i * 3 + 1] = h[1]; this.headP[i * 3 + 2] = h[2];
      // normalise the quaternion defensively
      let qx = h[3], qy = h[4], qz = h[5], qw = h[6];
      const ql = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
      if (ql > 1e-9) { qx /= ql; qy /= ql; qz /= ql; qw /= ql; } else { qx = 0; qy = 0; qz = 0; qw = 1; }
      this.headQ[i * 4] = qx; this.headQ[i * 4 + 1] = qy; this.headQ[i * 4 + 2] = qz; this.headQ[i * 4 + 3] = qw;
      const l = json.frames.left[i];
      this.leftP[i * 3] = l[0]; this.leftP[i * 3 + 1] = l[1]; this.leftP[i * 3 + 2] = l[2];
      const r = json.frames.right[i];
      this.rightP[i * 3] = r[0]; this.rightP[i * 3 + 1] = r[1]; this.rightP[i * 3 + 2] = r[2];
    }

    this._sample = createChoreoSample();
    this._qa = [0, 0, 0, 1];
    this._qb = [0, 0, 0, 1];

    this.fullBody = null;
    if (json.fullBody) {
      const fb = json.fullBody;
      const positions = decodeBase64Float32(fb.positions);
      const jointCount = fb.joints.length;
      const stride = jointCount * 3;
      const fbFrames = Math.floor(positions.length / stride);
      this.fullBody = {
        joints: fb.joints.slice(),
        parents: fb.parents.slice(),
        fps: fb.fps,
        jointCount,
        frameCount: fbFrames,
        positions: positions.subarray(0, fbFrames * stride),
      };
      this._fullBodyOut = new Float32Array(stride);
    }
  }

  get hasFullBody() {
    return this.fullBody !== null && this.fullBody.frameCount > 0;
  }

  beatToTime(beat) {
    return beat * this.beatDuration;
  }

  timeToBeat(t) {
    return t / this.beatDuration;
  }

  /** Frame index (float, clamped to [0, frameCount-1]) for time t. */
  frameIndexAt(t) {
    const f = t * this.fps;
    if (!(f > 0)) return 0;
    const max = this.frameCount - 1;
    return f > max ? max : f;
  }

  /**
   * Interpolated 3-point sample at time t (seconds since beat 0), clamped at both ends. Writes
   * into `out` (default: an internal reusable object - copy it if you need to keep it).
   */
  sampleAt(t, out = this._sample) {
    const f = this.frameIndexAt(t);
    const i0 = Math.floor(f);
    const i1 = i0 + 1 < this.frameCount ? i0 + 1 : i0;
    const a = f - i0;
    out.t = t;
    lerp3At(out.head.p, this.headP, i0 * 3, this.headP, i1 * 3, a);
    lerp3At(out.left.p, this.leftP, i0 * 3, this.leftP, i1 * 3, a);
    lerp3At(out.right.p, this.rightP, i0 * 3, this.rightP, i1 * 3, a);
    const qa = this._qa, qb = this._qb;
    qa[0] = this.headQ[i0 * 4]; qa[1] = this.headQ[i0 * 4 + 1]; qa[2] = this.headQ[i0 * 4 + 2]; qa[3] = this.headQ[i0 * 4 + 3];
    if (i1 === i0 || a === 0) {
      out.head.q[0] = qa[0]; out.head.q[1] = qa[1]; out.head.q[2] = qa[2]; out.head.q[3] = qa[3];
    } else {
      qb[0] = this.headQ[i1 * 4]; qb[1] = this.headQ[i1 * 4 + 1]; qb[2] = this.headQ[i1 * 4 + 2]; qb[3] = this.headQ[i1 * 4 + 3];
      slerp(out.head.q, qa, qb, a);
    }
    return out;
  }

  /** Index of the move whose [startBeat, endBeat) contains the beat at time t; -1 if none. */
  moveIndexAt(t) {
    const beat = this.timeToBeat(t);
    const moves = this.moves;
    for (let i = 0; i < moves.length; i++) {
      if (beat >= moves[i].startBeat && beat < moves[i].endBeat) return i;
    }
    // t exactly at the end -> last move (playing the final tick), otherwise none
    return -1;
  }

  /** Move object at time t or null. */
  moveAt(t) {
    const i = this.moveIndexAt(t);
    return i >= 0 ? this.moves[i] : null;
  }

  /** Index of the move containing `beat`, -1 if none. */
  moveIndexAtBeat(beat) {
    const moves = this.moves;
    for (let i = 0; i < moves.length; i++) {
      if (beat >= moves[i].startBeat && beat < moves[i].endBeat) return i;
    }
    return -1;
  }

  /**
   * Interpolated full-body joint positions (stage frame, jointCount*3 floats) at time t into
   * `out` (default: reusable internal buffer). Returns null when there is no fullBody block.
   */
  fullBodyAt(t, out = this._fullBodyOut) {
    const fb = this.fullBody;
    if (!fb || fb.frameCount === 0) return null;
    const stride = fb.jointCount * 3;
    let f = t * fb.fps;
    const max = fb.frameCount - 1;
    if (!(f > 0)) f = 0;
    else if (f > max) f = max;
    const i0 = Math.floor(f);
    const i1 = i0 + 1 <= max ? i0 + 1 : i0;
    const a = f - i0;
    const pos = fb.positions;
    const o0 = i0 * stride, o1 = i1 * stride;
    for (let k = 0; k < stride; k++) {
      const v0 = pos[o0 + k];
      out[k] = v0 + (pos[o1 + k] - v0) * a;
    }
    return out;
  }

  /** Reference frame (nearest, no interpolation) accessors for precomputation. */
  frameTime(i) {
    return i * this.frameDt;
  }

  /** Summary entry as used by choreos/index.json. */
  toIndexEntry(file = `${this.id}.json`) {
    return {
      id: this.id,
      title: this.title,
      artist: this.artist,
      bpm: this.bpm,
      durationBeats: this.durationBeats,
      difficulty: this.difficulty,
      file,
    };
  }

  toJSON() {
    return this.json;
  }
}

/**
 * Load a choreography from a URL (fetch + JSON) or wrap an already parsed object.
 * `fetchImpl` defaults to the global fetch. Rejects with an Error listing validation problems.
 */
export async function loadChoreo(urlOrObject, fetchImpl = globalThis.fetch) {
  let json = urlOrObject;
  if (typeof urlOrObject === 'string') {
    if (typeof fetchImpl !== 'function') throw new Error('no fetch implementation available');
    const res = await fetchImpl(urlOrObject, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`failed to load choreography ${urlOrObject}: HTTP ${res.status}`);
    json = await res.json();
    // resolve a relative audio url against the choreo file location
    if (json && json.audio && typeof json.audio.url === 'string' && json.audio.url && !/^[a-z]+:/i.test(json.audio.url) && !json.audio.url.startsWith('/')) {
      const base = urlOrObject.slice(0, urlOrObject.lastIndexOf('/') + 1);
      json.audio = { ...json.audio, url: base + json.audio.url };
    }
  }
  if (json instanceof Choreo) return json;
  return new Choreo(json);
}
