// recorder.js - Recorder: records a new choreography (DESIGN.md section 8) from 30 Hz
// stage-frame samples (head pose with quaternion, both hands) into a valid
// `dancing-points-choreo/1` JSON object, plus helpers to persist it (localStorage-like
// storage), download it (browser) or POST it to /api/choreos. Frames that were not sampled
// (dropped frames) are filled by holding the previous frame. No three.js; DOM only inside
// `download()`.

import { STAGE, STORAGE_KEYS } from '../config.js';
import { CHOREO_FORMAT, validateChoreo, expectedFrameCount } from './choreo.js';

const round4 = (x) => Math.round(x * 10000) / 10000;
const round6 = (x) => Math.round(x * 1000000) / 1000000;

export class Recorder {
  /** @param {object} [opts] { fps = 30 } */
  constructor(opts = {}) {
    this.fps = opts.fps > 0 ? opts.fps : 30;
    this.state = 'idle';            // 'idle' | 'recording' | 'stopped'
    this.config = null;
    this.frameCount = 0;
    this.head = null;               // Float32Array(frameCount * 7)
    this.left = null;               // Float32Array(frameCount * 3)
    this.right = null;
    this.filled = null;             // Uint8Array(frameCount)
    this.lastIndex = -1;
    this.sampleCount = 0;
    this.startedAt = null;
    this._cur = new Float64Array(13);   // packed sample: head p(3) q(4), left p(3), right p(3)
    this._prev = new Float64Array(13);
    this._prevT = NaN;
  }

  /**
   * Begin a recording. config: { title, bpm, bars, beatsPerBar = 4, countInBeats = 4,
   * synth = 'hiphop', mirror = false, audioUrl = null, audioOffsetSec = 0, audioGain = 0.8,
   * referenceHeight = STAGE.referenceHeight, id (optional), author = '', notes = '',
   * beatsPerMove = 8, difficulty = 2 }
   */
  start(config = {}) {
    const bpm = Number(config.bpm);
    if (!(bpm > 0)) throw new RangeError('bpm must be > 0');
    const beatsPerBar = config.beatsPerBar > 0 ? Math.floor(config.beatsPerBar) : 4;
    const bars = config.bars > 0 ? config.bars : 4;
    const durationBeats = config.durationBeats > 0 ? config.durationBeats : bars * beatsPerBar;
    const title = (config.title && String(config.title).trim()) || 'Eigener Tanz';
    this.config = {
      id: config.id && /^[a-z0-9-]+$/.test(config.id) ? config.id : Recorder.makeId(title),
      title,
      artist: config.artist || '',
      bpm,
      beatsPerBar,
      countInBeats: config.countInBeats >= 0 ? Math.floor(config.countInBeats) : 4,
      durationBeats,
      referenceHeight: config.referenceHeight > 0 ? config.referenceHeight : STAGE.referenceHeight,
      mirror: !!config.mirror,
      synth: config.synth || 'hiphop',
      audioUrl: config.audioUrl || null,
      audioOffsetSec: typeof config.audioOffsetSec === 'number' ? config.audioOffsetSec : 0,
      audioGain: typeof config.audioGain === 'number' ? config.audioGain : 0.8,
      author: config.author || '',
      notes: config.notes || '',
      beatsPerMove: config.beatsPerMove > 0 ? config.beatsPerMove : 8,
      difficulty: config.difficulty >= 1 && config.difficulty <= 5 ? config.difficulty : 2,
    };
    this.frameCount = expectedFrameCount({ durationBeats, bpm, fps: this.fps });
    this.head = new Float32Array(this.frameCount * 7);
    this.left = new Float32Array(this.frameCount * 3);
    this.right = new Float32Array(this.frameCount * 3);
    this.filled = new Uint8Array(this.frameCount);
    this.lastIndex = -1;
    this.sampleCount = 0;
    this._prevT = NaN;
    this.startedAt = new Date().toISOString();
    this.state = 'recording';
    return this;
  }

  get durationSec() {
    return this.config ? this.config.durationBeats * 60 / this.config.bpm : 0;
  }

  /** 0..1 fraction of frames recorded so far. */
  get progress() {
    if (!this.config || this.frameCount === 0) return 0;
    return Math.min(1, (this.lastIndex + 1) / this.frameCount);
  }

  /** True once the last frame has been recorded. */
  get complete() {
    return this.state !== 'idle' && this.lastIndex >= this.frameCount - 1;
  }

  /**
   * Store a stage-frame sample. `now` is either the game time in seconds (since beat 0) or a
   * BeatClock state object with `t`. Samples arrive at the render rate (72 Hz); every 30 Hz
   * frame whose time lies between the previous and this sample is written by linear
   * interpolation (nlerp for the head quaternion) at its exact frame time. Frames bracketed by
   * samples more than 2.5 frames apart are marked as gap-filled. Samples before beat 0 only
   * prime the interpolation. Returns the highest frame index written, or -1.
   */
  update(now, sampleS) {
    if (this.state !== 'recording' || !sampleS) return -1;
    const t = typeof now === 'number' ? now : (now && typeof now.t === 'number' ? now.t : NaN);
    if (!Number.isFinite(t)) return -1;
    const cur = this._cur;
    packSample(cur, sampleS);
    const prev = this._prev;
    const tPrev = this._prevT;
    let written = -1;
    if (t >= 0) {
      const dt = this.fps > 0 ? 1 / this.fps : 0;
      const gap = Number.isFinite(tPrev) && (t - tPrev) > 2.5 * dt;
      let idx = this.lastIndex + 1;
      while (idx < this.frameCount) {
        const ft = idx * dt;
        if (ft > t + 1e-9) break;
        if (Number.isFinite(tPrev) && tPrev < t && ft >= tPrev) {
          const a = (ft - tPrev) / (t - tPrev);
          this._writeLerp(idx, prev, cur, a);
        } else {
          this._writePacked(idx, cur);
        }
        this.filled[idx] = gap ? 2 : 1;
        this.lastIndex = idx;
        written = idx;
        idx++;
      }
      this.sampleCount++;
    }
    // remember this sample for the next interpolation (even during the count-in)
    if (!(Number.isFinite(tPrev) && t < tPrev)) {
      prev.set(cur);
      this._prevT = t;
    }
    return written;
  }

  _writePacked(i, s) {
    const o7 = i * 7, o3 = i * 3;
    for (let k = 0; k < 7; k++) this.head[o7 + k] = s[k];
    for (let k = 0; k < 3; k++) { this.left[o3 + k] = s[7 + k]; this.right[o3 + k] = s[10 + k]; }
  }

  _writeLerp(i, a, b, w) {
    const o7 = i * 7, o3 = i * 3;
    for (let k = 0; k < 3; k++) this.head[o7 + k] = a[k] + (b[k] - a[k]) * w;
    // nlerp with hemisphere check
    const dotq = a[3] * b[3] + a[4] * b[4] + a[5] * b[5] + a[6] * b[6];
    const sgn = dotq < 0 ? -1 : 1;
    let qx = a[3] + (sgn * b[3] - a[3]) * w;
    let qy = a[4] + (sgn * b[4] - a[4]) * w;
    let qz = a[5] + (sgn * b[5] - a[5]) * w;
    let qw = a[6] + (sgn * b[6] - a[6]) * w;
    const l = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (l > 1e-9) { qx /= l; qy /= l; qz /= l; qw /= l; } else { qx = 0; qy = 0; qz = 0; qw = 1; }
    this.head[o7 + 3] = qx; this.head[o7 + 4] = qy; this.head[o7 + 5] = qz; this.head[o7 + 6] = qw;
    for (let k = 0; k < 3; k++) {
      this.left[o3 + k] = a[7 + k] + (b[7 + k] - a[7 + k]) * w;
      this.right[o3 + k] = a[10 + k] + (b[10 + k] - a[10 + k]) * w;
    }
  }

  _copyFrame(dst, src) {
    for (let k = 0; k < 7; k++) this.head[dst * 7 + k] = this.head[src * 7 + k];
    for (let k = 0; k < 3; k++) {
      this.left[dst * 3 + k] = this.left[src * 3 + k];
      this.right[dst * 3 + k] = this.right[src * 3 + k];
    }
  }

  /** Abort without producing a choreography. */
  cancel() {
    this.state = 'idle';
  }

  /**
   * Finish and build the choreography JSON (trailing gaps are hold-filled). Throws when no
   * frame was recorded or the result does not validate.
   */
  stop() {
    if (this.state !== 'recording') throw new Error('recorder is not recording');
    if (this.lastIndex < 0) {
      this.state = 'idle';
      throw new Error('no frames recorded');
    }
    for (let i = this.lastIndex + 1; i < this.frameCount; i++) {
      this._copyFrame(i, this.lastIndex);
      this.filled[i] = 2;
    }
    this.state = 'stopped';
    const c = this.config;
    const n = this.frameCount;
    const head = new Array(n), left = new Array(n), right = new Array(n);
    for (let i = 0; i < n; i++) {
      const o7 = i * 7, o3 = i * 3;
      head[i] = [round4(this.head[o7]), round4(this.head[o7 + 1]), round4(this.head[o7 + 2]),
        round6(this.head[o7 + 3]), round6(this.head[o7 + 4]), round6(this.head[o7 + 5]), round6(this.head[o7 + 6])];
      left[i] = [round4(this.left[o3]), round4(this.left[o3 + 1]), round4(this.left[o3 + 2])];
      right[i] = [round4(this.right[o3]), round4(this.right[o3 + 1]), round4(this.right[o3 + 2])];
    }
    const json = {
      format: CHOREO_FORMAT,
      id: c.id,
      title: c.title,
      artist: c.artist,
      bpm: c.bpm,
      beatsPerBar: c.beatsPerBar,
      countInBeats: c.countInBeats,
      durationBeats: c.durationBeats,
      fps: this.fps,
      referenceHeight: c.referenceHeight,
      mirror: c.mirror,
      difficulty: c.difficulty,
      audio: { url: c.audioUrl, synth: c.synth, offsetSec: c.audioOffsetSec, gain: c.audioGain },
      moves: Recorder.defaultMoves(c.durationBeats, c.beatsPerMove),
      frames: { head, left, right },
      meta: {
        source: 'recorded',
        createdAt: this.startedAt || new Date().toISOString(),
        author: c.author,
        notes: c.notes,
        sampledFrames: this.sampleCount,
        holdFilledFrames: countFilled(this.filled, 2),
      },
    };
    const v = validateChoreo(json);
    if (!v.ok) throw new Error(`recorded choreography invalid: ${v.errors.join('; ')}`);
    return json;
  }

  // ---- static helpers ----------------------------------------------------------------------

  /** One move per `beatsPerMove` beats named "Teil 1..n" (last one may be shorter). */
  static defaultMoves(durationBeats, beatsPerMove = 8) {
    const moves = [];
    let b = 0, i = 1;
    while (b < durationBeats - 1e-9) {
      const end = Math.min(durationBeats, b + beatsPerMove);
      moves.push({ name: `Teil ${i}`, startBeat: b, endBeat: end, hint: '' });
      b = end;
      i++;
    }
    return moves;
  }

  /** Slug from a title plus a short time stamp so that ids are unique. */
  static makeId(title, now = Date.now()) {
    let slug = String(title || 'tanz').toLowerCase()
      .replace(/ä/g, 'ae').replace(/ö/g, 'oe').replace(/ü/g, 'ue').replace(/ß/g, 'ss')
      .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
    if (!slug) slug = 'tanz';
    if (slug.length > 32) slug = slug.slice(0, 32).replace(/-+$/g, '');
    return `${slug}-${now.toString(36)}`;
  }

  static toJSONString(json, pretty = false) {
    return pretty ? JSON.stringify(json, null, 1) : JSON.stringify(json);
  }

  static filename(json) {
    return `${json.id}.json`;
  }

  /** Keep a copy in storage[STORAGE_KEYS.choreos] (a map id -> choreo). Returns true on success. */
  static persist(json, storage, key = STORAGE_KEYS.choreos) {
    if (!storage || !json || !json.id) return false;
    try {
      const map = Recorder._readMap(storage, key);
      map[json.id] = json;
      storage.setItem(key, JSON.stringify(map));
      return true;
    } catch (e) {
      if (typeof console !== 'undefined') console.warn('[recorder] persist failed', e);
      return false;
    }
  }

  /** All persisted choreographies (array of JSON objects). */
  static listPersisted(storage, key = STORAGE_KEYS.choreos) {
    if (!storage) return [];
    try {
      const map = Recorder._readMap(storage, key);
      return Object.keys(map).map((id) => map[id]).filter((c) => c && validateChoreo(c).ok);
    } catch (e) {
      return [];
    }
  }

  static removePersisted(id, storage, key = STORAGE_KEYS.choreos) {
    if (!storage) return false;
    try {
      const map = Recorder._readMap(storage, key);
      if (!(id in map)) return false;
      delete map[id];
      storage.setItem(key, JSON.stringify(map));
      return true;
    } catch (e) {
      return false;
    }
  }

  /** Read the id -> choreo map; a missing or corrupt store reads as empty (and gets replaced). */
  static _readMap(storage, key) {
    const raw = storage.getItem(key);
    if (!raw) return {};
    try {
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
    } catch (e) {
      return {};
    }
  }

  /** Trigger a browser download of the JSON. Returns the file name, or null outside a browser. */
  static download(json, doc = globalThis.document) {
    if (!doc || typeof doc.createElement !== 'function' || typeof globalThis.Blob !== 'function' || !globalThis.URL || typeof globalThis.URL.createObjectURL !== 'function') {
      return null;
    }
    const name = Recorder.filename(json);
    const blob = new globalThis.Blob([Recorder.toJSONString(json)], { type: 'application/json' });
    const url = globalThis.URL.createObjectURL(blob);
    const a = doc.createElement('a');
    a.href = url;
    a.download = name;
    (doc.body || doc.documentElement).appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => globalThis.URL.revokeObjectURL(url), 1000);
    return name;
  }

  /** POST to `${baseUrl}/api/choreos`. Resolves to the parsed response; rejects on failure. */
  static async upload(json, baseUrl = '', fetchImpl = globalThis.fetch) {
    if (typeof fetchImpl !== 'function') throw new Error('no fetch available');
    const res = await fetchImpl(`${baseUrl.replace(/\/$/, '')}/api/choreos`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: Recorder.toJSONString(json),
    });
    if (!res.ok) throw new Error(`upload failed: HTTP ${res.status}`);
    try {
      return await res.json();
    } catch (e) {
      return { ok: true };
    }
  }
}

/** Pack a stage-frame sample into 13 floats: head p, head q (identity if absent), left p, right p. */
function packSample(out, s) {
  const h = s.head;
  out[0] = h.p[0]; out[1] = h.p[1]; out[2] = h.p[2];
  if (h.q) { out[3] = h.q[0]; out[4] = h.q[1]; out[5] = h.q[2]; out[6] = h.q[3]; }
  else { out[3] = 0; out[4] = 0; out[5] = 0; out[6] = 1; }
  out[7] = s.left.p[0]; out[8] = s.left.p[1]; out[9] = s.left.p[2];
  out[10] = s.right.p[0]; out[11] = s.right.p[1]; out[12] = s.right.p[2];
  return out;
}

function countFilled(arr, value) {
  let n = 0;
  for (let i = 0; i < arr.length; i++) if (arr[i] === value) n++;
  return n;
}
