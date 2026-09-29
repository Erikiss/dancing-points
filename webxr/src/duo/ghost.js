// ghost.js - GhostDuo: replay of a saved 3-point run (DESIGN.md section 9, "ghost duo").
// A run is the stage-frame stream of one play (`dancing-points-run/1` from
// PlaySession.runToJSON(): head[][7] / left[][3] / right[][3], or the compact
// {frames: [[hx,hy,hz, lx,ly,lz, rx,ry,rz], ...]} layout). This module normalises both into
// typed arrays, interpolates a sample for any time t (no allocation in sampleAt), and offers
// helpers to save/load runs in a localStorage-like store or on the LAN server (/api/runs).
// Pure JS: importable in Node, no three.js, no DOM.

import { slerp, quatNormalize, lerp3At } from '../util/math.js';

export const RUN_FORMAT = 'dancing-points-run/1';
export const RUN_STORAGE_KEY = 'dp.runs';
export const DEFAULT_MAX_RUNS_PER_CHOREO = 5;   // localStorage is small; a 32 s run is ~100 kB
export const DEFAULT_MAX_RUNS_TOTAL = 20;
export const MAX_RUN_FRAMES = 20000;
const ID_PATTERN = /^[a-z0-9-]+$/;

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

function isVec(v, n) {
  if (!Array.isArray(v) || v.length !== n) return false;
  for (let i = 0; i < n; i++) if (!isNum(v[i])) return false;
  return true;
}

function isTyped(v) {
  return v instanceof Float32Array || v instanceof Float64Array;
}

/** A sample object in the shape of Choreo.sampleAt(): {t, head:{p,q}, left:{p}, right:{p}}. */
export function createRunSample() {
  return { t: 0, head: { p: [0, 0, 0], q: [0, 0, 0, 1] }, left: { p: [0, 0, 0] }, right: { p: [0, 0, 0] } };
}

/**
 * Validate a run in JSON form (either layout) or a normalized run.
 * @returns {{ok: boolean, errors: string[], frameCount: number}}
 */
export function validateRun(run) {
  const errors = [];
  if (!run || typeof run !== 'object' || Array.isArray(run)) return { ok: false, errors: ['not an object'], frameCount: 0 };
  if (typeof run.choreoId !== 'string' || !ID_PATTERN.test(run.choreoId)) errors.push('choreoId must match [a-z0-9-]+');
  if (!isNum(run.fps) || run.fps <= 0 || run.fps > 240) errors.push('fps must be in (0, 240]');
  let n = -1;
  if (isTyped(run.head) && isTyped(run.left) && isTyped(run.right)) {
    n = Math.floor(run.head.length / 7);
    if (run.head.length !== n * 7 || run.left.length !== n * 3 || run.right.length !== n * 3) errors.push('typed arrays have inconsistent lengths');
  } else if (Array.isArray(run.frames)) {
    n = run.frames.length;
    for (let i = 0; i < n; i++) if (!isVec(run.frames[i], 9)) { errors.push(`frames[${i}] must be 9 finite numbers`); break; }
  } else if (Array.isArray(run.head) && Array.isArray(run.left) && Array.isArray(run.right)) {
    n = run.head.length;
    if (run.left.length !== n || run.right.length !== n) errors.push('head/left/right must have the same length');
    for (let i = 0; i < n; i++) if (!isVec(run.head[i], 7)) { errors.push(`head[${i}] must be 7 finite numbers`); break; }
    for (let i = 0; i < run.left.length; i++) if (!isVec(run.left[i], 3)) { errors.push(`left[${i}] must be 3 finite numbers`); break; }
    for (let i = 0; i < run.right.length; i++) if (!isVec(run.right[i], 3)) { errors.push(`right[${i}] must be 3 finite numbers`); break; }
  } else {
    errors.push('run needs frames[[9]] or head[[7]]/left[[3]]/right[[3]]');
  }
  if (n === 0) errors.push('run has no frames');
  if (n > MAX_RUN_FRAMES) errors.push(`run has ${n} frames, max ${MAX_RUN_FRAMES}`);
  if (run.frameCount !== undefined && isNum(run.frameCount) && n >= 0 && run.frameCount !== n) errors.push(`frameCount ${run.frameCount} does not match ${n} frames`);
  if (run.score !== undefined && run.score !== null && !isNum(run.score)) errors.push('score must be a number');
  if (run.player !== undefined && run.player !== null && typeof run.player !== 'string') errors.push('player must be a string');
  return { ok: errors.length === 0, errors, frameCount: Math.max(0, n) };
}

/**
 * Normalize a run into typed arrays. Idempotent (a normalized run is returned as is). Throws on
 * an invalid run (error has `.errors`).
 * @returns {{format, choreoId, fps, frameCount, head: Float32Array, left: Float32Array,
 *   right: Float32Array, hasQuat: boolean, score, player, playedAt, mode, id, duration}}
 */
export function normalizeRun(run) {
  if (run && isTyped(run.head) && run.normalized === true) return run;
  const v = validateRun(run);
  if (!v.ok) {
    const err = new Error(`invalid run: ${v.errors.join('; ')}`);
    err.errors = v.errors;
    throw err;
  }
  const n = v.frameCount;
  const head = new Float32Array(n * 7);
  const left = new Float32Array(n * 3);
  const right = new Float32Array(n * 3);
  let hasQuat = false;
  if (isTyped(run.head)) {
    head.set(run.head);
    left.set(run.left);
    right.set(run.right);
    hasQuat = run.hasQuat !== false;
  } else if (Array.isArray(run.frames)) {
    for (let i = 0; i < n; i++) {
      const f = run.frames[i];
      const o7 = i * 7, o3 = i * 3;
      head[o7] = f[0]; head[o7 + 1] = f[1]; head[o7 + 2] = f[2];
      head[o7 + 3] = 0; head[o7 + 4] = 0; head[o7 + 5] = 0; head[o7 + 6] = 1;
      left[o3] = f[3]; left[o3 + 1] = f[4]; left[o3 + 2] = f[5];
      right[o3] = f[6]; right[o3 + 1] = f[7]; right[o3 + 2] = f[8];
    }
  } else {
    hasQuat = true;
    for (let i = 0; i < n; i++) {
      const h = run.head[i], l = run.left[i], r = run.right[i];
      const o7 = i * 7, o3 = i * 3;
      for (let k = 0; k < 7; k++) head[o7 + k] = h[k];
      left[o3] = l[0]; left[o3 + 1] = l[1]; left[o3 + 2] = l[2];
      right[o3] = r[0]; right[o3 + 1] = r[1]; right[o3 + 2] = r[2];
    }
  }
  return {
    normalized: true,
    format: RUN_FORMAT,
    choreoId: run.choreoId,
    fps: run.fps,
    frameCount: n,
    duration: n > 0 ? (n - 1) / run.fps : 0,
    head,
    left,
    right,
    hasQuat,
    score: isNum(run.score) ? run.score : null,
    player: typeof run.player === 'string' ? run.player : '',
    playedAt: typeof run.playedAt === 'string' ? run.playedAt : null,
    mode: typeof run.mode === 'string' ? run.mode : 'solo',
    id: typeof run.id === 'string' ? run.id : null,
  };
}

/**
 * Convert a (normalized or JSON) run back to JSON. `compact` writes the 9-number frames layout
 * (drops the head quaternion), otherwise `dancing-points-run/1`.
 */
export function runToJSON(run, { compact = false, digits = 4 } = {}) {
  const r = normalizeRun(run);
  const f = Math.pow(10, digits);
  const rd = (x) => Math.round(x * f) / f;
  const n = r.frameCount;
  const out = {
    format: RUN_FORMAT,
    choreoId: r.choreoId,
    fps: r.fps,
    frameCount: n,
  };
  if (compact) {
    const frames = new Array(n);
    for (let i = 0; i < n; i++) {
      const o7 = i * 7, o3 = i * 3;
      frames[i] = [rd(r.head[o7]), rd(r.head[o7 + 1]), rd(r.head[o7 + 2]),
        rd(r.left[o3]), rd(r.left[o3 + 1]), rd(r.left[o3 + 2]),
        rd(r.right[o3]), rd(r.right[o3 + 1]), rd(r.right[o3 + 2])];
    }
    out.frames = frames;
  } else {
    const head = new Array(n), left = new Array(n), right = new Array(n);
    for (let i = 0; i < n; i++) {
      const o7 = i * 7, o3 = i * 3;
      head[i] = [rd(r.head[o7]), rd(r.head[o7 + 1]), rd(r.head[o7 + 2]), rd(r.head[o7 + 3]), rd(r.head[o7 + 4]), rd(r.head[o7 + 5]), rd(r.head[o7 + 6])];
      left[i] = [rd(r.left[o3]), rd(r.left[o3 + 1]), rd(r.left[o3 + 2])];
      right[i] = [rd(r.right[o3]), rd(r.right[o3 + 1]), rd(r.right[o3 + 2])];
    }
    out.head = head;
    out.left = left;
    out.right = right;
  }
  out.score = r.score;
  out.playedAt = r.playedAt;
  out.player = r.player;
  out.mode = r.mode;
  if (r.id) out.id = r.id;
  return out;
}

/**
 * Write the interpolated 3-point pose of a normalized run at time t (seconds since beat 0)
 * into flat output arrays (out*[off..off+2]). Clamped at both ends. No allocation.
 */
export function sampleRunInto(run, t, outHead, oh, outLeft, ol, outRight, or) {
  const n = run.frameCount;
  if (n === 0) return false;
  const x = t * run.fps;
  let i = Math.floor(x);
  let f = x - i;
  if (i < 0) { i = 0; f = 0; }
  if (i >= n - 1) { i = n - 1; f = 0; }
  const j = f > 0 ? i + 1 : i;
  outHead[oh] = run.head[i * 7] + (run.head[j * 7] - run.head[i * 7]) * f;
  outHead[oh + 1] = run.head[i * 7 + 1] + (run.head[j * 7 + 1] - run.head[i * 7 + 1]) * f;
  outHead[oh + 2] = run.head[i * 7 + 2] + (run.head[j * 7 + 2] - run.head[i * 7 + 2]) * f;
  outLeft[ol] = run.left[i * 3] + (run.left[j * 3] - run.left[i * 3]) * f;
  outLeft[ol + 1] = run.left[i * 3 + 1] + (run.left[j * 3 + 1] - run.left[i * 3 + 1]) * f;
  outLeft[ol + 2] = run.left[i * 3 + 2] + (run.left[j * 3 + 2] - run.left[i * 3 + 2]) * f;
  outRight[or] = run.right[i * 3] + (run.right[j * 3] - run.right[i * 3]) * f;
  outRight[or + 1] = run.right[i * 3 + 1] + (run.right[j * 3 + 1] - run.right[i * 3 + 1]) * f;
  outRight[or + 2] = run.right[i * 3 + 2] + (run.right[j * 3 + 2] - run.right[i * 3 + 2]) * f;
  return true;
}

/**
 * GhostDuo - the second dancer as a saved run of the same choreography.
 * `sampleAt(t)` returns a sample in the shape of `Choreo.sampleAt` so the renderer can pose a
 * PointAvatar/BodyAvatar exactly as for the teacher (translucent, 1.5 m to the side).
 */
export class GhostDuo {
  /**
   * @param {object} run a run in JSON form (either layout) or a normalized run
   * @param {object} [opts] { name = run.player || 'Aufzeichnung' }
   */
  constructor(run, opts = {}) {
    this.run = normalizeRun(run);
    this.choreoId = this.run.choreoId;
    this.fps = this.run.fps;
    this.frameCount = this.run.frameCount;
    this.duration = this.run.duration;
    this.score = this.run.score;
    this.player = this.run.player;
    this.playedAt = this.run.playedAt;
    this.hasQuat = this.run.hasQuat;
    this.name = opts.name || this.run.player || 'Aufzeichnung';
    this._sample = createRunSample();
    this._qa = [0, 0, 0, 1];
    this._qb = [0, 0, 0, 1];
  }

  /** Frame index (floor) for time t, clamped. */
  frameIndexAt(t) {
    const i = Math.floor(t * this.fps);
    return i < 0 ? 0 : i >= this.frameCount ? this.frameCount - 1 : i;
  }

  /**
   * Interpolated sample at time t (seconds since beat 0): lerp for positions, slerp for the head
   * quaternion, clamped at both ends. Writes into `out` (default: a reused internal object).
   */
  sampleAt(t, out = this._sample) {
    const run = this.run;
    const n = run.frameCount;
    out.t = t;
    if (n === 0) return out;
    const x = t * run.fps;
    let i = Math.floor(x);
    let f = x - i;
    if (i < 0) { i = 0; f = 0; }
    if (i >= n - 1) { i = n - 1; f = 0; }
    const j = f > 0 ? i + 1 : i;
    lerp3At(out.head.p, run.head, i * 7, run.head, j * 7, f);
    lerp3At(out.left.p, run.left, i * 3, run.left, j * 3, f);
    lerp3At(out.right.p, run.right, i * 3, run.right, j * 3, f);
    const q = out.head.q;
    if (!run.hasQuat) {
      q[0] = 0; q[1] = 0; q[2] = 0; q[3] = 1;
      return out;
    }
    const qa = this._qa, qb = this._qb;
    const oa = i * 7 + 3, ob = j * 7 + 3;
    qa[0] = run.head[oa]; qa[1] = run.head[oa + 1]; qa[2] = run.head[oa + 2]; qa[3] = run.head[oa + 3];
    if (f > 0) {
      qb[0] = run.head[ob]; qb[1] = run.head[ob + 1]; qb[2] = run.head[ob + 2]; qb[3] = run.head[ob + 3];
      slerp(q, qa, qb, f);
    } else {
      q[0] = qa[0]; q[1] = qa[1]; q[2] = qa[2]; q[3] = qa[3];
    }
    quatNormalize(q, q);
    return out;
  }

  /** Head speed (m/s) at time t from the bracketing frames (0 at the ends). */
  headSpeedAt(t) {
    const run = this.run;
    const n = run.frameCount;
    if (n < 2) return 0;
    let i = Math.floor(t * run.fps);
    if (i < 0) i = 0;
    if (i >= n - 1) i = n - 2;
    const a = i * 7, b = (i + 1) * 7;
    const dx = run.head[b] - run.head[a], dy = run.head[b + 1] - run.head[a + 1], dz = run.head[b + 2] - run.head[a + 2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz) * run.fps;
  }

  toJSON(opts) {
    return runToJSON(this.run, opts);
  }
}

// ---------------------------------------------------------------------------------------------
// Storage helpers (localStorage-like: getItem / setItem / removeItem)

function readStore(storage, key) {
  try {
    const raw = storage.getItem(key);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    const list = Array.isArray(parsed) ? parsed : parsed && Array.isArray(parsed.runs) ? parsed.runs : [];
    return list.filter((r) => validateRun(r).ok);
  } catch (e) {
    return [];
  }
}

function writeStore(storage, key, list) {
  try {
    storage.setItem(key, JSON.stringify(list));
    return true;
  } catch (e) {
    return false;
  }
}

/**
 * Append a run to the store (oldest first). Keeps at most `maxPerChoreo` runs per choreography
 * and `maxTotal` overall (oldest dropped). On a quota error the oldest runs are dropped and the
 * write retried once. Returns true when stored.
 */
export function saveRun(run, storage, { key = RUN_STORAGE_KEY, maxPerChoreo = DEFAULT_MAX_RUNS_PER_CHOREO, maxTotal = DEFAULT_MAX_RUNS_TOTAL } = {}) {
  if (!storage || typeof storage.setItem !== 'function') return false;
  const json = run && run.normalized ? runToJSON(run) : run;
  if (!validateRun(json).ok) return false;
  if (typeof json.playedAt !== 'string') json.playedAt = new Date().toISOString();
  let list = readStore(storage, key);
  list.push(json);
  list = capRuns(list, maxPerChoreo, maxTotal);
  if (writeStore(storage, key, list)) return true;
  // quota: drop the oldest half (but keep the new one) and retry once
  const keep = Math.max(1, Math.floor(list.length / 2));
  list = list.slice(list.length - keep);
  return writeStore(storage, key, list);
}

/** Apply the per-choreo / total caps to an oldest-first list (returns a new list). */
export function capRuns(list, maxPerChoreo = DEFAULT_MAX_RUNS_PER_CHOREO, maxTotal = DEFAULT_MAX_RUNS_TOTAL) {
  const counts = new Map();
  const kept = [];
  for (let i = list.length - 1; i >= 0; i--) {
    const r = list[i];
    const c = counts.get(r.choreoId) || 0;
    if (c >= maxPerChoreo) continue;
    counts.set(r.choreoId, c + 1);
    kept.push(r);
    if (kept.length >= maxTotal) break;
  }
  kept.reverse();
  return kept;
}

/** Runs from the store, newest first (optionally only for one choreography). */
export function loadRuns(storage, { key = RUN_STORAGE_KEY, choreoId = null } = {}) {
  if (!storage || typeof storage.getItem !== 'function') return [];
  const list = readStore(storage, key);
  const out = [];
  for (let i = list.length - 1; i >= 0; i--) if (!choreoId || list[i].choreoId === choreoId) out.push(list[i]);
  return out;
}

/** Remove all runs (or those of one choreography) from the store. */
export function clearRuns(storage, { key = RUN_STORAGE_KEY, choreoId = null } = {}) {
  if (!storage) return false;
  if (!choreoId) {
    try { storage.removeItem(key); } catch (e) { return false; }
    return true;
  }
  const list = readStore(storage, key).filter((r) => r.choreoId !== choreoId);
  return writeStore(storage, key, list);
}

/**
 * Pick one run of a list: 'latest' (default, by playedAt then order), 'best' (highest score),
 * or 'closest' to a target score (opts.score).
 */
export function pickRun(runs, { choreoId = null, strategy = 'latest', score = null } = {}) {
  let best = null;
  let bestKey = -Infinity;
  for (let i = 0; i < runs.length; i++) {
    const r = runs[i];
    if (!r || (choreoId && r.choreoId !== choreoId)) continue;
    let key;
    if (strategy === 'best') key = isNum(r.score) ? r.score : -1;
    else if (strategy === 'closest') key = isNum(r.score) && isNum(score) ? -Math.abs(r.score - score) : -1e9;
    else key = typeof r.playedAt === 'string' ? Date.parse(r.playedAt) || 0 : 0;
    if (key > bestKey) { bestKey = key; best = r; }
  }
  return best;
}

// ---------------------------------------------------------------------------------------------
// Server helpers (/api/runs); baseUrl '' = same origin, or 'https://host:port' (a `?server=`
// value like 'wss://host:port' is converted).

export function toHttpBase(url) {
  if (!url) return '';
  let s = String(url).trim();
  if (/^wss:\/\//i.test(s)) s = `https://${s.slice(6)}`;
  else if (/^ws:\/\//i.test(s)) s = `http://${s.slice(5)}`;
  s = s.replace(/\/ws\/?$/i, '');
  return s.replace(/\/+$/, '');
}

async function checkResponse(res, what) {
  if (!res.ok) {
    let detail = '';
    try { detail = (await res.json()).error || ''; } catch (e) { /* ignore */ }
    throw new Error(`${what}: HTTP ${res.status}${detail ? ` (${detail})` : ''}`);
  }
  return res.json();
}

/** POST a run (JSON or normalized) to the server; resolves with the server's reply {ok,id,url,run}. */
export async function uploadRun(run, baseUrl = '', fetchImpl = globalThis.fetch) {
  if (typeof fetchImpl !== 'function') throw new Error('fetch not available');
  const json = run && run.normalized ? runToJSON(run) : run;
  const res = await fetchImpl(`${toHttpBase(baseUrl)}/api/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(json),
  });
  return checkResponse(res, 'upload run');
}

/** Run metadata list from the server (newest first): [{id, choreoId, player, score, playedAt, url, ...}]. */
export async function fetchRunList(baseUrl = '', { choreoId = null, limit = 50, fetchImpl = globalThis.fetch } = {}) {
  if (typeof fetchImpl !== 'function') throw new Error('fetch not available');
  const q = new URLSearchParams();
  if (choreoId) q.set('choreoId', choreoId);
  if (limit) q.set('limit', String(limit));
  const res = await fetchImpl(`${toHttpBase(baseUrl)}/api/runs?${q.toString()}`);
  const data = await checkResponse(res, 'list runs');
  return Array.isArray(data.runs) ? data.runs : [];
}

/** Fetch one full run by id (or by its `url` from the list). */
export async function fetchRun(baseUrl = '', idOrUrl, fetchImpl = globalThis.fetch) {
  if (typeof fetchImpl !== 'function') throw new Error('fetch not available');
  const path = /^\//.test(idOrUrl) ? idOrUrl : `/api/runs/${idOrUrl}`;
  const res = await fetchImpl(`${toHttpBase(baseUrl)}${path}`);
  return checkResponse(res, 'fetch run');
}

/**
 * Find a ghost for a choreography: the server first (when `baseUrl` is given and reachable),
 * then the local store. Resolves with a GhostDuo or null when nothing is available.
 */
export async function loadGhost({ choreoId, storage = null, baseUrl = null, fetchImpl = globalThis.fetch, strategy = 'latest', score = null } = {}) {
  if (baseUrl !== null && baseUrl !== undefined && typeof fetchImpl === 'function') {
    try {
      const list = await fetchRunList(baseUrl, { choreoId, fetchImpl });
      const meta = pickRun(list, { choreoId, strategy, score });
      if (meta) {
        const run = await fetchRun(baseUrl, meta.url || meta.id, fetchImpl);
        return new GhostDuo(run);
      }
    } catch (e) {
      if (typeof console !== 'undefined' && console.warn) console.warn('[ghost] server runs unavailable:', e.message);
    }
  }
  if (storage) {
    const run = pickRun(loadRuns(storage, { choreoId }), { choreoId, strategy, score });
    if (run) return new GhostDuo(run);
  }
  return null;
}
