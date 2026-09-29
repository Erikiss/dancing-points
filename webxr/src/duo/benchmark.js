// benchmark.js - duo benchmark metrics and export (DESIGN.md section 9). Pure, deterministic
// functions over Result objects (src/game/scoring.js) and saved runs (src/duo/ghost.js):
// per-player summaries, pair metrics (syncDistance = mean 3-point distance after offset
// removal, syncLag = cross-correlation peak of the head speed signals), winner, the benchmark
// document, JSON/CSV export and leaderboard helpers (top N per choreography). Called once per
// results screen, so it may allocate. No three.js, no DOM, no network.

import { normalizeRun, sampleRunInto } from './ghost.js';

export const BENCHMARK_FORMAT = 'dancing-points-benchmark/1';
export const DEFAULT_MAX_LAG_SEC = 1.0;
export const DEFAULT_LAG_PENALTY = 0.1;   // correlation taper at the window edge (prefers small lags)
export const DEFAULT_TOP = 10;

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const round = (x, d = 3) => (isNum(x) ? Math.round(x * Math.pow(10, d)) / Math.pow(10, d) : x);

// ---------------------------------------------------------------------------------------------
// Sampling both runs onto a common time grid

/**
 * Sample two runs on a common grid (the overlap of both durations at `fps`). Returns flat
 * Float32Arrays (n*3 each) for head/left/right of A and B. `lagSec` shifts B in time
 * (B is read at t + lagSec, i.e. a positive lag "pulls B forward" to compensate B being late).
 * `mirror` mirrors B (x -> -x, hands swapped). `removeOffset` subtracts the mean head offset
 * between B and A from all B points (players stand on slightly different spots).
 */
export function alignRuns(runA, runB, { fps = null, lagSec = 0, mirror = false, removeOffset = true } = {}) {
  const a = normalizeRun(runA);
  const b = normalizeRun(runB);
  const rate = fps || a.fps;
  const tMax = Math.min(a.duration, b.duration - lagSec);
  const n = tMax >= 0 ? Math.floor(tMax * rate + 1e-6) + 1 : 0;
  const A = { head: new Float32Array(n * 3), left: new Float32Array(n * 3), right: new Float32Array(n * 3) };
  const B = { head: new Float32Array(n * 3), left: new Float32Array(n * 3), right: new Float32Array(n * 3) };
  const tmpL = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const t = i / rate;
    const o = i * 3;
    sampleRunInto(a, t, A.head, o, A.left, o, A.right, o);
    if (mirror) {
      // B's left hand becomes the right one (and vice versa); x is negated
      sampleRunInto(b, t + lagSec, B.head, o, tmpL, 0, B.left, o);
      B.right[o] = -tmpL[0]; B.right[o + 1] = tmpL[1]; B.right[o + 2] = tmpL[2];
      B.head[o] = -B.head[o];
      B.left[o] = -B.left[o];
    } else {
      sampleRunInto(b, t + lagSec, B.head, o, B.left, o, B.right, o);
    }
  }
  const offset = [0, 0, 0];
  if (removeOffset && n > 0) {
    for (let i = 0; i < n; i++) {
      const o = i * 3;
      offset[0] += B.head[o] - A.head[o];
      offset[1] += B.head[o + 1] - A.head[o + 1];
      offset[2] += B.head[o + 2] - A.head[o + 2];
    }
    offset[0] /= n; offset[1] /= n; offset[2] /= n;
    for (let i = 0; i < n; i++) {
      const o = i * 3;
      for (const arr of [B.head, B.left, B.right]) {
        arr[o] -= offset[0]; arr[o + 1] -= offset[1]; arr[o + 2] -= offset[2];
      }
    }
  }
  return { n, fps: rate, A, B, offset, lagSec, mirror };
}

function meanDistance(P, Q, n) {
  let s = 0;
  for (let i = 0; i < n; i++) {
    const o = i * 3;
    const dx = P[o] - Q[o], dy = P[o + 1] - Q[o + 1], dz = P[o + 2] - Q[o + 2];
    s += Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return n > 0 ? s / n : NaN;
}

// ---------------------------------------------------------------------------------------------
// Pair metrics

/**
 * Mean 3-point distance (metres) between the two players over the common part of the run,
 * broken down per point. Options: see alignRuns. NaN when the runs do not overlap.
 */
export function syncDistanceDetails(runA, runB, opts = {}) {
  const al = alignRuns(runA, runB, opts);
  const head = meanDistance(al.A.head, al.B.head, al.n);
  const left = meanDistance(al.A.left, al.B.left, al.n);
  const right = meanDistance(al.A.right, al.B.right, al.n);
  return {
    mean: al.n > 0 ? (head + left + right) / 3 : NaN,
    head,
    left,
    right,
    frames: al.n,
    durationSec: al.n > 0 ? (al.n - 1) / al.fps : 0,
    offset: al.offset,
    lagSec: al.lagSec,
    mirror: al.mirror,
  };
}

/** Mean 3-point distance between the two players in metres (after offset removal). */
export function syncDistance(runA, runB, opts = {}) {
  return syncDistanceDetails(runA, runB, opts).mean;
}

/** Head speed signal (m/s) of a run resampled at `fps` over `n` samples: speed[i] = |p(i+1) - p(i)| * fps. */
export function headSpeedSignal(run, fps, n) {
  const r = normalizeRun(run);
  const out = new Float32Array(n);
  const p0 = [0, 0, 0], p1 = [0, 0, 0], dummy = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const t0 = i / fps, t1 = (i + 1) / fps;
    sampleRunInto(r, t0, p0, 0, dummy, 0, dummy, 0);
    sampleRunInto(r, t1, p1, 0, dummy, 0, dummy, 0);
    const dx = p1[0] - p0[0], dy = p1[1] - p0[1], dz = p1[2] - p0[2];
    out[i] = Math.sqrt(dx * dx + dy * dy + dz * dz) * fps;
  }
  return out;
}

function standardize(x) {
  const n = x.length;
  if (n === 0) return { z: new Float32Array(0), ok: false };
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - mean) * (x[i] - mean);
  const sd = Math.sqrt(v / n);
  const z = new Float32Array(n);
  if (sd < 1e-9) return { z, ok: false };
  for (let i = 0; i < n; i++) z[i] = (x[i] - mean) / sd;
  return { z, ok: true };
}

/**
 * Cross-correlation of the two head speed signals. Returns
 * `{ lag, correlation, lagFrames, fps, maxLagSec, valid }` with `lag` in seconds:
 * **positive = B is behind A** (B performs the same motion `lag` seconds later; `B(t) ~ A(t - lag)`).
 * Dance motion repeats with the beat, so the correlation has near-equal peaks one period
 * apart; the peak search multiplies r(k) by `1 - lagPenalty * |k| / K` (default 10 % at the
 * window edge) to prefer the smallest lag among equal peaks. Sub-frame precision by parabolic
 * interpolation of the raw correlation around the peak. `valid` is false (lag 0, correlation
 * 0) for constant signals or when the overlap is shorter than 2 * maxLagSec.
 */
export function syncLagDetails(runA, runB, { maxLagSec = DEFAULT_MAX_LAG_SEC, fps = 30, lagPenalty = DEFAULT_LAG_PENALTY } = {}) {
  const a = normalizeRun(runA);
  const b = normalizeRun(runB);
  const tMax = Math.min(a.duration, b.duration);
  const n = Math.max(0, Math.floor(tMax * fps + 1e-6));
  const K = Math.max(1, Math.round(maxLagSec * fps));
  const invalid = { lag: 0, correlation: 0, lagFrames: 0, fps, maxLagSec, valid: false, frames: n };
  if (n < 2 * K + 2) return invalid;
  const sa = standardize(headSpeedSignal(a, fps, n));
  const sb = standardize(headSpeedSignal(b, fps, n));
  if (!sa.ok || !sb.ok) return invalid;
  const za = sa.z, zb = sb.z;
  // r(k) = mean over valid i of za[i] * zb[i + k]; k > 0 means B's signal is shifted later
  const corr = new Float64Array(2 * K + 1);
  let bestK = 0, bestScore = -Infinity;
  for (let k = -K; k <= K; k++) {
    let s = 0, m = 0;
    const iStart = Math.max(0, -k), iEnd = Math.min(n, n - k);
    for (let i = iStart; i < iEnd; i++) { s += za[i] * zb[i + k]; m++; }
    const r = m > 0 ? s / m : 0;
    corr[k + K] = r;
    const score = r * (1 - lagPenalty * Math.abs(k) / K);
    if (score > bestScore) { bestScore = score; bestK = k; }
  }
  const bestR = corr[bestK + K];
  // parabolic refinement
  let lagFrames = bestK;
  if (bestK > -K && bestK < K) {
    const y0 = corr[bestK + K - 1], y1 = corr[bestK + K], y2 = corr[bestK + K + 1];
    const denom = y0 - 2 * y1 + y2;
    if (Math.abs(denom) > 1e-12) {
      const delta = 0.5 * (y0 - y2) / denom;
      if (Math.abs(delta) <= 1) lagFrames = bestK + delta;
    }
  }
  return { lag: lagFrames / fps, correlation: Math.max(-1, Math.min(1, bestR)), lagFrames, fps, maxLagSec, valid: true, frames: n };
}

/** Lag of player B behind player A in seconds (cross-correlation peak of the head speed signals). */
export function syncLag(runA, runB, opts = {}) {
  return syncLagDetails(runA, runB, opts).lag;
}

// ---------------------------------------------------------------------------------------------
// Results

function num(v, fallback = 0) {
  return isNum(v) ? v : fallback;
}

/**
 * Best-first comparator for Result objects (or leaderboard entries): score desc, stars desc,
 * maxCombo desc, |timingBias| asc, playedAt asc (earlier keeps the rank on a full tie).
 */
export function compareResults(a, b) {
  const d = num(b.score) - num(a.score);
  if (d !== 0) return d;
  const ds = num(b.stars) - num(a.stars);
  if (ds !== 0) return ds;
  const dc = num(b.maxCombo) - num(a.maxCombo);
  if (dc !== 0) return dc;
  const dt = Math.abs(num(a.timingBias)) - Math.abs(num(b.timingBias));
  if (dt !== 0) return dt;
  const ta = typeof a.playedAt === 'string' ? Date.parse(a.playedAt) || 0 : 0;
  const tb = typeof b.playedAt === 'string' ? Date.parse(b.playedAt) || 0 : 0;
  return ta - tb;
}

/** 'A' | 'B' | 'tie' (score, then maxCombo, then the smaller |timingBias|). */
export function winner(resultA, resultB) {
  if (!resultA && !resultB) return 'tie';
  if (!resultA) return 'B';
  if (!resultB) return 'A';
  const d = num(resultB.score) - num(resultA.score);
  if (d !== 0) return d < 0 ? 'A' : 'B';
  const dc = num(resultB.maxCombo) - num(resultA.maxCombo);
  if (dc !== 0) return dc < 0 ? 'A' : 'B';
  const dt = Math.abs(num(resultA.timingBias)) - Math.abs(num(resultB.timingBias));
  if (dt !== 0) return dt < 0 ? 'A' : 'B';
  return 'tie';
}

/** Compact per-player summary of a Result for the benchmark document. */
export function playerSummary(result, name = '') {
  if (!result) return { name, score: null, stars: null, maxCombo: null, timingBias: null, durationSec: null, perMove: [], mode: null, playedAt: null };
  const moves = Array.isArray(result.moves) ? result.moves : [];
  return {
    name: name || result.player || '',
    score: num(result.score, null),
    stars: num(result.stars, null),
    maxCombo: num(result.maxCombo, null),
    timingBias: isNum(result.timingBias) ? result.timingBias : null,
    durationSec: isNum(result.durationSec) ? result.durationSec : null,
    perMove: moves.map((m) => ({ name: m.name, score: round(num(m.score), 3), grade: m.grade || null, gated: Boolean(m.gated) })),
    mode: result.mode || null,
    playedAt: result.playedAt || null,
    choreoId: result.choreoId || null,
  };
}

/**
 * Build the benchmark document for two players of the same choreography.
 * @param {object} resultA Result of player A (the local player)
 * @param {object} resultB Result of player B (ghost / online partner); may be null
 * @param {object} [runA] saved run of A (JSON or normalized) for the pair metrics
 * @param {object} [runB] saved run of B
 * @param {object} [opts] { nameA, nameB, mode='duo', mirror=false, maxLagSec, createdAt, choreoId }
 */
export function buildBenchmark(resultA, resultB, runA = null, runB = null, opts = {}) {
  const A = playerSummary(resultA, opts.nameA);
  const B = playerSummary(resultB, opts.nameB);
  const choreoId = opts.choreoId || (resultA && resultA.choreoId) || (resultB && resultB.choreoId) || (runA && runA.choreoId) || null;
  const pair = {
    syncDistance: null,
    syncDistanceAligned: null,
    syncDistanceHead: null,
    syncDistanceLeft: null,
    syncDistanceRight: null,
    syncLag: null,
    syncCorrelation: null,
    framesCompared: 0,
    scoreDiff: isNum(A.score) && isNum(B.score) ? A.score - B.score : null,
    comboDiff: isNum(A.maxCombo) && isNum(B.maxCombo) ? A.maxCombo - B.maxCombo : null,
  };
  if (runA && runB) {
    try {
      const mirror = Boolean(opts.mirror);
      const lag = syncLagDetails(runA, runB, { maxLagSec: opts.maxLagSec || DEFAULT_MAX_LAG_SEC });
      const d0 = syncDistanceDetails(runA, runB, { mirror });
      const d1 = lag.valid ? syncDistanceDetails(runA, runB, { mirror, lagSec: lag.lag }) : d0;
      pair.syncDistance = round(d0.mean, 4);
      pair.syncDistanceHead = round(d0.head, 4);
      pair.syncDistanceLeft = round(d0.left, 4);
      pair.syncDistanceRight = round(d0.right, 4);
      pair.syncDistanceAligned = round(d1.mean, 4);
      pair.syncLag = lag.valid ? round(lag.lag, 4) : null;
      pair.syncCorrelation = lag.valid ? round(lag.correlation, 4) : null;
      pair.framesCompared = d0.frames;
    } catch (e) {
      pair.error = e && e.message ? e.message : String(e);
    }
  }
  return {
    format: BENCHMARK_FORMAT,
    choreoId,
    createdAt: opts.createdAt || new Date().toISOString(),
    mode: opts.mode || 'duo',
    players: [A, B],
    pair,
    winner: winner(resultA, resultB),
  };
}

// ---------------------------------------------------------------------------------------------
// Export

export function toJSON(benchmark) {
  return JSON.stringify(benchmark, null, 2);
}

/** CSV-escape one cell. */
export function csvCell(v) {
  if (v === null || v === undefined) return '';
  const s = typeof v === 'number' ? String(v) : String(v);
  return /[",\r\n;]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/**
 * CSV export of a benchmark: `section,key,A,B` rows (meta, player, move, pair). Decimal point,
 * comma separator (`separator` option), CRLF line ends, UTF-8 without BOM.
 */
export function toCSV(benchmark, { separator = ',' } = {}) {
  const rows = [['section', 'key', 'A', 'B']];
  const [A, B] = benchmark.players || [{}, {}];
  rows.push(['meta', 'format', benchmark.format, '']);
  rows.push(['meta', 'choreoId', benchmark.choreoId, '']);
  rows.push(['meta', 'createdAt', benchmark.createdAt, '']);
  rows.push(['meta', 'mode', benchmark.mode, '']);
  for (const key of ['name', 'score', 'stars', 'maxCombo', 'timingBias', 'durationSec', 'playedAt']) {
    rows.push(['player', key, A ? A[key] : '', B ? B[key] : '']);
  }
  const movesA = (A && A.perMove) || [];
  const movesB = (B && B.perMove) || [];
  const count = Math.max(movesA.length, movesB.length);
  for (let i = 0; i < count; i++) {
    const name = (movesA[i] && movesA[i].name) || (movesB[i] && movesB[i].name) || `move ${i + 1}`;
    rows.push(['move', name, movesA[i] ? movesA[i].score : '', movesB[i] ? movesB[i].score : '']);
  }
  const pair = benchmark.pair || {};
  for (const key of Object.keys(pair)) rows.push(['pair', key, pair[key], '']);
  rows.push(['pair', 'winner', benchmark.winner, '']);
  return rows.map((r) => r.map(csvCell).join(separator)).join('\r\n') + '\r\n';
}

// ---------------------------------------------------------------------------------------------
// Leaderboard helpers (over arrays of Result / result JSON objects)

export function isResultLike(r) {
  return Boolean(r) && typeof r === 'object' && typeof r.choreoId === 'string' && isNum(r.score);
}

/** Identity key of a result (for de-duplication of local + server lists). */
export function resultKey(r) {
  return `${r.choreoId}|${r.playedAt || ''}|${r.player || ''}|${num(r.score)}|${r.mode || ''}`;
}

/** Union of several result lists without duplicates (first occurrence wins). */
export function mergeResults(...lists) {
  const seen = new Set();
  const out = [];
  for (const list of lists) {
    if (!Array.isArray(list)) continue;
    for (const r of list) {
      if (!isResultLike(r)) continue;
      const k = resultKey(r);
      if (seen.has(k)) continue;
      seen.add(k);
      out.push(r);
    }
  }
  return out;
}

/** Leaderboard entry (plain, small) from a result. */
export function toEntry(r, rank = 0) {
  return {
    rank,
    choreoId: r.choreoId,
    player: typeof r.player === 'string' ? r.player : '',
    score: num(r.score),
    stars: num(r.stars),
    maxCombo: num(r.maxCombo),
    timingBias: isNum(r.timingBias) ? r.timingBias : 0,
    playedAt: r.playedAt || null,
    mode: r.mode || 'solo',
    source: r.source || null,
  };
}

/**
 * Top `top` results of one choreography, best first, ranked 1..n. Filters: `mode`, `player`.
 * Results are de-duplicated (local + server copies of the same play count once).
 */
export function leaderboard(results, { choreoId, top = DEFAULT_TOP, mode = null, player = null } = {}) {
  const list = mergeResults(results).filter((r) => (!choreoId || r.choreoId === choreoId) && (!mode || r.mode === mode) && (!player || r.player === player));
  list.sort(compareResults);
  const n = top > 0 ? Math.min(top, list.length) : list.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = toEntry(list[i], i + 1);
  return out;
}

/** Leaderboards of every choreography present: { [choreoId]: entries }. */
export function leaderboards(results, { top = DEFAULT_TOP, mode = null } = {}) {
  const ids = new Set();
  for (const r of mergeResults(results)) ids.add(r.choreoId);
  const out = {};
  for (const id of [...ids].sort()) out[id] = leaderboard(results, { choreoId: id, top, mode });
  return out;
}

/** 1-based rank of `result` among `results` of the same choreography (-1 when it is not in the list). */
export function rankOf(results, result) {
  if (!isResultLike(result)) return -1;
  const list = mergeResults(results, [result]).filter((r) => r.choreoId === result.choreoId);
  list.sort(compareResults);
  const key = resultKey(result);
  for (let i = 0; i < list.length; i++) if (resultKey(list[i]) === key) return i + 1;
  return -1;
}

/** CSV of leaderboard entries (rank,player,score,stars,maxCombo,timingBias,playedAt,mode,choreoId). */
export function leaderboardToCSV(entries, { separator = ',' } = {}) {
  const cols = ['rank', 'player', 'score', 'stars', 'maxCombo', 'timingBias', 'playedAt', 'mode', 'choreoId'];
  const rows = [cols];
  for (const e of entries) rows.push(cols.map((c) => e[c]));
  return rows.map((r) => r.map(csvCell).join(separator)).join('\r\n') + '\r\n';
}
