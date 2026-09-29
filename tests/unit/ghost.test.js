// ghost.test.js - GhostDuo replay: run normalisation (both layouts), validation, interpolated
// sampling (positions + head quaternion), storage helpers with caps/quota handling and the
// /api/runs helpers against a fake fetch.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  GhostDuo, normalizeRun, validateRun, runToJSON, sampleRunInto, createRunSample,
  saveRun, loadRuns, clearRuns, capRuns, pickRun, toHttpBase,
  uploadRun, fetchRunList, fetchRun, loadGhost, RUN_FORMAT,
} from '../../webxr/src/duo/ghost.js';
import { quatFromYaw, yawFromQuat } from '../../webxr/src/util/math.js';
import { makeChoreo } from './helpers/make-choreo.js';

/** A run built from the frames of the synthetic choreography (exact reference playback). */
function runFromChoreo(opts = {}) {
  const c = makeChoreo(opts);
  return {
    format: RUN_FORMAT,
    choreoId: c.id,
    fps: c.fps,
    frameCount: c.frames.head.length,
    head: c.frames.head,
    left: c.frames.left,
    right: c.frames.right,
    score: opts.score ?? 90,
    playedAt: opts.playedAt || '2026-09-29T10:00:00.000Z',
    player: opts.player || 'Tester',
    mode: 'solo',
  };
}

function compactRun(n = 4, fps = 30, choreoId = 'compact') {
  const frames = [];
  for (let i = 0; i < n; i++) frames.push([i, 1.6, -i, -0.2, 1.0, i * 0.5, 0.2, 1.0, 0]);
  return { choreoId, fps, frames };
}

function makeStorage() {
  const map = new Map();
  const st = {
    calls: 0,
    failAbove: Infinity,
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => {
      st.calls++;
      if (v.length > st.failAbove) { const e = new Error('QuotaExceededError'); e.name = 'QuotaExceededError'; throw e; }
      map.set(k, String(v));
    },
    removeItem: (k) => { map.delete(k); },
    raw: (k) => map.get(k),
  };
  return st;
}

test('validateRun: both layouts accepted, broken runs rejected', () => {
  assert.equal(validateRun(runFromChoreo()).ok, true);
  assert.equal(validateRun(compactRun()).ok, true);
  assert.equal(validateRun(null).ok, false);
  assert.equal(validateRun({ choreoId: 'Bad Id', fps: 30, frames: [[0, 0, 0, 0, 0, 0, 0, 0, 0]] }).ok, false);
  assert.equal(validateRun({ choreoId: 'x', fps: 0, frames: [[0, 0, 0, 0, 0, 0, 0, 0, 0]] }).ok, false);
  assert.equal(validateRun({ choreoId: 'x', fps: 30, frames: [] }).ok, false);
  assert.equal(validateRun({ choreoId: 'x', fps: 30, frames: [[0, 0, 0]] }).ok, false);
  assert.equal(validateRun({ choreoId: 'x', fps: 30, frames: [[NaN, 0, 0, 0, 0, 0, 0, 0, 0]] }).ok, false);
  const r = runFromChoreo();
  const short = { ...r, left: r.left.slice(0, 3) };
  const v = validateRun(short);
  assert.equal(v.ok, false);
  assert.ok(v.errors.some((e) => /same length/.test(e)));
  assert.equal(validateRun({ ...r, frameCount: 3 }).ok, false);
  assert.throws(() => normalizeRun({ choreoId: 'x' }), /invalid run/);
});

test('normalizeRun: run/1 keeps quaternions, compact gets identity, idempotent', () => {
  const r = runFromChoreo();
  const n = normalizeRun(r);
  assert.equal(n.normalized, true);
  assert.equal(n.frameCount, r.head.length);
  assert.equal(n.hasQuat, true);
  assert.ok(n.head instanceof Float32Array && n.head.length === n.frameCount * 7);
  assert.ok(Math.abs(n.head[7 * 5 + 1] - r.head[5][1]) < 1e-6);
  assert.ok(Math.abs(n.right[3 * 5 + 2] - r.right[5][2]) < 1e-6);
  assert.ok(Math.abs(n.duration - (n.frameCount - 1) / 30) < 1e-9);
  assert.equal(normalizeRun(n), n);
  const c = normalizeRun(compactRun(3));
  assert.equal(c.hasQuat, false);
  assert.deepEqual(Array.from(c.head.subarray(3, 7)), [0, 0, 0, 1]);
  const l1 = Array.from(c.left.subarray(3, 6));
  assert.ok(Math.abs(l1[0] + 0.2) < 1e-6 && l1[1] === 1 && l1[2] === 0.5);
  assert.equal(c.score, null);
  assert.equal(c.player, '');
});

test('runToJSON round-trips both layouts', () => {
  const r = runFromChoreo();
  const j = runToJSON(r);
  assert.equal(j.format, RUN_FORMAT);
  assert.equal(j.frameCount, r.head.length);
  assert.deepEqual(j.head[10], r.head[10]);
  assert.deepEqual(j.left[10], r.left[10]);
  assert.equal(j.player, 'Tester');
  assert.equal(j.score, 90);
  const cj = runToJSON(r, { compact: true });
  assert.equal(cj.frames.length, r.head.length);
  assert.deepEqual(cj.frames[3].slice(0, 3), r.head[3].slice(0, 3));
  assert.deepEqual(cj.frames[3].slice(3, 6), r.left[3]);
  assert.deepEqual(cj.frames[3].slice(6, 9), r.right[3]);
  assert.equal(validateRun(cj).ok, true);
});

test('GhostDuo.sampleAt: exact at frame times, linear in between, clamped, slerp on the head', () => {
  const r = runFromChoreo();
  const g = new GhostDuo(r);
  assert.equal(g.choreoId, r.choreoId);
  assert.equal(g.fps, 30);
  assert.equal(g.frameCount, r.head.length);
  assert.equal(g.name, 'Tester');
  assert.equal(g.score, 90);
  const s = g.sampleAt(7 / 30);
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(s.head.p[k] - r.head[7][k]) < 1e-6);
    assert.ok(Math.abs(s.left.p[k] - r.left[7][k]) < 1e-6);
    assert.ok(Math.abs(s.right.p[k] - r.right[7][k]) < 1e-6);
  }
  for (let k = 0; k < 4; k++) assert.ok(Math.abs(s.head.q[k] - r.head[7][3 + k]) < 1e-4);
  const mid = g.sampleAt(7.5 / 30);
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(mid.head.p[k] - (r.head[7][k] + r.head[8][k]) / 2) < 1e-6);
    assert.ok(Math.abs(mid.left.p[k] - (r.left[7][k] + r.left[8][k]) / 2) < 1e-6);
  }
  const ql = Math.hypot(...mid.head.q);
  assert.ok(Math.abs(ql - 1) < 1e-6, 'unit quaternion');
  const before = g.sampleAt(-5);
  const last = g.sampleAt(1e9);
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(before.head.p[k] - r.head[0][k]) < 1e-6);
    assert.ok(Math.abs(last.head.p[k] - r.head[r.head.length - 1][k]) < 1e-6);
  }
  // the same reused object is returned by default; a custom out works too
  assert.equal(g.sampleAt(0.1), g.sampleAt(0.2));
  const own = createRunSample();
  assert.equal(g.sampleAt(0.3, own), own);
  assert.ok(g.headSpeedAt(0.5) >= 0);
});

test('GhostDuo slerp: yaw halfway between two frames; identity quaternion without quats', () => {
  const qa = quatFromYaw([0, 0, 0, 1], 0.2);
  const qb = quatFromYaw([0, 0, 0, 1], 0.6);
  const run = {
    choreoId: 'yaw', fps: 10,
    head: [[0, 1.6, 0, ...qa], [0, 1.6, 0, ...qb]],
    left: [[0, 1, 0], [0, 1, 0]],
    right: [[0, 1, 0], [0, 1, 0]],
  };
  const g = new GhostDuo(run);
  const s = g.sampleAt(0.05);
  assert.ok(Math.abs(yawFromQuat(s.head.q) - 0.4) < 1e-6);
  const c = new GhostDuo(compactRun(3));
  assert.equal(c.hasQuat, false);
  assert.deepEqual(Array.from(c.sampleAt(0.05).head.q), [0, 0, 0, 1]);
  assert.equal(c.name, 'Aufzeichnung');
});

test('sampleRunInto writes at offsets into flat arrays', () => {
  const n = normalizeRun(compactRun(3));
  const h = new Float32Array(6), l = new Float32Array(6), rr = new Float32Array(6);
  assert.equal(sampleRunInto(n, 0.5 / 30, h, 3, l, 3, rr, 3), true);
  assert.ok(Math.abs(h[3] - 0.5) < 1e-6 && Math.abs(h[5] + 0.5) < 1e-6);
  assert.ok(Math.abs(l[5] - 0.25) < 1e-6);
  assert.equal(h[0], 0);
});

test('storage: save/load newest first, caps per choreo and total, quota retry, clear', () => {
  const st = makeStorage();
  assert.deepEqual(loadRuns(st), []);
  for (let i = 0; i < 4; i++) {
    assert.equal(saveRun(runFromChoreo({ id: 'a', score: 50 + i, playedAt: `2026-01-0${i + 1}T00:00:00.000Z` }), st, { maxPerChoreo: 3, maxTotal: 5 }), true);
  }
  assert.equal(saveRun(runFromChoreo({ id: 'b', score: 70, playedAt: '2026-02-01T00:00:00.000Z' }), st, { maxPerChoreo: 3, maxTotal: 5 }), true);
  assert.equal(saveRun(runFromChoreo({ id: 'b', score: 71, playedAt: '2026-02-02T00:00:00.000Z' }), st, { maxPerChoreo: 3, maxTotal: 5 }), true);
  const all = loadRuns(st);
  assert.equal(all.length, 5, 'total cap');
  assert.equal(all[0].choreoId, 'b');
  assert.equal(all[0].score, 71, 'newest first');
  const a = loadRuns(st, { choreoId: 'a' });
  assert.equal(a.length, 3, 'per-choreo cap');
  assert.deepEqual(a.map((r) => r.score), [53, 52, 51], 'oldest dropped');
  assert.equal(saveRun({ choreoId: 'bad' }, st), false);
  // normalized runs are converted to JSON on save
  assert.equal(saveRun(normalizeRun(runFromChoreo({ id: 'c', playedAt: '2026-03-01T00:00:00.000Z' })), st), true);
  assert.equal(loadRuns(st, { choreoId: 'c' }).length, 1);
  // quota: the store rejects big writes -> the oldest half is dropped, the new run kept
  const size = st.raw('dp.runs').length;
  st.failAbove = size + 10;
  assert.equal(saveRun(runFromChoreo({ id: 'd', playedAt: '2026-04-01T00:00:00.000Z' }), st), true);
  assert.equal(loadRuns(st, { choreoId: 'd' }).length, 1);
  assert.ok(loadRuns(st).length < 7);
  st.failAbove = Infinity;
  // clear
  assert.equal(clearRuns(st, { choreoId: 'd' }), true);
  assert.equal(loadRuns(st, { choreoId: 'd' }).length, 0);
  assert.ok(loadRuns(st).length > 0);
  assert.equal(clearRuns(st), true);
  assert.equal(loadRuns(st).length, 0);
  // corrupt store is tolerated
  st.setItem('dp.runs', '{not json');
  assert.deepEqual(loadRuns(st), []);
  st.setItem('dp.runs', JSON.stringify({ runs: [compactRun(2, 30, 'x'), { broken: true }] }));
  assert.equal(loadRuns(st).length, 1);
  assert.deepEqual(loadRuns(null), []);
  assert.equal(saveRun(compactRun(), null), false);
});

test('capRuns and pickRun', () => {
  const list = [
    { choreoId: 'a', score: 10, playedAt: '2026-01-01T00:00:00Z' },
    { choreoId: 'a', score: 90, playedAt: '2026-01-02T00:00:00Z' },
    { choreoId: 'b', score: 50, playedAt: '2026-01-03T00:00:00Z' },
    { choreoId: 'a', score: 40, playedAt: '2026-01-04T00:00:00Z' },
  ];
  assert.deepEqual(capRuns(list, 2, 10).map((r) => r.score), [90, 50, 40]);
  assert.deepEqual(capRuns(list, 5, 2).map((r) => r.score), [50, 40]);
  assert.equal(pickRun(list, { choreoId: 'a' }).score, 40, 'latest');
  assert.equal(pickRun(list, { choreoId: 'a', strategy: 'best' }).score, 90);
  assert.equal(pickRun(list, { choreoId: 'a', strategy: 'closest', score: 20 }).score, 10);
  assert.equal(pickRun(list, { choreoId: 'zzz' }), null);
  assert.equal(pickRun([], {}), null);
});

test('toHttpBase converts ?server= values', () => {
  assert.equal(toHttpBase('wss://192.168.1.10:8443'), 'https://192.168.1.10:8443');
  assert.equal(toHttpBase('wss://host:8443/ws'), 'https://host:8443');
  assert.equal(toHttpBase('ws://localhost:8080/'), 'http://localhost:8080');
  assert.equal(toHttpBase('https://host/'), 'https://host');
  assert.equal(toHttpBase(''), '');
  assert.equal(toHttpBase(null), '');
});

test('server helpers use /api/runs with the right verbs and bodies', async () => {
  const calls = [];
  const runs = [{ id: 'run1', choreoId: 'a', score: 60, playedAt: '2026-01-01T00:00:00Z', url: '/api/runs/run1' },
    { id: 'run2', choreoId: 'a', score: 80, playedAt: '2026-01-02T00:00:00Z', url: '/api/runs/run2' }];
  const full = runToJSON(runFromChoreo({ id: 'a' }));
  const fetchImpl = async (url, init = {}) => {
    calls.push({ url, method: init.method || 'GET', body: init.body });
    const path = new URL(url).pathname;
    const json = (obj, status = 200) => ({ ok: status < 400, status, json: async () => obj });
    if (path === '/api/runs' && init.method === 'POST') return json({ ok: true, id: 'new', url: '/api/runs/new', run: {} }, 201);
    if (path === '/api/runs') return json({ runs });
    if (path === '/api/runs/run2') return json(full);
    return json({ error: 'nope' }, 404);
  };
  const up = await uploadRun(normalizeRun(full), 'wss://h:1/ws', fetchImpl);
  assert.equal(up.id, 'new');
  assert.equal(calls[0].url, 'https://h:1/api/runs');
  assert.equal(calls[0].method, 'POST');
  assert.equal(JSON.parse(calls[0].body).choreoId, 'a');
  const list = await fetchRunList('http://h:2', { choreoId: 'a', fetchImpl });
  assert.equal(list.length, 2);
  assert.ok(calls[1].url.startsWith('http://h:2/api/runs?choreoId=a'));
  const got = await fetchRun('http://h:2', 'run2', fetchImpl);
  assert.equal(got.choreoId, 'a');
  await assert.rejects(fetchRun('http://h:2', 'missing', fetchImpl), /404/);
  // loadGhost: best run from the server (run2), ghost is usable
  const g = await loadGhost({ choreoId: 'a', baseUrl: 'http://h:2', fetchImpl, strategy: 'best' });
  assert.ok(g instanceof GhostDuo);
  assert.equal(g.frameCount, full.frameCount);
  // loadGhost: server down -> local storage fallback -> null when nothing is there
  const st = makeStorage();
  const failing = async () => { throw new Error('ECONNREFUSED'); };
  const origWarn = console.warn;
  console.warn = () => {};
  try {
    assert.equal(await loadGhost({ choreoId: 'a', baseUrl: 'http://h:3', fetchImpl: failing, storage: st }), null);
    saveRun(full, st);
    const g2 = await loadGhost({ choreoId: 'a', baseUrl: 'http://h:3', fetchImpl: failing, storage: st });
    assert.ok(g2 instanceof GhostDuo);
  } finally {
    console.warn = origWarn;
  }
  assert.equal(await loadGhost({ choreoId: 'zzz', storage: st }), null);
});
