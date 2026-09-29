// choreo.test.js - validation, sampling, moves, fullBody decoding and loading.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  Choreo, validateChoreo, loadChoreo, expectedFrameCount, createChoreoSample, mirrorChoreoSample,
  encodeBase64Float32, decodeBase64Float32, CHOREO_FORMAT,
} from '../../webxr/src/game/choreo.js';
import { makeChoreo, cloneJSON } from './helpers/make-choreo.js';

const near = (a, b, eps = 1e-6) => Math.abs(a - b) <= eps;
const vecNear = (a, b, eps = 1e-6) => a.every((x, i) => near(x, b[i], eps));

test('synthetic choreo validates and has the expected length', () => {
  const json = makeChoreo();
  const v = validateChoreo(json);
  assert.deepEqual(v, { ok: true, errors: [] });
  assert.equal(expectedFrameCount(json), 241);
  assert.equal(json.frames.head.length, 241);
  const c = new Choreo(json);
  assert.equal(c.frameCount, 241);
  assert.ok(near(c.duration, 8));
  assert.ok(near(c.beatDuration, 0.5));
  assert.equal(c.moves.length, 2);
  assert.equal(c.hasFullBody, false);
  assert.equal(c.audio.synth, 'metronome');
  assert.equal(c.toIndexEntry().file, 'test-sway.json');
});

test('validation rejects wrong lengths, NaNs, non-contiguous moves, bad ids', () => {
  const bad = (mutate, expectSubstring) => {
    const j = cloneJSON(makeChoreo());
    mutate(j);
    const v = validateChoreo(j);
    assert.equal(v.ok, false, `expected failure for ${expectSubstring}`);
    assert.ok(v.errors.some((e) => e.includes(expectSubstring)), `errors ${JSON.stringify(v.errors)} should mention ${expectSubstring}`);
    assert.throws(() => new Choreo(j));
  };
  bad((j) => { j.frames.head.pop(); j.frames.head.pop(); }, 'frames.head has');
  bad((j) => { j.frames.left.push([0, 0, 0]); }, 'frames.left has');
  bad((j) => { j.frames.right[5][1] = NaN; }, 'frames.right[5]');
  bad((j) => { j.frames.head[3] = [0, 0, 0]; }, 'frames.head[3]');
  bad((j) => { j.moves[1].startBeat = 9; }, 'contiguous');
  bad((j) => { j.moves[1].endBeat = 15; }, 'durationBeats');
  bad((j) => { j.moves[0].endBeat = 0; }, 'endBeat must be');
  bad((j) => { j.moves = []; }, 'moves');
  bad((j) => { j.id = 'Bad Id!'; }, 'id');
  bad((j) => { j.format = 'other'; }, 'format');
  bad((j) => { j.bpm = 0; }, 'bpm');
  bad((j) => { j.audio.synth = 'techno'; }, 'audio.synth');
  bad((j) => { delete j.frames; }, 'frames missing');
  bad((j) => { j.fullBody = { joints: ['a'], parents: [-1], fps: 30, encoding: 'base64-float32', positions: 'AAAA' }; }, 'fullBody.positions byte length');
  assert.equal(validateChoreo(null).ok, false);
  assert.equal(validateChoreo('x').ok, false);
});

test('validation is tolerant to missing optional fields', () => {
  const j = cloneJSON(makeChoreo());
  delete j.artist;
  delete j.difficulty;
  delete j.audio;
  delete j.meta;
  delete j.mirror;
  delete j.moves[0].hint;
  assert.deepEqual(validateChoreo(j), { ok: true, errors: [] });
  const c = new Choreo(j);
  assert.equal(c.mirror, false);
  assert.equal(c.audio.synth, 'hiphop');
  assert.equal(c.artist, '');
  assert.equal(c.moves[0].hint, '');
  // one frame short (N instead of N+1) is accepted
  j.frames.head.pop(); j.frames.left.pop(); j.frames.right.pop();
  assert.equal(validateChoreo(j).ok, true);
});

test('sampleAt interpolates, clamps and reuses the output object', () => {
  const json = makeChoreo();
  const c = new Choreo(json);
  // exact frame times
  for (const i of [0, 1, 17, 120, 240]) {
    const s = c.sampleAt(i / 30);
    assert.ok(vecNear(s.head.p, json.frames.head[i].slice(0, 3)), `head frame ${i}`);
    assert.ok(vecNear(s.head.q, json.frames.head[i].slice(3, 7), 1e-4), `quat frame ${i}`);
    assert.ok(vecNear(s.left.p, json.frames.left[i]), `left frame ${i}`);
    assert.ok(vecNear(s.right.p, json.frames.right[i]), `right frame ${i}`);
  }
  // midpoint between frames 10 and 11
  const s = c.sampleAt(10.5 / 30);
  for (let k = 0; k < 3; k++) {
    const mid = 0.5 * (json.frames.head[10][k] + json.frames.head[11][k]);
    assert.ok(near(s.head.p[k], mid), `mid head ${k}`);
    const midL = 0.5 * (json.frames.left[10][k] + json.frames.left[11][k]);
    assert.ok(near(s.left.p[k], midL), `mid left ${k}`);
  }
  assert.ok(near(Math.hypot(...s.head.q), 1, 1e-6), 'slerp keeps unit length');
  // clamping
  const before = c.sampleAt(-5);
  assert.ok(vecNear(before.head.p, json.frames.head[0].slice(0, 3)));
  const after = c.sampleAt(500);
  assert.ok(vecNear(after.head.p, json.frames.head[240].slice(0, 3)));
  assert.ok(vecNear(c.sampleAt(NaN).head.p, json.frames.head[0].slice(0, 3)));
  // reuse
  const a = c.sampleAt(1);
  const b = c.sampleAt(2);
  assert.equal(a, b);
  const own = createChoreoSample();
  assert.equal(c.sampleAt(1, own), own);
  assert.ok(vecNear(own.head.p, c.sampleAt(1).head.p));
});

test('moveAt / beat conversions', () => {
  const c = new Choreo(makeChoreo());
  assert.equal(c.moveAt(0).name, 'Sway');
  assert.equal(c.moveAt(3.99).name, 'Sway');
  assert.equal(c.moveAt(4).name, 'Arms');
  assert.equal(c.moveAt(7.99).name, 'Arms');
  assert.equal(c.moveAt(8), null);
  assert.equal(c.moveAt(-0.1), null);
  assert.equal(c.moveIndexAt(5), 1);
  assert.equal(c.moveIndexAtBeat(8), 1);
  assert.equal(c.moveIndexAtBeat(16), -1);
  assert.ok(near(c.beatToTime(4), 2));
  assert.ok(near(c.timeToBeat(2), 4));
  assert.equal(c.moves[1].index, 1);
});

test('mirror helper flips x, swaps hands and mirrors the yaw', () => {
  const c = new Choreo(makeChoreo());
  const s = c.sampleAt(1.234, createChoreoSample());
  const m = mirrorChoreoSample(createChoreoSample(), s);
  assert.ok(near(m.head.p[0], -s.head.p[0]));
  assert.ok(near(m.head.p[1], s.head.p[1]));
  assert.ok(vecNear(m.left.p, [-s.right.p[0], s.right.p[1], s.right.p[2]]));
  assert.ok(vecNear(m.right.p, [-s.left.p[0], s.left.p[1], s.left.p[2]]));
  assert.ok(near(m.head.q[1], -s.head.q[1]));
  assert.ok(near(m.head.q[3], s.head.q[3]));
});

test('base64 float32 round trip and fullBodyAt interpolation', () => {
  const f = new Float32Array([0, 1.5, -2.25, 1e-3, 12345.678]);
  const dec = decodeBase64Float32(encodeBase64Float32(f));
  assert.equal(dec.length, 5);
  for (let i = 0; i < 5; i++) assert.equal(dec[i], f[i]);

  const json = makeChoreo({ fullBody: true });
  assert.equal(validateChoreo(json).ok, true);
  const c = new Choreo(json);
  assert.equal(c.hasFullBody, true);
  assert.equal(c.fullBody.jointCount, 3);
  assert.equal(c.fullBody.frameCount, 241);
  assert.deepEqual(c.fullBody.parents, [-1, 0, 0]);
  const p = c.fullBodyAt(7 / 30);
  assert.equal(p.length, 9);
  assert.ok(near(p[3], json.frames.head[7][0], 1e-6));
  assert.ok(near(p[4], json.frames.head[7][1], 1e-6));
  assert.ok(near(p[6], json.frames.left[7][0], 1e-6));
  // interpolation midpoint
  const mid = c.fullBodyAt(7.5 / 30, new Float32Array(9));
  assert.ok(near(mid[4], 0.5 * (json.frames.head[7][1] + json.frames.head[8][1]), 1e-6));
  // clamps
  assert.ok(near(c.fullBodyAt(99)[4], json.frames.head[240][1], 1e-6));
  assert.ok(near(c.fullBodyAt(-1)[4], json.frames.head[0][1], 1e-6));
  // no fullBody -> null
  assert.equal(new Choreo(makeChoreo()).fullBodyAt(0), null);
});

test('loadChoreo from object, from URL via injected fetch, error paths', async () => {
  const json = makeChoreo({ id: 'loaded-one' });
  json.audio.url = 'track.mp3';
  const c1 = await loadChoreo(json);
  assert.equal(c1.id, 'loaded-one');
  assert.equal(await loadChoreo(c1), c1);
  const calls = [];
  const fetchImpl = async (url) => {
    calls.push(url);
    if (url.endsWith('missing.json')) return { ok: false, status: 404 };
    if (url.endsWith('bad.json')) return { ok: true, status: 200, json: async () => ({ format: 'nope' }) };
    return { ok: true, status: 200, json: async () => cloneJSON(json) };
  };
  const c2 = await loadChoreo('/choreos/loaded-one.json', fetchImpl);
  assert.equal(c2.id, 'loaded-one');
  assert.equal(c2.audio.url, '/choreos/track.mp3', 'relative audio url resolved against the file');
  assert.deepEqual(calls, ['/choreos/loaded-one.json']);
  await assert.rejects(loadChoreo('/choreos/missing.json', fetchImpl), /404/);
  await assert.rejects(loadChoreo('/choreos/bad.json', fetchImpl), /invalid choreography/);
  assert.equal(CHOREO_FORMAT, 'dancing-points-choreo/1');
});
