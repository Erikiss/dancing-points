// recorder.test.js - Recorder produces a valid choreography from stage-frame samples.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Recorder } from '../../webxr/src/game/recorder.js';
import { Choreo, validateChoreo, createChoreoSample } from '../../webxr/src/game/choreo.js';
import { createSample } from '../../webxr/src/xr/input.js';
import { makeChoreo } from './helpers/make-choreo.js';

function makeStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => { map.set(k, String(v)); },
    removeItem: (k) => { map.delete(k); },
    map,
  };
}

/** Record the synthetic choreography back through the recorder at `hz` with dropped frames. */
function record(opts = {}) {
  const source = new Choreo(makeChoreo());
  const rec = new Recorder();
  rec.start({ title: 'Mein Test Tanz', bpm: 120, bars: 4, countInBeats: 4, synth: 'house', mirror: true, author: 'me', ...opts.config });
  assert.equal(rec.state, 'recording');
  assert.equal(rec.frameCount, 241);
  const hz = opts.hz || 72;
  const ref = createChoreoSample();
  const s = createSample();
  let written = 0;
  for (let i = 0; i <= Math.round(9.2 * hz); i++) {
    const t = i / hz - 1.0;          // starts 1 s before beat 0 (count-in samples must be ignored), ends past the last frame
    if (opts.dropEvery && i % opts.dropEvery === 0) continue;
    source.sampleAt(t, ref);
    s.head.p[0] = ref.head.p[0]; s.head.p[1] = ref.head.p[1]; s.head.p[2] = ref.head.p[2];
    s.head.q[0] = ref.head.q[0]; s.head.q[1] = ref.head.q[1]; s.head.q[2] = ref.head.q[2]; s.head.q[3] = ref.head.q[3];
    s.left.p[0] = ref.left.p[0]; s.left.p[1] = ref.left.p[1]; s.left.p[2] = ref.left.p[2];
    s.right.p[0] = ref.right.p[0]; s.right.p[1] = ref.right.p[1]; s.right.p[2] = ref.right.p[2];
    const idx = rec.update(t, s);
    if (t < 0) assert.equal(idx, -1);
    else if (idx >= 0) written++;
    if (opts.stopAt !== undefined && t >= opts.stopAt) break;
  }
  return { rec, source, written };
}

test('records a valid choreography with default moves and meta', () => {
  const { rec, source } = record();
  assert.equal(rec.complete, true);
  assert.ok(Math.abs(rec.progress - 1) < 1e-9);
  const json = rec.stop();
  assert.equal(rec.state, 'stopped');
  assert.deepEqual(validateChoreo(json), { ok: true, errors: [] });
  assert.equal(json.format, 'dancing-points-choreo/1');
  assert.ok(/^mein-test-tanz-[a-z0-9]+$/.test(json.id), json.id);
  assert.equal(json.title, 'Mein Test Tanz');
  assert.equal(json.bpm, 120);
  assert.equal(json.durationBeats, 16);
  assert.equal(json.fps, 30);
  assert.equal(json.mirror, true);
  assert.equal(json.audio.synth, 'house');
  assert.equal(json.audio.url, null);
  assert.deepEqual(json.moves.map((m) => m.name), ['Teil 1', 'Teil 2']);
  assert.deepEqual(json.moves.map((m) => [m.startBeat, m.endBeat]), [[0, 8], [8, 16]]);
  assert.equal(json.frames.head.length, 241);
  assert.equal(json.frames.head[0].length, 7);
  assert.equal(json.meta.source, 'recorded');
  assert.equal(json.meta.author, 'me');
  assert.equal(json.meta.holdFilledFrames, 0);
  // the recording reproduces the source within rounding (0.1 mm) at every frame
  const c = new Choreo(json);
  let maxErr = 0;
  for (let i = 0; i < 241; i++) {
    for (let k = 0; k < 3; k++) {
      maxErr = Math.max(maxErr, Math.abs(c.headP[i * 3 + k] - source.headP[i * 3 + k]));
      maxErr = Math.max(maxErr, Math.abs(c.leftP[i * 3 + k] - source.leftP[i * 3 + k]));
      maxErr = Math.max(maxErr, Math.abs(c.rightP[i * 3 + k] - source.rightP[i * 3 + k]));
    }
  }
  assert.ok(maxErr < 0.006, `max error ${maxErr}`);
  // sampleAt at frame times matches the head quaternion too
  const q = c.sampleAt(3).head.q;
  const q0 = source.sampleAt(3).head.q;
  for (let k = 0; k < 4; k++) assert.ok(Math.abs(q[k] - q0[k]) < 1e-4);
});

test('dropped frames and an early stop are hold-filled', () => {
  const { rec } = record({ hz: 30, dropEvery: 7, stopAt: 6.0 });
  assert.equal(rec.complete, false);
  const json = rec.stop();
  assert.equal(validateChoreo(json).ok, true);
  assert.equal(json.frames.head.length, 241);
  const last = rec.lastIndex;
  assert.ok(last === 180 || last === 181, `last index ${last}`);
  assert.equal(json.meta.holdFilledFrames, 240 - last, `hold filled ${json.meta.holdFilledFrames}`);
  // the trailing frames repeat the last recorded one
  assert.deepEqual(json.frames.head[240], json.frames.head[last]);
  assert.deepEqual(json.frames.left[240], json.frames.left[last]);
  assert.notDeepEqual(json.frames.head[last - 1], json.frames.head[last]);
  // dropped samples were bridged by interpolation, not counted as gaps
  assert.ok(Array.from(rec.filled.subarray(0, last + 1)).every((f) => f === 1));
});

test('errors and cancel', () => {
  const rec = new Recorder();
  assert.throws(() => rec.stop(), /not recording/);
  assert.throws(() => rec.start({ bpm: 0 }), RangeError);
  rec.start({ title: '', bpm: 100, bars: 2 });
  assert.equal(rec.config.title, 'Eigener Tanz');
  assert.equal(rec.update(-0.5, createSample()), -1);
  assert.throws(() => rec.stop(), /no frames/);
  assert.equal(rec.state, 'idle');
  rec.start({ bpm: 100, bars: 2 });
  rec.cancel();
  assert.equal(rec.state, 'idle');
  assert.equal(rec.update(0, createSample()), -1);
});

test('default moves, ids and JSON helpers', () => {
  assert.deepEqual(Recorder.defaultMoves(20, 8).map((m) => [m.name, m.startBeat, m.endBeat]), [['Teil 1', 0, 8], ['Teil 2', 8, 16], ['Teil 3', 16, 20]]);
  assert.equal(Recorder.defaultMoves(8, 8).length, 1);
  assert.equal(Recorder.makeId('Schöne Grüße! Tanz #1', 1000), 'schoene-gruesse-tanz-1-rs');
  assert.equal(Recorder.makeId('', 1000), 'tanz-rs');
  assert.ok(Recorder.makeId('x'.repeat(80), 1).length <= 34);
  const json = makeChoreo({ id: 'abc' });
  assert.equal(Recorder.filename(json), 'abc.json');
  assert.equal(JSON.parse(Recorder.toJSONString(json)).id, 'abc');
  assert.ok(Recorder.toJSONString(json, true).includes('\n'));
  // explicit id and durationBeats are honoured
  const rec = new Recorder();
  rec.start({ id: 'my-id', bpm: 60, durationBeats: 3, beatsPerMove: 2 });
  assert.equal(rec.config.id, 'my-id');
  assert.equal(rec.frameCount, 91);
  assert.equal(Recorder.download(json, null), null);
});

test('persist / list / remove in storage and upload via injected fetch', async () => {
  const storage = makeStorage();
  const a = makeChoreo({ id: 'rec-a' });
  const b = makeChoreo({ id: 'rec-b' });
  assert.equal(Recorder.persist(a, storage), true);
  assert.equal(Recorder.persist(b, storage), true);
  assert.equal(Recorder.persist(a, null), false);
  assert.deepEqual(Recorder.listPersisted(storage).map((c) => c.id).sort(), ['rec-a', 'rec-b']);
  assert.equal(Recorder.removePersisted('rec-a', storage), true);
  assert.equal(Recorder.removePersisted('rec-a', storage), false);
  assert.deepEqual(Recorder.listPersisted(storage).map((c) => c.id), ['rec-b']);
  storage.setItem('dp.choreos', '{broken');
  assert.deepEqual(Recorder.listPersisted(storage), []);
  assert.equal(Recorder.persist(a, storage), true, 'a broken store is replaced');
  assert.deepEqual(Recorder.listPersisted({ getItem: () => null }), []);

  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({ url, init });
    if (url.includes('fail')) return { ok: false, status: 500 };
    return { ok: true, status: 200, json: async () => ({ ok: true, id: 'rec-b' }) };
  };
  const res = await Recorder.upload(b, 'https://host:8443/', fetchImpl);
  assert.deepEqual(res, { ok: true, id: 'rec-b' });
  assert.equal(calls[0].url, 'https://host:8443/api/choreos');
  assert.equal(calls[0].init.method, 'POST');
  assert.equal(JSON.parse(calls[0].init.body).id, 'rec-b');
  await assert.rejects(Recorder.upload(b, 'https://fail', fetchImpl), /500/);
});
