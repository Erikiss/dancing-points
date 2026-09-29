// choreos-files.test.js - structural checks of the shipped, procedurally generated
// choreographies (webxr/choreos/snoop-cwalk.json, tutorial-basics.json, index.json) with an
// independent small validator of the DESIGN.md section 5 rules, plausibility checks (hand
// distances, head height, no teleports), the required move names/order, and a determinism
// check of tools/gen_choreos.js (regenerating into a temp dir yields byte-identical files).
// Also confirms the app's own validateChoreo/Choreo accept the files. No three.js, no DOM.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { writeChoreos, CHOREO_DEFS } from '../../tools/gen_choreos.js';
import { validateChoreo, Choreo } from '../../webxr/src/game/choreo.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const DIR = path.join(ROOT, 'webxr', 'choreos');

const readJSON = (file) => JSON.parse(fs.readFileSync(file, 'utf8'));
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const finiteRow = (row, width) => Array.isArray(row) && row.length === width && row.every((v) => typeof v === 'number' && Number.isFinite(v));

const SPEC = {
  'snoop-cwalk': {
    title: 'Snoop Dogg C-Walk – 6 Basic Combos',
    artist: 'nach Electro Breakers (TikTok)',
    bpm: 90, countInBeats: 4, durationBeats: 48, mirror: true, difficulty: 2, synth: 'hiphop',
    frames: 961,
    moves: ['Shoe Vibe', 'Restep', 'Heel Toe', 'Side Hopping', 'Shuffle Legs', 'Gangster Two Step'],
  },
  'tutorial-basics': {
    title: 'Tutorial: Grundschritte',
    bpm: 80, durationBeats: 32, mirror: false, difficulty: 1,
    frames: 721,
    moves: ['Wippen', 'Seitwärts', 'Arme hoch', 'Freestyle'],
  },
};

/** Independent validator of the structural rules (DESIGN.md section 5) plus plausibility. */
function checkChoreoFile(c, spec) {
  assert.equal(c.format, 'dancing-points-choreo/1');
  assert.match(c.id, /^[a-z0-9-]+$/);
  assert.equal(typeof c.title, 'string');
  assert.equal(c.fps, 30);
  assert.equal(c.beatsPerBar, 4);
  assert.equal(c.referenceHeight, 1.7);
  assert.equal(c.bpm, spec.bpm);
  assert.equal(c.durationBeats, spec.durationBeats);
  assert.equal(c.mirror, spec.mirror);
  assert.equal(c.difficulty, spec.difficulty);
  assert.equal(c.title, spec.title);
  if (spec.artist) assert.equal(c.artist, spec.artist);
  if (spec.countInBeats !== undefined) assert.equal(c.countInBeats, spec.countInBeats);
  assert.equal(c.audio.url, null);
  if (spec.synth) assert.equal(c.audio.synth, spec.synth);
  assert.equal(c.meta.source, 'procedural');

  // frame counts: durationBeats * 60 / bpm * fps + 1, index 0 = beat 0
  const n = Math.round(c.durationBeats * 60 / c.bpm * c.fps) + 1;
  assert.equal(n, spec.frames);
  assert.equal(c.frames.head.length, n, 'head frame count');
  assert.equal(c.frames.left.length, n, 'left frame count');
  assert.equal(c.frames.right.length, n, 'right frame count');

  // moves: contiguous from 0 to durationBeats, 8 beats each, exact names in order, German hints
  assert.deepEqual(c.moves.map((m) => m.name), spec.moves);
  let expectedStart = 0;
  for (const m of c.moves) {
    assert.equal(m.startBeat, expectedStart, `${m.name} starts at ${expectedStart}`);
    assert.equal(m.endBeat - m.startBeat, 8, `${m.name} is 8 beats`);
    assert.equal(typeof m.hint, 'string');
    assert.ok(m.hint.length > 20, `${m.name} has a hint`);
    expectedStart = m.endBeat;
  }
  assert.equal(expectedStart, c.durationBeats);

  // per frame: finite, widths, unit quaternion, head height, hands near the head, no teleports
  let maxHeadStep = 0, maxHandStep = 0;
  for (let i = 0; i < n; i++) {
    const h = c.frames.head[i], l = c.frames.left[i], r = c.frames.right[i];
    assert.ok(finiteRow(h, 7), `head[${i}] finite 7-vector`);
    assert.ok(finiteRow(l, 3), `left[${i}] finite 3-vector`);
    assert.ok(finiteRow(r, 3), `right[${i}] finite 3-vector`);
    const qn = Math.hypot(h[3], h[4], h[5], h[6]);
    assert.ok(Math.abs(qn - 1) < 1e-3, `head[${i}] quaternion unit length (${qn})`);
    assert.ok(h[1] >= 1.0 && h[1] <= 2.0, `head[${i}] y in 1..2 (${h[1]})`);
    assert.ok(dist(l, h) <= 0.9, `left[${i}] within 0.9 m of the head (${dist(l, h)})`);
    assert.ok(dist(r, h) <= 0.9, `right[${i}] within 0.9 m of the head (${dist(r, h)})`);
    assert.ok(dist(l, r) > 0.05, `hands[${i}] do not overlap`);
    if (i > 0) {
      maxHeadStep = Math.max(maxHeadStep, dist(h, c.frames.head[i - 1]));
      maxHandStep = Math.max(maxHandStep, dist(l, c.frames.left[i - 1]), dist(r, c.frames.right[i - 1]));
    }
  }
  assert.ok(maxHeadStep < 0.15, `head moves < 15 cm per frame (${maxHeadStep})`);
  assert.ok(maxHandStep < 0.25, `hands move < 25 cm per frame (${maxHandStep})`);

  // starts and ends in the neutral standing pose (head at rest height, facing -Z)
  const first = c.frames.head[0], last = c.frames.head[n - 1];
  for (const h of [first, last]) {
    assert.deepEqual(h.slice(0, 3), [0, 1.6, 0]);
    assert.deepEqual(h.slice(3), [0, 0, 0, 1]);
  }
  // rounding: at most 4 decimals
  for (const key of ['head', 'left', 'right']) {
    for (const row of c.frames[key]) {
      for (const v of row) assert.equal(Math.round(v * 1e4) / 1e4, v, 'values rounded to 4 decimals');
    }
  }
}

/** Per-move motion signature: head amplitudes and mean hand positions relative to the head. */
function moveSignature(c, move) {
  const beatDur = 60 / c.bpm;
  const i0 = Math.ceil((move.startBeat + 0.5) * beatDur * c.fps);
  const i1 = Math.floor((move.endBeat - 0.5) * beatDur * c.fps);
  let sx = 0, sy = 0, sz = 0, lx = 0, ly = 0, lz = 0, k = 0;
  for (let i = i0; i <= i1; i++) {
    const h = c.frames.head[i], l = c.frames.left[i];
    sx += h[0] * h[0]; sy += (h[1] - 1.6) * (h[1] - 1.6); sz += h[2] * h[2];
    lx += Math.abs(l[0] - h[0]); ly += l[1] - h[1]; lz += l[2] - h[2];
    k++;
  }
  return [Math.sqrt(sx / k), Math.sqrt(sy / k), Math.sqrt(sz / k), lx / k, ly / k, lz / k];
}

test('snoop-cwalk.json follows the format and the 6-combo spec', () => {
  const c = readJSON(path.join(DIR, 'snoop-cwalk.json'));
  checkChoreoFile(c, SPEC['snoop-cwalk']);
  assert.ok(fs.statSync(path.join(DIR, 'snoop-cwalk.json')).size <= 600 * 1024);
  // characteristic amplitudes (metres) of the moves, measured on the frames
  const sig = Object.fromEntries(c.moves.map((m) => [m.name, moveSignature(c, m)]));
  assert.ok(sig['Shoe Vibe'][0] > 0.03 && sig['Shoe Vibe'][0] < 0.07, 'Shoe Vibe sways about +-6 cm');
  assert.ok(sig['Restep'][2] > 0.08, 'Restep rocks fore/aft about +-12 cm');
  assert.ok(sig['Heel Toe'][4] > -0.45, 'Heel Toe keeps the hands at chest height');
  assert.ok(sig['Side Hopping'][0] > 0.2, 'Side Hopping hops about +-30 cm');
  assert.ok(sig['Shuffle Legs'][4] < sig['Heel Toe'][4] - 0.1, 'Shuffle Legs hands lower than Heel Toe');
  assert.ok(sig['Gangster Two Step'][0] > 0.15, 'Gangster Two Step sways about +-20 cm');
  // every move has a distinct, learnable signature (>= 3 cm difference in some feature)
  const names = Object.keys(sig);
  for (let a = 0; a < names.length; a++) {
    for (let b = a + 1; b < names.length; b++) {
      const d = Math.max(...sig[names[a]].map((v, i) => Math.abs(v - sig[names[b]][i])));
      assert.ok(d >= 0.03, `${names[a]} vs ${names[b]} differ (max feature delta ${d.toFixed(3)} m)`);
    }
  }
});

test('tutorial-basics.json follows the format and the 4-move spec', () => {
  const c = readJSON(path.join(DIR, 'tutorial-basics.json'));
  checkChoreoFile(c, SPEC['tutorial-basics']);
  assert.ok(fs.statSync(path.join(DIR, 'tutorial-basics.json')).size <= 600 * 1024);
  const sig = Object.fromEntries(c.moves.map((m) => [m.name, moveSignature(c, m)]));
  assert.ok(sig['Wippen'][1] > 0.015 && sig['Wippen'][0] < 0.001, 'Wippen bounces without sway');
  assert.ok(sig['Seitwärts'][0] > 0.08, 'Seitwärts sways about +-15 cm');
  // "Arme hoch": each hand rises above the head at some point
  const beatDur = 60 / c.bpm;
  const above = { left: false, right: false };
  for (let i = Math.round(16 * beatDur * 30); i < Math.round(24 * beatDur * 30); i++) {
    if (c.frames.left[i][1] > c.frames.head[i][1] + 0.15) above.left = true;
    if (c.frames.right[i][1] > c.frames.head[i][1] + 0.15) above.right = true;
  }
  assert.deepEqual(above, { left: true, right: true });
});

test('index.json lists the shipped choreographies with matching headers', () => {
  const index = readJSON(path.join(DIR, 'index.json'));
  assert.ok(Array.isArray(index.choreos));
  const ids = index.choreos.map((e) => e.id);
  assert.deepEqual(ids, ['snoop-cwalk', 'tutorial-basics', 'mocap-freestyle']);
  for (const e of index.choreos) {
    for (const key of ['id', 'title', 'artist', 'bpm', 'durationBeats', 'difficulty', 'file']) {
      assert.ok(e[key] !== undefined && e[key] !== null, `${e.id}.${key}`);
    }
    assert.equal(e.file, `${e.id}.json`);
    const file = path.join(DIR, e.file);
    assert.ok(fs.existsSync(file), `${e.file} exists`);
    const c = readJSON(file);
    assert.equal(c.id, e.id);
    assert.equal(c.title, e.title);
    assert.equal(c.bpm, e.bpm);
    assert.equal(c.durationBeats, e.durationBeats);
    assert.equal(c.difficulty, e.difficulty);
  }
  const mocap = index.choreos.find((e) => e.id === 'mocap-freestyle');
  assert.deepEqual(mocap, {
    id: 'mocap-freestyle', title: 'Freestyle (Mocap-Demo)', artist: 'Dancing Points Datensatz',
    bpm: 100, durationBeats: 50, difficulty: 3, file: 'mocap-freestyle.json',
  });
});

test('the app module validates and loads the generated files', () => {
  for (const id of Object.keys(SPEC)) {
    const json = readJSON(path.join(DIR, `${id}.json`));
    assert.deepEqual(validateChoreo(json), { ok: true, errors: [] });
    const c = new Choreo(json);
    assert.equal(c.frameCount, SPEC[id].frames);
    assert.equal(c.moveAt(0).name, SPEC[id].moves[0]);
    assert.equal(c.moveAt(c.beatToTime(9)).name, SPEC[id].moves[1]);
    const s = c.sampleAt(c.duration / 2);
    assert.ok([...s.head.p, ...s.head.q, ...s.left.p, ...s.right.p].every(Number.isFinite));
  }
});

test('the generator is deterministic and reproduces the committed files', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'dp-choreos-'));
  try {
    const report = writeChoreos(tmp);
    assert.equal(report.length, CHOREO_DEFS.length + 1);
    for (const name of ['snoop-cwalk.json', 'tutorial-basics.json', 'index.json']) {
      const fresh = fs.readFileSync(path.join(tmp, name), 'utf8');
      const committed = fs.readFileSync(path.join(DIR, name), 'utf8');
      assert.ok(fresh === committed, `${name} regenerated byte-identical (run: node tools/gen_choreos.js)`);
    }
    // a second run into another dir is identical as well
    const tmp2 = fs.mkdtempSync(path.join(os.tmpdir(), 'dp-choreos-'));
    try {
      writeChoreos(tmp2);
      for (const name of ['snoop-cwalk.json', 'tutorial-basics.json']) {
        assert.equal(fs.readFileSync(path.join(tmp2, name), 'utf8'), fs.readFileSync(path.join(tmp, name), 'utf8'));
      }
    } finally {
      fs.rmSync(tmp2, { recursive: true, force: true });
    }
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});
