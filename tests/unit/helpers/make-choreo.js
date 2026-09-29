// make-choreo.js - builds a small synthetic, valid `dancing-points-choreo/1` object for unit
// tests (16 beats @ 120 bpm, 30 fps: sinusoidal head sway/bounce + arm swings). No file I/O,
// no dependency on choreographies written by other lanes.

import { encodeBase64Float32 } from '../../../webxr/src/game/choreo.js';
import { quatFromYaw } from '../../../webxr/src/util/math.js';

/**
 * @param {object} [opts]
 * @param {string} [opts.id='test-sway']
 * @param {number} [opts.bpm=120]
 * @param {number} [opts.durationBeats=16]
 * @param {number} [opts.fps=30]
 * @param {boolean} [opts.mirror=false]
 * @param {boolean} [opts.fullBody=false]  add a tiny 3-joint fullBody block
 * @param {boolean} [opts.still=false]     all frames identical (static reference)
 * @param {number} [opts.referenceHeight=1.70]
 * @param {number} [opts.countInBeats=4]
 */
export function makeChoreo(opts = {}) {
  const bpm = opts.bpm || 120;
  const durationBeats = opts.durationBeats || 16;
  const fps = opts.fps || 30;
  const referenceHeight = opts.referenceHeight || 1.70;
  const beatDur = 60 / bpm;
  const n = Math.round(durationBeats * beatDur * fps) + 1;
  const head = new Array(n), left = new Array(n), right = new Array(n);
  const q = [0, 0, 0, 1];
  const r = (x) => Math.round(x * 10000) / 10000;
  for (let i = 0; i < n; i++) {
    const t = opts.still ? 0 : i / fps;
    const beat = t / beatDur;
    const move2 = beat >= durationBeats / 2 ? 1 : 0;
    // head: lateral sway (period 2 beats), bounce (period 1 beat), slight fore/aft (period 4 beats)
    const hx = 0.15 * Math.sin(Math.PI * beat);
    const hy = referenceHeight - 0.02 + 0.03 * Math.sin(2 * Math.PI * beat);
    const hz = 0.05 * Math.sin(Math.PI * beat / 2);
    quatFromYaw(q, 0.25 * Math.sin(Math.PI * beat / 2));
    head[i] = [r(hx), r(hy), r(hz), r(q[0]), r(q[1]), r(q[2]), r(q[3])];
    // arms: first half small swing, second half big alternating swings
    const amp = move2 ? 0.45 : 0.2;
    const swing = amp * Math.sin(Math.PI * beat);
    left[i] = [r(hx - 0.25 - 0.1 * swing), r(hy - 0.45 + swing), r(hz - 0.25 + 0.2 * swing)];
    right[i] = [r(hx + 0.25 + 0.1 * swing), r(hy - 0.45 - swing), r(hz - 0.25 - 0.2 * swing)];
  }
  const half = durationBeats / 2;
  const json = {
    format: 'dancing-points-choreo/1',
    id: opts.id || 'test-sway',
    title: 'Test Sway',
    artist: 'unit test',
    bpm,
    beatsPerBar: 4,
    countInBeats: opts.countInBeats !== undefined ? opts.countInBeats : 4,
    durationBeats,
    fps,
    referenceHeight,
    mirror: !!opts.mirror,
    difficulty: 1,
    audio: { url: null, synth: 'metronome', offsetSec: 0, gain: 0.8 },
    moves: [
      { name: 'Sway', startBeat: 0, endBeat: half, hint: 'hüfte' },
      { name: 'Arms', startBeat: half, endBeat: durationBeats, hint: 'arme' },
    ],
    frames: { head, left, right },
    meta: { source: 'procedural', createdAt: '2026-01-01T00:00:00.000Z', author: 'test', notes: '' },
  };
  if (opts.fullBody) {
    const joints = ['b_root', 'b_head', 'b_l_hand'];
    const parents = [-1, 0, 0];
    const pos = new Float32Array(n * joints.length * 3);
    for (let i = 0; i < n; i++) {
      const o = i * 9;
      pos[o] = head[i][0]; pos[o + 1] = 0; pos[o + 2] = head[i][2];
      pos[o + 3] = head[i][0]; pos[o + 4] = head[i][1]; pos[o + 5] = head[i][2];
      pos[o + 6] = left[i][0]; pos[o + 7] = left[i][1]; pos[o + 8] = left[i][2];
    }
    json.fullBody = { joints, parents, fps, encoding: 'base64-float32', positions: encodeBase64Float32(pos) };
  }
  return json;
}

/** Deep clone helper for tests that mutate the JSON. */
export function cloneJSON(obj) {
  return JSON.parse(JSON.stringify(obj));
}
