// scoring.test.js - Scorer behaviour per DESIGN.md section 7: exact playback scores ~100,
// noisy playback clearly lower, standing still is energy-gated, mirrored choreographies,
// timing bias detection, beat events, result shape and the per-tick time budget.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Choreo, createChoreoSample } from '../../webxr/src/game/choreo.js';
import { Scorer, gradeFor, starsFor, precomputeReference, resultToJSON } from '../../webxr/src/game/scoring.js';
import { createSample } from '../../webxr/src/xr/input.js';
import { makeRandom } from '../../webxr/src/emu/emulated-input.js';
import { makeChoreo } from './helpers/make-choreo.js';

const FPS = 30;

/**
 * Feed the scorer a synthetic player stream derived from the choreography's own reference.
 * opts: { lag (s), noise (m, OU sigma), seed, still, mirrorPlayer (player performs the mirrored
 * trajectory), tickHz }
 */
function runPlayer(choreo, opts = {}) {
  const scorer = new Scorer(choreo);
  const beats = [];
  scorer.onBeat = (ev) => beats.push({ ...ev });
  const ref = createChoreoSample();
  const s = createSample();
  const rnd = makeRandom(opts.seed || 7);
  const nz = new Float64Array(9);
  const sigma = opts.noise || 0;
  const tau = 0.25;
  const tickHz = opts.tickHz || FPS;
  const dt = 1 / tickHz;
  let first = true;
  let maxTickMs = 0, totalMs = 0, ticks = 0;
  const n = Math.round(choreo.duration * tickHz);
  for (let i = 0; i <= n; i++) {
    const t = i * dt;
    const tr = opts.still ? 0 : t - (opts.lag || 0);
    choreo.sampleAt(tr, ref);
    let head = ref.head.p, left = ref.left.p, right = ref.right.p;
    if (opts.mirrorPlayer) {
      head = [-ref.head.p[0], ref.head.p[1], ref.head.p[2]];
      left = [-ref.right.p[0], ref.right.p[1], ref.right.p[2]];
      right = [-ref.left.p[0], ref.left.p[1], ref.left.p[2]];
    }
    if (sigma > 0) {
      if (first) { for (let k = 0; k < 9; k++) nz[k] = sigma * rnd.gaussian(); first = false; }
      else {
        const a = Math.exp(-dt / tau), b = sigma * Math.sqrt(1 - a * a);
        for (let k = 0; k < 9; k++) nz[k] = nz[k] * a + b * rnd.gaussian();
      }
    }
    s.head.p[0] = head[0] + nz[0]; s.head.p[1] = head[1] + nz[1]; s.head.p[2] = head[2] + nz[2];
    s.left.p[0] = left[0] + nz[3]; s.left.p[1] = left[1] + nz[4]; s.left.p[2] = left[2] + nz[5];
    s.right.p[0] = right[0] + nz[6]; s.right.p[1] = right[1] + nz[7]; s.right.p[2] = right[2] + nz[8];
    const t0 = performance.now();
    const fs = scorer.scoreTick(s, t);
    const ms = performance.now() - t0;
    if (fs >= 0) { ticks++; totalMs += ms; if (ms > maxTickMs) maxTickMs = ms; }
  }
  const result = scorer.finalize({ mode: 'test', player: 'unit' });
  return { scorer, result, beats, avgTickMs: totalMs / Math.max(1, ticks), maxTickMs, ticks };
}

test('exact playback scores >= 95 with all moves Perfekt', () => {
  const choreo = new Choreo(makeChoreo());
  const { result, beats, ticks } = runPlayer(choreo);
  assert.ok(result.score >= 95, `score ${result.score}`);
  assert.equal(result.stars, 5);
  assert.equal(result.moves.length, 2);
  for (const m of result.moves) {
    assert.equal(m.grade, 'perfekt', `${m.name} ${m.score}`);
    assert.equal(m.gated, false);
  }
  assert.equal(beats.length, 16);
  assert.equal(result.maxCombo, 16);
  assert.ok(Math.abs(result.timingBias) <= 0.02, `timing bias ${result.timingBias}`);
  assert.equal(result.perBeat.length, 16);
  assert.ok(result.perBeat.every((v) => v >= 0.85));
  assert.equal(ticks, 241);
  assert.equal(result.choreoId, 'test-sway');
  assert.equal(result.mode, 'test');
  assert.equal(result.player, 'unit');
  assert.ok(typeof result.playedAt === 'string');
  assert.ok(Math.abs(result.durationSec - 8) < 1e-6);
});

test('playback with 0.35 m noise scores clearly lower (<= 70)', () => {
  const choreo = new Choreo(makeChoreo());
  const exact = runPlayer(choreo).result.score;
  const noisy = runPlayer(choreo, { noise: 0.35, seed: 3 }).result.score;
  assert.ok(noisy <= 70, `noisy score ${noisy}`);
  assert.ok(exact - noisy >= 30, `exact ${exact} vs noisy ${noisy}`);
  // deterministic for the same seed
  assert.equal(runPlayer(choreo, { noise: 0.35, seed: 3 }).result.score, noisy);
  // mild noise (3 cm) still scores well
  const mild = runPlayer(choreo, { noise: 0.03, seed: 5 }).result.score;
  assert.ok(mild >= 80, `mild noise score ${mild}`);
});

test('standing still: energy gate caps every move at <= 40', () => {
  const choreo = new Choreo(makeChoreo());
  const { result } = runPlayer(choreo, { still: true });
  for (const m of result.moves) {
    assert.ok(m.score <= 0.4 + 1e-9, `${m.name} ${m.score}`);
    assert.ok(m.grade === 'ok' || m.grade === 'daneben');
  }
  assert.ok(result.score <= 40, `total ${result.score}`);
  assert.ok(result.stars <= 2);
});

test('mirrored choreography: the mirrored performance scores high, the unmirrored one lower', () => {
  const choreo = new Choreo(makeChoreo({ mirror: true }));
  assert.equal(choreo.mirror, true);
  const mirrored = runPlayer(choreo, { mirrorPlayer: true }).result;
  assert.ok(mirrored.score >= 95, `mirrored ${mirrored.score}`);
  const straight = runPlayer(choreo, { mirrorPlayer: false }).result;
  assert.ok(straight.score < mirrored.score - 10, `straight ${straight.score} vs mirrored ${mirrored.score}`);
  // explicit override: scoring the same choreo unmirrored
  const s2 = new Scorer(choreo, { mirror: false });
  assert.equal(s2.mirror, false);
  const ref = precomputeReference(choreo, true);
  assert.equal(ref.n, choreo.frameCount);
  assert.ok(Math.abs(ref.head[0] + choreo.headP[0]) < 1e-6);
});

test('timing bias: a player 0.1 s late reports +0.1 s, early reports negative', () => {
  const choreo = new Choreo(makeChoreo());
  const late = runPlayer(choreo, { lag: 0.1 }).result;
  assert.ok(Math.abs(late.timingBias - 0.1) <= 0.04, `late bias ${late.timingBias}`);
  assert.ok(late.score >= 90, `late score ${late.score} (lag within the tolerance window)`);
  const early = runPlayer(choreo, { lag: -0.1 }).result;
  assert.ok(Math.abs(early.timingBias + 0.1) <= 0.04, `early bias ${early.timingBias}`);
  // way outside the window the score drops
  const veryLate = runPlayer(choreo, { lag: 0.5 }).result;
  assert.ok(veryLate.score < late.score - 10, `very late ${veryLate.score}`);
});

test('scoreTick stays within the 0.3 ms budget and does not depend on the tick rate', () => {
  const choreo = new Choreo(makeChoreo());
  // warm up the JIT, then measure
  runPlayer(choreo);
  const { avgTickMs, maxTickMs } = runPlayer(choreo);
  assert.ok(avgTickMs < 0.3, `avg tick ${avgTickMs.toFixed(4)} ms (max ${maxTickMs.toFixed(3)})`);
  // 72 Hz ticks (session caps at 30 Hz, but the scorer must cope with irregular rates)
  const fast = runPlayer(choreo, { tickHz: 72 }).result;
  assert.ok(fast.score >= 95, `72 Hz score ${fast.score}`);
  const slow = runPlayer(choreo, { tickHz: 15 }).result;
  assert.ok(slow.score >= 90, `15 Hz score ${slow.score}`);
});

test('beat events, combo, reset and out-of-range ticks', () => {
  const choreo = new Choreo(makeChoreo());
  const scorer = new Scorer(choreo);
  const events = [];
  scorer.onBeat = (ev) => events.push(ev);
  const s = createSample();
  assert.equal(scorer.scoreTick(s, -0.1), -1);
  assert.equal(scorer.scoreTick(s, 9), -1);
  const ref = createChoreoSample();
  // first 2 beats exact, then 2 beats standing at a wrong place, then exact again
  for (let i = 0; i <= 240; i++) {
    const t = i / 30;
    const beat = t * 2;
    choreo.sampleAt(t, ref);
    const wrong = beat >= 2 && beat < 4;
    s.head.p[0] = ref.head.p[0] + (wrong ? 1 : 0); s.head.p[1] = ref.head.p[1]; s.head.p[2] = ref.head.p[2];
    s.left.p[0] = ref.left.p[0]; s.left.p[1] = ref.left.p[1] + (wrong ? 1 : 0); s.left.p[2] = ref.left.p[2];
    s.right.p[0] = ref.right.p[0]; s.right.p[1] = ref.right.p[1]; s.right.p[2] = ref.right.p[2] + (wrong ? 1 : 0);
    scorer.scoreTick(s, t);
    assert.equal(scorer.lastFrameScore >= 0 && scorer.lastFrameScore <= 1, true);
  }
  assert.equal(events.length, 16, 'the tick at t = duration closes the last beat');
  assert.equal(events[0].combo, 1);
  assert.equal(events[1].combo, 2);
  assert.equal(events[2].combo, 0);
  assert.equal(events[2].grade, 'daneben');
  assert.equal(events[4].combo, 1);
  assert.equal(scorer.currentScore() > 0, true);
  const r = scorer.finalize();
  assert.equal(r.maxCombo, 12);
  assert.ok(r.perBeat[2] < 0.4 && r.perBeat[0] > 0.85);
  assert.equal(scorer.finalize(), r, 'finalize is idempotent');
  assert.equal(scorer.scoreTick(s, 1), -1, 'no ticks after finalize');
  scorer.reset();
  assert.equal(scorer.ticks, 0);
  assert.equal(scorer.combo, 0);
  assert.equal(scorer.finalized, false);
  const json = resultToJSON(r);
  assert.ok(Array.isArray(json.perBeat));
  assert.equal(json.perBeat.length, 16);
  assert.equal(JSON.parse(JSON.stringify(json)).score, r.score);
});

test('grade and star thresholds', () => {
  assert.equal(gradeFor(0.85), 'perfekt');
  assert.equal(gradeFor(0.849), 'gut');
  assert.equal(gradeFor(0.65), 'gut');
  assert.equal(gradeFor(0.5), 'ok');
  assert.equal(gradeFor(0.399), 'daneben');
  assert.equal(starsFor(100), 5);
  assert.equal(starsFor(90), 5);
  assert.equal(starsFor(89), 4);
  assert.equal(starsFor(75), 4);
  assert.equal(starsFor(60), 3);
  assert.equal(starsFor(40), 2);
  assert.equal(starsFor(39), 1);
  assert.equal(starsFor(0), 1);
});
