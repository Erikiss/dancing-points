// session.test.js - PlaySession end to end with the playback input backend, a deterministic
// BeatClock and a Calibration captured from the emulated standing pose.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Choreo } from '../../webxr/src/game/choreo.js';
import { BeatClock } from '../../webxr/src/game/clock.js';
import { Scorer } from '../../webxr/src/game/scoring.js';
import { PlaySession } from '../../webxr/src/game/session.js';
import { AudioEngine } from '../../webxr/src/game/audio.js';
import { Calibration } from '../../webxr/src/xr/calibration.js';
import { EmulatedInput } from '../../webxr/src/emu/emulated-input.js';
import { makeChoreo } from './helpers/make-choreo.js';

function makeSource() {
  let now = 10;
  const src = () => now;
  src.advance = (dt) => { now += dt; };
  return src;
}

/** Build a complete session; opts: { noise, lag, playerHeight, playerYaw, playerOrigin, speed, audio } */
function build(opts = {}) {
  const choreo = new Choreo(makeChoreo(opts.choreo || {}));
  const src = makeSource();
  const clock = new BeatClock({ timeSource: src });
  const input = new EmulatedInput({
    backend: 'playback', choreo, clock,
    noise: opts.noise || 0, lag: opts.lag || 0, seed: opts.seed || 1,
    playerHeight: opts.playerHeight, playerYaw: opts.playerYaw, playerOrigin: opts.playerOrigin,
  });
  const calibration = new Calibration();
  input.update(0);                       // standing pose (clock not running)
  assert.ok(calibration.capture(input.sample), 'calibration from the standing pose');
  const scorer = new Scorer(choreo);
  const audio = opts.audio ? new AudioEngine({ audioContext: null }) : null;
  const session = new PlaySession({ choreo, clock, scorer, input, calibration, audio, mode: 'solo', player: 'Tester' });
  return { choreo, src, clock, input, calibration, scorer, session };
}

/** Drive the session at 72 Hz until it finishes (or maxSeconds elapse). */
function drive(ctx, maxSeconds = 20, frameDt = 1 / 72) {
  let elapsed = 0, nowMs = 0;
  while (ctx.session.state !== 'finished' && elapsed < maxSeconds) {
    ctx.session.update(nowMs);
    ctx.src.advance(frameDt);
    nowMs += frameDt * 1000;
    elapsed += frameDt;
  }
  return elapsed;
}

test('full solo run: count-in, events, result >= 95, run recorded', () => {
  const ctx = build({ audio: true });
  const { session, choreo } = ctx;
  const ev = { countdown: [], beat: [], move: [], grade: [], tick: 0, finished: null, start: 0 };
  session.on('start', () => ev.start++);
  session.on('countdown', (e) => ev.countdown.push(e.beatsLeft));
  session.on('beat', (e) => ev.beat.push(e.beat));
  session.on('move', (e) => ev.move.push(e.move.name));
  session.on('grade', (e) => ev.grade.push(e));
  session.on('tick', () => ev.tick++);
  session.on('finished', (r) => { ev.finished = r; });

  assert.equal(session.state, 'idle');
  session.start();
  assert.equal(session.state, 'countdown');
  assert.equal(ev.start, 1);
  session.update(0);
  assert.equal(session.state, 'countdown');
  assert.equal(session.countdownBeatsLeft, 4);
  const elapsed = drive(ctx);
  assert.equal(session.state, 'finished');
  assert.ok(elapsed >= 10 && elapsed < 10.2, `elapsed ${elapsed} (2 s count-in + 8 s)`);
  assert.deepEqual(ev.countdown, [4, 3, 2, 1]);
  assert.deepEqual(ev.beat, Array.from({ length: 16 }, (_, i) => i));
  assert.deepEqual(ev.move, ['Sway', 'Arms']);
  assert.equal(ev.grade.length, 16);
  assert.ok(ev.grade.every((g) => g.grade === 'perfekt'), JSON.stringify(ev.grade.map((g) => g.grade)));
  assert.equal(ev.grade[15].combo, 16);
  assert.ok(ev.tick >= 235 && ev.tick <= 245, `ticks ${ev.tick} (30 Hz over 8 s)`);
  assert.ok(ev.finished);
  assert.equal(ev.finished, session.result);
  assert.ok(session.result.score >= 95, `score ${session.result.score}`);
  assert.equal(session.result.mode, 'solo');
  assert.equal(session.result.player, 'Tester');
  assert.equal(session.score, session.result.score);
  assert.equal(session.combo, 16);
  assert.equal(ctx.clock.running, false);
  // reference sample for the teacher is at the end
  assert.ok(Math.abs(session.refSample.t - choreo.duration) < 0.05);
  // run recorded for the ghost duo
  const run = session.run;
  assert.equal(run.frameCount, 241);
  assert.ok(run.filled.every((f) => f > 0), 'all run frames filled (sampled or hold-filled)');
  const sampled = run.filled.reduce((a, f) => a + (f === 1 ? 1 : 0), 0);
  assert.ok(sampled >= 230, `sampled frames ${sampled}`);
  const json = session.runToJSON();
  assert.equal(json.format, 'dancing-points-run/1');
  assert.equal(json.choreoId, 'test-sway');
  assert.equal(json.head.length, 241);
  assert.equal(json.head[0].length, 7);
  assert.equal(json.left[0].length, 3);
  assert.equal(json.score, session.result.score);
  // stage-frame head of the run is close to the reference
  assert.ok(Math.abs(json.head[100][1] - choreo.headP[100 * 3 + 1]) < 0.02);
  // further updates are no-ops
  assert.equal(session.update(99999), 'finished');
  session.dispose();
});

test('calibration round trip: a short, rotated, displaced virtual player still scores >= 95', () => {
  const ctx = build({ playerHeight: 1.45, playerYaw: 2.1, playerOrigin: [1.5, 0, -2.0] });
  assert.ok(Math.abs(ctx.calibration.h0 - 1.45) < 1e-9);
  assert.ok(Math.abs(ctx.calibration.yaw - 2.1) < 1e-9);
  ctx.session.start();
  drive(ctx);
  assert.equal(ctx.session.state, 'finished');
  assert.ok(ctx.session.result.score >= 95, `score ${ctx.session.result.score}`);
  assert.ok(Math.abs(ctx.session.playerS.k - 1.7 / 1.45) < 1e-9);
});

test('noise 0.35 m scores <= 70 and lag 0.1 s shows in the timing bias', () => {
  const noisy = build({ noise: 0.35, seed: 11 });
  noisy.session.start();
  drive(noisy);
  assert.ok(noisy.session.result.score <= 70, `noisy ${noisy.session.result.score}`);
  const late = build({ lag: 0.1 });
  late.session.start();
  drive(late);
  assert.ok(Math.abs(late.session.result.timingBias - 0.1) <= 0.04, `bias ${late.session.result.timingBias}`);
});

test('speed factor 10 finishes in a tenth of the wall time with the same score', () => {
  const ctx = build();
  ctx.session.start({ speed: 10 });
  const elapsed = drive(ctx, 5, 1 / 720);
  assert.equal(ctx.session.state, 'finished');
  assert.ok(elapsed < 1.1, `elapsed ${elapsed}`);
  assert.ok(ctx.session.result.score >= 95);
});

test('abort, pause/resume and zero count-in', () => {
  const ctx = build();
  const { session, src } = ctx;
  let aborted = null;
  session.on('abort', (e) => { aborted = e; });
  session.start();
  src.advance(3);
  session.update(3000);
  assert.equal(session.state, 'playing');
  assert.ok(session.progress > 0.1 && session.progress < 0.2, `progress ${session.progress}`);
  session.pause();
  assert.equal(session.paused, true);
  src.advance(5);
  session.update(8000);
  assert.ok(Math.abs(session.t - 1) < 0.02, `paused t ${session.t}`);
  session.resume();
  session.abort();
  assert.equal(session.state, 'aborted');
  assert.ok(aborted && Math.abs(aborted.t - 1) < 0.02);
  assert.equal(session.result, null);
  session.abort();
  assert.equal(session.state, 'aborted');

  const ctx2 = build({ choreo: { countInBeats: 0 } });
  ctx2.session.start();
  assert.equal(ctx2.session.state, 'playing');
  drive(ctx2);
  assert.equal(ctx2.session.state, 'finished');
  assert.ok(ctx2.session.result.score >= 95);
});

test('constructor validates dependencies', () => {
  assert.throws(() => new PlaySession({}), TypeError);
});
