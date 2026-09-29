// clock.test.js - BeatClock with an injected time source.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BeatClock } from '../../webxr/src/game/clock.js';

const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;

function makeSource(start = 100) {
  let now = start;
  const src = () => now;
  src.advance = (dt) => { now += dt; };
  src.set = (v) => { now = v; };
  return src;
}

test('count-in, beats and bars', () => {
  const src = makeSource();
  const clock = new BeatClock({ timeSource: src });
  assert.equal(clock.running, false);
  clock.start(120, 4);
  let s = clock.now();
  assert.ok(near(s.t, -2), `t at start ${s.t}`);
  assert.ok(near(s.beat, -4));
  assert.ok(near(s.countIn, 4));
  assert.equal(s.running, true);
  src.advance(2);
  s = clock.now();
  assert.ok(near(s.t, 0));
  assert.ok(near(s.beat, 0));
  assert.equal(s.bar, 0);
  assert.equal(s.countIn, 0);
  src.advance(2.25);
  s = clock.now();
  assert.ok(near(s.t, 2.25));
  assert.ok(near(s.beat, 4.5));
  assert.equal(s.bar, 1);
  assert.ok(near(clock.beatToTime(3), 1.5));
  assert.ok(near(clock.timeToBeat(1.5), 3));
  assert.ok(near(clock.countInSeconds, 2));
  // same reused state object
  assert.equal(clock.now(), s);
});

test('source <-> timeline mapping and beatToSource', () => {
  const src = makeSource(50);
  const clock = new BeatClock({ timeSource: src });
  clock.start(90, 2, { beatsPerBar: 4 });
  const beatDur = 60 / 90;
  assert.ok(near(clock.beatToSource(0), 50 + 2 * beatDur));
  assert.ok(near(clock.beatToSource(-2), 50));
  assert.ok(near(clock.beatToSource(8), 50 + 10 * beatDur));
  assert.ok(near(clock.sourceToTime(clock.timeToSource(3.3)), 3.3));
  assert.ok(near(clock.timeToSource(clock.sourceToTime(77)), 77));
  // startAt in the past shifts the timeline
  clock.start(90, 0, { startAt: 40 });
  assert.ok(near(clock.now().t, 10));
});

test('pause / resume / stop / seek', () => {
  const src = makeSource();
  const clock = new BeatClock({ timeSource: src });
  clock.start(60, 0);
  src.advance(1);
  clock.pause();
  assert.equal(clock.paused, true);
  src.advance(5);
  assert.ok(near(clock.now().t, 1), 'paused time does not advance');
  assert.equal(clock.now().paused, true);
  clock.resume();
  assert.ok(near(clock.now().t, 1));
  src.advance(1);
  assert.ok(near(clock.now().t, 2));
  // beatToSource accounts for the pause
  assert.ok(near(clock.beatToSource(2), src()));
  clock.seek(10);
  assert.ok(near(clock.now().t, 10));
  src.advance(0.5);
  assert.ok(near(clock.now().t, 10.5));
  clock.stop();
  assert.equal(clock.running, false);
  assert.ok(near(clock.now().t, 0));
  // pause when not running is a no-op
  clock.pause();
  assert.equal(clock.paused, false);
});

test('speed factor and audioContext time source', () => {
  const src = makeSource(0);
  const clock = new BeatClock({ timeSource: src, speed: 10 });
  clock.start(120, 4);
  src.advance(0.2); // 2 s of game time
  assert.ok(near(clock.now().t, 0));
  src.advance(0.1);
  assert.ok(near(clock.now().t, 1));
  // speed via start opts
  clock.start(120, 0, { speed: 2 });
  src.advance(1);
  assert.ok(near(clock.now().t, 2));
  // AudioContext-like source
  const ctx = { currentTime: 5 };
  const c2 = new BeatClock({ audioContext: ctx });
  c2.start(100, 1);
  ctx.currentTime = 5.6;
  assert.ok(near(c2.now().t, 0));
  assert.ok(near(c2.beatToSource(0), 5.6));
  assert.throws(() => c2.start(0, 4), RangeError);
});

test('default time source works without injection', () => {
  const clock = new BeatClock();
  clock.start(120, 0);
  const t0 = clock.now().t;
  assert.ok(t0 >= -1e-3 && t0 < 1, `t0 ${t0}`);
});
