// emulated-input.test.js - playback and desktop input backends (headless).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { EmulatedInput, makeRandom } from '../../webxr/src/emu/emulated-input.js';
import { Choreo, createChoreoSample } from '../../webxr/src/game/choreo.js';
import { Calibration } from '../../webxr/src/xr/calibration.js';
import { BeatClock } from '../../webxr/src/game/clock.js';
import { forwardFromQuat } from '../../webxr/src/util/math.js';
import { makeChoreo } from './helpers/make-choreo.js';

const near = (a, b, eps = 1e-6) => Math.abs(a - b) <= eps;
const vecNear = (a, b, eps = 1e-6) => a.every((x, i) => near(x, b[i], eps));

/** Minimal EventTarget stand-in for the desktop backend. */
function fakeWindow() {
  const handlers = new Map();
  const doc = { pointerLockElement: null };
  return {
    document: doc,
    addEventListener: (type, fn) => { handlers.set(type, fn); },
    removeEventListener: (type) => { handlers.delete(type); },
    dispatch: (type, ev) => { const h = handlers.get(type); if (h) h(ev); },
    handlers,
  };
}

test('deterministic PRNG', () => {
  const a = makeRandom(42), b = makeRandom(42), c = makeRandom(43);
  const xa = [a.next(), a.next(), a.gaussian()];
  const xb = [b.next(), b.next(), b.gaussian()];
  assert.deepEqual(xa, xb);
  assert.notDeepEqual(xa, [c.next(), c.next(), c.gaussian()]);
  assert.ok(xa.slice(0, 2).every((v) => v >= 0 && v < 1));
  // gaussian statistics
  const r = makeRandom(1);
  let s = 0, s2 = 0;
  const n = 20000;
  for (let i = 0; i < n; i++) { const g = r.gaussian(); s += g; s2 += g * g; }
  assert.ok(Math.abs(s / n) < 0.03);
  assert.ok(Math.abs(s2 / n - 1) < 0.05);
});

test('playback: standing pose before start, exact reference during play, lag applied', () => {
  const choreo = new Choreo(makeChoreo());
  const input = new EmulatedInput({ backend: 'playback', choreo });
  assert.equal(input.emulated, true);
  const s = input.update(0);
  assert.equal(input.standing, true);
  assert.ok(vecNear(s.head.p, [0, 1.7, 0]), `standing head ${s.head.p}`);
  const f = forwardFromQuat([0, 0, 0], s.head.q);
  assert.ok(vecNear(f, [0, 0, -1]));
  assert.ok(s.left.p[0] < 0 && s.right.p[0] > 0);
  assert.equal(s.head.valid && s.left.valid && s.right.valid, true);
  // explicit time (no clock)
  const ref = createChoreoSample();
  for (const t of [0, 1.234, 5, 7.99]) {
    input.setTime(t);
    input.update(0);
    choreo.sampleAt(t, ref);
    assert.ok(vecNear(input.sample.head.p, ref.head.p), `head at ${t}`);
    assert.ok(vecNear(input.sample.head.q, ref.head.q), `head q at ${t}`);
    assert.ok(vecNear(input.sample.left.p, ref.left.p), `left at ${t}`);
    assert.ok(vecNear(input.sample.right.p, ref.right.p), `right at ${t}`);
    assert.equal(input.standing, false);
  }
  // lag: sample at t equals reference at t - lag
  const lagged = new EmulatedInput({ backend: 'playback', choreo, lag: 0.1 });
  lagged.setTime(2.0);
  lagged.update(0);
  choreo.sampleAt(1.9, ref);
  assert.ok(vecNear(lagged.sample.head.p, ref.head.p));
  // with a clock: standing until the clock runs
  const src = { now: 5 };
  const clock = new BeatClock({ timeSource: () => src.now });
  const clocked = new EmulatedInput({ backend: 'playback', choreo, clock });
  clocked.update(0);
  assert.equal(clocked.standing, true);
  clock.start(120, 4);
  src.now += 2.5; // t = 0.5
  clocked.update(0);
  choreo.sampleAt(0.5, ref);
  assert.ok(vecNear(clocked.sample.head.p, ref.head.p));
  assert.throws(() => new EmulatedInput({ backend: 'playback' }), /choreo/);
});

test('playback through a virtual player: calibration from the standing pose recovers the reference', () => {
  const choreo = new Choreo(makeChoreo());
  const input = new EmulatedInput({ backend: 'playback', choreo, playerHeight: 1.5, playerYaw: -1.3, playerOrigin: [0.7, 0, 2.2] });
  input.update(0);
  assert.ok(vecNear(input.sample.head.p, [0.7, 1.5, 2.2]), `standing head ${input.sample.head.p}`);
  const cal = new Calibration();
  const data = cal.capture(input.sample);
  assert.ok(near(data.h0, 1.5) && near(data.yaw, -1.3));
  const ref = createChoreoSample();
  for (const t of [0.3, 4.4, 7.5]) {
    input.setTime(t);
    input.update(0);
    const s = cal.toStage(input.sample, choreo.referenceHeight);
    choreo.sampleAt(t, ref);
    assert.ok(vecNear(s.head.p, ref.head.p, 1e-6), `head ${s.head.p} vs ${ref.head.p}`);
    assert.ok(vecNear(s.left.p, ref.left.p, 1e-6));
    assert.ok(vecNear(s.right.p, ref.right.p, 1e-6));
    assert.ok(vecNear(s.head.q, ref.head.q, 1e-6));
    assert.ok(near(s.k, 1.7 / 1.5));
  }
  // copying an existing calibration as the virtual player
  const input2 = new EmulatedInput({ backend: 'playback', choreo, calibration: cal });
  input2.update(0);
  assert.ok(vecNear(input2.sample.head.p, [0.7, 1.5, 2.2]));
});

test('playback noise: deterministic per seed, stationary sigma about the requested value', () => {
  const choreo = new Choreo(makeChoreo());
  const mk = (seed) => new EmulatedInput({ backend: 'playback', choreo, noise: 0.35, seed });
  const a = mk(9), b = mk(9), c = mk(10);
  const ref = createChoreoSample();
  let sum2 = 0, n = 0, maxStep = 0;
  let prev = null;
  for (let i = 0; i <= 240 * 4; i++) {
    const t = (i / 120) % 8;   // wrap around: noise re-initialises when time goes backwards
    a.setTime(t); b.setTime(t); c.setTime(t);
    a.update(0); b.update(0); c.update(0);
    assert.ok(vecNear(a.sample.head.p, b.sample.head.p, 0), 'same seed -> identical');
    if (i === 3) assert.ok(!vecNear(a.sample.head.p, c.sample.head.p, 1e-9), 'different seed -> different');
    choreo.sampleAt(t, ref);
    for (let k = 0; k < 3; k++) {
      const d = a.sample.head.p[k] - ref.head.p[k];
      sum2 += d * d; n++;
    }
    if (prev !== null && i % 960 !== 0) maxStep = Math.max(maxStep, Math.abs(a.sample.head.p[0] - prev));
    prev = a.sample.head.p[0];
  }
  const sigma = Math.sqrt(sum2 / n);
  assert.ok(sigma > 0.25 && sigma < 0.45, `sigma ${sigma}`);
  assert.ok(maxStep < 0.6, `noise is temporally smooth (max step ${maxStep})`);
});

test('desktop: WASD moves the head, Q raises the left hand, Space is the trigger', () => {
  const win = fakeWindow();
  const input = new EmulatedInput({ backend: 'desktop', window: win });
  assert.equal(input.backend, 'desktop');
  assert.ok(win.handlers.has('keydown'));
  let s = input.update(0);
  assert.ok(vecNear(s.head.p, [0, 1.7, 0]));
  assert.ok(s.left.p[0] < s.head.p[0] && s.right.p[0] > s.head.p[0]);
  assert.ok(s.left.p[2] < s.head.p[2], 'hands in front (-Z)');
  win.dispatch('keydown', { code: 'KeyW', preventDefault() {} });
  for (let ms = 16; ms <= 1000; ms += 16) input.update(ms);
  s = input.sample;
  assert.ok(s.head.p[2] < -1.0 && s.head.p[2] > -1.3, `moved forward ${s.head.p[2]}`);
  assert.ok(near(s.head.p[0], 0, 1e-9));
  win.dispatch('keyup', { code: 'KeyW' });
  // look right by 90 degrees then walk forward -> moves along -X
  win.dispatch('mousedown', { button: 0 });
  win.dispatch('mousemove', { movementX: (Math.PI / 2) / input.lookSpeed, movementY: 0, buttons: 1 });
  win.dispatch('mouseup', { button: 0 });
  assert.ok(near(input.yaw, -Math.PI / 2, 1e-9), `yaw ${input.yaw}`);
  const z0 = s.head.p[2], x0 = s.head.p[0];
  win.dispatch('keydown', { code: 'KeyW', preventDefault() {} });
  for (let ms = 1016; ms <= 1500; ms += 16) input.update(ms);
  assert.ok(near(s.head.p[2], z0, 1e-6), 'no z change');
  assert.ok(s.head.p[0] > x0 + 0.4, `moved along +X: ${s.head.p[0]}`);
  win.dispatch('keyup', { code: 'KeyW' });
  // raise left hand
  const yBefore = s.left.p[1];
  win.dispatch('keydown', { code: 'KeyQ', preventDefault() {} });
  for (let ms = 1516; ms <= 2000; ms += 16) input.update(ms);
  assert.ok(s.left.p[1] > yBefore + 0.5, `left hand raised ${s.left.p[1]} vs ${yBefore}`);
  assert.ok(near(s.right.p[1], s.head.p[1] - 0.45, 1e-6));
  // trigger edge detection
  assert.equal(input.isDown('any', 'trigger'), false);
  win.dispatch('keydown', { code: 'Space', preventDefault() {} });
  input.update(2016);
  assert.equal(input.justPressed('any', 'trigger'), true);
  assert.equal(input.isDown('right', 'trigger'), true);
  input.update(2032);
  assert.equal(input.justPressed('any', 'trigger'), false);
  win.dispatch('keyup', { code: 'Space' });
  input.update(2048);
  assert.equal(input.anyTriggerDown(), false);
  // programmatic button
  input.setButton('left', 'primary', true);
  assert.equal(input.isDown('left', 'primary'), true);
  win.dispatch('blur', {});
  assert.equal(input.keys.size, 0);
  input.dispose();
  assert.equal(win.handlers.size, 0);
});
