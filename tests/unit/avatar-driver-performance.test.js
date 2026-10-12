// Deterministic driver regressions. Timings here are protocol inputs / a virtual clock,
// not a CPU or headset benchmark. The browser suite exercises real delayed Worker tasks.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { NeuralAvatarDriver, NOTICES, POSE_BUFFER_BYTES, POSE_FLOATS } from '../../webxr/src/net/avatar-driver.js';

class ControlledWorker {
  constructor() { this.messages = []; this.terminated = false; }
  postMessage(message) { this.messages.push(structuredClone(message)); }
  terminate() { this.terminated = true; }
  emit(message) { this.onmessage?.({ data: message }); }
  ready() { this.emit({ type: 'ready', joints: [], parents: [], boneLengths: [], loadMs: 1 }); }
}

const frame = new Float32Array([0, 1.6, 0, 0, 0, 0, 1, 0.25, 1, 0, -0.25, 1, 0]);
const pose = (seq, inferenceMs = 20, value = seq) => {
  const buffer = new ArrayBuffer(POSE_BUFFER_BYTES);
  new Float32Array(buffer, 0, POSE_FLOATS).fill(value);
  return { type: 'pose', seq, tick: seq - 1, t: seq / 30, buffer, inferenceMs,
    mappingMs: inferenceMs / 2, trackingMs: inferenceMs / 2, rootYawS: 0 };
};

async function setup(t, options = {}) {
  const driver = new NeuralAvatarDriver({ WorkerClass: ControlledWorker, ...options });
  t.after(() => driver.dispose());
  const ready = driver.init();
  const worker = driver.worker;
  worker.ready();
  await ready;
  driver.start();
  return { driver, worker };
}

function respond(driver, worker, ms) {
  driver.pushFrameDP(frame, driver.seq / 30);
  worker.emit(pose(driver.seq, ms));
}

for (const [ms, state, every] of [[20, 'running', 1], [25, 'running', 1],
  [25.25, 'slow', 2], [35, 'slow', 2], [60, 'slow', 2], [60.25, 'disabled', 1], [70, 'disabled', 1]]) {
  test(`guard defaults: ${ms} ms -> ${state}, only after ten responses`, async (t) => {
    const { driver, worker } = await setup(t);
    for (let i = 0; i < 9; i++) respond(driver, worker, ms);
    assert.equal(driver.state, 'running', 'nine samples cannot decide the guard');
    respond(driver, worker, ms);
    assert.equal(driver.state, state);
    assert.equal(driver.inferEvery, every);
    assert.equal(driver.stats.meanMs, ms);
    assert.equal(worker.terminated, state === 'disabled');
    if (state === 'disabled') {
      assert.equal(driver.notice, NOTICES.disabledSlow);
      assert.equal(driver.getPose(), null);
    }
  });
}

test('guard uses the rolling 60-response mean and the strict 17.5 ms recovery boundary', async (t) => {
  const { driver, worker } = await setup(t);
  for (let i = 0; i < 60; i++) respond(driver, worker, 20);
  respond(driver, worker, 70);
  assert.equal(driver.state, 'running', 'one slow outlier does not disable a healthy window');
  for (let i = 0; i < 60; i++) respond(driver, worker, 35);
  assert.equal(driver.state, 'slow');
  for (let i = 0; i < 60; i++) respond(driver, worker, 20);
  assert.equal(driver.state, 'slow', '20 ms is below slowdown, but above recovery');
  for (let i = 0; i < 60; i++) respond(driver, worker, 17.5);
  assert.equal(driver.state, 'slow', 'recovery comparison is strict');
  respond(driver, worker, 17);
  assert.equal(driver.state, 'running');
  assert.equal(driver.inferEvery, 1);
});

test('fast inference cannot undo an outstanding render-loop slowdown', async (t) => {
  const { driver, worker } = await setup(t);
  for (let i = 0; i < 90; i++) driver.reportFrameMs(18);
  assert.equal(driver.state, 'slow');
  for (let i = 0; i < 60; i++) respond(driver, worker, 5);
  assert.equal(driver.state, 'slow', 'render median is still 18 ms');
  assert.equal(driver.inferEvery, 2);
  for (let i = 0; i < 89; i++) driver.reportFrameMs(13.9);
  assert.equal(driver.state, 'slow', 'wait for a complete healthy render window');
  driver.reportFrameMs(13.9);
  assert.equal(driver.state, 'running');
  assert.equal(driver.inferEvery, 1);
});

test('stop/start preserves the chosen 15 Hz mode and matching status', async (t) => {
  const { driver, worker } = await setup(t);
  for (let i = 0; i < 10; i++) respond(driver, worker, 35);
  driver.stop();
  assert.equal(driver.state, 'ready');
  driver.start();
  assert.equal(driver.inferEvery, 2);
  assert.equal(driver.state, 'slow');
  assert.equal(driver.notice, NOTICES.slow);
});

test('a partial render window from the previous run cannot disable a healthy restarted run', async (t) => {
  const { driver, worker } = await setup(t);
  for (let i = 0; i < 10; i++) respond(driver, worker, 5);
  for (let i = 0; i < 150; i++) driver.reportFrameMs(30);
  assert.equal(driver.state, 'slow', 'one full bad window plus an incomplete one');
  driver.stop();
  driver.start();
  for (let i = 0; i < 30; i++) driver.reportFrameMs(13.9);
  assert.equal(driver.state, 'slow', 'old frames cannot complete a new disable window');
  for (let i = 0; i < 60; i++) driver.reportFrameMs(13.9);
  assert.equal(driver.state, 'running', 'a complete fresh healthy window permits recovery');
});

for (const action of ['reset', 'stop/start']) {
  test(`late pose across ${action} cannot populate the new avatar or trip its guard`, async (t) => {
    const { driver, worker } = await setup(t, { guard: { minTicks: 1 } });
    driver.pushFrameDP(frame, 0);
    const oldSeq = driver.seq;
    if (action === 'reset') driver.reset();
    else { driver.stop(); driver.start(); }
    driver.pushFrameDP(frame, 1);
    worker.emit(pose(oldSeq, 70, 99));
    assert.equal(driver.state, 'running');
    assert.equal(driver.stats.poses, 0);
    assert.equal(driver.getPose(), null);
    worker.emit(pose(driver.seq, 20, 2));
    assert.equal(driver.getPose()[0], 2);
    assert.equal(driver.stats.poses, 1);
  });
}

test('duplicate or out-of-order replies cannot regress the latest pose', async (t) => {
  const { driver, worker } = await setup(t);
  driver.pushFrameDP(frame, 0);
  driver.pushFrameDP(frame, 1 / 30);
  worker.emit(pose(2, 20, 2));
  worker.emit(pose(1, 70, 1));
  worker.emit(pose(2, 70, 99));
  assert.equal(driver.latest.seq, 2);
  assert.equal(driver.stats.poses, 1);
  assert.equal(driver.getPose()[0], 2);
  assert.equal(driver.stats.meanMs, 20);
});

test('dispose while loading settles init and ignores an already queued ready/error', async (t) => {
  const driver = new NeuralAvatarDriver({ WorkerClass: ControlledWorker });
  t.after(() => driver.dispose());
  let result = 'pending';
  const settled = driver.init().then(() => { result = 'resolved'; }, () => { result = 'rejected'; });
  const worker = driver.worker;
  const queued = worker.onmessage;
  driver.dispose();
  await Promise.resolve();
  assert.equal(result, 'rejected');
  queued({ data: { type: 'ready' } });
  queued({ data: { type: 'error', message: 'late error' } });
  assert.equal(driver.state, 'disposed');
  assert.equal(worker.terminated, true);
  await settled;
});

test('slow init is asynchronous, succeeds within its deadline, and cancels the deadline', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const driver = new NeuralAvatarDriver({ WorkerClass: ControlledWorker, initTimeoutMs: 1000 });
  t.after(() => driver.dispose());
  const pending = driver.init();
  t.mock.timers.tick(999);
  assert.equal(driver.state, 'loading');
  assert.equal(driver.update({}), false, 'loading never enters resampling');
  driver.worker.ready();
  await pending;
  t.mock.timers.tick(1001);
  assert.equal(driver.state, 'ready');
});

test('stalled initialization times out, terminates the worker, and rejects all pending requests', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const driver = new NeuralAvatarDriver({ WorkerClass: ControlledWorker, initTimeoutMs: 1000 });
  t.after(() => driver.dispose());
  let initResult = 'pending', telemetryResult = 'pending';
  const init = driver.init().catch(() => { initResult = 'rejected'; });
  const worker = driver.worker;
  const queued = worker.onmessage;
  const telemetry = driver.exportTelemetry().catch(() => { telemetryResult = 'rejected'; });
  t.mock.timers.tick(1000);
  await Promise.resolve();
  assert.equal(driver.state, 'error');
  assert.equal(initResult, 'rejected');
  assert.equal(telemetryResult, 'rejected');
  assert.equal(worker.terminated, true);
  queued({ data: { type: 'ready' } });
  assert.equal(driver.state, 'error');
  await Promise.all([init, telemetry]);
});

test('missing inference responses time out without a promise or wait in the render path', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { driver, worker } = await setup(t, { responseTimeoutMs: 1000 });
  let frames = 0;
  for (let i = 0; i < 29; i++) {
    assert.equal(driver.pushFrameDP(frame, i / 30), true);
    frames++;
    t.mock.timers.tick(33);
  }
  assert.equal(frames, 29, 'caller continues its render loop while no response arrives');
  assert.equal(driver.state, 'running');
  t.mock.timers.tick(43);
  assert.equal(driver.state, 'error');
  assert.equal(driver.getPose(), null);
  assert.equal(worker.terminated, true);
});

test('responses refresh the watchdog; stop cancels it and 15 Hz skipped frames do not arm it', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { driver, worker } = await setup(t, { responseTimeoutMs: 1000 });
  for (let i = 0; i < 12; i++) {
    driver.pushFrameDP(frame, i / 30);
    t.mock.timers.tick(70);
    worker.emit(pose(driver.seq, 35));
  }
  assert.equal(driver.state, 'slow');
  driver.pushFrameDP(frame, 1);
  t.mock.timers.tick(1500);
  assert.equal(driver.state, 'slow', 'one frame may intentionally be skipped at 15 Hz');
  driver.pushFrameDP(frame, 2);
  t.mock.timers.tick(900);
  driver.stop();
  t.mock.timers.tick(1500);
  assert.equal(driver.state, 'ready');
  assert.equal(worker.terminated, false);
});

for (const terminal of ['disabled', 'error']) {
  test(`${terminal} rejects pending telemetry and cannot be revived by late messages`, async (t) => {
    const { driver, worker } = await setup(t, { guard: { minTicks: 1 } });
    const queued = worker.onmessage;
    let result = 'pending';
    const telemetry = driver.exportTelemetry().then(() => { result = 'resolved'; }, () => { result = 'rejected'; });
    if (terminal === 'disabled') respond(driver, worker, 70);
    else worker.emit({ type: 'error', message: 'inference failed', stage: 'run' });
    await Promise.resolve();
    assert.equal(result, 'rejected');
    assert.equal(worker.terminated, true);
    queued({ data: { type: 'ready' } });
    queued({ data: { type: 'progress', loaded: 1 } });
    queued({ data: pose(100, 5) });
    assert.equal(driver.state, terminal);
    assert.equal(driver.getPose(), null);
    await telemetry;
  });
}
