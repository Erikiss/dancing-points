// CPU-only asynchronous regressions for the production inference worker. Existing fp32 output
// fixtures are replayed by a controllable ORT adapter, not by a fake worker or a rewritten
// inference loop. These tests check state / tensor consistency; they do not benchmark ONNX.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { readFileSync } from 'node:fs';
import { DPPipeline, MAPPING_INPUT_NAMES, TRACKING_INPUT_NAMES, TRACKING_OUTPUT_NAMES,
  N_JOINTS } from '../../webxr/src/net/pipeline.js';

const json = (url) => JSON.parse(readFileSync(new URL(url, import.meta.url), 'utf8'));
const mapping = json('../fixtures/mapping_io.json');
const tracking = json('../fixtures/tracking_io.json');
const skeleton = json('../../webxr/models/free/skeleton.json');
const initPose = json('../../webxr/models/free/init_pose.json');
const mappingOutputs = Object.fromEntries(['Positions', 'RootMotion'].map((n) => [n, Float32Array.from(mapping.outputsFp32[n].flat(Infinity))]));
const trackingOutputs = Object.fromEntries(TRACKING_OUTPUT_NAMES.map((n) => [n, Float32Array.from(tracking.outputsFp32[n].flat(Infinity))]));

class WorkerHarness {
  constructor(t) {
    this.events = [];
    this.waiters = [];
    this.barrierId = 0;
    this.worker = new Worker(new URL('./helpers/worker-regression-host.js', import.meta.url));
    this.worker.on('message', (event) => {
      this.events.push(event);
      for (const waiter of [...this.waiters]) {
        if (waiter.predicate(event)) waiter.resolve(event);
      }
    });
    this.worker.on('error', (error) => {
      this.error = error;
      for (const waiter of [...this.waiters]) waiter.reject(error);
    });
    t.after(() => this.worker.terminate());
  }

  send(message) { this.worker.postMessage(message); }

  waitFor(predicate) {
    if (this.error) return Promise.reject(this.error);
    const previous = this.events.find(predicate);
    if (previous) return Promise.resolve(previous);
    return new Promise((resolve, reject) => {
      const done = (fn, value) => {
        clearTimeout(timer);
        this.waiters.splice(this.waiters.indexOf(waiter), 1);
        fn(value);
      };
      const waiter = { predicate, resolve: (e) => done(resolve, e), reject: (e) => done(reject, e) };
      const timer = setTimeout(() => waiter.reject(new Error('worker event timeout; saw ' + this.events.map((e) => e.type + ':' + (e.id || e.seq || '')).join(', '))), 5000);
      this.waiters.push(waiter);
    });
  }

  async barrier() {
    const id = ++this.barrierId;
    this.send({ type: 'test:barrier', id });
    await this.waitFor((e) => e.type === 'test:barrier' && e.id === id);
  }

  stage(id, phase = 'start') { return this.waitFor((e) => e.type === 'test:stage' && e.id === id && e.phase === phase); }
  release(id, error) { this.send({ type: 'test:release', id, error }); }
  frame(frame, seq) { this.send({ type: 'frame', frame, seq, t: seq / 30 }); }
  pose(seq) { return this.waitFor((e) => e.type === 'pose' && e.seq === seq); }

  async init(options = {}, waitReady = true) {
    await this.waitFor((e) => e.type === 'test:host-ready');
    this.send({ type: 'test:configure', ...options });
    this.send({ type: 'init', modelsUrl: new URL('../../webxr/models/free/', import.meta.url).href,
      ortUrl: new URL('./helpers/worker-fixture-ort.js', import.meta.url).href,
      numThreads: 1, telemetry: true });
    if (waitReady) await this.waitFor((e) => e.type === 'ready');
  }

  assertNoError() { assert.deepEqual(this.events.filter((e) => e.type === 'error'), []); }
  assertNoPose() { assert.deepEqual(this.events.filter((e) => e.type === 'pose'), []); }
}

function frameAt(x) {
  const frame = Float32Array.from(mapping.framesDP[0].flat(Infinity));
  // Translate all three tracked points together. A mix of old mapping and new roots is obvious.
  for (const i of [0, 7, 10]) frame[i] += x;
  return frame;
}

function near(actual, expected, label) {
  assert.equal(actual.length, expected.length, label + ' length');
  let max = 0;
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Number.isFinite(actual[i]), label + ' finite');
    max = Math.max(max, Math.abs(actual[i] - expected[i]));
  }
  assert.ok(max < 1e-5, label + ' max abs error ' + max);
}

function assertPose(actual, expected) {
  near(new Float32Array(actual.buffer, 0, N_JOINTS * 3), expected.positionsWorldS, 'pose');
  near(new Float32Array(actual.buffer, N_JOINTS * 3 * 4, 3), expected.rootS, 'root');
  assert.equal(actual.tick, expected.tick, 'pose tick belongs to its input seq');
}

function finishReference(pipe) {
  pipe.applyMappingOutputs(mappingOutputs);
  const feeds = pipe.buildTrackingInputs();
  const trackingFeeds = Object.fromEntries(TRACKING_INPUT_NAMES.map((n) => [n, Array.from(feeds[n])]));
  const pose = structuredClone(pipe.applyTrackingOutputs(trackingOutputs));
  return { trackingFeeds, pose };
}

test('worker: busy frames retain history without changing an in-flight pose, inputs, root or tick', async (t) => {
  const h = new WorkerHarness(t);
  await h.init();
  h.send({ type: 'test:configure', gateRuns: true });
  const pipe = new DPPipeline({ skeleton, initPose });
  const frames = [frameAt(0), frameAt(3), frameAt(6), frameAt(9)];
  pipe.pushFrame(frames[0]);
  const mappingFeeds = pipe.buildMappingInputs();
  const expected = finishReference(pipe);

  h.frame(frames[0], 1);
  const firstMapping = await h.stage('mapping.run.2');
  for (const name of MAPPING_INPUT_NAMES) near(firstMapping.feeds[name], mappingFeeds[name], name);
  h.frame(frames[1], 2);
  await h.barrier();
  h.release('mapping.run.2');
  const firstTracking = await h.stage('tracking.run.2');
  for (const name of TRACKING_INPUT_NAMES) near(firstTracking.feeds[name], expected.trackingFeeds[name], name);
  h.frame(frames[2], 3);
  await h.barrier();
  h.release('tracking.run.2');
  const firstPose = await h.pose(1);
  assertPose(firstPose, expected.pose);
  assert.equal(firstPose.t, 1 / 30);
  assert.equal(firstPose.dropped, 2);

  // Busy frames must still appear in the *next* mapping history, even though they had no pose.
  for (const f of frames.slice(1)) pipe.pushFrame(f);
  const nextMappingFeeds = pipe.buildMappingInputs();
  const nextExpected = finishReference(pipe);
  h.frame(frames[3], 4);
  const nextMapping = await h.stage('mapping.run.3');
  for (const name of MAPPING_INPUT_NAMES) near(nextMapping.feeds[name], nextMappingFeeds[name], name);
  h.release('mapping.run.3');
  const nextTracking = await h.stage('tracking.run.3');
  for (const name of TRACKING_INPUT_NAMES) near(nextTracking.feeds[name], nextExpected.trackingFeeds[name], name);
  h.release('tracking.run.3');
  assertPose(await h.pose(4), nextExpected.pose);
  h.send({ type: 'telemetry' });
  const telemetry = await h.waitFor((e) => e.type === 'telemetry');
  assert.deepEqual(telemetry.data.ticks.map((tick) => tick.tick), [0, 3]);
  h.assertNoError();
});

test('worker: 30 Hz input yields 30 / 15 / 30 poses as inferEvery changes, preserving every history frame', async (t) => {
  const h = new WorkerHarness(t);
  await h.init();
  h.send({ type: 'test:configure', gateRuns: true });
  const pipe = new DPPipeline({ skeleton, initPose });
  const expectedTicks = [];
  let seq = 0;
  let run = 1; // Each session's first run is the initialization warm-up.
  for (const inferEvery of [1, 2, 1]) {
    h.send({ type: 'config', inferEvery });
    const firstSeq = seq + 1;
    for (let i = 0; i < 30; i++) {
      const frame = frameAt(++seq / 10);
      pipe.pushFrame(frame); // Include skipped inference ticks in the independent history.
      h.frame(frame, seq); // t = seq / 30; no wall-clock timing or CPU throughput assumption.
      if (seq % inferEvery !== 0) {
        await h.barrier();
        assert.equal(h.events.filter((e) => e.type === 'test:stage' && e.phase === 'start' && e.id.startsWith('mapping.run.')).length,
          run, 'the skipped tick must not start a model run');
        continue;
      }
      const feeds = pipe.buildMappingInputs();
      const expected = finishReference(pipe);
      const id = ++run;
      const actualMapping = await h.stage('mapping.run.' + id);
      for (const name of MAPPING_INPUT_NAMES) near(actualMapping.feeds[name], feeds[name], 'all-frame history: ' + name);
      h.release('mapping.run.' + id);
      await h.stage('tracking.run.' + id);
      h.release('tracking.run.' + id);
      const pose = await h.pose(seq); // Acknowledge completion before posting another input.
      assertPose(pose, expected.pose);
      assert.equal(pose.dropped, 0, 'test input never overlaps a running inference');
      expectedTicks.push(seq - 1);
    }
    const poses = h.events.filter((e) => e.type === 'pose' && e.seq >= firstSeq && e.seq <= seq);
    assert.equal(poses.length, 30 / inferEvery, `30 input ticks at inferEvery=${inferEvery}`);
  }
  h.send({ type: 'telemetry' });
  const telemetry = await h.waitFor((e) => e.type === 'telemetry');
  assert.deepEqual(telemetry.data.ticks.map((tick) => tick.tick), expectedTicks);
  assert.equal(expectedTicks.length, 75);
  h.assertNoError();
});

for (const stage of ['mapping', 'tracking']) {
  test(`worker: reset while ${stage} awaits discards the old run and starts clean at the new root`, async (t) => {
    const h = new WorkerHarness(t);
    await h.init();
    h.send({ type: 'test:configure', gateRuns: true });
    h.frame(frameAt(0), 1);
    await h.stage('mapping.run.2');
    if (stage === 'tracking') {
      h.release('mapping.run.2');
      await h.stage('tracking.run.2');
    }
    h.send({ type: 'reset' }); // Driver.stop() uses the same protocol message.
    h.frame(frameAt(10), 2);
    await h.barrier();
    h.release(stage + '.run.2');
    await h.barrier();
    h.assertNoError();
    h.assertNoPose();

    h.frame(frameAt(10), 3);
    await h.stage('mapping.run.3');
    h.release('mapping.run.3');
    const trackingId = stage === 'mapping' ? 'tracking.run.2' : 'tracking.run.3';
    await h.stage(trackingId);
    h.release(trackingId);
    const actual = await h.pose(3);
    const pipe = new DPPipeline({ skeleton, initPose });
    pipe.pushFrame(frameAt(10));
    pipe.pushFrame(frameAt(10));
    pipe.buildMappingInputs();
    assertPose(actual, finishReference(pipe).pose);
    assert.equal(actual.inferCount, 1);
    h.assertNoError();
  });
}

test('worker: a rejected old inference after reset cannot fail the new run', async (t) => {
  const h = new WorkerHarness(t);
  await h.init();
  h.send({ type: 'test:configure', gateRuns: true });
  h.frame(frameAt(0), 1);
  await h.stage('mapping.run.2');
  h.send({ type: 'reset' });
  h.release('mapping.run.2', 'late failure from cancelled inference');
  await h.barrier();
  h.assertNoError();
  h.assertNoPose();
  h.frame(frameAt(4), 2);
  await h.stage('mapping.run.3');
  h.release('mapping.run.3');
  await h.stage('tracking.run.2');
  h.release('tracking.run.2');
  assert.equal((await h.pose(2)).tick, 0);
});

for (const stage of ['mapping', 'tracking']) {
  test(`worker: dispose during ${stage} waits for run completion before releasing sessions`, async (t) => {
    const h = new WorkerHarness(t);
    await h.init();
    h.send({ type: 'test:configure', gateRuns: true });
    h.frame(frameAt(0), 1);
    await h.stage('mapping.run.2');
    if (stage === 'tracking') {
      h.release('mapping.run.2');
      await h.stage('tracking.run.2');
    }
    h.send({ type: 'dispose' });
    await h.barrier();
    assert.deepEqual(h.events.filter((e) => e.type === 'test:released'), [], 'do not release a running session');
    h.release(stage + '.run.2');
    await h.waitFor((e) => e.type === 'test:released' && e.model === 'tracking');
    await h.barrier();
    const released = h.events.filter((e) => e.type === 'test:released');
    assert.deepEqual(released.map((e) => [e.model, e.activeRuns, e.alreadyReleased]), [['mapping', 0, false], ['tracking', 0, false]]);
    h.assertNoPose();
    h.assertNoError();
  });
}

for (const stage of ['mapping.create.1', 'tracking.create.1', 'mapping.run.1', 'tracking.run.1']) {
  test(`worker: dispose during slow initialization (${stage}) cannot emit ready or leak sessions`, async (t) => {
    const h = new WorkerHarness(t);
    await h.init({ hold: [stage] }, false);
    await h.stage(stage);
    h.send({ type: 'dispose' });
    await h.barrier();
    assert.deepEqual(h.events.filter((e) => e.type === 'test:released'), [], 'finish pending create/warm-up before release');
    h.release(stage);
    await h.waitFor((e) => e.type === 'test:released' && e.model === 'mapping');
    await h.barrier();
    assert.deepEqual(h.events.filter((e) => e.type === 'ready'), []);
    const models = stage === 'mapping.create.1' ? ['mapping'] : ['mapping', 'tracking'];
    const released = h.events.filter((e) => e.type === 'test:released');
    assert.deepEqual(released.map((e) => [e.model, e.activeRuns, e.alreadyReleased]), models.map((m) => [m, 0, false]));
    h.assertNoError();
  });
}
