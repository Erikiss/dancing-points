// pipeline.test.js - parity of webxr/src/net/pipeline.js with the network I/O fixtures
// tests/fixtures/mapping_io.json and tracking_io.json (numpy oracle + fp32 ONNX outputs), and
// plumbing tests of the main-thread driver (webxr/src/net/avatar-driver.js) with a fake worker.
// node --test. Tolerances per DESIGN.md 6.4: input tensors 1e-5, post-processing 1e-4.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import * as P from '../../webxr/src/net/pipeline.js';
import { NeuralAvatarDriver, NOTICES } from '../../webxr/src/net/avatar-driver.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..', '..');
const mapping = JSON.parse(readFileSync(join(root, 'tests', 'fixtures', 'mapping_io.json'), 'utf8'));
const tracking = JSON.parse(readFileSync(join(root, 'tests', 'fixtures', 'tracking_io.json'), 'utf8'));
const skeleton = JSON.parse(readFileSync(join(root, 'webxr', 'models', 'free', 'skeleton.json'), 'utf8'));
const initPose = JSON.parse(readFileSync(join(root, 'webxr', 'models', 'free', 'init_pose.json'), 'utf8'));

const TOL_INPUT = 1e-5;
const TOL_POST = 1e-4;
const achieved = {}; // label -> max abs error

function flat(nested) {
  const out = [];
  P.flattenInto(out, Array.isArray(nested) ? nested : Array.from(nested));
  return out;
}

function maxAbsDiff(a, b) {
  const fa = flat(a);
  const fb = flat(b);
  assert.equal(fa.length, fb.length, 'length mismatch');
  let m = 0;
  for (let i = 0; i < fa.length; i++) {
    assert.ok(Number.isFinite(fa[i]), 'non-finite value at ' + i);
    m = Math.max(m, Math.abs(fa[i] - fb[i]));
  }
  return m;
}

function expectNear(got, exp, label, tol) {
  const err = maxAbsDiff(got, exp);
  achieved[label] = Math.max(achieved[label] || 0, err);
  assert.ok(err <= tol, `${label}: max abs error ${err} > ${tol}`);
  return err;
}

function f32(nested) {
  return Float32Array.from(flat(nested));
}

// ---------------------------------------------------------------------------------------------
// mapping_io.json

test('mapping_io: input tensors from the 15-frame window (channel-major, 1e-5)', () => {
  const frames = new Float64Array(P.N_HISTORY * P.FRAME_DIM);
  P.flattenInto(frames, mapping.framesDP);
  const work = P.createMappingWork();
  const feeds = P.mappingInputs(frames, mapping.params.prevYaw, work);
  expectNear(work.rootPos, mapping.roots.rootPos, 'mapping roots.rootPos', TOL_INPUT);
  expectNear(work.yaw, mapping.roots.yaw, 'mapping roots.yaw', TOL_INPUT);
  expectNear(work.rm, mapping.roots.rm, 'mapping roots.rm', TOL_INPUT);
  expectNear(work.rmRef, mapping.roots.rmRef, 'mapping roots.rmRef', TOL_INPUT);
  expectNear(work.rmRel, mapping.roots.rmRel, 'mapping roots.rmRel', TOL_INPUT);
  expectNear(work.local, mapping.roots.local, 'mapping roots.local', TOL_INPUT);
  assert.equal(feeds.input_leader_Positions.length, 135);
  assert.equal(feeds.input_leader_RootMotion.length, 60);
  expectNear(feeds.input_leader_Positions, mapping.inputs.input_leader_Positions, 'mapping input_leader_Positions', TOL_INPUT);
  expectNear(feeds.input_leader_RootMotion, mapping.inputs.input_leader_RootMotion, 'mapping input_leader_RootMotion', TOL_INPUT);
  // the S-frame frames convert to the same DP frames
  const framesS = new Float64Array(P.N_HISTORY * P.FRAME_DIM);
  P.flattenInto(framesS, mapping.framesS);
  for (let i = 0; i < P.N_HISTORY; i++) P.frameSToDp(framesS, i * P.FRAME_DIM, framesS, i * P.FRAME_DIM);
  expectNear(framesS, frames, 'mapping framesS -> DP', 1e-6);
});

test('mapping_io: decode fp32 outputs -> absolute roots, local and world positions (1e-4)', () => {
  const outputs = { Positions: f32(mapping.outputsFp32.Positions), RootMotion: f32(mapping.outputsFp32.RootMotion) };
  const dec = P.createMappingDecode();
  P.decodeMapping(outputs, Float64Array.from(mapping.roots.rmRef), dec);
  expectNear(dec.rmRel, mapping.decoded.rmRel, 'mapping decoded.rmRel', TOL_INPUT);
  expectNear(dec.rm, mapping.decoded.rm, 'mapping decoded.rm', TOL_POST);
  expectNear(dec.local, mapping.decoded.localPositions, 'mapping decoded.localPositions', TOL_POST);
  expectNear(dec.world, mapping.decoded.worldPositionsDP, 'mapping decoded.worldPositionsDP', TOL_POST);
  const worldS = new Float64Array(P.N_FUTURE * 9);
  P.dpToSPositions(worldS, dec.world, P.N_FUTURE * 3);
  expectNear(worldS, mapping.decoded.worldPositionsS, 'mapping decoded.worldPositionsS', TOL_POST);
  // sanity: the fp32 prediction is close to the ground-truth future (fixture says ~8.3 cm)
  const gt = flat(mapping.groundTruth.futureWorldDP);
  let sum = 0;
  for (let i = 0; i < P.N_FUTURE * 3; i++) {
    sum += Math.hypot(dec.world[i * 3] - gt[i * 3], dec.world[i * 3 + 1] - gt[i * 3 + 1], dec.world[i * 3 + 2] - gt[i * 3 + 2]);
  }
  const meanCm = 100 * sum / (P.N_FUTURE * 3);
  assert.ok(Math.abs(meanCm - mapping.groundTruth.meanErrorCm) < 0.05, `mean error ${meanCm} cm vs fixture ${mapping.groundTruth.meanErrorCm}`);
});

test('DPPipeline: pushing the 15 frames reproduces the mapping inputs (ring buffer + yaw fallback)', () => {
  const pipe = new P.DPPipeline({ skeleton, initPose });
  const frame = new Float64Array(P.FRAME_DIM);
  for (let i = 0; i < P.N_HISTORY; i++) {
    P.flattenInto(frame, mapping.framesDP[i]);
    assert.equal(pipe.pushFrame(frame), i);
  }
  assert.equal(pipe.frameCount, 15);
  const feeds = pipe.buildMappingInputs();
  expectNear(feeds.input_leader_Positions, mapping.inputs.input_leader_Positions, 'DPPipeline input_leader_Positions', TOL_INPUT);
  expectNear(feeds.input_leader_RootMotion, mapping.inputs.input_leader_RootMotion, 'DPPipeline input_leader_RootMotion', TOL_INPUT);
  expectNear(pipe.measuredRm, mapping.roots.rm[14], 'DPPipeline measuredRm', TOL_INPUT);
  const outputs = { Positions: f32(mapping.outputsFp32.Positions), RootMotion: f32(mapping.outputsFp32.RootMotion) };
  pipe.applyMappingOutputs(outputs);
  expectNear(pipe.mapped.world, mapping.decoded.worldPositionsDP, 'DPPipeline mapped.world', TOL_POST);
  // pushing more frames slides the window: the oldest frame becomes the fallback yaw source
  P.flattenInto(frame, mapping.framesDP[14]);
  pipe.pushFrame(frame);
  pipe.buildMappingInputs();
  assert.ok(Math.abs(pipe.windowPrevYaw - mapping.roots.yaw[0]) < 1e-6);
  assert.ok(Math.abs(pipe.mappingWork.yaw[0] - mapping.roots.yaw[1]) < 1e-6);
});

test('DPPipeline: the first frame pre-fills the history (Unity/Python clip semantics)', () => {
  const pipe = new P.DPPipeline({ skeleton, initPose });
  const frame = new Float64Array(P.FRAME_DIM);
  P.flattenInto(frame, mapping.framesDP[0]);
  pipe.pushFrame(frame);
  const feeds = pipe.buildMappingInputs();
  const rmRel = feeds.input_leader_RootMotion;
  for (let t = 0; t < P.N_HISTORY; t++) {
    assert.ok(Math.abs(rmRel[t] - 1) < 1e-12 && Math.abs(rmRel[P.N_HISTORY + t]) < 1e-12, 'all frames equal -> relative rm identity');
    assert.ok(Math.abs(rmRel[2 * P.N_HISTORY + t]) < 1e-12 && Math.abs(rmRel[3 * P.N_HISTORY + t]) < 1e-12);
  }
  expectNear(pipe.measuredRm, mapping.roots.rm[0], 'DPPipeline first-frame rm', TOL_INPUT);
});

// ---------------------------------------------------------------------------------------------
// tracking_io.json

function fixtureAvatar() {
  const av = P.createAvatarState();
  P.avatarFromPose(av, tracking.avatar, tracking.avatar.rm, 0);
  return av;
}

function trackingOutputs() {
  const o = {};
  for (const name of P.TRACKING_OUTPUT_NAMES) o[name] = f32(tracking.outputsFp32[name]);
  return o;
}

test('tracking_io: leader roots, relative roots and the 7 input tensors (1e-5)', () => {
  const T = P.N_LEADER_TRACKING;
  const frames = new Float64Array(T * P.FRAME_DIM);
  P.flattenInto(frames, tracking.leaderFramesDP);
  const pos = new Float64Array(T * 3);
  const yaw = new Float64Array(T);
  P.frameRoots(frames, T, 0, pos, yaw, new Float64Array(3));
  expectNear(pos, tracking.leaderRoots.rootPos, 'tracking leaderRoots.rootPos', TOL_INPUT);
  expectNear(yaw, tracking.leaderRoots.yaw, 'tracking leaderRoots.yaw', TOL_INPUT);
  const rm = new Float64Array(T * 4);
  for (let i = 0; i < T; i++) P.rmFromRoot(rm, i * 4, pos[i * 3], pos[i * 3 + 2], yaw[i]);
  expectNear(rm, tracking.leaderRoots.rm, 'tracking leaderRoots.rm', TOL_INPUT);
  const local = new Float64Array(T * 9);
  P.threePointLocal(frames, T, pos, yaw, local, new Float64Array(4), new Float64Array(16));
  expectNear(local, tracking.leaderRoots.local, 'tracking leaderRoots.local', TOL_INPUT);
  const av = fixtureAvatar();
  expectNear(av.world, tracking.avatar.worldDP, 'tracking avatar.worldDP', TOL_POST);
  expectNear(av.matrix, tracking.avatar.rootMatrix, 'tracking avatar.rootMatrix', TOL_INPUT);
  const feeds = P.createTrackingFeeds();
  P.trackingInputs(rm, local, av, feeds);
  expectNear(feeds.rel, tracking.leaderRoots.rmRelativeToAvatar, 'tracking leaderRoots.rmRelativeToAvatar', TOL_INPUT);
  for (const name of P.TRACKING_INPUT_NAMES) {
    assert.equal(feeds[name].length, P.TRACKING_INPUT_DIMS[name], name);
    expectNear(feeds[name], tracking.inputs[name], 'tracking ' + name, TOL_INPUT);
  }
  assert.deepEqual(Array.from(feeds.input_follower_RootMotion), [1, 0, 0, 0]);
});

test('tracking_io: decode frame 0, uncorrected root, raw world (1e-4)', () => {
  const pred = P.createTrackingPred();
  P.decodeTracking(trackingOutputs(), 0, pred);
  expectNear(pred.rm, tracking.decodedFrame0.rm, 'tracking decodedFrame0.rm', TOL_INPUT);
  expectNear(pred.positions, tracking.decodedFrame0.positions, 'tracking decodedFrame0.positions', TOL_INPUT);
  expectNear(pred.rotations, tracking.decodedFrame0.rotations, 'tracking decodedFrame0.rotations', TOL_INPUT);
  expectNear(pred.velocities, tracking.decodedFrame0.velocities, 'tracking decodedFrame0.velocities', TOL_INPUT);
  expectNear(pred.contacts, tracking.decodedFrame0.contacts, 'tracking decodedFrame0.contacts', TOL_INPUT);
  const newRm = new Float64Array(4);
  P.rmApply(newRm, 0, Float64Array.from(tracking.avatar.rm), 0, pred.rm, 0);
  expectNear(newRm, tracking.decodedFrame0.newRmUncorrected, 'tracking newRmUncorrected', TOL_POST);
  const m = new Float64Array(16);
  P.rmToMatrix(m, 0, newRm, 0);
  const raw = new Float64Array(P.N_JOINTS * 3);
  for (let j = 0; j < P.N_JOINTS; j++) P.toWorld(raw, j * 3, m, 0, pred.positions, j * 3);
  expectNear(raw, tracking.decodedFrame0.rawWorldDP, 'tracking rawWorldDP', TOL_POST);
});

for (const label of ['rc0_er0', 'rc035_er05']) {
  test(`tracking_io: post-processing ${label} (root correction, Euler blend, bone restore; 1e-4)`, () => {
    const exp = tracking.postProcessed[label];
    const av = fixtureAvatar();
    const pred = P.createTrackingPred();
    P.decodeTracking(trackingOutputs(), 0, pred);
    const sk = P.prepareSkeleton(tracking.skeleton);
    const work = P.createPostWork();
    P.postProcess(av, pred, Float64Array.from(tracking.measuredRm), sk, exp.rootCorrection, exp.eulerRatio, tracking.params.dt, work);
    expectNear(av.rm, exp.rm, `post ${label} rm`, TOL_POST);
    expectNear(av.matrix, exp.rootMatrix, `post ${label} rootMatrix`, TOL_POST);
    expectNear(av.world, exp.worldDP, `post ${label} worldDP`, TOL_POST);
    expectNear(av.positions, exp.positionsLocal, `post ${label} positionsLocal`, TOL_POST);
    expectNear(av.contacts, exp.contacts, `post ${label} contacts`, TOL_INPUT);
    const worldS = new Float64Array(P.N_JOINTS * 3);
    P.dpToSPositions(worldS, av.world, P.N_JOINTS);
    expectNear(worldS, exp.worldS, `post ${label} worldS`, TOL_POST);
    expectNear([av.rm[2], 0, -av.rm[3]], exp.rootS, `post ${label} rootS`, TOL_POST);
    // rc035_er05: b_l_foot, b_r_foot (zero-length, float noise) and b_l_forearm; rc0_er0 also clamps b_r_forearm
    assert.equal(work.clamps, label === 'rc035_er05' ? 3 : 4, 'bone-length restore is exercised');
    // bone lengths respected afterwards
    for (let j = 1; j < P.N_JOINTS; j++) {
      const p = sk.parents[j];
      const d = Math.hypot(av.world[j * 3] - av.world[p * 3], av.world[j * 3 + 1] - av.world[p * 3 + 1], av.world[j * 3 + 2] - av.world[p * 3 + 2]);
      assert.ok(d <= 1.05 * sk.boneLengths[j] + 1e-9, `bone ${j} too long after restore: ${d} > ${1.05 * sk.boneLengths[j]}`);
    }
    if (label === 'rc0_er0') {
      const gt = flat(tracking.groundTruth.nextWorldDP);
      let sum = 0;
      for (let j = 0; j < P.N_JOINTS; j++) sum += Math.hypot(av.world[j * 3] - gt[j * 3], av.world[j * 3 + 1] - gt[j * 3 + 1], av.world[j * 3 + 2] - gt[j * 3 + 2]);
      const meanCm = 100 * sum / P.N_JOINTS;
      assert.ok(Math.abs(meanCm - tracking.groundTruth.meanErrorCm_rc0_er0) < 0.05, `mean joint error ${meanCm} cm`);
    }
  });
}

test('restoreBoneLengths: subtree shift semantics and zero-length bones', () => {
  // 4 joints in a chain: 0 -> 1 -> 2 -> 3, bone lengths 1; stretch joint 1 to 2 units
  const parents = new Int32Array([-1, 0, 1, 2]);
  const lengths = new Float64Array([0, 1, 1, 1]);
  const world = new Float64Array([0, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0]);
  const clamps = P.restoreBoneLengths(world, parents, lengths, 1.05, new Uint8Array(4));
  assert.equal(clamps, 1);
  assert.ok(Math.abs(world[3] - 1.05) < 1e-12);
  assert.ok(Math.abs(world[6] - 2.05) < 1e-12, 'descendants move with the clamped joint');
  assert.ok(Math.abs(world[9] - 3.05) < 1e-12);
  // zero-length bone snaps onto its parent
  const w2 = new Float64Array([0, 0, 0, 0.5, 0.2, 0]);
  assert.equal(P.restoreBoneLengths(w2, new Int32Array([-1, 0]), new Float64Array([0, 0]), 1.05, new Uint8Array(2)), 1);
  assert.deepEqual(Array.from(w2.subarray(3)), [0, 0, 0]);
});

test('DPPipeline: tracking tick through the class API matches the fixture (rc 0.35 / er 0.5 and 0 / 0)', () => {
  for (const label of ['rc035_er05', 'rc0_er0']) {
    const exp = tracking.postProcessed[label];
    const pipe = new P.DPPipeline({ skeleton, initPose, rootCorrection: exp.rootCorrection, eulerRatio: exp.eulerRatio });
    const frame = new Float64Array(P.FRAME_DIM);
    P.flattenInto(frame, tracking.leaderFramesDP[0]);
    pipe.pushFrame(frame);
    pipe.buildMappingInputs();
    expectNear(pipe.measuredRm, tracking.measuredRm, 'DPPipeline tracking measuredRm', TOL_INPUT);
    // stand in for the mapping prediction: the ground-truth future frames 1..30
    const rmAll = flat(tracking.leaderRoots.rm);
    const localAll = flat(tracking.leaderRoots.local);
    pipe.mapped.rm.set(rmAll.slice(4));
    pipe.mapped.local.set(localAll.slice(9));
    pipe.setAvatarPose(tracking.avatar, tracking.avatar.rm, 0);
    const feeds = pipe.buildTrackingInputs();
    for (const name of P.TRACKING_INPUT_NAMES) expectNear(feeds[name], tracking.inputs[name], 'DPPipeline tracking ' + name, TOL_INPUT);
    const pose = pipe.applyTrackingOutputs(trackingOutputs(), 0);
    expectNear(pose.positionsWorldS, exp.worldS, `DPPipeline pose.positionsWorldS ${label}`, TOL_POST);
    expectNear(pose.positionsWorldDP, exp.worldDP, `DPPipeline pose.positionsWorldDP ${label}`, TOL_POST);
    expectNear(pose.rootS, exp.rootS, `DPPipeline pose.rootS ${label}`, TOL_POST);
    expectNear(pose.rm, exp.rm, `DPPipeline pose.rm ${label}`, TOL_POST);
    expectNear(pose.contacts, exp.contacts, `DPPipeline pose.contacts ${label}`, TOL_INPUT);
    assert.equal(pose.clamps, label === 'rc035_er05' ? 3 : 4);
    assert.equal(pose.tick, 0);
    // S yaw = -DP yaw
    assert.ok(Math.abs(pose.rootYawS + P.rmYaw(pose.rm, 0)) < 1e-12);
  }
});

test('DPPipeline.runTick with stub models runs the whole loop and re-uses its buffers', async () => {
  const pipe = new P.DPPipeline({ skeleton, initPose });
  let mappingCalls = 0;
  let trackingCalls = 0;
  const mout = { Positions: f32(mapping.outputsFp32.Positions), RootMotion: f32(mapping.outputsFp32.RootMotion) };
  const tout = trackingOutputs();
  const models = {
    runMapping(feeds) {
      mappingCalls++;
      assert.equal(feeds.input_leader_Positions.length, 135);
      return mout;
    },
    async runTracking(feeds) {
      trackingCalls++;
      assert.equal(feeds.input_leader_Positions.length, 279);
      return tout;
    },
  };
  const frame = new Float32Array(P.FRAME_DIM);
  const firstFeeds = pipe.trackingFeeds;
  for (let i = 0; i < 20; i++) {
    P.flattenInto(frame, mapping.framesDP[Math.min(i, 14)]);
    const pose = await pipe.runTick(frame, models);
    assert.equal(pose.tick, i);
    for (let k = 0; k < pose.positionsWorldS.length; k++) assert.ok(Number.isFinite(pose.positionsWorldS[k]));
  }
  assert.equal(mappingCalls, 20);
  assert.equal(trackingCalls, 20);
  assert.equal(pipe.trackingFeeds, firstFeeds, 'feeds are reused, not reallocated');
  // the avatar was initialised at the first frame's root
  assert.ok(pipe.avatarInitialised);
  pipe.reset();
  assert.equal(pipe.frameCount, 0);
  assert.throws(() => pipe.buildMappingInputs(), /no frame/);
});

test('sampleSToFrameDP converts a stage sample (scaled by k) into an unscaled DP frame', () => {
  const k = 1.25;
  const sample = {
    head: { p: [0.1 * k, 1.6 * k, -0.5 * k], q: [0.1, 0.2, 0.3, 0.9] },
    left: { p: [-0.3 * k, 1.0 * k, -0.7 * k] },
    right: { p: [0.3 * k, 1.1 * k, -0.6 * k] },
    k,
  };
  const out = new Float64Array(13);
  P.sampleSToFrameDP(out, 0, sample, k);
  const exp = [0.1, 1.6, 0.5, -0.1, -0.2, 0.3, 0.9, 0.3, 1.1, 0.6, -0.3, 1.0, 0.7];
  for (let i = 0; i < 13; i++) assert.ok(Math.abs(out[i] - exp[i]) < 1e-12, `component ${i}: ${out[i]} vs ${exp[i]}`);
});

// ---------------------------------------------------------------------------------------------
// NeuralAvatarDriver with a fake worker (no browser): sampling, protocol, interpolation, guard

class FakeWorker {
  constructor(url, opts) {
    FakeWorker.instances.push(this);
    this.url = String(url);
    this.opts = opts;
    this.onmessage = null;
    this.onerror = null;
    this.received = [];
    this.terminated = false;
    this.inferenceMs = 5;
    this.autoPose = true;
    this.tickCount = 0;
  }

  // main -> worker (structured clone of typed arrays, like a real worker)
  postMessage(msg) {
    if (msg && msg.frame) msg = Object.assign({}, msg, { frame: Float32Array.from(msg.frame) });
    this.received.push(msg);
    if (msg.type === 'init') {
      setTimeout(() => this.emit({ type: 'ready', loadMs: 1, ortVersion: 'fake', models: { mapping: 'm', tracking: 't' } }), 0);
    } else if (msg.type === 'frame' && this.autoPose) {
      this.tickCount++;
      const buffer = new ArrayBuffer(P.N_JOINTS * 3 * 4 + 12 + 16);
      const pos = new Float32Array(buffer, 0, P.N_JOINTS * 3);
      for (let i = 0; i < pos.length; i++) pos[i] = this.tickCount; // every joint = tick number
      setTimeout(() => this.emit({ type: 'pose', tick: this.tickCount - 1, t: msg.t, seq: msg.seq, buffer, rootYawS: 0, inferenceMs: this.inferenceMs, mappingMs: 1, trackingMs: this.inferenceMs - 1, dropped: 0 }), 0);
    } else if (msg.type === 'telemetry') {
      setTimeout(() => this.emit({ type: 'telemetry', data: { ticks: [1, 2, 3] } }), 0);
    }
  }

  emit(data) {
    if (this.terminated || !this.onmessage) return;
    this.onmessage({ data });
  }

  terminate() {
    this.terminated = true;
  }
}
FakeWorker.instances = [];

const tickMs = 1000 / 30;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

function sampleAt(t) {
  return {
    t,
    k: 1.1,
    head: { p: [0.05 * Math.sin(t), 1.65 + 0.01 * Math.sin(2 * t), 0], q: [0, 0, 0, 1], valid: true },
    left: { p: [-0.25, 1.0, -0.1], q: [0, 0, 0, 1], valid: true },
    right: { p: [0.25, 1.0, -0.1], q: [0, 0, 0, 1], valid: true },
  };
}

test('NeuralAvatarDriver: init handshake, 30 Hz sampling from a 72 Hz render loop, S->DP conversion', async () => {
  FakeWorker.instances.length = 0;
  let clock = 1000;
  const statuses = [];
  const driver = new NeuralAvatarDriver({
    WorkerClass: FakeWorker,
    now: () => clock,
    onStatus: (s) => statuses.push(s.state),
    workerUrl: 'fake://worker.js',
    modelsUrl: 'fake://models/free/',
  });
  await driver.init();
  const w = FakeWorker.instances[0];
  assert.equal(w.opts.type, 'module');
  assert.equal(w.received[0].type, 'init');
  assert.equal(w.received[0].modelsUrl, 'fake://models/free/');
  assert.equal(driver.state, 'ready');
  driver.start();
  assert.equal(driver.state, 'running');
  // 72 Hz samples for 1 s
  const n = 72;
  for (let i = 0; i <= n; i++) {
    clock = 1000 + (i / 72) * 1000;
    driver.update(sampleAt(i / 72));
  }
  const frames = w.received.filter((m) => m.type === 'frame');
  assert.ok(frames.length >= 29 && frames.length <= 31, `expected ~30 frames in 1 s, got ${frames.length}`);
  // frame content: DP, unscaled (divided by k), head z negated, quaternion mirrored
  const f = frames[0].frame;
  assert.equal(f.length, 13);
  assert.ok(Math.abs(f[1] - 1.65 / 1.1) < 1e-3, 'head height divided by k');
  assert.ok(Math.abs(f[9] - 0.1 / 1.1) < 1e-6, 'right wrist z negated (S -0.1 -> DP +0.1)');
  assert.ok(Math.abs(f[7] - 0.25 / 1.1) < 1e-6, 'right wrist x');
  assert.ok(Math.abs(f[10] + 0.25 / 1.1) < 1e-6, 'left wrist x');
  // frame times are on the 30 Hz grid (monotone, ~1/30 apart)
  for (let i = 1; i < frames.length; i++) assert.ok(Math.abs((frames[i].t - frames[i - 1].t) - 1 / 30) < 1e-6);
  await wait(10);
  assert.ok(driver.stats.poses >= 29, 'poses received: ' + driver.stats.poses);
  assert.ok(driver.inferenceMs > 0);
  assert.ok(statuses.includes('ready') && statuses.includes('running'));
  driver.dispose();
  assert.ok(w.terminated);
  assert.equal(driver.state, 'disposed');
});

test('NeuralAvatarDriver: pose interpolation between the last two poses and buffer recycling', async () => {
  FakeWorker.instances.length = 0;
  let clock = 0;
  const driver = new NeuralAvatarDriver({ WorkerClass: FakeWorker, now: () => clock, workerUrl: 'w', modelsUrl: 'm/' });
  await driver.init();
  driver.start();
  assert.equal(driver.getPose(), null, 'no pose yet');
  const frame = new Float32Array(13);
  frame[1] = 1.6; frame[6] = 1;
  driver.pushFrameDP(frame, 0);
  await wait(5);
  clock = 100;
  driver.pushFrameDP(frame, 1 / 30);
  await wait(5);
  const w = FakeWorker.instances[0];
  assert.equal(driver.stats.poses, 2);
  const recycled = w.received.filter((m) => m.type === 'recycle');
  assert.equal(recycled.length, 2, 'pose buffers are handed back to the worker');
  // second pose arrived at clock=100; poses are 1 (all joints) then 2. The interpolation window is
  // the smoothed arrival interval (EMA of the gaps, seeded with the nominal tick).
  const interval = driver.poseIntervalMs;
  assert.ok(interval > tickMs && interval < 100, 'smoothed interval between nominal tick and measured gap: ' + interval);
  clock = 100;
  let pose = driver.getPose();
  assert.ok(Math.abs(pose[0] - 1) < 1e-6, 'alpha 0 right at arrival -> previous pose');
  clock = 100 + interval / 2;
  pose = driver.getPose();
  assert.ok(Math.abs(pose[0] - 1.5) < 1e-6, 'half an interval later -> midway, got ' + pose[0]);
  clock = 100 + 10 * tickMs;
  pose = driver.getPose();
  assert.ok(Math.abs(pose[0] - 2) < 1e-6, 'clamped to the latest pose');
  assert.ok(Math.abs(driver.rootS[0] - driver.rootLast[0]) < 1e-6);
  assert.equal(driver.latest.tick, 1);
  assert.equal(driver.scale, 1);
  driver.dispose();
});

test('NeuralAvatarDriver: performance guard (slow -> 15 Hz, too slow -> disabled with German notice)', async () => {
  FakeWorker.instances.length = 0;
  let clock = 0;
  const statuses = [];
  const driver = new NeuralAvatarDriver({
    WorkerClass: FakeWorker, now: () => clock, workerUrl: 'w', modelsUrl: 'm/',
    guard: { windowTicks: 10, slowMs: 25, disableMs: 60, minTicks: 5 },
    onStatus: (s) => statuses.push({ state: s.state, notice: s.notice }),
  });
  await driver.init();
  driver.start();
  const w = FakeWorker.instances[0];
  const frame = new Float32Array(13);
  frame[1] = 1.6; frame[6] = 1;
  w.inferenceMs = 40;
  for (let i = 0; i < 8; i++) {
    driver.pushFrameDP(frame, i / 30);
    await wait(2);
  }
  assert.equal(driver.state, 'slow');
  const cfg = w.received.filter((m) => m.type === 'config');
  assert.ok(cfg.length >= 1 && cfg[cfg.length - 1].inferEvery === 2, 'worker told to infer every 2nd frame');
  assert.equal(driver.inferEvery, 2);
  // recovery: fast again
  w.inferenceMs = 5;
  for (let i = 8; i < 30; i++) {
    driver.pushFrameDP(frame, i / 30);
    await wait(2);
  }
  assert.equal(driver.state, 'running');
  assert.equal(driver.inferEvery, 1);
  // way too slow -> disabled
  w.inferenceMs = 90;
  for (let i = 30; i < 50; i++) {
    driver.pushFrameDP(frame, i / 30);
    await wait(2);
  }
  assert.equal(driver.state, 'disabled');
  assert.ok(w.terminated, 'worker terminated when disabled');
  const last = statuses[statuses.length - 1];
  assert.equal(last.state, 'disabled');
  assert.equal(last.notice, NOTICES.disabledSlow);
  assert.match(last.notice, /deaktiviert/);
  assert.equal(driver.getPose(), null, 'no pose after disable');
});

test('NeuralAvatarDriver: telemetry export round trip and error handling', async () => {
  FakeWorker.instances.length = 0;
  const driver = new NeuralAvatarDriver({ WorkerClass: FakeWorker, workerUrl: 'w', modelsUrl: 'm/', telemetry: true });
  await driver.init();
  const w = FakeWorker.instances[0];
  assert.equal(w.received[0].telemetry, true);
  const data = await driver.exportTelemetry();
  assert.deepEqual(data, { ticks: [1, 2, 3] });
  const errors = [];
  driver.onError = (e) => errors.push(e);
  w.emit({ type: 'error', message: 'kaputt', stage: 'run' });
  assert.equal(errors.length, 1);
  assert.equal(driver.state, 'error');
  driver.dispose();
});

test('report achieved tolerances', () => {
  const groups = { input: 0, post: 0 };
  for (const [k, v] of Object.entries(achieved)) {
    if (/input_|roots|measuredRm|decodedFrame0|rmRel|contacts|rootMatrix/.test(k)) groups.input = Math.max(groups.input, v);
    else groups.post = Math.max(groups.post, v);
  }
  console.log(`fixture parity -> inputs/decoding max abs error ${groups.input.toExponential(2)} (limit 1e-5), ` +
    `post-processing/world max abs error ${groups.post.toExponential(2)} (limit 1e-4)`);
  assert.ok(groups.input <= TOL_INPUT);
  assert.ok(groups.post <= TOL_POST);
});
