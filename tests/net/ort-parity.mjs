#!/usr/bin/env node
// ort-parity.mjs - int8 vs fp32 parity of the shipped models through the JS pipeline (optional,
// NOT part of `npm test`). Requires `onnxruntime-node`; when it cannot be required the script
// prints SKIP and exits 0. It installs nothing.
//
// Runs webxr/models/free/{mapping,tracking}_leader.int8.onnx on the exact input tensors of
// tests/fixtures/{mapping_io,tracking_io}.json (assembled by src/net/pipeline.js), compares the
// int8 outputs with the fp32 expected outputs stored in the fixtures, decodes / post-processes both
// and reports the errors in centimetres. Also times a closed-loop DPPipeline run.
//
// Usage: node tests/net/ort-parity.mjs   (ORT_NODE=<dir containing node_modules/onnxruntime-node>)
import { createRequire } from 'node:module';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import * as P from '../../webxr/src/net/pipeline.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..', '..');
const require = createRequire(import.meta.url);

function loadOrt() {
  const candidates = [];
  if (process.env.ORT_NODE) candidates.push(join(process.env.ORT_NODE, 'node_modules', 'onnxruntime-node'));
  candidates.push('/tmp/claude-0/-home-user-dancing-points/17584e77-ec43-5660-97ee-ff4b7176d577/scratchpad/net-js/node_modules/onnxruntime-node');
  candidates.push(join(root, 'node_modules', 'onnxruntime-node'));
  candidates.push('onnxruntime-node');
  for (const c of candidates) {
    try { return require(c); } catch (e) { /* next */ }
  }
  return null;
}

const flat = (nested) => { const o = []; P.flattenInto(o, Array.isArray(nested) ? nested : Array.from(nested)); return o; };
const f32 = (nested) => Float32Array.from(flat(nested));

function errStats(got, exp) {
  const g = flat(got), e = flat(exp);
  if (g.length !== e.length) throw new Error('length mismatch ' + g.length + ' vs ' + e.length);
  let maxAbs = 0, sumAbs = 0, sumSq = 0, sumSqRef = 0;
  for (let i = 0; i < g.length; i++) {
    const d = g[i] - e[i];
    if (!Number.isFinite(g[i])) throw new Error('non-finite output');
    maxAbs = Math.max(maxAbs, Math.abs(d));
    sumAbs += Math.abs(d); sumSq += d * d; sumSqRef += e[i] * e[i];
  }
  return { n: g.length, maxAbs, meanAbs: sumAbs / g.length, rel: Math.sqrt(sumSq / Math.max(sumSqRef, 1e-30)) };
}

function meanPointErrCm(got, exp) {
  const g = flat(got), e = flat(exp);
  let s = 0;
  const n = g.length / 3;
  for (let i = 0; i < n; i++) s += Math.hypot(g[i * 3] - e[i * 3], g[i * 3 + 1] - e[i * 3 + 1], g[i * 3 + 2] - e[i * 3 + 2]);
  return 100 * s / n;
}

const fmt = (s) => `max|d| ${s.maxAbs.toExponential(2)}, mean|d| ${s.meanAbs.toExponential(2)}, rel ${(100 * s.rel).toFixed(2)} %`;

async function main() {
  const ort = loadOrt();
  if (!ort) { console.log('SKIP: onnxruntime-node not installed (set ORT_NODE or npm install onnxruntime-node in a scratch dir)'); return 0; }
  const modelsDir = join(root, 'webxr', 'models', 'free');
  const mapPath = join(modelsDir, 'mapping_leader.int8.onnx');
  const trackPath = join(modelsDir, 'tracking_leader.int8.onnx');
  if (!existsSync(mapPath) || !existsSync(trackPath)) { console.log('SKIP: int8 models missing in ' + modelsDir); return 0; }
  const mapping = JSON.parse(readFileSync(join(root, 'tests', 'fixtures', 'mapping_io.json'), 'utf8'));
  const tracking = JSON.parse(readFileSync(join(root, 'tests', 'fixtures', 'tracking_io.json'), 'utf8'));
  const skeleton = JSON.parse(readFileSync(join(modelsDir, 'skeleton.json'), 'utf8'));
  const initPose = JSON.parse(readFileSync(join(modelsDir, 'init_pose.json'), 'utf8'));
  const opts = { executionProviders: ['cpu'], intraOpNumThreads: 1, interOpNumThreads: 1, graphOptimizationLevel: 'all' };
  console.log('onnxruntime-node', ort.env.versions ? JSON.stringify(ort.env.versions) : '', '| models', modelsDir);
  const t0 = Date.now();
  const mapSession = await ort.InferenceSession.create(mapPath, opts);
  const trackSession = await ort.InferenceSession.create(trackPath, opts);
  console.log('sessions created in', Date.now() - t0, 'ms');
  const tensor = (data, n) => new ort.Tensor('float32', data, [1, n]);
  const run = async (session, feeds) => {
    const f = {};
    for (const n of session.inputNames) f[n] = tensor(feeds[n], feeds[n].length);
    const r = await session.run(f);
    const o = {};
    for (const n of session.outputNames) o[n] = r[n].data;
    return o;
  };
  let ok = true;

  // ---- mapping ----
  console.log('\nmapping_leader int8 on the mapping_io fixture (pivot frame ' + mapping.params.pivotFrame + ')');
  const frames = new Float64Array(P.N_HISTORY * P.FRAME_DIM);
  P.flattenInto(frames, mapping.framesDP);
  const work = P.createMappingWork();
  const mfeeds = P.mappingInputs(frames, mapping.params.prevYaw, work);
  const mIn = errStats(mfeeds.input_leader_Positions, mapping.inputs.input_leader_Positions);
  console.log('  JS input tensors vs fixture:', fmt(mIn));
  const mt0 = process.hrtime.bigint();
  const mout = await run(mapSession, mfeeds);
  const mapMs = Number(process.hrtime.bigint() - mt0) / 1e6;
  for (const n of P.MAPPING_OUTPUT_NAMES) console.log(`  int8 ${n} vs fp32:`, fmt(errStats(mout[n], mapping.outputsFp32[n])));
  const decInt8 = P.createMappingDecode();
  P.decodeMapping(mout, work.rmRef, decInt8);
  const decFp32 = P.createMappingDecode();
  P.decodeMapping({ Positions: f32(mapping.outputsFp32.Positions), RootMotion: f32(mapping.outputsFp32.RootMotion) }, work.rmRef, decFp32);
  const mapWorldInt8VsFp32 = meanPointErrCm(decInt8.world, decFp32.world);
  const mapWorldInt8VsGt = meanPointErrCm(decInt8.world, mapping.groundTruth.futureWorldDP);
  const mapWorldFp32VsGt = meanPointErrCm(decFp32.world, mapping.groundTruth.futureWorldDP);
  console.log(`  decoded future 3-point world (30 frames): int8 vs fp32 ${mapWorldInt8VsFp32.toFixed(2)} cm | int8 vs GT ${mapWorldInt8VsGt.toFixed(2)} cm | fp32 vs GT ${mapWorldFp32VsGt.toFixed(2)} cm (fixture ${mapping.groundTruth.meanErrorCm.toFixed(2)})`);
  console.log(`  mapping run time (first call): ${mapMs.toFixed(1)} ms`);
  if (mapWorldInt8VsFp32 > 3) { ok = false; console.log('  FAIL: int8 mapping deviates > 3 cm from fp32'); }

  // ---- tracking ----
  console.log('\ntracking_leader int8 on the tracking_io fixture');
  const T = P.N_LEADER_TRACKING;
  const lframes = new Float64Array(T * P.FRAME_DIM);
  P.flattenInto(lframes, tracking.leaderFramesDP);
  const pos = new Float64Array(T * 3), yaw = new Float64Array(T);
  P.frameRoots(lframes, T, 0, pos, yaw, new Float64Array(3));
  const rm = new Float64Array(T * 4);
  for (let i = 0; i < T; i++) P.rmFromRoot(rm, i * 4, pos[i * 3], pos[i * 3 + 2], yaw[i]);
  const local = new Float64Array(T * 9);
  P.threePointLocal(lframes, T, pos, yaw, local, new Float64Array(4), new Float64Array(16));
  const avatar = P.createAvatarState();
  P.avatarFromPose(avatar, tracking.avatar, tracking.avatar.rm, 0);
  const tfeeds = P.trackingInputs(rm, local, avatar, P.createTrackingFeeds());
  let maxIn = 0;
  for (const n of P.TRACKING_INPUT_NAMES) maxIn = Math.max(maxIn, errStats(tfeeds[n], tracking.inputs[n]).maxAbs);
  console.log('  JS input tensors vs fixture: max|d|', maxIn.toExponential(2));
  const tt0 = process.hrtime.bigint();
  const tout = await run(trackSession, tfeeds);
  const trackMs = Number(process.hrtime.bigint() - tt0) / 1e6;
  for (const n of P.TRACKING_OUTPUT_NAMES) console.log(`  int8 ${n} vs fp32:`, fmt(errStats(tout[n], tracking.outputsFp32[n])));
  const sk = P.prepareSkeleton(tracking.skeleton);
  const results = {};
  for (const label of ['rc0_er0', 'rc035_er05']) {
    const exp = tracking.postProcessed[label];
    const av = P.createAvatarState();
    P.avatarFromPose(av, tracking.avatar, tracking.avatar.rm, 0);
    const pred = P.createTrackingPred();
    P.decodeTracking(tout, 0, pred);
    P.postProcess(av, pred, Float64Array.from(tracking.measuredRm), sk, exp.rootCorrection, exp.eulerRatio, tracking.params.dt, P.createPostWork());
    const vsFp32 = meanPointErrCm(av.world, exp.worldDP);
    const vsGt = meanPointErrCm(av.world, tracking.groundTruth.nextWorldDP);
    const fp32VsGt = meanPointErrCm(exp.worldDP, tracking.groundTruth.nextWorldDP);
    results[label] = vsFp32;
    console.log(`  post-processed world ${label}: int8 vs fp32 ${vsFp32.toFixed(2)} cm | int8 vs GT next frame ${vsGt.toFixed(2)} cm | fp32 vs GT ${fp32VsGt.toFixed(2)} cm`);
    if (vsFp32 > 3) { ok = false; console.log('  FAIL: int8 tracking deviates > 3 cm from fp32'); }
  }
  console.log(`  tracking run time (first call): ${trackMs.toFixed(1)} ms`);

  // ---- closed loop timing (DPPipeline.runTick with the real int8 models) ----
  console.log('\nclosed loop: DPPipeline.runTick over the 31 fixture frames (int8, onnxruntime-node CPU, 1 thread)');
  const pipe = new P.DPPipeline({ skeleton, initPose });
  const models = { runMapping: (f) => run(mapSession, f), runTracking: (f) => run(trackSession, f) };
  const frame = new Float64Array(P.FRAME_DIM);
  const times = [];
  let lastPose = null;
  for (let i = 0; i < T; i++) {
    P.flattenInto(frame, tracking.leaderFramesDP[i]);
    const s = process.hrtime.bigint();
    lastPose = await pipe.runTick(frame, models);
    times.push(Number(process.hrtime.bigint() - s) / 1e6);
  }
  let finite = true;
  for (let i = 0; i < lastPose.positionsWorldS.length; i++) if (!Number.isFinite(lastPose.positionsWorldS[i])) finite = false;
  const headJ = P.HEAD;
  const headIn = tracking.leaderFramesDP[T - 1];
  const headErr = Math.hypot(lastPose.positionsWorldDP[headJ * 3] - headIn[0], lastPose.positionsWorldDP[headJ * 3 + 2] - headIn[2]);
  times.sort((a, b) => a - b);
  console.log(`  ${T} ticks: median ${times[Math.floor(T / 2)].toFixed(1)} ms, min ${times[0].toFixed(1)} ms, max ${times[T - 1].toFixed(1)} ms | finite ${finite} | avatar head xz vs measured head ${(100 * headErr).toFixed(1)} cm | clamps last tick ${lastPose.clamps}`);
  if (!finite) { ok = false; console.log('  FAIL: non-finite pose'); }

  console.log('\n' + (ok ? 'PASS' : 'FAIL') + ` (int8 vs fp32: mapping world ${mapWorldInt8VsFp32.toFixed(2)} cm, tracking world rc0/er0 ${results.rc0_er0.toFixed(2)} cm, rc0.35/er0.5 ${results.rc035_er05.toFixed(2)} cm)`);
  return ok ? 0 : 1;
}

main().then((c) => process.exit(c), (e) => { console.error(e); process.exit(1); });
