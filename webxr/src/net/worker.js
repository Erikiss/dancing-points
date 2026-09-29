// worker.js - inference Web Worker for the neural avatar (onnxruntime-web, WASM backend).
//
// This is an ES MODULE worker (`new Worker(url, { type: 'module' })`): it imports the pure-JS
// pipeline (./pipeline.js) directly and loads onnxruntime-web through a dynamic import of the ESM
// build ../../vendor/ort/ort.wasm.min.mjs (copied from onnxruntime-web/dist). The wasm glue
// (ort-wasm-simd-threaded.{mjs,wasm}) is resolved from the same vendor directory via
// `ort.env.wasm.wasmPaths`. numThreads = 1 (no SharedArrayBuffer / cross-origin isolation needed),
// SIMD on. Fallback when the ESM build is missing: the classic bundle ort.wasm.min.js is fetched
// and evaluated (it assigns `globalThis.ort`).
//
// Responsibility: load meta/skeleton/init pose + the two int8 models of one set, keep a DPPipeline,
// run mapping + tracking per 30 Hz frame (or every n-th frame when told to), post poses (S frame)
// with timing, keep an optional telemetry ring of network inputs/outputs.
// Must not: import three.js; allocate per tick beyond what onnxruntime does internally (feed
// tensors are created once over the pipeline's buffers, pose buffers are recycled by the main thread).
//
// Message protocol: see ./README.md.

import { DPPipeline, MAPPING_INPUT_NAMES, MAPPING_OUTPUT_NAMES, TRACKING_INPUT_NAMES,
  TRACKING_OUTPUT_NAMES, MAPPING_INPUT_DIMS, TRACKING_INPUT_DIMS, N_JOINTS } from './pipeline.js';

const POSE_FLOATS = N_JOINTS * 3;
const POSE_BUFFER_BYTES = (POSE_FLOATS + 3 + 4) * 4;

const S = {
  ort: null,
  ortSource: null,
  pipeline: null,
  mapping: null,
  tracking: null,
  mapFeeds: null,     // { name: ort.Tensor } over pipeline.mappingWork.feeds
  trackFeeds: null,   // { name: ort.Tensor } over pipeline.trackingFeeds
  mapOut: { Positions: null, RootMotion: null },
  trackOut: { VelocitiesV2: null, Positions: null, Rotations: null, FootContactLabels: null, RootMotion: null },
  cfg: { inferEvery: 1, telemetry: false, telemetryTicks: 2000, set: 'leader' },
  ready: false,
  busy: false,
  disposed: false,
  frameCount: 0,
  inferCount: 0,
  dropped: 0,
  pool: [],
  telemetry: [],
  telemetryNext: 0,
};

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

function post(msg, transfer) {
  if (transfer) self.postMessage(msg, transfer); else self.postMessage(msg);
}

function fail(stage, err) {
  const message = (err && err.message) ? err.message : String(err);
  console.warn('[dp-net worker] ' + stage + ': ' + message, err);
  post({ type: 'error', stage, message });
}

function dirUrl(u) {
  return u.endsWith('/') ? u : u + '/';
}

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error('HTTP ' + r.status + ' for ' + url);
  return r.json();
}

/** Fetch a binary file with progress messages; returns a Uint8Array. */
async function fetchBytes(url, label) {
  const r = await fetch(url);
  if (!r.ok) throw new Error('HTTP ' + r.status + ' for ' + url);
  const total = Number(r.headers.get('content-length')) || 0;
  if (!r.body || !r.body.getReader) {
    const buf = await r.arrayBuffer();
    post({ type: 'progress', label, loaded: buf.byteLength, total: buf.byteLength });
    return new Uint8Array(buf);
  }
  const reader = r.body.getReader();
  const chunks = [];
  let loaded = 0;
  let lastReport = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    if (loaded - lastReport > (1 << 21)) {
      lastReport = loaded;
      post({ type: 'progress', label, loaded, total });
    }
  }
  post({ type: 'progress', label, loaded, total: total || loaded });
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.byteLength; }
  return out;
}

/** Load onnxruntime-web: ESM build via dynamic import, classic bundle via eval as a fallback. */
async function loadOrt(ortUrl, wasmDir, numThreads, simd) {
  const vendor = dirUrl(wasmDir || new URL('../../vendor/ort/', import.meta.url).href);
  const esmUrl = ortUrl || (vendor + 'ort.wasm.min.mjs');
  let ort = null;
  let source = null;
  try {
    const mod = await import(/* webpackIgnore: true */ esmUrl);
    ort = (mod && mod.InferenceSession) ? mod : (mod && mod.default && mod.default.InferenceSession ? mod.default : null);
    if (!ort) throw new Error('module has no InferenceSession export');
    source = esmUrl;
  } catch (e) {
    console.warn('[dp-net worker] ESM onnxruntime-web not usable (' + e.message + '), trying the classic bundle');
    const classicUrl = vendor + 'ort.wasm.min.js';
    const r = await fetch(classicUrl);
    if (!r.ok) throw new Error('cannot load onnxruntime-web: ' + esmUrl + ' (' + e.message + ') nor ' + classicUrl + ' (HTTP ' + r.status + ')');
    const text = await r.text();
    // The bundle is strict-mode code declaring a top-level `var ort`; evaluated as a function body
    // that variable is local, so return it explicitly (an indirect eval would not expose it).
    ort = new Function(text + '\n;return typeof ort !== "undefined" ? ort : self.ort;')(); // eslint-disable-line no-new-func
    if (!ort || !ort.InferenceSession) throw new Error('classic bundle did not define ort');
    self.ort = ort;
    source = classicUrl;
  }
  ort.env.wasm.wasmPaths = { mjs: vendor + 'ort-wasm-simd-threaded.mjs', wasm: vendor + 'ort-wasm-simd-threaded.wasm' };
  ort.env.wasm.numThreads = numThreads > 0 ? numThreads : 1;
  ort.env.wasm.simd = simd !== false;
  ort.env.wasm.proxy = false;
  if (ort.env.logLevel !== undefined) ort.env.logLevel = 'warning';
  S.ortSource = source;
  return ort;
}

function checkSignature(session, name, inputNames, outputNames) {
  const inp = session.inputNames || [];
  const out = session.outputNames || [];
  for (const n of inputNames) if (!inp.includes(n)) throw new Error(name + ': model has no input ' + n + ' (has ' + inp.join(',') + ')');
  for (const n of outputNames) if (!out.includes(n)) throw new Error(name + ': model has no output ' + n + ' (has ' + out.join(',') + ')');
}

async function createSession(ort, bytes, label) {
  const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all', intraOpNumThreads: 1, interOpNumThreads: 1 };
  const t0 = now();
  const session = await ort.InferenceSession.create(bytes, opts);
  return { session, createMs: now() - t0, label };
}

function makeFeedTensors(ort, buffers, names, dims) {
  const feeds = {};
  for (const n of names) feeds[n] = new ort.Tensor('float32', buffers[n], [1, dims[n]]);
  return feeds;
}

async function init(msg) {
  const t0 = now();
  const modelsUrl = dirUrl(msg.modelsUrl || new URL('../../models/free/', import.meta.url).href);
  S.cfg.set = msg.set || 'leader';
  S.cfg.inferEvery = msg.inferEvery > 0 ? Math.floor(msg.inferEvery) : 1;
  S.cfg.telemetry = !!msg.telemetry;
  S.cfg.telemetryTicks = msg.telemetryTicks > 0 ? msg.telemetryTicks : 2000;

  const [meta, skeleton, initPose] = await Promise.all([
    fetchJson(modelsUrl + 'meta.json'), fetchJson(modelsUrl + 'skeleton.json'), fetchJson(modelsUrl + 'init_pose.json'),
  ]);
  const mapMeta = meta.models && meta.models['mapping_' + S.cfg.set];
  const trackMeta = meta.models && meta.models['tracking_' + S.cfg.set];
  if (!mapMeta || !trackMeta) throw new Error('meta.json has no mapping_/tracking_' + S.cfg.set);

  const ort = await loadOrt(msg.ortUrl, msg.wasmDir, msg.numThreads, msg.simd);
  S.ort = ort;

  const mapBytes = await fetchBytes(modelsUrl + mapMeta.file, 'mapping');
  const trackBytes = await fetchBytes(modelsUrl + trackMeta.file, 'tracking');
  const m = await createSession(ort, mapBytes, 'mapping');
  const t = await createSession(ort, trackBytes, 'tracking');
  checkSignature(m.session, 'mapping', MAPPING_INPUT_NAMES, MAPPING_OUTPUT_NAMES);
  checkSignature(t.session, 'tracking', TRACKING_INPUT_NAMES, TRACKING_OUTPUT_NAMES);
  S.mapping = m.session;
  S.tracking = t.session;

  S.pipeline = new DPPipeline({
    skeleton, initPose,
    rootCorrection: msg.rootCorrection !== undefined ? msg.rootCorrection : 0.35,
    eulerRatio: msg.eulerRatio !== undefined ? msg.eulerRatio : 0.5,
    useMeasuredFrame: msg.useMeasuredFrame !== false,
  });
  S.mapFeeds = makeFeedTensors(ort, S.pipeline.mappingWork.feeds, MAPPING_INPUT_NAMES, MAPPING_INPUT_DIMS);
  S.trackFeeds = makeFeedTensors(ort, S.pipeline.trackingFeeds, TRACKING_INPUT_NAMES, TRACKING_INPUT_DIMS);

  // warm-up (JIT of the wasm kernels); the pipeline state is untouched
  const w0 = now();
  await S.mapping.run(S.mapFeeds);
  const w1 = now();
  await S.tracking.run(S.trackFeeds);
  const w2 = now();

  S.ready = true;
  post({
    type: 'ready',
    loadMs: now() - t0,
    createMs: { mapping: m.createMs, tracking: t.createMs },
    warmupMs: { mapping: w1 - w0, tracking: w2 - w1 },
    ortVersion: (ort.env.versions && (ort.env.versions.web || ort.env.versions.common)) || null,
    ortSource: S.ortSource,
    simd: ort.env.wasm.simd,
    numThreads: ort.env.wasm.numThreads,
    style: meta.style, character: meta.character, set: S.cfg.set,
    models: { mapping: { file: mapMeta.file, bytes: mapBytes.byteLength }, tracking: { file: trackMeta.file, bytes: trackBytes.byteLength } },
    joints: skeleton.joints,
    parents: skeleton.parents,
    boneLengths: skeleton.boneLengths,
  });
}

function onFrame(msg) {
  if (!S.ready || S.disposed) return;
  S.pipeline.pushFrame(msg.frame, 0);
  S.frameCount++;
  if (S.busy) { S.dropped++; return; }
  if (S.cfg.inferEvery > 1 && (S.frameCount % S.cfg.inferEvery) !== 0) return;
  runInference(msg.t, msg.seq).catch((e) => { S.busy = false; fail('run', e); });
}

async function runInference(t, seq) {
  S.busy = true;
  const p = S.pipeline;
  const t0 = now();
  p.buildMappingInputs();
  const mres = await S.mapping.run(S.mapFeeds);
  S.mapOut.Positions = mres.Positions.data;
  S.mapOut.RootMotion = mres.RootMotion.data;
  p.applyMappingOutputs(S.mapOut);
  const t1 = now();
  p.buildTrackingInputs();
  const tres = await S.tracking.run(S.trackFeeds);
  for (const n of TRACKING_OUTPUT_NAMES) S.trackOut[n] = tres[n].data;
  const pose = p.applyTrackingOutputs(S.trackOut, 0);
  const t2 = now();
  S.inferCount++;

  const buffer = S.pool.length ? S.pool.pop() : new ArrayBuffer(POSE_BUFFER_BYTES);
  new Float32Array(buffer, 0, POSE_FLOATS).set(pose.positionsWorldS);
  new Float32Array(buffer, POSE_FLOATS * 4, 3).set(pose.rootS);
  new Float32Array(buffer, (POSE_FLOATS + 3) * 4, 4).set(pose.contacts);
  post({
    type: 'pose', tick: pose.tick, t, seq, buffer, rootYawS: pose.rootYawS, clamps: pose.clamps,
    inferenceMs: t2 - t0, mappingMs: t1 - t0, trackingMs: t2 - t1, dropped: S.dropped, inferCount: S.inferCount,
  }, [buffer]);

  if (S.cfg.telemetry) recordTelemetry(pose.tick, t);
  S.busy = false;
}

function copyF32(a) {
  return Float32Array.from(a);
}

function recordTelemetry(tick, t) {
  const p = S.pipeline;
  const entry = {
    tick, t,
    mappingInputs: {}, mappingOutputs: { Positions: copyF32(S.mapOut.Positions), RootMotion: copyF32(S.mapOut.RootMotion) },
    trackingInputs: {},
    trackingOutputFrame0: {
      VelocitiesV2: copyF32(p.pred.velocities), Positions: copyF32(p.pred.positions), Rotations: copyF32(p.pred.rotations),
      FootContactLabels: copyF32(p.pred.contacts), RootMotion: copyF32(p.pred.rm),
    },
    measuredRm: copyF32(p.measuredRm),
    avatarRm: copyF32(p.avatar.rm),
    worldDP: copyF32(p.avatar.world),
  };
  for (const n of MAPPING_INPUT_NAMES) entry.mappingInputs[n] = copyF32(p.mappingWork.feeds[n]);
  for (const n of TRACKING_INPUT_NAMES) entry.trackingInputs[n] = copyF32(p.trackingFeeds[n]);
  if (S.telemetry.length < S.cfg.telemetryTicks) S.telemetry.push(entry);
  else { S.telemetry[S.telemetryNext] = entry; S.telemetryNext = (S.telemetryNext + 1) % S.cfg.telemetryTicks; }
}

function round7(v) {
  return v === 0 ? 0 : Number(v.toPrecision(7));
}

function jsonify(v) {
  if (v instanceof Float32Array || v instanceof Float64Array) return Array.from(v, round7);
  if (Array.isArray(v)) return v.map(jsonify);
  if (v && typeof v === 'object') {
    const o = {};
    for (const k of Object.keys(v)) o[k] = jsonify(v[k]);
    return o;
  }
  return v;
}

function exportTelemetry() {
  // chronological order
  const n = S.telemetry.length;
  const ticks = [];
  const start = n < S.cfg.telemetryTicks ? 0 : S.telemetryNext;
  for (let i = 0; i < n; i++) ticks.push(jsonify(S.telemetry[(start + i) % n]));
  return {
    format: 'dancing-points-telemetry/1',
    set: S.cfg.set,
    layout: 'channel-major flat[c*T + t]; rotations 9 per joint row-major; frames DP space',
    inferEvery: S.cfg.inferEvery,
    ortSource: S.ortSource,
    ticks,
  };
}

function dispose() {
  S.disposed = true;
  S.ready = false;
  const rel = async () => {
    try { if (S.mapping && S.mapping.release) await S.mapping.release(); } catch (e) { /* ignore */ }
    try { if (S.tracking && S.tracking.release) await S.tracking.release(); } catch (e) { /* ignore */ }
    S.mapping = S.tracking = null;
  };
  rel();
}

self.onmessage = (ev) => {
  const msg = ev.data;
  if (!msg || typeof msg !== 'object') return;
  try {
    switch (msg.type) {
      case 'init':
        init(msg).catch((e) => fail('init', e));
        break;
      case 'frame':
        onFrame(msg);
        break;
      case 'recycle':
        if (msg.buffer && msg.buffer.byteLength === POSE_BUFFER_BYTES && S.pool.length < 8) S.pool.push(msg.buffer);
        break;
      case 'config':
        if (msg.inferEvery > 0) S.cfg.inferEvery = Math.floor(msg.inferEvery);
        if (msg.telemetry !== undefined) S.cfg.telemetry = !!msg.telemetry;
        if (S.pipeline) {
          if (msg.rootCorrection !== undefined) S.pipeline.rootCorrection = msg.rootCorrection;
          if (msg.eulerRatio !== undefined) S.pipeline.eulerRatio = msg.eulerRatio;
        }
        break;
      case 'reset':
        if (S.pipeline) S.pipeline.reset();
        S.frameCount = 0;
        S.dropped = 0;
        break;
      case 'telemetry':
        post({ type: 'telemetry', data: exportTelemetry() });
        break;
      case 'dispose':
        dispose();
        break;
      default:
        break;
    }
  } catch (e) {
    fail(msg.type, e);
  }
};
