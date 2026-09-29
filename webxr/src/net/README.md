# `src/net` — the neural pipeline in the browser

Port of the Dancing Points runtime (Python `data_process/*` + Unity `ONNXMappingAndTracking.cs`)
to plain JavaScript, running the int8 ONNX models with onnxruntime-web (WASM, SIMD) in a Web
Worker. The binding spec is `docs/DESIGN.md` §6; this file explains how the three modules fit
together, the worker message protocol and how to swap models.

| file | runs on | responsibility |
|---|---|---|
| `pipeline.js` | anywhere (Node, worker, page) | pure maths: roots / root motion, DP↔S, tensor assembly, output decoding, post‑processing, `DPPipeline` closed‑loop state. No three.js, no onnxruntime. |
| `worker.js` | Web Worker (ES module) | loads onnxruntime-web + the models, owns one `DPPipeline`, runs mapping + tracking per frame, posts poses, keeps telemetry. |
| `avatar-driver.js` | main thread | `NeuralAvatarDriver`: resamples the 72 Hz input to 30 Hz, talks to the worker, interpolates poses for rendering, performance guard, telemetry export. No three.js. |

Tests: `tests/unit/pipeline-math.test.js`, `tests/unit/pipeline.test.js` (fixture parity with the
numpy oracle `tools/dp_pipeline.py`; `npm test`), `tests/net/ort-parity.mjs` (int8 vs fp32 through
onnxruntime-node, optional), `tests/net/worker-smoke.mjs` (headless Chromium, real worker + models).

## 1. What the pipeline does (per 30 Hz tick)

Input: one **3‑point frame** in DP space (Unity, left‑handed, +Z forward, metres, *unscaled*):
`[head x y z, head quat x y z w, right wrist x y z, left wrist x y z]`. The driver builds it from
the stage‑frame sample (`sampleSToFrameDP`: divide by the calibration scale `k`, `z → −z`,
quaternion `(x,y,z,w) → (−x,−y,z,w)`).

1. **Root** of every frame: `pos = (head.x, 0, head.z)`, `yaw = atan2(fwd.x, fwd.z)` with
   `fwd = quat·(0,0,1)`; if the ground projection is < 1e‑3 the previous yaw is kept.
   Root motion `rm = [cos yaw, −sin yaw, x, z]`, matrix `M = T(x,0,z)·R_y(yaw)`.
2. **Mapping** (`mapping_leader`, 15 frames `t−14…t`, ring buffer pre‑filled with the first frame):
   root‑local positions of head/right/left and the roots relative to the *oldest* frame, channel‑major
   (`flat[c·T + t]`) → 30 predicted future frames `t+1…t+30` (local positions + relative roots).
3. **Tracking** (`tracking_leader`): leader window = `[measured frame t] + 30 mapping frames`, roots
   relative to the **avatar root**; follower inputs = the avatar state (velocities, positions,
   rotations, contacts, `rm = [1,0,0,0]`). Output frame 0 (= `t+1`) is used.
4. **Post‑processing** (`postProcess`, bit‑exact with the Python oracle to ~6e‑7): new root
   `apply(avatar.rm, Δrm)` → root correction towards the measured head root (`rootCorrection = 0.35`)
   → world positions → Euler blend with the previous world positions and the predicted velocities
   (`eulerRatio = 0.5`) → bone‑length restore (Unity `Transform` semantics: a clamped joint drags its
   whole subtree, tolerance 1.05) → positions re‑expressed in the new root, contacts clamped to [0,1].
5. Pose out: `positionsWorldS` (34 joints × xyz, stage frame S, **unscaled** metres), `rootS`,
   `rootYawS = −yaw_DP`, contacts, timings.

`DPPipeline` exposes the steps separately so the worker can await the ONNX runs in between:
`pushFrame → buildMappingInputs → applyMappingOutputs → buildTrackingInputs → applyTrackingOutputs`,
or `await pipeline.runTick(frame, { runMapping, runTracking })`. All buffers are allocated once.

## 2. The worker

`worker.js` is an **ES module worker** (`new Worker(url, { type: 'module' })`) so that it can
`import './pipeline.js'` (a classic worker cannot import ES modules). onnxruntime-web is loaded by
dynamic `import()` of `../../vendor/ort/ort.wasm.min.mjs` (the ESM build from `onnxruntime-web/dist`);
the wasm glue `ort-wasm-simd-threaded.{mjs,wasm}` is taken from the same directory through
`ort.env.wasm.wasmPaths`. If the `.mjs` is missing the classic bundle `ort.wasm.min.js` is fetched
and evaluated instead (both paths are exercised by `worker-smoke.mjs`). Settings: `numThreads` as
requested (1 unless the page is cross‑origin isolated), `simd = true`, `proxy = false`.

Models are fetched as bytes (with progress messages) and turned into `InferenceSession`s
(`executionProviders: ['wasm']`, graph optimisation `all`). Feed tensors are created **once** over
the pipeline's `Float32Array`s; per run only the outputs are allocated by onnxruntime. A warm‑up
run of both models happens before `ready`.

Frames may arrive faster than inference runs: every frame is pushed into the 30 Hz history ring,
but a new inference starts only when the previous one has finished (`busy` → the frame is counted
as `dropped`) and only every `inferEvery`‑th frame (the guard sets 2 = 15 Hz).

### Message protocol

Main → worker

| message | fields | effect |
|---|---|---|
| `init` | `modelsUrl` (dir, `meta.json`/`skeleton.json`/`init_pose.json`/onnx), `set` (`leader`), `ortUrl?`, `wasmDir?`, `rootCorrection`, `eulerRatio`, `inferEvery`, `telemetry`, `telemetryTicks`, `numThreads`, `simd`, `useMeasuredFrame?` | load everything, warm up, reply `ready` or `error` |
| `frame` | `seq`, `t` (s), `frame: Float32Array(13)` DP | push into the history; run inference if idle |
| `config` | any of `inferEvery`, `rootCorrection`, `eulerRatio`, `telemetry` | change at runtime |
| `reset` | – | forget history + avatar (re‑initialised at the next frame's root) |
| `telemetry` | – | reply `telemetry` |
| `recycle` | `buffer` (transferred) | give a pose buffer back to the pool |
| `dispose` | – | release the sessions |

Worker → main

| message | fields |
|---|---|
| `ready` | `loadMs`, `createMs{mapping,tracking}`, `warmupMs{…}`, `ortVersion`, `ortSource`, `simd`, `numThreads`, `style`, `character`, `set`, `models{mapping,tracking:{file,bytes}}`, `joints`, `parents`, `boneLengths` |
| `progress` | `label`, `loaded`, `total` (bytes) |
| `pose` | `tick`, `t`, `seq`, `buffer` (transferred `ArrayBuffer`, 436 B: `Float32Array(102)` positions S at 0, `Float32Array(3)` rootS at 408, `Float32Array(4)` contacts at 420), `rootYawS`, `clamps`, `inferenceMs`, `mappingMs`, `trackingMs`, `dropped`, `inferCount` |
| `telemetry` | `data`: `{format:'dancing-points-telemetry/1', set, layout, inferEvery, ortSource, ticks:[{tick, t, mappingInputs{…}, mappingOutputs{Positions[270], RootMotion[120]}, trackingInputs{7 tensors}, trackingOutputFrame0{VelocitiesV2[102], Positions[102], Rotations[306], FootContactLabels[4], RootMotion[4]}, measuredRm, avatarRm, worldDP[102]}]}` (values rounded to 7 significant digits) |
| `error` | `stage` (`init`, `run`, …), `message` |

The telemetry ring keeps the last `telemetryTicks` (default 2000) ticks in memory (~8 KB/tick:
all inputs, full mapping outputs, tracking output frame 0 — the other 29 predicted frames are not
stored to keep the Quest's memory in check; change `recordTelemetry` if you need them).

## 3. The driver (main thread)

```js
import { NeuralAvatarDriver } from './src/net/avatar-driver.js';

const driver = new NeuralAvatarDriver({
  style: 'free',                    // -> ./models/free/ (relative to src/net/)
  telemetry: params.telemetry,
  onStatus: ({ state, notice }) => hud.notice(notice),   // German texts in NOTICES
  onError: (e) => console.warn(e),
});
await driver.init();                // loads ~86 MB of models, resolves with the worker's ready info
driver.start();

// render loop (72 Hz)
const sampleS = calibration.toStage(input.sample, choreo.referenceHeight);
driver.update(sampleS);             // resamples to 30 Hz and posts DP frames
const joints = driver.getPose();    // Float32Array(102), stage frame S, UNSCALED metres, or null
// to world (three.js) at the player's physical place:
calibration.stageToWorld(out, p, calibration.h0);   // scale 1
// to compare with the teacher at reference height: multiply by driver.scale (= sample.k)
```

* `update(sampleS, tSec?)` linearly resamples the incoming samples onto the 30 Hz grid
  (`sample.t` is the clock), up to 4 catch‑up ticks per call, resyncs when far behind.
  `pushFrameDP(frame13, t)` feeds a ready DP frame (tests, tools).
* `getPose(out?, tMs?)` interpolates between the last two poses using a smoothed arrival interval
  (EMA of the gaps, clamped to 1–4 ticks), so 15 Hz mode and slow devices still render smoothly.
  `driver.rootS`, `driver.latest` (`{tick, t, positionsWorldS, rootS, rootYawS, contacts,
  inferenceMs, mappingMs, trackingMs, arrivedMs}`) accompany it.
* States: `idle → loading → ready → running ⇄ slow → disabled | error | disposed`. Notices
  (`NOTICES`): `'Neural-Avatar wird geladen …'`, `'Neural-Avatar bereit'`, `'Neural-Avatar:
  reduzierte Rate (15 Hz)'`, `'Neural-Avatar deaktiviert (zu langsam)'`, `'Neural-Avatar nicht
  verfügbar'`.
* **Performance guard**: mean `inferenceMs` over the last 60 inferences (after 10): > 25 ms →
  `inferEvery = 2` (15 Hz, state `slow`), back to 30 Hz below 17.5 ms; > 60 ms → the worker is
  terminated, state `disabled`, notice `'Neural-Avatar deaktiviert (zu langsam)'` — the app then
  shows the `PointAvatar`. `?avatar=points|off` is handled by the app (do not create the driver).
* `setConfig({rootCorrection, eulerRatio, telemetry})`, `reset()`, `stop()`, `dispose()`,
  `exportTelemetry() → Promise<object>`, `downloadTelemetry(filename?)` (browser only).
* Pose buffers are transferred worker → main and handed back (`recycle`) — no per‑tick garbage
  beyond the small message objects and the 52‑byte frame copy.

## 4. Swapping / improving models

The worker only needs a directory with:

* `meta.json` (`dancing-points-models/1`): `models.mapping_<set>` and `models.tracking_<set>` with
  `file` and the `inputs`/`outputs` names — the names must be exactly those of §6.1
  (`input_leader_Positions`, …); tensors are `float32 [1, N]`.
* `skeleton.json` (34 joints, `parents` before children, `boneLengths`) and `init_pose.json`
  (root‑local `positions`, `rotations` 34×9, `velocities`, `contacts`).

`tools/prepare_models.py --checkpoints <dir|checkpoints.tar> --style <style> --set leader --out
webxr/models` produces all of it (int8 dynamic quantisation, deterministic). Then:

* `?style=<style>` (app) or `new NeuralAvatarDriver({ style })` / `{ modelsUrl }` for another
  directory (e.g. an fp32 export, a fine‑tuned or interpretability‑edited network exported to ONNX
  with the same signature).
* Check parity and speed without a headset: `node tests/net/ort-parity.mjs` (needs
  `onnxruntime-node`, otherwise prints SKIP) and `node tests/net/worker-smoke.mjs --models <dir>`.
* Record what the network sees: `?telemetry=1`, then "Telemetrie exportieren" (or
  `driver.downloadTelemetry()`) — the JSON has the exact input tensors per tick, so an offline
  script can replay them into a modified network (`tools/dp_pipeline.py` reads the same layout).

Post‑processing parameters can be changed live (`setConfig`) to compare drift/jitter trade‑offs
(`rootCorrection` 0 … 1, `eulerRatio` 0 … 1); the numpy reference reports the same numbers
(`python tools/dp_pipeline.py --validate …`).

## 5. Measured (headless Chromium, Playwright build 1194, this container, 4 cores; `tests/net/worker-smoke.mjs`)

| configuration | mapping | tracking | total per inference | effect |
|---|---|---|---|---|
| int8, 1 wasm thread (no cross‑origin isolation) | 10 ms | 24 ms | **34 ms** mean (28–48) | guard → 15 Hz mode |
| fp32, 1 thread | 18 ms | 25 ms | 44 ms | slower and 4× larger → int8 stays |
| int8, 4 threads (page cross‑origin isolated) | 5.7 ms | 10 ms | **16 ms** mean | full 30 Hz |
| int8, onnxruntime‑node (native CPU, 1 thread) | – | – | 5.5 ms | reference |

Model load in Chromium: ~1.1 s from localhost (session creation 0.6 + 0.1 s, warm‑up 40 + 35 ms).
A Quest 2 (single WASM thread) will be several times slower than this desktop CPU; the two levers
are (1) serving the app **cross‑origin isolated** (`Cross-Origin-Opener-Policy: same-origin`,
`Cross-Origin-Embedder-Policy: require-corp` — the LAN server can send them; GitHub Pages cannot
set headers, but a service worker can add them to its responses, "coi-serviceworker" style),
which lets the driver use `min(4, cores − 1)` threads automatically, and (2) the guard's 15 Hz
mode. Below 60 ms per inference the neural avatar stays on. The pipeline's own JS costs ~11 µs
per tick and allocates ≈ 40 B (onnxruntime allocates the 62 KB of output tensors per run).
