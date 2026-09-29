# Native Port: Unity 2022.3 + Meta XR + Sentis

The WebXR app (`webxr/`) is the first, immediately testable version. For an arcade launcher that
expects an APK, or for more headroom on the Quest 2 (multi-threaded inference, hand IK, richer
avatars), a native build is the next step. This note describes how to get there **without
re-inventing the pieces that are already specified and tested** in this repository.

## What already exists natively

The authors' Unity framework — <https://github.com/PeizhuoLi/dancing-points-unity> (Unity
**2022.3.54f1**, `com.unity.sentis` 2.0.0) — already contains the runtime the WebXR app ports:

| C# class | Role | WebXR equivalent |
|---|---|---|
| `ONNXMappingAndTracking` (`Assets/Projects/DancingPoints/Scripts/`) | per-frame mapping → tracking chain, relative root handling, Euler blend, bone-length restore, foot/hand IK | `webxr/src/net/pipeline.js` + `worker.js` (§6 of `docs/DESIGN.md`) |
| `SentisONNXNetworkWithMetaInfo` (`Assets/Scripts/Animation/AI/`) | loads the ONNX + `metadata_props` (window sizes, input joints, fps), allocates input buffers, runs the Sentis worker | `worker.js` (`meta.json`) |
| `FeatureExtractor` (`Assets/Scripts/Animation/`) | Positions / Rotations / VelocitiesV2 / RootMotion / FootContactLabels features in root space | `pipeline.js` (`toLocal`, `rmFromRoot`, …) |
| `RootMotionEdits.RootMotionType` | `[cos φ, sin φ, x, z]` ⇄ `Matrix4x4`, relative root maths | `pipeline.js` (`rmRelative`, `rmApply`) |
| `RootModule` (`TOPOLOGY.ThreePts`) | root = head projected to the floor, forward = head facing | `pipeline.js` (`rootFromHead`) |
| `Actor`, `Blueman` | 34-joint simplified skeleton, bone lengths | `webxr/models/free/skeleton.json` |

The demo scenes replay dataset motion from `MotionEditor` assets. For a live VR build the leader
input comes from the headset instead; everything downstream of "3-point window in world space"
stays the same.

## What the WebXR app defines and the port must keep 1:1

1. **Choreography format** `dancing-points-choreo/1` (`docs/DESIGN.md` §5): stage frame S
   (right-handed, Y up, forward −Z, metres at `referenceHeight` 1.70), head pose + hand positions
   at 30 fps, moves in beats, optional `fullBody` (base64 float32). Unity is left-handed: convert
   with `z → −z` and quaternion `(x, y, z, w) → (−x, −y, z, w)` — the same mapping the WebXR app
   uses for the network ("S ⇄ DP" in `pipeline.js`), verified in `tests/fixtures/math.json`.
2. **Calibration** (`webxr/src/xr/calibration.js`): standing head height `h0`, facing yaw, floor
   origin; scale `k = referenceHeight / h0`; persisted per headset.
3. **Scoring** (`docs/DESIGN.md` §7, `webxr/src/game/scoring.js`): per-tick features (head,
   hands relative to head, velocities), σ values from `config.SCORING`, ±0.2 s best-δ window,
   energy gate, grades, stars, combo, timing bias. Port it as a pure C# class and validate it with
   the same numbers the JS unit tests assert (`tests/unit/scoring.test.js`): exact playback 100,
   0.35 m noise ≤ 70, standing still ≤ 40 per move.
4. **Duo protocol** (§9): WebSocket JSON messages (`hello`, `ping/pong`, `create/join`, `start`
   with server `startAt`, `state` at 15 Hz, `result`), room codes, clock sync (median of 5). The
   Node server in `server/` stays the backend; a native client only needs a WebSocket library.
5. **Results/benchmark** (`webxr/src/duo/benchmark.js`): `syncDistance` (offset-removed mean
   3-point distance), `syncLag` (cross-correlation of head speed), winner, CSV/JSON layout.
6. **Model directory contract** (`webxr/models/<style>/meta.json`, `skeleton.json`,
   `init_pose.json`; produced by `tools/prepare_models.py`). Sentis loads the same `.onnx` files
   (fp32 or int8 — Sentis 2.x supports the `DynamicQuantizeLinear`/`MatMulInteger` ops; if a build
   complains, use the fp32 checkpoints and let Sentis quantize with `ModelQuantizer`).

## Quest build settings (Unity 2022.3)

- Packages: `com.unity.xr.openxr` (Meta Quest feature group) **or** the Meta XR SDK
  (`com.meta.xr.sdk.core`); `com.unity.sentis` 2.x; Input System.
- Player: Android, IL2CPP, ARM64, Vulkan or OpenGLES3 (test both; Vulkan is the default for
  Quest), Minimum API 29, *Multithreaded Rendering* on, *Optimized Frame Pacing* off, texture
  compression ASTC.
- XR: OpenXR with Oculus Touch Controller Profile + Hand Interaction; render mode *Multi-view*;
  target 72 Hz; fixed foveated rendering level 1–2.
- Sentis: `BackendType.GPUCompute` for the mapping/tracking nets (the CVAE decoder's 2048×15540
  GEMM is the hot spot) and run inference **asynchronously** across frames
  (`Worker.ScheduleIterable` / `Schedule` + `ReadbackRequest`) at 30 Hz so the 72 Hz render loop
  never waits. Fall back to `BackendType.CPU` with the int8 model if the GPU path stalls.

## Step list

1. Fork `dancing-points-unity`, remove the editor-only demo pipelines, keep
   `Assets/Scripts/Animation/AI`, `FeatureExtractor`, `RootMotionEdits`, `Actor`, `Blueman`,
   `ONNXMappingAndTracking` (strip the `MotionEditor` dependency: feed the leader 3-point window
   from a ring buffer filled by the XR camera + controller transforms; use `init_pose.json` for the
   first autoregressive frame; keep `SolveFootIK` — it is a real advantage over the web version).
2. Add a `ChoreoLoader` (JSON → structs; converting S → Unity), `BeatClock` (AudioSettings.dspTime),
   `AudioEngine` (synth patterns as AudioClips or a small procedural drum machine; operator tracks via
   `UnityWebRequestMultimedia`), `Scorer` (port), `PlaySession`, `Recorder`.
3. UI: world-space canvas panels (same texts as `webxr/src/ui/texts.js`), XR Interaction Toolkit
   ray interactors; HUD on the stage, never head-locked.
4. Duo: `System.Net.WebSockets.ClientWebSocket` against `server/` (`/ws`), same messages.
5. Persistence: results/runs/choreos as JSON in `Application.persistentDataPath`, plus the REST
   endpoints of `server/` (`/api/choreos`, `/api/results`, `/api/runs`, optional `X-Dp-Token`).
6. Validate the neural pipeline with the committed fixtures (`tests/fixtures/*.json`): a C# test
   that builds the tensors from `mapping_io.json` / `tracking_io.json` and compares to the expected
   values (inputs 1e-5, post-processing 1e-4) proves the port before the first headset run.
7. Build the APK, sideload with `adb install`, and hand it to the arcade launcher.

## Effort estimate

Roughly two to three weeks for one Unity developer familiar with XR: one week for the runtime
pipeline + avatar (most of it exists), one week for game flow/UI/scoring/recorder, a few days for
duo/networking and store/launcher packaging. The WebXR version remains the fastest way to iterate
on choreographies, scoring parameters and network variants, because all of those are shared files.
