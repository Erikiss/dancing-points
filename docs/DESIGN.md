# Dancing Points VR — Design Document (developer contract)

This document is the binding contract for everything under `webxr/`, `server/`, `tools/` and
`tests/`. Several people/agents implement modules in parallel; **if you need to deviate from an
interface described here, change this document in the same change.**

The product: a Just‑Dance‑style VR dance game for **Meta Quest 2** (standalone, in the Quest
Browser as a WebXR PWA), built on the *Dancing Points* three‑point (head + two hands) technology of
this repository. The operator use case is a VR arcade ("Fun and Play"‑style): a player picks a
~30 s dance from a menu, imitates it, gets scored; a duo mode lets two players compete/benchmark;
operators can add their own songs and TikTok dances over WLAN.

Language: code + comments in English, **player‑facing UI text in German** (arcade in Chemnitz).

---

## 1. Why WebXR first (and how native comes later)

| Requirement (from the brief) | Decision |
|---|---|
| "Fast‑shot solution that runs stable on Quest 2 and that I can try right away" | Static WebXR app (three.js), zero build step, opened in the Quest Browser. Also installable as PWA and packageable to an APK with Meta's `ovr-platform-util create-pwa`. |
| Performant | 72 Hz render loop, tiny scene, all heavy work (neural nets) in a Web Worker at 30 Hz, int8‑quantized ONNX, automatic fallback to a light avatar when the device is too slow. |
| Extendable for online/WLAN operation, own songs, own TikTok dances | Choreographies are JSON files (§5). A small Node server (`server/`) serves the app over HTTPS in the LAN, stores uploaded choreographies/results and relays duo‑mode state over WebSocket. The app needs an HTTP(S) origin (ES modules + WebXR secure context: it does not boot from `file://`, so there is no USB‑stick mode); without a server it runs from GitHub Pages with results in `localStorage`. Offline start / model caching through the service worker need a **trusted** certificate (GitHub Pages, or `server.js --cert/--key`): Chromium refuses a service worker on a clicked‑through self‑signed certificate. |
| Improve the network later (mechanistic interpretability etc.) | The network is a drop‑in ONNX file + `meta.json`; the runtime pipeline is a documented, tested port of the Python/Unity feature pipeline (§6). A telemetry toggle records every network input/output frame for offline analysis. |
| Menu, scored dances, 30 s sequences, Snoop Dogg C‑Walk with 6 combos | §4, §5, `webxr/choreos/snoop-cwalk.json`. |
| Duo mode with a metric / small benchmark | §7. |

A native Unity/Meta‑XR port is described in `docs/NATIVE_PORT.md` (later). Everything platform
independent (choreo format, scoring, pipeline math, protocols) is specified here so that the native
port reuses it 1:1.

---

## 2. Repository layout (new files)

```
webxr/                      the app (static; open index.html via any static server)
  index.html                import map (three -> ./vendor/three.module.js), boots src/app.js
  manifest.webmanifest      PWA manifest (name "Dancing Points VR")
  sw.js                     service worker: network-first shell (src/, choreos/), cache-first assets (vendor/, models), network-first /api
  vendor/three.module.js    three r170 (vendored, MIT)
  vendor/ort/ort.wasm.min.js, ort-wasm-simd-threaded.{wasm,mjs}   onnxruntime-web 1.20.1 (vendored, MIT)
  models/free/              free-style models (int8) + meta.json + skeleton.json + init_pose.json   (§6)
  choreos/index.json        list of shipped choreographies
  choreos/*.json            choreographies (§5)
  src/app.js                App: boot, state machine, main loop
  src/config.js             constants, URL params, feature flags
  src/util/math.js          vec3 / quat / complex helpers on plain arrays (no three.js)
  src/util/events.js        tiny EventEmitter
  src/xr/input.js           XRInput (WebXR) and input backend interface
  src/xr/calibration.js     Calibration -> stage frame + height normalisation
  src/emu/emulated-input.js desktop/scripted input backend (used by tests)
  src/game/clock.js         BeatClock (audio-clock based master timeline, speed factor for tests)
  src/game/audio.js         AudioEngine (synth beat, custom track, count-in, SFX)
  src/game/choreo.js        loadChoreo / Choreo class / validation / interpolation
  src/game/scoring.js       Scorer (pure, deterministic, unit-tested)
  src/game/session.js       PlaySession (clock + choreo + input + scorer + events)
  src/game/recorder.js      Recorder (record a new choreography in the headset)
  src/render/scene.js       SceneKit (renderer, XR, floor, lights, stage)
  src/render/avatar.js      PointAvatar, BodyAvatar, TeacherAvatar
  src/render/hud.js         HUD (world-space canvas panels)
  src/ui/menu.js            Menu (world-space panels, laser pointer, mouse in emu mode)
  src/ui/texts.js           all German UI strings
  src/net/pipeline.js       DPPipeline: Dancing Points feature pipeline port (pure JS)
  src/net/worker.js         inference Web Worker (onnxruntime-web)
  src/net/avatar-driver.js  NeuralAvatarDriver (main-thread glue + perf guard)
  src/duo/ghost.js          GhostDuo (replay of a saved run)
  src/duo/online.js         OnlineDuo (WebSocket client)
  src/duo/benchmark.js      metrics + export
server/
  server.js                 Node 18+: HTTPS/HTTP static + REST + WebSocket relay
  package.json              deps: ws, selfsigned
tools/
  prepare_models.py         checkpoints -> webxr/models/<style>/ (int8 + meta + skeleton + init pose)
  dp_pipeline.py            numpy reference implementation of §6 (source of truth for fixtures)
  gen_fixtures.py           writes tests/fixtures/*.json from dp_pipeline.py + onnxruntime
  precompute_teacher.py     runs tracking net offline over a choreo -> fullBody frames in the JSON
  gen_choreos.js            procedural choreography generator (Node) -> webxr/choreos/*.json
tests/
  unit/*.test.js            node --test (no browser): math, pipeline parity, scoring, choreo, benchmark
  e2e/smoke.test.js         Playwright (headless Chromium, emu mode): menu, autoplay, scores, no errors
  fixtures/*.json           generated by tools/gen_fixtures.py (committed)
docs/DESIGN.md              this file
docs/NATIVE_PORT.md         notes for a Unity/Meta XR native port
README.md                   operator/player documentation (German section added on top)
.github/workflows/ci.yml    node unit tests + e2e
.github/workflows/pages.yml deploy webxr/ to GitHub Pages
```

ES modules only, no bundler, no TypeScript. Node ≥ 20 for the repository tooling (root
`package.json`: `playwright-core` requires 20), the server alone runs on Node ≥ 18. Python ≥ 3.11
with `numpy`, `onnx`, `onnxruntime` for `tools/*.py` (no PyTorch needed at runtime).

---

## 3. Coordinate frames

All positions in metres, Y up.

| Frame | Handedness | Forward | Used by |
|---|---|---|---|
| **W** — three.js/WebXR world (`local-floor` reference space) | right | −Z | rendering, XR input |
| **S** — Stage frame | right | −Z | choreographies, scoring, recorder, duo sync, teacher avatar |
| **DP** — Dancing Points / Unity space | left | +Z | neural pipeline only (`src/net/*`) |

* **W→S** (`Calibration.toStage`): translate so the calibration point is the origin, rotate about Y
  so the player's calibrated facing direction becomes −Z, then **scale by `k = referenceHeight /
  h0`** where `h0` is the calibrated standing head height and `referenceHeight = 1.70` (from the
  choreography). Choreographies are authored/recorded at reference height; scaling makes a short
  and a tall player comparable. Head orientation quaternions are rotated by the same yaw and not
  scaled.
* **S→DP** and back (`src/net/pipeline.js: toDP / fromDP`): `p_DP = (x, y, −z)`; quaternion
  `(x, y, z, w)_S → (−x, −y, z, w)_DP` (mirror through the XY plane; verified in unit tests against
  `M·R·M`, `M = diag(1,1,−1)`). Use the **unscaled** stage frame (divide by `k` again) for DP, the
  network wants real‑world metres.
* Quaternion layout everywhere: `[x, y, z, w]` (three.js order).

---

## 4. App flow and states

`App` (`src/app.js`) owns one `SceneKit`, one input backend, the `Menu`, `HUD`, and the current
mode object. States:

```
BOOT → MENU ⇄ SETTINGS
MENU → CALIBRATE → COUNTDOWN → PLAYING → RESULTS → MENU
MENU → RECORD_SETUP → CALIBRATE → COUNTDOWN → RECORDING → RECORD_REVIEW → MENU
MENU → DUO_LOBBY → (ghost | online) → CALIBRATE → COUNTDOWN → PLAYING → RESULTS → MENU
```

* **Calibrate**: the player stands on the marked spot facing the stage screen and holds a trigger
  (or waits 3 s in emu mode). We capture `h0` (head height) and the facing yaw. Calibration
  persists in `localStorage` per headset; the menu offers "Neu kalibrieren".
* **Countdown**: count‑in of `countInBeats` beats with metronome clicks; the reference avatar
  starts on beat 0.
* **Playing**: each render frame: sample input → `Calibration.toStage` → `PlaySession.update(t)`
  (scoring at ≤ 30 Hz) → HUD/avatars. The teacher avatar dances in front of the player
  (2.5 m ahead, facing the same direction as the player, i.e. seen from behind — "follow the
  leader"; `mirror: true` choreographies are displayed mirrored and facing the player).
* **Results**: total score, stars, per‑move grades, max combo, timing bias, "Nochmal", "Menü".
  Results are appended to `localStorage['dp.results']` and POSTed to `/api/results` when a server
  is configured.

Global debug handle (always present, used by e2e tests): `window.__dp = { app, version }` with
`app.state`, `app.startSolo(choreoId, opts)`, `app.lastResult`, `app.clock`, `app.input`.

URL parameters (`src/config.js`): `emu=1` (no WebXR, desktop camera + mouse/keyboard input),
`emu=playback` (scripted input: the reference trajectory itself, plus `noise=<metres>` and
`lag=<seconds>`), `choreo=<id>`, `autostart=1`, `speed=<factor>` (clock speed for tests), `server=
<wss://host:port>`, `avatar=neural|points|off`, `style=free`, `telemetry=1`, `lang=de|en`.

Frame budget on Quest 2: render ≤ 8 ms on the main thread; scoring ≤ 0.5 ms; nothing allocates in
the hot loop (reuse typed arrays); the neural worker is off the main thread.

---

## 5. Choreography format `dancing-points-choreo/1`

```jsonc
{
  "format": "dancing-points-choreo/1",
  "id": "snoop-cwalk",                       // [a-z0-9-]+ ; file name = id + ".json"
  "title": "Snoop Dogg C-Walk – 6 Basic Combos",
  "artist": "nach Electro Breakers (TikTok)",
  "bpm": 90, "beatsPerBar": 4, "countInBeats": 4,
  "durationBeats": 48,                       // playable length after count-in (48 beats @ 90 bpm = 32 s)
  "fps": 30,                                 // sampling rate of "frames"
  "referenceHeight": 1.70,
  "mirror": true,                            // reference shown as mirror (TikTok-style); scoring mirrors x and swaps hands
  "difficulty": 2,                           // 1..5 (UI only)
  "audio": { "url": null, "synth": "hiphop", "offsetSec": 0.0, "gain": 0.8 },
  "moves": [                                 // contiguous, ordered, in beats from beat 0
    { "name": "Shoe Vibe", "startBeat": 0, "endBeat": 8, "hint": "…" }
  ],
  "frames": {                                // length = durationBeats*60/bpm*fps (+1), index 0 = beat 0
    "head":  [[x, y, z, qx, qy, qz, qw], ...],   // stage frame S (metres at referenceHeight)
    "left":  [[x, y, z], ...],
    "right": [[x, y, z], ...]
  },
  "fullBody": {                              // optional, produced by tools/precompute_teacher.py
    "joints": ["b_root", "..."], "parents": [-1, 0, ...],
    "fps": 30, "encoding": "base64-float32", "positions": "<base64 of frames*joints*3 float32, stage frame S>"
  },
  "meta": { "source": "procedural|recorded|mocap", "createdAt": "ISO", "author": "", "notes": "" }
}
```

Rules: `frames.head[i]` is at time `i / fps` seconds after beat 0. Beat `b` is at `b * 60 / bpm` s.
`Choreo.sampleAt(t)` linearly interpolates positions (slerp for the head quaternion) and clamps at
the ends. `Choreo.moveAt(t)` returns the move whose `[startBeat, endBeat)` contains the beat.
Validation (`validateChoreo`) rejects wrong lengths, NaNs, non‑contiguous moves; it must be
tolerant to missing optional fields. `choreos/index.json` is `{"choreos": [{"id","title","artist",
"bpm","durationBeats","difficulty","file"}]}`; the server merges its uploaded ones into the list.

Audio: the app never ships copyrighted music. `audio.synth` selects a WebAudio drum pattern
(`hiphop`, `house`, `metronome`); `audio.url` (absolute or relative to the choreo file) plays an
operator‑supplied track instead (licensing is the operator's responsibility; see README).

The Snoop Dogg C‑Walk choreo ships as a **procedural v0** (`tools/gen_choreos.js`, moves in this
order and with these names, 8 beats each): *Shoe Vibe, Restep, Heel Toe, Side Hopping, Shuffle
Legs, Gangster Two Step*. C‑Walk is footwork; with head + hands only, the trackable signal is the
head bounce/sway/hop and the arm swing. The generator encodes for each move the characteristic
head amplitude (vertical bounce, lateral sway, fore/aft), tempo relative to the beat (1×, 2×, ½×),
and arm patterns. The recommended path for a faithful version is to **record it in the headset**
(Recorder, §8) while watching the TikTok video; the recording replaces the file.

---

## 6. Neural pipeline (port of Python `data_process/*` + Unity `ONNXMappingAndTracking.cs`)

### 6.1 Models shipped

`webxr/models/free/meta.json` (generated by `tools/prepare_models.py`; fields are copied from the
ONNX `metadata_props`):

| model | inputs (name → length) | outputs (name → length) |
|---|---|---|
| `mapping_leader` (MLP, 34.8M params, int8 35 MB) | `input_leader_Positions` 135 = 9 feat × 15 frames; `input_leader_RootMotion` 60 = 4 × 15 | `Positions` 270 = 9 × 30; `RootMotion` 120 = 4 × 30 |
| `tracking_leader` (CVAE, 50.4M params, int8 51 MB) | `input_leader_Positions` 279 = 9 × 31; `input_leader_RootMotion` 124 = 4 × 31; `input_follower_VelocitiesV2` 102; `input_follower_Positions` 102; `input_follower_Rotations` 306; `input_follower_FootContactLabels` 4; `input_follower_RootMotion` 4 | `VelocitiesV2` 3060 = 102 × 30; `Positions` 3060; `Rotations` 9180 = 306 × 30; `FootContactLabels` 120 = 4 × 30; `RootMotion` 120 = 4 × 30 |

Normalisation (mean/std) is baked into the ONNX. Model fps = 30. `mapping_follower` /
`tracking_follower` (AI partner) have identical signatures and are optional (`prepare_models.py
--style chacha --set follower`).

**Tensor layout**: every multi‑frame tensor is **channel‑major**: `flat[c * T + t]` for feature
channel `c` (0..C‑1) and frame `t` (0..T‑1, oldest first). Single‑frame inputs are just the feature
vector. Feature channel order within a frame: `Positions` = for each input joint in the order
`b_head, b_r_wrist, b_l_wrist` (**right before left!**) the local `x, y, z`; `RootMotion` = `[cos φ,
sin φ, x, z]`; `VelocitiesV2`/`Positions` (34 joints) = joint order of `skeleton.json`, `x, y, z`
each; `Rotations` = 9 per joint `m00 m01 m02 m10 m11 m12 m20 m21 m22` (Unity row/col naming,
rotation of the joint relative to the root); `FootContactLabels` = `[l_ankle, l_ball, r_ankle,
r_ball]` in 0..1.

### 6.2 The root and root motion (DP space)

For a head pose (position `hp`, rotation `hq`) the **3‑point root** is
`rootPos = (hp.x, 0, hp.z)` and `rootYaw` = yaw of the head's forward direction projected onto the
ground. In the dataset the head bone's local +Y axis is the forward direction (verified on
`Freestyle_Solo_01`: mean angle to root +Z = 0.0°). For the VR headset the forward direction is the
gaze direction (`hq * (0,0,1)` in DP; `hq * (0,0,−1)` in W/S, converted). If the projected vector
is shorter than 1e‑3 (looking straight down/up) keep the previous yaw.

Root as a 4×4 matrix `M = T(rootPos) · R_y(rootYaw)` (Unity: `Quaternion.Euler(0, yawDeg, 0)`).
Root motion 4‑vector `rm = [cos φ, sin φ, x, z]` with `φ = −rootYaw` (Unity
`RootMotionType.FromMatrix4x4`, angle sign flipped because the (x, z) plane is used as a complex
plane). `AsMatrix4x4`: `φ = atan2(sin, cos)`, matrix = `T(x, 0, z) · R_y(−φ)`.

**Relative root** (`relative_motion.transform_root_motion`, `CalculateRelativeRoot` in C#), with
complex numbers `rot = cos + i·sin`, `pos = x + i·z`:
`rot_rel = rot / rot_ref`, `pos_rel = (pos − pos_ref) / rot_ref`. Applying a relative root to a
reference root (Unity `referenceRoot * deltaRoot`) is the inverse: `rot = rot_rel · rot_ref`,
`pos = pos_rel · rot_ref + pos_ref`.

Local joint position = `M⁻¹ · p_world` (Unity `PositionTo`). World = `M · p_local`.

### 6.3 Per‑tick algorithm (30 Hz, in the worker) — self tracking of the player

State: ring buffer of the last 15 DP 3‑point frames `(headPos, headQuat, rWristPos, lWristPos)`
(resampled to the 30 Hz tick), the avatar's current pose `A = {root M_A, positions[34]
(root‑local), rotations[34] (root‑local 3×3), velocities[34] (root‑local), contacts[4]}`
initialised from `init_pose.json` with `M_A` = the player's current 3‑point root.

1. **Mapping input**: frames `t−14 … t` (oldest first). For frame `i`: root `M_i` from the head;
   `Positions_i` = the three input joints in `M_i`‑local coordinates (order head, right, left);
   `rm_i` relative to `rm_0` (the oldest frame = reference; `reference_char = 0`,
   `center_frame_idx = 0`). Fill channel‑major.
2. Run `mapping_leader` → for output frames `f = 0..29` (= `t+1 … t+30`): `rmOut_f` relative to
   `rm_0` → absolute root `M_f = apply(rm_0, rmOut_f)`; `Positions_f` are local to `M_f`.
3. **Tracking input**: leader window of 31 frames: frame 0 = **measured** current frame `t`
   (root `M_t`, local positions), frames 1..30 = mapping outputs `f = 0..29`. (The Unity demo
   duplicates output 0 instead; using the measured frame is the training‑time semantic.)
   Reference root for the relative transform is the **follower/avatar root** `M_A`
   (`reference_char = 1`): all 31 leader `rm` are made relative to `rm_A`; the follower's own
   `input_follower_RootMotion` = `rm_A` relative to itself = `[1, 0, 0, 0]`. Follower inputs are
   the avatar state (velocities, positions, rotations, contacts) verbatim.
4. Run `tracking_leader`. Take output frame `0` (= `t+1`): `rmΔ`, positions/rotations (local to
   the **new** root), velocities (in new‑root orientation), contacts.
   New root `M_A' = M_A · asMatrix(rmΔ)` (Unity `currentRoot * predictedRoot`).
   Optional **root correction** (`config.net.rootCorrection = 0.35`): blend `M_A'` position/yaw
   towards the measured head root `M_t` by that factor (prevents drift; local features unchanged).
   Optional **Euler blend** (`config.net.eulerRatio = 0.5`, Unity `DoEulerIntegrate`): world
   position of joint j = `lerp(M_A'·p_j, prevWorld_j + R(M_A')·v_j·dt, ratio)`, then re‑express
   locally. Then **bone‑length restore**: for every joint with a parent, clamp the world distance to
   `≤ 1.05 × boneLengths[j]` (move the child towards the parent).
   Contacts are clamped to `[0, 1]` and become the next `input_follower_FootContactLabels`.
5. Post `{ positionsWorldS: Float32Array(34*3), rootS, inferenceMs }` to the main thread (converted
   DP→S). The main thread interpolates between the last two poses for smooth 72 Hz rendering.

Performance guard (`avatar-driver.js`): average `inferenceMs` over 60 ticks; > 25 ms → tick at
15 Hz; > 60 ms → disable neural avatar, show `PointAvatar`, HUD notice "Neural‑Avatar deaktiviert
(zu langsam)". A `?avatar=` URL parameter overrides.

**AI partner mode** (optional): same code with the follower model set; the leader window comes
from the player's history and the mapping_follower prediction; the partner avatar root starts at
the player's root offset by 1 m in +Z(DP) facing the player (rotated 180°).

### 6.4 Reference implementation and parity tests

`tools/dp_pipeline.py` (numpy) implements §6.2–6.3 for offline use and is the oracle for the JS
port. `tools/gen_fixtures.py` produces `tests/fixtures/`:

* `math.json` — root/relative‑root/coordinate conversion cases (inputs + expected outputs).
* `mapping_io.json` — a real 15‑frame window from `Freestyle_Solo_01` (dataset frames converted to
  world 3‑point trajectories) → exact expected `input_leader_*` tensors (fp32) and the fp32 ONNX
  outputs (from `ckpt/.../model.onnx`, not int8).
* `tracking_io.json` — same for one tracking tick (with the dataset's own future as the "mapping
  prediction") → expected input tensors + fp32 outputs + the post‑processed world positions.

JS unit tests must reproduce the input tensors to `1e‑5` and the post‑processing to `1e‑4`;
int8 vs fp32 outputs are compared with a loose tolerance in a separate test that only runs when
`onnxruntime-node` is installed (optional, skipped in CI).

`tools/precompute_teacher.py` runs the tracking net (fp32 or int8) offline over a choreography's
3‑point frames (future window = the actual future frames, i.e. no mapping net) and stores
`fullBody` in the JSON. It uses `dp_pipeline.py`, so it is also an end‑to‑end check of the port.

Telemetry (`?telemetry=1`): the worker keeps the last N=2000 ticks of `{inputs, outputs}` in memory;
"Telemetrie exportieren" in the settings menu downloads them as JSON (for offline analysis /
interpretability work on the network).

---

## 7. Scoring (`src/game/scoring.js`) — pure, deterministic

Inputs per scored tick (30 Hz): player sample `P` and reference `R`, both in stage frame S at
reference height: `head [x,y,z]`, `left [x,y,z]`, `right [x,y,z]`, and velocities (finite
differences of the stage‑frame positions, smoothed with a 2‑tap average). If `mirror` is set, the
reference is mirrored before scoring: `x → −x` and left/right swapped.

Feature distances for player tick at time `t` vs reference at time `t + δ`, δ ∈ {−0.2 … +0.2 s}
in 1/30 s steps (best δ wins = reaction/lag tolerance):

```
dHead   = |P.head − R.head|                                  σ = 0.10 m
dLeft   = |(P.left − P.head) − (R.left − R.head)|            σ = 0.18 m   (hand relative to head: robust to drift)
dRight  = same for right                                      σ = 0.18 m
dVel    = |P.v − R.v| summed over the three points / 3         σ = 0.80 m/s
frameScore = 0.25·g(dHead) + 0.25·g(dLeft) + 0.25·g(dRight) + 0.25·g(dVel),  g(d) = exp(−(d/σ)²)
```

Per move: `moveScore = mean(frameScore over the move's ticks)`. **Energy gate**: if the player's
RMS speed over the move is < 30 % of the reference's, `moveScore = min(moveScore, 0.4)` (standing
still never scores well). Grades: `Perfekt ≥ 0.85`, `Gut ≥ 0.65`, `OK ≥ 0.40`, else `Daneben`.
Beat feedback: every beat, the mean frameScore of that beat gives a popup grade and drives the
combo (consecutive beats ≥ 0.65). Total: `score = round(100 · weightedMean(moveScore, by duration))`;
stars: ≥90 → 5, ≥75 → 4, ≥60 → 3, ≥40 → 2, else 1. `timingBias` = mean best‑δ (seconds; negative
= early). `Result` object:

```js
{ choreoId, score, stars, maxCombo, timingBias, durationSec,
  moves: [{ name, score, grade, startBeat, endBeat }], perBeat: Float32Array, playedAt: ISO, mode, player }
```

All of this must be implementable in ≤ 0.3 ms per tick (precompute reference features once per
choreography as typed arrays).

---

## 8. Recorder (`src/game/recorder.js`)

Setup panel: title, BPM (tap‑tempo button + text), bars, count‑in, synth pattern, mirror flag,
optional audio URL. Then calibrate → count‑in → record `durationBeats` worth of 30 Hz stage‑frame
frames (head pose with quaternion, both hands). Review: play back as teacher avatar, "Speichern"
(POST `/api/choreos` if a server is configured, otherwise download the JSON and keep a copy in
`localStorage['dp.choreos']`), "Verwerfen". Moves default to one move per 8 beats named
"Teil 1..n"; the operator can edit names later in the JSON. Recorded choreographies appear in the
menu list under "Eigene Tänze".

---

## 9. Duo mode and benchmark (`src/duo/*`, `server/`)

* **Ghost duo** (single headset): the second dancer is a saved run (`localStorage['dp.runs']` or
  `/api/runs`) of the same choreography, shown as a translucent `PointAvatar`/`BodyAvatar` next to
  the teacher. Both scores are shown side by side.
* **Online duo** (two headsets in the same WLAN + `server/`): room code (4 letters) shown in the
  lobby; both players calibrate; the host presses start; the server broadcasts
  `{type:'start', choreoId, startAt}` using its own clock; clients sync clocks with
  `{type:'ping', t0}` / `{type:'pong', t0, t1}` (NTP‑style offset, median of 5). During play each
  client sends `{type:'state', t, head:[7], left:[3], right:[3], score, combo}` at 15 Hz; the
  server relays to the room. The other player is rendered from interpolated states, 1.5 m to the
  side of the teacher.
* **Benchmark metrics** (`benchmark.js`, pure functions, unit‑tested): per player `score`, `stars`,
  `perMove`; pair metrics `syncDistance` (mean 3‑point distance between the two players over the
  run, after mirroring/offset removal), `syncLag` (cross‑correlation peak of head speed signals,
  seconds), `winner`. Export: JSON and CSV download; results are stored in
  `localStorage['dp.results']` and POSTed to `/api/results`. The menu has a "Bestenliste" panel
  (top 10 per choreography, local + server).

WebSocket messages are JSON; every message has `type`; server adds `from` (client id) and `room`.
Rooms die when empty. No auth (LAN only). Message size ≤ 4 KB, rate‑limited to 30 msg/s per client.

`server/server.js`: `node server/server.js [--port 8443] [--http] [--dir webxr] [--data ./data]`.
Default: HTTPS with a self‑signed certificate generated into `./data/cert/` on first start
(`selfsigned`), plus a plain HTTP listener on `--http-port 8080` for localhost/dev. REST: `GET
/api/choreos` (index incl. uploaded), `GET /api/choreos/:id`, `POST /api/choreos` (JSON body,
validated, written to `data/choreos/<id>.json`), `GET/POST /api/results`, `GET/POST /api/runs`
(saved 3‑point runs for ghost duo, capped at 50 per choreo), `GET /api/info` (version, LAN URLs,
QR‑code‑able https URL). Static files from `--dir` with correct MIME (`.wasm`, `.mjs`, `.onnx`
as `application/octet-stream`) and `Cache-Control: no-cache` for HTML/JSON.

---

## 10. Testing strategy

* `npm test` at the repository root runs `node --test tests/unit/`.
* `npm run test:e2e` starts `server/server.js --http --http-port 8090` and runs Playwright
  (`playwright-core` with the pre‑installed Chromium at `/opt/pw-browsers/chromium*` or the
  default) against `http://localhost:8090/?emu=playback&choreo=snoop-cwalk&autostart=1&speed=10`.
  Assertions: `window.__dp.app.state` reaches `RESULTS` within the time budget, `lastResult.score ≥
  90` for exact playback, `≤ 70` for `noise=0.35`, no `console.error`, no uncaught exceptions,
  and with `avatar=neural` the worker reports at least one pose (or a documented fallback reason
  in headless where WASM SIMD may be unavailable).
* Every module that does maths is written so it can be imported in Node without `three` (plain
  arrays / typed arrays). Rendering modules import `three` and are only exercised by e2e.

---

## 11. Conventions

* Files: 2‑space indent, semicolons, single quotes, `camelCase`, classes `PascalCase`, one class per
  file where sensible. No global mutable state except `window.__dp`.
* Every module starts with a short header comment stating its responsibility and what it must not
  do (e.g. "no three.js imports").
* Hot paths: no allocations per frame; reuse `Float32Array`s; never `JSON.parse` in the loop.
* Errors: user‑visible problems become a HUD/menu notice in German; details go to `console.warn`.
* Version string `APP_VERSION` in `src/config.js`, shown in the menu footer and in `/api/info`.

---

## Appendix: net-python (tools/dp_pipeline.py, prepare_models.py, gen_fixtures.py, precompute_teacher.py) API

Reference implementation of §6 in numpy (`tools/dp_pipeline.py`, importable, numpy-only; onnxruntime
is imported lazily inside `OnnxModels`). All internal maths is float64, network tensors float32.
Validated against the training-time slicing (`MappingProcessor`/`TrackingProcessor.reshape_data`)
on `Freestyle_Solo_01`: runtime-assembled inputs match the official ones to < 2e-6; fp32
`mapping_leader` predicts the 30 future 3-point frames with 7.5 cm mean local error (a wrong
layout gives metres); fp32 `tracking_leader` reproduces the next ground-truth frame with 3.3 cm
mean joint error; closed loop (10 s) drifts 1.3 cm with `rootCorrection = 0.35` vs 8.6 cm without.
int8 numbers are within 0.1 cm of fp32.

### Constants / layouts

* `FPS = 30`, `DT = 1/30`, `DATASET_FPS = 120`, `SAMPLE_STEP = 4`, `N_JOINTS = 34`, `JOINT_NAMES`,
  `PARENTS` (parents always precede children), `HEAD = 27`, `R_WRIST = 33`, `L_WRIST = 25`,
  `INPUT_JOINTS = (27, 33, 25)` (head, right, left), `CONTACT_JOINTS = (5, 8, 13, 16)`
  (`l_talocrural, l_ball, r_talocrural, r_ball`), `N_HISTORY = 15`, `N_FUTURE = 30`,
  `N_LEADER_TRACKING = 31`, `BONE_LENGTH_TOLERANCE = 1.05`, `YAW_EPS = 1e-3`.
* **3-point frame** (`FRAME_DIM = 13`, DP space, metres): `[head x y z, head quat x y z w,
  right wrist x y z, left wrist x y z]`. The head forward direction is `quat * (0, 0, 1)`.
  Slices `FRAME_HEAD/FRAME_QUAT/FRAME_RW/FRAME_LW`.
* `R_FIX` (3×3): "VR headset axes" → head-bone axes of the dataset character. Measured on the clip:
  bone +Y = forward, bone −X = up, bone −Z = right. A dataset frame's head quaternion is
  `quat(R_y(yaw) · R_local_head · R_FIX)`; its forward is the bone's +Y, its up is world-up-ish
  (the dancer looks ~20° down on average), so the quaternion is also fine for rendering a head.
* Rotation matrices in tensors: 9 values row-major `m00 m01 m02 m10 … m22`, `R` maps joint-local
  vectors to root-local vectors (`R_rel = R_root^T · R_joint_world`).
* `VelocitiesV2` (dataset semantics, verified to 6e-6): `(p_local(t) − M(t)^-1 · p_world(t−1)) / dt`
  — both positions expressed in the **current** root, per second (dataset dt = 1/120).
* Dataset clip columns (`COL_VEL/COL_POS/COL_ROT/COL_RM/COL_CONTACT`): `[0:102] VelocitiesV2,
  [102:204] Positions, [204:510] Rotations, [510:514] RootMotion, [514:518] FootContactLabels`.

### Functions (signatures)

```
quat_to_mat(q) -> (...,3,3); mat_to_quat(R) -> (4,) w ≥ 0; quat_mul(a, b); quat_rotate(q, v);
quat_normalize(q); quat_from_axis_angle(axis, angle); rot_y(yaw) -> 3x3 ([c 0 s; 0 1 0; -s 0 c]);
wrap_angle(a)
dp_to_s_pos(p) = p*(1,1,-1) = s_to_dp_pos;  dp_to_s_quat(q) = q*(-1,-1,1,1) = s_to_dp_quat;
frame_dp_to_s(frames) = frame_s_to_dp (involution on (...,13))
yaw_from_forward(fwd, prev_yaw=0, eps=1e-3) -> atan2(fwd.x, fwd.z) or prev_yaw
root_from_head(head_pos, head_fwd, prev_yaw=0) -> (pos (x,0,z), yaw)
root_matrix(pos, yaw) -> 4x4 T(pos)·R_y(yaw)
rm_from_root(pos, yaw) -> [cos yaw, −sin yaw, x, z]  (= [cos φ, sin φ, x, z], φ = −yaw); vectorised
rm_from_matrix(M); rm_to_matrix(rm); rm_to_matrices(rm (...,4)) -> (...,4,4); rm_yaw(rm); rm_pos(rm)
rm_relative(rm, ref) -> rel   (rot/rot_ref, (pos−pos_ref)/rot_ref; == inv(M_ref)·M_rm)
rm_apply(ref, rel) -> rm      (inverse; == M_ref·M_rel)
to_local(M, p) = M⁻¹p; to_world(M, p) = Mp; to_local_batch(Ms, Ps); to_world_batch(Ms, Ps)
channel_major(x (T,C)) -> float32 flat[c*T+t]; split_channels(flat, C) -> (T,C)
head_forward(quat) -> quat*(0,0,1); make_frame(...); frame_roots(frames, prev_yaw=0) -> (pos (T,3), yaw (T,))
three_point_local(frames, root_pos, yaw) -> (T,3,3) [head, right, left]
mapping_inputs(frames (15,13), prev_yaw=0) -> (feeds, info{rm, rm_ref, rm_rel, yaw, root_pos, local})
decode_mapping(outputs, rm_ref) -> {rm (30,4) absolute, rm_rel, local (30,3,3), world (30,3,3)}
AvatarState(rm, positions (34,3), rotations (34,3,3), velocities (34,3), contacts (4), world=None)
avatar_from_pose(pose_dict, rm)   (pose_dict = init_pose.json or clip_pose(clip, i))
tracking_inputs(leader_rm (31,4), leader_local (31,3,3), avatar) -> feeds (7 tensors, follower rm = [1,0,0,0])
decode_tracking(outputs, frame=0) -> {velocities, positions, rotations (34,3,3), contacts, rm}; decode_tracking_all
correct_root(rm, measured_rm, factor) -> rm    (x,z lerp; yaw += factor·wrap(Δyaw))
restore_bone_lengths(world, parents, bone_lengths, tol=1.05) -> world
post_process(avatar, pred, measured_rm, skeleton, root_correction=0.35, euler_ratio=0.5, dt=DT) -> AvatarState
OnnxModels(models_dir=None, mapping_path=None, tracking_path=None, char_set='leader', threads=1)
    .skeleton .init_pose .meta .run_mapping(feeds) .run_tracking(feeds)  (dict name -> float32 flat)
run_self_tracking(frames (N,13), models, root_correction=0.35, euler_ratio=0.5, history=15,
                  use_measured_frame=True, init_rm=None) -> {world (N,34,3), rm (N,4), head_rm (N,4), mapping_world (N,30,3,3), contacts}
run_future_tracking(frames, models, root_correction=0.35, euler_ratio=0.5, init_rm=None) -> same without mapping_world
load_clip(npy) -> {vel, pos, rot, rm, contacts, n, raw}; clip_world_positions(clip, idx); clip_head_world_rot(clip, idx);
clip_three_point_frames(clip, idx) -> (n,13); clip_pose(clip, i); sampled_indices(start, count, step=4);
official_mapping_inputs(clip, pivot) / official_tracking_inputs(clip, pivot)  (training-time slicing, oracle for (b))
validate(clip_path, models, pivots_sec, loop_start_sec, loop_seconds) -> dict of numbers
```

CLI: `python tools/dp_pipeline.py --validate <clip.npy> [--ckpt <dir with mapping_leader/model.onnx>]
[--models webxr/models/free] [--pivots-sec 5,20,…] [--loop-start-sec 24] [--loop-seconds 10] [--json out]`.

### Post-processing details that the JS port must match (fixture `tracking_io.json`, tolerance 1e-4)

1. `new_rm = rm_apply(avatar.rm, pred.rm)`; then `correct_root(new_rm, measured_rm, rootCorrection)`
   (`measured_rm` = root of the **measured** current frame t, i.e. `leaderRoots.rm[0]`).
2. `world_j = M_new · p_j`; Euler blend (if `eulerRatio > 0`):
   `world_j = lerp(world_j, avatar.world_j + R_new · v_j · dt, eulerRatio)` with `avatar.world` = the
   final world positions of the previous tick (init: `M_A · init_pose.positions`).
3. Bone-length restore in joint order (parents first): if `|world_j − world_parent| > 1.05 · boneLengths[j]`
   the joint is moved onto the parent→joint direction at that distance **and the same delta is
   applied to all of its descendants** (Unity `Transform` semantics of `Bone.SetLength`). Zero-length
   bones (`b_l_foot`, `b_r_foot`, ~`b_*_talocrural`) therefore snap onto their parent. In the fixture
   tick 3 bones exceed the tolerance, so this step is exercised.
4. `positions = M_new⁻¹ · world`; rotations/velocities = network output verbatim; contacts clipped to
   [0, 1]; `avatar.world = world`. Unity's `RestoreAlignment` (rotation touch-up) and foot/hand IK
   are intentionally not ported.
5. The tracking leader window is `[measured frame t] + mapping outputs 0..29`
   (`use_measured_frame=True`); `False` reproduces the Unity demo (prediction 0 duplicated).
6. Mapping history: ring buffer pre-filled with the first frame (`idx = clip(t−14…t, 0)`), yaw
   fallback = yaw of the frame preceding the window (0 for the very first).

### Fixtures (`tests/fixtures/*.json`, `format: dancing-points-fixture/1`, numbers float32 → 7 significant digits, expectations computed from the rounded inputs)

* `math.json`: `cases[]` with `id`, `kind` ∈ {`rootFromHead` (11, incl. tilted/rolled heads, straight-down
  → `prevYaw`, 3 dataset frames with the dataset `datasetRm` for comparison), `rmMatrix` (6),
  `rmRelative` (6, incl. `applied` = `rmApply(ref, rel)` and `selfRelative`), `localWorld` (4),
  `dpToS` (5, with `rotS = M·R·M`, `forwardS`), `sToDp` (3), `headForward` (3), `frameRoots` (1, sequence
  with a degenerate frame)}, `input`, `expected`, `note`. Matrices are row-major (16 or 9 values).
* `mapping_io.json`: `params` (pivot = dataset frame 7200 = 60 s, `datasetFrames`), `framesDP` /
  `framesS` (15×13, unscaled world), `roots` {rootPos, yaw, rm, rmRef, rmRel, local}, `inputs`
  {`input_leader_Positions`[135], `input_leader_RootMotion`[60]}, `outputsFp32` {`Positions`[270],
  `RootMotion`[120]}, `decoded` {rm, rmRel, localPositions, worldPositionsDP, worldPositionsS},
  `groundTruth` {futureWorldDP, meanErrorCm ≈ 8.3}.
* `tracking_io.json`: `leaderFramesDP/S` (31×13: frame 0 = measured t, 1..30 = ground-truth future),
  `leaderRoots` {rootPos, yaw, rm, local, rmRelativeToAvatar}, `avatar` {rm, rootMatrix, positions,
  rotations (34×9), velocities, contacts, worldDP} (= ground-truth pose at t), `measuredRm`, `inputs`
  (7 tensors), `outputsFp32` (5 tensors, 15 540 floats), `decodedFrame0` {…, newRmUncorrected,
  rawWorldDP}, `postProcessed.rc0_er0` / `.rc035_er05` {rm, rootMatrix, worldDP, worldS, rootS,
  positionsLocal, contacts}, `skeleton` {joints, parents, boneLengths}, `groundTruth`
  {nextWorldDP, meanErrorCm_rc0_er0 ≈ 2.2}.

### tools/prepare_models.py

`--checkpoints <dir | checkpoints.tar> --style free --set leader|follower|both --out webxr/models
[--clip <npy>] [--clip-label Freestyle_Solo_01] [--no-quantize] [--no-smoke]`. Accepts
`<dir>/dancing/<style>/<model>/model.onnx`, `<dir>/<style>/<model>/…` or `<dir>/<model>/…`; from a
tar only `model.onnx`/`args.txt` of the requested models are extracted. Output is deterministic:
re-running for style `free` reproduces the committed `meta.json`, `skeleton.json`, `init_pose.json`
and both int8 files byte-for-byte (`quantize_dynamic`, QInt8 weights, `Gemm`/`MatMul`). `meta.json`
keeps models of other sets already present. `skeleton.boneLengths` = mean child–parent distance
over the clip (float32); `init_pose.json` = calmest both-feet-down frame of the first 30 s
(`positions` (34×3), `rotations` (34×9), `velocities` zeros, `contacts`, `headHeight`).

### tools/precompute_teacher.py

* `--choreo <json> --models webxr/models/free [--fp32 <ckpt dir>] --out <json> [--root-correction 0.35]
  [--euler-ratio 0.5] [--net-scale k] [--replace]`: choreography frames (S, at `referenceHeight`) →
  DP, divided by `net-scale` (default `meta.scale` if present, else 1) → `run_future_tracking`
  (future window = actual frames, start pose `init_pose.json` at the root of frame 0) → `fullBody`
  {joints, parents, fps 30, encoding `base64-float32`, positions (frames×34×3, S, little-endian,
  frame-major)}. Writes `meta.fullBodySource` (`'tracking_leader int8|fp32'`) and
  `meta.fullBodyParams`. If the file already carries `meta.fullBodySource == 'ground-truth'` it only
  reports the network error and keeps the ground truth unless `--replace`.
* `--from-dataset <clip.npy> --start-sec S --duration-sec 30 [--bpm 100] [--reference-height 1.70]
  [--gt-fullbody] [--compare-net] [--id/--title/--artist/--difficulty]`: cuts an excerpt into a
  choreography: frames relative to the first frame's root (origin, facing −Z in S), scaled by
  `k = referenceHeight / standingHeadHeight` (95th percentile of the head height over frames with all
  four foot contacts), moves "Teil 1..n" of 8 beats, `meta.scale = k`, `meta.standingHeadHeight`,
  `meta.datasetStartFrame/EndFrame/Step`, `meta.clip`. `--gt-fullbody` stores the ground-truth joints.

### webxr/choreos/mocap-freestyle.json

Excerpt 13 s–43 s of `Freestyle_Solo_01` (dataset frames 1560..5160, step 4; the liveliest 30 s
window by mean joint speed), 901 frames, `durationBeats 50 @ 100 bpm`, `referenceHeight 1.70`
(`meta.scale = 1.1553`, standing head height 1.4715 m), `fullBody` = ground truth
(`meta.fullBodySource: 'ground-truth'`), 568 KB. **`webxr/choreos/index.json` must list it**
(`{"id":"mocap-freestyle","title":"Freestyle (Mocap-Demo)","artist":"Dancing Points Datensatz","bpm":100,
"durationBeats":50,"difficulty":3,"file":"mocap-freestyle.json"}`) — owned by the choreo lane.
The int8 tracking net driven by these frames (via `--choreo --replace`) reproduces the ground-truth
body with 9.1 cm mean joint error (head 3.9 cm, wrists 14.3 cm).

## Appendix: game engine API

Platform‑independent modules under `webxr/src/` (lane *game-engine*): everything here runs in
Node without `three` and is covered by `tests/unit/*.test.js` (`npm test` =
`node --test tests/unit/*.test.js`; the shell expands the pattern, which works on Node 18–22 —
Node 22 no longer treats a bare directory argument as a search root). All positions are plain
arrays `[x, y, z]`, quaternions `[x, y, z, w]`, out‑parameter style (`fn(out, …)` writes into
`out` and returns it), no allocation in anything called per frame.

### Shared sample shape (`src/xr/input.js`)

```js
sample = { t, k, head: { p:[x,y,z], q:[x,y,z,w], valid }, left: {p,q,valid}, right: {p,q,valid} }
```
`t` = wall time in seconds (XR: rAF timestamp / 1000; playback backend: game time), `k` = the
calibration scale (1 for W‑frame samples). `createSample()` allocates one, `copySample(out, s)`
deep‑copies. **Input backend interface** (implemented by `XRInput` and `EmulatedInput`): `sample`
(latest, reused object), `update(...)`, `buttons.{left,right}` (`createButtonState()`:
`trigger 0..1, squeeze 0..1, stick, primary (A/X), secondary (B/Y), menu, thumb:[x,y], connected,
isHand`), `justPressed(hand, name)` / `isDown(hand, name)` with `hand ∈ 'left'|'right'|'any'`,
`name ∈ BUTTONS = ['trigger','squeeze','stick','primary','secondary','menu']`, `anyTriggerDown()`,
`tracking` (getter), `emulated` (boolean), `dispose()`. Poses keep their last value when tracking
is lost (`valid` becomes false).

### `src/config.js`
`APP_VERSION = '0.1.0'`, `APP_NAME`, `SCORING` (`tickHz 30, sigmaHead 0.10, sigmaHand 0.18,
sigmaVel 0.80, deltaMaxSec 0.20, weight* 0.25, energyGateRatio 0.30, energyGateCap 0.40,
gradePerfekt 0.85, gradeGut 0.65, gradeOk 0.40, comboThreshold 0.65, stars [90,75,60,40]`),
`NET` (`tickHz 30, historyFrames 15, futureFrames 30, rootCorrection 0.35, eulerRatio 0.5,
boneLengthTolerance 1.05, guardWindowTicks 60, slowMs 25, disableMs 60, telemetryTicks 2000`),
`STAGE` (`referenceHeight 1.70, teacherDistance 2.5, duoSideOffset 1.5, floorSize 6`), `FLOW`
(`defaultCountInBeats 4, calibrationHoldSec 1, calibrationEmuSec 3, audioLookaheadSec 0.1,
runRecordHz 30, duoStateHz 15`), `STORAGE_KEYS` (`calibration 'dp.calibration', results
'dp.results', runs 'dp.runs', choreos 'dp.choreos', settings 'dp.settings', telemetry
'dp.telemetry'`), `DEFAULT_PARAMS`, `parseParams(search) → {emu: null|'desktop'|'playback', noise,
lag, seed, choreo, autostart, speed, server, avatar, style, telemetry, lang}` (invalid values fall
back to the defaults; `emu=1|true|desktop` → `'desktop'`), `params` (parsed from
`location.search`, defaults in Node).

### `src/util/math.js`
Scalars: `clamp, lerpScalar, degToRad, radToDeg, wrapAngle (→ (−π, π]), angleDiff(a, b)`.
vec3: `vec3Create, set, copy, add, sub, scale, addScaled(out,a,b,s), dot, cross, length, lengthSq,
dist, distSq, distAt(a,ia,b,ib), normalize, lerp, negate, equalsApprox, rotateY(out, v, angle)`.
quat: `quatCreate, quatIdentity, quatCopy, quatSet, quatLength, quatNormalize, quatDot,
quatConjugate, quatInverse, quatMul(out,a,b) (Hamilton, apply b then a), quatFromAxisAngle(out,
axis, angle), quatFromYaw(out, yaw), quatMulVec(out, q, v) (out may alias v), forwardFromQuat(out,
q) = q·(0,0,−1), upFromQuat, yawFromQuat(q, fallback=0), yawFromForward(f, fallback), slerp(out,
a, b, t) (shortest path), quatAngle, quatEqualsApprox`. Yaw convention: `quatFromYaw(yaw)·(0,0,−1)
= (−sin yaw, 0, −cos yaw)`; `yawFromQuat` returns `fallback` when the ground projection of the
forward vector is < 1e‑3. Flat arrays: `put3, get3, lerp3At(out, a, ia, b, ib, t)`,
`matrixFromYawPosScale(out16, yaw, pos, s)` (column‑major `T·R_y·S`, `Matrix4.fromArray`).

### `src/util/events.js`
`class EventEmitter { on(type, fn) → unsubscribe; once(type, fn); off(type, fn?); emit(type, a, b)
→ listeners called; listenerCount(type); removeAllListeners() }`. `emit` has fixed arity (two
payload arguments) and never allocates; listeners may unsubscribe during emit; a throwing listener
is reported with `console.warn` and does not stop the others.

### `src/xr/input.js`
`class XRInput { constructor({session?}); attachSession(session); detachSession();
update(xrFrame, refSpace, session?, timeMs?) → sample; justPressed(hand, name='trigger');
isDown(hand, name); anyTriggerDown(); tracking; emulated=false; frameCount; dispose() }`.
Head from `xrFrame.getViewerPose(refSpace)`; hands from `inputSource.gripSpace`, else the XRHand
`'wrist'` joint (`getJointPose`; pinch index‑tip/thumb‑tip < 2 cm acts as trigger), else
`targetRaySpace`. Buttons from the xr‑standard gamepad (`0 trigger, 1 squeeze, 3 stick, 4 A/X, 5
B/Y, 6/7 menu`, axes 2/3 thumbstick) OR‑ed with the session's `select*`/`squeeze*` events.
`emulatedPosition` poses count as `valid=false`.

### `src/xr/calibration.js`
`class Calibration { constructor({storage?, key='dp.calibration', referenceHeight=1.70});
data: null | {h0, yaw, origin:[x,0,z], capturedAt, version:1}; valid; h0; yaw; origin;
scale(referenceHeight?) = referenceHeight / h0; set(data) → bool; capture(sampleW, {fallbackYaw?})
→ data|null; beginCapture(); addSample(sampleW) → n; endCapture() → data|null (circular mean of the
yaw); toStage(sampleW, referenceHeight?, out?) → sampleS (sets out.k); fromStage(sampleS,
referenceHeight?, out?) → sampleW; worldToStage(out, pW, referenceHeight?); stageToWorld(out, pS,
referenceHeight?); worldToStageQuat(out, qW); stageToWorldQuat(out, qS); stageToWorldMatrix(out16,
referenceHeight?); save() → bool; load() → bool; clear(); toJSON() }`.
`h0` = **standing head height** in W (0.8–2.4 m accepted); `referenceHeight` is therefore the
standing head height of the choreography in S (recorded choreographies get exactly that by
construction, `precompute_teacher.py --from-dataset` scales the dataset to it). Without data the
transform is the identity (h0 = referenceHeight, yaw 0, origin 0). `MIN_HEAD_HEIGHT`,
`MAX_HEAD_HEIGHT`, `CALIBRATION_VERSION` are exported.

### `src/emu/emulated-input.js`
`class EmulatedInput { constructor({backend='desktop'|'playback', choreo, clock?, noise=0,
noiseTau=0.25, lag=0, seed=1, playerHeight?, playerYaw=0, playerOrigin=[0,0,0], calibration?,
window?, document?}); update(nowMs?) → sample; setTime(t); gameTime(); setButton(hand, name,
value); attach(win, doc); detach(); standing; emulated=true; …input backend interface }`,
`makeRandom(seed) → {next(), gaussian()}` (mulberry32 + Box–Muller, deterministic).
*playback*: while no clock is running (or `setTime` was never called) the sample is the **standing
calibration pose** (head at S `(0, referenceHeight, 0)` facing −Z, hands at the sides) so that the
app's calibration step yields `h0 = playerHeight`, `yaw = playerYaw`, `origin = playerOrigin`;
during play the sample is `choreo.sampleAt(t − lag)` (unmirrored reference), plus Ornstein–Uhlenbeck
noise per coordinate (stationary σ = `noise` metres per axis, correlation time `noiseTau`),
converted S → W through the virtual player (a private `Calibration`). `clock.now().t` is the
time source when a `BeatClock` is given. *desktop*: `W/A/S/D` (or arrows) move, `R/F` height,
mouse‑look while the left button is held or the pointer is locked, `Q/E` raise the left/right hand,
`Space`/left mouse button = trigger, `Shift` = squeeze, `X/Y` = left primary/secondary,
`Enter/B` = right primary/secondary, `Escape` = menu. `update()` is idempotent for a repeated
wall time, so the app may call it in its render loop even though `PlaySession` also drives it.

### `src/game/clock.js`
`class BeatClock { constructor({audioContext?, timeSource?, speed=1}); start(bpm, countInBeats=4,
{audioContext?, timeSource?, speed?, beatsPerBar?, startAt?}) → this; now() → {t, beat, bar,
sourceTime, countIn, running, paused} (reused object); pause(); resume(); stop(); seek(t);
sourceNow(); timeToSource(t); sourceToTime(s); beatToSource(beat); beatToTime(beat);
timeToBeat(t); countInSeconds; bpm; beatsPerBar; countInBeats; beatDuration; speed; running;
paused }`. `t` is in seconds since beat 0 (negative during the count‑in), `bar = floor(beat /
beatsPerBar)`. Time source priority: injected `timeSource()` → `audioContext.currentTime` →
`performance.now()/1000`. `speed` multiplies source time (tests: `?speed=10`).

### `src/game/audio.js`
`class AudioEngine { constructor({audioContext?, fetchImpl?, lookahead=0.1}); available; resume()
→ Promise<bool>; setMasterGain(g); getMasterGain(); setPattern('hiphop'|'house'|'metronome'|null);
loadTrack(url, {offsetSec=0, gain=0.8}) → Promise<bool> (never throws); start(clock, {synth?,
countInBeats?, beatsPerBar?, durationBeats?, useTrack=true}); update() (call every frame: schedules
16th‑note steps inside the lookahead window on `clock.beatToSource(beat)`); playClick(accent);
playSfx('perfekt'|'gut'|'ok'|'daneben'); stop(); dispose(); playing; scheduledEvents }`,
`PATTERNS`, `patternStep(name, step16) → bitmask`. Count‑in: a click on every count‑in beat
(accent on the first). With a loaded track the synth pattern is muted (`'metronome'` stays audible).
Every method is a no‑op without an `AudioContext`.

### `src/game/choreo.js`
`validateChoreo(obj) → {ok, errors[]}` (lengths `N` or `N+1` with `N = round(durationBeats·60/bpm·fps)`,
finite numbers, contiguous moves covering `[0, durationBeats]`, optional fields tolerated),
`expectedFrameCount(c)`, `loadChoreo(urlOrObject, fetchImpl?) → Promise<Choreo>` (relative
`audio.url` resolved against the file), `createChoreoSample() → {t, head:{p,q}, left:{p},
right:{p}}`, `mirrorChoreoSample(out, s)` (x → −x, hands swapped, yaw mirrored),
`encodeBase64Float32(f32)`, `decodeBase64Float32(str)`, `CHOREO_FORMAT`, `ID_PATTERN`,
`SYNTH_PATTERNS`.
`class Choreo { constructor(json) (throws with .errors); id, title, artist, bpm, beatsPerBar,
countInBeats, durationBeats, fps, referenceHeight, mirror, difficulty, audio {url, synth,
offsetSec, gain}, moves [{index, name, startBeat, endBeat, hint}], meta, json; beatDuration,
duration (s), frameDt, frameCount; headP/headQ/leftP/rightP (Float32Array N·3 / N·4);
beatToTime(b); timeToBeat(t); frameIndexAt(t); sampleAt(t, out?) (lerp + slerp, clamped, reused
internal object by default); moveAt(t) → move|null; moveIndexAt(t); moveIndexAtBeat(b);
hasFullBody; fullBody: null | {joints, parents, fps, jointCount, frameCount, positions
Float32Array}; fullBodyAt(t, out?) → Float32Array(jointCount·3)|null; toIndexEntry(file?);
toJSON() }`.

### `src/game/scoring.js`
`class Scorer { constructor(choreo, {mirror?=choreo.mirror, onBeat?, sigmaHead?, sigmaHand?,
sigmaVel?, deltaMaxSec?}); reset(); scoreTick(sampleS, t) → frameScore 0..1 | −1 (t outside
[0, duration] or after finalize); onBeat: (ev) => void with ev = {beat, score, grade, combo}
(called when a beat closes; the last beat closes in finalize() or on the tick at t = duration);
moveScore(i); moveGated(i); currentScore() → 0..100 running total; timingBias() → seconds;
finalize({mode='solo', player='', playedAt?}) → Result (idempotent); ref (precomputed
features); perBeat Float32Array; combo; maxCombo; ticks; lastFrameScore; lastDelta;
lastBeatEvent; finalized; result }`, `precomputeReference(choreo, mirror) → {n, fps, head,
leftRel, rightRel, vel (N·9), speed, mirror}`, `gradeFor(score01) → 'perfekt'|'gut'|'ok'|'daneben'`,
`starsFor(score100) → 1..5`, `GRADE_IDS`, `resultToJSON(result)` (perBeat as a rounded array).
Implements §7 verbatim; the player's velocity is the finite difference between successive ticks
(actual `dt`), 2‑tap smoothed, and for the first two ticks the velocity term equals the mean of the
three position terms. **`timingBias` sign**: `> 0` means the player is *late* (behind the
reference), `< 0` early — i.e. `timingBias = −mean(best δ)` with δ as defined in §7
(`player(t) ≈ reference(t + δ)`); a playback with `lag = 0.1` reports `+0.1`. `Result` =
`{choreoId, score, stars, maxCombo, timingBias, durationSec, moves:[{name, score, grade,
startBeat, endBeat, ticks, gated}], perBeat, playedAt, mode, player, ticks, meanFrameScore,
mirror}`. Measured: 0.0012 ms per tick, 0 B allocated per tick.

### `src/game/session.js`
`class PlaySession extends EventEmitter { constructor({choreo, clock, scorer, input, calibration,
audio?, mode='solo', player='', scoreHz=30, recordRun=true, driveInput?=input.emulated});
start({speed?}) → this; update(nowMs?) → state; abort(); pause(); resume(); paused;
countdownBeatsLeft; progress; runToJSON() → run|null; dispose(); state 'idle'|'countdown'|
'playing'|'finished'|'aborted'; t; beat; bar; frameScore; combo; score (running 0..100);
moveIndex; currentMove; lastBeatEvent; result; playerS (latest stage‑frame sample); refSample
(choreo.sampleAt(t), unmirrored — the renderer mirrors for `mirror: true`); run {choreoId, fps 30,
frameCount, head Float32Array(N·7), left, right Float32Array(N·3), filled Uint8Array, lastIndex};
ticks; frames; startedAt }`.
Events: `'start' {choreoId, mode}`, `'countdown' {beatsLeft, beat}` (each count‑in beat),
`'beat' {beat, bar, beatInBar}` (each playable beat, from the clock), `'move' {index, move, prev}`,
`'grade' {beat, score, grade, combo}` (scorer's beat verdict; drives the HUD popup, combo and
`audio.playSfx`), `'tick' {t, frameScore, delta}` (≤ 30 Hz, **reused object**), `'finished'
(Result)`, `'abort' {t}`. `update()` reads `calibration.toStage(input.sample)`; when `driveInput`
it first calls `input.update(nowMs)` (default for emulated backends; `XRInput.update` must be
called by the render loop with the XRFrame). The run (`dancing-points-run/1`:
`{format, choreoId, fps, frameCount, head number[][7], left, right number[][3], score, playedAt,
player, mode}`) is the stage‑frame 3‑point stream for the ghost duo, gaps hold‑filled.
Render loop recipe: `xrInput.update(frame, refSpace, session, time); session.update(time);
teacher.pose(session.refSample); hud.show(session.frameScore, session.combo, session.score)`.

### `src/game/recorder.js`
`class Recorder { constructor({fps=30}); start(config) → this; update(tOrClockState, sampleS) →
highest frame index written | −1; stop() → choreo JSON (validateChoreo passes; throws when no
frame); cancel(); state 'idle'|'recording'|'stopped'; config; frameCount; progress 0..1; complete;
durationSec; lastIndex; sampleCount; filled Uint8Array (1 sampled/interpolated, 2 gap‑filled) }`.
`config = {title, bpm, bars=4 | durationBeats, beatsPerBar=4, countInBeats=4, synth='hiphop',
mirror=false, audioUrl=null, audioOffsetSec=0, audioGain=0.8, referenceHeight=1.70, id?, artist,
author, notes, beatsPerMove=8, difficulty=2}`. Samples arrive at the render rate; every 30 Hz frame
is written by linear interpolation (nlerp for the head quaternion) between the bracketing samples
at its exact frame time, so a 72 Hz stream reproduces a sinusoidal motion to < 1 mm; frames
bracketed by samples > 2.5 frames apart and trailing frames after an early `stop()` are
hold‑filled (`meta.holdFilledFrames`). Values are rounded to 0.1 mm / 1e‑6. Statics:
`Recorder.defaultMoves(durationBeats, beatsPerMove=8)` ("Teil 1..n"), `makeId(title, now?)`
(slug + base‑36 time stamp, umlauts transliterated), `toJSONString(json, pretty=false)`,
`filename(json)`, `persist(json, storage, key='dp.choreos') → bool` (storage holds a map id → choreo;
a corrupt store is replaced), `listPersisted(storage, key?) → json[]` (validated),
`removePersisted(id, storage, key?)`, `download(json, doc?) → filename|null` (browser only),
`upload(json, baseUrl='', fetchImpl?) → Promise<response json>` (`POST {baseUrl}/api/choreos`).
Recording loop recipe (no `PlaySession` needed): `clock.start(cfg.bpm, cfg.countInBeats);
audio.start(clock, {synth: cfg.synth, countInBeats, beatsPerBar, durationBeats}); recorder.start(cfg);`
per frame `recorder.update(clock.now(), calibration.toStage(input.sample, cfg.referenceHeight))`
until `recorder.complete`, then `const json = recorder.stop()`.

### `src/ui/texts.js`
`de` (German, primary), `en` (same keys), `LANGUAGES = ['de','en']`, `GRADE_KEYS {perfekt:
'gradePerfekt', gut: 'gradeGut', ok: 'gradeOk', daneben: 'gradeDaneben'}`, `getTexts(lang) →
frozen object (unknown language / missing key → German)`, `formatText(texts, key, vars?)` with
`{name}` placeholders. Key groups: `menu*`, `settings*`, `calibrate*`, `countdown*`, `hud*`,
`grade*`, `results*`, `record*`, `duo*`, `highscores*`, generic (`ok, cancel, error, loading,
offline, online, player, errorChoreoLoad, errorChoreoInvalid, errorAudioLoad, errorXr,
errorModelLoad`). The test asserts that every German key has an English counterpart.

### Test helper `tests/unit/helpers/make-choreo.js`
`makeChoreo({id, bpm=120, durationBeats=16, fps=30, mirror, fullBody, still, referenceHeight,
countInBeats}) → choreo JSON` (sinusoidal sway/bounce + arm swings, moves "Sway"/"Arms",
optional 3‑joint fullBody), `cloneJSON(obj)`. Measured with it: exact playback 100, `noise 0.35`
→ 2, `noise 0.03` → 80, standing still → 24 (gated), `lag ±0.1` → timingBias ±0.1 with score 99;
the shipped `mocap-freestyle.json` validates and scores 100 on exact playback.

## Appendix: choreos (tools/gen_choreos.js, webxr/choreos/*) API

`tools/gen_choreos.js` is a Node ES module (no three.js; imports only `webxr/src/util/math.js`
and `validateChoreo` from `webxr/src/game/choreo.js`). It is importable without side effects and
runnable as `node tools/gen_choreos.js [--out <dir>] [--only <id,id>] [--no-index] [--quiet]`
(`npm run gen:choreos`). Output is **deterministic** (no randomness, fixed `meta.createdAt =
GENERATED_AT`); `tests/unit/choreos-files.test.js` regenerates into a temp dir and requires
byte-identical files, so **after changing the generator, re-run it and commit the JSON**.

### Shipped files (`webxr/choreos/`)

| file | frames | bytes | notes |
|---|---|---|---|
| `snoop-cwalk.json` | 961 (48 beats @ 90 bpm, 32 s) | ~105 kB | mirror `true`, synth `hiphop`, 6 moves × 8 beats: Shoe Vibe, Restep, Heel Toe, Side Hopping, Shuffle Legs, Gangster Two Step |
| `tutorial-basics.json` | 721 (32 beats @ 80 bpm, 24 s) | ~77 kB | mirror `false`, synth `metronome`, 4 moves × 8 beats: Wippen, Seitwärts, Arme hoch, Freestyle |
| `index.json` | – | – | snoop-cwalk, tutorial-basics, mocap-freestyle (external entry, `EXTERNAL_INDEX_ENTRIES`) |
| `README.md` | – | – | German operator guide: recording / writing / placing own dances, audio licensing |

Conventions of the generated frames (stage frame S): frame 0 and the last frame are the neutral
standing pose (head `[0, 1.60, 0]`, identity quaternion, hands hanging at `[±0.24, 0.88, −0.10]`);
motion eases in/out over 0.5 beat at both ends and crossfades over ±0.5 beat around every move
boundary (smoothstep), so there are no velocity spikes at boundaries. Head y stays in 1.55..1.66,
hands are ≤ 0.81 m from the head and 0.30..0.52 m from the shoulder joints; all values are
rounded to 4 decimals, quaternions are unit length within 1e-4. Title uses an ASCII hyphen
("C-Walk") so canvas fonts render it.

### Body model (`BODY`, `ANCHORS`)

`BODY = { headRestY: 1.60, shoulderHalfWidth: 0.20, shoulderDrop: 0.22, armReach: 0.62,
maxHandHeadDist: 0.88, torsoLever: 0.55 }`. Hands are authored relative to the head as offsets
from `ANCHORS.rest` (`left [−0.24, −0.72, −0.10]`, `right [0.24, −0.72, −0.10]`); further anchors
`chest`, `hip`, `crossed`. After summing all primitives each hand is clamped to `armReach`
around its shoulder (`head + [±0.20, −0.22, 0]`) and to `maxHandHeadDist` around the head.

### Motion primitives (each returns `(beat, pose) => void` that ADDS into a pose accumulator)

Pose accumulator (`createPose()`): `{ head: [dx, dy, dz], yaw, pitch, roll, left: [dx, dy, dz],
right: [dx, dy, dz] }` — offsets from the rest pose in metres / radians. Yaw > 0 turns the face
to the player's left (−X), pitch > 0 looks up, roll > 0 tilts the head to the left.
`timesPerBeat` = cycles per beat (1 = every beat, 2 = eighths, 0.5 = every 2 beats), `phase` in
beats. Wave shapes: `'sin'` (`sineWave`) or `'step'` (`stepWave(u, blend)`: smooth square wave
that reaches ±1 exactly on the cycle boundary / half cycle, i.e. a weight shift that lands on the
beat); pulses use `bump(u, width)` (raised cosine, 1 on the cycle start).

```
bounce(amplitude, timesPerBeat=1, phase=0, { nod=0.8 })            knee-bend dip, lowest on the beat, head nods
sway(amplitude, timesPerBeat=0.5, phase=0, { shape='sin', blend=0.3, look=0.6 })   lateral (+ = right), yaw follows
rock(amplitude, timesPerBeat=0.5, phase=0, { shape='step', blend=0.35, nod=0.4 })  fore/aft (+ wave = forward −Z)
hop(lateral, height, timesPerBeat=1, phase=0, { flight=0.4, dip=0.03, dipLen=0.3, look=0.25 })
      alternating side hop between x = ±lateral; flight ends on the beat, arc of `height`, landing dip
lean(angle, timesPerBeat=0.25, phase=0, { shape, blend })           hip lean (+ wave = right): roll + lever displacement
headTurn(angle, timesPerBeat=0.25, phase=0, { shape, blend, pitch=0 })   + wave = look right (+X)
armSwing(angle, timesPerBeat=0.5, phase=0, { alternate=true, shape, blend })  pendulum about the shoulders (+ = left arm forward)
armFlare(angle, timesPerBeat=1, phase=0, { width=0.6 })            both arms abduct outward on the pulse
armRaise(amount=1, timesPerBeat=0.25, phase=0, { alternate=true, lookUp=0.12, maxAngle=2.8 })
      front-arc raise above the head; alternate: left in the first half cycle, right in the second
armPump(depth, timesPerBeat=1, phase=0, { width=0.5, forward=0 })  both hands push down on the pulse
armCross(amount=1, timesPerBeat=0.5, phase=0, { width=0.7 })       hands move to ANCHORS.crossed on the pulse
handsAt(anchor|{left,right}, weight=1)                             constant base offset (e.g. hands at the chest)
```

Helpers: `cycles(b, timesPerBeat, phase)`, `smoothstep`, `clamp01`, `fract`, `sineWave`,
`stepWave`, `bump`, `createPose/resetPose/mixPose/scalePose`, `headQuat(out, yaw, pitch, roll)`
(`q = qY(yaw)·qX(pitch)·qZ(roll)`, normalised).

### Definitions and building

```
CHOREO_DEFS: [{ id, title, artist, bpm, beatsPerBar, countInBeats, durationBeats, mirror, difficulty,
                audio, notes, moves: [{ name, beats, hint, parts: [primitive, ...] }] }]
EXTERNAL_INDEX_ENTRIES: index entries of shipped files not produced here (mocap-freestyle)
frameCountFor(def) -> durationBeats*60/bpm*30 + 1
samplePose(def, movesWithBeats, beat, pose, tmp, { blendBeats=0.5, edgeBeats=0.5 }) -> pose
buildChoreo(def) -> validated choreography JSON object (throws if validateChoreo fails)
indexEntry(json) -> { id, title, artist, bpm, durationBeats, difficulty, file }
serializeChoreo(json) -> string (2-space header, one frame row per line, trailing newline)
serializeIndex(entries) -> string
writeChoreos(outDir, { only=null, index=true }) -> [{ id, file, frames, bytes }]   (index.json only when !only)
DEFAULT_OUT_DIR = <repo>/webxr/choreos;  GENERATOR_VERSION, FPS = 30, REFERENCE_HEIGHT = 1.70, GENERATED_AT
```

To add a procedural dance: append a definition to `CHOREO_DEFS` (moves must sum to
`durationBeats`), run the generator, add tests/expectations if needed. Recorded dances do not go
through the generator (see `webxr/choreos/README.md`).

## Appendix: net-js (src/net/pipeline.js, worker.js, avatar-driver.js) API

JS port of §6 (lane *net-js*), verified against the Python oracle fixtures: `tests/unit/pipeline-math.test.js`
(39 `math.json` cases, max abs error 5.7e‑7) and `tests/unit/pipeline.test.js` (`mapping_io.json` /
`tracking_io.json`: input tensors max abs error 5.7e‑7 vs limit 1e‑5, decoding + post‑processing
5.6e‑7 vs limit 1e‑4). Everything in `pipeline.js` is importable in Node without three.js or
onnxruntime; hot paths write into preallocated typed arrays (out‑parameter + offset style:
`fn(out, outOffset, …, in, inOffset)`), the `create*` helpers allocate once. Full usage notes:
`webxr/src/net/README.md`.

### `src/net/pipeline.js`

Constants: `FPS 30, DT, N_JOINTS 34, JOINT_NAMES, PARENTS, HEAD 27, R_WRIST 33, L_WRIST 25,
INPUT_JOINTS [27,33,25], CONTACT_JOINTS [5,8,13,16], N_HISTORY 15, N_FUTURE 30, N_LEADER_TRACKING 31,
FRAME_DIM 13, FRAME_HEAD 0, FRAME_QUAT 3, FRAME_RW 7, FRAME_LW 10, YAW_EPS 1e-3,
BONE_LENGTH_TOLERANCE 1.05, MAPPING_INPUT_NAMES/OUTPUT_NAMES, TRACKING_INPUT_NAMES/OUTPUT_NAMES,
MAPPING_INPUT_DIMS, MAPPING_OUTPUT_DIMS, TRACKING_INPUT_DIMS, TRACKING_OUTPUT_DIMS` (name → flat length).

```
wrapAngle(a) -> [-pi, pi)                         quatToMat(out9, oo, q, qo)   (row-major, not normalised)
quatRotate(out, oo, q, qo, v, vo)                 headForward(out3, oo, q, qo) = q*(0,0,1)
yawFromForward(fx, fz, prevYaw, eps=1e-3)         rootFromHead(outPos, po, head, ho, fwd, fo, prevYaw) -> yaw
rmFromRoot(out4, oo, x, z, yaw)                   rmYaw(rm, o=0) = -atan2(rm[1], rm[0])
rootMatrix(out16, oo, x, y, z, yaw)               rmToMatrix(out16, oo, rm, ro)   rmFromMatrix(out4, oo, m16, mo)
rmRelative(out4, oo, rm, ro, ref, fo)             rmApply(out4, oo, ref, fo, rel, ro)
toLocal(out3, oo, m16, mo, p, po) = M^-1 p        toWorld(out3, oo, m16, mo, p, po) = M p   rotateByMatrix(out3, oo, m16, mo, v, vo) = R v
dpToSPos / sToDpPos (out, oo, p, po)              dpToSQuat / sToDpQuat (out, oo, q, qo)
frameDpToS / frameSToDp (out13, oo, f, fo)        dpToSPositions(out, p, n)
channelMajor(out, x, T, C, xo=0)                  splitFrame(out, flat, T, C, t)      flattenInto(out, nestedJson, offset=0) -> count
frameRoots(frames(T*13), T, prevYaw, outPos(T*3), outYaw(T), fwd3)
threePointLocal(frames, T, rootPos, yaw, outLocal(T*9), rm4, m16)
createMappingWork() -> {rootPos, yaw, rm, rmRef, rmRel, local, feeds{input_leader_Positions F32[135], input_leader_RootMotion F32[60]}, …}
mappingInputs(frames15, prevYaw, work) -> work.feeds
createMappingDecode() -> {rm(30*4), rmRel, local(30*9), world(30*9)};  decodeMapping(outputs{Positions, RootMotion}, rmRef, out)
createAvatarState() -> {rm(4), positions(102), rotations(306), velocities(102), contacts(4), world(102), matrix(16)}
avatarFromPose(avatar, pose{positions, rotations, velocities?, contacts?}, rm, ro=0);  avatarUpdateWorld(avatar);  copyAvatar(dst, src)
createTrackingFeeds() -> the 7 Float32Array inputs (+ scratch rel);  trackingInputs(leaderRm(31*4), leaderLocal(31*9), avatar, feeds)
createTrackingPred() -> {velocities, positions, rotations, contacts, rm};  decodeTracking(outputs, frame, pred)
correctRoot(out4, rm, measuredRm, factor)         prepareSkeleton(skeletonJson) -> {joints, parents Int32Array, boneLengths Float64Array, n}
restoreBoneLengths(world(n*3), parents, boneLengths, tolerance, mark Uint8Array(n)) -> clamps   (subtree shift, parents first)
createPostWork();  postProcess(avatar, pred, measuredRm, skeleton, rootCorrection, eulerRatio, dt, work) -> avatar (in place)
sampleSToFrameDP(out13, oo, sampleS, k)           (stage sample at reference height -> unscaled DP frame)
class DPPipeline({skeleton, initPose, rootCorrection=0.35, eulerRatio=0.5, dt=DT, useMeasuredFrame=true})
  pushFrame(frame13, offset=0) -> tick   (first frame pre-fills the 16-slot ring; yaw fallback chain)
  buildMappingInputs() -> feeds;  applyMappingOutputs(outputs) -> this.mapped;  setAvatarPose(pose|null, rm, ro=0)
  buildTrackingInputs() -> feeds (avatar auto-initialised from initPose at the measured root on first use)
  applyTrackingOutputs(outputs, frame=0) -> this.pose {tick, positionsWorldDP F32[102], positionsWorldS F32[102], rootDP, rootS F32[3],
                                                        rootYawDP, rootYawS (= -rootYawDP), rm F64[4], contacts F32[4], clamps}
  async runTick(frame, models{runMapping(feeds), runTracking(feeds)}, offset=0) -> pose;  reset();  frameCount;  lastYaw;  measuredRm;
  mappingWork, mapped, leaderRm, leaderLocal, trackingFeeds, pred, avatar, skeleton, windowPrevYaw
```

### `src/net/worker.js` (ES module worker, `new Worker(url, {type:'module'})`)

Loads onnxruntime-web by dynamic `import('../../vendor/ort/ort.wasm.min.mjs')` (ESM build, vendored
from `onnxruntime-web@1.20.1/dist`), falls back to evaluating the classic `ort.wasm.min.js`;
`ort.env.wasm.wasmPaths = {mjs, wasm}` in the vendor dir, `numThreads` from `init` (1 unless
cross‑origin isolated), `simd` on, `proxy` off. Feed tensors are built once over the pipeline
buffers; pose buffers (436 B: positions S 102 floats @0, rootS @408, contacts @420) are transferred
and recycled. Protocol (details in `src/net/README.md`): main→worker `init{modelsUrl, set, ortUrl?,
wasmDir?, rootCorrection, eulerRatio, inferEvery, telemetry, telemetryTicks, numThreads, simd}`,
`frame{seq, t, frame F32[13] DP}`, `config{inferEvery?, rootCorrection?, eulerRatio?, telemetry?}`,
`reset`, `telemetry`, `recycle{buffer}`, `dispose`; worker→main `ready{loadMs, createMs, warmupMs,
ortVersion, ortSource, simd, numThreads, models, joints, parents, boneLengths}`, `progress{label,
loaded, total}`, `pose{tick, t, seq, buffer, rootYawS, clamps, inferenceMs, mappingMs, trackingMs,
dropped}`, `telemetry{data}` (`dancing-points-telemetry/1`: per tick all inputs, mapping outputs,
tracking output frame 0, measuredRm, avatarRm, worldDP; 7 significant digits), `error{stage,
message}`. Frames arriving while an inference runs are pushed into the history but not inferred
(`dropped` counter); `inferEvery = 2` implements the 15 Hz guard mode.

### `src/net/avatar-driver.js`

`class NeuralAvatarDriver({workerUrl?, modelsUrl?, style='free', set='leader', tickHz=30,
rootCorrection, eulerRatio, telemetry, telemetryTicks, numThreads?, guard{windowTicks 60, slowMs 25,
disableMs 60, minTicks 10, recoverRatio 0.7}, onPose, onStatus, onError, WorkerClass?, now?, ortUrl?,
wasmDir?})`: `init() → Promise<readyInfo>`, `start()`, `stop()`, `update(sampleS, tSec?)` (render
loop; stage sample at reference height with `sample.k`; linear resampling onto the 30 Hz grid),
`pushFrameDP(frame13, tSec)`, `getPose(out?, tMs?) → Float32Array(102) | null` (positions in the
**unscaled** stage frame S, interpolated between the last two poses over a smoothed arrival
interval `poseIntervalMs`), `rootS`, `latest {tick, t, seq, positionsWorldS, rootS, rootYawS, contacts,
inferenceMs, mappingMs, trackingMs, arrivedMs}`, `scale` (= last `sample.k`), `state` (`idle,
loading, ready, running, slow, disabled, error, disposed`), `notice`, `inferenceMs` (rolling mean),
`inferEvery`, `stats {frames, poses, dropped, meanMs, maxMs, minMs}`, `setConfig({rootCorrection,
eulerRatio, telemetry})`, `reset()`, `exportTelemetry() → Promise<object>`, `downloadTelemetry(name?)`,
`dispose()`. `NOTICES` = the German strings (`disabledSlow = 'Neural-Avatar deaktiviert (zu
langsam)'`, `slow = 'Neural-Avatar: reduzierte Rate (15 Hz)'`, `loading`, `ready`, `error`).
Default thread count: 1, or `min(4, hardwareConcurrency − 1)` when the page is cross‑origin
isolated. Rendering recipe: `calibration.stageToWorld(out, p, calibration.h0)` for W coordinates
at the player's place; `× driver.scale` to compare with a choreography at reference height.

### Measured (headless Chromium, `tests/net/worker-smoke.mjs`, this container)

int8 + 1 wasm thread: 34 ms per inference (mapping 10 + tracking 24) → the guard selects 15 Hz;
fp32: 44 ms; int8 + 4 threads on a cross‑origin‑isolated page: 16 ms (30 Hz). Model load ~1.1 s from
localhost. `tests/net/ort-parity.mjs` (onnxruntime-node): int8 vs fp32 outputs 0.2–2.3 % relative,
decoded mapping world 0.45 cm, post‑processed tracking world 0.22–0.50 cm; 5.5 ms per closed‑loop
tick natively. **Recommendation for `server/server.js`**: send `Cross-Origin-Opener-Policy:
same-origin` and `Cross-Origin-Embedder-Policy: require-corp` so the Quest Browser can use WASM
threads (all app resources are same‑origin); without them the worker runs single‑threaded.

## Appendix: presentation API

Lane *game-presentation*: `webxr/index.html`, `manifest.webmanifest`, `sw.js`, `assets/icon*`,
`src/app.js`, `src/render/{scene,avatar,hud}.js`, `src/ui/menu.js`, `tests/e2e/README.md`.
These modules import `three` (import map in `index.html`) and are exercised only in the browser
(headless Chromium smoke test, see `tests/e2e/README.md`). Verified headless: exact playback
scores `snoop-cwalk` 100, `tutorial-basics` 100, `mocap-freestyle` 99 (BodyAvatar path),
`noise=0.35` → 1; no console errors; the menu works with mouse + keyboard; the record flow
produces a valid 61-frame dance; the service worker serves the shell offline.

### `src/app.js` — `App` (extends `EventEmitter`), `STATES`, `parseExtraParams`, `mirroredChoreoView`

Auto-boots in the browser (`window.__dp = { app, version }`; set `window.__dpNoAutoBoot = true`
before importing to construct it yourself: `new App({ params, storage, document, window })`).
`STATES = { BOOT, MENU, SETTINGS, CALIBRATE, COUNTDOWN, PLAYING, RESULTS, RECORD_SETUP,
RECORDING, RECORD_REVIEW, DUO_LOBBY }` (strings, §4). Public surface:

```
state, prevState, version, params, texts, local (extra strings), settings {avatar, volume, lang, playerName, telemetry}
sceneKit, menu, hud, teacher (TeacherAvatar at S (0,0,-teacherDistance)), markers (ControllerMarkers, W),
playerBody (BodyAvatar | null), skeleton (models/<style>/skeleton.json, loaded at boot)
input, inputKind 'xr'|'desktop'|'playback', xrSession, calibration, audioCtx, audio (AudioEngine), clock (BeatClock)
choreos [{id,title,artist,bpm,durationBeats,difficulty,source:'shipped'|'server'|'own', url?|json?}], selectedChoreoId, selectedEntry
choreo (current Choreo), session (PlaySession | null), scorer, recorder, lastResult, lastRun (run JSON), playerS (stage-frame sample, every frame)
modes Map, modeName, mode, bodyPoseProvider, telemetryExporter, frameCount

boot() → Promise<App>; enterVR() → Promise<XRSession> (user gesture); enterEmu('desktop'|'playback')
loadChoreoList() → entries (index.json + `${server}/api/choreos` + Recorder.listPersisted); loadChoreoRef(idOrUrlOrJsonOrChoreo) → Promise<Choreo> (cached)
startSolo(choreoRef, {mode='solo', recalibrate, player}) → Promise<Choreo> (resolves when CALIBRATE is entered)
startMode(name, choreoRef?, opts?)  (= startSolo with mode); playAgain(); abortToMenu(); backToMenu()
openSettings(); openHighscores(); openDuoLobby(); startRecordSetup(); beginRecording(); saveRecording() → Promise<bool>; discardRecording()
setAvatarMode('neural'|'points'|'off'); setVolume(0..1); saveSettings(); exportTelemetry()
localResults() → dp.results array; highscores(choreoId, n=10) → sorted results (local + cached server)
registerMode(name, mode); setBodyAvatarPoseProvider(fn, skeleton?) → BodyAvatar|null; setTelemetryExporter(fn)
```

Events (`app.on(type, fn)`): `'boot'`, `'input'` (backend replaced), `'state' {from, to}`,
`'choreos'`, `'calibrated'` (data), `'session'` (PlaySession created + started), `'result'`,
`'recording'` (Recorder), `'recorded'` (json), `'saved'` (json), `'lobby'` (ctx), `'mode'`,
`'settings'`, `'frame' (timeMs, dtSec)` (every render frame, after the state update).

**Main loop** (`_frame(time, xrFrame)` via `renderer.setAnimationLoop`): input update (`XRInput.update(frame,
refSpace, session, time)` / `EmulatedInput.update(time)`) → `calibration.toStage(sample, choreo.referenceHeight,
app.playerS)` → camera from the head pose when not presenting (`?cam=third` = third-person) → menu pointer
(XR: laser from `sceneKit.controllers[hand].ray`, the hand that last pressed its trigger; trigger =
`menu.press()`) → state update → body-pose provider → `menu.update()`, `hud.update(dt)` → render.

**Calibration** (state CALIBRATE, `_enterCalibrate({then:'play'|'record'|'menu', recalibrate})`): XR = hold any
trigger for `FLOW.calibrationHoldSec` (averaged capture); a persisted calibration younger than 10 min whose
`h0` is within 10 cm of the current head height and whose origin is within 0.5 m is reused after 0.6 s
("Kalibrierung übernommen") unless `recalibrate`; emu = automatic after `FLOW.calibrationEmuSec / speed`.
Afterwards `sceneKit.setStageMatrix(calibration.stageToWorldMatrix(m16, choreo.referenceHeight))` — every
S-frame object (teacher, HUD, menu, ghost/duo avatars, player body) lives under `sceneKit.stage`.

**Playback determinism**: with `emu=playback` the App owns a virtual time source (`app._virtual.source`)
for `app.clock` and advances it in `_updatePlay` in sub-steps ≤ one scoring tick (`PlaySession.scoreHz`)
per render frame, so scoring runs at the full 30 Hz whatever the frame rate; the backend gets
`mirroredChoreoView(choreo)` (a `{referenceHeight, sampleAt}` view returning `mirrorChoreoSample` for
`mirror: true`, so the ideal player performs what the scorer expects). Audio is disabled in playback mode.

**Persistence**: `localStorage['dp.results']` = JSON array of `resultToJSON(result) + {choreoTitle}` (max
500, POSTed to `${httpBase}/api/results`); `localStorage['dp.runs']` = JSON array of
`dancing-points-run/1` objects (newest last, max 3 per `choreoId`, POSTed to `/api/runs`) — **the ghost duo
reads this**; `dp.settings`, `dp.calibration` (Calibration), `dp.choreos` (Recorder map). `params.server`
(`wss://` or `https://`) is turned into the REST base by replacing `ws` → `http`.

**Mode interface** (duo lane; `app.registerMode('ghost' | 'online', mode)`; the "Duo-Modus" panel enables
its buttons when a mode is registered; `mode.lobbyPanelId` names a menu panel shown alongside the duo panel
in DUO_LOBBY):

```
mode = {
  init(app)?,                            // once at registration
  lobbyPanelId?,                         // menu panel id to show in the lobby (menu.addPanel first)
  async prepare(ctx)?,                   // after the choreo is loaded, before CALIBRATE (throw → notice + MENU)
  onSessionCreated(session, ctx)?,       // PlaySession built, not yet started: attach listeners / avatars
  update(timeMs, dtSec, session)?,       // every frame in COUNTDOWN/PLAYING
  onFinished(result, ctx) → { lines: string[], moveLines: string[] } | null,   // extra rows on the results panels
  exit(ctx)?,                            // leaving the run (results → menu, abort)
}
ctx = { app, choreo, calibration, sceneKit, stage, menu, hud, texts, teacher, input, clock, storage, server, httpBase }
```
`PlaySession.mode` is the mode name; the second dancer belongs at
`stage` position `(STAGE.duoSideOffset, 0, -STAGE.teacherDistance)` (e.g. `new PointAvatar({color:
AVATAR_COLORS.ghost, opacity: 0.6})`, `setSample(sampleS)` per frame; a `BodyAvatar` for full-body runs).

**Neural avatar hook**: `app.setBodyAvatarPoseProvider((timeMs, playerS, app) => Float32Array(34*3) | null,
skeleton?)` renders the player's body (translucent `BodyAvatar` under `stage`, stage-frame positions)
whenever the provider returns a pose and `settings.avatar === 'neural'`; `app.playerS` is the current
stage-frame sample for the driver's input; `app.setTelemetryExporter(fn)` backs "Telemetrie exportieren"
(return `false` when nothing was recorded → HUD notice).

Extra URL parameters parsed by `parseExtraParams(location.search)`: `cam=third`, `height=<m>`,
`yaw=<rad>` (virtual playback player), and `choreo=<URL>` (config.parseParams rejects URLs, the App
accepts absolute/relative URLs ending in `.json` or containing `/`).

### `src/render/scene.js` — `SceneKit`, `SCENE_COLORS`, `makeGridTexture`

```
new SceneKit({ container=document.body, canvas?, xr=true, pixelRatio=1, floorSize, teacherDistance })
renderer (WebGLRenderer: xr.enabled, 'local-floor', foveation 1, framebuffer scale 1, pixel ratio 1, no shadows)
scene, camera (PerspectiveCamera 70°, 0.05–80 m), stage (Group, matrixAutoUpdate=false, S→W), sky, floor, marker,
platform (2.4×0.06×1.6 m at S z=-teacherDistance), screen (wall 4.4×2.6 m at z = screenZ = -(teacherDistance+0.7)),
screenWidth, screenHeight, screenZ, hemi/key/fill lights, controllers {left|right: {ray, grip, source, index, hand} | null}
setStageMatrix(array16); resetStage(); setMarkerHighlight(on, phase); setAnimationLoop(fn); render();
startXR(session) → Promise<XRReferenceSpace> (onSessionEnd callback); endXR(); referenceSpace; presenting;
resize(); setCameraFromPose({p, q}); setThirdPersonCamera(); dispose()
```

### `src/render/avatar.js` — `PointAvatar`, `BodyAvatar`, `TeacherAvatar`, `ControllerMarkers`, `AVATAR_COLORS`

* `new PointAvatar({color, opacity=1, headRadius=0.12, handRadius=0.055, lines=true, mirror=false, showNose=true})`
  (Group): `setSample({head:{p,q}, left:{p}, right:{p}})` in the parent's frame (no allocation), `mirror`
  property (x → −x, hands swapped, yaw mirrored), `setColor`, `setOpacity`, `dispose`.
* `new BodyAvatar({skeleton: {joints, parents, boneLengths?}, color, opacity=1, radiusScale=1, headSphere=true})`:
  one `InstancedMesh` of unit cylinders (one draw call, bones with parent; skeleton bones < 1.5 cm skipped)
  + a head sphere between `b_head` and `b_head_null`; `setPose(Float32Array jointCount*3)` (parent frame),
  `hasPose`, `bones`, `setColor`, `setOpacity`, `dispose`. Works with the 34-joint `skeleton.json` and with
  a choreo's `fullBody.joints/parents`.
* `new TeacherAvatar({color, opacity, mode:'auto'|'points'})`: `setChoreo(choreo)` (BodyAvatar when
  `choreo.hasFullBody`, else PointAvatar; `mirror: true` → rotated 180° to face the player), `setMode`,
  `usesBody`, `update(t, refSample?)` (uses `choreo.fullBodyAt(t)` or the sample), `setColor/Opacity`.
* `new ControllerMarkers({color, opacity, radius=0.035})`: `setSample(sampleW, showHead=false)`.
* `AVATAR_COLORS = { teacher 0x4cc9f0, player 0xffd166, ghost 0xb388ff, opponent 0xff6b6b }`.

### `src/render/hud.js` — `HUD`, `TextPanel`, helpers

`new TextPanel({width, height (m), pxPerMeter=320, background, border, rounded})` (Mesh with a
`CanvasTexture`): `draw((ctx, w, h) => …, key)` redraws only when `key` changes, `invalidate()`,
`setOpacity`, `dispose`. Helpers: `roundRect`, `fitText(ctx, text, x, y, maxWidth, size, weight, align,
color)`, `starsString(n)`, `formatTime(sec)`, `HUD_COLORS`, `FONT`.
`new HUD({texts, screenWidth, screenHeight})` (Group placed at `(0, 0, sceneKit.screenZ)` under `stage`):
panels `scorePanel`, `titlePanel`, `timePanel` (top row), `movePanel` (above the teacher's head),
progress bar (top edge), `countdownPanel` (left of the teacher), `gradePanel` (right, animated),
`noticePanel` (bottom). Setters compare with the last value (cheap per frame): `showScore(score, combo)`,
`showMove(current, next, hint)`, `showTitle(text|null)`, `showTime(secondsLeft)`, `showProgress(0..1)`,
`showCountdown(n | 'go' | null)`, `showGrade(grade, combo)`, `showNotice(text, seconds)`, `setPlayMode(on)`,
`reset()`, `update(dt)`; extra panels for other lanes: `addPanel(id, {width, height, position, background,
border}) → TextPanel`, `getPanel(id)`, `removePanel(id)`.

### `src/ui/menu.js` — `Menu`, `MenuPanel`, `MENU_COLORS`, `DEFAULT_PPM`

`new Menu({texts, camera, app, pxPerMeter=640})` (Group; the App places it at S `(0, 0, -1.9)` under
`stage`; add `menu.laser` and `menu.reticle` to the scene):

```
addPanel(id, spec) → MenuPanel; setPanel(id, spec); removePanel(id); getPanel(id); show(id, alongsideIds?);
showAlso(id); hidePanel(id); hideAll(); refresh(id?); activeId; active
setPointer(originV3, dirV3, hand); setPointerFromObject(obj3d, hand) (−Z of its world matrix, e.g. the XR
controller ray); setPointerFromPose({p,q}, hand); clearPointer(); setMouse(ndcX, ndcY); press() → bool;
activateFocused(); back(); handleKey(code) → bool (arrows, Enter, Escape/Backspace, Digit1-9); update()
(raycast, hover, laser, redraw dirty panels); on('action', ({panelId, itemId, item, option?, step?}) => …)
```
Panel spec: `{ title, width=1.6 (m), position=[x,y,z], rotationY, items | () => items, footer, onBack(app),
persistent, minHeight, background, pxPerMeter }`. Items (`text`, `value`, `options`, `items`, `disabled`
may be functions of the app): `label {text, size:'small'|'normal'|'large', align, color}`, `text {text}`
(wrapped), `button {id, text, onClick(app, item), disabled, primary, color}`, `toggle {id, text, value,
onChange(v, app)}`, `row {items: [button|toggle…]}`, `list {id, options:[{id, text, sub, color}], selected,
pageSize=5, onSelect(id, option, app), emptyText}` (paged), `stepper {id, text, value, onDec, onInc, onDec2?,
onInc2?}`, `progress {value, color}`, `spacer {height}`. A panel's height follows its content (the canvas
texture is re-created on resize); `panel.hits` are the clickable regions in canvas pixels (`panel.hitAt(px,
py)`), `panel.refresh()` re-layouts, `panel.invalidate()` redraws. Panels created by the App: `main`,
`settings`, `duo`, `calibrate`, `record-setup`, `record-review`, `results`, `results-moves`, `highscores`.

### `index.html`, `manifest.webmanifest`, `sw.js`, `assets/`

`index.html`: import map `three → ./vendor/three.module.js`, start overlay (`#overlay`, `#btn-vr` "VR
starten" → `app.enterVR()` requesting `immersive-vr` with `requiredFeatures: ['local-floor']`,
`optionalFeatures: ['hand-tracking', 'bounded-floor']`; `#btn-emu` "Am PC testen" → `app.enterEmu('desktop')`;
`#status`; `#version`), loads `./src/app.js`. No external requests. `sw.js` is registered by the App as
`./sw.js?v=<APP_VERSION>` (+ `registration.update()` at every boot; a `controllerchange` after an
update reloads the page once it is back in the menu). Strategy (superseding the first version, see the
fixer appendix): the **shell** (`index.html`, manifest, `src/`, `choreos/`, `assets/`, `models/*/*.json`)
is served **network-first** with a conditional request (`cache: 'no-cache'`, ETag/304) and falls back to
the cache `dp-shell-<SW_VERSION>` offline; it is precached on install (all `src/` modules, every entry of
`choreos/index.json`). The big immutable **assets** (`vendor/`, `models/*.onnx`) are **cache-first** in the
unversioned cache `dp-assets` (returned to the page immediately, stored via `event.waitUntil`; never deleted
on activate, revalidated once by ETag after a version bump). `/api/*` is network-first with cache fallback
(`/api/info`, `/api/health` never cached). `SW_VERSION` inside `sw.js` must equal `APP_VERSION`
(`tests/unit/fixes.test.js`); old `dp-shell-*` caches are deleted on activate; `postMessage({type:'clearCache'})`
empties both caches. A service worker only registers on a trusted‑certificate origin (see §1). Icons:
`assets/icon.svg` (source) and `icon-192.png` / `icon-512.png` rendered from it with headless Chromium.

Addendum (presentation): `app.setBodyAvatarPoseProvider(fn, skeleton?, { unscaled: true })` multiplies the
provider's pose by `playerS.k` before rendering — use it with `NeuralAvatarDriver.getPose` (unscaled S, real
metres): `app.setBodyAvatarPoseProvider((t) => driver.getPose(buf, t), skeleton, { unscaled: true })`.

## Appendix: server & duo API

Lane *server-duo*: `server/server.js` (Node ≥ 18, deps `ws`, `selfsigned`; `npm install` inside
`server/`), `webxr/src/duo/ghost.js`, `webxr/src/duo/online.js`, `webxr/src/duo/benchmark.js`.
Tests: `tests/unit/{ghost,benchmark,server}.test.js` (the server test spawns the server as a
child process with `--http --http-port 0 --no-https --data <tmp>` and skips itself when `ws` is
not installed). All duo modules are pure JS (no three.js/DOM), importable in Node.

### `server/server.js`

`node server/server.js [--port 8443] [--http-port 8080] [--http|--no-http] [--no-https]
[--host 0.0.0.0] [--dir ../webxr] [--data server/data] [--cert x.pem --key y.pem] [--token secret]
[--no-coep] [--verbose] [--quiet]`. Defaults: HTTPS **and** HTTP listeners on; `--http` only forces the HTTP
listener on, `--no-https` drops the HTTPS listener (no certificate needed – use it in tests/CI,
e.g. `--http --http-port 8090 --no-https`, otherwise a first start also generates a certificate
and binds 8443). `--http-port 0` picks a free port. After start‑up stdout carries one
machine‑readable line `READY {"https":8443|null,"http":8080|null,"urls":[…],"data":"…"}`.
Certificate: `<data>/cert/server.{key,crt}` + `meta.json`, self‑signed, 10 years, SANs =
`localhost, 127.0.0.1` + all LAN IPv4s; regenerated automatically when a LAN IP is missing from
the SANs or the cert expires within 30 days. Exports (for programmatic use): `DancingPointsServer`
(`new DancingPointsServer(parseArgs(argv)).start()`, `.close()`, `.info()`, `.urls()`,
`.httpPort/.httpsPort`), `parseArgs`, `ensureCertificate`, `lanAddresses`, `validateResult`,
`validateRun`, `fallbackValidateChoreo`, constants `SERVER_VERSION, WS_PATH='/ws',
MAX_WS_MESSAGE=4096, RATE_PER_SEC=30, RATE_BURST=30, MAX_ROOM_PLAYERS=2, MAX_RUNS_PER_CHOREO=50,
MAX_RESULTS=5000, MAX_BODY_CHOREO=5 MB, MAX_BODY_DEFAULT=1 MB`.

Static: `--dir` root, directory → `index.html` (301 to the trailing slash), MIME per extension
(`.js/.mjs` `text/javascript`, `.wasm` `application/wasm`, `.onnx/.bin` `application/octet-stream`,
`.json` `application/json`, `.webmanifest` `application/manifest+json`, audio/image/font types),
`Cache-Control: no-cache` for `.html/.json/.webmanifest`, `sw.js` and everything under `src/` and
`choreos/` (ETag revalidation: a fix reaches the headsets at the next load), `public, max-age=3600`
otherwise (vendor, models), weak ETag + `If-None-Match` → 304, `Range` → 206, HEAD, path traversal
blocked. Every response carries `Cross-Origin-Opener-Policy: same-origin`,
`Cross-Origin-Embedder-Policy: require-corp` (unless `--no-coep`), `Cross-Origin-Resource-Policy:
cross-origin`, `Access-Control-Allow-Origin: *` (static files too: a Pages‑hosted app loads choreo
JSON/audio with `fetch()`), `X-Content-Type-Options: nosniff`; `/api/*` additionally the CORS
method/header allowances (+ OPTIONS preflight, `X-Dp-Token` allowed), so the app may also be served
from GitHub Pages and talk to a LAN server via `?server=`.

**Write access** (`_authorize`): with `--token <secret>` every `POST`/`DELETE` under `/api/` must carry
`X-Dp-Token: <secret>` (or `?token=`), otherwise 401; without a token `POST` is open (headsets upload
results/runs/recordings) and `DELETE` is only accepted from the loopback interface (403 otherwise).
The app takes the token once from `?token=` (`config.parseParams`), persists it in `dp.settings.apiToken`
and sends the header on every upload. `GET /api/info` reports `auth: 'token' | 'none'`.

REST (JSON; errors are `{error, errors?}` with 400/404/405/413/500):

| route | behaviour |
|---|---|
| `GET /api/info` | `{name, version (APP_VERSION), serverVersion, urls[], primaryUrl, https, http, httpsPort, httpPort, wsPath, serverTime, uptimeSec, rooms, clients, choreoValidator:'app'\|'fallback', limits{…}}` |
| `GET /api/health` | `{ok:true, serverTime}` |
| `GET /api/choreos` | `{choreos:[entry]}`: `webxr/choreos/index.json` entries (`+ url:'/choreos/<file>', source:'shipped'`) merged with uploads (`{id,title,artist,bpm,durationBeats,difficulty,file:'<id>.json', url:'/api/choreos/<id>', source:'upload', uploadedAt}`); an upload with a shipped id replaces that entry in place (`replaces:'shipped'`). **Clients should load `entry.url`** (absolute path on the server origin). |
| `GET /api/choreos/:id` | the JSON (upload first, then the shipped file), `no-cache` |
| `POST /api/choreos` | body ≤ 5 MB, `validateChoreo` from `webxr/src/game/choreo.js` (structural fallback when the app tree is absent) → `data/choreos/<id>.json`; 201 `{ok, id, url, entry}` / 400 `{error, errors[]}` |
| `DELETE /api/choreos/:id` | removes an upload (404 for shipped ids) |
| `GET /api/results?choreoId=&player=&limit=100` | `{results:[…]}` newest first |
| `POST /api/results` | ≤ 1 MB; `validateResult` (`choreoId` id‑pattern, `score` 0..100, optional `stars 0..5, maxCombo ≥ 0, timingBias, player, mode, playedAt ISO, moves ≤ 256, perBeat ≤ 8192`); **`sanitizeResult`** keeps only the known fields (`choreoId, score, stars, maxCombo, timingBias, durationSec, ticks, meanFrameScore, mirror, player ≤ 64, mode, playedAt, choreoTitle ≤ 200, moves ≤ 256 × {name, score, grade, startBeat, endBeat, ticks, gated}, perBeat ≤ 8192 numbers, benchmark {mode, winner, pair (numbers), partner {name, score, stars, maxCombo}} ≤ 8 KB`); adds `id, receivedAt` (+ `playedAt` if missing); stored in `data/results.json` (debounced atomic write, ≤ 5000 kept); 201 `{ok, id, result}`. An over‑limit body gets a real 413 (chunked uploads are drained, not reset). |
| `GET /api/runs?choreoId=&limit=100` | `{runs:[meta]}` newest first, **metadata only** `{id, choreoId, player, score, mode, playedAt, receivedAt, fps, frameCount, url:'/api/runs/<id>'}` |
| `GET /api/runs/:id` | the full run JSON as uploaded (+ `id, receivedAt`) |
| `POST /api/runs` | ≤ 5 MB; `validateRun` accepts `dancing-points-run/1` (`head[][7]/left[][3]/right[][3]`) or the compact `{frames:[[hx,hy,hz,lx,ly,lz,rx,ry,rz]]}` layout, `fps` in (0,240], 1..20000 frames; stored as `data/runs/<choreoId>/<id>.json`; **cap 50 per choreography** (oldest by `receivedAt` deleted, ids returned in `removed[]`); 201 `{ok, id, url, run: meta, removed}` |
| `DELETE /api/runs/:id` | removes one run |
| `GET /api/rooms` | `{rooms:[{code, choreoId, players:[{id,name,host}], maxPlayers, startAt, createdAt}]}` |

WebSocket relay at `/ws` on both listeners (`wss://ip:8443/ws`, `ws://ip:8080/ws`). Text
frames with JSON, `type` required, ≤ 4 KB (`ws` closes with 1009 above), token bucket 30 msg/s
with burst 30 per client (excess dropped silently; one `{type:'error', code:'RATE_LIMIT',
dropped}` per second). Server → client on connect: `{type:'hello', id, serverTime, version,
maxPlayers}`. Client → server:

| message | reply / effect |
|---|---|
| `{type:'ping', t0}` | `{type:'pong', t0, t1}` (`t1` = server `Date.now()`) |
| `{type:'create', name, choreoId?}` | creates a room (code: 4 letters from `ABCDEFGHJKLMNPQRSTUVWXYZ`), sender = host; reply `{type:'room', code, choreoId, players, maxPlayers, startAt, createdAt, you, host:true}` |
| `{type:'join', code, name}` | reply `room` (`host:false`); others get `{type:'peer', event:'joined', room, player, players}`; errors `ROOM_NOT_FOUND`, `ROOM_FULL` |
| `{type:'leave'}` | reply `{type:'left', room:null}`; others get `peer` `event:'left'` and, if the host left, `peer` `event:'host'` (first remaining player becomes host). Closing the socket leaves too. Empty rooms are deleted. |
| `{type:'setChoreo', choreoId}` | host only; **everyone** in the room (incl. sender) gets `{type:'choreo', room, from, choreoId}` |
| `{type:'start', choreoId?, delayMs=3000}` | host only (`NOT_HOST`); `delayMs` clamped 200..30000; **everyone** gets `{type:'start', room, from, choreoId, startAt, delayMs, serverTime, players}` with `startAt` = server clock ms at which the **count‑in** begins |
| `{type:'state', t, head:[7], left:[3], right:[3], score, combo}` | relayed to the **other** room member(s) only, with `from` (client id) and `room` added |
| any other type (`result`, `abort`, `emote`, …) | relayed likewise (`NOT_IN_ROOM` when not in a room). Types `hello/pong/error/room/peer/left` are reserved (`BAD_MESSAGE`). |

Errors: `{type:'error', code, message (German, player‑facing), …}` with codes `ROOM_NOT_FOUND,
ROOM_FULL, NOT_HOST, NOT_IN_ROOM, BAD_REQUEST, RATE_LIMIT, BAD_JSON, BAD_MESSAGE`. Heartbeat: ws
ping every 30 s, unresponsive sockets are terminated.

### `src/duo/ghost.js`

Run layouts accepted everywhere: `dancing-points-run/1` (from `PlaySession.runToJSON()`), the
compact `{choreoId, fps, frames:[[hx,hy,hz, lx,ly,lz, rx,ry,rz], …]}` (head quaternion = identity,
`hasQuat=false`), or an already normalized run.
`validateRun(run) → {ok, errors, frameCount}`, `normalizeRun(run) → {normalized:true, format,
choreoId, fps, frameCount, duration, head Float32Array(N·7), left/right Float32Array(N·3),
hasQuat, score, player, playedAt, mode, id}` (idempotent, throws with `.errors`),
`runToJSON(run, {compact=false, digits=4})`, `sampleRunInto(run, t, outHead, oh, outLeft, ol,
outRight, or) → bool` (lerp into flat arrays at offsets, clamped, no allocation),
`createRunSample() → {t, head:{p,q}, left:{p}, right:{p}}` (same shape as `Choreo.sampleAt`).
`class GhostDuo { constructor(run, {name?}); run, choreoId, fps, frameCount, duration, score,
player, playedAt, hasQuat, name; sampleAt(t, out?) (lerp + slerp, clamped, reused internal object,
0 allocations); frameIndexAt(t); headSpeedAt(t); toJSON(opts) }`. Render recipe: `ghost.sampleAt(
session.t)` → pose a translucent `PointAvatar` at `STAGE.duoSideOffset` beside the teacher.
Storage (localStorage‑like `{getItem,setItem,removeItem}`, key `STORAGE_KEYS.runs = 'dp.runs'`,
a plain array oldest‑first): `saveRun(run, storage, {key, maxPerChoreo=5, maxTotal=20}) → bool`
(quota error → drops the oldest half and retries once), `loadRuns(storage, {key, choreoId}) →
run[]` newest first (invalid/corrupt entries skipped), `clearRuns(storage, {key, choreoId})`,
`capRuns(list, maxPerChoreo, maxTotal)`, `pickRun(runs, {choreoId, strategy:'latest'|'best'|
'closest', score})`. Server: `toHttpBase(url)` (`wss://h:p/ws` → `https://h:p`),
`uploadRun(run, baseUrl='', fetchImpl?) → reply`, `fetchRunList(baseUrl, {choreoId, limit,
fetchImpl}) → meta[]`, `fetchRun(baseUrl, idOrUrl, fetchImpl?) → run`, `loadGhost({choreoId,
storage?, baseUrl?, fetchImpl?, strategy, score}) → Promise<GhostDuo|null>` (server first, then
local store; server failures are only `console.warn`ed). Constants `RUN_FORMAT,
RUN_STORAGE_KEY, DEFAULT_MAX_RUNS_PER_CHOREO, DEFAULT_MAX_RUNS_TOTAL, MAX_RUN_FRAMES`.

### `src/duo/online.js`

`toWsUrl(serverUrl)` (`https://h:p` | `http://` | `wss://` | `h:p` → `wss://h:p/ws`; `?server=`
values work directly), `toHttpUrl(serverUrl)` (for REST). `class OnlineDuo extends EventEmitter {
constructor({WebSocketImpl = globalThis.WebSocket (Node: pass `ws`), now = Date.now, stateHz=15,
pingCount=5, pingTimeoutMs=2000, requestTimeoutMs=5000, interpolationDelaySec=0.1,
bufferSize=16}) }`. Properties: `state 'idle'|'connecting'|'connected'|'lobby'|'playing'|'closed'`,
`connected`, `id`, `serverVersion`, `room {code, choreoId, players[{id,name,host}], maxPlayers,
startAt}`, `roomCode`, `isHost`, `peer` (the other player or null), `offsetMs` (server − local),
`rttMs`, `synced`, `startAt` (server ms), `startAtLocalMs`, `startChoreoId`, `lastError`,
`stats {sent, received, droppedRate, remoteStates}`, `remoteStateCount`, `remoteLatest`,
`remoteScore`, `remoteCombo`. Methods (promises reject with `err.code` = server error code or on
timeout): `connect(url) → Promise<this>` (resolves after `hello`), `disconnect(code, reason)`,
`dispose()`, `syncClock(count=5) → {offsetMs, rttMs, samples}` (sequential pings, offset =
median of `t1 − (t0 + t3)/2`), `serverNow()`, `toLocalMs(serverMs)`, `toServerMs(localMs)`,
`createRoom(playerName, choreoId?) → room`, `joinRoom(code, playerName) → room`, `leaveRoom()`,
`setChoreo(choreoId)` (host), `start(choreoId, {delayMs=3000}) → {choreoId, startAt,
startAtLocalMs, delayMs}` (host; resolves when the server's broadcast arrives), `sendState(t,
headP, headQ, leftP, rightP, score, combo) → bool` (≤ `stateHz`, one reused message object,
values rounded to 0.1 mm; `false` when throttled or not in a room), `sendStateFromSample(t,
sampleS, score, combo)`, `sendResult(result)` (compact copy: no `perBeat`, moves reduced to
`{name, score, grade}`), `send(type, payload)` / `sendRaw(obj)` (generic relay),
`getRemoteSample(tNow, out?) → {t, head:{p,q}, left:{p}, right:{p}, score, combo, age} | null`
(interpolates the buffered partner states at `tNow − interpolationDelaySec`, nlerp for the
quaternion, holds the newest/oldest state outside the buffer, `age` = seconds since the last
state arrived; reused object, 0 allocations), `resetRemote()`.
Events: `open`, `hello`, `room` (room), `peer` `{event:'joined'|'left'|'host', player, players}`,
`choreo` `{choreoId}`, `start` `{choreoId, startAt, startAtLocalMs, delayMs, from}` (also for the
guest), `state` (raw relayed message), `result` `{from, result}`, `message` (other types),
`error` `{code, message, handled}`, `sync`, `left`, `close` `{code, reason}`, `disconnected`
(unexpected close after a successful connect). The remote buffer is reset on `room`, `start`
and when the peer leaves.
Start recipe on both headsets (after `syncClock()`): `const wait = (startAtLocalMs − Date.now()) /
1000; clock.start(choreo.bpm, choreo.countInBeats, {startAt: clock.sourceNow() + wait});` then
per scored tick `online.sendStateFromSample(session.t, session.playerS, session.score,
session.combo)` and per render frame `online.getRemoteSample(session.t)` → partner avatar
at `STAGE.duoSideOffset`.

### `src/duo/benchmark.js`

Pure functions (may allocate; called once per results screen). Runs may be JSON or normalized.
`alignRuns(runA, runB, {fps=A.fps, lagSec=0, mirror=false, removeOffset=true}) → {n, fps, A, B,
offset, lagSec, mirror}` (both resampled on A's grid over the common duration; `mirror` mirrors
B: x → −x, hands swapped; `removeOffset` subtracts the mean head offset B−A from all B points),
`syncDistanceDetails(runA, runB, opts) → {mean, head, left, right, frames, durationSec, offset,
lagSec, mirror}`, **`syncDistance(runA, runB, opts) → metres`** (mean of the three point
distances; `NaN` without overlap), `headSpeedSignal(run, fps, n) → Float32Array`,
`syncLagDetails(runA, runB, {maxLagSec=1.0, fps=30, lagPenalty=0.1}) → {lag, correlation,
lagFrames, fps, maxLagSec, valid, frames}`, **`syncLag(runA, runB, opts) → seconds, positive = B
is behind A`** (`B(t) ≈ A(t − lag)`; normalized cross‑correlation of the zero‑mean, unit‑variance
head speed signals, peak search tapered by `1 − lagPenalty·|k|/K` so the smallest of the
beat‑periodic peaks wins, parabolic sub‑frame refinement; `valid=false`, lag 0 for constant
signals or overlap < 2·maxLagSec). Measured on the synthetic choreography: lag 0.1 s recovered
within 0.02 s, correlation > 0.9; identical runs → 0 s.
`compareResults(a, b)` (best‑first comparator: score, stars, maxCombo desc, |timingBias| asc,
playedAt asc), **`winner(resultA, resultB) → 'A'|'B'|'tie'`** (score → maxCombo → smaller
|timingBias|; a missing result loses), `playerSummary(result, name?) → {name, score, stars,
maxCombo, timingBias, durationSec, perMove:[{name, score, grade, gated}], mode, playedAt,
choreoId}`, **`buildBenchmark(resultA, resultB, runA?, runB?, {nameA, nameB, mode='duo', mirror,
maxLagSec, createdAt, choreoId})`** →
```
{ format:'dancing-points-benchmark/1', choreoId, createdAt, mode, players:[A, B],
  pair:{ syncDistance, syncDistanceAligned (with the measured lag removed), syncDistanceHead/Left/Right,
         syncLag, syncCorrelation, framesCompared, scoreDiff (A−B), comboDiff }, winner }
```
(pair metrics `null` without both runs). Export: `toJSON(benchmark)` (pretty string),
**`toCSV(benchmark, {separator=','})`** (`section,key,A,B` rows: `meta`, `player` (name, score,
stars, maxCombo, timingBias, durationSec, playedAt), one `move` row per move, `pair` rows incl.
`winner`; CRLF, RFC‑4180 quoting via `csvCell`). Leaderboard: `isResultLike(r)`, `resultKey(r)`,
`mergeResults(...lists)` (de‑duplicates local + server copies), `toEntry(r, rank)`,
**`leaderboard(results, {choreoId, top=10, mode?, player?}) → [{rank, choreoId, player, score,
stars, maxCombo, timingBias, playedAt, mode, source}]`** (`top=0` = all), `leaderboards(results,
{top, mode}) → {choreoId: entries}`, `rankOf(results, result) → 1‑based rank` (the result is
inserted if not present), `leaderboardToCSV(entries)`. Constants `BENCHMARK_FORMAT,
DEFAULT_MAX_LAG_SEC, DEFAULT_LAG_PENALTY, DEFAULT_TOP`.

## Appendix: integration (src/app.js wiring, src/render/mirror.js, src/duo/modes.js, tests/e2e, CI)

Lane *integration*: everything is wired end to end and verified by `npm run test:e2e`
(`tests/e2e/smoke.test.js`, headless Chromium against `server/server.js`). Measured in this
container (SwiftShader, 4 cores): exact playback `snoop-cwalk` 100 / `tutorial-basics` 100,
`noise=0.35` → 1, neural avatar 35 poses in a 4 s run (mean 31.5 ms per inference, 3 wasm
threads because the server's COOP/COEP headers make the page cross‑origin isolated; model load
2.9–3.3 s from localhost), ghost duo benchmark of two exact runs `syncDistance` 0.001 m /
`syncLag` −1 ms, online duo (two pages, one room, `startAt` sync) both 100 with the partner's
result and ~700 recorded partner frames on each side.

### `src/app.js` additions

* **Avatar default**: `settings.avatar` is `'neural'` on the headset; with `?emu=…` it becomes
  `'points'` unless `?avatar=neural` is given explicitly (`parseExtraParams().avatarExplicit`).
* **Neural avatar** (`app.neural = { driver, ready, info, poses, state, notice, fallback, loadMs }`):
  `_neuralEnsure()` creates the `NeuralAvatarDriver` once (at boot when `avatar === 'neural'`, or
  from the settings button) with `config.NET` parameters; the models load while the player is in
  the menu. `_neuralSync()` starts/stops feeding by state (`NEURAL_STATES` = CALIBRATE,
  COUNTDOWN, PLAYING, RESULTS, RECORDING, RECORD_REVIEW); `_frame` feeds
  `driver.update(app.playerS, time / 1000)` (wall time keeps the 30 Hz resampling monotonic in
  `speed ≠ 1` playback). On `ready` the driver's `{joints, parents, boneLengths}` become the body
  skeleton (`setBodyAvatarPoseProvider(…, {unscaled: true})` + `mirror.setSkeleton`). Guard
  fallback (`disabled`), load errors and missing Worker support set `neural.fallback` (reason
  string), show `hudNeuralDisabled` / `errorModelLoad` and leave the mirror on points.
  `app.on('neural', {state, notice, inferenceMs, inferEvery})`. The settings button "Telemetrie
  exportieren" calls `driver.downloadTelemetry()` (false → "Keine Telemetrie" notice).
* **Player mirror** (`app.mirror`, `PlayerMirror` under `stage` at S `(−duoSideOffset, 0,
  −teacherDistance)`): shown in `MIRROR_STATES` (COUNTDOWN, PLAYING, RESULTS, RECORDING,
  RECORD_REVIEW) unless `avatar === 'off'`; body when a neural pose exists, otherwise the three
  points. The self body (`playerBody`) is only visible when not presenting in XR (desktop /
  `?cam=third`) and has no head sphere.
* **Server detection** (`_probeServer()` at boot, before the choreo list): `?server=` wins
  (`app.serverUrl` as given, `app.serverHttp` = REST base with `ws→http`); otherwise
  `GET api/info` next to `index.html` (the server must answer with `serverVersion`) makes the
  page's own origin the server (`app.serverInfo`). Every REST call (`/api/choreos`, `/api/results`,
  `/api/runs`, `Recorder.upload`) and the mode context (`ctx.server`, `ctx.httpBase`) use these.
  `loadChoreoList` loads server entries from `entry.url` (uploads replace a shipped id).
  `sw.js` never caches `/api/info` and `/api/health`, so the probe fails honestly offline.
* **Mode interface additions**: `mode.startOptions(ctx) → {startAt?}` (merged into
  `PlaySession.start`, which now forwards `startAt` to `BeatClock.start`); `onFinished` may
  return `{lines, moveLines, buttons}` where lines can be functions (re-evaluated on
  `menu.refresh('results')`) and `buttons` are menu button items added as a row on the results
  panel; a thrown error with `err.notice` in `prepare` shows that German text instead of
  `errorChoreoLoad`. `playAgain()` of the online mode returns to the duo lobby. `app.lastBenchmark`
  holds the last duo benchmark; `_storeResult` stores a compact `benchmark {mode, winner, pair,
  partner}` with the result (localStorage + `/api/results`, the server keeps unknown fields).
* Duo modes are registered at boot (`registerDuoModes(app)`); the duo panel's "Online (WLAN)"
  button (`_onlineButton`) shows the lobby panel and connects when a server is known.

### `src/render/mirror.js` — `PlayerMirror`, `MIRROR_MODES`

`new PlayerMirror({color, opacity=0.85, skeleton?, mode='points'})` (Group): `setSkeleton({joints,
parents, boneLengths?})` builds the `BodyAvatar`; `setMode('body'|'points'|'off')`;
`setPointSample(sampleS)` and `setBodyPose(Float32Array(jointCount·3))` mirror the player's
stage‑frame data through the plane halfway to the stage: in the group's frame `p' = (x, y, −z)`
(hands keep their side, the image moves to the player's right when the player does) and the head
quaternion `q' = (−q.z, q.w, −q.x, q.y)` (= reflection `M R M`, `M = diag(1,1,−1)`, followed by a
180° yaw so the nose points at the player). `usesBody`, `hasPose`, `setColor`, `setOpacity`,
`dispose`. No allocation in the setters.

### `src/duo/modes.js` — `createGhostMode(app)`, `createOnlineMode(app, {WebSocketImpl?})`, `registerDuoModes(app)`, `downloadText`, `START_DELAY_MS = 8000`, `ROOM_ALPHABET`

* **ghost**: `prepare` → `loadGhost({choreoId, storage, baseUrl: httpBase, strategy: 'best'})`
  (server first, then `dp.runs`; none → `err.notice = duoNoRuns`); `onSessionCreated` adds a
  translucent `PointAvatar` (`AVATAR_COLORS.ghost`) at `(duoSideOffset, 0, −teacherDistance)`, a
  HUD panel `'duo'` (`{width 1.0, height 0.4, position [1.6, 1.84, 0.02]}`, "Du 87 / Aufzeichnung
  91") and scores the ghost **live** with a second `Scorer` on the session's `'tick'` events
  (`mode.scorer.currentScore()`); `onFinished` finalizes the ghost's Result and builds
  `buildBenchmark(result, ghostResult, app.lastRun, ghost.run, {mode: 'ghost'})` → lines
  (partner score/stars/combo, winner, `Synchronität: Abstand x cm · Versatz ±n ms`), a per‑move
  "A / B" line and the export buttons (`dp-benchmark-<choreo>-<mode>-<time>.json|csv` via
  `downloadText`, notice `duoExported`).
* **online**: `init` adds the lobby panel `'duo-online'` (shown alongside `'duo'` in
  `DUO_LOBBY`; status, room code, players, "Raum erstellen", code entry — `window.prompt` on the
  desktop, four letter steppers over `ROOM_ALPHABET` in VR — "Raum beitreten", "Als Host
  starten", "Raum verlassen"; footer server + RTT) and connects on the `'lobby'` event
  (`ensureConnected()` = `OnlineDuo.connect(serverUrl)` + `syncClock()`). `createRoom()`,
  `joinRoom(code)`, `leaveRoom()`, `startAsHost(choreoId, {delayMs})` (re‑syncs the clock, then
  `online.start`). The server's `start` broadcast (`_onStart`) calls `app.startMode('online',
  choreoId)` on **both** headsets; `startOptions(ctx)` returns `{startAt: ctx.clock.sourceNow() +
  (startAtLocalMs − Date.now()) / 1000}` so both count‑ins begin at the same wall time (a longer
  count‑in is shown while waiting). Per tick: `sendStateFromSample` (15 Hz) and the partner's
  interpolated sample is recorded into a 30 Hz run (`mode.remote` → `mode.remoteRun` as
  `dancing-points-run/1`, hold‑filled); per frame the partner `PointAvatar`
  (`AVATAR_COLORS.opponent`) and the HUD panel (`online.remoteScore`). `onFinished` sends the
  result (`sendResult`); when the partner's `result` message has arrived (before or after) the
  benchmark is built (`mode: 'online'`) and the results panel refreshed. `exit` removes avatar,
  panel and listeners but keeps the connection (the room survives for the next round);
  `dispose()` closes it. Properties for tests: `online` (the `OnlineDuo`), `status`, `roomCode`,
  `isHost`, `startInfo`, `remoteResult`, `benchmark`, `lastResult`.
* New `texts.js` keys (de/en): `duoLeaveRoom, duoConnecting, duoConnected, duoEnterCode, duoHost,
  duoWaitingResult, duoRemoteMissing, duoGhostLoaded, duoExported, duoExportFailed,
  duoStartFailed, duoConnectFailed, duoOnlyHost, duoStartingSoon, duoSyncDistance`.

### Shipped full‑body teacher for the procedural dances

`webxr/choreos/snoop-cwalk.json` and `tutorial-basics.json` now carry `fullBody` from
`tools/precompute_teacher.py --models webxr/models/free --net-scale 1.1553` (int8 tracking net,
3.5 s / 2.7 s runtime; the net scale is `referenceHeight / standingHeadHeight` of the dataset
character, as for the mocap demo). Plausibility (numpy): no NaN, bone lengths ≤ 1.05 × (scaled)
skeleton, root within ±0.42 m, feet 2–27 cm above the floor, joint speeds ≤ 3.4 m/s; head error vs
the 3‑point reference 7.2 cm / 6.1 cm mean (max 22 / 12 cm), wrists ~11 cm mean — comparable
with the mocap demo's 9 cm, so the `TeacherAvatar` renders a body for all three shipped dances.
`tools/gen_choreos.js` keeps an existing file's `fullBody` + `meta.fullBodySource/fullBodyParams`
when regenerating in place (`carryOverFullBody`, frame count must match; `stripFullBody` for
comparisons); `tests/unit/choreos-files.test.js` compares the regenerated three‑point content
(`serializeChoreo(stripFullBody(committed)) === fresh`) and validates the body block. File sizes:
629 kB / 471 kB (limit in the test 1 MB).

### Tests, scripts and CI

* Root `package.json`: `test` = `node --test tests/unit/*.test.js` (Node 22 rejects a bare
  directory), `test:e2e` = `node --test tests/e2e/*.test.js`, `test:all`, `start`, `start:http`,
  `gen:choreos`; devDependency `playwright-core@1.63.0`; `postinstall` installs `server/`.
* `tests/e2e/smoke.test.js`: starts `server/server.js --http --http-port 0 --no-https --data
  tests/.tmp/e2e-data-*` (READY line), launches the Chromium from `PLAYWRIGHT_CHROMIUM`,
  `/opt/pw-browsers/chromium-*/chrome-linux/chrome` or `chromium.executablePath()` (skips with a
  message otherwise) with SwiftShader + no background throttling; one shared context for (a)–(h)
  (localStorage carries results/runs between tests), two small separate contexts for the online
  duo (i). Assertions: (a) boot, 3 shipped dances, same‑origin server detected, `avatar ==
  'points'`; (b) snoop ≥ 90, result + run local and on the server, results panels, "Menü" →
  main menu; (c) noise 0.35 ≤ 70; (d) `avatar=neural` ≥ 1 finite pose with the mirror body or a
  fallback reason; (e) tutorial ≥ 90; (g) record → save → upload → play; (h) ghost benchmark
  (`syncDistance < 0.05`, `|syncLag| < 0.1`, export buttons); (i) room code, join, host start,
  both RESULTS ≥ 90, partner scores/results/frames, `syncDistanceAligned < 0.25`.
* `.github/workflows/ci.yml`: unit job (Node 22, `npm ci`, `npm test`, `py_compile` of the tools)
  and e2e job (`npx playwright-core install --with-deps chromium`, `npm run test:e2e`).
  `.github/workflows/pages.yml`: `upload-pages-artifact` of `webxr/` + `deploy-pages` on push to
  `main` / manual dispatch. `.gitignore` adds `node_modules/`, `server/data/`, `tests/.tmp/`,
  `.playwright/`.

Addendum (integration): the teacher's representation is no longer coupled to the avatar setting —
`TeacherAvatar` is always in `'auto'` mode (full body whenever the choreography carries
`fullBody`, otherwise points); the settings entry "Avatar: Neural / Punkte / Aus" only selects
how the *player* is shown (neural body, three points, or no mirror). Reason: the shipped dances
now all have a body, the body is one instanced draw call, and the emu default of `'points'`
would otherwise hide the teacher's body on the desktop.

## Appendix: fixer (review findings applied)

Lane *fixer*: the reviewers' findings were applied across the presentation, duo, net, server and
docs lanes; this appendix lists the interface changes (everything else in the appendices above
still holds). Regression tests: `tests/unit/fixes.test.js`, the `hardening` test in
`tests/unit/server.test.js`; the e2e suite is unchanged and green.

### Versions and the service worker

* `APP_VERSION` (`src/config.js`) and `SW_VERSION` (`sw.js`) and `SERVER_VERSION` are `0.1.1` and
  **must be bumped together** (`fixes.test.js` compares them). `sw.js` is now network‑first for the
  shell (conditional requests, offline fallback) and cache‑first only for `vendor/` and
  `models/*.onnx` in the unversioned `dp-assets` cache (see the presentation appendix). The App
  calls `registration.update()` at boot and reloads once after a `controllerchange` when it is in
  the menu (`app._swReload` defers it during a run). A failed registration (self‑signed
  certificate: Chromium refuses the script) is shown on the start overlay (`app.swError`,
  German notice `swFailed`) instead of only `console.warn`.
* `server.js`: `src/` and `choreos/` are `no-cache`; `Access-Control-Allow-Origin: *` on all
  responses; `--token`; `sanitizeResult` (exported); 413 for over‑limit chunked bodies;
  `GET /api/info.auth`.

### App (`src/app.js`)

* Boot: `_probeXR()` runs before the network probes (the VR button never waits for a server);
  `_fetch(url, init, ms = FLOW.networkTimeoutMs)` wraps every REST call with
  `AbortSignal.timeout`; `loadChoreoList()` publishes shipped + own entries first (`_publishChoreos`)
  and merges the server list when it arrives; `loadChoreoRef(id)` falls back to
  `${serverHttp}/api/choreos/<id>` for an id that is not in the list (a partner's upload).
  `openDuoLobby()` refreshes the list in the background.
* Calibration: `_calibrationStillValid()` additionally requires the head yaw within ~30°
  (`CALIBRATION_MAX_YAW_DIFF`) of the stored yaw; a released trigger cancels the partial capture
  (`Calibration.cancelCapture()`, new; `captureCount` getter) instead of installing it; a reused
  calibration waits `FLOW.calibrationReuseWaitSec` (2 s, progress bar + "Bereit?" notice) before
  the count‑in, and a trigger hold during that wait recalibrates. In XR the trigger activates the
  calibrate panel's buttons when the laser hovers one (`menu.hovering`), otherwise it is the hold.
* Quitting in XR: `_updateQuitHold(time)` runs in `_frame` for CALIBRATE, COUNTDOWN, PLAYING and
  RECORDING: hold B/Y for `FLOW.quitHoldSec` (2.5 s) with a countdown notice; release resets.
* Recentre / visibility / GPU: `sceneKit.onReferenceSpaceReset` → `_onRecenter()` (clears the
  calibration, resets the stage, aborts a run, notice `recentered`, event `'recenter'`);
  `sceneKit.onVisibilityChange` → `_onXRVisibility(state)` pauses/resumes the `PlaySession`
  (+ `AudioEngine.pauseTrack()/resumeTrack()`, new) while the system menu is open / the headset
  is off; `onContextRestored` → `_reload()`.
* Results: `startSolo` resets `app.lastBenchmark`; `_storeResult` uploads immediately except for
  the online duo, where the POST waits (≤ `RESULT_POST_DELAY_MS` = 10 s) for the benchmark;
  **`app.attachBenchmark(benchmark, result?)`** (new, called by the online mode) updates the stored
  local copy and releases the pending upload. `highscores()` de‑duplicates local + server copies
  (`mergeResults`; only server‑only entries carry `fromServer`); `_fetchServerHighscores` asks
  for `limit=1000`. Uploads send `X-Dp-Token` when `settings.apiToken` is set (`?token=` once).
* Neural avatar: progress notices while the models download (`neuralProgress`); a guard verdict
  is persisted in `settings.neuralVerdict = {disabled, reason, inferenceMs, at, version}` and
  `_neuralEnsure()` skips the driver on later boots until "Neural" is chosen in the settings
  again (or `?avatar=neural`); `_frame` feeds `driver.reportFrameMs(dt·1000)` while presenting.
* `Menu`: `visible` is `false` after `hideAll()` (no laser during a dance) and `true` after
  `show()/showAlso()`; `press()` ignores presses within `pressLockMs` (250 ms) of `show()`;
  `hovering` getter. HUD/menu textures use mipmaps; the notice panel sits at `NOTICE_Y = 0.95` m.

### Input, driver, duo

* `XRInput`: `isPressed('trigger'|'squeeze')` uses the gamepad's digital `pressed` flag when
  reported, otherwise a Schmitt trigger (`PRESS_ON = 0.6`, `PRESS_OFF = 0.4`); button state
  gains `triggerDown, squeezeDown, triggerPressed, squeezePressed`.
* `NeuralAvatarDriver`: `defaultThreads(nav?, isolated?)` and `isMobileXR(nav?)` are exported;
  the count is capped at `MOBILE_MAX_THREADS = 2` on mobile user agents or ≤ 8 cores.
  `reportFrameMs(ms)` (render‑loop guard: median frame time per `guard.frameWindow` = 90 frames,
  > `frameSlowMs` 15 → 15 Hz, > `frameDisableMs` 22 twice or while already slow → disabled);
  `stats.frameMs`, `stats.frameSlowWindows`.
* `OnlineDuo._pushRemote` rejects states with non‑finite numbers and clamps `score` to 0..100,
  `combo` to 0..1e6.
* Online mode (`src/duo/modes.js`): the partner is recorded at the sample's own time
  (`r.t`, i.e. `t − interpolationDelaySec`), so `syncLag`/`syncDistance` no longer include the
  100 ms render delay; `startInfo` survives `exit()` (rematch from the results screen keeps the
  synchronised start; `startOptions` ignores a broadcast older than `START_INFO_MAX_AGE_MS`);
  `partnerGone` (`'left' | 'disconnected' | 'timeout' | 'abort'`) finishes the round without the
  partner (`RESULT_TIMEOUT_MS` = 10 s after the own result, `peer left`, `disconnected`, or a
  relayed `{type:'abort', reason}` which a guest sends when it cannot start the dance); the
  partner avatar is hidden when `age > PARTNER_STALE_SEC` (2 s); `reconnect()` + a "Neu verbinden"
  button in the lobby (one automatic retry after `RECONNECT_DELAY_MS` while in DUO_LOBBY);
  `ensureConnected()` disposes a dead client before creating a new one; an unknown `choreoId`
  in `start`/`choreo` triggers `app.loadChoreoList()` first. New texts: `duoReconnect,
  duoPartnerGone, duoPartnerAborted, duoPartnerStartFailed`.

### Documentation and deployment

* DESIGN §1: no USB‑stick/`file://` mode; offline start / model cache / PWA need a trusted
  certificate (`server/README.md` describes Let's Encrypt DNS‑01 and a private CA).
* Root `package.json` `engines.node >= 20` (playwright‑core); the server alone runs on 18.
* `.github/workflows/pages.yml`: `configure-pages` with `enablement: true` + a comment on the
  one‑time Pages source setting.
