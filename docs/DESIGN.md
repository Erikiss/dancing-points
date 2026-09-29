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
| Extendable for online/WLAN operation, own songs, own TikTok dances | Choreographies are JSON files (§5). A small Node server (`server/`) serves the app over HTTPS in the LAN, stores uploaded choreographies/results and relays duo‑mode state over WebSocket. The app also works fully offline from GitHub Pages / a USB stick. |
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
  sw.js                     service worker: cache-first for app shell + models, network-first for /api
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

ES modules only, no bundler, no TypeScript. Node ≥ 18 for tools/tests/server. Python ≥ 3.11 with
`numpy`, `onnx`, `onnxruntime` for `tools/*.py` (no PyTorch needed at runtime).

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
