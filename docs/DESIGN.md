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
