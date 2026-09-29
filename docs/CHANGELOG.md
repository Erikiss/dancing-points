# Changelog – Dancing Points VR

## 0.1.1 (2026-09-29)

Review fixes after the first integrated build (36 findings from three independent reviews:
correctness, Quest 2 performance/UX, server/security/operations):

- Service worker: network-first app shell with an unversioned asset cache (models, vendor
  libraries survive version bumps); `SW_VERSION` must match `APP_VERSION` (unit test); clear notice
  when the service worker cannot register (self-signed certificate) instead of silent failure.
- App: network probes with timeouts (boot never blocks on a hanging server); calibration can be
  cancelled in VR and a partial capture is discarded; recordings can be aborted; the quit gesture
  requires a 2 s hold with visible progress; WebXR reference-space `reset` (recenter) invalidates
  the persisted calibration yaw; WebGL context loss and XR visibility changes are handled; HUD
  textures use mipmaps; leaderboard merges local and server results without duplicates and keeps
  all-time bests; model download shows progress.
- Neural avatar: wasm thread cap on mobile XR, render-loop aware guard, driver disposes cleanly.
- Duo: partner recorded at the sample's own time (correct lag/distance), rematch keeps the
  synchronised start, reconnect with partner timeout, numeric validation of relayed state.
- Server: optional `--token` for mutating API calls, CORS for static assets (GitHub Pages app +
  LAN server), `--cert/--key` for trusted certificates, result sanitiser, 413 instead of socket
  resets on oversized bodies; README documents the certificate limitations.
- Docs: German operator README, native port notes, this changelog.

## 0.1.0 (2026-09-29)

First integrated build of the WebXR dance game for Meta Quest 2:

- three.js WebXR app without build step (menu, calibration, count-in, live scoring, results,
  recorder, settings, PWA manifest + service worker).
- Choreographies: Snoop Dogg C-Walk (6 combos, procedural v0), tutorial, mocap demo; JSON format
  `dancing-points-choreo/1` with procedural generator `tools/gen_choreos.js`.
- Scoring engine (3-point features, ±0.2 s window, energy gate, grades, stars, combo, timing bias).
- Neural mirror avatar: JS port of the Dancing Points runtime pipeline (fixture parity < 1e-6
  against a numpy reference validated on the dataset), onnxruntime-web worker, int8 models for the
  free style (35 MB + 51 MB), performance guard with fallback.
- Duo modes (ghost replay, online with room codes and clock sync), benchmark metrics with
  JSON/CSV export, leaderboard.
- Node LAN server (HTTPS with generated certificate, REST for choreos/results/runs, WebSocket
  relay, COOP/COEP headers).
- Tests: 100+ unit tests (node --test), Playwright e2e smoke suite, Python pipeline validation and
  fixture generation; GitHub Actions CI and Pages deployment.
