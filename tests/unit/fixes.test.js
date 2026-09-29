// fixes.test.js - regression tests for the review findings (lane "fixer"): service-worker
// version parity, XR trigger hysteresis, calibration capture cancel, OnlineDuo state
// validation, the online duo's remote recording (partner recorded at the sample's own time,
// startInfo surviving mode.exit), the driver's thread cap + render-loop guard and the server's
// result sanitizer. Node only (no three.js): the online mode is created against a stub App and
// a stubbed render/avatar module is not needed because only the pure parts are exercised.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { APP_VERSION, parseParams, FLOW } from '../../webxr/src/config.js';
import { XRInput, createButtonState, PRESS_ON, PRESS_OFF } from '../../webxr/src/xr/input.js';
import { Calibration } from '../../webxr/src/xr/calibration.js';
import { OnlineDuo } from '../../webxr/src/duo/online.js';
import { syncLagDetails, syncDistanceDetails, mergeResults } from '../../webxr/src/duo/benchmark.js';
import { defaultThreads, isMobileXR, MOBILE_MAX_THREADS, NeuralAvatarDriver } from '../../webxr/src/net/avatar-driver.js';
import { sanitizeResult } from '../../server/server.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

test('sw.js SW_VERSION equals APP_VERSION (bump both on a release, otherwise clients keep the old shell)', () => {
  const sw = fs.readFileSync(path.join(ROOT, 'webxr', 'sw.js'), 'utf8');
  const m = /const SW_VERSION = '([^']+)'/.exec(sw);
  assert.ok(m, 'SW_VERSION constant present');
  assert.equal(m[1], APP_VERSION);
  assert.match(sw, /dp-assets/, 'unversioned asset cache for models/vendor');
  assert.ok(!/cacheFirst\(request, event\)\)[\s\S]*index\.html/.test(sw.split('self.addEventListener(\'fetch\'')[1].split('navigate')[0]), 'navigations are not cache-first');
  const server = fs.readFileSync(path.join(ROOT, 'server', 'server.js'), 'utf8');
  assert.match(server, new RegExp(`SERVER_VERSION = '${APP_VERSION.replace(/\./g, '\\.')}'`));
});

test('config: ?token= is parsed (and validated), new FLOW constants', () => {
  assert.equal(parseParams('').token, null);
  assert.equal(parseParams('?token=abc-123').token, 'abc-123');
  assert.equal(parseParams('?token=bad%20value').token, null);
  assert.ok(FLOW.quitHoldSec >= 2 && FLOW.calibrationReuseWaitSec >= 1 && FLOW.networkTimeoutMs > 0);
});

test('XRInput: analog trigger has hysteresis (no double activations around 0.5), gamepad pressed flag wins', () => {
  const input = new XRInput();
  const gp = { buttons: [{ value: 0, pressed: false }, { value: 0 }, {}, {}, {}, {}], axes: [0, 0, 0, 0] };
  const frame = {
    getViewerPose: () => ({ transform: { position: { x: 0, y: 1.6, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } } }),
    getPose: () => ({ transform: { position: { x: 0.2, y: 1.0, z: -0.3 }, orientation: { x: 0, y: 0, z: 0, w: 1 } } }),
  };
  const session = { inputSources: [{ handedness: 'right', gripSpace: {}, gamepad: gp }] };
  const ref = {};
  const step = (value, pressed) => { gp.buttons[0] = { value, pressed }; input.update(frame, ref, session, 0); return { down: input.isDown('right', 'trigger'), just: input.justPressed('right', 'trigger') }; };
  // the flag is authoritative when present
  assert.deepEqual(step(0.1, false), { down: false, just: false });
  assert.deepEqual(step(0.55, false), { down: false, just: false });
  assert.deepEqual(step(0.55, true), { down: true, just: true });
  assert.deepEqual(step(0.45, true), { down: true, just: false });
  assert.deepEqual(step(0.3, false), { down: false, just: false });
  // analog only (hand tracking / controllers without the flag): Schmitt trigger 0.6 / 0.4
  const analog = (value) => { gp.buttons[0] = { value }; input.update(frame, ref, session, 0); return { down: input.isDown('right', 'trigger'), just: input.justPressed('right', 'trigger') }; };
  assert.deepEqual(analog(0.5), { down: false, just: false });
  assert.deepEqual(analog(PRESS_ON), { down: true, just: true });
  let edges = 0;
  for (const v of [0.55, 0.45, 0.5, 0.58, 0.42, 0.5]) if (analog(v).just) edges++;
  assert.equal(edges, 0, 'bouncing between 0.4 and 0.6 stays pressed');
  assert.deepEqual(analog(PRESS_OFF), { down: false, just: false });
  assert.deepEqual(analog(0.5), { down: false, just: false });
  assert.deepEqual(analog(0.9), { down: true, just: true });
  const st = createButtonState();
  assert.equal(st.triggerDown, false);
});

test('Calibration.cancelCapture drops a partial capture (a trigger tap keeps the stored data)', () => {
  const c = new Calibration({});
  c.set({ h0: 1.80, yaw: 0.5, origin: [1, 0, 1] });
  c.beginCapture();
  assert.equal(c.addSample({ head: { p: [0, 1.55, 0], q: [0, 0, 0, 1], valid: true } }), 1);
  assert.equal(c.captureCount, 1);
  c.cancelCapture();
  assert.equal(c.captureCount, 0);
  assert.equal(c.endCapture(), null, 'nothing to install after a cancel');
  assert.equal(c.h0, 1.80);
  assert.equal(c.yaw, 0.5);
  assert.deepEqual(c.origin, [1, 0, 1]);
});

test('OnlineDuo: relayed states with non-numeric values are rejected, score/combo clamped', () => {
  const duo = new OnlineDuo({ WebSocketImpl: class {}, now: () => 0 });
  duo._onMessage(JSON.stringify({ type: 'state', t: 1, head: ['a', 'b', 'c'], left: [0, 0, 0], right: [0, 0, 0] }));
  assert.equal(duo.remoteStateCount, 0);
  duo._onMessage(JSON.stringify({ type: 'state', t: 1, head: [0, 1, 0, 0, 0, null, 1], left: [0, 0, 0], right: [0, 0, 0] }));
  assert.equal(duo.remoteStateCount, 0);
  duo._onMessage(JSON.stringify({ type: 'state', t: 1, head: [0, 1, 0], left: [0, 0, 0], right: [0, 0, 'x'] }));
  assert.equal(duo.remoteStateCount, 0);
  duo._onMessage(JSON.stringify({ type: 'state', t: 1, head: [0, 1.6, 0, 0, 0, 0, 1], left: [0, 1, 0], right: [0, 1, 0], score: 1e9, combo: -5 }));
  assert.equal(duo.remoteStateCount, 1);
  assert.equal(duo.remoteScore, 100);
  assert.equal(duo.remoteCombo, 0);
  const s = duo.getRemoteSample(2);
  assert.ok(Number.isFinite(s.head.p[1]) && Math.abs(s.head.p[1] - 1.6) < 1e-6);
});

/** Simulated online duo: partner states at 15 Hz (+ latency), local ticks at 30 Hz. */
function simulateRemoteRecording(recordAtSampleTime) {
  const fps = 30, dur = 20, n = Math.round(dur * fps) + 1;
  const traj = (t) => ({
    head: [0.15 * Math.sin(2 * Math.PI * 0.75 * t), 1.6 + 0.05 * Math.sin(2 * Math.PI * 1.5 * t), 0.1 * Math.cos(2 * Math.PI * 0.4 * t), 0, 0, 0, 1],
    left: [-0.3, 1.0 + 0.2 * Math.sin(2 * Math.PI * 0.75 * t), -0.2],
    right: [0.3, 1.0 + 0.2 * Math.cos(2 * Math.PI * 0.75 * t), -0.2],
  });
  let wall = 0;
  const duo = new OnlineDuo({ WebSocketImpl: class {}, now: () => wall });
  const A = { format: 'dancing-points-run/1', choreoId: 'x', fps, frameCount: n, head: [], left: [], right: [] };
  for (let i = 0; i < n; i++) { const s = traj(i / fps); A.head.push(s.head); A.left.push(s.left); A.right.push(s.right); }
  const rem = { head: new Float32Array(n * 7), left: new Float32Array(n * 3), right: new Float32Array(n * 3), filled: new Uint8Array(n) };
  const latency = 0.02;
  let nextSend = 0;
  for (let i = 0; i < n; i++) {
    const t = i / fps; wall = t * 1000;
    while (nextSend + latency <= t + 1e-9) { const s = traj(nextSend); duo._pushRemote({ type: 'state', t: nextSend, ...s, score: 0, combo: 0 }); nextSend += 1 / 15; }
    const r = duo.getRemoteSample(t);
    if (!r || duo.remoteStateCount === 0) continue;
    // modes.js _recordRemote(t, r): index by t
    let j = Math.round((recordAtSampleTime ? r.t : t) * fps);
    if (j < 0) j = 0; if (j >= n) j = n - 1;
    const o7 = j * 7, o3 = j * 3;
    rem.head[o7] = r.head.p[0]; rem.head[o7 + 1] = r.head.p[1]; rem.head[o7 + 2] = r.head.p[2]; rem.head[o7 + 6] = 1;
    rem.left[o3] = r.left.p[0]; rem.left[o3 + 1] = r.left.p[1]; rem.left[o3 + 2] = r.left.p[2];
    rem.right[o3] = r.right.p[0]; rem.right[o3 + 1] = r.right.p[1]; rem.right[o3 + 2] = r.right.p[2];
    rem.filled[j] = 1;
  }
  const B = { format: 'dancing-points-run/1', choreoId: 'x', fps, frameCount: n, head: [], left: [], right: [] };
  let src = 0; while (src < n && !rem.filled[src]) src++;
  for (let i = 0; i < n; i++) { if (rem.filled[i]) src = i; B.head.push(Array.from(rem.head.subarray(src * 7, src * 7 + 7))); B.left.push(Array.from(rem.left.subarray(src * 3, src * 3 + 3))); B.right.push(Array.from(rem.right.subarray(src * 3, src * 3 + 3))); }
  const lag = syncLagDetails(A, B, { maxLagSec: 1.0, fps });
  return { lag: lag.lag, distance: syncDistanceDetails(A, B, {}).mean };
}

test('online duo: recording the partner at the sample time removes the interpolation delay from the benchmark', async () => {
  const wrong = simulateRemoteRecording(false);
  const right = simulateRemoteRecording(true);
  assert.ok(Math.abs(wrong.lag - 0.1) < 0.02, `indexing by tick time shows the 100 ms delay (${wrong.lag})`);
  assert.ok(Math.abs(right.lag) < 0.01, `indexing by sample time: no lag (${right.lag})`);
  assert.ok(right.distance < 0.005, `raw distance a few mm (${right.distance})`);
  // and modes.js really records at the sample's time
  const src = fs.readFileSync(path.join(ROOT, 'webxr', 'src', 'duo', 'modes.js'), 'utf8');
  assert.match(src, /m\._recordRemote\(r\.t, r\)/);
});

test('online mode: startInfo survives exit() (rematch from the results screen) and stale starts are ignored', async () => {
  // render/avatar.js imports three; stub the two render modules the mode pulls in
  const { createOnlineMode, START_DELAY_MS, START_INFO_MAX_AGE_MS } = await importModes();
  const listeners = {};
  const app = {
    texts: new Proxy({}, { get: (_, k) => String(k) }), settings: {}, choreos: [{ id: 'tutorial-basics' }], selectedChoreoId: 'tutorial-basics', state: 'RESULTS',
    menu: { addPanel() {}, getPanel() { return null; }, refresh() {} }, hud: { showNotice() {}, addPanel() { return { draw() {} }; }, removePanel() {} },
    on(type, fn) { (listeners[type] ||= []).push(fn); }, startMode: () => Promise.resolve(null), win: null, inputKind: 'xr', loadChoreoList: () => Promise.resolve([]),
  };
  const mode = createOnlineMode(app, { WebSocketImpl: class {} });
  const clock = { sourceNow: () => 100 };
  mode._onStart({ choreoId: 'tutorial-basics', startAt: 0, startAtLocalMs: Date.now() + START_DELAY_MS, delayMs: START_DELAY_MS });
  const before = mode.startOptions({ clock });
  assert.ok(before && Math.abs(before.startAt - (100 + START_DELAY_MS / 1000)) < 0.1);
  mode.exit({ stage: { remove() {} }, hud: app.hud });
  const after = mode.startOptions({ clock });
  assert.ok(after && Math.abs(after.startAt - before.startAt) < 0.1, 'exit() keeps the synchronised start');
  mode.startInfo = { startAtLocalMs: Date.now() - START_INFO_MAX_AGE_MS - 1000 };
  assert.equal(mode.startOptions({ clock }), null, 'a broadcast from an earlier round is not reused');
  // partner timeout / abort finish the round without the partner
  mode.lastResult = { choreoId: 'tutorial-basics', score: 90, stars: 5, maxCombo: 3, timingBias: 0, moves: [], perBeat: new Float32Array(0), player: 'A', mode: 'online', playedAt: new Date().toISOString() };
  assert.equal(mode.benchmark, null);
  mode._partnerGone('timeout');
  assert.ok(mode.benchmark && mode.benchmark.winner === 'A', 'benchmark built with a missing partner result');
  assert.equal(mode.partnerGone, 'timeout');
  await new Promise((r) => setTimeout(r, 10));
  mode.dispose();
});

async function importModes() {
  // modes.js -> render/avatar.js + render/hud.js import three.js; resolve them through a stub
  // module map by registering a loader hook is heavier than needed: use the stubbed copy
  const stubDir = path.join(ROOT, 'tests', '.tmp', 'modes-stub');
  fs.mkdirSync(path.join(stubDir, 'render'), { recursive: true });
  fs.mkdirSync(path.join(stubDir, 'duo'), { recursive: true });
  fs.writeFileSync(path.join(stubDir, 'render', 'avatar.js'), 'export class PointAvatar { constructor() { this.position = { set() {} }; this.visible = true; } setSample() {} dispose() {} }\nexport const AVATAR_COLORS = { ghost: 1, opponent: 2 };\n');
  fs.writeFileSync(path.join(stubDir, 'render', 'hud.js'), 'export function fitText() {}\nexport function starsString(n) { return String(n); }\nexport const HUD_COLORS = { text: "#fff" };\n');
  const src = fs.readFileSync(path.join(ROOT, 'webxr', 'src', 'duo', 'modes.js'), 'utf8')
    .replace("from '../config.js'", `from '${path.join(ROOT, 'webxr', 'src', 'config.js')}'`)
    .replace("from '../game/scoring.js'", `from '${path.join(ROOT, 'webxr', 'src', 'game', 'scoring.js')}'`)
    .replace("from './ghost.js'", `from '${path.join(ROOT, 'webxr', 'src', 'duo', 'ghost.js')}'`)
    .replace("from './online.js'", `from '${path.join(ROOT, 'webxr', 'src', 'duo', 'online.js')}'`)
    .replace("from './benchmark.js'", `from '${path.join(ROOT, 'webxr', 'src', 'duo', 'benchmark.js')}'`);
  const file = path.join(stubDir, 'duo', 'modes.js');
  fs.writeFileSync(file, src);
  return import(`${file}?t=${Date.now()}`);
}

test('NeuralAvatarDriver: thread cap on mobile / small machines, render-loop guard', () => {
  assert.equal(defaultThreads({ hardwareConcurrency: 8, userAgent: 'Mozilla/5.0 (X11; Linux x86_64; Quest 2) OculusBrowser/30' }, true), MOBILE_MAX_THREADS);
  assert.equal(defaultThreads({ hardwareConcurrency: 8, userAgent: 'desktop' }, true), 2, '<= 8 cores: 2 threads');
  assert.equal(defaultThreads({ hardwareConcurrency: 16, userAgent: 'desktop' }, true), 4);
  assert.equal(defaultThreads({ hardwareConcurrency: 16, userAgent: 'Quest' }, true), 2);
  assert.equal(defaultThreads({ hardwareConcurrency: 16, userAgent: 'desktop' }, false), 1, 'no SharedArrayBuffer -> 1');
  assert.equal(isMobileXR({ userAgent: 'OculusBrowser' }), true);
  assert.equal(isMobileXR({ userAgent: 'Chrome' }), false);
  // render-loop guard: fed the frame interval while presenting
  const posted = [];
  class FakeWorker { constructor() { this.onmessage = null; } postMessage(m) { posted.push(m); if (m.type === 'init') setTimeout(() => this.onmessage && this.onmessage({ data: { type: 'ready', joints: [], parents: [], boneLengths: [] } }), 0); } terminate() {} }
  const states = [];
  const driver = new NeuralAvatarDriver({ WorkerClass: FakeWorker, onStatus: (s) => states.push(s.state), guard: { frameWindow: 8, frameSlowMs: 15, frameDisableMs: 22 } });
  return driver.init().then(() => {
    driver.start();
    assert.equal(driver.state, 'running');
    for (let i = 0; i < 8; i++) driver.reportFrameMs(13.9);
    assert.equal(driver.inferEvery, 1, '72 Hz keeps 30 Hz inference');
    for (let i = 0; i < 8; i++) driver.reportFrameMs(18);
    assert.equal(driver.inferEvery, 2, 'a slow window -> 15 Hz');
    assert.equal(driver.state, 'slow');
    for (let i = 0; i < 8; i++) driver.reportFrameMs(30);
    assert.equal(driver.state, 'disabled', 'judder while already at 15 Hz -> disabled');
    assert.ok(posted.some((m) => m.type === 'config' && m.inferEvery === 2));
    driver.dispose();
  });
});

test('server sanitizeResult keeps only the known fields with caps', () => {
  const r = sanitizeResult({ choreoId: 'a', score: 5, junk: 1, player: 'p'.repeat(100), moves: new Array(300).fill({ name: 'n', score: 1, grade: 'ok', x: 1 }), perBeat: new Array(9000).fill(0.5), benchmark: { mode: 'online', winner: 'B', pair: { syncDistance: 0.1 }, partner: { name: 'x', score: 3 }, extra: 'no' } });
  assert.equal(r.junk, undefined);
  assert.equal(r.player.length, 64);
  assert.equal(r.moves.length, 256);
  assert.deepEqual(r.moves[0], { name: 'n', score: 1, grade: 'ok' });
  assert.equal(r.perBeat.length, 8192);
  assert.deepEqual(r.benchmark, { mode: 'online', winner: 'B', pair: { syncDistance: 0.1 }, partner: { name: 'x', score: 3 } });
  assert.equal(sanitizeResult({ choreoId: 'a', score: 5, benchmark: 'nope' }).benchmark, undefined);
});

test('highscores: local + server copies of one play count once (mergeResults)', () => {
  const local = [{ choreoId: 'a', score: 90, player: 'E', playedAt: '2026-01-01T10:00:00.000Z', mode: 'solo' }];
  const server = local.map((r) => ({ ...r, id: 'r1', receivedAt: 'x', fromServer: true }));
  server.push({ choreoId: 'a', score: 70, player: 'F', playedAt: '2026-01-01T11:00:00.000Z', mode: 'solo', fromServer: true });
  const merged = mergeResults(local, server);
  assert.equal(merged.length, 2);
  assert.equal(merged[0].fromServer, undefined, 'the local copy wins');
  assert.equal(merged[1].fromServer, true, 'server-only entries keep the tag');
});
