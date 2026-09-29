// smoke.test.js - end-to-end smoke test of the WebXR app in headless Chromium (DESIGN.md
// section 10). Starts server/server.js (HTTP only, free port, throw-away data dir) so that the
// same-origin REST API (results, runs, choreos) and the duo WebSocket relay are exercised too,
// then drives the app through its emu modes with playwright-core:
//   (a) index.html boots in desktop emu mode without console errors, same-origin server detected
//   (b) exact playback of snoop-cwalk reaches RESULTS with score >= 90 (result + run persisted
//       locally and on the server); (f) the results menu is shown and leads back to the main menu
//   (c) noise=0.35 scores <= 70
//   (d) avatar=neural: the inference worker produces >= 1 pose (mirror shows the body) or reports
//       a documented fallback reason
//   (e) tutorial-basics completes
//   (g) record flow: record -> save -> own dance listed + uploaded -> playable
//   (h) ghost duo against the best saved run (benchmark on the results screen)
//   (i) online duo: two pages in one room, synchronised start, both results + benchmark
// Skips (with a clear message) when playwright-core or a Chromium binary is missing:
// env PLAYWRIGHT_CHROMIUM=<chrome>, /opt/pw-browsers/chromium-*/chrome-linux/chrome, or the
// browser installed by `npx playwright-core install chromium`.
// Run: npm run test:e2e   (node --test tests/e2e/*.test.js)

import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const TMP = path.join(ROOT, 'tests', '.tmp');
const TIMEOUT = 240000;

async function loadPlaywright() {
  try {
    return await import('playwright-core');
  } catch (e) {
    return null;
  }
}

function findChromium(pw) {
  if (process.env.PLAYWRIGHT_CHROMIUM && fs.existsSync(process.env.PLAYWRIGHT_CHROMIUM)) return process.env.PLAYWRIGHT_CHROMIUM;
  try {
    const base = '/opt/pw-browsers';
    const dirs = fs.readdirSync(base).filter((d) => /^chromium-\d+$/.test(d)).sort().reverse();
    for (const d of dirs) {
      const p = path.join(base, d, 'chrome-linux', 'chrome');
      if (fs.existsSync(p)) return p;
    }
  } catch (e) { /* no pre-installed browsers */ }
  try {
    const p = pw.chromium.executablePath();
    if (p && fs.existsSync(p)) return p;
  } catch (e) { /* not installed */ }
  return null;
}

const pw = await loadPlaywright();
const chromiumPath = pw ? findChromium(pw) : null;
const skip = !pw
  ? 'playwright-core is not installed (npm install)'
  : (!chromiumPath ? 'no Chromium found: set PLAYWRIGHT_CHROMIUM or run `npx playwright-core install chromium`' : false);
if (skip) console.log(`# e2e skipped: ${skip}`);

const S = { server: null, port: 0, base: '', browser: null, context: null, dataDir: '' };

function startServer(dataDir) {
  return new Promise((resolve, reject) => {
    const proc = spawn(process.execPath, [path.join(ROOT, 'server', 'server.js'), '--http', '--http-port', '0', '--no-https', '--data', dataDir], { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    let err = '';
    const timer = setTimeout(() => reject(new Error(`server did not report READY:\n${out}\n${err}`)), 30000);
    proc.stdout.on('data', (d) => {
      out += d.toString();
      const m = /READY (\{.*\})/.exec(out);
      if (m) {
        clearTimeout(timer);
        resolve({ proc, info: JSON.parse(m[1]) });
      }
    });
    proc.stderr.on('data', (d) => { err += d.toString(); });
    proc.on('exit', (code) => { clearTimeout(timer); reject(new Error(`server exited with ${code}: ${err}`)); });
  });
}

before(async () => {
  if (skip) return;
  fs.mkdirSync(TMP, { recursive: true });
  S.dataDir = fs.mkdtempSync(path.join(TMP, 'e2e-data-'));
  const { proc, info } = await startServer(S.dataDir);
  S.server = proc;
  S.port = info.http;
  S.base = `http://127.0.0.1:${S.port}`;
  S.browser = await pw.chromium.launch({
    executablePath: chromiumPath,
    headless: true,
    args: [
      '--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader', '--use-gl=swiftshader', '--ignore-gpu-blocklist',
      // two pages must run their render loops at full rate at the same time (online duo)
      '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding',
    ],
  });
  // one context: localStorage (results, runs, own dances) is shared between the tests
  S.context = await S.browser.newContext({ viewport: { width: 1280, height: 720 } });
  console.log(`# e2e: server ${S.base}, chromium ${chromiumPath}`);
});

after(async () => {
  if (S.browser) await S.browser.close().catch(() => null);
  if (S.server) {
    S.server.kill('SIGTERM');
    await new Promise((r) => { S.server.on('exit', r); setTimeout(r, 3000); });
  }
  if (S.dataDir) fs.rmSync(S.dataDir, { recursive: true, force: true });
});

const BENIGN = [/GL Driver Message/, /ReadPixels/, /Automatic fallback to software WebGL/];

async function newPage(context = S.context) {
  const page = await context.newPage();
  const errors = [];
  page.on('console', (m) => {
    if (m.type() === 'error') {
      const t = m.text();
      if (!BENIGN.some((re) => re.test(t))) errors.push(`console.error: ${t}`);
    }
  });
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('requestfailed', (r) => errors.push(`requestfailed: ${r.url()} ${r.failure() ? r.failure().errorText : ''}`));
  return { page, errors };
}

const appState = () => window.__dp && window.__dp.app && window.__dp.app.state;

async function waitState(page, state, timeout = 120000) {
  await page.waitForFunction((s) => window.__dp && window.__dp.app && window.__dp.app.state === s, state, { timeout, polling: 50 });
}

async function resultInfo(page) {
  return page.evaluate(() => {
    const app = window.__dp.app;
    const r = app.lastResult;
    return {
      state: app.state,
      score: r ? r.score : null,
      stars: r ? r.stars : null,
      maxCombo: r ? r.maxCombo : null,
      timingBias: r ? r.timingBias : null,
      mode: r ? r.mode : null,
      moves: r ? r.moves.map((m) => `${m.name}:${Math.round(m.score * 100)}/${m.grade}`) : null,
      choreo: app.choreo ? app.choreo.id : null,
      results: JSON.parse(localStorage.getItem('dp.results') || '[]').length,
      runs: JSON.parse(localStorage.getItem('dp.runs') || '[]').length,
      panels: [...app.menu.panels.entries()].filter(([, p]) => p.visible).map(([id]) => id),
      menuVisible: app.menu.visible,
      activePanel: app.menu.activeId,
      mirrorVisible: app.mirror.visible,
      mirrorBody: app.mirror.usesBody,
      avatar: app.settings.avatar,
      neural: app.neural ? { state: app.neural.state, poses: app.neural.poses, fallback: app.neural.fallback, ready: app.neural.ready } : null,
      serverHttp: app.serverHttp,
      frames: app.frameCount,
    };
  });
}

async function playback(query, { expectScoreMin = null, expectScoreMax = null, label = query } = {}) {
  const { page, errors } = await newPage();
  const t0 = Date.now();
  await page.goto(`${S.base}/?${query}`);
  await waitState(page, 'RESULTS', 150000);
  const info = await resultInfo(page);
  console.log(`# ${label}: ${JSON.stringify({ score: info.score, stars: info.stars, maxCombo: info.maxCombo, timingBias: info.timingBias, moves: info.moves, frames: info.frames })} (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
  assert.equal(info.state, 'RESULTS');
  assert.deepEqual(errors, [], 'no console errors / page errors / failed requests');
  if (expectScoreMin !== null) assert.ok(info.score >= expectScoreMin, `score ${info.score} >= ${expectScoreMin}`);
  if (expectScoreMax !== null) assert.ok(info.score <= expectScoreMax, `score ${info.score} <= ${expectScoreMax}`);
  return { page, errors, info };
}

test('(a) index.html boots in emu mode without console errors and finds the same-origin server', { skip, timeout: TIMEOUT }, async () => {
  const { page, errors } = await newPage();
  await page.goto(`${S.base}/?emu=1`);
  await page.waitForFunction(() => window.__dp && window.__dp.app.state === 'MENU' && window.__dp.app.choreos.length > 0, null, { timeout: 60000 });
  const info = await page.evaluate(() => {
    const app = window.__dp.app;
    return {
      version: window.__dp.version,
      choreos: app.choreos.map((c) => `${c.id}(${c.source})`),
      serverHttp: app.serverHttp,
      serverVersion: app.serverInfo && app.serverInfo.serverVersion,
      avatar: app.settings.avatar,
      modes: [...app.modes.keys()],
      panel: app.menu.activeId,
      mirrorVisible: app.mirror.visible,
      neural: app.neural,
      title: document.title,
    };
  });
  console.log(`# boot: ${JSON.stringify(info)}`);
  assert.deepEqual(errors, []);
  assert.equal(info.title, 'Dancing Points VR');
  assert.ok(info.choreos.length >= 3, 'shipped choreographies listed');
  assert.ok(info.choreos.includes('snoop-cwalk(shipped)') && info.choreos.includes('tutorial-basics(shipped)') && info.choreos.includes('mocap-freestyle(shipped)'));
  assert.equal(info.serverHttp, S.base, 'same-origin /api/info detected');
  assert.equal(info.avatar, 'points', 'emu mode defaults to the point avatar');
  assert.equal(info.neural, null, 'no neural driver without ?avatar=neural in emu mode');
  assert.deepEqual(info.modes.sort(), ['ghost', 'online']);
  assert.equal(info.panel, 'main');
  assert.equal(info.mirrorVisible, false);
  await page.close();
});

test('(b)+(f) exact playback of snoop-cwalk scores >= 90, persists, shows the results menu and returns to the main menu', { skip, timeout: TIMEOUT }, async () => {
  const { page, info } = await playback('emu=playback&choreo=snoop-cwalk&autostart=1&speed=10', { expectScoreMin: 90, label: 'snoop exact' });
  assert.equal(info.choreo, 'snoop-cwalk');
  assert.equal(info.mode, 'solo');
  assert.ok(info.results >= 1 && info.runs >= 1, `result + run persisted locally (${info.results}/${info.runs})`);
  assert.ok(info.menuVisible && info.activePanel === 'results' && info.panels.includes('results-moves'), `results panels visible (${info.panels})`);
  assert.equal(info.mirrorVisible, true, 'the player mirror is shown during/after the dance');
  assert.equal(info.mirrorBody, false, 'point mirror in points mode');
  assert.equal(info.serverHttp, S.base);
  // the server received the result and the run (same-origin API)
  await page.waitForTimeout(500);
  const res = await (await fetch(`${S.base}/api/results?choreoId=snoop-cwalk`)).json();
  const runs = await (await fetch(`${S.base}/api/runs?choreoId=snoop-cwalk`)).json();
  console.log(`# server: ${res.results.length} results, ${runs.runs.length} runs for snoop-cwalk`);
  assert.ok(res.results.length >= 1 && res.results[0].score === info.score, 'result POSTed to /api/results');
  assert.ok(runs.runs.length >= 1, 'run POSTed to /api/runs');
  // (f) the "Menü" button of the results panel leads back to the main menu
  const back = await page.evaluate(() => {
    const app = window.__dp.app;
    const panel = app.menu.getPanel('results');
    const hit = panel.hits.find((h) => h.item && h.item.id === 'menu');
    if (!hit) return { error: 'no menu button' };
    hit.action();
    return { state: app.state, panel: app.menu.activeId, visible: app.menu.visible, session: app.session, mirror: app.mirror.visible };
  });
  console.log(`# after Menü: ${JSON.stringify(back)}`);
  assert.equal(back.state, 'MENU');
  assert.equal(back.panel, 'main');
  assert.equal(back.visible, true);
  assert.equal(back.session, null);
  await page.close();
});

test('(c) playback with noise=0.35 scores <= 70', { skip, timeout: TIMEOUT }, async () => {
  const { page, info } = await playback('emu=playback&choreo=snoop-cwalk&autostart=1&speed=10&noise=0.35', { expectScoreMax: 70, label: 'snoop noise 0.35' });
  assert.ok(info.score < 90);
  await page.close();
});

test('(e) tutorial-basics completes with score >= 90', { skip, timeout: TIMEOUT }, async () => {
  const { page } = await playback('emu=playback&choreo=tutorial-basics&autostart=1&speed=10', { expectScoreMin: 90, label: 'tutorial exact' });
  await page.close();
});

test('(d) avatar=neural: the worker produces poses (mirror body) or reports a fallback reason', { skip, timeout: TIMEOUT }, async () => {
  const { page, errors } = await newPage();
  const t0 = Date.now();
  await page.goto(`${S.base}/?emu=playback&choreo=tutorial-basics&autostart=1&speed=10&avatar=neural`);
  await waitState(page, 'RESULTS', 150000);
  // the worker keeps running on the results screen; wait for poses or a decided fallback
  await page.waitForFunction(() => {
    const n = window.__dp.app.neural;
    return n && (n.poses >= 1 || n.fallback || n.state === 'error' || n.state === 'disabled');
  }, null, { timeout: 120000, polling: 100 });
  await page.waitForTimeout(800);
  const info = await page.evaluate(() => {
    const app = window.__dp.app;
    const n = app.neural;
    const d = n.driver;
    return {
      score: app.lastResult.score,
      avatar: app.settings.avatar,
      state: n.state, poses: n.poses, fallback: n.fallback, ready: n.ready, loadMs: n.loadMs,
      driverState: d ? d.state : null,
      inferenceMs: d ? d.inferenceMs : null,
      inferEvery: d ? d.inferEvery : null,
      stats: d ? { frames: d.stats.frames, poses: d.stats.poses, dropped: d.stats.dropped, meanMs: d.stats.meanMs, maxMs: d.stats.maxMs, minMs: d.stats.minMs } : null,
      info: n.info ? { ortSource: n.info.ortSource, simd: n.info.simd, numThreads: n.info.numThreads, loadMs: n.info.loadMs } : null,
      mirrorBody: app.mirror.usesBody,
      mirrorVisible: app.mirror.visible,
      selfBody: app.playerBody ? app.playerBody.visible : null,
      pose: (() => {
        if (!d || !n.poses) return null;
        const p = d.latest.positionsWorldS;
        let finite = true;
        for (let i = 0; i < p.length; i++) if (!Number.isFinite(p[i])) finite = false;
        return { finite, headY: p[27 * 3 + 1], rootS: Array.from(d.latest.rootS) };
      })(),
    };
  });
  console.log(`# neural: ${JSON.stringify(info)} (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
  assert.deepEqual(errors, []);
  assert.equal(info.avatar, 'neural');
  assert.ok(info.score >= 90, 'scoring unaffected by the neural avatar');
  if (info.poses >= 1) {
    assert.ok(info.pose.finite, 'pose values finite');
    assert.ok(info.pose.headY > 1.0 && info.pose.headY < 2.0, `plausible head height ${info.pose.headY}`);
    assert.equal(info.mirrorBody, true, 'the mirror shows the full body');
    assert.ok(['running', 'slow'].includes(info.driverState), `driver state ${info.driverState}`);
  } else {
    assert.ok(typeof info.fallback === 'string' && info.fallback.length > 0, 'fallback reason documented');
    console.log(`# neural fallback: ${info.fallback}`);
    assert.equal(info.mirrorBody, false, 'point mirror after the fallback');
  }
  await page.close();
});

test('(g) record flow: record -> save -> own dance listed, uploaded and playable', { skip, timeout: TIMEOUT }, async () => {
  const { page, errors } = await newPage();
  await page.goto(`${S.base}/?emu=1&speed=10`);
  await page.waitForFunction(() => window.__dp && window.__dp.app.state === 'MENU' && window.__dp.app.choreos.length > 0, null, { timeout: 60000 });
  await page.evaluate(() => {
    const app = window.__dp.app;
    app.startRecordSetup();
    app._recordCfg.bpm = 120;
    app._recordCfg.bars = 1;
    app._recordCfg.countInBeats = 2;
    app._recordCfg.title = 'E2E Aufnahme';
    app.beginRecording();
  });
  await waitState(page, 'RECORD_REVIEW', 60000);
  const rec = await page.evaluate(() => ({ frames: window.__dp.app._review.json.frames.head.length, id: window.__dp.app._review.json.id }));
  console.log(`# recorded: ${JSON.stringify(rec)}`);
  assert.equal(rec.frames, 61);
  const saved = await page.evaluate(() => window.__dp.app.saveRecording());
  assert.equal(saved, true, 'saveRecording resolved true (uploaded)');
  const own = await page.evaluate(() => window.__dp.app.choreos.filter((c) => c.source === 'own' || c.source === 'server').map((c) => `${c.id}(${c.source})`));
  console.log(`# own/server dances: ${own}`);
  assert.ok(own.some((s) => s.startsWith(rec.id)), 'own dance listed');
  const list = await (await fetch(`${S.base}/api/choreos`)).json();
  assert.ok(list.choreos.some((c) => c.id === rec.id && c.source === 'upload'), 'dance uploaded to /api/choreos');
  await page.evaluate((id) => window.__dp.app.startSolo(id), rec.id);
  await waitState(page, 'RESULTS', 60000);
  assert.deepEqual(errors, []);
  await page.close();
});

test('(h) ghost duo against the best saved run shows both scores and a benchmark', { skip, timeout: TIMEOUT }, async () => {
  const { page, errors } = await newPage();
  await page.goto(`${S.base}/?emu=playback&speed=10`);
  await page.waitForFunction(() => window.__dp && window.__dp.app.state === 'MENU' && window.__dp.app.choreos.length > 0, null, { timeout: 60000 });
  await page.evaluate(() => window.__dp.app.startMode('ghost', 'snoop-cwalk'));
  await waitState(page, 'RESULTS', 150000);
  const info = await page.evaluate(() => {
    const app = window.__dp.app;
    const b = app.lastBenchmark;
    const mode = app.modes.get('ghost');
    return {
      mode: app.lastResult.mode, score: app.lastResult.score,
      ghost: mode.ghost ? { name: mode.ghost.name, score: mode.ghost.score, frames: mode.ghost.frameCount } : null,
      ghostResult: mode.ghostResult ? mode.ghostResult.score : null,
      benchmark: b ? { winner: b.winner, players: b.players.map((p) => `${p.name}:${p.score}`), syncDistance: b.pair.syncDistance, syncDistanceAligned: b.pair.syncDistanceAligned, syncLag: b.pair.syncLag, framesCompared: b.pair.framesCompared } : null,
      buttons: app.menu.getPanel('results').hits.map((h) => h.item && h.item.id).filter(Boolean),
      duoPanel: !!app.hud.getPanel('duo'),
      stored: JSON.parse(localStorage.getItem('dp.results') || '[]').filter((r) => r.mode === 'ghost').map((r) => r.benchmark && r.benchmark.winner),
    };
  });
  console.log(`# ghost: ${JSON.stringify(info)}`);
  assert.deepEqual(errors, []);
  assert.equal(info.mode, 'ghost');
  assert.ok(info.score >= 90);
  assert.ok(info.ghost && info.ghost.frames > 900, 'ghost run loaded');
  assert.ok(info.ghostResult >= 90, `the exact ghost run scores >= 90 live (${info.ghostResult})`);
  assert.ok(info.benchmark && info.benchmark.framesCompared > 900, 'benchmark built from both runs');
  assert.ok(info.benchmark.syncDistance < 0.05, `two exact playbacks are in sync (${info.benchmark.syncDistance} m)`);
  assert.ok(Math.abs(info.benchmark.syncLag) < 0.1, `no lag between them (${info.benchmark.syncLag} s)`);
  assert.ok(info.buttons.includes('export-json') && info.buttons.includes('export-csv'), 'export buttons on the results panel');
  assert.equal(info.duoPanel, true, 'duo HUD panel present on the results screen');
  assert.ok(info.stored.length >= 1 && ['A', 'B', 'tie'].includes(info.stored[0]), 'benchmark stored with the result');
  await page.evaluate(() => window.__dp.app.abortToMenu());
  assert.equal(await page.evaluate(() => !!window.__dp.app.hud.getPanel('duo')), false, 'duo HUD panel removed on exit');
  await page.close();
});

test('(i) online duo: two pages in one room start together and exchange results', { skip, timeout: TIMEOUT }, async () => {
  const query = `emu=playback&speed=3&choreo=tutorial-basics&server=${encodeURIComponent(S.base)}`;
  // two "headsets": separate contexts, otherwise the second tab backgrounds the first one and
  // Chromium throttles its requestAnimationFrame loop; small viewports keep both software-
  // rendered pages at a usable frame rate (the remote states are sampled per render frame)
  const hostContext = await S.browser.newContext({ viewport: { width: 640, height: 360 } });
  const guestContext = await S.browser.newContext({ viewport: { width: 640, height: 360 } });
  const host = await newPage(hostContext);
  const guest = await newPage(guestContext);
  await host.page.goto(`${S.base}/?${query}`);
  await guest.page.goto(`${S.base}/?${query}`);
  for (const p of [host.page, guest.page]) {
    await p.waitForFunction(() => window.__dp && window.__dp.app.state === 'MENU' && window.__dp.app.choreos.length > 0, null, { timeout: 60000 });
  }
  const code = await host.page.evaluate(async () => {
    const app = window.__dp.app;
    app.openDuoLobby();
    const mode = app.modes.get('online');
    const room = await mode.createRoom();
    return room.code;
  });
  console.log(`# online: room ${code}`);
  assert.match(code, /^[A-HJ-NP-Z]{4}$/);
  const joined = await guest.page.evaluate(async (c) => {
    const app = window.__dp.app;
    app.openDuoLobby();
    const mode = app.modes.get('online');
    const room = await mode.joinRoom(c);
    return { players: room.players.length, isHost: mode.isHost, state: app.state, panel: app.menu.getPanel('duo-online').visible };
  }, code);
  console.log(`# online: guest joined ${JSON.stringify(joined)}`);
  assert.equal(joined.players, 2);
  assert.equal(joined.isHost, false);
  assert.equal(joined.state, 'DUO_LOBBY');
  assert.equal(joined.panel, true, 'online lobby panel shown');
  const start = await host.page.evaluate(async () => {
    const mode = window.__dp.app.modes.get('online');
    const s = await mode.startAsHost('tutorial-basics', { delayMs: 2500 });
    return { choreoId: s.choreoId, delayMs: s.delayMs, wait: (s.startAtLocalMs - Date.now()) / 1000, offsetMs: mode.online.offsetMs, rttMs: mode.online.rttMs };
  });
  console.log(`# online: start ${JSON.stringify(start)}`);
  assert.equal(start.choreoId, 'tutorial-basics');
  await waitState(host.page, 'RESULTS', 120000);
  await waitState(guest.page, 'RESULTS', 120000);
  // both results travel through the relay; the benchmark appears when the partner's arrives
  for (const p of [host.page, guest.page]) {
    await p.waitForFunction(() => !!window.__dp.app.lastBenchmark, null, { timeout: 20000, polling: 100 });
  }
  const collect = (page) => page.evaluate(() => {
    const app = window.__dp.app;
    const mode = app.modes.get('online');
    const b = app.lastBenchmark;
    return {
      score: app.lastResult.score, mode: app.lastResult.mode, remoteScore: mode.online.remoteScore,
      remoteStates: mode.online.stats.remoteStates, sent: mode.online.stats.sent,
      remoteFrames: mode.remoteRun ? mode.remoteRun.frameCount : 0,
      benchmark: { winner: b.winner, players: b.players.map((p) => `${p.name}:${p.score}`), syncDistance: b.pair.syncDistance, syncDistanceAligned: b.pair.syncDistanceAligned, syncLag: b.pair.syncLag, framesCompared: b.pair.framesCompared },
      startedT: app.session ? app.session.t : null,
    };
  });
  const h = await collect(host.page);
  const g = await collect(guest.page);
  console.log(`# online host: ${JSON.stringify(h)}`);
  console.log(`# online guest: ${JSON.stringify(g)}`);
  assert.deepEqual(host.errors, [], 'host: no errors');
  assert.deepEqual(guest.errors, [], 'guest: no errors');
  for (const r of [h, g]) {
    assert.equal(r.mode, 'online');
    assert.ok(r.score >= 90, `own score ${r.score}`);
    assert.ok(r.remoteScore >= 90, `partner score on the HUD ${r.remoteScore}`);
    assert.ok(r.remoteStates > 20, `remote states received ${r.remoteStates}`);
    assert.ok(r.benchmark.players[1].endsWith(`:${r.remoteScore}`) || r.benchmark.players[1].split(':')[1] >= 90, 'partner result in the benchmark');
    assert.ok(r.remoteFrames > 600, `partner states recorded as a run (${r.remoteFrames} frames)`);
    assert.ok(r.benchmark.framesCompared > 600, 'pair metrics from both runs');
    assert.ok(r.benchmark.syncDistanceAligned < 0.25, `synchronised dancers (${r.benchmark.syncDistanceAligned} m aligned, ${r.benchmark.syncDistance} m raw)`);
  }
  await host.page.close();
  await guest.page.close();
  await hostContext.close();
  await guestContext.close();
});
