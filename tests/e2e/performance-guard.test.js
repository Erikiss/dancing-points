// CPU/headless regression of the real emulated app + module inference worker.
// Only the ORT sessions/model bytes are replaced, using existing Python fixture outputs.
// 20/35/70 ms are real asynchronous waits, not claims about Quest 2 or ONNX throughput.
// Exact threshold boundaries and tick scheduling belong to the deterministic unit tests.
// Run: node --test tests/e2e/performance-guard.test.js
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import fs from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const WEB = path.join(ROOT, 'webxr');
const TIMEOUT = 90000;
const WAIT = 20000;

let pw = null;
try { pw = await import('playwright-core'); } catch { /* reported as an explicit skip below */ }

function findChromium() {
  if (!pw) return null;
  if (process.env.PLAYWRIGHT_CHROMIUM && fs.existsSync(process.env.PLAYWRIGHT_CHROMIUM)) return process.env.PLAYWRIGHT_CHROMIUM;
  try {
    for (const dir of fs.readdirSync('/opt/pw-browsers').filter((d) => /^chromium-\d+$/.test(d)).sort().reverse()) {
      for (const platform of ['chrome-linux', 'chrome-linux64']) {
        const file = path.join('/opt/pw-browsers', dir, platform, 'chrome');
        if (fs.existsSync(file)) return file;
      }
    }
  } catch { /* use Playwright's default install next */ }
  const file = pw.chromium.executablePath();
  return fs.existsSync(file) ? file : null;
}

const chromiumPath = findChromium();
const skip = !pw ? 'playwright-core missing (npm ci)' : !chromiumPath
  ? 'no Chromium: set PLAYWRIGHT_CHROMIUM or run npx playwright-core install chromium' : false;
if (skip) console.log(`# performance-guard e2e skipped: ${skip}`);
let browser;
before(async () => {
  if (skip) return;
  browser = await pw.chromium.launch({
    executablePath: chromiumPath, headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader',
      '--use-gl=swiftshader', '--ignore-gpu-blocklist', '--disable-background-timer-throttling',
      '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding'],
  });
});
after(async () => { if (browser) await browser.close(); });

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml',
  '.webmanifest': 'application/manifest+json', '.css': 'text/css',
};

async function testServer(isolated) {
  const config = { delayMs: 20 };
  const outputs = {};
  for (const kind of ['mapping', 'tracking']) {
    outputs[kind] = JSON.parse(await readFile(path.join(ROOT, 'tests/fixtures', kind + '_io.json'), 'utf8')).outputsFp32;
  }
  const stub = await readFile(path.join(ROOT, 'tests/e2e/fixtures/controlled-ort.mjs'));
  let initOpen = false;
  let runOpen = false;
  const initWaiters = new Set();
  const runWaiters = new Set();
  const send = (res, data = '', type = 'text/plain', status = 200) => {
    const bytes = Buffer.isBuffer(data) ? data : Buffer.from(data);
    if (res.destroyed) return;
    res.writeHead(status, { 'Content-Type': type, 'Content-Length': bytes.length });
    res.end(bytes);
  };
  const gate = (waiters, res) => {
    waiters.add(res);
    res.once('close', () => waiters.delete(res));
  };
  const release = (waiters) => { for (const res of waiters) send(res); waiters.clear(); };
  const server = http.createServer(async (req, res) => {
    // These are real document AND worker response headers. No navigator/isolation shim.
    res.setHeader('Cache-Control', 'no-store');
    if (isolated) {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    }
    try {
      const url = new URL(req.url, 'http://localhost');
      switch (url.pathname) {
        case '/__guard__/config': return send(res, JSON.stringify(config), MIME['.json']);
        case '/__guard__/outputs': return send(res, JSON.stringify(outputs), MIME['.json']);
        case '/__guard__/init-gate': return initOpen ? send(res) : gate(initWaiters, res);
        case '/__guard__/run-gate': return runOpen ? send(res) : gate(runWaiters, res);
        case '/vendor/ort/ort.wasm.min.mjs': return send(res, stub, MIME['.mjs']);
        case '/models/free/mapping_leader.int8.onnx': return send(res, Buffer.from([1]), 'application/octet-stream');
        case '/models/free/tracking_leader.int8.onnx': return send(res, Buffer.from([2]), 'application/octet-stream');
        // Static-site mode: no result API. A neutral JSON response avoids expected 404 noise.
        case '/api/info': return send(res, '{}', MIME['.json']);
        case '/favicon.ico': return send(res);
        default: break;
      }
      const pathname = decodeURIComponent(url.pathname === '/' ? '/index.html' : url.pathname);
      const file = path.resolve(WEB, '.' + pathname);
      if (!file.startsWith(WEB + path.sep)) return send(res, 'forbidden', 'text/plain', 403);
      send(res, await readFile(file), MIME[path.extname(file)] || 'application/octet-stream');
    } catch (err) { send(res, String(err.message), 'text/plain', 404); }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  return {
    base: 'http://127.0.0.1:' + server.address().port, config,
    releaseInit() { initOpen = true; release(initWaiters); },
    releaseRuns() { runOpen = true; release(runWaiters); },
    async close() {
      release(initWaiters); release(runWaiters);
      const closed = new Promise((resolve) => server.close(resolve));
      server.closeAllConnections();
      await closed;
    },
  };
}

async function scenario(t, isolated = false) {
  const server = await testServer(isolated);
  const context = await browser.newContext({ viewport: { width: 480, height: 320 }, serviceWorkers: 'block' });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (err) => errors.push(err.message));
  page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()); });
  t.after(async () => { await context.close(); await server.close(); });
  await page.goto(server.base + '/?emu=playback&choreo=tutorial-basics&autostart=1&speed=10&avatar=neural');
  await page.waitForFunction(() => window.__dp?.app.neural?.driver?.state === 'loading', null, { timeout: WAIT });
  await page.evaluate(() => {
    window.__guard = { statuses: [], poses: [], observations: [], sent: [] };
    window.__dp.app.on('neural', (state) => window.__guard.statuses.push({ ...state }));
  });
  await attachWorkerProbe(page);
  return { page, server, errors };
}

async function attachWorkerProbe(page) {
  await page.evaluate(() => {
    const d = window.__dp.app.neural.driver;
    const w = d.worker;
    const log = window.__guard;
    const post = w.postMessage.bind(w);
    w.postMessage = (msg, ...args) => {
      // Do not retain the transferable buffers or individual 30 Hz input frames.
      if (msg.type !== 'frame' && msg.type !== 'recycle') log.sent.push({ type: msg.type, inferEvery: msg.inferEvery });
      return post(msg, ...args);
    };
    w.addEventListener('message', ({ data: msg }) => {
      if (msg.type === 'pose') log.poses.push({ seq: msg.seq, tick: msg.tick, ms: msg.inferenceMs });
      if (msg.type === '__guard_observation') log.observations.push(msg);
    });
  });
}

let controlId = 0;
async function control(page, values) {
  const id = ++controlId;
  await page.evaluate(({ id, values }) => window.__dp.app.neural.driver.worker.postMessage({
    type: '__guard_control', id, ...values,
  }), { id, values });
  await page.waitForFunction((id) => window.__guard.observations.some((msg) => msg.event === 'configured' && msg.id === id), id, { timeout: WAIT });
  return page.evaluate((id) => window.__guard.observations.find((msg) => msg.event === 'configured' && msg.id === id), id);
}

async function waitNeural(page, state, minPoses = 0) {
  await page.waitForFunction(({ state, minPoses }) => {
    const n = window.__dp.app.neural;
    return n.driver.state === state && n.driver.stats.poses >= minPoses;
  }, { state, minPoses }, { timeout: WAIT });
}

async function framesAdvance(page, count = 10) {
  const before = await page.evaluate(() => window.__dp.app.frameCount);
  await page.waitForFunction(({ before, count }) => window.__dp.app.frameCount >= before + count,
    { before, count }, { timeout: WAIT });
}

async function result(page) {
  await page.waitForFunction(() => window.__dp.app.state === 'RESULTS', null, { timeout: WAIT });
  const score = await page.evaluate(() => window.__dp.app.lastResult.score);
  assert.ok(score >= 90, 'the existing exact-playback scoring baseline survives the neural workload: ' + score);
}

async function snapshot(page) {
  return page.evaluate(() => {
    const app = window.__dp.app;
    const n = app.neural;
    const d = n.driver;
    return {
      state: d.state, appState: app.state, inferEvery: d.inferEvery, meanMs: d.stats.meanMs,
      poses: d.stats.poses, callbackPoses: n.poses, frames: app.frameCount,
      fallback: n.fallback, body: app.mirror.usesBody, visible: app.mirror.visible,
      worker: !!d.worker, ready: n.ready, disabledSaved: !!app.settings.neuralVerdict?.disabled,
      numThreads: n.info?.numThreads, isolated: crossOriginIsolated,
      sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    };
  });
}

for (const isolated of [false, true]) {
  test(`guard: 20 → 35 → 70 ms and settings retry (COOP/COEP ${isolated ? 'on' : 'off'})`, { skip, timeout: TIMEOUT }, async (t) => {
    const { page, server, errors } = await scenario(t, isolated);
    server.releaseInit();
    await waitNeural(page, 'running', 12);
    const full = await snapshot(page);
    assert.equal(full.isolated, isolated, 'document isolation follows real server headers');
    assert.equal(full.sharedArrayBuffer, isolated, 'SharedArrayBuffer availability follows isolation');
    assert.equal(full.inferEvery, 1, 'full 30 Hz inference schedule selected');
    assert.ok(full.meanMs >= 19 && full.meanMs <= 25, '20 ms wait + pipeline overhead stays in full-rate range: ' + full.meanMs);
    const worker = await control(page, { delayMs: 35 });
    assert.equal(worker.isolated, isolated, 'worker isolation follows real server headers');
    assert.equal(worker.sharedArrayBuffer, isolated);
    assert.equal(worker.numThreads, full.numThreads, 'selected thread budget reaches the ORT session boundary');
    if (!isolated) assert.equal(full.numThreads, 1);
    else {
      const expected = await page.evaluate(() => {
        const cores = navigator.hardwareConcurrency || 2;
        return Math.min(cores <= 8 ? 2 : 4, Math.max(1, cores - 1));
      });
      assert.equal(full.numThreads, expected);
    }

    await waitNeural(page, 'slow');
    await framesAdvance(page);
    const slow = await snapshot(page);
    assert.equal(slow.inferEvery, 2, 'every second 30 Hz input tick selects nominal 15 Hz');
    assert.ok(slow.meanMs > 25 && slow.meanMs < 60);
    assert.equal(slow.body, true, 'slow mode keeps the neural mirror body');
    assert.equal(slow.fallback, null);
    assert.ok(await page.evaluate(() => window.__guard.sent.some((m) => m.type === 'config' && m.inferEvery === 2)), '15 Hz configuration sent to the real worker');
    const slowStart = await page.evaluate(() => window.__guard.poses.length);
    await page.waitForFunction((index) => window.__guard.poses.length >= index + 6, slowStart, { timeout: WAIT });
    const slowSeqs = await page.evaluate((index) => window.__guard.poses.slice(index, index + 6).map((p) => p.seq), slowStart);
    assert.ok(slowSeqs.every((seq) => seq % 2 === slowSeqs[0] % 2), 'real worker infers only every second input tick: ' + slowSeqs);
    assert.ok(slowSeqs.slice(1).every((seq, i) => seq - slowSeqs[i] >= 2), 'slow-mode pose sequences advance by at least two input ticks');
    await control(page, { delayMs: 70 });
    await waitNeural(page, 'disabled');
    await framesAdvance(page);
    const disabled = await snapshot(page);
    assert.ok(disabled.meanMs > 60, 'moving average crosses 60 ms');
    assert.ok(disabled.callbackPoses > 0, 'fallback also works after successful neural poses');
    assert.equal(disabled.body, false, 'mirror returns to points');
    assert.equal(disabled.worker, false, 'too-slow worker terminated');
    assert.equal(disabled.disabledSaved, true, 'guard verdict persisted');
    assert.ok(disabled.fallback);
    await result(page);

    // The existing Settings action promises to retry after a stored performance verdict.
    server.config.delayMs = 20;
    await page.evaluate(() => {
      window.__guard.oldDriver = window.__dp.app.neural.driver;
      window.__dp.app.setAvatarMode('neural');
    });
    await page.waitForFunction(() => window.__dp.app.neural.driver !== window.__guard.oldDriver,
      null, { timeout: WAIT });
    await waitNeural(page, 'running', 10);
    await framesAdvance(page);
    const retried = await snapshot(page);
    assert.equal(retried.inferEvery, 1);
    assert.equal(retried.body, true);
    assert.equal(retried.disabledSaved, false);
    assert.equal(retried.fallback, null);
    assert.deepEqual(errors, []);
    console.log('# guard modes ' + JSON.stringify({ isolated, full, slow, disabled, retried }));
  });
}

test('slow model initialization keeps rendering/scoring; a delayed old run cannot cross reset', { skip, timeout: TIMEOUT }, async (t) => {
  const { page, server, errors } = await scenario(t);
  await framesAdvance(page);
  await result(page);
  const loading = await snapshot(page);
  assert.equal(loading.state, 'loading', 'session.create remains gated throughout the dance');
  assert.equal(loading.ready, false);
  assert.equal(loading.body, false, 'point mirror while neural initialization is pending');
  assert.equal(loading.poses, 0);
  server.releaseInit();
  await waitNeural(page, 'running', 10);

  await control(page, { blockNextRun: true });
  await page.waitForFunction(() => window.__guard.observations.some((msg) => msg.event === 'run-blocked'), null, { timeout: WAIT });
  const blocked = await page.evaluate(() => ({ frames: window.__dp.app.frameCount, poses: window.__guard.poses.length }));
  await framesAdvance(page, 8);
  assert.equal(await page.evaluate(() => window.__guard.poses.length), blocked.poses, 'worker reply is genuinely pending');
  const reset = await page.evaluate(async () => {
    const app = window.__dp.app;
    app.abortToMenu();
    const resetSeq = app.neural.driver.seq;
    const poseIndex = window.__guard.poses.length;
    await app.startSolo('tutorial-basics');
    return { resetSeq, poseIndex };
  });
  server.releaseRuns();
  await page.waitForFunction((index) => window.__guard.poses.length >= index + 3, reset.poseIndex, { timeout: WAIT });
  const arrived = await page.evaluate((index) => window.__guard.poses.slice(index), reset.poseIndex);
  assert.ok(arrived.every((pose) => pose.seq > reset.resetSeq), 'no completion from the pre-reset sequence reaches the new dance: ' + JSON.stringify({ reset, arrived }));
  await result(page);
  await framesAdvance(page);
  assert.equal((await snapshot(page)).fallback, null);
  assert.deepEqual(errors, []);
});

test('worker reply timeout falls back to points while rendering and emulated scoring continue', { skip, timeout: TIMEOUT }, async (t) => {
  const { page, server, errors } = await scenario(t);
  server.releaseInit();
  await waitNeural(page, 'running', 10);
  await control(page, { blockNextRun: true });
  await page.waitForFunction(() => window.__guard.observations.some((msg) => msg.event === 'run-blocked'), null, { timeout: WAIT });
  await framesAdvance(page, 8);
  await waitNeural(page, 'error');
  await framesAdvance(page);
  const info = await snapshot(page);
  assert.equal(info.worker, false, 'unresponsive worker terminated');
  assert.equal(info.body, false, 'error fallback hides the stale neural body');
  assert.match(info.fallback, /timeout|timed out|response/i, 'timeout reason is visible');
  assert.equal(info.disabledSaved, false, 'worker timeout does not save a performance verdict');
  await result(page);
  assert.deepEqual(errors, []);
  console.log('# gated worker timeout ' + JSON.stringify(info));
});
