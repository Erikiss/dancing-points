#!/usr/bin/env node
// worker-smoke.mjs - Playwright smoke test of the inference worker (not part of `npm test`).
//
// Serves the repository root over a tiny static HTTP server on a free port, opens
// tests/net/worker-smoke.html in headless Chromium (playwright-core + the pre-installed browser),
// which starts src/net/worker.js with the free models and a synthetic standing person, and asserts
// that poses come back with finite values and plausible bone lengths. Prints inference timings.
//
// Usage: node tests/net/worker-smoke.mjs [--ticks 60] [--timeout 180000] [--ort-fallback]
//                                        [--models <dir>] [--threads N] [--coi]
//   --ort-fallback  point the worker at a missing ESM build so it falls back to ort.wasm.min.js
//   --models <dir>  serve another model directory (meta/skeleton/init_pose + onnx) instead of webxr/models/free
//   --threads N     onnxruntime wasm threads (needs --coi: COOP/COEP headers -> crossOriginIsolated)
//   PLAYWRIGHT_CORE=<dir containing playwright-core>  (default: a scratchpad vendor-npm dir or NODE_PATH)
//   CHROMIUM=<path to chrome binary>                  (default: /opt/pw-browsers/chromium-*/chrome-linux/chrome)
import http from 'node:http';
import { createRequire } from 'node:module';
import { readFile, readdir, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve, extname, normalize, sep } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..', '..');
const args = process.argv.slice(2);
const argOf = (name, dflt) => { const i = args.indexOf(name); return i >= 0 && i + 1 < args.length ? args[i + 1] : dflt; };
const ticks = Number(argOf('--ticks', 60));
const timeout = Number(argOf('--timeout', 180000));
const ortFallback = args.includes('--ort-fallback');
const extraModels = argOf('--models', null) ? resolve(argOf('--models', null)) : null;
const threads = Number(argOf('--threads', 1));
const coi = args.includes('--coi');
const EXTRA_PREFIX = '/extra-models/';

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream',
  '.css': 'text/css; charset=utf-8', '.png': 'image/png', '.svg': 'image/svg+xml', '.map': 'application/json',
};

function serveStatic() {
  return new Promise((resolveServer) => {
    const server = http.createServer(async (req, res) => {
      try {
        const url = new URL(req.url, 'http://localhost');
        let base = root;
        let pathname = decodeURIComponent(url.pathname);
        if (extraModels && pathname.startsWith(EXTRA_PREFIX)) { base = extraModels; pathname = pathname.slice(EXTRA_PREFIX.length); }
        const rel = normalize(pathname).replace(/^([/\\])+/, '');
        const file = resolve(base, rel);
        if (!file.startsWith(base + sep) && file !== base) { res.writeHead(403); res.end(); return; }
        const st = await stat(file);
        if (st.isDirectory()) { res.writeHead(403); res.end(); return; }
        const data = await readFile(file);
        const headers = {
          'Content-Type': MIME[extname(file).toLowerCase()] || 'application/octet-stream',
          'Content-Length': data.length,
          'Cache-Control': 'no-cache',
        };
        if (coi) { headers['Cross-Origin-Opener-Policy'] = 'same-origin'; headers['Cross-Origin-Embedder-Policy'] = 'require-corp'; }
        res.writeHead(200, headers);
        res.end(data);
      } catch (e) {
        res.writeHead(404); res.end('not found');
      }
    });
    server.listen(0, '127.0.0.1', () => resolveServer(server));
  });
}

async function findChromium() {
  if (process.env.CHROMIUM && existsSync(process.env.CHROMIUM)) return process.env.CHROMIUM;
  const base = '/opt/pw-browsers';
  try {
    const dirs = (await readdir(base)).filter((d) => /^chromium-\d+$/.test(d)).sort();
    for (const d of dirs.reverse()) {
      const p = join(base, d, 'chrome-linux', 'chrome');
      if (existsSync(p)) return p;
    }
  } catch (e) { /* fall through */ }
  return null;
}

function loadPlaywright() {
  const require = createRequire(import.meta.url);
  const candidates = [];
  if (process.env.PLAYWRIGHT_CORE) candidates.push(join(process.env.PLAYWRIGHT_CORE, 'playwright-core'));
  candidates.push('/tmp/claude-0/-home-user-dancing-points/17584e77-ec43-5660-97ee-ff4b7176d577/scratchpad/vendor-npm/node_modules/playwright-core');
  candidates.push(join(root, 'node_modules', 'playwright-core'));
  candidates.push('playwright-core');
  for (const c of candidates) {
    try { return require(c); } catch (e) { /* next */ }
  }
  return null;
}

async function main() {
  const pw = loadPlaywright();
  if (!pw) { console.log('SKIP: playwright-core not found (set PLAYWRIGHT_CORE)'); return 0; }
  const executablePath = await findChromium();
  if (!executablePath) { console.log('SKIP: no Chromium found under /opt/pw-browsers (set CHROMIUM)'); return 0; }
  const server = await serveStatic();
  const port = server.address().port;
  const query = new URLSearchParams({ ticks: String(ticks), timeout: String(timeout), threads: String(threads) });
  if (ortFallback) query.set('ortUrl', `http://127.0.0.1:${port}/webxr/vendor/ort/does-not-exist.mjs`);
  if (extraModels) query.set('models', EXTRA_PREFIX);
  const url = `http://127.0.0.1:${port}/tests/net/worker-smoke.html?${query}`;
  console.log('serving', root, 'at', url);
  const browser = await pw.chromium.launch({ executablePath, headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  let code = 1;
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('console', (m) => { const t = m.text(); if (m.type() === 'error') errors.push(t); if (!/^\[smoke\] pose /.test(t) || /pose \d*0:/.test(t)) console.log('  [page]', t); });
    page.on('pageerror', (e) => { errors.push('pageerror: ' + e.message); console.log('  [pageerror]', e.message); });
    page.on('response', (r) => { if (r.status() >= 400) console.log('  [http ' + r.status() + ']', r.url()); });
    page.on('requestfailed', (r) => console.log('  [request failed]', r.url(), r.failure() && r.failure().errorText));
    await page.goto(url);
    await page.waitForFunction(() => window.__smoke && window.__smoke.done, null, { timeout: timeout + 30000 });
    const smoke = await page.evaluate(() => window.__smoke);
    if (smoke.error) throw new Error('worker reported: ' + smoke.error);
    const poses = smoke.poses;
    if (poses.length < Math.min(ticks, 10)) throw new Error('too few poses: ' + poses.length);
    let nonFinite = 0;
    let maxRatio = 0;
    let maxBone = '';
    let headErr = 0;
    let headYmin = Infinity;
    let headYmax = -Infinity;
    for (const p of poses) {
      if (!p.finite) nonFinite++;
      if (p.maxBoneRatio > maxRatio) { maxRatio = p.maxBoneRatio; maxBone = p.maxBone; }
      headErr = Math.max(headErr, Math.hypot(p.headS[0], p.headS[2]));
      headYmin = Math.min(headYmin, p.headS[1]);
      headYmax = Math.max(headYmax, p.headS[1]);
    }
    const last = poses[poses.length - 1];
    console.log('\nresult:');
    console.log('  ort:', smoke.info.ortVersion, 'from', smoke.info.ortSource, '| simd', smoke.info.simd, '| threads', smoke.info.numThreads, '| crossOriginIsolated', smoke.crossOriginIsolated);
    console.log('  models:', smoke.info.models.mapping.file, (smoke.info.models.mapping.bytes / 1e6).toFixed(1), 'MB,', smoke.info.models.tracking.file, (smoke.info.models.tracking.bytes / 1e6).toFixed(1), 'MB');
    console.log('  load:', smoke.info.loadMs.toFixed(0), 'ms (session create map', smoke.info.createMs.mapping.toFixed(0), 'ms, track', smoke.info.createMs.tracking.toFixed(0), 'ms; warm-up map', smoke.info.warmupMs.mapping.toFixed(1), 'ms, track', smoke.info.warmupMs.tracking.toFixed(1), 'ms)');
    console.log('  poses:', poses.length, '| frames posted:', smoke.timing.framesPosted, '| dropped (busy):', smoke.timing.dropped, '| inferEvery:', smoke.timing.inferEvery, '| driver state:', smoke.driverState);
    console.log('  inference ms: mean', smoke.timing.meanMs.toFixed(1), 'median', smoke.timing.medianMs.toFixed(1), 'min', smoke.timing.minMs.toFixed(1), 'max', smoke.timing.maxMs.toFixed(1), '(mapping', smoke.timing.meanMappingMs.toFixed(1), '+ tracking', smoke.timing.meanTrackingMs.toFixed(1), ') | poses/s', smoke.timing.posesPerSec.toFixed(1));
    console.log('  non-finite poses:', nonFinite, '| max bone length ratio:', maxRatio.toFixed(3), '(' + maxBone + ')', '| head xz distance from input head max', headErr.toFixed(3), 'm | head y range', headYmin.toFixed(3), '..', headYmax.toFixed(3));
    console.log('  last pose: tick', last.tick, 'root', last.rootS.map((v) => v.toFixed(3)).join(','), 'yawS', last.rootYawS.toFixed(3), 'contacts', last.contacts.map((v) => v.toFixed(2)).join(','));
    console.log('  telemetry ticks recorded:', smoke.telemetryTicks, '| console errors:', errors.length);
    const problems = [];
    if (nonFinite) problems.push('non-finite poses');
    if (maxRatio > 1.05 + 1e-3) problems.push('bone length ratio ' + maxRatio);
    if (!(headYmin > 1.0 && headYmax < 2.0)) problems.push('implausible head height');
    if (headErr > 0.5) problems.push('avatar head far from the tracked head');
    if (errors.length) problems.push('console errors: ' + errors.join(' | '));
    if (!(smoke.telemetryTicks > 0)) problems.push('telemetry export returned no ticks');
    if (ortFallback && !/ort\.wasm\.min\.js$/.test(smoke.info.ortSource)) problems.push('fallback did not use the classic bundle');
    if (problems.length) throw new Error('FAIL: ' + problems.join('; '));
    console.log('\nPASS');
    code = 0;
  } catch (e) {
    console.error(e.message);
    code = 1;
  } finally {
    await browser.close();
    server.close();
  }
  return code;
}

main().then((c) => process.exit(c), (e) => { console.error(e); process.exit(1); });
