// sw.js - service worker of Dancing Points VR.
// Strategy (an arcade headset must pick up every deployed change, but never re-download the
// 85 MB of int8 models because of a text fix):
//   * shell (index.html, manifest, src/*, choreos/*, models/*/*.json, assets/*): NETWORK-FIRST
//     with a conditional request (`cache: 'no-cache'` = ETag revalidation, cheap 304s on the
//     LAN) and the cache as the offline fallback. The shell cache is versioned by SW_VERSION
//     (must equal APP_VERSION in src/config.js - tests/unit/sw.test.js checks it) and old shell
//     caches are deleted on activate.
//   * big immutable assets (vendor/ = three.js + onnxruntime wasm, models/*.onnx): CACHE-FIRST
//     in one unversioned cache that survives version bumps; the response is returned to the
//     page immediately and stored in the background (`event.waitUntil`). After a version bump
//     the stored assets are revalidated once (ETag; a replaced model file is fetched again).
//   * /api/*: network-first with cache fallback; /api/info and /api/health are never cached
//     (the server probe must fail honestly when the server is gone).
// The App registers `./sw.js?v=<APP_VERSION>` and calls `registration.update()` at every boot;
// a byte change of this file (the SW_VERSION bump) is what makes the browser install the new
// worker. Never throws on missing files or while offline. Classic script (no ES modules): must
// not import from src/. Note: Chromium refuses to register a service worker on an origin whose
// certificate error was clicked through (self-signed LAN certificate) - see server/README.md.

const SW_VERSION = '0.1.1';               // == APP_VERSION (src/config.js); bump both on release
const SHELL_CACHE = `dp-shell-${SW_VERSION}`;
const ASSET_CACHE = 'dp-assets';
const SHELL_PREFIX = 'dp-shell-';
const CACHE_PREFIXES = [SHELL_PREFIX, ASSET_CACHE];

// small files fetched on install (missing ones are skipped); every entry of choreos/index.json
// is added at install time as well
const PRECACHE = [
  './',
  './index.html',
  './manifest.webmanifest',
  './assets/icon.svg',
  './assets/icon-192.png',
  './assets/icon-512.png',
  './src/app.js',
  './src/config.js',
  './src/util/math.js',
  './src/util/events.js',
  './src/xr/input.js',
  './src/xr/calibration.js',
  './src/emu/emulated-input.js',
  './src/game/clock.js',
  './src/game/audio.js',
  './src/game/choreo.js',
  './src/game/scoring.js',
  './src/game/session.js',
  './src/game/recorder.js',
  './src/render/scene.js',
  './src/render/avatar.js',
  './src/render/hud.js',
  './src/render/mirror.js',
  './src/ui/menu.js',
  './src/ui/texts.js',
  './src/net/pipeline.js',
  './src/net/worker.js',
  './src/net/avatar-driver.js',
  './src/duo/ghost.js',
  './src/duo/online.js',
  './src/duo/benchmark.js',
  './src/duo/modes.js',
  './choreos/index.json',
  './models/free/meta.json',
  './models/free/skeleton.json',
  './models/free/init_pose.json',
];
// big immutable assets precached on install (the models are cached on first use instead)
const PRECACHE_ASSETS = ['./vendor/three.module.js'];

// path prefixes (relative to the scope) handled by the two strategies
const SHELL_PREFIXES = ['src/', 'choreos/', 'assets/'];
const ASSET_PREFIXES = ['vendor/', 'models/'];

const scopePath = (() => {
  try { return new URL(self.registration ? self.registration.scope : './', self.location.href).pathname; } catch (e) { return '/'; }
})();

let assetsRevalidated = false;

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    try {
      const shell = await caches.open(SHELL_CACHE);
      const list = PRECACHE.slice();
      // every listed choreography (the mocap demo is 0.6 MB - still a shell file)
      try {
        const idx = await (await fetch(new Request('./choreos/index.json', { cache: 'reload' }))).json();
        for (const e of (idx && idx.choreos) || []) if (e && (e.file || e.id)) list.push(`./choreos/${e.file || e.id + '.json'}`);
      } catch (e) { /* offline: cached on first use */ }
      await Promise.allSettled(list.map(async (url) => {
        try {
          const res = await fetch(new Request(url, { cache: 'reload' }));
          if (res && res.ok) await shell.put(url, res);
        } catch (e) { /* offline or missing: cached on first use */ }
      }));
      const assets = await caches.open(ASSET_CACHE);
      await Promise.allSettled(PRECACHE_ASSETS.map(async (url) => {
        try {
          if (await assets.match(url, { ignoreSearch: true })) return;
          const res = await fetch(new Request(url, { cache: 'no-cache' }));
          if (res && res.ok) await assets.put(url, res);
        } catch (e) { /* cached on first use */ }
      }));
    } catch (e) { /* never fail the install */ }
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    try {
      const keys = await caches.keys();
      // old shells go, the asset cache stays (85 MB of models survive a text fix)
      await Promise.all(keys.filter((k) => k.startsWith(SHELL_PREFIX) && k !== SHELL_CACHE).map((k) => caches.delete(k)));
    } catch (e) { /* ignore */ }
    await self.clients.claim();
  })());
});

function isApi(url) {
  return url.pathname.startsWith(scopePath + 'api/') || url.pathname.includes('/api/');
}

function relPath(url) {
  return url.pathname.startsWith(scopePath) ? url.pathname.slice(scopePath.length) : url.pathname;
}

function isShell(url) {
  const rel = relPath(url);
  if (rel === '' || rel === 'index.html' || rel === 'manifest.webmanifest') return true;
  if (/^models\/[^/]+\/[^/]+\.json$/.test(rel)) return true;   // meta/skeleton/init_pose
  return SHELL_PREFIXES.some((p) => rel.startsWith(p));
}

function isAsset(url) {
  const rel = relPath(url);
  return ASSET_PREFIXES.some((p) => rel.startsWith(p));
}

function cacheableResponse(res) {
  return res && res.ok && (res.type === 'basic' || res.type === 'default');
}

/** Conditional request (ETag/If-None-Match through the HTTP cache): fresh when online, cheap. */
function revalidating(request) {
  try { return new Request(request, { cache: 'no-cache' }); } catch (e) { return request; }
}

async function networkFirst(request, cacheName, { navigate = false, offlineJson = false } = {}) {
  const cache = await caches.open(cacheName);
  try {
    const res = await fetch(revalidating(request));
    if (cacheableResponse(res) && request.method === 'GET') {
      try { await cache.put(request, res.clone()); } catch (e) { /* quota: ignore */ }
    }
    return res;
  } catch (e) {
    const hit = await cache.match(request, { ignoreSearch: true });
    if (hit) return hit;
    if (navigate) {
      const index = await cache.match('./index.html') || await cache.match('./');
      if (index) return index;
    }
    if (offlineJson) return new Response(JSON.stringify({ error: 'offline' }), { status: 503, headers: { 'content-type': 'application/json' } });
    return new Response('offline', { status: 503, headers: { 'content-type': 'text/plain' } });
  }
}

/** Cache-first for the big assets; a miss streams to the page while the copy is stored. */
async function cacheFirst(request, event) {
  const cache = await caches.open(ASSET_CACHE);
  const hit = await cache.match(request, { ignoreSearch: true });
  if (hit) return hit;
  try {
    const res = await fetch(request);
    if (cacheableResponse(res)) {
      const copy = res.clone();
      const store = cache.put(request, copy).catch(() => { /* quota: ignore */ });
      if (event && typeof event.waitUntil === 'function') event.waitUntil(store);
    }
    return res;
  } catch (e) {
    return new Response('offline', { status: 503, headers: { 'content-type': 'text/plain' } });
  }
}

/**
 * Once per worker lifetime after a version bump: conditional re-fetch of every stored asset so
 * that a replaced model/vendor file (same name, new bytes) is picked up. 304 answers cost
 * nothing; a changed file is downloaded in the background.
 */
async function revalidateAssets() {
  if (assetsRevalidated) return;
  assetsRevalidated = true;
  try {
    const cache = await caches.open(ASSET_CACHE);
    const meta = await cache.match('./__meta');
    const seen = meta ? await meta.text() : '';
    if (seen === SW_VERSION) return;
    for (const req of await cache.keys()) {
      if (relPath(new URL(req.url)) === '__meta') continue;
      try {
        const old = await cache.match(req);
        const headers = {};
        const etag = old && old.headers.get('etag');
        if (etag) headers['If-None-Match'] = etag;
        const res = await fetch(new Request(req.url, { headers, cache: 'no-cache' }));
        if (res.status === 304) continue;
        if (cacheableResponse(res)) await cache.put(req, res);
      } catch (e) { /* offline: keep the copy */ }
    }
    await cache.put('./__meta', new Response(SW_VERSION, { headers: { 'content-type': 'text/plain' } }));
  } catch (e) { /* ignore */ }
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  let url;
  try { url = new URL(request.url); } catch (e) { return; }
  if (url.origin !== self.location.origin) return;
  if (isApi(url)) {
    if (/\/api\/(info|health)$/.test(url.pathname)) return;
    event.respondWith(networkFirst(request, SHELL_CACHE, { offlineJson: true }));
    return;
  }
  if (request.mode === 'navigate') {
    event.respondWith(networkFirst(request, SHELL_CACHE, { navigate: true }));
    return;
  }
  if (isAsset(url) && !isShell(url)) {
    event.respondWith(cacheFirst(request, event));
    if (!assetsRevalidated) event.waitUntil(revalidateAssets());
    return;
  }
  if (isShell(url)) {
    event.respondWith(networkFirst(request, SHELL_CACHE));
  }
});

self.addEventListener('message', (event) => {
  const data = event.data;
  if (data && data.type === 'skipWaiting') self.skipWaiting();
  if (data && data.type === 'clearCache') {
    event.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => CACHE_PREFIXES.some((p) => k.startsWith(p))).map((k) => caches.delete(k)))));
  }
});
