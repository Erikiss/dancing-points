// sw.js - service worker: cache-first app shell (index, src/, vendor/, models/free/*, choreos/,
// assets/) and network-first for /api/*. The cache name is versioned with the APP_VERSION the
// app passes in the registration URL (`sw.js?v=<APP_VERSION>`, see src/config.js), so a new
// version replaces the old shell on activate. Large files (models, mocap choreo) are cached on
// first use, not on install. Never throws on missing files or while offline. Classic script
// (no ES modules): must not import from src/.

const VERSION = (() => {
  try { return new URL(self.location.href).searchParams.get('v') || '0.1.0'; } catch (e) { return '0.1.0'; }
})();
const CACHE = `dp-shell-${VERSION}`;
const CACHE_PREFIX = 'dp-shell-';

// small files fetched on install (missing ones are skipped)
const PRECACHE = [
  './',
  './index.html',
  './manifest.webmanifest',
  './assets/icon.svg',
  './assets/icon-192.png',
  './assets/icon-512.png',
  './vendor/three.module.js',
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
  './choreos/snoop-cwalk.json',
  './choreos/tutorial-basics.json',
  './models/free/meta.json',
  './models/free/skeleton.json',
  './models/free/init_pose.json',
];

// runtime cache-first for these path prefixes (relative to the scope)
const RUNTIME_PREFIXES = ['src/', 'vendor/', 'models/', 'choreos/', 'assets/'];

const scopePath = (() => {
  try { return new URL(self.registration ? self.registration.scope : './', self.location.href).pathname; } catch (e) { return '/'; }
})();

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    try {
      const cache = await caches.open(CACHE);
      await Promise.allSettled(PRECACHE.map(async (url) => {
        try {
          const res = await fetch(new Request(url, { cache: 'reload' }));
          if (res && res.ok) await cache.put(url, res);
        } catch (e) { /* offline or missing: cached on first use */ }
      }));
    } catch (e) { /* never fail the install */ }
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    try {
      const keys = await caches.keys();
      await Promise.all(keys.filter((k) => k.startsWith(CACHE_PREFIX) && k !== CACHE).map((k) => caches.delete(k)));
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

function cacheable(url) {
  const rel = relPath(url);
  if (rel === '' || rel === 'index.html' || rel === 'manifest.webmanifest' || rel === 'sw.js') return true;
  return RUNTIME_PREFIXES.some((p) => rel.startsWith(p));
}

async function networkFirst(request) {
  const cache = await caches.open(CACHE);
  try {
    const res = await fetch(request);
    if (res && res.ok && request.method === 'GET') {
      try { await cache.put(request, res.clone()); } catch (e) { /* ignore */ }
    }
    return res;
  } catch (e) {
    const hit = await cache.match(request);
    if (hit) return hit;
    return new Response(JSON.stringify({ error: 'offline' }), { status: 503, headers: { 'content-type': 'application/json' } });
  }
}

async function cacheFirst(request, event) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(request, { ignoreSearch: true });
  if (hit) return hit;
  try {
    const res = await fetch(request);
    if (res && res.ok && (res.type === 'basic' || res.type === 'default')) {
      try { await cache.put(request, res.clone()); } catch (e) { /* quota: ignore */ }
    }
    return res;
  } catch (e) {
    if (request.mode === 'navigate') {
      const index = await cache.match('./index.html') || await cache.match('./');
      if (index) return index;
    }
    return new Response('offline', { status: 503, headers: { 'content-type': 'text/plain' } });
  }
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  let url;
  try { url = new URL(request.url); } catch (e) { return; }
  if (url.origin !== self.location.origin) return;
  if (isApi(url)) {
    // the server probe must fail honestly when the server is gone (no cached "alive" answer)
    if (/\/api\/(info|health)$/.test(url.pathname)) return;
    event.respondWith(networkFirst(request));
    return;
  }
  if (request.mode === 'navigate' || cacheable(url)) {
    event.respondWith(cacheFirst(request, event));
  }
});

self.addEventListener('message', (event) => {
  const data = event.data;
  if (data && data.type === 'skipWaiting') self.skipWaiting();
  if (data && data.type === 'clearCache') {
    event.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k.startsWith(CACHE_PREFIX)).map((k) => caches.delete(k)))));
  }
});
