// server.test.js - spawns server/server.js (HTTP only, temp data dir) and tests the static
// hosting headers/MIME, the REST API (choreo index merge with uploads, results, runs cap) and
// the WebSocket relay (rooms, start broadcast, state relay, rate limiting, cleanup) with two
// `ws` clients, plus the OnlineDuo client class end to end. Needs `npm install` in server/
// (the `ws` package); the tests are skipped with a message when it is missing.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { OnlineDuo, toWsUrl, toHttpUrl } from '../../webxr/src/duo/online.js';
import { makeChoreo } from './helpers/make-choreo.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const serverDir = path.join(repoRoot, 'server');
const serverJs = path.join(serverDir, 'server.js');

let WS = null;
try {
  WS = createRequire(path.join(serverDir, 'package.json'))('ws');
} catch (e) {
  WS = null;
}
const skip = WS ? false : 'ws package missing - run "npm install" inside server/ first';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------------------------
// Server process

let proc = null;
let base = '';
let wsBase = '';
let dataDir = '';
let stdout = '';

async function startServer() {
  dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'dp-server-test-'));
  proc = spawn(process.execPath, [serverJs, '--http', '--http-port', '0', '--no-https', '--data', dataDir, '--dir', path.join(repoRoot, 'webxr')], {
    cwd: repoRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  proc.stdout.setEncoding('utf8');
  proc.stderr.setEncoding('utf8');
  proc.stderr.on('data', (d) => { stdout += d; });
  const ready = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`server did not become ready:\n${stdout}`)), 15000);
    proc.stdout.on('data', (d) => {
      stdout += d;
      const m = /^READY (\{.*\})$/m.exec(stdout);
      if (m) { clearTimeout(timer); resolve(JSON.parse(m[1])); }
    });
    proc.on('exit', (code) => { clearTimeout(timer); reject(new Error(`server exited with ${code}:\n${stdout}`)); });
  });
  const info = await ready;
  base = `http://127.0.0.1:${info.http}`;
  wsBase = `ws://127.0.0.1:${info.http}/ws`;
  return info;
}

async function stopServer() {
  if (!proc) return;
  const p = proc;
  proc = null;
  const exited = new Promise((resolve) => p.once('exit', resolve));
  p.kill('SIGTERM');
  await Promise.race([exited, sleep(3000).then(() => p.kill('SIGKILL'))]);
  fs.rmSync(dataDir, { recursive: true, force: true });
}

async function api(method, route, body, headers = {}) {
  const init = { method, headers: { ...headers } };
  if (body !== undefined) {
    init.body = typeof body === 'string' ? body : JSON.stringify(body);
    init.headers['Content-Type'] = 'application/json';
  }
  const res = await fetch(`${base}${route}`, init);
  const text = await res.text();
  let json = null;
  try { json = JSON.parse(text); } catch (e) { /* not json */ }
  return { status: res.status, headers: res.headers, json, text };
}

// ---------------------------------------------------------------------------------------------
// WebSocket test client with a message queue

function openClient() {
  return new Promise((resolve, reject) => {
    const ws = new WS(wsBase);
    const queue = [];
    const waiters = [];
    const client = {
      ws,
      id: null,
      closed: null,
      all: [],
      send: (obj) => ws.send(JSON.stringify(obj)),
      /** next message of the given type (any when null) within `timeout` ms */
      next: (type = null, timeout = 3000) => new Promise((res, rej) => {
        const idx = queue.findIndex((m) => !type || m.type === type);
        if (idx >= 0) { res(queue.splice(idx, 1)[0]); return; }
        const w = { type, res, rej, timer: setTimeout(() => {
          waiters.splice(waiters.indexOf(w), 1);
          rej(new Error(`timeout waiting for "${type}" (queue: ${queue.map((m) => m.type).join(',')})`));
        }, timeout) };
        waiters.push(w);
      }),
      /** true when a message of the type arrives within `ms` */
      expectNone: async (type, ms = 300) => {
        try { await client.next(type, ms); return false; } catch (e) { return true; }
      },
      drain: () => { const out = queue.splice(0); return out; },
      close: () => new Promise((res) => { if (ws.readyState === WS.CLOSED) res(); else { ws.once('close', () => res()); ws.close(); } }),
    };
    ws.on('message', (data) => {
      const msg = JSON.parse(data.toString());
      client.all.push(msg);
      if (msg.type === 'hello' && !client.id) { client.id = msg.id; resolve(client); return; }
      const wi = waiters.findIndex((w) => !w.type || w.type === msg.type);
      if (wi >= 0) { const w = waiters.splice(wi, 1)[0]; clearTimeout(w.timer); w.res(msg); }
      else queue.push(msg);
    });
    ws.on('close', (code, reason) => { client.closed = { code, reason: reason.toString() }; });
    ws.on('error', reject);
  });
}

before(async () => { if (!skip) await startServer(); });
after(async () => { await stopServer(); });

// ---------------------------------------------------------------------------------------------

test('GET /api/info: version, urls, https flag', { skip }, async () => {
  const r = await api('GET', '/api/info');
  assert.equal(r.status, 200);
  assert.equal(r.json.https, false);
  assert.equal(r.json.http, true);
  assert.match(r.json.version, /^\d+\.\d+\.\d+/);
  assert.ok(Array.isArray(r.json.urls) && r.json.urls.length > 0);
  assert.ok(r.json.urls.every((u) => /^http:\/\/.+:\d+\/$/.test(u)));
  assert.ok(r.json.urls.includes(`http://localhost:${new URL(base).port}/`));
  assert.equal(r.json.wsPath, '/ws');
  assert.equal(r.json.limits.runsPerChoreo, 50);
  assert.equal(r.headers.get('access-control-allow-origin'), '*');
  const opt = await fetch(`${base}/api/results`, { method: 'OPTIONS' });
  assert.equal(opt.status, 204);
  assert.equal((await api('GET', '/api/nope')).status, 404);
  assert.equal((await api('POST', '/api/info', {})).status, 405);
});

test('static files: MIME types, isolation headers, caching, 404, traversal, range', { skip }, async () => {
  const idx = await fetch(`${base}/`);
  assert.equal(idx.status, 200);
  assert.match(idx.headers.get('content-type'), /^text\/html/);
  assert.equal(idx.headers.get('cross-origin-opener-policy'), 'same-origin');
  assert.equal(idx.headers.get('cross-origin-embedder-policy'), 'require-corp');
  assert.equal(idx.headers.get('cross-origin-resource-policy'), 'cross-origin');
  assert.equal(idx.headers.get('cache-control'), 'no-cache');
  const html = await idx.text();
  assert.match(html, /<!doctype html>/i);
  const etag = idx.headers.get('etag');
  const notMod = await fetch(`${base}/index.html`, { headers: { 'If-None-Match': etag } });
  assert.equal(notMod.status, 304);

  const checks = [
    ['/src/app.js', /^text\/javascript/],
    ['/vendor/ort/ort-wasm-simd-threaded.mjs', /^text\/javascript/],
    ['/vendor/ort/ort-wasm-simd-threaded.wasm', /^application\/wasm/],
    ['/choreos/index.json', /^application\/json/],
    ['/manifest.webmanifest', /^application\/manifest\+json/],
  ];
  for (const [route, re] of checks) {
    const r = await fetch(`${base}${route}`, { method: 'HEAD' });
    assert.equal(r.status, 200, route);
    assert.match(r.headers.get('content-type'), re, route);
    assert.ok(Number(r.headers.get('content-length')) > 0, `${route} content-length`);
  }
  const json = await fetch(`${base}/choreos/index.json`);
  assert.equal(json.headers.get('cache-control'), 'no-cache');
  const js = await fetch(`${base}/src/config.js`);
  assert.match(js.headers.get('cache-control'), /max-age/);
  const missing = await fetch(`${base}/does-not-exist.js`);
  assert.equal(missing.status, 404);
  for (const bad of ['/%2e%2e/%2e%2e/etc/passwd', '/src/%2e%2e/%2e%2e/%2e%2e/etc/passwd', '/..%2f..%2fetc/passwd']) {
    const r = await fetch(`${base}${bad}`);
    assert.notEqual(r.status, 200, bad);
    assert.ok(!(await r.text()).includes('root:'), bad);
  }
  const range = await fetch(`${base}/index.html`, { headers: { Range: 'bytes=0-9' } });
  assert.equal(range.status, 206);
  assert.equal((await range.text()).length, 10);
  assert.equal(range.headers.get('content-range').startsWith('bytes 0-9/'), true);
  const dir = await fetch(`${base}/choreos`, { redirect: 'manual' });
  assert.equal(dir.status, 301);
  assert.equal(dir.headers.get('location'), '/choreos/');
  const post = await fetch(`${base}/index.html`, { method: 'POST', body: 'x' });
  assert.equal(post.status, 405);
  const ws426 = await fetch(`${base}/ws`);
  assert.equal(ws426.status, 426);
});

test('choreos: shipped index, upload validation, merge, fetch by id, override, delete', { skip }, async () => {
  const before = await api('GET', '/api/choreos');
  assert.equal(before.status, 200);
  assert.ok(Array.isArray(before.json.choreos));
  const shippedIds = before.json.choreos.map((c) => c.id);
  assert.ok(shippedIds.includes('snoop-cwalk'), 'shipped index is served');
  assert.ok(before.json.choreos.every((c) => c.source === 'shipped' && c.url === `/choreos/${c.file}`));

  const bad = await api('POST', '/api/choreos', { format: 'dancing-points-choreo/1', id: 'bad' });
  assert.equal(bad.status, 400);
  assert.ok(Array.isArray(bad.json.errors) && bad.json.errors.length > 0);
  assert.equal((await api('POST', '/api/choreos', '{oops')).status, 400);
  assert.equal((await api('POST', '/api/choreos', '')).status, 400);

  const choreo = makeChoreo({ id: 'test-upload' });
  choreo.title = 'Upload-Test';
  const up = await api('POST', '/api/choreos', choreo);
  assert.equal(up.status, 201, up.text);
  assert.equal(up.json.id, 'test-upload');
  assert.equal(up.json.url, '/api/choreos/test-upload');
  assert.ok(fs.existsSync(path.join(dataDir, 'choreos', 'test-upload.json')));

  const after1 = await api('GET', '/api/choreos');
  const entry = after1.json.choreos.find((c) => c.id === 'test-upload');
  assert.ok(entry, 'uploaded choreo is merged into the index');
  assert.equal(entry.source, 'upload');
  assert.equal(entry.title, 'Upload-Test');
  assert.equal(entry.bpm, choreo.bpm);
  assert.equal(entry.durationBeats, choreo.durationBeats);
  assert.equal(entry.file, 'test-upload.json');
  assert.equal(after1.json.choreos.length, before.json.choreos.length + 1);
  assert.deepEqual(after1.json.choreos.slice(0, before.json.choreos.length).map((c) => c.id), shippedIds, 'shipped order kept');

  const get = await api('GET', '/api/choreos/test-upload');
  assert.equal(get.status, 200);
  assert.match(get.headers.get('content-type'), /^application\/json/);
  assert.equal(get.headers.get('cache-control'), 'no-cache');
  assert.equal(get.json.title, 'Upload-Test');
  assert.deepEqual(get.json.frames.head[3], choreo.frames.head[3]);
  const shipped = await api('GET', '/api/choreos/snoop-cwalk');
  assert.equal(shipped.status, 200);
  assert.equal(shipped.json.id, 'snoop-cwalk');
  assert.equal((await api('GET', '/api/choreos/nope-nope')).status, 404);
  assert.equal((await api('GET', '/api/choreos/Bad!')).status, 400);

  // an upload with a shipped id replaces the shipped entry (recording replaces the procedural v0)
  const override = makeChoreo({ id: 'snoop-cwalk' });
  override.title = 'Aufgenommen';
  assert.equal((await api('POST', '/api/choreos', override)).status, 201);
  const after2 = await api('GET', '/api/choreos');
  const ov = after2.json.choreos.find((c) => c.id === 'snoop-cwalk');
  assert.equal(ov.source, 'upload');
  assert.equal(ov.replaces, 'shipped');
  assert.equal(after2.json.choreos.filter((c) => c.id === 'snoop-cwalk').length, 1);
  assert.equal((await api('GET', '/api/choreos/snoop-cwalk')).json.title, 'Aufgenommen');
  assert.equal((await api('DELETE', '/api/choreos/snoop-cwalk')).status, 200);
  assert.equal((await api('GET', '/api/choreos/snoop-cwalk')).json.title !== 'Aufgenommen', true, 'shipped file is back');
  assert.equal((await api('DELETE', '/api/choreos/snoop-cwalk')).status, 404);
  assert.equal((await api('DELETE', '/api/choreos/test-upload')).status, 200);
  assert.ok(!fs.existsSync(path.join(dataDir, 'choreos', 'test-upload.json')));

  // body limit: 5 MB for choreos
  const big = `{"pad":"${'x'.repeat(5 * 1024 * 1024 + 10)}"}`;
  const tooBig = await fetch(`${base}/api/choreos`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: big }).catch(() => null);
  if (tooBig) assert.equal(tooBig.status, 413);
  const big1 = `{"pad":"${'x'.repeat(1024 * 1024 + 10)}"}`;
  const tooBig1 = await fetch(`${base}/api/results`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: big1 }).catch(() => null);
  if (tooBig1) assert.equal(tooBig1.status, 413);
});

test('results: POST/GET, validation, filter, persisted to results.json', { skip }, async () => {
  assert.equal((await api('POST', '/api/results', { choreoId: 'snoop-cwalk', score: 200 })).status, 400);
  assert.equal((await api('POST', '/api/results', { score: 50 })).status, 400);
  assert.equal((await api('POST', '/api/results', [1, 2])).status, 400);
  const r1 = await api('POST', '/api/results', { choreoId: 'snoop-cwalk', score: 81, stars: 4, maxCombo: 7, timingBias: 0.03, player: 'Erik', mode: 'solo', moves: [{ name: 'Shoe Vibe', score: 0.9, grade: 'perfekt' }], perBeat: [0.9, 0.8] });
  assert.equal(r1.status, 201, r1.text);
  assert.ok(r1.json.id);
  assert.ok(r1.json.result.receivedAt);
  const r2 = await api('POST', '/api/results', { choreoId: 'tutorial-basics', score: 55, player: 'Anna' });
  assert.equal(r2.status, 201);
  const r3 = await api('POST', '/api/results', { choreoId: 'snoop-cwalk', score: 62, player: 'Anna', playedAt: '2026-09-29T09:00:00.000Z' });
  assert.equal(r3.status, 201);
  const all = await api('GET', '/api/results');
  assert.ok(all.json.results.length >= 3);
  assert.equal(all.json.results[0].id, r3.json.id, 'newest first');
  const only = await api('GET', '/api/results?choreoId=snoop-cwalk');
  assert.ok(only.json.results.every((r) => r.choreoId === 'snoop-cwalk'));
  assert.equal(only.json.results.length, 2);
  assert.equal(only.json.results[1].moves[0].name, 'Shoe Vibe');
  const lim = await api('GET', '/api/results?limit=1');
  assert.equal(lim.json.results.length, 1);
  const byPlayer = await api('GET', '/api/results?player=Anna');
  assert.equal(byPlayer.json.results.length, 2);
  assert.equal((await api('DELETE', '/api/results')).status, 405);
  await sleep(500);
  const persisted = JSON.parse(fs.readFileSync(path.join(dataDir, 'results.json'), 'utf8'));
  assert.ok(persisted.results.some((r) => r.id === r1.json.id));
});

test('runs: validation, both layouts, fetch by id, cap of 50 per choreo, delete', { skip }, async () => {
  assert.equal((await api('POST', '/api/runs', { choreoId: 'cap-test', fps: 30 })).status, 400);
  assert.equal((await api('POST', '/api/runs', { choreoId: 'cap-test', fps: 30, frames: [[1, 2]] })).status, 400);
  assert.equal((await api('POST', '/api/runs', { choreoId: 'cap-test', fps: 30, frames: [], head: [] })).status, 400);
  const full = { format: 'dancing-points-run/1', choreoId: 'cap-test', fps: 30, frameCount: 2, head: [[0, 1.6, 0, 0, 0, 0, 1], [0.1, 1.6, 0, 0, 0, 0, 1]], left: [[0, 1, 0], [0, 1, 0]], right: [[0, 1, 0], [0, 1, 0]], score: 77, player: 'Erik' };
  const first = await api('POST', '/api/runs', full);
  assert.equal(first.status, 201, first.text);
  assert.equal(first.json.run.choreoId, 'cap-test');
  assert.equal(first.json.run.frameCount, 2);
  assert.equal(first.json.run.url, `/api/runs/${first.json.id}`);
  const got = await api('GET', `/api/runs/${first.json.id}`);
  assert.equal(got.status, 200);
  assert.deepEqual(got.json.head, full.head);
  assert.equal(got.json.score, 77);
  assert.ok(got.json.receivedAt);
  const ids = [first.json.id];
  for (let i = 1; i < 52; i++) {
    const compact = { choreoId: 'cap-test', fps: 30, frames: [[i, 1.6, 0, 0, 1, 0, 0, 1, 0]], score: i, player: `P${i}` };
    const r = await api('POST', '/api/runs', compact);
    assert.equal(r.status, 201, r.text);
    ids.push(r.json.id);
    if (i >= 50) assert.equal(r.json.removed.length, 1, 'oldest run removed once over the cap');
  }
  const list = await api('GET', '/api/runs?choreoId=cap-test&limit=1000');
  assert.equal(list.json.runs.length, 50, 'capped at 50 per choreo');
  const listed = new Set(list.json.runs.map((r) => r.id));
  assert.ok(!listed.has(ids[0]) && !listed.has(ids[1]), 'the two oldest are gone');
  assert.ok(listed.has(ids[51]) && listed.has(ids[2]));
  assert.equal(list.json.runs[0].id, ids[51], 'newest first');
  assert.ok(list.json.runs.every((r) => r.url && r.choreoId === 'cap-test' && !r.frames && !r.head), 'list has metadata only');
  assert.equal((await api('GET', `/api/runs/${ids[0]}`)).status, 404);
  assert.ok(!fs.existsSync(path.join(dataDir, 'runs', 'cap-test', `${ids[0]}.json`)));
  assert.equal(fs.readdirSync(path.join(dataDir, 'runs', 'cap-test')).length, 50);
  const other = await api('POST', '/api/runs', { choreoId: 'other-choreo', fps: 30, frames: [[0, 1.6, 0, 0, 1, 0, 0, 1, 0]] });
  assert.equal(other.status, 201);
  assert.equal((await api('GET', '/api/runs?choreoId=other-choreo')).json.runs.length, 1);
  assert.equal((await api('GET', '/api/runs?limit=5')).json.runs.length, 5);
  assert.equal((await api('DELETE', `/api/runs/${other.json.id}`)).status, 200);
  assert.equal((await api('GET', `/api/runs/${other.json.id}`)).status, 404);
  assert.equal((await api('GET', '/api/runs/nope')).status, 404);
});

test('websocket relay: rooms, start broadcast, state relay to the other client only', { skip }, async () => {
  const a = await openClient();
  const b = await openClient();
  const c = await openClient();
  assert.ok(a.id && b.id && c.id && a.id !== b.id);
  assert.equal(a.all[0].type, 'hello');
  assert.ok(Math.abs(a.all[0].serverTime - Date.now()) < 5000);

  // ping / pong
  const t0 = Date.now();
  a.send({ type: 'ping', t0 });
  const pong = await a.next('pong');
  assert.equal(pong.t0, t0);
  assert.ok(Math.abs(pong.t1 - Date.now()) < 2000);

  // create / join
  a.send({ type: 'create', name: 'Erik', choreoId: 'snoop-cwalk' });
  const roomA = await a.next('room');
  assert.match(roomA.code, /^[A-HJ-NP-Z]{4}$/);
  assert.equal(roomA.host, true);
  assert.equal(roomA.you, a.id);
  assert.equal(roomA.choreoId, 'snoop-cwalk');
  assert.deepEqual(roomA.players, [{ id: a.id, name: 'Erik', host: true }]);
  b.send({ type: 'join', code: 'ZZZZ', name: 'Anna' });
  assert.equal((await b.next('error')).code, 'ROOM_NOT_FOUND');
  b.send({ type: 'join', code: roomA.code.toLowerCase(), name: 'Anna' });
  const roomB = await b.next('room');
  assert.equal(roomB.code, roomA.code);
  assert.equal(roomB.host, false);
  assert.equal(roomB.players.length, 2);
  const joined = await a.next('peer');
  assert.equal(joined.event, 'joined');
  assert.equal(joined.player.id, b.id);
  assert.equal(joined.player.name, 'Anna');
  c.send({ type: 'join', code: roomA.code, name: 'Carl' });
  assert.equal((await c.next('error')).code, 'ROOM_FULL');
  const rooms = await api('GET', '/api/rooms');
  assert.equal(rooms.json.rooms.length, 1);
  assert.equal(rooms.json.rooms[0].code, roomA.code);
  assert.equal(rooms.json.rooms[0].players.length, 2);

  // not in a room / not host
  c.send({ type: 'state', t: 0 });
  assert.equal((await c.next('error')).code, 'NOT_IN_ROOM');
  b.send({ type: 'start', choreoId: 'snoop-cwalk' });
  assert.equal((await b.next('error')).code, 'NOT_HOST');
  b.send({ type: 'setChoreo', choreoId: 'tutorial-basics' });
  assert.equal((await b.next('error')).code, 'NOT_HOST');
  a.send({ type: 'setChoreo', choreoId: 'tutorial-basics' });
  assert.equal((await a.next('choreo')).choreoId, 'tutorial-basics');
  assert.equal((await b.next('choreo')).choreoId, 'tutorial-basics');

  // start: server stamps startAt with its own clock and sends it to everyone in the room
  const sent = Date.now();
  a.send({ type: 'start', choreoId: 'snoop-cwalk', delayMs: 500 });
  const startA = await a.next('start');
  const startB = await b.next('start');
  assert.equal(startA.startAt, startB.startAt);
  assert.equal(startA.choreoId, 'snoop-cwalk');
  assert.equal(startA.from, a.id);
  assert.equal(startA.room, roomA.code);
  assert.equal(startA.delayMs, 500);
  assert.ok(startA.startAt >= sent + 500 - 5 && startA.startAt < sent + 3000, `startAt ${startA.startAt - sent} ms after send`);
  assert.ok(await c.expectNone('start'), 'a client outside the room gets no start');

  // state relay: only the other room member receives it, with from + room added
  a.send({ type: 'state', t: 1.25, head: [0, 1.6, 0, 0, 0, 0, 1], left: [-0.2, 1, 0], right: [0.2, 1, 0], score: 42, combo: 3 });
  const st = await b.next('state');
  assert.equal(st.from, a.id);
  assert.equal(st.room, roomA.code);
  assert.equal(st.t, 1.25);
  assert.deepEqual(st.left, [-0.2, 1, 0]);
  assert.equal(st.score, 42);
  assert.ok(await a.expectNone('state'), 'sender does not get its own state');
  assert.ok(await c.expectNone('state'), 'other rooms do not get it');
  // custom message types are relayed too
  b.send({ type: 'result', result: { score: 77 } });
  const res = await a.next('result');
  assert.equal(res.from, b.id);
  assert.equal(res.result.score, 77);
  // reserved types are rejected
  a.send({ type: 'hello' });
  assert.equal((await a.next('error')).code, 'BAD_MESSAGE');
  a.send({ type: 'pong', t0: 1, t1: 2 });
  assert.equal((await a.next('error')).code, 'BAD_MESSAGE');
  a.ws.send('not json');
  assert.equal((await a.next('error')).code, 'BAD_JSON');
  a.ws.send(Buffer.from([1, 2, 3]));
  assert.equal((await a.next('error')).code, 'BAD_MESSAGE');

  // rate limiting: a burst of 120 messages, only ~30 get through, sender is told once
  b.drain();
  await sleep(1100);   // refill the bucket after the messages above
  for (let i = 0; i < 120; i++) a.send({ type: 'state', t: i, head: [0, 0, 0], left: [0, 0, 0], right: [0, 0, 0] });
  await sleep(400);
  const relayed = b.drain().filter((m) => m.type === 'state');
  assert.ok(relayed.length >= 20 && relayed.length <= 45, `rate limited: ${relayed.length} of 120 relayed`);
  const rateErr = a.all.filter((m) => m.type === 'error' && m.code === 'RATE_LIMIT');
  assert.ok(rateErr.length >= 1 && rateErr.length <= 2, `RATE_LIMIT errors: ${rateErr.length}`);
  assert.ok(rateErr[0].dropped >= 1);
  await sleep(1100);   // tokens refill
  a.send({ type: 'state', t: 999, head: [0, 0, 0], left: [0, 0, 0], right: [0, 0, 0] });
  assert.equal((await b.next('state')).t, 999, 'after the refill messages go through again');

  // message size limit: 4 KB -> connection closed with 1009
  const d = await openClient();
  d.send({ type: 'create', name: 'Big' });
  await d.next('room');
  d.send({ type: 'state', pad: 'x'.repeat(5000) });
  await sleep(300);
  assert.ok(d.closed, 'oversized message closes the connection');
  assert.equal(d.closed.code, 1009);

  // leave / host hand-over / room cleanup
  a.send({ type: 'leave' });
  assert.equal((await a.next('left')).room, null);
  const left = await b.next('peer');
  assert.equal(left.event, 'left');
  assert.equal(left.player.id, a.id);
  const host = await b.next('peer');
  assert.equal(host.event, 'host');
  assert.equal(host.player.id, b.id);
  assert.equal(host.players[0].host, true);
  b.send({ type: 'start', choreoId: 'snoop-cwalk', delayMs: 300 });
  assert.equal((await b.next('start')).from, b.id, 'the new host may start');
  await b.close();
  await sleep(200);
  const roomsAfter = await api('GET', '/api/rooms');
  assert.ok(!roomsAfter.json.rooms.some((r) => r.code === roomA.code), 'room is gone when empty');
  // rooms of closed sockets die too (d was closed by the server)
  assert.ok(!roomsAfter.json.rooms.some((r) => r.players.some((p) => p.id === d.id)));
  await a.close();
  await c.close();
  await sleep(100);
  const info = await api('GET', '/api/info');
  assert.equal(info.json.clients, 0);
  assert.equal(info.json.rooms, 0);
});

test('OnlineDuo client end to end: rooms, clock sync, start, state interpolation, result', { skip }, async () => {
  assert.equal(toWsUrl('https://h:8443'), 'wss://h:8443/ws');
  assert.equal(toWsUrl('http://h:8080/'), 'ws://h:8080/ws');
  assert.equal(toWsUrl('wss://h:8443/ws'), 'wss://h:8443/ws');
  assert.equal(toWsUrl('h:8443'), 'wss://h:8443/ws');
  assert.equal(toHttpUrl('wss://h:8443/ws'), 'https://h:8443');
  assert.equal(toHttpUrl('ws://h:8080'), 'http://h:8080');

  const host = new OnlineDuo({ WebSocketImpl: WS, stateHz: 15 });
  const guest = new OnlineDuo({ WebSocketImpl: WS, stateHz: 15 });
  const events = { host: [], guest: [] };
  for (const [name, d] of [['host', host], ['guest', guest]]) {
    for (const type of ['open', 'hello', 'room', 'peer', 'choreo', 'start', 'state', 'result', 'error', 'close', 'sync', 'left']) {
      d.on(type, () => events[name].push(type));
    }
  }
  await host.connect(base);
  assert.equal(host.connected, true);
  assert.equal(host.state, 'connected');
  assert.ok(host.id);
  assert.ok(events.host.includes('open') && events.host.includes('hello'));
  await guest.connect(base.replace('http://', 'ws://') + '/ws');
  assert.ok(guest.id && guest.id !== host.id);

  const sync = await host.syncClock();
  assert.equal(sync.samples, 5);
  assert.ok(Math.abs(sync.offsetMs) < 100, `offset ${sync.offsetMs}`);
  assert.ok(sync.rttMs >= 0 && sync.rttMs < 500);
  assert.equal(host.synced, true);
  assert.ok(Math.abs(host.serverNow() - Date.now()) < 200);
  assert.equal(host.toLocalMs(host.toServerMs(1234)), 1234);
  await guest.syncClock(3);

  const room = await host.createRoom('Erik', 'snoop-cwalk');
  assert.match(room.code, /^[A-Z]{4}$/);
  assert.equal(host.isHost, true);
  assert.equal(host.state, 'lobby');
  assert.equal(host.peer, null);
  await assert.rejects(guest.joinRoom('QQQQ', 'Anna'), (e) => e.code === 'ROOM_NOT_FOUND');
  const joinedRoom = await guest.joinRoom(room.code.toLowerCase(), 'Anna');
  assert.equal(joinedRoom.code, room.code);
  assert.equal(guest.isHost, false);
  assert.equal(guest.peer.name, 'Erik');
  await sleep(50);
  assert.equal(host.peer.name, 'Anna');
  assert.equal(host.room.players.length, 2);
  assert.ok(events.host.includes('peer'));
  await assert.rejects(guest.setChoreo('tutorial-basics'), (e) => e.code === 'NOT_HOST');
  assert.equal(await host.setChoreo('tutorial-basics'), 'tutorial-basics');
  await sleep(50);
  assert.equal(guest.room.choreoId, 'tutorial-basics');

  // start
  const guestStart = new Promise((resolve) => guest.once('start', resolve));
  const started = await host.start('snoop-cwalk', { delayMs: 400 });
  const gs = await guestStart;
  assert.equal(started.choreoId, 'snoop-cwalk');
  assert.equal(gs.startAt, started.startAt);
  assert.ok(Math.abs(started.startAtLocalMs - (Date.now() + 400)) < 300, 'local start time ~ now + delay');
  assert.ok(Math.abs(gs.startAtLocalMs - started.startAtLocalMs) < 100, 'both headsets agree on the local start time');
  assert.equal(host.state, 'playing');
  assert.equal(guest.state, 'playing');
  assert.equal(guest.startChoreoId, 'snoop-cwalk');

  // state upload (rate limited to 15 Hz, one reused object) and interpolation on the other side
  assert.equal(host.getRemoteSample(0), null);
  const q = [0, 0, 0, 1];
  assert.equal(guest.sendState(0.0, [0, 1.6, 0], q, [-0.2, 1, 0], [0.2, 1, 0], 10, 1), true);
  assert.equal(guest.sendState(0.01, [9, 9, 9], q, [9, 9, 9], [9, 9, 9], 99, 9), false, 'second call within 1/15 s is dropped');
  await sleep(80);
  assert.equal(guest.sendState(0.1, [0.3, 1.6, 0], q, [-0.2, 1.1, 0], [0.2, 1.1, 0], 20, 2), true);
  await sleep(80);
  assert.equal(guest.sendState(0.2, [0.6, 1.6, 0], q, [-0.2, 1.2, 0], [0.2, 1.2, 0], 30, 3), true);
  await sleep(100);
  assert.equal(host.remoteStateCount, 3);
  assert.equal(host.remoteScore, 30);
  assert.equal(host.remoteCombo, 3);
  assert.equal(guest.stats.droppedRate, 1);
  const delay = host.interpolationDelaySec;
  const mid = host.getRemoteSample(0.15 + delay);
  assert.ok(mid, 'sample available');
  assert.ok(Math.abs(mid.head.p[0] - 0.45) < 1e-6, `interpolated x ${mid.head.p[0]}`);
  assert.ok(Math.abs(mid.left.p[1] - 1.15) < 1e-6);
  assert.ok(Math.abs(mid.t - 0.15) < 1e-9);
  assert.deepEqual(Array.from(mid.head.q), [0, 0, 0, 1]);
  const past = host.getRemoteSample(0.17 + delay);
  assert.equal(past.score, 30, 'score of the newer state past the half-way point');
  assert.equal(host.getRemoteSample(0.12 + delay).score, 20);
  assert.ok(mid.age >= 0 && mid.age < 5);
  const early = host.getRemoteSample(-1);
  assert.ok(Math.abs(early.head.p[0]) < 1e-6, 'before the first state: oldest held');
  const late = host.getRemoteSample(10);
  assert.ok(Math.abs(late.head.p[0] - 0.6) < 1e-6, 'after the last state: newest held');
  const own = { t: 0, head: { p: [0, 0, 0], q: [0, 0, 0, 1] }, left: { p: [0, 0, 0] }, right: { p: [0, 0, 0] } };
  assert.equal(host.getRemoteSample(0.05 + delay, own), own);
  assert.ok(Math.abs(own.head.p[0] - 0.15) < 1e-6);
  assert.ok(events.host.filter((e) => e === 'state').length === 3);
  assert.equal(guest.getRemoteSample(0.1), null, 'the guest got nothing from the host yet');
  // sendStateFromSample
  await sleep(80);
  assert.equal(host.sendStateFromSample(0.3, own, 5, 0), true);
  await sleep(80);
  assert.equal(guest.remoteStateCount, 1);
  assert.ok(Math.abs(guest.getRemoteSample(0.3 + delay).head.p[0] - 0.15) < 1e-6);

  // result relay
  const resultP = new Promise((resolve) => guest.once('result', resolve));
  assert.equal(host.sendResult({ choreoId: 'snoop-cwalk', score: 91, stars: 5, maxCombo: 8, timingBias: 0.01, durationSec: 32, moves: [{ name: 'Shoe Vibe', score: 0.9, grade: 'perfekt', ticks: 240 }], player: 'Erik', mode: 'online', playedAt: 'x', perBeat: new Float32Array(48) }), true);
  const rmsg = await resultP;
  assert.equal(rmsg.result.score, 91);
  assert.equal(rmsg.from, host.id);
  assert.equal(rmsg.result.perBeat, undefined, 'perBeat is not sent');
  assert.equal(rmsg.result.moves[0].ticks, undefined, 'moves are compacted');

  // an unrelated RATE_LIMIT error must not reject a pending request
  const errs = [];
  host.on('error', (e) => errs.push(e.code));
  for (let i = 0; i < 60; i++) host.sendRaw({ type: 'emote', i });
  await sleep(200);
  assert.ok(errs.includes('RATE_LIMIT'));

  // leave and disconnect
  await sleep(1100);
  await guest.leaveRoom();
  assert.equal(guest.room, null);
  await sleep(50);
  assert.equal(host.peer, null);
  assert.equal(host.remoteStateCount, 0, 'remote buffer reset when the partner leaves');
  guest.disconnect();
  assert.equal(guest.connected, false);
  assert.equal(guest.state, 'closed');
  const closeP = new Promise((resolve) => host.once('close', resolve));
  host.dispose();
  await Promise.race([closeP, sleep(500)]);
  assert.equal(host.connected, false);
  await sleep(100);
  assert.equal((await api('GET', '/api/rooms')).json.rooms.length, 0);
  // connect to a dead port fails cleanly
  const lonely = new OnlineDuo({ WebSocketImpl: WS, requestTimeoutMs: 1500 });
  lonely.on('error', () => {});
  await assert.rejects(lonely.connect('ws://127.0.0.1:1/ws'));
  assert.equal(lonely.state, 'closed');
  const noImpl = new OnlineDuo({ WebSocketImpl: null });
  if (!globalThis.WebSocket) await assert.rejects(noImpl.connect('ws://x'), /not available/);
});
