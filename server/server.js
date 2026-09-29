#!/usr/bin/env node
// server.js - Dancing Points VR LAN server (DESIGN.md section 9).
// Responsibilities: serve the static WebXR app over HTTPS (self-signed certificate with the LAN
// IPs as SANs, generated on first start) and plain HTTP, the REST API (/api/info, /api/choreos,
// /api/results, /api/runs, /api/rooms) backed by JSON files in the data directory, and the duo
// mode WebSocket relay on /ws (rooms with 4-letter codes, clock sync, start broadcast, state
// relay, rate limiting). Node >= 18, ES module, dependencies: ws, selfsigned.
// Must not: implement game logic, scoring or any UI; the browser app decides what to do with
// the relayed messages. No auth (LAN only).

import http from 'node:http';
import https from 'node:https';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { WebSocketServer } from 'ws';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------------------------------------------------------------------------------------------
// Constants

export const SERVER_VERSION = '0.1.0';
export const WS_PATH = '/ws';
export const MAX_WS_MESSAGE = 4096;          // bytes per WebSocket message
export const RATE_PER_SEC = 30;              // messages per second per client
export const RATE_BURST = 30;                // token bucket capacity
export const MAX_ROOM_PLAYERS = 2;
export const MAX_BODY_CHOREO = 5 * 1024 * 1024;
export const MAX_BODY_DEFAULT = 1 * 1024 * 1024;
export const MAX_RUNS_PER_CHOREO = 50;
export const MAX_RESULTS = 5000;
export const MAX_RUN_FRAMES = 20000;
export const DEFAULT_START_DELAY_MS = 3000;
export const HEARTBEAT_MS = 30000;
const ID_PATTERN = /^[a-z0-9-]+$/;
const ROOM_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ';   // no I / O (ambiguous on a HUD)
const ROOM_CODE_LEN = 4;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.htm': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.cjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.webmanifest': 'application/manifest+json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.bin': 'application/octet-stream',
  '.data': 'application/octet-stream',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.mp3': 'audio/mpeg',
  '.ogg': 'audio/ogg',
  '.oga': 'audio/ogg',
  '.wav': 'audio/wav',
  '.m4a': 'audio/mp4',
  '.aac': 'audio/aac',
  '.flac': 'audio/flac',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.glb': 'model/gltf-binary',
  '.gltf': 'model/gltf+json',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.otf': 'font/otf',
  '.xml': 'application/xml; charset=utf-8',
  '.pdf': 'application/pdf',
};
// files whose content changes between deployments and must always be revalidated
const NO_CACHE_EXT = new Set(['.html', '.htm', '.json', '.webmanifest']);
const NO_CACHE_NAMES = new Set(['sw.js']);

// ---------------------------------------------------------------------------------------------
// CLI

function printHelp() {
  process.stdout.write(`Dancing Points VR - LAN-Server v${SERVER_VERSION}

Aufruf: node server/server.js [Optionen]

  --port <n>        HTTPS-Port (Standard 8443)
  --http-port <n>   HTTP-Port (Standard 8080, 0 = freier Port)
  --http            HTTP-Listener einschalten (Standard: an)
  --no-http         HTTP-Listener ausschalten
  --no-https        HTTPS-Listener ausschalten (kein Zertifikat noetig; nur fuer Tests/Dev)
  --host <addr>     Bind-Adresse (Standard 0.0.0.0)
  --dir <pfad>      Verzeichnis der Web-App (Standard: ../webxr relativ zu server.js)
  --data <pfad>     Datenverzeichnis fuer Zertifikat, Uploads, Ergebnisse (Standard: ./data neben server.js)
  --cert <pem> --key <pem>   eigenes Zertifikat statt des selbstsignierten
  --no-coep         Cross-Origin-Isolation-Header (COOP/COEP) weglassen
  --verbose         jede Anfrage loggen
  --quiet           nur Fehler ausgeben
  --version, --help
`);
}

export function parseArgs(argv) {
  const opts = {
    port: 8443,
    httpPort: 8080,
    http: true,
    https: true,
    host: '0.0.0.0',
    dir: path.resolve(__dirname, '..', 'webxr'),
    data: path.resolve(__dirname, 'data'),
    cert: null,
    key: null,
    coep: true,
    verbose: false,
    quiet: false,
    help: false,
    version: false,
  };
  const next = (i, name) => {
    if (i + 1 >= argv.length) throw new Error(`Option ${name} braucht einen Wert`);
    return argv[i + 1];
  };
  const num = (v, name) => {
    const n = Number(v);
    if (!Number.isInteger(n) || n < 0 || n > 65535) throw new Error(`Ungueltiger Wert fuer ${name}: ${v}`);
    return n;
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    switch (a) {
      case '--port': opts.port = num(next(i, a), a); i++; break;
      case '--http-port': opts.httpPort = num(next(i, a), a); i++; break;
      case '--http': opts.http = true; break;
      case '--no-http': opts.http = false; break;
      case '--https': opts.https = true; break;
      case '--no-https': opts.https = false; break;
      case '--host': opts.host = next(i, a); i++; break;
      case '--dir': opts.dir = path.resolve(process.cwd(), next(i, a)); i++; break;
      case '--data': opts.data = path.resolve(process.cwd(), next(i, a)); i++; break;
      case '--cert': opts.cert = path.resolve(process.cwd(), next(i, a)); i++; break;
      case '--key': opts.key = path.resolve(process.cwd(), next(i, a)); i++; break;
      case '--no-coep': opts.coep = false; break;
      case '--verbose': opts.verbose = true; break;
      case '--quiet': opts.quiet = true; break;
      case '--help': case '-h': opts.help = true; break;
      case '--version': case '-v': opts.version = true; break;
      default: {
        const m = /^--([a-z-]+)=(.*)$/.exec(a);
        if (m) { argv.splice(i + 1, 0, m[2]); argv[i] = `--${m[1]}`; i--; break; }
        throw new Error(`Unbekannte Option: ${a}`);
      }
    }
  }
  if (!opts.http && !opts.https) throw new Error('Mindestens einer von HTTP/HTTPS muss aktiv sein');
  return opts;
}

// ---------------------------------------------------------------------------------------------
// Small helpers

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

function isVec(v, n) {
  if (!Array.isArray(v) || v.length !== n) return false;
  for (let i = 0; i < n; i++) if (!isNum(v[i])) return false;
  return true;
}

function shortId(bytes = 6) {
  return crypto.randomBytes(bytes).toString('hex');
}

function timeId(prefix) {
  return `${prefix}${Date.now().toString(36)}${crypto.randomBytes(3).toString('hex')}`;
}

function cleanName(v, fallback = 'Spieler') {
  if (typeof v !== 'string') return fallback;
  const s = v.replace(/[\u0000-\u001f\u007f]/g, '').trim().slice(0, 24);
  return s.length ? s : fallback;
}

function cleanString(v, max = 200) {
  return typeof v === 'string' ? v.slice(0, max) : undefined;
}

export function lanAddresses() {
  const out = [];
  const ifaces = os.networkInterfaces();
  for (const name of Object.keys(ifaces)) {
    for (const info of ifaces[name] || []) {
      const family = typeof info.family === 'number' ? (info.family === 4 ? 'IPv4' : 'IPv6') : info.family;
      if (family !== 'IPv4' || info.internal) continue;
      out.push(info.address);
    }
  }
  // private ranges first (that is what the Quest sees in the WLAN)
  const rank = (ip) => (/^192\.168\./.test(ip) ? 0 : /^10\./.test(ip) ? 1 : /^172\.(1[6-9]|2\d|3[01])\./.test(ip) ? 2 : 3);
  out.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  return out;
}

async function writeFileAtomic(file, text) {
  const tmp = `${file}.${process.pid}.tmp`;
  await fsp.writeFile(tmp, text, 'utf8');
  await fsp.rename(tmp, file);
}

async function readJSON(file, fallback = null) {
  try {
    return JSON.parse(await fsp.readFile(file, 'utf8'));
  } catch (e) {
    return fallback;
  }
}

function safeStatSync(p) {
  try { return fs.statSync(p); } catch (e) { return null; }
}

// ---------------------------------------------------------------------------------------------
// Choreography validation: use the app's validator when available (repo layout), otherwise a
// structural fallback (the server can run stand-alone with only the static files).

let validateChoreoImpl = null;

async function loadChoreoValidator() {
  try {
    const mod = await import(new URL('../webxr/src/game/choreo.js', import.meta.url).href);
    if (typeof mod.validateChoreo === 'function') {
      validateChoreoImpl = mod.validateChoreo;
      return 'app';
    }
  } catch (e) {
    // fall through
  }
  validateChoreoImpl = fallbackValidateChoreo;
  return 'fallback';
}

export function fallbackValidateChoreo(c) {
  const errors = [];
  if (!c || typeof c !== 'object') return { ok: false, errors: ['not an object'] };
  if (c.format !== 'dancing-points-choreo/1') errors.push('format must be "dancing-points-choreo/1"');
  if (typeof c.id !== 'string' || !ID_PATTERN.test(c.id)) errors.push('id must match [a-z0-9-]+');
  if (typeof c.title !== 'string' || !c.title) errors.push('title missing');
  if (!isNum(c.bpm) || c.bpm <= 0) errors.push('bpm must be > 0');
  if (!isNum(c.durationBeats) || c.durationBeats <= 0) errors.push('durationBeats must be > 0');
  if (!isNum(c.fps) || c.fps <= 0) errors.push('fps must be > 0');
  if (!Array.isArray(c.moves) || c.moves.length === 0) errors.push('moves must be a non-empty array');
  if (!c.frames || typeof c.frames !== 'object') errors.push('frames missing');
  else {
    const n = Math.round((c.durationBeats * 60 / c.bpm) * c.fps) + 1;
    for (const [key, w] of [['head', 7], ['left', 3], ['right', 3]]) {
      const arr = c.frames[key];
      if (!Array.isArray(arr)) { errors.push(`frames.${key} missing`); continue; }
      if (arr.length !== n && arr.length !== n - 1) errors.push(`frames.${key} has ${arr.length} entries, expected ${n}`);
      for (let i = 0; i < arr.length; i++) if (!isVec(arr[i], w)) { errors.push(`frames.${key}[${i}] must be ${w} finite numbers`); break; }
    }
  }
  return { ok: errors.length === 0, errors };
}

// ---------------------------------------------------------------------------------------------
// Result / run validation (REST bodies)

export function validateResult(r) {
  const errors = [];
  if (!r || typeof r !== 'object' || Array.isArray(r)) return { ok: false, errors: ['not an object'] };
  if (typeof r.choreoId !== 'string' || !ID_PATTERN.test(r.choreoId)) errors.push('choreoId must match [a-z0-9-]+');
  if (!isNum(r.score) || r.score < 0 || r.score > 100) errors.push('score must be a number in [0, 100]');
  if (r.stars !== undefined && (!isNum(r.stars) || r.stars < 0 || r.stars > 5)) errors.push('stars must be 0..5');
  if (r.maxCombo !== undefined && (!isNum(r.maxCombo) || r.maxCombo < 0)) errors.push('maxCombo must be >= 0');
  if (r.timingBias !== undefined && r.timingBias !== null && !isNum(r.timingBias)) errors.push('timingBias must be a number');
  if (r.player !== undefined && r.player !== null && typeof r.player !== 'string') errors.push('player must be a string');
  if (r.mode !== undefined && r.mode !== null && typeof r.mode !== 'string') errors.push('mode must be a string');
  if (r.playedAt !== undefined && r.playedAt !== null && (typeof r.playedAt !== 'string' || Number.isNaN(Date.parse(r.playedAt)))) errors.push('playedAt must be an ISO date string');
  if (r.moves !== undefined && (!Array.isArray(r.moves) || r.moves.length > 256)) errors.push('moves must be an array (<= 256)');
  if (r.perBeat !== undefined && (!Array.isArray(r.perBeat) || r.perBeat.length > 8192)) errors.push('perBeat must be an array (<= 8192)');
  return { ok: errors.length === 0, errors };
}

/** Accepts `dancing-points-run/1` (head[7]/left[3]/right[3]) or the compact `frames[[9]]` layout. */
export function validateRun(r) {
  const errors = [];
  if (!r || typeof r !== 'object' || Array.isArray(r)) return { ok: false, errors: ['not an object'] };
  if (typeof r.choreoId !== 'string' || !ID_PATTERN.test(r.choreoId)) errors.push('choreoId must match [a-z0-9-]+');
  if (!isNum(r.fps) || r.fps <= 0 || r.fps > 240) errors.push('fps must be in (0, 240]');
  let n = -1;
  if (Array.isArray(r.frames)) {
    n = r.frames.length;
    for (let i = 0; i < n; i++) if (!isVec(r.frames[i], 9)) { errors.push(`frames[${i}] must be 9 finite numbers`); break; }
  } else if (Array.isArray(r.head) && Array.isArray(r.left) && Array.isArray(r.right)) {
    n = r.head.length;
    if (r.left.length !== n || r.right.length !== n) errors.push('head/left/right must have the same length');
    for (let i = 0; i < n; i++) if (!isVec(r.head[i], 7)) { errors.push(`head[${i}] must be 7 finite numbers`); break; }
    for (let i = 0; i < r.left.length; i++) if (!isVec(r.left[i], 3)) { errors.push(`left[${i}] must be 3 finite numbers`); break; }
    for (let i = 0; i < r.right.length; i++) if (!isVec(r.right[i], 3)) { errors.push(`right[${i}] must be 3 finite numbers`); break; }
  } else {
    errors.push('run needs frames[[9]] or head[[7]]/left[[3]]/right[[3]]');
  }
  if (n === 0) errors.push('run has no frames');
  if (n > MAX_RUN_FRAMES) errors.push(`run has ${n} frames, max ${MAX_RUN_FRAMES}`);
  if (r.frameCount !== undefined && isNum(r.frameCount) && n >= 0 && r.frameCount !== n) errors.push(`frameCount ${r.frameCount} does not match ${n} frames`);
  if (r.player !== undefined && r.player !== null && typeof r.player !== 'string') errors.push('player must be a string');
  if (r.score !== undefined && r.score !== null && !isNum(r.score)) errors.push('score must be a number');
  if (r.playedAt !== undefined && r.playedAt !== null && typeof r.playedAt !== 'string') errors.push('playedAt must be a string');
  return { ok: errors.length === 0, errors, frameCount: n };
}

// ---------------------------------------------------------------------------------------------
// Data store (JSON files under --data)

class DataStore {
  constructor(dataDir, log) {
    this.dir = dataDir;
    this.log = log;
    this.choreoDir = path.join(dataDir, 'choreos');
    this.runDir = path.join(dataDir, 'runs');
    this.resultsFile = path.join(dataDir, 'results.json');
    this.uploaded = new Map();   // id -> index entry of an uploaded choreography
    this.runs = new Map();       // runId -> run meta
    this.results = [];           // newest last
    this._resultsTimer = null;
    this._resultsDirty = false;
    this._writing = Promise.resolve();
  }

  async init() {
    await fsp.mkdir(this.choreoDir, { recursive: true });
    await fsp.mkdir(this.runDir, { recursive: true });
    // uploaded choreographies
    for (const f of await fsp.readdir(this.choreoDir)) {
      if (!f.endsWith('.json')) continue;
      const obj = await readJSON(path.join(this.choreoDir, f));
      if (!obj || typeof obj.id !== 'string' || `${obj.id}.json` !== f) { this.log.warn(`Ignoriere ${f} in ${this.choreoDir}`); continue; }
      this.uploaded.set(obj.id, this._choreoEntry(obj, (await fsp.stat(path.join(this.choreoDir, f))).mtime.toISOString()));
    }
    // runs
    for (const choreoId of await fsp.readdir(this.runDir)) {
      const sub = path.join(this.runDir, choreoId);
      const st = safeStatSync(sub);
      if (!st || !st.isDirectory()) continue;
      for (const f of await fsp.readdir(sub)) {
        if (!f.endsWith('.json')) continue;
        const obj = await readJSON(path.join(sub, f));
        if (!obj || typeof obj.id !== 'string') continue;
        this.runs.set(obj.id, this._runMeta(obj));
      }
    }
    // results
    const res = await readJSON(this.resultsFile, null);
    if (res && Array.isArray(res.results)) this.results = res.results.filter((r) => r && typeof r === 'object');
  }

  _choreoEntry(c, uploadedAt) {
    return {
      id: c.id,
      title: c.title,
      artist: typeof c.artist === 'string' ? c.artist : '',
      bpm: c.bpm,
      durationBeats: c.durationBeats,
      difficulty: isNum(c.difficulty) ? c.difficulty : 2,
      file: `${c.id}.json`,
      url: `/api/choreos/${c.id}`,
      source: 'upload',
      uploadedAt,
    };
  }

  _runMeta(run) {
    return {
      id: run.id,
      choreoId: run.choreoId,
      player: typeof run.player === 'string' ? run.player : '',
      score: isNum(run.score) ? run.score : null,
      mode: typeof run.mode === 'string' ? run.mode : 'solo',
      playedAt: typeof run.playedAt === 'string' ? run.playedAt : null,
      receivedAt: run.receivedAt || null,
      fps: run.fps,
      frameCount: Array.isArray(run.frames) ? run.frames.length : (Array.isArray(run.head) ? run.head.length : 0),
      url: `/api/runs/${run.id}`,
    };
  }

  // -- choreographies ------------------------------------------------------------------------

  async saveChoreo(c) {
    const text = JSON.stringify(c);
    await fsp.mkdir(this.choreoDir, { recursive: true });
    await writeFileAtomic(path.join(this.choreoDir, `${c.id}.json`), text);
    const entry = this._choreoEntry(c, new Date().toISOString());
    this.uploaded.set(c.id, entry);
    return entry;
  }

  async deleteChoreo(id) {
    if (!this.uploaded.has(id)) return false;
    this.uploaded.delete(id);
    await fsp.rm(path.join(this.choreoDir, `${id}.json`), { force: true });
    return true;
  }

  uploadedChoreoFile(id) {
    return this.uploaded.has(id) ? path.join(this.choreoDir, `${id}.json`) : null;
  }

  // -- results ---------------------------------------------------------------------------------

  addResult(r) {
    const stored = { ...r };
    stored.id = timeId('r');
    stored.receivedAt = new Date().toISOString();
    if (typeof stored.playedAt !== 'string') stored.playedAt = stored.receivedAt;
    if (typeof stored.player === 'string') stored.player = stored.player.slice(0, 64);
    this.results.push(stored);
    if (this.results.length > MAX_RESULTS) this.results.splice(0, this.results.length - MAX_RESULTS);
    this._scheduleResultsWrite();
    return stored;
  }

  listResults({ choreoId = null, limit = 100, player = null } = {}) {
    const out = [];
    for (let i = this.results.length - 1; i >= 0 && out.length < limit; i--) {
      const r = this.results[i];
      if (choreoId && r.choreoId !== choreoId) continue;
      if (player && r.player !== player) continue;
      out.push(r);
    }
    return out;
  }

  _scheduleResultsWrite() {
    this._resultsDirty = true;
    if (this._resultsTimer) return;
    this._resultsTimer = setTimeout(() => {
      this._resultsTimer = null;
      this.flush().catch((e) => this.log.warn(`Ergebnisse konnten nicht gespeichert werden: ${e.message}`));
    }, 200);
    this._resultsTimer.unref();
  }

  async flush() {
    if (!this._resultsDirty) return;
    this._resultsDirty = false;
    this._writing = this._writing.then(() => writeFileAtomic(this.resultsFile, JSON.stringify({ results: this.results })));
    await this._writing;
  }

  // -- runs ------------------------------------------------------------------------------------

  async saveRun(run) {
    const stored = { ...run };
    stored.id = timeId('run');
    stored.receivedAt = new Date().toISOString();
    if (typeof stored.playedAt !== 'string') stored.playedAt = stored.receivedAt;
    if (typeof stored.player === 'string') stored.player = stored.player.slice(0, 64);
    const dir = path.join(this.runDir, stored.choreoId);
    await fsp.mkdir(dir, { recursive: true });
    await writeFileAtomic(path.join(dir, `${stored.id}.json`), JSON.stringify(stored));
    const meta = this._runMeta(stored);
    this.runs.set(stored.id, meta);
    // cap per choreography: drop the oldest (by receivedAt)
    const same = this.listRuns({ choreoId: stored.choreoId, limit: Infinity, oldestFirst: true });
    const removed = [];
    while (same.length > MAX_RUNS_PER_CHOREO) {
      const old = same.shift();
      this.runs.delete(old.id);
      removed.push(old.id);
      await fsp.rm(path.join(dir, `${old.id}.json`), { force: true });
    }
    return { meta, removed };
  }

  listRuns({ choreoId = null, limit = 100, oldestFirst = false } = {}) {
    const all = [];
    for (const m of this.runs.values()) if (!choreoId || m.choreoId === choreoId) all.push(m);
    all.sort((a, b) => (a.receivedAt || '').localeCompare(b.receivedAt || '') || a.id.localeCompare(b.id));
    if (!oldestFirst) all.reverse();
    return all.length > limit ? all.slice(0, limit) : all;
  }

  runFile(id) {
    const m = this.runs.get(id);
    return m ? path.join(this.runDir, m.choreoId, `${id}.json`) : null;
  }

  async deleteRun(id) {
    const file = this.runFile(id);
    if (!file) return false;
    this.runs.delete(id);
    await fsp.rm(file, { force: true });
    return true;
  }
}

// ---------------------------------------------------------------------------------------------
// Certificate

function certFilesOf(dataDir) {
  const dir = path.join(dataDir, 'cert');
  return { dir, key: path.join(dir, 'server.key'), cert: path.join(dir, 'server.crt'), meta: path.join(dir, 'meta.json') };
}

/** Load or (re)generate the self-signed certificate; returns { key, cert, generated, sans }. */
export async function ensureCertificate(dataDir, ips, log) {
  const files = certFilesOf(dataDir);
  const wanted = ['localhost', '127.0.0.1', ...ips];
  const meta = await readJSON(files.meta, null);
  const now = Date.now();
  if (meta && Array.isArray(meta.sans) && fs.existsSync(files.key) && fs.existsSync(files.cert)) {
    const missing = wanted.filter((h) => !meta.sans.includes(h));
    const expiresSoon = !meta.notAfter || Date.parse(meta.notAfter) - now < 30 * 86400e3;
    if (missing.length === 0 && !expiresSoon) {
      return { key: await fsp.readFile(files.key, 'utf8'), cert: await fsp.readFile(files.cert, 'utf8'), generated: false, sans: meta.sans };
    }
    log.info(missing.length ? `Neue LAN-Adresse(n) ${missing.join(', ')} - erzeuge Zertifikat neu` : 'Zertifikat laeuft ab - erzeuge es neu');
  }
  const { default: selfsigned } = await import('selfsigned');
  log.info('Erzeuge selbstsigniertes Zertifikat (einmalig, dauert ein paar Sekunden) ...');
  const altNames = [];
  for (const h of wanted) {
    if (/^\d+\.\d+\.\d+\.\d+$/.test(h)) altNames.push({ type: 7, ip: h });
    else altNames.push({ type: 2, value: h });
  }
  const days = 3650;
  const pems = selfsigned.generate(
    [{ name: 'commonName', value: 'Dancing Points VR' }, { name: 'organizationName', value: 'Dancing Points VR LAN' }],
    {
      keySize: 2048,
      days,
      algorithm: 'sha256',
      extensions: [
        { name: 'basicConstraints', cA: true },
        { name: 'keyUsage', keyCertSign: true, digitalSignature: true, nonRepudiation: true, keyEncipherment: true, dataEncipherment: true },
        { name: 'extKeyUsage', serverAuth: true, clientAuth: true },
        { name: 'subjectAltName', altNames },
      ],
    }
  );
  await fsp.mkdir(files.dir, { recursive: true });
  await fsp.writeFile(files.key, pems.private, { mode: 0o600 });
  await fsp.writeFile(files.cert, pems.cert);
  await fsp.writeFile(files.meta, JSON.stringify({
    sans: wanted,
    createdAt: new Date(now).toISOString(),
    notAfter: new Date(now + days * 86400e3).toISOString(),
    fingerprint: pems.fingerprint,
  }, null, 2));
  return { key: pems.private, cert: pems.cert, generated: true, sans: wanted };
}

// ---------------------------------------------------------------------------------------------
// HTTP helpers

function sendJSON(res, status, obj, extraHeaders) {
  const body = JSON.stringify(obj);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
    'Cache-Control': 'no-cache',
    ...extraHeaders,
  });
  res.end(body);
}

function sendError(res, status, message, errors) {
  const obj = { error: message };
  if (errors) obj.errors = errors;
  sendJSON(res, status, obj);
}

function readBody(req, limit) {
  return new Promise((resolve, reject) => {
    const declared = Number(req.headers['content-length']);
    if (Number.isFinite(declared) && declared > limit) {
      const err = new Error('body too large');
      err.status = 413;
      req.resume();
      reject(err);
      return;
    }
    const chunks = [];
    let size = 0;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > limit) {
        const err = new Error('body too large');
        err.status = 413;
        req.destroy();
        reject(err);
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

async function readJSONBody(req, limit) {
  const buf = await readBody(req, limit);
  if (buf.length === 0) {
    const err = new Error('empty body');
    err.status = 400;
    throw err;
  }
  try {
    return JSON.parse(buf.toString('utf8'));
  } catch (e) {
    const err = new Error(`invalid JSON: ${e.message}`);
    err.status = 400;
    throw err;
  }
}

function sendFile(req, res, file, stat, headers) {
  const ext = path.extname(file).toLowerCase();
  const base = path.basename(file);
  const type = MIME[ext] || 'application/octet-stream';
  const etag = `W/"${stat.size.toString(16)}-${Math.floor(stat.mtimeMs).toString(16)}"`;
  const h = {
    'Content-Type': type,
    'ETag': etag,
    'Last-Modified': stat.mtime.toUTCString(),
    'Accept-Ranges': 'bytes',
    'Cache-Control': NO_CACHE_EXT.has(ext) || NO_CACHE_NAMES.has(base) ? 'no-cache' : 'public, max-age=3600',
    ...headers,
  };
  if (req.headers['if-none-match'] === etag) {
    res.writeHead(304, h);
    res.end();
    return;
  }
  let start = 0;
  let end = stat.size - 1;
  let status = 200;
  const range = req.headers.range;
  if (range && stat.size > 0) {
    const m = /^bytes=(\d*)-(\d*)$/.exec(range);
    if (m && (m[1] !== '' || m[2] !== '')) {
      if (m[1] === '') { start = Math.max(0, stat.size - Number(m[2])); }
      else { start = Number(m[1]); if (m[2] !== '') end = Math.min(stat.size - 1, Number(m[2])); }
      if (start > end || start >= stat.size) {
        res.writeHead(416, { 'Content-Range': `bytes */${stat.size}` });
        res.end();
        return;
      }
      status = 206;
      h['Content-Range'] = `bytes ${start}-${end}/${stat.size}`;
    }
  }
  h['Content-Length'] = end - start + 1;
  res.writeHead(status, h);
  if (req.method === 'HEAD' || stat.size === 0) {
    res.end();
    return;
  }
  const stream = fs.createReadStream(file, { start, end });
  stream.on('error', () => { if (!res.headersSent) res.writeHead(500); res.end(); });
  stream.pipe(res);
}

// ---------------------------------------------------------------------------------------------
// Rooms / WebSocket relay

class Relay {
  constructor(log) {
    this.log = log;
    this.clients = new Map();   // id -> client
    this.rooms = new Map();     // code -> room
    this.wss = new WebSocketServer({ noServer: true, maxPayload: MAX_WS_MESSAGE, perMessageDeflate: false });
    this.wss.on('connection', (ws, req) => this._onConnection(ws, req));
    this._heartbeat = setInterval(() => this._beat(), HEARTBEAT_MS);
    this._heartbeat.unref();
  }

  handleUpgrade(req, socket, head) {
    this.wss.handleUpgrade(req, socket, head, (ws) => this.wss.emit('connection', ws, req));
  }

  now() {
    return Date.now();
  }

  _beat() {
    for (const c of this.clients.values()) {
      if (!c.alive) { c.ws.terminate(); continue; }
      c.alive = false;
      try { c.ws.ping(); } catch (e) { /* ignore */ }
    }
  }

  _newRoomCode() {
    for (let attempt = 0; attempt < 1000; attempt++) {
      let code = '';
      const bytes = crypto.randomBytes(ROOM_CODE_LEN);
      for (let i = 0; i < ROOM_CODE_LEN; i++) code += ROOM_ALPHABET[bytes[i] % ROOM_ALPHABET.length];
      if (!this.rooms.has(code)) return code;
    }
    throw new Error('no free room code');
  }

  _onConnection(ws, req) {
    const client = {
      id: shortId(4),
      ws,
      name: 'Spieler',
      room: null,
      alive: true,
      tokens: RATE_BURST,
      lastRefill: this.now(),
      dropped: 0,
      lastRateError: 0,
      remote: req && req.socket ? req.socket.remoteAddress : '',
    };
    this.clients.set(client.id, client);
    ws.on('pong', () => { client.alive = true; });
    ws.on('message', (data, isBinary) => this._onMessage(client, data, isBinary));
    ws.on('close', () => this._onClose(client));
    ws.on('error', (err) => this.log.debug(`ws ${client.id}: ${err.message}`));
    this.send(client, { type: 'hello', id: client.id, serverTime: this.now(), version: SERVER_VERSION, maxPlayers: MAX_ROOM_PLAYERS });
    this.log.debug(`ws connect ${client.id} from ${client.remote}`);
  }

  _onClose(client) {
    this.clients.delete(client.id);
    this._leaveRoom(client, 'closed');
    this.log.debug(`ws close ${client.id}`);
  }

  send(client, obj) {
    if (client.ws.readyState !== 1) return false;
    try {
      client.ws.send(JSON.stringify(obj));
      return true;
    } catch (e) {
      return false;
    }
  }

  sendError(client, code, message, extra) {
    this.send(client, { type: 'error', code, message, ...extra });
  }

  /** Send `obj` to every member of the room (optionally except one client). */
  broadcast(room, obj, except = null) {
    const text = JSON.stringify(obj);
    let n = 0;
    for (const c of room.players) {
      if (c === except || c.ws.readyState !== 1) continue;
      try { c.ws.send(text); n++; } catch (e) { /* ignore */ }
    }
    return n;
  }

  roomInfo(room) {
    return {
      code: room.code,
      choreoId: room.choreoId,
      players: room.players.map((c) => ({ id: c.id, name: c.name, host: c === room.host })),
      maxPlayers: MAX_ROOM_PLAYERS,
      startAt: room.startAt,
      createdAt: room.createdAt,
    };
  }

  listRooms() {
    return [...this.rooms.values()].map((r) => this.roomInfo(r));
  }

  _rateLimited(client) {
    const now = this.now();
    client.tokens = Math.min(RATE_BURST, client.tokens + ((now - client.lastRefill) / 1000) * RATE_PER_SEC);
    client.lastRefill = now;
    if (client.tokens >= 1) {
      client.tokens -= 1;
      return false;
    }
    client.dropped++;
    if (now - client.lastRateError > 1000) {
      client.lastRateError = now;
      this.sendError(client, 'RATE_LIMIT', `Zu viele Nachrichten (max. ${RATE_PER_SEC}/s)`, { dropped: client.dropped });
    }
    return true;
  }

  _onMessage(client, data, isBinary) {
    if (isBinary) { this.sendError(client, 'BAD_MESSAGE', 'Nur Text-Nachrichten (JSON) erlaubt'); return; }
    if (this._rateLimited(client)) return;
    let msg;
    try {
      msg = JSON.parse(typeof data === 'string' ? data : data.toString('utf8'));
    } catch (e) {
      this.sendError(client, 'BAD_JSON', 'Nachricht ist kein gueltiges JSON');
      return;
    }
    if (!msg || typeof msg !== 'object' || typeof msg.type !== 'string' || msg.type.length > 32) {
      this.sendError(client, 'BAD_MESSAGE', 'Nachricht braucht ein Feld "type"');
      return;
    }
    switch (msg.type) {
      case 'ping':
        this.send(client, { type: 'pong', t0: msg.t0, t1: this.now() });
        return;
      case 'create': this._create(client, msg); return;
      case 'join': this._join(client, msg); return;
      case 'leave': this._leaveRoom(client, 'leave'); this.send(client, { type: 'left', room: null }); return;
      case 'setChoreo': this._setChoreo(client, msg); return;
      case 'start': this._start(client, msg); return;
      case 'hello': case 'pong': case 'error': case 'room': case 'peer': case 'left':
        this.sendError(client, 'BAD_MESSAGE', `Nachrichtentyp "${msg.type}" ist dem Server vorbehalten`);
        return;
      default:
        this._relay(client, msg);
    }
  }

  _create(client, msg) {
    this._leaveRoom(client, 'switch');
    client.name = cleanName(msg.name, client.name);
    const room = {
      code: this._newRoomCode(),
      host: client,
      players: [client],
      choreoId: typeof msg.choreoId === 'string' && ID_PATTERN.test(msg.choreoId) ? msg.choreoId : null,
      startAt: null,
      createdAt: new Date().toISOString(),
    };
    this.rooms.set(room.code, room);
    client.room = room;
    this.send(client, { type: 'room', ...this.roomInfo(room), you: client.id, host: true });
    this.log.info(`Raum ${room.code} erstellt von ${client.name} (${client.id})`);
  }

  _join(client, msg) {
    const code = typeof msg.code === 'string' ? msg.code.trim().toUpperCase() : '';
    const room = this.rooms.get(code);
    if (!room) { this.sendError(client, 'ROOM_NOT_FOUND', `Raum ${code || '?'} gibt es nicht`); return; }
    if (room.players.includes(client)) { this.send(client, { type: 'room', ...this.roomInfo(room), you: client.id, host: room.host === client }); return; }
    if (room.players.length >= MAX_ROOM_PLAYERS) { this.sendError(client, 'ROOM_FULL', `Raum ${code} ist voll`); return; }
    this._leaveRoom(client, 'switch');
    client.name = cleanName(msg.name, client.name);
    room.players.push(client);
    client.room = room;
    this.send(client, { type: 'room', ...this.roomInfo(room), you: client.id, host: false });
    this.broadcast(room, { type: 'peer', event: 'joined', room: room.code, player: { id: client.id, name: client.name, host: false }, players: this.roomInfo(room).players }, client);
    this.log.info(`${client.name} (${client.id}) ist Raum ${room.code} beigetreten`);
  }

  _leaveRoom(client, reason) {
    const room = client.room;
    if (!room) return;
    client.room = null;
    const idx = room.players.indexOf(client);
    if (idx >= 0) room.players.splice(idx, 1);
    if (room.players.length === 0) {
      this.rooms.delete(room.code);
      this.log.info(`Raum ${room.code} geschlossen (${reason})`);
      return;
    }
    let hostChanged = false;
    if (room.host === client) {
      room.host = room.players[0];
      hostChanged = true;
    }
    const players = this.roomInfo(room).players;
    this.broadcast(room, { type: 'peer', event: 'left', room: room.code, player: { id: client.id, name: client.name, host: false }, players, reason });
    if (hostChanged) {
      this.broadcast(room, { type: 'peer', event: 'host', room: room.code, player: { id: room.host.id, name: room.host.name, host: true }, players });
    }
  }

  _setChoreo(client, msg) {
    const room = client.room;
    if (!room) { this.sendError(client, 'NOT_IN_ROOM', 'Du bist in keinem Raum'); return; }
    if (room.host !== client) { this.sendError(client, 'NOT_HOST', 'Nur der Host darf den Tanz waehlen'); return; }
    if (typeof msg.choreoId !== 'string' || !ID_PATTERN.test(msg.choreoId)) { this.sendError(client, 'BAD_REQUEST', 'Ungueltige choreoId'); return; }
    room.choreoId = msg.choreoId;
    this.broadcast(room, { type: 'choreo', room: room.code, from: client.id, choreoId: room.choreoId });
  }

  _start(client, msg) {
    const room = client.room;
    if (!room) { this.sendError(client, 'NOT_IN_ROOM', 'Du bist in keinem Raum'); return; }
    if (room.host !== client) { this.sendError(client, 'NOT_HOST', 'Nur der Host darf starten'); return; }
    const choreoId = typeof msg.choreoId === 'string' && ID_PATTERN.test(msg.choreoId) ? msg.choreoId : room.choreoId;
    if (!choreoId) { this.sendError(client, 'BAD_REQUEST', 'Kein Tanz gewaehlt (choreoId fehlt)'); return; }
    let delay = isNum(msg.delayMs) ? msg.delayMs : DEFAULT_START_DELAY_MS;
    delay = Math.min(30000, Math.max(200, delay));
    room.choreoId = choreoId;
    room.startAt = this.now() + delay;
    this.broadcast(room, {
      type: 'start',
      room: room.code,
      from: client.id,
      choreoId,
      startAt: room.startAt,
      delayMs: delay,
      serverTime: this.now(),
      players: this.roomInfo(room).players,
    });
    this.log.info(`Raum ${room.code}: Start ${choreoId} in ${delay} ms`);
  }

  /** Everything else (state, result, abort, emotes, ...) goes to the other room members. */
  _relay(client, msg) {
    const room = client.room;
    if (!room) { this.sendError(client, 'NOT_IN_ROOM', 'Du bist in keinem Raum'); return; }
    msg.from = client.id;
    msg.room = room.code;
    this.broadcast(room, msg, client);
  }

  close() {
    clearInterval(this._heartbeat);
    for (const c of this.clients.values()) {
      try { c.ws.close(1001, 'server shutdown'); } catch (e) { /* ignore */ }
    }
    this.wss.close();
  }
}

// ---------------------------------------------------------------------------------------------
// The server

export class DancingPointsServer {
  constructor(opts) {
    this.opts = opts;
    this.log = makeLogger(opts);
    this.store = new DataStore(opts.data, this.log);
    this.relay = new Relay(this.log);
    this.httpServer = null;
    this.httpsServer = null;
    this.startedAt = Date.now();
    this.appVersion = SERVER_VERSION;
    this.validatorKind = 'fallback';
    this.certInfo = null;
    this._closed = false;
  }

  async start() {
    const o = this.opts;
    if (!safeStatSync(o.dir) || !safeStatSync(o.dir).isDirectory()) {
      throw new Error(`App-Verzeichnis nicht gefunden: ${o.dir} (Option --dir)`);
    }
    await fsp.mkdir(o.data, { recursive: true });
    await this.store.init();
    this.validatorKind = await loadChoreoValidator();
    try {
      const cfg = await import(new URL('../webxr/src/config.js', import.meta.url).href);
      if (typeof cfg.APP_VERSION === 'string') this.appVersion = cfg.APP_VERSION;
    } catch (e) { /* stand-alone deployment */ }

    const handler = (req, res) => this._handle(req, res).catch((err) => {
      this.log.warn(`Fehler bei ${req.method} ${req.url}: ${err && err.stack ? err.stack : err}`);
      if (!res.headersSent) sendError(res, 500, 'internal error');
      else res.end();
    });
    const upgrade = (req, socket, head) => {
      let pathname = '/';
      try { pathname = new URL(req.url, 'http://localhost').pathname; } catch (e) { /* ignore */ }
      if (pathname !== WS_PATH) { socket.destroy(); return; }
      this.relay.handleUpgrade(req, socket, head);
    };

    const ips = lanAddresses();
    if (o.https) {
      let key, cert;
      if (o.cert && o.key) {
        key = await fsp.readFile(o.key, 'utf8');
        cert = await fsp.readFile(o.cert, 'utf8');
        this.certInfo = { generated: false, sans: [], custom: true };
      } else {
        this.certInfo = await ensureCertificate(o.data, ips, this.log);
        key = this.certInfo.key;
        cert = this.certInfo.cert;
      }
      this.httpsServer = https.createServer({ key, cert }, handler);
      this.httpsServer.on('upgrade', upgrade);
      await listen(this.httpsServer, o.port, o.host);
    }
    if (o.http) {
      this.httpServer = http.createServer(handler);
      this.httpServer.on('upgrade', upgrade);
      await listen(this.httpServer, o.httpPort, o.host);
    }
    return this;
  }

  get httpsPort() { return this.httpsServer ? this.httpsServer.address().port : null; }
  get httpPort() { return this.httpServer ? this.httpServer.address().port : null; }

  /** URLs a headset in the WLAN can open (HTTPS first, the Quest needs it for WebXR). */
  urls() {
    const ips = lanAddresses();
    const out = [];
    if (this.httpsServer) for (const ip of ips) out.push(`https://${ip}:${this.httpsPort}/`);
    if (this.httpServer) for (const ip of ips) out.push(`http://${ip}:${this.httpPort}/`);
    if (this.httpsServer) out.push(`https://localhost:${this.httpsPort}/`);
    if (this.httpServer) out.push(`http://localhost:${this.httpPort}/`);
    return out;
  }

  info() {
    const primary = this.urls()[0] || null;
    return {
      name: 'Dancing Points VR Server',
      version: this.appVersion,
      serverVersion: SERVER_VERSION,
      urls: this.urls(),
      primaryUrl: primary,
      https: Boolean(this.httpsServer),
      http: Boolean(this.httpServer),
      httpsPort: this.httpsPort,
      httpPort: this.httpPort,
      wsPath: WS_PATH,
      serverTime: Date.now(),
      uptimeSec: Math.round((Date.now() - this.startedAt) / 1000),
      rooms: this.relay.rooms.size,
      clients: this.relay.clients.size,
      choreoValidator: this.validatorKind,
      limits: { wsMessageBytes: MAX_WS_MESSAGE, wsMessagesPerSec: RATE_PER_SEC, runsPerChoreo: MAX_RUNS_PER_CHOREO, maxPlayers: MAX_ROOM_PLAYERS },
    };
  }

  async close() {
    if (this._closed) return;
    this._closed = true;
    this.relay.close();
    await this.store.flush().catch(() => {});
    const closers = [];
    for (const s of [this.httpServer, this.httpsServer]) {
      if (!s) continue;
      closers.push(new Promise((resolve) => s.close(() => resolve())));
      if (typeof s.closeAllConnections === 'function') s.closeAllConnections();
    }
    await Promise.all(closers);
  }

  // -- request routing ---------------------------------------------------------------------------

  _commonHeaders(res) {
    if (this.opts.coep) {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    }
    // lets other origins (e.g. the app served from GitHub Pages with ?server=) embed our files
    res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
    res.setHeader('X-Content-Type-Options', 'nosniff');
  }

  async _handle(req, res) {
    const t0 = process.hrtime.bigint();
    if (this.opts.verbose) {
      res.on('finish', () => {
        const ms = Number(process.hrtime.bigint() - t0) / 1e6;
        this.log.info(`${req.method} ${req.url} -> ${res.statusCode} (${ms.toFixed(1)} ms)`);
      });
    }
    this._commonHeaders(res);
    let url;
    try {
      url = new URL(req.url, 'http://localhost');
    } catch (e) {
      sendError(res, 400, 'bad url');
      return;
    }
    if (url.pathname === '/api' || url.pathname.startsWith('/api/')) {
      await this._api(req, res, url);
      return;
    }
    if (url.pathname === WS_PATH) {
      res.writeHead(426, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('WebSocket endpoint - use a ws:// or wss:// connection');
      return;
    }
    await this._static(req, res, url);
  }

  async _static(req, res, url) {
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405, { 'Allow': 'GET, HEAD' });
      res.end();
      return;
    }
    let pathname;
    try {
      pathname = decodeURIComponent(url.pathname);
    } catch (e) {
      sendError(res, 400, 'bad path');
      return;
    }
    if (pathname.includes('\0')) { sendError(res, 400, 'bad path'); return; }
    const root = this.opts.dir;
    let file = path.resolve(root, `.${path.posix.normalize(`/${pathname}`)}`);
    if (file !== root && !file.startsWith(root + path.sep)) { sendError(res, 403, 'forbidden'); return; }
    let stat = safeStatSync(file);
    if (stat && stat.isDirectory()) {
      if (!pathname.endsWith('/')) {
        res.writeHead(301, { Location: `${url.pathname}/${url.search}` });
        res.end();
        return;
      }
      file = path.join(file, 'index.html');
      stat = safeStatSync(file);
    }
    if (!stat || !stat.isFile()) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-cache' });
      res.end('404 Not Found');
      return;
    }
    sendFile(req, res, file, stat);
  }

  async _api(req, res, url) {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    res.setHeader('Access-Control-Max-Age', '600');
    if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
    const parts = url.pathname.split('/').filter(Boolean);   // ['api', 'choreos', ':id']
    const resource = parts[1] || '';
    const id = parts[2];
    if (parts.length > 3) { sendError(res, 404, 'not found'); return; }
    const method = req.method;
    try {
      switch (resource) {
        case 'info':
          if (method !== 'GET' && method !== 'HEAD') return methodNotAllowed(res, 'GET');
          return sendJSON(res, 200, this.info());
        case 'health':
          return sendJSON(res, 200, { ok: true, serverTime: Date.now() });
        case 'rooms':
          if (method !== 'GET') return methodNotAllowed(res, 'GET');
          return sendJSON(res, 200, { rooms: this.relay.listRooms() });
        case 'choreos':
          return await this._apiChoreos(req, res, url, method, id);
        case 'results':
          return await this._apiResults(req, res, url, method, id);
        case 'runs':
          return await this._apiRuns(req, res, url, method, id);
        default:
          return sendError(res, 404, 'not found');
      }
    } catch (err) {
      if (err && err.status) return sendError(res, err.status, err.message);
      throw err;
    }
  }

  async shippedIndex() {
    const idx = await readJSON(path.join(this.opts.dir, 'choreos', 'index.json'), null);
    const list = idx && Array.isArray(idx.choreos) ? idx.choreos : [];
    return list
      .filter((e) => e && typeof e.id === 'string' && typeof e.file === 'string')
      .map((e) => ({ ...e, url: `/choreos/${e.file}`, source: 'shipped' }));
  }

  /** Merged index: shipped entries (an uploaded one with the same id replaces it) + uploads. */
  async choreoIndex() {
    const shipped = await this.shippedIndex();
    const out = [];
    const seen = new Set();
    for (const e of shipped) {
      const up = this.store.uploaded.get(e.id);
      out.push(up ? { ...up, replaces: 'shipped' } : e);
      seen.add(e.id);
    }
    for (const up of this.store.uploaded.values()) if (!seen.has(up.id)) out.push(up);
    return out;
  }

  async _apiChoreos(req, res, url, method, id) {
    if (!id) {
      if (method === 'GET' || method === 'HEAD') return sendJSON(res, 200, { choreos: await this.choreoIndex() });
      if (method === 'POST') {
        const body = await readJSONBody(req, MAX_BODY_CHOREO);
        const v = validateChoreoImpl(body);
        if (!v.ok) return sendJSON(res, 400, { error: 'invalid choreography', errors: v.errors.slice(0, 20) });
        const entry = await this.store.saveChoreo(body);
        this.log.info(`Tanz hochgeladen: ${entry.id} ("${entry.title}")`);
        return sendJSON(res, 201, { ok: true, id: entry.id, url: entry.url, entry });
      }
      return methodNotAllowed(res, 'GET, POST');
    }
    if (!ID_PATTERN.test(id)) return sendError(res, 400, 'bad id');
    if (method === 'DELETE') {
      const ok = await this.store.deleteChoreo(id);
      return ok ? sendJSON(res, 200, { ok: true, id }) : sendError(res, 404, 'not an uploaded choreography');
    }
    if (method !== 'GET' && method !== 'HEAD') return methodNotAllowed(res, 'GET, DELETE');
    let file = this.store.uploadedChoreoFile(id);
    if (!file) {
      const entry = (await this.shippedIndex()).find((e) => e.id === id);
      if (entry) {
        const candidate = path.resolve(this.opts.dir, 'choreos', entry.file);
        if (candidate.startsWith(path.resolve(this.opts.dir, 'choreos') + path.sep)) file = candidate;
      }
    }
    const stat = file ? safeStatSync(file) : null;
    if (!stat || !stat.isFile()) return sendError(res, 404, 'choreography not found');
    return sendFile(req, res, file, stat, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-cache' });
  }

  async _apiResults(req, res, url, method, id) {
    if (id) return sendError(res, 404, 'not found');
    if (method === 'GET' || method === 'HEAD') {
      const limit = clampInt(url.searchParams.get('limit'), 1, 1000, 100);
      const choreoId = url.searchParams.get('choreoId');
      const player = url.searchParams.get('player');
      return sendJSON(res, 200, { results: this.store.listResults({ choreoId, limit, player }) });
    }
    if (method === 'POST') {
      const body = await readJSONBody(req, MAX_BODY_DEFAULT);
      const v = validateResult(body);
      if (!v.ok) return sendJSON(res, 400, { error: 'invalid result', errors: v.errors });
      const stored = this.store.addResult(body);
      return sendJSON(res, 201, { ok: true, id: stored.id, result: stored });
    }
    return methodNotAllowed(res, 'GET, POST');
  }

  async _apiRuns(req, res, url, method, id) {
    if (!id) {
      if (method === 'GET' || method === 'HEAD') {
        const limit = clampInt(url.searchParams.get('limit'), 1, 1000, 100);
        const choreoId = url.searchParams.get('choreoId');
        const runs = this.store.listRuns({ choreoId, limit });
        return sendJSON(res, 200, { runs });
      }
      if (method === 'POST') {
        const body = await readJSONBody(req, MAX_BODY_CHOREO);
        const v = validateRun(body);
        if (!v.ok) return sendJSON(res, 400, { error: 'invalid run', errors: v.errors });
        const { meta, removed } = await this.store.saveRun(body);
        return sendJSON(res, 201, { ok: true, id: meta.id, url: meta.url, run: meta, removed });
      }
      return methodNotAllowed(res, 'GET, POST');
    }
    if (!/^[a-z0-9]+$/.test(id)) return sendError(res, 400, 'bad id');
    if (method === 'DELETE') {
      const ok = await this.store.deleteRun(id);
      return ok ? sendJSON(res, 200, { ok: true, id }) : sendError(res, 404, 'run not found');
    }
    if (method !== 'GET' && method !== 'HEAD') return methodNotAllowed(res, 'GET, DELETE');
    const file = this.store.runFile(id);
    const stat = file ? safeStatSync(file) : null;
    if (!stat || !stat.isFile()) return sendError(res, 404, 'run not found');
    return sendFile(req, res, file, stat, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-cache' });
  }
}

function methodNotAllowed(res, allow) {
  res.writeHead(405, { 'Allow': allow, 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify({ error: 'method not allowed' }));
}

function clampInt(v, min, max, fallback) {
  if (v === null || v === undefined || v === '') return fallback;
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.floor(n)));
}

function listen(server, port, host) {
  return new Promise((resolve, reject) => {
    const onError = (err) => {
      server.removeListener('listening', onListening);
      if (err && err.code === 'EADDRINUSE') err.message = `Port ${port} ist schon belegt (${err.code}). Anderen Port waehlen: --port / --http-port`;
      reject(err);
    };
    const onListening = () => { server.removeListener('error', onError); resolve(server.address()); };
    server.once('error', onError);
    server.once('listening', onListening);
    server.listen(port, host);
  });
}

function makeLogger(opts) {
  const ts = () => new Date().toISOString().slice(11, 19);
  const quiet = Boolean(opts.quiet);
  const verbose = Boolean(opts.verbose);
  return {
    info: (m) => { if (!quiet) process.stdout.write(`[dp ${ts()}] ${m}\n`); },
    warn: (m) => { process.stderr.write(`[dp ${ts()}] WARNUNG: ${m}\n`); },
    debug: (m) => { if (verbose && !quiet) process.stdout.write(`[dp ${ts()}] ${m}\n`); },
  };
}

// ---------------------------------------------------------------------------------------------
// main

async function main() {
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (err) {
    process.stderr.write(`${err.message}\n`);
    printHelp();
    process.exit(2);
  }
  if (opts.help) { printHelp(); return; }
  if (opts.version) { process.stdout.write(`${SERVER_VERSION}\n`); return; }

  const server = new DancingPointsServer(opts);
  try {
    await server.start();
  } catch (err) {
    process.stderr.write(`Start fehlgeschlagen: ${err.message}\n`);
    process.exit(1);
  }
  const log = server.log;
  const info = server.info();
  log.info(`Dancing Points VR - LAN-Server v${SERVER_VERSION} (App v${info.version})`);
  log.info(`App-Verzeichnis: ${opts.dir}`);
  log.info(`Datenverzeichnis: ${opts.data}  (Tänze: ${server.store.uploaded.size} hochgeladen, ${server.store.runs.size} Läufe, ${server.store.results.length} Ergebnisse)`);
  if (server.httpsServer) {
    const cert = server.certInfo || {};
    log.info(`HTTPS auf Port ${server.httpsPort}${cert.custom ? ' (eigenes Zertifikat)' : cert.generated ? ' (neues selbstsigniertes Zertifikat)' : ' (selbstsigniertes Zertifikat)'}`);
  }
  if (server.httpServer) log.info(`HTTP auf Port ${server.httpPort} (nur fuer localhost/Tests - WebXR braucht HTTPS)`);
  const urls = server.urls();
  if (urls.length) {
    log.info('Auf der Quest im Browser oeffnen:');
    for (const u of urls) log.info(`    ${u}`);
    if (server.httpsServer) log.info('Beim ersten Aufruf die Zertifikatswarnung bestaetigen ("Erweitert" -> "Trotzdem fortfahren").');
  }
  if (!lanAddresses().length) log.warn('Keine LAN-IPv4-Adresse gefunden - ist das WLAN verbunden?');
  // machine-readable readiness line (tests / e2e runner)
  process.stdout.write(`READY ${JSON.stringify({ https: server.httpsPort, http: server.httpPort, urls, data: opts.data })}\n`);

  const shutdown = (sig) => {
    log.info(`${sig} - beende ...`);
    server.close().then(() => process.exit(0), () => process.exit(0));
    setTimeout(() => process.exit(0), 2000).unref();
  };
  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGTERM', () => shutdown('SIGTERM'));
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === __filename;
if (isMain) {
  main().catch((err) => {
    process.stderr.write(`${err && err.stack ? err.stack : err}\n`);
    process.exit(1);
  });
}
