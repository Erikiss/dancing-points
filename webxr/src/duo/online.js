// online.js - OnlineDuo: WebSocket client of the duo-mode relay in server/server.js
// (DESIGN.md section 9). Rooms with 4-letter codes, NTP-style clock sync (median of 5 pings),
// host-driven start with a server-stamped `startAt`, 15 Hz state upload (one reused message
// object) and interpolation of the partner's states for rendering. Works in the browser with
// the global WebSocket and in Node with the `ws` package passed as `WebSocketImpl`.
// Pure JS: no three.js, no DOM. Must not: drive the clock or score anything - it only moves
// messages; the app owns the BeatClock and the PlaySession.

import { EventEmitter } from '../util/events.js';
import { createRunSample } from './ghost.js';

export const DEFAULT_STATE_HZ = 15;
export const DEFAULT_PING_COUNT = 5;
export const DEFAULT_INTERPOLATION_DELAY_SEC = 0.1;   // ~1.5 state intervals at 15 Hz
export const STATES = Object.freeze(['idle', 'connecting', 'connected', 'lobby', 'playing', 'closed']);
export const WS_PATH = '/ws';

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const r4 = (x) => Math.round(x * 10000) / 10000;

/** 'https://host:port', 'wss://host:port/ws', 'host:port' ... -> 'wss://host:port/ws'. */
export function toWsUrl(serverUrl) {
  if (!serverUrl) return null;
  let s = String(serverUrl).trim();
  if (!/^[a-z]+:\/\//i.test(s)) s = `wss://${s}`;
  s = s.replace(/^https:\/\//i, 'wss://').replace(/^http:\/\//i, 'ws://');
  s = s.replace(/\/+$/, '');
  if (!/\/ws$/i.test(s)) s += WS_PATH;
  return s;
}

/** Inverse of toWsUrl for REST calls: 'wss://host:port/ws' -> 'https://host:port'. */
export function toHttpUrl(serverUrl) {
  if (!serverUrl) return '';
  let s = String(serverUrl).trim();
  if (!/^[a-z]+:\/\//i.test(s)) s = `https://${s}`;
  s = s.replace(/^wss:\/\//i, 'https://').replace(/^ws:\/\//i, 'http://');
  return s.replace(/\/ws\/?$/i, '').replace(/\/+$/, '');
}

function median(values) {
  const v = values.slice().sort((a, b) => a - b);
  const n = v.length;
  if (n === 0) return 0;
  return n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

function makeSlot() {
  return { t: 0, head: new Float32Array(7), left: new Float32Array(3), right: new Float32Array(3), score: 0, combo: 0, recvAt: 0 };
}

export class OnlineDuo extends EventEmitter {
  /**
   * @param {object} [opts]
   * @param {Function} [opts.WebSocketImpl] WebSocket constructor (Node: `ws`), default global
   * @param {Function} [opts.now] ms clock, default Date.now (must match the server's notion:
   *   wall time; the offset is measured anyway)
   * @param {number} [opts.stateHz=15] maximum sendState rate
   * @param {number} [opts.pingCount=5] pings per syncClock
   * @param {number} [opts.pingTimeoutMs=2000]
   * @param {number} [opts.requestTimeoutMs=5000]
   * @param {number} [opts.interpolationDelaySec=0.1] render delay of the remote player
   * @param {number} [opts.bufferSize=16] remote state ring buffer
   */
  constructor(opts = {}) {
    super();
    this.WebSocketImpl = opts.WebSocketImpl || globalThis.WebSocket || null;
    this.now = typeof opts.now === 'function' ? opts.now : () => Date.now();
    this.stateHz = opts.stateHz > 0 ? opts.stateHz : DEFAULT_STATE_HZ;
    this.pingCount = opts.pingCount > 0 ? opts.pingCount : DEFAULT_PING_COUNT;
    this.pingTimeoutMs = opts.pingTimeoutMs > 0 ? opts.pingTimeoutMs : 2000;
    this.requestTimeoutMs = opts.requestTimeoutMs > 0 ? opts.requestTimeoutMs : 5000;
    this.interpolationDelaySec = isNum(opts.interpolationDelaySec) ? opts.interpolationDelaySec : DEFAULT_INTERPOLATION_DELAY_SEC;

    this.state = 'idle';
    this.url = null;
    this.ws = null;
    this.id = null;
    this.serverVersion = null;
    this.serverTimeAtHello = null;
    this.room = null;            // { code, choreoId, players:[{id,name,host}], maxPlayers, startAt }
    this.offsetMs = 0;           // server clock - local clock
    this.rttMs = 0;
    this.synced = false;
    this.startAt = null;         // server ms (count-in start)
    this.startAtLocalMs = null;
    this.startChoreoId = null;
    this.lastError = null;
    this.stats = { sent: 0, received: 0, droppedRate: 0, remoteStates: 0 };

    this._sendIntervalMs = 1000 / this.stateHz;
    this._lastSendMs = -Infinity;
    this._stateMsg = { type: 'state', t: 0, head: [0, 0, 0, 0, 0, 0, 1], left: [0, 0, 0], right: [0, 0, 0], score: 0, combo: 0 };
    this._pending = new Map();   // message type -> { resolve, reject, timer }
    this._pingPending = null;
    this._closingByUser = false;

    const size = opts.bufferSize > 1 ? Math.floor(opts.bufferSize) : 16;
    this._buf = new Array(size);
    for (let i = 0; i < size; i++) this._buf[i] = makeSlot();
    this._bufCount = 0;
    this._bufHead = -1;          // index of the newest slot
    this._remoteSample = createRunSample();
    this._remoteSample.score = 0;
    this._remoteSample.combo = 0;
    this._remoteSample.age = 0;
    this._qTmp = [0, 0, 0, 1];
  }

  // -- connection ------------------------------------------------------------------------------

  get connected() {
    return Boolean(this.ws) && this.ws.readyState === 1 && this.id !== null;
  }

  get isHost() {
    if (!this.room || !this.id) return false;
    const me = this.room.players.find((p) => p.id === this.id);
    return Boolean(me && me.host);
  }

  /** The other player in the room ({id, name, host}) or null. */
  get peer() {
    if (!this.room) return null;
    return this.room.players.find((p) => p.id !== this.id) || null;
  }

  get roomCode() {
    return this.room ? this.room.code : null;
  }

  /**
   * Connect to a server ('wss://host:port', 'https://host:port' or a full /ws URL). Resolves
   * after the server's `hello` (we know our client id then).
   */
  connect(url) {
    if (!this.WebSocketImpl) return Promise.reject(new Error('WebSocket not available'));
    if (this.ws) this.disconnect();
    this.url = toWsUrl(url);
    this.state = 'connecting';
    this._closingByUser = false;
    return new Promise((resolve, reject) => {
      let ws;
      try {
        ws = new this.WebSocketImpl(this.url);
      } catch (err) {
        this.state = 'idle';
        reject(err);
        return;
      }
      this.ws = ws;
      let settled = false;
      const fail = (err) => {
        if (settled) return;
        settled = true;
        this.state = 'closed';
        reject(err);
      };
      const timer = setTimeout(() => fail(new Error(`connect timeout (${this.url})`)), this.requestTimeoutMs);
      ws.onopen = () => {
        this.state = 'connected';
        this.emit('open', { url: this.url });
      };
      ws.onmessage = (ev) => {
        this._onMessage(ev.data);
        if (!settled && this.id) {
          settled = true;
          clearTimeout(timer);
          resolve(this);
        }
      };
      ws.onerror = (ev) => {
        const err = new Error((ev && ev.message) || (ev && ev.error && ev.error.message) || `WebSocket error (${this.url})`);
        this.lastError = err;
        this.emit('error', { code: 'SOCKET', message: err.message, error: err });
        fail(err);
      };
      ws.onclose = (ev) => {
        clearTimeout(timer);
        const wasConnected = this.id !== null;
        this._onClose(ev);
        if (!settled) fail(new Error(`connection closed (${(ev && ev.code) || 0})`));
        else if (wasConnected && !this._closingByUser) this.emit('disconnected', { code: ev && ev.code, reason: ev && ev.reason });
      };
    });
  }

  /** Close the connection (no reconnect). */
  disconnect(code = 1000, reason = 'bye') {
    this._closingByUser = true;
    const ws = this.ws;
    if (ws) {
      this.ws = null;
      try { ws.close(code, reason); } catch (e) { /* ignore */ }
    }
    this._rejectAll(new Error('disconnected'));
    this.id = null;
    this.room = null;
    this.synced = false;
    this.state = 'closed';
  }

  dispose() {
    this.disconnect();
    this.removeAllListeners();
  }

  _onClose(ev) {
    this.ws = null;
    this.id = null;
    this.room = null;
    this.synced = false;
    this.state = 'closed';
    this._rejectAll(new Error('connection closed'));
    this.emit('close', { code: ev ? ev.code : 0, reason: ev ? ev.reason : '' });
  }

  _rejectAll(err) {
    for (const p of this._pending.values()) {
      clearTimeout(p.timer);
      p.reject(err);
    }
    this._pending.clear();
    if (this._pingPending) {
      clearTimeout(this._pingPending.timer);
      this._pingPending.reject(err);
      this._pingPending = null;
    }
  }

  // -- low level send / request ------------------------------------------------------------------

  /** Send a JSON message; returns false when not connected. */
  sendRaw(obj) {
    const ws = this.ws;
    if (!ws || ws.readyState !== 1) return false;
    try {
      ws.send(JSON.stringify(obj));
      this.stats.sent++;
      return true;
    } catch (e) {
      return false;
    }
  }

  /** Generic relay message to the other room member(s): {type, ...payload}. */
  send(type, payload = null) {
    const msg = payload && typeof payload === 'object' ? { ...payload, type } : { type };
    return this.sendRaw(msg);
  }

  _request(obj, replyTypes) {
    return new Promise((resolve, reject) => {
      if (!this.ws || this.ws.readyState !== 1) { reject(new Error('not connected')); return; }
      const key = replyTypes[0];
      if (this._pending.has(key)) { reject(new Error(`request "${obj.type}" already pending`)); return; }
      const timer = setTimeout(() => {
        this._pending.delete(key);
        reject(new Error(`timeout waiting for "${key}" (request "${obj.type}")`));
      }, this.requestTimeoutMs);
      this._pending.set(key, { resolve, reject, timer, types: replyTypes });
      if (!this.sendRaw(obj)) {
        clearTimeout(timer);
        this._pending.delete(key);
        reject(new Error('send failed'));
      }
    });
  }

  _settle(type, msg) {
    // errors that are not a reply to a request (rate limit, malformed message) settle nothing
    if (type === 'error' && (msg.code === 'RATE_LIMIT' || msg.code === 'BAD_JSON' || msg.code === 'BAD_MESSAGE')) return false;
    for (const [key, p] of this._pending) {
      if (!p.types.includes(type)) continue;
      clearTimeout(p.timer);
      this._pending.delete(key);
      if (type === 'error') {
        const err = new Error(msg.message || msg.code || 'server error');
        err.code = msg.code;
        p.reject(err);
      } else {
        p.resolve(msg);
      }
      return true;
    }
    return false;
  }

  // -- clock sync ------------------------------------------------------------------------------

  _pingOnce() {
    return new Promise((resolve, reject) => {
      if (!this.ws || this.ws.readyState !== 1) { reject(new Error('not connected')); return; }
      const t0 = this.now();
      const timer = setTimeout(() => {
        this._pingPending = null;
        reject(new Error('ping timeout'));
      }, this.pingTimeoutMs);
      this._pingPending = { t0, timer, resolve, reject };
      if (!this.sendRaw({ type: 'ping', t0 })) {
        clearTimeout(timer);
        this._pingPending = null;
        reject(new Error('send failed'));
      }
    });
  }

  /**
   * NTP-style clock sync: `count` sequential pings, offset = median of
   * t1 - (t0 + t3) / 2 (t1 = server receive/send time, t0/t3 = local send/receive time).
   * Resolves with {offsetMs, rttMs, samples}; updates `offsetMs`, `rttMs`, `synced`.
   */
  async syncClock(count = this.pingCount) {
    const offsets = [];
    const rtts = [];
    let failures = 0;
    for (let i = 0; i < count; i++) {
      try {
        const s = await this._pingOnce();
        offsets.push(s.offset);
        rtts.push(s.rtt);
      } catch (e) {
        failures++;
        if (failures > count) throw e;
      }
    }
    if (offsets.length === 0) throw new Error('clock sync failed (no pong)');
    this.offsetMs = median(offsets);
    this.rttMs = median(rtts);
    this.synced = true;
    const info = { offsetMs: this.offsetMs, rttMs: this.rttMs, samples: offsets.length };
    this.emit('sync', info);
    return info;
  }

  /** Server time now (ms) according to the last sync. */
  serverNow() {
    return this.now() + this.offsetMs;
  }

  toLocalMs(serverMs) {
    return serverMs - this.offsetMs;
  }

  toServerMs(localMs) {
    return localMs + this.offsetMs;
  }

  // -- rooms -----------------------------------------------------------------------------------

  /** Create a room; resolves with the room ({code, players, choreoId, ...}). */
  async createRoom(playerName = 'Spieler', choreoId = null) {
    await this._request({ type: 'create', name: playerName, choreoId }, ['room', 'error']);
    return this.room;
  }

  /** Join a room by its 4-letter code. */
  async joinRoom(code, playerName = 'Spieler') {
    await this._request({ type: 'join', code: String(code || '').trim().toUpperCase(), name: playerName }, ['room', 'error']);
    return this.room;
  }

  async leaveRoom() {
    if (!this.room) return;
    await this._request({ type: 'leave' }, ['left', 'error']);
    this.room = null;
    this.state = this.connected ? 'connected' : this.state;
  }

  /** Host only: announce the choreography to the room (resolves on the server's echo). */
  async setChoreo(choreoId) {
    const msg = await this._request({ type: 'setChoreo', choreoId }, ['choreo', 'error']);
    return msg.choreoId;
  }

  /**
   * Host only: start the dance. The server stamps `startAt` (its clock, ms) = now + delayMs and
   * broadcasts it; resolves with {choreoId, startAt, startAtLocalMs, delayMs} once it arrives.
   * The count-in begins at startAt on both headsets: `clock.start(bpm, countIn, {startAt:
   * clock.sourceNow() + (startAtLocalMs - Date.now()) / 1000})`.
   */
  async start(choreoId, { delayMs = 3000 } = {}) {
    const msg = await this._request({ type: 'start', choreoId, delayMs }, ['start', 'error']);
    return { choreoId: msg.choreoId, startAt: msg.startAt, startAtLocalMs: this.toLocalMs(msg.startAt), delayMs: msg.delayMs };
  }

  // -- state upload ----------------------------------------------------------------------------

  /**
   * Send the local 3-point state (stage frame S) at most `stateHz` times per second. Reuses one
   * message object; values are rounded to 0.1 mm. Returns true when a message went out.
   * @param {number} t game time in seconds since beat 0
   * @param {number[]} headP [x,y,z]  @param {number[]} headQ [x,y,z,w]
   * @param {number[]} leftP  @param {number[]} rightP
   */
  sendState(t, headP, headQ, leftP, rightP, score = 0, combo = 0) {
    const nowMs = this.now();
    if (nowMs - this._lastSendMs < this._sendIntervalMs) { this.stats.droppedRate++; return false; }
    const ws = this.ws;
    if (!ws || ws.readyState !== 1 || !this.room) return false;
    this._lastSendMs = nowMs;
    const m = this._stateMsg;
    m.t = r4(t);
    m.head[0] = r4(headP[0]); m.head[1] = r4(headP[1]); m.head[2] = r4(headP[2]);
    if (headQ) {
      m.head[3] = r4(headQ[0]); m.head[4] = r4(headQ[1]); m.head[5] = r4(headQ[2]); m.head[6] = r4(headQ[3]);
    } else {
      m.head[3] = 0; m.head[4] = 0; m.head[5] = 0; m.head[6] = 1;
    }
    m.left[0] = r4(leftP[0]); m.left[1] = r4(leftP[1]); m.left[2] = r4(leftP[2]);
    m.right[0] = r4(rightP[0]); m.right[1] = r4(rightP[1]); m.right[2] = r4(rightP[2]);
    m.score = Math.round(score);
    m.combo = combo | 0;
    try {
      ws.send(JSON.stringify(m));
      this.stats.sent++;
      return true;
    } catch (e) {
      return false;
    }
  }

  /** Convenience: sendState from a stage-frame sample ({head:{p,q}, left:{p}, right:{p}}). */
  sendStateFromSample(t, sampleS, score = 0, combo = 0) {
    return this.sendState(t, sampleS.head.p, sampleS.head.q, sampleS.left.p, sampleS.right.p, score, combo);
  }

  /** Send the final Result (compact: no perBeat) to the partner. */
  sendResult(result) {
    if (!result) return false;
    return this.sendRaw({
      type: 'result',
      result: {
        choreoId: result.choreoId,
        score: result.score,
        stars: result.stars,
        maxCombo: result.maxCombo,
        timingBias: result.timingBias,
        durationSec: result.durationSec,
        moves: Array.isArray(result.moves) ? result.moves.map((m) => ({ name: m.name, score: m.score, grade: m.grade })) : [],
        player: result.player,
        mode: result.mode,
        playedAt: result.playedAt,
      },
    });
  }

  // -- remote state ----------------------------------------------------------------------------

  /** Number of buffered remote states. */
  get remoteStateCount() {
    return this._bufCount;
  }

  /** Newest remote state slot (or null). */
  get remoteLatest() {
    return this._bufCount > 0 ? this._buf[this._bufHead] : null;
  }

  get remoteScore() {
    const s = this.remoteLatest;
    return s ? s.score : 0;
  }

  get remoteCombo() {
    const s = this.remoteLatest;
    return s ? s.combo : 0;
  }

  /** Forget buffered remote states (call at the start of a round). */
  resetRemote() {
    this._bufCount = 0;
    this._bufHead = -1;
  }

  _pushRemote(msg) {
    const head = msg.head, left = msg.left, right = msg.right;
    if (!isNum(msg.t) || !Array.isArray(head) || head.length < 3 || !Array.isArray(left) || left.length < 3 || !Array.isArray(right) || right.length < 3) return;
    // the server relays states verbatim: a misbehaving peer must not inject NaN into the avatar
    // and the benchmark metrics
    const nq = head.length >= 7 ? 7 : 3;
    for (let i = 0; i < nq; i++) if (!isNum(head[i])) return;
    for (let i = 0; i < 3; i++) if (!isNum(left[i]) || !isNum(right[i])) return;
    const size = this._buf.length;
    this._bufHead = (this._bufHead + 1) % size;
    if (this._bufCount < size) this._bufCount++;
    const s = this._buf[this._bufHead];
    s.t = msg.t;
    for (let i = 0; i < 3; i++) { s.head[i] = head[i]; s.left[i] = left[i]; s.right[i] = right[i]; }
    if (head.length >= 7) { s.head[3] = head[3]; s.head[4] = head[4]; s.head[5] = head[5]; s.head[6] = head[6]; }
    else { s.head[3] = 0; s.head[4] = 0; s.head[5] = 0; s.head[6] = 1; }
    s.score = isNum(msg.score) ? Math.min(100, Math.max(0, msg.score)) : 0;
    s.combo = isNum(msg.combo) ? Math.min(1e6, Math.max(0, Math.floor(msg.combo))) : 0;
    s.recvAt = this.now();
    this.stats.remoteStates++;
  }

  /**
   * Interpolated remote sample for game time `tNow` (seconds since beat 0), rendered
   * `interpolationDelaySec` in the past so that there is normally a newer state to interpolate
   * towards; holds the newest state when the buffer is behind. Returns null when no remote
   * state has arrived yet. `out` (default: reused internal object) has the shape
   * {t, head:{p,q}, left:{p}, right:{p}, score, combo, age} - `age` is seconds since the newest
   * state was received (use it to fade the avatar when the partner drops out).
   */
  getRemoteSample(tNow, out = this._remoteSample) {
    const n = this._bufCount;
    if (n === 0) return null;
    const size = this._buf.length;
    const tq = tNow - this.interpolationDelaySec;
    const newest = this._buf[this._bufHead];
    // find the newest slot with t <= tq (walk back from the head), and its successor
    let older = null, newer = null;
    for (let k = 0; k < n; k++) {
      const idx = (this._bufHead - k + size) % size;
      const s = this._buf[idx];
      if (s.t <= tq) { older = s; break; }
      newer = s;
    }
    let f = 0;
    if (!older) { older = newer; newer = null; }        // tq before the oldest buffered state
    else if (newer && newer.t > older.t) f = Math.min(1, Math.max(0, (tq - older.t) / (newer.t - older.t)));
    else newer = null;
    const a = older, b = newer;
    out.t = tq;
    const hp = out.head.p, lp = out.left.p, rp = out.right.p, q = out.head.q;
    if (b && f > 0) {
      for (let i = 0; i < 3; i++) {
        hp[i] = a.head[i] + (b.head[i] - a.head[i]) * f;
        lp[i] = a.left[i] + (b.left[i] - a.left[i]) * f;
        rp[i] = a.right[i] + (b.right[i] - a.right[i]) * f;
      }
      // nlerp (shortest path) of the head quaternion
      let dot = a.head[3] * b.head[3] + a.head[4] * b.head[4] + a.head[5] * b.head[5] + a.head[6] * b.head[6];
      const sign = dot < 0 ? -1 : 1;
      let len = 0;
      for (let i = 0; i < 4; i++) {
        const v = a.head[3 + i] + (sign * b.head[3 + i] - a.head[3 + i]) * f;
        q[i] = v;
        len += v * v;
      }
      len = Math.sqrt(len) || 1;
      for (let i = 0; i < 4; i++) q[i] /= len;
      out.score = f < 0.5 ? a.score : b.score;
      out.combo = f < 0.5 ? a.combo : b.combo;
    } else {
      for (let i = 0; i < 3; i++) { hp[i] = a.head[i]; lp[i] = a.left[i]; rp[i] = a.right[i]; }
      for (let i = 0; i < 4; i++) q[i] = a.head[3 + i];
      out.score = a.score;
      out.combo = a.combo;
    }
    out.age = (this.now() - newest.recvAt) / 1000;
    return out;
  }

  // -- incoming messages ---------------------------------------------------------------------

  _onMessage(data) {
    let msg;
    try {
      msg = JSON.parse(typeof data === 'string' ? data : String(data));
    } catch (e) {
      return;
    }
    if (!msg || typeof msg.type !== 'string') return;
    this.stats.received++;
    switch (msg.type) {
      case 'hello':
        this.id = msg.id;
        this.serverVersion = msg.version || null;
        this.serverTimeAtHello = msg.serverTime || null;
        this.state = 'connected';
        this.emit('hello', msg);
        return;
      case 'pong': {
        const p = this._pingPending;
        if (p && msg.t0 === p.t0) {
          clearTimeout(p.timer);
          this._pingPending = null;
          const t3 = this.now();
          p.resolve({ offset: msg.t1 - (p.t0 + t3) / 2, rtt: t3 - p.t0 });
        }
        return;
      }
      case 'room':
        this.room = { code: msg.code, choreoId: msg.choreoId || null, players: msg.players || [], maxPlayers: msg.maxPlayers || 2, startAt: msg.startAt || null };
        this.state = 'lobby';
        this.resetRemote();
        this._settle('room', msg);
        this.emit('room', this.room);
        return;
      case 'left':
        this.room = null;
        this._settle('left', msg);
        this.emit('left', msg);
        return;
      case 'peer':
        if (this.room && Array.isArray(msg.players)) this.room.players = msg.players;
        if (msg.event === 'left') this.resetRemote();
        this.emit('peer', msg);
        return;
      case 'choreo':
        if (this.room) this.room.choreoId = msg.choreoId;
        this._settle('choreo', msg);
        this.emit('choreo', msg);
        return;
      case 'start': {
        this.startAt = msg.startAt;
        this.startChoreoId = msg.choreoId;
        this.startAtLocalMs = this.toLocalMs(msg.startAt);
        if (this.room) { this.room.startAt = msg.startAt; if (msg.choreoId) this.room.choreoId = msg.choreoId; }
        this.state = 'playing';
        this.resetRemote();
        this._settle('start', msg);
        this.emit('start', { choreoId: msg.choreoId, startAt: msg.startAt, startAtLocalMs: this.startAtLocalMs, delayMs: msg.delayMs, from: msg.from });
        return;
      }
      case 'state':
        this._pushRemote(msg);
        this.emit('state', msg);
        return;
      case 'result':
        this.emit('result', msg);
        return;
      case 'error': {
        this.lastError = new Error(msg.message || msg.code || 'server error');
        this.lastError.code = msg.code;
        const handled = this._settle('error', msg);
        this.emit('error', { code: msg.code, message: msg.message, handled });
        return;
      }
      default:
        this.emit('message', msg);
    }
  }
}
