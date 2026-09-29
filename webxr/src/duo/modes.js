// modes.js - the two duo game modes of DESIGN.md section 9 as App modes (`app.registerMode`):
//   'ghost'  - the second dancer is a saved run of the same choreography (localStorage['dp.runs']
//              or GET /api/runs); it is scored live with a second Scorer so both scores race on the
//              HUD, and the results screen gets the benchmark (winner, sync distance, lag) with
//              JSON/CSV export buttons.
//   'online' - two headsets in one WLAN through server/server.js: lobby panel with a 4-letter room
//              code (create / join / start as host), NTP-style clock sync, synchronised start
//              (`startOptions` -> BeatClock startAt), 15 Hz state relay, the partner rendered from
//              interpolated states next to the teacher, both scores on the HUD, the partner's
//              result + a recorded copy of its states feed the same benchmark.
// Imports three.js only through render/avatar.js (PointAvatar); the App owns the state machine,
// the PlaySession and the clock - modes only observe and decorate. No allocation per frame in
// `update` (sampleAt/getRemoteSample reuse their objects; HUD panels redraw on key changes only).

import { STAGE } from '../config.js';
import { Scorer } from '../game/scoring.js';
import { PointAvatar, AVATAR_COLORS } from '../render/avatar.js';
import { fitText, starsString, HUD_COLORS } from '../render/hud.js';
import { loadGhost, RUN_FORMAT } from './ghost.js';
import { OnlineDuo } from './online.js';
import { buildBenchmark, toCSV, toJSON } from './benchmark.js';

export const START_DELAY_MS = 8000;            // host start -> count-in (time for both calibrations)
export const ROOM_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ';   // the server's code alphabet (no I/O)
const DUO_PANEL = { width: 1.0, height: 0.4, position: [1.6, 1.84, 0.02] };

// ---------------------------------------------------------------------------------------------
// shared helpers

function partnerPosition(avatar) {
  avatar.position.set(STAGE.duoSideOffset, 0, -STAGE.teacherDistance);
}

/** Two-line score panel: "Du 87" / "<name> 91". Redraws only when the key changes. */
function drawScores(panel, texts, nameA, scoreA, nameB, scoreB, colorB) {
  const key = `${scoreA}|${scoreB}|${nameB}`;
  panel.draw((ctx, w, h) => {
    fitText(ctx, `${nameA}  ${scoreA}`, w / 2, h * 0.3, w * 0.9, 34, 'bold', 'center', HUD_COLORS.text);
    fitText(ctx, `${nameB}  ${scoreB}`, w / 2, h * 0.72, w * 0.9, 34, 'bold', 'center', colorB);
  }, key);
}

function hexColor(c) {
  return '#' + c.toString(16).padStart(6, '0');
}

/** Browser download of a text file; returns false when not possible (no DOM). */
export function downloadText(filename, text, mime = 'application/json') {
  if (typeof document === 'undefined' || typeof Blob === 'undefined' || typeof URL === 'undefined' || !URL.createObjectURL) return false;
  try {
    const blob = new Blob([text], { type: mime });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    return true;
  } catch (e) {
    console.warn('[duo] download failed', e);
    return false;
  }
}

function safeName(s) {
  return String(s || '').replace(/[^a-z0-9-]+/gi, '-').replace(/^-+|-+$/g, '').toLowerCase() || 'duo';
}

/** Results-panel rows shared by both modes (lines may be functions so they refresh). */
function benchmarkLines(app, mode, getBenchmark, nameB, waitingText) {
  const t = app.texts;
  const pct = (x) => `${Math.round(x)}`;
  const lines = [
    () => {
      const b = getBenchmark();
      if (!b) return waitingText;
      const B = b.players[1];
      return `${nameB()}: ${B.score === null ? '–' : pct(B.score)} ${t.resultsScore} · ${starsString(B.stars || 1)} · ${t.resultsMaxCombo} ×${B.maxCombo || 0}`;
    },
    () => {
      const b = getBenchmark();
      if (!b) return '';
      const w = b.winner === 'A' ? t.duoYou : (b.winner === 'B' ? nameB() : t.resultsDraw);
      return b.winner === 'tie' ? `${t.resultsDraw}` : `${t.resultsWinner}: ${w}`;
    },
    () => {
      const b = getBenchmark();
      if (!b || b.pair.syncDistance === null || !Number.isFinite(b.pair.syncDistance)) return '';
      const lag = Number.isFinite(b.pair.syncLag) ? Math.round(b.pair.syncLag * 1000) : 0;
      return `${t.resultsSync}: ${t.duoSyncDistance} ${Math.round(b.pair.syncDistance * 100)} cm · ${t.resultsSyncLag} ${lag > 0 ? '+' : ''}${lag} ms`;
    },
  ];
  const moveLines = [
    () => {
      const b = getBenchmark();
      if (!b) return '';
      const A = b.players[0].perMove, B = b.players[1].perMove;
      return A.map((m, i) => `${m.name}: ${pct(m.score * 100)} / ${B[i] ? pct(B[i].score * 100) : '–'}`).join('   ');
    },
  ];
  const exporter = (kind) => {
    const b = getBenchmark();
    if (!b) { app.hud.showNotice(waitingText, 2); return; }
    const name = `dp-benchmark-${safeName(b.choreoId)}-${safeName(mode)}-${(b.createdAt || new Date().toISOString()).replace(/[:.]/g, '-')}.${kind}`;
    const ok = kind === 'csv' ? downloadText(name, toCSV(b), 'text/csv') : downloadText(name, toJSON(b), 'application/json');
    app.hud.showNotice(ok ? t.duoExported : t.duoExportFailed, 2);
  };
  const buttons = [
    { type: 'button', id: 'export-json', text: t.duoExportJson, onClick: () => exporter('json') },
    { type: 'button', id: 'export-csv', text: t.duoExportCsv, onClick: () => exporter('csv') },
  ];
  return { lines, moveLines, buttons };
}

/** Hold-fill a frame ring recorded from interpolated remote states into a run/1 JSON. */
function framesToRun(choreoId, fps, n, head, left, right, filled, extra = {}) {
  let count = 0;
  for (let i = 0; i < n; i++) if (filled[i]) count++;
  if (count === 0) return null;
  const r4 = (x) => Math.round(x * 10000) / 10000;
  const H = new Array(n), L = new Array(n), R = new Array(n);
  let src = -1;
  for (let i = 0; i < n; i++) if (filled[i]) { src = i; break; }
  for (let i = 0; i < n; i++) {
    if (filled[i]) src = i;
    const o7 = src * 7, o3 = src * 3;
    H[i] = [r4(head[o7]), r4(head[o7 + 1]), r4(head[o7 + 2]), r4(head[o7 + 3]), r4(head[o7 + 4]), r4(head[o7 + 5]), r4(head[o7 + 6])];
    L[i] = [r4(left[o3]), r4(left[o3 + 1]), r4(left[o3 + 2])];
    R[i] = [r4(right[o3]), r4(right[o3 + 1]), r4(right[o3 + 2])];
  }
  return { format: RUN_FORMAT, choreoId, fps, frameCount: n, head: H, left: L, right: R, ...extra };
}

// ---------------------------------------------------------------------------------------------
// ghost mode

export function createGhostMode(app) {
  const mode = {
    name: 'ghost',
    ghost: null,
    scorer: null,
    avatar: null,
    panel: null,
    ghostResult: null,
    benchmark: null,
    _unsub: [],

    async prepare(ctx) {
      const t = ctx.texts;
      this.ghost = null;
      this.ghostResult = null;
      this.benchmark = null;
      const ghost = await loadGhost({ choreoId: ctx.choreo.id, storage: ctx.storage, baseUrl: ctx.httpBase, strategy: 'best' });
      if (!ghost) {
        const err = new Error(`no saved run for ${ctx.choreo.id}`);
        err.notice = t.duoNoRuns;
        throw err;
      }
      this.ghost = ghost;
      this.scorer = new Scorer(ctx.choreo);
      ctx.hud.showNotice(`${t.duoGhostLoaded}: ${ghost.name} (${ghost.score === null || ghost.score === undefined ? '–' : ghost.score} ${t.resultsScore})`, 3);
    },

    onSessionCreated(session, ctx) {
      this.avatar = new PointAvatar({ color: AVATAR_COLORS.ghost, opacity: 0.6 });
      partnerPosition(this.avatar);
      ctx.stage.add(this.avatar);
      this.panel = ctx.hud.addPanel('duo', DUO_PANEL);
      this.scorer.reset();
      this.scorer.onBeat = null;
      const ghost = this.ghost, scorer = this.scorer;
      this._unsub.push(session.on('tick', (e) => { scorer.scoreTick(ghost.sampleAt(e.t), e.t); }));
      drawScores(this.panel, ctx.texts, ctx.texts.duoYou, 0, ghost.name, 0, hexColor(AVATAR_COLORS.ghost));
    },

    update(timeMs, dt, session) {
      if (!this.avatar || !this.ghost) return;
      const t = session.t < 0 ? 0 : session.t;
      this.avatar.setSample(this.ghost.sampleAt(t));
      if (this.panel) drawScores(this.panel, app.texts, app.texts.duoYou, session.score, this.ghost.name, this.scorer.currentScore(), hexColor(AVATAR_COLORS.ghost));
    },

    onFinished(result, ctx) {
      const t = ctx.texts;
      this.ghostResult = this.scorer.finalize({ mode: 'ghost', player: this.ghost.name, playedAt: this.ghost.playedAt || undefined });
      const nameA = result.player || t.duoYou;
      this.benchmark = buildBenchmark(result, this.ghostResult, ctx.app.lastRun, this.ghost.run, {
        nameA, nameB: this.ghost.name, mode: 'ghost', choreoId: ctx.choreo.id,
      });
      app.lastBenchmark = this.benchmark;
      if (this.panel) drawScores(this.panel, t, t.duoYou, result.score, this.ghost.name, this.ghostResult.score, hexColor(AVATAR_COLORS.ghost));
      return benchmarkLines(app, 'ghost', () => this.benchmark, () => this.ghost.name, t.duoWaitingResult);
    },

    exit(ctx) {
      for (const off of this._unsub) { try { off(); } catch (e) { /* ignore */ } }
      this._unsub.length = 0;
      if (this.avatar) {
        ctx.stage.remove(this.avatar);
        this.avatar.dispose();
        this.avatar = null;
      }
      if (this.panel) {
        ctx.hud.removePanel('duo');
        this.panel = null;
      }
    },
  };
  return mode;
}

// ---------------------------------------------------------------------------------------------
// online mode

export function createOnlineMode(app, opts = {}) {
  const WebSocketImpl = opts.WebSocketImpl || (typeof WebSocket !== 'undefined' ? WebSocket : null);
  const mode = {
    name: 'online',
    lobbyPanelId: 'duo-online',
    online: null,
    serverUrl: null,
    status: 'idle',          // idle | connecting | connected | failed
    code: ['A', 'A', 'A', 'A'],
    codeEdit: false,
    startInfo: null,
    avatar: null,
    panel: null,
    remote: null,            // recorded partner frames
    remoteRun: null,
    remoteResult: null,
    benchmark: null,
    lastResult: null,
    _unsub: [],
    _connectPromise: null,

    init(application) {
      const t = application.texts;
      const m = this;
      application.menu.addPanel('duo-online', {
        title: t.duoOnline,
        width: 1.6,
        position: [1.75, 1.5, 0.15],
        rotationY: -0.4,
        items: () => m._lobbyItems(application),
        footer: () => `${t.settingsServer}: ${m.serverUrl || t.offline}${m.online && m.online.synced ? ` · RTT ${Math.round(m.online.rttMs)} ms` : ''}`,
        onBack: () => application.backToMenu(),
      });
      application.on('lobby', (ctx) => { m.serverUrl = ctx.server || ctx.httpBase || null; if (m.serverUrl) m.ensureConnected().catch(() => null); });
      application.on('state', (e) => { if (e.to === 'DUO_LOBBY') m._refresh(); });
    },

    // -- connection --------------------------------------------------------------------------

    _refresh() {
      const p = app.menu.getPanel('duo-online');
      if (p && p.visible) p.refresh();
    },

    async ensureConnected() {
      if (this.online && this.online.connected) return this.online;
      if (this._connectPromise) return this._connectPromise;
      if (!this.serverUrl) throw new Error('no server');
      const t = app.texts;
      this.status = 'connecting';
      this._refresh();
      const online = new OnlineDuo({ WebSocketImpl });
      this.online = online;
      this._bindOnline(online);
      this._connectPromise = (async () => {
        try {
          await online.connect(this.serverUrl);
          await online.syncClock();
          this.status = 'connected';
          return online;
        } catch (e) {
          this.status = 'failed';
          console.warn('[duo] connect failed', e);
          app.hud.showNotice(t.duoConnectFailed, 4);
          throw e;
        } finally {
          this._connectPromise = null;
          this._refresh();
        }
      })();
      return this._connectPromise;
    },

    _bindOnline(online) {
      const t = app.texts;
      online.on('start', (ev) => this._onStart(ev));
      online.on('peer', (ev) => {
        app.hud.showNotice(ev.event === 'joined' ? t.duoPlayerJoined : (ev.event === 'left' ? t.duoPlayerLeft : `${t.duoHost}: ${ev.player ? ev.player.name : ''}`), 3);
        this._refresh();
      });
      online.on('room', () => this._refresh());
      online.on('choreo', (ev) => { if (ev.choreoId && app.choreos.some((c) => c.id === ev.choreoId)) app.selectedChoreoId = ev.choreoId; this._refresh(); });
      online.on('result', (ev) => this._onRemoteResult(ev));
      online.on('disconnected', () => { this.status = 'failed'; app.hud.showNotice(t.duoConnectionLost, 4); this._refresh(); });
      online.on('error', (ev) => { if (ev.code !== 'RATE_LIMIT') console.warn('[duo] server error', ev.code, ev.message); });
    },

    get connected() {
      return !!(this.online && this.online.connected);
    },

    get roomCode() {
      return this.online ? this.online.roomCode : null;
    },

    get isHost() {
      return !!(this.online && this.online.isHost);
    },

    playerName() {
      return (app.settings && app.settings.playerName) || app.texts.player;
    },

    async createRoom() {
      const online = await this.ensureConnected();
      await online.createRoom(this.playerName(), app.selectedChoreoId);
      this._refresh();
      return online.room;
    },

    async joinRoom(code) {
      const online = await this.ensureConnected();
      const c = String(code || this.code.join('')).toUpperCase();
      await online.joinRoom(c, this.playerName());
      this.codeEdit = false;
      if (online.room && online.room.choreoId && app.choreos.some((x) => x.id === online.room.choreoId)) app.selectedChoreoId = online.room.choreoId;
      this._refresh();
      return online.room;
    },

    async leaveRoom() {
      if (this.online && this.online.room) await this.online.leaveRoom().catch(() => null);
      this._refresh();
    },

    /** Host: announce the start; the server's broadcast starts the run on both headsets. */
    async startAsHost(choreoId = app.selectedChoreoId, { delayMs = START_DELAY_MS } = {}) {
      const t = app.texts;
      if (!this.connected || !this.online.room) throw new Error('not in a room');
      if (!this.isHost) { app.hud.showNotice(t.duoOnlyHost, 3); throw new Error('not the host'); }
      if (!choreoId) { app.hud.showNotice(t.menuNoChoreos, 3); throw new Error('no choreography'); }
      try {
        await this.online.syncClock();
        return await this.online.start(choreoId, { delayMs });
      } catch (e) {
        app.hud.showNotice(`${t.duoStartFailed}${e && e.code ? ` (${e.code})` : ''}`, 4);
        throw e;
      }
    },

    _onStart(ev) {
      this.startInfo = ev;
      this.remoteResult = null;
      this.benchmark = null;
      const busyStates = ['CALIBRATE', 'COUNTDOWN', 'PLAYING', 'RECORDING', 'RECORD_REVIEW'];
      if (busyStates.includes(app.state)) { console.warn('[duo] start ignored in state', app.state); return; }
      app.hud.showNotice(app.texts.duoStartingSoon, 3);
      app.startMode('online', ev.choreoId || app.selectedChoreoId).catch((e) => console.warn('[duo] online start failed', e));
    },

    _onRemoteResult(ev) {
      this.remoteResult = ev && ev.result ? ev.result : null;
      if (this.lastResult && this.remoteResult) {
        this._buildBenchmark(this.lastResult);
        app.menu.refresh('results');
      }
    },

    // -- lobby panel ------------------------------------------------------------------------

    _lobbyItems(application) {
      const t = application.texts;
      const items = [];
      if (!this.serverUrl) {
        items.push({ type: 'text', text: t.duoNoServer });
        return items;
      }
      const statusText = this.status === 'connecting' ? t.duoConnecting : (this.connected ? t.duoConnected : (this.status === 'failed' ? t.duoConnectFailed : t.offline));
      items.push({ type: 'label', text: statusText, size: 'small' });
      const online = this.online;
      const room = online && online.room;
      if (room) {
        items.push({ type: 'label', text: `${t.duoRoomCode}: ${room.code}`, size: 'large', align: 'center', color: '#4cc9f0' });
        const players = room.players.map((p) => `${p.id === online.id ? t.duoYou : p.name}${p.host ? ` (${t.duoHost})` : ''}`).join(' · ');
        items.push({ type: 'label', text: players, size: 'small', align: 'center' });
        if (this.isHost) {
          items.push({ type: 'button', id: 'start', text: t.duoHostStart, primary: true, onClick: () => this.startAsHost().catch(() => null) });
        } else {
          items.push({ type: 'label', text: t.duoWaiting, size: 'small', align: 'center' });
        }
        items.push({ type: 'button', id: 'leave', text: t.duoLeaveRoom, onClick: () => this.leaveRoom() });
        return items;
      }
      items.push({ type: 'button', id: 'create', text: t.duoCreateRoom, primary: true, disabled: () => !this.connected, onClick: () => this.createRoom().catch(() => null) });
      if (this.codeEdit) {
        const letters = this.code;
        letters.forEach((ch, i) => {
          const step = (d) => { letters[i] = ROOM_ALPHABET[(ROOM_ALPHABET.indexOf(letters[i]) + d + ROOM_ALPHABET.length) % ROOM_ALPHABET.length]; };
          items.push({ type: 'stepper', id: `code${i}`, text: `${t.duoRoomCode} ${i + 1}`, value: () => letters[i], onDec: () => step(-1), onInc: () => step(1), onDec2: () => step(-6), onInc2: () => step(6) });
        });
        items.push({
          type: 'row',
          items: [
            { type: 'button', id: 'join', text: `${t.duoJoinRoom}: ${letters.join('')}`, primary: true, disabled: () => !this.connected, onClick: () => this.joinRoom().catch(() => null) },
            { type: 'button', id: 'code-cancel', text: t.cancel, onClick: () => { this.codeEdit = false; this._refresh(); } },
          ],
        });
      } else {
        items.push({ type: 'button', id: 'code', text: `${t.duoEnterCode}: ${this.code.join('')}`, disabled: () => !this.connected, onClick: () => this._editCode() });
        items.push({ type: 'button', id: 'join', text: t.duoJoinRoom, disabled: () => !this.connected, onClick: () => this.joinRoom().catch(() => null) });
      }
      return items;
    },

    _editCode() {
      const t = app.texts;
      const win = app.win;
      if (app.inputKind !== 'xr' && win && typeof win.prompt === 'function') {
        const v = win.prompt(t.duoRoomCode, this.code.join(''));
        if (v && /^[a-z]{4}$/i.test(v.trim())) this.code = v.trim().toUpperCase().split('');
        this._refresh();
        return;
      }
      this.codeEdit = true;
      this._refresh();
    },

    // -- run ----------------------------------------------------------------------------------

    startOptions(ctx) {
      const info = this.startInfo;
      if (!info || !Number.isFinite(info.startAtLocalMs)) return null;
      const wait = (info.startAtLocalMs - Date.now()) / 1000;
      return { startAt: ctx.clock.sourceNow() + wait };
    },

    onSessionCreated(session, ctx) {
      const online = this.online;
      this.lastResult = null;
      this.remoteRun = null;
      this.benchmark = null;
      this.avatar = new PointAvatar({ color: AVATAR_COLORS.opponent, opacity: 0.7 });
      partnerPosition(this.avatar);
      ctx.stage.add(this.avatar);
      this.panel = ctx.hud.addPanel('duo', DUO_PANEL);
      const n = session.run ? session.run.frameCount : Math.round(ctx.choreo.duration * 30) + 1;
      const rem = this.remote;
      if (!rem || rem.n !== n) {
        this.remote = { n, fps: 30, head: new Float32Array(n * 7), left: new Float32Array(n * 3), right: new Float32Array(n * 3), filled: new Uint8Array(n) };
      } else {
        rem.filled.fill(0);
      }
      if (online) online.resetRemote();
      const m = this;
      this._unsub.push(session.on('tick', (e) => {
        if (!online) return;
        online.sendStateFromSample(e.t, session.playerS, session.score, session.combo);
        const r = online.getRemoteSample(e.t);
        if (r && online.remoteStateCount > 0) m._recordRemote(e.t, r);
      }));
      drawScores(this.panel, ctx.texts, ctx.texts.duoYou, 0, this.peerName(), 0, hexColor(AVATAR_COLORS.opponent));
    },

    _recordRemote(t, r) {
      const rem = this.remote;
      let idx = Math.round(t * rem.fps);
      if (idx < 0) idx = 0;
      if (idx >= rem.n) idx = rem.n - 1;
      const o7 = idx * 7, o3 = idx * 3;
      rem.head[o7] = r.head.p[0]; rem.head[o7 + 1] = r.head.p[1]; rem.head[o7 + 2] = r.head.p[2];
      rem.head[o7 + 3] = r.head.q[0]; rem.head[o7 + 4] = r.head.q[1]; rem.head[o7 + 5] = r.head.q[2]; rem.head[o7 + 6] = r.head.q[3];
      rem.left[o3] = r.left.p[0]; rem.left[o3 + 1] = r.left.p[1]; rem.left[o3 + 2] = r.left.p[2];
      rem.right[o3] = r.right.p[0]; rem.right[o3 + 1] = r.right.p[1]; rem.right[o3 + 2] = r.right.p[2];
      rem.filled[idx] = 1;
    },

    peerName() {
      const p = this.online ? this.online.peer : null;
      return p && p.name ? p.name : app.texts.duoOpponent;
    },

    update(timeMs, dt, session) {
      const online = this.online;
      if (!this.avatar || !online) return;
      const r = online.getRemoteSample(session.t);
      if (r && online.remoteStateCount > 0) {
        this.avatar.setSample(r);
        this.avatar.visible = true;
      }
      if (this.panel) drawScores(this.panel, app.texts, app.texts.duoYou, session.score, this.peerName(), online.remoteScore, hexColor(AVATAR_COLORS.opponent));
    },

    _buildBenchmark(result) {
      const rem = this.remote;
      if (!this.remoteRun && rem) {
        this.remoteRun = framesToRun(result.choreoId, rem.fps, rem.n, rem.head, rem.left, rem.right, rem.filled, { player: this.peerName(), mode: 'online' });
      }
      this.benchmark = buildBenchmark(result, this.remoteResult, app.lastRun, this.remoteRun, {
        nameA: result.player || app.texts.duoYou, nameB: this.peerName(), mode: 'online', choreoId: result.choreoId,
      });
      app.lastBenchmark = this.benchmark;
      return this.benchmark;
    },

    onFinished(result, ctx) {
      const t = ctx.texts;
      this.lastResult = result;
      if (this.online) this.online.sendResult(result);
      if (this.remoteResult) this._buildBenchmark(result);
      if (this.panel) drawScores(this.panel, t, t.duoYou, result.score, this.peerName(), this.online ? this.online.remoteScore : 0, hexColor(AVATAR_COLORS.opponent));
      return benchmarkLines(app, 'online', () => this.benchmark, () => this.peerName(), t.duoWaitingResult);
    },

    exit(ctx) {
      for (const off of this._unsub) { try { off(); } catch (e) { /* ignore */ } }
      this._unsub.length = 0;
      if (this.avatar) {
        ctx.stage.remove(this.avatar);
        this.avatar.dispose();
        this.avatar = null;
      }
      if (this.panel) {
        ctx.hud.removePanel('duo');
        this.panel = null;
      }
      this.startInfo = null;
    },

    dispose() {
      if (this.online) { this.online.dispose(); this.online = null; }
      this.status = 'idle';
    },
  };
  return mode;
}

/** Register both duo modes on an App. Returns { ghost, online }. */
export function registerDuoModes(app, opts = {}) {
  const ghost = app.registerMode('ghost', createGhostMode(app));
  const online = app.registerMode('online', createOnlineMode(app, opts));
  return { ghost, online };
}
