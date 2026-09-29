// app.js - App: boot, the state machine of DESIGN.md section 4, the main loop
// (renderer.setAnimationLoop), input backend selection (XRInput vs EmulatedInput by URL params),
// calibration flow, count-in / play / results, the record flow (RECORD_SETUP / RECORDING /
// RECORD_REVIEW with Recorder), settings, highscores, persistence (localStorage + optional
// server) and the extension points for the duo modes (`app.registerMode`) and the neural
// avatar (`app.setBodyAvatarPoseProvider`). Owns one SceneKit, one input backend, the Menu,
// the HUD and the current PlaySession. Exposes `window.__dp = { app, version }`.
// Imports three.js through the render modules (browser only); no allocation in `_frame`.

import { APP_VERSION, APP_NAME, STAGE, FLOW, NET, STORAGE_KEYS, params as pageParams } from './config.js';
import { EventEmitter } from './util/events.js';
import { XRInput, createSample } from './xr/input.js';
import { Calibration } from './xr/calibration.js';
import { EmulatedInput } from './emu/emulated-input.js';
import { BeatClock } from './game/clock.js';
import { AudioEngine } from './game/audio.js';
import { loadChoreo, Choreo, validateChoreo, mirrorChoreoSample, createChoreoSample } from './game/choreo.js';
import { Scorer, resultToJSON } from './game/scoring.js';
import { PlaySession } from './game/session.js';
import { Recorder } from './game/recorder.js';
import { getTexts, GRADE_KEYS } from './ui/texts.js';
import { SceneKit } from './render/scene.js';
import { TeacherAvatar, BodyAvatar, ControllerMarkers, AVATAR_COLORS } from './render/avatar.js';
import { HUD, starsString } from './render/hud.js';
import { PlayerMirror } from './render/mirror.js';
import { Menu } from './ui/menu.js';
import { NeuralAvatarDriver } from './net/avatar-driver.js';
import { registerDuoModes } from './duo/modes.js';
import { mergeResults, resultKey } from './duo/benchmark.js';
import { yawFromQuat, angleDiff } from './util/math.js';

export const STATES = Object.freeze({
  BOOT: 'BOOT',
  MENU: 'MENU',
  SETTINGS: 'SETTINGS',
  CALIBRATE: 'CALIBRATE',
  COUNTDOWN: 'COUNTDOWN',
  PLAYING: 'PLAYING',
  RESULTS: 'RESULTS',
  RECORD_SETUP: 'RECORD_SETUP',
  RECORDING: 'RECORDING',
  RECORD_REVIEW: 'RECORD_REVIEW',
  DUO_LOBBY: 'DUO_LOBBY',
});

// strings that ui/texts.js does not provide (German first, English fallback)
const LOCAL_TEXTS = {
  de: {
    notAvailable: 'Noch nicht verfügbar',
    vrEnded: 'VR-Sitzung beendet',
    desktopHint: 'Desktop: Maus zeigt & klickt · WASD bewegen · Q/E Hände heben · Esc Abbrechen',
    calibrateKept: 'Kalibrierung übernommen',
    chooseDance: 'Bitte zuerst einen Tanz auswählen',
    recordAborted: 'Aufnahme abgebrochen',
    ownDance: 'Eigener Tanz',
    fromServer: 'Server',
    holdToQuit: 'B/Y {sec} s gedrückt halten zum Abbrechen',
    readyHold: 'Bereit? Gleich geht es los …',
    swFailed: 'Kein Offline-Cache (Zertifikat nicht vertrauenswürdig – siehe server/README.md)',
    swUpdated: 'Neue Version verfügbar – Seite wird neu geladen …',
    recentered: 'Ansicht neu zentriert – bitte neu kalibrieren',
    contextLost: 'Grafik zurückgesetzt – Seite wird neu geladen …',
    paused: 'Pause – Headset aufsetzen zum Weitermachen',
    resumed: 'Weiter geht es',
    neuralProgress: 'Neural-Avatar lädt: {loaded} / {total} MB',
    neuralDisabledSaved: 'Neural-Avatar deaktiviert (zu langsam) – in den Einstellungen wieder einschaltbar',
    reviewInfo: '{frames} Frames · {seconds} s · {bpm} BPM',
    modeSolo: 'Solo',
    modeGhost: 'Duo (Geist)',
    modeOnline: 'Duo (Online)',
    stars: 'Sterne',
    ms: 'ms',
    dateAt: 'am',
    telemetryNone: 'Keine Telemetrie vorhanden (Neural-Avatar mit ?telemetry=1 starten)',
    startWhileBusy: 'Bitte warten …',
    loadingDance: 'Tanz wird geladen …',
    bars: 'Takte',
    perMoveHeader: 'Move · Punkte · Bewertung',
  },
  en: {
    notAvailable: 'Not available yet',
    vrEnded: 'VR session ended',
    desktopHint: 'Desktop: mouse points & clicks · WASD move · Q/E raise hands · Esc abort',
    calibrateKept: 'Calibration kept',
    chooseDance: 'Please choose a dance first',
    recordAborted: 'Recording aborted',
    ownDance: 'My dance',
    fromServer: 'Server',
    holdToQuit: 'Hold B/Y for {sec} s to quit',
    readyHold: 'Ready? Starting in a moment ...',
    swFailed: 'No offline cache (certificate not trusted - see server/README.md)',
    swUpdated: 'New version available - reloading ...',
    recentered: 'View recentred - please recalibrate',
    contextLost: 'Graphics reset - reloading ...',
    paused: 'Paused - put the headset on to continue',
    resumed: 'Resuming',
    neuralProgress: 'Neural avatar loading: {loaded} / {total} MB',
    neuralDisabledSaved: 'Neural avatar disabled (too slow) - re-enable it in the settings',
    reviewInfo: '{frames} frames · {seconds} s · {bpm} BPM',
    modeSolo: 'Solo',
    modeGhost: 'Duo (ghost)',
    modeOnline: 'Duo (online)',
    stars: 'Stars',
    ms: 'ms',
    dateAt: 'on',
    telemetryNone: 'No telemetry recorded (start the neural avatar with ?telemetry=1)',
    startWhileBusy: 'Please wait ...',
    loadingDance: 'Loading dance ...',
    bars: 'bars',
    perMoveHeader: 'Move · score · grade',
  },
};

const AVATAR_MODES = ['neural', 'points', 'off'];
const SYNTHS = ['hiphop', 'house', 'metronome'];
const MAX_RESULTS = 500;
const MAX_RUNS_PER_CHOREO = 3;
const CALIBRATION_REUSE_SEC = 600;
const CALIBRATION_MAX_YAW_DIFF = 0.55;   // rad (~30 deg): a persisted calibration whose facing differs more is redone
const QUIT_HOLD_SEC = FLOW.quitHoldSec;
const QUIT_STATES = new Set([STATES.CALIBRATE, STATES.COUNTDOWN, STATES.PLAYING, STATES.RECORDING]);
const SERVER_HIGHSCORE_LIMIT = 1000;
const RESULT_POST_DELAY_MS = 10000;      // online duo: wait this long for the benchmark before uploading the result
// states in which the neural avatar worker is fed and the player's mirror image is shown
const NEURAL_STATES = new Set([STATES.CALIBRATE, STATES.COUNTDOWN, STATES.PLAYING, STATES.RESULTS, STATES.RECORDING, STATES.RECORD_REVIEW]);
const MIRROR_STATES = new Set([STATES.COUNTDOWN, STATES.PLAYING, STATES.RESULTS, STATES.RECORDING, STATES.RECORD_REVIEW]);

function isChoreoUrl(ref) {
  return typeof ref === 'string' && (/^https?:\/\//i.test(ref) || ref.includes('/') || ref.endsWith('.json'));
}

/** Parse the URL parameters that config.parseParams does not know (or rejects). */
export function parseExtraParams(search = '') {
  const out = { cam: 'first', height: 0, yaw: 0, choreoUrl: null, avatarExplicit: false };
  let sp;
  try { sp = new URLSearchParams(search || ''); } catch (e) { return out; }
  out.avatarExplicit = sp.has('avatar');
  const cam = sp.get('cam');
  if (cam === 'third' || cam === '3') out.cam = 'third';
  const h = Number(sp.get('height'));
  if (h > 0.8 && h < 2.4) out.height = h;
  const yaw = Number(sp.get('yaw'));
  if (Number.isFinite(yaw)) out.yaw = yaw;
  const ch = sp.get('choreo');
  if (ch && isChoreoUrl(ch)) out.choreoUrl = ch;
  return out;
}

/**
 * The "ideal player" of a `mirror: true` choreography performs the mirror image of the reference
 * (that is what the scorer expects and what a player copying the facing teacher does). The
 * playback input backend only needs `referenceHeight` and `sampleAt(t, out)`, so it gets this
 * mirrored view instead of the raw choreography.
 */
export function mirroredChoreoView(choreo) {
  if (!choreo.mirror) return choreo;
  const tmp = createChoreoSample();
  const own = createChoreoSample();
  return {
    id: choreo.id,
    mirror: true,
    referenceHeight: choreo.referenceHeight,
    duration: choreo.duration,
    sampleAt: (t, out = own) => mirrorChoreoSample(out, choreo.sampleAt(t, tmp)),
  };
}

function httpBaseOf(server) {
  if (!server) return null;
  return server.replace(/^ws(s?):\/\//i, 'http$1://').replace(/\/$/, '');
}

function formatDate(iso) {
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    const p = (n) => (n < 10 ? '0' : '') + n;
    return `${p(d.getDate())}.${p(d.getMonth() + 1)}. ${p(d.getHours())}:${p(d.getMinutes())}`;
  } catch (e) {
    return '';
  }
}

export class App extends EventEmitter {
  /**
   * @param {object} [opts] { params (config.params), storage (localStorage), document, window,
   *   container, xr (allow WebXR, default true) }
   */
  constructor(opts = {}) {
    super();
    this.params = opts.params || pageParams;
    this.win = opts.window || (typeof window !== 'undefined' ? window : null);
    this.doc = opts.document || (typeof document !== 'undefined' ? document : null);
    this.extra = parseExtraParams(this.win && this.win.location ? this.win.location.search : '');
    this.storage = opts.storage !== undefined ? opts.storage : safeStorage(this.win);
    this.texts = getTexts(this.params.lang);
    this.local = LOCAL_TEXTS[this.params.lang] || LOCAL_TEXTS.de;
    this.version = APP_VERSION;

    this.state = STATES.BOOT;
    this.prevState = null;
    this.sceneKit = null;
    this.menu = null;
    this.hud = null;
    this.teacher = null;
    this.markers = null;
    this.playerBody = null;
    this.skeleton = null;

    this.input = null;
    this.inputKind = null;            // 'xr' | 'desktop' | 'playback'
    this.xrSession = null;
    this.calibration = new Calibration({ storage: this.storage });
    this.audioCtx = null;
    this.audio = new AudioEngine({ audioContext: null });
    this.clock = new BeatClock({ speed: this.params.speed });

    this.choreos = [];                // list entries {id, title, artist, bpm, durationBeats, difficulty, source, url|json}
    this.choreoCache = new Map();
    this.choreo = null;
    this.selectedChoreoId = null;

    this.session = null;
    this.scorer = null;
    this.recorder = null;
    this.lastResult = null;
    this.lastRun = null;
    this.modes = new Map();
    this.modeName = 'solo';
    this.mode = null;
    this.bodyPoseProvider = null;
    this.telemetryExporter = null;
    this.mirror = null;               // PlayerMirror (the player's image on the stage)
    this.neural = null;               // { driver, ready, info, poses, state, notice, fallback }
    this.lastBenchmark = null;        // duo benchmark of the last run (modes.js)
    this.serverHttp = null;           // REST base ('https://host:port'), from ?server= or same origin
    this.serverUrl = null;            // as given (ws(s):// or http(s)://) for the online duo
    this.serverInfo = null;           // /api/info of the same-origin server (when probed)
    this._neuralPoseBuf = null;

    this.playerS = createSample();
    this.settings = this._loadSettings();
    this.pointerHand = 'right';
    this._m16 = new Float64Array(16);
    this._time = NaN;
    this._dt = 0;
    this._calib = null;
    this._review = null;
    this._recordCfg = { title: '', bpm: 90, bars: 8, countInBeats: 4, synth: 'hiphop', mirror: true };
    this._taps = [];
    this._quitHold = NaN;
    this._pendingStart = null;
    this._virtual = null;             // { source } virtual time source of the playback clock
    this._busy = false;
    this._goHideAt = Infinity;
    this._highscoreIndex = 0;
    this._serverHighscores = null;
    this._pendingPost = null;         // { json, timer } result upload waiting for the duo benchmark
    this._swReload = false;           // a new service worker took over: reload when back in the menu
    this._pausedByXR = false;
    this.frameCount = 0;
    this._frame = this._frame.bind(this);
    this._overlay = null;
  }

  // ---- boot ---------------------------------------------------------------------------------

  async boot() {
    const t = this.texts;
    this.calibration.load();
    this.sceneKit = new SceneKit({ container: this.doc ? this.doc.body : null, xr: this.params.emu === null && !!(this.win && this.win.navigator && this.win.navigator.xr) });
    const sk = this.sceneKit;
    this.hud = new HUD({ texts: t, screenWidth: sk.screenWidth, screenHeight: sk.screenHeight });
    this.hud.position.set(0, 0, sk.screenZ);
    sk.stage.add(this.hud);
    this.menu = new Menu({ texts: t, camera: sk.camera, app: this });
    this.menu.position.set(0, 0, -1.9);
    sk.stage.add(this.menu);
    sk.scene.add(this.menu.laser, this.menu.reticle);
    this.teacher = new TeacherAvatar({ color: AVATAR_COLORS.teacher });
    this.teacher.position.set(0, 0, -sk.teacherDistance);
    this.teacher.visible = false;
    sk.stage.add(this.teacher);
    this.markers = new ControllerMarkers({ color: AVATAR_COLORS.player });
    sk.scene.add(this.markers);
    // the player's mirror image on the stage, left of the teacher (the duo partner is right)
    this.mirror = new PlayerMirror({ color: AVATAR_COLORS.player, opacity: 0.85, skeleton: this.skeleton });
    this.mirror.position.set(-STAGE.duoSideOffset, 0, -sk.teacherDistance);
    this.mirror.visible = false;
    sk.stage.add(this.mirror);
    this._applyStageMatrix();
    this._buildPanels();
    registerDuoModes(this);
    this._bindOverlay();
    sk.onContextLost = () => this._setStatus(this.local.contextLost);
    sk.onContextRestored = () => this._reload(this.local.contextLost);
    this._registerServiceWorker();
    this._loadSkeleton();
    this._neuralEnsure();
    sk.setAnimationLoop(this._frame);
    this.setState(STATES.MENU);
    this.menu.show('main');
    // the VR button does not depend on the network: enable it before the (bounded) probes
    if (!this.params.emu) await this._probeXR();
    await this._probeServer();
    await this.loadChoreoList();

    if (this.params.emu) {
      this.enterEmu(this.params.emu);
      const ref = this.extra.choreoUrl || this.params.choreo;
      if (ref && !this.params.autostart) this.selectedChoreoId = ref;
      if (this.params.autostart && ref) {
        this.startSolo(ref).catch((e) => this._fail(t.errorChoreoLoad, e));
      }
    }
    this.emit('boot', this);
    return this;
  }

  /** fetch with a timeout (a wedged server / captive portal must not block the boot or a run). */
  _fetch(url, init = {}, ms = FLOW.networkTimeoutMs) {
    const opts = { ...init };
    if (!opts.signal && typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function') {
      try { opts.signal = AbortSignal.timeout(ms); } catch (e) { /* older browsers */ }
    }
    return fetch(url, opts);
  }

  /** Headers of a mutating API call (server/server.js --token). */
  _apiHeaders(extra = {}) {
    const h = { ...extra };
    if (this.settings.apiToken) h['X-Dp-Token'] = this.settings.apiToken;
    return h;
  }

  _reload(notice) {
    if (notice) { this.hud.showNotice(notice, 3); this._setStatus(notice); }
    const loc = this.win && this.win.location;
    if (loc && typeof loc.reload === 'function') setTimeout(() => loc.reload(), 300);
  }

  _bindOverlay() {
    const doc = this.doc;
    if (!doc) return;
    const $ = (id) => doc.getElementById(id);
    this._overlay = { root: $('overlay'), vr: $('btn-vr'), emu: $('btn-emu'), status: $('status'), version: $('version') };
    const o = this._overlay;
    if (o.version) o.version.textContent = `${APP_NAME} ${this.texts.menuVersion} ${APP_VERSION}`;
    if (o.vr) o.vr.addEventListener('click', () => this.enterVR().catch((e) => this._fail(this.texts.errorXr, e)));
    if (o.emu) o.emu.addEventListener('click', () => this.enterEmu('desktop'));
  }

  _setStatus(text) {
    const o = this._overlay;
    if (o && o.status) o.status.textContent = text || '';
  }

  _showOverlay(show) {
    const o = this._overlay;
    if (o && o.root) o.root.style.display = show ? '' : 'none';
  }

  async _probeXR() {
    const nav = this.win ? this.win.navigator : null;
    const t = this.texts;
    let ok = false;
    if (nav && nav.xr && typeof nav.xr.isSessionSupported === 'function') {
      try { ok = await nav.xr.isSessionSupported('immersive-vr'); } catch (e) { ok = false; }
    }
    const o = this._overlay;
    if (o && o.vr) o.vr.disabled = !ok;
    this._setStatus(ok ? '' : t.menuVrUnavailable);
    this.xrSupported = ok;
  }

  /**
   * Register sw.js. Chromium refuses a service worker on an origin whose certificate error was
   * clicked through (self-signed LAN certificate): the app still runs, but without offline
   * start / model cache - the operator sees it on the start overlay (details: console).
   * Updates: sw.js serves the shell network-first, `registration.update()` checks for a new
   * worker at every boot and a change of controller reloads the page once it is idle.
   */
  _registerServiceWorker() {
    const nav = this.win ? this.win.navigator : null;
    if (!nav || !('serviceWorker' in nav) || !this.win.location || this.win.location.protocol === 'file:') return;
    const sw = nav.serviceWorker;
    const hadController = !!sw.controller;
    try {
      sw.register(`./sw.js?v=${encodeURIComponent(APP_VERSION)}`)
        .then((reg) => { if (reg && typeof reg.update === 'function') reg.update().catch(() => null); })
        .catch((e) => {
          console.warn('[app] service worker registration failed', e);
          const cert = /SSL|certificate/i.test(String(e && e.message));
          this.swError = cert ? 'certificate' : (e && e.message) || 'error';
          this._setStatus(cert ? this.local.swFailed : `${this.texts.error}: ${this.swError}`);
        });
      if (typeof sw.addEventListener === 'function') {
        sw.addEventListener('controllerchange', () => {
          if (!hadController) return;   // first install: the page is already the fresh one
          this._swReload = true;
          this.hud.showNotice(this.local.swUpdated, 4);
          if (this.state === STATES.MENU || this.state === STATES.BOOT) this._reload();
        });
      }
    } catch (e) {
      console.warn('[app] service worker registration failed', e);
    }
  }

  async _loadSkeleton() {
    try {
      const res = await fetch(`./models/${this.params.style}/skeleton.json`, { cache: 'no-cache' });
      if (!res.ok) return;
      const sk = await res.json();
      if (sk && Array.isArray(sk.joints) && Array.isArray(sk.parents)) {
        this.skeleton = sk;
        if (this.mirror && !this.mirror.body) this.mirror.setSkeleton(sk);
      }
    } catch (e) {
      /* the neural lane will provide its own skeleton */
    }
  }

  // ---- input backends ------------------------------------------------------------------------

  _ensureAudioContext() {
    if (this.audioCtx || !this.win) return;
    const AC = this.win.AudioContext || this.win.webkitAudioContext;
    if (!AC) return;
    try {
      this.audioCtx = new AC();
      this.audio = new AudioEngine({ audioContext: this.audioCtx });
      this.audio.setMasterGain(this.settings.volume);
    } catch (e) {
      console.warn('[app] AudioContext unavailable', e);
    }
  }

  _audioLive() {
    return !!(this.audioCtx && this.audioCtx.state === 'running');
  }

  async _resumeAudio() {
    this._ensureAudioContext();
    if (this.audio) await this.audio.resume();
  }

  /** Request an immersive-vr session (must be called from a user gesture). */
  async enterVR() {
    const nav = this.win.navigator;
    if (!nav.xr) throw new Error('WebXR not available');
    this._ensureAudioContext();
    await this._resumeAudio();
    const session = await nav.xr.requestSession('immersive-vr', {
      requiredFeatures: ['local-floor'],
      optionalFeatures: ['hand-tracking', 'bounded-floor'],
    });
    this.xrSession = session;
    this.sceneKit.onSessionEnd = () => this._onXRSessionEnd();
    this.sceneKit.onReferenceSpaceReset = () => this._onRecenter();
    this.sceneKit.onVisibilityChange = (v) => this._onXRVisibility(v);
    await this.sceneKit.startXR(session);
    if (this.input && this.input.dispose) this.input.dispose();
    this.input = new XRInput({ session });
    this.inputKind = 'xr';
    this._showOverlay(false);
    this._setStatus('');
    if (this.state !== STATES.MENU) this.setState(STATES.MENU);
    this.menu.show('main');
    this.emit('input', this.input);
    return session;
  }

  /**
   * The player recentred the view (long press on the Oculus button): 'local-floor' has a new
   * origin and yaw, so the stored calibration (old W frame) is wrong. Drop it, reset the stage
   * and leave any run - the next start calibrates again.
   */
  _onRecenter() {
    this.calibration.cancelCapture();
    this.calibration.clear();
    this.sceneKit.resetStage();
    this._applyStageMatrix();
    const inRun = this.session || (this.recorder && this.recorder.state === 'recording') || this.state === STATES.CALIBRATE;
    if (inRun) this.abortToMenu();
    this.hud.showNotice(this.local.recentered, 4);
    this.emit('recenter', this.calibration.data);
  }

  /** XR visibility (system menu open, headset doffed): pause the run, resume when visible. */
  _onXRVisibility(state) {
    const s = this.session;
    const running = s && (s.state === 'countdown' || s.state === 'playing');
    if (state !== 'visible') {
      if (running && !s.paused) {
        s.pause();
        this._pausedByXR = true;
        if (this._audioLive() && typeof this.audio.pauseTrack === 'function') this.audio.pauseTrack();
        this.hud.showNotice(this.local.paused, 60);
      }
    } else if (this._pausedByXR) {
      this._pausedByXR = false;
      if (running && s.paused) {
        s.resume();
        if (this._audioLive() && typeof this.audio.resumeTrack === 'function') this.audio.resumeTrack();
        this.hud.showNotice(this.local.resumed, 2);
      } else {
        this.hud.showNotice(null);
      }
    }
  }

  _onXRSessionEnd() {
    if (this.session) this.session.abort();
    this._disposeSession();
    this.xrSession = null;
    if (this.input && this.input.dispose) this.input.dispose();
    this.input = new EmulatedInput({ backend: 'desktop', window: this.win, document: this.doc });
    this.inputKind = 'desktop';
    this._showOverlay(true);
    this._setStatus(this.local.vrEnded);
    this.setState(STATES.MENU);
    this.menu.show('main');
    this.hud.setPlayMode(false);
    this.teacher.visible = false;
  }

  /** Enter the desktop/playback emulation (no headset). kind: 'desktop' | 'playback'. */
  enterEmu(kind = 'desktop') {
    this._ensureAudioContext();
    if (this.input && this.input.dispose) this.input.dispose();
    this.input = new EmulatedInput({ backend: 'desktop', window: this.win, document: this.doc, playerHeight: this.extra.height || undefined, playerYaw: this.extra.yaw });
    this.inputKind = kind === 'playback' ? 'playback' : 'desktop';
    this._showOverlay(false);
    this._bindEmuDom();
    this.hud.showNotice(this.local.desktopHint, 6);
    this.emit('input', this.input);
  }

  _bindEmuDom() {
    if (this._emuBound || !this.win || !this.doc) return;
    this._emuBound = true;
    const el = this.sceneKit.renderer.domElement;
    el.addEventListener('mousemove', (e) => {
      const r = el.getBoundingClientRect();
      const x = ((e.clientX - r.left) / Math.max(1, r.width)) * 2 - 1;
      const y = -(((e.clientY - r.top) / Math.max(1, r.height)) * 2 - 1);
      this.menu.setMouse(x, y);
    });
    el.addEventListener('click', () => {
      this._resumeAudio();
      if (this.menu.visible) this.menu.press();
    });
    this.doc.addEventListener('keydown', (e) => {
      if (e.code === 'Escape' && (this.state === STATES.COUNTDOWN || this.state === STATES.PLAYING || this.state === STATES.RECORDING)) {
        this.abortToMenu();
        e.preventDefault();
        return;
      }
      if (this.menu.visible && this.menu.handleKey(e.code)) e.preventDefault();
    });
  }

  /**
   * Playback runs need a backend bound to the choreography; desktop/XR keep their backend.
   * The playback clock is a virtual time source that the App advances in sub-steps of at most
   * one scoring tick (`_updatePlay`), so scripted runs are deterministic and score the full
   * 30 Hz whatever the render frame rate is (headless SwiftShader renders at ~20 fps).
   */
  _inputForChoreo(choreo) {
    if (this.inputKind !== 'playback') return this.input;
    if (this.input && this.input.dispose) this.input.dispose();
    const p = this.params;
    this._virtual = { source: 0 };
    this.clock = new BeatClock({ timeSource: () => this._virtual.source, speed: p.speed });
    this.input = new EmulatedInput({
      backend: 'playback',
      choreo: mirroredChoreoView(choreo),
      clock: this.clock,
      noise: p.noise,
      lag: p.lag,
      seed: p.seed,
      playerHeight: this.extra.height || undefined,
      playerYaw: this.extra.yaw,
    });
    this.emit('input', this.input);
    return this.input;
  }

  // ---- state ----------------------------------------------------------------------------------

  setState(next) {
    if (next === this.state) return;
    this.prevState = this.state;
    this.state = next;
    this.emit('state', { from: this.prevState, to: next });
    this._neuralSync();
    // a new service worker took over during a run: pick up the new version once idle
    if (this._swReload && next === STATES.MENU) { this._swReload = false; this._reload(this.local.swUpdated); }
  }

  /** The stage transform from the calibration (S -> W incl. height scale). */
  _applyStageMatrix() {
    const ref = this.choreo ? this.choreo.referenceHeight : STAGE.referenceHeight;
    this.calibration.stageToWorldMatrix(this._m16, ref);
    this.sceneKit.setStageMatrix(this._m16);
  }

  _fail(text, err) {
    console.warn('[app]', text, err);
    this.hud.showNotice(text, 5);
    this._setStatus(text);
    this._busy = false;
    if (this.state !== STATES.MENU) {
      this.setState(STATES.MENU);
      this.menu.show('main');
    }
  }

  // ---- choreographies ------------------------------------------------------------------------

  /**
   * Build the menu list: shipped entries (index.json) + own recordings are shown at once, the
   * server's entries (uploads, which replace a shipped id) are merged in when its answer arrives
   * (bounded by the network timeout, so a wedged server never blanks the menu).
   */
  async loadChoreoList() {
    const list = [];
    try {
      const res = await this._fetch('./choreos/index.json', { cache: 'no-cache' });
      if (res.ok) {
        const j = await res.json();
        for (const e of (j && j.choreos) || []) {
          if (!e || !e.id) continue;
          list.push({ ...e, source: 'shipped', url: `./choreos/${e.file || e.id + '.json'}` });
        }
      }
    } catch (e) {
      console.warn('[app] choreos/index.json not available', e);
    }
    const addOwn = () => {
      for (const j of Recorder.listPersisted(this.storage)) {
        if (list.some((x) => x.id === j.id)) continue;
        list.push({ id: j.id, title: j.title, artist: j.artist || '', bpm: j.bpm, durationBeats: j.durationBeats, difficulty: j.difficulty || 2, source: 'own', json: j });
      }
    };
    addOwn();
    this._publishChoreos(list);
    const base = this.serverHttp;
    if (base) {
      try {
        const res = await this._fetch(`${base}/api/choreos`, { cache: 'no-cache' });
        if (res.ok) {
          const j = await res.json();
          for (const e of (j && j.choreos) || []) {
            if (!e || !e.id) continue;
            // the server says where to load it from ('/choreos/<file>' shipped, '/api/choreos/<id>' uploads)
            const url = typeof e.url === 'string' && e.url.startsWith('/') ? `${base}${e.url}` : `${base}/api/choreos/${encodeURIComponent(e.id)}`;
            const existing = list.find((x) => x.id === e.id);
            if (existing) {
              if (e.source === 'upload' && existing.source !== 'own') { existing.url = url; existing.source = 'server'; existing.title = e.title || existing.title; }
              continue;
            }
            list.push({ ...e, source: 'server', url });
          }
          this.choreoCache.clear();   // an upload may have replaced a shipped/own id
        }
      } catch (e) {
        console.warn('[app] server choreo list not available', e);
      }
      this._publishChoreos(list);
    }
    return list;
  }

  _publishChoreos(list) {
    this.choreos = list;
    if (!this.selectedChoreoId && list.length > 0) this.selectedChoreoId = list[0].id;
    const main = this.menu.getPanel('main');
    if (main) {
      const li = main.spec.items.find((i) => i.type === 'list');
      if (li) li.selected = this.selectedChoreoId;
      main.refresh();
    }
    this.emit('choreos', list);
  }

  /** Resolve an id, a URL, a JSON object or a Choreo to a Choreo (cached by id/url). */
  async loadChoreoRef(ref) {
    if (ref instanceof Choreo) return ref;
    if (ref && typeof ref === 'object') return new Choreo(ref);
    if (typeof ref !== 'string' || !ref) throw new Error('no choreography reference');
    const cached = this.choreoCache.get(ref);
    if (cached) return cached;
    let choreo;
    if (isChoreoUrl(ref)) {
      choreo = await loadChoreo(ref);
    } else {
      const entry = this.choreos.find((e) => e.id === ref);
      if (entry && entry.json) choreo = new Choreo(entry.json);
      else if (entry && entry.url) choreo = await loadChoreo(entry.url);
      else if (this.serverHttp) {
        // unknown id (e.g. a duo partner's dance uploaded after this headset booted): the server
        // serves uploads and shipped files alike under /api/choreos/<id>
        try { choreo = await loadChoreo(`${this.serverHttp}/api/choreos/${encodeURIComponent(ref)}`); } catch (e) { choreo = await loadChoreo(`./choreos/${ref}.json`); }
      } else choreo = await loadChoreo(`./choreos/${ref}.json`);
    }
    this.choreoCache.set(ref, choreo);
    this.choreoCache.set(choreo.id, choreo);
    return choreo;
  }

  get selectedEntry() {
    return this.choreos.find((e) => e.id === this.selectedChoreoId) || null;
  }

  // ---- play flow -----------------------------------------------------------------------------

  /**
   * Start a run of a choreography (id, URL, JSON or Choreo). opts: { mode='solo', recalibrate,
   * player }. Resolves once the choreography is loaded and the calibration step is entered.
   */
  async startSolo(choreoRef, opts = {}) {
    if (this._busy) { this.hud.showNotice(this.local.startWhileBusy, 2); return null; }
    const ref = choreoRef || this.selectedChoreoId;
    if (!ref) { this.hud.showNotice(this.local.chooseDance, 3); return null; }
    this._busy = true;
    const t = this.texts;
    try {
      if (this.session) { this.session.abort(); this._disposeSession(); }
      this.lastBenchmark = null;      // never attach a previous round's benchmark to this run
      this.menu.hideAll();
      this.hud.setPlayMode(false);
      this.hud.showTitle(this.local.loadingDance);
      const choreo = await this.loadChoreoRef(ref);
      this.choreo = choreo;
      if (this.choreos.some((e) => e.id === choreo.id)) this.selectedChoreoId = choreo.id;
      this.teacher.setChoreo(choreo);
      this.teacher.setMode('auto');   // body whenever the choreography has one (the avatar setting concerns the player)
      this._inputForChoreo(choreo);
      this._applyStageMatrix();
      const modeName = opts.mode || 'solo';
      if (modeName !== 'solo') {
        const mode = this.modes.get(modeName);
        if (!mode) throw new Error(`mode '${modeName}' is not registered`);
        this.mode = mode;
        this.modeName = modeName;
        if (typeof mode.prepare === 'function') await mode.prepare(this._modeContext());
      } else {
        this.mode = null;
        this.modeName = 'solo';
      }
      this._pendingStart = { ref, opts };
      this._enterCalibrate({ then: 'play', recalibrate: !!opts.recalibrate });
      this._busy = false;
      return choreo;
    } catch (e) {
      this._busy = false;
      const msg = e && e.notice ? e.notice : (e && e.errors ? t.errorChoreoInvalid : t.errorChoreoLoad);
      this._fail(msg, e);
      throw e;
    }
  }

  /** Start a registered mode (e.g. 'ghost', 'online') with the selected or given choreography. */
  startMode(name, choreoRef = null, opts = {}) {
    return this.startSolo(choreoRef || this.selectedChoreoId, { ...opts, mode: name });
  }

  _modeContext() {
    return {
      app: this,
      choreo: this.choreo,
      calibration: this.calibration,
      sceneKit: this.sceneKit,
      stage: this.sceneKit.stage,
      menu: this.menu,
      hud: this.hud,
      texts: this.texts,
      teacher: this.teacher,
      input: this.input,
      clock: this.clock,
      storage: this.storage,
      server: this.serverUrl,
      httpBase: this.serverHttp,
    };
  }

  _enterCalibrate({ then, recalibrate }) {
    this._calib = { then, recalibrate: !!recalibrate, startedAt: NaN, holdSince: NaN, holding: false, progress: 0, drawnProgress: -1, done: false };
    this.hud.setPlayMode(false);
    this.hud.showTitle(this.texts.calibrateTitle);
    this.teacher.visible = false;
    this.menu.show('calibrate');
    this.setState(STATES.CALIBRATE);
  }

  _calibrationStillValid() {
    const c = this.calibration;
    if (!c.valid || !this.input) return false;
    const age = (Date.now() - Date.parse(c.data.capturedAt || 0)) / 1000;
    if (!(age >= 0 && age < CALIBRATION_REUSE_SEC)) return false;
    const head = this.input.sample.head;
    if (!head.valid) return false;
    if (Math.abs(head.p[1] - c.h0) > 0.10) return false;
    const dx = head.p[0] - c.origin[0], dz = head.p[2] - c.origin[2];
    if (dx * dx + dz * dz >= 0.5 * 0.5) return false;
    // facing: after a recentre (or standing on the marker turned around) the saved yaw is wrong
    const yaw = yawFromQuat(head.q, NaN);
    if (!Number.isFinite(yaw)) return false;
    return Math.abs(angleDiff(yaw, c.yaw)) < CALIBRATION_MAX_YAW_DIFF;
  }

  _updateCalibrate(time) {
    const c = this._calib;
    if (!c || c.done) return;
    if (!Number.isFinite(c.startedAt)) c.startedAt = time;
    const elapsed = (time - c.startedAt) / 1000;
    const input = this.input;
    this.sceneKit.setMarkerHighlight(true, time / 250);
    if (!input) return;
    if (input.emulated) {
      const wait = FLOW.calibrationEmuSec / this.clock.speed;
      c.progress = Math.min(1, elapsed / wait);
      if (elapsed >= wait) {
        const data = this.calibration.capture(input.sample);
        this._finishCalibrate(!!data);
      }
      return;
    }
    // trigger hold = fresh capture (also overrides a reusable calibration); not while the laser
    // is on a button of the panel ('Abbrechen' is pressed with the same trigger)
    if (input.anyTriggerDown() && !this.menu.hovering) {
      if (!c.holding) {
        c.holding = true;
        c.holdSince = time;
        this.calibration.beginCapture();
      }
      this.calibration.addSample(input.sample);
      c.progress = Math.min(1, (time - c.holdSince) / 1000 / FLOW.calibrationHoldSec);
      if (c.progress >= 1) {
        const data = this.calibration.endCapture();
        c.holding = false;
        this._finishCalibrate(!!data);
      }
      return;
    }
    if (c.holding) {
      c.holding = false;
      c.progress = 0;
      this.calibration.cancelCapture();   // a short tap must not install a 1-2 frame average
    }
    // reuse of the persisted calibration: announce it, then give the player a moment to lower
    // the controller and get ready before the count-in starts
    if (!c.recalibrate && elapsed > 0.6 && this._calibrationStillValid()) {
      if (!c.keptSince) {
        c.keptSince = time;
        this.hud.showNotice(`${this.local.calibrateKept} · ${this.local.readyHold}`, FLOW.calibrationReuseWaitSec + 0.5);
      }
      const wait = FLOW.calibrationReuseWaitSec / this.clock.speed;
      c.progress = Math.min(1, (time - c.keptSince) / 1000 / wait);
      if (c.progress >= 1) this._finishCalibrate(true, true);
    } else if (c.keptSince) {
      c.keptSince = 0;
      c.progress = 0;
    }
  }

  _finishCalibrate(ok, kept = false) {
    const c = this._calib;
    const t = this.texts;
    if (!ok) {
      this.hud.showNotice(t.calibrateFailed, 4);
      c.progress = 0;
      c.holding = false;
      return;
    }
    c.done = true;
    if (!kept) {
      this.calibration.save();
      this.hud.showNotice(`${t.calibrateDone} ${t.calibrateHeight} ${this.calibration.h0.toFixed(2)} m`, 2.5);
    }
    this.sceneKit.setMarkerHighlight(false);
    this._applyStageMatrix();
    this.menu.hideAll();
    this.emit('calibrated', this.calibration.data);
    if (c.then === 'play') {
      this._startPlaySession().catch((e) => this._fail(t.errorChoreoLoad, e));
    } else if (c.then === 'record') {
      this._startRecording();
    } else {
      this.setState(STATES.MENU);
      this.menu.show('main');
      this.hud.showTitle(null);
    }
  }

  async _startPlaySession() {
    const choreo = this.choreo;
    const t = this.texts;
    this.scorer = new Scorer(choreo);
    // the playback (virtual) clock cannot drive WebAudio scheduling: no audio in that mode
    const live = this._audioLive() && !this._virtual;
    this.clock.audioContext = live ? this.audioCtx : null;
    this.clock.speed = this.params.speed;
    let audio = null;
    if (live) {
      audio = this.audio;
      audio.setMasterGain(this.settings.volume);
      audio.track = null;
      if (choreo.audio && choreo.audio.url) {
        const ok = await audio.loadTrack(choreo.audio.url, { offsetSec: choreo.audio.offsetSec, gain: choreo.audio.gain });
        if (!ok) this.hud.showNotice(t.errorAudioLoad, 4);
      }
    }
    const session = new PlaySession({
      choreo,
      clock: this.clock,
      scorer: this.scorer,
      input: this.input,
      calibration: this.calibration,
      audio,
      mode: this.modeName,
      player: this.settings.playerName || '',
    });
    this.session = session;
    const hud = this.hud;
    session.on('countdown', (e) => hud.showCountdown(e.beatsLeft));
    session.on('beat', (e) => {
      if (e.beat === 0) {
        hud.showCountdown('go');
        this._goHideAt = choreo.beatDuration * 0.75;
      }
    });
    session.on('move', (e) => {
      const next = choreo.moves[e.index + 1];
      hud.showMove(e.move.name, next ? next.name : null, e.move.hint);
    });
    session.on('grade', (e) => {
      hud.showGrade(e.grade, e.combo);
      hud.showScore(session.score, e.combo);
    });
    session.on('finished', (result) => this._onFinished(result));
    if (this.mode && typeof this.mode.onSessionCreated === 'function') this.mode.onSessionCreated(session, this._modeContext());
    hud.reset();
    hud.setPlayMode(true);
    hud.showTitle(choreo.title);
    hud.showScore(0, 0);
    hud.showMove(choreo.moves[0] ? choreo.moves[0].name : '', choreo.moves[1] ? choreo.moves[1].name : null);
    hud.showTime(choreo.duration);
    this.teacher.visible = true;
    this.teacher.update(0);
    this._quitHold = NaN;
    // modes may pin the start to a synchronised clock time (online duo: BeatClock startAt)
    const startOpts = { speed: this.params.speed };
    if (this.mode && typeof this.mode.startOptions === 'function') {
      try { Object.assign(startOpts, this.mode.startOptions(this._modeContext()) || {}); } catch (e) { console.warn('[app] mode.startOptions failed', e); }
    }
    session.start(startOpts);
    this.setState(STATES.COUNTDOWN);
    this.emit('session', session);
    return session;
  }

  _updatePlay(time, dt) {
    const s = this.session;
    if (!s) return;
    if (this._virtual) {
      // deterministic playback: advance the virtual source clock by the real frame time in
      // sub-steps no longer than one scoring tick (in game time), scoring every tick
      const stepSource = (1 / s.scoreHz) / this.clock.speed;
      const advance = dt > 0 ? dt : 1 / 72;
      const n = Math.max(1, Math.ceil(advance / stepSource - 1e-9));
      const ds = advance / n;
      for (let i = 0; i < n; i++) {
        this._virtual.source += ds;
        s.update(time);
        if (s.state === 'finished' || s.state === 'aborted') break;
      }
    } else {
      s.update(time);
    }
    const st = s.state;
    if (st === 'countdown' && this.state !== STATES.COUNTDOWN) this.setState(STATES.COUNTDOWN);
    else if (st === 'playing' && this.state !== STATES.PLAYING) this.setState(STATES.PLAYING);
    if (st === 'finished' || st === 'aborted') return;
    const t = s.t;
    this.teacher.update(t < 0 ? 0 : t, s.refSample);
    this.hud.showProgress(s.progress);
    this.hud.showTime(Math.max(0, this.choreo.duration - t));
    if (t >= this._goHideAt) { this.hud.showCountdown(null); this._goHideAt = Infinity; }
    if (this.mode && typeof this.mode.update === 'function') this.mode.update(time, dt, s);
  }

  /**
   * Hold B/Y (secondary) for QUIT_HOLD_SEC to leave a run, a recording or the calibration in XR.
   * A resting thumb presses B/Y easily, hence the long hold with a visible countdown on the
   * wall notice; releasing resets it.
   */
  _updateQuitHold(time) {
    if (this.inputKind !== 'xr' || !this.input) return;
    if (this.input.isDown('any', 'secondary')) {
      if (!Number.isFinite(this._quitHold)) { this._quitHold = time; this._quitShown = -1; }
      const held = (time - this._quitHold) / 1000;
      if (held >= QUIT_HOLD_SEC) { this._quitHold = NaN; this.abortToMenu(); return; }
      const left = Math.ceil(QUIT_HOLD_SEC - held);
      if (left !== this._quitShown) {
        this._quitShown = left;
        this.hud.showNotice(this.local.holdToQuit.replace('{sec}', String(left)), 1.5);
      }
    } else if (Number.isFinite(this._quitHold)) {
      this._quitHold = NaN;
      this.hud.showNotice(null);
    }
  }

  _onFinished(result) {
    this.lastResult = result;
    const s = this.session;
    this.lastRun = s ? s.runToJSON() : null;
    this.hud.showCountdown(null);
    this.hud.showScore(result.score, s ? s.combo : 0);
    let extra = null;
    if (this.mode && typeof this.mode.onFinished === 'function') {
      try { extra = this.mode.onFinished(result, this._modeContext()) || null; } catch (e) { console.warn('[app] mode.onFinished failed', e); }
    }
    this._storeResult(result);
    this._storeRun(this.lastRun);
    this._showResults(result, extra);
    this.setState(STATES.RESULTS);
    this.emit('result', result);
  }

  _disposeSession() {
    if (this.session) {
      this.session.dispose();
      this.session = null;
    }
    if (this.mode && typeof this.mode.exit === 'function') {
      try { this.mode.exit(this._modeContext()); } catch (e) { console.warn('[app] mode.exit failed', e); }
    }
  }

  /** Leave any run / recording and return to the main menu. */
  abortToMenu() {
    this.calibration.cancelCapture();
    this._quitHold = NaN;
    if (this.session) this.session.abort();
    if (this.recorder && this.recorder.state === 'recording') {
      this.recorder.cancel();
      this.clock.stop();
      this.audio.stop();
      this.hud.showNotice(this.local.recordAborted, 3);
    }
    this._disposeSession();
    this._review = null;
    this._calib = null;
    this.sceneKit.setMarkerHighlight(false);
    this.hud.setPlayMode(false);
    this.hud.showCountdown(null);
    this.hud.showTitle(null);
    this.teacher.visible = false;
    this.setState(STATES.MENU);
    this.menu.show('main');
  }

  /** Play the last choreography again (same mode; the online duo returns to its lobby). */
  playAgain() {
    const p = this._pendingStart;
    this._disposeSession();
    if (this.modeName === 'online') {
      this.hud.setPlayMode(false);
      this.hud.showTitle(null);
      this.teacher.visible = false;
      this.openDuoLobby();
      return Promise.resolve(null);
    }
    if (!p) return this.startSolo(this.selectedChoreoId);
    return this.startSolo(p.ref, p.opts).catch(() => null);
  }

  // ---- results / persistence -----------------------------------------------------------------

  _showResults(result, extra) {
    const t = this.texts, l = this.local;
    const bias = result.timingBias;
    const timingText = Math.abs(bias) < 0.03 ? t.resultsTimingOnBeat : (bias > 0 ? t.resultsTimingLate : t.resultsTimingEarly);
    const items = [
      { type: 'label', text: this.choreo ? this.choreo.title : result.choreoId, size: 'small', align: 'center' },
      { type: 'label', text: `${result.score} ${t.resultsScore}`, size: 'large', align: 'center', color: '#4cc9f0' },
      { type: 'label', text: starsString(result.stars), size: 'large', align: 'center', color: '#ffd166' },
      { type: 'label', text: `${t.resultsMaxCombo}: ×${result.maxCombo}`, align: 'center' },
      { type: 'label', text: `${timingText} (${Math.round(bias * 1000)} ${l.ms})`, size: 'small', align: 'center' },
    ];
    if (extra && Array.isArray(extra.lines)) {
      // lines may be functions (duo modes refresh them when the partner's result arrives)
      for (const line of extra.lines) items.push({ type: 'label', text: typeof line === 'function' ? line : String(line), align: 'center', size: 'small', color: '#b388ff' });
    }
    items.push({ type: 'spacer' });
    items.push({
      type: 'row',
      items: [
        { type: 'button', id: 'again', text: t.resultsAgain, primary: true, onClick: () => this.playAgain() },
        { type: 'button', id: 'menu', text: t.resultsMenu, onClick: () => this.abortToMenu() },
      ],
    });
    if (extra && Array.isArray(extra.buttons) && extra.buttons.length) items.push({ type: 'row', items: extra.buttons });
    this.menu.setPanel('results', { title: t.resultsTitle, width: 1.5, position: [-0.8, 1.55, 0], rotationY: 0.12, items, onBack: () => this.abortToMenu() });
    const moveItems = [{ type: 'label', text: l.perMoveHeader, size: 'small' }];
    for (const m of result.moves) {
      const grade = t[GRADE_KEYS[m.grade]] || m.grade;
      moveItems.push({ type: 'label', text: `${m.name}: ${Math.round(m.score * 100)} – ${grade}${m.gated ? ' ⚠' : ''}`, size: 'small', color: gradeColor(m.grade) });
    }
    if (extra && Array.isArray(extra.moveLines)) {
      for (const line of extra.moveLines) moveItems.push({ type: 'label', text: typeof line === 'function' ? line : String(line), size: 'small' });
    }
    this.menu.setPanel('results-moves', { title: t.resultsPerMove, width: 1.5, position: [0.8, 1.55, 0], rotationY: -0.12, items: moveItems });
    this.menu.show('results', ['results-moves']);
  }

  _storeResult(result) {
    if (!this.storage) return;
    try {
      const raw = this.storage.getItem(STORAGE_KEYS.results);
      let arr = raw ? JSON.parse(raw) : [];
      if (!Array.isArray(arr)) arr = [];
      const json = resultToJSON(result);
      json.choreoTitle = this.choreo ? this.choreo.title : '';
      const b = this.lastBenchmark;
      if (b && b.choreoId === result.choreoId && result.mode !== 'solo') json.benchmark = compactBenchmark(b);
      arr.push(json);
      if (arr.length > MAX_RESULTS) arr = arr.slice(arr.length - MAX_RESULTS);
      this.storage.setItem(STORAGE_KEYS.results, JSON.stringify(arr));
      this._flushPendingPost();
      if (result.mode === 'online' && !json.benchmark) {
        // the partner's result usually arrives a few ms later: upload once the benchmark is
        // attached (attachBenchmark) or after a bounded wait
        this._pendingPost = { json, timer: setTimeout(() => this._flushPendingPost(), RESULT_POST_DELAY_MS) };
      } else {
        this._postResult(json);
      }
    } catch (e) {
      console.warn('[app] storing the result failed', e);
    }
  }

  _postResult(json) {
    const base = this.serverHttp;
    if (!base) return;
    this._fetch(`${base}/api/results`, { method: 'POST', headers: this._apiHeaders({ 'content-type': 'application/json' }), body: JSON.stringify(json) }, 8000)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); this.hud.showNotice(this.texts.resultsSaved, 2); })
      .catch((e) => { console.warn('[app] result upload failed', e); this.hud.showNotice(this.texts.resultsSaveFailed, 3); });
  }

  _flushPendingPost() {
    const p = this._pendingPost;
    if (!p) return;
    this._pendingPost = null;
    clearTimeout(p.timer);
    this._postResult(p.json);
  }

  /**
   * Duo modes: the benchmark became known after the result was stored (the partner's result
   * arrived late). Update the stored copy and release the pending upload with it.
   */
  attachBenchmark(benchmark, result = this.lastResult) {
    if (!benchmark || !result || benchmark.choreoId !== result.choreoId) return false;
    this.lastBenchmark = benchmark;
    const compact = compactBenchmark(benchmark);
    let updated = false;
    if (this.storage) {
      try {
        const raw = this.storage.getItem(STORAGE_KEYS.results);
        const arr = raw ? JSON.parse(raw) : [];
        if (Array.isArray(arr)) {
          const key = resultKey(result);
          for (let i = arr.length - 1; i >= 0; i--) {
            if (arr[i] && resultKey(arr[i]) === key) { arr[i].benchmark = compact; updated = true; break; }
          }
          if (updated) this.storage.setItem(STORAGE_KEYS.results, JSON.stringify(arr));
        }
      } catch (e) {
        console.warn('[app] updating the stored benchmark failed', e);
      }
    }
    if (this._pendingPost && resultKey(this._pendingPost.json) === resultKey(result)) {
      this._pendingPost.json.benchmark = compact;
      this._flushPendingPost();
    }
    return updated;
  }

  /** localStorage['dp.runs'] = array of dancing-points-run/1 objects (newest last, 3 per choreo). */
  _storeRun(run) {
    if (!run || !this.storage) return;
    try {
      const raw = this.storage.getItem(STORAGE_KEYS.runs);
      let arr = raw ? JSON.parse(raw) : [];
      if (!Array.isArray(arr)) arr = [];
      arr.push(run);
      const same = arr.filter((r) => r && r.choreoId === run.choreoId);
      while (same.length > MAX_RUNS_PER_CHOREO) {
        const victim = same.shift();
        arr.splice(arr.indexOf(victim), 1);
      }
      this.storage.setItem(STORAGE_KEYS.runs, JSON.stringify(arr));
      const base = this.serverHttp;
      if (base) {
        this._fetch(`${base}/api/runs`, { method: 'POST', headers: this._apiHeaders({ 'content-type': 'application/json' }), body: JSON.stringify(run) }, 15000)
          .catch((e) => console.warn('[app] run upload failed', e));
      }
    } catch (e) {
      console.warn('[app] storing the run failed', e);   // quota: runs are optional (ghost duo)
    }
  }

  /** Local results (parsed array) or []. */
  localResults() {
    if (!this.storage) return [];
    try {
      const raw = this.storage.getItem(STORAGE_KEYS.results);
      const arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  /**
   * Top-n results of a choreography: local + (cached) server results, de-duplicated (every
   * play of this headset is on the server too; only server-only entries are tagged
   * `fromServer` = played on another headset).
   */
  highscores(choreoId, n = 10) {
    const local = this.localResults().filter((r) => r && r.choreoId === choreoId);
    const server = [];
    if (Array.isArray(this._serverHighscores)) {
      for (const r of this._serverHighscores) if (r && r.choreoId === choreoId) server.push({ ...r, fromServer: true });
    }
    const all = mergeResults(local, server);
    all.sort((a, b) => (b.score - a.score) || (Date.parse(b.playedAt || 0) - Date.parse(a.playedAt || 0)));
    return all.slice(0, n);
  }

  async _fetchServerHighscores() {
    const base = this.serverHttp;
    if (!base) return;
    try {
      // the server returns the newest results first (default 100): ask for its maximum so that
      // an all-time best of a busy day is not pushed out by the afternoon's plays
      const res = await this._fetch(`${base}/api/results?limit=${SERVER_HIGHSCORE_LIMIT}`, { cache: 'no-cache' }, 8000);
      if (!res.ok) return;
      const j = await res.json();
      const arr = Array.isArray(j) ? j : (j && Array.isArray(j.results) ? j.results : []);
      this._serverHighscores = arr;
      this.menu.refresh('highscores');
    } catch (e) {
      console.warn('[app] server results not available', e);
    }
  }

  // ---- record flow ---------------------------------------------------------------------------

  startRecordSetup() {
    const cfg = this._recordCfg;
    if (!cfg.title) cfg.title = `${this.local.ownDance} ${Recorder.listPersisted(this.storage).length + 1}`;
    this._taps.length = 0;
    this.setState(STATES.RECORD_SETUP);
    this.menu.show('record-setup');
  }

  _tapTempo() {
    const now = performance.now();
    const taps = this._taps;
    if (taps.length && now - taps[taps.length - 1] > 2500) taps.length = 0;
    taps.push(now);
    if (taps.length > 8) taps.shift();
    if (taps.length >= 2) {
      const span = (taps[taps.length - 1] - taps[0]) / 1000;
      const bpm = Math.round(60 * (taps.length - 1) / span);
      if (bpm >= 40 && bpm <= 240) this._recordCfg.bpm = bpm;
    }
  }

  /** From the setup panel: calibrate, then count in and record. */
  beginRecording() {
    if (this.session) { this.session.abort(); this._disposeSession(); }
    this.choreo = null;
    this.teacher.visible = false;
    this._applyStageMatrix();
    if (this.inputKind === 'playback') {
      // playback emu has no live performer; record the desktop pose with a real-time clock
      if (this.input && this.input.dispose) this.input.dispose();
      this.input = new EmulatedInput({ backend: 'desktop', window: this.win, document: this.doc });
      this.inputKind = 'desktop';
      this._virtual = null;
      this.clock = new BeatClock({ speed: this.params.speed });
      this.emit('input', this.input);
    }
    this._enterCalibrate({ then: 'record' });
  }

  _startRecording() {
    const cfg = this._recordCfg;
    const live = this._audioLive();
    this.clock.audioContext = live ? this.audioCtx : null;
    this.clock.speed = this.params.speed;
    this.recorder = new Recorder({ fps: 30 });
    const config = {
      title: cfg.title,
      bpm: cfg.bpm,
      bars: cfg.bars,
      beatsPerBar: 4,
      countInBeats: cfg.countInBeats,
      synth: cfg.synth,
      mirror: cfg.mirror,
      referenceHeight: STAGE.referenceHeight,
      author: this.settings.playerName || '',
    };
    this.recorder.start(config);
    this.clock.start(cfg.bpm, cfg.countInBeats, { beatsPerBar: 4, speed: this.params.speed });
    if (live) {
      this.audio.setMasterGain(this.settings.volume);
      this.audio.track = null;
      this.audio.start(this.clock, { synth: cfg.synth, countInBeats: cfg.countInBeats, beatsPerBar: 4, durationBeats: this.recorder.config.durationBeats });
    }
    this.hud.reset();
    this.hud.setPlayMode(true);
    this.hud.showTitle(cfg.title);
    this.hud.showScore(0, 0);
    this.hud.showMove(this.texts.recording, null);
    this.hud.scorePanel.visible = false;
    this.setState(STATES.COUNTDOWN);
    this.emit('recording', this.recorder);
  }

  _updateRecording(time) {
    const rec = this.recorder;
    if (!rec || rec.state !== 'recording') return;
    const st = this.clock.now();
    if (this._audioLive() && this.audio.playing) this.audio.update();
    if (st.t < 0) {
      this.hud.showCountdown(Math.ceil(-st.beat));
      if (this.state !== STATES.COUNTDOWN) this.setState(STATES.COUNTDOWN);
      return;
    }
    if (this.state !== STATES.RECORDING) {
      this.setState(STATES.RECORDING);
      this.hud.showCountdown('go');
      this._goHideAt = this.clock.beatDuration * 0.75;
    }
    if (st.t >= this._goHideAt) { this.hud.showCountdown(null); this._goHideAt = Infinity; }
    rec.update(st, this.playerS);
    this.hud.showProgress(rec.progress);
    this.hud.showTime(Math.max(0, rec.durationSec - st.t));
    if (rec.complete) this._finishRecording();
  }

  _finishRecording() {
    const t = this.texts;
    let json;
    try {
      json = this.recorder.stop();
    } catch (e) {
      this.clock.stop();
      this.audio.stop();
      this._fail(t.recordTooShort, e);
      return;
    }
    this.clock.stop();
    this.audio.stop();
    this.hud.setPlayMode(false);
    this.hud.showCountdown(null);
    let choreo = null;
    try { choreo = new Choreo(json); } catch (e) { console.warn('[app] recorded choreo invalid', e); }
    this._review = { json, choreo, t: 0, playing: true };
    if (choreo) {
      this.teacher.setChoreo(choreo);
      this.teacher.setMode('points');
      this.teacher.visible = true;
    }
    this.hud.showTitle(t.recordReviewTitle);
    this.menu.refresh('record-review');
    this.menu.show('record-review');
    this.setState(STATES.RECORD_REVIEW);
    this.emit('recorded', json);
  }

  _updateReview(dt) {
    const r = this._review;
    if (!r || !r.choreo || !r.playing) return;
    r.t += dt * this.clock.speed;
    if (r.t > r.choreo.duration + 0.5) r.t = 0;
    this.teacher.update(r.t);
  }

  async saveRecording() {
    const r = this._review;
    const t = this.texts;
    if (!r || !r.json) return false;
    const json = r.json;
    Recorder.persist(json, this.storage);
    const base = this.serverHttp;
    let ok = true;
    if (base) {
      try {
        const token = this.settings.apiToken;
        const fetchImpl = token ? (url, init = {}) => fetch(url, { ...init, headers: { ...(init.headers || {}), 'X-Dp-Token': token } }) : undefined;
        await Recorder.upload(json, base, fetchImpl);
        this.hud.showNotice(t.recordSaved, 3);
      } catch (e) {
        ok = false;
        console.warn('[app] upload failed', e);
        this.hud.showNotice(`${t.recordSaveFailed} – ${t.recordSavedLocal}`, 4);
      }
    } else {
      if (this.inputKind !== 'xr') Recorder.download(json, this.doc);
      this.hud.showNotice(t.recordSavedLocal, 3);
    }
    this.choreoCache.set(json.id, new Choreo(json));
    await this.loadChoreoList();
    this.selectedChoreoId = json.id;
    const main = this.menu.getPanel('main');
    if (main) { const li = main.spec.items.find((i) => i.type === 'list'); if (li) li.selected = json.id; main.refresh(); }
    this.discardRecording(false);
    this.emit('saved', json);
    return ok;
  }

  discardRecording(notify = true) {
    this._review = null;
    this.teacher.visible = false;
    this.teacher.setMode('auto');
    if (notify) this.hud.showNotice(this.local.recordAborted, 2);
    this.hud.showTitle(null);
    this.setState(STATES.MENU);
    this.menu.show('main');
  }

  // ---- settings ------------------------------------------------------------------------------

  _loadSettings() {
    const s = { avatar: 'neural', volume: 0.8, lang: 'de', playerName: '', telemetry: false };
    if (this.storage) {
      try {
        const raw = this.storage.getItem(STORAGE_KEYS.settings);
        if (raw) Object.assign(s, JSON.parse(raw));
      } catch (e) { /* ignore */ }
    }
    if (this.params.avatar) s.avatar = this.params.avatar;
    if (this.params.lang) s.lang = this.params.lang;
    if (this.params.telemetry) s.telemetry = true;
    if (this.params.token) s.apiToken = this.params.token;   // ?token= once, then persisted per headset
    if (!AVATAR_MODES.includes(s.avatar)) s.avatar = 'neural';
    // an explicit ?avatar=neural retries even after the guard disabled the network on this headset
    if (this.params.avatar === 'neural' && this.extra && this.extra.avatarExplicit) s.neuralVerdict = null;
    // the neural avatar is the default on the headset; the desktop/test emulation uses the
    // light point avatar unless ?avatar=neural asks for the network explicitly
    if (this.params.emu && s.avatar === 'neural' && !(this.extra && this.extra.avatarExplicit && this.params.avatar === 'neural')) s.avatar = 'points';
    if (!(s.volume >= 0 && s.volume <= 1)) s.volume = 0.8;
    return s;
  }

  saveSettings() {
    if (!this.storage) return;
    try { this.storage.setItem(STORAGE_KEYS.settings, JSON.stringify(this.settings)); } catch (e) { /* ignore */ }
    this.emit('settings', this.settings);
  }

  setAvatarMode(mode) {
    if (!AVATAR_MODES.includes(mode)) return;
    this.settings.avatar = mode;
    if (this.playerBody) this.playerBody.visible = mode === 'neural' && this.playerBody.hasPose;
    if (mode === 'neural') {
      // choosing 'Neural' in the settings is the operator's decision to try the network again
      if (this.settings.neuralVerdict) {
        this.settings.neuralVerdict = null;
        if (this.neural && !this.neural.driver) this.neural = null;
      }
      this._neuralEnsure();
    }
    this._neuralSync();
    if (this.mirror && mode === 'off') this.mirror.visible = false;
    this.saveSettings();
  }

  setVolume(v) {
    this.settings.volume = Math.max(0, Math.min(1, Math.round(v * 100) / 100));
    if (this.audio) this.audio.setMasterGain(this.settings.volume);
    this.saveSettings();
  }

  openSettings() {
    this.setState(STATES.SETTINGS);
    this.menu.show('settings');
  }

  openHighscores() {
    this._highscoreIndex = Math.max(0, this.choreos.findIndex((e) => e.id === this.selectedChoreoId));
    this.setState(STATES.MENU);
    this.menu.show('highscores');
    this._fetchServerHighscores();
  }

  openDuoLobby() {
    this.setState(STATES.DUO_LOBBY);
    const also = [];
    for (const m of this.modes.values()) if (m.lobbyPanelId && this.menu.getPanel(m.lobbyPanelId)) also.push(m.lobbyPanelId);
    this.menu.show('duo', also);
    this.emit('lobby', this._modeContext());
    // dances uploaded by the other headset since boot must be startable here
    if (this.serverHttp) this.loadChoreoList().catch(() => null);
  }

  backToMenu() {
    this.setState(STATES.MENU);
    this.menu.show('main');
  }

  exportTelemetry() {
    if (typeof this.telemetryExporter === 'function') {
      try {
        const r = this.telemetryExporter(this);
        if (r === false) this.hud.showNotice(this.local.telemetryNone, 4);
      } catch (e) {
        console.warn('[app] telemetry export failed', e);
        this.hud.showNotice(this.texts.error, 3);
      }
    } else {
      this.hud.showNotice(this.local.telemetryNone, 4);
    }
  }

  // ---- extension points ----------------------------------------------------------------------

  /**
   * Register a game mode (duo lanes). `mode` = { label?, lobbyPanelId?, init(app)?,
   * prepare(ctx) async?, onSessionCreated(session, ctx)?, update(time, dt, session)?,
   * onFinished(result, ctx) -> {lines[], moveLines[]}?, exit(ctx)? }.
   * Start it with `app.startMode(name, choreoId)`.
   */
  registerMode(name, mode) {
    if (!name || !mode) throw new TypeError('registerMode(name, mode)');
    mode.name = name;
    this.modes.set(name, mode);
    if (typeof mode.init === 'function') {
      try { mode.init(this); } catch (e) { console.warn(`[app] mode '${name}' init failed`, e); }
    }
    this.menu.refresh('duo');
    this.emit('mode', { name, mode });
    return mode;
  }

  /**
   * Provide full-body joint positions of the player (neural lane): `fn(timeMs, playerS, app)`
   * returns a Float32Array(jointCount*3) in the stage frame S or null. `skeleton` = {joints,
   * parents} (defaults to models/<style>/skeleton.json). Pass null to remove the avatar.
   */
  setBodyAvatarPoseProvider(fn, skeleton = null, opts = {}) {
    this.bodyPoseProvider = typeof fn === 'function' ? fn : null;
    // opts.unscaled: the provider returns real-world metres (NeuralAvatarDriver.getPose); the
    // stage group is at reference height, so the pose is multiplied by k = playerS.k first
    this._bodyPoseUnscaled = opts.unscaled === true;
    this._bodyPoseScaled = null;
    const sk = skeleton || this.skeleton;
    if (this.playerBody) {
      this.sceneKit.stage.remove(this.playerBody);
      this.playerBody.dispose();
      this.playerBody = null;
    }
    if (this.bodyPoseProvider && sk) {
      // the self body has no head sphere (the camera sits inside the head in VR)
      this.playerBody = new BodyAvatar({ skeleton: sk, color: AVATAR_COLORS.player, opacity: 0.75, headSphere: false });
      this.playerBody.visible = false;
      this.sceneKit.stage.add(this.playerBody);
    }
    return this.playerBody;
  }

  // ---- neural avatar -------------------------------------------------------------------------

  /**
   * Create + load the NeuralAvatarDriver once (settings.avatar === 'neural'). The models load in
   * the background while the player is in the menu; the worker is fed only during a dance
   * (`_neuralSync`). The guard's fallback ('disabled') and load errors leave `neural.fallback`
   * set and the mirror shows the three points instead (PointAvatar).
   */
  _neuralEnsure() {
    if (this.neural || this.settings.avatar !== 'neural') return this.neural;
    const t = this.texts;
    const n = { driver: null, ready: false, info: null, poses: 0, state: 'idle', notice: '', fallback: null, loadMs: 0 };
    this.neural = n;
    const verdict = this.settings.neuralVerdict;
    if (verdict && verdict.disabled) {
      // the guard disabled the network on this headset earlier: do not download and re-test
      // 85 MB of models on every boot (the settings button 'Neural' clears the verdict)
      n.fallback = `disabled earlier (${verdict.reason || 'too slow'}${verdict.inferenceMs ? `, ${Math.round(verdict.inferenceMs)} ms` : ''})`;
      n.state = 'disabled';
      console.warn('[app] neural avatar skipped: ' + n.fallback);
      return n;
    }
    if (typeof Worker === 'undefined') {
      n.fallback = 'no Web Worker support';
      n.state = 'error';
      console.warn('[app] neural avatar unavailable: ' + n.fallback);
      return n;
    }
    let driver;
    try {
      driver = new NeuralAvatarDriver({
        style: this.params.style,
        telemetry: !!this.settings.telemetry,
        telemetryTicks: NET.telemetryTicks,
        rootCorrection: NET.rootCorrection,
        eulerRatio: NET.eulerRatio,
        guard: { windowTicks: NET.guardWindowTicks, slowMs: NET.slowMs, disableMs: NET.disableMs },
        onStatus: (s) => this._onNeuralStatus(s),
        onError: (e) => { if (!n.fallback) n.fallback = e && e.message ? e.message : String(e); },
        onPose: () => { n.poses++; },
      });
    } catch (e) {
      n.fallback = e && e.message ? e.message : String(e);
      n.state = 'error';
      console.warn('[app] neural avatar unavailable: ' + n.fallback);
      return n;
    }
    n.driver = driver;
    this.setTelemetryExporter(() => (n.driver && n.driver.worker && n.driver.telemetry ? n.driver.downloadTelemetry() : false));
    const t0 = performance.now();
    driver.init().then((info) => {
      n.ready = true;
      n.info = info;
      n.loadMs = performance.now() - t0;
      const skeleton = { joints: info.joints, parents: info.parents, boneLengths: info.boneLengths };
      this._neuralPoseBuf = new Float32Array(info.joints.length * 3);
      this.setBodyAvatarPoseProvider((tMs) => (driver.active ? driver.getPose(this._neuralPoseBuf, tMs) : null), skeleton, { unscaled: true });
      if (this.mirror) this.mirror.setSkeleton(skeleton);
      console.log(`[app] neural avatar ready in ${n.loadMs.toFixed(0)} ms (${info.ortSource ? info.ortSource.split('/').pop() : 'ort'}, simd ${info.simd}, threads ${info.numThreads})`);
      this._neuralSync();
    }).catch((e) => {
      n.fallback = n.fallback || (e && e.message ? e.message : String(e));
      n.state = 'error';
      console.warn('[app] neural avatar unavailable: ' + n.fallback);
      this.hud.showNotice(t.errorModelLoad, 4);
    });
    return n;
  }

  _onNeuralStatus(s) {
    const n = this.neural;
    if (!n) return;
    if (s.progress) {
      // 35 + 51 MB over arcade WLAN take a while: keep the player informed (throttled to whole MB)
      const p = s.progress;
      const mb = (x) => Math.round(x / 1048576);
      const loaded = mb(p.loaded), total = p.total ? mb(p.total) : 0;
      if (loaded !== n.progressMb || p.label !== n.progressLabel) {
        n.progressMb = loaded;
        n.progressLabel = p.label;
        const text = this.local.neuralProgress.replace('{loaded}', String(loaded)).replace('{total}', total ? String(total) : '?');
        this.hud.showNotice(`${text}${p.label ? ` (${p.label})` : ''}`, 2.5);
      }
      return;
    }
    n.state = s.state;
    n.notice = s.notice || '';
    const t = this.texts;
    if (s.state === 'disabled') {
      n.fallback = s.notice || 'disabled';
      console.warn('[app] neural avatar fallback: ' + n.fallback + ` (mean ${s.inferenceMs.toFixed(1)} ms)`);
      // remember the verdict per headset (settings); 'Neural' in the settings clears it
      this.settings.neuralVerdict = { disabled: true, reason: 'too slow', inferenceMs: s.inferenceMs, at: new Date().toISOString(), version: APP_VERSION };
      this.saveSettings();
      this.hud.showNotice(this.local.neuralDisabledSaved, 6);
    } else if (s.state === 'error') {
      n.fallback = n.fallback || s.notice || 'error';
      this.hud.showNotice(t.errorModelLoad, 4);
    } else if (s.state === 'slow') {
      this.hud.showNotice(s.notice, 2);
    } else if (s.state === 'loading') {
      this.hud.showNotice(t.hudNeuralLoading, 2);
    }
    this.emit('neural', { state: s.state, notice: n.notice, inferenceMs: s.inferenceMs, inferEvery: s.inferEvery });
  }

  /** Feed the worker only while a dance (or its results) is on screen; idle in the menu. */
  _neuralSync() {
    const n = this.neural;
    if (!n || !n.driver) return;
    const d = n.driver;
    const want = NEURAL_STATES.has(this.state) && this.settings.avatar === 'neural';
    if (want && d.state === 'ready') d.start();
    else if (!want && d.active) d.stop();
  }

  // ---- server ---------------------------------------------------------------------------------

  /**
   * Decide where the REST API lives: `?server=` wins; otherwise the page's own origin when
   * `api/info` next to index.html answers (server/server.js serves both). GitHub Pages has no
   * server: results and runs then stay in localStorage only. Bounded by FLOW.networkTimeoutMs.
   */
  async _probeServer() {
    const explicit = httpBaseOf(this.params.server);
    if (explicit) {
      this.serverHttp = explicit;
      this.serverUrl = this.params.server;
      return explicit;
    }
    const loc = this.win && this.win.location;
    if (!loc || !/^https?:$/.test(loc.protocol) || typeof fetch !== 'function') return null;
    try {
      const url = new URL('api/info', loc.href);
      const res = await this._fetch(url.href, { cache: 'no-store' });
      if (!res.ok) return null;
      const info = await res.json();
      if (!info || typeof info !== 'object' || !('serverVersion' in info)) return null;
      this.serverInfo = info;
      this.serverHttp = new URL('.', loc.href).href.replace(/\/$/, '');
      this.serverUrl = this.serverHttp;
      return this.serverHttp;
    } catch (e) {
      return null;
    }
  }

  /** `fn(app)` downloads/returns telemetry; return false when there is nothing to export. */
  setTelemetryExporter(fn) {
    this.telemetryExporter = typeof fn === 'function' ? fn : null;
  }

  // ---- main loop -----------------------------------------------------------------------------

  _frame(time, xrFrame) {
    const dt = Number.isFinite(this._time) ? Math.min(0.1, Math.max(0, (time - this._time) / 1000)) : 0;
    this._time = time;
    this._dt = dt;
    this.frameCount++;
    const sk = this.sceneKit;
    const input = this.input;

    // 1. input
    if (input) {
      if (this.inputKind === 'xr') input.update(xrFrame, sk.referenceSpace, this.xrSession, time);
      else input.update(time);
      const ref = this.choreo ? this.choreo.referenceHeight : STAGE.referenceHeight;
      this.calibration.toStage(input.sample, ref, this.playerS);
      this.markers.setSample(input.sample, this.extra.cam === 'third');
      // neural avatar: 30 Hz resampling + S->DP happen in the driver (wall time keeps it monotonic)
      const nd = this.neural ? this.neural.driver : null;
      if (nd && nd.active && input.sample.head.valid) nd.update(this.playerS, time / 1000);
      // render-loop guard: only the headset's frame rate matters (the desktop/headless emu is slow anyway)
      if (nd && nd.active && sk.presenting && dt > 0) nd.reportFrameMs(dt * 1000);
      // 2. camera when not presenting
      if (!sk.presenting) {
        if (this.extra.cam === 'third') sk.setThirdPersonCamera();
        else sk.setCameraFromPose(input.sample.head);
      }
      // 3. menu pointer (XR laser / emu mouse) + trigger press
      if (this.menu.visible && this.inputKind === 'xr') {
        if (input.justPressed('left', 'trigger')) this.pointerHand = 'left';
        else if (input.justPressed('right', 'trigger')) this.pointerHand = 'right';
        const ctrl = sk.controllers[this.pointerHand];
        if (ctrl) this.menu.setPointerFromObject(ctrl.ray, this.pointerHand);
        else if (input.sample[this.pointerHand].valid) this.menu.setPointerFromPose(input.sample[this.pointerHand], this.pointerHand);
        else this.menu.clearPointer();
        // in CALIBRATE the trigger is the hold-to-calibrate button - unless the laser is on a
        // button of the panel ('Abbrechen')
        if ((this.state !== STATES.CALIBRATE || this.menu.hovering) && input.justPressed(this.pointerHand, 'trigger')) this.menu.press();
      }
      // B/Y hold leaves a run, a recording or the calibration (XR only)
      if (QUIT_STATES.has(this.state)) this._updateQuitHold(time);
    }

    // 4. state
    switch (this.state) {
      case STATES.CALIBRATE: {
        this._updateCalibrate(time);
        // redraw the progress bar at ~10 Hz, not every frame
        const c = this._calib;
        if (c && Math.abs(c.progress - c.drawnProgress) >= 0.05) {
          c.drawnProgress = c.progress;
          const panel = this.menu.getPanel('calibrate');
          if (panel) panel.invalidate();
        }
        break;
      }
      case STATES.COUNTDOWN:
        if (this.session) this._updatePlay(time, dt);
        else this._updateRecording(time);
        break;
      case STATES.PLAYING:
        this._updatePlay(time, dt);
        break;
      case STATES.RECORDING:
        this._updateRecording(time);
        break;
      case STATES.RECORD_REVIEW:
        this._updateReview(dt);
        break;
      default:
        break;
    }

    // 5. the player's own body (neural avatar provider) + the mirror image on the stage
    const neural = this.settings.avatar === 'neural';
    const showMirror = !!(this.mirror && input && this.settings.avatar !== 'off' && MIRROR_STATES.has(this.state));
    let pose = null;
    if (this.bodyPoseProvider && this.playerBody) {
      try { pose = this.bodyPoseProvider(time, this.playerS, this); } catch (e) { pose = null; }
      if (pose && this._bodyPoseUnscaled) {
        const k = this.playerS.k || 1;
        if (!this._bodyPoseScaled || this._bodyPoseScaled.length !== pose.length) this._bodyPoseScaled = new Float32Array(pose.length);
        const out = this._bodyPoseScaled;
        for (let i = 0; i < pose.length; i++) out[i] = pose[i] * k;
        pose = out;
      }
      if (pose) this.playerBody.setPose(pose);
      // the self body is for the desktop/third-person views; in the headset the mirror is enough
      this.playerBody.visible = !!pose && neural && showMirror && !sk.presenting;
    }
    if (showMirror) {
      const m = this.mirror;
      if (pose && neural && m.body) {
        m.setMode('body');
        m.setBodyPose(pose);
      } else {
        m.setMode('points');
        m.setPointSample(this.playerS);
      }
      m.visible = true;
    } else if (this.mirror) {
      this.mirror.visible = false;
    }

    // 6. UI + render
    this.menu.update();
    this.hud.update(dt);
    this.emit('frame', time, dt);
    sk.render();
  }

  // ---- panels --------------------------------------------------------------------------------

  _buildPanels() {
    const t = this.texts, l = this.local;
    const menu = this.menu;
    const app = this;

    menu.addPanel('main', {
      title: t.menuTitle,
      width: 1.7,
      position: [0, 1.5, 0],
      items: [
        {
          type: 'list',
          id: 'choreos',
          pageSize: 5,
          selected: null,
          emptyText: t.menuNoChoreos,
          options: () => app.choreos.map((e) => ({
            id: e.id,
            text: e.title,
            sub: `${Math.round(e.durationBeats * 60 / e.bpm)} ${t.secondsShort} · ${'★'.repeat(Math.max(1, Math.min(5, e.difficulty || 1)))}${e.source === 'own' ? ' · ' + l.ownDance : (e.source === 'server' ? ' · ' + l.fromServer : '')}`,
          })),
          onSelect: (id) => { app.selectedChoreoId = id; },
        },
        {
          type: 'row',
          items: [
            { type: 'button', id: 'play', text: t.menuPlay, primary: true, onClick: () => app.startSolo(app.selectedChoreoId).catch(() => null) },
            { type: 'button', id: 'duo', text: t.menuDuo, onClick: () => app.openDuoLobby() },
          ],
        },
        {
          type: 'row',
          items: [
            { type: 'button', id: 'record', text: t.menuRecord, onClick: () => app.startRecordSetup() },
            { type: 'button', id: 'settings', text: t.menuSettings, onClick: () => app.openSettings() },
            { type: 'button', id: 'highscores', text: t.menuHighscores, onClick: () => app.openHighscores() },
          ],
        },
      ],
      footer: () => `${APP_NAME} · ${t.menuVersion} ${APP_VERSION} · ${app.inputKind === 'xr' ? 'VR' : 'Desktop'} · ${app.calibration.valid ? `${t.calibrateHeight} ${app.calibration.h0.toFixed(2)} m` : t.calibrateTitle + ' –'}`,
    });

    const avatarLabel = () => ({ neural: t.settingsAvatarNeural, points: t.settingsAvatarPoints, off: t.settingsAvatarOff }[app.settings.avatar]);
    menu.addPanel('settings', {
      title: t.settingsTitle,
      width: 1.6,
      position: [0, 1.5, 0],
      items: [
        { type: 'button', id: 'avatar', text: () => `${t.settingsAvatar}: ${avatarLabel()}`, onClick: () => app.setAvatarMode(AVATAR_MODES[(AVATAR_MODES.indexOf(app.settings.avatar) + 1) % AVATAR_MODES.length]) },
        { type: 'stepper', id: 'volume', text: t.settingsAudio, value: () => `${Math.round(app.settings.volume * 100)} %`, onDec: () => app.setVolume(app.settings.volume - 0.1), onInc: () => app.setVolume(app.settings.volume + 0.1) },
        { type: 'button', id: 'lang', text: () => `${t.settingsLanguage}: ${app.settings.lang === 'en' ? 'English' : 'Deutsch'}`, onClick: () => app._switchLanguage() },
        { type: 'toggle', id: 'telemetry', text: t.settingsTelemetry, value: () => !!app.settings.telemetry, onChange: (v) => { app.settings.telemetry = v; app.saveSettings(); } },
        { type: 'button', id: 'telemetry-export', text: t.settingsTelemetryExport, onClick: () => app.exportTelemetry() },
        { type: 'button', id: 'recalibrate', text: t.menuRecalibrate, onClick: () => app._enterCalibrate({ then: 'menu', recalibrate: true }) },
        { type: 'button', id: 'back', text: t.menuBack, onClick: () => app.backToMenu() },
      ],
      footer: () => `${t.settingsServer}: ${app.serverHttp || t.offline}`,
      onBack: () => app.backToMenu(),
    });

    menu.addPanel('duo', {
      title: t.duoTitle,
      width: 1.6,
      position: [0, 1.5, 0],
      items: [
        { type: 'label', text: () => (app.selectedEntry ? app.selectedEntry.title : t.menuNoChoreos), size: 'small' },
        { type: 'button', id: 'ghost', text: () => `${t.duoGhost}${app.modes.has('ghost') ? '' : ' – ' + l.notAvailable}`, primary: true, onClick: () => (app.modes.has('ghost') ? app.startMode('ghost').catch(() => null) : app.hud.showNotice(l.notAvailable, 2)) },
        { type: 'button', id: 'online', text: () => `${t.duoOnline}${app.modes.has('online') ? '' : ' – ' + l.notAvailable}`, onClick: () => app._onlineButton() },
        { type: 'button', id: 'back', text: t.menuBack, onClick: () => app.backToMenu() },
      ],
      footer: () => (app.serverHttp ? `${t.settingsServer}: ${app.serverHttp}` : t.duoNoServer),
      onBack: () => app.backToMenu(),
    });

    menu.addPanel('calibrate', {
      title: t.calibrateTitle,
      width: 1.6,
      position: [0, 1.5, 0],
      items: [
        { type: 'text', text: () => (app.input && app.input.emulated ? t.calibrateInstructionEmu : t.calibrateInstruction) },
        { type: 'progress', value: () => (app._calib ? app._calib.progress : 0), color: '#ffd166' },
        { type: 'label', text: () => (app._calib && app._calib.holding ? t.calibrateHold : ''), size: 'small', align: 'center' },
        { type: 'button', id: 'cancel', text: t.cancel, onClick: () => app.abortToMenu() },
      ],
      onBack: () => app.abortToMenu(),
    });

    const cfg = this._recordCfg;
    menu.addPanel('record-setup', {
      title: t.recordTitle,
      width: 1.7,
      position: [0, 1.55, 0],
      items: [
        { type: 'button', id: 'title', text: () => `${t.recordName}: ${cfg.title}`, onClick: () => app._editRecordTitle() },
        { type: 'stepper', id: 'bpm', text: t.recordBpm, value: () => cfg.bpm, onDec: () => { cfg.bpm = Math.max(40, cfg.bpm - 1); }, onInc: () => { cfg.bpm = Math.min(240, cfg.bpm + 1); }, onDec2: () => { cfg.bpm = Math.max(40, cfg.bpm - 10); }, onInc2: () => { cfg.bpm = Math.min(240, cfg.bpm + 10); } },
        { type: 'button', id: 'tap', text: () => `${t.recordTapTempo} (${cfg.bpm} BPM)`, onClick: () => app._tapTempo() },
        { type: 'stepper', id: 'bars', text: () => `${t.recordBars} (${Math.round(cfg.bars * 4 * 60 / cfg.bpm)} ${t.secondsShort})`, value: () => cfg.bars, onDec: () => { cfg.bars = Math.max(1, cfg.bars - 1); }, onInc: () => { cfg.bars = Math.min(32, cfg.bars + 1); } },
        { type: 'stepper', id: 'countin', text: t.recordCountIn, value: () => cfg.countInBeats, onDec: () => { cfg.countInBeats = Math.max(0, cfg.countInBeats - 1); }, onInc: () => { cfg.countInBeats = Math.min(8, cfg.countInBeats + 1); } },
        {
          type: 'row',
          items: [
            { type: 'button', id: 'synth', text: () => `${t.recordSynth}: ${cfg.synth}`, onClick: () => { cfg.synth = SYNTHS[(SYNTHS.indexOf(cfg.synth) + 1) % SYNTHS.length]; } },
            { type: 'toggle', id: 'mirror', text: t.recordMirror, value: () => cfg.mirror, onChange: (v) => { cfg.mirror = v; } },
          ],
        },
        {
          type: 'row',
          items: [
            { type: 'button', id: 'start', text: t.recordStart, primary: true, onClick: () => app.beginRecording() },
            { type: 'button', id: 'back', text: t.menuBack, onClick: () => app.backToMenu() },
          ],
        },
      ],
      onBack: () => app.backToMenu(),
    });

    menu.addPanel('record-review', {
      title: t.recordReviewTitle,
      width: 1.6,
      position: [0, 1.5, 0],
      items: [
        { type: 'label', text: () => (app._review ? app._review.json.title : ''), size: 'large' },
        { type: 'label', text: () => (app._review ? l.reviewInfo.replace('{frames}', app._review.json.frames.head.length).replace('{seconds}', Math.round(app._review.json.durationBeats * 60 / app._review.json.bpm)).replace('{bpm}', app._review.json.bpm) : ''), size: 'small' },
        {
          type: 'row',
          items: [
            { type: 'button', id: 'play', text: () => (app._review && app._review.playing ? t.hudPaused : t.recordPlayback), onClick: () => { if (app._review) app._review.playing = !app._review.playing; } },
            { type: 'button', id: 'save', text: t.recordSave, primary: true, onClick: () => app.saveRecording() },
            { type: 'button', id: 'discard', text: t.recordDiscard, onClick: () => app.discardRecording() },
          ],
        },
      ],
      onBack: () => app.discardRecording(),
    });

    menu.addPanel('highscores', {
      title: t.highscoresTitle,
      width: 1.7,
      position: [0, 1.55, 0],
      items: [
        {
          type: 'row',
          items: [
            { type: 'button', id: 'prev', text: '‹', onClick: () => { const n = app.choreos.length; if (n) app._highscoreIndex = (app._highscoreIndex - 1 + n) % n; } },
            { type: 'button', id: 'name', text: () => (app.choreos[app._highscoreIndex] ? app.choreos[app._highscoreIndex].title : t.menuNoChoreos), onClick: () => {} },
            { type: 'button', id: 'next', text: '›', onClick: () => { const n = app.choreos.length; if (n) app._highscoreIndex = (app._highscoreIndex + 1) % n; } },
          ],
        },
        {
          type: 'list',
          id: 'scores',
          pageSize: 10,
          emptyText: t.highscoresEmpty,
          options: () => {
            const e = app.choreos[app._highscoreIndex];
            if (!e) return [];
            return app.highscores(e.id, 10).map((r, i) => ({
              id: `${i}`,
              text: `${i + 1}. ${r.score} ${t.resultsScore} · ${starsString(r.stars || 1)} · ${r.player || t.player}`,
              sub: `${formatDate(r.playedAt)}${r.fromServer ? ' · ' + t.highscoresServer : ''}${r.mode && r.mode !== 'solo' ? ' · ' + r.mode : ''}`,
            }));
          },
          onSelect: () => {},
        },
        { type: 'button', id: 'back', text: t.menuBack, onClick: () => app.backToMenu() },
      ],
      onBack: () => app.backToMenu(),
    });
  }

  /** 'Duo (Online)' in the duo panel: connect to the server (lobby panel) or explain what is missing. */
  _onlineButton() {
    const t = this.texts, l = this.local;
    const mode = this.modes.get('online');
    if (!mode) { this.hud.showNotice(l.notAvailable, 2); return; }
    if (!this.serverHttp) { this.hud.showNotice(t.duoNoServer, 3); return; }
    if (mode.lobbyPanelId && this.menu.getPanel(mode.lobbyPanelId)) {
      this.menu.showAlso(mode.lobbyPanelId);
      if (typeof mode.ensureConnected === 'function') mode.ensureConnected().catch(() => null);
    } else {
      this.startMode('online').catch(() => null);
    }
  }

  _editRecordTitle() {
    const cfg = this._recordCfg;
    if (this.inputKind !== 'xr' && this.win && typeof this.win.prompt === 'function') {
      const v = this.win.prompt(this.texts.recordName, cfg.title);
      if (v && v.trim()) cfg.title = v.trim().slice(0, 40);
      return;
    }
    // no keyboard in VR: cycle through a few preset names
    const presets = [this.local.ownDance, 'TikTok', 'Freestyle', 'Hip-Hop', 'Ballroom'];
    const n = Recorder.listPersisted(this.storage).length + 1;
    const cur = presets.findIndex((p) => cfg.title.startsWith(p));
    cfg.title = `${presets[(cur + 1) % presets.length]} ${n}`;
  }

  _switchLanguage() {
    this.settings.lang = this.settings.lang === 'en' ? 'de' : 'en';
    this.saveSettings();
    if (!this.win || !this.win.location) return;
    try {
      const url = new URL(this.win.location.href);
      url.searchParams.set('lang', this.settings.lang);
      this.win.location.href = url.toString();
    } catch (e) { /* ignore */ }
  }

  dispose() {
    this.sceneKit.setAnimationLoop(null);
    if (this.session) this.session.abort();
    this._disposeSession();
    if (this.neural && this.neural.driver) this.neural.driver.dispose();
    for (const m of this.modes.values()) if (typeof m.dispose === 'function') { try { m.dispose(); } catch (e) { /* ignore */ } }
    if (this.mirror) this.mirror.dispose();
    if (this.input && this.input.dispose) this.input.dispose();
    this.menu.dispose();
    this.hud.dispose();
    this.teacher.dispose();
    this.markers.dispose();
    this.sceneKit.dispose();
  }
}

function gradeColor(grade) {
  return { perfekt: '#4cff9a', gut: '#4cc9f0', ok: '#ffd166', daneben: '#ff6b6b' }[grade] || '#f4f6ff';
}

/** The part of a duo benchmark stored with a result (localStorage + /api/results). */
function compactBenchmark(b) {
  const p = b.players && b.players[1];
  return { mode: b.mode, winner: b.winner, pair: b.pair, partner: p ? { name: p.name, score: p.score, stars: p.stars, maxCombo: p.maxCombo } : null };
}

/** localStorage may throw (privacy mode); fall back to an in-memory map. */
function safeStorage(win) {
  try {
    const ls = win && win.localStorage;
    if (ls) { ls.getItem('dp.probe'); return ls; }
  } catch (e) { /* fall through */ }
  const mem = new Map();
  return {
    getItem: (k) => (mem.has(k) ? mem.get(k) : null),
    setItem: (k, v) => { mem.set(k, String(v)); },
    removeItem: (k) => { mem.delete(k); },
  };
}

export { validateChoreo };

// ---- auto boot in the browser ------------------------------------------------------------------
if (typeof window !== 'undefined' && typeof document !== 'undefined' && !window.__dpNoAutoBoot) {
  const app = new App();
  window.__dp = { app, version: APP_VERSION };
  app.boot().catch((e) => {
    console.error('[app] boot failed', e);
    const s = document.getElementById('status');
    if (s) s.textContent = `${app.texts.error}: ${e && e.message ? e.message : e}`;
  });
}
