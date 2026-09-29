// config.js - constants, URL parameters and feature flags for Dancing Points VR.
// Pure data + a tiny parser; importable in Node (no window/location access at import time
// except through the guarded `params` export). No three.js imports.

export const APP_VERSION = '0.1.1';   // bump together with SW_VERSION in webxr/sw.js (tests/unit/sw.test.js)
export const APP_NAME = 'Dancing Points VR';

// Scoring constants (DESIGN.md section 7).
export const SCORING = Object.freeze({
  tickHz: 30,               // scoring rate (ticks per second)
  sigmaHead: 0.10,          // m
  sigmaHand: 0.18,          // m, hand position relative to the head
  sigmaVel: 0.80,           // m/s
  deltaMaxSec: 0.20,        // reaction/lag tolerance window (+/-), searched in 1/tickHz steps
  weightHead: 0.25,
  weightLeft: 0.25,
  weightRight: 0.25,
  weightVel: 0.25,
  energyGateRatio: 0.30,    // player RMS speed < 30 % of reference -> gate
  energyGateCap: 0.40,      // capped move score when gated
  gradePerfekt: 0.85,
  gradeGut: 0.65,
  gradeOk: 0.40,
  comboThreshold: 0.65,     // beat score needed to keep the combo alive
  stars: Object.freeze([90, 75, 60, 40]), // >= 90 -> 5 stars, >= 75 -> 4, >= 60 -> 3, >= 40 -> 2, else 1
});

// Neural pipeline guard thresholds (DESIGN.md section 6.3).
export const NET = Object.freeze({
  tickHz: 30,
  historyFrames: 15,
  futureFrames: 30,
  rootCorrection: 0.35,
  eulerRatio: 0.5,
  boneLengthTolerance: 1.05,
  guardWindowTicks: 60,     // average inferenceMs over this many ticks
  slowMs: 25,               // > slowMs  -> tick at 15 Hz
  disableMs: 60,            // > disableMs -> disable the neural avatar
  telemetryTicks: 2000,
});

// Stage geometry (metres, stage frame S).
export const STAGE = Object.freeze({
  referenceHeight: 1.70,    // default reference height when a choreography does not define one
  teacherDistance: 2.5,     // teacher avatar this far in front of the player (-Z in S)
  duoSideOffset: 1.5,       // second dancer this far to the side of the teacher (+X in S)
  floorSize: 6.0,
});

// Timing / flow constants.
export const FLOW = Object.freeze({
  defaultCountInBeats: 4,
  calibrationHoldSec: 1.0,  // hold the trigger this long to calibrate (XR)
  calibrationEmuSec: 3.0,   // in emu mode calibration completes after this delay
  calibrationReuseWaitSec: 2.0, // 'Bereit?' pause before the count-in when a persisted calibration is reused
  quitHoldSec: 2.5,         // hold B/Y this long to abort a run / recording / calibration (XR)
  networkTimeoutMs: 3000,   // boot-time probes (api/info, choreo list) and result uploads
  audioLookaheadSec: 0.1,   // WebAudio scheduling lookahead
  runRecordHz: 30,          // player run recording rate for ghost duo
  duoStateHz: 15,           // online duo state send rate
});

// localStorage keys.
export const STORAGE_KEYS = Object.freeze({
  calibration: 'dp.calibration',
  results: 'dp.results',
  runs: 'dp.runs',
  choreos: 'dp.choreos',
  settings: 'dp.settings',
  telemetry: 'dp.telemetry',
});

export const DEFAULT_PARAMS = Object.freeze({
  emu: null,          // null | 'desktop' | 'playback'
  noise: 0,           // metres, playback emu only
  lag: 0,             // seconds, playback emu only
  seed: 1,            // playback emu PRNG seed
  choreo: null,       // choreography id
  autostart: false,
  speed: 1,           // clock speed factor (tests)
  server: null,       // wss://host:port (or https://host:port)
  avatar: 'neural',   // 'neural' | 'points' | 'off'
  style: 'free',      // model set
  telemetry: false,
  lang: 'de',
  token: null,        // API token of server/server.js --token (sent as X-Dp-Token on POST/DELETE)
});

function toNumber(value, fallback) {
  if (value === null || value === undefined || value === '') return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function toBool(value, fallback) {
  if (value === null || value === undefined) return fallback;
  const v = String(value).toLowerCase();
  if (v === '' || v === '1' || v === 'true' || v === 'yes' || v === 'on') return true;
  if (v === '0' || v === 'false' || v === 'no' || v === 'off') return false;
  return fallback;
}

/**
 * Parse URL parameters (a search string like "?emu=playback&noise=0.1") into a params object.
 * Unknown keys are ignored; invalid values fall back to DEFAULT_PARAMS.
 */
export function parseParams(search = '') {
  const out = { ...DEFAULT_PARAMS };
  let sp;
  try {
    sp = new URLSearchParams(search || '');
  } catch (e) {
    return out;
  }
  const get = (k) => (sp.has(k) ? sp.get(k) : null);

  const emu = get('emu');
  if (emu !== null) {
    const v = emu.toLowerCase();
    if (v === 'playback' || v === 'scripted') out.emu = 'playback';
    else if (v === 'desktop' || toBool(v, false)) out.emu = 'desktop';
    else out.emu = null;
  }
  out.noise = Math.max(0, toNumber(get('noise'), out.noise));
  out.lag = toNumber(get('lag'), out.lag);
  out.seed = Math.floor(toNumber(get('seed'), out.seed));
  const choreo = get('choreo');
  if (choreo && /^[a-z0-9-]+$/.test(choreo)) out.choreo = choreo;
  out.autostart = toBool(get('autostart'), out.autostart);
  const speed = toNumber(get('speed'), out.speed);
  out.speed = speed > 0 ? speed : out.speed;
  const server = get('server');
  if (server && /^(wss?|https?):\/\//.test(server)) out.server = server;
  const avatar = get('avatar');
  if (avatar && ['neural', 'points', 'off'].includes(avatar)) out.avatar = avatar;
  const style = get('style');
  if (style && /^[a-z0-9_-]+$/.test(style)) out.style = style;
  out.telemetry = toBool(get('telemetry'), out.telemetry);
  const lang = get('lang');
  if (lang && (lang === 'de' || lang === 'en')) out.lang = lang;
  const token = get('token');
  if (token && /^[A-Za-z0-9_.-]{1,128}$/.test(token)) out.token = token;
  return out;
}

/** Params of the current page (defaults when there is no `location`, e.g. in Node). */
export const params = parseParams(
  typeof globalThis.location !== 'undefined' && globalThis.location ? globalThis.location.search : ''
);
