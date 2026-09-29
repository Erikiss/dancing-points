// emulated-input.js - EmulatedInput: input backends that replace XRInput when there is no
// headset. Same sample shape and button interface as XRInput (see xr/input.js).
//   'playback': feeds the choreography's own reference trajectory, converted from the stage
//               frame back to W through a virtual player (height / yaw / origin), with optional
//               temporally smooth gaussian noise (metres) and a lag (seconds). Deterministic
//               (seeded PRNG). Runs headless in Node.
//   'desktop':  WASD / mouse-look head, hands follow the head; Q/E raise the left/right hand,
//               Space = trigger. Attaches to a window/document when given.
// No three.js. Before the clock runs, both backends output the standing calibration pose.

import { STAGE } from '../config.js';
import { Calibration } from '../xr/calibration.js';
import { createSample, createButtonState } from '../xr/input.js';
import { createChoreoSample } from '../game/choreo.js';
import { quatFromYaw, quatMul, quatFromAxisAngle, clamp } from '../util/math.js';

/** mulberry32 PRNG: deterministic, tiny, good enough for noise. */
export function makeRandom(seed = 1) {
  let a = (seed >>> 0) || 1;
  const next = () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  // Box-Muller gaussian
  let spare = NaN;
  const gaussian = () => {
    if (!Number.isNaN(spare)) { const s = spare; spare = NaN; return s; }
    let u = 0, v = 0;
    while (u === 0) u = next();
    v = next();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
  return { next, gaussian };
}

const PRESS = 0.5;

export class EmulatedInput {
  /**
   * @param {object} opts
   * @param {'desktop'|'playback'} [opts.backend='desktop']
   * @param {import('../game/choreo.js').Choreo} [opts.choreo]  required for 'playback'
   * @param {{now: function}} [opts.clock]  BeatClock providing the game time (playback)
   * @param {number} [opts.noise=0]  stationary sigma of the smooth noise, metres (per axis)
   * @param {number} [opts.noiseTau=0.25]  noise correlation time, seconds
   * @param {number} [opts.lag=0]  seconds the virtual player is behind the reference
   * @param {number} [opts.seed=1]
   * @param {number} [opts.playerHeight]  virtual player's head height (default referenceHeight)
   * @param {number} [opts.playerYaw=0]   virtual player's facing yaw in W
   * @param {number[]} [opts.playerOrigin=[0,0,0]]  virtual player's floor position in W
   * @param {Calibration} [opts.calibration]  alternative: copy the virtual player from this
   * @param {Window} [opts.window] / @param {Document} [opts.document]  for 'desktop'
   */
  constructor(opts = {}) {
    this.backend = opts.backend === 'playback' ? 'playback' : 'desktop';
    this.emulated = true;
    this.sample = createSample();
    this.buttons = { left: createButtonState(), right: createButtonState() };
    this._prev = { left: createButtonState(), right: createButtonState() };
    this.choreo = opts.choreo || null;
    this.clock = opts.clock || null;
    this.referenceHeight = this.choreo ? this.choreo.referenceHeight : STAGE.referenceHeight;
    this.noise = opts.noise > 0 ? opts.noise : 0;
    this.noiseTau = opts.noiseTau > 0 ? opts.noiseTau : 0.25;
    this.lag = typeof opts.lag === 'number' ? opts.lag : 0;
    this.seed = opts.seed !== undefined ? opts.seed : 1;
    this.random = makeRandom(this.seed);
    this.time = NaN;             // explicit game time when there is no clock (setTime)
    this.frameCount = 0;

    // virtual player transform (S -> W)
    this.virtual = new Calibration({ referenceHeight: this.referenceHeight });
    if (opts.calibration && opts.calibration.valid) {
      this.virtual.set(opts.calibration.data);
    } else {
      this.virtual.set({
        h0: opts.playerHeight > 0 ? opts.playerHeight : this.referenceHeight,
        yaw: typeof opts.playerYaw === 'number' ? opts.playerYaw : 0,
        origin: Array.isArray(opts.playerOrigin) ? opts.playerOrigin : [0, 0, 0],
      });
    }

    this._refS = createChoreoSample();
    this._stageS = createSample();
    this._noise = new Float64Array(9);
    this._noiseT = NaN;
    this._standing = true;

    // desktop state
    this.keys = new Set();
    this.mouseButtons = 0;
    this.pos = [0, this.virtual.h0, 0];
    this.yaw = this.virtual.yaw;
    this.pitch = 0;
    this.handRaise = [0, 0];
    this.moveSpeed = 1.2;        // m/s
    this.lookSpeed = 0.0025;     // rad per pixel
    this._lastNow = NaN;
    this._win = null;
    this._doc = null;
    this._onKeyDown = (e) => { this.keys.add(e.code); if (e.code === 'Space' || e.code.startsWith('Arrow')) { if (e.preventDefault) e.preventDefault(); } };
    this._onKeyUp = (e) => { this.keys.delete(e.code); };
    this._onMouseMove = (e) => {
      const locked = this._doc && this._doc.pointerLockElement;
      if (locked || (e.buttons & 1)) {
        this.yaw -= (e.movementX || 0) * this.lookSpeed;
        this.pitch = clamp(this.pitch - (e.movementY || 0) * this.lookSpeed, -1.2, 1.2);
      }
    };
    this._onMouseDown = (e) => { this.mouseButtons |= (1 << e.button); };
    this._onMouseUp = (e) => { this.mouseButtons &= ~(1 << e.button); };
    this._onBlur = () => { this.keys.clear(); this.mouseButtons = 0; };
    if (this.backend === 'desktop' && (opts.window || opts.document)) this.attach(opts.window, opts.document);
    if (this.backend === 'playback' && !this.choreo) throw new Error('playback backend needs a choreo');
    this._writeStanding();
  }

  /** Attach DOM listeners (desktop backend). */
  attach(win, doc) {
    this.detach();
    this._win = win || null;
    this._doc = doc || (win ? win.document : null) || null;
    const target = this._win || this._doc;
    if (!target || typeof target.addEventListener !== 'function') return;
    target.addEventListener('keydown', this._onKeyDown);
    target.addEventListener('keyup', this._onKeyUp);
    target.addEventListener('mousemove', this._onMouseMove);
    target.addEventListener('mousedown', this._onMouseDown);
    target.addEventListener('mouseup', this._onMouseUp);
    target.addEventListener('blur', this._onBlur);
  }

  detach() {
    const target = this._win || this._doc;
    if (target && typeof target.removeEventListener === 'function') {
      target.removeEventListener('keydown', this._onKeyDown);
      target.removeEventListener('keyup', this._onKeyUp);
      target.removeEventListener('mousemove', this._onMouseMove);
      target.removeEventListener('mousedown', this._onMouseDown);
      target.removeEventListener('mouseup', this._onMouseUp);
      target.removeEventListener('blur', this._onBlur);
    }
    this._win = null;
    this._doc = null;
  }

  /** Explicit game time for the playback backend when no clock is attached. */
  setTime(t) {
    this.time = t;
  }

  /** Game time (seconds since beat 0) or NaN when not playing. */
  gameTime() {
    if (this.clock) {
      const s = this.clock.now();
      return s.running ? s.t : NaN;
    }
    return this.time;
  }

  /**
   * Produce the next sample. `nowMs` is wall time in ms (desktop integration; defaults to
   * performance.now()). Returns the reused sample.
   */
  update(nowMs) {
    if (typeof nowMs !== 'number') {
      nowMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
    }
    this.frameCount++;
    copyBtn(this._prev.left, this.buttons.left);
    copyBtn(this._prev.right, this.buttons.right);
    this.sample.t = nowMs / 1000;
    if (this.backend === 'playback') this._updatePlayback();
    else this._updateDesktop(nowMs);
    return this.sample;
  }

  // ---- playback ----------------------------------------------------------------------------

  _updatePlayback() {
    const t = this.gameTime();
    if (!Number.isFinite(t)) {
      this._writeStanding();
      return;
    }
    this._standing = false;
    const ref = this.choreo.sampleAt(t - this.lag, this._refS);
    const s = this._stageS;
    s.t = t;
    s.head.p[0] = ref.head.p[0]; s.head.p[1] = ref.head.p[1]; s.head.p[2] = ref.head.p[2];
    s.head.q[0] = ref.head.q[0]; s.head.q[1] = ref.head.q[1]; s.head.q[2] = ref.head.q[2]; s.head.q[3] = ref.head.q[3];
    s.left.p[0] = ref.left.p[0]; s.left.p[1] = ref.left.p[1]; s.left.p[2] = ref.left.p[2];
    s.right.p[0] = ref.right.p[0]; s.right.p[1] = ref.right.p[1]; s.right.p[2] = ref.right.p[2];
    s.left.q[0] = 0; s.left.q[1] = 0; s.left.q[2] = 0; s.left.q[3] = 1;
    s.right.q[0] = 0; s.right.q[1] = 0; s.right.q[2] = 0; s.right.q[3] = 1;
    s.head.valid = true; s.left.valid = true; s.right.valid = true;
    if (this.noise > 0) this._applyNoise(s, t);
    // S -> W through the virtual player
    this.virtual.fromStage(s, this.referenceHeight, this.sample);
    this.sample.t = t;
  }

  /** Ornstein-Uhlenbeck noise per coordinate: stationary sigma = noise, correlation time tau. */
  _applyNoise(s, t) {
    const nz = this._noise;
    let dt = t - this._noiseT;
    if (!Number.isFinite(dt) || dt < 0) {
      // (re)initialise from the stationary distribution
      for (let i = 0; i < 9; i++) nz[i] = this.noise * this.random.gaussian();
    } else if (dt > 0) {
      const a = Math.exp(-dt / this.noiseTau);
      const b = this.noise * Math.sqrt(1 - a * a);
      for (let i = 0; i < 9; i++) nz[i] = nz[i] * a + b * this.random.gaussian();
    }
    this._noiseT = t;
    s.head.p[0] += nz[0]; s.head.p[1] += nz[1]; s.head.p[2] += nz[2];
    s.left.p[0] += nz[3]; s.left.p[1] += nz[4]; s.left.p[2] += nz[5];
    s.right.p[0] += nz[6]; s.right.p[1] += nz[7]; s.right.p[2] += nz[8];
  }

  /** Standing calibration pose: head at (0, referenceHeight, 0) facing -Z in S, hands at the sides. */
  _writeStanding() {
    const s = this._stageS;
    s.head.p[0] = 0; s.head.p[1] = this.referenceHeight; s.head.p[2] = 0;
    s.head.q[0] = 0; s.head.q[1] = 0; s.head.q[2] = 0; s.head.q[3] = 1;
    s.left.p[0] = -0.2; s.left.p[1] = this.referenceHeight * 0.55; s.left.p[2] = -0.05;
    s.right.p[0] = 0.2; s.right.p[1] = this.referenceHeight * 0.55; s.right.p[2] = -0.05;
    s.left.q[0] = 0; s.left.q[1] = 0; s.left.q[2] = 0; s.left.q[3] = 1;
    s.right.q[0] = 0; s.right.q[1] = 0; s.right.q[2] = 0; s.right.q[3] = 1;
    s.head.valid = true; s.left.valid = true; s.right.valid = true;
    this._noiseT = NaN;
    this._standing = true;
    const t = this.sample.t;
    this.virtual.fromStage(s, this.referenceHeight, this.sample);
    this.sample.t = t;
  }

  // ---- desktop -----------------------------------------------------------------------------

  _updateDesktop(nowMs) {
    let dt = (nowMs - this._lastNow) / 1000;
    this._lastNow = nowMs;
    if (!Number.isFinite(dt) || dt < 0) dt = 0;
    if (dt > 0.1) dt = 0.1;
    const k = this.keys;
    // movement in the yaw frame: forward = -Z
    let fwd = 0, side = 0, up = 0;
    if (k.has('KeyW') || k.has('ArrowUp')) fwd += 1;
    if (k.has('KeyS') || k.has('ArrowDown')) fwd -= 1;
    if (k.has('KeyD') || k.has('ArrowRight')) side += 1;
    if (k.has('KeyA') || k.has('ArrowLeft')) side -= 1;
    if (k.has('KeyR')) up += 1;
    if (k.has('KeyF')) up -= 1;
    const c = Math.cos(this.yaw), sn = Math.sin(this.yaw);
    // forward vector (-sin yaw, 0, -cos yaw), right vector (cos yaw, 0, -sin yaw)
    const v = this.moveSpeed * dt;
    this.pos[0] += (-sn * fwd + c * side) * v;
    this.pos[2] += (-c * fwd - sn * side) * v;
    this.pos[1] = clamp(this.pos[1] + up * v, 0.5, 2.4);
    if (k.has('KeyQ')) this.handRaise[0] = Math.min(1, this.handRaise[0] + dt * 4);
    else this.handRaise[0] = Math.max(0, this.handRaise[0] - dt * 4);
    if (k.has('KeyE')) this.handRaise[1] = Math.min(1, this.handRaise[1] + dt * 4);
    else this.handRaise[1] = Math.max(0, this.handRaise[1] - dt * 4);

    const s = this.sample;
    s.head.p[0] = this.pos[0]; s.head.p[1] = this.pos[1]; s.head.p[2] = this.pos[2];
    // orientation: yaw about Y then pitch about local X
    quatFromYaw(QA, this.yaw);
    quatFromAxisAngle(QB, X_AXIS, this.pitch);
    quatMul(s.head.q, QA, QB);
    s.head.valid = true;
    // hands follow the head (yaw frame offsets), raised by Q/E
    for (let h = 0; h < 2; h++) {
      const pt = h === 0 ? s.left : s.right;
      const ox = h === 0 ? -0.22 : 0.22;
      const oy = -0.45 + 0.75 * this.handRaise[h];
      const oz = -0.35;
      pt.p[0] = this.pos[0] + ox * c + oz * sn;
      pt.p[1] = this.pos[1] + oy;
      pt.p[2] = this.pos[2] - ox * sn + oz * c;
      pt.q[0] = QA[0]; pt.q[1] = QA[1]; pt.q[2] = QA[2]; pt.q[3] = QA[3];
      pt.valid = true;
    }
    // buttons
    const trig = k.has('Space') || (this.mouseButtons & 1) !== 0 ? 1 : 0;
    const bl = this.buttons.left, br = this.buttons.right;
    bl.trigger = trig; br.trigger = trig;
    bl.squeeze = k.has('ShiftLeft') ? 1 : 0; br.squeeze = k.has('ShiftRight') ? 1 : 0;
    bl.primary = k.has('KeyX'); br.primary = k.has('Enter');
    bl.secondary = k.has('KeyY'); br.secondary = k.has('KeyB');
    bl.menu = k.has('Escape'); br.menu = k.has('Escape');
    bl.connected = true; br.connected = true;
  }

  // ---- buttons (same interface as XRInput) ---------------------------------------------------

  /** Programmatically press/release a button (tests, e2e autoplay). */
  setButton(hand, name, value) {
    const hands = hand === 'any' ? ['left', 'right'] : [hand];
    for (const h of hands) {
      const b = this.buttons[h];
      if (name === 'trigger' || name === 'squeeze') b[name] = value ? 1 : 0;
      else b[name] = !!value;
    }
  }

  justPressed(hand, name = 'trigger') {
    if (hand === 'any') return this.justPressed('left', name) || this.justPressed('right', name);
    return isPressed(this.buttons[hand], name) && !isPressed(this._prev[hand], name);
  }

  isDown(hand, name = 'trigger') {
    if (hand === 'any') return this.isDown('left', name) || this.isDown('right', name);
    return isPressed(this.buttons[hand], name);
  }

  anyTriggerDown() {
    return this.isDown('any', 'trigger');
  }

  get tracking() {
    return true;
  }

  get standing() {
    return this._standing;
  }

  dispose() {
    this.detach();
  }
}

const QA = [0, 0, 0, 1];
const QB = [0, 0, 0, 1];
const X_AXIS = [1, 0, 0];

function isPressed(b, name) {
  if (!b) return false;
  if (name === 'trigger' || name === 'squeeze') return b[name] >= PRESS;
  return b[name] === true;
}

function copyBtn(dst, src) {
  dst.trigger = src.trigger; dst.squeeze = src.squeeze; dst.stick = src.stick;
  dst.primary = src.primary; dst.secondary = src.secondary; dst.menu = src.menu;
  dst.thumb[0] = src.thumb[0]; dst.thumb[1] = src.thumb[1];
  dst.connected = src.connected; dst.isHand = src.isHand;
}
