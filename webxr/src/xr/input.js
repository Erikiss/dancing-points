// input.js - XRInput: reads head + hand poses and controller buttons from the raw WebXR API
// (XRFrame / XRReferenceSpace / XRSession), no three.js. Also defines the input backend
// interface and the sample object shape shared with the emulated backends:
//
//   sample = { t, head: {p:[x,y,z], q:[x,y,z,w], valid}, left: {...}, right: {...} }   (frame W)
//
// Backend interface: `sample` (latest, reused object), `update(...)`, `buttons`,
// `justPressed(hand, name)`, `emulated` flag, `dispose()`. Poses are kept from the last valid
// frame when tracking is lost (`valid` becomes false). Importable in Node (no DOM at import).

export const BUTTONS = Object.freeze(['trigger', 'squeeze', 'stick', 'primary', 'secondary', 'menu']);

/** Allocate a sample object (frame W or S). */
export function createSample() {
  return {
    t: 0,
    k: 1,
    head: { p: [0, 0, 0], q: [0, 0, 0, 1], valid: false },
    left: { p: [0, 0, 0], q: [0, 0, 0, 1], valid: false },
    right: { p: [0, 0, 0], q: [0, 0, 0, 1], valid: false },
  };
}

/** Deep-copy a sample into another (no allocation). */
export function copySample(out, s) {
  out.t = s.t;
  out.k = s.k !== undefined ? s.k : 1;
  copyPoint(out.head, s.head);
  copyPoint(out.left, s.left);
  copyPoint(out.right, s.right);
  return out;
}

function copyPoint(o, p) {
  o.p[0] = p.p[0]; o.p[1] = p.p[1]; o.p[2] = p.p[2];
  if (p.q && o.q) { o.q[0] = p.q[0]; o.q[1] = p.q[1]; o.q[2] = p.q[2]; o.q[3] = p.q[3]; }
  o.valid = p.valid !== false;
}

/** Allocate a per-hand button state object (shared shape with the emulated backends). */
export function createButtonState() {
  return {
    trigger: 0,        // analog 0..1
    squeeze: 0,        // analog 0..1
    stick: false,      // thumbstick press
    primary: false,    // A (right) / X (left)
    secondary: false,  // B (right) / Y (left)
    menu: false,
    thumb: [0, 0],     // thumbstick axes
    connected: false,
    isHand: false,
  };
}

function setPose(point, pose) {
  if (!pose) {
    point.valid = false;
    return false;
  }
  const t = pose.transform;
  const p = t.position, q = t.orientation;
  if (!p || !q || !Number.isFinite(p.x)) {
    point.valid = false;
    return false;
  }
  point.p[0] = p.x; point.p[1] = p.y; point.p[2] = p.z;
  point.q[0] = q.x; point.q[1] = q.y; point.q[2] = q.z; point.q[3] = q.w;
  point.valid = pose.emulatedPosition ? false : true;
  return true;
}

const PRESS_THRESHOLD = 0.5;

export class XRInput {
  /** @param {object} [opts] { session } the XRSession (may also be passed to update()) */
  constructor(opts = {}) {
    this.emulated = false;
    this.sample = createSample();
    this.buttons = { left: createButtonState(), right: createButtonState() };
    this._prev = { left: createButtonState(), right: createButtonState() };
    this._selectHeld = { left: false, right: false, none: false };
    this._squeezeHeld = { left: false, right: false, none: false };
    this.session = null;
    this.frameCount = 0;
    this._onSelectStart = (ev) => { this._selectHeld[handOf(ev)] = true; };
    this._onSelectEnd = (ev) => { this._selectHeld[handOf(ev)] = false; };
    this._onSqueezeStart = (ev) => { this._squeezeHeld[handOf(ev)] = true; };
    this._onSqueezeEnd = (ev) => { this._squeezeHeld[handOf(ev)] = false; };
    if (opts.session) this.attachSession(opts.session);
  }

  /** Listen to select/squeeze events (fallback for input sources without a gamepad). */
  attachSession(session) {
    if (this.session === session) return;
    this.detachSession();
    this.session = session;
    if (session && typeof session.addEventListener === 'function') {
      session.addEventListener('selectstart', this._onSelectStart);
      session.addEventListener('selectend', this._onSelectEnd);
      session.addEventListener('squeezestart', this._onSqueezeStart);
      session.addEventListener('squeezeend', this._onSqueezeEnd);
    }
  }

  detachSession() {
    const s = this.session;
    if (s && typeof s.removeEventListener === 'function') {
      s.removeEventListener('selectstart', this._onSelectStart);
      s.removeEventListener('selectend', this._onSelectEnd);
      s.removeEventListener('squeezestart', this._onSqueezeStart);
      s.removeEventListener('squeezeend', this._onSqueezeEnd);
    }
    this.session = null;
  }

  /**
   * Read the current frame. Call once per XR animation frame.
   * @param {XRFrame} xrFrame
   * @param {XRReferenceSpace} refSpace  ('local-floor')
   * @param {XRSession} [session]        defaults to the attached session / xrFrame.session
   * @param {number} [timeMs]            rAF timestamp (ms); defaults to performance.now()
   * @returns the reused sample
   */
  update(xrFrame, refSpace, session, timeMs) {
    const s = this.sample;
    if (typeof timeMs === 'number') s.t = timeMs / 1000;
    else if (typeof performance !== 'undefined') s.t = performance.now() / 1000;
    else s.t = Date.now() / 1000;
    this.frameCount++;

    const sess = session || (xrFrame && xrFrame.session) || this.session;
    if (sess && sess !== this.session) this.attachSession(sess);

    // previous button state for edge detection
    copyButtons(this._prev.left, this.buttons.left);
    copyButtons(this._prev.right, this.buttons.right);
    resetButtons(this.buttons.left);
    resetButtons(this.buttons.right);

    if (!xrFrame || !refSpace) {
      s.head.valid = false; s.left.valid = false; s.right.valid = false;
      return s;
    }

    // head
    let viewerPose = null;
    try {
      viewerPose = xrFrame.getViewerPose(refSpace);
    } catch (e) {
      viewerPose = null;
    }
    setPose(s.head, viewerPose);

    // hands / controllers
    let leftSeen = false, rightSeen = false;
    const sources = sess ? sess.inputSources : null;
    if (sources) {
      for (let i = 0; i < sources.length; i++) {
        const src = sources[i];
        const hand = src.handedness;
        if (hand !== 'left' && hand !== 'right') continue;
        const point = s[hand];
        const btn = this.buttons[hand];
        btn.connected = true;
        let pose = null;
        if (src.gripSpace) {
          try { pose = xrFrame.getPose(src.gripSpace, refSpace); } catch (e) { pose = null; }
        }
        if (!pose && src.hand && typeof src.hand.get === 'function' && typeof xrFrame.getJointPose === 'function') {
          btn.isHand = true;
          const wrist = src.hand.get('wrist');
          if (wrist) {
            try { pose = xrFrame.getJointPose(wrist, refSpace); } catch (e) { pose = null; }
          }
          // pinch (index tip to thumb tip) acts as trigger for tracked hands
          const tip = src.hand.get('index-finger-tip');
          const thumb = src.hand.get('thumb-tip');
          if (tip && thumb) {
            let a = null, b = null;
            try { a = xrFrame.getJointPose(tip, refSpace); b = xrFrame.getJointPose(thumb, refSpace); } catch (e) { a = null; }
            if (a && b) {
              const dx = a.transform.position.x - b.transform.position.x;
              const dy = a.transform.position.y - b.transform.position.y;
              const dz = a.transform.position.z - b.transform.position.z;
              const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
              btn.trigger = d < 0.02 ? 1 : 0;
            }
          }
        }
        if (!pose && src.targetRaySpace) {
          try { pose = xrFrame.getPose(src.targetRaySpace, refSpace); } catch (e) { pose = null; }
        }
        if (setPose(point, pose)) {
          if (hand === 'left') leftSeen = true; else rightSeen = true;
        }
        readGamepad(btn, src.gamepad);
        if (this._selectHeld[hand]) btn.trigger = Math.max(btn.trigger, 1);
        if (this._squeezeHeld[hand]) btn.squeeze = Math.max(btn.squeeze, 1);
      }
    }
    if (!leftSeen) s.left.valid = false;
    if (!rightSeen) s.right.valid = false;
    // 'none' handedness select (e.g. gaze/tap input) counts as a right trigger
    if (this._selectHeld.none) this.buttons.right.trigger = Math.max(this.buttons.right.trigger, 1);
    return s;
  }

  /** True on the frame the button went from released to pressed. hand: 'left'|'right'|'any'. */
  justPressed(hand, name = 'trigger') {
    if (hand === 'any') return this.justPressed('left', name) || this.justPressed('right', name);
    const cur = this.buttons[hand], prev = this._prev[hand];
    if (!cur || !prev) return false;
    return isPressed(cur, name) && !isPressed(prev, name);
  }

  /** True while the button is held. */
  isDown(hand, name = 'trigger') {
    if (hand === 'any') return this.isDown('left', name) || this.isDown('right', name);
    const cur = this.buttons[hand];
    return cur ? isPressed(cur, name) : false;
  }

  anyTriggerDown() {
    return this.isDown('any', 'trigger');
  }

  get tracking() {
    return this.sample.head.valid;
  }

  dispose() {
    this.detachSession();
  }
}

function handOf(ev) {
  const h = ev && ev.inputSource ? ev.inputSource.handedness : 'none';
  return h === 'left' || h === 'right' ? h : 'none';
}

function isPressed(b, name) {
  switch (name) {
    case 'trigger': return b.trigger >= PRESS_THRESHOLD;
    case 'squeeze': return b.squeeze >= PRESS_THRESHOLD;
    case 'stick': return b.stick;
    case 'primary': return b.primary;
    case 'secondary': return b.secondary;
    case 'menu': return b.menu;
    default: return false;
  }
}

function copyButtons(dst, src) {
  dst.trigger = src.trigger; dst.squeeze = src.squeeze; dst.stick = src.stick;
  dst.primary = src.primary; dst.secondary = src.secondary; dst.menu = src.menu;
  dst.thumb[0] = src.thumb[0]; dst.thumb[1] = src.thumb[1];
  dst.connected = src.connected; dst.isHand = src.isHand;
}

function resetButtons(b) {
  b.trigger = 0; b.squeeze = 0; b.stick = false; b.primary = false; b.secondary = false; b.menu = false;
  b.thumb[0] = 0; b.thumb[1] = 0; b.connected = false; b.isHand = false;
}

/** xr-standard gamepad mapping (Oculus Touch): 0 trigger, 1 squeeze, 3 stick, 4 A/X, 5 B/Y. */
function readGamepad(btn, gp) {
  if (!gp || !gp.buttons) return;
  const b = gp.buttons;
  const val = (i) => (b[i] ? (typeof b[i].value === 'number' ? b[i].value : (b[i].pressed ? 1 : 0)) : 0);
  const pressed = (i) => (b[i] ? (b[i].pressed === true || val(i) >= PRESS_THRESHOLD) : false);
  btn.trigger = Math.max(btn.trigger, val(0));
  btn.squeeze = Math.max(btn.squeeze, val(1));
  btn.stick = pressed(3);
  btn.primary = pressed(4);
  btn.secondary = pressed(5);
  btn.menu = pressed(6) || pressed(7);
  const ax = gp.axes;
  if (ax && ax.length >= 4) { btn.thumb[0] = ax[2]; btn.thumb[1] = ax[3]; }
  else if (ax && ax.length >= 2) { btn.thumb[0] = ax[0]; btn.thumb[1] = ax[1]; }
}
