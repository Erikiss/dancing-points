// hud.js - HUD: world-space canvas panels (score / combo, current + next move, progress bar,
// countdown, grade popups, notices, title) placed at the stage screen, never head-locked.
// `TextPanel` is a generic canvas-textured plane that only redraws when its content changes;
// the HUD setters compare against the last values so per-frame calls are cheap. All strings
// come from ui/texts.js (German). Imports three.js (browser only).

import * as THREE from 'three';
import { getTexts, GRADE_KEYS } from '../ui/texts.js';

export const HUD_COLORS = Object.freeze({
  text: '#f4f6ff',
  dim: '#aab3d6',
  accent: '#4cc9f0',
  warm: '#ffd166',
  panel: 'rgba(14, 18, 38, 0.82)',
  panelEdge: 'rgba(106, 127, 255, 0.55)',
  perfekt: '#4cff9a',
  gut: '#4cc9f0',
  ok: '#ffd166',
  daneben: '#ff6b6b',
});

export const FONT = '"Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
export const NOTICE_Y = 0.95;   // height (m) of the notice panel on the stage wall

/**
 * A plane with a canvas texture. `draw(fn, key)` runs `fn(ctx, w, h)` only when `key` differs
 * from the last key (compare primitives or preformatted strings; do not build keys per frame).
 */
export class TextPanel extends THREE.Mesh {
  /**
   * @param {object} opts { width (m), height (m), pxPerMeter=320, background=HUD_COLORS.panel,
   *   rounded=true, transparent=true, border=true }
   */
  constructor(opts = {}) {
    const width = opts.width > 0 ? opts.width : 1;
    const height = opts.height > 0 ? opts.height : 0.4;
    const ppm = opts.pxPerMeter > 0 ? opts.pxPerMeter : 320;
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(8, Math.round(width * ppm));
    canvas.height = Math.max(8, Math.round(height * ppm));
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    // mipmaps: the stage is scaled by h0/1.70, so the panels are minified for most players and
    // shimmer during head motion without them (NPOT mipmaps are fine on WebGL2 / Quest)
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.generateMipmaps = true;
    texture.anisotropy = 4;
    const material = new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false });
    super(new THREE.PlaneGeometry(width, height), material);
    this.name = opts.name || 'TextPanel';
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.texture = texture;
    this.width = width;
    this.height = height;
    this.pxPerMeter = ppm;
    this.background = opts.background !== undefined ? opts.background : HUD_COLORS.panel;
    this.border = opts.border !== false;
    this.rounded = opts.rounded !== false;
    this._key = undefined;
    this.renderOrder = 5;
  }

  /** Clear + paint the background. Called by draw(). */
  clear() {
    const ctx = this.ctx, w = this.canvas.width, h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (this.background) {
      ctx.fillStyle = this.background;
      roundRect(ctx, 2, 2, w - 4, h - 4, this.rounded ? Math.min(28, h * 0.18) : 0);
      ctx.fill();
      if (this.border) {
        ctx.strokeStyle = HUD_COLORS.panelEdge;
        ctx.lineWidth = 3;
        ctx.stroke();
      }
    }
  }

  /** Redraw when `key` changed. Returns true if it redrew. */
  draw(fn, key) {
    if (key !== undefined && key === this._key) return false;
    this._key = key;
    this.clear();
    fn(this.ctx, this.canvas.width, this.canvas.height);
    this.texture.needsUpdate = true;
    return true;
  }

  /** Force the next draw. */
  invalidate() {
    this._key = undefined;
  }

  setOpacity(o) {
    this.material.opacity = o;
  }

  dispose() {
    this.geometry.dispose();
    this.material.dispose();
    this.texture.dispose();
  }
}

export function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
  ctx.lineTo(x + rr, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
  ctx.lineTo(x, y + rr);
  ctx.quadraticCurveTo(x, y, x + rr, y);
  ctx.closePath();
}

/** Draw text that fits `maxWidth` by shrinking the font size (px). Returns the size used. */
export function fitText(ctx, text, x, y, maxWidth, size, weight = 'bold', align = 'center', color = HUD_COLORS.text) {
  let s = size;
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  ctx.fillStyle = color;
  ctx.font = `${weight} ${s}px ${FONT}`;
  while (s > 10 && ctx.measureText(text).width > maxWidth) {
    s -= 2;
    ctx.font = `${weight} ${s}px ${FONT}`;
  }
  ctx.fillText(text, x, y);
  return s;
}

const STAR_FULL = '★';
const STAR_EMPTY = '☆';

export function starsString(n, max = 5) {
  let s = '';
  for (let i = 0; i < max; i++) s += i < n ? STAR_FULL : STAR_EMPTY;
  return s;
}

/** Format seconds as m:ss. */
export function formatTime(sec) {
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${r < 10 ? '0' : ''}${r}`;
}

// ---------------------------------------------------------------------------------------------

export class HUD extends THREE.Group {
  /**
   * @param {object} [opts] { texts (getTexts(lang)), screenWidth=4.4, screenHeight=2.6, z=0 }
   * Position this group at the stage screen (SceneKit.screen position) under SceneKit.stage.
   */
  constructor(opts = {}) {
    super();
    this.name = 'HUD';
    this.texts = opts.texts || getTexts('de');
    const sw = opts.screenWidth > 0 ? opts.screenWidth : 4.4;
    const sh = opts.screenHeight > 0 ? opts.screenHeight : 2.6;
    const zf = 0.02; // in front of the wall
    // Layout (stage frame, the teacher stands 0.7 m in front of this wall with the head at
    // ~1.6 m): a top row above the teacher's head, the move panel just above it, the countdown
    // and grade popups left/right of the teacher so nothing important is occluded.
    const topY = sh - 0.32;

    // score + combo (top left)
    this.scorePanel = new TextPanel({ width: 1.3, height: 0.5, name: 'hud-score' });
    this.scorePanel.position.set(-sw / 2 + 0.75, topY, zf);
    this.add(this.scorePanel);

    // title / status (top centre)
    this.titlePanel = new TextPanel({ width: 1.6, height: 0.36, name: 'hud-title', background: 'rgba(14,18,38,0.6)' });
    this.titlePanel.position.set(0, topY, zf);
    this.add(this.titlePanel);

    // time (top right)
    this.timePanel = new TextPanel({ width: 1.0, height: 0.36, name: 'hud-time' });
    this.timePanel.position.set(sw / 2 - 0.6, topY, zf);
    this.add(this.timePanel);

    // current + next move (centre, above the teacher's head)
    this.movePanel = new TextPanel({ width: 2.4, height: 0.44, name: 'hud-move' });
    this.movePanel.position.set(0, topY - 0.48, zf);
    this.add(this.movePanel);

    // progress bar (thin, along the top edge of the wall)
    this.progressBg = new THREE.Mesh(new THREE.PlaneGeometry(3.6, 0.05), new THREE.MeshBasicMaterial({ color: 0x2a3358 }));
    this.progressBg.position.set(0, sh - 0.04, zf);
    this.add(this.progressBg);
    this.progressBar = new THREE.Mesh(new THREE.PlaneGeometry(3.6, 0.05), new THREE.MeshBasicMaterial({ color: 0x4cc9f0 }));
    this.progressBar.position.set(-1.8, sh - 0.04, zf + 0.002);
    this.progressBar.geometry.translate(1.8, 0, 0);   // scale from the left edge
    this.progressBar.scale.x = 0.0001;
    this.add(this.progressBar);

    // countdown (big, left of the teacher)
    this.countdownPanel = new TextPanel({ width: 1.0, height: 1.0, name: 'hud-countdown', background: 'rgba(14,18,38,0.5)' });
    this.countdownPanel.position.set(-1.45, 1.35, zf + 0.01);
    this.countdownPanel.visible = false;
    this.add(this.countdownPanel);

    // grade popup (right of the teacher, animated)
    this.gradePanel = new TextPanel({ width: 1.4, height: 0.5, name: 'hud-grade', background: null, border: false });
    this._gradeY = 1.45;
    this.gradePanel.position.set(1.4, this._gradeY, zf + 0.02);
    this.gradePanel.visible = false;
    this.add(this.gradePanel);
    this._gradeAge = 0;
    this._gradeLife = 0.9;

    // notice (temporary; at hip height of the teacher so it sits in the player's field of view -
    // at floor level it was ~24 degrees below the eye line and easy to miss)
    this.noticePanel = new TextPanel({ width: 3.0, height: 0.3, name: 'hud-notice', background: 'rgba(60, 20, 30, 0.85)' });
    this.noticePanel.position.set(0, NOTICE_Y, zf + 0.01);
    this.noticePanel.visible = false;
    this.add(this.noticePanel);
    this._noticeLeft = 0;

    // extra panels other lanes register (e.g. the duo's second score)
    this.extra = new Map();

    // last values (avoid redraws)
    this._score = -1; this._combo = -1;
    this._move = null; this._next = null; this._moveKey = '';
    this._title = null;
    this._time = -1;
    this._countdown = null;

    this.setPlayMode(false);
  }

  /** Show/hide the play-time panels (score, move, time, progress). */
  setPlayMode(on) {
    this.scorePanel.visible = on;
    this.movePanel.visible = on;
    this.timePanel.visible = on;
    this.progressBg.visible = on;
    this.progressBar.visible = on;
    if (!on) {
      this.countdownPanel.visible = false;
      this.gradePanel.visible = false;
    }
  }

  /** Reset the cached values (start of a run). */
  reset() {
    this._score = -1; this._combo = -1; this._move = null; this._next = null; this._moveKey = '';
    this._time = -1; this._countdown = null;
    this.progressBar.scale.x = 0.0001;
    this.gradePanel.visible = false;
    this.countdownPanel.visible = false;
    this.scorePanel.invalidate();
    this.movePanel.invalidate();
    this.timePanel.invalidate();
  }

  showScore(score, combo) {
    if (score === this._score && combo === this._combo) return;
    this._score = score; this._combo = combo;
    const t = this.texts;
    this.scorePanel.draw((ctx, w, h) => {
      fitText(ctx, t.hudScore, w * 0.3, h * 0.28, w * 0.5, 30, '600', 'center', HUD_COLORS.dim);
      fitText(ctx, String(score), w * 0.3, h * 0.68, w * 0.5, 84, 'bold', 'center', HUD_COLORS.text);
      fitText(ctx, t.hudCombo, w * 0.76, h * 0.28, w * 0.4, 30, '600', 'center', HUD_COLORS.dim);
      fitText(ctx, `×${combo}`, w * 0.76, h * 0.68, w * 0.4, 64, 'bold', 'center', combo >= 4 ? HUD_COLORS.warm : HUD_COLORS.text);
    });
  }

  /** Current move name and the next one (strings or null). */
  showMove(current, next, hint = '') {
    if (current === this._move && next === this._next) return;
    this._move = current; this._next = next;
    const t = this.texts;
    this.movePanel.draw((ctx, w, h) => {
      const main = current || '';
      fitText(ctx, main, w / 2, h * 0.36, w * 0.9, 64, 'bold', 'center', HUD_COLORS.accent);
      const sub = next ? `${t.hudNextMove}: ${next}` : (hint || '');
      if (sub) fitText(ctx, sub, w / 2, h * 0.76, w * 0.9, 32, '500', 'center', HUD_COLORS.dim);
    });
  }

  showTitle(text) {
    if (text === this._title) return;
    this._title = text;
    this.titlePanel.visible = !!text;
    if (!text) return;
    this.titlePanel.draw((ctx, w, h) => {
      fitText(ctx, text, w / 2, h / 2, w * 0.92, 40, 'bold', 'center', HUD_COLORS.text);
    });
  }

  /** Remaining time (seconds); redraws once per second (no allocation otherwise). */
  showTime(secondsLeft) {
    const sec = secondsLeft > 0 ? Math.round(secondsLeft) : 0;
    if (sec === this._time) return;
    this._time = sec;
    const t = this.texts;
    const s = formatTime(sec);
    this.timePanel.draw((ctx, w, h) => {
      fitText(ctx, `${t.hudTime} ${s}`, w / 2, h / 2, w * 0.9, 40, 'bold', 'center', HUD_COLORS.text);
    });
  }

  showProgress(p) {
    const v = p < 0 ? 0 : (p > 1 ? 1 : p);
    this.progressBar.scale.x = v < 0.0001 ? 0.0001 : v;
  }

  /** Countdown: a number (beats left), 'go' for "Los!", or null to hide. */
  showCountdown(value) {
    if (value === this._countdown) return;
    this._countdown = value;
    if (value === null || value === undefined) {
      this.countdownPanel.visible = false;
      return;
    }
    const t = this.texts;
    this.countdownPanel.visible = true;
    this.countdownPanel.draw((ctx, w, h) => {
      if (value === 'go') {
        fitText(ctx, t.countdownGo, w / 2, h / 2, w * 0.9, 150, 'bold', 'center', HUD_COLORS.perfekt);
      } else {
        fitText(ctx, t.countdownReady, w / 2, h * 0.2, w * 0.9, 44, '600', 'center', HUD_COLORS.dim);
        fitText(ctx, String(value), w / 2, h * 0.6, w * 0.9, 220, 'bold', 'center', HUD_COLORS.warm);
      }
    }, value);
  }

  /** Grade popup ('perfekt'|'gut'|'ok'|'daneben') with the combo; animated in update(). */
  showGrade(grade, combo = 0) {
    const key = GRADE_KEYS[grade] || 'gradeDaneben';
    const label = this.texts[key];
    const color = HUD_COLORS[grade] || HUD_COLORS.text;
    this.gradePanel.draw((ctx, w, h) => {
      fitText(ctx, label, w / 2, h * 0.4, w * 0.9, 96, 'bold', 'center', color);
      if (combo >= 2) fitText(ctx, `×${combo}`, w / 2, h * 0.82, w * 0.9, 40, '600', 'center', HUD_COLORS.dim);
    }, grade + '|' + (combo >= 2 ? combo : 0));
    this.gradePanel.visible = true;
    this._gradeAge = 0;
    this.gradePanel.scale.setScalar(1.25);
    this.gradePanel.material.opacity = 1;
  }

  /** Temporary message (seconds). */
  showNotice(text, seconds = 3) {
    if (!text) { this.noticePanel.visible = false; return; }
    this.noticePanel.draw((ctx, w, h) => {
      fitText(ctx, text, w / 2, h / 2, w * 0.94, 34, '600', 'center', HUD_COLORS.text);
    }, text);
    this.noticePanel.visible = true;
    this._noticeLeft = seconds;
  }

  /**
   * Register an extra panel (other lanes): `hud.addPanel('duo', {width, height, position:[x,y,z]})`
   * returns the TextPanel; draw with `panel.draw(fn, key)`.
   */
  addPanel(id, spec = {}) {
    this.removePanel(id);
    const p = new TextPanel({ width: spec.width, height: spec.height, name: `hud-${id}`, background: spec.background, border: spec.border });
    const pos = spec.position || [0, 1.2, 0.02];
    p.position.set(pos[0], pos[1], pos[2]);
    this.add(p);
    this.extra.set(id, p);
    return p;
  }

  removePanel(id) {
    const p = this.extra.get(id);
    if (!p) return;
    this.remove(p);
    p.dispose();
    this.extra.delete(id);
  }

  getPanel(id) {
    return this.extra.get(id) || null;
  }

  /** Animate popups / notices. dt in seconds. */
  update(dt) {
    if (this.gradePanel.visible) {
      this._gradeAge += dt;
      const a = this._gradeAge / this._gradeLife;
      if (a >= 1) {
        this.gradePanel.visible = false;
      } else {
        const s = 1.25 - 0.25 * Math.min(1, a * 4);
        this.gradePanel.scale.setScalar(s);
        this.gradePanel.material.opacity = a < 0.6 ? 1 : 1 - (a - 0.6) / 0.4;
        this.gradePanel.position.y += dt * 0.15;
        if (a < 0.05) this.gradePanel.position.y = this._gradeBaseY();
      }
    }
    if (this.noticePanel.visible) {
      this._noticeLeft -= dt;
      if (this._noticeLeft <= 0) this.noticePanel.visible = false;
    }
  }

  _gradeBaseY() {
    return this._gradeY;
  }

  dispose() {
    for (const p of [this.scorePanel, this.titlePanel, this.timePanel, this.movePanel, this.countdownPanel, this.gradePanel, this.noticePanel]) p.dispose();
    for (const p of this.extra.values()) p.dispose();
    this.extra.clear();
  }
}
