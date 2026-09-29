// menu.js - Menu: a data-driven system of world-space panels (canvas textures on planes) with
// buttons, lists, steppers, labels and progress bars. Interaction: a laser pointer from an XR
// controller (`setPointerFromObject` / `setPointer` + `press()` on trigger), the mouse in emu
// mode (`setMouse(ndcX, ndcY)` + `press()`), and keyboard shortcuts (`handleKey(code)`).
// Panels are described by plain specs so other lanes can add their own via
// `menu.addPanel(id, spec)`. Panels only redraw when hover/focus/content changes. Player-facing
// strings are passed in by the caller (ui/texts.js). Imports three.js (browser only).

import * as THREE from 'three';
import { EventEmitter } from '../util/events.js';
import { getTexts } from './texts.js';
import { roundRect, FONT } from '../render/hud.js';

export const MENU_COLORS = Object.freeze({
  panel: 'rgba(14, 18, 38, 0.92)',
  panelEdge: 'rgba(106, 127, 255, 0.6)',
  text: '#f4f6ff',
  dim: '#aab3d6',
  accent: '#4cc9f0',
  accentText: '#08111f',
  button: 'rgba(58, 68, 104, 0.9)',
  buttonHover: 'rgba(90, 104, 160, 0.95)',
  buttonDisabled: 'rgba(40, 46, 70, 0.7)',
  focus: '#ffd166',
  selected: 'rgba(76, 201, 240, 0.25)',
  danger: '#ff6b6b',
  good: '#4cff9a',
  bar: '#4cc9f0',
  barBg: 'rgba(58, 68, 104, 0.9)',
});

// layout in canvas pixels (at DEFAULT_PPM px per metre)
export const DEFAULT_PPM = 640;
const PAD = 36;
const GAP = 14;
const TITLE_H = 84;
const LABEL_H = { small: 40, normal: 50, large: 66 };
const TEXT_LINE = 38;
const BUTTON_H = 76;
const LIST_ROW = 70;
const STEPPER_H = 76;
const PROGRESS_H = 30;
const FOOTER_H = 44;
const SPACER_H = 24;

const _origin = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _ndc = new THREE.Vector2();

function val(v, ...args) {
  return typeof v === 'function' ? v(...args) : v;
}

function wrapLines(ctx, text, maxWidth) {
  const out = [];
  const paragraphs = String(text).split('\n');
  for (const para of paragraphs) {
    const words = para.split(' ');
    let line = '';
    for (const w of words) {
      const test = line ? line + ' ' + w : w;
      if (ctx.measureText(test).width > maxWidth && line) {
        out.push(line);
        line = w;
      } else {
        line = test;
      }
    }
    out.push(line);
  }
  return out;
}

/** Stepper button/box x positions, laid out from the right edge: [«] [−] [value] [+] [»]. */
function stepperRects(item, x, inner) {
  const bw = 84, boxW = 200;
  const right = x + inner;
  const inc2 = item.onInc2 ? right - bw : null;
  const r1 = inc2 !== null ? inc2 - GAP : right;
  const inc = r1 - bw;
  const box = inc - GAP - boxW;
  const dec = box - GAP - bw;
  const dec2 = item.onDec2 ? dec - GAP - bw : null;
  return { bw, boxW, inc2, inc, box, dec, dec2 };
}

/** One panel: spec + canvas + mesh + hit regions. Created through Menu.addPanel. */
export class MenuPanel {
  constructor(menu, id, spec) {
    this.menu = menu;
    this.id = id;
    this.spec = null;
    this.ppm = spec.pxPerMeter > 0 ? spec.pxPerMeter : menu.pxPerMeter;
    this.widthM = spec.width > 0 ? spec.width : 1.6;
    this.pxW = Math.round(this.widthM * this.ppm);
    this.pxH = 0;
    this.heightM = 0;
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.pxW;
    this.canvas.height = 64;
    this.ctx = this.canvas.getContext('2d');
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    // mipmaps: the panel is minified for players shorter than the reference height (stage scale)
    // and at ~640 px/m on a Quest 2; without them text shimmers during head motion
    this.texture.minFilter = THREE.LinearMipmapLinearFilter;
    this.texture.generateMipmaps = true;
    this.texture.anisotropy = 4;
    this.material = new THREE.MeshBasicMaterial({ map: this.texture, transparent: true, depthWrite: false });
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(this.widthM, 0.1), this.material);
    this.mesh.name = `menu-${id}`;
    this.mesh.renderOrder = 6;
    this.mesh.userData.panel = this;
    this.hits = [];
    this.hover = -1;
    this.focus = -1;
    this.dirty = true;
    this.visible = false;
    this.setSpec(spec);
  }

  setSpec(spec) {
    this.spec = spec;
    const pos = spec.position || [0, 0, 0];
    this.mesh.position.set(pos[0], pos[1], pos[2]);
    this.mesh.rotation.set(0, spec.rotationY || 0, 0);
    this.layout();
    this.dirty = true;
  }

  /** Compute item rectangles + total height; rebuilds the geometry when the height changes. */
  layout() {
    const spec = this.spec;
    const ctx = this.ctx;
    const w = this.pxW;
    const inner = w - 2 * PAD;
    const rows = [];       // {item, y, h, extra}
    let y = PAD;
    if (spec.title) { rows.push({ kind: 'title', y, h: TITLE_H }); y += TITLE_H + GAP; }
    const items = val(spec.items, this.menu.app) || [];
    for (const item of items) {
      const type = item.type || 'label';
      let h = 0, extra = null;
      switch (type) {
        case 'label': h = LABEL_H[item.size] || LABEL_H.normal; break;
        case 'text': {
          ctx.font = `500 30px ${FONT}`;
          const lines = wrapLines(ctx, val(item.text, this.menu.app), inner);
          h = lines.length * TEXT_LINE + 8;
          extra = lines;
          break;
        }
        case 'button': case 'toggle': h = BUTTON_H; break;
        case 'row': h = BUTTON_H; break;
        case 'list': {
          const options = val(item.options, this.menu.app) || [];
          const pageSize = item.pageSize > 0 ? item.pageSize : 5;
          const pages = Math.max(1, Math.ceil(options.length / pageSize));
          if (!(item.page >= 0)) item.page = 0;
          if (item.page >= pages) item.page = pages - 1;
          const start = item.page * pageSize;
          const shown = options.slice(start, start + pageSize);
          h = Math.max(1, shown.length) * LIST_ROW + (pages > 1 ? BUTTON_H + GAP : 0);
          extra = { options, shown, pages, pageSize, start };
          break;
        }
        case 'stepper': h = STEPPER_H; break;
        case 'progress': h = PROGRESS_H; break;
        case 'spacer': h = item.height > 0 ? item.height : SPACER_H; break;
        default: h = LABEL_H.normal;
      }
      rows.push({ kind: type, item, y, h, extra });
      y += h + GAP;
    }
    if (spec.footer) { rows.push({ kind: 'footer', y, h: FOOTER_H }); y += FOOTER_H; }
    y += PAD;
    const minPx = spec.minHeight > 0 ? Math.round(spec.minHeight * this.ppm) : 0;
    const pxH = Math.max(minPx, Math.round(y));
    this.rows = rows;
    if (pxH !== this.pxH) {
      this.pxH = pxH;
      this.heightM = pxH / this.ppm;
      this.canvas.height = pxH;
      this.mesh.geometry.dispose();
      this.mesh.geometry = new THREE.PlaneGeometry(this.widthM, this.heightM);
      // the GPU texture keeps its old size; dispose it so the next upload re-creates it
      this.texture.dispose();
      this.texture.needsUpdate = true;
      this.dirty = true;
    }
    // hit regions
    const hits = [];
    for (const r of rows) {
      const item = r.item;
      if (!item) continue;
      const x = PAD;
      switch (r.kind) {
        case 'button': case 'toggle':
          hits.push({ x, y: r.y, w: inner, h: r.h, item, kind: r.kind, action: () => this._activate(item) });
          break;
        case 'row': {
          const subs = val(item.items, this.menu.app) || [];
          const n = subs.length || 1;
          const bw = (inner - GAP * (n - 1)) / n;
          subs.forEach((sub, i) => {
            hits.push({ x: x + i * (bw + GAP), y: r.y, w: bw, h: r.h, item: sub, kind: sub.type || 'button', action: () => this._activate(sub) });
          });
          break;
        }
        case 'list': {
          const { shown, pages, start } = r.extra;
          shown.forEach((opt, i) => {
            hits.push({ x, y: r.y + i * LIST_ROW, w: inner, h: LIST_ROW - 4, item, option: opt, kind: 'option', index: start + i, action: () => this._select(item, opt) });
          });
          if (pages > 1) {
            const py = r.y + shown.length * LIST_ROW + GAP;
            hits.push({ x, y: py, w: 120, h: BUTTON_H, item, kind: 'page', label: '‹', action: () => this._page(item, -1, pages) });
            hits.push({ x: x + inner - 120, y: py, w: 120, h: BUTTON_H, item, kind: 'page', label: '›', action: () => this._page(item, 1, pages) });
          }
          break;
        }
        case 'stepper': {
          const rc = stepperRects(item, x, inner);
          if (rc.dec2) hits.push({ x: rc.dec2, y: r.y, w: rc.bw, h: r.h, item, kind: 'step', label: '«', action: () => this._step(item, 'onDec2') });
          hits.push({ x: rc.dec, y: r.y, w: rc.bw, h: r.h, item, kind: 'step', label: '−', action: () => this._step(item, 'onDec') });
          hits.push({ x: rc.inc, y: r.y, w: rc.bw, h: r.h, item, kind: 'step', label: '+', action: () => this._step(item, 'onInc') });
          if (rc.inc2) hits.push({ x: rc.inc2, y: r.y, w: rc.bw, h: r.h, item, kind: 'step', label: '»', action: () => this._step(item, 'onInc2') });
          break;
        }
        default:
          break;
      }
    }
    this.hits = hits;
    if (this.hover >= hits.length) this.hover = -1;
    if (this.focus >= hits.length) this.focus = -1;
  }

  _activate(item) {
    if (val(item.disabled, this.menu.app)) return;
    if (item.type === 'toggle' && typeof item.onChange === 'function') item.onChange(!val(item.value, this.menu.app), this.menu.app);
    else if (typeof item.onClick === 'function') item.onClick(this.menu.app, item);
    this.menu.emit('action', { panelId: this.id, itemId: item.id || null, item });
    this.refresh();
  }

  _select(item, opt) {
    item.selected = opt.id;
    if (typeof item.onSelect === 'function') item.onSelect(opt.id, opt, this.menu.app);
    this.menu.emit('action', { panelId: this.id, itemId: item.id || null, item, option: opt });
    this.refresh();
  }

  _page(item, delta, pages) {
    item.page = ((item.page + delta) % pages + pages) % pages;
    this.refresh();
  }

  _step(item, fn) {
    if (typeof item[fn] === 'function') item[fn](this.menu.app, item);
    this.menu.emit('action', { panelId: this.id, itemId: item.id || null, item, step: fn });
    this.refresh();
  }

  /** Re-layout + redraw (content may have changed). */
  refresh() {
    this.layout();
    this.dirty = true;
  }

  invalidate() {
    this.dirty = true;
  }

  /** Hit index for canvas pixel coordinates, -1 if none. */
  hitAt(px, py) {
    const hits = this.hits;
    for (let i = 0; i < hits.length; i++) {
      const h = hits[i];
      if (px >= h.x && px <= h.x + h.w && py >= h.y && py <= h.y + h.h) return i;
    }
    return -1;
  }

  setHover(i) {
    if (i === this.hover) return;
    this.hover = i;
    this.dirty = true;
  }

  setFocus(i) {
    if (i === this.focus) return;
    this.focus = i;
    this.dirty = true;
  }

  draw() {
    this.dirty = false;
    const ctx = this.ctx, w = this.pxW, h = this.pxH;
    const spec = this.spec;
    const app = this.menu.app;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = spec.background || MENU_COLORS.panel;
    roundRect(ctx, 2, 2, w - 4, h - 4, 30);
    ctx.fill();
    ctx.strokeStyle = MENU_COLORS.panelEdge;
    ctx.lineWidth = 3;
    ctx.stroke();
    const inner = w - 2 * PAD;
    ctx.textBaseline = 'middle';
    for (const r of this.rows) {
      const item = r.item;
      switch (r.kind) {
        case 'title':
          ctx.textAlign = 'left';
          ctx.fillStyle = MENU_COLORS.accent;
          ctx.font = `bold 52px ${FONT}`;
          ctx.fillText(val(spec.title, app), PAD, r.y + TITLE_H * 0.45);
          ctx.fillStyle = MENU_COLORS.panelEdge;
          ctx.fillRect(PAD, r.y + TITLE_H - 6, inner, 3);
          break;
        case 'footer':
          ctx.textAlign = 'center';
          ctx.fillStyle = MENU_COLORS.dim;
          ctx.font = `500 24px ${FONT}`;
          ctx.fillText(val(spec.footer, app), w / 2, r.y + FOOTER_H / 2);
          break;
        case 'label': {
          const size = item.size === 'small' ? 28 : (item.size === 'large' ? 46 : 34);
          ctx.textAlign = item.align || 'left';
          ctx.fillStyle = item.color || (item.size === 'small' ? MENU_COLORS.dim : MENU_COLORS.text);
          ctx.font = `${item.size === 'large' ? 'bold' : '600'} ${size}px ${FONT}`;
          const tx = item.align === 'center' ? w / 2 : (item.align === 'right' ? w - PAD : PAD);
          ctx.fillText(String(val(item.text, app)), tx, r.y + r.h / 2);
          break;
        }
        case 'text': {
          ctx.textAlign = 'left';
          ctx.fillStyle = item.color || MENU_COLORS.text;
          ctx.font = `500 30px ${FONT}`;
          r.extra.forEach((line, i) => ctx.fillText(line, PAD, r.y + 4 + i * TEXT_LINE + TEXT_LINE / 2));
          break;
        }
        case 'progress': {
          const v = Math.max(0, Math.min(1, Number(val(item.value, app)) || 0));
          ctx.fillStyle = MENU_COLORS.barBg;
          roundRect(ctx, PAD, r.y + 4, inner, r.h - 8, 10);
          ctx.fill();
          if (v > 0) {
            ctx.fillStyle = item.color || MENU_COLORS.bar;
            roundRect(ctx, PAD, r.y + 4, inner * v, r.h - 8, 10);
            ctx.fill();
          }
          break;
        }
        case 'stepper': {
          ctx.textAlign = 'left';
          ctx.fillStyle = MENU_COLORS.text;
          ctx.font = `600 34px ${FONT}`;
          ctx.fillText(String(val(item.text, app)), PAD, r.y + r.h / 2);
          // value box
          const rc = stepperRects(item, PAD, inner);
          ctx.fillStyle = MENU_COLORS.buttonDisabled;
          roundRect(ctx, rc.box, r.y + 6, rc.boxW, r.h - 12, 12);
          ctx.fill();
          ctx.textAlign = 'center';
          ctx.fillStyle = MENU_COLORS.accent;
          ctx.font = `bold 36px ${FONT}`;
          ctx.fillText(String(val(item.value, app)), rc.box + rc.boxW / 2, r.y + r.h / 2);
          break;
        }
        case 'list': {
          const { shown, pages } = r.extra;
          if (shown.length === 0) {
            ctx.textAlign = 'center';
            ctx.fillStyle = MENU_COLORS.dim;
            ctx.font = `500 30px ${FONT}`;
            ctx.fillText(String(val(item.emptyText, app) || this.menu.texts.menuNoChoreos), w / 2, r.y + LIST_ROW / 2);
          }
          if (pages > 1) {
            ctx.textAlign = 'center';
            ctx.fillStyle = MENU_COLORS.dim;
            ctx.font = `500 28px ${FONT}`;
            ctx.fillText(`${this.menu.texts.menuPage} ${item.page + 1}/${pages}`, w / 2, r.y + shown.length * LIST_ROW + GAP + BUTTON_H / 2);
          }
          break;
        }
        default:
          break;
      }
    }
    // interactive regions
    for (let i = 0; i < this.hits.length; i++) {
      const hit = this.hits[i];
      const hovered = i === this.hover;
      const focused = i === this.focus;
      switch (hit.kind) {
        case 'button': case 'toggle': {
          const item = hit.item;
          const disabled = !!val(item.disabled, app);
          const primary = !!item.primary;
          let text = String(val(item.text, app));
          if (hit.kind === 'toggle') {
            const v = val(item.value, app);
            const on = this.menu.texts.settingsOn, off = this.menu.texts.settingsOff;
            text = `${text}: ${typeof v === 'boolean' ? (v ? on : off) : v}`;
          }
          ctx.fillStyle = disabled ? MENU_COLORS.buttonDisabled : (primary ? MENU_COLORS.accent : (hovered ? MENU_COLORS.buttonHover : MENU_COLORS.button));
          roundRect(ctx, hit.x, hit.y, hit.w, hit.h, 16);
          ctx.fill();
          if (hovered || focused) {
            ctx.strokeStyle = focused ? MENU_COLORS.focus : MENU_COLORS.text;
            ctx.lineWidth = 4;
            ctx.stroke();
          }
          ctx.textAlign = 'center';
          ctx.fillStyle = disabled ? MENU_COLORS.dim : (primary ? MENU_COLORS.accentText : (item.color || MENU_COLORS.text));
          let size = 36;
          ctx.font = `bold ${size}px ${FONT}`;
          while (size > 18 && ctx.measureText(text).width > hit.w - 24) { size -= 2; ctx.font = `bold ${size}px ${FONT}`; }
          ctx.fillText(text, hit.x + hit.w / 2, hit.y + hit.h / 2);
          break;
        }
        case 'option': {
          const opt = hit.option;
          const selected = hit.item.selected === opt.id;
          ctx.fillStyle = selected ? MENU_COLORS.selected : (hovered ? MENU_COLORS.buttonHover : 'rgba(58, 68, 104, 0.45)');
          roundRect(ctx, hit.x, hit.y, hit.w, hit.h, 12);
          ctx.fill();
          if (selected || hovered || focused) {
            ctx.strokeStyle = focused ? MENU_COLORS.focus : (selected ? MENU_COLORS.accent : MENU_COLORS.text);
            ctx.lineWidth = selected ? 4 : 3;
            ctx.stroke();
          }
          ctx.textAlign = 'left';
          ctx.fillStyle = opt.color || MENU_COLORS.text;
          let size = 32;
          ctx.font = `bold ${size}px ${FONT}`;
          const subText = opt.sub ? String(opt.sub) : '';
          ctx.font = `500 24px ${FONT}`;
          const subW = subText ? ctx.measureText(subText).width : 0;
          ctx.font = `bold ${size}px ${FONT}`;
          const maxW = hit.w - 32 - subW - (subText ? 24 : 0);
          while (size > 18 && ctx.measureText(String(opt.text)).width > maxW) { size -= 2; ctx.font = `bold ${size}px ${FONT}`; }
          ctx.fillText(String(opt.text), hit.x + 16, hit.y + hit.h / 2);
          if (subText) {
            ctx.textAlign = 'right';
            ctx.fillStyle = MENU_COLORS.dim;
            ctx.font = `500 24px ${FONT}`;
            ctx.fillText(subText, hit.x + hit.w - 16, hit.y + hit.h / 2);
          }
          break;
        }
        case 'page': case 'step': {
          ctx.fillStyle = hovered ? MENU_COLORS.buttonHover : MENU_COLORS.button;
          roundRect(ctx, hit.x, hit.y, hit.w, hit.h, 14);
          ctx.fill();
          if (hovered || focused) {
            ctx.strokeStyle = focused ? MENU_COLORS.focus : MENU_COLORS.text;
            ctx.lineWidth = 4;
            ctx.stroke();
          }
          ctx.textAlign = 'center';
          ctx.fillStyle = MENU_COLORS.text;
          ctx.font = `bold 40px ${FONT}`;
          ctx.fillText(hit.label, hit.x + hit.w / 2, hit.y + hit.h / 2);
          break;
        }
        default:
          break;
      }
    }
    this.texture.needsUpdate = true;
  }

  dispose() {
    this.mesh.geometry.dispose();
    this.material.dispose();
    this.texture.dispose();
  }
}

// ---------------------------------------------------------------------------------------------

export class Menu extends THREE.Group {
  /**
   * @param {object} [opts] { texts, camera (for the mouse raycast), app (passed to callbacks),
   *   pxPerMeter=640, laserColor }
   */
  constructor(opts = {}) {
    super();
    this.name = 'Menu';
    this.texts = opts.texts || getTexts('de');
    this.camera = opts.camera || null;
    this.app = opts.app || null;
    this.pxPerMeter = opts.pxPerMeter > 0 ? opts.pxPerMeter : DEFAULT_PPM;
    this.events = new EventEmitter();
    this.panels = new Map();
    this.activeId = null;
    this._meshes = [];
    this._raycaster = new THREE.Raycaster();
    this._raycaster.far = 8;
    this._hits = [];
    this._pointer = { active: false, origin: new THREE.Vector3(), dir: new THREE.Vector3(0, 0, -1), hand: 'right' };
    this._mouse = { active: false, x: 0, y: 0 };
    this.hoverPanel = null;
    this.hoverHit = -1;
    this.hitPoint = new THREE.Vector3();
    this.hasHit = false;
    this.pressLockMs = opts.pressLockMs >= 0 ? opts.pressLockMs : DEFAULT_PRESS_LOCK_MS;
    this._shownAt = -Infinity;
    this.visible = false;   // no panel shown yet (also gates the XR laser in the App loop)

    // laser + reticle live in world space (added to the scene by the app)
    const lgeo = new THREE.BufferGeometry();
    this._laserPos = new Float32Array(6);
    lgeo.setAttribute('position', new THREE.BufferAttribute(this._laserPos, 3));
    this.laser = new THREE.Line(lgeo, new THREE.LineBasicMaterial({ color: opts.laserColor !== undefined ? opts.laserColor : 0x4cc9f0, transparent: true, opacity: 0.8 }));
    this.laser.frustumCulled = false;
    this.laser.visible = false;
    this.reticle = new THREE.Mesh(new THREE.SphereGeometry(0.012, 10, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    this.reticle.visible = false;
    this.reticle.renderOrder = 10;
  }

  on(type, fn) { return this.events.on(type, fn); }
  off(type, fn) { return this.events.off(type, fn); }
  emit(type, a, b) { return this.events.emit(type, a, b); }

  // ---- panels --------------------------------------------------------------------------------

  /**
   * Add (or replace) a panel. spec: { title, width=1.6, position=[x,y,z], rotationY, items: [...]
   * | () => [...], footer, onBack, persistent, minHeight, background, pxPerMeter }.
   * Item types: label {text,size,align,color}, text {text}, button {id,text,onClick,disabled,
   * primary,color}, toggle {id,text,value,onChange}, row {items:[button|toggle]}, list {id,
   * options:[{id,text,sub,color}], selected, pageSize, onSelect, emptyText}, stepper {text,value,
   * onDec,onInc,onDec2,onInc2}, progress {value,color}, spacer {height}. `text`, `value`,
   * `options`, `items`, `disabled` may be functions of the app.
   */
  addPanel(id, spec) {
    this.removePanel(id);
    const panel = new MenuPanel(this, id, spec);
    this.panels.set(id, panel);
    this.add(panel.mesh);
    panel.mesh.visible = false;
    panel.visible = false;
    return panel;
  }

  removePanel(id) {
    const p = this.panels.get(id);
    if (!p) return;
    this.remove(p.mesh);
    p.dispose();
    this.panels.delete(id);
    if (this.activeId === id) this.activeId = null;
    this._rebuildMeshList();
  }

  getPanel(id) {
    return this.panels.get(id) || null;
  }

  /** Replace a panel's spec in place (keeps identity, re-layouts). */
  setPanel(id, spec) {
    const p = this.panels.get(id);
    if (!p) return this.addPanel(id, spec);
    p.setSpec(spec);
    return p;
  }

  /** Show `id` (hides other non-persistent panels). Extra ids may be shown alongside. */
  show(id, alongside = null) {
    for (const [pid, p] of this.panels) {
      const on = pid === id || (alongside !== null && alongside.includes(pid)) || p.spec.persistent === true;
      this._setPanelVisible(p, on);
    }
    this.activeId = id;
    const p = this.panels.get(id);
    if (p) { p.hover = -1; p.focus = -1; p.refresh(); }
    this._rebuildMeshList();
    this.visible = true;
    this._shownAt = now();
  }

  /** Show a panel in addition to the active one. */
  showAlso(id) {
    const p = this.panels.get(id);
    if (p) { this._setPanelVisible(p, true); p.refresh(); }
    this._rebuildMeshList();
    this.visible = true;
  }

  hidePanel(id) {
    const p = this.panels.get(id);
    if (p) this._setPanelVisible(p, false);
    this._rebuildMeshList();
  }

  hideAll() {
    for (const p of this.panels.values()) this._setPanelVisible(p, false);
    this.activeId = null;
    this._rebuildMeshList();
    this.laser.visible = false;
    this.reticle.visible = false;
    this.hoverPanel = null;
    this.hoverHit = -1;
    this.visible = false;   // no laser / trigger handling while nothing is shown (during a dance)
  }

  _setPanelVisible(p, on) {
    p.visible = on;
    p.mesh.visible = on;
    if (!on) { p.hover = -1; }
  }

  _rebuildMeshList() {
    this._meshes.length = 0;
    for (const p of this.panels.values()) if (p.visible) this._meshes.push(p.mesh);
  }

  /** Re-layout + redraw one panel or all visible panels (after data changes). */
  refresh(id = null) {
    if (id) { const p = this.panels.get(id); if (p) p.refresh(); return; }
    for (const p of this.panels.values()) if (p.visible) p.refresh();
  }

  get active() {
    return this.activeId ? this.panels.get(this.activeId) || null : null;
  }

  // ---- pointing ------------------------------------------------------------------------------

  /** Laser from a world-space origin + direction (XR controller targetRay). */
  setPointer(origin, dir, hand = 'right') {
    this._pointer.origin.copy(origin);
    this._pointer.dir.copy(dir).normalize();
    this._pointer.active = true;
    this._pointer.hand = hand;
    this._mouse.active = false;
  }

  /** Laser from an Object3D's world transform (-Z forward), e.g. renderer.xr.getController(i). */
  setPointerFromObject(obj, hand = 'right') {
    obj.updateMatrixWorld(true);
    _origin.setFromMatrixPosition(obj.matrixWorld);
    _dir.set(0, 0, -1).transformDirection(obj.matrixWorld);
    this.setPointer(_origin, _dir, hand);
  }

  /** Laser from a W-frame pose {p:[x,y,z], q:[x,y,z,w]} (e.g. an input sample's hand). */
  setPointerFromPose(pose, hand = 'right') {
    _origin.set(pose.p[0], pose.p[1], pose.p[2]);
    _quat.set(pose.q[0], pose.q[1], pose.q[2], pose.q[3]);
    _dir.set(0, 0, -1).applyQuaternion(_quat);
    this.setPointer(_origin, _dir, hand);
  }

  clearPointer() {
    this._pointer.active = false;
    this.laser.visible = false;
    this.reticle.visible = false;
  }

  /** Mouse position in normalised device coordinates (emu mode). */
  setMouse(ndcX, ndcY) {
    this._mouse.x = ndcX;
    this._mouse.y = ndcY;
    this._mouse.active = true;
    this._pointer.active = false;
  }

  /** True while the reticle/mouse is over an interactive item of a visible panel. */
  get hovering() {
    return !!(this.hoverPanel && this.hoverPanel.visible && this.hoverHit >= 0);
  }

  /** Activate the hovered item. Returns true when something was activated. */
  press() {
    const p = this.hoverPanel;
    if (!p || this.hoverHit < 0 || !p.visible) return false;
    // a squeeze that outlasts the previous screen must not hit a freshly shown panel
    if (now() - this._shownAt < this.pressLockMs) return false;
    const hit = p.hits[this.hoverHit];
    if (!hit) return false;
    p.setFocus(this.hoverHit);
    hit.action();
    return true;
  }

  /** Activate the focused item of the active panel (keyboard). */
  activateFocused() {
    const p = this.active;
    if (!p || p.focus < 0) return false;
    const hit = p.hits[p.focus];
    if (!hit) return false;
    hit.action();
    return true;
  }

  /** Call the active panel's onBack. Returns true if handled. */
  back() {
    const p = this.active;
    if (p && typeof p.spec.onBack === 'function') { p.spec.onBack(this.app); return true; }
    return false;
  }

  /**
   * Keyboard shortcuts (emu mode): arrows move the focus, Enter activates, Escape = back,
   * digits 1..9 pick the n-th list entry / button. Returns true when consumed.
   */
  handleKey(code) {
    const p = this.active;
    if (!p || !this.visible) return false;
    const n = p.hits.length;
    switch (code) {
      case 'ArrowDown': case 'ArrowRight':
        if (n === 0) return false;
        p.setFocus((p.focus + 1) % n);
        return true;
      case 'ArrowUp': case 'ArrowLeft':
        if (n === 0) return false;
        p.setFocus((p.focus - 1 + n) % n);
        return true;
      case 'Enter': case 'NumpadEnter':
        if (p.focus < 0 && n > 0) p.setFocus(0);
        return this.activateFocused();
      case 'Escape': case 'Backspace':
        return this.back();
      default: {
        const m = /^(?:Digit|Numpad)([1-9])$/.exec(code);
        if (m) {
          const k = Number(m[1]) - 1;
          // prefer list options, then buttons
          const options = p.hits.filter((h) => h.kind === 'option');
          const target = options.length > k ? options[k] : p.hits.filter((h) => h.kind === 'button' || h.kind === 'toggle')[k];
          if (target) { p.setFocus(p.hits.indexOf(target)); target.action(); return true; }
        }
        return false;
      }
    }
  }

  /** Per-frame: raycast the pointer/mouse against visible panels, update hover + laser, redraw. */
  update() {
    if (!this.visible) {
      this.laser.visible = false;
      this.reticle.visible = false;
      return;
    }
    let cast = false;
    if (this._pointer.active) {
      this._raycaster.set(this._pointer.origin, this._pointer.dir);
      cast = true;
    } else if (this._mouse.active && this.camera) {
      _ndc.set(this._mouse.x, this._mouse.y);
      this._raycaster.setFromCamera(_ndc, this.camera);
      cast = true;
    }
    let hoverPanel = null, hoverHit = -1;
    this.hasHit = false;
    if (cast && this._meshes.length > 0) {
      this._hits.length = 0;
      this._raycaster.intersectObjects(this._meshes, false, this._hits);
      if (this._hits.length > 0) {
        const hit = this._hits[0];
        const panel = hit.object.userData.panel;
        if (panel && hit.uv) {
          const px = hit.uv.x * panel.pxW;
          const py = (1 - hit.uv.y) * panel.pxH;
          hoverPanel = panel;
          hoverHit = panel.hitAt(px, py);
          this.hitPoint.copy(hit.point);
          this.hasHit = true;
        }
      }
    }
    if (this.hoverPanel && this.hoverPanel !== hoverPanel) this.hoverPanel.setHover(-1);
    if (hoverPanel) hoverPanel.setHover(hoverHit);
    this.hoverPanel = hoverPanel;
    this.hoverHit = hoverHit;

    // laser
    if (this._pointer.active) {
      const o = this._pointer.origin, d = this._pointer.dir;
      const len = this.hasHit ? o.distanceTo(this.hitPoint) : 3;
      const lp = this._laserPos;
      lp[0] = o.x; lp[1] = o.y; lp[2] = o.z;
      lp[3] = o.x + d.x * len; lp[4] = o.y + d.y * len; lp[5] = o.z + d.z * len;
      this.laser.geometry.attributes.position.needsUpdate = true;
      this.laser.visible = true;
      this.reticle.visible = this.hasHit;
      if (this.hasHit) this.reticle.position.copy(this.hitPoint);
    } else {
      this.laser.visible = false;
      this.reticle.visible = false;
    }

    for (const p of this.panels.values()) {
      if (p.visible && p.dirty) p.draw();
    }
  }

  dispose() {
    for (const p of this.panels.values()) p.dispose();
    this.panels.clear();
    this.laser.geometry.dispose();
    this.laser.material.dispose();
    this.reticle.geometry.dispose();
    this.reticle.material.dispose();
  }
}

const _quat = new THREE.Quaternion();
const DEFAULT_PRESS_LOCK_MS = 250;

function now() {
  return (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
}
