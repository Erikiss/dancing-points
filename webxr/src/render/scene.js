// scene.js - SceneKit: the three.js WebGLRenderer (XR enabled, 'local-floor'), camera, sky
// gradient, floor with the marked calibration spot, the stage (platform + screen wall 2.5 m in
// front of the player), lights, resize handling and the XR controller objects. Everything that
// lives in the stage frame S is parented under `stage`, whose matrix is set from
// `Calibration.stageToWorldMatrix`. Quest 2 settings: pixel ratio 1, no shadows, foveation 1.
// Imports three.js; only exercised in the browser (never imported by unit tests).

import * as THREE from 'three';
import { STAGE } from '../config.js';

const UP = new THREE.Vector3(0, 1, 0);

export const SCENE_COLORS = Object.freeze({
  skyTop: 0x0a0c1e,
  skyHorizon: 0x35245a,
  skyBottom: 0x120c20,
  floor: 0x1a1f33,
  grid: 0x3a4468,
  marker: 0x4cc9f0,
  screen: 0x0f1424,
  screenFrame: 0x6a7fff,
  platform: 0x232a48,
});

export class SceneKit {
  /**
   * @param {object} [opts]
   * @param {HTMLElement} [opts.container=document.body]  the canvas is appended here
   * @param {HTMLCanvasElement} [opts.canvas]
   * @param {boolean} [opts.xr=true]  enable the WebXR manager
   * @param {number} [opts.pixelRatio=1]
   * @param {number} [opts.floorSize=STAGE.floorSize]
   * @param {number} [opts.teacherDistance=STAGE.teacherDistance]
   */
  constructor(opts = {}) {
    const doc = opts.document || globalThis.document;
    this.container = opts.container || (doc ? doc.body : null);
    this.teacherDistance = opts.teacherDistance > 0 ? opts.teacherDistance : STAGE.teacherDistance;
    this.floorSize = opts.floorSize > 0 ? opts.floorSize : STAGE.floorSize;
    this.screenDistance = this.teacherDistance + 0.7;   // the stage wall/screen behind the teacher
    this.screenWidth = 4.4;
    this.screenHeight = 2.6;

    const renderer = new THREE.WebGLRenderer({
      canvas: opts.canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
      stencil: false,
    });
    renderer.setPixelRatio(opts.pixelRatio > 0 ? opts.pixelRatio : 1);
    renderer.shadowMap.enabled = false;
    renderer.xr.enabled = opts.xr !== false;
    renderer.xr.setReferenceSpaceType('local-floor');
    renderer.xr.setFoveation(1);
    renderer.xr.setFramebufferScaleFactor(1);
    renderer.setClearColor(SCENE_COLORS.skyTop, 1);
    this.renderer = renderer;
    if (this.container && !opts.canvas) this.container.appendChild(renderer.domElement);
    renderer.domElement.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      this.contextLost = true;
      console.warn('[scene] WebGL context lost');
      if (typeof this.onContextLost === 'function') this.onContextLost();
    });
    // three.js cannot rebuild every GPU resource after a loss (memory pressure when the Quest
    // browser is backgrounded); the App reloads the page instead of showing a black canvas
    renderer.domElement.addEventListener('webglcontextrestored', () => {
      console.warn('[scene] WebGL context restored');
      if (typeof this.onContextRestored === 'function') this.onContextRestored();
    });
    this.contextLost = false;
    this.onContextLost = null;
    this.onContextRestored = null;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(70, 1, 0.05, 80);
    this.camera.position.set(0, 1.6, 0);
    this.scene.add(this.camera);

    // stage group: S -> W (identity until calibrated)
    this.stage = new THREE.Group();
    this.stage.name = 'stage';
    this.stage.matrixAutoUpdate = false;
    this.scene.add(this.stage);

    this._buildSky();
    this._buildFloor();
    this._buildStage();
    this._buildLights();

    // XR controllers (three.js tracks targetRay + grip spaces for us)
    this.controllers = { left: null, right: null };
    this._controllerObjects = [];
    if (renderer.xr.enabled) this._setupControllers();

    this._onResize = () => this.resize();
    if (typeof window !== 'undefined') window.addEventListener('resize', this._onResize);
    this.resize();

    this.xrSession = null;
    this.onSessionEnd = null;
    this.onReferenceSpaceReset = null;   // Oculus-button recentre: the local-floor frame moved
    this.onVisibilityChange = null;      // session.visibilityState: 'visible' | 'visible-blurred' | 'hidden'
    this._onSessionEndEvent = () => {
      const s = this.xrSession;
      this.xrSession = null;
      if (typeof this.onSessionEnd === 'function') this.onSessionEnd(s);
    };
    this._onResetEvent = (ev) => {
      if (typeof this.onReferenceSpaceReset === 'function') this.onReferenceSpaceReset(ev);
    };
    this._onVisibilityEvent = () => {
      const s = this.xrSession;
      if (s && typeof this.onVisibilityChange === 'function') this.onVisibilityChange(s.visibilityState);
    };
  }

  // ---- construction --------------------------------------------------------------------------

  _buildSky() {
    const geo = new THREE.SphereGeometry(45, 24, 12);
    const pos = geo.attributes.position;
    const colors = new Float32Array(pos.count * 3);
    const top = new THREE.Color(SCENE_COLORS.skyTop);
    const hor = new THREE.Color(SCENE_COLORS.skyHorizon);
    const bot = new THREE.Color(SCENE_COLORS.skyBottom);
    const c = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const y = pos.getY(i) / 45;      // -1..1
      if (y >= 0) c.copy(hor).lerp(top, Math.pow(y, 0.6));
      else c.copy(hor).lerp(bot, Math.pow(-y, 0.5));
      colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, depthWrite: false, fog: false });
    this.sky = new THREE.Mesh(geo, mat);
    this.sky.name = 'sky';
    this.sky.renderOrder = -10;
    this.scene.add(this.sky);
  }

  _buildFloor() {
    const size = this.floorSize;
    const tex = makeGridTexture(512, SCENE_COLORS.floor, SCENE_COLORS.grid);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(size, size);
    const geo = new THREE.PlaneGeometry(size, size);
    geo.rotateX(-Math.PI / 2);
    const mat = new THREE.MeshLambertMaterial({ map: tex });
    this.floor = new THREE.Mesh(geo, mat);
    this.floor.name = 'floor';
    this.stage.add(this.floor);

    // calibration spot: ring + forward arrow at the stage origin
    const marker = new THREE.Group();
    marker.name = 'marker';
    const ringMat = new THREE.MeshBasicMaterial({ color: SCENE_COLORS.marker, transparent: true, opacity: 0.9 });
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.26, 0.32, 40), ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.004;
    marker.add(ring);
    const inner = new THREE.Mesh(new THREE.CircleGeometry(0.06, 20), ringMat);
    inner.rotation.x = -Math.PI / 2;
    inner.position.y = 0.004;
    marker.add(inner);
    // arrow (triangle) pointing to -Z (towards the stage)
    const tri = new THREE.Shape();
    tri.moveTo(-0.09, 0); tri.lineTo(0.09, 0); tri.lineTo(0, 0.2); tri.closePath();
    const arrow = new THREE.Mesh(new THREE.ShapeGeometry(tri), ringMat);
    arrow.rotation.x = -Math.PI / 2;
    arrow.rotation.z = 0;               // shape +y maps to -z after the rotation
    arrow.position.set(0, 0.004, -0.40);
    marker.add(arrow);
    // two foot outlines
    const footMat = new THREE.MeshBasicMaterial({ color: SCENE_COLORS.marker, transparent: true, opacity: 0.35 });
    for (const sx of [-1, 1]) {
      const foot = new THREE.Mesh(new THREE.PlaneGeometry(0.09, 0.24), footMat);
      foot.rotation.x = -Math.PI / 2;
      foot.position.set(sx * 0.11, 0.003, 0);
      marker.add(foot);
    }
    this.marker = marker;
    this.markerMaterial = ringMat;
    this.stage.add(marker);
  }

  _buildStage() {
    const d = this.teacherDistance;
    // platform under the teacher
    const platGeo = new THREE.BoxGeometry(2.4, 0.06, 1.6);
    const platMat = new THREE.MeshLambertMaterial({ color: SCENE_COLORS.platform });
    this.platform = new THREE.Mesh(platGeo, platMat);
    this.platform.position.set(0, 0.03, -d);
    this.platform.name = 'platform';
    this.stage.add(this.platform);
    const edge = new THREE.LineSegments(new THREE.EdgesGeometry(platGeo), new THREE.LineBasicMaterial({ color: SCENE_COLORS.screenFrame }));
    edge.position.copy(this.platform.position);
    this.stage.add(edge);

    // screen wall behind the teacher (the HUD is placed in front of it)
    const sw = this.screenWidth, sh = this.screenHeight;
    const screenGeo = new THREE.PlaneGeometry(sw, sh);
    const screenMat = new THREE.MeshBasicMaterial({ color: SCENE_COLORS.screen });
    this.screen = new THREE.Mesh(screenGeo, screenMat);
    this.screen.position.set(0, sh / 2 + 0.02, -this.screenDistance);
    this.screen.name = 'screen';
    this.stage.add(this.screen);
    const frame = new THREE.LineSegments(new THREE.EdgesGeometry(screenGeo), new THREE.LineBasicMaterial({ color: SCENE_COLORS.screenFrame }));
    frame.position.copy(this.screen.position);
    frame.position.z += 0.005;
    this.stage.add(frame);
    // side pillars for a bit of depth
    const pillarGeo = new THREE.BoxGeometry(0.12, sh, 0.12);
    const pillarMat = new THREE.MeshLambertMaterial({ color: 0x2a3358 });
    for (const sx of [-1, 1]) {
      const p = new THREE.Mesh(pillarGeo, pillarMat);
      p.position.set(sx * (sw / 2 + 0.1), sh / 2, -this.screenDistance + 0.1);
      this.stage.add(p);
    }
  }

  _buildLights() {
    this.hemi = new THREE.HemisphereLight(0xcfe0ff, 0x2a1e3a, 1.1);
    this.scene.add(this.hemi);
    this.key = new THREE.DirectionalLight(0xffffff, 1.4);
    this.key.position.set(2.5, 4, 3);
    this.scene.add(this.key);
    this.fill = new THREE.DirectionalLight(0x8090ff, 0.5);
    this.fill.position.set(-3, 2, -2);
    this.scene.add(this.fill);
  }

  _setupControllers() {
    const xr = this.renderer.xr;
    for (let i = 0; i < 2; i++) {
      const ray = xr.getController(i);
      const grip = xr.getControllerGrip(i);
      ray.name = `controller-ray-${i}`;
      grip.name = `controller-grip-${i}`;
      ray.addEventListener('connected', (ev) => {
        const src = ev.data;
        const hand = src && (src.handedness === 'left' || src.handedness === 'right') ? src.handedness : null;
        if (!hand) return;
        this.controllers[hand] = { ray, grip, source: src, index: i, hand };
      });
      ray.addEventListener('disconnected', () => {
        for (const hand of ['left', 'right']) {
          if (this.controllers[hand] && this.controllers[hand].ray === ray) this.controllers[hand] = null;
        }
      });
      this.scene.add(ray);
      this.scene.add(grip);
      this._controllerObjects.push(ray, grip);
    }
  }

  // ---- stage transform -----------------------------------------------------------------------

  /** Set the S -> W matrix (column-major 16 floats, e.g. from Calibration.stageToWorldMatrix). */
  setStageMatrix(array16) {
    this.stage.matrix.fromArray(array16);
    this.stage.matrixWorldNeedsUpdate = true;
  }

  /** Reset the stage to the identity (uncalibrated). */
  resetStage() {
    this.stage.matrix.identity();
    this.stage.matrixWorldNeedsUpdate = true;
  }

  /** Position (stage frame) of the HUD/screen plane: use it to place panels on the wall. */
  get screenZ() {
    return -this.screenDistance;
  }

  /** Highlight the calibration marker (pulsing while calibrating). */
  setMarkerHighlight(on, phase = 0) {
    const m = this.markerMaterial;
    if (!m) return;
    m.opacity = on ? 0.6 + 0.4 * Math.abs(Math.sin(phase)) : 0.9;
    m.color.setHex(on ? 0xffd166 : SCENE_COLORS.marker);
  }

  // ---- loop / XR -----------------------------------------------------------------------------

  setAnimationLoop(fn) {
    this.renderer.setAnimationLoop(fn);
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  /** Attach a WebXR session to the renderer. Resolves to the reference space. */
  async startXR(session) {
    this.xrSession = session;
    session.addEventListener('end', this._onSessionEndEvent);
    session.addEventListener('visibilitychange', this._onVisibilityEvent);
    await this.renderer.xr.setSession(session);
    const ref = this.renderer.xr.getReferenceSpace();
    // a recentre (long press on the Oculus button) changes origin + yaw of 'local-floor'; the
    // persisted calibration is then wrong and must be redone (App.onReferenceSpaceReset)
    if (ref && typeof ref.addEventListener === 'function') ref.addEventListener('reset', this._onResetEvent);
    return ref;
  }

  get referenceSpace() {
    return this.renderer.xr.enabled ? this.renderer.xr.getReferenceSpace() : null;
  }

  get presenting() {
    return this.renderer.xr.enabled && this.renderer.xr.isPresenting;
  }

  async endXR() {
    const s = this.xrSession;
    if (!s) return;
    try { await s.end(); } catch (e) { /* already ended */ }
  }

  resize() {
    if (this.renderer.xr && this.renderer.xr.isPresenting) return;
    const w = typeof window !== 'undefined' ? window.innerWidth : 1280;
    const h = typeof window !== 'undefined' ? window.innerHeight : 720;
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, true);
  }

  /** Place the (non-XR) camera from a W-frame head pose {p:[x,y,z], q:[x,y,z,w]}. */
  setCameraFromPose(pose) {
    const c = this.camera;
    c.position.set(pose.p[0], pose.p[1], pose.p[2]);
    c.quaternion.set(pose.q[0], pose.q[1], pose.q[2], pose.q[3]);
  }

  /** Third-person camera (desktop/screenshots): behind and above the player looking at the stage. */
  setThirdPersonCamera() {
    const c = this.camera;
    c.position.set(1.6, 2.1, 2.6);
    _lookTarget.set(0, 1.1, -1.6);
    c.up.copy(UP);
    c.lookAt(_lookTarget);
  }

  dispose() {
    if (typeof window !== 'undefined') window.removeEventListener('resize', this._onResize);
    this.renderer.setAnimationLoop(null);
    this.renderer.dispose();
  }
}

const _lookTarget = new THREE.Vector3();

/** A tiling grid texture (dark tile with a lighter line) drawn on a canvas. */
export function makeGridTexture(size = 512, fill = 0x1a1f33, line = 0x3a4468) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#' + fill.toString(16).padStart(6, '0');
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = '#' + line.toString(16).padStart(6, '0');
  ctx.lineWidth = Math.max(2, size / 128);
  ctx.strokeRect(ctx.lineWidth / 2, ctx.lineWidth / 2, size - ctx.lineWidth, size - ctx.lineWidth);
  ctx.globalAlpha = 0.35;
  ctx.lineWidth = Math.max(1, size / 256);
  ctx.beginPath();
  ctx.moveTo(size / 2, 0); ctx.lineTo(size / 2, size);
  ctx.moveTo(0, size / 2); ctx.lineTo(size, size / 2);
  ctx.stroke();
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}
