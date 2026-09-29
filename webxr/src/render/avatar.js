// avatar.js - avatars rendered with three.js:
//   PointAvatar      head sphere (+ nose cone showing the facing) + two hand spheres + lines,
//                    fed with a 3-point sample {head:{p,q}, left:{p}, right:{p}} in the parent's
//                    frame (stage frame S when parented under SceneKit.stage). Mirror support
//                    (x -> -x, hands swapped, yaw mirrored).
//   BodyAvatar       34-joint figure (skeleton.json joints/parents, or the choreo's fullBody
//                    joints) drawn as one InstancedMesh of cylinders (one draw call) plus a head
//                    sphere; setPose(Float32Array jointCount*3).
//   TeacherAvatar    BodyAvatar when the choreography has fullBody frames, else PointAvatar;
//                    faces the player (rotated 180 deg) for mirror choreographies.
//   ControllerMarkers  the player's own controllers as small spheres (W frame).
// No allocation in the per-frame setters. Imports three.js (browser only).

import * as THREE from 'three';

export const AVATAR_COLORS = Object.freeze({
  teacher: 0x4cc9f0,
  player: 0xffd166,
  ghost: 0xb388ff,
  opponent: 0xff6b6b,
});

const Y_AXIS = new THREE.Vector3(0, 1, 0);
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _mid = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _s = new THREE.Vector3();
const _m = new THREE.Matrix4();

// shared geometries (unit size, scaled per instance/mesh)
let _sphereGeo = null;
let _cylGeo = null;
let _coneGeo = null;
function sphereGeo() { return _sphereGeo || (_sphereGeo = new THREE.SphereGeometry(1, 14, 10)); }
function cylGeo() { return _cylGeo || (_cylGeo = new THREE.CylinderGeometry(1, 1, 1, 7, 1)); }
function coneGeo() {
  if (!_coneGeo) {
    _coneGeo = new THREE.ConeGeometry(1, 1, 8, 1);
    _coneGeo.rotateX(-Math.PI / 2);   // point along -Z (forward)
  }
  return _coneGeo;
}

function makeMaterial(color, opacity) {
  const transparent = opacity < 1;
  return new THREE.MeshLambertMaterial({
    color,
    transparent,
    opacity,
    depthWrite: !transparent,
  });
}

// ---------------------------------------------------------------------------------------------

export class PointAvatar extends THREE.Group {
  /**
   * @param {object} [opts] { color, opacity=1, headRadius=0.12, handRadius=0.055, lines=true,
   *   mirror=false, showNose=true }
   */
  constructor(opts = {}) {
    super();
    this.name = 'PointAvatar';
    this.color = opts.color !== undefined ? opts.color : AVATAR_COLORS.teacher;
    this.opacity = opts.opacity !== undefined ? opts.opacity : 1;
    this.mirror = !!opts.mirror;
    this.headRadius = opts.headRadius > 0 ? opts.headRadius : 0.12;
    this.handRadius = opts.handRadius > 0 ? opts.handRadius : 0.055;

    this.material = makeMaterial(this.color, this.opacity);
    this.head = new THREE.Mesh(sphereGeo(), this.material);
    this.head.scale.setScalar(this.headRadius);
    this.add(this.head);
    this.nose = new THREE.Mesh(coneGeo(), this.material);
    this.nose.scale.set(0.045, 0.045, 0.08);
    this.nose.position.set(0, -0.02, -this.headRadius);
    this.nose.visible = opts.showNose !== false;
    this.head.add(this.nose);
    // the nose is a child of the head: undo the head's scale so it keeps its own size
    this.nose.scale.divideScalar(this.headRadius);
    this.nose.position.divideScalar(this.headRadius);

    this.leftHand = new THREE.Mesh(sphereGeo(), this.material);
    this.leftHand.scale.setScalar(this.handRadius);
    this.rightHand = new THREE.Mesh(sphereGeo(), this.material);
    this.rightHand.scale.setScalar(this.handRadius);
    this.add(this.leftHand, this.rightHand);

    // lines: head -> left, head -> right (4 vertices)
    this.lines = null;
    if (opts.lines !== false) {
      const geo = new THREE.BufferGeometry();
      this._linePos = new Float32Array(4 * 3);
      geo.setAttribute('position', new THREE.BufferAttribute(this._linePos, 3));
      this.lineMaterial = new THREE.LineBasicMaterial({ color: this.color, transparent: this.opacity < 1, opacity: Math.min(1, this.opacity + 0.1) });
      this.lines = new THREE.LineSegments(geo, this.lineMaterial);
      this.lines.frustumCulled = false;
      this.add(this.lines);
    }
    this.hasPose = false;
  }

  /** Feed a 3-point sample (positions in this group's frame). No allocation. */
  setSample(s) {
    const hp = s.head.p, hq = s.head.q;
    const lp = this.mirror ? s.right.p : s.left.p;
    const rp = this.mirror ? s.left.p : s.right.p;
    const sx = this.mirror ? -1 : 1;
    this.head.position.set(sx * hp[0], hp[1], hp[2]);
    if (hq) {
      if (this.mirror) this.head.quaternion.set(hq[0], -hq[1], -hq[2], hq[3]);
      else this.head.quaternion.set(hq[0], hq[1], hq[2], hq[3]);
    }
    this.leftHand.position.set(sx * lp[0], lp[1], lp[2]);
    this.rightHand.position.set(sx * rp[0], rp[1], rp[2]);
    if (this.lines) {
      const p = this._linePos;
      p[0] = sx * hp[0]; p[1] = hp[1] - this.headRadius * 0.6; p[2] = hp[2];
      p[3] = sx * lp[0]; p[4] = lp[1]; p[5] = lp[2];
      p[6] = p[0]; p[7] = p[1]; p[8] = p[2];
      p[9] = sx * rp[0]; p[10] = rp[1]; p[11] = rp[2];
      this.lines.geometry.attributes.position.needsUpdate = true;
    }
    this.hasPose = true;
  }

  setColor(color) {
    this.color = color;
    this.material.color.setHex(color);
    if (this.lineMaterial) this.lineMaterial.color.setHex(color);
  }

  setOpacity(opacity) {
    this.opacity = opacity;
    this.material.opacity = opacity;
    this.material.transparent = opacity < 1;
    this.material.depthWrite = opacity >= 1;
    if (this.lineMaterial) {
      this.lineMaterial.opacity = Math.min(1, opacity + 0.1);
      this.lineMaterial.transparent = opacity < 1;
    }
  }

  dispose() {
    this.material.dispose();
    if (this.lines) { this.lines.geometry.dispose(); this.lineMaterial.dispose(); }
  }
}

// ---------------------------------------------------------------------------------------------

/** Bone radius heuristics by joint name (metres). */
function radiusForJoint(name) {
  const n = String(name || '').toLowerCase();
  if (n.includes('spine') || n === 'b_root' || n.includes('hips') || n.includes('pelvis')) return 0.075;
  if (n.includes('neck')) return 0.045;
  if (n.includes('head')) return 0.05;
  if (n.includes('upleg')) return 0.06;
  if (n.includes('leg')) return 0.05;
  if (n.includes('shoulder')) return 0.045;
  if (n.includes('forearm')) return 0.035;
  if (n.includes('arm')) return 0.042;
  if (n.includes('wrist') || n.includes('hand')) return 0.028;
  if (n.includes('foot') || n.includes('talo') || n.includes('ball') || n.includes('tarsal') || n.includes('subtalar')) return 0.03;
  return 0.035;
}

export class BodyAvatar extends THREE.Group {
  /**
   * @param {object} opts
   * @param {{joints: string[], parents: number[], boneLengths?: number[]}} opts.skeleton
   * @param {number} [opts.color]
   * @param {number} [opts.opacity=1]
   * @param {number} [opts.radiusScale=1]
   * @param {boolean} [opts.headSphere=true]
   */
  constructor(opts = {}) {
    super();
    this.name = 'BodyAvatar';
    const sk = opts.skeleton;
    if (!sk || !Array.isArray(sk.joints) || !Array.isArray(sk.parents)) throw new TypeError('BodyAvatar needs skeleton {joints, parents}');
    this.joints = sk.joints.slice();
    this.parents = sk.parents.slice();
    this.jointCount = this.joints.length;
    this.color = opts.color !== undefined ? opts.color : AVATAR_COLORS.teacher;
    this.opacity = opts.opacity !== undefined ? opts.opacity : 1;
    const radiusScale = opts.radiusScale > 0 ? opts.radiusScale : 1;

    // bones = joints with a parent (zero-length bones of the skeleton are skipped)
    this.bones = [];
    for (let j = 0; j < this.jointCount; j++) {
      const p = this.parents[j];
      if (p < 0 || p >= this.jointCount) continue;
      if (sk.boneLengths && Number.isFinite(sk.boneLengths[j]) && sk.boneLengths[j] < 0.015) continue;
      this.bones.push({ child: j, parent: p, radius: radiusForJoint(this.joints[j]) * radiusScale });
    }
    this.boneCount = this.bones.length;
    this._boneChild = new Int32Array(this.boneCount);
    this._boneParent = new Int32Array(this.boneCount);
    this._boneRadius = new Float32Array(this.boneCount);
    for (let i = 0; i < this.boneCount; i++) {
      this._boneChild[i] = this.bones[i].child;
      this._boneParent[i] = this.bones[i].parent;
      this._boneRadius[i] = this.bones[i].radius;
    }

    this.material = makeMaterial(this.color, this.opacity);
    this.instanced = new THREE.InstancedMesh(cylGeo(), this.material, Math.max(1, this.boneCount));
    this.instanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.instanced.frustumCulled = false;
    this.instanced.count = this.boneCount;
    this.add(this.instanced);

    // head sphere between b_head and b_head_null (or above the head joint)
    this.headIndex = this.joints.indexOf('b_head');
    if (this.headIndex < 0) this.headIndex = this.joints.findIndex((n) => /head/i.test(n));
    this.headTopIndex = this.joints.indexOf('b_head_null');
    this.headSphere = null;
    if (opts.headSphere !== false && this.headIndex >= 0) {
      this.headSphere = new THREE.Mesh(sphereGeo(), this.material);
      this.headSphere.scale.setScalar(0.105);
      this.add(this.headSphere);
    }
    this.hasPose = false;
    // hide until the first pose arrives
    this.instanced.visible = false;
    if (this.headSphere) this.headSphere.visible = false;
  }

  /**
   * Set the pose from flat joint positions (jointCount*3 floats, this group's frame).
   * Positions may be a Float32Array or a plain array. No allocation.
   */
  setPose(positions) {
    if (!positions || positions.length < this.jointCount * 3) return;
    const inst = this.instanced;
    for (let i = 0; i < this.boneCount; i++) {
      const c = this._boneChild[i] * 3, p = this._boneParent[i] * 3;
      _a.set(positions[p], positions[p + 1], positions[p + 2]);
      _b.set(positions[c], positions[c + 1], positions[c + 2]);
      _dir.subVectors(_b, _a);
      const len = _dir.length();
      _mid.addVectors(_a, _b).multiplyScalar(0.5);
      if (len < 1e-4) {
        _q.identity();
        _s.set(0, 0, 0);
      } else {
        _dir.multiplyScalar(1 / len);
        _q.setFromUnitVectors(Y_AXIS, _dir);
        const r = this._boneRadius[i];
        _s.set(r, len + r * 0.6, r);   // slightly longer so joints overlap (capsule look)
      }
      _m.compose(_mid, _q, _s);
      inst.setMatrixAt(i, _m);
    }
    inst.instanceMatrix.needsUpdate = true;
    if (this.headSphere) {
      const h = this.headIndex * 3;
      if (this.headTopIndex >= 0) {
        const t = this.headTopIndex * 3;
        this.headSphere.position.set(
          (positions[h] + positions[t]) * 0.5,
          (positions[h + 1] + positions[t + 1]) * 0.5,
          (positions[h + 2] + positions[t + 2]) * 0.5
        );
      } else {
        this.headSphere.position.set(positions[h], positions[h + 1] + 0.08, positions[h + 2]);
      }
    }
    if (!this.hasPose) {
      this.hasPose = true;
      inst.visible = true;
      if (this.headSphere) this.headSphere.visible = true;
    }
  }

  setColor(color) {
    this.color = color;
    this.material.color.setHex(color);
  }

  setOpacity(opacity) {
    this.opacity = opacity;
    this.material.opacity = opacity;
    this.material.transparent = opacity < 1;
    this.material.depthWrite = opacity >= 1;
  }

  dispose() {
    this.material.dispose();
    this.instanced.dispose();
  }
}

// ---------------------------------------------------------------------------------------------

export class TeacherAvatar extends THREE.Group {
  /**
   * @param {object} [opts] { color, opacity, skeleton (fallback joints/parents when the choreo
   *   has no fullBody: unused then), mode: 'auto' | 'points' }
   */
  constructor(opts = {}) {
    super();
    this.name = 'TeacherAvatar';
    this.color = opts.color !== undefined ? opts.color : AVATAR_COLORS.teacher;
    this.opacity = opts.opacity !== undefined ? opts.opacity : 1;
    this.mode = opts.mode === 'points' ? 'points' : 'auto';
    this.choreo = null;
    this.points = new PointAvatar({ color: this.color, opacity: this.opacity });
    this.add(this.points);
    this.body = null;              // built per choreography (joint layout may differ)
    this.mirror = false;
    this._facing = new THREE.Group();   // rotated 180 deg for mirror choreographies
    this.remove(this.points);
    this._facing.add(this.points);
    this.add(this._facing);
    this._fbOut = null;
  }

  /** Bind a choreography: picks the body (fullBody) or point representation. */
  setChoreo(choreo) {
    this.choreo = choreo;
    this.mirror = !!(choreo && choreo.mirror);
    // mirror choreographies face the player (the rotation makes the display a mirror image)
    this._facing.rotation.y = this.mirror ? Math.PI : 0;
    if (this.body) {
      this._facing.remove(this.body);
      this.body.dispose();
      this.body = null;
    }
    if (choreo && choreo.hasFullBody) {
      const fb = choreo.fullBody;
      this.body = new BodyAvatar({ skeleton: { joints: fb.joints, parents: fb.parents }, color: this.color, opacity: this.opacity });
      this._facing.add(this.body);
      this._fbOut = new Float32Array(fb.jointCount * 3);
    }
    this._applyMode();
  }

  /** 'auto' (body when available) or 'points'. */
  setMode(mode) {
    this.mode = mode === 'points' ? 'points' : 'auto';
    this._applyMode();
  }

  _applyMode() {
    const useBody = this.mode === 'auto' && this.body !== null;
    if (this.body) this.body.visible = useBody;
    this.points.visible = !useBody;
  }

  get usesBody() {
    return this.mode === 'auto' && this.body !== null;
  }

  /**
   * Update for game time t (seconds since beat 0). `refSample` (choreo.sampleAt(t), unmirrored)
   * may be passed to avoid a second sampling; otherwise the choreography is sampled here.
   */
  update(t, refSample = null) {
    const c = this.choreo;
    if (!c) return;
    if (this.usesBody) {
      const pose = c.fullBodyAt(t, this._fbOut);
      if (pose) this.body.setPose(pose);
    } else {
      const s = refSample || c.sampleAt(t);
      this.points.setSample(s);
    }
  }

  setColor(color) {
    this.color = color;
    this.points.setColor(color);
    if (this.body) this.body.setColor(color);
  }

  setOpacity(opacity) {
    this.opacity = opacity;
    this.points.setOpacity(opacity);
    if (this.body) this.body.setOpacity(opacity);
  }

  dispose() {
    this.points.dispose();
    if (this.body) this.body.dispose();
  }
}

// ---------------------------------------------------------------------------------------------

/** The player's own controllers (and optionally head) as small spheres, W frame. */
export class ControllerMarkers extends THREE.Group {
  constructor(opts = {}) {
    super();
    this.name = 'ControllerMarkers';
    this.color = opts.color !== undefined ? opts.color : AVATAR_COLORS.player;
    this.material = makeMaterial(this.color, opts.opacity !== undefined ? opts.opacity : 1);
    const r = opts.radius > 0 ? opts.radius : 0.035;
    this.left = new THREE.Mesh(sphereGeo(), this.material);
    this.left.scale.setScalar(r);
    this.right = new THREE.Mesh(sphereGeo(), this.material);
    this.right.scale.setScalar(r);
    this.head = new THREE.Mesh(sphereGeo(), this.material);
    this.head.scale.setScalar(0.11);
    this.head.visible = false;
    this.add(this.left, this.right, this.head);
  }

  /** Update from a W-frame input sample; `showHead` for third-person views. */
  setSample(s, showHead = false) {
    this.left.visible = s.left.valid !== false;
    this.right.visible = s.right.valid !== false;
    this.left.position.set(s.left.p[0], s.left.p[1], s.left.p[2]);
    this.right.position.set(s.right.p[0], s.right.p[1], s.right.p[2]);
    this.head.visible = showHead && s.head.valid !== false;
    if (showHead) {
      this.head.position.set(s.head.p[0], s.head.p[1], s.head.p[2]);
      this.head.quaternion.set(s.head.q[0], s.head.q[1], s.head.q[2], s.head.q[3]);
    }
  }

  setColor(color) {
    this.color = color;
    this.material.color.setHex(color);
  }

  dispose() {
    this.material.dispose();
  }
}
