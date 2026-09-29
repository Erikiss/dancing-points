// mirror.js - PlayerMirror: the player's own dancer shown as a mirror image on the stage
// (next to the teacher), either as a full body (neural avatar pose, BodyAvatar) or as the three
// tracked points (PointAvatar). A real mirror standing halfway between the player and the stage
// reflects the player's stage-frame point p = (x, y, z) to (x, y, -z) in this group's frame,
// which the App places at S (-STAGE.duoSideOffset, 0, -STAGE.teacherDistance): the image faces
// the player, and moves to the player's right when the player moves right, exactly like a
// dance-studio mirror. Hands keep their side (a mirror does not swap them). Head orientation:
// reflection through the XY plane followed by a 180 degree yaw so the nose points at the player.
// Imports three.js (render module, browser only); no allocation in the per-frame setters.

import * as THREE from 'three';
import { PointAvatar, BodyAvatar, AVATAR_COLORS } from './avatar.js';
import { createChoreoSample } from '../game/choreo.js';

export const MIRROR_MODES = Object.freeze(['body', 'points', 'off']);

export class PlayerMirror extends THREE.Group {
  /**
   * @param {object} [opts] { color=AVATAR_COLORS.player, opacity=0.85, skeleton ({joints, parents,
   *   boneLengths?}) for the body representation, mode='points' }
   */
  constructor(opts = {}) {
    super();
    this.name = 'PlayerMirror';
    this.color = opts.color !== undefined ? opts.color : AVATAR_COLORS.player;
    this.opacity = opts.opacity !== undefined ? opts.opacity : 0.85;
    this.mode = MIRROR_MODES.includes(opts.mode) ? opts.mode : 'points';
    this.points = new PointAvatar({ color: this.color, opacity: this.opacity });
    this.add(this.points);
    this.body = null;
    this._bodyPose = null;
    this._sample = createChoreoSample();
    this.hasPose = false;
    if (opts.skeleton) this.setSkeleton(opts.skeleton);
    this._applyMode();
  }

  /** (Re)build the body representation for a skeleton {joints, parents, boneLengths?}. */
  setSkeleton(skeleton) {
    if (this.body) {
      this.remove(this.body);
      this.body.dispose();
      this.body = null;
      this._bodyPose = null;
    }
    if (!skeleton || !Array.isArray(skeleton.joints) || !Array.isArray(skeleton.parents)) return null;
    this.body = new BodyAvatar({ skeleton, color: this.color, opacity: this.opacity });
    this._bodyPose = new Float32Array(skeleton.joints.length * 3);
    this.add(this.body);
    this._applyMode();
    return this.body;
  }

  /** 'body' (needs a skeleton and body poses), 'points' or 'off'. */
  setMode(mode) {
    if (!MIRROR_MODES.includes(mode)) return;
    this.mode = mode;
    this._applyMode();
  }

  _applyMode() {
    const useBody = this.mode === 'body' && this.body !== null;
    if (this.body) this.body.visible = useBody && this.body.hasPose;
    this.points.visible = this.mode === 'points' || (this.mode === 'body' && !useBody);
    this.visible = this.mode !== 'off';
  }

  get usesBody() {
    return this.mode === 'body' && this.body !== null && this.body.hasPose;
  }

  /**
   * Mirror the player's stage-frame 3-point sample ({head:{p,q}, left:{p}, right:{p}}) into the
   * point representation. No allocation.
   */
  setPointSample(s) {
    const m = this._sample;
    const hp = s.head.p, hq = s.head.q, lp = s.left.p, rp = s.right.p;
    m.head.p[0] = hp[0]; m.head.p[1] = hp[1]; m.head.p[2] = -hp[2];
    // q' = (M R M) * R_y(180deg) with M = diag(1, 1, -1): (x, y, z, w) -> (-z, w, -x, y)
    m.head.q[0] = -hq[2]; m.head.q[1] = hq[3]; m.head.q[2] = -hq[0]; m.head.q[3] = hq[1];
    m.left.p[0] = lp[0]; m.left.p[1] = lp[1]; m.left.p[2] = -lp[2];
    m.right.p[0] = rp[0]; m.right.p[1] = rp[1]; m.right.p[2] = -rp[2];
    this.points.setSample(m);
    this.hasPose = true;
    if (this.mode === 'points' || (this.mode === 'body' && !(this.body && this.body.hasPose))) {
      this.points.visible = true;
      if (this.body) this.body.visible = false;
    }
  }

  /**
   * Mirror full-body joint positions (jointCount*3, stage frame at reference height) into the
   * body representation. Ignored without a skeleton. No allocation.
   */
  setBodyPose(positions) {
    const out = this._bodyPose;
    if (!out || !positions || positions.length < out.length) return false;
    for (let i = 0; i < out.length; i += 3) {
      out[i] = positions[i];
      out[i + 1] = positions[i + 1];
      out[i + 2] = -positions[i + 2];
    }
    this.body.setPose(out);
    this.hasPose = true;
    if (this.mode === 'body') {
      this.body.visible = true;
      this.points.visible = false;
    }
    return true;
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
