// scoring.js - Scorer: pure, deterministic Just-Dance-style scoring of a player's stage-frame
// 3-point stream against a choreography (DESIGN.md section 7). Reference features are
// precomputed once per choreography into typed arrays; `scoreTick` does no allocation and
// runs in well under 0.3 ms. Emits per-beat grades through an optional callback. No three.js,
// no DOM.

import { SCORING } from '../config.js';

export const GRADE_IDS = Object.freeze(['daneben', 'ok', 'gut', 'perfekt']);

/** Grade id for a 0..1 score. */
export function gradeFor(score) {
  if (score >= SCORING.gradePerfekt) return 'perfekt';
  if (score >= SCORING.gradeGut) return 'gut';
  if (score >= SCORING.gradeOk) return 'ok';
  return 'daneben';
}

/** Stars (1..5) for a 0..100 score. */
export function starsFor(score100) {
  const s = SCORING.stars;
  if (score100 >= s[0]) return 5;
  if (score100 >= s[1]) return 4;
  if (score100 >= s[2]) return 3;
  if (score100 >= s[3]) return 2;
  return 1;
}

/**
 * Precompute the reference feature arrays of a choreography (optionally mirrored):
 * head positions, hand positions relative to the head, 2-tap smoothed velocities and the mean
 * 3-point speed per frame. Exported for tests and the duo benchmark.
 */
export function precomputeReference(choreo, mirror = choreo.mirror) {
  const n = choreo.frameCount;
  const fps = choreo.fps;
  const head = new Float32Array(n * 3);
  const leftRel = new Float32Array(n * 3);
  const rightRel = new Float32Array(n * 3);
  const vel = new Float32Array(n * 9);     // head, left, right
  const speed = new Float32Array(n);
  const sx = mirror ? -1 : 1;
  const srcLeft = mirror ? choreo.rightP : choreo.leftP;
  const srcRight = mirror ? choreo.leftP : choreo.rightP;
  const hp = choreo.headP;
  for (let i = 0; i < n; i++) {
    const o = i * 3;
    head[o] = sx * hp[o]; head[o + 1] = hp[o + 1]; head[o + 2] = hp[o + 2];
    leftRel[o] = sx * srcLeft[o] - head[o];
    leftRel[o + 1] = srcLeft[o + 1] - head[o + 1];
    leftRel[o + 2] = srcLeft[o + 2] - head[o + 2];
    rightRel[o] = sx * srcRight[o] - head[o];
    rightRel[o + 1] = srcRight[o + 1] - head[o + 1];
    rightRel[o + 2] = srcRight[o + 2] - head[o + 2];
  }
  // absolute (mirrored) positions of the three points, 9 per frame: head, left, right
  const absPos = new Float32Array(n * 9);
  for (let i = 0; i < n; i++) {
    const o = i * 3, a = i * 9;
    absPos[a] = head[o]; absPos[a + 1] = head[o + 1]; absPos[a + 2] = head[o + 2];
    absPos[a + 3] = sx * srcLeft[o]; absPos[a + 4] = srcLeft[o + 1]; absPos[a + 5] = srcLeft[o + 2];
    absPos[a + 6] = sx * srcRight[o]; absPos[a + 7] = srcRight[o + 1]; absPos[a + 8] = srcRight[o + 2];
  }
  // velocities: v_i = (p_i - p_{i-1}) * fps, smoothed with a 2-tap average -> (p_i - p_{i-2}) * fps / 2
  for (let i = 0; i < n; i++) {
    const i0 = i >= 2 ? i - 2 : 0;
    const steps = i - i0;
    const scale = steps > 0 ? fps / steps : 0;
    let sum = 0;
    for (let c = 0; c < 9; c++) {
      vel[i * 9 + c] = (absPos[i * 9 + c] - absPos[i0 * 9 + c]) * scale;
    }
    for (let p = 0; p < 3; p++) {
      const vx = vel[i * 9 + p * 3], vy = vel[i * 9 + p * 3 + 1], vz = vel[i * 9 + p * 3 + 2];
      sum += Math.sqrt(vx * vx + vy * vy + vz * vz);
    }
    speed[i] = sum / 3;
  }
  // frames 0 and 1 have no history: copy frame 2's velocity so the start is not penalised
  if (n > 2) {
    for (let i = 0; i < 2; i++) {
      for (let c = 0; c < 9; c++) vel[i * 9 + c] = vel[18 + c];
      speed[i] = speed[2];
    }
  }
  return { n, fps, head, leftRel, rightRel, vel, speed, mirror };
}

export class Scorer {
  /**
   * @param {import('./choreo.js').Choreo} choreo
   * @param {object} [opts] { mirror (default choreo.mirror), onBeat(event), sigma overrides }
   */
  constructor(choreo, opts = {}) {
    this.choreo = choreo;
    this.mirror = opts.mirror !== undefined ? !!opts.mirror : choreo.mirror;
    this.onBeat = typeof opts.onBeat === 'function' ? opts.onBeat : null;
    this.sigmaHead = opts.sigmaHead || SCORING.sigmaHead;
    this.sigmaHand = opts.sigmaHand || SCORING.sigmaHand;
    this.sigmaVel = opts.sigmaVel || SCORING.sigmaVel;
    this.deltaSteps = Math.round((opts.deltaMaxSec !== undefined ? opts.deltaMaxSec : SCORING.deltaMaxSec) * choreo.fps);
    this.ref = precomputeReference(choreo, this.mirror);
    this.fps = choreo.fps;
    this.beatDuration = choreo.beatDuration;
    this.duration = choreo.duration;
    this.beatCount = Math.ceil(choreo.durationBeats - 1e-9);
    this.moveCount = choreo.moves.length;
    this._moveStartBeat = new Float64Array(this.moveCount);
    this._moveEndBeat = new Float64Array(this.moveCount);
    for (let i = 0; i < this.moveCount; i++) {
      this._moveStartBeat[i] = choreo.moves[i].startBeat;
      this._moveEndBeat[i] = choreo.moves[i].endBeat;
    }
    this._invSigHead2 = 1 / (this.sigmaHead * this.sigmaHead);
    this._invSigHand2 = 1 / (this.sigmaHand * this.sigmaHand);
    this._invSigVel2 = 1 / (this.sigmaVel * this.sigmaVel);
    // accumulators
    this.moveSum = new Float64Array(this.moveCount);
    this.moveTicks = new Uint32Array(this.moveCount);
    this.movePlayerSpeedSq = new Float64Array(this.moveCount);
    this.moveRefSpeedSq = new Float64Array(this.moveCount);
    this.perBeat = new Float32Array(this.beatCount);
    this.perBeatScored = new Uint8Array(this.beatCount);
    this._prevP = new Float64Array(9);   // previous player positions (head, left, right)
    this._prevV = new Float64Array(9);   // previous raw velocity
    this._curV = new Float64Array(9);    // current smoothed velocity
    this.reset();
  }

  /** Reset all accumulators (a new run of the same choreography). */
  reset() {
    this.moveSum.fill(0);
    this.moveTicks.fill(0);
    this.movePlayerSpeedSq.fill(0);
    this.moveRefSpeedSq.fill(0);
    this.perBeat.fill(0);
    this.perBeatScored.fill(0);
    this._prevP.fill(0);
    this._prevV.fill(0);
    this._curV.fill(0);
    this._prevT = NaN;
    this._historyTicks = 0;       // ticks with velocity history
    this.ticks = 0;
    this.frameSum = 0;
    this.lastFrameScore = 0;
    this.lastDelta = 0;
    this.deltaSum = 0;
    this.deltaCount = 0;
    this.beatSum = 0;
    this.beatTicks = 0;
    this.currentBeat = -1;
    this.combo = 0;
    this.maxCombo = 0;
    this.lastBeatEvent = null;
    this.finalized = false;
    this.result = null;
    this.lastMoveIndex = -1;
  }

  /**
   * Score one tick. `sampleS` = player sample in stage frame at reference height
   * ({head:{p}, left:{p}, right:{p}}), `t` = seconds since beat 0 (game time).
   * Returns the frame score (0..1) or -1 when t is outside the playable range (ignored).
   */
  scoreTick(sampleS, t) {
    if (this.finalized) return -1;
    if (!(t >= 0) || t > this.duration) return -1;
    const beatF = t / this.beatDuration;
    const beatIdx = Math.floor(beatF);
    const moveIdx = this._moveIndexForBeat(beatF);
    if (moveIdx < 0) return -1;

    const hp = sampleS.head.p, lp = sampleS.left.p, rp = sampleS.right.p;
    const prevP = this._prevP, prevV = this._prevV, curV = this._curV;

    // --- player velocity (finite difference, 2-tap smoothed)
    let dt = t - this._prevT;
    const hasHistory = Number.isFinite(dt) && dt > 1e-4;
    if (hasHistory) {
      if (dt < 0.25 / this.fps) dt = 0.25 / this.fps;
      const inv = 1 / dt;
      const vh0 = (hp[0] - prevP[0]) * inv, vh1 = (hp[1] - prevP[1]) * inv, vh2 = (hp[2] - prevP[2]) * inv;
      const vl0 = (lp[0] - prevP[3]) * inv, vl1 = (lp[1] - prevP[4]) * inv, vl2 = (lp[2] - prevP[5]) * inv;
      const vr0 = (rp[0] - prevP[6]) * inv, vr1 = (rp[1] - prevP[7]) * inv, vr2 = (rp[2] - prevP[8]) * inv;
      if (this._historyTicks >= 1) {
        curV[0] = 0.5 * (vh0 + prevV[0]); curV[1] = 0.5 * (vh1 + prevV[1]); curV[2] = 0.5 * (vh2 + prevV[2]);
        curV[3] = 0.5 * (vl0 + prevV[3]); curV[4] = 0.5 * (vl1 + prevV[4]); curV[5] = 0.5 * (vl2 + prevV[5]);
        curV[6] = 0.5 * (vr0 + prevV[6]); curV[7] = 0.5 * (vr1 + prevV[7]); curV[8] = 0.5 * (vr2 + prevV[8]);
      } else {
        curV[0] = vh0; curV[1] = vh1; curV[2] = vh2;
        curV[3] = vl0; curV[4] = vl1; curV[5] = vl2;
        curV[6] = vr0; curV[7] = vr1; curV[8] = vr2;
      }
      prevV[0] = vh0; prevV[1] = vh1; prevV[2] = vh2;
      prevV[3] = vl0; prevV[4] = vl1; prevV[5] = vl2;
      prevV[6] = vr0; prevV[7] = vr1; prevV[8] = vr2;
      this._historyTicks++;
    }
    prevP[0] = hp[0]; prevP[1] = hp[1]; prevP[2] = hp[2];
    prevP[3] = lp[0]; prevP[4] = lp[1]; prevP[5] = lp[2];
    prevP[6] = rp[0]; prevP[7] = rp[1]; prevP[8] = rp[2];
    this._prevT = t;
    const useVel = this._historyTicks >= 2;

    // hand positions relative to the head
    const lr0 = lp[0] - hp[0], lr1 = lp[1] - hp[1], lr2 = lp[2] - hp[2];
    const rr0 = rp[0] - hp[0], rr1 = rp[1] - hp[1], rr2 = rp[2] - hp[2];

    // --- best reference frame within +/- deltaSteps
    const ref = this.ref;
    const n = ref.n;
    const base = Math.round(t * this.fps);
    let best = -1, bestK = 0;
    const kMin = -this.deltaSteps, kMax = this.deltaSteps;
    for (let k = kMin; k <= kMax; k++) {
      let i = base + k;
      if (i < 0 || i >= n) continue;
      const o = i * 3;
      const rh = ref.head, rl = ref.leftRel, rr = ref.rightRel;
      let dx = hp[0] - rh[o], dy = hp[1] - rh[o + 1], dz = hp[2] - rh[o + 2];
      const dHead2 = dx * dx + dy * dy + dz * dz;
      dx = lr0 - rl[o]; dy = lr1 - rl[o + 1]; dz = lr2 - rl[o + 2];
      const dLeft2 = dx * dx + dy * dy + dz * dz;
      dx = rr0 - rr[o]; dy = rr1 - rr[o + 1]; dz = rr2 - rr[o + 2];
      const dRight2 = dx * dx + dy * dy + dz * dz;
      const gHead = Math.exp(-dHead2 * this._invSigHead2);
      const gLeft = Math.exp(-dLeft2 * this._invSigHand2);
      const gRight = Math.exp(-dRight2 * this._invSigHand2);
      let gVel;
      if (useVel) {
        const ov = i * 9;
        const rv = ref.vel;
        let dVel = 0;
        for (let p = 0; p < 3; p++) {
          const q = p * 3;
          const ax = curV[q] - rv[ov + q], ay = curV[q + 1] - rv[ov + q + 1], az = curV[q + 2] - rv[ov + q + 2];
          dVel += Math.sqrt(ax * ax + ay * ay + az * az);
        }
        dVel /= 3;
        gVel = Math.exp(-dVel * dVel * this._invSigVel2);
      } else {
        gVel = (gHead + gLeft + gRight) / 3;
      }
      const s = SCORING.weightHead * gHead + SCORING.weightLeft * gLeft + SCORING.weightRight * gRight + SCORING.weightVel * gVel;
      if (s > best) { best = s; bestK = k; }
    }
    if (best < 0) return -1;

    // --- accumulate
    this.lastFrameScore = best;
    this.lastDelta = bestK / this.fps;
    this.ticks++;
    this.frameSum += best;
    if (best > 0.05) {       // only count timing when the match is meaningful
      this.deltaSum += this.lastDelta;
      this.deltaCount++;
    }
    this.moveSum[moveIdx] += best;
    this.moveTicks[moveIdx]++;
    if (useVel) {
      let ps = 0;
      for (let p = 0; p < 3; p++) {
        const q = p * 3;
        ps += Math.sqrt(curV[q] * curV[q] + curV[q + 1] * curV[q + 1] + curV[q + 2] * curV[q + 2]);
      }
      ps /= 3;
      this.movePlayerSpeedSq[moveIdx] += ps * ps;
      const bi = base < 0 ? 0 : (base >= n ? n - 1 : base);
      const rs = ref.speed[bi];
      this.moveRefSpeedSq[moveIdx] += rs * rs;
    }
    this.lastMoveIndex = moveIdx;

    // --- beats
    if (beatIdx !== this.currentBeat) {
      if (this.currentBeat >= 0) this._closeBeat();
      this.currentBeat = beatIdx;
      this.beatSum = 0;
      this.beatTicks = 0;
    }
    this.beatSum += best;
    this.beatTicks++;
    return best;
  }

  _moveIndexForBeat(beatF) {
    const s = this._moveStartBeat, e = this._moveEndBeat;
    // fast path: same move as last tick
    const li = this.lastMoveIndex;
    if (li >= 0 && beatF >= s[li] && beatF < e[li]) return li;
    for (let i = 0; i < this.moveCount; i++) {
      if (beatF >= s[i] && beatF < e[i]) return i;
    }
    // exactly at the end of the choreography -> last move
    if (this.moveCount > 0 && Math.abs(beatF - e[this.moveCount - 1]) < 1e-6) return this.moveCount - 1;
    return -1;
  }

  _closeBeat() {
    const b = this.currentBeat;
    if (b < 0 || b >= this.beatCount || this.beatTicks === 0) return;
    const score = this.beatSum / this.beatTicks;
    this.perBeat[b] = score;
    this.perBeatScored[b] = 1;
    if (score >= SCORING.comboThreshold) {
      this.combo++;
      if (this.combo > this.maxCombo) this.maxCombo = this.combo;
    } else {
      this.combo = 0;
    }
    const ev = { beat: b, score, grade: gradeFor(score), combo: this.combo };
    this.lastBeatEvent = ev;
    if (this.onBeat) this.onBeat(ev);
  }

  /** Move score (0..1) with the energy gate applied, for a move index. */
  moveScore(i) {
    const ticks = this.moveTicks[i];
    if (ticks === 0) return 0;
    let s = this.moveSum[i] / ticks;
    const rms = Math.sqrt(this.moveRefSpeedSq[i] / ticks);
    const prms = Math.sqrt(this.movePlayerSpeedSq[i] / ticks);
    if (rms > 1e-6 && prms < SCORING.energyGateRatio * rms) {
      s = Math.min(s, SCORING.energyGateCap);
    }
    return s;
  }

  /** True when the energy gate capped move i. */
  moveGated(i) {
    const ticks = this.moveTicks[i];
    if (ticks === 0) return false;
    const rms = Math.sqrt(this.moveRefSpeedSq[i] / ticks);
    const prms = Math.sqrt(this.movePlayerSpeedSq[i] / ticks);
    return rms > 1e-6 && prms < SCORING.energyGateRatio * rms && this.moveSum[i] / ticks > SCORING.energyGateCap;
  }

  /** Running total (0..100) over the moves played so far (weighted by move duration). */
  currentScore() {
    let wsum = 0, w = 0;
    for (let i = 0; i < this.moveCount; i++) {
      if (this.moveTicks[i] === 0) continue;
      const dur = this._moveEndBeat[i] - this._moveStartBeat[i];
      wsum += this.moveScore(i) * dur;
      w += dur;
    }
    return w > 0 ? Math.round(100 * wsum / w) : 0;
  }

  /** Timing bias in seconds: > 0 = the player is behind (late), < 0 = early. */
  timingBias() {
    return this.deltaCount > 0 ? -this.deltaSum / this.deltaCount : 0;
  }

  /**
   * Close the last beat and build the Result object (DESIGN.md section 7).
   * @param {object} [info] { mode = 'solo', player = '', playedAt }
   */
  finalize(info = {}) {
    if (this.finalized) return this.result;
    if (this.currentBeat >= 0) this._closeBeat();
    this.finalized = true;
    const choreo = this.choreo;
    const moves = new Array(this.moveCount);
    let wsum = 0, w = 0;
    for (let i = 0; i < this.moveCount; i++) {
      const m = choreo.moves[i];
      const score = this.moveScore(i);
      const dur = m.endBeat - m.startBeat;
      wsum += score * dur;
      w += dur;
      moves[i] = {
        name: m.name,
        score: Math.round(score * 1000) / 1000,
        grade: gradeFor(score),
        startBeat: m.startBeat,
        endBeat: m.endBeat,
        ticks: this.moveTicks[i],
        gated: this.moveGated(i),
      };
    }
    const total = w > 0 ? wsum / w : 0;
    const score = Math.round(100 * total);
    this.result = {
      choreoId: choreo.id,
      score,
      stars: starsFor(score),
      maxCombo: this.maxCombo,
      timingBias: Math.round(this.timingBias() * 1000) / 1000,
      durationSec: Math.round(this.duration * 100) / 100,
      moves,
      perBeat: this.perBeat,
      playedAt: info.playedAt || new Date().toISOString(),
      mode: info.mode || 'solo',
      player: info.player || '',
      ticks: this.ticks,
      meanFrameScore: this.ticks > 0 ? this.frameSum / this.ticks : 0,
      mirror: this.mirror,
    };
    return this.result;
  }
}

/** Plain-JSON form of a Result (perBeat as a rounded array) for storage / POST /api/results. */
export function resultToJSON(result) {
  const perBeat = new Array(result.perBeat.length);
  for (let i = 0; i < perBeat.length; i++) perBeat[i] = Math.round(result.perBeat[i] * 1000) / 1000;
  return { ...result, perBeat };
}
