// pipeline.js - DPPipeline: pure-JS port of the Dancing Points runtime feature pipeline
// (docs/DESIGN.md section 6.2/6.3; oracle: tools/dp_pipeline.py, fixtures: tests/fixtures/*.json).
//
// Responsibility: 3-point root / root motion maths, DP <-> S conversion, assembly of the mapping
// and tracking network inputs (channel-major), decoding of their outputs and the per-tick
// post-processing (new root, root correction, Euler blend, bone-length restore). The DPPipeline
// class keeps the 30 Hz history ring buffer and the avatar state for the closed loop.
//
// Must not: import three.js or onnxruntime (importable in Node and in a Web Worker), allocate in
// anything called per tick (every function writes into preallocated typed arrays; the `create*`
// helpers allocate once). All internal maths is float64 (JS numbers), network feeds are Float32Array.
//
// Conventions: DP space (Unity, left-handed, Y up, forward +Z). A 3-point frame is 13 numbers
// [head x y z, head quat x y z w, right wrist x y z, left wrist x y z]; head forward = quat*(0,0,1).
// Root: pos = (head.x, 0, head.z), yaw = atan2(fwd.x, fwd.z), M = T(pos) * R_y(yaw).
// Root motion rm = [cos phi, sin phi, x, z], phi = -yaw. Matrices are row-major (16 or 9 values).
// Multi-frame tensors are channel-major: flat[c * T + t].

export const FPS = 30;
export const DT = 1 / FPS;
export const N_JOINTS = 34;
export const JOINT_NAMES = Object.freeze([
  'b_root', 'b_l_upleg', 'b_l_leg', 'b_l_foot_twist', 'b_l_foot', 'b_l_talocrural', 'b_l_subtalar',
  'b_l_transversetarsal', 'b_l_ball', 'b_r_upleg', 'b_r_leg', 'b_r_foot_twist', 'b_r_foot',
  'b_r_talocrural', 'b_r_subtalar', 'b_r_transversetarsal', 'b_r_ball', 'b_spine0', 'b_spine1',
  'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist',
  'b_neck0', 'b_head', 'b_head_null', 'b_r_shoulder', 'b_r_arm', 'b_r_forearm', 'b_r_wrist_twist',
  'b_r_wrist',
]);
export const PARENTS = Object.freeze([-1, 0, 1, 2, 3, 3, 5, 6, 7, 0, 9, 10, 11, 11, 13, 14, 15, 0, 17,
  18, 19, 20, 21, 22, 23, 24, 20, 26, 27, 20, 29, 30, 31, 32]);
export const HEAD = 27;
export const R_WRIST = 33;
export const L_WRIST = 25;
export const INPUT_JOINTS = Object.freeze([HEAD, R_WRIST, L_WRIST]); // head, right, left
export const CONTACT_JOINTS = Object.freeze([5, 8, 13, 16]);         // l_ankle, l_ball, r_ankle, r_ball

export const N_HISTORY = 15;          // mapping input frames t-14 .. t
export const N_FUTURE = 30;           // mapping output frames t+1 .. t+30
export const N_LEADER_TRACKING = 31;  // tracking leader window t .. t+30

export const FRAME_DIM = 13;
export const FRAME_HEAD = 0;
export const FRAME_QUAT = 3;
export const FRAME_RW = 7;
export const FRAME_LW = 10;

export const YAW_EPS = 1e-3;
export const BONE_LENGTH_TOLERANCE = 1.05;

export const MAPPING_INPUT_NAMES = Object.freeze(['input_leader_Positions', 'input_leader_RootMotion']);
export const MAPPING_OUTPUT_NAMES = Object.freeze(['Positions', 'RootMotion']);
export const TRACKING_INPUT_NAMES = Object.freeze(['input_leader_Positions', 'input_leader_RootMotion',
  'input_follower_VelocitiesV2', 'input_follower_Positions', 'input_follower_Rotations',
  'input_follower_FootContactLabels', 'input_follower_RootMotion']);
export const TRACKING_OUTPUT_NAMES = Object.freeze(['VelocitiesV2', 'Positions', 'Rotations',
  'FootContactLabels', 'RootMotion']);
export const MAPPING_INPUT_DIMS = Object.freeze({ input_leader_Positions: 9 * N_HISTORY, input_leader_RootMotion: 4 * N_HISTORY });
export const MAPPING_OUTPUT_DIMS = Object.freeze({ Positions: 9 * N_FUTURE, RootMotion: 4 * N_FUTURE });
export const TRACKING_INPUT_DIMS = Object.freeze({
  input_leader_Positions: 9 * N_LEADER_TRACKING, input_leader_RootMotion: 4 * N_LEADER_TRACKING,
  input_follower_VelocitiesV2: 3 * N_JOINTS, input_follower_Positions: 3 * N_JOINTS,
  input_follower_Rotations: 9 * N_JOINTS, input_follower_FootContactLabels: 4, input_follower_RootMotion: 4,
});
export const TRACKING_OUTPUT_DIMS = Object.freeze({
  VelocitiesV2: 3 * N_JOINTS * N_FUTURE, Positions: 3 * N_JOINTS * N_FUTURE, Rotations: 9 * N_JOINTS * N_FUTURE,
  FootContactLabels: 4 * N_FUTURE, RootMotion: 4 * N_FUTURE,
});

const TWO_PI = 2 * Math.PI;

// ---------------------------------------------------------------------------------------------
// Scalars, quaternions, small matrices
// ---------------------------------------------------------------------------------------------

/** Wrap an angle to [-pi, pi) (Python `(a + pi) % (2 pi) - pi`). */
export function wrapAngle(a) {
  let x = (a + Math.PI) % TWO_PI;
  if (x < 0) x += TWO_PI;
  return x - Math.PI;
}

/** Row-major 3x3 rotation matrix of quaternion q = [x, y, z, w] at q[qo..] into out[oo..oo+9). Not normalised. */
export function quatToMat(out, oo, q, qo) {
  const x = q[qo], y = q[qo + 1], z = q[qo + 2], w = q[qo + 3];
  out[oo] = 1 - 2 * (y * y + z * z);
  out[oo + 1] = 2 * (x * y - z * w);
  out[oo + 2] = 2 * (x * z + y * w);
  out[oo + 3] = 2 * (x * y + z * w);
  out[oo + 4] = 1 - 2 * (x * x + z * z);
  out[oo + 5] = 2 * (y * z - x * w);
  out[oo + 6] = 2 * (x * z - y * w);
  out[oo + 7] = 2 * (y * z + x * w);
  out[oo + 8] = 1 - 2 * (x * x + y * y);
  return out;
}

/** out[oo..oo+3) = q * v (rotate v by q; q at q[qo..], v at v[vo..]). out may alias v. */
export function quatRotate(out, oo, q, qo, v, vo) {
  const x = q[qo], y = q[qo + 1], z = q[qo + 2], w = q[qo + 3];
  const vx = v[vo], vy = v[vo + 1], vz = v[vo + 2];
  const rx = (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - z * w) * vy + 2 * (x * z + y * w) * vz;
  const ry = 2 * (x * y + z * w) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - x * w) * vz;
  const rz = 2 * (x * z - y * w) * vx + 2 * (y * z + x * w) * vy + (1 - 2 * (x * x + y * y)) * vz;
  out[oo] = rx; out[oo + 1] = ry; out[oo + 2] = rz;
  return out;
}

/** Head forward (gaze) direction in DP: q * (0, 0, 1) = third column of the rotation matrix. */
export function headForward(out, oo, q, qo) {
  const x = q[qo], y = q[qo + 1], z = q[qo + 2], w = q[qo + 3];
  out[oo] = 2 * (x * z + y * w);
  out[oo + 1] = 2 * (y * z - x * w);
  out[oo + 2] = 1 - 2 * (x * x + y * y);
  return out;
}

/** Yaw of a forward vector projected on the ground; keeps prevYaw when the projection is degenerate. */
export function yawFromForward(fx, fz, prevYaw, eps = YAW_EPS) {
  // sqrt instead of Math.hypot: hypot is variadic and allocates in V8 (this runs 16x per tick)
  if (Math.sqrt(fx * fx + fz * fz) < eps) return prevYaw;
  return Math.atan2(fx, fz);
}

/** 3-point root of a head: outPos = (head.x, 0, head.z); returns the yaw of the forward direction. */
export function rootFromHead(outPos, po, head, ho, fwd, fo, prevYaw) {
  outPos[po] = head[ho]; outPos[po + 1] = 0; outPos[po + 2] = head[ho + 2];
  return yawFromForward(fwd[fo], fwd[fo + 2], prevYaw);
}

/** Root motion [cos yaw, -sin yaw, x, z] (= [cos phi, sin phi, x, z], phi = -yaw). */
export function rmFromRoot(out, oo, x, z, yaw) {
  out[oo] = Math.cos(yaw); out[oo + 1] = -Math.sin(yaw); out[oo + 2] = x; out[oo + 3] = z;
  return out;
}

export function rmYaw(rm, o = 0) {
  return -Math.atan2(rm[o + 1], rm[o]);
}

/** Row-major 4x4 root matrix T(x, y, z) * R_y(yaw). */
export function rootMatrix(out, oo, x, y, z, yaw) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  out[oo] = c; out[oo + 1] = 0; out[oo + 2] = s; out[oo + 3] = x;
  out[oo + 4] = 0; out[oo + 5] = 1; out[oo + 6] = 0; out[oo + 7] = y;
  out[oo + 8] = -s; out[oo + 9] = 0; out[oo + 10] = c; out[oo + 11] = z;
  out[oo + 12] = 0; out[oo + 13] = 0; out[oo + 14] = 0; out[oo + 15] = 1;
  return out;
}

/** RootMotionType.AsMatrix4x4: phi = atan2(sin, cos), M = T(x, 0, z) * R_y(-phi). */
export function rmToMatrix(out, oo, rm, ro) {
  // self-contained (no scalar-argument call: avoids HeapNumber boxing when not inlined)
  const yaw = -Math.atan2(rm[ro + 1], rm[ro]);
  const c = Math.cos(yaw), s = Math.sin(yaw);
  out[oo] = c; out[oo + 1] = 0; out[oo + 2] = s; out[oo + 3] = rm[ro + 2];
  out[oo + 4] = 0; out[oo + 5] = 1; out[oo + 6] = 0; out[oo + 7] = 0;
  out[oo + 8] = -s; out[oo + 9] = 0; out[oo + 10] = c; out[oo + 11] = rm[ro + 3];
  out[oo + 12] = 0; out[oo + 13] = 0; out[oo + 14] = 0; out[oo + 15] = 1;
  return out;
}

/** Inverse of rmToMatrix for a row-major 4x4 root matrix. */
export function rmFromMatrix(out, oo, m, mo) {
  return rmFromRoot(out, oo, m[mo + 3], m[mo + 11], Math.atan2(m[mo + 2], m[mo]));
}

/** rel = relative(rm, ref): rot_rel = rot / rot_ref, pos_rel = (pos - pos_ref) / rot_ref (complex x + i z). */
export function rmRelative(out, oo, rm, ro, ref, fo) {
  const a = rm[ro], b = rm[ro + 1];
  const c = ref[fo], d = ref[fo + 1];
  const px = rm[ro + 2] - ref[fo + 2], pz = rm[ro + 3] - ref[fo + 3];
  const n = c * c + d * d;
  const rr = (a * c + b * d) / n, ri = (b * c - a * d) / n;
  const qr = (px * c + pz * d) / n, qi = (pz * c - px * d) / n;
  out[oo] = rr; out[oo + 1] = ri; out[oo + 2] = qr; out[oo + 3] = qi;
  return out;
}

/** rm = apply(ref, rel): rot = rot_rel * rot_ref, pos = pos_rel * rot_ref + pos_ref (Unity referenceRoot * deltaRoot). */
export function rmApply(out, oo, ref, fo, rel, ro) {
  const a = rel[ro], b = rel[ro + 1], px = rel[ro + 2], pz = rel[ro + 3];
  const c = ref[fo], d = ref[fo + 1], e = ref[fo + 2], f = ref[fo + 3];
  const rr = a * c - b * d, ri = a * d + b * c;
  const qr = px * c - pz * d + e, qi = px * d + pz * c + f;
  out[oo] = rr; out[oo + 1] = ri; out[oo + 2] = qr; out[oo + 3] = qi;
  return out;
}

/** out = M^-1 * p for a rigid row-major 4x4 M (Unity PositionTo). out may alias p. */
export function toLocal(out, oo, m, mo, p, po) {
  const dx = p[po] - m[mo + 3], dy = p[po + 1] - m[mo + 7], dz = p[po + 2] - m[mo + 11];
  const x = m[mo] * dx + m[mo + 4] * dy + m[mo + 8] * dz;
  const y = m[mo + 1] * dx + m[mo + 5] * dy + m[mo + 9] * dz;
  const z = m[mo + 2] * dx + m[mo + 6] * dy + m[mo + 10] * dz;
  out[oo] = x; out[oo + 1] = y; out[oo + 2] = z;
  return out;
}

/** out = M * p (Unity PositionFrom). out may alias p. */
export function toWorld(out, oo, m, mo, p, po) {
  const px = p[po], py = p[po + 1], pz = p[po + 2];
  const x = m[mo] * px + m[mo + 1] * py + m[mo + 2] * pz + m[mo + 3];
  const y = m[mo + 4] * px + m[mo + 5] * py + m[mo + 6] * pz + m[mo + 7];
  const z = m[mo + 8] * px + m[mo + 9] * py + m[mo + 10] * pz + m[mo + 11];
  out[oo] = x; out[oo + 1] = y; out[oo + 2] = z;
  return out;
}

/** out = R * v with the 3x3 rotation part of a row-major 4x4 M (no translation). */
export function rotateByMatrix(out, oo, m, mo, v, vo) {
  const px = v[vo], py = v[vo + 1], pz = v[vo + 2];
  const x = m[mo] * px + m[mo + 1] * py + m[mo + 2] * pz;
  const y = m[mo + 4] * px + m[mo + 5] * py + m[mo + 6] * pz;
  const z = m[mo + 8] * px + m[mo + 9] * py + m[mo + 10] * pz;
  out[oo] = x; out[oo + 1] = y; out[oo + 2] = z;
  return out;
}

// ---------------------------------------------------------------------------------------------
// DP <-> S (DESIGN 3): p_DP = (x, y, -z); q_S (x,y,z,w) -> q_DP (-x, -y, z, w). Both are involutions.
// ---------------------------------------------------------------------------------------------

export function dpToSPos(out, oo, p, po) {
  out[oo] = p[po]; out[oo + 1] = p[po + 1]; out[oo + 2] = -p[po + 2];
  return out;
}
export const sToDpPos = dpToSPos;

export function dpToSQuat(out, oo, q, qo) {
  out[oo] = -q[qo]; out[oo + 1] = -q[qo + 1]; out[oo + 2] = q[qo + 2]; out[oo + 3] = q[qo + 3];
  return out;
}
export const sToDpQuat = dpToSQuat;

/** Convert a 13-value 3-point frame between DP and S (in place allowed). */
export function frameDpToS(out, oo, f, fo) {
  dpToSPos(out, oo + FRAME_HEAD, f, fo + FRAME_HEAD);
  dpToSQuat(out, oo + FRAME_QUAT, f, fo + FRAME_QUAT);
  dpToSPos(out, oo + FRAME_RW, f, fo + FRAME_RW);
  dpToSPos(out, oo + FRAME_LW, f, fo + FRAME_LW);
  return out;
}
export const frameSToDp = frameDpToS;

/** Convert n xyz positions DP -> S (or back) into out. out may alias p. */
export function dpToSPositions(out, p, n) {
  for (let i = 0; i < n; i++) dpToSPos(out, i * 3, p, i * 3);
  return out;
}

// ---------------------------------------------------------------------------------------------
// Tensor layout helpers
// ---------------------------------------------------------------------------------------------

/** Frame-major x[t*C + c] (T frames, C channels) -> channel-major out[c*T + t]. */
export function channelMajor(out, x, T, C, xo = 0) {
  for (let t = 0; t < T; t++) {
    const base = xo + t * C;
    for (let c = 0; c < C; c++) out[c * T + t] = x[base + c];
  }
  return out;
}

/** One frame `t` of a channel-major tensor flat[c*T + t] (T frames, C channels) -> out[c]. */
export function splitFrame(out, flat, T, C, t) {
  for (let c = 0; c < C; c++) out[c] = flat[c * T + t];
  return out;
}

/** Flatten a (possibly nested) JSON array of numbers into out (returns the number of values written). */
export function flattenInto(out, nested, offset = 0) {
  let n = offset;
  for (let i = 0; i < nested.length; i++) {
    const v = nested[i];
    if (typeof v === 'number') out[n++] = v;
    else n = flattenInto(out, v, n);
  }
  return n;
}

// ---------------------------------------------------------------------------------------------
// 3-point frames
// ---------------------------------------------------------------------------------------------

/**
 * Roots of T consecutive DP frames (T*13, oldest first): outPos (T*3), outYaw (T). The yaw fallback
 * of a degenerate frame is the previous frame's yaw (prevYaw for the first one).
 */
export function frameRoots(frames, T, prevYaw, outPos, outYaw, fwd3) {
  let last = +prevYaw; // unary plus: keeps `last` a float64 phi (a tagged parameter would box every atan2 result)
  for (let i = 0; i < T; i++) {
    const fo = i * FRAME_DIM;
    // forward = quat * (0, 0, 1) (inline headForward / yawFromForward: no scalar calls in the hot loop)
    const q = fo + FRAME_QUAT;
    const x = frames[q], y = frames[q + 1], z = frames[q + 2], w = frames[q + 3];
    const fx = 2 * (x * z + y * w);
    const fz = 1 - 2 * (x * x + y * y);
    if (fwd3) { fwd3[0] = fx; fwd3[1] = 2 * (y * z - x * w); fwd3[2] = fz; }
    if (Math.sqrt(fx * fx + fz * fz) >= YAW_EPS) last = Math.atan2(fx, fz);
    outPos[i * 3] = frames[fo + FRAME_HEAD];
    outPos[i * 3 + 1] = 0;
    outPos[i * 3 + 2] = frames[fo + FRAME_HEAD + 2];
    outYaw[i] = last;
  }
  return outYaw;
}

/** Root-local positions (T*9) of head, right wrist, left wrist for T frames (rm4 / m16 scratch are unused, kept for API compatibility). */
export function threePointLocal(frames, T, rootPos, yaw, outLocal, rm4, m16) {
  for (let i = 0; i < T; i++) {
    // M = T(pos) * R_y(yaw); local = R^T (p - t) with R = [c 0 s; 0 1 0; -s 0 c]
    const c = Math.cos(yaw[i]), s = Math.sin(yaw[i]);
    const tx = rootPos[i * 3], tz = rootPos[i * 3 + 2];
    const fo = i * FRAME_DIM, lo = i * 9;
    for (let k = 0; k < 3; k++) {
      const po = fo + (k === 0 ? FRAME_HEAD : (k === 1 ? FRAME_RW : FRAME_LW));
      const dx = frames[po] - tx, dy = frames[po + 1], dz = frames[po + 2] - tz;
      outLocal[lo + k * 3] = c * dx - s * dz;
      outLocal[lo + k * 3 + 1] = dy;
      outLocal[lo + k * 3 + 2] = s * dx + c * dz;
    }
  }
  return outLocal;
}

// ---------------------------------------------------------------------------------------------
// Mapping network (DESIGN 6.3 steps 1-2)
// ---------------------------------------------------------------------------------------------

export function createMappingWork() {
  return {
    rootPos: new Float64Array(N_HISTORY * 3),
    yaw: new Float64Array(N_HISTORY),
    rm: new Float64Array(N_HISTORY * 4),
    rmRef: new Float64Array(4),
    rmRel: new Float64Array(N_HISTORY * 4),
    local: new Float64Array(N_HISTORY * 9),
    feeds: {
      input_leader_Positions: new Float32Array(9 * N_HISTORY),
      input_leader_RootMotion: new Float32Array(4 * N_HISTORY),
    },
    fwd: new Float64Array(3),
    rm4: new Float64Array(4),
    m16: new Float64Array(16),
  };
}

/**
 * Assemble the mapping inputs from the last 15 DP frames (15*13, oldest first). Fills
 * work.feeds (Float32Array 135 / 60) plus work.rm, rmRef (= rm of the oldest frame), rmRel, yaw,
 * rootPos and local (15*9) and returns work.feeds.
 */
export function mappingInputs(frames, prevYaw, work) {
  frameRoots(frames, N_HISTORY, prevYaw, work.rootPos, work.yaw, work.fwd);
  threePointLocal(frames, N_HISTORY, work.rootPos, work.yaw, work.local, work.rm4, work.m16);
  for (let i = 0; i < N_HISTORY; i++) { // rm = [cos yaw, -sin yaw, x, z] (inline rmFromRoot)
    work.rm[i * 4] = Math.cos(work.yaw[i]);
    work.rm[i * 4 + 1] = -Math.sin(work.yaw[i]);
    work.rm[i * 4 + 2] = work.rootPos[i * 3];
    work.rm[i * 4 + 3] = work.rootPos[i * 3 + 2];
  }
  work.rmRef[0] = work.rm[0]; work.rmRef[1] = work.rm[1]; work.rmRef[2] = work.rm[2]; work.rmRef[3] = work.rm[3];
  for (let i = 0; i < N_HISTORY; i++) rmRelative(work.rmRel, i * 4, work.rm, i * 4, work.rmRef, 0);
  channelMajor(work.feeds.input_leader_Positions, work.local, N_HISTORY, 9);
  channelMajor(work.feeds.input_leader_RootMotion, work.rmRel, N_HISTORY, 4);
  return work.feeds;
}

export function createMappingDecode() {
  return {
    rm: new Float64Array(N_FUTURE * 4),      // absolute roots of frames t+1 .. t+30
    rmRel: new Float64Array(N_FUTURE * 4),   // relative to rmRef
    local: new Float64Array(N_FUTURE * 9),   // root-local [head, right, left]
    world: new Float64Array(N_FUTURE * 9),   // world DP
    m16: new Float64Array(16),
  };
}

/** Decode mapping outputs {Positions[270], RootMotion[120]} relative to rmRef into out (createMappingDecode). */
export function decodeMapping(outputs, rmRef, out) {
  const pos = outputs.Positions, rmo = outputs.RootMotion;
  for (let f = 0; f < N_FUTURE; f++) {
    for (let c = 0; c < 9; c++) out.local[f * 9 + c] = pos[c * N_FUTURE + f];
    for (let c = 0; c < 4; c++) out.rmRel[f * 4 + c] = rmo[c * N_FUTURE + f];
    rmApply(out.rm, f * 4, rmRef, 0, out.rmRel, f * 4);
    rmToMatrix(out.m16, 0, out.rm, f * 4);
    for (let k = 0; k < 3; k++) toWorld(out.world, f * 9 + k * 3, out.m16, 0, out.local, f * 9 + k * 3);
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Avatar state and tracking network (DESIGN 6.3 steps 3-4)
// ---------------------------------------------------------------------------------------------

/** Full-body avatar state in DP: root motion, root-local positions/rotations/velocities, contacts, world. */
export function createAvatarState() {
  return {
    rm: new Float64Array(4),
    positions: new Float64Array(N_JOINTS * 3),
    rotations: new Float64Array(N_JOINTS * 9),
    velocities: new Float64Array(N_JOINTS * 3),
    contacts: new Float64Array(4),
    world: new Float64Array(N_JOINTS * 3),
    matrix: new Float64Array(16),
  };
}

/** Recompute avatar.matrix and avatar.world from rm and positions. */
export function avatarUpdateWorld(avatar) {
  rmToMatrix(avatar.matrix, 0, avatar.rm, 0);
  for (let j = 0; j < N_JOINTS; j++) toWorld(avatar.world, j * 3, avatar.matrix, 0, avatar.positions, j * 3);
  return avatar;
}

/** Fill an avatar from a pose object (init_pose.json / fixture: positions, rotations, velocities, contacts; nested or flat) at root rm. */
export function avatarFromPose(avatar, pose, rm, ro = 0) {
  avatar.rm[0] = rm[ro]; avatar.rm[1] = rm[ro + 1]; avatar.rm[2] = rm[ro + 2]; avatar.rm[3] = rm[ro + 3];
  flattenInto(avatar.positions, pose.positions);
  flattenInto(avatar.rotations, pose.rotations);
  if (pose.velocities) flattenInto(avatar.velocities, pose.velocities); else avatar.velocities.fill(0);
  if (pose.contacts) flattenInto(avatar.contacts, pose.contacts); else avatar.contacts.fill(1);
  return avatarUpdateWorld(avatar);
}

export function copyAvatar(dst, src) {
  dst.rm.set(src.rm); dst.positions.set(src.positions); dst.rotations.set(src.rotations);
  dst.velocities.set(src.velocities); dst.contacts.set(src.contacts); dst.world.set(src.world); dst.matrix.set(src.matrix);
  return dst;
}

export function createTrackingFeeds() {
  return {
    input_leader_Positions: new Float32Array(9 * N_LEADER_TRACKING),
    input_leader_RootMotion: new Float32Array(4 * N_LEADER_TRACKING),
    input_follower_VelocitiesV2: new Float32Array(3 * N_JOINTS),
    input_follower_Positions: new Float32Array(3 * N_JOINTS),
    input_follower_Rotations: new Float32Array(9 * N_JOINTS),
    input_follower_FootContactLabels: new Float32Array(4),
    input_follower_RootMotion: new Float32Array(4),
    rel: new Float64Array(4 * N_LEADER_TRACKING), // scratch (not a network input)
  };
}

/**
 * Assemble the tracking inputs: leaderRm (31*4) absolute roots of the leader window (frame 0 =
 * measured current frame, 1..30 = future), leaderLocal (31*9) root-local head/right/left, avatar =
 * reference root (reference_char = 1, follower rm relative to itself = [1, 0, 0, 0]).
 */
export function trackingInputs(leaderRm, leaderLocal, avatar, feeds) {
  for (let i = 0; i < N_LEADER_TRACKING; i++) rmRelative(feeds.rel, i * 4, leaderRm, i * 4, avatar.rm, 0);
  channelMajor(feeds.input_leader_Positions, leaderLocal, N_LEADER_TRACKING, 9);
  channelMajor(feeds.input_leader_RootMotion, feeds.rel, N_LEADER_TRACKING, 4);
  feeds.input_follower_VelocitiesV2.set(avatar.velocities);
  feeds.input_follower_Positions.set(avatar.positions);
  feeds.input_follower_Rotations.set(avatar.rotations);
  feeds.input_follower_FootContactLabels.set(avatar.contacts);
  feeds.input_follower_RootMotion[0] = 1; feeds.input_follower_RootMotion[1] = 0;
  feeds.input_follower_RootMotion[2] = 0; feeds.input_follower_RootMotion[3] = 0;
  return feeds;
}

export function createTrackingPred() {
  return {
    velocities: new Float64Array(N_JOINTS * 3),
    positions: new Float64Array(N_JOINTS * 3),
    rotations: new Float64Array(N_JOINTS * 9),
    contacts: new Float64Array(4),
    rm: new Float64Array(4),
  };
}

/** Decode one output frame (default 0 = t+1) of the tracking outputs into pred (createTrackingPred). */
export function decodeTracking(outputs, frame, pred) {
  splitFrame(pred.velocities, outputs.VelocitiesV2, N_FUTURE, 3 * N_JOINTS, frame);
  splitFrame(pred.positions, outputs.Positions, N_FUTURE, 3 * N_JOINTS, frame);
  splitFrame(pred.rotations, outputs.Rotations, N_FUTURE, 9 * N_JOINTS, frame);
  splitFrame(pred.contacts, outputs.FootContactLabels, N_FUTURE, 4, frame);
  splitFrame(pred.rm, outputs.RootMotion, N_FUTURE, 4, frame);
  return pred;
}

/** Blend the (x, z) position and the yaw of rm towards measuredRm by factor (0 = copy). out may alias rm. */
export function correctRoot(out, rm, measuredRm, factor) {
  if (factor <= 0) {
    out[0] = rm[0]; out[1] = rm[1]; out[2] = rm[2]; out[3] = rm[3];
    return out;
  }
  const x = rm[2] + factor * (measuredRm[2] - rm[2]);
  const z = rm[3] + factor * (measuredRm[3] - rm[3]);
  const yaw = -Math.atan2(rm[1], rm[0]);
  const myaw = -Math.atan2(measuredRm[1], measuredRm[0]);
  let d = (myaw - yaw + Math.PI) % TWO_PI; // wrapAngle inline
  if (d < 0) d += TWO_PI;
  const newYaw = yaw + factor * (d - Math.PI);
  out[0] = Math.cos(newYaw); out[1] = -Math.sin(newYaw); out[2] = x; out[3] = z;
  return out;
}

/** Validated skeleton with typed arrays; parents must precede children (checked once). */
export function prepareSkeleton(skeleton) {
  const n = skeleton.parents.length;
  const parents = new Int32Array(n);
  const boneLengths = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    parents[j] = skeleton.parents[j];
    boneLengths[j] = skeleton.boneLengths ? skeleton.boneLengths[j] : 0;
    if (parents[j] >= j) throw new Error('skeleton: parent of joint ' + j + ' must precede it');
  }
  return { joints: skeleton.joints ? skeleton.joints.slice() : null, parents, boneLengths, n };
}

/**
 * Unity RestoreBoneLength with Transform semantics: joints in order (parents first); if
 * |world_j - world_parent| > tolerance * boneLengths[j] the joint is moved to that distance along the
 * parent -> joint direction and the same delta is applied to all of its descendants. world is a
 * flat (n*3) array modified in place; mark is a Uint8Array(n) scratch. Returns the number of clamps.
 */
export function restoreBoneLengths(world, parents, boneLengths, tolerance, mark) {
  const n = parents.length;
  let clamps = 0;
  for (let j = 1; j < n; j++) {
    const p = parents[j];
    if (p < 0) continue;
    const jo = j * 3, po = p * 3;
    const dx = world[jo] - world[po], dy = world[jo + 1] - world[po + 1], dz = world[jo + 2] - world[po + 2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const maxLen = tolerance * boneLengths[j];
    if (len > maxLen) {
      const s = maxLen / len;
      const ddx = world[po] + dx * s - world[jo];
      const ddy = world[po + 1] + dy * s - world[jo + 1];
      const ddz = world[po + 2] + dz * s - world[jo + 2];
      mark.fill(0);
      mark[j] = 1;
      world[jo] += ddx; world[jo + 1] += ddy; world[jo + 2] += ddz;
      for (let k = j + 1; k < n; k++) {
        if (mark[parents[k]]) {
          mark[k] = 1;
          world[k * 3] += ddx; world[k * 3 + 1] += ddy; world[k * 3 + 2] += ddz;
        }
      }
      clamps++;
    }
  }
  return clamps;
}

export function createPostWork() {
  return {
    rm: new Float64Array(4),
    m: new Float64Array(16),
    world: new Float64Array(N_JOINTS * 3),
    mark: new Uint8Array(N_JOINTS),
    v: new Float64Array(3),
    clamps: 0,
  };
}

/**
 * One tracking tick post-processing (DESIGN 6.3 step 4 / appendix): new root = apply(avatar.rm,
 * pred.rm) corrected towards measuredRm; world = M_new * p; Euler blend with the previous world
 * positions; bone-length restore; positions re-expressed locally; rotations/velocities verbatim;
 * contacts clamped to [0, 1]. Updates avatar in place (rm, positions, rotations, velocities,
 * contacts, world, matrix) and returns it. skeleton = prepareSkeleton(...).
 */
export function postProcess(avatar, pred, measuredRm, skeleton, rootCorrection, eulerRatio, dt, work) {
  rmApply(work.rm, 0, avatar.rm, 0, pred.rm, 0);
  correctRoot(work.rm, work.rm, measuredRm, rootCorrection);
  rmToMatrix(work.m, 0, work.rm, 0);
  const world = work.world, m = work.m;
  for (let j = 0; j < N_JOINTS; j++) toWorld(world, j * 3, m, 0, pred.positions, j * 3);
  if (eulerRatio > 0) {
    for (let j = 0; j < N_JOINTS; j++) {
      const o = j * 3;
      rotateByMatrix(work.v, 0, m, 0, pred.velocities, o);
      const mx = avatar.world[o] + work.v[0] * dt;
      const my = avatar.world[o + 1] + work.v[1] * dt;
      const mz = avatar.world[o + 2] + work.v[2] * dt;
      world[o] = world[o] + eulerRatio * (mx - world[o]);
      world[o + 1] = world[o + 1] + eulerRatio * (my - world[o + 1]);
      world[o + 2] = world[o + 2] + eulerRatio * (mz - world[o + 2]);
    }
  }
  work.clamps = restoreBoneLengths(world, skeleton.parents, skeleton.boneLengths, BONE_LENGTH_TOLERANCE, work.mark);
  for (let j = 0; j < N_JOINTS; j++) toLocal(avatar.positions, j * 3, m, 0, world, j * 3);
  avatar.rotations.set(pred.rotations);
  avatar.velocities.set(pred.velocities);
  for (let i = 0; i < 4; i++) {
    const c = pred.contacts[i];
    avatar.contacts[i] = c < 0 ? 0 : (c > 1 ? 1 : c);
  }
  avatar.world.set(world);
  avatar.matrix.set(m);
  avatar.rm.set(work.rm);
  return avatar;
}

// ---------------------------------------------------------------------------------------------
// DPPipeline: closed-loop state machine used by the worker (and by tests / tools)
// ---------------------------------------------------------------------------------------------

const RING = N_HISTORY + 1; // window + the frame preceding it (its yaw is the fallback of the window)

/**
 * Per-tick usage (see runTick):
 *   pipeline.pushFrame(frameDP)              -> tick index
 *   feeds = pipeline.buildMappingInputs()    -> run mapping_leader
 *   pipeline.applyMappingOutputs(outputs)    -> pipeline.mapped (30 future roots / local / world)
 *   feeds = pipeline.buildTrackingInputs()   -> run tracking_leader
 *   pose = pipeline.applyTrackingOutputs(outputs)   -> pose.positionsWorldS etc.
 * Frames may be pushed more often than inference runs (the history stays a 30 Hz ring buffer).
 */
export class DPPipeline {
  constructor({ skeleton, initPose, rootCorrection = 0.35, eulerRatio = 0.5, dt = DT, useMeasuredFrame = true } = {}) {
    if (!skeleton || !initPose) throw new Error('DPPipeline needs skeleton and initPose');
    this.skeleton = prepareSkeleton(skeleton);
    if (this.skeleton.n !== N_JOINTS) throw new Error('DPPipeline expects a ' + N_JOINTS + '-joint skeleton');
    this.initPose = initPose;
    this.rootCorrection = rootCorrection;
    this.eulerRatio = eulerRatio;
    this.dt = dt;
    this.useMeasuredFrame = useMeasuredFrame;

    this.histFrames = new Float64Array(RING * FRAME_DIM);
    this.histYaw = new Float64Array(RING);
    this.histCount = 0;   // frames pushed since reset
    this.histNext = 0;    // ring slot of the next push
    this.window = new Float64Array(N_HISTORY * FRAME_DIM);
    this.windowPrevYaw = 0;
    this.fwd = new Float64Array(3);
    this.tmpPos = new Float64Array(3);

    this.mappingWork = createMappingWork();
    this.mapped = createMappingDecode();
    this.leaderRm = new Float64Array(N_LEADER_TRACKING * 4);
    this.leaderLocal = new Float64Array(N_LEADER_TRACKING * 9);
    this.trackingFeeds = createTrackingFeeds();
    this.pred = createTrackingPred();
    this.postWork = createPostWork();
    this.avatar = createAvatarState();
    this.avatarInitialised = false;
    this.measuredRm = new Float64Array(4); // root of the measured frame used by the current tick
    this.mappingBuilt = false;

    this.pose = {
      tick: -1,
      positionsWorldDP: new Float32Array(N_JOINTS * 3),
      positionsWorldS: new Float32Array(N_JOINTS * 3),
      rootDP: new Float32Array(3),
      rootS: new Float32Array(3),
      rootYawDP: 0,
      rootYawS: 0,
      rm: new Float64Array(4),
      contacts: new Float32Array(4),
      clamps: 0,
    };
    this.tick = -1;
  }

  /** Forget the history and the avatar; the avatar is re-initialised at the next tick's root. */
  reset() {
    this.histCount = 0;
    this.histNext = 0;
    this.avatarInitialised = false;
    this.mappingBuilt = false;
    this.tick = -1;
    this.pose.tick = -1;
  }

  /** Number of frames pushed since the last reset. */
  get frameCount() {
    return this.histCount;
  }

  /** Yaw of the most recently pushed frame (fallback chain applied). */
  get lastYaw() {
    return this.histCount === 0 ? 0 : this.histYaw[(this.histNext + RING - 1) % RING];
  }

  /**
   * Push one DP 3-point frame (13 values at frame[offset..]). The first frame pre-fills the whole
   * ring (Unity/Python semantics: idx = clip(t-14..t, 0)). Returns the tick index (0-based).
   */
  pushFrame(frame, offset = 0) {
    // previous yaw as a float64 load (not through the getter: a tagged value would box the phi below)
    let yaw = this.histCount === 0 ? 0.0 : this.histYaw[(this.histNext + RING - 1) % RING];
    const q = offset + FRAME_QUAT;
    const x = frame[q], y = frame[q + 1], z = frame[q + 2], w = frame[q + 3];
    const fx = 2 * (x * z + y * w), fz = 1 - 2 * (x * x + y * y); // forward = quat * (0,0,1)
    if (Math.sqrt(fx * fx + fz * fz) >= YAW_EPS) yaw = Math.atan2(fx, fz);
    if (this.histCount === 0) {
      for (let s = 0; s < RING; s++) {
        const so = s * FRAME_DIM;
        for (let i = 0; i < FRAME_DIM; i++) this.histFrames[so + i] = frame[offset + i];
        this.histYaw[s] = yaw;
      }
      this.histNext = 0;
    } else {
      const so = this.histNext * FRAME_DIM;
      for (let i = 0; i < FRAME_DIM; i++) this.histFrames[so + i] = frame[offset + i];
      this.histYaw[this.histNext] = yaw;
      this.histNext = (this.histNext + 1) % RING;
    }
    this.histCount++;
    this.tick++;
    return this.tick;
  }

  /** Copy the last 15 frames (oldest first) into this.window and set this.windowPrevYaw. */
  assembleWindow() {
    // Slots in push order: the oldest slot is histNext (the one about to be overwritten).
    const oldest = this.histNext; // slot of frame t-15 (the frame preceding the window)
    this.windowPrevYaw = this.histYaw[oldest];
    for (let i = 0; i < N_HISTORY; i++) {
      const s = (oldest + 1 + i) % RING;
      const so = s * FRAME_DIM, wo = i * FRAME_DIM;
      for (let k = 0; k < FRAME_DIM; k++) this.window[wo + k] = this.histFrames[so + k];
    }
    return this.window;
  }

  /** Build the mapping feeds for the current history; returns {input_leader_Positions, input_leader_RootMotion}. */
  buildMappingInputs() {
    if (this.histCount === 0) throw new Error('DPPipeline: no frame pushed');
    this.assembleWindow();
    const feeds = mappingInputs(this.window, this.windowPrevYaw, this.mappingWork);
    const w = this.mappingWork;
    const o = (N_HISTORY - 1) * 4;
    this.measuredRm[0] = w.rm[o]; this.measuredRm[1] = w.rm[o + 1]; this.measuredRm[2] = w.rm[o + 2]; this.measuredRm[3] = w.rm[o + 3];
    this.mappingBuilt = true;
    return feeds;
  }

  /** Decode the mapping outputs (relative to the oldest history frame) into this.mapped. */
  applyMappingOutputs(outputs) {
    if (!this.mappingBuilt) throw new Error('DPPipeline: buildMappingInputs() first');
    return decodeMapping(outputs, this.mappingWork.rmRef, this.mapped);
  }

  /** Place the avatar (init pose by default) at root rm. */
  setAvatarPose(pose, rm, ro = 0) {
    avatarFromPose(this.avatar, pose || this.initPose, rm, ro);
    this.avatarInitialised = true;
    return this.avatar;
  }

  /** Build the tracking feeds: leader window [measured t] + mapped t+1..t+30, avatar as reference. */
  buildTrackingInputs() {
    if (!this.mappingBuilt) throw new Error('DPPipeline: buildMappingInputs() first');
    if (!this.avatarInitialised) this.setAvatarPose(this.initPose, this.measuredRm, 0);
    const w = this.mappingWork;
    const lastLocal = (N_HISTORY - 1) * 9;
    if (this.useMeasuredFrame) {
      for (let c = 0; c < 4; c++) this.leaderRm[c] = this.measuredRm[c];
      for (let c = 0; c < 9; c++) this.leaderLocal[c] = w.local[lastLocal + c];
    } else { // Unity demo: prediction 0 duplicated
      for (let c = 0; c < 4; c++) this.leaderRm[c] = this.mapped.rm[c];
      for (let c = 0; c < 9; c++) this.leaderLocal[c] = this.mapped.local[c];
    }
    this.leaderRm.set(this.mapped.rm, 4);
    this.leaderLocal.set(this.mapped.local, 9);
    return trackingInputs(this.leaderRm, this.leaderLocal, this.avatar, this.trackingFeeds);
  }

  /** Decode output frame `frame` (default 0 = t+1), post-process and fill this.pose (returned). */
  applyTrackingOutputs(outputs, frame = 0) {
    if (!this.avatarInitialised) throw new Error('DPPipeline: buildTrackingInputs() first');
    decodeTracking(outputs, frame, this.pred);
    postProcess(this.avatar, this.pred, this.measuredRm, this.skeleton, this.rootCorrection, this.eulerRatio, this.dt, this.postWork);
    const pose = this.pose, av = this.avatar;
    pose.tick = this.tick;
    pose.positionsWorldDP.set(av.world);
    dpToSPositions(pose.positionsWorldS, av.world, N_JOINTS);
    pose.rm.set(av.rm);
    pose.rootDP[0] = av.rm[2]; pose.rootDP[1] = 0; pose.rootDP[2] = av.rm[3];
    pose.rootS[0] = av.rm[2]; pose.rootS[1] = 0; pose.rootS[2] = -av.rm[3];
    pose.rootYawDP = -Math.atan2(av.rm[1], av.rm[0]);
    pose.rootYawS = -pose.rootYawDP;
    pose.contacts.set(av.contacts);
    pose.clamps = this.postWork.clamps;
    return pose;
  }

  /**
   * One full tick: push frame, mapping, tracking, post-processing. models = { runMapping(feeds),
   * runTracking(feeds) } returning (a promise of) { name: Float32Array }. Returns this.pose.
   */
  async runTick(frame, models, offset = 0) {
    this.pushFrame(frame, offset);
    const mfeeds = this.buildMappingInputs();
    const mout = await models.runMapping(mfeeds);
    this.applyMappingOutputs(mout);
    const tfeeds = this.buildTrackingInputs();
    const tout = await models.runTracking(tfeeds);
    return this.applyTrackingOutputs(tout, 0);
  }
}

/**
 * Convert a stage-frame sample (src/xr/input.js shape: {head:{p,q}, left:{p}, right:{p}, k}) at
 * reference height into an unscaled DP 3-point frame (13 values) at out[oo..]. k = calibration
 * scale (sample.k or 1); positions are divided by it (the network wants real-world metres).
 */
export function sampleSToFrameDP(out, oo, sample, k) {
  const inv = k > 0 ? 1 / k : 1;
  const hp = sample.head.p, hq = sample.head.q, lp = sample.left.p, rp = sample.right.p;
  out[oo] = hp[0] * inv; out[oo + 1] = hp[1] * inv; out[oo + 2] = -hp[2] * inv;
  out[oo + 3] = -hq[0]; out[oo + 4] = -hq[1]; out[oo + 5] = hq[2]; out[oo + 6] = hq[3];
  out[oo + 7] = rp[0] * inv; out[oo + 8] = rp[1] * inv; out[oo + 9] = -rp[2] * inv;
  out[oo + 10] = lp[0] * inv; out[oo + 11] = lp[1] * inv; out[oo + 12] = -lp[2] * inv;
  return out;
}
