// math.js - vec3 / quaternion / angle helpers on plain arrays or typed arrays.
// gl-matrix style: functions take an `out` array first and never allocate unless they are
// explicitly named `*Create`. Quaternion layout is [x, y, z, w] (three.js order). Forward is -Z.
// No three.js imports; used by every module including the Node unit tests.

export const EPS = 1e-9;

// ---------------------------------------------------------------------------------------------
// scalars

export function clamp(x, lo, hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

export function lerpScalar(a, b, t) {
  return a + (b - a) * t;
}

export function degToRad(d) {
  return d * (Math.PI / 180);
}

export function radToDeg(r) {
  return r * (180 / Math.PI);
}

/** Wrap an angle to (-pi, pi]. */
export function wrapAngle(a) {
  let x = a % (2 * Math.PI);
  if (x <= -Math.PI) x += 2 * Math.PI;
  else if (x > Math.PI) x -= 2 * Math.PI;
  return x;
}

/** Signed shortest difference b - a in (-pi, pi]. */
export function angleDiff(a, b) {
  return wrapAngle(b - a);
}

// ---------------------------------------------------------------------------------------------
// vec3

export function vec3Create(x = 0, y = 0, z = 0) {
  return [x, y, z];
}

export function set(out, x, y, z) {
  out[0] = x; out[1] = y; out[2] = z;
  return out;
}

export function copy(out, a) {
  out[0] = a[0]; out[1] = a[1]; out[2] = a[2];
  return out;
}

export function add(out, a, b) {
  out[0] = a[0] + b[0]; out[1] = a[1] + b[1]; out[2] = a[2] + b[2];
  return out;
}

export function sub(out, a, b) {
  out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2];
  return out;
}

export function scale(out, a, s) {
  out[0] = a[0] * s; out[1] = a[1] * s; out[2] = a[2] * s;
  return out;
}

/** out = a + b * s */
export function addScaled(out, a, b, s) {
  out[0] = a[0] + b[0] * s; out[1] = a[1] + b[1] * s; out[2] = a[2] + b[2] * s;
  return out;
}

export function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross(out, a, b) {
  const ax = a[0], ay = a[1], az = a[2];
  const bx = b[0], by = b[1], bz = b[2];
  out[0] = ay * bz - az * by;
  out[1] = az * bx - ax * bz;
  out[2] = ax * by - ay * bx;
  return out;
}

export function length(a) {
  return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

export function lengthSq(a) {
  return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}

export function dist(a, b) {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export function distSq(a, b) {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

/** Distance between two points stored in flat arrays at offsets ia and ib. */
export function distAt(a, ia, b, ib) {
  const dx = a[ia] - b[ib], dy = a[ia + 1] - b[ib + 1], dz = a[ia + 2] - b[ib + 2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export function normalize(out, a) {
  const l = length(a);
  if (l < EPS) { out[0] = 0; out[1] = 0; out[2] = 0; return out; }
  const inv = 1 / l;
  out[0] = a[0] * inv; out[1] = a[1] * inv; out[2] = a[2] * inv;
  return out;
}

export function lerp(out, a, b, t) {
  out[0] = a[0] + (b[0] - a[0]) * t;
  out[1] = a[1] + (b[1] - a[1]) * t;
  out[2] = a[2] + (b[2] - a[2]) * t;
  return out;
}

export function negate(out, a) {
  out[0] = -a[0]; out[1] = -a[1]; out[2] = -a[2];
  return out;
}

export function equalsApprox(a, b, eps = 1e-6) {
  return Math.abs(a[0] - b[0]) <= eps && Math.abs(a[1] - b[1]) <= eps && Math.abs(a[2] - b[2]) <= eps;
}

/** Rotate a vector about the Y axis by `angle` radians (right handed: +angle turns +X towards -Z). */
export function rotateY(out, a, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  const x = a[0], z = a[2];
  out[0] = x * c + z * s;
  out[1] = a[1];
  out[2] = -x * s + z * c;
  return out;
}

// ---------------------------------------------------------------------------------------------
// quaternion [x, y, z, w]

export function quatCreate(x = 0, y = 0, z = 0, w = 1) {
  return [x, y, z, w];
}

export function quatIdentity(out) {
  out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1;
  return out;
}

export function quatCopy(out, q) {
  out[0] = q[0]; out[1] = q[1]; out[2] = q[2]; out[3] = q[3];
  return out;
}

export function quatSet(out, x, y, z, w) {
  out[0] = x; out[1] = y; out[2] = z; out[3] = w;
  return out;
}

export function quatLength(q) {
  return Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
}

export function quatNormalize(out, q) {
  const l = quatLength(q);
  if (l < EPS) return quatIdentity(out);
  const inv = 1 / l;
  out[0] = q[0] * inv; out[1] = q[1] * inv; out[2] = q[2] * inv; out[3] = q[3] * inv;
  return out;
}

export function quatDot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

/** Conjugate (== inverse for unit quaternions). */
export function quatConjugate(out, q) {
  out[0] = -q[0]; out[1] = -q[1]; out[2] = -q[2]; out[3] = q[3];
  return out;
}

/** Inverse for arbitrary (non-zero) quaternions. */
export function quatInverse(out, q) {
  const d = quatDot(q, q);
  if (d < EPS) return quatIdentity(out);
  const inv = 1 / d;
  out[0] = -q[0] * inv; out[1] = -q[1] * inv; out[2] = -q[2] * inv; out[3] = q[3] * inv;
  return out;
}

/** Hamilton product out = a * b (apply b first, then a - three.js convention). */
export function quatMul(out, a, b) {
  const ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const bx = b[0], by = b[1], bz = b[2], bw = b[3];
  out[0] = ax * bw + aw * bx + ay * bz - az * by;
  out[1] = ay * bw + aw * by + az * bx - ax * bz;
  out[2] = az * bw + aw * bz + ax * by - ay * bx;
  out[3] = aw * bw - ax * bx - ay * by - az * bz;
  return out;
}

export function quatFromAxisAngle(out, axis, angle) {
  const h = angle * 0.5;
  const s = Math.sin(h);
  out[0] = axis[0] * s; out[1] = axis[1] * s; out[2] = axis[2] * s; out[3] = Math.cos(h);
  return out;
}

/** Rotation about +Y by `yaw` radians. */
export function quatFromYaw(out, yaw) {
  const h = yaw * 0.5;
  out[0] = 0; out[1] = Math.sin(h); out[2] = 0; out[3] = Math.cos(h);
  return out;
}

/** Rotate vector v by unit quaternion q: out = q * v * q^-1. `out` may alias `v`. */
export function quatMulVec(out, q, v) {
  const x = v[0], y = v[1], z = v[2];
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  // t = 2 * cross(q.xyz, v)
  const tx = 2 * (qy * z - qz * y);
  const ty = 2 * (qz * x - qx * z);
  const tz = 2 * (qx * y - qy * x);
  // out = v + qw * t + cross(q.xyz, t)
  out[0] = x + qw * tx + (qy * tz - qz * ty);
  out[1] = y + qw * ty + (qz * tx - qx * tz);
  out[2] = z + qw * tz + (qx * ty - qy * tx);
  return out;
}

/** Forward vector of an orientation: q * (0, 0, -1). */
export function forwardFromQuat(out, q) {
  out[0] = 0; out[1] = 0; out[2] = -1;
  return quatMulVec(out, q, out);
}

/** Up vector of an orientation: q * (0, 1, 0). */
export function upFromQuat(out, q) {
  out[0] = 0; out[1] = 1; out[2] = 0;
  return quatMulVec(out, q, out);
}

/**
 * Yaw (rotation about +Y) of the orientation's forward direction projected on the ground, such
 * that quatFromYaw(yaw) * (0,0,-1) == normalized ground projection of q * (0,0,-1).
 * Returns `fallback` when the projection is shorter than 1e-3 (looking straight up/down).
 */
export function yawFromQuat(q, fallback = 0) {
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  // forward = q * (0,0,-1)
  const fx = -(2 * (qx * qz + qw * qy));
  const fz = -(1 - 2 * (qx * qx + qy * qy));
  if (fx * fx + fz * fz < 1e-6) return fallback;
  return Math.atan2(-fx, -fz);
}

/** Yaw of a direction vector (same convention as yawFromQuat). */
export function yawFromForward(f, fallback = 0) {
  if (f[0] * f[0] + f[2] * f[2] < 1e-6) return fallback;
  return Math.atan2(-f[0], -f[2]);
}

/** Spherical linear interpolation with shortest-path handling. */
export function slerp(out, a, b, t) {
  let ax = a[0], ay = a[1], az = a[2], aw = a[3];
  let bx = b[0], by = b[1], bz = b[2], bw = b[3];
  let cosom = ax * bx + ay * by + az * bz + aw * bw;
  if (cosom < 0) {
    cosom = -cosom;
    bx = -bx; by = -by; bz = -bz; bw = -bw;
  }
  let s0, s1;
  if (1 - cosom > 1e-6) {
    const omega = Math.acos(clamp(cosom, -1, 1));
    const sinom = Math.sin(omega);
    s0 = Math.sin((1 - t) * omega) / sinom;
    s1 = Math.sin(t * omega) / sinom;
  } else {
    s0 = 1 - t;
    s1 = t;
  }
  out[0] = s0 * ax + s1 * bx;
  out[1] = s0 * ay + s1 * by;
  out[2] = s0 * az + s1 * bz;
  out[3] = s0 * aw + s1 * bw;
  return out;
}

/** Angle in radians between two unit quaternions. */
export function quatAngle(a, b) {
  const d = Math.abs(quatDot(a, b));
  return 2 * Math.acos(clamp(d, -1, 1));
}

export function quatEqualsApprox(a, b, eps = 1e-6) {
  return quatAngle(a, b) <= eps;
}

// ---------------------------------------------------------------------------------------------
// helpers on flat Float32Array layouts (x,y,z triples)

/** Copy a 3-vector into a flat array at offset. */
export function put3(dst, offset, v) {
  dst[offset] = v[0]; dst[offset + 1] = v[1]; dst[offset + 2] = v[2];
  return dst;
}

/** Read a 3-vector from a flat array at offset. */
export function get3(out, src, offset) {
  out[0] = src[offset]; out[1] = src[offset + 1]; out[2] = src[offset + 2];
  return out;
}

/** Linear interpolation between two flat triples into `out`. */
export function lerp3At(out, a, ia, b, ib, t) {
  out[0] = a[ia] + (b[ib] - a[ia]) * t;
  out[1] = a[ia + 1] + (b[ib + 1] - a[ia + 1]) * t;
  out[2] = a[ia + 2] + (b[ib + 2] - a[ia + 2]) * t;
  return out;
}

/** Fill a 4x4 column-major matrix (three.js Matrix4.elements layout) from T(pos) * R_y(yaw) * S(s). */
export function matrixFromYawPosScale(out16, yaw, pos, s = 1) {
  const c = Math.cos(yaw), sn = Math.sin(yaw);
  // column 0 = R * (s,0,0)
  out16[0] = c * s; out16[1] = 0; out16[2] = -sn * s; out16[3] = 0;
  // column 1
  out16[4] = 0; out16[5] = s; out16[6] = 0; out16[7] = 0;
  // column 2 = R * (0,0,s)
  out16[8] = sn * s; out16[9] = 0; out16[10] = c * s; out16[11] = 0;
  // column 3 translation
  out16[12] = pos[0]; out16[13] = pos[1]; out16[14] = pos[2]; out16[15] = 1;
  return out16;
}
