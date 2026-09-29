// math.test.js - unit tests for webxr/src/util/math.js (node --test).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as m from '../../webxr/src/util/math.js';

const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;
const vecNear = (a, b, eps = 1e-9) => a.length === b.length && a.every((x, i) => near(x, b[i], eps));

test('vec3 basics', () => {
  const out = [0, 0, 0];
  assert.deepEqual(m.add(out, [1, 2, 3], [4, 5, 6]), [5, 7, 9]);
  assert.deepEqual(m.sub(out, [1, 2, 3], [4, 5, 6]), [-3, -3, -3]);
  assert.deepEqual(m.scale(out, [1, 2, 3], 2), [2, 4, 6]);
  assert.deepEqual(m.addScaled(out, [1, 1, 1], [1, 2, 3], 2), [3, 5, 7]);
  assert.equal(m.dot([1, 2, 3], [4, 5, 6]), 32);
  assert.deepEqual(m.cross(out, [1, 0, 0], [0, 1, 0]), [0, 0, 1]);
  assert.equal(m.length([3, 4, 0]), 5);
  assert.equal(m.dist([1, 1, 1], [4, 5, 1]), 5);
  assert.equal(m.distSq([0, 0, 0], [1, 2, 2]), 9);
  assert.deepEqual(m.lerp(out, [0, 0, 0], [2, 4, 6], 0.5), [1, 2, 3]);
  assert.ok(vecNear(m.normalize(out, [0, 3, 4]), [0, 0.6, 0.8]));
  assert.deepEqual(m.normalize(out, [0, 0, 0]), [0, 0, 0]);
  assert.equal(m.clamp(5, 0, 1), 1);
  assert.equal(m.clamp(-5, 0, 1), 0);
  assert.ok(near(m.wrapAngle(3 * Math.PI), Math.PI));
  assert.ok(near(m.wrapAngle(-3 * Math.PI), Math.PI));
  assert.ok(near(m.angleDiff(3, -3), 2 * Math.PI - 6));
  assert.equal(m.distAt([0, 0, 0, 1, 1, 1], 3, [0, 0, 0, 1, 1, 1], 0), Math.sqrt(3));
});

test('quaternion axis-angle / yaw conventions (forward = -Z, right-handed)', () => {
  const q = [0, 0, 0, 1];
  const v = [0, 0, 0];
  m.quatFromAxisAngle(q, [0, 1, 0], Math.PI / 2);
  m.forwardFromQuat(v, q);
  assert.ok(vecNear(v, [-1, 0, 0], 1e-12), `yaw +90deg forward should be -X, got ${v}`);
  // rotateY matches the quaternion
  const r = m.rotateY([0, 0, 0], [0, 0, -1], Math.PI / 2);
  assert.ok(vecNear(r, v, 1e-12));
  // quatFromYaw == axis-angle about Y
  const q2 = m.quatFromYaw([0, 0, 0, 1], Math.PI / 2);
  assert.ok(m.quatEqualsApprox(q, q2, 1e-12));
  // yaw round trip
  for (const yaw of [-3, -1.5, -0.3, 0, 0.7, 2, 3.1]) {
    m.quatFromYaw(q, yaw);
    assert.ok(near(m.yawFromQuat(q), yaw, 1e-9), `yaw ${yaw}`);
    m.forwardFromQuat(v, q);
    assert.ok(near(m.yawFromForward(v), yaw, 1e-9));
  }
  // looking straight down -> fallback
  m.quatFromAxisAngle(q, [1, 0, 0], -Math.PI / 2);
  assert.equal(m.yawFromQuat(q, 42), 42);
  // yaw with pitch mixed in keeps the yaw
  const qy = m.quatFromYaw([0, 0, 0, 1], 1.2);
  const qp = m.quatFromAxisAngle([0, 0, 0, 1], [1, 0, 0], 0.5);
  m.quatMul(q, qy, qp);
  assert.ok(near(m.yawFromQuat(q), 1.2, 1e-9));
  m.upFromQuat(v, qy);
  assert.ok(vecNear(v, [0, 1, 0], 1e-12));
});

test('quaternion multiply / inverse / rotate vector', () => {
  const a = m.quatFromAxisAngle([0, 0, 0, 1], [0, 1, 0], 0.9);
  const b = m.quatFromAxisAngle([0, 0, 0, 1], [1, 0, 0], -0.4);
  const ab = m.quatMul([0, 0, 0, 1], a, b);
  const inv = m.quatInverse([0, 0, 0, 1], ab);
  const id = m.quatMul([0, 0, 0, 1], ab, inv);
  assert.ok(m.quatEqualsApprox(id, [0, 0, 0, 1], 1e-9));
  const conj = m.quatConjugate([0, 0, 0, 1], ab);
  assert.ok(vecNear(conj, inv, 1e-12));
  // q * v * q^-1 == applying b then a
  const v = [0.3, -0.2, 0.9];
  const v1 = m.quatMulVec([0, 0, 0], b, v);
  const v2 = m.quatMulVec([0, 0, 0], a, v1);
  const v3 = m.quatMulVec([0, 0, 0], ab, v);
  assert.ok(vecNear(v2, v3, 1e-12));
  // length preserved
  assert.ok(near(m.length(v3), m.length(v), 1e-12));
  // aliasing out == v works
  const v4 = v.slice();
  m.quatMulVec(v4, ab, v4);
  assert.ok(vecNear(v4, v3, 1e-12));
  // normalize
  const qn = m.quatNormalize([0, 0, 0, 1], [0, 0, 0, 4]);
  assert.deepEqual(qn, [0, 0, 0, 1]);
  assert.deepEqual(m.quatNormalize([1, 1, 1, 1], [0, 0, 0, 0]), [0, 0, 0, 1]);
});

test('slerp endpoints, midpoint and shortest path', () => {
  const a = m.quatFromYaw([0, 0, 0, 1], 0);
  const b = m.quatFromYaw([0, 0, 0, 1], 1.0);
  const out = [0, 0, 0, 1];
  m.slerp(out, a, b, 0);
  assert.ok(m.quatEqualsApprox(out, a, 1e-9));
  m.slerp(out, a, b, 1);
  assert.ok(m.quatEqualsApprox(out, b, 1e-9));
  m.slerp(out, a, b, 0.5);
  assert.ok(near(m.yawFromQuat(out), 0.5, 1e-9));
  assert.ok(near(m.quatLength(out), 1, 1e-12));
  // -b is the same rotation; slerp must take the short way
  const nb = [-b[0], -b[1], -b[2], -b[3]];
  m.slerp(out, a, nb, 0.5);
  assert.ok(near(m.yawFromQuat(out), 0.5, 1e-9));
  // identical inputs
  m.slerp(out, a, a, 0.3);
  assert.ok(m.quatEqualsApprox(out, a, 1e-9));
  assert.ok(near(m.quatAngle(a, b), 1.0, 1e-9));
});

test('flat array helpers and yaw/pos/scale matrix', () => {
  const flat = new Float32Array([1, 2, 3, 4, 5, 6]);
  const out = [0, 0, 0];
  assert.deepEqual(m.get3(out, flat, 3), [4, 5, 6]);
  m.put3(flat, 0, [7, 8, 9]);
  assert.equal(flat[2], 9);
  m.lerp3At(out, flat, 0, flat, 3, 0.5);
  assert.deepEqual(out, [5.5, 6.5, 7.5]);
  // matrix: M * p == R_y(yaw) * (s * p) + pos
  const yaw = 0.8, pos = [1, 2, 3], s = 1.5;
  const M = m.matrixFromYawPosScale(new Array(16).fill(0), yaw, pos, s);
  const p = [0.3, -0.7, 0.2];
  const expect = m.rotateY([0, 0, 0], m.scale([0, 0, 0], p, s), yaw);
  m.add(expect, expect, pos);
  // column-major multiply
  const got = [
    M[0] * p[0] + M[4] * p[1] + M[8] * p[2] + M[12],
    M[1] * p[0] + M[5] * p[1] + M[9] * p[2] + M[13],
    M[2] * p[0] + M[6] * p[1] + M[10] * p[2] + M[14],
  ];
  assert.ok(vecNear(got, expect, 1e-12), `${got} vs ${expect}`);
});
