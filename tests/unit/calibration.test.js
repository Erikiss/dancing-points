// calibration.test.js - W <-> S transform, height normalisation and persistence.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Calibration } from '../../webxr/src/xr/calibration.js';
import { createSample } from '../../webxr/src/xr/input.js';
import { quatFromYaw, forwardFromQuat, yawFromQuat } from '../../webxr/src/util/math.js';

const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;
const vecNear = (a, b, eps = 1e-9) => a.every((x, i) => near(x, b[i], eps));

function makeStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => { map.set(k, String(v)); },
    removeItem: (k) => { map.delete(k); },
    map,
  };
}

/** Player standing at `origin` (floor), head height h, facing yaw (W). */
function standingSample(origin, h, yaw) {
  const s = createSample();
  s.head.p[0] = origin[0]; s.head.p[1] = h; s.head.p[2] = origin[2];
  quatFromYaw(s.head.q, yaw);
  s.head.valid = true;
  // hands: 0.25 m to each side, 0.45 m below the head, 0.3 m in front (in the facing frame)
  const c = Math.cos(yaw), sn = Math.sin(yaw);
  const place = (pt, ox, oz) => {
    pt.p[0] = origin[0] + ox * c + oz * sn;
    pt.p[1] = h - 0.45;
    pt.p[2] = origin[2] - ox * sn + oz * c;
    quatFromYaw(pt.q, yaw);
    pt.valid = true;
  };
  place(s.left, -0.25, -0.3);
  place(s.right, 0.25, -0.3);
  return s;
}

test('capture + toStage: origin, facing -Z, scale k = referenceHeight / h0', () => {
  const cal = new Calibration();
  const yaw = -Math.PI / 2; // facing +X in W
  const sample = standingSample([1, 0, 2], 1.5, yaw);
  const data = cal.capture(sample);
  assert.ok(data);
  assert.ok(near(data.h0, 1.5));
  assert.ok(near(data.yaw, yaw));
  assert.deepEqual(data.origin, [1, 0, 2]);
  assert.ok(near(cal.scale(1.7), 1.7 / 1.5));

  const s = cal.toStage(sample, 1.7);
  assert.ok(near(s.k, 1.7 / 1.5));
  assert.ok(vecNear(s.head.p, [0, 1.7, 0], 1e-9), `head ${s.head.p}`);
  // head forward in S is -Z
  const f = forwardFromQuat([0, 0, 0], s.head.q);
  assert.ok(vecNear(f, [0, 0, -1], 1e-9), `forward ${f}`);
  assert.ok(near(yawFromQuat(s.head.q), 0));
  // hands: left is at -X, in front (-Z), scaled
  const k = 1.7 / 1.5;
  assert.ok(vecNear(s.left.p, [-0.25 * k, (1.5 - 0.45) * k, -0.3 * k], 1e-9), `left ${s.left.p}`);
  assert.ok(vecNear(s.right.p, [0.25 * k, (1.5 - 0.45) * k, -0.3 * k], 1e-9), `right ${s.right.p}`);
  assert.equal(s.left.valid, true);

  // a point 1 m ahead of the player in W (+X) is 1*k ahead (-Z) in S
  const p = cal.worldToStage([0, 0, 0], [2, 1.5, 2], 1.7);
  assert.ok(vecNear(p, [0, 1.7, -k], 1e-9), `ahead ${p}`);
  // and back
  const w = cal.stageToWorld([0, 0, 0], p, 1.7);
  assert.ok(vecNear(w, [2, 1.5, 2], 1e-9), `back ${w}`);
});

test('toStage/fromStage round trip and matrix agree', () => {
  const cal = new Calibration();
  cal.set({ h0: 1.62, yaw: 2.3, origin: [-0.4, 0, 0.9] });
  const sample = standingSample([-0.1, 0, 1.4], 1.55, 1.1);
  const s = cal.toStage(sample, 1.7, createSample());
  const back = cal.fromStage(s, 1.7, createSample());
  for (const key of ['head', 'left', 'right']) {
    assert.ok(vecNear(back[key].p, sample[key].p, 1e-9), `${key} pos`);
    assert.ok(vecNear(back[key].q, sample[key].q, 1e-9), `${key} quat`);
  }
  // matrix S -> W (column major)
  const M = cal.stageToWorldMatrix(new Array(16).fill(0), 1.7);
  const p = s.left.p;
  const got = [
    M[0] * p[0] + M[4] * p[1] + M[8] * p[2] + M[12],
    M[1] * p[0] + M[5] * p[1] + M[9] * p[2] + M[13],
    M[2] * p[0] + M[6] * p[1] + M[10] * p[2] + M[14],
  ];
  assert.ok(vecNear(got, sample.left.p, 1e-9), `matrix ${got} vs ${sample.left.p}`);
  // default reference height on the instance is used when omitted
  const s2 = cal.toStage(sample);
  assert.ok(near(s2.k, 1.7 / 1.62));
});

test('identity when not calibrated; invalid samples rejected', () => {
  const cal = new Calibration({ referenceHeight: 1.7 });
  assert.equal(cal.valid, false);
  const p = cal.worldToStage([0, 0, 0], [1, 2, 3]);
  assert.ok(vecNear(p, [1, 2, 3]));
  const s = createSample();
  s.head.valid = false;
  assert.equal(cal.capture(s), null);
  s.head.valid = true;
  s.head.p[1] = 0.3; // sitting on the floor
  assert.equal(cal.capture(s), null);
  assert.equal(cal.set({ h0: NaN, yaw: 0 }), false);
  assert.equal(cal.valid, false);
});

test('averaged capture', () => {
  const cal = new Calibration();
  cal.beginCapture();
  assert.equal(cal.addSample(standingSample([0, 0, 0], 1.6, 0.1)), 1);
  assert.equal(cal.addSample(standingSample([0.2, 0, 0], 1.7, -0.1)), 2);
  const bad = createSample();
  bad.head.valid = false;
  assert.equal(cal.addSample(bad), 2);
  const data = cal.endCapture();
  assert.ok(data);
  assert.ok(near(data.h0, 1.65, 1e-9));
  assert.ok(near(data.yaw, 0, 1e-9));
  assert.ok(near(data.origin[0], 0.1, 1e-9));
  assert.equal(cal.endCapture(), null);
});

test('persistence via injectable storage', () => {
  const storage = makeStorage();
  const cal = new Calibration({ storage });
  assert.equal(cal.save(), false);           // nothing to save yet
  assert.equal(cal.load(), false);
  cal.capture(standingSample([0.5, 0, -0.5], 1.58, 0.4));
  assert.equal(cal.save(), true);
  assert.ok(storage.map.has('dp.calibration'));
  const cal2 = new Calibration({ storage });
  assert.equal(cal2.load(), true);
  assert.ok(near(cal2.h0, 1.58));
  assert.ok(near(cal2.yaw, 0.4));
  assert.deepEqual(cal2.origin, [0.5, 0, -0.5]);
  cal2.clear();
  assert.equal(cal2.valid, false);
  assert.equal(storage.map.has('dp.calibration'), false);
  // corrupt data is ignored
  storage.setItem('dp.calibration', '{not json');
  assert.equal(new Calibration({ storage }).load(), false);
  storage.setItem('dp.calibration', JSON.stringify({ version: 99, h0: 1.6, yaw: 0 }));
  assert.equal(new Calibration({ storage }).load(), false);
});
