#!/usr/bin/env python3
"""Generate the parity fixtures tests/fixtures/{math,mapping_io,tracking_io}.json (DESIGN.md 6.4).

Responsibility: derive every expected value from tools/dp_pipeline.py (the oracle) and the fp32
ONNX checkpoints via onnxruntime; store plain JSON arrays, float32-rounded to 7 significant
digits. Expected values are computed FROM the rounded inputs, so a port that reads the fixture
can reproduce them exactly (inputs to 1e-5, post-processing to 1e-4).
Must not: use the int8 models for expected outputs (they are only referenced by name).

Usage:
  python tools/gen_fixtures.py --clip <clip.npy> --ckpt <fp32 checkpoint dir> \
      [--models webxr/models/free] [--out tests/fixtures] [--pivot-sec 60.0]
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dp_pipeline as dp  # noqa: E402

FORMAT = 'dancing-points-fixture/1'


def r7(x):
  """float32 -> 7 significant digits, recursively over lists/arrays."""
  if isinstance(x, (np.ndarray, list, tuple)):
    return [r7(v) for v in np.asarray(x).tolist()] if isinstance(x, np.ndarray) else [r7(v) for v in x]
  if isinstance(x, (bool, np.bool_)):
    return bool(x)
  if isinstance(x, (int, np.integer)):
    return int(x)
  v = float(np.float32(x))
  if v == 0.0:
    return 0.0
  return float('%.7g' % v)


def a(x):
  """Rounded value back as a float64 numpy array (so expectations use the stored inputs)."""
  return np.asarray(r7(x), dtype=np.float64)


def mat16(m):
  return r7(np.asarray(m).reshape(-1))


def write(path, obj):
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(obj, f, separators=(',', ':'))
  print('wrote %s (%.1f KB)' % (path, os.path.getsize(path) / 1024))


# --------------------------------------------------------------------------------------------
# math.json
# --------------------------------------------------------------------------------------------

def quat_from_yaw_pitch_roll(yaw, pitch, roll):
  """Head orientation in DP: R_y(yaw) * R_x(pitch) * R_z(roll) as a quaternion."""
  qy = dp.quat_from_axis_angle([0, 1, 0], yaw)
  qx = dp.quat_from_axis_angle([1, 0, 0], pitch)
  qz = dp.quat_from_axis_angle([0, 0, 1], roll)
  return dp.quat_mul(qy, dp.quat_mul(qx, qz))


def math_cases(clip):
  rng = np.random.default_rng(20240917)
  cases = []

  def add(kind, inp, exp, note=''):
    cases.append({'id': '%s_%02d' % (kind, sum(1 for c in cases if c['kind'] == kind) + 1),
                  'kind': kind, 'input': inp, 'expected': exp, 'note': note})

  # root from head: level, tilted, rolled, looking down (degenerate), from the clip
  heads = [
    ([0.0, 1.6, 0.0], (0.0, 0.0, 0.0), 0.0, 'level head at origin'),
    ([1.2, 1.55, -0.7], (0.8, 0.0, 0.0), 0.0, 'yawed head'),
    ([-0.4, 1.7, 2.1], (-2.3, 0.6, 0.0), 0.0, 'pitched down 34 deg'),
    ([0.3, 1.5, 0.2], (2.9, -0.4, 0.5), 0.0, 'pitched up and rolled'),
    ([0.5, 1.62, -1.1], (1.0, math.pi / 2 - 1e-4, 0.0), 0.7, 'looking straight down -> prevYaw'),
    ([0.5, 1.62, -1.1], (1.0, 1.5698, 0.0), 0.7, 'almost straight down (|xz| < 1e-3) -> prevYaw'),
    ([0.5, 1.62, -1.1], (1.0, 1.55, 0.0), 0.7, 'steep but valid (|xz| > 1e-3)'),
    ([2.0, 1.4, 3.0], (-3.1, 0.2, -0.3), 0.0, 'yaw near -pi'),
  ]
  for pos, (yaw, pitch, roll), prev, note in heads:
    q = a(quat_from_yaw_pitch_roll(yaw, pitch, roll))
    pos = a(pos)
    fwd = dp.head_forward(q)
    rpos, ryaw = dp.root_from_head(pos, fwd, prev)
    rm = dp.rm_from_root(rpos, ryaw)
    add('rootFromHead', {'headPos': r7(pos), 'headQuat': r7(q), 'prevYaw': r7(prev)},
        {'forward': r7(fwd), 'rootPos': r7(rpos), 'yaw': r7(ryaw), 'rm': r7(rm),
         'matrix': mat16(dp.root_matrix(rpos, ryaw))}, note)
  for i in [1000, 9876, 20000]:
    fr = a(dp.clip_three_point_frames(clip, np.array([i]))[0])
    fwd = dp.head_forward(fr[dp.FRAME_QUAT])
    rpos, ryaw = dp.root_from_head(fr[dp.FRAME_HEAD], fwd, 0.0)
    add('rootFromHead', {'headPos': r7(fr[dp.FRAME_HEAD]), 'headQuat': r7(fr[dp.FRAME_QUAT]), 'prevYaw': 0.0},
        {'forward': r7(fwd), 'rootPos': r7(rpos), 'yaw': r7(ryaw), 'rm': r7(dp.rm_from_root(rpos, ryaw)),
         'matrix': mat16(dp.root_matrix(rpos, ryaw)),
         'datasetRm': r7(clip['rm'][i])}, 'dataset frame %d (datasetRm = dataset root motion)' % i)

  # root motion <-> matrix round trips
  for k in range(6):
    pos = a([rng.uniform(-3, 3), 0.0, rng.uniform(-3, 3)])
    yaw = r7(rng.uniform(-math.pi, math.pi)) if k < 5 else r7(math.pi - 1e-6)
    rm = a(dp.rm_from_root(pos, yaw))
    m = dp.rm_to_matrix(rm)
    add('rmMatrix', {'rm': r7(rm)},
        {'matrix': mat16(m), 'yaw': r7(dp.rm_yaw(rm)), 'pos': r7(dp.rm_pos(rm)), 'rmFromMatrix': r7(dp.rm_from_matrix(m))},
        'rm -> matrix (row-major 4x4) -> rm')

  # relative root and apply
  for k in range(6):
    ref = a(dp.rm_from_root([rng.uniform(-3, 3), 0, rng.uniform(-3, 3)], rng.uniform(-math.pi, math.pi)))
    rm = a(dp.rm_from_root([rng.uniform(-3, 3), 0, rng.uniform(-3, 3)], rng.uniform(-math.pi, math.pi)))
    rel = dp.rm_relative(rm, ref)
    add('rmRelative', {'rm': r7(rm), 'ref': r7(ref)},
        {'rel': r7(rel), 'relMatrix': mat16(dp.rm_to_matrix(rel)),
         'applied': r7(dp.rm_apply(ref, a(rel))), 'selfRelative': r7(dp.rm_relative(ref, ref))},
        'rel = relative(rm, ref); applied = apply(ref, round7(rel)) ~ rm; relMatrix = inv(M_ref) * M_rm')

  # local <-> world
  for k in range(4):
    rm = a(dp.rm_from_root([rng.uniform(-3, 3), 0, rng.uniform(-3, 3)], rng.uniform(-math.pi, math.pi)))
    pts = a(rng.uniform(-2, 2, size=(3, 3)))
    m = dp.rm_to_matrix(rm)
    loc = dp.to_local(m, pts)
    add('localWorld', {'rm': r7(rm), 'world': r7(pts)},
        {'local': r7(loc), 'worldBack': r7(dp.to_world(m, a(loc)))}, 'local = inv(M) * world')

  # DP <-> S with quaternions (verified against M R M, M = diag(1,1,-1))
  mirror = np.diag([1.0, 1.0, -1.0])
  for k in range(5):
    p = a(rng.uniform(-3, 3, size=3))
    q = a(dp.quat_normalize(rng.normal(size=4)))
    qs = dp.dp_to_s_quat(q)
    add('dpToS', {'posDP': r7(p), 'quatDP': r7(q)},
        {'posS': r7(dp.dp_to_s_pos(p)), 'quatS': r7(qs), 'rotS': r7(mirror @ dp.quat_to_mat(q) @ mirror),
         'forwardDP': r7(dp.head_forward(q)), 'forwardS': r7(dp.quat_rotate(qs, [0, 0, -1]))},
        'rotS = M R_DP M; forwardS = quatS * (0,0,-1) = dpToS(forwardDP)')
  for k in range(3):
    p = a(rng.uniform(-3, 3, size=3))
    q = a(dp.quat_normalize(rng.normal(size=4)))
    add('sToDp', {'posS': r7(p), 'quatS': r7(q)},
        {'posDP': r7(dp.s_to_dp_pos(p)), 'quatDP': r7(dp.s_to_dp_quat(q)),
         'rotDP': r7(mirror @ dp.quat_to_mat(q) @ mirror)}, 'involution of dpToS')

  # head forward
  for k in range(3):
    q = a(dp.quat_normalize(rng.normal(size=4)))
    add('headForward', {'quatDP': r7(q)}, {'forwardDP': r7(dp.head_forward(q)), 'rot': r7(dp.quat_to_mat(q))},
        'forward = quat * (0,0,1); rot = quaternion -> 3x3 (row-major)')

  # frame roots over a sequence with a degenerate frame in the middle
  frames = a(dp.clip_three_point_frames(clip, dp.sampled_indices(3000, 6)))
  frames[3, dp.FRAME_QUAT] = a(quat_from_yaw_pitch_roll(0.3, math.pi / 2, 0.0))  # straight down
  rp, ry = dp.frame_roots(frames, prev_yaw=0.25)
  add('frameRoots', {'frames': r7(frames), 'prevYaw': 0.25},
      {'rootPos': r7(rp), 'yaw': r7(ry), 'rm': r7(dp.rm_from_root(rp, ry)),
       'local': r7(dp.three_point_local(frames, rp, ry))},
      'frame 3 looks straight down and keeps the yaw of frame 2; local = [head, right, left] per frame')

  return cases


# --------------------------------------------------------------------------------------------
# mapping_io.json
# --------------------------------------------------------------------------------------------

def mapping_fixture(clip, models, pivot, clip_name):
  idx = dp.sampled_indices(pivot - (dp.N_HISTORY - 1) * dp.SAMPLE_STEP, dp.N_HISTORY)
  frames = a(dp.clip_three_point_frames(clip, idx))
  feeds, info = dp.mapping_inputs(frames, prev_yaw=0.0)
  feeds = {k: a(v).astype(np.float32) for k, v in feeds.items()}
  outputs = models.run_mapping(feeds)
  outputs = {k: a(v).astype(np.float32) for k, v in outputs.items()}
  dec = dp.decode_mapping(outputs, a(info['rm_ref']))
  fidx = dp.sampled_indices(pivot + dp.SAMPLE_STEP, dp.N_FUTURE)
  gt_world = dp.clip_world_positions(clip, fidx)[:, list(dp.INPUT_JOINTS)]
  return {
    'format': FORMAT,
    'kind': 'mapping_io',
    'generator': 'tools/gen_fixtures.py',
    'params': {
      'clip': clip_name, 'datasetFps': dp.DATASET_FPS, 'step': dp.SAMPLE_STEP, 'modelFps': dp.FPS,
      'pivotFrame': int(pivot), 'pivotSec': pivot / dp.DATASET_FPS, 'datasetFrames': [int(i) for i in idx],
      'nHistory': dp.N_HISTORY, 'nFuture': dp.N_FUTURE, 'referenceFrame': 0, 'prevYaw': 0.0,
      'inputJoints': ['b_head', 'b_r_wrist', 'b_l_wrist'], 'layout': 'channel-major flat[c*T + t]',
      'frameLayout': '[head x y z, head quat x y z w, right wrist x y z, left wrist x y z]',
      'fp32Model': os.path.relpath(models.mapping_path, os.getcwd()) if models.mapping_path else None,
      'int8Model': 'webxr/models/free/mapping_leader.int8.onnx',
    },
    'framesDP': r7(frames),
    'framesS': r7(dp.frame_dp_to_s(frames)),
    'roots': {'rootPos': r7(info['root_pos']), 'yaw': r7(info['yaw']), 'rm': r7(info['rm']),
              'rmRef': r7(info['rm_ref']), 'rmRel': r7(info['rm_rel']), 'local': r7(info['local'])},
    'inputs': {k: r7(v) for k, v in feeds.items()},
    'outputsFp32': {k: r7(v) for k, v in outputs.items()},
    'decoded': {'rm': r7(dec['rm']), 'rmRel': r7(dec['rm_rel']), 'localPositions': r7(dec['local']),
                'worldPositionsDP': r7(dec['world']), 'worldPositionsS': r7(dp.dp_to_s_pos(dec['world']))},
    'groundTruth': {'futureWorldDP': r7(gt_world), 'futureDatasetFrames': [int(i) for i in fidx],
                    'meanErrorCm': r7(np.linalg.norm(dec['world'] - gt_world, axis=-1).mean() * 100)},
  }


# --------------------------------------------------------------------------------------------
# tracking_io.json
# --------------------------------------------------------------------------------------------

def tracking_fixture(clip, models, pivot, clip_name, skeleton):
  idx = dp.sampled_indices(pivot, dp.N_LEADER_TRACKING)
  frames = a(dp.clip_three_point_frames(clip, idx))
  root_pos, yaw = dp.frame_roots(frames, 0.0)
  leader_rm = dp.rm_from_root(root_pos, yaw)
  leader_local = dp.three_point_local(frames, root_pos, yaw)
  pose = dp.clip_pose(clip, pivot)
  pose = {k: a(v) for k, v in pose.items()}
  avatar = dp.avatar_from_pose(pose, a(leader_rm[0]))
  feeds = dp.tracking_inputs(leader_rm, leader_local, avatar)
  feeds = {k: a(v).astype(np.float32) for k, v in feeds.items()}
  outputs = models.run_tracking(feeds)
  outputs = {k: a(v).astype(np.float32) for k, v in outputs.items()}
  pred = dp.decode_tracking(outputs, 0)
  measured_rm = a(leader_rm[0])
  post = {}
  for label, rc, er in (('rc0_er0', 0.0, 0.0), ('rc035_er05', 0.35, 0.5)):
    st = dp.post_process(avatar, pred, measured_rm, skeleton, rc, er)
    post[label] = {'rootCorrection': rc, 'eulerRatio': er, 'rm': r7(st.rm), 'rootMatrix': mat16(st.matrix),
                   'worldDP': r7(st.world), 'worldS': r7(dp.dp_to_s_pos(st.world)), 'rootS': r7(dp.dp_to_s_pos(dp.rm_pos(st.rm))),
                   'positionsLocal': r7(st.positions), 'contacts': r7(st.contacts)}
  # uncorrected new root and the raw world positions before Euler blend / bone restore (debug aid)
  new_rm = dp.rm_apply(avatar.rm, pred['rm'])
  raw_world = dp.to_world(dp.rm_to_matrix(new_rm), pred['positions'])
  nxt = pivot + dp.SAMPLE_STEP
  gt_world = dp.clip_world_positions(clip, np.array([nxt]))[0]
  return {
    'format': FORMAT,
    'kind': 'tracking_io',
    'generator': 'tools/gen_fixtures.py',
    'params': {
      'clip': clip_name, 'datasetFps': dp.DATASET_FPS, 'step': dp.SAMPLE_STEP, 'modelFps': dp.FPS, 'dt': dp.DT,
      'pivotFrame': int(pivot), 'pivotSec': pivot / dp.DATASET_FPS, 'datasetFrames': [int(i) for i in idx],
      'nLeader': dp.N_LEADER_TRACKING, 'nOutputFrames': dp.N_FUTURE, 'usedOutputFrame': 0,
      'referenceChar': 1, 'leaderWindow': 'frame 0 = measured current frame t, frames 1..30 = ground-truth future (stands in for the mapping prediction)',
      'avatar': 'ground-truth pose at t (velocities/positions/rotations/contacts) placed at the measured root of t',
      'boneLengthTolerance': dp.BONE_LENGTH_TOLERANCE,
      'boneLengthRestore': 'joints in order (parents first); if |child-parent| > tol*len: child moved to that distance, same delta applied to all descendants (Unity Transform semantics)',
      'eulerBlend': 'world = lerp(M_new*p, prevWorld + R_new*v*dt, eulerRatio) before bone restore',
      'rootCorrection': 'new root (x,z) and yaw blended towards measuredRm by rootCorrection (0 = off)',
      'layout': 'channel-major flat[c*T + t]; rotations 9 per joint row-major m00 m01 m02 m10 .. m22',
      'fp32Model': os.path.relpath(models.tracking_path, os.getcwd()) if models.tracking_path else None,
      'int8Model': 'webxr/models/free/tracking_leader.int8.onnx',
    },
    'leaderFramesDP': r7(frames),
    'leaderFramesS': r7(dp.frame_dp_to_s(frames)),
    'leaderRoots': {'rootPos': r7(root_pos), 'yaw': r7(yaw), 'rm': r7(leader_rm), 'local': r7(leader_local),
                    'rmRelativeToAvatar': r7(dp.rm_relative(leader_rm, avatar.rm))},
    'avatar': {'rm': r7(avatar.rm), 'rootMatrix': mat16(avatar.matrix), 'positions': r7(avatar.positions),
               'rotations': r7(avatar.rotations.reshape(dp.N_JOINTS, 9)), 'velocities': r7(avatar.velocities),
               'contacts': r7(avatar.contacts), 'worldDP': r7(avatar.world)},
    'measuredRm': r7(measured_rm),
    'inputs': {k: r7(v) for k, v in feeds.items()},
    'outputsFp32': {k: r7(v) for k, v in outputs.items()},
    'decodedFrame0': {'rm': r7(pred['rm']), 'positions': r7(pred['positions']),
                      'rotations': r7(pred['rotations'].reshape(dp.N_JOINTS, 9)), 'velocities': r7(pred['velocities']),
                      'contacts': r7(pred['contacts']), 'newRmUncorrected': r7(new_rm), 'rawWorldDP': r7(raw_world)},
    'postProcessed': post,
    'skeleton': {'joints': skeleton['joints'], 'parents': skeleton['parents'], 'boneLengths': r7(skeleton['boneLengths'])},
    'groundTruth': {'nextWorldDP': r7(gt_world), 'nextDatasetFrame': int(nxt),
                    'meanErrorCm_rc0_er0': r7(np.linalg.norm(a(post['rc0_er0']['worldDP']) - gt_world, axis=-1).mean() * 100)},
  }


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--clip', required=True)
  ap.add_argument('--ckpt', required=True, help='fp32 checkpoint dir with mapping_leader/model.onnx, tracking_leader/model.onnx')
  ap.add_argument('--models', default=dp.default_models_dir())
  ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests', 'fixtures'))
  ap.add_argument('--pivot-sec', type=float, default=60.0, help='time of the "current" frame t in the clip')
  ap.add_argument('--clip-name', default='Freestyle_Solo_01_male_ROM_20221018_PM_01 (Dance-All-2-3pt, Standard)')
  args = ap.parse_args(argv)
  os.makedirs(args.out, exist_ok=True)
  clip = dp.load_clip(args.clip)
  models = dp.OnnxModels(args.models, os.path.join(args.ckpt, 'mapping_leader', 'model.onnx'),
                         os.path.join(args.ckpt, 'tracking_leader', 'model.onnx'))
  pivot = int(round(args.pivot_sec * dp.DATASET_FPS))
  write(os.path.join(args.out, 'math.json'), {
    'format': FORMAT, 'kind': 'math', 'generator': 'tools/gen_fixtures.py',
    'conventions': {
      'space': 'DP (Unity, left-handed, Y up, forward +Z) unless suffixed S (stage: right-handed, forward -Z)',
      'quaternion': '[x, y, z, w]', 'matrix': 'row-major 4x4 (16 values) or 3x3 (9 values)',
      'rm': '[cos phi, sin phi, x, z] with phi = -yaw; yaw = atan2(fwd.x, fwd.z); M = T(x,0,z) * R_y(yaw)',
      'rootFromHead': 'rootPos = (head.x, 0, head.z); forward = quat * (0,0,1); |forward.xz| < 1e-3 -> prevYaw',
      'rmRelative': 'rot_rel = rot/rot_ref, pos_rel = (pos - pos_ref)/rot_ref (complex, x + i z)',
      'rmApply': 'rot = rot_rel*rot_ref, pos = pos_rel*rot_ref + pos_ref',
      'dpToS': 'pos (x,y,-z); quat (-x,-y,z,w)',
      'rounding': 'float32 rounded to 7 significant digits; expectations computed from the rounded inputs',
    },
    'cases': math_cases(clip),
  })
  write(os.path.join(args.out, 'mapping_io.json'), mapping_fixture(clip, models, pivot, args.clip_name))
  write(os.path.join(args.out, 'tracking_io.json'), tracking_fixture(clip, models, pivot, args.clip_name, models.skeleton))
  return 0


if __name__ == '__main__':
  sys.exit(main())
