#!/usr/bin/env python3
"""Dancing Points runtime pipeline: numpy reference implementation (docs/DESIGN.md section 6).

Responsibility
  * coordinate frames: DP (Unity, left-handed, forward +Z) <-> S (stage frame, right-handed,
    forward -Z), quaternion helpers on plain arrays;
  * the 3-point root (head projected to the floor + head forward yaw), root motion 4-vectors
    [cos phi, sin phi, x, z] <-> 4x4 matrices, relative roots (complex plane), local <-> world;
  * assembly of the mapping / tracking network inputs (channel-major), decoding of the outputs;
  * post-processing of a tracking tick (new root, root correction, Euler blend, bone-length
    restore) and the closed-loop drivers `run_self_tracking` / `run_future_tracking`;
  * dataset helpers for the Dance-All-2-3pt clips (world reconstruction, "official" slicing that
    mirrors data_process/data_processors.py) and a `--validate` self-check.

This module is the oracle for webxr/src/net/pipeline.js and for tools/gen_fixtures.py.
It must stay numpy-only and deterministic: no torch, onnxruntime is imported lazily and only
inside `OnnxModels`. All internal maths is float64; network tensors are float32.

Conventions (see DESIGN.md 3 and 6.2)
  * A 3-point frame in DP space is a float array of FRAME_DIM = 13 values:
    [head x y z, head quaternion x y z w, right wrist x y z, left wrist x y z].
    The head forward direction is quat * (0, 0, 1)  (gaze direction of a VR headset in DP).
  * Root: pos = (head.x, 0, head.z), yaw = atan2(fwd.x, fwd.z); M = T(pos) * R_y(yaw).
  * Root motion rm = [cos phi, sin phi, x, z] with phi = -yaw (Unity RootMotionType).
  * Multi-frame tensors are channel-major: flat[c * T + t].
"""

import argparse
import json
import math
import os
import sys

import numpy as np

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------

FPS = 30
DT = 1.0 / FPS
DATASET_FPS = 120
SAMPLE_STEP = DATASET_FPS // FPS  # 4

N_JOINTS = 34
JOINT_NAMES = [
  'b_root', 'b_l_upleg', 'b_l_leg', 'b_l_foot_twist', 'b_l_foot', 'b_l_talocrural', 'b_l_subtalar',
  'b_l_transversetarsal', 'b_l_ball', 'b_r_upleg', 'b_r_leg', 'b_r_foot_twist', 'b_r_foot',
  'b_r_talocrural', 'b_r_subtalar', 'b_r_transversetarsal', 'b_r_ball', 'b_spine0', 'b_spine1',
  'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist',
  'b_neck0', 'b_head', 'b_head_null', 'b_r_shoulder', 'b_r_arm', 'b_r_forearm', 'b_r_wrist_twist',
  'b_r_wrist',
]
PARENTS = [-1, 0, 1, 2, 3, 3, 5, 6, 7, 0, 9, 10, 11, 11, 13, 14, 15, 0, 17, 18, 19, 20, 21, 22, 23,
           24, 20, 26, 27, 20, 29, 30, 31, 32]
HEAD = 27
R_WRIST = 33
L_WRIST = 25
INPUT_JOINTS = (HEAD, R_WRIST, L_WRIST)  # order head, right, left (metadata input_joints)
# foot contact order: [l_ankle, l_ball, r_ankle, r_ball]
CONTACT_JOINTS = (5, 8, 13, 16)

N_HISTORY = 15          # mapping input frames (t-14 .. t)
N_FUTURE = 30           # mapping output frames (t+1 .. t+30)
N_LEADER_TRACKING = 31  # tracking leader window (t .. t+30)

FRAME_DIM = 13
FRAME_HEAD = slice(0, 3)
FRAME_QUAT = slice(3, 7)
FRAME_RW = slice(7, 10)
FRAME_LW = slice(10, 13)

# Dance-All-2-3pt column layout (518 float32 per 120 fps frame), see Description.txt
COL_VEL = slice(0, 102)
COL_POS = slice(102, 204)
COL_ROT = slice(204, 510)
COL_RM = slice(510, 514)
COL_CONTACT = slice(514, 518)

# Fixed rotation from "VR headset axes" to the head bone's local axes. Measured on
# Freestyle_Solo_01: the bone's local +Y is forward (mean angle to root +Z = 0.0 deg), local -X is
# up (27 deg from world +Y, the dancer looks slightly down) and local -Z is right. Columns are the
# images of (right, up, forward): R_FIX @ (0,0,1) = (0,1,0), R_FIX @ (0,1,0) = (-1,0,0),
# R_FIX @ (1,0,0) = (0,0,-1), det = +1. Head quaternion of a dataset frame = quat(R_world @ R_FIX).
R_FIX = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])

DP_TO_S_POS = np.array([1.0, 1.0, -1.0])
DP_TO_S_QUAT = np.array([-1.0, -1.0, 1.0, 1.0])

YAW_EPS = 1e-3
BONE_LENGTH_TOLERANCE = 1.05

MAPPING_INPUT_NAMES = ('input_leader_Positions', 'input_leader_RootMotion')
MAPPING_OUTPUT_NAMES = ('Positions', 'RootMotion')
TRACKING_INPUT_NAMES = ('input_leader_Positions', 'input_leader_RootMotion',
                        'input_follower_VelocitiesV2', 'input_follower_Positions',
                        'input_follower_Rotations', 'input_follower_FootContactLabels',
                        'input_follower_RootMotion')
TRACKING_OUTPUT_NAMES = ('VelocitiesV2', 'Positions', 'Rotations', 'FootContactLabels', 'RootMotion')


# --------------------------------------------------------------------------------------------
# Quaternions (x, y, z, w) and small matrices
# --------------------------------------------------------------------------------------------

def quat_normalize(q):
  q = np.asarray(q, dtype=np.float64)
  return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_to_mat(q):
  """Rotation matrix (..., 3, 3) of unit quaternion(s) (..., 4) in x, y, z, w order."""
  q = np.asarray(q, dtype=np.float64)
  x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
  m[..., 0, 0] = 1 - 2 * (y * y + z * z)
  m[..., 0, 1] = 2 * (x * y - z * w)
  m[..., 0, 2] = 2 * (x * z + y * w)
  m[..., 1, 0] = 2 * (x * y + z * w)
  m[..., 1, 1] = 1 - 2 * (x * x + z * z)
  m[..., 1, 2] = 2 * (y * z - x * w)
  m[..., 2, 0] = 2 * (x * z - y * w)
  m[..., 2, 1] = 2 * (y * z + x * w)
  m[..., 2, 2] = 1 - 2 * (x * x + y * y)
  return m


def mat_to_quat(m):
  """Unit quaternion (x, y, z, w) of a rotation matrix (3, 3). Positive w."""
  m = np.asarray(m, dtype=np.float64)
  t = m[0, 0] + m[1, 1] + m[2, 2]
  if t > 0:
    s = math.sqrt(t + 1.0) * 2
    q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
  elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
    s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
    q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
  elif m[1, 1] > m[2, 2]:
    s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
    q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
  else:
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
  q = np.array(q, dtype=np.float64)
  if q[3] < 0:
    q = -q
  return q / np.linalg.norm(q)


def quat_mul(a, b):
  """Hamilton product a * b (apply b first, then a) for (..., 4) arrays."""
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
  bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
  return np.stack([
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ], axis=-1)


def quat_rotate(q, v):
  """Rotate vector(s) v (..., 3) by quaternion(s) q (..., 4)."""
  m = quat_to_mat(q)
  v = np.asarray(v, dtype=np.float64)
  return np.einsum('...ij,...j->...i', m, np.broadcast_to(v, m.shape[:-2] + (3,)))


def quat_from_axis_angle(axis, angle):
  axis = np.asarray(axis, dtype=np.float64)
  axis = axis / np.linalg.norm(axis)
  s = math.sin(angle / 2)
  return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)])


def rot_y(yaw):
  """Rotation about +Y by yaw (Unity Quaternion.Euler(0, yawDeg, 0)): +Z -> (sin, 0, cos)."""
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_y_batch(yaw):
  yaw = np.asarray(yaw, dtype=np.float64)
  c, s = np.cos(yaw), np.sin(yaw)
  m = np.zeros(yaw.shape + (3, 3), dtype=np.float64)
  m[..., 0, 0] = c
  m[..., 0, 2] = s
  m[..., 1, 1] = 1.0
  m[..., 2, 0] = -s
  m[..., 2, 2] = c
  return m


def wrap_angle(a):
  """Wrap angle(s) to (-pi, pi]."""
  return (np.asarray(a, dtype=np.float64) + math.pi) % (2 * math.pi) - math.pi


# --------------------------------------------------------------------------------------------
# DP <-> S conversion (DESIGN 3): p_DP = (x, y, -z); q_S (x,y,z,w) -> q_DP (-x, -y, z, w)
# --------------------------------------------------------------------------------------------

def dp_to_s_pos(p):
  return np.asarray(p, dtype=np.float64) * DP_TO_S_POS


def s_to_dp_pos(p):
  return np.asarray(p, dtype=np.float64) * DP_TO_S_POS


def dp_to_s_quat(q):
  return np.asarray(q, dtype=np.float64) * DP_TO_S_QUAT


def s_to_dp_quat(q):
  return np.asarray(q, dtype=np.float64) * DP_TO_S_QUAT


def frame_dp_to_s(frame):
  """Convert a 3-point frame (..., 13) from DP to S (positions and head quaternion)."""
  f = np.array(frame, dtype=np.float64, copy=True)
  f[..., FRAME_HEAD] = dp_to_s_pos(f[..., FRAME_HEAD])
  f[..., FRAME_QUAT] = dp_to_s_quat(f[..., FRAME_QUAT])
  f[..., FRAME_RW] = dp_to_s_pos(f[..., FRAME_RW])
  f[..., FRAME_LW] = dp_to_s_pos(f[..., FRAME_LW])
  return f


frame_s_to_dp = frame_dp_to_s  # the conversion is an involution


# --------------------------------------------------------------------------------------------
# Root, root motion, relative roots (DESIGN 6.2)
# --------------------------------------------------------------------------------------------

def yaw_from_forward(fwd, prev_yaw=0.0, eps=YAW_EPS):
  """Yaw of a forward vector projected onto the ground; keeps prev_yaw when degenerate."""
  x, z = float(fwd[0]), float(fwd[2])
  if math.hypot(x, z) < eps:
    return float(prev_yaw)
  return math.atan2(x, z)


def root_from_head(head_pos, head_fwd, prev_yaw=0.0):
  """3-point root: position (head.x, 0, head.z) and yaw from the head forward direction."""
  pos = np.array([float(head_pos[0]), 0.0, float(head_pos[2])])
  return pos, yaw_from_forward(head_fwd, prev_yaw)


def root_matrix(pos, yaw):
  """4x4 root matrix M = T(pos) * R_y(yaw) (pos.y is kept as given, normally 0)."""
  m = np.eye(4)
  m[:3, :3] = rot_y(yaw)
  m[:3, 3] = np.asarray(pos, dtype=np.float64)
  return m


def rm_from_root(pos, yaw):
  """Root motion [cos phi, sin phi, x, z] with phi = -yaw (Unity RootMotionType.FromMatrix4x4)."""
  pos = np.asarray(pos, dtype=np.float64)
  yaw = np.asarray(yaw, dtype=np.float64)
  return np.stack([np.cos(yaw), -np.sin(yaw), pos[..., 0], pos[..., 2]], axis=-1)


def rm_from_matrix(m):
  yaw = math.atan2(m[0, 2], m[0, 0])
  return rm_from_root(m[:3, 3], yaw)


def rm_yaw(rm):
  rm = np.asarray(rm, dtype=np.float64)
  return -np.arctan2(rm[..., 1], rm[..., 0])


def rm_pos(rm):
  rm = np.asarray(rm, dtype=np.float64)
  return np.stack([rm[..., 2], np.zeros_like(rm[..., 2]), rm[..., 3]], axis=-1)


def rm_to_matrix(rm):
  """RootMotionType.AsMatrix4x4: phi = atan2(sin, cos), M = T(x, 0, z) * R_y(-phi)."""
  rm = np.asarray(rm, dtype=np.float64)
  phi = math.atan2(rm[1], rm[0])
  return root_matrix((rm[2], 0.0, rm[3]), -phi)


def rm_to_matrices(rm):
  """Batched rm_to_matrix: (..., 4) -> (..., 4, 4)."""
  rm = np.asarray(rm, dtype=np.float64)
  yaw = rm_yaw(rm)
  m = np.zeros(rm.shape[:-1] + (4, 4), dtype=np.float64)
  m[..., :3, :3] = rot_y_batch(yaw)
  m[..., 0, 3] = rm[..., 2]
  m[..., 2, 3] = rm[..., 3]
  m[..., 3, 3] = 1.0
  return m


def _cplx(rm):
  rm = np.asarray(rm, dtype=np.float64)
  return rm[..., 0] + 1j * rm[..., 1], rm[..., 2] + 1j * rm[..., 3]


def rm_relative(rm, ref):
  """relative_motion.transform_root_motion: rot_rel = rot / rot_ref, pos_rel = (pos - pos_ref) / rot_ref."""
  rot, pos = _cplx(rm)
  rot_ref, pos_ref = _cplx(ref)
  rot_rel = rot / rot_ref
  pos_rel = (pos - pos_ref) / rot_ref
  return np.stack([rot_rel.real, rot_rel.imag, pos_rel.real, pos_rel.imag], axis=-1)


def rm_apply(ref, rel):
  """Inverse of rm_relative (Unity referenceRoot * deltaRoot): rot = rot_rel * rot_ref,
  pos = pos_rel * rot_ref + pos_ref."""
  rot_rel, pos_rel = _cplx(rel)
  rot_ref, pos_ref = _cplx(ref)
  rot = rot_rel * rot_ref
  pos = pos_rel * rot_ref + pos_ref
  return np.stack([rot.real, rot.imag, pos.real, pos.imag], axis=-1)


def to_local(m, p):
  """M^-1 * p for a rigid 4x4 M and points p (..., 3) (Unity PositionTo)."""
  r = m[:3, :3]
  t = m[:3, 3]
  return (np.asarray(p, dtype=np.float64) - t) @ r


def to_world(m, p):
  """M * p for points p (..., 3) (Unity PositionFrom)."""
  r = m[:3, :3]
  t = m[:3, 3]
  return np.asarray(p, dtype=np.float64) @ r.T + t


def to_local_batch(ms, ps):
  """ms (N, 4, 4), ps (N, K, 3) -> (N, K, 3) local points."""
  r = ms[:, :3, :3]
  t = ms[:, :3, 3]
  return np.einsum('nkj,nji->nki', ps - t[:, None, :], r)


def to_world_batch(ms, ps):
  r = ms[:, :3, :3]
  t = ms[:, :3, 3]
  return np.einsum('nij,nkj->nki', r, ps) + t[:, None, :]


# --------------------------------------------------------------------------------------------
# Tensor layout helpers
# --------------------------------------------------------------------------------------------

def channel_major(x):
  """(T, C) frame-major -> flat float32 (C*T) with flat[c*T + t] = x[t, c]."""
  x = np.asarray(x, dtype=np.float64)
  return np.ascontiguousarray(x.T).reshape(-1).astype(np.float32)


def split_channels(flat, n_channels):
  """flat (C*T) channel-major -> (T, C) float64."""
  flat = np.asarray(flat, dtype=np.float64).reshape(-1)
  t = flat.shape[0] // n_channels
  return flat.reshape(n_channels, t).T.copy()


# --------------------------------------------------------------------------------------------
# 3-point frames
# --------------------------------------------------------------------------------------------

def make_frame(head_pos, head_quat, r_wrist, l_wrist):
  f = np.empty(FRAME_DIM, dtype=np.float64)
  f[FRAME_HEAD] = head_pos
  f[FRAME_QUAT] = head_quat
  f[FRAME_RW] = r_wrist
  f[FRAME_LW] = l_wrist
  return f


def head_forward(head_quat):
  """Forward (gaze) direction of the head in DP: quat * (0, 0, 1)."""
  return quat_rotate(head_quat, np.array([0.0, 0.0, 1.0]))


def frame_roots(frames, prev_yaw=0.0):
  """Roots of a sequence of DP frames (T, 13): positions (T, 3), yaws (T,). The yaw fallback
  for a degenerate forward vector is the previous frame's yaw (prev_yaw for the first frame)."""
  frames = np.asarray(frames, dtype=np.float64)
  fwd = head_forward(frames[:, FRAME_QUAT])
  n = frames.shape[0]
  pos = np.zeros((n, 3))
  yaw = np.zeros(n)
  last = float(prev_yaw)
  for i in range(n):
    p, last = root_from_head(frames[i, FRAME_HEAD], fwd[i], last)
    pos[i] = p
    yaw[i] = last
  return pos, yaw


def three_point_local(frames, root_pos, yaw):
  """Root-local positions (T, 3, 3) of head, right wrist, left wrist."""
  frames = np.asarray(frames, dtype=np.float64)
  world = np.stack([frames[:, FRAME_HEAD], frames[:, FRAME_RW], frames[:, FRAME_LW]], axis=1)
  ms = rm_to_matrices(rm_from_root(root_pos, yaw))
  return to_local_batch(ms, world)


# --------------------------------------------------------------------------------------------
# Mapping network (DESIGN 6.3 steps 1-2)
# --------------------------------------------------------------------------------------------

def mapping_inputs(frames, prev_yaw=0.0):
  """Assemble the mapping inputs from the last N_HISTORY DP frames (oldest first).

  Returns (feeds, info): feeds = {'input_leader_Positions': float32[135],
  'input_leader_RootMotion': float32[60]}; info holds the absolute root motions 'rm' (15, 4),
  the reference 'rm_ref' (= rm[0], oldest frame), 'yaw', 'root_pos' and 'local' (15, 3, 3)."""
  frames = np.asarray(frames, dtype=np.float64)
  if frames.shape != (N_HISTORY, FRAME_DIM):
    raise ValueError('mapping_inputs expects (%d, %d) frames' % (N_HISTORY, FRAME_DIM))
  root_pos, yaw = frame_roots(frames, prev_yaw)
  local = three_point_local(frames, root_pos, yaw)
  rm = rm_from_root(root_pos, yaw)
  rm_rel = rm_relative(rm, rm[0])
  feeds = {
    'input_leader_Positions': channel_major(local.reshape(N_HISTORY, 9)),
    'input_leader_RootMotion': channel_major(rm_rel),
  }
  info = {'rm': rm, 'rm_ref': rm[0], 'rm_rel': rm_rel, 'yaw': yaw, 'root_pos': root_pos, 'local': local}
  return feeds, info


def decode_mapping(outputs, rm_ref):
  """Decode mapping outputs into absolute roots (30, 4), local (30, 3, 3) and world (30, 3, 3)
  positions of the 3 input joints for frames t+1 .. t+30."""
  local = split_channels(outputs['Positions'], 9).reshape(N_FUTURE, 3, 3)
  rm_rel = split_channels(outputs['RootMotion'], 4)
  rm_abs = rm_apply(rm_ref, rm_rel)
  world = to_world_batch(rm_to_matrices(rm_abs), local)
  return {'rm': rm_abs, 'rm_rel': rm_rel, 'local': local, 'world': world}


# --------------------------------------------------------------------------------------------
# Tracking network (DESIGN 6.3 steps 3-4)
# --------------------------------------------------------------------------------------------

class AvatarState:
  """Full-body avatar state in DP space: root motion rm (4,), root-local positions (34, 3),
  root-local rotations (34, 3, 3), velocities (34, 3) in root orientation, contacts (4,) and the
  cached world positions (34, 3) of the last tick (for the Euler blend)."""

  __slots__ = ('rm', 'positions', 'rotations', 'velocities', 'contacts', 'world')

  def __init__(self, rm, positions, rotations, velocities, contacts, world=None):
    self.rm = np.array(rm, dtype=np.float64).reshape(4)
    self.positions = np.array(positions, dtype=np.float64).reshape(N_JOINTS, 3)
    self.rotations = np.array(rotations, dtype=np.float64).reshape(N_JOINTS, 3, 3)
    self.velocities = np.array(velocities, dtype=np.float64).reshape(N_JOINTS, 3)
    self.contacts = np.array(contacts, dtype=np.float64).reshape(4)
    if world is None:
      world = to_world(rm_to_matrix(self.rm), self.positions)
    self.world = np.array(world, dtype=np.float64).reshape(N_JOINTS, 3)

  @property
  def matrix(self):
    return rm_to_matrix(self.rm)

  def copy(self):
    return AvatarState(self.rm, self.positions, self.rotations, self.velocities, self.contacts, self.world)


def avatar_from_pose(pose, rm):
  """AvatarState from a pose dict (init_pose.json or clip_pose) placed at root motion rm."""
  return AvatarState(rm, pose['positions'], pose['rotations'], pose['velocities'], pose['contacts'])


def tracking_inputs(leader_rm, leader_local, avatar):
  """Assemble the tracking inputs. leader_rm (31, 4) absolute root motions of the leader window
  (frame 0 = measured current frame, 1..30 = future), leader_local (31, 3, 3) root-local
  positions of head/right/left; avatar = AvatarState (reference root, reference_char = 1)."""
  leader_rm = np.asarray(leader_rm, dtype=np.float64)
  leader_local = np.asarray(leader_local, dtype=np.float64)
  if leader_rm.shape != (N_LEADER_TRACKING, 4) or leader_local.shape != (N_LEADER_TRACKING, 3, 3):
    raise ValueError('tracking_inputs expects a %d-frame leader window' % N_LEADER_TRACKING)
  rel = rm_relative(leader_rm, avatar.rm)
  return {
    'input_leader_Positions': channel_major(leader_local.reshape(N_LEADER_TRACKING, 9)),
    'input_leader_RootMotion': channel_major(rel),
    'input_follower_VelocitiesV2': avatar.velocities.reshape(-1).astype(np.float32),
    'input_follower_Positions': avatar.positions.reshape(-1).astype(np.float32),
    'input_follower_Rotations': avatar.rotations.reshape(-1).astype(np.float32),
    'input_follower_FootContactLabels': avatar.contacts.astype(np.float32),
    'input_follower_RootMotion': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
  }


def decode_tracking(outputs, frame=0):
  """Decode one output frame of the tracking network (default 0 = t+1)."""
  return {
    'velocities': split_channels(outputs['VelocitiesV2'], 102)[frame].reshape(N_JOINTS, 3),
    'positions': split_channels(outputs['Positions'], 102)[frame].reshape(N_JOINTS, 3),
    'rotations': split_channels(outputs['Rotations'], 306)[frame].reshape(N_JOINTS, 3, 3),
    'contacts': split_channels(outputs['FootContactLabels'], 4)[frame],
    'rm': split_channels(outputs['RootMotion'], 4)[frame],
  }


def decode_tracking_all(outputs):
  """All 30 output frames: dict of arrays with a leading frame axis."""
  return {
    'velocities': split_channels(outputs['VelocitiesV2'], 102).reshape(N_FUTURE, N_JOINTS, 3),
    'positions': split_channels(outputs['Positions'], 102).reshape(N_FUTURE, N_JOINTS, 3),
    'rotations': split_channels(outputs['Rotations'], 306).reshape(N_FUTURE, N_JOINTS, 3, 3),
    'contacts': split_channels(outputs['FootContactLabels'], 4),
    'rm': split_channels(outputs['RootMotion'], 4),
  }


def correct_root(rm, measured_rm, factor):
  """Blend the (x, z) position and the yaw of rm towards measured_rm by factor."""
  if factor <= 0:
    return np.array(rm, dtype=np.float64)
  pos = rm_pos(rm)
  mpos = rm_pos(measured_rm)
  yaw = float(rm_yaw(rm))
  myaw = float(rm_yaw(measured_rm))
  pos = pos + factor * (mpos - pos)
  yaw = yaw + factor * float(wrap_angle(myaw - yaw))
  return rm_from_root(pos, yaw)


def restore_bone_lengths(world, parents, bone_lengths, tolerance=BONE_LENGTH_TOLERANCE):
  """Unity ONNXMappingAndTracking.RestoreBoneLength: for every joint (parents first) whose
  distance to its parent exceeds tolerance * boneLength, move it to that distance along the
  parent->child direction. The joint is a Unity Transform, so the same displacement is applied
  to all of its descendants (subtree shift)."""
  world = np.array(world, dtype=np.float64, copy=True)
  n = world.shape[0]
  children = [[] for _ in range(n)]
  for j in range(n):
    if parents[j] >= 0:
      children[parents[j]].append(j)
  for j in range(1, n):
    p = parents[j]
    if p < 0:
      continue
    d = world[j] - world[p]
    length = float(np.linalg.norm(d))
    max_len = tolerance * float(bone_lengths[j])
    if length > max_len:
      new = world[p] + d * (max_len / length)
      delta = new - world[j]
      stack = [j]
      while stack:
        k = stack.pop()
        world[k] += delta
        stack.extend(children[k])
  return world


def post_process(avatar, pred, measured_rm, skeleton, root_correction=0.35, euler_ratio=0.5, dt=DT):
  """One tracking tick post-processing (DESIGN 6.3 step 4). pred = decode_tracking(...);
  measured_rm = root motion of the measured current 3-point frame. Returns the new AvatarState
  (positions re-expressed locally to the new root; rotations/velocities verbatim; contacts
  clamped to [0, 1]; world = final world positions)."""
  new_rm = rm_apply(avatar.rm, pred['rm'])
  new_rm = correct_root(new_rm, measured_rm, root_correction)
  m = rm_to_matrix(new_rm)
  world = to_world(m, pred['positions'])
  if euler_ratio > 0:
    momentum = avatar.world + (pred['velocities'] @ m[:3, :3].T) * dt
    world = world + euler_ratio * (momentum - world)
  world = restore_bone_lengths(world, skeleton['parents'], skeleton['boneLengths'])
  positions = to_local(m, world)
  contacts = np.clip(pred['contacts'], 0.0, 1.0)
  return AvatarState(new_rm, positions, pred['rotations'], pred['velocities'], contacts, world)


# --------------------------------------------------------------------------------------------
# Model wrapper and closed-loop drivers
# --------------------------------------------------------------------------------------------

def load_json(path):
  with open(path, 'r', encoding='utf-8') as f:
    return json.load(f)


def default_models_dir():
  return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'webxr', 'models', 'free')


class OnnxModels:
  """onnxruntime sessions for mapping_<set> and tracking_<set> plus skeleton/init pose.

  models_dir: directory with meta.json, skeleton.json, init_pose.json and the int8 files.
  mapping_path / tracking_path override the ONNX files (e.g. fp32 checkpoints)."""

  def __init__(self, models_dir=None, mapping_path=None, tracking_path=None, char_set='leader',
               threads=1):
    import onnxruntime as ort  # lazy: keep the module importable without onnxruntime
    self.models_dir = models_dir or default_models_dir()
    self.skeleton = load_json(os.path.join(self.models_dir, 'skeleton.json'))
    self.init_pose = load_json(os.path.join(self.models_dir, 'init_pose.json'))
    meta_path = os.path.join(self.models_dir, 'meta.json')
    self.meta = load_json(meta_path) if os.path.exists(meta_path) else None
    if mapping_path is None:
      mapping_path = os.path.join(self.models_dir, 'mapping_%s.int8.onnx' % char_set)
    if tracking_path is None:
      tracking_path = os.path.join(self.models_dir, 'tracking_%s.int8.onnx' % char_set)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    self.mapping_path = mapping_path
    self.tracking_path = tracking_path
    self._mapping = ort.InferenceSession(mapping_path, so, providers=['CPUExecutionProvider']) \
      if mapping_path and os.path.exists(mapping_path) else None
    self._tracking = ort.InferenceSession(tracking_path, so, providers=['CPUExecutionProvider']) \
      if tracking_path and os.path.exists(tracking_path) else None

  @staticmethod
  def _run(session, feeds):
    names = [i.name for i in session.get_inputs()]
    batch = {n: np.asarray(feeds[n], dtype=np.float32).reshape(1, -1) for n in names}
    outs = session.run(None, batch)
    return {o.name: np.asarray(v, dtype=np.float32).reshape(-1) for o, v in zip(session.get_outputs(), outs)}

  def run_mapping(self, feeds):
    if self._mapping is None:
      raise RuntimeError('mapping model not loaded: %s' % self.mapping_path)
    return self._run(self._mapping, feeds)

  def run_tracking(self, feeds):
    if self._tracking is None:
      raise RuntimeError('tracking model not loaded: %s' % self.tracking_path)
    return self._run(self._tracking, feeds)


def run_self_tracking(frames, models, root_correction=0.35, euler_ratio=0.5, history=N_HISTORY,
                      use_measured_frame=True, init_rm=None, progress=False):
  """Closed-loop self tracking of a DP 3-point trajectory frames (N, 13) at 30 fps.

  Per tick t: mapping on frames t-14..t (history pre-filled with frame 0, as the Unity demo) ->
  tracking on [measured t, 30 predicted] with the avatar as reference -> post-processing.
  Returns dict: 'world' (N, 34, 3) DP world joint positions, 'rm' (N, 4) avatar root after the
  tick, 'head_rm' (N, 4) measured 3-point root at t, 'mapping_world' (N, 30, 3, 3) predicted
  future 3-point world positions, 'contacts' (N, 4)."""
  frames = np.asarray(frames, dtype=np.float64)
  n = frames.shape[0]
  skeleton = models.skeleton
  root_pos, yaw = frame_roots(frames, 0.0)
  head_rm = rm_from_root(root_pos, yaw)
  if init_rm is None:
    init_rm = head_rm[0]
  avatar = avatar_from_pose(models.init_pose, init_rm)
  world = np.zeros((n, N_JOINTS, 3))
  rms = np.zeros((n, 4))
  contacts = np.zeros((n, 4))
  mapping_world = np.zeros((n, N_FUTURE, 3, 3))
  for t in range(n):
    if progress and t % 30 == 0:
      print('  tick %d / %d' % (t, n), file=sys.stderr)
    idx = np.clip(np.arange(t - history + 1, t + 1), 0, n - 1)
    window = frames[idx]
    feeds, info = mapping_inputs(window, prev_yaw=yaw[max(t - history, 0)])
    mapped = decode_mapping(models.run_mapping(feeds), info['rm_ref'])
    mapping_world[t] = mapped['world']
    if use_measured_frame:
      leader_rm = np.concatenate([head_rm[t][None, :], mapped['rm']], axis=0)
      leader_local = np.concatenate([info['local'][-1][None], mapped['local']], axis=0)
    else:  # Unity demo: duplicate prediction 0
      leader_rm = np.concatenate([mapped['rm'][:1], mapped['rm']], axis=0)
      leader_local = np.concatenate([mapped['local'][:1], mapped['local']], axis=0)
    tfeeds = tracking_inputs(leader_rm, leader_local, avatar)
    pred = decode_tracking(models.run_tracking(tfeeds), 0)
    avatar = post_process(avatar, pred, head_rm[t], skeleton, root_correction, euler_ratio)
    world[t] = avatar.world
    rms[t] = avatar.rm
    contacts[t] = avatar.contacts
  return {'world': world, 'rm': rms, 'head_rm': head_rm, 'mapping_world': mapping_world, 'contacts': contacts}


def run_future_tracking(frames, models, root_correction=0.35, euler_ratio=0.5, init_rm=None, progress=False):
  """Teacher mode (tools/precompute_teacher.py): tracking only, the leader window of tick t is
  the actual frames t .. t+30 (clamped at the end). Same return dict as run_self_tracking
  (without 'mapping_world')."""
  frames = np.asarray(frames, dtype=np.float64)
  n = frames.shape[0]
  skeleton = models.skeleton
  root_pos, yaw = frame_roots(frames, 0.0)
  head_rm = rm_from_root(root_pos, yaw)
  local_all = three_point_local(frames, root_pos, yaw)
  if init_rm is None:
    init_rm = head_rm[0]
  avatar = avatar_from_pose(models.init_pose, init_rm)
  world = np.zeros((n, N_JOINTS, 3))
  rms = np.zeros((n, 4))
  contacts = np.zeros((n, 4))
  for t in range(n):
    if progress and t % 30 == 0:
      print('  tick %d / %d' % (t, n), file=sys.stderr)
    idx = np.clip(np.arange(t, t + N_LEADER_TRACKING), 0, n - 1)
    tfeeds = tracking_inputs(head_rm[idx], local_all[idx], avatar)
    pred = decode_tracking(models.run_tracking(tfeeds), 0)
    avatar = post_process(avatar, pred, head_rm[t], skeleton, root_correction, euler_ratio)
    world[t] = avatar.world
    rms[t] = avatar.rm
    contacts[t] = avatar.contacts
  return {'world': world, 'rm': rms, 'head_rm': head_rm, 'contacts': contacts}


# --------------------------------------------------------------------------------------------
# Dataset helpers (Dance-All-2-3pt clips, 120 fps, DP space)
# --------------------------------------------------------------------------------------------

def load_clip(path):
  """Load a (N, 518) float32 clip -> dict of float64 arrays: vel/pos (N, 34, 3), rot (N, 34, 3, 3),
  rm (N, 4), contacts (N, 4), n."""
  data = np.load(path)
  if data.ndim != 2 or data.shape[1] != 518:
    raise ValueError('expected a (N, 518) array, got %s' % (data.shape,))
  n = data.shape[0]
  return {
    'vel': data[:, COL_VEL].astype(np.float64).reshape(n, N_JOINTS, 3),
    'pos': data[:, COL_POS].astype(np.float64).reshape(n, N_JOINTS, 3),
    'rot': data[:, COL_ROT].astype(np.float64).reshape(n, N_JOINTS, 3, 3),
    'rm': data[:, COL_RM].astype(np.float64),
    'contacts': data[:, COL_CONTACT].astype(np.float64),
    'n': n,
    'raw': data,
  }


def clip_root_matrices(clip, idx=None):
  rm = clip['rm'] if idx is None else clip['rm'][idx]
  return rm_to_matrices(rm)


def clip_world_positions(clip, idx=None):
  """World joint positions (n, 34, 3) = M_root * local."""
  pos = clip['pos'] if idx is None else clip['pos'][idx]
  return to_world_batch(clip_root_matrices(clip, idx), pos)


def clip_head_world_rot(clip, idx=None):
  """World head rotation (n, 3, 3) = R_y(yaw) @ R_local_head."""
  rm = clip['rm'] if idx is None else clip['rm'][idx]
  rot = clip['rot'] if idx is None else clip['rot'][idx]
  return np.einsum('nij,njk->nik', rot_y_batch(rm_yaw(rm)), rot[:, HEAD])


def clip_three_point_frames(clip, idx):
  """DP 3-point frames (n, 13) for dataset frame indices idx; the head quaternion is the
  world head rotation composed with R_FIX so that its +Z forward equals the head bone's +Y."""
  idx = np.asarray(idx)
  world = clip_world_positions(clip, idx)
  rot = clip_head_world_rot(clip, idx) @ R_FIX
  frames = np.zeros((len(idx), FRAME_DIM))
  frames[:, FRAME_HEAD] = world[:, HEAD]
  frames[:, FRAME_RW] = world[:, R_WRIST]
  frames[:, FRAME_LW] = world[:, L_WRIST]
  for i in range(len(idx)):
    frames[i, FRAME_QUAT] = mat_to_quat(rot[i])
  return frames


def clip_pose(clip, i):
  """Pose dict (positions, rotations, velocities, contacts) at dataset frame i."""
  return {
    'positions': clip['pos'][i],
    'rotations': clip['rot'][i],
    'velocities': clip['vel'][i],
    'contacts': clip['contacts'][i],
  }


def sampled_indices(start, count, step=SAMPLE_STEP):
  return start + step * np.arange(count)


def official_mapping_inputs(clip, pivot, step=SAMPLE_STEP):
  """Mapping inputs sliced from the dataset features exactly like MappingProcessor.reshape_data
  (window 2 s @ 120 fps sub-sampled by 4 -> 61 frames centred on pivot; leader = first 31 frames,
  use_partial_lead 0.5 -> last 15 of them = frames pivot-14*step .. pivot; reference root =
  frame 0 of that slice; channel-major flatten). Returns (feeds, gt) with the ground-truth
  future 3-point local positions 'future_local' (30, 3, 3) and roots 'future_rm' (30, 4)."""
  idx = sampled_indices(pivot - (N_HISTORY - 1) * step, N_HISTORY, step)
  rm = clip['rm'][idx]
  local = clip['pos'][idx][:, list(INPUT_JOINTS)]
  rm_rel = rm_relative(rm, rm[0])
  feeds = {
    'input_leader_Positions': channel_major(local.reshape(N_HISTORY, 9)),
    'input_leader_RootMotion': channel_major(rm_rel),
  }
  fidx = sampled_indices(pivot + step, N_FUTURE, step)
  gt = {
    'idx': idx, 'future_idx': fidx, 'rm': rm, 'rm_ref': rm[0],
    'future_local': clip['pos'][fidx][:, list(INPUT_JOINTS)],
    'future_rm': clip['rm'][fidx],
    'future_world': clip_world_positions(clip, fidx)[:, list(INPUT_JOINTS)],
  }
  return feeds, gt


def official_tracking_inputs(clip, pivot, step=SAMPLE_STEP):
  """Tracking inputs sliced like TrackingProcessor.reshape_data (window 1 s -> 31 frames
  pivot .. pivot+30*step, reference_char 1 -> reference root = follower frame 0 = pivot;
  follower input = pose at pivot with root [1, 0, 0, 0])."""
  idx = sampled_indices(pivot, N_LEADER_TRACKING, step)
  avatar = avatar_from_pose(clip_pose(clip, pivot), clip['rm'][pivot])
  leader_local = clip['pos'][idx][:, list(INPUT_JOINTS)]
  feeds = tracking_inputs(clip['rm'][idx], leader_local, avatar)
  gt = {'idx': idx, 'avatar': avatar, 'next': pivot + step}
  return feeds, gt


# --------------------------------------------------------------------------------------------
# Validation (--validate)
# --------------------------------------------------------------------------------------------

def _fmt(x):
  return '%.6g' % x


def validate(clip_path, models, pivots_sec, loop_start_sec, loop_seconds, out=None):
  """Run the checks (a)-(e) of the lane brief and return a dict of numbers."""
  res = {}
  clip = load_clip(clip_path)
  n = clip['n']
  log = print if out is None else (lambda *a: print(*a, file=out))

  # (a) world reconstruction -> root re-derivation
  idx_all = np.arange(n)
  frames = clip_three_point_frames(clip, idx_all)
  root_pos, yaw = frame_roots(frames, 0.0)
  gt_pos = rm_pos(clip['rm'])
  gt_yaw = rm_yaw(clip['rm'])
  pos_err = np.linalg.norm(root_pos - gt_pos, axis=1)
  yaw_err = np.degrees(np.abs(wrap_angle(yaw - gt_yaw)))
  res['a_root_pos_max_err_m'] = float(pos_err.max())
  res['a_root_yaw_max_err_deg'] = float(yaw_err.max())
  res['a_root_yaw_mean_err_deg'] = float(yaw_err.mean())
  # also directly through the dataset forward (+Y of the world head rotation)
  fwd_y = clip_head_world_rot(clip) @ np.array([0.0, 1.0, 0.0])
  yaw2 = np.arctan2(fwd_y[:, 0], fwd_y[:, 2])
  res['a_root_yaw_max_err_deg_headY'] = float(np.degrees(np.abs(wrap_angle(yaw2 - gt_yaw))).max())
  log('(a) root from reconstructed head: max |pos err| = %s m, max |yaw err| = %s deg (mean %s deg); '
      'via head +Y directly: %s deg' % (_fmt(res['a_root_pos_max_err_m']), _fmt(res['a_root_yaw_max_err_deg']),
                                        _fmt(res['a_root_yaw_mean_err_deg']), _fmt(res['a_root_yaw_max_err_deg_headY'])))

  pivots = [int(round(s * DATASET_FPS)) for s in pivots_sec]
  pivots = [p for p in pivots if p - (N_HISTORY - 1) * SAMPLE_STEP >= 0 and p + N_FUTURE * SAMPLE_STEP < n]

  # (b) runtime vs official mapping inputs
  max_diff_pos = 0.0
  max_diff_rm = 0.0
  for p in pivots:
    off, gt = official_mapping_inputs(clip, p)
    win = frames[gt['idx']]
    rt, info = mapping_inputs(win, prev_yaw=0.0)
    max_diff_pos = max(max_diff_pos, float(np.abs(rt['input_leader_Positions'] - off['input_leader_Positions']).max()))
    max_diff_rm = max(max_diff_rm, float(np.abs(rt['input_leader_RootMotion'] - off['input_leader_RootMotion']).max()))
  res['b_mapping_input_max_diff_positions'] = max_diff_pos
  res['b_mapping_input_max_diff_rootmotion'] = max_diff_rm
  log('(b) mapping inputs runtime vs official over %d pivots: max |dPositions| = %s, max |dRootMotion| = %s'
      % (len(pivots), _fmt(max_diff_pos), _fmt(max_diff_rm)))

  # (c) fp32 mapping prediction vs ground-truth future
  errs_local = []
  errs_world = []
  errs_root = []
  for p in pivots:
    off, gt = official_mapping_inputs(clip, p)
    dec = decode_mapping(models.run_mapping(off), gt['rm_ref'])
    errs_local.append(np.linalg.norm(dec['local'] - gt['future_local'], axis=-1).mean())
    errs_world.append(np.linalg.norm(dec['world'] - gt['future_world'], axis=-1).mean())
    errs_root.append(np.linalg.norm(rm_pos(dec['rm']) - rm_pos(gt['future_rm']), axis=-1).mean())
  res['c_mapping_mean_local_err_cm'] = float(np.mean(errs_local) * 100)
  res['c_mapping_mean_world_err_cm'] = float(np.mean(errs_world) * 100)
  res['c_mapping_mean_root_err_cm'] = float(np.mean(errs_root) * 100)
  log('(c) mapping_leader (%s): mean 3-point error over 30 future frames: local %.2f cm, world %.2f cm, root pos %.2f cm'
      % (os.path.basename(os.path.dirname(models.mapping_path)) or models.mapping_path,
         res['c_mapping_mean_local_err_cm'], res['c_mapping_mean_world_err_cm'], res['c_mapping_mean_root_err_cm']))

  # (d) tracking with ground-truth future + ground-truth current pose
  errs_pos = []
  errs_world = []
  errs_rm_pos = []
  errs_rm_yaw = []
  errs_vel = []
  for p in pivots:
    feeds, gt = official_tracking_inputs(clip, p)
    pred = decode_tracking(models.run_tracking(feeds), 0)
    nxt = gt['next']
    errs_pos.append(np.linalg.norm(pred['positions'] - clip['pos'][nxt], axis=-1).mean())
    new_rm = rm_apply(gt['avatar'].rm, pred['rm'])
    world_pred = to_world(rm_to_matrix(new_rm), pred['positions'])
    world_gt = clip_world_positions(clip, np.array([nxt]))[0]
    errs_world.append(np.linalg.norm(world_pred - world_gt, axis=-1).mean())
    rm_delta_gt = rm_relative(clip['rm'][nxt], clip['rm'][p])
    errs_rm_pos.append(np.linalg.norm(rm_pos(pred['rm']) - rm_pos(rm_delta_gt)))
    errs_rm_yaw.append(np.degrees(abs(float(wrap_angle(rm_yaw(pred['rm']) - rm_yaw(rm_delta_gt))))))
    errs_vel.append(np.linalg.norm(pred['velocities'] - clip['vel'][nxt], axis=-1).mean())
  res['d_tracking_mean_local_pos_err_cm'] = float(np.mean(errs_pos) * 100)
  res['d_tracking_mean_world_pos_err_cm'] = float(np.mean(errs_world) * 100)
  res['d_tracking_mean_root_delta_pos_err_cm'] = float(np.mean(errs_rm_pos) * 100)
  res['d_tracking_mean_root_delta_yaw_err_deg'] = float(np.mean(errs_rm_yaw))
  res['d_tracking_mean_vel_err_m_s'] = float(np.mean(errs_vel))
  log('(d) tracking_leader: output frame 0 vs GT next frame over %d pivots: mean joint pos error local %.2f cm, '
      'world %.2f cm; root delta error pos %.2f cm, yaw %.3f deg; velocity error %.3f m/s'
      % (len(pivots), res['d_tracking_mean_local_pos_err_cm'], res['d_tracking_mean_world_pos_err_cm'],
         res['d_tracking_mean_root_delta_pos_err_cm'], res['d_tracking_mean_root_delta_yaw_err_deg'],
         res['d_tracking_mean_vel_err_m_s']))

  # (e) closed loop
  start = int(round(loop_start_sec * DATASET_FPS))
  count = int(round(loop_seconds * FPS))
  idx = sampled_indices(start, count)
  idx = idx[idx < n]
  loop_frames = frames[idx]
  gt_world = clip_world_positions(clip, idx)
  gt_rm = clip['rm'][idx]
  for label, rc, er in (('rc=0.35 er=0.5', 0.35, 0.5), ('rc=0 er=0.5', 0.0, 0.5), ('rc=0 er=0', 0.0, 0.0)):
    r = run_self_tracking(loop_frames, models, root_correction=rc, euler_ratio=er)
    drift = np.linalg.norm(rm_pos(r['rm']) - rm_pos(r['head_rm']), axis=-1)
    yaw_d = np.degrees(np.abs(wrap_angle(rm_yaw(r['rm']) - rm_yaw(r['head_rm']))))
    joint_err = np.linalg.norm(r['world'] - gt_world, axis=-1).mean(axis=1)
    key = label.replace('=', '').replace(' ', '_').replace('.', '')
    res['e_%s_root_drift_mean_cm' % key] = float(drift.mean() * 100)
    res['e_%s_root_drift_max_cm' % key] = float(drift.max() * 100)
    res['e_%s_root_yaw_drift_mean_deg' % key] = float(yaw_d.mean())
    res['e_%s_joint_err_mean_cm' % key] = float(joint_err.mean() * 100)
    res['e_%s_joint_err_last_sec_cm' % key] = float(joint_err[-FPS:].mean() * 100)
    map_err = np.linalg.norm(r['mapping_world'][:, 0] - np.stack(
      [gt_world[np.minimum(np.arange(len(idx)) + 1, len(idx) - 1)][:, j] for j in INPUT_JOINTS], axis=1), axis=-1).mean()
    res['e_%s_mapping_next_frame_err_cm' % key] = float(map_err * 100)
    log('(e) closed loop %d ticks (%s): avatar root vs head root drift mean %.2f cm (max %.2f cm), yaw %.2f deg; '
        'mean joint error vs GT %.2f cm (last second %.2f cm); mapping t+1 3-point error %.2f cm'
        % (len(idx), label, res['e_%s_root_drift_mean_cm' % key], res['e_%s_root_drift_max_cm' % key],
           res['e_%s_root_yaw_drift_mean_deg' % key], res['e_%s_joint_err_mean_cm' % key],
           res['e_%s_joint_err_last_sec_cm' % key], res['e_%s_mapping_next_frame_err_cm' % key]))
  return res


def main(argv=None):
  ap = argparse.ArgumentParser(description='Dancing Points pipeline reference (numpy). '
                               'Use --validate <clip.npy> to run the semantic checks against a dataset clip.')
  ap.add_argument('--validate', metavar='CLIP_NPY', help='(N, 518) float32 clip of Dance-All-2-3pt at 120 fps')
  ap.add_argument('--models', default=default_models_dir(), help='models dir (skeleton.json, init_pose.json, int8 onnx)')
  ap.add_argument('--ckpt', default=None, help='fp32 checkpoint dir containing mapping_leader/model.onnx and tracking_leader/model.onnx')
  ap.add_argument('--set', default='leader', choices=['leader', 'follower'])
  ap.add_argument('--pivots-sec', default='5,20,35,50,65,80,95,110,125,140,155,170,185',
                  help='comma separated pivot times (s) for checks (b)-(d)')
  ap.add_argument('--loop-start-sec', type=float, default=24.0)
  ap.add_argument('--loop-seconds', type=float, default=10.0)
  ap.add_argument('--json', default=None, help='write the validation numbers to this JSON file')
  args = ap.parse_args(argv)
  if not args.validate:
    ap.print_help()
    return 0
  mapping_path = tracking_path = None
  if args.ckpt:
    mapping_path = os.path.join(args.ckpt, 'mapping_%s' % args.set, 'model.onnx')
    tracking_path = os.path.join(args.ckpt, 'tracking_%s' % args.set, 'model.onnx')
  models = OnnxModels(args.models, mapping_path, tracking_path, char_set=args.set)
  print('mapping: %s\ntracking: %s' % (models.mapping_path, models.tracking_path))
  pivots = [float(x) for x in args.pivots_sec.split(',') if x.strip()]
  res = validate(args.validate, models, pivots, args.loop_start_sec, args.loop_seconds)
  if args.json:
    with open(args.json, 'w', encoding='utf-8') as f:
      json.dump(res, f, indent=1)
  return 0


if __name__ == '__main__':
  sys.exit(main())
