#!/usr/bin/env python3
"""Precompute the teacher avatar's full body for a choreography (DESIGN.md 5 / 6.4).

Responsibility
  --choreo mode: run the tracking network offline over a choreography's 3-point frames (the
    leader window of tick t is the actual frames t..t+30, no mapping network; start pose from
    init_pose.json placed at the root of frame 0) and write `fullBody` (base64 float32 joint
    positions in stage frame S) into the choreography JSON.
  --from-dataset mode: cut an excerpt of a Dance-All-2-3pt clip into a new choreography
    (3-point frames in S relative to the first frame's root, scaled to referenceHeight) with the
    ground-truth full body (`--gt-fullbody`, used for webxr/choreos/mocap-freestyle.json) or the
    network's full body, and report the network-vs-ground-truth error (`--compare-net`).
Must not: ship copyrighted audio; alter anything but `fullBody` and `meta.fullBody*` in --choreo mode.

Usage:
  python tools/precompute_teacher.py --choreo webxr/choreos/x.json --models webxr/models/free \
      [--fp32 <ckpt dir>] --out webxr/choreos/x.json [--root-correction 0.35] [--euler-ratio 0.5]
  python tools/precompute_teacher.py --from-dataset <clip.npy> --start-sec 24 --duration-sec 30 \
      --models webxr/models/free [--fp32 <ckpt dir>] --out webxr/choreos/mocap-freestyle.json \
      [--gt-fullbody] [--compare-net] [--id mocap-freestyle] [--bpm 100] [--reference-height 1.70]
"""

import argparse
import base64
import datetime
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dp_pipeline as dp  # noqa: E402

CHOREO_FORMAT = 'dancing-points-choreo/1'


def encode_positions(world_s):
  """(N, 34, 3) float -> base64 of float32 little-endian, frame-major."""
  arr = np.ascontiguousarray(np.asarray(world_s, dtype=np.float32).reshape(-1))
  return base64.b64encode(arr.astype('<f4').tobytes()).decode('ascii')


def decode_positions(b64, n_joints=dp.N_JOINTS):
  arr = np.frombuffer(base64.b64decode(b64), dtype='<f4').astype(np.float64)
  return arr.reshape(-1, n_joints, 3)


def full_body_block(world_s, skeleton):
  return {
    'joints': list(skeleton['joints']),
    'parents': list(skeleton['parents']),
    'fps': dp.FPS,
    'encoding': 'base64-float32',
    'positions': encode_positions(world_s),
  }


def choreo_frames_to_dp(choreo, net_scale=1.0):
  """Choreography frames (S, at referenceHeight) -> DP 3-point frames (N, 13) in metres."""
  head = np.asarray(choreo['frames']['head'], dtype=np.float64)
  left = np.asarray(choreo['frames']['left'], dtype=np.float64)
  right = np.asarray(choreo['frames']['right'], dtype=np.float64)
  n = head.shape[0]
  frames = np.zeros((n, dp.FRAME_DIM))
  frames[:, dp.FRAME_HEAD] = dp.s_to_dp_pos(head[:, :3]) / net_scale
  frames[:, dp.FRAME_QUAT] = dp.quat_normalize(dp.s_to_dp_quat(head[:, 3:7]))
  frames[:, dp.FRAME_RW] = dp.s_to_dp_pos(right) / net_scale
  frames[:, dp.FRAME_LW] = dp.s_to_dp_pos(left) / net_scale
  return frames


def load_models(models_dir, fp32_dir=None, char_set='leader'):
  mapping_path = tracking_path = None
  if fp32_dir:
    mapping_path = os.path.join(fp32_dir, 'mapping_%s' % char_set, 'model.onnx')
    tracking_path = os.path.join(fp32_dir, 'tracking_%s' % char_set, 'model.onnx')
  return dp.OnnxModels(models_dir, mapping_path, tracking_path, char_set=char_set)


def run_teacher(frames_dp, models, root_correction, euler_ratio, net_scale=1.0):
  """Tracking-only closed loop; returns world joint positions in S at choreography scale."""
  res = dp.run_future_tracking(frames_dp, models, root_correction, euler_ratio, progress=True)
  return dp.dp_to_s_pos(res['world'] * net_scale), res


def report_error(world_a, world_b, label):
  err = np.linalg.norm(np.asarray(world_a) - np.asarray(world_b), axis=-1)
  print('%s: mean joint error %.2f cm (max %.2f cm); head %.2f cm, wrists %.2f cm' % (
    label, err.mean() * 100, err.max() * 100, err[:, dp.HEAD].mean() * 100,
    err[:, [dp.L_WRIST, dp.R_WRIST]].mean() * 100))
  return float(err.mean())


# --------------------------------------------------------------------------------------------
# --choreo mode
# --------------------------------------------------------------------------------------------

def run_choreo_mode(args):
  choreo = dp.load_json(args.choreo)
  if choreo.get('format') != CHOREO_FORMAT:
    raise ValueError('unexpected choreo format %r' % choreo.get('format'))
  meta = choreo.setdefault('meta', {})
  net_scale = args.net_scale if args.net_scale else float(meta.get('scale', 1.0))
  models = load_models(args.models, args.fp32)
  frames = choreo_frames_to_dp(choreo, net_scale)
  print('choreo %s: %d frames @ %d fps, net scale %.4f, model %s' % (
    choreo.get('id'), frames.shape[0], choreo.get('fps', dp.FPS), net_scale, os.path.basename(models.tracking_path)))
  world_s, _ = run_teacher(frames, models, args.root_correction, args.euler_ratio, net_scale)
  existing = choreo.get('fullBody')
  if existing and meta.get('fullBodySource') == 'ground-truth':
    gt = decode_positions(existing['positions'])
    report_error(world_s, gt, 'network vs ground-truth full body')
    if not args.replace:
      print('keeping the ground-truth fullBody (use --replace to overwrite it with the network output)')
      if args.out != args.choreo:
        with open(args.out, 'w', encoding='utf-8') as f:
          json.dump(choreo, f, separators=(',', ':'))
        print('wrote %s (unchanged fullBody)' % args.out)
      return 0
  choreo['fullBody'] = full_body_block(world_s, models.skeleton)
  meta['fullBodySource'] = 'tracking_leader ' + ('fp32' if args.fp32 else 'int8')
  meta['fullBodyParams'] = {'rootCorrection': args.root_correction, 'eulerRatio': args.euler_ratio,
                            'netScale': net_scale, 'model': os.path.basename(models.tracking_path),
                            'tool': 'tools/precompute_teacher.py'}
  with open(args.out, 'w', encoding='utf-8') as f:
    json.dump(choreo, f, separators=(',', ':'))
  print('wrote %s (%.1f KB, fullBody %d frames)' % (args.out, os.path.getsize(args.out) / 1024, frames.shape[0]))
  return 0


# --------------------------------------------------------------------------------------------
# --from-dataset mode
# --------------------------------------------------------------------------------------------

def standing_head_height(clip, min_contact=0.5, percentile=95):
  """Head height of an upright stance: high percentile of the head height over frames where all
  four foot contacts are set (both feet flat on the floor)."""
  mask = (clip['contacts'] >= min_contact).all(axis=1)
  heights = clip['pos'][:, dp.HEAD, 1]
  if mask.sum() < 100:
    mask = np.ones_like(mask)
  return float(np.percentile(heights[mask], percentile))


def excerpt_to_choreo(clip, start, count, args, models=None):
  idx = dp.sampled_indices(start, count)
  if idx[-1] >= clip['n']:
    raise ValueError('excerpt exceeds the clip (%d frames needed, %d available)' % (idx[-1] + 1, clip['n']))
  frames_dp = dp.clip_three_point_frames(clip, idx)          # DP world, metres
  gt_world = dp.clip_world_positions(clip, idx)               # DP world joints, metres
  h0 = standing_head_height(clip)
  k = args.reference_height / h0
  # express everything relative to the first frame's root (origin, facing +Z in DP = -Z in S)
  root_pos, yaw = dp.frame_roots(frames_dp, 0.0)
  m0 = dp.root_matrix(root_pos[0], yaw[0])
  r0_inv = dp.rot_y(-yaw[0])
  rel = np.zeros_like(frames_dp)
  rel[:, dp.FRAME_HEAD] = dp.to_local(m0, frames_dp[:, dp.FRAME_HEAD]) * k
  rel[:, dp.FRAME_RW] = dp.to_local(m0, frames_dp[:, dp.FRAME_RW]) * k
  rel[:, dp.FRAME_LW] = dp.to_local(m0, frames_dp[:, dp.FRAME_LW]) * k
  for i in range(rel.shape[0]):
    rel[i, dp.FRAME_QUAT] = dp.mat_to_quat(r0_inv @ dp.quat_to_mat(frames_dp[i, dp.FRAME_QUAT]))
  rel_s = dp.frame_dp_to_s(rel)
  gt_rel_s = dp.dp_to_s_pos(dp.to_local(m0, gt_world.reshape(-1, 3)).reshape(gt_world.shape) * k)

  duration_beats = int(round(args.duration_sec * args.bpm / 60.0))
  n_frames = int(round(duration_beats * 60.0 / args.bpm * dp.FPS)) + 1
  if n_frames != count:
    raise ValueError('frame count %d does not match durationBeats %d (%d frames)' % (count, duration_beats, n_frames))
  moves = []
  b = 0
  i = 1
  while b < duration_beats:
    e = min(b + 8, duration_beats)
    moves.append({'name': 'Teil %d' % i, 'startBeat': b, 'endBeat': e, 'hint': 'Bewegung nachmachen'})
    b = e
    i += 1

  def rnd(x, d=4):
    return [[round(float(v), d) for v in row] for row in np.asarray(x)]

  head = np.concatenate([rel_s[:, dp.FRAME_HEAD], rel_s[:, dp.FRAME_QUAT]], axis=1)
  choreo = {
    'format': CHOREO_FORMAT,
    'id': args.id,
    'title': args.title,
    'artist': args.artist,
    'bpm': args.bpm, 'beatsPerBar': 4, 'countInBeats': 4,
    'durationBeats': duration_beats,
    'fps': dp.FPS,
    'referenceHeight': args.reference_height,
    'mirror': False,
    'difficulty': args.difficulty,
    'audio': {'url': None, 'synth': 'hiphop', 'offsetSec': 0.0, 'gain': 0.8},
    'moves': moves,
    'frames': {'head': rnd(head), 'left': rnd(rel_s[:, dp.FRAME_LW]), 'right': rnd(rel_s[:, dp.FRAME_RW])},
    'meta': {
      'source': 'mocap',
      'createdAt': datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
      'author': 'tools/precompute_teacher.py --from-dataset',
      'notes': ('Excerpt of the research dataset Dance-All-2-3pt (Dancing Points, Li et al. 2026), clip %s, '
                'dataset frames %d..%d (120 fps, sub-sampled to 30 fps), leader character. For demo and '
                'testing only, not a choreography to a song; the motion is not beat-aligned. / '
                'Ausschnitt aus dem Forschungsdatensatz, nur fuer Demo und Tests.'
                % (args.clip_name, int(idx[0]), int(idx[-1]))),
      'clip': args.clip_name,
      'datasetStartFrame': int(idx[0]),
      'datasetEndFrame': int(idx[-1]),
      'datasetStep': dp.SAMPLE_STEP,
      'standingHeadHeight': round(h0, 4),
      'scale': round(k, 6),
    },
  }
  skeleton = models.skeleton if models is not None else dp.load_json(os.path.join(args.models, 'skeleton.json'))
  net_world_s = None
  if models is not None and (args.compare_net or not args.gt_fullbody):
    # the network runs in the original dataset world (metres); express its result like the ground
    # truth: relative to frame 0's root and scaled to the reference height
    _, res = run_teacher(frames_dp, models, args.root_correction, args.euler_ratio, 1.0)
    net_world_s = dp.dp_to_s_pos(dp.to_local(m0, res['world'].reshape(-1, 3)).reshape(gt_world.shape) * k)
    report_error(net_world_s, gt_rel_s, 'network (%s) vs ground-truth full body, at choreography scale'
                 % os.path.basename(models.tracking_path))
  if args.gt_fullbody or net_world_s is None:
    choreo['fullBody'] = full_body_block(gt_rel_s, skeleton)
    choreo['meta']['fullBodySource'] = 'ground-truth'
  else:
    choreo['fullBody'] = full_body_block(net_world_s, skeleton)
    choreo['meta']['fullBodySource'] = 'tracking_leader ' + ('fp32' if args.fp32 else 'int8')
    choreo['meta']['fullBodyParams'] = {'rootCorrection': args.root_correction, 'eulerRatio': args.euler_ratio,
                                        'netScale': k, 'model': os.path.basename(models.tracking_path),
                                        'tool': 'tools/precompute_teacher.py'}
  print('standing head height %.4f m -> scale %.4f; %d frames, %d beats, %d moves; head height (S) min %.2f max %.2f m' % (
    h0, k, count, duration_beats, len(moves), head[:, 1].min(), head[:, 1].max()))
  return choreo


def run_dataset_mode(args):
  clip = dp.load_clip(args.from_dataset)
  duration_beats = int(round(args.duration_sec * args.bpm / 60.0))
  count = int(round(duration_beats * 60.0 / args.bpm * dp.FPS)) + 1
  start = int(round(args.start_sec * dp.DATASET_FPS))
  models = None
  if args.compare_net or not args.gt_fullbody:
    models = load_models(args.models, args.fp32)
  choreo = excerpt_to_choreo(clip, start, count, args, models)
  with open(args.out, 'w', encoding='utf-8') as f:
    json.dump(choreo, f, separators=(',', ':'))
  print('wrote %s (%.1f KB)' % (args.out, os.path.getsize(args.out) / 1024))
  return 0


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--choreo', help='choreography JSON to process (--choreo mode)')
  ap.add_argument('--from-dataset', help='(N, 518) float32 clip npy (--from-dataset mode)')
  ap.add_argument('--models', default=dp.default_models_dir())
  ap.add_argument('--fp32', default=None, help='fp32 checkpoint dir (dancing/<style>) instead of the int8 files')
  ap.add_argument('--out', required=True)
  ap.add_argument('--root-correction', type=float, default=0.35)
  ap.add_argument('--euler-ratio', type=float, default=0.5)
  ap.add_argument('--net-scale', type=float, default=None,
                  help='divide choreo positions by this before the network (default: meta.scale or 1)')
  ap.add_argument('--replace', action='store_true', help='overwrite a ground-truth fullBody with the network output')
  # --from-dataset options
  ap.add_argument('--start-sec', type=float, default=0.0)
  ap.add_argument('--duration-sec', type=float, default=30.0)
  ap.add_argument('--bpm', type=int, default=100)
  ap.add_argument('--id', default='mocap-freestyle')
  ap.add_argument('--title', default='Freestyle (Mocap-Demo)')
  ap.add_argument('--artist', default='Dancing Points Datensatz')
  ap.add_argument('--difficulty', type=int, default=3)
  ap.add_argument('--reference-height', type=float, default=1.70)
  ap.add_argument('--clip-name', default='Freestyle_Solo_01_male_ROM_20221018_PM_01')
  ap.add_argument('--gt-fullbody', action='store_true', help='write the ground-truth joint positions as fullBody')
  ap.add_argument('--compare-net', action='store_true', help='also run the network and print its error vs ground truth')
  args = ap.parse_args(argv)
  if bool(args.choreo) == bool(args.from_dataset):
    ap.error('give exactly one of --choreo or --from-dataset')
  if args.choreo:
    return run_choreo_mode(args)
  return run_dataset_mode(args)


if __name__ == '__main__':
  sys.exit(main())
