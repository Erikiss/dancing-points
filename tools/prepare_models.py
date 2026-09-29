#!/usr/bin/env python3
"""Prepare the Dancing Points ONNX checkpoints for the WebXR app (docs/DESIGN.md 6.1).

Responsibility: from the released checkpoints (directory or checkpoints.tar) produce
  webxr/models/<style>/{mapping_,tracking_}<set>.int8.onnx   dynamic int8 (Gemm/MatMul weights)
  webxr/models/<style>/meta.json      format, style, character, fps, per model: file, quantization,
                                      all ONNX metadata_props, inputs/outputs with dims
  webxr/models/<style>/skeleton.json  joints, parents, boneLengths (mean over a dataset clip),
                                      input joints and joint roles
  webxr/models/<style>/init_pose.json a standing frame of the clip (positions, rotations,
                                      zero velocities, contacts, headHeight)
Must not: need torch or a GPU; must not modify the checkpoints.

Usage:
  python tools/prepare_models.py --checkpoints <dir or checkpoints.tar> --style free \
      --set leader|follower|both --out webxr/models --clip <clip.npy>

The clip is a (N, 518) float32 array of Dance-All-2-3pt rows at 120 fps (see tools/dp_pipeline.py);
it is optional when skeleton.json / init_pose.json already exist in the output directory.
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dp_pipeline as dp  # noqa: E402

MODEL_NAMES = {
  'leader': ['mapping_leader', 'tracking_leader'],
  'follower': ['mapping_follower', 'tracking_follower'],
  'both': ['mapping_leader', 'tracking_leader', 'mapping_follower', 'tracking_follower'],
}
CHARACTER = 'Blueman-simplified'


def find_checkpoint(root, style, model_name):
  """Locate <model_name>/model.onnx for a style below root (several layouts accepted)."""
  candidates = [
    os.path.join(root, 'dancing', style, model_name, 'model.onnx'),
    os.path.join(root, style, model_name, 'model.onnx'),
    os.path.join(root, model_name, 'model.onnx'),
  ]
  for c in candidates:
    if os.path.exists(c):
      return c
  raise FileNotFoundError('no model.onnx for %s/%s below %s' % (style, model_name, root))


def extract_from_tar(tar_path, style, model_names, dest):
  """Extract only model.onnx + args.txt of the requested models from checkpoints.tar."""
  wanted = set()
  for m in model_names:
    wanted.add('dancing/%s/%s/model.onnx' % (style, m))
    wanted.add('dancing/%s/%s/args.txt' % (style, m))
  found = set()
  with tarfile.open(tar_path, 'r') as tf:
    for member in tf:
      name = member.name.lstrip('./')
      if name in wanted:
        member.name = name
        tf.extract(member, dest, filter='data')
        found.add(name)
        if found == wanted:
          break
  missing = [w for w in wanted if w not in found and w.endswith('model.onnx')]
  if missing:
    raise FileNotFoundError('missing in tar: %s' % ', '.join(missing))
  return dest


def quantize(src, dst):
  from onnxruntime.quantization import quantize_dynamic, QuantType
  quantize_dynamic(src, dst, weight_type=QuantType.QInt8, op_types_to_quantize=['Gemm', 'MatMul'])


def model_meta(onnx_path, file_name):
  import onnx
  m = onnx.load(onnx_path, load_external_data=False)
  md = {mp.key: mp.value for mp in m.metadata_props}
  ins = [{'name': i.name, 'dim': int(i.type.tensor_type.shape.dim[1].dim_value)} for i in m.graph.input]
  outs = [{'name': o.name, 'dim': int(o.type.tensor_type.shape.dim[1].dim_value)} for o in m.graph.output]
  return {'file': file_name, 'quantization': 'int8-dynamic', 'metadata': md, 'inputs': ins, 'outputs': outs}


def build_skeleton(clip_raw):
  """skeleton.json content; boneLengths = mean distance child-parent over the clip (float32 maths,
  identical to the first version of this file)."""
  pos = clip_raw[:, dp.COL_POS].reshape(-1, dp.N_JOINTS, 3)
  lens = []
  for i in range(dp.N_JOINTS):
    p = dp.PARENTS[i]
    lens.append(0.0 if p < 0 else float(np.linalg.norm(pos[:, i] - pos[:, p], axis=1).mean()))
  return {
    'format': 'dancing-points-skeleton/1',
    'character': CHARACTER,
    'joints': list(dp.JOINT_NAMES),
    'parents': list(dp.PARENTS),
    'boneLengths': lens,
    'inputJoints': ['b_head', 'b_r_wrist', 'b_l_wrist'],
    'headJoint': 'b_head',
    'leftWrist': 'b_l_wrist',
    'rightWrist': 'b_r_wrist',
    'leftAnkle': 'b_l_talocrural',
    'rightAnkle': 'b_r_talocrural',
    'leftBall': 'b_l_ball',
    'rightBall': 'b_r_ball',
    'coordinate': 'unity-left-handed-y-up (DP space)',
    'note': 'root = head position projected to floor; forward = head gaze direction projected to floor',
  }


def pick_init_frame(clip_raw, search_seconds=30):
  """Calm frame with both feet down within the first search_seconds: argmin of
  mean joint speed + 0.5 * (2 - sum of contacts)."""
  vel = clip_raw[:, dp.COL_VEL].reshape(-1, dp.N_JOINTS, 3)
  contact = clip_raw[:, dp.COL_CONTACT]
  speed = np.linalg.norm(vel, axis=2).mean(1)
  cand = np.arange(0, min(search_seconds * dp.DATASET_FPS, clip_raw.shape[0]))
  return int(cand[np.argmin(speed[cand] + 0.5 * (2 - contact[cand].sum(1)))])


def build_init_pose(clip_raw, k, clip_label):
  pos = clip_raw[:, dp.COL_POS].reshape(-1, dp.N_JOINTS, 3)
  rot = clip_raw[:, dp.COL_ROT]
  contact = clip_raw[:, dp.COL_CONTACT]
  return {
    'format': 'dancing-points-pose/1',
    'source': 'Dance-All-2-3pt %s frame %d' % (clip_label, k),
    'positions': pos[k].round(5).tolist(),
    'rotations': rot[k].reshape(dp.N_JOINTS, 9).round(5).tolist(),
    'velocities': np.zeros((dp.N_JOINTS, 3)).tolist(),
    'contacts': contact[k].tolist(),
    'headHeight': float(pos[k, dp.HEAD, 1]),
  }


def smoke_test(fp32_path, int8_path, clip_raw=None):
  """Run fp32 and int8 once on the same input and report the relative output error."""
  import onnxruntime as ort
  so = ort.SessionOptions()
  so.intra_op_num_threads = 1
  s32 = ort.InferenceSession(fp32_path, so, providers=['CPUExecutionProvider'])
  s8 = ort.InferenceSession(int8_path, so, providers=['CPUExecutionProvider'])
  rng = np.random.default_rng(0)
  feeds = {i.name: (rng.standard_normal((1, i.shape[1])) * 0.3).astype(np.float32) for i in s32.get_inputs()}
  o32 = s32.run(None, feeds)
  o8 = s8.run(None, feeds)
  report = []
  for o, a, b in zip(s32.get_outputs(), o32, o8):
    err = np.abs(a - b)
    report.append('%s rel %.4f' % (o.name, err.mean() / (np.abs(a).mean() + 1e-9)))
  return ', '.join(report)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--checkpoints', required=True, help='checkpoint directory or checkpoints.tar')
  ap.add_argument('--style', default='free', help='balboa|chacha|foxtrot|free|hustle|vwaltz')
  ap.add_argument('--set', default='leader', choices=['leader', 'follower', 'both'])
  ap.add_argument('--out', default='webxr/models', help='output root (a <style>/ folder is created)')
  ap.add_argument('--clip', default=None, help='(N, 518) float32 npy for skeleton.json / init_pose.json')
  ap.add_argument('--clip-label', default='Freestyle_Solo_01', help='clip name written into init_pose.json')
  ap.add_argument('--no-quantize', action='store_true', help='skip the int8 export (only json files)')
  ap.add_argument('--no-smoke', action='store_true', help='skip the fp32 vs int8 smoke test')
  args = ap.parse_args(argv)

  model_names = MODEL_NAMES[args.set]
  out_dir = os.path.join(args.out, args.style)
  os.makedirs(out_dir, exist_ok=True)

  tmp = None
  root = args.checkpoints
  if os.path.isfile(root) and tarfile.is_tarfile(root):
    tmp = tempfile.mkdtemp(prefix='dp-ckpt-')
    print('extracting %s from %s ...' % (', '.join(model_names), root))
    root = extract_from_tar(root, args.style, model_names, tmp)

  meta_path = os.path.join(out_dir, 'meta.json')
  meta = {'format': 'dancing-points-models/1', 'style': args.style, 'character': CHARACTER,
          'fps': dp.FPS, 'models': {}}
  if os.path.exists(meta_path):
    try:
      old = dp.load_json(meta_path)
      if old.get('style') == args.style:
        meta['models'].update(old.get('models', {}))
    except (OSError, ValueError):
      pass
  for name in model_names:
    src = find_checkpoint(root, args.style, name)
    file_name = '%s.int8.onnx' % name
    dst = os.path.join(out_dir, file_name)
    if not args.no_quantize:
      print('quantizing %s -> %s' % (src, dst))
      quantize(src, dst)
      print('  %.1f MB -> %.1f MB' % (os.path.getsize(src) / 2 ** 20, os.path.getsize(dst) / 2 ** 20))
      if not args.no_smoke:
        print('  smoke test (random input): %s' % smoke_test(src, dst))
    meta['models'][name] = model_meta(src, file_name)
  # keep a stable, readable order: leader models first
  ordered = {}
  for name in MODEL_NAMES['both']:
    if name in meta['models']:
      ordered[name] = meta['models'][name]
  meta['models'] = ordered
  with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=1)
  print('wrote %s' % meta_path)

  skel_path = os.path.join(out_dir, 'skeleton.json')
  pose_path = os.path.join(out_dir, 'init_pose.json')
  if args.clip:
    clip_raw = np.load(args.clip)
    if clip_raw.dtype != np.float32 or clip_raw.ndim != 2 or clip_raw.shape[1] != 518:
      raise ValueError('clip must be a (N, 518) float32 array')
    with open(skel_path, 'w', encoding='utf-8') as f:
      json.dump(build_skeleton(clip_raw), f, indent=1)
    print('wrote %s' % skel_path)
    k = pick_init_frame(clip_raw)
    pose = build_init_pose(clip_raw, k, args.clip_label)
    with open(pose_path, 'w', encoding='utf-8') as f:
      json.dump(pose, f)
    print('wrote %s (frame %d, head height %.4f m, contacts %s)' % (pose_path, k, pose['headHeight'], pose['contacts']))
  else:
    for p in (skel_path, pose_path):
      if not os.path.exists(p):
        print('WARNING: %s missing and no --clip given' % p)
  if tmp:
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
