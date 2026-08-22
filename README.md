# Dancing Points: Synthesizing Ballroom Dancing with Three-Point Inputs

![Python](https://img.shields.io/badge/Python->=3.11-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=2.1-Red?logo=pytorch)

This repository provides the implementation for our leader/follower ballroom-dancing mapping and
tracking networks. Given only the three-point (head + both wrists) trajectory of a VR headset and
controllers, our method predicts both the counterpart dancer's three-point trajectory and each
dancer's full-body motion, with a purely deterministic, non-generative model. It is based on our
work [Dancing Points: Synthesizing Ballroom Dancing with Three-Point Inputs](https://peizhuoli.github.io/dancing-points/).

For the Unity project for visualization and real-time playback, a separate repository is provided
[here](https://github.com/PeizhuoLi/dancing-points-unity).

```bibtex
@inproceedings{Li2026dancingpoints,
  title={Dancing Points: Synthesizing Ballroom Dancing with Three-Point Inputs},
  author={Li, Peizhuo and Starke, Sebastian and Ye, Yuting and Sorkine-Hornung, Olga},
  booktitle = {Computer Graphics Forum},
  doi = {https://doi.org/10.1111/cgf.70588},
  year = {2026}
}
```

## Prerequisites

This code has been tested under Ubuntu 20.04 with Python 3.11. Please install the following
packages (and their dependencies):

- pytorch == 2.1.1
- onnx == 1.13.1
- numpy == 1.26.0
- matplotlib == 3.8.0
- tqdm == 4.65.0
- tensorboard == 2.12.1

For a reproducible Python 3.11 environment, use the checked-in dependency files:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA systems, install the PyTorch 2.1.1 wheel matching the machine's CUDA runtime before the
remaining dependencies. Development and smoke-test dependencies are in `requirements-dev.txt`.

## Quick Start

We provide pre-trained checkpoints for six ballroom-dancing styles (balboa, cha-cha, foxtrot, free,
hustle, viennese waltz), one `mapping_leader`/`mapping_follower`/`tracking_leader`/`tracking_follower`
set per style. Download `checkpoints.tar` [here](https://drive.google.com/file/d/1l8pY825MRyPjIqVu025L07C1I_2uLlQP/view?usp=share_link)
and extract it at the root of the repository:

```bash
tar -xf checkpoints.tar -C results/release
```

so that e.g. `results/release/dancing/chacha/tracking_follower/args.txt` exists.

Download the corresponding pre-processed dataset, `datasets.tar`
[here](https://drive.google.com/file/d/1Kx1uK9pQ3h3veSlN_EolNIBHDxuUewpM/view?usp=share_link), and
extract it under `./Datasets`:

```bash
tar -xf datasets.tar -C Datasets
```

which produces `Datasets/Dance-All-2-3pt` (leader) and `Datasets/Dance-All-1` (follower) — the
combined, multi-style datasets used by every style above (see `--data_name_filter` below).
Checkpoints and datasets for [LaFAN](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
(a single `mapping`/`tracking` pair trained on the whole dataset) are not yet bundled in these
archives.

To evaluate a checkpoint and re-export its ONNX model, run:

```bash
python test_unified_autoregressive_mapping.py --save=./results/release/dancing/chacha/tracking_follower --export_onnx=1
```

`--save` also works for any `mapping_*` checkpoint. The script reads `args.txt` and the newest
`.pt` file from `--save`, rebuilds the matching dataset/model, and reports reconstruction losses.
Prebuilt `model.onnx` files for real-time playback in Unity are included with each checkpoint, and
are also shipped directly with the [Unity project](https://github.com/PeizhuoLi/dancing-points-unity).

## Training from Scratch

Each of the four networks per style/character pair — `mapping_leader`, `mapping_follower`,
`tracking_leader`, `tracking_follower` — is trained as a separate run via
`train_unified_autoregressive_mapping.py`. Every run takes `--paths=<dataset_a>,<dataset_b>`, two
dataset folder names under `./Datasets` separated by a comma; *which* dataset goes first/second
depends on the network:

| Network            | `--model_type` | `--paths`             | Predicts                                                                    |
|---------------------|-----------------|------------------------|------------------------------------------------------------------------------|
| `mapping_leader`   | `mlp_pose`      | `<leader>,<leader>`   | the leader's own future 3-point trajectory, from the leader's 3-point history |
| `mapping_follower` | `mlp_pose`      | `<leader>,<follower>` | the follower's future 3-point trajectory, from the leader's 3-point history   |
| `tracking_leader`  | `cvae`          | `<leader>,<leader>`   | the leader's full-body motion, from the leader's own future 3-point + current pose |
| `tracking_follower`| `cvae`          | `<follower>,<follower>` | the follower's full-body motion, from the follower's own future 3-point + current pose |

In other words: tracking is always self-tracking (both `--paths` entries are the same character's
data), so `<leader>`/`<follower>` above just tells you which character's dataset to point it at.
Mapping always takes the leader's data as its *first* `--paths` entry (the input); the second entry
is the *target* — the leader's own dataset again for `mapping_leader`, or the follower's dataset for
`mapping_follower`. See `option.py`, `models/CVAE.py` and `models/MLPPoseMapping.py` for the full
set of training/architecture flags — in particular `--data_name_filter` to restrict training to a
single dance style within a combined multi-style dataset.

The rest of each network's recipe (epochs, network size, learning rate, the 3-point input joints,
loss weights, ...) is baked in as the argparse defaults for `--model_type=cvae`
(`AutoregressiveCVAEOption` in `models/CVAE.py`) and `--model_type=mlp_pose`
(`OneFrameMappingOption` in `models/MLPPoseMapping.py`), so a training command only needs to state
what actually varies per run: `--paths`/`--data_name_filter` (which dataset(s)) and `--save` (where
to write it). Every flag is still overridable on the command line if you want to deviate from the
shipped recipe.

### Reproducible and non-interactive runs

Training now defaults to seed `23456`, records it in `args.txt`, seeds DataLoader workers, and uses
deterministic cuDNN behavior. `--device=auto` (or the legacy empty default) chooses CUDA when it is
available and otherwise uses CPU; pass `--device=cuda` or `--device=cpu` to require a specific
device. Change `--num_workers` when the host has different multiprocessing constraints.

If a run's TensorBoard log already exists, training exits instead of prompting or deleting it.
Only an explicit `--overwrite_log=1` replaces that run's `log/` directory. Evaluation reuses the
training seed by default; `--seed=<n>` can override it for a controlled comparison.

### Full example: training all four networks for balboa

All four networks use the same combined, multi-style `Dance-All-2-3pt` (leader) /
`Dance-All-1` (follower) datasets, restricted to one style via `--data_name_filter`. This is a
single unified recipe — swap `balboa` for any other style name (chacha, foxtrot, free, hustle,
vwaltz) in `--data_name_filter` and `--save` to reproduce that style instead, everything else
unchanged.

```bash
# tracking_leader: leader's full-body motion, from the leader's own future 3pt + current pose
python train_unified_autoregressive_mapping.py --model_type=cvae \
    --paths=Dance-All-2-3pt,Dance-All-2-3pt \
    --data_name_filter=balboa --save=./results/balboa/tracking_leader

# tracking_follower: same recipe, follower's own dataset instead of the leader's
python train_unified_autoregressive_mapping.py --model_type=cvae \
    --paths=Dance-All-1,Dance-All-1 \
    --data_name_filter=balboa --save=./results/balboa/tracking_follower

# mapping_leader: leader's own future 3pt trajectory, from the leader's 3pt history
python train_unified_autoregressive_mapping.py --model_type=mlp_pose \
    --paths=Dance-All-2-3pt,Dance-All-2-3pt \
    --data_name_filter=balboa --save=./results/balboa/mapping_leader

# mapping_follower: same recipe, follower's dataset as the second --paths entry (the target)
python train_unified_autoregressive_mapping.py --model_type=mlp_pose \
    --paths=Dance-All-2-3pt,Dance-All-1 \
    --data_name_filter=balboa --save=./results/balboa/mapping_follower
```

> **Note:** this is the recommended recipe for reproducing/retraining balboa going forward. The
> checkpoint actually shipped under `results/release/dancing/balboa/mapping_*` was trained with an
> older, balboa-specific recipe instead (standalone `Balboa-2-V3-3pt`/`Balboa-1-V3` datasets and a
> different root-position loss weight) — see the caveat under Released Checkpoints below.

Checkpoints are written to each `--save` dir every `--save_freq` epochs. After training, run
`test_unified_autoregressive_mapping.py --save=<save_dir> --export_onnx=1` on each of the four
dirs, same as Quick Start, to evaluate and export its ONNX model. At inference time, chaining
`mapping_follower`'s output into `tracking_follower`'s input (and `mapping_leader`'s into
`tracking_leader`'s) is what lets a single VR three-point stream drive both avatars.

## Released Checkpoints

```
results/release/
  dancing/<style>/{tracking_leader, tracking_follower, mapping_leader, mapping_follower}/
  lafan/{tracking, mapping}/
```

`<style>` is one of `balboa`, `chacha`, `foxtrot`, `free`, `hustle`, `vwaltz`. Each checkpoint
directory holds the training `args.txt`, the final `.pt` weights, and a `model.onnx` export.

A couple of caveats about this specific set of checkpoints:
- Balboa's `mapping_leader`/`mapping_follower` pair was trained slightly differently from the other
  five styles (an earlier sweep, different root-position loss weight); its ONNX input/output
  signature is identical to the others, so it is still a drop-in replacement, just not from the
  exact same training recipe.
- The LaFAN checkpoints were trained with a longer, cyclic-LR schedule than they ended up running
  for, so the learning-rate restart never fully completed. They are shipped as-is and still perform
  well, but a from-scratch retrain with the schedule matched to the actual epoch count would be
  expected to do slightly better.

## Original Motion Data

The preprocessed datasets above are derived from raw motion-captured skeletal animation. The
original data, in FBX format, is available [here](https://drive.google.com/file/d/1FqiUV1l014aDtSXyd-tE3aIEs70Y5rlg/view?usp=share_link).

## Tests

The CPU smoke suite covers deterministic runtime setup, safe log handling, model dispatch, MLP and
CVAE forward/loss shapes, and deterministic CVAE inference:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The same suite runs automatically for pull requests through GitHub Actions. Full dataset and
checkpoint evaluation remains an explicit integration test because those assets are distributed
separately.

## Acknowledgments

The optimizer and cyclic learning-rate scheduler in `Library/AdamWR/` are adapted from an
implementation of ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) and
["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/abs/1506.01186).
