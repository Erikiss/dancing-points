import random

import numpy as np
import pytest
import torch

from runtime_utils import prepare_log_dir, resolve_device, seed_everything


def _random_triplet():
    return random.random(), np.random.random(), torch.rand(1).item()


def test_seed_everything_replays_all_random_sources():
    seed_everything(123)
    first = _random_triplet()
    seed_everything(123)
    second = _random_triplet()

    assert first == second


def test_negative_seed_is_rejected():
    with pytest.raises(ValueError, match='non-negative'):
        seed_everything(-1)


def test_resolve_device_auto_returns_available_device():
    device = resolve_device('auto')
    expected = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device.type == expected


def test_unavailable_cuda_is_rejected():
    if torch.cuda.is_available():
        assert resolve_device('cuda').type == 'cuda'
    else:
        with pytest.raises(RuntimeError, match='CUDA was requested'):
            resolve_device('cuda')


def test_prepare_log_dir_requires_explicit_overwrite(tmp_path):
    log_dir = tmp_path / 'run' / 'log'
    prepare_log_dir(log_dir)
    sentinel = log_dir / 'keep.txt'
    sentinel.write_text('keep')

    with pytest.raises(FileExistsError, match='overwrite_log'):
        prepare_log_dir(log_dir)
    assert sentinel.read_text() == 'keep'

    prepare_log_dir(log_dir, overwrite=True)
    assert log_dir.is_dir()
    assert not sentinel.exists()


def test_prepare_log_dir_rejects_unexpected_target(tmp_path):
    with pytest.raises(ValueError, match='unsafe'):
        prepare_log_dir(tmp_path / 'not-a-log')
