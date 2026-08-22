import pytest

from models.MLPPoseMapping import OneFrameMappingOption
from option import TestOptionParser as EvaluationOptionParser


def test_training_defaults_are_reproducible_and_non_destructive():
    parser = OneFrameMappingOption()
    args = parser.parse_args(['--paths=leader,follower'])

    assert args.seed == 23456
    assert args.deterministic == 1
    assert args.overwrite_log == 0
    assert args.num_workers == 4


def test_training_option_validation():
    parser = OneFrameMappingOption()
    args = parser.parse_args(['--paths=leader,follower', '--seed=-1'])

    with pytest.raises(ValueError, match='seed'):
        parser.post_process(args)


def test_evaluation_defaults_to_training_seed_and_auto_device():
    args = EvaluationOptionParser().parse_args([])

    assert args.seed is None
    assert args.device == 'auto'
    assert args.num_samples == 100


def test_evaluation_option_validation():
    with pytest.raises(ValueError, match='num_samples'):
        EvaluationOptionParser().parse_args(['--num_samples=1'])
