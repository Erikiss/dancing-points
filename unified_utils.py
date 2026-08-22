import argparse


def model_dispatch(args=None):
    # Let the selected model parser render the complete --help output.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model_type', type=str, default='cvae')
    args, remaining_args = parser.parse_known_args(args)
    model_str = '--model_type=' + args.model_type
    if args.model_type == 'cvae':
        import models.CVAE as model
    elif args.model_type == 'mlp_pose':
        import models.MLPPoseMapping as model
    else:
        raise ValueError(f"Unknown model_type: {args.model_type!r}; expected 'cvae' or 'mlp_pose'")

    return model, remaining_args, model_str
