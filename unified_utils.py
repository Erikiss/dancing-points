import argparse


def model_dispatch(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='cvae')
    args, remaining_args = parser.parse_known_args(args)
    model_str = '--model_type=' + args.model_type
    if args.model_type == 'cvae':
        import models.CVAE as model
    elif args.model_type == 'mlp_pose':
        import models.MLPPoseMapping as model
    else:
        model = None

    return model, remaining_args, model_str