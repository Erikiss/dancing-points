import argparse
import sys
import os


class BaseOptionParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--use_tqdm', type=int, default=0)

    @staticmethod
    def checker(args):
        return args

    @staticmethod
    def text_serialize(args):
        d = vars(args)
        res = ''
        for k, v in d.items():
            res += f'--{k}={v} '
        return res

    @staticmethod
    def serialize(args):
        d = vars(args)
        return d

    def deserialize(self, d):
        args = self.parse_args('')
        args.__dict__.update(d)
        args = self.checker(args)
        return args

    def text_deserialize(self, d):
        args = self.parse_args(d)
        return args

    def parse_args(self, args_str=None):
        return self.checker(self.parser.parse_args(args_str))

    def get_parser(self):
        return self.parser

    def save(self, filename, args_str=None):
        if args_str is None:
            args_str = ' '.join(sys.argv[1:])
        path = '/'.join(filename.split('/')[:-1])
        os.makedirs(path, exist_ok=True)
        with open(filename, 'w') as file:
            file.write(args_str)

    def load(self, filename):
        with open(filename, 'r') as file:
            args_str = file.readline()
        return self.parse_args(args_str.split())


class TestOptionParser(BaseOptionParser):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--save', type=str, default='./test')
        self.parser.add_argument('--plot_save', type=str, default='./results/plots')
        self.parser.add_argument('--plot_cnt', type=int, default=5)
        self.parser.add_argument('--load_epoch', type=int, default=-1)
        self.parser.add_argument('--unity_input', type=str, default='')
        self.parser.add_argument('--export_onnx', type=int, default=1)


class MLPMappingOption(BaseOptionParser):
    def __init__(self):
        super().__init__()
        
        # Data related
        self.parser.add_argument('--save', type=str, default='./results/test')
        self.parser.add_argument('--paths', type=str)
        self.parser.add_argument('--input_features', type=str)
        self.parser.add_argument('--output_features', type=str)
        self.parser.add_argument('--follower_input_use_input_features', type=int, default=0)
        self.parser.add_argument('--path4manifolds', type=str)
        self.parser.add_argument('--test_sequence_ratio', type=float, default=0.2)
        self.parser.add_argument('--use_random_test_sequence', type=int, default=0)
        self.parser.add_argument('--use_relative_root_motion', type=int, default=1)
        self.parser.add_argument('--extra_weight_root_position', type=float, default=10)
        self.parser.add_argument('--extra_weight_input_root_position', type=float, default=0)
        self.parser.add_argument('--extra_weight_foot_contact', type=float, default=0)
        self.parser.add_argument('--use_delta_root_motion', type=int, default=0)
        self.parser.add_argument('--use_3pt_input', type=str, default='b_head,b_r_wrist,b_l_wrist')
        self.parser.add_argument('--follower_in_leader_coordinate', type=int, default=0)
        self.parser.add_argument('--running_normalize', type=int, default=0)
        self.parser.add_argument('--no_root_derivative', type=int, default=1)
        self.parser.add_argument('--reference_char', type=int, default=0)
        self.parser.add_argument('--apply_3pt_input', type=int, default=1)
        self.parser.add_argument('--apply_3pt_output', type=int, default=0)
        self.parser.add_argument('--all_input_in_transformed_coordinate', type=int, default=0)
        self.parser.add_argument('--data_name_filter', type=str, default='')
        self.parser.add_argument('--use_partial_lead', type=float, default=0)
        self.parser.add_argument('--no_mirror', type=int, default=0)
        self.parser.add_argument('--data_processor_type', type=str, default='Unknown')
        self.parser.add_argument('--no_autoregressive', type=int, default=0)

        # Network arch related
        self.parser.add_argument('--num_layers', type=int, default=10)
        self.parser.add_argument('--hidden_size', type=int, default=2048)
        self.parser.add_argument('--dropout', type=float, default=0.2)
        self.parser.add_argument('--noise_level', type=float, default=1e-1)

        # Training related
        self.parser.add_argument('--debug', type=int, default=0)
        self.parser.add_argument('--epochs', type=int, default=10)
        self.parser.add_argument('--save_freq', type=int, default=20)
        self.parser.add_argument('--lambda_rec', type=float, default=1)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--device', type=str, default='')

        self.parser.add_argument('--learning_rate', type=float, default=5e-5)
        self.parser.add_argument('--weight_decay', type=float, default=1e-4)
        self.parser.add_argument('--restart_period', type=int, default=10)
        self.parser.add_argument('--restart_mult', type=int, default=2)


    @staticmethod
    def post_process(args):
        if isinstance(args.paths, str):
            args.paths = args.paths.split(',')
        if isinstance(args.path4manifolds, str):
            args.path4manifolds = args.path4manifolds.split(',')
        if isinstance(args.input_features, str):
            args.input_features = args.input_features.split(',')
        if isinstance(args.output_features, str):
            args.output_features = args.output_features.split(',')
        if args.use_3pt_input == 'None':
            args.use_3pt_input = None
        if args.use_3pt_input == 'RootOnly':
            args.use_3pt_input = []
        if isinstance(args.use_3pt_input, str):
            args.use_3pt_input = args.use_3pt_input.split(',')
        if args.follower_in_leader_coordinate:
            raise Exception('Follower in leader coordinate deprecated')
        return args


class AutoregressiveMLPMappingOption(MLPMappingOption):
    def __init__(self):
        super().__init__()
        # Data related
        self.parser.add_argument('--window', type=float, default=1.0)
        self.parser.add_argument('--future_ratio', type=float, default=1.0)
        self.parser.add_argument('--model_fps', type=int, default=30)

        # Network arch related


        # Training related
        self.parser.add_argument('--use_future', type=int, default=1)


    @staticmethod
    def post_process(args):
        # if isinstance(args.use_3pt_input, str):
        #     args.use_3pt_input = args.use_3pt_input.split(',')
        # if isinstance(args.paths, str):
        #     args.paths = args.paths.split(',')
        # if isinstance(args.path4manifolds, str):
        #     args.path4manifolds = args.path4manifolds.split(',')
        # if isinstance(args.input_features, str):
        #     args.input_features = args.input_features.split(',')
        # if isinstance(args.output_features, str):
        #     args.output_features = args.output_features.split(',')
        return MLPMappingOption.post_process(args)
        # return args
