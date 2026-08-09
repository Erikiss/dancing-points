import torch
import torch.nn as nn
import numpy as np

import data_process.relative_motion as relative_motion
import data_process.data_processors as data_processors
from modules import MLPChannels
from option import MLPMappingOption


ExpectedDataProcessor = data_processors.MappingProcessor


class OneFrameMappingOption(MLPMappingOption):
    def __init__(self):
        super().__init__()

        # Data related
        self.parser.add_argument('--window', type=float, default=1.0)
        self.parser.add_argument('--model_fps', type=int, default=30)
        self.parser.add_argument('--use_input_window', type=int, default=1)
        self.parser.add_argument('--history_portion', type=float, default=1)
        self.parser.add_argument('--follower_autoregressive', type=int, default=0)
        self.parser.add_argument('--follower_full_history', type=int, default=0)
        self.parser.add_argument('--center_frame_idx', type=int, default=0)

        # Mapping recipe defaults (see README "Training from Scratch"); still overridable per-run
        self.parser.set_defaults(
            input_features='Positions,RootMotion',
            output_features='Positions,RootMotion',
            apply_3pt_output=1,
            use_partial_lead=0.5,
            epochs=143,
            extra_weight_root_position=5.,
        )

    @staticmethod
    def post_process(args):
        args = MLPMappingOption.post_process(args)
        args.original_window = args.window
        args.window = args.window * (1 + args.history_portion)
        # if args.follower_autoregressive and args.reference_char == 0:
        #     raise Exception("follower_autoregressive requires reference_char to be 1")
        return args


def create_model_from_args(args, motion_data, data_processor, requires_dims=False):
    one_batch = motion_data[0]
    one_batch[0] = one_batch[0].unsqueeze(0)
    one_batch[1] = one_batch[1].unsqueeze(0)

    lead, follow_input, follow_output = data_processor.reshape_data(one_batch, args, motion_data)

    input_dims = lead.shape[-1] + follow_input.shape[-1]
    output_dims = follow_output.shape[-1]

    channels = [input_dims] + [args.hidden_size] * (args.num_layers - 1) + [output_dims]

    model = OneFrameMLP(channels, bn=False, dropout=args.dropout)

    if requires_dims:
        return model, input_dims, output_dims

    return model


TrainOption = OneFrameMappingOption


class OneFrameMLP(MLPChannels):
    def __init__(self, channels, bn, dropout):
        super().__init__(channels, bn, dropout)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return super().forward(x), None

    def learn(self, input, output):
        self.train()
        self.zero_grad()
        model_output, _ = self(input)
        loss_rec = self.loss_fn(model_output, output)
        losses = {"rec": loss_rec}
        return losses, None


class NamedModel(nn.Module):
    def __init__(self, flatten_model, feature_dims, std_in, mean_in, std_out, mean_out):
        super().__init__()
        self.flatten_model = flatten_model
        self.feature_dims = feature_dims

        def from_numpy(x):
            if isinstance(x, np.ndarray):
                device = list(flatten_model.parameters())[0].device
                x = torch.from_numpy(x).to(device).to(torch.float32)
            return x

        self.std_in = from_numpy(std_in) if std_in is not None else None
        self.mean_in = from_numpy(mean_in) if mean_in is not None else None
        self.std_out = from_numpy(std_out) if std_out is not None else None
        self.mean_out = from_numpy(mean_out) if mean_out is not None else None

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        batch_size = x.shape[0]

        if self.std_in is not None:
            x = (x - self.mean_in) / self.std_in

        output = self.flatten_model(x)[0]

        if self.std_out is not None:
            output = output * self.std_out + self.mean_out

        outputs = []
        for d in self.feature_dims:
            outputs.append(output[..., :d])
            output = output[..., d:]
        return outputs

@data_processors.reshape_data_verifier(data_processors.MappingProcessor)
def reshape_data(one_batch, args, dataset):
    lead = one_batch[0]
    follow_original = follow = one_batch[1]

    if not args.use_input_window:
        lead = lead[..., :1]

    if args.history_portion > 0:
        leader_window_size = int(args.original_window * dataset.datas[0].target_fps) + 1
        follower_window_size = leader_window_size - 1
        lead = lead[..., :leader_window_size]
        if args.use_partial_lead != 0:
            length = int(lead.shape[-1] * args.use_partial_lead)
            lead = lead[..., -length:]
        if args.follower_autoregressive:
            follow = follow_original[..., -(follower_window_size+1):]
        else:
            follow = follow[..., -follower_window_size:]

    lead, follow = relative_motion.create_relative_root_motion(lead, follow, 0, dataset)

    follow_output = follow
    if args.follower_autoregressive:
        follow_input = follow[..., :1]
        follow_output = follow[..., 1:]
    else:
        follow_input = follow[..., :0]

    batch_size = lead.shape[0]
    lead = lead.reshape(batch_size, -1)
    follow_input = follow_input.reshape(batch_size, -1)
    follow_output = follow_output.reshape(batch_size, -1)
    return lead, follow_input, follow_output


@data_processors.get_data_shapes_verifier(data_processors.MappingProcessor)
def get_data_shapes(motion_data, args):
    one_batch = motion_data[0]
    lead = one_batch[0]
    follow_original = follow = one_batch[1]

    t_axis = lead.shape[-1]

    if not args.use_input_window:
        lead = lead[..., :1]

    if args.history_portion > 0:
        leader_window_size = int(args.original_window * motion_data.datas[0].target_fps) + 1
        follower_window_size = leader_window_size - 1
        lead = lead[..., :leader_window_size]
        if args.use_partial_lead != 0:
            length = int(lead.shape[-1] * args.use_partial_lead)
            t_axis -= lead.shape[-1] - length
            lead = lead[..., -length:]
        follow = follow[..., -follower_window_size:]
        follower_output_offset = t_axis - follower_window_size
    else:
        follower_output_offset = 0
        follower_window_size = t_axis

    follow_output = follow
    if args.follower_autoregressive:
        follow_input = follow_original[..., -follower_window_size - 1:-follower_window_size]
    else:
        follow_input = follow[..., :0]

    batch_size = lead.shape[0]
    lead = lead.reshape(batch_size, -1)

    in_data = motion_data.datas[0]
    out_data = motion_data.datas[1]

    batch_size = lead.shape[0]

    n_dims_in_leader = [n * lead.shape[-1] for n in in_data.feature_dims]
    n_dims_in_follower = [n * follow_input.shape[-1] for n in out_data.feature_dims]
    n_dims_out_follower = [n * follow_output.shape[-1] for n in out_data.feature_dims]

    meta_dict = {"window_size": str(t_axis), "n_frame_leader_input": str(lead.shape[-1]), "n_frame_follower_input": "0",
                 "n_frame_follower_output": str(follower_window_size), "follower_output_offset": str(follower_output_offset)}
    return n_dims_in_leader, n_dims_in_follower, n_dims_out_follower, (lead.shape[-1], follow_input.shape[-1], follow_output.shape[-1]), meta_dict
