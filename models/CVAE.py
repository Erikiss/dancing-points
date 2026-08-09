import torch
import torch.nn as nn

from data_process import data_processors
import models.CVAEModule as CVAEModule
import option

import numpy as np


class AutoregressiveCVAEOption(option.AutoregressiveMLPMappingOption):
    def __init__(self):
        super().__init__()

        # Tracking recipe defaults (see README "Training from Scratch"); still overridable per-run
        self.parser.set_defaults(
            input_features='Positions,RootMotion',
            output_features='VelocitiesV2,Positions,Rotations,FootContactLabels,RootMotion',
            reference_char=1,
            running_normalize=1,
            no_mirror=1,
            epochs=140,
        )

        # Network arch related
        self.parser.add_argument('--encoder_dim', type=int, default=2048)
        self.parser.add_argument('--estimator_dim', type=int, default=1024)
        self.parser.add_argument('--decoder_dim', type=int, default=2048)
        self.parser.add_argument('--codebook_channels', type=int, default=256)
        self.parser.add_argument('--codebook_dim', type=int, default=8)

        self.parser.add_argument('--activation_codebook', type=str, default='softmax')

        # Training related
        self.parser.add_argument('--lambda_matching', type=float, default=1.0)

    @staticmethod
    def post_process(args):
        args = option.AutoregressiveMLPMappingOption.post_process(args)
        return args


ExpectedDataProcessor = data_processors.TrackingProcessor
TrainOption = AutoregressiveCVAEOption


def create_model_from_args(args, motion_data, data_processor, requires_dims=False):
    one_batch = motion_data[0]
    one_batch[0] = one_batch[0].unsqueeze(0)
    one_batch[1] = one_batch[1].unsqueeze(0)

    lead, follow_input, follow_output = data_processor.reshape_data(one_batch, args, motion_data)

    input_dims = lead.shape[-1] + follow_input.shape[-1]
    output_dims = follow_output.shape[-1]

    model = CVAEModule.CVAEModel(input_dims, output_dims, args.encoder_dim, args.estimator_dim, args.decoder_dim,
                      args.codebook_channels, args.codebook_dim, args.dropout, args.activation_codebook)

    if requires_dims:
        return model, input_dims, output_dims

    return model


class NamedModel(nn.Module):
    def __init__(self, flatten_model, feature_dims, std_in, mean_in, std_out, mean_out):
        super().__init__()
        self.flatten_model = flatten_model
        self.feature_dims = feature_dims
        self.consts_for_k = torch.zeros(1, device=list(flatten_model.parameters())[0].device)

        def from_numpy(x):
            if isinstance(x, np.ndarray):
                device = list(flatten_model.parameters())[0].device
                x = torch.from_numpy(x).to(device).to(torch.float32)
            return x

        self.std_in = from_numpy(std_in)
        self.mean_in = from_numpy(mean_in)
        self.std_out = from_numpy(std_out)
        self.mean_out = from_numpy(mean_out)

    def forward(self, *args):
        x = torch.cat(args, dim=-1)
        batch_size = x.shape[0]

        if self.std_in is not None:
            x = (x - self.mean_in) / self.std_in

        output, estimate = self.flatten_model(x)

        if self.std_out is not None:
            output = output * self.std_out + self.mean_out

        outputs = []
        for d in self.feature_dims:
            outputs.append(output[..., :d])
            output = output[..., d:]
        # `estimate` (code/mu/std/kl_loss) is deliberately not returned: exporting it makes the
        # LogVar branch and the KL subgraph graph outputs, so they survive into the ONNX and the
        # engine reads them back every frame. Dropping it prunes 16 nodes with bit-identical
        # motion outputs.
        return outputs
