import os
import os.path as osp

from tqdm import tqdm
import pickle

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from option import TestOptionParser

from paired_dataset import create_dataset_from_args, PairedDatasetTestView
from onnx_helpers import export_named_onnx_autoregressive_mlp
from modules import RunningNormalizeModel

from unified_utils import model_dispatch
import data_process.data_processors as data_processors


class RootPositionLoss:
    def __init__(self, motion_data):
        idx = motion_data.channel_names.index("RootMotion")
        s = sum(motion_data.feature_dims[:idx])
        e = sum(motion_data.feature_dims[:idx + 1])
        self.sli = slice(s, e)

    def __call__(self, y, gt):
        y = y[..., self.sli, :1]
        y = y[..., 2:4, :]
        gt = gt[..., self.sli, :1]
        gt = gt[..., 2:4, :]
        err = ((y - gt)**2).sum(axis=1) ** 0.5
        err = err.mean()
        return err


class SpecificChannelLoss:
    def __init__(self, motion_data, channel_name):
        idx = motion_data.channel_names.index(channel_name)
        s = sum(motion_data.feature_dims[:idx])
        e = sum(motion_data.feature_dims[:idx + 1])
        self.sli = slice(s, e)

    def __call__(self, y, gt):
        y = y[..., self.sli, :1]
        gt = gt[..., self.sli, :1]
        err = ((y - gt)**2).sum(axis=1) ** 0.5
        err = err.mean()
        return err


def main():
    test_option_parser = TestOptionParser()
    test_args = test_option_parser.parse_args()
    n_sample = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    to_save = {}

    if osp.exists(osp.join(test_args.save, "args.pkl")):
        raise Exception("This script is not compatible with the old args.pkl format")
    else:
        with open(osp.join(test_args.save, "args.txt"), "r") as f:
            text_args = f.read().split()
            model, remaining_args, _ = model_dispatch(text_args)
            option_parser = model.TrainOption()
            args = option_parser.text_deserialize(remaining_args)
            args = option_parser.post_process(args)

    if args.data_processor_type != 'Unknown':
        data_processor = data_processors.processor_dispatcher(args.data_processor_type)
    else:
        data_processor = model.ExpectedDataProcessor()

    Save = test_args.save

    motion_data = create_dataset_from_args(args)
    #Build network model
    network, in_dims, out_dims = model.create_model_from_args(args, motion_data, data_processor, requires_dims=True)
    if args.running_normalize:
        network = RunningNormalizeModel(network, in_dims, out_dims)

    #Load model
    ref_files = [f for f in os.listdir(Save) if f.endswith(".pt")]
    ref_files.sort(key=lambda x: int(x.split('.')[0]))

    if test_args.load_epoch != -1:
        target_file = f'{test_args.load_epoch:04d}.pt'
    else:
        target_file = ref_files[-1]

    state_dict = torch.load(osp.join(Save, target_file), map_location='cpu')
    network.load_state_dict(state_dict)
    network.eval()

    n_dims_in_leader, n_dims_in_follower, n_dims_out_follower, t_axis = data_processor.get_data_shapes(motion_data, args)
    meta_dict = data_processor.get_meta_dict(motion_data, args)

    input_feature_dims = n_dims_in_leader + n_dims_in_follower
    leader_input_feature_names = [f'input_leader_{name}' for name in args.input_features] if sum(n_dims_in_leader) > 0 else []
    if args.follower_input_use_input_features:
        follower_input_feature_names = [f'input_follower_{name}' for name in args.input_features]
    else:
        follower_input_feature_names = [f'input_follower_{name}' for name in args.output_features] if sum(
            n_dims_in_follower) > 0 else []

    input_feature_names = leader_input_feature_names + follower_input_feature_names
    
    input_shapes = []
    input_names = []

    for i in range(len(input_feature_dims)):
        if input_feature_dims[i] == 0:
            continue
        input_shapes.append((1, input_feature_dims[i]))
        input_names.append(input_feature_names[i])

    output_feature_dims = n_dims_out_follower
    dynamic_axes = {}

    for name in input_names:
        dynamic_axes[name] = {0: 'batch_size'}

    in_data = motion_data.datas[0]
    out_data = motion_data.datas[1]

    def expand_statistics(s, size):
        s = torch.from_numpy(s)
        s = s.unsqueeze(-1)
        s = s.expand(-1, size)
        s = s.flatten()
        return s

    std_in0 = expand_statistics(in_data.data_std, t_axis[0])
    mean_in0 = expand_statistics(in_data.data_mean, t_axis[0])

    if args.follower_input_use_input_features:
        follower_sli = out_data.get_feature_slice_by_name(args.input_features)

        std_in1 = expand_statistics(out_data.data_std[..., follower_sli], t_axis[1])
        mean_in1 = expand_statistics(out_data.data_mean[..., follower_sli], t_axis[1])
    else:
        std_in1 = expand_statistics(out_data.data_std, t_axis[1])
        mean_in1 = expand_statistics(out_data.data_mean, t_axis[1])

    std_out = expand_statistics(out_data.data_std, t_axis[2])
    mean_out = expand_statistics(out_data.data_mean, t_axis[2])

    std_in = torch.cat((std_in0, std_in1))
    mean_in = torch.cat((mean_in0, mean_in1))

    root_sli = out_data.get_feature_slice_by_name(['RootMotion'])
    print('Root motion std:', out_data.data_std[..., root_sli])

    meta_dict['fps'] = str(motion_data.datas[0].target_fps)
    meta_dict['use_relative_root_motion'] = str(motion_data.use_relative_root_motion)
    meta_dict['use_delta_root_motion'] = str(motion_data.use_delta_root_motion)
    meta_dict['no_root_derivative'] = str(motion_data.no_root_derivative)
    if isinstance(args.use_3pt_input, list) and len(args.use_3pt_input) == 0:
        meta_dict['input_joints'] = 'RootOnly'
    if args.use_3pt_input:
        meta_dict['input_joints'] = ','.join(args.use_3pt_input)
    meta_dict['reference_char'] = str(motion_data.reference_char)
    meta_dict['all_input_in_transformed_coordinate'] = str(args.all_input_in_transformed_coordinate)
    meta_dict['apply_3pt_input'] = str(args.apply_3pt_input)
    meta_dict['apply_3pt_output'] = str(args.apply_3pt_output)
    meta_dict['use_future'] = str(args.use_future) if hasattr(args, 'use_future') else '0'
    if hasattr(args, 'follower_autoregressive'):
        meta_dict['follower_autoregressive'] = str(args.follower_autoregressive)

    if test_args.export_onnx:
        export_named_onnx_autoregressive_mlp(network, osp.join(Save, 'model.onnx'), input_names, input_shapes,
                                             args.output_features, output_feature_dims,
                                             model.NamedModel, std_in, mean_in, std_out, mean_out,
                                             dynamic_axes=dynamic_axes, meta_dict=meta_dict)

    loss_function = torch.nn.MSELoss()
    data_loader = DataLoader(motion_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    if args.test_sequence_ratio == 0:
        test_data_loader = data_loader
    else:
        test_data_loader = DataLoader(PairedDatasetTestView(motion_data), batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

    # Short sequence evaluation
    iterator = iter(data_loader)
    test_iterator = iter(test_data_loader)
    n_sample = min(n_sample, len(test_data_loader) - 2)
    loop = tqdm(range(n_sample))
    losses = []
    losses_test = []

    std_magnitudes = []
    std_magnitudes_test = []

    root_losses = []
    root_losses_test = []
    root_loss_function = RootPositionLoss(motion_data.datas[1])

    contact_losses = []
    contact_losses_test = []
    contact_loss_function = SpecificChannelLoss(motion_data.datas[1], 'Positions')

    entropy_pi = []
    entropy_pi_test = []
    
    diffusion_losses = []

    network = network.to(device)
    network.eval()
    for i in loop:
        # Run model prediction
        training = i > n_sample // 2

        if training:
            losses_array = losses
            root_losses_array = root_losses
            contact_loss_array = contact_losses
            std_magnitudes_array = std_magnitudes
            entropy_pi_array = entropy_pi
            train_batch = next(iterator)
        else:
            losses_array = losses_test
            root_losses_array = root_losses_test
            contact_loss_array = contact_losses_test
            std_magnitudes_array = std_magnitudes_test
            entropy_pi_array = entropy_pi_test
            train_batch = next(test_iterator)

        input_data, follow_output = data_processor.ready_input_output(train_batch, args, motion_data)
        input_data = input_data.to(device)
        follow_output = follow_output.to(device)
        
        # diffusion_loss, _ = network.learn(input_data, follow_output)
        # diffusion_losses.append(diffusion_loss['rec'].detach().cpu().item())

        with torch.no_grad():
            if isinstance(network, RunningNormalizeModel):
                yPred_denorm, ex_info = network(input_data)
                yPred = network.out_s.normalize(yPred_denorm)
                gt_denorm = follow_output
                follow_output = network.out_s.normalize(gt_denorm)

                yPred_denorm = yPred_denorm.reshape(yPred_denorm.shape[0], -1, t_axis[2])
                gt_denorm = gt_denorm.reshape(gt_denorm.shape[0], -1, t_axis[2])
            else:
                yPred, ex_info = network(input_data)

                yPred = yPred.reshape(yPred.shape[0], -1, t_axis[2])
                follow_output = follow_output.reshape(yPred.shape[0], -1, t_axis[2])

                yPred_denorm = yPred * torch.from_numpy(motion_data.datas[1].data_std[..., None]).to(device)
                gt_denorm = follow_output * torch.from_numpy(motion_data.datas[1].data_std[..., None]).to(device)

        # Compute loss
        loss = loss_function(yPred, follow_output)
        root_loss = root_loss_function(yPred_denorm, gt_denorm)
        contact_loss = contact_loss_function(yPred_denorm, gt_denorm)

        losses_array.append(loss.detach().cpu().item())
        root_losses_array.append(root_loss.detach().cpu().item())
        contact_loss_array.append(contact_loss.detach().cpu().item())

        if isinstance(ex_info, dict) and 'pi' in ex_info:
            pi = ex_info['pi']
            e = -torch.sum(pi * torch.log(pi + 1e-8), dim=-1)
            entropy_pi_array.append(e.mean().detach().cpu().item())

        if isinstance(ex_info, dict) and 'std' in ex_info:
            std = ex_info['std']
            # std_magnitudes_array.append(std.max(dim=-1)[0].mean().detach().cpu().item())
            std_magnitudes_array.append(std.mean().detach().cpu().item())

        if isinstance(ex_info, dict) and 'logvar' in ex_info:
            logvar = ex_info['logvar']
            std = torch.exp(0.5 * logvar)
            # std_magnitudes_array.append(std.max(dim=-1)[0].mean().detach().cpu().item())
            std_magnitudes_array.append(std.mean().detach().cpu().item())


    losses = np.array(losses)
    losses_test = np.array(losses_test)
    root_losses = np.array(root_losses)
    root_losses_test = np.array(root_losses_test)
    contact_loss = np.array(contact_losses)
    contact_loss_test = np.array(contact_losses_test)
    to_save[f'loss_mean'] = losses.mean()
    to_save[f'test_loss_mean'] = losses_test.mean()
    to_save[f'root_loss_mean'] = root_losses.mean()
    to_save[f'test_root_loss_mean'] = root_losses_test.mean()
    # to_save['diffusion_losses'] = np.array(diffusion_losses).mean()

    if len(entropy_pi):
        entropy_pi = np.array(entropy_pi)
        entropy_pi_test = np.array(entropy_pi_test)
        to_save[f'entropy_pi_mean'] = entropy_pi.mean()
        to_save[f'test_entropy_pi_mean'] = entropy_pi_test.mean()

    # to_save[f'contact_loss_mean'] = contact_loss.mean()
    # to_save[f'test_contact_loss_mean'] = contact_loss_test.mean()

    if (len(std_magnitudes) > 0):
        std_magnitudes = np.array(std_magnitudes)
        std_magnitudes_test = np.array(std_magnitudes_test)
        to_save[f'std_magnitude_mean'] = f'{std_magnitudes.mean():.5e}'
        to_save[f'test_std_magnitude_mean'] = f'{std_magnitudes_test.mean():.5e}'

    print(to_save)
    with open(osp.join(test_args.save, 'summary_data.pickle'), 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
