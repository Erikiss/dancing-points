import numpy as np
import torch
from torch.utils.data import Dataset
from dataset import SequenceAndManifold


def create_dataset_from_args(args):
    def separate_names(names):
        names = names.copy()
        manifold_names = []
        if 'manifold' in names:
            names.remove('manifold')
            manifold_names += ['manifold']
        return names, manifold_names

    window = args.window if hasattr(args, 'window') else 1

    channel_names = [[], []]
    manifold_names = [[], []]

    channel_names[0], manifold_names[0] = separate_names(args.input_features)
    channel_names[1], manifold_names[1] = separate_names(args.output_features)

    target_fps = args.model_fps if hasattr(args, 'model_fps') else None
    use_relative_root_motion = args.use_relative_root_motion if hasattr(args, 'use_relative_root_motion') else False

    inplace_normalize = not args.running_normalize

    dataset = PairedDataset(args.paths, path4manifolds=args.path4manifolds, needed_channel_names=channel_names,
                            needed_manifold_names=manifold_names, use_relative_root_motion=use_relative_root_motion,
                            extra_weight_root_position=args.extra_weight_root_position,
                            use_delta_root_motion=args.use_delta_root_motion,
                            window=window, test_sequence_ratio=args.test_sequence_ratio,
                            normalize=inplace_normalize, use_manifold_ori=False, std_cap=1e-3,
                            normalize_manifold=True, use_random_test_sequence=args.use_random_test_sequence,
                            target_fps=target_fps, use_3pt_input=args.use_3pt_input,
                            no_root_derivative=args.no_root_derivative, reference_char=args.reference_char,
                            apply_3pt_input=args.apply_3pt_input, apply_3pt_output=args.apply_3pt_output,
                            data_name_filter=args.data_name_filter, no_mirror=args.no_mirror,
                            extra_weight_foot_contact=args.extra_weight_foot_contact,
                            extra_weight_input_root_position=args.extra_weight_input_root_position)
    return dataset


class PairedDataset(Dataset):
    def __init__(self, paths, path4manifolds, needed_channel_names, needed_manifold_names,
                 use_relative_root_motion, extra_weight_root_position, use_delta_root_motion, reference_char,
                 apply_3pt_input, apply_3pt_output, extra_weight_foot_contact, extra_weight_input_root_position, *args, **kwargs):
        if isinstance(paths, str):
            paths = paths.split(',')
        self.datas = []
        use_3pt_input = kwargs.get('use_3pt_input', None)

        for i in range(len(paths)):
            kwargs['use_3pt_input'] = None
            if i == 0 and apply_3pt_input:
                kwargs['use_3pt_input'] = use_3pt_input
            if i == 1 and apply_3pt_output:
                kwargs['use_3pt_input'] = use_3pt_input
            self.datas.append(SequenceAndManifold(paths[i], path4manifold=path4manifolds[i], needed_manifold_names=needed_manifold_names[i],
                                                  needed_channel_names=needed_channel_names[i], *args, **kwargs))

        self.use_relative_root_motion = use_relative_root_motion
        self.use_delta_root_motion = use_delta_root_motion
        self.reference_char = reference_char
        self.no_root_derivative = kwargs.get('no_root_derivative', False)

        assert not (use_delta_root_motion and use_relative_root_motion)

        def get_foot_contact(motion_data):
            idx = motion_data.channel_names.index("FootContactLabels")
            s = sum(motion_data.feature_dims[:idx])
            e = sum(motion_data.feature_dims[:idx + 1])
            return motion_data.Data[:, s:e] * motion_data.data_std[s:e] + motion_data.data_mean[s:e], s, e

        def get_root_motion(motion_data):
            idx = motion_data.channel_names.index("RootMotion")
            s = sum(motion_data.feature_dims[:idx])
            e = sum(motion_data.feature_dims[:idx + 1])
            return motion_data.Data[:, s:e] * motion_data.data_std[s:e] + motion_data.data_mean[s:e], s, e

        def calculate_delta_root_motion(motion_data):
            root_motion, s, e = get_root_motion(motion_data)
            rotation = root_motion[:, :2]
            translation = root_motion[:, 2:]

            rotation = (rotation[:, 0] + 1j * rotation[:, 1]).astype(np.complex64)
            rotation[1:] = rotation[1:] / rotation[:-1]
            rotation[0] = 1
            translation[1:] = translation[1:] - translation[:-1]
            translation[0] = 0

            rotation = np.stack([rotation.real, rotation.imag], axis=-1)
            root_motion = np.concatenate([rotation, translation], axis=-1)

            r_std = root_motion.std(axis=0)
            r_mean = root_motion.mean(axis=0)
            motion_data.Data[:, s:e] = (root_motion - r_mean) / r_std

        if use_delta_root_motion:
            calculate_delta_root_motion(self.datas[0])
            calculate_delta_root_motion(self.datas[1])

        if extra_weight_root_position:
            _, s1, e1 = get_root_motion(self.datas[1])
            s, e = s1 + 2, s1 + 4
            self.datas[1].data_std[s:e] /= extra_weight_root_position
            self.datas[1].Data[:, s:e] *= extra_weight_root_position

        if extra_weight_input_root_position:
            _, s1, e1 = get_root_motion(self.datas[0])
            s, e = s1 + 2, s1 + 4
            self.datas[0].data_std[s:e] /= extra_weight_input_root_position
            self.datas[0].Data[:, s:e] *= extra_weight_input_root_position

        if extra_weight_foot_contact:
            contact, s, e = get_foot_contact(self.datas[1])
            self.datas[1].data_std[s:e] = 1 / extra_weight_foot_contact
            self.datas[1].data_mean[s:e] = 0
            self.datas[1].Data[:, s:e] = contact * extra_weight_foot_contact



    def __getitem__(self, idx):
        data_in = self.datas[0][idx]
        data_out = self.datas[1][idx]
        # if self.datas[0].slicing_3pt is not None:
        #     data_in = data_in[self.datas[0].slicing_3pt]
        return [data_in, data_out]

    def get_test_item(self, idx):
        data_in = self.datas[0].get_test_item(idx)
        data_out = self.datas[1].get_test_item(idx)
        return [data_in, data_out]

    def get_feature_by_names(self, idx, f, names):
        return self.datas[idx].get_feature_by_names(f, names)

    def __len__(self):
        return len(self.datas[0])

    def test_len(self):
        return len(self.datas[0].test_sequences)


class PairedDatasetTestView(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset.get_test_item(idx)

    def __len__(self):
        return self.dataset.test_len()
