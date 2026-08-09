import os.path

import numpy as np
import torch
import Library.Utility as utility
from torch.utils.data import Dataset
import os.path as osp


def get_shape(Load):
    try:
        return utility.LoadTxtAsInt(Load + "/DataShape.txt")
    except:
        _, d1, d0 = get_combined_shape(Load)
        return np.array([d0, sum(d1)])


def check_path(path):
    if 'Datasets' not in path:
        path = osp.join('Datasets', path)
    return path


def get_combined_shape(prefix):
    filename = osp.join(prefix, 'Description.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()
    channel_names = lines[0].strip().split(',')
    channel_dims = [int(x) for x in lines[1].strip().split(',')]
    n_frames = int(lines[2].strip())
    return channel_names, channel_dims, n_frames


def get_joint_mapping(prefix):
    filename = osp.join(prefix, 'Description.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()
    if len(lines) < 5:
        return None, None
    joint_names = lines[3].strip().split(',')
    joint_mappings = [int(x) for x in lines[4].strip().split(',')]
    return joint_names, joint_mappings


def get_fps(prefix):
    filename = osp.join(prefix, 'Description.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()
    if len(lines) < 6:
        return 60
    return int(lines[5].strip())


def load_single_dataset_bin(path, normalize, needed_feature_names, std_cap):
    channel_names, channel_dims, n_frames = get_combined_shape(path)
    if needed_feature_names == None or needed_feature_names == 'all':
        needed_feature_names = channel_names
    shape = (n_frames, sum(channel_dims))
    data = path + "/Data.bin"
    data = utility.ReadBinary(data, shape[0], shape[1])

    # Reorder the data channel according to the needed_feature_names
    named_data = {}
    for i, name in enumerate(channel_names):
        named_data[name] = data[:, :channel_dims[i]]
        data = data[:, channel_dims[i]:]
    assert data.shape[-1] == 0

    channel_dims = []
    data = []
    for name in needed_feature_names:
        data.append(named_data[name])
        channel_dims.append(named_data[name].shape[-1])
    data = np.concatenate(data, axis=-1)

    data_std = data.std(axis=0)
    data_mean = data.mean(axis=0)
    if not normalize:
        data_std[:] = 1.0
        data_mean[:] = 0.0
    else:
        set_std_cap(data_std, std_cap)
    data = (data - data_mean) / data_std

    return data, data_mean, data_std, channel_dims, needed_feature_names


def get_with_gather_numpy(Data, gather_window, sequence):
    gather = gather_window
    pivot = sequence[0]
    _min = sequence[1]
    _max = sequence[2]

    gather = np.clip(gather + pivot, _min, _max)

    data = Data[gather].astype(np.float32)
    return data


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.single_frame = False

    def set_single_frame(self, val):
        self.single_frame = val

    def prepare_sequence(self, frames, Load, extra_frames, test_sequence_ratio,
                         use_random_test_sequence=True, step=1, filter=None):
        Shape = get_shape(Load)

        if filter is not None:
            Sequences, mask, uuid_map = utility.LoadFilteredSequence(Load + "/Sequences.txt", filter)
            self.data_mask = mask
            self.uuid_map = uuid_map
        else:
            Sequences = utility.LoadSequences(Load + "/Sequences.txt", True, Shape[0])
            self.data_mask = None

        feature_dim = Shape[1]
        gather_padding = (int((frames - 1) / 2))
        gather_window = np.arange(frames + extra_frames, step=step) - gather_padding
        gather_window_test = np.arange(frames) - gather_padding

        print("Generating Data Sequences")
        data_sequences = []
        test_sequences = []

        for i in range(Sequences[-1]):
            utility.PrintProgress(i, Sequences[-1])
            indices = np.where(Sequences == (i + 1))[0]
            for j in range(indices.shape[0]):
                slice = [indices[j], indices[0], indices[-1]]
                if use_random_test_sequence:
                    if np.random.uniform(0, 1) < test_sequence_ratio and (
                            indices[0] + gather_padding) <= indices[j] <= (indices[-1] - gather_padding):
                        test_sequences.append(slice)
                    else:
                        data_sequences.append(slice)
                else:
                    if j < indices.shape[0] * (1 - test_sequence_ratio):
                        data_sequences.append(slice)
                    elif (indices[0] + gather_padding) <= indices[j] <= (indices[-1] - gather_padding):
                        test_sequences.append(slice)

        print("Data Sequences:", len(data_sequences))
        print("Test Sequences:", len(test_sequences))
        data_sequences = np.array(data_sequences)

        self.Sequences = Sequences
        self.data_sequences = data_sequences
        self.test_sequences = test_sequences
        self.sample_count = len(data_sequences)
        self.gather_window = gather_window
        self.gather_window_test = gather_window_test
        self.window_size = len(gather_window)
        self.window_size_test = len(gather_window_test)
        self.feature_dim = feature_dim

        self.Data = None
        self.data_mean = 0.
        self.data_std = 1.

    def get_window_starting_frame_index(self, item):
        gather = self.gather_window
        sequence = self.data_sequences[item]
        pivot = sequence[0]
        _min = sequence[1]
        _max = sequence[2]

        gather = np.clip(gather + pivot, _min, _max)
        return gather[0]

    def get_window_with_sequence(self, sequence):
        gather = self.gather_window
        pivot = sequence[0]
        _min = sequence[1]
        _max = sequence[2]

        gather = np.clip(gather + pivot, _min, _max)

        data = self.Data[gather]
        data = torch.from_numpy(data).float()

        data = data.permute(1, 0)

        return data

    def __getitem__(self, item):
        return self.get_window_with_sequence(self.data_sequences[item])

    def get_test_item(self, item):
        return self.get_window_with_sequence(self.test_sequences[item])

    def get_window_bound(self, item):
        sequence = self.data_sequences[item]
        _min = sequence[1]
        _max = sequence[2]
        return _min, _max

    def sample_continuous_test_window(self):
        indices = self.gather_window_test + np.random.choice(self.test_sequences)
        if self.single_frame:
            return torch.from_numpy(self.Data[indices].astype(np.float32))
        return self.load_batches(indices)[..., :self.window_size_test]

    def load_batches(self, indices):
        res = []
        for i in indices:
            res.append(self[i])
        res = torch.stack(res, dim=0)
        return res

    def __len__(self):
        if self.single_frame:
            return self.Data.shape[0]
        return self.sample_count

    def sample_long_sequence(self, length):
        seq = self.Data[:length]
        seq = torch.from_numpy(seq).permute(1, 0)
        return seq


def get_dataset_name_from_path(path: str):
    path = path.strip().lower()
    if 'human' in path:
        if 'loco' in path:
            return 'human_loco'
        return 'human'
    if 'dog' in path:
        return 'dog'
    if 'mocha' in path:
        return path[path.index('mocha'):]
    if 'paireddance-simplified-1' in path:
        return 'PD-S-1'
    if 'paireddance-simplified-2' in path:
        return 'PD-S-2'
    if 'balboa-1' in path:
        return 'balboa-1'
    if 'balboa-2' in path:
        return 'balboa-2'
    if 'dance-all-leader' in path:
        return 'dance-all-leader'
    if 'dance-all-follower' in path:
        return 'dance-all-follower'
    return 'unknown'


def set_std_cap(data_std, cap):
    print(f"Set {(data_std < cap).sum()} entries cap to", cap)
    print("The entries are", np.where(data_std < cap)[0])
    data_std[data_std < cap] = cap


class SequenceAndManifold(BaseDataset):
    def __init__(self, path, window, test_sequence_ratio, path4manifold, needed_channel_names, normalize, use_manifold_ori,
                 std_cap, extra_frames=0, frames=None, needed_manifold_names=['manifold', ], normalize_manifold=True,
                 requires_full_sequence=False, additional_manifold_names=[], use_random_test_sequence=True,
                 target_fps=None, use_3pt_input=None, no_root_derivative=False, data_name_filter=None, no_mirror=False):
        super().__init__()
        path = check_path(path)

        # Prepare metadata
        self.fps = get_fps(path)
        if target_fps is None:
            target_fps = self.fps

        self.target_fps = target_fps

        assert self.fps % target_fps == 0

        def filter(x):
            res = True
            if data_name_filter is not None:
                if data_name_filter == 'None':
                    res = True
                elif data_name_filter == 'lafan_no_push':
                    res = 'push' not in str(x[3]).lower()
                elif data_name_filter == 'lafan_no_push_aiming':
                    low = str(x[3]).lower()
                    res = 'push' not in low and 'aiming' not in low
                else:
                    res = data_name_filter.lower() in str(x[3]).lower()
            if no_mirror:
                res = res and 'mirrored' not in str(x[2]).lower()
            return res

        if frames is None:
            frames = int(window * self.fps) + 1
        self.sample_step = self.fps // target_fps
        self.prepare_sequence(frames, path, extra_frames, test_sequence_ratio,
                              use_random_test_sequence, step=self.sample_step, filter=filter)
        self.frames_per_window = self.gather_window.shape[0]


        if len(needed_channel_names) > 0:
            data, _, _, channel_dims, needed_channel_names = load_single_dataset_bin(path, normalize=False,
                                                                                     needed_feature_names=needed_channel_names,
                                                                                     std_cap=std_cap)

            if self.data_mask is not None:
                data = data[self.data_mask]

            # if use_3pt_input is None:
            #     joint_names, joint_mapping = get_joint_mapping(path)
            #     use_3pt_input = joint_names

            joint_names, joint_mapping = get_joint_mapping(path)
            self.num_joints = len(joint_mapping)

            if use_3pt_input is None:
                use_3pt_input = joint_names

            if use_3pt_input is not None:
                target_idx = []
                for joint in use_3pt_input:
                    idx = joint_names.index(joint)
                    target_idx.append(joint_mapping[idx])

                interested_slicing = []
                offset = 0
                for i in range(len(channel_dims)):
                    n_dim_per_joint = channel_dims[i] // len(joint_mapping)
                    if n_dim_per_joint > 0:  # Perjoint features
                        for idx in target_idx:
                            interested_slicing += list(range(offset + n_dim_per_joint * idx, offset + n_dim_per_joint * (idx + 1)))
                        offset += channel_dims[i]
                        channel_dims[i] = n_dim_per_joint * len(target_idx)
                    else: # Single features, for now it's only root motion
                        if 'RootMotion' in needed_channel_names[i]:
                            if no_root_derivative and channel_dims[i] == 8:
                                channel_dims[i] -= 4
                                compensate = 4
                            else:
                                compensate = 0
                            interested_slicing += list(range(offset, offset + channel_dims[i]))
                            if no_root_derivative:
                                offset += compensate # Compensate for the removed derivative
                        else:
                            interested_slicing += list(range(offset, offset + channel_dims[i]))
                        offset += channel_dims[i]


                self.slicing_3pt = interested_slicing
                self.slicing_3pt = 'this should not be used at all'

                data = data[..., interested_slicing]

                self.num_joints = len(target_idx)
            else:
                self.slicing_3pt = None
        else:
            data = None
            channel_dims = []
            needed_channel_names = []
            self.slicing_3pt = None

        manifold = np.load(path4manifold) if os.path.exists(path4manifold) else None
        manifold_features = []
        manifold_dims = []

        additional_manifold_features = []
        additional_manifold_dims = []
        if manifold is not None and path4manifold.endswith('.npz'):
            for name in needed_manifold_names:
                manifold_features.append(manifold[name])
                manifold_dims.append(manifold[name].shape[-1])

            for name in additional_manifold_names:
                additional_manifold_features.append(manifold[name])
                additional_manifold_dims.append(manifold[name].shape[-1])

        if len(manifold_features):
            manifold = np.concatenate(manifold_features, axis=-1)
        else:
            manifold = np.empty(data.shape[:-1] + (0,), dtype=np.float32)
        if data is not None:
            data = np.concatenate((manifold, data), axis=-1)
        else:
            data = manifold
        self.n_channel_manifold = manifold.shape[-1]

        if self.slicing_3pt is not None and (manifold is not None and manifold.shape[1] > 0):
            raise Exception("3pt input is not supported for manifold, need to add additional offset")

        self.additional_manifold_names = additional_manifold_names
        self.additional_manifold_features = additional_manifold_features
        self.additional_manifold_dims = additional_manifold_dims

        data_std = data.std(axis=0)
        data_mean = data.mean(axis=0)
        if not normalize:
            data_std[:] = 1.0
            data_mean[:] = 0.0
        if not normalize_manifold:
            manifold_dim = manifold.shape[-1]
            data_std[:manifold_dim] = 1.0
            data_mean[:manifold_dim] = 0.0

        set_std_cap(data_std, std_cap)
        data = (data - data_mean) / data_std

        if requires_full_sequence:
            self.full_sequence = utility.LoadFullSequence(path + "/Sequences.txt", True, data.shape[0])
            self.restore_full_sequence_mapping()
        self.name = get_dataset_name_from_path(path)
        if use_manifold_ori:
            self.name += '_ori'
        self.Data = data
        self.data_mean = data_mean
        self.data_std = data_std
        self.feature_dims = manifold_dims + channel_dims
        self.channel_names = needed_manifold_names + needed_channel_names
        self.fps = get_fps(path)

    def get_manifold_feature(self, name):
        feature_idx = self.additional_manifold_names.index(name)
        return self.additional_manifold_features[feature_idx]

    def get_motion_feature(self, name):
        feature_idx = self.channel_names.index(name)
        all_features = self.Data
        for i in range(feature_idx):
            all_features = all_features[..., self.feature_dims[i]:]
        return all_features[..., :self.feature_dims[feature_idx]]

    def get_motion_window(self, name, item):
        data = self.get_motion_feature(name)
        return get_with_gather_numpy(data, self.gather_window, self.data_sequences[item])

    def get_one_cycle(self):
        if self.name.startswith('dog') or self.name.startswith('human'):
            return 1
        elif self.name.startswith('mocha'):
            return 2
        elif self.name.startswith('PD-S'):
            return 1
        else:
            raise Exception("Unknown dataset")

    def get_num_states(self):
        return self.get_manifold_feature('index').max() + 1

    def get_manifold_window(self, item, extra_frames, name, extra_frames_rear_only=False):
        feature_idx = self.additional_manifold_names.index(name)
        data = self.additional_manifold_features[feature_idx]
        gather = self.gather_window
        if extra_frames > 0:
            gather0 = np.arange(-extra_frames, 0, dtype=np.int64) + gather[0] if not extra_frames_rear_only else np.zeros((0,), dtype=np.int64)
            gather1 = np.arange(0, extra_frames, dtype=np.int64) + gather[-1] + 1
            gather = np.concatenate([gather0, gather, gather1])
        return get_with_gather_numpy(data, gather, self.data_sequences[item])

    def get_phase(self, item, extra_frames=0):
        return self.get_manifold_window(item, extra_frames, 'phase')

    def restore_full_sequence_mapping(self):
        """
        This function exists because the id for motion is modified in order to remove breaking frames
        by cutting the motion into multiple sequences.
        """
        self.full_sequence_mapping = {}
        current_count = 1
        for i in range(1, max(self.Sequences) + 1):
            self.full_sequence_mapping[i] = current_count
            indices = np.where(self.Sequences == i)[0]
            start = indices[0]
            if start == 0 or \
                (self.full_sequence[start][-1] != self.full_sequence[start-1][-1] or  \
                        self.full_sequence[start][2] != self.full_sequence[start-1][2]):
                current_count += 1
            else:
                # print('Something is wrong')
                pass

    def get_feature_slice_by_name(self, names):
        slices = []
        for name in names:
            idx = self.channel_names.index(name)
            s = sum(self.feature_dims[:idx])
            e = sum(self.feature_dims[:idx + 1])
            slices += list(range(s, e))
        return slices

    def get_feature_size_by_name(self, names):
        if isinstance(names, str):
            idx = self.channel_names.index(names)
            return self.feature_dims[idx]
        else:
            res = []
            for name in names:
                idx = self.channel_names.index(name)
                res.append(self.feature_dims[idx])
            return res

    def get_feature_by_name(self, d, name, dim=-1, denormalize=True):
        # Put the dim-th dimension of d to the last dimension
        d_perm = list(range(d.ndim))
        d_perm[dim], d_perm[-1] = d_perm[-1], d_perm[dim]
        d = d.permute(*d_perm)
        d_shape = d.shape
        d = d.reshape(-1, d.shape[-1])

        idx = self.channel_names.index(name)
        s = sum(self.feature_dims[:idx])
        e = sum(self.feature_dims[:idx + 1])

        d = d[..., s:e]
        if denormalize:
            d = d * self.data_std[s:e] + self.data_mean[s:e]

        d = d.reshape(d_shape)
        d = d.permute(*d_perm)
        return d

    def get_a_window_with_step(self, start_idx, window_size):
        gather_window = np.arange(window_size * self.sample_step, step=self.sample_step) + start_idx
        window = get_with_gather_numpy(self.Data, gather_window, self.data_sequences[start_idx])
        window = torch.from_numpy(window)
        window = window.permute(1, 0)
        return window
