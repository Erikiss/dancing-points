import torch
from abc import ABC, abstractmethod
import data_process.relative_motion as relative_motion


def reshape_data_verifier(processor_class):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not issubclass(processor_class, BaseProcessor):
                raise TypeError(f"processor_class must be a subclass of BaseProcessor, got {processor_class}")
            func_res = func(*args, **kwargs)
            new_imp_res = processor_class().reshape_data(*args, **kwargs)

            assert len(func_res) == len(new_imp_res)
            for i in range(len(func_res)):
                assert torch.allclose(func_res[i], new_imp_res[i])
            return func_res
        return wrapper
    return decorator


def get_data_shapes_verifier(processor_class):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not issubclass(processor_class, BaseProcessor):
                raise TypeError(f"processor_class must be a subclass of BaseProcessor, got {processor_class}")
            func_res = func(*args, **kwargs)
            func_res, func_dict = func_res[:-1], func_res[-1]
            new_imp_res = processor_class().get_data_shapes(*args, **kwargs)
            new_imp_dict = processor_class().get_meta_dict(*args, **kwargs)

            assert len(func_res) == len(new_imp_res)
            for i in range(len(func_res)):
                assert len(func_res[i]) == len(new_imp_res[i])
                for j in range(len(func_res[i])):
                    assert func_res[i][j] == new_imp_res[i][j]

            assert len(func_dict) == len(new_imp_dict)
            for k in func_dict.keys():
                assert k in new_imp_dict
                assert func_dict[k] == new_imp_dict[k]
            return func_res + (func_dict,)
        return wrapper
    return decorator


class BaseProcessor(ABC):
    # The shape should always be (batch_size, n_channels, n_frames) or (batch_size, n_channels * n_frames) when flatten
    @abstractmethod
    def reshape_data(self, one_batch, args, dataset, flatten=True):
        pass

    def get_data_shapes(self, motion_data, args):
        in_data = motion_data.datas[0]
        out_data = motion_data.datas[1]

        one_batch = motion_data[0]
        one_batch[0] = one_batch[0].unsqueeze(0)
        one_batch[1] = one_batch[1].unsqueeze(0)

        lead, follow_input, follow_output = self.reshape_data(one_batch, args, motion_data,
                                                              flatten=False)

        lead_t = lead.shape[-1]
        follow_input_t = follow_input.shape[-1]
        follow_output_t = follow_output.shape[-1]
        t_axes = (lead_t, follow_input_t, follow_output_t)

        # Returns the num_channels in leader in a list, ordered by required feature names
        list_n_channels_input_leader = [n * lead_t for n in in_data.feature_dims]

        if args.follower_input_use_input_features:
            list_n_channels_input_follower = out_data.get_feature_size_by_name(args.input_features)
        else:
            list_n_channels_input_follower = out_data.feature_dims
        list_n_channels_input_follower = [n * follow_input_t for n in list_n_channels_input_follower]

        list_n_channels_output_follower = [n * follow_output_t for n in out_data.feature_dims]

        return list_n_channels_input_leader, list_n_channels_input_follower, list_n_channels_output_follower, t_axes

    def ready_input_output(self, one_batch, args, dataset):
        lead, follow_input, follow_output = self.reshape_data(one_batch, args, dataset,
                                                              flatten=True)

        if hasattr(args, 'no_autoregressive') and args.no_autoregressive:
            input_data = lead
        else:
            input_data = torch.cat((lead, follow_input), dim=-1)

        return input_data, follow_output

    def ready_input_output_no_flatten(self, one_batch, args, dataset):
        lead, follow_input, follow_output = self.reshape_data(one_batch, args, dataset,
                                                              flatten=False)

        batch_size = lead.shape[0]
        lead = lead.reshape(batch_size, -1)
        follow_input = follow_input.reshape(batch_size, -1)

        if hasattr(args, 'no_autoregressive') and args.no_autoregressive:
            input_data = lead
        else:
            input_data = torch.cat((lead, follow_input), dim=-1)

        input_data = input_data.unsqueeze(-1)

        return input_data, follow_output

    @abstractmethod
    def get_meta_dict(self, motion_data, args):
        pass


class TrackingProcessor(BaseProcessor):
    def reshape_data(self, one_batch, args, dataset, flatten=True):
        lead = one_batch[0]
        follow = one_batch[1]
        t_axis = lead.shape[-1]
        n_future = int(t_axis * args.future_ratio)

        if n_future == t_axis:
            n_future -= 1

        center_frame_idx = t_axis - n_future - 1

        if args.all_input_in_transformed_coordinate:
            lead, follow = relative_motion.create_all_input_in_transformed_coordinate(lead, follow, center_frame_idx,
                                                                                      dataset)
        if args.use_relative_root_motion:
            lead, follow = relative_motion.create_relative_root_motion(lead, follow, center_frame_idx, dataset)
        if args.follower_in_leader_coordinate:
            lead, follow = relative_motion.create_fully_relative_root_motion(lead, follow, center_frame_idx, dataset)

        if not args.use_future:
            lead = lead[..., :-n_future + 1]

        follow_input = follow[..., -n_future - 1:-n_future]

        if args.follower_input_use_input_features:
            follower_sli = []
            for f in args.input_features:
                idx = dataset.datas[1].channel_names.index(f)
                s = sum(dataset.datas[1].feature_dims[:idx])
                e = s + dataset.datas[1].feature_dims[idx]
                follower_sli += list(range(s, e))
            follow_input = follow_input[..., follower_sli, :]

        follow_output = follow[..., -n_future:]

        if flatten:
            batch_size = lead.shape[0]
            lead = lead.reshape(batch_size, -1)
            follow_input = follow_input.reshape(batch_size, -1)
            follow_output = follow_output.reshape(batch_size, -1)

        if hasattr(args, 'no_autoregressive') and args.no_autoregressive:
            follow_input = follow_input[..., :0]
        return lead, follow_input, follow_output

    def get_meta_dict(self, motion_data, args):
        one_batch = motion_data[0]
        one_batch[0] = one_batch[0].unsqueeze(0)
        one_batch[1] = one_batch[1].unsqueeze(0)

        lead, follow_input, follow_output = self.reshape_data(one_batch, args, motion_data,
                                                              flatten=False)

        meta_dict = {"window_size": str(one_batch[0].shape[-1]),
                     "n_frame_leader_input": str(lead.shape[-1]),
                     "n_frame_follower_input": str(follow_input.shape[-1]),
                     "n_frame_follower_output": str(follow_output.shape[-1]),
                     "follower_output_offset": str(follow_input.shape[-1])}

        return meta_dict


class MappingProcessor(BaseProcessor):
    def reshape_data(self, one_batch, args, dataset, flatten=True):
        lead = one_batch[0]
        follow_original = follow = one_batch[1]

        if not args.use_input_window:
            lead = lead[..., :1]

        if args.history_portion > 0:
            leader_window_size = int(args.original_window * dataset.datas[0].target_fps) + 1
            follower_prediction_window_size = leader_window_size - 1
            lead = lead[..., :leader_window_size]
            leader_history_length = leader_window_size
            if args.use_partial_lead != 0:
                leader_history_length = int(lead.shape[-1] * args.use_partial_lead)
            lead = lead[..., -leader_history_length:]

            follower_history_length = 0
            if args.follower_autoregressive:
                follower_history_length = 1
                if args.follower_full_history:
                    follower_history_length = leader_history_length

            follow = follow[..., -(follower_history_length + follower_prediction_window_size):]
        else:
            raise Exception("history_portion should be > 0")

        lead, follow = relative_motion.create_relative_root_motion(lead, follow, args.center_frame_idx, dataset)

        follow_input = follow[..., :follower_history_length]
        follow_output = follow[..., -follower_prediction_window_size:]

        if flatten:
            batch_size = lead.shape[0]
            lead = lead.reshape(batch_size, -1)
            follow_input = follow_input.reshape(batch_size, -1)
            follow_output = follow_output.reshape(batch_size, -1)
        return lead, follow_input, follow_output

    def get_meta_dict(self, motion_data, args):
        one_batch = motion_data[0]
        one_batch[0] = one_batch[0].unsqueeze(0)
        one_batch[1] = one_batch[1].unsqueeze(0)

        lead, follow_input, follow_output = self.reshape_data(one_batch, args, motion_data,
                                                              flatten=False)

        meta_dict = {"window_size": str(lead.shape[-1] + follow_output.shape[-1]),
                     "n_frame_leader_input": str(lead.shape[-1]),
                     "n_frame_follower_input": str(follow_input.shape[-1]),
                     "n_frame_follower_output": str(follow_output.shape[-1]),
                     "follower_output_offset": str(lead.shape[-1])}

        return meta_dict


def processor_dispatcher(process_type: str) -> BaseProcessor:
    process_type = process_type.lower()
    if process_type == 'tracking':
        return TrackingProcessor()
    elif process_type == 'mapping':
        return MappingProcessor()
    else:
        raise ValueError(f"Unknown process type: {process_type}. Supported types are 'tracking' and 'mapping'.")
