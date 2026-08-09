import torch


def make_complex(a):
    return a[..., 0] + 1j * a[..., 1]


def transform_root_motion(root_motion, ref_rot, ref_pos):
    res = torch.empty_like(root_motion)

    rot = make_complex(root_motion[..., :2])
    rot = rot / ref_rot
    res[..., 0] = rot.real
    res[..., 1] = rot.imag

    pos = make_complex(root_motion[..., 2:4])
    pos = (pos - ref_pos) / ref_rot
    res[..., 2] = pos.real
    res[..., 3] = pos.imag

    if res.shape[-1] > 4:
        res[..., 4:6] = root_motion[..., 4:6]

        velo = make_complex(root_motion[..., 6:])
        velo = velo / ref_rot
        res[..., 6] = velo.real
        res[..., 7] = velo.imag

    return res


def extract_unnomralized_root(motion, motion_data):
    idx = motion_data.channel_names.index("RootMotion")
    s = sum(motion_data.feature_dims[:idx])
    e = sum(motion_data.feature_dims[:idx + 1])
    return motion[..., s:e] * motion_data.data_std[s:e] + motion_data.data_mean[s:e]


def put_back_root(motion, motion_data, root_motion):
    motion = motion.clone()
    idx = motion_data.channel_names.index("RootMotion")
    s = sum(motion_data.feature_dims[:idx])
    e = sum(motion_data.feature_dims[:idx + 1])
    root_motion = (root_motion - motion_data.data_mean[s:e]) / motion_data.data_std[s:e]
    motion[..., s:e] = root_motion
    return motion


def create_relative_root_motion(lead, follow, center_frame_idx, dataset):
    """
    Args:
        root_motion0: (batch_size, n_dim, n_frame)
        root_motion1: (batch_size, n_dim, n_frame)
    Returns:

    """
    lead = lead.permute(0, 2, 1)
    follow = follow.permute(0, 2, 1)

    root_lead = extract_unnomralized_root(lead, dataset.datas[0])
    root_follow = extract_unnomralized_root(follow, dataset.datas[1])

    if dataset.reference_char == 0:
        center_root_motion = root_lead[:, center_frame_idx, :].unsqueeze(1)
    elif dataset.reference_char == 1:
        center_root_motion = root_follow[:, center_frame_idx, :].unsqueeze(1)

    ref_rot = make_complex(center_root_motion[..., :2])
    ref_pos = make_complex(center_root_motion[..., 2:4])

    relative_root_follow = transform_root_motion(root_follow, ref_rot, ref_pos)
    relative_root_lead = transform_root_motion(root_lead, ref_rot, ref_pos)

    new_lead = put_back_root(lead, dataset.datas[0], relative_root_lead)
    new_follow = put_back_root(follow, dataset.datas[1], relative_root_follow)

    new_lead = new_lead.permute(0, 2, 1)
    new_follow = new_follow.permute(0, 2, 1)

    return new_lead, new_follow


# Not used for now


def get_features(motion, dataset):
    motion = motion * dataset.data_std + dataset.data_mean
    features = []
    for i in range(len(dataset.feature_dims)):
        s = sum(dataset.feature_dims[:i])
        e = sum(dataset.feature_dims[:i + 1])
        f = motion[..., s:e]
        if dataset.channel_names[i] != "RootMotion" and f.shape[-1] != 0:
            f = f.reshape(f.shape[0], f.shape[1], dataset.num_joints, f.shape[2] // dataset.num_joints)
        features.append(f)
    return features


def assemble_features(features, dataset):
    motion = []
    for i in range(len(features)):
        f = features[i]
        motion.append(f.reshape(f.shape[0], f.shape[1], -1))
    motion = torch.cat(motion, dim=-1)
    return (motion - dataset.data_mean) / dataset.data_std


def construct_mat44(root_motion):
    shape = root_motion.shape
    root_motion = root_motion.reshape(-1, shape[-1])
    mat = torch.eye(4, device=root_motion.device).repeat(root_motion.shape[0], 1, 1)

    rot = root_motion[..., :2]
    pos = root_motion[..., 2:4]

    angle = torch.atan2(rot[..., 1], rot[..., 0])
    angle = -angle
    c = torch.cos(angle)
    s = torch.sin(angle)

    # Construct rotation matrix along y-axis
    mat[..., 0, 0] = c
    mat[..., 0, 2] = s
    mat[..., 2, 0] = -s
    mat[..., 2, 2] = c

    mat[..., 0, 3] = pos[..., 0]
    mat[..., 2, 3] = pos[..., 1]

    mat = mat.reshape(shape[:-1] + (4, 4))
    return mat


def invert_rigid_transform(batch_matrix):
    # Extract the rotation and translation components
    matrix_shape = batch_matrix.shape
    batch_matrix = batch_matrix.reshape(-1, 4, 4)
    rotation = batch_matrix[:, :3, :3]  # (batch_size, 3, 3)
    translation = batch_matrix[:, :3, 3]  # (batch_size, 3)

    # Transpose the rotation matrix (equivalent to its inverse)
    rotation_transpose = rotation.transpose(1, 2)  # (batch_size, 3, 3)

    # Compute the inverted translation
    inverted_translation = -torch.bmm(rotation_transpose, translation.unsqueeze(-1)).squeeze(-1)  # (batch_size, 3)

    # Construct the inverted transformation matrix
    inverted_matrix = torch.eye(4, device=batch_matrix.device).repeat(batch_matrix.shape[0], 1, 1)  # (batch_size, 4, 4)
    inverted_matrix[:, :3, :3] = rotation_transpose
    inverted_matrix[:, :3, 3] = inverted_translation
    return inverted_matrix.reshape(matrix_shape)


def apply_affine(mat, vec):
    rot = mat[..., :3, :3]
    translation = mat[..., :3, 3]
    return (rot @ vec.unsqueeze(-1)).squeeze(-1) + translation


def create_fully_relative_root_motion(lead, follow, center_frame_idx, dataset):
    """
        Args:
            lead: (batch_size, n_dim, n_frame)
            follow: (batch_size, n_dim, n_frame)
        Returns:

    """
    lead = lead.permute(0, 2, 1)
    follow = follow.permute(0, 2, 1)

    motions = [lead, follow]


    root_lead = extract_unnomralized_root(lead, dataset.datas[0])
    ref_root_mat = construct_mat44(root_lead[:, center_frame_idx, :])
    ref_root_mat_inv = invert_rigid_transform(ref_root_mat)

    for choice in range(2):
        target_motion = motions[choice]
        target_features = get_features(target_motion, dataset.datas[choice])

        current_root_mat = construct_mat44(target_motion)
        transitional_mat = ref_root_mat_inv.unsqueeze(1) @ current_root_mat

        for i in range(len(dataset.datas[choice].channel_names)):
            channel_name = dataset.datas[choice].channel_names[i]
            if channel_name == "RootMotion":
                continue
            elif channel_name == "Rotations":
                r = target_features[i]
                r = r.reshape(r.shape[:-1] + (3, 3))
                r = transitional_mat[..., None, :3, :3] @ r
                target_features[i] = r
            elif channel_name == "Positions":
                p = target_features[i]
                p = apply_affine(transitional_mat.unsqueeze(2), p)
                target_features[i] = p
            elif channel_name == "VelocitiesV2":
                v = target_features[i]
                v = v.unsqueeze(-1)
                rot = transitional_mat[..., :3, :3]
                rot = rot.reshape(rot.shape[0], rot.shape[1], 1, 3, 3)
                v = (rot @ v).squeeze(-1)
                target_features[i] = v
            else:
                raise ValueError("Unsupported channel name: " + channel_name)

        motions[choice] = assemble_features(target_features, dataset.datas[choice])

    lead, follow = motions
    lead = lead.permute(0, 2, 1)
    follow = follow.permute(0, 2, 1)

    create_relative_root_motion(lead, follow, center_frame_idx, dataset)

    return lead, follow


def transform_root(input_features, center_transform, dataset):
    """
            Args:
                lead: (batch_size, n_dim, n_frame)
                follow: (batch_size, n_dim, n_frame)
            Returns:

        """
    input_features = input_features.permute(0, 2, 1)

    root_motion = extract_unnomralized_root(input_features, dataset)
    root_mat = construct_mat44(root_motion)
    center_transform_inv = invert_rigid_transform(center_transform)

    transitional_mat = center_transform_inv.unsqueeze(1) @ root_mat

    input_features = get_features(input_features, dataset)

    for i in range(len(dataset.channel_names)):
        channel_name = dataset.channel_names[i]
        if channel_name == "RootMotion":
            continue
        elif channel_name == "Rotations":
            r = input_features[i]
            r = r.reshape(r.shape[:-1] + (3, 3))
            r = transitional_mat[..., None, :3, :3] @ r
            input_features[i] = r
        elif channel_name == "Positions":
            p = input_features[i]
            p = apply_affine(transitional_mat.unsqueeze(2), p)
            input_features[i] = p
        elif channel_name == "VelocitiesV2":
            v = input_features[i]
            v = v.unsqueeze(-1)
            rot = transitional_mat[..., :3, :3]
            rot = rot.reshape(rot.shape[0], rot.shape[1], 1, 3, 3)
            v = (rot @ v).squeeze(-1)
            input_features[i] = v
        else:
            raise ValueError("Unsupported channel name: " + channel_name)

    transformed_features = assemble_features(input_features, dataset)

    transformed_features = transformed_features.permute(0, 2, 1)

    return transformed_features


def create_all_input_in_transformed_coordinate(lead, follow, center_frame_idx, dataset):
    root_follow = extract_unnomralized_root(follow.permute(0, 2, 1), dataset.datas[1])
    lead = transform_root(lead, construct_mat44(root_follow[:, center_frame_idx, :]), dataset.datas[0])
    return lead, follow
