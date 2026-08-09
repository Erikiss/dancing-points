import torch
import torch.nn as nn


class RunningStatistics(nn.Module):
    def __init__(self, cap=1e-3, dims=None, last_dim_only=False, additional_weight=1.0):
        super().__init__()
        self.r_mean = None
        self.r_var = None

        if dims is not None:
            self.r_mean = nn.Parameter(torch.zeros(dims), requires_grad=False)
            self.r_var = nn.Parameter(torch.ones(dims), requires_grad=False)

        self.r_count = 0
        self.cap = cap
        self.last_dim_only = last_dim_only
        if not isinstance(additional_weight, torch.Tensor) and additional_weight == 1.0:
            self.additional_weight = additional_weight
        else:
            if not isinstance(additional_weight, torch.Tensor):
                additional_weight = torch.tensor(additional_weight)
            self.additional_weight = nn.Parameter(additional_weight, requires_grad=False)

    def initialize_from_dataset(self, dataset):
        self.r_mean = nn.Parameter(torch.from_numpy(dataset.data_mean))
        self.r_var = nn.Parameter(torch.from_numpy(dataset.data_std ** 2))
        self.additional_weight = 1.

    def my_load_state_dict(self, state_dict):
        if 'additional_weight' in state_dict:
            self.additional_weight = nn.Parameter(state_dict['additional_weight'])
        self.load_state_dict(state_dict)

    @property
    def count(self):
        return self.r_count

    def update(self, x):
        """
        Args:
            x: (batch_size, dim)
        """
        if x is None:
            return

        if self.last_dim_only:
            x = x.reshape(-1, x.shape[-1])
        x = x.detach()
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]

        if self.r_mean is None:
            self.r_mean = nn.Parameter(batch_mean, requires_grad=False)
            self.r_var = nn.Parameter(batch_var, requires_grad=False)
            self.r_count = batch_count
            return

        delta = batch_mean - self.r_mean
        new_mean = self.r_mean + delta * batch_count / (self.r_count + batch_count)
        m_a = self.r_var * self.r_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.r_count * batch_count / (self.r_count + batch_count)
        new_var = M2 / (self.r_count + batch_count)
        self.r_mean[:] = new_mean
        self.r_var[:] = new_var
        self.r_count += batch_count

    def normalize(self, x):
        if x is None:
            return None
        std = torch.sqrt(self.r_var + self.cap ** 2) * self.additional_weight
        return (x - self.r_mean) / std

    def denormalize(self, x):
        if x is None:
            return None
        std = torch.sqrt(self.r_var + self.cap ** 2) * self.additional_weight
        return x * std + self.r_mean
