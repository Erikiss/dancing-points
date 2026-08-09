import torch.nn as nn

from utils.running_statics import RunningStatistics


class MLPChannels(nn.Module):
    def __init__(self, n_channels, bn, dropout=0):
        super().__init__()
        self.layers = []
        self.n_channels = n_channels
        dropout = nn.Dropout(dropout, inplace=True) if dropout > 0 else None
        for i in range(len(n_channels) - 1):
            if dropout is not None:
                self.layers.append(dropout)
            self.layers.append(nn.Linear(n_channels[i], n_channels[i + 1]))
            if i != len(n_channels) - 2:
                if bn:
                    self.layers.append(nn.BatchNorm1d(n_channels[i + 1]))
                self.layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class RunningNormalizeModel(nn.Module):
    def __init__(self, model, in_dims=None, out_dims=None):
        super().__init__()
        self.in_s = RunningStatistics(dims=in_dims)
        self.out_s = RunningStatistics(dims=out_dims)
        self.model = model

    def forward(self, x, *args, **kargs):
        x = self.in_s.normalize(x)
        y, info = self.model(x, *args, **kargs)
        y = self.out_s.denormalize(y)
        return y, info

    def learn(self, input, output):
        if self.training:
            self.in_s.update(input)
            self.out_s.update(output)

        if self.out_s.r_count >= 1000:
            return self.model.learn(self.in_s.normalize(input), self.out_s.normalize(output))
        else:
            return {}, None
