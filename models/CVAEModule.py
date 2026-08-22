import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalLayer(torch.nn.Module):
    def __init__(self, samples_size, mean=True, sigma=True):
        super(VariationalLayer, self).__init__()

        self.Mu = nn.Linear(samples_size, samples_size)
        self.LogVar = nn.Linear(samples_size, samples_size)

        self.Mean = mean
        self.Sigma = sigma

    def forward(self, x, activation):
        x, _ = activation(x)

        mu = self.Mu(x) if self.Mean else 0.0
        logvar = self.LogVar(x)
        std = torch.exp(0.5 * logvar) if self.Sigma else 1.0

        if self.training:
            z = torch.randn_like(std) * std + mu
        else:
            # Deterministic inference/export: skip sampling, i.e. act as if variance == 0.
            z = mu

        if True:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
            return z, mu, std, kld_loss
        else:
            return z


# BATTLE-TESTED
class LinearEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, dropout):
        super(LinearEncoder, self).__init__()

        self.InputSize = input_size
        self.OutputSize = output_size

        self.Dropout = dropout

        self.L1 = nn.Linear(input_size, hidden1_size)
        self.L2 = nn.Linear(hidden1_size, hidden2_size)
        self.L3 = nn.Linear(hidden2_size, output_size)

    def forward(self, z, initial=True):
        if initial:
            z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L1(z)
        z = F.elu(z)

        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L2(z)
        z = F.elu(z)

        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L3(z)

        return z


class CVAEModel(nn.Module):
    def __init__(self, input_dim, output_dim, encoder_dim, estimator_dim, decoder_dim, codebook_channels, codebook_dim,
                 dropout, activation_codebook):
        super(CVAEModel, self).__init__()

        self.XDim = input_dim
        self.YDim = output_dim

        self.Dropout = dropout

        self.C = codebook_channels
        self.D = codebook_dim

        codebook_size = codebook_channels * codebook_dim

        if activation_codebook == 'softmax':
            self.activation = self.softmax
        elif activation_codebook == 'normalize':
            self.activation = self.normalize
        elif activation_codebook == 'none':
            self.activation = lambda x: (x, None)
        else:
            raise ValueError(
                "activation_codebook must be one of: softmax, normalize, none; "
                f"got {activation_codebook!r}"
            )

        print("Training End-To-End")
        self.Encoder = LinearEncoder(output_dim + input_dim, encoder_dim, encoder_dim, codebook_size, dropout)
        self.Estimator = LinearEncoder(input_dim, estimator_dim, estimator_dim, codebook_size, dropout)
        self.EncoderSampler = VariationalLayer(codebook_size, True, True)
        self.EstimatorSampler = VariationalLayer(codebook_size, True, True)
        self.Decoder = LinearEncoder(codebook_size + input_dim, decoder_dim, decoder_dim, output_dim,
                                             dropout)

        print("Input Dim", input_dim)
        print("Output Dim", output_dim)

    def softmax(self, z):
        z = z.reshape(-1, self.C, self.D)
        z = F.softmax(z, dim=-1)

        z_soft = z

        shape = z.size()
        _, ind = z.max(dim=-1)
        z_hard = torch.zeros_like(z).view(-1, shape[-1])
        z_hard.scatter_(1, ind.view(-1, 1), 1)
        z_hard = z_hard.view(*shape)
        z_hard = (z_hard - z).detach() + z

        z_soft = z_soft.reshape(-1, self.C * self.D)
        z_hard = z_hard.reshape(-1, self.C * self.D)
        return z_soft, z_hard

    def normalize(self, z):
        z = z.reshape(-1, self.C, self.D)
        z = z / z.norm(dim=-1, keepdim=True)
        z = z.reshape(-1, self.C * self.D)

        return z, None

    def learn(self, x, t):  # x: Inputs t: Outputs
        mse_fn = nn.MSELoss()

        # Select
        state = x

        # Encode Y
        target = self.Encoder(torch.cat((t, x), dim=-1))
        target, target_mu, target_std, target_kld = self.EncoderSampler(target, self.activation)

        # Encode X
        estimate = self.Estimator(x)
        estimate, estimate_mu, estimate_std, estimate_kld = self.EstimatorSampler(estimate, self.activation)

        # Compute Matching Loss
        match_loss = mse_fn(target, estimate)
        # match_loss = 0.5 * (mse_fn(target_mu, estimate_mu) + mse_fn(target_std, estimate_std))

        # Construct Code
        code = torch.cat((target, state), dim=-1)

        # Decode
        y = self.Decoder(code)

        # Compute Reconstruction Loss
        rec_loss = mse_fn(t, y)

        loss = {
            "rec": rec_loss,
            "matching": match_loss
        }

        return loss, y

    def forward(self, x):

        # Select
        state = x

        # Encode X
        estimate = self.Estimator(x)
        estimate, mu, std, kl_loss = self.EstimatorSampler(estimate, self.activation)

        # Construct Code
        code = torch.cat((estimate, state), dim=-1)

        # Decode
        y = self.Decoder(code)

        return y, {'code': code, 'mu': mu, 'std': std, 'kl_loss': kl_loss}
