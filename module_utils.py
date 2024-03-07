import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


def conv_blocks(in_channels, out_channels, kernel_size=1, **kwargs):
    return nn.Sequential(
        nn.BatchNorm1d(in_channels),
        GELU(),
        nn.Conv1d(in_channels, out_channels, kernel_size, padding="same", **kwargs),
    )


class Residual(nn.Module):
    """residuel block"""

    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, x, *args, **kwargs):
        return x + self._module(x, *args, **kwargs)


class SoftmaxPooling1D(nn.Module):

    def __init__(self, channels, pool_size=2, w_init_scale=2.0):
        """
        Args:
            channels: number of channels
            pool_size: pooling size
            w_init_scale: scale on the diagonal element.
        """
        super().__init__()
        self._pool_size = pool_size
        self._w_init_scale = w_init_scale
        self._logit_linear = nn.Linear(channels, channels, bias=False)
        self._logit_linear.weight.data.copy_(torch.eye(channels) * self._w_init_scale)

    def forward(self, x):
        assert x.shape[-1] % self._pool_size == 0, (
            "input length must " "by divisible by pool_size"
        )
        x = rearrange(x, "b c (l p) -> b l p c", p=self._pool_size)
        x = x * F.softmax(self._logit_linear(x), dim=-2)
        x = torch.sum(x, dim=-2)
        return rearrange(x, "b l c -> b c l")


class TargetLengthCrop(nn.Module):

    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length
        if seq_len < target_len:
            raise ValueError(
                f"sequence length {seq_len} is less than target length {target_len}"
            )

        trim = (target_len - seq_len) // 2
        return x[:, -trim:trim, :]
