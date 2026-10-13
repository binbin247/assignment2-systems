
import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    def __init__(self, context_length: int, dim: int, theta: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache", RotaryEmbedding._init_cache(context_length, dim, theta), persistent=False
        )

    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float) -> Tensor:
        assert dim % 2 == 0

        d = torch.arange(0, dim, 2) / dim
        freqs = torch.tensor(theta) ** -d
        t = torch.arange(context_length)

        freqs = t.unsqueeze(-1) * freqs.unsqueeze(0)

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(self, x: Tensor, pos_ids: Tensor | None) -> Tensor:
        x1, x2 = x[..., 0::2], x[..., 1::2]

        if pos_ids is not None:
            cos, sin = self._freq_cis_cache[:, pos_ids, :].unbind(0)
        else:
            seq_len = x.size(-2)
            cos, sin = self._freq_cis_cache[:, :seq_len, :].unbind(0)

        # 2D rotation matrix applied to pairs in x
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        return torch.cat((x1_rot, x2_rot), dim=-1)

    def extra_repr(self):
        return f"context_length={self._freq_cis_cache.shape[1]}, dim/2={self._freq_cis_cache.shape[2]}"
