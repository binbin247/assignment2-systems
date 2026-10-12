import torch
from torch import nn, Tensor
import torch.nn.functional as F

print(torch.__version__)

class CausalMultiHeadAttention(nn.Module):
    def __init__(
            self, 
            d_model: int,
            n_heads: int,
            d_head: int | None = None
            ):

        super().__init__()
        self.d_model = d_model # model dimension
        self.n_heads = n_heads # number of head
        self.d_head = d_head if d_head else d_model//n_heads # dimension of head

        self.q_proj = nn.Linear(in_features=self.d_model, out_features=self.n_heads*self.d_head, bias=False)
        self.k_proj = nn.Linear(in_features=self.d_model, out_features=self.n_heads*self.d_head, bias=False)
        self.v_proj = nn.Linear(in_features=self.d_model, out_features=self.n_heads*self.d_head, bias=False)

        self.out_proj = nn.Linear(in_features=self.n_heads*self.d_head, out_features=self.d_model, bias=False)


    def forward(
            self,
            x: Tensor # size = (B, T, C)
    ) -> Tensor:
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.d_model, "model dimension is not correct"
        # size = (B, n_heads, T, d_head)
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(-2, -3)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(-2, -3)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(-2, -3)

        atten = F.scaled_dot_product_attention(
            query=Q, key=K, value=V, is_causal=True, dropout_p=0
            )
        atten = atten.transpose(-2, -3).contiguous().view(batch_size, seq_len, self.n_heads*self.d_head)
        return self.out_proj(atten)
