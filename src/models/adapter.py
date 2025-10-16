# models/adapter.py
import torch
import torch.nn as nn

class Adapter(nn.Module):
    """
    Lightweight bottleneck adapter to map LLaVA features -> 49-d per-frame attention vector.
    Input: (B, T, in_dim)
    Output: (B, T, out_dim) where out_dim is 49 to match downstream att_frame_feature
    """
    def __init__(self, in_dim: int, bottleneck: int = 256, out_dim: int = 49, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.down = nn.Linear(in_dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # residual projection if dims don't align (here in_dim != out_dim typically)
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D_in)
        assert x.dim() == 3, f"Adapter expects 3D input (B,T,D), got {x.shape}"
        res = x
        x = self.down(x)      # (B,T,bottleneck)
        x = self.act(x)
        x = self.up(x)        # (B,T,out_dim)
        x = self.dropout(x)
        if self.res_proj is not None:
            res = self.res_proj(res)
        out = self.norm(x + res)
        return out
