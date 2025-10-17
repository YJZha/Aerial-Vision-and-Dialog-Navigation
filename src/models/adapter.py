# models/adapter.py
import torch
import torch.nn as nn
import math

class Adapter(nn.Module):
    """
    Adapter that projects fused LLaVA vector to (512,49) per sample.
    Input: (B, in_dim)
    Output: (B, 512, 49)
    """
    def __init__(self, in_dim=768*2, mid_dim=1024, out_c=512, out_w=49, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.mid = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = out_c * out_w
        self.proj = nn.Linear(mid_dim, self.out_dim)
        self.out_c = out_c
        self.out_w = out_w

    def forward(self, x):
        # x: (B, in_dim)
        h = self.mid(x)
        p = self.proj(h)  # (B, out_c*out_w)
        p = p.view(-1, self.out_c, self.out_w)
        return p
