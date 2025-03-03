"""
Vision Transformer (ViT) for diffusion model based on DiT architecture.
Adapted from https://github.com/facebookresearch/DiT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + self.b


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C

        qkv = (
            self.qkv(x)
            .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim, mlp_hidden_dim, dropout=drop)

    def forward(self, x, scale_shift=None):
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x + self.attn(self.norm1(x))
            x = x * (scale + 1) + shift
            x = x + self.mlp(self.norm2(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        time_emb_dim=None,
        cond_emb_dim=0,
    ):
        super().__init__()
        self.cond_emb_dim = cond_emb_dim
        self.tc_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(int(time_emb_dim) + int(cond_emb_dim), dim * 2)
        )
        self.transformer = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
        )

    def forward(self, x, time_emb=None, cond_emb=None):
        scale_shift = None
        # Add time step and condition embeddings
        if cond_emb is not None:
            assert self.cond_emb_dim > 0
            tc_emb = tuple((time_emb, cond_emb))
            tc_emb = torch.cat(tc_emb, dim=1)
        else:
            assert self.cond_emb_dim == 0
            tc_emb = time_emb
        tc_emb = self.tc_mlp(tc_emb)
        tc_emb = rearrange(tc_emb, "b c -> b c 1 1")
        scale_shift = tc_emb.chunk(2, dim=1)

        return self.transformer(x, scale_shift)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.view(-1)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ViT(nn.Module):
    def __init__(
        self,
        dim,
        embed_input_dim=128,  # embedding dim of regression label
        cond_drop_prob=0.5,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        in_channels=3,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        attn_dim_head=32,
        attn_heads=4,
        patch_size=2,
        num_blocks=8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_drop_prob = cond_drop_prob
        self.patch_size = patch_size

        # Determine dimensions
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(in_channels, init_dim, kernel_size=7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time embeddings
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Condition embeddings
        self.cond_mlp_1 = nn.Sequential(
            nn.Linear(embed_input_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )

        self.null_cond_emb = nn.Parameter(
            -1 * torch.abs(torch.randn(dim)), requires_grad=True
        )

        cond_emb_dim = dim * 4
        self.cond_mlp_2 = nn.Sequential(
            nn.Linear(dim, cond_emb_dim),
            nn.BatchNorm1d(cond_emb_dim),
            nn.ReLU(),
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dims[-1],
                    num_heads=attn_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    time_emb_dim=time_dim,
                    cond_emb_dim=cond_emb_dim,
                )
                for _ in range(num_blocks)
            ]
        )

        # Output
        default_out_dim = in_channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_conv = nn.Conv2d(dims[-1], self.out_dim, 1)

    def forward(
        self,
        x,
        timesteps,
        labels_emb,
        cond_drop_prob=None,
        keep_mask=None,
        return_bottleneck=False,
    ):
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # Get condition embedding
        c_emb = self.cond_mlp_1(labels_emb)

        if cond_drop_prob > 0:
            if keep_mask is not None:
                self.keep_mask = keep_mask
            else:
                self.keep_mask = prob_mask_like(
                    (batch,), 1 - cond_drop_prob, device=device
                )

            null_cond_emb = repeat(self.null_cond_emb, "d -> b d", b=batch)

            c_emb = torch.where(
                rearrange(self.keep_mask, "b -> b 1"), c_emb, null_cond_emb
            )

        c_emb = self.cond_mlp_2(c_emb)

        # Process input through ViT
        x = self.init_conv(x)

        t_emb = self.time_mlp(timesteps)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, t_emb, c_emb)

        if return_bottleneck:
            return x

        # Final output
        return self.final_conv(x)


# Helper functions
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered
