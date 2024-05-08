import math
from typing import Optional

import torch
from torch import nn
from torch.autograd import forward_ad
import torch.nn.functional as F

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# Code inspired by these sources:
# hhttps://github.com/hkproj/pytorch-stable-diffusion/blob/main/sd/diffusion.py#L26
# https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/netjorks/nets/diffusion_model_unet.py#L589
#---------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float=0.0, timestep_emb_dim: int=1280) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.silu = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_lin = nn.Linear(timestep_emb_dim, out_channels)

        self.batch_norm_combined = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, time: torch.Tensor):
        remnant = x
        x = self.batch_norm(x)
        x = self.silu(x)
        x = self.conv1(x)

        time_emb = self.time_lin(self.silu(time))[:, :, None, None]

        x = x + time_emb

        x = self.batch_norm_combined(x)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return self.residual(remnant) + x
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

class Embedding(nn.Module):
    def __init__(self, in_dim: int, patch_size: int, img_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patch = (img_size // patch_size) ** 2
        
        self.patch_embed = self.PatchEmbed(in_dim, patch_size)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim))
        self.position_embed = nn.Parameter(torch.randn(1, self.num_patch, in_dim))

    def forward(self, x :torch.Tensor):
        x = self.patch_embed(x)
        # token = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((x, token), dim=1)

        x = x + self.position_embed

        return x

    class PatchEmbed(nn.Module):
        def __init__(self, in_dim: int, patch_size: int) -> None:
            super().__init__()
            self.patch_size = patch_size
            self.patch_dim = in_dim * patch_size ** 2

            self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

            # TODO, commented out for now, created problems in dimnesionality, maybe not needed
            # self.norm = nn.LayerNorm(self.patch_dim)
            self.linear = nn.Linear(self.patch_dim, in_dim)

        def forward(self, x: torch.Tensor):
            # -> b, (c * p * p), p_num    ... where (c * p * p) is (patch_dim)
            x = self.unfold(x)
            # print(f"PE (b, c*p*p, pnum): {x.shape}")

            # b, (c * p * p), p_num -> b, c, p, p, p_num
            # x = x.view(b, c, self.patch_size, self.patch_size, self.num_patches).permute(0, 4, 1, 2, 3)

            # x = self.norm(x)
            x = x.transpose(-1, -2)
            # print(f"PE prelin: {x.shape}")
            x = self.linear(x)
            # print(f"PE afterlin: {x.shape}")
            return x


class TransformerBlock(nn.Module):
    def __init__(self, in_dim: int, num_heads: int, patch_size: int, img_size: int, dropout: float=0.0, cross_attn_ctx_dim: Optional[int] = None) -> None:
        super().__init__()
        self.cross_attn_dim = cross_attn_ctx_dim
        
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(in_dim)

        # self.embedding = Embedding(in_dim, patch_size, img_size)

        self.pre_attn_norm1 = nn.LayerNorm(in_dim)
        self.attn1 = SelfAttn(in_dim=in_dim, num_heads=num_heads)

        self.pre_attn_norm2 = nn.LayerNorm(in_dim)
        self.attn2 = SelfAttn(in_dim=in_dim, num_heads=num_heads)
        if cross_attn_ctx_dim is not None:
            self.attn2 = CrossAttn(in_dim=in_dim, num_heads=num_heads, ctx_dim=cross_attn_ctx_dim)

        self.pre_out_norm = nn.LayerNorm(in_dim)
# also inspired by 
# https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        self.feed_forward = nn.Sequential(
            GeGLU(in_dim, in_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(in_dim * 4, in_dim)
        )
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # self.img_restore_linear = nn.Linear(in_dim, in_dim * patch_size ** 2)
        # self.fold = nn.Fold(output_size=img_size, kernel_size=patch_size, stride=patch_size)

        self.conv2 = nn.Conv2d(in_dim, in_dim, kernel_size=1) 

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        first_remnant = x
        
        x = self.batch_norm(x)
        x = self.conv1(x)
        
        _, _, h, _ = x.shape
        x = self.prepare_for_attention(x)
        # x = self.embedding(x)

        remnant = x
        x = self.pre_attn_norm1(x)
        x = self.attn1(x)
        x = x + remnant

        remnant = x
        x = self.pre_attn_norm2(x)
        if self.cross_attn_dim is None:
            x = self.attn2(x)
        elif context is not None:
            context = self.prepare_for_attention(context)
            x = self.attn2(x, context)
        else:
            raise AttributeError(
                f"if cross_attn_ctx_dim is defined, must pass context to forward method, but got None"
            )

        x = x + remnant

        remnant = x
        x = self.pre_out_norm(x)
        x = self.feed_forward(x)
        x = x + remnant

        x = self.unprepare_from_attention(x, h)
        # x = self.img_restore_linear(x)
        # x = x.transpose(-1, -2)
        # x = self.fold(x)

        x = self.conv2(x) + first_remnant

        return x

    def prepare_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise AttributeError(
                f"expected shape dimension of input and context for attention is 4, got {x.ndim}"
            )
        batch, dim, height, width = x.shape
        x = x.view((batch, dim, height * width))
        # 'move' the last dimension to be the second last: (batch, dim, height*width) -> (batch, height*width, dim)
        x = x.transpose(-1, -2)

        return x


    def unprepare_from_attention(self, x: torch.Tensor, height: int) -> torch.Tensor:
        batch, hw, dim = x.shape
        x = x.transpose(-1, -2)
        x = x.view((batch, dim, height, hw // height))
        return x


# taken from
# https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
class GeGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.proj = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x: torch.Tensor):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class SelfAttn(nn.Module):
    def __init__(self, in_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = in_dim // num_heads
        self.scale = 1 / math.sqrt(head_dim)

        self.to_q = nn.Linear(in_dim, in_dim)
        self.to_k = nn.Linear(in_dim, in_dim)
        self.to_v = nn.Linear(in_dim, in_dim)
        
        self.to_out = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        return self.attention(query, key, value)

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # source of the code for attention:
    # https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention
    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        # the last (inferred) shape dim should be the same as self.head_dim
        q = query.view(*query.shape[:2], self.num_heads, -1)
        k = key.view(*key.shape[:2], self.num_heads, -1)
        v = value.view(*value.shape[:2], self.num_heads, -1)

        # ------------------------------------------------------------
        # replacement for the query-key-value calculations
        out = F.scaled_dot_product_attention(q,k,v, scale=self.scale)
        # ------------------------------------------------------------

        out = out.reshape(*out.shape[:2], -1)
        
        return self.to_out(out)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class CrossAttn(nn.Module):
    def __init__(self, in_dim: int, num_heads: int, ctx_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = in_dim // num_heads
        self.scale = 1 / math.sqrt(head_dim)

        self.to_q = nn.Linear(in_dim, in_dim)
        self.to_k = nn.Linear(ctx_dim, in_dim)
        self.to_v = nn.Linear(ctx_dim, in_dim)
        
        self.to_out = nn.Linear(in_dim, in_dim)

    def forward(self, x, context):
        query = self.to_q(x)
        key = self.to_k(context)
        value = self.to_v(context)

        return self.attention(query, key, value)

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # source of the code for attention:
    # https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention
    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        q = query.view(*query.shape[:2], self.num_heads, -1)
        k = key.view(*key.shape[:2], self.num_heads, -1)
        v = value.view(*value.shape[:2], self.num_heads, -1)

        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale

        # perform the attention in-place to save memory (according to source)
        half = attn.shape[0] // 2
        attn[half:] = attn[half:].softmax(dim=-1)
        attn[:half] = attn[:half].softmax(dim=-1)
        
        out = torch.einsum('bhij,bjhd->bihd', attn, v)

        out = out.reshape(*out.shape[:2], -1)
        
        return self.to_out(out)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

class UNETR(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_heads: int,
                 patch_size: int,
                 img_size: int,
                 dropout: float = 0.0,
                 cross_attn_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.cross_attn_dim = cross_attn_dim

        first_channel = 32
        time_embed_dim = first_channel * 4
        # Time embedding
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # code from MONAI-Generative:
        # https://github.com/Project-MONAI/GenerativeModels/blob/ef6b7e6356ff15a201585fc4e0f9f1f58228d5b1/generative/networks/nets/diffusion_model_unet.py#L1758
        self.time_embed = nn.Sequential(
            nn.Linear(first_channel, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # Encoder
        self.encoder_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(in_channels, first_channel, kernel_size=3, padding=1),
            ]),

            nn.ModuleList([
                ResBlock(first_channel, 64, timestep_emb_dim=time_embed_dim, dropout=dropout),
                ResBlock(64, 64, timestep_emb_dim=time_embed_dim, dropout=dropout),
            ]),
            
            # (H, W) -> (H / 2, W / 2)
            nn.ModuleList([
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),

            ]),

            nn.ModuleList([
                ResBlock(128, 128, timestep_emb_dim=time_embed_dim, dropout=dropout),
                TransformerBlock(128, num_heads=num_heads, patch_size=patch_size, img_size=img_size // 2, dropout=dropout),
            ]),
            
            # (H / 2, W / 2) -> (H / 4, W / 4)
            nn.ModuleList([
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            ]),
            
            nn.ModuleList([
                ResBlock(256, 256, timestep_emb_dim=time_embed_dim, dropout=dropout),
                TransformerBlock(256, num_heads=num_heads, patch_size=patch_size, img_size=img_size // 4, dropout=dropout),
            ]),
        ])

        # Bottle-neck
        self.bottleneck = nn.ModuleList([
            ResBlock(256, 256, timestep_emb_dim=time_embed_dim, dropout=dropout),
            TransformerBlock(256, num_heads=num_heads, patch_size=patch_size, img_size=img_size // 8, dropout=dropout),
            ResBlock(256, 256, timestep_emb_dim=time_embed_dim, dropout=dropout),
        ])

        # Decoder (symetric to the Encoder)
        self.decoder_layers = nn.ModuleList([            
            nn.ModuleList([
                ResBlock(256 + 256, 256, timestep_emb_dim=time_embed_dim, dropout=dropout),
                TransformerBlock(256, num_heads=num_heads, patch_size=patch_size, img_size=img_size // 4, dropout=dropout),
            ]),

            nn.ModuleList([
                ResBlock(256 + 256, 256, timestep_emb_dim=time_embed_dim, dropout=dropout),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            ]),

            nn.ModuleList([
                ResBlock(128 + 128, 128, timestep_emb_dim=time_embed_dim, dropout=dropout),
                TransformerBlock(128, num_heads=num_heads, patch_size=patch_size, img_size=img_size // 2, dropout=dropout),
            ]),
            
            nn.ModuleList([
                ResBlock(128 + 128, 128, timestep_emb_dim=time_embed_dim, dropout=dropout),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            ]),
                        
            nn.ModuleList([
                ResBlock(64 + 64, 64, timestep_emb_dim=time_embed_dim, dropout=dropout),
                ResBlock(64, 64, timestep_emb_dim=time_embed_dim, dropout=dropout),
            ]),

            nn.ModuleList([
                ResBlock(64 + first_channel, first_channel, timestep_emb_dim=time_embed_dim, dropout=dropout),
            ]),
        ])

        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # source:
        # https://github.com/hkproj/pytorch-stable-diffusion/blob/main/sd/diffusion.py#L306
        self.out_layer = nn.Sequential(
            nn.BatchNorm2d(num_features=first_channel),
            nn.SiLU(),
            nn.Conv2d(first_channel, out_channels, kernel_size=3, padding=1),
        )
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def forward(self,
                x: torch.Tensor,
                time: torch.Tensor,
                context: Optional[torch.Tensor] = None
    ):
        
        time = time.to(x.dtype)
        time = self._get_timestep_embeddings(time, 32)

        time_embed = self.time_embed(time)

        residues = []
        for layers in self.encoder_layers:
            x = self._forward_layer(layers, x, time_embed, context)
            residues.append(x)

        x = self._forward_layer(self.bottleneck, x, time_embed, context)

        for layers in self.decoder_layers:
            x = torch.cat((residues.pop(), x), dim=1)
            x = self._forward_layer(layers, x, time_embed, context)

        x = self.out_layer(x)
        return x

    def _forward_layer(self,
                       layers: nn.ModuleList,
                       x: torch.Tensor,
                       time_embed: torch.Tensor,
                       context: Optional[torch.Tensor]
    ):
        for layer in layers:
            if isinstance(layer, ResBlock):
                x = layer(x, time_embed)
            elif isinstance(layer, TransformerBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # Code from MONAI-Generative
    # https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/networks/nets/diffusion_model_unet.py#L461
    def _get_timestep_embeddings(self, timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError("Timesteps should be a 1d-array")

        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        freqs = torch.exp(exponent / half_dim)

        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # zero pad
        if embedding_dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

        return embedding
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

