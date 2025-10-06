import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, frames_per_patch: int = 6, expansion: int = 2):
        super().__init__()
        self.d_model = d_model
        self.frames_per_patch = frames_per_patch
        # Frame embedding (collapse pitch dim)
        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(88, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.norm_frame = nn.LayerNorm(d_model)
        # anti-aliasing conv on time axis
        self.aa = nn.Conv1d(d_model, d_model, kernel_size=3, stride=1,
                            padding=1, groups=d_model, bias=False)
        
        # Late temporal pooling (downsample frames -> patches)
        self.glu_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * expansion * 2,
            kernel_size=frames_per_patch,
            stride=frames_per_patch,
            padding=0,
            bias=True,
        )
        self.project = nn.Conv1d(
            in_channels=d_model * expansion,
            out_channels=d_model,
            kernel_size=1,
        )
        self.norm_temporal = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 88, T)
        x = self.conv2d(x)                      # (B, C, 1, T)
        x = x.squeeze(2).transpose(1, 2)        # (B, T, C)
        x = self.norm_frame(x)

        # anti-aliased and temporal pooling
        x = x.transpose(1, 2)                   # (B, C, T)
        x = self.aa(x)                          # (B, C, T)
        v, g = self.glu_conv(x).chunk(2, dim=1)
        x = self.project(v * torch.sigmoid(g))  # (B, C, T//k)
        x = x.transpose(1, 2)                   # (B, T//k, C)
        return self.norm_temporal(x)


def downsample_key_padding_mask(mask: Optional[torch.Tensor], frames_per_patch: int) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    bsz, total_len = mask.shape
    if total_len < frames_per_patch:
        return mask.new_zeros((bsz, 0), dtype=mask.dtype)
    out_len = total_len // frames_per_patch
    trimmed = mask[:, : out_len * frames_per_patch]
    grouped = trimmed.view(bsz, out_len, frames_per_patch)
    return grouped.all(dim=-1)


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int) -> None:
        super().__init__()
        if max_distance < 1:
            raise ValueError("max_distance must be >= 1")
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bias = nn.Parameter(torch.zeros(2 * max_distance - 1, num_heads))

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device)
        rel = pos[:, None] - pos[None, :]
        rel = rel.clamp(-self.max_distance + 1, self.max_distance - 1)
        rel = rel + self.max_distance - 1
        bias = self.bias[rel]
        return bias.permute(2, 0, 1).to(dtype=dtype)


class RelativeTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ff_dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation_fn = F.gelu
        elif activation == "relu":
            self.activation_fn = F.relu
        else:
            raise ValueError(f"unsupported activation: {activation}")

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = src.size()
        qkv = self.qkv_proj(src)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.nhead, self.head_dim)
        k = k.view(bsz, seq_len, self.nhead, self.head_dim)
        v = v.view(bsz, seq_len, self.nhead, self.head_dim)

        attn_scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.head_dim)

        if src_mask is not None:
            if src_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(src_mask.unsqueeze(0), float("-inf"))
            else:
                attn_scores = attn_scores + src_mask.unsqueeze(0)

        if src_key_padding_mask is not None:
            key_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_mask, float("-inf"))

        if attn_bias is not None:
            if attn_bias.dim() == 3:
                attn_scores = attn_scores + attn_bias.unsqueeze(0)
            elif attn_bias.dim() == 4:
                attn_scores = attn_scores + attn_bias
            else:
                raise ValueError("attn_bias must be 3D or 4D tensor")

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.einsum("bhts,bshd->bthd", attn_weights, v)
        context = context.contiguous().view(bsz, seq_len, self.d_model)
        attn_out = self.out_proj(context)
        src = self.norm1(src + self.resid_dropout(attn_out))

        ff_out = self.linear2(self.ff_dropout(self.activation_fn(self.linear1(src))))
        src = self.norm2(src + self.resid_dropout(ff_out))
        return src


class RelativeTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        relative_position_bias: Optional[RelativePositionBias],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RelativeTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.rpb = relative_position_bias

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        output = src
        if self.rpb is not None:
            attn_bias = self.rpb(src.size(1), device=src.device, dtype=src.dtype)
        else:
            attn_bias = None

        for layer in self.layers:
            output = layer(
                output,
                src_key_padding_mask=src_key_padding_mask,
                attn_bias=attn_bias,
            )

        output = self.norm(output)
        return output


class ChordProjectionHead(nn.Module):
    def __init__(self, d_model: int, vocab_sizes: Dict[str, int]) -> None:
        super().__init__()
        self.boundary_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        projection_heads: Dict[str, nn.Module] = {}
        for name, size in vocab_sizes.items():
            projection_heads[name] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, size),
            )
        self.projection_heads = nn.ModuleDict(projection_heads)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        boundary_logits = self.boundary_head(x).squeeze(-1)
        outputs: Dict[str, torch.Tensor] = {"boundary": boundary_logits}
        for comp, head in self.projection_heads.items():
            outputs[comp] = head(x)
        return outputs


class KTokenDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.sa_qkv = nn.Linear(d_model, 3 * d_model)
        self.sa_out = nn.Linear(d_model, d_model)
        self.sa_ln = nn.LayerNorm(d_model)
        self.sa_drop = nn.Dropout(dropout)

        self.ca_q = nn.Linear(d_model, d_model)
        self.ca_kv = nn.Linear(d_model, 2 * d_model)
        self.ca_out = nn.Linear(d_model, d_model)
        self.ca_ln = nn.LayerNorm(d_model)
        self.ca_drop = nn.Dropout(dropout)

        hidden = d_model * mlp_ratio
        self.ff_ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.ff_drop = nn.Dropout(dropout)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn = torch.einsum("nlhd,nshd->nhls", q, k) / math.sqrt(q.size(-1))
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.einsum("nhls,nshd->nlhd", attn, v)
        ctx = ctx.contiguous().view(q.size(0), q.size(1), -1)
        return ctx

    def forward(self, x_tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        bsz, length, k_tokens, d_model = x_tokens.shape
        c_len = context.size(2)
        batch_tokens = bsz * length

        x = x_tokens.view(batch_tokens, k_tokens, d_model)
        c = context.view(batch_tokens, c_len, d_model)

        x_norm = self.sa_ln(x)
        qkv = self.sa_qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_tokens, k_tokens, self.nhead, self.head_dim)
        k = k.view(batch_tokens, k_tokens, self.nhead, self.head_dim)
        v = v.view(batch_tokens, k_tokens, self.nhead, self.head_dim)
        sa_ctx = self._attn(q, k, v)
        x = x + self.sa_drop(self.sa_out(sa_ctx))

        x_norm = self.ca_ln(x)
        q = self.ca_q(x_norm).view(batch_tokens, k_tokens, self.nhead, self.head_dim)
        kv = self.ca_kv(c)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(batch_tokens, c_len, self.nhead, self.head_dim)
        v = v.view(batch_tokens, c_len, self.nhead, self.head_dim)
        ca_ctx = self._attn(q, k, v)
        x = x + self.ca_drop(self.ca_out(ca_ctx))

        x_norm = self.ff_ln(x)
        x = x + self.ff_drop(self.ff(x_norm))

        return x.view(bsz, length, k_tokens, d_model)


def build_rpb(config: Dict[str, Any]) -> RelativePositionBias:
    return RelativePositionBias(
        num_heads=config["n_head"],
        max_distance=config["n_beats"] * config["label_resolution"],
    )


def build_encoder(config: Dict[str, Any]) -> RelativeTransformerEncoder:
    rpb = build_rpb(config)
    return RelativeTransformerEncoder(
        num_layers=config["num_encoder_layers"],
        d_model=config["d_model"],
        nhead=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        activation="gelu",
        relative_position_bias=rpb,
    )


def build_patch_embedding(config: Dict[str, Any]) -> PatchEmbedding:
    return PatchEmbedding(
        d_model=config["d_model"],
        frames_per_patch=config["frames_per_patch"],
        expansion=2,
    )


