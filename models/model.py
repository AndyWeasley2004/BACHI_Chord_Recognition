import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x.transpose(0, 1)


class RelativeTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, activation: str = 'gelu'):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
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

        if activation == 'gelu':
            self.activation_fn = F.gelu
        elif activation == 'relu':
            self.activation_fn = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # src: (B, T_new, C)
        bsz, seq_len_new, _ = src.size()

        qkv = self.qkv_proj(src)  # (B, T_new, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len_new, self.nhead, self.head_dim)
        k_all = k.view(bsz, seq_len_new, self.nhead, self.head_dim)
        v_all = v.view(bsz, seq_len_new, self.nhead, self.head_dim)

        # Attention: queries are only for current tokens; keys include past+current
        attn_scores = torch.einsum('bthd,bshd->bhts', q, k_all) / math.sqrt(self.head_dim)  # (B, H, T_new, T_total)

        # Additive or boolean mask over attention logits
        if src_mask is not None:
            if src_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(src_mask.unsqueeze(0), float('-inf'))
            else:
                attn_scores = attn_scores + src_mask.unsqueeze(0)

        # Key padding mask
        if src_key_padding_mask is not None:
            key_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
            attn_scores = attn_scores.masked_fill(key_mask, float('-inf'))

        if attn_bias is not None:
            # Support 3D (H, T, T) or 4D (B, H, T, T)
            if attn_bias.dim() == 3:
                attn_scores = attn_scores + attn_bias.unsqueeze(0)
            elif attn_bias.dim() == 4:
                attn_scores = attn_scores + attn_bias
            else:
                raise ValueError("attn_bias must be 3D or 4D tensor if provided")

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.einsum('bhts,bshd->bthd', attn_weights, v_all)  # (B, T_new, H, D)
        context = context.contiguous().view(bsz, seq_len_new, self.d_model)  # (B, T_new, C)
        attn_out = self.out_proj(context)

        src = src + self.resid_dropout(attn_out)
        src = self.norm1(src)

        ff = self.linear2(self.ff_dropout(self.activation_fn(self.linear1(src))))
        src = src + self.resid_dropout(ff)
        src = self.norm2(src)

        return src

def downsample_key_padding_mask(mask: torch.Tensor, frames_per_patch: int) -> torch.Tensor:
    # mask: (B, T) where True denotes padding.
    bsz, total_len = mask.shape
    if total_len < frames_per_patch:
        # No valid output tokens from temporal pooling
        return mask.new_ones((bsz, 0), dtype=mask.dtype)
    out_len = total_len // frames_per_patch
    trimmed = mask[:, :out_len * frames_per_patch]
    grouped = trimmed.view(bsz, out_len, frames_per_patch)
    return grouped.all(dim=-1)


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int):
        super().__init__()
        if max_distance < 1:
            raise ValueError("max_distance must be >= 1")
        self.max_distance = max_distance
        self.num_heads = num_heads
        # Table over relative distances in [-max_distance+1, max_distance-1]
        self.bias = nn.Parameter(torch.zeros(2 * max_distance - 1, num_heads))

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Compute clipped relative position indices
        pos = torch.arange(seq_len, device=device)
        rel = pos[:, None] - pos[None, :]  # (T, T)
        rel = rel.clamp(-self.max_distance + 1, self.max_distance - 1)
        rel = rel + self.max_distance - 1  # shift to [0, 2*max_distance-2]
        bias = self.bias[rel]  # (T, T, H)
        return bias.permute(2, 0, 1).to(dtype=dtype)  # (H, T, T)


class RelativeTransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int, dim_feedforward: int, 
                 dropout: float, activation: str = 'gelu', relative_position_bias = None):
        super().__init__()
        self.layers = nn.ModuleList([
            RelativeTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.rpb = relative_position_bias
        self.nhead = nhead

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = src

        if self.rpb is not None:
            attn_bias = self.rpb(src.size(1), device=src.device, dtype=src.dtype)
        else:
            attn_bias = None

        for mod in self.layers:
            output = mod(
                output,
                src_key_padding_mask=src_key_padding_mask,
                attn_bias=attn_bias,
            )

        output = self.norm(output)
        return output


class ChordDecomposeProjection(nn.Module):
    def __init__(self, d_model: int, vocab_sizes: Dict[str, int]):
        super().__init__()
        self.d_model = d_model
        self.vocab_sizes = vocab_sizes
        self.boundary_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.projection_heads = nn.ModuleDict()
        for comp, size in self.vocab_sizes.items():
            self.projection_heads[comp] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        boundary_logits = self.boundary_head(x)

        output = {'boundary': boundary_logits.squeeze(-1)}
        for comp, head in self.projection_heads.items():
            output[comp] = head(x)

        return output


class ChordRecognitionModel(nn.Module):
    def __init__(self, model_config: Dict[str, Any], vocab_sizes: Dict[str, int]):
        super().__init__()
        self.config = model_config
        self.vocab_sizes = vocab_sizes
        self.d_model = self.config['d_model']
        # Encoder: shared patch embedding and relative transformer (unchanged)
        self.embedding = PatchEmbedding(
            d_model=self.d_model,
            frames_per_patch=self.config['frames_per_patch'],
            expansion=2,
        )

        self.input_dropout = nn.Dropout(self.config['dropout'])

        rpb = RelativePositionBias(
            num_heads=self.config['n_head'],
            max_distance=self.config['n_beats'] * self.config['label_resolution']
        )
        self.relative_transformer_encoder = RelativeTransformerEncoder(
            num_layers=self.config['num_encoder_layers'],
            d_model=self.d_model,
            nhead=self.config['n_head'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout'],
            activation='gelu',
            relative_position_bias=rpb,
        )

        # Boundary head, smoother, and FiLM gating
        d_b = max(1, self.d_model // 4)
        k_b = int(self.config.get('boundary_kernel', 5))
        self.boundary_head = nn.Linear(self.d_model, 1)
        self.boundary_smoother = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_b,
            padding=k_b // 2,
            groups=1,
            bias=True,
        )
        self.boundary_e0 = nn.Parameter(torch.zeros(d_b))
        self.boundary_e1 = nn.Parameter(torch.randn(d_b) * 0.02)
        # Optional key context (3-part setting). Infer size from vocab_sizes if provided
        self.Vq = int(vocab_sizes.get('quality', 0))
        self.Vr = int(vocab_sizes.get('root', 0))
        self.Vb = int(vocab_sizes.get('bass', 0))

        # FiLM layers take boundary embedding
        self.film_ln_in = nn.LayerNorm(self.d_model + d_b)
        self.film_ln_h = nn.LayerNorm(self.d_model)
        self.film_mlp = nn.Linear(self.d_model + d_b, 2 * self.d_model)

        # Triple-token decoder: embeddings and heads
        self.mask_id_q = int(self.config.get('mask_id_q', self.Vq))
        self.mask_id_r = int(self.config.get('mask_id_r', self.Vr))
        self.mask_id_b = int(self.config.get('mask_id_b', self.Vb))
        self.emb_q = nn.Embedding(self.Vq + 1, self.d_model)
        self.emb_r = nn.Embedding(self.Vr + 1, self.d_model)
        self.emb_b = nn.Embedding(self.Vb + 1, self.d_model)

        dec_heads = int(self.config.get('dec_heads', 4))
        dec_mlp_ratio = int(self.config.get('dec_mlp_ratio', 4))
        dec_layers = int(self.config.get('dec_layers', 1))
        dec_dropout = float(self.config.get('dec_dropout', 0.1))
        self.window_radius = int(self.config.get('window_radius', 2))
        self.decoder_layers = nn.ModuleList([
            KTokenDecoderLayer(
                d_model=self.d_model,
                nhead=dec_heads,
                mlp_ratio=dec_mlp_ratio,
                dropout=dec_dropout,
            ) for _ in range(dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(self.d_model)
        self.head_q = nn.Linear(self.d_model, self.Vq)
        self.head_r = nn.Linear(self.d_model, self.Vr)
        self.head_b = nn.Linear(self.d_model, self.Vb)
        
        # Legacy decompose projection for compatibility when training in decompose mode
        self.chord_decompose_projection = ChordDecomposeProjection(self.d_model, self.vocab_sizes)

    def forward(self, encoder_input: torch.Tensor, 
                src_key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        H, _ = self._encode(encoder_input, src_key_padding_mask)
        out = self.chord_decompose_projection(H)
        return out

    # ===== Utilities =====
    def _encode(self, encoder_input: torch.Tensor, 
                src_key_padding_mask: Optional[torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        x = self.embedding(encoder_input)
        mask_down = None
        if src_key_padding_mask is not None:
            mask_down = downsample_key_padding_mask(src_key_padding_mask, self.config['frames_per_patch'])
        x = self.input_dropout(x)
        H = self.relative_transformer_encoder(x, src_key_padding_mask=mask_down)
        boundary_logits = self.boundary_head(H).squeeze(-1)
        return H, boundary_logits

    def _smooth_boundary(self, boundary_logits: torch.Tensor) -> torch.Tensor:
        b = boundary_logits.unsqueeze(1)
        smoothed = self.boundary_smoother(b)
        return torch.sigmoid(smoothed.squeeze(1))

    def _apply_film(self, H: torch.Tensor, b_soft: torch.Tensor) -> torch.Tensor:
        B, T, D = H.shape
        e0 = self.boundary_e0.view(1, 1, -1).expand(B, T, -1)
        e1 = self.boundary_e1.view(1, 1, -1).expand(B, T, -1)
        # soft embedding for boundary
        eb = b_soft.unsqueeze(-1) * e1 + (1.0 - b_soft).unsqueeze(-1) * e0
        film_in = torch.cat([H, eb], dim=-1)

        # layer norm and linear projection
        film_in = self.film_ln_in(film_in)
        gamma, beta = self.film_mlp(film_in).chunk(2, dim=-1)
        Z = self.film_ln_h(H) * (1.0 + gamma) + beta
        return Z

    def _build_local_windows(self, H: torch.Tensor, radius: int) -> torch.Tensor:
        x = H.transpose(1, 2)
        padded = F.pad(x, (radius, radius), mode='replicate')
        win = padded.unfold(dimension=2, size=2 * radius + 1, step=1)
        win = win.permute(0, 2, 3, 1).contiguous()
        return win

    def _build_context(self, H: torch.Tensor, Z: torch.Tensor, b_soft: torch.Tensor) -> torch.Tensor:
        local = self._build_local_windows(H, self.window_radius)
        z = Z.unsqueeze(2)
        parts = [z, local]
        C = torch.cat(parts, dim=2)
        return C

    def _embed_tokens(self, ids_q: torch.Tensor, ids_r: torch.Tensor, ids_b: torch.Tensor) -> torch.Tensor:
        xq = self.emb_q(ids_q)
        xr = self.emb_r(ids_r)
        xb = self.emb_b(ids_b)
        X = torch.stack([xq, xr, xb], dim=2)
        return X

    def _run_decoder(self, X: torch.Tensor, C: torch.Tensor):
        x = X
        for layer in self.decoder_layers:
            x = layer(x, C)
        x = self.dec_norm(x)
        xq = x[:, :, 0, :]
        xr = x[:, :, 1, :]
        xb = x[:, :, 2, :]
        logits_q = self.head_q(xq)
        logits_r = self.head_r(xr)
        logits_b = self.head_b(xb)
        return logits_q, logits_r, logits_b

    # ===== Training forward =====
    def forward_train(self, encoder_input: torch.Tensor,
                      targets: Dict[str, torch.Tensor],
                      src_key_padding_mask: Optional[torch.Tensor] = None,
                      target_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        device = encoder_input.device
        H, boundary_logits = self._encode(encoder_input, src_key_padding_mask)
        # No key prediction/context in training
        b_soft = self._smooth_boundary(boundary_logits)
        Z = self._apply_film(H, b_soft)
        C = self._build_context(H, Z, b_soft)

        tgt_q = targets['quality']
        tgt_r = targets['root']
        tgt_b = targets['bass']
        B, T = tgt_q.shape
        if target_mask is None:
            target_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        # mask n slots randomly per (B,T) across 3 slots [q,r,b]
        k_rand = torch.randint(1, 4, (B, T), device=device)
        rand_scores = torch.rand(B, T, 3, device=device)
        top_vals, top_idx = torch.topk(rand_scores, k=3, dim=-1)
        mask_slots = torch.zeros(B, T, 3, dtype=torch.bool, device=device)
        # enable first k indices per position
        for kk in range(1, 4):
            sel = (k_rand == kk)
            if sel.any():
                idx_sel = top_idx[sel][:, :kk]
                row = mask_slots[sel]
                if idx_sel.numel() > 0:
                    row.scatter_(dim=1, index=idx_sel, value=True)
                mask_slots[sel] = row

        ids_q = tgt_q.clone()
        ids_r = tgt_r.clone()
        ids_b = tgt_b.clone()
        ids_q[mask_slots[:, :, 0]] = self.mask_id_q
        ids_r[mask_slots[:, :, 1]] = self.mask_id_r
        ids_b[mask_slots[:, :, 2]] = self.mask_id_b

        X = self._embed_tokens(ids_q, ids_r, ids_b)
        logits_q, logits_r, logits_b = self._run_decoder(X, C)

        def ce_masked(logits: Optional[torch.Tensor], target: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
            # Build supervision mask and safe targets to avoid CUDA asserts from out-of-range labels
            m = slot_mask & target_mask  # (B,T) supervised locations
            num_classes = logits.size(-1)
            safe_target = torch.where(
                m,
                target.clamp(min=0, max=num_classes - 1),
                torch.zeros_like(target)
            )
            ce = F.cross_entropy(logits.transpose(1, 2), safe_target, reduction='none')
            denom = m.float().sum().clamp(min=1.0)
            return (ce * m.float()).sum() / denom

        loss_q = ce_masked(logits_q, tgt_q, mask_slots[:, :, 0])
        loss_r = ce_masked(logits_r, tgt_r, mask_slots[:, :, 1])
        loss_b = ce_masked(logits_b, tgt_b, mask_slots[:, :, 2])

        bce = F.binary_cross_entropy_with_logits(boundary_logits, 
                                                 targets['boundary'].to(boundary_logits.dtype), 
                                                 pos_weight=torch.tensor(2.0, device=device),
                                                 reduction='none')
        loss_boundary = (bce * target_mask.float()).sum() / target_mask.float().sum().clamp(min=1.0)
        total_loss = loss_q + loss_r + loss_b + loss_boundary * 3

        with torch.no_grad():
            stats = {}
            for name, logits, target, m in [
                ('quality', logits_q, tgt_q, mask_slots[:, :, 0]),
                ('root', logits_r, tgt_r, mask_slots[:, :, 1]),
                ('bass', logits_b, tgt_b, mask_slots[:, :, 2]),
            ]:
                if logits is None:
                    stats[f'acc_{name}'] = 0.0
                    stats[f'conf_{name}'] = 0.0
                    stats[f'ece_{name}'] = 0.0
                else:
                    pred = logits.argmax(dim=-1)
                    sel = (m & target_mask)
                    denom = sel.float().sum().clamp(min=1.0)
                    acc = (pred[sel] == target[sel]).float().sum() / denom
                    prob = logits.float().softmax(dim=-1)
                    conf = prob.max(dim=-1).values
                    mean_conf = conf[sel].sum() / denom
                    # simple ECE
                    ece = torch.tensor(0.0, device=device)
                    bins = torch.linspace(0, 1, steps=11, device=device)
                    conf_flat = conf[sel]
                    pred_flat = pred[sel]
                    tgt_flat = target[sel]
                    for i in range(10):
                        lo, hi = bins[i], bins[i+1]
                        mask_bin = (conf_flat >= lo) & (conf_flat < hi if i < 9 else conf_flat <= hi)
                        if mask_bin.sum() > 0:
                            acc_bin = (pred_flat[mask_bin] == tgt_flat[mask_bin]).float().mean()
                            conf_bin = conf_flat[mask_bin].mean()
                            ece = ece + (mask_bin.float().mean() * (acc_bin - conf_bin).abs())
                    stats[f'acc_{name}'] = acc.item()
                    stats[f'conf_{name}'] = mean_conf.item()
                    stats[f'ece_{name}'] = ece.item()

        return {
            'loss': total_loss,
            'loss_map': {
                'quality': loss_q,
                'root': loss_r,
                'bass': loss_b,
                'boundary': loss_boundary,
            },
            'logits': {
                'quality': logits_q,
                'root': logits_r,
                'bass': logits_b,
            },
            'mask_slots': mask_slots,  # (B,T,3) bool in order [q,r,b]
            'boundary_logits': boundary_logits,
            'stats': stats,
        }

    # ===== Inference forward =====
    def forward_infer(self, encoder_input: torch.Tensor,
                      src_key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        device = encoder_input.device
        H, boundary_logits = self._encode(encoder_input, src_key_padding_mask)
        # No key prediction/context in inference
        b_soft = self._smooth_boundary(boundary_logits)
        Z = self._apply_film(H, b_soft)
        C = self._build_context(H, Z, b_soft)

        B, T, _ = H.shape
        ids_q = torch.full((B, T), self.mask_id_q, dtype=torch.long, device=device)
        ids_r = torch.full((B, T), self.mask_id_r, dtype=torch.long, device=device)
        ids_b = torch.full((B, T), self.mask_id_b, dtype=torch.long, device=device)
        filled_q = torch.zeros((B, T), dtype=torch.bool, device=device)
        filled_r = torch.zeros((B, T), dtype=torch.bool, device=device)
        filled_b = torch.zeros((B, T), dtype=torch.bool, device=device)

        # Track decode order per time step: 0=quality, 1=root, 2=bass
        decode_order = torch.full((B, T, 3), -1, dtype=torch.long, device=device)
        order_pos = 0

        for step in (3, 2, 1):
            X = self._embed_tokens(ids_q, ids_r, ids_b)
            logits_q, logits_r, logits_b = self._run_decoder(X, C)
            pq = logits_q.softmax(dim=-1)
            pr = logits_r.softmax(dim=-1)
            pb = logits_b.softmax(dim=-1)
            conf_q = pq.max(dim=-1).values
            conf_r = pr.max(dim=-1).values
            conf_b = pb.max(dim=-1).values
            conf_q = conf_q.masked_fill(filled_q, float('-inf'))
            conf_r = conf_r.masked_fill(filled_r, float('-inf'))
            conf_b = conf_b.masked_fill(filled_b, float('-inf'))
            conf = torch.stack([conf_q, conf_r, conf_b], dim=-1)
            take_slot = conf.argmax(dim=-1)

            # record order at this step
            decode_order[:, :, order_pos] = take_slot
            order_pos += 1
            
            pred_q = logits_q.argmax(dim=-1)
            commit_q = (take_slot == 0) | ((step == 1) & (~filled_q))
            ids_q[commit_q] = pred_q[commit_q]
            filled_q = filled_q | commit_q
            
            pred_r = logits_r.argmax(dim=-1)
            commit_r = (take_slot == 1) | ((step == 1) & (~filled_r))
            ids_r[commit_r] = pred_r[commit_r]
            filled_r = filled_r | commit_r
            
            pred_b = logits_b.argmax(dim=-1)
            commit_b = (take_slot == 2) | ((step == 1) & (~filled_b))
            ids_b[commit_b] = pred_b[commit_b]
            filled_b = filled_b | commit_b

        # final confidences
        X = self._embed_tokens(ids_q, ids_r, ids_b)
        logits_q, logits_r, logits_b = self._run_decoder(X, C)
        conf_q = logits_q.softmax(dim=-1).max(dim=-1).values
        conf_r = logits_r.softmax(dim=-1).max(dim=-1).values
        conf_b = logits_b.softmax(dim=-1).max(dim=-1).values
        
        return {
            'quality': ids_q,
            'root': ids_r,
            'bass': ids_b,
            'conf_quality': conf_q,
            'conf_root': conf_r,
            'conf_bass': conf_b,
            'boundary': boundary_logits,
            'decode_order': decode_order,
        }


class KTokenDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        if self.head_dim * nhead != d_model:
            raise ValueError("d_model must be divisible by nhead")

        # self-attention over K tokens
        self.sa_qkv = nn.Linear(d_model, 3 * d_model)
        self.sa_out = nn.Linear(d_model, d_model)
        self.sa_ln = nn.LayerNorm(d_model)
        self.sa_drop = nn.Dropout(dropout)

        # cross-attention to context
        self.ca_q = nn.Linear(d_model, d_model)
        self.ca_kv = nn.Linear(d_model, 2 * d_model)
        self.ca_out = nn.Linear(d_model, d_model)
        self.ca_ln = nn.LayerNorm(d_model)
        self.ca_drop = nn.Dropout(dropout)

        # ffn
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
        # q,k,v: (N, L, H, D)
        attn = torch.einsum('nlhd,nshd->nhls', q, k) / math.sqrt(q.size(-1))  # (N,H,L,S)
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.einsum('nhls,nshd->nlhd', attn, v)  # (N,L,H,D)
        ctx = ctx.contiguous().view(q.size(0), q.size(1), -1)  # (N,L,C)
        return ctx

    def forward(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # X: (B,T,K,D), C: (B,T,Lc,D)
        B, T, K, D = X.shape
        Lc = C.size(2)
        N = B * T
        # reshape
        x = X.view(N, K, D)
        c = C.view(N, Lc, D)

        # self-attn (over K)
        x_norm = self.sa_ln(x)
        qkv = self.sa_qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(N, K, self.nhead, self.head_dim)
        k = k.view(N, K, self.nhead, self.head_dim)
        v = v.view(N, K, self.nhead, self.head_dim)
        sa_ctx = self._attn(q, k, v)  # (N,K,C)
        x = x + self.sa_drop(self.sa_out(sa_ctx))

        # cross-attn (queries = tokens, keys/values = context)
        x_norm = self.ca_ln(x)
        q = self.ca_q(x_norm).view(N, K, self.nhead, self.head_dim)
        kv = self.ca_kv(c)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(N, Lc, self.nhead, self.head_dim)
        v = v.view(N, Lc, self.nhead, self.head_dim)
        ca_ctx = self._attn(q, k, v)  # (N,K,C)
        x = x + self.ca_drop(self.ca_out(ca_ctx))

        # ffn
        x_norm = self.ff_ln(x)
        x = x + self.ff_drop(self.ff(x_norm))

        return x.view(B, T, K, D)
