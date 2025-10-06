import math
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    downsample_key_padding_mask,
    KTokenDecoderLayer,
    ChordProjectionHead,
    build_encoder,
    build_patch_embedding,
)


def _to_patch_input(x_bt88: torch.Tensor) -> torch.Tensor:
    # Accepts (B, T, 88) -> returns (B, 1, 88, T)
    return x_bt88.transpose(1, 2).unsqueeze(1).contiguous()

class BaseEncoder(nn.Module):
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__()
        self.config = model_config
        self.d_model = self.config["d_model"]
        self.embedding = build_patch_embedding(self.config)
        self.input_dropout = nn.Dropout(self.config["dropout"])
        self.encoder = build_encoder(self.config)
        self.boundary_head = nn.Linear(self.d_model, 1)

    def encode(self, encoder_input_bt88: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_pr = _to_patch_input(encoder_input_bt88)
        x = self.embedding(x_pr)
        mask_down = None
        if src_key_padding_mask is not None:
            mask_down = downsample_key_padding_mask(src_key_padding_mask, self.config["frames_per_patch"])
        x = self.input_dropout(x)
        h = self.encoder(x, src_key_padding_mask=mask_down)
        boundary_logits = self.boundary_head(h).squeeze(-1)
        return h, boundary_logits


class BaselineLinearModel(nn.Module):
    """PatchEmbedding + TransformerEncoder + linear heads (baseline)."""

    def __init__(self, model_config: Dict[str, Any], vocab_sizes: Dict[str, int]):
        super().__init__()
        self.config = model_config
        self.encoder = BaseEncoder(model_config)
        self.proj = ChordProjectionHead(model_config["d_model"], vocab_sizes)
        self.use_key = ('key' in vocab_sizes)

    def forward_train(
        self,
        encoder_input: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        src_key_padding_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        vocabs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        device = encoder_input.device
        h, boundary_logits = self.encoder.encode(encoder_input, src_key_padding_mask)
        outputs = self.proj(h)

        bsz, t_len = targets["quality"].shape
        if target_mask is None:
            target_mask = torch.ones(bsz, t_len, dtype=torch.bool, device=device)

        def ce_masked(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, comp_name: str) -> torch.Tensor:
            num_classes = logits.size(-1)
            comp_pad = vocabs.get(f"{comp_name}_pad_idx", vocabs["pad_idx"]) if vocabs is not None else 0
            valid = mask & (target != comp_pad)
            safe_target = torch.where(valid, target.clamp(min=0, max=num_classes - 1), torch.zeros_like(target))
            ce = F.cross_entropy(logits.transpose(1, 2), safe_target, reduction="none")
            denom = valid.float().sum().clamp(min=1.0)
            return (ce * valid.float()).sum() / denom

        loss_q = ce_masked(outputs["quality"], targets["quality"], target_mask, "quality")
        loss_r = ce_masked(outputs["root"], targets["root"], target_mask, "root")
        loss_b = ce_masked(outputs["bass"], targets["bass"], target_mask, "bass")
        if self.use_key and ("key" in outputs) and ("key" in targets):
            loss_k = ce_masked(outputs["key"], targets["key"], target_mask, "key")
        else:
            loss_k = torch.tensor(0.0, device=device)
        bce = F.binary_cross_entropy_with_logits(boundary_logits, targets["boundary"].to(boundary_logits.dtype), reduction="none")
        loss_boundary = (bce * target_mask.float()).sum() / target_mask.float().sum().clamp(min=1.0)
        total_loss = loss_q + loss_r + loss_b + loss_k + loss_boundary * 3.0

        with torch.no_grad():
            logits = {k: v for k, v in outputs.items() if k in ("quality", "root", "bass")}
        return {
            "loss": total_loss,
            "loss_map": {
                "quality": loss_q,
                "root": loss_r,
                "bass": loss_b,
                "key": loss_k,
                "boundary": loss_boundary,
            },
            "logits": logits,
            "boundary_logits": boundary_logits,
        }

    def forward_infer(self, encoder_input: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h, boundary_logits = self.encoder.encode(encoder_input, src_key_padding_mask)
        outputs = self.proj(h)
        comps = ["quality","root","bass"] + (["key"] if self.use_key and ("key" in outputs) else [])
        ids = {k: outputs[k].argmax(dim=-1) for k in comps}
        conf = {f"conf_{k}": outputs[k].softmax(dim=-1).max(dim=-1).values for k in comps}
        ids.update(conf)
        ids["boundary"] = boundary_logits
        return ids


class FiLMContextLinearModel(nn.Module):
    """FiLM injection + local context window + per-component context projector (linear heads)."""

    def __init__(self, model_config: Dict[str, Any], vocab_sizes: Dict[str, int]):
        super().__init__()
        self.config = model_config
        self.encoder = BaseEncoder(model_config)
        d_model = model_config["d_model"]
        self.window_radius = int(model_config["window_radius"])  # Lc = 2R+1 + 1(Z)

        # FiLM parameters (copy of CR-model style)
        self.d_b = max(1, d_model // 4)
        k_b = int(model_config["boundary_kernel"])
        self.boundary_smoother = nn.Conv1d(1, 1, kernel_size=k_b, padding=k_b // 2, bias=True)
        self.boundary_e0 = nn.Parameter(torch.zeros(self.d_b))
        self.boundary_e1 = nn.Parameter(torch.randn(self.d_b) * 0.02)

        self.film_ln_in = nn.LayerNorm(d_model + self.d_b)
        self.film_ln_h = nn.LayerNorm(d_model)
        self.film_mlp = nn.Linear(d_model + self.d_b, 2 * d_model)

        # Per-component context heads
        self.comp_names = ["quality", "root", "bass"]
        if 'key' in vocab_sizes:
            self.comp_names.append('key')
        self.vocab_sizes = vocab_sizes
        self.attn_mlp = nn.ModuleDict()
        self.comp_proj = nn.ModuleDict()
        for comp in self.comp_names:
            self.attn_mlp[comp] = nn.Sequential(
                nn.Linear(model_config["d_model"], model_config["d_model"] // 2),
                nn.GELU(),
                nn.Linear(model_config["d_model"] // 2, 1),  # score per context token
            )
            self.comp_proj[comp] = nn.Sequential(
                nn.Linear(model_config["d_model"], model_config["d_model"]),
                nn.GELU(),
                nn.Linear(model_config["d_model"], self.vocab_sizes[comp]),
            )

    def _smooth_boundary(self, boundary_logits: torch.Tensor) -> torch.Tensor:
        b = boundary_logits.unsqueeze(1)
        smoothed = self.boundary_smoother(b)
        return torch.sigmoid(smoothed.squeeze(1))

    def _apply_film(self, h: torch.Tensor, b_soft: torch.Tensor) -> torch.Tensor:
        bsz, t_len, d_model = h.shape
        e0 = self.boundary_e0.view(1, 1, -1).expand(bsz, t_len, -1)
        e1 = self.boundary_e1.view(1, 1, -1).expand(bsz, t_len, -1)
        eb = b_soft.unsqueeze(-1) * e1 + (1.0 - b_soft).unsqueeze(-1) * e0
        film_in = torch.cat([h, eb], dim=-1)
        film_in = self.film_ln_in(film_in)
        gamma, beta = self.film_mlp(film_in).chunk(2, dim=-1)
        z = self.film_ln_h(h) * (1.0 + gamma) + beta
        return z

    def _build_local_windows(self, h: torch.Tensor, radius: int) -> torch.Tensor:
        x = h.transpose(1, 2)
        padded = F.pad(x, (radius, radius), mode="replicate")
        win = padded.unfold(dimension=2, size=2 * radius + 1, step=1)
        win = win.permute(0, 2, 3, 1).contiguous()
        return win

    def _build_context(self, h: torch.Tensor, z: torch.Tensor, b_soft: torch.Tensor) -> torch.Tensor:
        local = self._build_local_windows(h, self.window_radius)
        z_exp = z.unsqueeze(2)  # (B,T,1,D)
        c = torch.cat([z_exp, local], dim=2)  # (B,T,Lc,D)
        return c

    def _context_logits(self, c: torch.Tensor, comp: str) -> torch.Tensor:
        # c: (B,T,L,D)
        scores = self.attn_mlp[comp](c)  # (B,T,L,1)
        attn = torch.softmax(scores, dim=2)
        pooled = (attn * c).sum(dim=2)  # (B,T,D)
        logits = self.comp_proj[comp](pooled)  # (B,T,V)
        return logits

    def forward_train(
        self,
        encoder_input: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        src_key_padding_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        vocabs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        device = encoder_input.device
        h, boundary_logits = self.encoder.encode(encoder_input, src_key_padding_mask)
        b_soft = self._smooth_boundary(boundary_logits)
        z = self._apply_film(h, b_soft)
        c = self._build_context(h, z, b_soft)

        logits = {comp: self._context_logits(c, comp) for comp in self.comp_names}
        bsz, t_len = targets["quality"].shape
        if target_mask is None:
            target_mask = torch.ones(bsz, t_len, dtype=torch.bool, device=device)

        def ce_masked(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, comp_name: str) -> torch.Tensor:
            num_classes = logits.size(-1)
            comp_pad = vocabs.get(f"{comp_name}_pad_idx", vocabs["pad_idx"]) if vocabs is not None else 0
            valid = mask & (target != comp_pad)
            safe_target = torch.where(valid, target.clamp(min=0, max=num_classes - 1), torch.zeros_like(target))
            ce = F.cross_entropy(logits.transpose(1, 2), safe_target, reduction="none")
            denom = valid.float().sum().clamp(min=1.0)
            return (ce * valid.float()).sum() / denom

        loss_q = ce_masked(logits["quality"], targets["quality"], target_mask, "quality")
        loss_r = ce_masked(logits["root"], targets["root"], target_mask, "root")
        loss_b = ce_masked(logits["bass"], targets["bass"], target_mask, "bass")
        loss_k = torch.tensor(0.0, device=device)
        if 'key' in self.comp_names and ('key' in targets):
            loss_k = ce_masked(logits['key'], targets['key'], target_mask, 'key')
        bce = F.binary_cross_entropy_with_logits(boundary_logits, targets["boundary"].to(boundary_logits.dtype), reduction="none")
        loss_boundary = (bce * target_mask.float()).sum() / target_mask.float().sum().clamp(min=1.0)
        total_loss = loss_q + loss_r + loss_b + loss_k + loss_boundary * 3.0

        return {
            "loss": total_loss,
            "loss_map": {
                "quality": loss_q,
                "root": loss_r,
                "bass": loss_b,
                "key": loss_k,
                "boundary": loss_boundary,
            },
            "logits": logits,
            "boundary_logits": boundary_logits,
        }

    def forward_infer(self, encoder_input: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h, boundary_logits = self.encoder.encode(encoder_input, src_key_padding_mask)
        b_soft = self._smooth_boundary(boundary_logits)
        z = self._apply_film(h, b_soft)
        c = self._build_context(h, z, b_soft)
        logits = {comp: self._context_logits(c, comp) for comp in self.comp_names}
        ids = {k: logits[k].argmax(dim=-1) for k in self.comp_names}
        conf = {f"conf_{k}": logits[k].softmax(dim=-1).max(dim=-1).values for k in self.comp_names}
        ids.update(conf)
        ids["boundary"] = boundary_logits
        return ids


class ChordRecognitionModelWrapper(nn.Module):
    """A thin wrapper around scripts.model.ChordRecognitionModel to normalize inputs."""

    def __init__(self, model_config: Dict[str, Any], vocab_sizes: Dict[str, int]):
        super().__init__()
        from .model import ChordRecognitionModel as CR

        self.inner = CR(model_config, vocab_sizes)

    def forward_train(
        self,
        encoder_input: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        src_key_padding_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        vocabs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        x = _to_patch_input(encoder_input)
        return self.inner.forward_train(x, targets, src_key_padding_mask, target_mask)

    def forward_infer(self, encoder_input: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = _to_patch_input(encoder_input)
        return self.inner.forward_infer(x, src_key_padding_mask)


class FiLMKTokenKeyModel(nn.Module):
    """FiLM + KToken decoder with key-conditioned FiLM injection (key used as auxiliary condition only)."""

    def __init__(self, model_config: Dict[str, Any], vocab_sizes: Dict[str, int], key_vocab_size: int):
        super().__init__()
        self.config = model_config
        self.vocab_sizes = vocab_sizes
        d_model = model_config["d_model"]
        self.encoder = BaseEncoder(model_config)

        # FiLM
        self.d_b = max(1, d_model // 4)
        k_b = int(model_config["boundary_kernel"])
        self.boundary_smoother = nn.Conv1d(1, 1, kernel_size=k_b, padding=k_b // 2, bias=True)
        self.boundary_e0 = nn.Parameter(torch.zeros(self.d_b))
        self.boundary_e1 = nn.Parameter(torch.randn(self.d_b) * 0.02)

        # Key soft embedding
        self.d_k = max(1, d_model // 8)
        self.key_embed = nn.Embedding(key_vocab_size, self.d_k)
        self.key_head = nn.Linear(d_model, key_vocab_size)

        # FiLM projection with concatenated boundary/key embeddings
        self.film_ln_in = nn.LayerNorm(d_model + self.d_b + self.d_k)
        self.film_ln_h = nn.LayerNorm(d_model)
        self.film_mlp = nn.Linear(d_model + self.d_b + self.d_k, 2 * d_model)

        # Decoder (same as CR model)
        self.mask_id_q = self.vocab_sizes["quality"]
        self.mask_id_r = self.vocab_sizes["root"]
        self.mask_id_b = self.vocab_sizes["bass"]
        self.emb_q = nn.Embedding(self.vocab_sizes["quality"] + 1, d_model)
        self.emb_r = nn.Embedding(self.vocab_sizes["root"] + 1, d_model)
        self.emb_b = nn.Embedding(self.vocab_sizes["bass"] + 1, d_model)

        dec_heads = int(model_config["dec_heads"])
        dec_mlp_ratio = int(model_config["dec_mlp_ratio"])
        dec_layers = int(model_config["dec_layers"])
        dec_dropout = float(model_config["dec_dropout"])
        self.window_radius = int(model_config["window_radius"])
        self.decoder_layers = nn.ModuleList(
            [
                KTokenDecoderLayer(
                    d_model=d_model,
                    nhead=dec_heads,
                    mlp_ratio=dec_mlp_ratio,
                    dropout=dec_dropout,
                )
                for _ in range(dec_layers)
            ]
        )
        self.dec_norm = nn.LayerNorm(d_model)
        self.head_q = nn.Linear(d_model, self.vocab_sizes["quality"])
        self.head_r = nn.Linear(d_model, self.vocab_sizes["root"])
        self.head_b = nn.Linear(d_model, self.vocab_sizes["bass"])
        self.use_key = ("key" in self.vocab_sizes)

    def _smooth_boundary(self, boundary_logits: torch.Tensor) -> torch.Tensor:
        b = boundary_logits.unsqueeze(1)
        smoothed = self.boundary_smoother(b)
        return torch.sigmoid(smoothed.squeeze(1))

    def _apply_film(self, h: torch.Tensor, b_soft: torch.Tensor, key_soft: torch.Tensor) -> torch.Tensor:
        bsz, t_len, d_model = h.shape
        # boundary embedding
        e0 = self.boundary_e0.to(h.dtype).view(1, 1, -1).expand(bsz, t_len, -1)
        e1 = self.boundary_e1.to(h.dtype).view(1, 1, -1).expand(bsz, t_len, -1)
        eb = b_soft.unsqueeze(-1) * e1 + (1.0 - b_soft).unsqueeze(-1) * e0
        # key soft embedding via expectation over embeddings
        key_indices = torch.arange(self.key_embed.num_embeddings, device=h.device)
        key_emb_table = self.key_embed(key_indices)  # (V_k, d_k)
        key_emb_table = key_emb_table.to(dtype=key_soft.dtype)
        ek = key_soft @ key_emb_table  # (B,T,d_k)

        film_in = torch.cat([h, eb, ek], dim=-1)
        film_in = self.film_ln_in(film_in)
        gamma, beta = self.film_mlp(film_in).chunk(2, dim=-1)
        z = self.film_ln_h(h) * (1.0 + gamma) + beta
        return z

    def _build_local_windows(self, h: torch.Tensor, radius: int) -> torch.Tensor:
        x = h.transpose(1, 2)
        padded = F.pad(x, (radius, radius), mode="replicate")
        win = padded.unfold(dimension=2, size=2 * radius + 1, step=1)
        win = win.permute(0, 2, 3, 1).contiguous()
        return win

    def _build_context(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        local = self._build_local_windows(h, self.window_radius)
        z_exp = z.unsqueeze(2)
        c = torch.cat([z_exp, local], dim=2)
        return c

    def _embed_tokens(self, ids_q: torch.Tensor, ids_r: torch.Tensor, ids_b: torch.Tensor) -> torch.Tensor:
        xq = self.emb_q(ids_q)
        xr = self.emb_r(ids_r)
        xb = self.emb_b(ids_b)
        x = torch.stack([xq, xr, xb], dim=2)
        return x

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

    def forward_train(
        self,
        encoder_input: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        src_key_padding_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        vocabs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        device = encoder_input.device
        h, boundary_logits = self.encoder.encode(encoder_input, src_key_padding_mask)
        b_soft = self._smooth_boundary(boundary_logits)

        key_logits = self.key_head(h)
        key_soft = key_logits.softmax(dim=-1)

        z = self._apply_film(h, b_soft, key_soft)
        C = self._build_context(h, z)

        tgt_q = targets["quality"]
        tgt_r = targets["root"]
        tgt_b = targets["bass"]
        bsz, t_len = tgt_q.shape
        if target_mask is None:
            target_mask = torch.ones(bsz, t_len, dtype=torch.bool, device=device)

        # Random mask-filling per CR model
        k_rand = torch.randint(1, 4, (bsz, t_len), device=device)
        rand_scores = torch.rand(bsz, t_len, 3, device=device)
        top_vals, top_idx = torch.topk(rand_scores, k=3, dim=-1)
        mask_slots = torch.zeros(bsz, t_len, 3, dtype=torch.bool, device=device)
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

        def ce_masked(logits: torch.Tensor, target: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
            m = slot_mask & target_mask
            num_classes = logits.size(-1)
            safe_target = torch.where(m, target.clamp(min=0, max=num_classes - 1), torch.zeros_like(target))
            ce = F.cross_entropy(logits.transpose(1, 2), safe_target, reduction="none")
            denom = m.float().sum().clamp(min=1.0)
            return (ce * m.float()).sum() / denom

        loss_q = ce_masked(logits_q, tgt_q, mask_slots[:, :, 0])
        loss_r = ce_masked(logits_r, tgt_r, mask_slots[:, :, 1])
        loss_b = ce_masked(logits_b, tgt_b, mask_slots[:, :, 2])

        # Optional key cross-entropy (auxiliary, weight 1.0)
        loss_k = torch.tensor(0.0, device=device)
        if self.use_key and ("key" in targets):
            key_logits = self.key_head(h)
            num_classes_k = key_logits.size(-1)
            safe_k = targets["key"].clamp(min=0, max=num_classes_k - 1)
            loss_k = F.cross_entropy(key_logits.transpose(1, 2), safe_k, reduction="none")
            denom_k = target_mask.float().sum().clamp(min=1.0)
            loss_k = (loss_k * target_mask.float()).sum() / denom_k

        bce = F.binary_cross_entropy_with_logits(boundary_logits, targets["boundary"].to(boundary_logits.dtype), reduction="none")
        loss_boundary = (bce * target_mask.float()).sum() / target_mask.float().sum().clamp(min=1.0)
        total_loss = loss_q + loss_r + loss_b + loss_k + loss_boundary * 3.0

        return {
            "loss": total_loss,
            "loss_map": {
                "quality": loss_q,
                "root": loss_r,
                "bass": loss_b,
                "key": loss_k,
                "boundary": loss_boundary,
            },
            "logits": {
                "quality": logits_q,
                "root": logits_r,
                "bass": logits_b,
            },
            "mask_slots": mask_slots,
            "boundary_logits": boundary_logits,
        }

    def forward_infer(self, encoder_input: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        device = encoder_input.device
        h, boundary_logits = self.encoder.encode(encoder_input, src_key_padding_mask)
        b_soft = self._smooth_boundary(boundary_logits)
        key_logits = self.key_head(h)
        key_soft = key_logits.softmax(dim=-1)
        z = self._apply_film(h, b_soft, key_soft)
        C = self._build_context(h, z)

        bsz, t_len, _ = h.shape
        ids_q = torch.full((bsz, t_len), self.mask_id_q, dtype=torch.long, device=device)
        ids_r = torch.full((bsz, t_len), self.mask_id_r, dtype=torch.long, device=device)
        ids_b = torch.full((bsz, t_len), self.mask_id_b, dtype=torch.long, device=device)
        filled_q = torch.zeros((bsz, t_len), dtype=torch.bool, device=device)
        filled_r = torch.zeros((bsz, t_len), dtype=torch.bool, device=device)
        filled_b = torch.zeros((bsz, t_len), dtype=torch.bool, device=device)

        for step in (3, 2, 1):
            X = self._embed_tokens(ids_q, ids_r, ids_b)
            logits_q, logits_r, logits_b = self._run_decoder(X, C)
            pq = logits_q.softmax(dim=-1)
            pr = logits_r.softmax(dim=-1)
            pb = logits_b.softmax(dim=-1)
            conf_q = pq.max(dim=-1).values
            conf_r = pr.max(dim=-1).values
            conf_b = pb.max(dim=-1).values
            conf_q = conf_q.masked_fill(filled_q, float("-inf"))
            conf_r = conf_r.masked_fill(filled_r, float("-inf"))
            conf_b = conf_b.masked_fill(filled_b, float("-inf"))
            conf = torch.stack([conf_q, conf_r, conf_b], dim=-1)
            take_slot = conf.argmax(dim=-1)

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

        X = self._embed_tokens(ids_q, ids_r, ids_b)
        logits_q, logits_r, logits_b = self._run_decoder(X, C)
        conf_q = logits_q.softmax(dim=-1).max(dim=-1).values
        conf_r = logits_r.softmax(dim=-1).max(dim=-1).values
        conf_b = logits_b.softmax(dim=-1).max(dim=-1).values
        return {
            "quality": ids_q,
            "root": ids_r,
            "bass": ids_b,
            "conf_quality": conf_q,
            "conf_root": conf_r,
            "conf_bass": conf_b,
            "boundary": boundary_logits,
        }


class HTAdapter(nn.Module):
    """Adapter to unify HT with the common training/eval interface."""

    def __init__(self, ht_config: Dict[str, Any], vocab_sizes: Dict[str, int]):
        super().__init__()
        from .HT import HarmonyTransformer

        self.inner = HarmonyTransformer(ht_config, vocab_sizes)
        self.vocab_sizes = vocab_sizes

    def forward_train(
        self,
        encoder_input: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        src_key_padding_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        vocabs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        device = encoder_input.device
        out = self.inner(encoder_input, src_key_padding_mask)
        logits_q = out["quality"]
        logits_r = out["root"]
        logits_b = out["bass"]
        boundary_logits = out["boundary"].squeeze(-1)

        bsz, t_len, _ = logits_q.shape
        if target_mask is None:
            target_mask = torch.ones(bsz, t_len, dtype=torch.bool, device=device)

        def ce_masked(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, comp_name: str) -> torch.Tensor:
            num_classes = logits.size(-1)
            comp_pad = vocabs.get(f"{comp_name}_pad_idx", vocabs["pad_idx"]) if vocabs is not None else 0
            valid = mask & (target != comp_pad)
            safe_target = torch.where(valid, target.clamp(min=0, max=num_classes - 1), torch.zeros_like(target))
            ce = F.cross_entropy(logits.transpose(1, 2), safe_target, reduction="none")
            denom = valid.float().sum().clamp(min=1.0)
            return (ce * valid.float()).sum() / denom

        loss_q = ce_masked(logits_q, targets["quality"], target_mask, "quality")
        loss_r = ce_masked(logits_r, targets["root"], target_mask, "root")
        loss_b = ce_masked(logits_b, targets["bass"], target_mask, "bass")
        bce = F.binary_cross_entropy_with_logits(boundary_logits, targets["boundary"].to(boundary_logits.dtype), reduction="none")
        loss_boundary = (bce * target_mask.float()).sum() / target_mask.float().sum().clamp(min=1.0)
        total_loss = loss_q + loss_r + loss_b + 3.0 * loss_boundary

        return {
            "loss": total_loss,
            "loss_map": {
                "quality": loss_q,
                "root": loss_r,
                "bass": loss_b,
                "boundary": loss_boundary,
            },
            "logits": {
                "quality": logits_q,
                "root": logits_r,
                "bass": logits_b,
            },
            "boundary_logits": boundary_logits,
        }

    def forward_infer(self, encoder_input: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = self.inner(encoder_input, src_key_padding_mask)
        ids = {
            "quality": out["quality"].argmax(dim=-1),
            "root": out["root"].argmax(dim=-1),
            "bass": out["bass"].argmax(dim=-1),
            "boundary": out["boundary"].squeeze(-1),
        }
        return ids


def build_model(experiment: str, model_config: Dict[str, Any], vocabs: Dict[str, Any], use_key: bool = False) -> nn.Module:
    # Build vocab sizes for components (include key optionally)
    chord_components = ["root", "quality", "bass"] + (["key"] if use_key and ("key" in vocabs) else [])
    vocab_sizes = {comp: len(vocabs[comp]) for comp in chord_components}

    exp = experiment.lower()
    if exp == "baseline":
        return BaselineLinearModel(model_config, vocab_sizes)
    if exp == "film_ctx":
        return FiLMContextLinearModel(model_config, vocab_sizes)
    if exp == "film_kdec":
        return ChordRecognitionModelWrapper(model_config, vocab_sizes)
    if exp == "film_kdec_key":
        key_vocab_size = len(vocabs["key"]) if ("key" in vocabs) else 24
        return FiLMKTokenKeyModel(model_config, vocab_sizes, key_vocab_size)
    if exp == "ht":
        # The HT config expects specific keys; reuse model_config where possible
        ht_cfg = {
            "input_size": model_config["input_size"],
            "d_model": model_config["d_model"],
            "n_layers": model_config["num_encoder_layers"],
            "n_heads": model_config["n_head"],
            "dropout": model_config["dropout"],
            "train_boundary": True,
            "slope": 1.0,
            "n_beats": model_config["n_beats"],
            "beat_resolution": model_config["beat_resolution"],
        }
        return HTAdapter(ht_cfg, vocab_sizes)

    raise ValueError(f"Unknown experiment '{experiment}'")


