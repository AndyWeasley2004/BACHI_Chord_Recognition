import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

def get_relative_position_encoding(n_steps, d_model, max_dist=10):
    """
    Generates relative positional encodings, similar to Transformer-XL.
    """
    vocab_size = 2 * max_dist + 1
    position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe = torch.zeros(vocab_size, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    range_vec = torch.arange(n_steps)
    distance_mat = range_vec[None, :] - range_vec[:, None]
    distance_mat_clipped = torch.clamp(distance_mat, -max_dist, max_dist)
    final_mat = distance_mat_clipped + max_dist

    embeddings = F.embedding(final_mat.long(), pe)
    return embeddings

class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with relative positional encoding, inspired by Transformer-XL and HTv2.
    """
    def __init__(self, d_model, n_heads, dropout=0.1, max_dist=10):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_dist = max_dist

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.u_bias = nn.Parameter(torch.randn(self.n_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.randn(self.n_heads, self.d_head))
        self.w_r = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, pos_emb, key_padding_mask=None, attn_mask=None, values_provided=False):
        batch_size, seq_len_q, _ = q.size()
        seq_len_k = k.size(1)

        residual = q

        q = self.w_q(q).view(batch_size, seq_len_q, self.n_heads, self.d_head)
        k = self.w_k(k).view(batch_size, seq_len_k, self.n_heads, self.d_head)
        v_transformed = self.w_v(v).view(batch_size, seq_len_k, self.n_heads, self.d_head)
        
        q_with_u = (q + self.u_bias).transpose(1, 2)  # (B, h, T_q, d_h)
        q_with_v = (q + self.v_bias).transpose(1, 2)  # (B, h, T_q, d_h)
        k = k.transpose(1, 2) # (B, h, T_k, d_h)
        v_transformed = v_transformed.transpose(1, 2) # (B, h, T_k, d_h)

        pos_emb = self.w_r(pos_emb).view(seq_len_q, seq_len_k, self.n_heads, self.d_head)
        pos_emb = pos_emb.permute(2, 0, 3, 1) # (h, T_q, d_h, T_k)
        
        # Content-based addressing
        ac = torch.matmul(q_with_u, k.transpose(-2, -1)) # (B, h, T_q, T_k)
        
        # Position-based addressing
        q_for_bd = q_with_v.permute(1, 2, 0, 3) # (h, T_q, B, d_h)
        bd_t = torch.matmul(q_for_bd, pos_emb) # (h, T_q, B, T_k)
        bd = bd_t.permute(2, 0, 1, 3) # (B, h, T_q, T_k)

        attn_score = (ac + bd) / math.sqrt(self.d_head)

        # Apply key and (optional) attention masks
        if key_padding_mask is not None:
            attn_score = attn_score.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        if attn_mask is not None:
            attn_score = attn_score.masked_fill(attn_mask.unsqueeze(0), float('-inf'))

        # ------------------------------------------------------------------
        # Guard against rows where *all* keys were masked (e.g., an attention
        # block that consists purely of padding tokens). A softmax over a
        # vector of \(-\infty\) would otherwise produce NaNs. We detect such
        # rows and set their scores to zero, ensuring a well-defined softmax
        # (which will yield a uniform distribution) before later resetting
        # their weights to zero.
        # ------------------------------------------------------------------
        all_masked = torch.isinf(attn_score) & (attn_score < 0)  # True where -inf
        all_masked = all_masked.all(dim=-1, keepdim=True)        # (B, h, T_q, 1)

        # For rows with all_masked == True, replace -inf with 0 so that the
        # subsequent softmax does not generate NaNs.
        attn_score = attn_score.masked_fill(all_masked, 0.0)
        
        attn_weights = F.softmax(attn_score, dim=-1)  # (B, h, T_q, T_k)

        # After the softmax, zero-out rows that correspond to fully-padded
        # queries so they don’t influence the output.
        attn_weights = attn_weights.masked_fill(all_masked, 0.0)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v_transformed) # (B, h, T_q, d_h)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        attn_output = self.w_o(attn_output)
        attn_output = self.dropout(attn_output)

        if values_provided:
            output = v + attn_output
        else:
            output = residual + attn_output
        
        return self.layer_norm(output), attn_weights

class IntraBlockMHA(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_dist=3):
        super().__init__()
        self.mha = RelativeMultiHeadAttention(d_model, n_heads, dropout, max_dist=max_dist)
    
    def forward(self, x, n_blocks, key_padding_mask=None):
        batch_size, seq_len, _ = x.shape
        block_len = seq_len // n_blocks
        
        x = x.reshape(batch_size * n_blocks, block_len, -1)
        
        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.reshape(batch_size * n_blocks, block_len)

        pos_emb = get_relative_position_encoding(block_len, x.size(-1), self.mha.max_dist).to(x.device)

        x, _ = self.mha(x, x, x, pos_emb, key_padding_mask=mask)
        return x.reshape(batch_size, seq_len, -1)

class ConvFFN(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # piano roll resolution: 4 -> 12 (3x)
        # kernel size: 3 -> 9 (3x)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=9, padding=4)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = x + residual
        return self.norm(x)

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_dist, use_conv_ffn=True):
        super().__init__()
        self.self_attn = RelativeMultiHeadAttention(d_model, n_heads, dropout, max_dist)
        self.cross_attn = RelativeMultiHeadAttention(d_model, n_heads, dropout, max_dist)
        self.pos_attn = RelativeMultiHeadAttention(d_model, n_heads, dropout, max_dist)
        if use_conv_ffn:
            self.ffn = ConvFFN(d_model, dropout)
        else: # Fallback to original FFN if needed
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
            self.norm = nn.LayerNorm(d_model)

        self.use_conv_ffn = use_conv_ffn

    def forward(self, dec_input, enc_output, pos_emb, dec_pos_emb,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention
        dec_output, _ = self.self_attn(dec_input, dec_input, dec_input, pos_emb, key_padding_mask=tgt_key_padding_mask)

        # Positional Attention
        dec_output = self.pos_attn(dec_pos_emb.unsqueeze(0).repeat(dec_input.size(0), 1, 1), 
                                   dec_pos_emb.unsqueeze(0).repeat(dec_input.size(0), 1, 1), 
                                   dec_output, pos_emb=pos_emb, 
                                   key_padding_mask=tgt_key_padding_mask, values_provided=True)[0]

        # Cross-attention
        dec_output, _ = self.cross_attn(dec_output, enc_output, enc_output, pos_emb, key_padding_mask=memory_key_padding_mask)

        # FFN
        if self.use_conv_ffn:
            dec_output = self.ffn(dec_output)
        else:
            dec_output = self.norm(dec_output + self.ffn(dec_output))
            
        return dec_output

class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens in the sequence."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class BinaryRound(torch.autograd.Function):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    """
    @staticmethod
    def forward(ctx, input):
        return torch.round(input).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class HarmonyTransformer(nn.Module):
    def __init__(self, config: Dict, vocab_sizes: Dict[str, int]):
        super().__init__()
        self.input_size = config['input_size']
        self.d_model = config['d_model']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.dropout_rate = config['dropout']
        self.train_boundary = config['train_boundary']
        self.slope = config.get('slope', 1.0)
        self.max_len = config['n_beats'] * config['beat_resolution']
        self.config = config
        
        # Embeddings
        self.enc_input_embed = nn.Linear(self.input_size, self.d_model)
        self.dec_input_embed = nn.Linear(self.input_size, self.d_model)
        
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout_rate, self.max_len)
        self.register_buffer('pos_emb', get_relative_position_encoding(self.max_len, self.d_model, self.max_len - 1))
        
        self.enc_intra_block_mha = IntraBlockMHA(self.d_model, self.n_heads, self.dropout_rate, max_dist=3)
        self.dec_intra_block_mha = IntraBlockMHA(self.d_model, self.n_heads, self.dropout_rate, max_dist=3)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            RelativeMultiHeadAttention(self.d_model, self.n_heads, self.dropout_rate, self.max_len-1)
            for _ in range(self.n_layers)
        ])
        self.encoder_ffns = nn.ModuleList([ConvFFN(self.d_model, self.dropout_rate) for _ in range(self.n_layers)])
        self.enc_weights = nn.Parameter(torch.zeros(self.n_layers + 1))

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerLayer(self.d_model, self.n_heads, self.dropout_rate, self.max_len-1)
            for _ in range(self.n_layers)
        ])
        self.dec_weights = nn.Parameter(torch.zeros(self.n_layers + 1))

        # Chord Change Prediction
        self.chord_change_predictor = nn.Linear(self.d_model, 1)

        # Output layers
        self.root_predictor = nn.Linear(self.d_model, vocab_sizes['root'])
        self.quality_predictor = nn.Linear(self.d_model, vocab_sizes['quality'])
        self.bass_predictor = nn.Linear(self.d_model, vocab_sizes['bass'])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def chord_block_compression(self, hidden_states, chord_changes):
        """Compress hidden states according to chord changes."""
        block_ids = torch.cumsum(chord_changes, dim=1) - chord_changes[:, 0].unsqueeze(1)
        # Ensure integer dtype for one-hot encoding
        block_ids = block_ids.long()

        max_blocks = (torch.max(block_ids).item() + 1) if block_ids.numel() > 0 else 1
        one_hot_ids = F.one_hot(block_ids, num_classes=max_blocks).float()  # (B, S, M)
        
        summed_states = torch.bmm(one_hot_ids.transpose(1, 2), hidden_states)  # (B, M, H)
        block_counts = one_hot_ids.sum(dim=1).unsqueeze(-1).clamp(min=1)
        
        mean_states = summed_states / block_counts
        # block_ids already of integer dtype
        return mean_states, block_ids

    def decode_compressed_sequences(self, compressed_sequences, block_ids):
        """Decode chord sequences according to chords_pred and block_ids."""
        return torch.gather(compressed_sequences, 1, block_ids.unsqueeze(-1).expand(-1, -1, compressed_sequences.size(-1)))

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            src: (batch_size, seq_len, input_size)
            src_key_padding_mask: (batch_size, seq_len), True for valid, False for pad
        Returns:
            Dictionary of predictions.
        """
        # --- Encoder ---
        enc_output = self.pos_encoder(self.enc_input_embed(src))
        
        # Intra-block MHA
        enc_output = self.enc_intra_block_mha(enc_output, n_blocks=self.max_len//4, key_padding_mask=src_key_padding_mask)

        # Main encoder layers
        enc_layer_outputs = [enc_output]
        for i in range(self.n_layers):
            enc_output, _ = self.encoder_layers[i](enc_output, enc_output, enc_output, 
                                                   self.pos_emb, key_padding_mask=src_key_padding_mask)
            enc_output = self.encoder_ffns[i](enc_output)
            enc_layer_outputs.append(enc_output)

        # Weighted sum of encoder layers
        enc_weights = F.softmax(self.enc_weights, dim=0)
        enc_output = torch.stack(enc_layer_outputs, dim=-1)  # (B, S, H, L+1)
        enc_output = (enc_output * enc_weights).sum(dim=-1)  # (B, S, H)

        # --- Chord‐change prediction ---
        boundary_logits = self.chord_change_predictor(enc_output)  # (B, S, 1)
        chord_change_prob = torch.sigmoid(self.slope * boundary_logits)
        chord_change_pred = BinaryRound.apply(chord_change_prob).squeeze(-1)  # (B, S) in {0,1}

        # --- Decoder input embedding with regionalization ---
        dec_input_embed = self.dec_input_embed(src)
        dec_input_embed = F.dropout(dec_input_embed, p=self.dropout_rate, training=self.training)
        dec_input_embed = self.dec_intra_block_mha(dec_input_embed, n_blocks=self.max_len // 4, key_padding_mask=src_key_padding_mask)

        # Compress by predicted chord boundaries and expand back
        dec_input_embed_reg, block_ids = self.chord_block_compression(dec_input_embed, chord_change_pred.long())
        dec_input_embed_reg = self.decode_compressed_sequences(dec_input_embed_reg, block_ids)

        # Combine embeddings
        dec_input_embed = dec_input_embed + dec_input_embed_reg + enc_output

        # Positional encoding
        dec_input_embed = self.pos_encoder(dec_input_embed)
        dec_pos_emb = self.pos_encoder.pe[:, :dec_input_embed.size(1), :].squeeze(0)

        # --- Decoder layers with layer weighting ---
        dec_layer_outputs = [dec_input_embed]
        dec_output = dec_input_embed
        for i in range(self.n_layers):
            dec_output = self.decoder_layers[i](dec_output, enc_output, self.pos_emb, dec_pos_emb,
                                                tgt_key_padding_mask=src_key_padding_mask,
                                                memory_key_padding_mask=src_key_padding_mask)
            dec_layer_outputs.append(dec_output)

        dec_weights = F.softmax(self.dec_weights, dim=0)
        dec_output = torch.stack(dec_layer_outputs, dim=-1)  # (B, S, H, L+1)
        dec_output = (dec_output * dec_weights).sum(dim=-1)

        # --- Output Projections ---
        root_logits = self.root_predictor(dec_output)
        quality_logits = self.quality_predictor(dec_output)
        bass_logits = self.bass_predictor(dec_output)

        preds = {
            'root': root_logits,
            'quality': quality_logits,
            'bass': bass_logits,
            'boundary': boundary_logits
        }

        return preds
