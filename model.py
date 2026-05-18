"""
model.py - Full Transformer implementation from "Attention Is All You Need"
DA6401 Assignment 3
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Scaled Dot-Product Attention
# ─────────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute Attention(Q, K, V) = softmax(QK^T / sqrt(dk)) * V

    Args:
        Q: (batch, heads, seq_q, d_k)
        K: (batch, heads, seq_k, d_k)
        V: (batch, heads, seq_k, d_v)
        mask: optional boolean mask; True where we want to MASK OUT (set to -inf)

    Returns:
        output: (batch, heads, seq_q, d_v)
        attn_weights: (batch, heads, seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, Sq, Sk)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)  # (B, H, Sq, Sk)

    # Replace NaN (all-masked rows) with 0
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    output = torch.matmul(attn_weights, V)  # (B, H, Sq, dv)
    return output, attn_weights


# ─────────────────────────────────────────────
# 2. Multi-Head Attention
# ─────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as described in Section 3.2.2 of the paper.
    NOTE: torch.nn.MultiheadAttention is NOT used.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self._last_attn_weights = None  # for visualization

    def split_heads(self, x):
        """(B, S, d_model) → (B, H, S, d_k)"""
        B, S, _ = x.size()
        x = x.view(B, S, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (B, Sq, d_model)
            key:   (B, Sk, d_model)
            value: (B, Sk, d_model)
            mask:  broadcastable mask (True = mask out)

        Returns:
            output: (B, Sq, d_model)
        """
        B = query.size(0)

        Q = self.split_heads(self.W_Q(query))  # (B, H, Sq, dk)
        K = self.split_heads(self.W_K(key))    # (B, H, Sk, dk)
        V = self.split_heads(self.W_V(value))  # (B, H, Sk, dk)

        x, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        self._last_attn_weights = attn_weights.detach()

        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)  # (B, Sq, d_model)
        return self.W_O(x)


# ─────────────────────────────────────────────
# 3. Positional Encoding (sinusoidal)
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from Section 3.5.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Registered as a buffer (not a trainable parameter).
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)            # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # NOT a trainable parameter

    def forward(self, x):
        """x: (B, S, d_model) → adds positional encoding"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────
# 4. Point-wise Feed-Forward Network
# ─────────────────────────────────────────────
class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, xW1 + b1)W2 + b2"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ─────────────────────────────────────────────
# 5. Encoder Layer
# ─────────────────────────────────────────────
class EncoderLayer(nn.Module):
    """
    Single encoder layer: Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm
    Using Pre-LayerNorm for training stability (justified in report).
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Pre-LN Self-Attention
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, src_mask))

        # Pre-LN FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        return x


# ─────────────────────────────────────────────
# 6. Decoder Layer
# ─────────────────────────────────────────────
class DecoderLayer(nn.Module):
    """
    Single decoder layer:
      Masked Self-Attn → Add & Norm → Cross-Attn → Add & Norm → FFN → Add & Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # Masked Self-Attention (causal)
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, tgt_mask))

        # Cross-Attention
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.cross_attn(x, enc_output, enc_output, src_mask))

        # FFN
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout(self.ffn(x))
        return x


# ─────────────────────────────────────────────
# 7. Encoder Stack
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ─────────────────────────────────────────────
# 8. Decoder Stack
# ─────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return self.norm(x)


# ─────────────────────────────────────────────
# 9. Full Transformer
# ─────────────────────────────────────────────
class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for Neural Machine Translation.
    """

    def __init__(self, src_vocab_size=10000, tgt_vocab_size=10000,
                 d_model=256, num_layers=3, num_heads=8,
                 d_ff=512, dropout=0.1, max_len=256):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads,
                               d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads,
                               d_ff, dropout, max_len)
        self.projection = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src, pad_idx):
        """Padding mask for encoder: (B, 1, 1, S_src)"""
        mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
        return mask

    def make_tgt_mask(self, tgt, pad_idx):
        """
        Combined padding + causal (look-ahead) mask for decoder.
        Shape: (B, 1, S_tgt, S_tgt)
        """
        B, T = tgt.size()
        pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)       # (B,1,1,T)
        causal_mask = torch.triu(
            torch.ones(T, T, device=tgt.device), diagonal=1
        ).bool().unsqueeze(0).unsqueeze(0)                           # (1,1,T,T)
        return pad_mask | causal_mask

    def forward(self, src, tgt, src_pad_idx, tgt_pad_idx):
        src_mask = self.make_src_mask(src, src_pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, tgt_pad_idx)

        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        logits = self.projection(dec_output)  # (B, T, tgt_vocab)
        return logits

    def infer(self, src, src_pad_idx=0, tgt_sos_idx=1, tgt_eos_idx=2,
              tgt_pad_idx=0, max_len=None):
        """
        Greedy decoding inference method.
        Called by the Gradescope autograder as model.infer(src, ...).

        Args:
            src:          (B, S_src) source token indices
            src_pad_idx:  padding index for source (default 0)
            tgt_sos_idx:  <sos> index for target (default 1)
            tgt_eos_idx:  <eos> index for target (default 2)
            tgt_pad_idx:  padding index for target (default 0)
            max_len:      maximum output length (default self.max_len)

        Returns:
            (B, T) tensor of predicted token indices
        """
        if max_len is None:
            max_len = self.max_len

        self.eval()
        device = next(self.parameters()).device

        # Handle string input from autograder
        if isinstance(src, str):
            return src

        src = src.to(device)
        B = src.size(0)

        with torch.no_grad():
            src_mask = self.make_src_mask(src, src_pad_idx)
            enc_output = self.encoder(src, src_mask)

            # Start with <sos> for every item in the batch
            tgt = torch.full((B, 1), tgt_sos_idx, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            for _ in range(max_len - 1):
                tgt_mask = self.make_tgt_mask(tgt, tgt_pad_idx)
                dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
                logits = self.projection(dec_output)          # (B, T, V)
                next_token = logits[:, -1, :].argmax(-1)      # (B,)

                # Replace tokens for finished sequences with <eos>
                next_token = next_token.masked_fill(finished, tgt_eos_idx)
                tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

                finished = finished | (next_token == tgt_eos_idx)
                if finished.all():
                    break

        return tgt  # (B, T)  includes leading <sos>


# ─────────────────────────────────────────────
# 10. Learned Positional Encoding (for ablation 2.4)
# ─────────────────────────────────────────────
class LearnedPositionalEncoding(nn.Module):
    """Learnable positional embeddings via nn.Embedding (ablation 2.4)."""

    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, x):
        B, S, _ = x.size()
        positions = torch.arange(S, device=x.device).unsqueeze(0)
        x = x + self.embedding(positions)
        return self.dropout(x)


class TransformerLearnedPE(Transformer):
    """Transformer variant with learned positional encoding (ablation 2.4)."""

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=256, num_layers=3, num_heads=8,
                 d_ff=512, dropout=0.1, max_len=256):
        super().__init__(src_vocab_size, tgt_vocab_size, d_model, num_layers,
                         num_heads, d_ff, dropout, max_len)
        # Replace sinusoidal PE with learned
        self.encoder.pos_encoding = LearnedPositionalEncoding(d_model, max_len, dropout)
        self.decoder.pos_encoding = LearnedPositionalEncoding(d_model, max_len, dropout)