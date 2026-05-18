"""
model_no_scale.py - Transformer variant WITHOUT the 1/sqrt(dk) scaling factor.
Used for ablation study 2.2.

Import this instead of model.py when running with --no_scale_attn.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-import everything from model.py and only override scaled_dot_product_attention
from model import (
    MultiHeadAttention, PositionalEncoding, PositionwiseFeedForward,
    EncoderLayer, DecoderLayer, Encoder, Decoder, Transformer
)


def unscaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention WITHOUT the 1/sqrt(dk) scaling factor.
    Used for ablation 2.2.
    """
    # NO division by sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    output = torch.matmul(attn_weights, V)
    return output, attn_weights


class MultiHeadAttentionNoScale(MultiHeadAttention):
    """Multi-Head Attention without scaling factor (ablation 2.2)."""

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q = self.split_heads(self.W_Q(query))
        K = self.split_heads(self.W_K(key))
        V = self.split_heads(self.W_V(value))

        x, attn_weights = unscaled_dot_product_attention(Q, K, V, mask)
        self._last_attn_weights = attn_weights.detach()

        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_O(x)


class EncoderLayerNoScale(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionNoScale(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, src_mask))
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        return x


class DecoderLayerNoScale(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttentionNoScale(d_model, num_heads)
        self.cross_attn = MultiHeadAttentionNoScale(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, tgt_mask))
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.cross_attn(x, enc_output, enc_output, src_mask))
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout(self.ffn(x))
        return x


class EncoderNoScale(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayerNoScale(d_model, num_heads, d_ff, dropout)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderNoScale(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayerNoScale(d_model, num_heads, d_ff, dropout)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return self.norm(x)


class TransformerNoScale(Transformer):
    """Transformer without attention scaling factor (ablation 2.2)."""

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=256, num_layers=3, num_heads=8,
                 d_ff=512, dropout=0.1, max_len=256):
        nn.Module.__init__(self)
        self.encoder = EncoderNoScale(src_vocab_size, d_model, num_layers,
                                      num_heads, d_ff, dropout, max_len)
        self.decoder = DecoderNoScale(tgt_vocab_size, d_model, num_layers,
                                      num_heads, d_ff, dropout, max_len)
        self.projection = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()