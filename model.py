"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"
"""

import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_w = F.softmax(scores, dim=-1)

    attn_w = torch.nan_to_num(attn_w, nan=0.0)

    output = torch.matmul(attn_w, V)

    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:

    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:

    tgt_len = tgt.size(1)

    device = tgt.device

    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    causal_mask = torch.triu(
        torch.ones(
            tgt_len,
            tgt_len,
            dtype=torch.bool,
            device=device
        ),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)

    tgt_mask = pad_mask | causal_mask

    return tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTIHEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):

        B, T, _ = x.shape

        x = x.view(B, T, self.num_heads, self.d_k)

        return x.transpose(1, 2)

    def forward(
        self,
        query,
        key,
        value,
        mask=None,
    ):

        B = query.size(0)

        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        attn_out, _ = scaled_dot_product_attention(
            Q,
            K,
            V,
            mask
        )

        attn_out = attn_out.transpose(1, 2).contiguous()

        attn_out = attn_out.view(B, -1, self.d_model)

        output = self.W_o(attn_out)

        return output


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float
            )
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED FORWARD
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
        dropout=0.1,
    ):

        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)

        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        return self.linear2(
            self.dropout(
                F.relu(
                    self.linear1(x)
                )
            )
        )


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1,
    ):

        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout
        )

        self.ffn = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):

        x = x + self.dropout(
            self.self_attn(
                self.norm1(x),
                self.norm1(x),
                self.norm1(x),
                src_mask
            )
        )

        x = x + self.dropout(
            self.ffn(
                self.norm2(x)
            )
        )

        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1,
    ):

        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout
        )

        self.cross_attn = MultiHeadAttention(
            d_model,
            num_heads,
            dropout
        )

        self.ffn = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.norm2 = nn.LayerNorm(d_model)

        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        memory,
        src_mask,
        tgt_mask,
    ):

        x = x + self.dropout(
            self.self_attn(
                self.norm1(x),
                self.norm1(x),
                self.norm1(x),
                tgt_mask
            )
        )

        x = x + self.dropout(
            self.cross_attn(
                self.norm2(x),
                memory,
                memory,
                src_mask
            )
        )

        x = x + self.dropout(
            self.ffn(
                self.norm3(x)
            )
        )

        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):

    def __init__(self, layer, N):

        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )

        self.norm = nn.LayerNorm(
            layer.norm1.normalized_shape
        )

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  DECODER
# ══════════════════════════════════════════════════════════════════════

class Decoder(nn.Module):

    def __init__(self, layer, N):

        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )

        self.norm = nn.LayerNorm(
            layer.norm1.normalized_shape
        )

    def forward(
        self,
        x,
        memory,
        src_mask,
        tgt_mask,
    ):

        for layer in self.layers:
            x = layer(
                x,
                memory,
                src_mask,
                tgt_mask
            )

        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size: int = 10000,
        tgt_vocab_size: int = 10000,
        d_model: int = 256,
        N: int = 3,
        d_ff: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        pad_idx: int = 1,
    ) -> None:

        super().__init__()

        import spacy
        from dataset import get_dataloaders

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # self.de_nlp = spacy.load("de_core_news_sm")
        self.de_nlp = spacy.blank("de")

        _, _, _, self.src_vocab, self.tgt_vocab = get_dataloaders(
            batch_size=1
        )

        src_vocab_size = len(self.src_vocab)
        tgt_vocab_size = len(self.tgt_vocab)

        self.d_model = d_model

        self.pad_idx = pad_idx

        self.src_embed = nn.Embedding(
            src_vocab_size,
            d_model,
            padding_idx=pad_idx
        )

        self.tgt_embed = nn.Embedding(
            tgt_vocab_size,
            d_model,
            padding_idx=pad_idx
        )

        self.src_pe = PositionalEncoding(
            d_model,
            dropout
        )

        self.tgt_pe = PositionalEncoding(
            d_model,
            dropout
        )

        enc_layer = EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        dec_layer = DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        self.encoder = Encoder(enc_layer, N)

        self.decoder = Decoder(dec_layer, N)

        self.fc_out = nn.Linear(
            d_model,
            tgt_vocab_size
        )

        self._init_weights()

        self._load_from_drive(
            "best_checkpoint.pt"
        )

    def _init_weights(self):

        for p in self.parameters():

            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _load_from_drive(self, path):

        try:

            import gdown

            gdown.download(
                id="10ZkESdbBEproTxuYRDkDw0slEh31uzzw",
                output=path,
                quiet=False
            )

            state = torch.load(
                path,
                map_location="cpu"
            )

            self.load_state_dict(state)

            print(
                "[Transformer] Weights loaded successfully."
            )

        except Exception as e:

            print(
                f"[Transformer] Warning: could not load checkpoint — {e}"
            )

    def encode(
        self,
        src,
        src_mask,
    ):

        src_emb = self.src_pe(
            self.src_embed(src)
            * math.sqrt(self.d_model)
        )

        memory = self.encoder(
            src_emb,
            src_mask
        )

        return memory

    def decode(
        self,
        memory,
        src_mask,
        tgt,
        tgt_mask,
    ):

        tgt_emb = self.tgt_pe(
            self.tgt_embed(tgt)
            * math.sqrt(self.d_model)
        )

        dec_out = self.decoder(
            tgt_emb,
            memory,
            src_mask,
            tgt_mask
        )

        logits = self.fc_out(dec_out)

        return logits

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
    ):

        memory = self.encode(
            src,
            src_mask
        )

        logits = self.decode(
            memory,
            src_mask,
            tgt,
            tgt_mask
        )

        return logits

    def infer(
        self,
        src_sentence: str,
        device: str = "cpu",
        max_len: int = 20,
    ) -> str:

        self.eval()

        with torch.no_grad():

            tokens = [
                tok.text.lower()
                for tok in self.de_nlp(
                    src_sentence.strip()
                )
            ]

            indices = self.src_vocab.encode(
                tokens
            )

            src = torch.tensor(
                indices,
                dtype=torch.long
            ).unsqueeze(0).to(device)

            src_mask = make_src_mask(
                src,
                pad_idx=self.pad_idx
            )

            memory = self.encode(
                src,
                src_mask
            )

            sos_idx = self.tgt_vocab.SOS_IDX

            eos_idx = self.tgt_vocab.EOS_IDX

            ys = torch.tensor(
                [[sos_idx]],
                dtype=torch.long,
                device=device
            )

            for _ in range(max_len):

                tgt_mask = make_tgt_mask(
                    ys,
                    pad_idx=self.pad_idx
                )

                logits = self.decode(
                    memory,
                    src_mask,
                    ys,
                    tgt_mask
                )

                next_tok = logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True
                )

                ys = torch.cat(
                    [ys, next_tok],
                    dim=1
                )

                if next_tok.item() == eos_idx:
                    break

            generated = ys.squeeze(0).tolist()[1:]

            words = self.tgt_vocab.decode(
                generated,
                skip_special=True
            )

            return " ".join(words)