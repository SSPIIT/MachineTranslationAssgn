"""
dataset.py — Multi30k Dataset Loading, Tokenization, and Vocabulary
DA6401 Assignment 3: "Attention Is All You Need"
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy
from collections import Counter
from typing import List, Tuple, Dict, Optional


# ══════════════════════════════════════════════════════════════════════
#  VOCABULARY
# ══════════════════════════════════════════════════════════════════════

class Vocabulary:
    """
    Simple vocabulary class that maps tokens ↔ integer indices.

    Special tokens:
        <unk>  index 0  – unknown token
        <pad>  index 1  – padding token
        <sos>  index 2  – start of sequence
        <eos>  index 3  – end of sequence
    """

    UNK = "<unk>"
    PAD = "<pad>"
    SOS = "<sos>"
    EOS = "<eos>"

    UNK_IDX = 0
    PAD_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3

    def __init__(self) -> None:
        self.stoi: Dict[str, int] = {
            self.UNK: self.UNK_IDX,
            self.PAD: self.PAD_IDX,
            self.SOS: self.SOS_IDX,
            self.EOS: self.EOS_IDX,
        }
        self.itos: Dict[int, str] = {v: k for k, v in self.stoi.items()}

    def build_from_counter(self, counter: Counter, min_freq: int = 2) -> None:
        """
        Populate vocab from a token frequency Counter.

        Args:
            counter  : Counter mapping token → frequency.
            min_freq : Minimum frequency to include a token (default 2).
        """
        for token, freq in counter.most_common():
            if freq < min_freq:
                break
            if token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

    def __len__(self) -> int:
        return len(self.stoi)

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, self.UNK)

    def lookup_index(self, token: str) -> int:
        return self.stoi.get(token, self.UNK_IDX)

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens → list of indices (with SOS/EOS)."""
        return (
            [self.SOS_IDX]
            + [self.stoi.get(t, self.UNK_IDX) for t in tokens]
            + [self.EOS_IDX]
        )

    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """Convert list of indices → list of tokens."""
        special = {self.UNK_IDX, self.PAD_IDX, self.SOS_IDX, self.EOS_IDX}
        result = []
        for idx in indices:
            if skip_special and idx in special:
                continue
            result.append(self.itos.get(idx, self.UNK))
        return result


# ══════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    """
    Wrapper around the bentrevett/multi30k HuggingFace dataset.

    Handles:
      - Loading the dataset splits
      - Tokenization with spaCy (de_core_news_sm / en_core_web_sm)
      - Vocabulary construction (shared across train/val/test)
      - Conversion of raw sentences → padded integer tensors

    Args:
        split      : 'train', 'validation', or 'test'
        src_vocab  : Pre-built Vocabulary for German  (None → build from data)
        tgt_vocab  : Pre-built Vocabulary for English (None → build from data)
        min_freq   : Minimum token frequency for vocabulary inclusion
        max_len    : Maximum sequence length (longer sequences are dropped)
    """

    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[Vocabulary] = None,
        tgt_vocab: Optional[Vocabulary] = None,
        min_freq: int = 2,
        max_len: int = 100,
    ) -> None:
        self.split = split
        self.max_len = max_len

        # ── load spaCy tokenizers ──────────────────────────────────────
        try:
            # self.de_nlp = spacy.load("de_core_news_sm")
            self.de_nlp = spacy.blank("de")
        except OSError:
            raise OSError(
                "German spaCy model not found. "
                "Run: python -m spacy download de_core_news_sm"
            )
        try:
            # self.en_nlp = spacy.load("en_core_web_sm")
            self.en_nlp = spacy.blank("en")
        except OSError:
            raise OSError(
                "English spaCy model not found. "
                "Run: python -m spacy download en_core_web_sm"
            )

        # ── load HuggingFace dataset ───────────────────────────────────
        raw = load_dataset("bentrevett/multi30k", trust_remote_code=True)
        self.raw_data = raw[split]

        # ── tokenize all sentences ────────────────────────────────────
        self.src_tokens: List[List[str]] = []
        self.tgt_tokens: List[List[str]] = []
        self._tokenize_all()

        # ── build or reuse vocabularies ───────────────────────────────
        if src_vocab is None:
            self.src_vocab = Vocabulary()
            src_counter = Counter(
                tok for sent in self.src_tokens for tok in sent
            )
            self.src_vocab.build_from_counter(src_counter, min_freq=min_freq)
        else:
            self.src_vocab = src_vocab

        if tgt_vocab is None:
            self.tgt_vocab = Vocabulary()
            tgt_counter = Counter(
                tok for sent in self.tgt_tokens for tok in sent
            )
            self.tgt_vocab.build_from_counter(tgt_counter, min_freq=min_freq)
        else:
            self.tgt_vocab = tgt_vocab

        # ── convert tokens → integer lists ────────────────────────────
        self.src_data, self.tgt_data = self._encode_all()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenize_de(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.de_nlp(text.strip())]

    def _tokenize_en(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.en_nlp(text.strip())]

    def _tokenize_all(self) -> None:
        for example in self.raw_data:
            de_toks = self._tokenize_de(example["de"])
            en_toks = self._tokenize_en(example["en"])
            self.src_tokens.append(de_toks)
            self.tgt_tokens.append(en_toks)

    def _encode_all(self) -> Tuple[List[List[int]], List[List[int]]]:
        src_data, tgt_data = [], []
        for src_toks, tgt_toks in zip(self.src_tokens, self.tgt_tokens):
            # +2 for SOS/EOS
            if len(src_toks) + 2 > self.max_len or len(tgt_toks) + 2 > self.max_len:
                continue
            src_data.append(self.src_vocab.encode(src_toks))
            tgt_data.append(self.tgt_vocab.encode(tgt_toks))
        return src_data, tgt_data

    # ------------------------------------------------------------------
    # Public methods required by assignment skeleton
    # ------------------------------------------------------------------

    def build_vocab(self) -> Tuple[Vocabulary, Vocabulary]:
        """Return the (src_vocab, tgt_vocab) built during __init__."""
        return self.src_vocab, self.tgt_vocab

    def process_data(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Return (src_data, tgt_data) as lists of integer index lists."""
        return self.src_data, self.tgt_data

    def tokenize_src(self, text: str) -> List[str]:
        """Tokenize a raw German sentence."""
        return self._tokenize_de(text)

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = torch.tensor(self.src_data[idx], dtype=torch.long)
        tgt = torch.tensor(self.tgt_data[idx], dtype=torch.long)
        return src, tgt


# ══════════════════════════════════════════════════════════════════════
#  COLLATE & DATALOADER FACTORY
# ══════════════════════════════════════════════════════════════════════

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int = Vocabulary.PAD_IDX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a batch of (src, tgt) tensor pairs to the same length.

    Returns:
        src_batch : [batch, src_len]
        tgt_batch : [batch, tgt_len]
    """
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_batch, tgt_batch


def get_dataloaders(
    batch_size: int = 128,
    min_freq: int = 2,
    max_len: int = 100,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """
    Build train/val/test DataLoaders sharing the same vocabulary.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    # Build vocab from training split only
    train_ds = Multi30kDataset(
        split="train", min_freq=min_freq, max_len=max_len
    )
    src_vocab, tgt_vocab = train_ds.build_vocab()

    val_ds = Multi30kDataset(
        split="validation",
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=max_len,
    )
    test_ds = Multi30kDataset(
        split="test",
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=max_len,
    )

    from functools import partial
    _collate = partial(collate_fn, pad_idx=Vocabulary.PAD_IDX)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=2, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab