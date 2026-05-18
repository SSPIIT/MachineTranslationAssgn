"""
dataset.py - Data loading, tokenization, and vocabulary for Multi30k
DA6401 Assignment 3

Tokenization: spaCy (de_core_news_sm for German, en_core_web_sm for English)
"""

import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from datasets import load_dataset


# ─────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────
class Vocabulary:
    """Simple vocabulary with special tokens."""

    PAD = '<pad>'
    SOS = '<sos>'
    EOS = '<eos>'
    UNK = '<unk>'

    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.token2idx = {}
        self.idx2token = {}
        self._special = [self.PAD, self.SOS, self.EOS, self.UNK]
        for tok in self._special:
            self._add(tok)

    def _add(self, token):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def build(self, tokenized_sentences):
        counter = Counter()
        for tokens in tokenized_sentences:
            counter.update(tokens)
        for tok, freq in counter.items():
            if freq >= self.min_freq:
                self._add(tok)

    def encode(self, tokens):
        unk = self.token2idx[self.UNK]
        return [self.token2idx.get(t, unk) for t in tokens]

    def decode(self, indices):
        return [self.idx2token.get(i, self.UNK) for i in indices]

    def __len__(self):
        return len(self.token2idx)

    @property
    def pad_idx(self):
        return self.token2idx[self.PAD]

    @property
    def sos_idx(self):
        return self.token2idx[self.SOS]

    @property
    def eos_idx(self):
        return self.token2idx[self.EOS]


# ─────────────────────────────────────────────
# Tokenizer (spaCy)
# ─────────────────────────────────────────────
class SpacyTokenizer:
    def __init__(self, lang):
        try:
            self.nlp = spacy.load(lang)
        except OSError:
            raise OSError(
                f"spaCy model '{lang}' not found. "
                f"Run: python -m spacy download {lang}"
            )

    def __call__(self, text):
        return [tok.text.lower() for tok in self.nlp.tokenizer(text)]


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class Multi30kDataset(Dataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer,
                 src_vocab, tgt_vocab, max_len=256):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.pairs = []

        for example in data:
            src_tokens = src_tokenizer(example['de'])[:max_len - 2]
            tgt_tokens = tgt_tokenizer(example['en'])[:max_len - 2]

            src_ids = ([src_vocab.sos_idx]
                       + src_vocab.encode(src_tokens)
                       + [src_vocab.eos_idx])
            tgt_ids = ([tgt_vocab.sos_idx]
                       + tgt_vocab.encode(tgt_tokens)
                       + [tgt_vocab.eos_idx])

            self.pairs.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch, src_pad_idx, tgt_pad_idx):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)
    return src_padded, tgt_padded


# ─────────────────────────────────────────────
# Main build function
# ─────────────────────────────────────────────
def build_dataloaders(batch_size=128, max_len=256, min_freq=2):
    """
    Download Multi30k, tokenize with spaCy, build vocabularies,
    and return DataLoaders.
    """
    print("Loading Multi30k dataset...")
    dataset = load_dataset("bentrevett/multi30k")
    train_data = list(dataset['train'])
    val_data   = list(dataset['validation'])
    test_data  = list(dataset['test'])

    print("Loading spaCy tokenizers...")
    de_tokenizer = SpacyTokenizer('de_core_news_sm')
    en_tokenizer = SpacyTokenizer('en_core_web_sm')

    print("Tokenizing training data for vocabulary...")
    train_de = [de_tokenizer(ex['de'])[:max_len - 2] for ex in train_data]
    train_en = [en_tokenizer(ex['en'])[:max_len - 2] for ex in train_data]

    src_vocab = Vocabulary(min_freq=min_freq)
    tgt_vocab = Vocabulary(min_freq=min_freq)
    src_vocab.build(train_de)
    tgt_vocab.build(train_en)
    print(f"Vocab sizes — DE: {len(src_vocab):,}  EN: {len(tgt_vocab):,}")

    train_ds = Multi30kDataset(train_data, de_tokenizer, en_tokenizer,
                               src_vocab, tgt_vocab, max_len)
    val_ds   = Multi30kDataset(val_data,   de_tokenizer, en_tokenizer,
                               src_vocab, tgt_vocab, max_len)
    test_ds  = Multi30kDataset(test_data,  de_tokenizer, en_tokenizer,
                               src_vocab, tgt_vocab, max_len)

    from functools import partial
    _collate = partial(collate_fn,
                       src_pad_idx=src_vocab.pad_idx,
                       tgt_pad_idx=tgt_vocab.pad_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=_collate, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=_collate, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, collate_fn=_collate, num_workers=2)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab