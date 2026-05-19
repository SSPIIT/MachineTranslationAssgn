# python

"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import math
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth[correct] = 1 - eps + eps / (vocab_size - 1)
        y_smooth[other]   = eps / (vocab_size - 1)
        y_smooth[pad]     = 0   (ignored entirely)

    Args:
        vocab_size : Number of output classes.
        pad_idx    : Index of <pad> token — receives 0 probability.
        smoothing  : Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing
        # KLDiv expects log-probabilities for input
        self.criterion  = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value (mean over non-pad tokens).
        """
        # Log-softmax for KLDiv
        log_probs = torch.log_softmax(logits, dim=-1)

        # Build smoothed distribution
        smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_dist[:, self.pad_idx] = 0.0

        # Zero out rows where target is padding
        pad_mask = (target == self.pad_idx)
        smooth_dist[pad_mask] = 0.0

        # KL divergence: sum / number of non-pad tokens
        loss = self.criterion(log_probs, smooth_dist)
        n_tokens = (~pad_mask).sum().item()
        return loss / max(n_tokens, 1)


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0
    pad_idx = model.pad_idx

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    pbar = tqdm(data_iter, desc=f"{'Train' if is_train else 'Val  '} Epoch {epoch_num}")

    with ctx:
        for src, tgt in pbar:
            src = src.to(device)          # [B, src_len]
            tgt = tgt.to(device)          # [B, tgt_len]

            # Decoder input: all but last token
            # Decoder target: all but first token (SOS)
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx=pad_idx)
            tgt_mask = make_tgt_mask(tgt_inp, pad_idx=pad_idx)

            logits = model(src, tgt_inp, src_mask, tgt_mask)
            # logits: [B, tgt_len-1, vocab_size]

            # Flatten for loss
            B, T, V = logits.shape
            loss = loss_fn(
                logits.contiguous().view(B * T, V),
                tgt_out.contiguous().view(B * T),
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            n_tokens = (tgt_out != pad_idx).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys     = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=model.pad_idx)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys       = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
        tgt_vocab       : Vocabulary object.
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU × 100 (float, range 0–100).
    """
    from nltk.translate.bleu_score import corpus_bleu

    model.eval()
    pad_idx = model.pad_idx
    sos_idx = tgt_vocab.SOS_IDX
    eos_idx = tgt_vocab.EOS_IDX

    hypotheses  = []
    references  = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval"):
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i     = src[i].unsqueeze(0)
                src_mask  = make_src_mask(src_i, pad_idx=pad_idx)
                pred_ids  = greedy_decode(
                    model, src_i, src_mask,
                    max_len=max_len,
                    start_symbol=sos_idx,
                    end_symbol=eos_idx,
                    device=device,
                ).squeeze(0).tolist()

                # Strip SOS and everything from EOS onward
                if sos_idx in pred_ids:
                    pred_ids = pred_ids[pred_ids.index(sos_idx) + 1:]
                if eos_idx in pred_ids:
                    pred_ids = pred_ids[: pred_ids.index(eos_idx)]

                pred_tokens = tgt_vocab.decode(pred_ids, skip_special=True)

                # Reference: strip SOS, EOS, PAD
                ref_ids    = tgt[i].tolist()
                ref_tokens = tgt_vocab.decode(ref_ids, skip_special=True)

                hypotheses.append(pred_tokens)
                references.append([ref_tokens])

    # torchtext bleu_score expects:
    #   candidate_corpus : List[List[str]]
    #   references_corpus: List[List[List[str]]]
    score = corpus_bleu(references, hypotheses) * 100.0
    return score


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    """
    # Collect model config so it can be reconstructed
    src_embed = model.src_embed
    tgt_embed = model.tgt_embed
    enc_layer = model.encoder.layers[0]
    model_config = {
        "src_vocab_size": src_embed.num_embeddings,
        "tgt_vocab_size": tgt_embed.num_embeddings,
        "d_model":        model.d_model,
        "N":              len(model.encoder.layers),
        "num_heads":      enc_layer.self_attn.num_heads,
        "d_ff":           enc_layer.ffn.linear1.out_features,
        "dropout":        enc_layer.dropout.p,
        "pad_idx":        model.pad_idx,
    }

    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config":         model_config,
        },
        path,
    )
    print(f"[Checkpoint] Saved epoch {epoch} → {path}")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = ckpt.get("epoch", 0)
    print(f"[Checkpoint] Loaded epoch {epoch} from {path}")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.
    """
    import wandb
    from dataset import get_dataloaders, Vocabulary

    # ── Hyperparameters ───────────────────────────────────────────────
    config = {
        "d_model":      256,
        "N":            3,
        "num_heads":    8,
        "d_ff":         512,
        "dropout":      0.1,
        "warmup_steps": 4000,
        "batch_size":   128,
        "num_epochs":   20,
        "min_freq":     2,
        "max_len":      100,
        "label_smoothing": 0.1,
    }

    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=cfg.batch_size,
        min_freq=cfg.min_freq,
        max_len=cfg.max_len,
    )
    print(f"[Data] src_vocab={len(src_vocab)}  tgt_vocab={len(tgt_vocab)}")

    # ── Model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.d_model,
        N=cfg.N,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        pad_idx=Vocabulary.PAD_IDX,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {n_params:,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)

    # ── Loss ──────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=Vocabulary.PAD_IDX,
        smoothing=cfg.label_smoothing,
    )

    # ── Training Loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler, epoch,
            is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            None, None, epoch,
            is_train=False, device=device,
        )

        lr = optimizer.param_groups[0]["lr"]
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "lr": lr, "epoch": epoch})
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path="best_checkpoint.pt")

    # ── Final BLEU ────────────────────────────────────────────────────
    # Load best checkpoint
    load_checkpoint("best_checkpoint.pt", model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    wandb.log({"test_bleu": bleu})
    print(f"[Eval] Test BLEU: {bleu:.2f}")
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()