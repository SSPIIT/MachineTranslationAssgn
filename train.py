"""
train.py - Main Training Script
DA6401 Assignment 3

Trains the Transformer on Multi30k (DE → EN) with:
- Noam learning rate schedule
- Label smoothing (eps=0.1)
- W&B logging (loss, BLEU, attention maps)
- Model checkpointing (best validation BLEU)
"""

import os
import math
import argparse
import torch
import torch.nn as nn
import wandb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import Transformer, TransformerLearnedPE
from model_no_scale import TransformerNoScale
from scheduler import NoamScheduler
from loss import LabelSmoothingLoss
from dataset import build_dataloaders
from inference import evaluate_bleu, translate_sentence


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, optimizer, scheduler, criterion,
                src_vocab, tgt_vocab, device, clip=1.0, log_grad_norm=False):
    model.train()
    total_loss = 0
    total_tokens = 0
    grad_norms = []

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_input  = tgt[:, :-1]   # feed all but last token
        tgt_output = tgt[:, 1:]    # predict all but first token

        logits = model(src, tgt_input,
                       src_vocab.pad_idx, tgt_vocab.pad_idx)  # (B, T, V)

        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()

        if log_grad_norm:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(math.sqrt(total_norm))

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()

        n_tokens = (tgt_output != tgt_vocab.pad_idx).sum().item()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, grad_norms


def validate(model, loader, criterion, src_vocab, tgt_vocab, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input,
                           src_vocab.pad_idx, tgt_vocab.pad_idx)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), tgt_output.reshape(-1))

            n_tokens = (tgt_output != tgt_vocab.pad_idx).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


# ─────────────────────────────────────────────
# Attention map visualization (W&B 2.3)
# ─────────────────────────────────────────────
def log_attention_maps(model, src_sentence, de_tokenizer,
                       src_vocab, tgt_vocab, device, step):
    """Extract and log last encoder layer attention heads to W&B."""
    model.eval()
    tokens = de_tokenizer(src_sentence)[:30]
    src_ids = [src_vocab.sos_idx] + src_vocab.encode(tokens) + [src_vocab.eos_idx]
    display_tokens = ['<sos>'] + tokens + ['<eos>']
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        src_mask = model.make_src_mask(src_tensor, src_vocab.pad_idx)
        model.encoder(src_tensor, src_mask)

    last_layer = model.encoder.layers[-1]
    attn = last_layer.self_attn._last_attn_weights  # (1, H, S, S)
    if attn is None:
        return

    attn = attn[0].cpu().numpy()  # (H, S, S)
    num_heads = attn.shape[0]
    S = len(display_tokens)
    attn = attn[:, :S, :S]

    fig, axes = plt.subplots(2, num_heads // 2, figsize=(num_heads * 3, 6))
    axes = axes.flatten()

    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attn[h], cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Head {h + 1}', fontsize=10)
        ax.set_xticks(range(S))
        ax.set_yticks(range(S))
        ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(display_tokens, fontsize=7)

    plt.suptitle(f'Encoder Last Layer – Attention Heads (step {step})')
    plt.tight_layout()
    wandb.log({"attention_maps": wandb.Image(fig)}, step=step)
    plt.close(fig)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # W&B init
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args)
    )

    # Data
    print("Building dataloaders...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=args.batch_size,
        max_len=args.max_len,
        min_freq=args.min_freq
    )

    # Model
    if args.no_scale_attn:
        ModelClass = TransformerNoScale
    elif args.learned_pe:
        ModelClass = TransformerLearnedPE
    else:
        ModelClass = Transformer
    model = ModelClass(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")
    wandb.config.update({"params": count_parameters(model)})

    # Loss
    criterion = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=tgt_vocab.pad_idx,
        smoothing=args.label_smoothing
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,                # actual lr is entirely controlled by Noam
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # Scheduler
    if args.fixed_lr > 0:
        # Ablation 2.1 – fixed learning rate
        for pg in optimizer.param_groups:
            pg['lr'] = args.fixed_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        # wrap to expose .step() uniformly
        class _FixedSched:
            def step(self): pass
            @property
            def current_lr(self): return args.fixed_lr
        scheduler = _FixedSched()
    else:
        scheduler = NoamScheduler(optimizer, d_model=args.d_model,
                                  warmup_steps=args.warmup_steps)

    # Training
    best_bleu = -1.0
    global_step = 0

    # Sample sentence for attention visualization
    sample_de = "Ein Hund läuft über eine Wiese."

    for epoch in range(1, args.epochs + 1):
        log_grad = (epoch == 1)  # log grad norms only in first epoch for ablation 2.2

        train_loss, grad_norms = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, src_vocab, tgt_vocab, device,
            clip=args.clip, log_grad_norm=log_grad
        )
        val_loss = validate(model, val_loader, criterion,
                            src_vocab, tgt_vocab, device)

        current_lr = (scheduler.current_lr
                      if hasattr(scheduler, 'current_lr')
                      else optimizer.param_groups[0]['lr'])

        global_step += len(train_loader)

        # BLEU on validation
        val_bleu = evaluate_bleu(model, val_loader, src_vocab, tgt_vocab,
                                 device, max_len=args.max_len)

        log_dict = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/perplexity": math.exp(min(train_loss, 20)),
            "val/loss": val_loss,
            "val/perplexity": math.exp(min(val_loss, 20)),
            "val/bleu": val_bleu,
            "lr": current_lr,
        }

        # Log gradient norms (ablation 2.2) for first 1000 steps
        if grad_norms:
            log_dict["grad_norm/mean"] = np.mean(grad_norms)
            log_dict["grad_norm/max"]  = np.max(grad_norms)

        wandb.log(log_dict, step=global_step)

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val BLEU: {val_bleu:.2f} | "
              f"LR: {current_lr:.6f}")

        # Checkpoint
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            ckpt_path = os.path.join(args.save_dir, f"{args.run_name}_best.pt")
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_bleu': val_bleu,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'args': vars(args),
            }, ckpt_path)
            print(f"  → Saved best checkpoint (BLEU {val_bleu:.2f})")

        # Log attention maps every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            from dataset import SpacyTokenizer
            de_tokenizer = SpacyTokenizer('de_core_news_sm')
            log_attention_maps(model, sample_de, de_tokenizer,
                               src_vocab, tgt_vocab, device, global_step)

    # Final test BLEU using best checkpoint
    print("\nEvaluating best checkpoint on test set...")
    ckpt = torch.load(
        os.path.join(args.save_dir, f"{args.run_name}_best.pt"),
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(ckpt['model_state_dict'])
    test_bleu = evaluate_bleu(model, test_loader, src_vocab, tgt_vocab,
                               device, max_len=args.max_len)
    print(f"Test BLEU: {test_bleu:.2f}")
    wandb.log({"test/bleu": test_bleu})

    # Log a few example translations
    from dataset import SpacyTokenizer
    de_tokenizer = SpacyTokenizer('de_core_news_sm')
    examples = [
        "Ein Mann sitzt auf einer Bank.",
        "Zwei Hunde spielen im Schnee.",
        "Eine Frau liest ein Buch.",
    ]
    translations_table = wandb.Table(columns=["German", "English (predicted)"])
    for de in examples:
        en = translate_sentence(model, de, de_tokenizer, src_vocab, tgt_vocab, device)
        translations_table.add_data(de, en)
    wandb.log({"example_translations": translations_table})

    wandb.finish()
    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer NMT (DE→EN)')

    # Model
    parser.add_argument('--d_model',    type=int,   default=256)
    parser.add_argument('--num_layers', type=int,   default=3)
    parser.add_argument('--num_heads',  type=int,   default=8)
    parser.add_argument('--d_ff',       type=int,   default=512)
    parser.add_argument('--dropout',    type=float, default=0.1)
    parser.add_argument('--max_len',    type=int,   default=256)

    # Training
    parser.add_argument('--epochs',          type=int,   default=20)
    parser.add_argument('--batch_size',      type=int,   default=128)
    parser.add_argument('--warmup_steps',    type=int,   default=4000)
    parser.add_argument('--clip',            type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--min_freq',        type=int,   default=2)

    # Ablations
    parser.add_argument('--no_scale_attn', action='store_true',
                        help='Disable 1/sqrt(dk) scaling (ablation 2.2)')
    parser.add_argument('--fixed_lr',   type=float, default=0.0,
                        help='Use fixed LR instead of Noam (0 = use Noam)')
    parser.add_argument('--learned_pe', action='store_true',
                        help='Use learned positional embeddings (ablation 2.4)')

    # Misc
    parser.add_argument('--save_dir',       type=str, default='checkpoints')
    parser.add_argument('--run_name',       type=str, default='transformer_base')
    parser.add_argument('--wandb_project',  type=str, default='da6401_assignment3')

    args = parser.parse_args()
    main(args)