"""
loss.py - Label Smoothing Cross-Entropy Loss
DA6401 Assignment 3

Label smoothing with epsilon=0.1 as described in Section 5.4 of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss.

    Instead of a one-hot target, we use:
        target_smooth = (1 - eps) * one_hot + eps / vocab_size

    This prevents the model from becoming over-confident.

    Args:
        vocab_size:  size of the target vocabulary
        pad_idx:     index of the padding token (ignored in loss)
        smoothing:   epsilon_ls (default 0.1)
    """

    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        """
        Args:
            logits:  (B * T, vocab_size) – raw (unnormalized) scores
            targets: (B * T,)            – integer target indices

        Returns:
            scalar loss
        """
        log_probs = F.log_softmax(logits, dim=-1)  # (N, V)

        # Build smoothed target distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0.0  # never assign probability mass to <pad>
            # Zero-out padding positions entirely
            pad_mask = targets.eq(self.pad_idx)
            smooth_dist[pad_mask] = 0.0

        loss = -(smooth_dist * log_probs).sum(dim=-1)

        # Mean over non-pad tokens
        non_pad = (~pad_mask).sum()
        return loss.sum() / non_pad.clamp(min=1)