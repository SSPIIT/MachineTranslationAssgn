"""
evaluate_checkpoint.py - Evaluate a saved checkpoint on the test set
DA6401 Assignment 3

Usage:
    python evaluate_checkpoint.py --checkpoint checkpoints/transformer_base_best.pt
"""

import argparse
import torch
from model import Transformer, TransformerLearnedPE
from dataset import build_dataloaders
from inference import evaluate_bleu, translate_sentence


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt['args']
    src_vocab = ckpt['src_vocab']
    tgt_vocab = ckpt['tgt_vocab']

    ModelClass = TransformerLearnedPE if cfg.get('learned_pe', False) else Transformer
    model = ModelClass(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg.get('d_model', 256),
        num_layers=cfg.get('num_layers', 3),
        num_heads=cfg.get('num_heads', 8),
        d_ff=cfg.get('d_ff', 512),
        dropout=0.0,  # no dropout at test time
        max_len=cfg.get('max_len', 256)
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded. Epoch: {ckpt.get('epoch', '?')} | Val BLEU: {ckpt.get('val_bleu', '?'):.2f}")

    # Re-build test loader only
    _, _, test_loader, _, _ = build_dataloaders(
        batch_size=args.batch_size,
        max_len=cfg.get('max_len', 256)
    )

    print("Computing test BLEU...")
    test_bleu = evaluate_bleu(model, test_loader, src_vocab, tgt_vocab,
                               device, max_len=cfg.get('max_len', 256))
    print(f"\nTest BLEU: {test_bleu:.2f}")

    # Sample translations
    from dataset import SpacyTokenizer
    de_tokenizer = SpacyTokenizer('de_core_news_sm')
    samples = [
        "Ein Mann sitzt auf einer Bank.",
        "Zwei Hunde spielen im Schnee.",
        "Eine Frau liest ein Buch.",
        "Kinder spielen im Park.",
        "Ein Auto fährt durch die Stadt.",
    ]
    print("\nSample Translations:")
    print(f"{'German':<50} {'English (predicted)'}")
    print('-' * 90)
    for de in samples:
        en = translate_sentence(model, de, de_tokenizer, src_vocab, tgt_vocab, device)
        print(f"{de:<50} {en}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to saved .pt checkpoint')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    main(args)