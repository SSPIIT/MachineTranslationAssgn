"""
inference.py - Greedy Decoding and BLEU Evaluation
DA6401 Assignment 3
"""

import torch
import evaluate as hf_evaluate


def greedy_decode(model, src, src_vocab, tgt_vocab, device, max_len=100):
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model:     trained Transformer
        src:       (1, S_src) source token indices
        src_vocab: source Vocabulary
        tgt_vocab: target Vocabulary
        device:    torch device
        max_len:   maximum output length

    Returns:
        List[int] – predicted token indices (excluding <sos>/<eos>)
    """
    model.eval()
    src = src.to(device)

    with torch.no_grad():
        src_mask = model.make_src_mask(src, src_vocab.pad_idx)
        enc_output = model.encoder(src, src_mask)

        # Start with <sos>
        tgt_ids = [tgt_vocab.sos_idx]

        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=device)
            tgt_mask = model.make_tgt_mask(tgt_tensor, tgt_vocab.pad_idx)

            dec_output = model.decoder(tgt_tensor, enc_output, tgt_mask, src_mask)
            logits = model.projection(dec_output)  # (1, T, V)

            next_token = logits[0, -1, :].argmax(-1).item()
            tgt_ids.append(next_token)

            if next_token == tgt_vocab.eos_idx:
                break

    # Strip <sos> and <eos>
    return tgt_ids[1:-1]


def translate_sentence(model, sentence, de_tokenizer, src_vocab, tgt_vocab,
                        device, max_len=100):
    """
    Translate a single German sentence string to English.
    """
    tokens = de_tokenizer(sentence)[:max_len - 2]
    src_ids = [src_vocab.sos_idx] + src_vocab.encode(tokens) + [src_vocab.eos_idx]
    src_tensor = torch.tensor([src_ids], dtype=torch.long)

    pred_ids = greedy_decode(model, src_tensor, src_vocab, tgt_vocab, device, max_len)
    return ' '.join(tgt_vocab.decode(pred_ids))


def evaluate_bleu(model, data_loader, src_vocab, tgt_vocab, device, max_len=100):
    """
    Compute corpus-level BLEU score on a DataLoader.

    Returns:
        bleu_score (float, 0-100)
    """
    bleu_metric = hf_evaluate.load("bleu")
    model.eval()

    predictions = []
    references  = []

    with torch.no_grad():
        for src, tgt in data_loader:
            for i in range(src.size(0)):
                src_i = src[i].unsqueeze(0)
                pred_ids = greedy_decode(model, src_i, src_vocab, tgt_vocab,
                                         device, max_len)
                pred_tokens = tgt_vocab.decode(pred_ids)

                # Reference: strip <sos>, <eos>, <pad>
                ref_ids = tgt[i].tolist()
                ref_tokens = [
                    tgt_vocab.idx2token[t] for t in ref_ids
                    if t not in (tgt_vocab.sos_idx, tgt_vocab.eos_idx, tgt_vocab.pad_idx)
                ]

                pred_tokens = [t for t in pred_tokens if t not in ["<sos>", "<eos>", "<pad>"]]
                ref_tokens = [t for t in ref_tokens if t not in ["<sos>", "<eos>", "<pad>"]]

                predictions.append(" ".join(pred_tokens))
                references.append([" ".join(ref_tokens)])

    result = bleu_metric.compute(predictions=predictions, references=references)
    return result['bleu'] * 100