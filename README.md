# DA6401 Assignment 3 — Transformer for Machine Translation

This project implements the Transformer architecture from the paper **“Attention Is All You Need”** using PyTorch for German-to-English Neural Machine Translation on the Multi30k dataset.

## Features

- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
- Transformer Encoder-Decoder Architecture
- Noam Learning Rate Scheduler
- Label Smoothing
- Greedy Decoding
- BLEU Score Evaluation
- Weights & Biases Experiment Tracking

## Experiments Performed

- Noam Scheduler vs Fixed Learning Rate
- Scaling Factor Ablation
- Sinusoidal vs Learned Positional Encoding
- Effect of Label Smoothing
- Attention Visualization

## W&B Report

Public W&B report link:

https://api.wandb.ai/links/sspiit-iitm-india/5hoccm7x

## Dataset

- Multi30k Dataset

## Libraries Used

- torch
- numpy
- matplotlib
- scikit-learn
- wandb
- datasets
- spacy
- tqdm

## Running the Project

### Train Model

```bash
python train.py