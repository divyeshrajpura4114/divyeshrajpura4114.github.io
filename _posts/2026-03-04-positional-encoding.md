---
title: Positional Encodings in Transformers – Types and Comparison
date: 2026-03-04 10:00:00 +0530
categories: [Transformers, PositionalEncoding, LLM]
tags: [Transformers, PositionalEncoding, LLM]
author: Divyesh Rajpura
---

## Introduction

Imagine reading a book where every word has been cut out and tossed into a hat. You still have all the words, but the story is gone. This is exactly how a Transformer "sees" language by default.

Unlike Recurrent Neural Networks (RNNs), which process text word-by-word (like a human reading left-to-right), or Convolutional Neural Networks (CNNs), which look at local chunks, Transformers process the entire sequence simultaneously. This makes them incredibly fast, but it leaves them with a peculiar form of amnesia: they have no inherent sense of word order.

To fix this, we use **Positional Encodings**. These are essentially "positonal details" injected into each word so the model knows not just what the word is, but where it sits in the sentence.

Consider these two sentences:
- Dog bites man
- Man bites dog

To a raw Transformer (without Positional Encodings), these sentences are identical because they contain the same tokens. However, they convey significant difference in meaning. Positional encodings ensure the model treats these as distinct structural sequences.

---

## Types of Positional Encodings

### 1. Sinusoidal Positional Encoding
### 2. Learned Positional Embeddings
### 3. Relative Positional Encoding
### 4. Rotary Positional Embeddings (RoPE)
### 5. ALiBi (Attention with Linear Biases)

---

## Comparison of Positional Encoding Methods

| Method | Parameters | Handles Long Context | Used In |
|------|------|------|------|
| Sinusoidal | No | Good | Original Transformer |
| Learned | Yes | Limited | BERT |
| Relative | Few | Good | T5, Transformer-XL |
| RoPE | No | Very Good | LLaMA, GPT-NeoX |
| ALiBi | No | Excellent | Long-context LLMs |

---

## References
- [Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. 2017. Attention is All You Need. In Advances in Neural Information Processing Systems (NeurIPS).](https://arxiv.org/abs/1706.03762)
- [Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.](https://arxiv.org/abs/1810.04805)
- [Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Self-Attention with Relative Position Representations.](https://arxiv.org/abs/1803.02155)
- [Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. RoFormer: Enhanced Transformer with Rotary Position Embedding.](https://arxiv.org/abs/2104.09864)
- [Ofir Press, Noah A Smith, and Mike Lewis. 2021. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.](https://arxiv.org/abs/2108.12409)
