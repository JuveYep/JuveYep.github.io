---
layout:       post
title:        "Self Attention"
author:       "Ke"
header-style: text
catalog:      true
tags:
    - Transformer
---
>self-attention
# Attention is All You Need [[paper]](https://arxiv.org/abs/1706.03762),[[code]](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)

$\mathnormal{Q,K,V}$由同一个值$F\in\mathcal{R}^{N\times d}$经过线性变换得到,$Q:query,K:key,V:value$.