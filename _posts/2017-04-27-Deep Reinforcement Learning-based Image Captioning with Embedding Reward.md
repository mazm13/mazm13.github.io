---
layout: post
title: Deep Reinforcement Learning-based Image Captioning with Embedding Reward
---

这篇是对最新关于image caption的文章的总结。和大部分state-of-the-art的方法大相径庭的是，这篇文章没有使用Encoder-Decoder框架，而是剑走偏锋受到增强学习中actor-critic方法的启发，提出了一种新的称之为“决策制定”(decision-making)的框架。并且据他们的实验结果显示，在各种评测标准中都达到了state-of-the-art的水平。本文从增强学习中的相关概念出发，来剖析这篇文章的方法思路以及可以带给我们的启发。

Zhou Ren, Xiaoyu Wang, Ning Zhang, Xutao Lv, Li-Jia Li. "**Deep Reinforcement Learning-based Image Captioning with Embedding Reward**"[[https://arxiv.org/abs/1704.03899]](https://arxiv.org/abs/1704.03899)

