---
layout: post
title: Reading Notes
---

### Paper Reading

[1] Satoshi Tsutsui, David Crandall(2017): Using Artificial Tokens to Control Languages for Multilingual Image Caption Generation. *CoRR, abs/1706.06275*, [https://arxiv.org/abs/1706.06275](https://arxiv.org/abs/1706.06275)

训练一个模型，能够生成多种语言的标注。

在多语言的图像标注任务中（如英文或日文）用人工符号在控制语言。整个文章的唯一亮点就是：“在起始符上下文章，不同的语言用不同的起始符，从而起到控制语言的作用”。具体来说，文章中所用到的模型是为Show and Tell模型，通过最大似然法进行训练。而在起始符的取值上，对于英文取“<en>”，日文取“<jp>”，而不是像原来那样取“<sos>”。整个模型中，日文标注和英文标注是混杂的。所以整个单词表的大小为英文单词个数+日文单词个数。为什么选择日文呢？为了验证P这种方法的有效性，取了两种差别比较大的自然语言（P.S. 本文作者之一是个日本人）。

实验部分：数据集采用YJ Caption 26k Dataset，是在MSCOCO数据集的基础上取了一个子集做了人工的日文标注，共包含26,500张图片及其日文标注，同时在MSCOCO中取这些图片的英文标注，构成一个完整的实验的数据集。其中22,500用于训练，2,000用于验证，2,000用于测试。

实验结果：混合模型的分数都比单个语言的模型要差。

感觉并没有什么用。模型还是依赖特定语言的数据集，比如要生成中文的标注就得需要中文的数据集，以及构建中文单词表，并且加以训练。文中所用的方法不如单独训练简单一点儿。如果要做语言迁移的话，一种更加现成的方法是生成英文的标注然后通过机器翻译进行翻译。

[2] Marcella Cornia, Lorenzo Baraldi, Giuseppe Serra, Rita Cucchiara(2017): UPaying More Attention to Saliency: Image Captioning with Saliency and Context Attention. *CoRR, abs/1706.08474*, [https://arxiv.org/abs/1706.08474](https://arxiv.org/abs/1706.08474)

通过显著性检测来辅助标注生成。之前没有文章是这么做的，之前的文章要么是做目标检测，要不什么都不做。

### Computer Vision Course

整体上看一下内容，前面的内容是不包括深度学习方法，如颜色和光照模型、2D特征、形状和轮廓等等，后面的内容主要讲的还是深度学习方法，也包括了CNN、RNN、LSTM等内容，非常贴切。里面的内容我也看了一下（RNN和LSTM这一章），讲得非常的细致。我觉得有必要静下心来好好补充一下这些基础知识。

### Deep Learning & Reinforcement Learning Book

**no more**