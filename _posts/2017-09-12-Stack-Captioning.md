---
layout: post
title: Stack-Captioning
---

本周所做的工作：
* 实验进入训练阶段，并且已经实现了模型的inference部分[https://github.com/mazm13/attim2txt](https://github.com/mazm13/attim2txt)
* 读了2篇论文

### Experiment

[TensorBoard](http://219.223.173.255:6006)

目前已经完成了代码的编写，但是训练的效果没有达到预期水平，loss抖动太过剧烈且仍然在一个较大值附近波动，几个可能的原因：1.模型的搭建上出了问题（可能性较小）；2.VGGnet的代码使用的不太对；3.训练方法不对，出现抖动可能是因为训练时学习速率太大（目前已经改到一个较小的值）

### Stack-Captioning: Coarse-to-Fine Learning for Image Captioning

[1] Jiuxiang Gu, Jianfei Cai, Gang Wang, Tsuhan Chen."Stack-Captioning: Coarse-to-Fine Learning for Image Captioning." *CoRR, abs/1709.03376*, [https://arxiv.org/abs/1709.03376](https://arxiv.org/abs1709.03376)

这篇文章对于single-stage的decoder部分进行了改进，改成了multi-stage，并且解决了由于multi-stage造成的诸多困难：难以训练、梯度消失。这篇文章的想法是这样的，使用一个多级的语言模型，首先在第一级生成一个粗糙的caption，然后在此基础上通过attention的机制生成更加细致的caption，层层递进，最终生成一个细致丰富的caption。每级生成的结果会作为下一级的先验或者说是条件。

这篇文章同样也给出了目前的image caption方法（最大似然估计方法）的3个主要问题及原因：首先，它很难去生成丰富细致的描述，这是因为丰富的描述需要更为复杂的模型去生成，而复杂的模型经常会出现梯度消失问题（与深层网络的梯度消失问题类似）；其次，exposure bias；还有就是交叉熵无法保证句子的有效程度。举出这些问题是为了给自己的模型做铺垫的，那么文章会推出一个更为复杂的模型，并解决了梯度消失问题，解决了exposure bias问题，并且把一些evalution metircs作为reword。（和之前一篇拿RL做image caption的是一个套路）

coarse-to-fine multi-stage prediction framework. 见图。

这篇文章主要贡献：1）一个可以逐渐提高标注质量的框架（refined attention weights）2）归一化的intermediate rewards进行优化的增强学习的方法。




