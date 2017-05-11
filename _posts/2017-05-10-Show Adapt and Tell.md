---
layout: post
title: Show, Adapt and Tell
---


Show, Adapt and Tell: Adversarial Training of Cross-domain Image Captioner[[arvix:1705.00930]](http://arvix.org/pdf/1705.00930.pdf)


## Notes
从文章题目来看，这篇文章是通过GAN来解决image caption中的“跨领域”问题。文章题目模仿了比较经典的image caption的文章“show and tell”，通过”adapt“来表明这篇文章的主要目的是通过对原领域的训练，使之适应新的领域。当然，我们在这里先搞清楚，什么是“跨领域”问题，实际上在文中值得是标注风格的迁移，更具体来讲，是在不同数据集上的迁移。不同的数据集它们的标注风格实际上是有明显的不同的，如文章中所说，mscoco中的图片场景更为宏观，拥有多个物体，而且标注会主要描述物体的位置、大小、颜色等，而cub-200包含了各种鸟的图片，标注会更加注重描述鸟的细节部分。

问题来由：一些领域数据集不够大，不够支撑现有的image caption的方法。训练数据：source domain pairs:(x, y), target domain sentences: y

主要思路：参考现有的image caption gan的方法，在判别模型上做文章，判别模型由两个部分构成：domain cirtic：判断生成的句子是否符合target domain；mutil-model cirtic：判断生成的句子是否符合图片的内容。将两者结合起来作为生成模型的回报，然后采用polocy gradient求梯度，然后update生成模型。

具体：

captioner: 采用show-and-tell这个经典的cnn-rnn模型

domain-cirtic：A sentence y is first encoded by CNN [18] with highway connection [19] into a sentence representation. Then, we pass the representation through a fully connected layer and a softmax layer to generate probability. 句子会被分成三类：source, target, generated. 分别得到这三类的概率。

multi-modal critic: 句子y会被lstm转化为向量c，然后通过全连接层和tanh层得到值a，对应的，转码后的图片（这里encoded存疑，是否值得是image embeddings）通过全连接层和tanh层得到值b；将a和b做元素间相乘，结果做softmax之后即生成y和x分别为paired, unpaired, generated的概率。

reward: 将生成的句子为target domain的概率和句子与图片paired的概率相乘

训练方法：前面将判别模型弄成了分类问题，所以训练方法为监督学习的方法


## Thinking
这篇文章很好的解决了cross-domain的问题，但是对于简单的image caption问题来讲，该文章的方法与普通的通过gan来实现image caption的方法本质上还是一样的。尽管判别模型转化成了分类问题。


## 实验部分
自己的实验部分：以及搭好了数据预处理部分（将输入整理好之后放入队列中）、判别模型部分（采用toward那篇文章的方法），部分生成模型
欠缺的：policy gradien部分，这部分比较难写，尤其是做mc-rollout的时候需要另一个生成模型

一些问题：生成模型中噪声的引入是否是必要的？


