---
layout: post
title: Holiday Paper Reading 2
---

## OBJ2TEXT

[1] Xuwang Yin, Vicente Ordonez(2017): OBJ2TEXT: Generating Visually Descriptive Language from Object Layouts. *CoRR, abs/1707.07102*, [https://arxiv.org/abs/1707.07102](https://arxiv.org/abs/1707.07102)

这篇文章是我最近在arXiv上看到的关于Image Caption的文章，这篇文章和我之前看的另外一篇文章所做的内容非常相似——都是利用detection辅助主流Image Caption方法。另外一篇文章：

Zhongliang Yang, Yu-Jin Zhang, Sadaqat ur Rehman & Yongfeng Huang(2017): Image Captioning with Object Detection and Localization. *CoRR, abs/1706.02430*, [https://arxiv.org/abs/1706.02430](https://arxiv.org/abs/1706.02430)，更多内容参见Post: [Reading Notes](/2017/06/21/Reading-Notes/)

这种想法比较复古。正如我之前所讲到的，人类对于图像进行标注时，也会按照大致的思路，先进行检测，再根据目标的位置属性以及场景进行描述。在show-and-tell之前遵从类似思路的方法性能大都很差，主要原因还是他们的语言模型部分采用的是模板，而不是一个可训练的语言模型。待会会讲到两者之前的区别。

首先讲一下这篇文章所用到的方法。这篇文章的贡献共有三个方面：
* 证明了将目标层作为一个序列运用LSTM去编码，模型仍能为标注任务有效地捕捉空间信息；
* 展示了一个依赖于目标标记而非像素数据的模型，在图像标注任务上不俗的竞争力，尽管这项任务本身具有歧义性；
* 将本文的OBJ2TEXT模型与标准的图像标注模型相结合可以生成更好的标注。

这篇文章要解决的问题是：描述目标层（Object layouts），这里输入场景中的所有物体的种类和位置已知。目标层通常被用来做story-boarding，sketching还有computer graphics application。这里用来提升现有的Image Caption模型。目标层包含丰富的语义信息，但是这些信息比较abstract，抽象或者精简，滤去了诸如颜色、文本、外观之类的其他信息，所以和传统的Image Captioning任务比较起来，会有更多的挑战。

![](/images/2017-07-25-1.png)

这篇文章推出了OBJ2TEXT，是一个seq2seq的模型，通过一个基于LSTM的神经语言模型将目标层进行编码。整个神经语言生成系统包含两步：第一步决定图像上哪些内容会被用来生成文本，第二部运用结构化的语言性质连接这些概念concepts。在这篇文章的模型OBJ2TEXT中，第一步被作为encoder，第二部被作为decoder。训练的数据系是标准的MS-COCO数据集，实际上这个数据集同时包含detection和image caption的数据，并且之前的工作都是致力于其中一个任务的，而本文推出的方法是第一个去训练object annotations和文本描述之间的映射的。

### Model
#### OBJ2TEXT

OBJ2TEXT是一个seq2seq的模型，将目标层作为一个序列来输入，然后预测每次输出单词从而生成文本。训练数据集是$N$和观察结果$\\{\\langle o^{(n)},l^{(n)}\\rangle\\}$，$\\langle o^{(n)},l^{(n)}\\rangle$是一张图像中物体种类和位置的序列，它与目标标注${s^{(n)}}$相关联。encoder与decoder被训练最小化下面这个损失函数，所用的优化方法是随机梯度下降法

\\[W^*=arg\min_{W}\sum\limits_{n=1}^{N}\mathcal{L}(\langle o^{(n)},l^{(n)}\rangle,s^{(n)})\\]

这里$W$是整个系统的参数。而损失函数为负log likehood。

\\[\mathcal{L}(\langle o^{(n)},l^{(n)}\rangle,s^{(n)})=-\log p(s^n\|h_L^n,W_2)\\]

这里$h_L^n$由encoder计算得到。

##### Encoder

encoder每次输入一个$\langle o_t,l_t\rangle$，即目标类别-位置对。$o^t$是大小为V的one-hot向量，而$l^t$则包含了目标的左边框、上边框以及目标的宽度和高度，坐标会被正则化到$[0,1]$范围之内。然后$o^t$和$l^t$会通过下式转化为一个$k$维的向量$x_t$，作为$t$步的输入：

\\[x_t=W_oo_t+(W_ll_t+b_l),x_t\in\mathbb{R}^k\\]

设置LSTM的初始状态和初始输入为0，每次输入$x_t$，然后将最后的隐藏状态$h_L$作为最终的编码结果，并用来生成标注$s$。

后面的Decoder和之前的是一样的。在此不做赘述。

#### OBJ2TEXT-YOLO

利用YOLO将输入图片转化为目标层，然后利用上述方法生成标注。

#### OBJ2TEXT-YOLO+CNN-RNN

这是一个结合模型，将输入的图片一方面利用CNN转化为固定长度的向量，另一方面利用上文中的方法生成向量$x_t$，然后将两者做一个拼接作为语言模型的输入。

![](/images/2017-07-25-2.png)

### RESULTS

和CNN-RNN比较起来，有10%左右的提升。比如C5的CIDEr的分数在0.932，和CNN-RNN的0.865相比有8%的提升，但是这个分数在mscoco-leaderboard上不算高，现在最高的在1.123。这篇文章所使用的框架和之前的一篇文章相比差别很大，可以说是完全不一样。之前那篇文章对目标的处理和位置的处理有很大的不一样，并且新增了一个“attention”机制。

另外，这篇文章有源代码。[http://www.cs.virginia.edu/~xy4cm/obj2text/](http://www.cs.virginia.edu/~xy4cm/obj2text/)，上面有demo和源码，torch平台上在neuraltalk2的基础上改的。如果在tensorflow上拿im2txt改的话，也挺容易的。

## 数值最优化算法与理论

这本书开始看了，这部分涉及到很多数学证明的东西。打算往过看，并且做后面的习题。