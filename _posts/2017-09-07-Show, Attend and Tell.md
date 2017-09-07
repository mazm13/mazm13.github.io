---
layout: post
title: Show, Attend and Tell
---

这两周所做的工作：
* Faster RCNN的基础上，实现原论文的方法（Faster RCNN的代码不太好用）
* 用一种简单的方法获取图片的features，目前的做法是取VGG16的pool5之后得到的结果$[7\times7\times512]$，展开之后为$[49\times512]$，作为49个features.
* NIC代码中LSTM的修改（按原论文的方式）
* 以上两项是正在实现过程中的[https://github.com/mazm13/attim2txt](https://github.com/mazm13/attim2txt)
* 读了2篇论文

### Show, Attend and Tell

[1] Show, Attend and Tell: Neural Image Caption Generation with Visual Attention *CoRR, abs/1502.03044*, [https://arxiv.org/abs/1502.03044](https://arxiv.org/abs/1502.03044)

*Show, Attend and Tell*是一篇比较经典的Image Captioning的文章，在原来的*Show and Tell*的基础上，新增了Attention机制。这篇文章所做的主要贡献如下：
* 推出了2个不同的基于*Show and Tell*框架的visual attention模型：1) "soft"通过反向传播方法训练的确定注意力机制；2) "hard"同REINFORCE训练的随机注意力机制；
* 它们这种注意力是可视化的；
* 在Flickr8k, Flickr30k和MS COCO这3个不同的数据集上做了测试，并与其它方法进行比较。

#### Framework

我们直接看整个模型的框架。上面列举了2中不同的注意力机制，但是在整个模型中只是feature权重的计算方式不同而已，即文中所说的两者的区别在于$\phi$函数不同，这会在后面讲到。模型的输入是一张图片，输出是生成的caption $\boldsymbol{y}$，$y = y_1, y_2, \cdots, y_C, y_i\in\mathbb{R}^K$。这里$K$是单词表的大小，$C$是序列长度。通过卷及神经网络提取出$L$个$D$维的feature向量。记为
$$a=\{a_1, \cdots, a_L\}, a_i\in\mathbb{R}^D$$。
这里为了使提取出的向量与原图片中的位置有更好的相关性，这里提取的方法取底层卷积层的特征。

语言模型部分仍然使用LSTM，取$T_{s,t}:\mathbb{R}^s\rightarrow\mathbb{R}^t$为$s$维到$t$维的仿射变换，为可学习变量。则
![](/images/2017-09-07-1.png)
实际上与原来的LSTM相比，只是输入不同，原LSTM的输入是$Ey_{t-1}$，而这里的输入可以认为是$Ey_{t-1}$和$\hat{z_t}$的拼接。其中$\hat{z_t}$为上下文向量，为特征的线性组合，其计算方式如下：
\\[e_{ti}=f_{att}(a_i,h_{t-1})\\]
\\[\alpha_{ti}=\frac{ exp(e_{ti}) }{\sum_{k=1}^{L}exp(e_{tk})}\\]
$a_i$的权重与LSTM前一时刻的状态$h_{t-1}$有关，通过注意力计算函数$f_{att}$结果做softmax得到。另外要注意的是LSTM状态的初始化，记忆状态和隐含状态分别用两个多层感知机对$a_i$的均值做一个转化。

我们需要知道的是$\boldsymbol{a}$和$f_{att}$分别如何得到。

#### 特征获取

经imageNet预训练过的VGGnet，第四层卷积层之后maxpooling前的$14\times14\times512$的feature map，提取为$196\times512$的特征。这个和我之前的设想非常相近，我取得是maxpooling之后的$7\times7\times512=49\times512$的feature。

#### 注意力机制

这里有两个不同方案的注意力机制，一个称为hard的随机注意力机制，即在$L$个feature中通过某种方法选择一个feature出来，作为此时$\hat{z_t}$，另一种是称之为soft的随即注意力机制，与前者不同，它是对各个feature进行加权求和或者认为是求出一个期望feature，这种方法更加smooth。

#### 总结

现有的image caption方法，主要包含这几部分：
* 图像信息提取
* 注意力所需的特征提取（optional）
* 解析生成单词
现有的方法也都是在这些方面做着改进。就本文而言特征提取是在CNN底层获取，而注意力机制是通过将特征加权注入LSTM的输入得到的，想法比较简单。但是在训练方面，尤其是Stochatic hard这一部分通过Reinforce进行训练十分具有借鉴意义。作为对比，BUTD这篇文章特征提取是通过Faster RCNN进行目标检测，然后进行CNN卷积得到比如$36\times2048$的特征，也就是说可能BUTD的特征选取更加具有目标性，另外，BUTD的解析部分也更加高级。

我目前的实现情况：特征提取为VGGnet的第5层池化层之后的$7\times7\times512$的feature map，Decoder部分采用的是BUTD的双层LSTM。


