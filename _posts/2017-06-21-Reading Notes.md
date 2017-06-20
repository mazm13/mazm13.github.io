---
layout: post
title: Reading Notes
---
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

Zhongliang Yang, Yu-Jin Zhang, Sadaqat ur Rehman & Yongfeng Huang(2017): Image Captioning with Object Detection and Localization. *CoRR, abs/1706.02430*, [https://arxiv.org/abs/1706.02430](https://arxiv.org/abs/1706.02430)

清华大学电子系本月发的论文，关于Image Caption的。从其标题来看，是通过目标检测和定位来进行Image Caption，这种想法比较复古。之前我讲到最初的方法大都是这种思路，先进行目标检测，再根据目标的属性位置以及所进行的活动等信息进行标注。当然这种思路也比较自然，其实人对于一张图片进行自然语言时，也会对图片做类似的处理。首先识别出其中包含的各个物体（Object）尤其是人物，然后根据他们的位置关系、动作信息以及外貌细节来构建出对这张图片的描述。但是在Show and Tell之前的方法的效果并不好，主要原因还是语言的生成是基于模板的，而不是一个可训练的语言模型。

本文的模型也是Encoder-Decoder模型——Encoder将图片信息编码成一系列的目标检测和定位信息，Decoder利用这些信息生成自然语言。

### Encoder part

#### Object Detection

Fast R-CNN，由于其效率和效力。主要包含两部分：第一个模块是一个深度卷积神经网络对整个图片进行处理，得到一些特定的区域，这些特定的区域包含要处理的物体；第二个模块是Fast R-CNN目标检测器，分别对上一步得到的区域进行检测。为得到检测候选区，使用一个滑动窗口来对图片进行检测，最后取这些窗口中得分前n个区域（每个区域都是一个矩阵），每一个物体用一个d维的向量来表示\\(obj_i\\)。

#### Object Localization

对于每个矩形区域，在原图的基础上分别保留每个矩形区域之内（包含边框）的图片内容而将其他区域的值置为原图每个像素的均值，这样会得到n个大小与原图片一致而只保留选定区域的图片。将每张新得到的图片送入VGG网络，结果经“fc7”全连接层得到一个表示物体位置的向量（t维）。最终我们得到其他n个表示目标位置的向量\\(loc_i\\)。

最终我们将图片中的每一个物体\\(i\\)分别表示内容和位置的向量拼接起来，即得到注解向量\\(A_i=[obj_i;loc_i]\\)。这是Encoder要做的事情。

### Decoder part

关注机制的LSTM网络。

The key idea of attention mechanism is that when a sentence is used to describe an image, not every word in the sentence is "translated" from the whole image but actually it just has relation to a few subregions of an image. It can be viewed as a form of alignment from words of the sentence to subregions of the image. 

LSTM network products one word at every step \\(j\\) conditioned on a context vector \\(z_j\\), the previous hidden state \\(h_{j-1}\\) and the previously generated words \\(w_{j-1}\\).

与Show and Tell不同的是，这里新增了一个上下文向量\\(z_j\\)

关于\\(z_j\\)的计算：首先将LSTM前一个隐藏状态\\(h_{j-1}\\)和之前得到的注解向量\\(A_i\\)作为一个多层感知机的输入，得到输出\\(e_{ji}\\)，然后经过类似Softmax的方法使所有值加起来为1，得到系数\\(\\alpha_{ji}\\)，即为注释向量$A_i$的权值，最终得到
\\[z_j=\sum_{i=1}^{n}\\alpha_{ji}A_i\\]

### Training

只训练Decoder部分。损失函数目标单词概率的负log，另加对系数的正则化。就这么完了。。实现结果显示，他们的模型比Show and Tell的要好。但是没有跟最近一些对于ST进行优化的模型比较，如经强化学习改进的、经GAN改进的。也没有标注多样性的分析。


### 分析

比较依赖于目标检测的结果，如果一张图片（如风景图）本来就没有什么特定的目标，这样得到的注释向量就没有什么信息量。