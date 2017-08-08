---
layout: post
title: Holiday Paper Reading 3
---

这是holiday reading的最后一次了，明天就要奔赴深圳继续自己的研究生生涯了。本次带来的内容是OBJ2TEXT的实现以及新的一篇论文的阅读。

## OBJ2TEXT

[1] Xuwang Yin, Vicente Ordonez(2017): OBJ2TEXT: Generating Visually Descriptive Language from Object Layouts. *CoRR, abs/1707.07102*, [https://arxiv.org/abs/1707.07102](https://arxiv.org/abs/1707.07102)，
更多内容参见上一篇Post: [Holiday Paper Reading 2](/2017/07/25/Holiday-Paper-Reading-2/)。文章的demo和源代码参见：[http://www.cs.virginia.edu/~xy4cm/obj2text/](http://www.cs.virginia.edu/~xy4cm/obj2text/)。另外从github上可以看到，这篇文章还有[pyTorch](https://github.com/yunjey/pytorch-tutorial)的实现版本。目前正在学习pytorch以及在看代码。另一方面，我也在看有关于detection的[yolo(you only look once)](https://pjreddie.com/darknet/yolo/)的部分。打算自己用tensorflow实现以下yolo的部分，并且将训练好的参数打包给image caption部分使用，查看最后的训练结果。最后将yolo的参数设置为可训练，进行微调，将调整之后的模型与之前的做一下对比。看训练出来的yolo长什么样。

## Bottom-Up and Top-Down Attention for Image Captioning and VQA

[2] Peter Anderson, Xiaodong He et.al.(2017): Bottom-Up and Top-Down Attention for Image Captioning and VQA. *CoRR, abs/1707.07998*, [https://arxiv.org/abs/1707.07998](https://arxiv.org/abs/1707.07998)

这篇文章的题目比较饶人，但是其训练结果非常吸引人，ms coco leaderboard榜单之首！

首先我们看一下top-down和bottom-up之间的区别，top-down指的是自上而下，而bottom-up是自下而上。这里对于image caption和VQA任务来说，top-down指的是non-visual, task-specific context，而bottom-up指的是purely visual feed-forward attention mechanisms。

下面我们来看一下具体的方法吧。

### APPROACH

对于一张图片$I$，这里的image captioning模型和VQA模型都会输入这张图片的features，$V=\\{v_1,v_2,\cdots,v_k\\},v_i\in\mathbb{R}^D$，这里的$k$是可变大小的。而$V$由我们的自下而上的注意力机制模型决定，或按照一般的方法输入CNN的最后一层。

#### Bottom-Up Attention Model

这里使用的是Faster R-CNN，Faster R-CNN检测目标分两步。第一步：

## 数值最优化算法与理论

这本书开始看了，这部分涉及到很多数学证明的东西。打算往过看，并且做后面的习题。