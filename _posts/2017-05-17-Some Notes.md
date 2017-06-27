---
layout: post
title: Some Notes
---
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

本笔记是对Google开源的Show-and-Tell代码的学习的一些笔记，并对TensorFlow的一些用法以及训练方法的整理。

### 输入
读取数据到数据队列当中。在输入数据很大的情况下，实际上只要构建出合理的输入队列，我们就可以不用去操心数据读入的事情。TensorFlow支持多种的数据读入方式，一种比较简单的方式是使用placeholder，但是这需要在使用的过程中feed_data，并不适合大量数据的读入。另一种方式是使用二进制文件和输入队列的组合形式。

二进制文件，在代码中有一个独立于训练之外的代码，需要训练之前去生成的，生成TFRecord file，这样的话在训练的时候就可以使用TFRecordReader就可以读取该文件。

* pattern->文件名队列data_files
* tf.train.string_input_producer->filename_queue，这里可以设置shuffle=True均匀打乱文件名
* Reader读取文件得到数据，存储到values_queue
* tf.train.batch_join，从values_queue中读取到足够多的数据，生成batch数据

### LSTM部分

先讲讲代码中的lstm的操作

* 构建lstm_cell
* 构建零状态，设置batch_size
* 将image embedding作为输入，得到初始状态（这时有一个初始输出，没有什么用，或者默认第一个输出为BOS）
* 在构建输入输出的时候有一个input_seqs，实际上为完整的句子去掉最后一个单词，然后利用下面的函数

{% highlight python linenos %}
lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                    inputs=self.seq_embeddings,
                                    sequence_length=sequence_length,
                                    initial_state=initial_state,
                                    dtype=tf.float32,
                                    scope=lstm_scope)
{% endhighlight %}

### 实验

有以下问题
* 做sequence embedding，我看文章中是用LSTM做的，但是没有找到相关方法，我个人的想法是，lstm的初始输入为全零，但是每次将单词输入进去，然后将最后的输出作为sequence，不知道这个方法可行不？
* 生成模型一遍train，一遍inference