---
layout: post
title: Documents for github repo. im2txtViaGAN
---
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
github repo. link[(https://github.com/mazm13/im2txtViaGAN)](https://github.com/mazm13/im2txtViaGAN)

***TODO LIST***
* ~~begin document~~
* figure out inputs ops
* target seqs' information? difference with cross-entropy loss, think about how to use that information(e.g. make loss be the difference between their scores)
* sequence embedding in Discriminator
* word embedding in Generator, pretrained or unpretrained, trainable or untrainable
* noize in Generator, add it or not
* why this score is not differential by G's parameter

## Generator: G
First of all, we should figure out what G does in the framework following policy \\(\pi_{\theta}\\).

At each step, policy \\(\pi_{\theta}\\) takes conditions \\(f(I)\\), \\(z\\), which means image embedding vector and noise respecetively, and the preceding words \\(S_{1:t-1}\\) as inputs, then yields a distribution \\(\pi_{\theta}(w_t\|f(I),z,S_{1:t-1})\\) over the extended vocabulary. Besides, if no words are proceded yet, the state is doneted by \\(S_{1:0}\\) and begin indicator is BOS(begin of sentence), on the contrary, end indicator is doneted by EOS(end of sentence, while it is doneted by letter E in [^fn-ref-1]). So if we get \\(w_t=EOS\\) at step t, the sentence will be terminated, otherwize \\(w_t\\) will be appended to the end of the sentence. Reward of an action on the conditions of state is given by Discriminator D, \\(r=D(I, S_{1:t})\\).

So we already know what should a generator do at each step. Remeber inputs and outputs again:

**Inputs**(if we use patch, and path_size in conguration file is 32):
* Images(Encoded by inputs ops), a float32 Tensor with shape [batch_size, height, width, channels]
* input_seqs, a float32 Tensor with shape [batch_size, padded_length]

**Outputs**
* Actions \\(w_t\\)

Let's see the framework of it. At first, map image I into fixed length vector \\(f(I)\\) named image embedding later by utilizing inception v3 and a fully-connected layer. Make it be the first input of LSTM(a kind of RNN, to know more about LSTM, refer colah's blog:[http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)). 

At the same time, input_seq are imbedded into vector

### Image embedding
Map an image I to a vector \\(f(I)\\) with inception v3 which is a convolutional neural network and can be imported from TensorFlow library. We can get an inception v3 network by this:

{% highlight python linenos %}
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base

with tf.variable_scope(scope, "InceptionV3", [images]) as scope:
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=weights_regularizer,
      trainable=trainable):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):
      net, end_points = inception_v3_base(images, scope=scope)
      with tf.variable_scope("logits"):
        shape = net.get_shape()
        net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
        net = slim.dropout(
            net,
            keep_prob=dropout_keep_prob,
            is_training=is_inception_model_training,
            scope="dropout")
        net = slim.flatten(net, scope="flatten")
{% endhighlight%}

### Sequence embedding

Map a caption sentence into a vector(the same length as image embedding) named sequence embedding. As paper[^fn-ref-1], we utilize a LSTM network.

### Drafts

There are something about experiment. We train G by utilizing D's score, however this score is not differential by G's parameters \\(\theta\\). We need Policy Gradient(PG, a method in reinforcement learning) to compute gradient \\(\nabla_{\theta}\pi(\theta)\\) and update \\(\theta\\) by gradient descent(GD) or stochastic gradient descent(SGD). And we also need D's score to update D's parameters, it's important to note that **D's parameters are not updated every time when it evaluates a sentence.** 

#### Inputs

I want to discuss inputs in TensorFlow. In ops/Inputs.py

{% highlight python linenos %}
def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  if is_training:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=True, capacity=16, name=shard_queue_name)
    min_queue_examples = values_per_shard * input_queue_capacity_factor
    capacity = min_queue_examples + 100 * batch_size
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_queue_examples,
        dtypes=[tf.string],
        name="random_" + value_queue_name)
  else:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard + 3 * batch_size
    values_queue = tf.FIFOQueue(
        capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    _, value = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([value]))
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))
  tf.summary.scalar(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

  return values_queueiles.extended(tf.fgile.Glob(pattern))

{% endhighlight %}

#### Reference
[^fn-ref-1]: B. Dai, D. Lin, R. Urtasun, S. Fidler, "Towards Diverse and Natural Image Descriptions via a Conditional GAN", CoRR, vol. abs/1703.06029, 2017. [online]. Available:[https://arxiv.org/abs/1703.06029](https://arxiv.org/abs/1703.06029)

