---
layout: post
title: documents for github repo. im2txtViaGAN
---

github repo. link[(https://github.com/mazm13/im2txtViaGAN)](https://github.com/mazm13/im2txtViaGAN)

***TODO LIST***
* ~~begin document~~
* figure out inputs ops
* sequence embedding in Discriminator
* word embedding in Generator, pretrained or unpretrained, trainable or untrainable
* noize in Generator, add it or not

## Generator: G
First of all, we should figure out what G does in the framework following policy \pi_{\theta}.

At each step, policy \pi_{\theta} takes conditions f(I), z, which means image embedding vector and noise respecetively, and the preceding words S_{1:t-1} as inputs, then yields a distribution \pi_{\theta}(w_t\|f(I),z,S_{1:t-1}) over the extended vocabulary. Besides, if no words are proceded yet, the state is doneted by S_{1:0} and begin indicator is BOS(begin of sentence), on the contrary, end indicator is doneted by EOS(end of sentence, while it is doneted by letter e in [1]). So if we get w_t=EOS at step t, the sentence will be terminated, otherwize w_t will be appended to the end of the sentence. Reward of an action on the conditions of state is given by Discriminator D, r=D(I, S_{1:t}).

So we already know what should a generator do at each step. Remeber inputs and outputs again:

**Inputs**(if we use patch, and path_size in conguration file is 32):
* Images(Encoded by inputs ops), a float32 Tensor with shape [batch_size, height, width, channels]
* input_seqs, a float32 Tensor with shape [batch_size, padded_length]

**Outputs**
* Actions w_t

Let's see the framework of it. At first, map image I into fixed length vector f(I) named image embedding later by utilizing inception v3 and a fully-connected layer. Make it be the first input of LSTM(a kind of RNN). 

At the same time, input_seq are imbedded into vector

### image embedding
Map an image I to a vector f(I) with inception v3 which is a convolutional neural network and can be imported from TensorFlow library. We can get an inception v3 network by this:

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

