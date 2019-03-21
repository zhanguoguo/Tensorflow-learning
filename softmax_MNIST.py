#%%
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

import tensorflow as tf
sess = tf.InteractiveSession()
# 创建一个placeholder，即输入数据的地方，第一个参数是数据类型，第二个参数是tensor的shape，None代表不限尺寸
x = tf.placeholder(tf.float32, [None, 784])
# 给softmax Regression模型中的weights和bias创建一个Variable对象，variable是用来存储模型参数的
# 不同于存储数据的tensor一旦使用就会消失，但variable是持久化的，长期存储，迭代更新
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# tf.matmul是Tensorflow中的矩阵乘法函数

# 用交叉熵来定义损失函数
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
# 定义一个优化算法，采用随机梯度下降算法，来更新参数减小loss，优化目标为cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 采用tensorflow的全局参数初始化器tf.global_variables_initializer,并直接执行run方法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()
# 最后，开始迭代的执行训练操作train_step，每次随机从训练样本中抽取100个样本构成一个mini-batch，
# 并feed给placeholder,然后调用traain_step对这些样本进行训练。
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})
# 用tf.argmax(y,1)是求各个预测的数字中概率最大的一个
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 用tf.cast将之前correct_prediction输出的bool值转换为float32
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))