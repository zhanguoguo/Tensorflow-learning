import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 在自编码中，会使用到一种参数初始化Xavier initialization
# 它所做的事就是让权重被初始化为不大不小正好合适的数，0均值，2/(n_in+n_out)方差的分布
# 创建一个（-sqrt(6/(n_in+n_out)),sqrt(6/(n_in+n_out))）范围类的均匀分布

# 实现标准的均匀分布的Xaiver初始化器，其中fan_in是输入节点的个数，fan_out是输出节点的个数
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant*np.sqrt(6.0/(fan_in+fan_out))
    high = constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(),scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

    # 定义网络结构，为输入x创建一个维度为n_input的placeholder
    # 然后建立一个能提取特征的隐藏层，输入是加了噪声的
        self.x = tf.placeholder(tf.float32,[None, self.n_input])
        # self.transfer 对结果进行激活函数的处理
        self.hidden = self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal((n_input,)),
                                                     self.weights['w1']),self.weights['b1']))
        # 隐含层后，需要对输出层进行数据复原，重建操作，
        # 这里不需要激活函数，直接将隐层的输出乘上输出成的权重w2，加上偏置
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

        #接下来定义自编码的损失函数，采用平方差作为cost，tf.substract计算输出与输入之差，用tf.pow求差的平方
        # 最后使用tf.reduce_sum求和即可得到平方误差

        # 定义优化器对损失cost进行优化，创建session，并初始化自编码器的全部模型参数
        self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        self.optimizer = optimizer.minimize((self.cost))

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def _initialize_weights(self):
        all_weights = dict()
        # w1需要使用前面定义的xavier_init函数初始化
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))

        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input],dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))

        return all_weights

    def partial_fit(self,X):
        # 计算损失cost及执行一步训练的函数
        # 函数让session执行计算两个图的节点，分别是损失和训练过程
        # 函数要做的就是用一个batch数据进行训练并返回当前的损失cost
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x:X, self.scale:self.training_scale})
        return cost

    # 定义一个只求损失cost的函数，这个函数是在自编码训练完毕后，在测试集上对模型性能进行评测时用到的
    def calc_total_cost(self, X):
        return self.sess.run(self.cost,
                             feed_dict={self.x:X, self.scale : self.training_scale})

    # 定义transform函数，返回自编码器隐含层的输出结果，自编码隐含层的主要功能就是学习出数据种的高阶特征
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x:X, self.scale : self.training_scale})


    # 定义generate函数，将隐含层的输出结果作为输入，通过之后的重建层将渠道的高阶特征复原为原始数据
    def generate(self,hidden = None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])

        return self.sess.run(self.reconstruction, feed_dict={self.hidden:hidden})

    # 定义reconstruction函数，将整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction, feed_dict={self.x:X, self.scale:self.training_scale})


    def getweights(self):
        return self.sess.run(self.weights['w1'])

    def getBias(self):
        return self.sess.run(self.weights['b1'])

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test

#定义随机获取block数据的函数，不放回抽样
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epochs = 30
batch_size = 128
display_step = 1

# 创建一个AGN自编码器，定义输入是784
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,
                                               n_hidden=200,
                                               transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale=0.01)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost/n_samples*batch_size


    if epoch % display_step == 0:
        print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))

print("Total cost:"+str(autoencoder.calc_total_cost(X_test)))