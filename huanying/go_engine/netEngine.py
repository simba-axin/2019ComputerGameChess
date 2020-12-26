# -*- coding: utf-8 -*-
# 此.py文件是利用什么神经网络来实现，
# 一下采用CNN来实现，
# 最后面有一种GAN神经网络处理的

###解决AUX的过程###
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
###解决AUX的过程###
import os
import sys
import logging
import math
import tensorflow as tf
import go_engine.features as features
import go_engine.go as go
import utils.go_utils as utils

logger = logging.getLogger(__name__)
sys.path.append("..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"#屏蔽通知信息、警告信息和报错信（INFO\WARNING\FATAL）


class PolicyNetWork():
    def __init__(self, features=features.DEFAULT_FEATURES, use_cpu=False):
        self.num_input_planes = sum(f.planes for f in features)
        self.features = features
        # self.session = tf.Session()
        # 更新语法
        self.session = tf.compat.v1.Session()
        self.test_summary_writer = None
        self.training_summary_writer = None
        self.test_stats = StatisticsCollector()  # 测试统计
        self.training_stats = StatisticsCollector()  # 训练统计
        if use_cpu:
            with tf.device("/cpu:0"):
                self.set_up_network()
        else:
            self.set_up_network()

    def set_up_network(self):

        global_step = tf.Variable(0, name="global_step", trainable=False)
        '''
        x = tf.placeholder(tf.float32, [None, go.N, go.N, self.num_input_planes])
        y = tf.placeholder(tf.float32, [None, go.N ** 2])  # go.N ** 2
        '''
        # 更新语法
        x = tf.compat.v1.placeholder(tf.float32, [None, go.N, go.N, self.num_input_planes])
        y = tf.compat.v1.placeholder(tf.float32, [None, go.N ** 2])  # go.N ** 2
        # # 方便初始化权重（weights）和偏差（biases）的函数
        # def _weight_variable(shape, name):
        #     number_inputs_added = utils.product(shape[:-1])  # 返回除最后一位外其他几个数的乘积
        #     stddev = 1 / math.sqrt(number_inputs_added)
        #     return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)  #从截断的正态分布中输出随机值,shape表示生成张量的维度,stddev是标准差
        #
        # W_conv_init = _weight_variable([5, 5, self.num_input_planes, 192], name="W_conv_init")
        # W_conv_intermediate = []
        # for i in range(3):
        #     W_conv_intermediate.append(_weight_variable([3, 3, 192, 192], name="W_conv"))
        #
        # W_conv_final = _weight_variable([1, 1, 192, 1], name="W_conv_final")

        # 卷积层定义
        conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=3, strides=1,
                                 padding='same', activation=tf.nn.relu, data_format="channels_last")
        conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3, strides=1,
                                 padding='same', activation=tf.nn.relu, data_format="channels_last")
        conv3 = tf.layers.conv2d(conv2, filters=128, kernel_size=3, strides=1,
                                 padding='same', activation=tf.nn.relu, data_format="channels_last")
        # 卷积结果
        action_conv = tf.layers.conv2d(conv3, filters=self.num_input_planes, kernel_size=1, strides=1,
                                       padding='same', activation=tf.nn.relu, data_format="channels_last")
        '''
        # 更新语法
        # 卷积层定义
        conv1 = tf.keras.layers.Conv2D(inputs=x, filters=32, kernel_size=3, strides=1,
                                 padding='same', activation=tf.nn.relu, data_format="channels_last")
        conv2 = tf.keras.layers.Conv2D(conv1, filters=64, kernel_size=3, strides=1,
                                 padding='same', activation=tf.nn.relu, data_format="channels_last")
        conv3 = tf.keras.layers.Conv2D(conv2, filters=128, kernel_size=3, strides=1,
                                 padding='same', activation=tf.nn.relu, data_format="channels_last")
        # 卷积结果
        action_conv = tf.keras.layers.Conv2D(conv3, filters=self.num_input_planes, kernel_size=1, strides=1,
                                       padding='same', activation=tf.nn.relu, data_format="channels_last")
        '''

        # 扁平化张量
        action_conv_flat = tf.reshape(action_conv, [-1, self.num_input_planes * go.N * go.N])


        # 输出层（全连接层）：
        action_fc = tf.layers.dense(inputs=action_conv_flat, units=go.N * go.N, activation=tf.nn.log_softmax)
        '''
        # 更新语法
        action_fc = tf.keras.layers.dense(inputs=action_conv_flat, units=go.N * go.N, activation=tf.nn.log_softmax)
        '''
        b_conv_final = tf.Variable(tf.constant(0, shape=[go.N ** 2], dtype=tf.float32))

        evaluation_fc = tf.layers.dense(inputs=conv3, units=go.N ** 2, activation=tf.nn.tanh)
        evaluation_conv_flat_0 = tf.nn.dropout(evaluation_fc, 0.8)
        evaluation_conv_flat_1 = tf.reshape(evaluation_conv_flat_0, [-1, go.N ** 2])
        evaluation_conv_flat_2 = tf.layers.dense(evaluation_conv_flat_1, units=64, activation=tf.nn.relu)
        logits_out = tf.layers.dense(evaluation_conv_flat_2, units=1, activation=tf.nn.tanh)

        logits = tf.reshape(logits_out, [-1, go.N ** 2])
        self.output = tf.nn.softmax(logits + b_conv_final)
        # 计算损失
        # log_likelihood_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y))
        log_likelihood_cost = tf.losses.mean_squared_error(y, logits)
        # l2正则化
        l2_penalty_beta = 1e-4
        vars = tf.trainable_variables()
        l2_penalty = l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        loss = log_likelihood_cost + l2_penalty

        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=global_step)
        # 存储判断是否正确，argmax函数用来寻找logits和y中的同一维度（1轴）的最大值位置。equal用来判断是否相等（返回结果为一个和判断张量一样的张量）
        was_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        # 精准度的计算，计算所有维度的平均正确率（把所有维度正确糅合成一个正确率）
        accuracy = tf.reduce_mean(tf.cast(was_correct, tf.float32))

        # #优化器
        # weight_summaries = tf.summary.merge([
        #     tf.summary.histogram(weight_var.name, weight_var)
        #     for weight_var in [W_conv_init] + W_conv_intermediate + [W_conv_final, b_conv_final]],
        #     name="weight_summaries"
        # )

        activation_summaries = tf.summary.merge([
            tf.summary.histogram(act_var.name, act_var)
            for act_var in [conv1] + [conv2] + [conv3] + [action_fc]],
        )

        # 模型存储
        saver = tf.train.Saver()

        # save everything to self.
        for name, thing in locals().items():
            if not name.startswith('_'):
                setattr(self, name, thing)

    # 初始化训练日志
    def initialize_logging(self, tensorboard_logdir):
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(tensorboard_logdir, "test"),
                                                         self.session.graph)
        self.training_summary_writer = tf.summary.FileWriter(os.path.join(tensorboard_logdir, "training"),
                                                             self.session.graph)

    # 初始化tensorFlow的变量
    def initialize_variables(self, save_file=None):  # save_file是模型保存的路径，在cmd窗口中执行的时候进行传值
        self.session.run(tf.global_variables_initializer())
        if save_file is not None:
            self.saver.restore(self.session, save_file)

    def get_global_step(self):
        return self.session.run(self.global_step)

    # 训练完多少次后保存训练数据
    def save_variables(self, save_file):
        if save_file is not None:
            print("Saving checkpoint to %s" % save_file, file=sys.stderr)
            self.saver.save(self.session, save_file)

    # 开始训练API
    def train(self, training_data, batch_size=32):
        num_minibatches = training_data.data_size // batch_size
        for i in range(num_minibatches):
            batch_x, batch_y = training_data.get_batch(batch_size)
            _, accuracy, cost = self.session.run(
                [self.train_step, self.accuracy, self.log_likelihood_cost],
                feed_dict={self.x: batch_x, self.y: batch_y})
            self.training_stats.report(accuracy, cost)

        avg_accuracy, avg_cost, accuracy_summaries = self.training_stats.collect()
        global_step = self.get_global_step()
        print("Step %d training data accuracy: %g; cost: %g" % (global_step, avg_accuracy, avg_cost))
        if self.training_summary_writer is not None:
            activation_summaries = self.session.run(
                self.activation_summaries,
                feed_dict={self.x: batch_x, self.y: batch_y})
            self.training_summary_writer.add_summary(activation_summaries, global_step)
            self.training_summary_writer.add_summary(accuracy_summaries, global_step)

    # 预测数据API
    def run(self, position):
        """Return a sorted list of (probability, move) tuples"""
        processed_position = features.extract_features(position, features=self.features)
        probabilities = self.session.run(self.output, feed_dict={self.x: processed_position[None, :]})[0]
        # print("可能落子点：",probabilities.reshape([go.N, go.N]))
        return probabilities.reshape([go.N, go.N])

    # 检测精确度
    def check_accuracy(self, test_data, batch_size=128):
        num_minibatches = test_data.data_size // batch_size
        # weight_summaries = self.session.run(self.weight_summaries)

        for i in range(num_minibatches):
            batch_x, batch_y = test_data.get_batch(batch_size)
            accuracy, cost = self.session.run(
                [self.accuracy, self.log_likelihood_cost],
                feed_dict={self.x: batch_x, self.y: batch_y})
            self.test_stats.report(accuracy, cost)

        avg_accuracy, avg_cost, accuracy_summaries = self.test_stats.collect()
        global_step = self.get_global_step()
        print("Step %s test data accuracy: %g; cost: %g" % (global_step, avg_accuracy, avg_cost))

        if self.test_summary_writer is not None:
            # self.test_summary_writer.add_summary(weight_summaries, global_step)
            self.test_summary_writer.add_summary(accuracy_summaries, global_step)


class StatisticsCollector(object):
    '''
    Accuracy and cost cannot be calculated with the full test dataset
    in one pass, so they must be computed in batches. Unfortunately,
    the built-in TF summary nodes cannot be told to aggregate multiple
    executions. Therefore, we aggregate the accuracy/cost ourselves at
    the python level, and then shove it through the accuracy/cost summary
    nodes to generate the appropriate summary protobufs for writing.
    无法使用完整的测试数据集计算准确性和成本
    一次通过，所以它们必须分批计算。不幸的是，
    不能告诉内置tf summary节点聚合多个
    执行。因此，我们将准确度/成本汇总为
    python级别，然后将其推送到准确性/成本摘要中
    节点以生成用于写入的适当摘要协议。
    '''
    graph = tf.Graph()
    with tf.device("/cpu:0"), graph.as_default():
        accuracy = tf.compat.v1.placeholder(tf.float32, [])
        cost = tf.compat.v1.placeholder(tf.float32, [])
        accuracy_summary = tf.compat.v1.summary.scalar("accuracy", accuracy)
        cost_summary = tf.compat.v1.summary.scalar("log_likelihood_cost", cost)
        accuracy_summaries = tf.compat.v1.summary.merge([accuracy_summary, cost_summary], name="accuracy_summaries")
    session = tf.compat.v1.Session(graph=graph)

    def __init__(self):
        self.accuracies = []
        self.costs = []

    def report(self, accuracy, cost):
        self.accuracies.append(accuracy)
        self.costs.append(cost)

    def collect(self):
        avg_acc = sum(self.accuracies) / len(self.accuracies)
        avg_cost = sum(self.costs) / len(self.costs)
        self.accuracies = []
        self.costs = []
        summary = self.session.run(self.accuracy_summaries,
                                   feed_dict={self.accuracy: avg_acc, self.cost: avg_cost})
        return avg_acc, avg_cost, summary


'''
import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import go_evaluable.features as features
import go_boardControl.boardControl as go
sys.path.append("..")

# Tensorflow日志，在对战中关闭错误等级以下的日志
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
EPSILON = 1e-35
learning_rate = 0.0002
beta1 = 0.5

class PolicyNetwork(object):
    def __init__(self, features=features.DEFAULT_FEATURES, k=192, num_int_conv_layers=3, use_cpu=False):
        self.num_input_planes = sum(f.planes for f in features)
        self.features = features
        self.k = k
        self.num_int_conv_layers = num_int_conv_layers  # 卷积层数量
        self.test_summary_writer = None
        self.training_summary_writer = None
        self.test_stats = StatisticsCollector()  # 测试统计
        self.training_stats = StatisticsCollector()  # 训练统计
        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))  # tensorFlow的会话
        # 在GPU无法使用的时候选择CPU, 主要是在训练的时候，选择GPU速度会快至少一倍
        if use_cpu:
            with tf.device("/cpu:0"):
                self.set_up_network()
        else:
            with tf.device("/gpu:0"):
                self.set_up_network()
    # 初始化设置神经网路，输入输出，卷积层等
    def set_up_network(self):
        def discriminator(x, scope='D', reuse=False):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                d_1 = conv2d(x, self.k, 5, name='d_1')
                d_1 = slim.dropout(d_1, keep_prob=0.8, is_training=True, scope='d_1_conv/')
                d_2 = conv2d(d_1, self.k, name='d_2')
                d_2 = slim.dropout(d_2, keep_prob=0.8, is_training=True, scope='d_2_conv/')
                d_3 = conv2d(d_2, self.k, name='d_3')
                d_3 = slim.dropout(d_3, keep_prob=0.8, is_training=True, scope='d_3_conv/')
                d_4 = conv2d(d_3, self.k, name='d_4')
                d_4 = slim.dropout(d_4, keep_prob=0.8, is_training=True, scope='d_4_conv/')
                d_5 = conv2d(d_4, 1, 1, name='d_5')
                d_5 = slim.dropout(d_5, keep_prob=0.8, is_training=True, scope='d_5_conv/')
                d_6 = slim.fully_connected(
                        tf.reshape(d_5,[-1, go.N ** 2]), go.N**2 + 1, scope='d_6_fc',
                        activation_fn=None)
            return tf.nn.softmax(d_6),d_6
        def generator(z, scope='G'):
            with tf.variable_scope(scope) as scope:
                layer1 = tf.layers.dense(z, go.N ** 2)
                layer1 = tf.reshape(layer1, [-1, go.N, go.N, 1])
                g_1 = deconv2d(layer1, self.k, 1, name='g_1')
                g_2 = deconv2d(g_1, self.k, name='g_2')
                g_3 = deconv2d(g_2, self.k, name='g_3')
                g_4 = deconv2d(g_3, self.k, name='g_4')
                g_5 = deconv2d(g_4, self.num_input_planes, 5, name='g_5')
            return g_5
        def deconv2d(x, channels, kernel=3, name='deconv2d', stride=1):
            with tf.variable_scope(name):
                deconv = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel,
                                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.08),
                                           kernel_regularizer=None,
                                           strides=(1,1),
                                           padding='SAME',use_bias=True)
                deconv = tf.nn.relu(deconv)
                bn = tf.contrib.layers.batch_norm(deconv, center=True,scale=True,
                                                  decay=0.9,is_training=True,
                                                  updates_collections=None)
            return bn

        def conv2d(x, channels, kernel=3, pad=0, pad_type='zero', name='conv2d'):
            with tf.variable_scope(name):
                w = tf.Variable(tf.truncated_normal([kernel, kernel, int(x.shape[-1]), channels], stddev=0.08), name='w')
                conv = tf.nn.conv2d(x, w ,strides=[1,1,1,1], padding='SAME')
                biases = tf.get_variable('biases', [channels], initializer=tf.constant_initializer(0.0))
                conv = lrelu(tf.reshape(tf.nn.bias_add(conv, biases),[-1, int(conv.shape[1]), int(conv.shape[2]), int(conv.shape[3])]))
                #return conv
            return tf.contrib.layers.batch_norm(conv,center=True,scale=True,decay=0.9,is_training=True,updates_collections=None)

        def lrelu(x, leak=0.2, name='lrelu'):
            with tf.variable_scope(name):
                f1 = 0.5 * (1 + leak)
                f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        def huber_loss(labels, predictions, delta = 1.0):
            residual = tf.abs(predictions - labels)
            condition = tf.less(residual, delta)
            small_res = 0.5 * tf.square(residual)
            large_res = delta * residual - 0.5 * tf.square(delta)
            return tf.where(condition, small_res, large_res)

        def get_loss(x, z, y_real, y_fake):
            g_outputs = generator(z)
            d_outputs_real, d_logits_real= discriminator(x)
            d_outputs_fake, d_logits_fake= discriminator(g_outputs, reuse=True)

            was_correct = tf.equal(tf.argmax(d_logits_real, 1), tf.argmax(y_real, 1))
            accuracy = tf.reduce_mean(tf.cast(was_correct, tf.float32))
            g_loss = tf.reduce_mean(tf.log(d_outputs_fake[:, -1]))
            g_loss += tf.reduce_mean(huber_loss(x, g_outputs))*0.0001
            g_loss_l2 = tf.reduce_mean(tf.square(g_outputs - x))
            g_loss = g_loss + g_loss_l2

            d_loss_real = tf.nn.softmax_cross_entropy_with_logits_v2(logits=d_logits_real, labels=y_real)
            d_loss_fake = tf.nn.softmax_cross_entropy_with_logits_v2(logits=d_logits_fake, labels=y_fake)
            d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)
            return accuracy, g_loss, d_loss

        global_step = tf.Variable(0, name="global_step", trainable=False)
        # 输入为9x9x48的图像堆栈
        z = tf.placeholder(tf.float32, [None, go.N ** 2 + 1])

        x = tf.placeholder(tf.float32, [None, go.N, go.N, self.num_input_planes])

        y_real = tf.placeholder(tf.float32, shape=[None, go.N ** 2 + 1])

        y_fake = tf.placeholder(tf.float32, shape=[None, go.N ** 2 + 1])

        #batch_size = int(x.shape[0])

        accuracy, g_loss, d_loss = get_loss(x, z, y_real, y_fake)
        d_sum = tf.summary.scalar("d_loss", d_loss)
        g_sum = tf.summary.scalar("g_loss", g_loss)
        train_vars = tf.trainable_variables()
        g_vars = [var for var in train_vars if var.name.startswith("G")]
        d_vars = [var for var in train_vars if var.name.startswith("D")]
        g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
        d_opt = tf.train.AdamOptimizer(learning_rate*0.5, beta1=beta1).minimize(d_loss, var_list=d_vars, global_step=global_step)

        saver = tf.train.Saver()

        # save everything to self.
        for name, thing in locals().items():
            if not name.startswith('_'):
                setattr(self, name, thing)

    # 初始化训练日志
    def initialize_logging(self, tensorboard_logdir):
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(tensorboard_logdir, "test"), self.session.graph)
        self.training_summary_writer = tf.summary.FileWriter(os.path.join(tensorboard_logdir, "training"), self.session.graph)

    # 初始化tensorFlow的变量
    def initialize_variables(self, save_file=None):
        self.session.run(tf.global_variables_initializer())
        if save_file is not None:
            self.saver.restore(self.session, save_file)

    def get_global_step(self):
        return self.session.run(self.global_step)

    # 训练完多少次后保存训练数据
    def save_variables(self, save_file, step):
        if save_file is not None:
            print("Saving checkpoint to %s" % save_file, file=sys.stderr)
            self.saver.save(self.session, save_file, step)

    # 开始训练API
    def train(self, training_data, batch_size=32):
        num_minibatches = training_data.data_size // batch_size
        for i in range(num_minibatches):
            batch_x, batch_y = training_data.get_batch(batch_size)
            batch_real_y = np.concatenate((batch_y, np.zeros([batch_size, 1])), axis=1)
            #batch_fake_y = np.concatenate((0.1*np.ones([batch_size, go.N**2])/10, 0.9*np.ones([batch_size, 1])), axis=1)
            batch_fake_y = np.concatenate((np.zeros([batch_size, go.N ** 2]), np.ones([batch_size, 1])), axis=1)
            batch_z = np.random.uniform(-1, 1, [batch_size, go.N ** 2 + 1])

            _, accuracy, cost = self.session.run(
                [self.d_opt, self.accuracy, self.d_loss], feed_dict={self.x: batch_x,
                self.y_real: batch_real_y, 
                self.y_fake: batch_fake_y,
                self.z: batch_z
                }
                )
            _,g_loss = self.session.run(
                [self.g_opt, self.g_loss],feed_dict={
                            self.x: batch_x,
                            self.y_real: batch_real_y, 
                            self.z: batch_z, self.y_fake:batch_fake_y}
                    )
            self.training_stats.report(accuracy, cost)

        avg_accuracy, avg_cost, accuracy_summaries = self.training_stats.collect()
        global_step = self.get_global_step()
        print("Step %d training data accuracy: %g; cost: %g" % (global_step, avg_accuracy, avg_cost))
        if self.training_summary_writer is not None:
            activation_summaries = self.session.run(
                self.d_vars,
                feed_dict={self.x: batch_x,self.y_real: batch_real_y, self.y_fake: batch_fake_y,self.z: batch_z})
            self.training_summary_writer.add_summary(activation_summaries, global_step)
            self.training_summary_writer.add_summary(accuracy_summaries, global_step)

    # 预测数据API
    def run(self, position):
        """Return a sorted list of (probability, move) tuples"""
        processed_position = features.extract_features(position, features=self.features)
        probabilities = self.session.run(self.d_outputs_real, feed_dict={self.x: processed_position[None, :]})[0]
        # print("可能落子点：",probabilities.reshape([go.N, go.N]))
        probabilities = probabilities[:-1]
        return probabilities.reshape([go.N, go.N])

    # 检测精确度
    def check_accuracy(self, test_data, batch_size=128):
        num_minibatches = test_data.data_size // batch_size
        weight_summaries = self.session.run(self.d_vars)

        for i in range(num_minibatches):
            batch_x, batch_y = test_data.get_batch(batch_size)
            batch_real_y = np.concatenate((batch_y, np.zeros([batch_size, 1])), axis=1)
            #batch_fake_y = np.concatenate((0.1 * np.ones([batch_size, go.N ** 2]) / 10, 0.9 * np.ones([batch_size, 1])), axis = 1)
            batch_fake_y = np.concatenate((np.zeros([batch_size, go.N ** 2]), np.ones([batch_size, 1])), axis=1)
            batch_z = np.random.uniform(-1, 1, [batch_size, go.N ** 2 + 1])
            accuracy, cost = self.session.run(
                [self.accuracy, self.d_loss],
                feed_dict={
                            self.x: batch_x,
                            self.y_real: batch_real_y,
                            self.z: batch_z, self.y_fake:batch_fake_y
                })
            _ = self.session.run(
                [self.g_loss], feed_dict={
                    self.x: batch_x,
                    self.y_real: batch_real_y,
                    self.z: batch_z, self.y_fake: batch_fake_y}
            )
            self.test_stats.report(accuracy, cost)

        avg_accuracy, avg_cost, accuracy_summaries = self.test_stats.collect()
        global_step = self.get_global_step()
        print("Step %s test data accuracy: %g; cost: %g" % (global_step, avg_accuracy, avg_cost))

        if self.test_summary_writer is not None:
            self.test_summary_writer.add_summary(weight_summaries, global_step)
            self.test_summary_writer.add_summary(accuracy_summaries, global_step)


class StatisticsCollector(object):

    Accuracy and cost cannot be calculated with the full test dataset
    in one pass, so they must be computed in batches. Unfortunately,
    the built-in TF summary nodes cannot be told to aggregate multiple
    executions. Therefore, we aggregate the accuracy/cost ourselves at
    the python level, and then shove it through the accuracy/cost summary
    nodes to generate the appropriate summary protobufs for writing.

    graph = tf.Graph()
    with tf.device("/cpu:0"), graph.as_default():
        accuracy = tf.placeholder(tf.float32, [])
        cost = tf.placeholder(tf.float32, [])
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)
        cost_summary = tf.summary.scalar("log_likelihood_cost", cost)
        accuracy_summaries = tf.summary.merge([accuracy_summary, cost_summary], name="accuracy_summaries")
    session = tf.Session(graph=graph)

    def __init__(self):
        self.accuracies = []
        self.costs = []

    def report(self, accuracy, cost):
        self.accuracies.append(accuracy)
        self.costs.append(cost)

    def collect(self):
        avg_acc = sum(self.accuracies) / len(self.accuracies)
        avg_cost = sum(self.costs) / len(self.costs)
        self.accuracies = []
        self.costs = []
        summary = self.session.run(self.accuracy_summaries,
            feed_dict={self.accuracy:avg_acc, self.cost: avg_cost})
        return avg_acc, avg_cost, summary

'''