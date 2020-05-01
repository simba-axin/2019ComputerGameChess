# -*- coding: utf-8 -*-
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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class PolicyNetWork():
    def __init__(self,features=features.DEFAULT_FEATURES,use_cpu=False):
        self.num_input_planes = sum(f.planes for f in features)
        self.features = features
        self.session = tf.Session()

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

        x = tf.placeholder(tf.float32, [None, go.N, go.N, self.num_input_planes])
        y = tf.placeholder(tf.float32, [None, go.N ** 2])  # go.N ** 2

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

        #卷积层定义
        conv1 = tf.layers.conv2d(inputs = x,filters=32,kernel_size=3,strides=1,
                                 padding='same',activation=tf.nn.relu,data_format="channels_last")
        conv2 = tf.layers.conv2d(conv1,filters=64,kernel_size=3,strides=1,
                                 padding='same',activation=tf.nn.relu,data_format="channels_last")
        conv3 = tf.layers.conv2d(conv2,filters=128,kernel_size=3,strides=1,
                                 padding='same',activation=tf.nn.relu,data_format="channels_last")
        #卷积结果
        action_conv = tf.layers.conv2d(conv3,filters=self.num_input_planes,kernel_size=1,strides=1,
                                       padding='same',activation=tf.nn.relu,data_format="channels_last")
        #扁平化张量
        action_conv_flat = tf.reshape(action_conv,[-1,self.num_input_planes*go.N*go.N])
        #输出层（全连接层）：
        action_fc = tf.layers.dense(inputs=action_conv_flat,units=go.N*go.N,activation=tf.nn.log_softmax)

        b_conv_final = tf.Variable(tf.constant(0, shape=[go.N ** 2], dtype=tf.float32))

        evaluation_fc = tf.layers.dense(inputs=conv3,units=go.N ** 2,activation=tf.nn.tanh)
        evaluation_conv_flat_0 = tf.nn.dropout(evaluation_fc,0.8)
        evaluation_conv_flat_1 = tf.reshape(evaluation_conv_flat_0,[-1,go.N ** 2])
        evaluation_conv_flat_2 = tf.layers.dense(evaluation_conv_flat_1,units=64,activation=tf.nn.relu)
        logits_out = tf.layers.dense(evaluation_conv_flat_2,units=1,activation=tf.nn.tanh)

        logits = tf.reshape(logits_out,[-1,go.N ** 2])
        self.output = tf.nn.softmax(logits + b_conv_final)
        #计算损失
        #log_likelihood_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y))
        log_likelihood_cost = tf.losses.mean_squared_error(y,logits)
        #l2正则化
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

        #模型存储
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
    def initialize_variables(self, save_file=None):           #save_file是模型保存的路径，在cmd窗口中执行的时候进行传值
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
        #weight_summaries = self.session.run(self.weight_summaries)

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
            #self.test_summary_writer.add_summary(weight_summaries, global_step)
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
