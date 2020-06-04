import numpy as np
import os
import tensorflow as tf
import datetime
from tensorflow.python import debug as tf_debug
from utils import fully_connected_layer, stateful_lstm, huber_loss




class DRQN():
    """
    Base class for deep Q learning
    """

    def __init__(self, config, num_actions):

        # super(DRQN, self).__init__(config, "drqn")
        self.num_actions = num_actions
        self.num_lstm_layers = 1
        self.lstm_size = config.lstm_size
        self.min_history = config.min_history
        self.states_to_update = config.states_to_update

        self.image_size = config.image_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma
        self.dir_save = config.dir_save
        self.learning_rate_minimum = config.learning_rate_minimum
        self.debug = not True
        self.keep_prob = config.keep_prob
        self.batch_size = config.batch_size
        self.lr_method = config.lr_method
        self.learning_rate = config.learning_rate
        self.lr_decay = config.lr_decay
        self.sess = None
        self.saver = None
        # delete ./out
        self.dir_output = "./out/DRQN/"+ str(datetime.datetime.utcnow()) + "/"
        self.dir_model = self.dir_save + "/net/" + str(datetime.datetime.utcnow()) + "/"

        self.train_steps = 0
        self.is_training = False

        self.sequence_length = 6

    def add_train_op(self, lr_method, lr, loss, clip=-1):
        _lr_m = lr_method.lower()  # lower to make sure

        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(lr)

            if clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    def initialize_session(self):
        print("Initializing tf session")
        self.sess = tf.Session()
        if self.debug:
            self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "localhost:6064")
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def close_session(self):
        self.sess.close()

    def add_summary(self, summary_tags, histogram_tags):
        self.summary_placeholders = {}
        self.summary_ops = {}
        for tag in summary_tags:
            self.summary_placeholders[tag] = tf.placeholder(tf.float32, None, name=tag)
            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
        for tag in histogram_tags:
            self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
            self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])
        self.file_writer = tf.summary.FileWriter(self.dir_output + "/train",
                                                 self.sess.graph)

    def inject_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summ in summary_str_lists:
            self.file_writer.add_summary(summ, step)


    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)
        self.saver.save(self.sess, self.dir_model)

    def restore_session(self, path=None):
        if path is not None:
            self.saver.restore(self.sess, path)
        else:
            self.saver.restore(self.sess, self.dir_model)

    def add_placeholders(self):
        self.w = {}
        self.w_target = {}
        self.state = tf.placeholder(tf.float32, shape=[None, self.image_size* self.num_classes],
                                    name="input_state")
        self.action = tf.placeholder(tf.int32, shape=[None], name="action_input")
        self.reward = tf.placeholder(tf.int32, shape=[None], name="reward")
        self.state_target = tf.placeholder(tf.float32,
                                           shape=[None, self.image_size* self.num_classes],
                                           name="input_target")

        self.sequence_size = tf.placeholder(dtype=tf.float32, name="sequence_size")
        # create placeholder to fill in lstm state
        self.c_state_train = tf.placeholder(tf.float32, [None, self.lstm_size], name="train_c")
        self.h_state_train = tf.placeholder(tf.float32, [None, self.lstm_size], name="train_h")
        self.lstm_state_train = tf.nn.rnn_cell.LSTMStateTuple(self.c_state_train, self.h_state_train)

        self.c_state_target = tf.placeholder(tf.float32, [None, self.lstm_size], name="target_c")
        self.h_state_target = tf.placeholder(tf.float32, [None, self.lstm_size], name="target_h")
        self.lstm_state_target = tf.nn.rnn_cell.LSTMStateTuple(self.c_state_target, self.h_state_target)

        # initial zero state to be used when starting episode
        self.initial_zero_state_batch = np.zeros((self.batch_size, self.lstm_size))
        self.initial_zero_state_single = np.zeros((1, self.lstm_size))
        self.initial_zero_complete = np.zeros((self.num_lstm_layers, 2, self.batch_size, self.lstm_size))

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")
        self.target_val = tf.placeholder(dtype=tf.float32, shape=[None], name="target_val")
        self.terminal = tf.placeholder(dtype=tf.float32, shape=[None], name="terminal")
        self.target_val_tf = tf.placeholder(dtype=tf.float32, shape=[None, self.num_actions])

    def add_logits_op_train(self):
        '''
        if self.cnn_format == "NHWC":
            x = tf.transpose(self.state, [0, 2, 3, 1])
        else:
            x = self.state
        self.image_summary = []
        w, b, out, summary = conv2d_layer(x, 32, [8, 8], [4, 4], scope_name="conv1_train",
                                          summary_tag="conv1_out",
                                          activation=tf.nn.relu, data_format=self.cnn_format)
        self.w["wc1"] = w
        self.w["bc1"] = b
        self.image_summary.append(summary)

        w, b, out, summary = conv2d_layer(out, 64, [4, 4], [2, 2], scope_name="conv2_train", summary_tag="conv2_out",
                                          activation=tf.nn.relu, data_format=self.cnn_format)
        self.w["wc2"] = w
        self.w["bc2"] = b
        self.image_summary.append(summary)

        w, b, out, summary = conv2d_layer(out, 64, [3, 3], [1, 1], scope_name="conv3_train", summary_tag="conv3_out",
                                          activation=tf.nn.relu, data_format=self.cnn_format)
        self.w["wc3"] = w
        self.w["bc3"] = b
        self.image_summary.append(summary)
        shape = out.get_shape().as_list()
        '''

        self.state_flat = tf.reshape(self.state, [self.batch_size, self.sequence_size, self.image_size * self.num_classes])
        self.dense1 = tf.contrib.layers.fully_connected(inputs=self.state_flat, num_outputs=256)
        self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=64)

        out, state = stateful_lstm(self.dense2, self.num_lstm_layers, self.lstm_size, tuple([self.lstm_state_train]),
                                   scope_name="lstm_train")

        self.state_output_c = state[0][0]
        self.state_output_h = state[0][1]

        out = tf.reshape(out, [1, 512])

        w, b, out = fully_connected_layer(out, self.num_actions, scope_name="out_train", activation=None)

        self.w["wout"] = w
        self.w["bout"] = b

        self.q_out = out
        self.q_action = tf.argmax(self.q_out, axis=1)


    def add_logits_op_target(self):
        '''
        if self.cnn_format == "NHWC":
            x = tf.transpose(self.state_target, [0, 2, 3, 1])
        else:
            x = self.state_target
        w, b, out, _ = conv2d_layer(x, 32, [8, 8], [4, 4], scope_name="conv1_target", summary_tag=None,
                                    activation=tf.nn.relu, data_format=self.cnn_format)
        self.w_target["wc1"] = w
        self.w_target["bc1"] = b

        w, b, out, _ = conv2d_layer(out, 64, [4, 4], [2, 2], scope_name="conv2_target", summary_tag=None,
                                    activation=tf.nn.relu, data_format=self.cnn_format)
        self.w_target["wc2"] = w
        self.w_target["bc2"] = b

        w, b, out, _ = conv2d_layer(out, 64, [3, 3], [1, 1], scope_name="conv3_target", summary_tag=None,
                                    activation=tf.nn.relu, data_format=self.cnn_format)
        self.w_target["wc3"] = w
        self.w_target["bc3"] = b
        '''
        self.state_flat = tf.reshape(self.state,
                                     [self.batch_size, self.sequence_size, self.image_size * self.num_classes])
        self.dense1 = tf.contrib.layers.fully_connected(inputs=self.state_flat, num_outputs=256)
        self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=64)

        out, state = stateful_lstm(self.dense2, self.num_lstm_layers, self.lstm_size, tuple([self.lstm_state_train]),
                                   scope_name="lstm_target")


        self.state_output_target_c = state[0][0]
        self.state_output_target_h = state[0][1]

        out = tf.reshape(out, [1, 512])

        w, b, out = fully_connected_layer(out, self.num_actions, scope_name="out_train", activation=None)

        self.w_target["wout"] = w
        self.w_target["bout"] = b

        self.q_target_out = out
        self.q_target_action = tf.argmax(self.q_target_out, axis=1)

    def train_on_batch_target(self, states, action, reward, terminal, steps):
        states = states / 255.0
        q, loss = np.zeros((self.batch_size, self.num_actions)), 0
        states = np.transpose(states, [1, 0, 2, 3])
        action = np.transpose(action, [1, 0])
        reward = np.transpose(reward, [1, 0])
        terminal = np.transpose(terminal, [1, 0])
        states = np.reshape(states, [states.shape[0], states.shape[1], states.shape[2] * states.shape[3]])
        lstm_state_c, lstm_state_h = self.initial_zero_state_batch, self.initial_zero_state_batch

        lstm_state_target_c, lstm_state_target_h = self.sess.run(
            [self.state_output_target_c, self.state_output_target_h],
            {
                self.sequence_size: 1,
                self.state: states[0],
                self.state_target: states[0],
                self.c_state_target: self.initial_zero_state_batch,
                self.h_state_target: self.initial_zero_state_batch
            }
        )

        for i in range(self.min_history):
            j = i + 1
            lstm_state_c, lstm_state_h, lstm_state_target_c, lstm_state_target_h = self.sess.run(
                [self.state_output_c, self.state_output_h, self.state_output_target_c, self.state_output_target_h],
                {
                    self.sequence_size: 1,
                    self.state: states[i],
                    self.state_target: states[j],
                    self.c_state_target: lstm_state_target_c,
                    self.h_state_target: lstm_state_target_h,
                    self.c_state_train: lstm_state_c,
                    self.h_state_train: lstm_state_h
                }
            )
        for i in range(self.min_history, self.min_history + self.states_to_update):
            j = i + 1
            target_val, lstm_state_target_c, lstm_state_target_h = self.sess.run(
                [self.q_target_out, self.state_output_target_c, self.state_output_target_h],
                {
                    self.sequence_size: 1,
                    self.state: states[i],
                    self.state_target: states[j],
                    self.c_state_target: lstm_state_target_c,
                    self.h_state_target: lstm_state_target_h
                }
            )
            max_target = np.max(target_val, axis=1)
            target = (1. - terminal[i]) * self.gamma * max_target + reward[i]
            _, q_, train_loss_, lstm_state_c, lstm_state_h = self.sess.run(
                [self.train_op, self.q_out, self.loss, self.state_output_c, self.state_output_h],
                feed_dict={
                    self.sequence_size: 1,
                    self.state: states[i],
                    self.c_state_train: lstm_state_c,
                    self.h_state_train: lstm_state_h,
                    self.action: action[i],
                    self.target_val: target,
                    self.lr: self.learning_rate
                }
            )
            q += q_
            loss += train_loss_

        if steps % 20000 == 0 and steps > 50000:
            self.learning_rate *= self.lr_decay  # decay learning rate
            if self.learning_rate < self.learning_rate_minimum:
                self.learning_rate = self.learning_rate_minimum
        self.train_steps += 1
        return q.mean(), loss / (self.states_to_update)

    def add_loss_op_target(self):
        action_one_hot = tf.one_hot(self.action, self.num_actions, 1.0, 0.0, name='action_one_hot')
        train = tf.reduce_sum(self.q_out * action_one_hot, reduction_indices=1, name='q_acted')
        self.delta = train - self.target_val
        self.loss = tf.reduce_mean(huber_loss(self.delta))

        avg_q = tf.reduce_mean(self.q_out, 0)
        q_summary = []
        for i in range(self.num_actions):
            q_summary.append(tf.summary.histogram('q/{}'.format(i), avg_q[i]))
        self.avg_q_summary = tf.summary.merge(q_summary, 'q_summary')
        self.loss_summary = tf.summary.scalar("loss", self.loss)

    def build(self):
        self.add_placeholders()
        self.add_logits_op_train()
        self.add_logits_op_target()
        self.add_loss_op_target()
        self.add_train_op(self.lr_method, self.lr, self.loss, clip=10)
        self.initialize_session()
        self.init_update()

    def update_target(self):
        for name in self.w:
            self.target_w_assign[name].eval({self.target_w_in[name]: self.w[name].eval(session=self.sess)},
                                            session=self.sess)
        for var in self.lstm_vars:
            self.target_w_assign[var.name].eval({self.target_w_in[var.name]: var.eval(session=self.sess)},
                                                session=self.sess)

    def init_update(self):
        self.target_w_in = {}
        self.target_w_assign = {}
        for name in self.w:
            self.target_w_in[name] = tf.placeholder(tf.float32, self.w_target[name].get_shape().as_list(), name=name)
            self.target_w_assign[name] = self.w_target[name].assign(self.target_w_in[name])

        self.lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="lstm_train")
        lstm_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="lstm_target")

        for i, var in enumerate(self.lstm_vars):
            self.target_w_in[var.name] = tf.placeholder(tf.float32, var.get_shape().as_list())
            self.target_w_assign[var.name] = lstm_target_vars[i].assign(self.target_w_in[var.name])