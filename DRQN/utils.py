import json
import numpy as np
import tensorflow as tf


def rgb2gray(screen):
    return np.dot(screen[..., :3], [0.299, 0.587, 0.114])


def load_config(config_file):
    pass


def save_config(config_file, config_dict):
    with open(config_file, 'w') as fp:
        json.dump(config_dict, fp)


def fully_connected_layer(x, output_dim, scope_name="fully", initializer=tf.random_normal_initializer(stddev=0.02),
                          activation=tf.nn.relu):
    shape = x.get_shape().as_list()
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        w = tf.get_variable("w", [shape[1], output_dim], dtype=tf.float32,
                            initializer=initializer)
        b = tf.get_variable("b", [output_dim], dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        out = tf.nn.xw_plus_b(x, w, b)
        if activation is not None:
            out = activation(out)

        return w, b, out

def stateful_lstm(x, num_layers, lstm_size, state_input, scope_name="lstm"):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size=1, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)
        return outputs, state

def huber_loss(x, delta=1.0):
    return tf.where(tf.abs(x) < delta, 0.5 * tf.square(x), delta * tf.abs(x) - 0.5* delta)

def integer_product(x):
    return int(np.prod(x))

def initializer_bounds_filter(filter_shape):
    fan_in = integer_product(filter_shape[:3])
    fan_out = integer_product(filter_shape[:2]) * filter_shape[3]
    return np.sqrt(6. / (fan_in + fan_out))