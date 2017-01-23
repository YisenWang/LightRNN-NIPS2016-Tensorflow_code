import tensorflow as tf
import numpy as np

class MultiLSTM(object):
    def __init__(self, input_size, lstm_size, layers, forget_bias=1.0, activation=tf.tanh, dtype=tf.float32, scope=None):
        self._input_size = input_size  # =word_embedding_dim in RNNLM
        self._lstm_size = lstm_size
        self._layers = layers
        self._forget_bias = forget_bias
        self._activation = activation

        with tf.variable_scope(scope or type(self).__name__):
            self._matrix = []
            self._bias = []
            for layer in range(layers):
                with tf.variable_scope("Layer{}".format(layer)):
                    dim = (input_size + self._lstm_size) if layer == 0 else (self._lstm_size * 2)
                    initializer = tf.random_uniform_initializer(-np.sqrt(1./dim), np.sqrt(1./dim))
                    matrix = tf.get_variable("Matrix", [dim, self._lstm_size * 4], 
                        dtype=dtype, initializer=initializer)
                    bias = tf.get_variable("bias", [self._lstm_size * 4], 
                        initializer=tf.constant_initializer(0.0, dtype=dtype))
                    self._matrix.append(matrix)
                    self._bias.append(bias)

    def __call__(self, inputs, state, input_dropout=None, output_dropout=None, scope=None):
        cur_inputs = inputs
        new_states = []
        for layer in range(self._layers):
            cur_state = tf.slice(state, [0, self._lstm_size * 2 * layer], \
                [-1, self._lstm_size * 2])
            c, h = tf.split(1, 2, cur_state)
            if input_dropout is not None and input_dropout > 0:
                cur_inputs = tf.nn.dropout(cur_inputs, 1. - input_dropout)
            concat = tf.matmul(tf.concat(1, [cur_inputs, h]), self._matrix[layer]) + \
                self._bias[layer]

            i, j, f, o = tf.split(1, 4, concat)
            new_c = (c * tf.sigmoid(f + self._forget_bias) + 
                tf.sigmoid(i) * self._activation(j))

            cur_inputs = self._activation(new_c) * tf.sigmoid(o) # cur_inputs = new_h
            new_states.append(tf.concat(1, [new_c, cur_inputs]))
            if output_dropout is not None and output_dropout > 0:
               cur_inputs = tf.nn.dropout(cur_inputs, 1. - output_dropout) 

        new_states = tf.concat(1, new_states)
        return cur_inputs, new_states
