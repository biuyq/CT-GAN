import tflib as lib
import tflib.ops.linear

import numpy as np
import tensorflow as tf

class RNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, name, n_in, n_hid, activation=tf.tanh):
        self._n_in = n_in
        self._n_hid = n_hid
        self._activation = activation
        self._name = name

    @property
    def state_size(self):
        return self._n_hid

    @property
    def output_size(self):
        return self._n_hid

    def __call__(self, inputs, state, scope=None):
        """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""
        output = self._activation(
            lib.ops.linear.Linear(self._name + '.InputToHidden', self._n_in, self._n_hid, inputs) + \
            lib.ops.linear.Linear(self._name + '.HiddenToHidden', self._n_hid, self._n_hid, state)
        )
        return output, output

def RNN(name, n_in, n_hid, inputs):
    h0 = lib.param(name+'.h0', np.zeros(n_hid, dtype='float32'))
    batch_size = tf.shape(inputs)[0]
    h0 = tf.reshape(tf.tile(h0, tf.pack([batch_size])), tf.pack([batch_size, n_hid]))
    return tf.nn.dynamic_rnn(RNNCell(name, n_in, n_hid), inputs, initial_state=h0, swap_memory=True)[0]