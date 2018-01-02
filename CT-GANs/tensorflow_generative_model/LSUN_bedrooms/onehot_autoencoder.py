import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True)
except ImportError:
    pass

import tflib as lib
import tflib.debug
import tflib.ops.linear
import tflib.ops.batchnorm

import numpy as np
import tensorflow as tf
import sklearn.datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import functools

BATCH_SIZE = 128
ITERS = 100000
OUTPUT_DIM = 256

# MODE = 'lsgan' # dcgan, wgan, wgan++, lsgan

def ReLULayer(name, n_in, n_out, inputs, alpha=0., wn=False):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, weightnorm=wn)
    output = tf.nn.relu(output)
    return output


def Autoencoder(inputs):
    DIM = 256
    BOTTLENECK = 32
    output = inputs
    # output = ReLULayer('Discriminator.1', OUTPUT_DIM, DIM, output)
    output = lib.ops.linear.Linear('Discriminator.2', OUTPUT_DIM, BOTTLENECK, output)
    # output = ReLULayer('Discriminator.3', BOTTLENECK, DIM, output)
    output = lib.ops.linear.Linear('Discriminator.Out', BOTTLENECK, OUTPUT_DIM, output)
    return output

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])

outs = Autoencoder(real_data)

disc_cost = tf.nn.softmax_cross_entropy_with_logits(logits=Autoencoder(real_data), labels=real_data)

disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))

# Dataset iterator
def inf_train_gen():
    while True:
        mb = []
        for i in xrange(BATCH_SIZE):
            example = np.zeros(OUTPUT_DIM, dtype='float32')
            example[np.random.randint(OUTPUT_DIM)] = 1.
            mb.append(example)
        yield np.array(mb, dtype='float32')

# Train loop!
with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()
    disc_costs, all_disc_gradients, gen_costs = [], [], []
    for iteration in xrange(ITERS):
        _data = gen.next()

        disc_iters = 8 # you should be able to reduce this dramatically but let's be safe
        for i in xrange(disc_iters):
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data: _data})
            disc_costs.append(_disc_cost)
            # all_disc_gradients.append(__disc_gradients)

        # if gen_train_op is not None:
        #     _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={real_data: _data})
        #     gen_costs.append(_gen_cost)

        if iteration % 100 == 0:
            print "iter:{}\tdisc:{:.6f} disc_gradients:{:.3f}\tgen:{:.3f}\tscore:{:.5f}".format(iteration, np.mean(disc_costs), np.mean(all_disc_gradients), np.mean(gen_costs), 0.)
            disc_costs, all_disc_gradients, gen_costs = [], [], []