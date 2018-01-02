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
ITERS = 10000
OUTPUT_DIM = 512

# MODE = 'lsgan' # dcgan, wgan, wgan++, lsgan

def gumbel_softmax_logits(logits, temp):
    """gumbel-softmax, minus the final softmax step"""
    gumbel_noise = -tf.log(-tf.log(tf.random_uniform(tf.shape(logits), minval=.1, maxval=.99)))
    logits += gumbel_noise
    logits /= temp # gumbel temp; same shape as logits (or broadcastable)
    return logits

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def MLayer(name, n_in, n_out, inputs, alpha=0., wn=False):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, weightnorm=wn)
    output2 = lib.ops.linear.Linear(name+'.Linear2', n_in, n_out, inputs, weightnorm=wn)
    # output = tf.nn.relu(output)
    output = output * output2
    # output = output * tf.nn.sigmoid(output2)
    # output = tf.tanh(output) * tf.nn.sigmoid(output2)
    return output

def ReLULayer(name, n_in, n_out, inputs, alpha=0., wn=False):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, weightnorm=wn)
    output = tf.nn.relu(output)
    return output

def MiniLayer(name, n_in, n_out, inputs, alpha=0., wn=False):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, weightnorm=wn)
    output = LeakyReLU(output)
    return output

def Generator(n_samples, softmax=True):
    # print 'warning noise generator'
    # if softmax:
    #     return tf.nn.softmax(tf.random_normal([n_samples, OUTPUT_DIM]))
    # else:
    #     return tf.random_normal([n_samples, OUTPUT_DIM])

    # normal
    noise = tf.random_normal([n_samples, 128])
    # gumbel
    # noise = -tf.log(-tf.log(tf.random_uniform([n_samples, 128], minval=1e-7, maxval=1-1e-7)))
    # uniform
    # noise = tf.random_uniform([n_samples, 128], minval=-2, maxval=2)
    # one-hot
    # noise = tf.nn.softmax(gumbel_softmax_logits(tf.random_normal([n_samples, 512]), 0.01))
    # noise *= 10.

    DIM = 256
    output = ReLULayer('Generator.1', noise.get_shape().as_list()[1], DIM, noise)
    output = ReLULayer('Generator.2', DIM, DIM, output)
    output = MLayer('Generator.3', DIM, DIM, output)
    output = MLayer('Generator.4', DIM, DIM, output)
    output = MLayer('Generator.5', DIM, DIM, output)
    # output = MLayer('Generator.6', DIM, DIM, output)
    # output = MLayer('Generator.7', DIM, DIM, output)

    output = lib.ops.linear.Linear('Generator.Out', DIM, OUTPUT_DIM, output)

    # output = lib.ops.linear.Linear('Generator.Out', DIM, OUTPUT_DIM+1, output)
    # output = gumbel_softmax_logits(output[:,1:], 1e-1 + tf.nn.softplus(output[:,:1]))

    if softmax:
        output = tf.nn.softmax(output)

    return output

def Discriminator(inputs):
    DIM = 256

    # entropy = -tf.reduce_mean(inputs * tf.log(inputs), axis=1, keep_dims=True)
    # inputs = tf.concat([entropy, inputs], axis=1)

    # output = tf.reshape(inputs, [-1, 1])
    # output = MiniLayer('Discriminator.Pre1', 1, 8, output)
    # output = MiniLayer('Discriminator.Pre3', 8, 1, output)
    # output = tf.reshape(output, [-1, OUTPUT_DIM])
    inputs = (.99*inputs) + (.01/OUTPUT_DIM) # smooth dist. to prevent nans
    output = inputs*tf.log(inputs)# + ((1-inputs)*tf.log(1-inputs))
    output = tf.concat([output, inputs], 1)

    # output = MLayer('Discriminator.1', OUTPUT_DIM, DIM, inputs)
    output = ReLULayer('Discriminator.2', 2*OUTPUT_DIM, DIM, output)
    output = ReLULayer('Discriminator.3', DIM, DIM, output)
    output = ReLULayer('Discriminator.4', DIM, DIM, output)

    output = lib.ops.linear.Linear('Discriminator.Out', DIM, 1, output)
    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE)

disc_out = Discriminator(tf.concat([fake_data, real_data], 0))
disc_fake, disc_real = disc_out[:BATCH_SIZE], disc_out[BATCH_SIZE:]

# WGAN loss: disc tries to push fakes down and reals up, gen tries to push fakes up
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(Discriminator(Generator(4*BATCH_SIZE)))

# WGAN gradient loss term (this is my thing)
alpha = tf.random_uniform(
    shape=[BATCH_SIZE,1], 
    minval=0.,
    maxval=1.
)
differences = fake_data - real_data
interpolates = real_data + (alpha*differences)
disc_interpolates = Discriminator(interpolates)
gradients = tf.gradients(disc_interpolates, [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
lipschitz_penalty = tf.reduce_mean((slopes-1)**2)
disc_cost += 10*lipschitz_penalty
disc_gradients = tf.reduce_mean(slopes)

if len(lib.params_with_name('Generator')):
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
else:
    gen_train_op = None
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))

# Dataset iterator
def inf_train_gen():
    random_dist = np.random.uniform(size=OUTPUT_DIM)
    random_dist /= np.sum(random_dist)
    while True:
        mb = []
        for i in xrange(BATCH_SIZE):
            example = np.zeros(OUTPUT_DIM, dtype='float32')
            # example[np.argmax(np.random.uniform() < np.cumsum(random_dist))] = 1.
            example[np.random.randint(OUTPUT_DIM)] = 1.
            # print example
            mb.append(example)
        yield np.array(mb, dtype='float32')

fake_data_10K = Generator(10000, softmax=False)
def score():
    '''based on inception score; KL(p(x|z)||p(x))'''
    all_logits = []
    for i in xrange(5):
        all_logits.append(session.run(fake_data_10K))
    all_logits = np.concatenate(all_logits, axis=0).astype('float64')
    probs = np.exp(all_logits - np.max(all_logits, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    probs = (.99998*probs)+.00001 # smooth them ever so slightly to prevent nans
    kl = probs * (np.log(probs) - np.log(np.mean(probs, axis=0, keepdims=True)))
    return np.exp(np.mean(np.sum(kl, axis=1)))


# Train loop!
with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()
    disc_costs, all_disc_gradients, gen_costs = [], [], []
    for iteration in xrange(ITERS):
        _data = gen.next()

        disc_iters = 16 # you should be able to reduce this dramatically but let's be safe
        for i in xrange(disc_iters):
            _disc_cost, __disc_gradients, _ = session.run([disc_cost, disc_gradients, disc_train_op], feed_dict={real_data: _data})
            disc_costs.append(_disc_cost)
            all_disc_gradients.append(__disc_gradients)

        if gen_train_op is not None:
            _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={real_data: _data})
            gen_costs.append(_gen_cost)

        if iteration % 100 == 0:
            print "iter:{}\tdisc:{:.6f} disc_gradients:{:.3f}\tgen:{:.3f}\tscore:{:.5f}".format(iteration, np.mean(disc_costs), np.mean(all_disc_gradients), np.mean(gen_costs), score())
            disc_costs, all_disc_gradients, gen_costs = [], [], []