"""Generative Adversarial Network for MNIST."""

# Goal: push training as fast as possible while maintaining stability
# Attempt 1: baseline. batch size 200 to reduce SGD noise. curve looks very very stable.
# noBeta2: set adam's beta2 to default value. seems to work just as well, maybe a little better? (curve drops a little faster)
# highLR: increase LR to 5e-4. learning happens way faster, still reasonably stable.
# relu: replace leakyrelu with relu in D. 
# bn: batchnorm everywhere: looks substantially less stable, actually...
# bnG: generator only batchnorm. looks incredibly stable.
# bnD: discrim only batchnorm. looks incredibly unstable.
# cIters10: no bn anywhere, 10x critic iters. not profoundly more stable, maybe a little more?
# cIters1: 1x critic iters. pretty unstable... but it converges to decent digits, surprisingly?
# cIters1_lambda1: doesn't seem to help, really.
# cIters1_5xLR: critic iters 1, gen LR 1/5th of disc LR
# cIters1_tanh: critic iters 1x, tanh everywhere. annoyingly, seems to work [though the loss curve is kind of bad]
# cIters1_tanh_noGradPenalty: no grad penalty. loss is now totally meaningless, but the motherfucker still kind of works?


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
import tflib.ops.conv2d
import tflib.ops.adamax
import tflib.ops.batchnorm
import tflib.ops.layernorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.lsun_bedrooms
import tflib.plot

import numpy as np
import tensorflow as tf
import sklearn.datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import functools

BATCH_SIZE = 200
ITERS = 10000
# DIM = 32
# DIM_G = 32
MODE = 'wgan-gp' # dcgan, wgan, wgan-gp
KEEP_PROB = 1.

DIM = 64
OUTPUT_DIM = 28*28

def Batchnorm(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception()
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def FCGenerator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = ReLULayer('Generator.1', 128, 512, noise)
    output = ReLULayer('Generator.2', 512, 512, output)
    output = ReLULayer('Generator.3', 512, 512, output)
    output = lib.ops.linear.Linear('Generator.Out', 512, OUTPUT_DIM, output)

    output = tf.nn.sigmoid(output)

    return output

def FCDiscriminator(inputs):

    output = LeakyReLULayer('Discriminator.1', OUTPUT_DIM, 512, inputs)
    output = LeakyReLULayer('Discriminator.2', 512, 512, output)
    output = LeakyReLULayer('Discriminator.3', 512, 512, output)
    output = lib.ops.linear.Linear('Discriminator.Out', OUTPUT_DIM, 1, output)

    return tf.reshape(output, [-1])

def DCGANGenerator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    # if MODE == 'wgan':
    # output = Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    # output = tf.tanh(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    # if MODE == 'wgan':
    # output = Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    # output = tf.tanh(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    # if MODE == 'wgan':
    # output = Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)
    # output = tf.tanh(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def DCGANDiscriminator(inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 1, DIM, 5, output, stride=2)
    output = tf.nn.relu(output)
    # output = tf.tanh(output)

    # output = tf.nn.dropout(output, KEEP_PROB, noise_shape=[BATCH_SIZE, DIM, 1, 1])

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    # if MODE == 'wgan':
    # output = Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    # output = tf.tanh(output)

    # output = tf.nn.dropout(output, KEEP_PROB, noise_shape=[BATCH_SIZE, 2*DIM, 1, 1])

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    # if MODE == 'wgan':
    # output = Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)
    # output = tf.tanh(output)

    # output = tf.nn.dropout(output, KEEP_PROB, noise_shape=[BATCH_SIZE, 4*DIM, 1, 1])

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

Generator = DCGANGenerator
Discriminator = DCGANDiscriminator

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'))

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        print "Clipping {}".format(var.name)
        clip_bounds = [-.01, .01]
        clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
    clip_disc_weights = tf.group(*clip_ops)

    wasserstein = tf.constant(0.)
    lipschitz_penalty = tf.constant(0.)

elif MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    lipschitz_penalty = tf.reduce_mean((slopes-1.)**2)
    wasserstein = disc_cost
    # disc_cost += 10*lipschitz_penalty

    gen_train_op = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'))

    clip_disc_weights = None

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_train_op = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'))

frame_i = [0]
fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples = Generator(128, noise=fixed_noise)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    # samples = ((samples+1.)*(255./2)).astype('int32')
    lib.save_images.save_images(samples.reshape((128, 28, 28)), 'samples_{}.png'.format(frame))


train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _data = gen.next()
            _ = session.run([gen_train_op], feed_dict={real_data: _data})

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = 1
        for i in xrange(disc_iters):
            _data = gen.next()
            _disc_cost, _wasserstein, _lipschitz_penalty, _ = session.run([disc_cost, wasserstein, lipschitz_penalty, disc_train_op], feed_dict={real_data: _data})
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('tr disc cost', _disc_cost)

        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs, dev_wassersteins, dev_lipschitz_penalties = [], [], []
            for images,_ in dev_gen():
                _dev_disc_cost, _dev_wasserstein, _dev_lipschitz_penalty = session.run([disc_cost, wasserstein, lipschitz_penalty], feed_dict={real_data: images}) 
                dev_disc_costs.append(_dev_disc_cost)
                dev_wassersteins.append(_dev_wasserstein)
                dev_lipschitz_penalties.append(_dev_lipschitz_penalty)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration, _data)

        lib.plot.flush()

        lib.plot.tick()