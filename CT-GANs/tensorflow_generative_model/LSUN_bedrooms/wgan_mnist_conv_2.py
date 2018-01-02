"""Generative Adversarial Network for MNIST."""

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
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.ops.batchnorm

import numpy as np
import tensorflow as tf
import sklearn.datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import functools

import inception_score

BATCH_SIZE = 100
ITERS = 100000
DIM = 16
DIM_G = 16
GRAD_LOSS = False

EVALUATOR_DISC_ITERS = 20

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ConvGenerator(n_samples):
    noise = tf.random_uniform(
        shape=[n_samples, 128], 
        minval=-np.sqrt(3),
        maxval=np.sqrt(3)
    )

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*(4*DIM_G), noise)
    output = tf.reshape(output, [-1, 4*DIM_G, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.1', 4*DIM_G, 2*DIM_G, 5, output)
    output = output[:,:,::7,::7] # crop 8x8 to 7x7
    output = tf.nn.relu(output)
    output = lib.ops.deconv2d.Deconv2D('Generator.2', 2*DIM_G, 1*DIM_G, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.deconv2d.Deconv2D('Generator.3', 1*DIM_G, 1, 5, output, he_init=False)
    output = tf.nn.sigmoid(output)

    output = tf.reshape(output, [-1, 784])
    return output

def ConvDiscriminator(name, inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])
    output = lib.ops.conv2d.Conv2D(name+'.1', 1, 1*DIM, 5, output, stride=2)
    output = LeakyReLU(output)
    output = lib.ops.conv2d.Conv2D(name+'.2', 1*DIM, 2*DIM, 5, output, stride=2)
    output = LeakyReLU(output)
    output = lib.ops.conv2d.Conv2D(name+'.3', 2*DIM, 4*DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear(name+'.Out', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

Generator = ConvGenerator

Discriminator = functools.partial(ConvDiscriminator, 'Discriminator')
Discriminator2 = functools.partial(ConvDiscriminator, 'Discriminator2')
Discriminator3 = functools.partial(ConvDiscriminator, 'Discriminator3')

real_data = tf.placeholder(tf.float32, shape=[None, 784])
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

disc_2_real = Discriminator2(real_data)
disc_2_fake = Discriminator2(fake_data)

disc_3_real = Discriminator3(real_data)
disc_3_fake = Discriminator3(fake_data)

# WGAN generator loss
gen_cost = -tf.reduce_mean(disc_fake)

# WGAN discriminator loss
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
disc_2_cost = tf.reduce_mean(disc_2_fake) - tf.reduce_mean(disc_2_real)
disc_3_cost = tf.reduce_mean(disc_3_fake) - tf.reduce_mean(disc_3_real)

# WGAN lipschitz-penalty
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
disc_cost += 10*lipschitz_penalty

gradients_2 = tf.gradients(Discriminator2(interpolates), [interpolates])[0]
slopes_2 = tf.sqrt(tf.reduce_sum(tf.square(gradients_2), reduction_indices=[1]))
lipschitz_penalty_2 = tf.reduce_mean((slopes_2-1.)**2)
disc_2_cost += 10*lipschitz_penalty_2

gradients_3 = tf.gradients(Discriminator3(interpolates), [interpolates])[0]
slopes_3 = tf.sqrt(tf.reduce_sum(tf.square(gradients_3), reduction_indices=[1]))
lipschitz_penalty_3 = tf.reduce_mean((slopes_3-1.)**2)
disc_3_cost += 10*lipschitz_penalty_3


gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator.'))
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'))
disc_2_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_2_cost, var_list=lib.params_with_name('Discriminator2.'))
disc_3_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_2_cost, var_list=lib.params_with_name('Discriminator3.'))

clip_ops = []
# for var in lib.params_with_name('Discriminator'):
    # print "Clipping {}".format(var.name)
    # clip_bounds = [-.01, .01]
    # clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
clip_disc_weights = tf.group(*clip_ops)

def generate_image(iteration):
    samples = session.run(fake_data)
    lib.save_images.save_images(samples[:100], 'samples_{}.jpg'.format(iteration))

scorer = inception_score.InceptionScore()
fake_data_1000 = Generator(1000)
def calculate_inception_score():
    samples = []
    for i in xrange(10):
        samples.append(session.run(fake_data_1000))
    samples = np.concatenate(samples, axis=0)
    return scorer.score(samples)

train_gen, _, _ = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()
    disc_costs, disc_2_costs, disc_3_costs = [], [], []

    for iteration in xrange(ITERS):

        start_time = time.time()

        if (iteration < 200):
            disc_iters = 100
        else:
            disc_iters = 5

        for i in xrange(disc_iters):
            _data = gen.next()
            _disc_cost, _disc_2_cost, _disc_3_cost, _, _, _ = session.run([disc_cost, disc_2_cost, disc_3_cost, disc_train_op, disc_2_train_op, disc_3_train_op], feed_dict={real_data: _data})
            _ = session.run([clip_disc_weights])

        for i in xrange(EVALUATOR_DISC_ITERS - disc_iters):
            _data = gen.next()
            _disc_2_cost, _disc_3_cost, _, _= session.run([disc_2_cost, disc_3_cost, disc_2_train_op, disc_3_train_op], feed_dict={real_data: _data})
            _ = session.run([clip_disc_weights])

        disc_costs.append(_disc_cost)
        disc_2_costs.append(_disc_3_cost)
        disc_3_costs.append(_disc_2_cost)

        _ = session.run([gen_train_op])

        delta_time = time.time() - start_time

        if iteration % 100 == 0:
            inception = calculate_inception_score()
            print "iter:\t{}\tdisc:\t{:.5f}\t{:.5f}\t{:.5f}\ttime:\t{:.5f}\tinception:{:.5f}".format(
                iteration, 
                np.mean(disc_costs), 
                np.mean(disc_2_costs), 
                np.mean(disc_3_costs), 
                delta_time,
                inception
            )
            disc_costs, disc_2_costs, disc_3_costs = [], [], []
            generate_image(iteration)