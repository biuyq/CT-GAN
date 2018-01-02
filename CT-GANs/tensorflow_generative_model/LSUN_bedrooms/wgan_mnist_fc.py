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
import tflib.save_images
import tflib.mnist

import numpy as np
import tensorflow as tf
import sklearn.datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import functools

BATCH_SIZE = 100
ITERS = 100000
DIM = 16
DIM_G = 64

def LeakyReLU(x, alpha=0.25):
    return tf.maximum(alpha*x, x)

def GenLayer(name, n_in, n_out, inputs, alpha=0.25):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, 2*n_out, inputs)
    output_1, output_2 = tf.split(1,2,output)
    return tf.nn.sigmoid(output_1) * output_2

def DiscLayer(name, n_in, n_out, inputs, alpha=0.25):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.tanh(output)

def DiscLayer2(name, n_in, n_out, inputs, alpha=0.25):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def FCGenerator(n_samples):
    noise = tf.random_uniform(
        shape=[n_samples, 128], 
        minval=-np.sqrt(3),
        maxval=np.sqrt(3)
    )

    output = GenLayer('Generator.1', 128, 512, noise)
    output = GenLayer('Generator.2', 512, 512, output)
    output = GenLayer('Generator.3', 512, 512, output)
    output = lib.ops.linear.Linear('Generator.Out', 512, 784, output)

    output = tf.nn.sigmoid(output)

    return output

def FCDiscriminator(inputs, name='Discriminator'):

    output = DiscLayer(name+'.1', 784, 512, inputs)
    output = DiscLayer(name+'.2', 512, 512, output)
    output = DiscLayer(name+'.3', 512, 512, output)
    output = lib.ops.linear.Linear(name+'.Out', 512, 1, output)

    return tf.reshape(output, [-1])

def FCDiscriminator2(inputs, name='Discriminator2'):

    output = DiscLayer(name+'.1', 784, 512, inputs)
    output = DiscLayer(name+'.2', 512, 512, output)
    output = DiscLayer(name+'.3', 512, 512, output)
    output = lib.ops.linear.Linear(name+'.Out', 512, 1, output)

    return tf.reshape(output, [-1])

def FCDiscriminator3(inputs, name='Discriminator3'):

    output = DiscLayer2(name+'.1', 784, 512, inputs)
    output = DiscLayer2(name+'.2', 512, 512, output)
    output = DiscLayer2(name+'.3', 512, 512, output)
    output = lib.ops.linear.Linear(name+'.Out', 512, 1, output)

    return tf.reshape(output, [-1])

# def SubpixelConv2D(*args, **kwargs):
#     kwargs['output_dim'] = 4*kwargs['output_dim']
#     output = lib.ops.conv2d.Conv2D(*args, **kwargs)
#     output = tf.transpose(output, [0,2,3,1])
#     output = tf.depth_to_space(output, 2)
#     output = tf.transpose(output, [0,3,1,2])
#     return output

# def ResBlock(name, dim, inputs):
#     output = tf.nn.relu(inputs)
#     output = lib.ops.conv2d.Conv2D(name+'.1', dim, dim, 3, output)
#     output = tf.nn.relu(output)
#     output = lib.ops.conv2d.Conv2D(name+'.2', dim, dim, 3, output)
#     return output + inputs

# def ResBlockG(name, dim, inputs):
#     output = tf.nn.relu(inputs)
#     output = lib.ops.conv2d.Conv2D(name+'.1', dim, dim, 3, output)
#     output = tf.nn.relu(output)
#     output = lib.ops.conv2d.Conv2D(name+'.2', dim, dim, 3, output)
#     return output + inputs

# def ResBlockDownsample(name, dim, output_dim, inputs):
#     output = tf.nn.relu(inputs)
#     output = lib.ops.conv2d.Conv2D(name+'.1', dim, dim, 3, output)
#     output = tf.nn.relu(output)
#     output = lib.ops.conv2d.Conv2D(name+'.2', dim, output_dim, 3, output, stride=2)
#     return output + lib.ops.conv2d.Conv2D(name+'.skip', dim, output_dim, 1, inputs, stride=2)

# def ResBlockUpsample(name, dim, output_dim, inputs):
#     output = tf.nn.relu(inputs)
#     output = SubpixelConv2D(name+'.1', input_dim=dim, output_dim=output_dim, filter_size=3, inputs=output)
#     output = tf.nn.relu(output)
#     output = lib.ops.conv2d.Conv2D(name+'.2', output_dim, output_dim, 3, output)
#     return output + SubpixelConv2D(name+'.skip', input_dim=dim, output_dim=output_dim, filter_size=1, inputs=inputs)

# def ConvGenerator(n_samples):
#     noise = tf.random_uniform(
#         shape=[n_samples, 128], 
#         minval=-np.sqrt(3),
#         maxval=np.sqrt(3)
#     )

#     output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*(8*DIM_G), noise)
#     output = tf.reshape(output, [-1, 8*DIM_G, 4, 4])

#     output = ResBlockG('Generator.1Pre', 8*DIM_G, output)
#     output = ResBlockG('Generator.1', 8*DIM_G, output)
#     output = ResBlockUpsample('Generator.2', 8*DIM_G, 4*DIM_G, output)
#     output = output[:, :, :7, :7]
#     output = ResBlockG('Generator.3', 4*DIM_G, output)
#     output = ResBlockG('Generator.4', 4*DIM_G, output)
#     output = ResBlockUpsample('Generator.5', 4*DIM_G, 2*DIM_G, output)
#     output = ResBlockG('Generator.6', 2*DIM_G, output)
#     output = ResBlockG('Generator.7', 2*DIM_G, output)
#     output = ResBlockUpsample('Generator.8', 2*DIM_G, DIM_G, output)
#     output = ResBlockG('Generator.9', DIM_G, output)

#     output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 1, 1, output, he_init=False)
#     output = tf.nn.sigmoid(output / 5.)

#     output = tf.reshape(output, [-1, 784])
#     return output

# def ConvDiscriminator(inputs):
#     output = tf.reshape(inputs, [-1, 1, 28, 28])
#     output = lib.ops.conv2d.Conv2D('Discriminator.Input', 1, DIM, 1, output)

#     output = ResBlock('Discriminator.1', DIM, output)
#     output = ResBlockDownsample('Discriminator.2', DIM, 2*DIM, output)
#     output = ResBlock('Discriminator.3', 2*DIM, output)
#     output = ResBlock('Discriminator.4', 2*DIM, output)
#     output = ResBlockDownsample('Discriminator.5', 2*DIM, 4*DIM, output)
#     output = ResBlock('Discriminator.6', 4*DIM, output)
#     output = ResBlock('Discriminator.7', 4*DIM, output)
#     output = ResBlockDownsample('Discriminator.8', 4*DIM, 8*DIM, output)
#     output = ResBlock('Discriminator.9', 8*DIM, output)
#     # output = ResBlock('Discriminator.9Post', 8*DIM, output)

#     output = tf.reshape(output, [-1, 4*4*(8*DIM)])
#     output = lib.ops.linear.Linear('Discriminator.Out', 4*4*(8*DIM), 1, output)

#     return tf.reshape(output, [-1])

Generator = FCGenerator

import functools
Discriminator = FCDiscriminator
Discriminator2 = FCDiscriminator2
Discriminator3 = FCDiscriminator3

real_data = tf.placeholder(tf.float32, shape=[None, 784])
fake_data = Generator(BATCH_SIZE)

disc_out = Discriminator(tf.concat(0, [real_data, fake_data]))
disc_real = disc_out[:BATCH_SIZE]
disc_fake = disc_out[BATCH_SIZE:2*BATCH_SIZE]

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

interpolates_batch = interpolates
gradients = tf.gradients(Discriminator(interpolates_batch), [interpolates_batch])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
lipschitz_penalty = tf.reduce_mean((slopes-1.)**2)
wgan_disc_cost = disc_cost
disc_cost += 10*lipschitz_penalty
lipschitz_penalty = tf.reduce_mean(slopes)

gradients_2 = tf.gradients(Discriminator2(interpolates), [interpolates])[0]
slopes_2 = tf.sqrt(tf.reduce_sum(tf.square(gradients_2), reduction_indices=[1]))
lipschitz_penalty_2 = tf.reduce_mean((slopes_2-1.)**2)
wgan_disc_2_cost = disc_2_cost
disc_2_cost += 10*lipschitz_penalty_2
lipschitz_penalty_2 = tf.reduce_mean(slopes_2)

gradients_3 = tf.gradients(Discriminator3(interpolates), [interpolates])[0]
slopes_3 = tf.sqrt(tf.reduce_sum(tf.square(gradients_3), reduction_indices=[1]))
lipschitz_penalty_3 = tf.reduce_mean((slopes_3-1.)**2)
wgan_disc_3_cost = disc_3_cost
disc_3_cost += 10*lipschitz_penalty_3
lipschitz_penalty_3 = tf.reduce_mean(slopes_3)


if len(lib.params_with_name('Generator')):
    gen_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(gen_cost, var_list=lib.params_with_name('Generator.'))
    # gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
else:
    gen_train_op = None
disc_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'))
disc_2_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(disc_2_cost, var_list=lib.params_with_name('Discriminator2.'))
disc_3_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(disc_3_cost, var_list=lib.params_with_name('Discriminator3.'))

# disc_train_op = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))

frame_i = [0]
def generate_image(frame, true_dist):
    samples = session.run(fake_data)
    lib.save_images.save_images(samples[:100], 'samples_{}.jpg'.format(frame))

train_gen, _, _ = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    def generate_samples(iteration):
        samples = session.run(fake_images)
        lib.save_images.save_images(samples.reshape((-1,28,28)), 'samples_{}.jpg'.format(iteration))

    gen = inf_train_gen()
    disc_costs, wgan_disc_costs, lipschitz_penalties, disc_2_costs, wgan_disc_2_costs, lipschitz_penalties_2, disc_3_costs, wgan_disc_3_costs, lipschitz_penalties_3, gen_costs = [], [], [], [], [], [], [], [], [], []

    start_time = time.time()

    for iteration in xrange(ITERS):
        _data = gen.next()

        if iteration % 2 == 0:
            if (iteration < 200):
                disc_iters = 100
            else:
                disc_iters = 5
            for i in xrange(disc_iters):
                # _disc_cost, _wgan_disc_cost, _lipschitz_penalty, _ = session.run([disc_cost, wgan_disc_cost, lipschitz_penalty, disc_train_op], feed_dict={real_data: _data})
                _disc_cost, _wgan_disc_cost, _lipschitz_penalty, _disc_2_cost, _wgan_disc_2_cost, _lipschitz_penalty_2, _disc_3_cost, _wgan_disc_3_cost, _lipschitz_penalty_3, _, _, _ = session.run([disc_cost, wgan_disc_cost, lipschitz_penalty, disc_2_cost, wgan_disc_2_cost, lipschitz_penalty_2, disc_3_cost, wgan_disc_3_cost, lipschitz_penalty_3, disc_train_op, disc_2_train_op, disc_3_train_op], feed_dict={real_data: _data})
                _data = gen.next()
            disc_costs.append(_disc_cost)
            wgan_disc_costs.append(_wgan_disc_cost)
            lipschitz_penalties.append(_lipschitz_penalty)
            disc_2_costs.append(_disc_2_cost)
            wgan_disc_2_costs.append(_wgan_disc_2_cost)
            lipschitz_penalties_2.append(_lipschitz_penalty_2)
            disc_3_costs.append(_disc_3_cost)
            wgan_disc_3_costs.append(_wgan_disc_3_cost)
            lipschitz_penalties_3.append(_lipschitz_penalty_3)

        else:
            if gen_train_op is not None:
                _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={real_data: _data})
                gen_costs.append(_gen_cost)

        if iteration % 100 == 0:
            print "iter:\t{}\tdisc:\t{:.3f}\t{:.3f}\t{:.3f}\tgen:\t{:.3f}\ttime:\t{:.3f}".format(iteration, np.mean(disc_costs), np.mean(disc_2_costs), np.mean(disc_3_costs), np.mean(gen_costs), time.time() - start_time)
            disc_costs, wgan_disc_costs, lipschitz_penalties, disc_2_costs, wgan_disc_2_costs, lipschitz_penalties_2, disc_3_costs, wgan_disc_3_costs, lipschitz_penalties_3, gen_costs = [], [], [], [], [], [], [], [], [], []
            generate_image(iteration, _data)
            start_time = time.time()