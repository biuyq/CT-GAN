"""Lots of GAN architectures on LSUN bedrooms."""

import os, sys
sys.path.append(os.getcwd())

N_GPUS = 8

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True, n_gpus=N_GPUS)
except ImportError:
    pass

import tflib as lib
import tflib.debug
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.adamax
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.ops.layernorm
import tflib.save_images
import tflib.mnist
import tflib.lsun_bedrooms
import tflib.small_imagenet
import tflib.imagenet

# import tflib.ops.gru

import numpy as np
import tensorflow as tf
import sklearn.datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import functools
import collections

BATCH_SIZE = 64
ITERS = 100000

DIM_128 = 32
DIM_64 = 64
DIM_32 = 128
DIM_16 = 256
DIM_8 = 512
DIM_4 = 1024

NOISE_DIM = 256

def GeneratorAndDiscriminator():
    return ResnetGenerator, ResnetDiscriminator

OUTPUT_DIM = 128*128*3

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

lib.ops.conv2d.enable_default_weightnorm()
lib.ops.deconv2d.enable_default_weightnorm()
lib.ops.linear.enable_default_weightnorm()

# ! Layers

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='glorot')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='glorot')
    return LeakyReLU(output)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, generator=False):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim, stride=2)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    if generator:
        inputs_shape = inputs.get_shape().as_list()
        noise = tf.random_normal([inputs_shape[0], 16, inputs_shape[2], inputs_shape[3]])
        output += 0.5 * lib.ops.conv2d.Conv2D(name+'.NoiseConv', input_dim=16, output_dim=input_dim, inputs=noise, filter_size=1, he_init=False)

    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=True)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=True)

    return shortcut + (0.3 * output)

# ! Generators

def ResnetGenerator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, NOISE_DIM])

    output = lib.ops.linear.Linear('Generator.Input', NOISE_DIM, 4*4*DIM_4, noise)
    output = tf.reshape(output, [-1, DIM_4, 4, 4])

    output = ResidualBlock('Generator.3', DIM_4, DIM_8, 3, output, resample='up', generator=True)
    output = ResidualBlock('Generator.3B', DIM_8, DIM_8, 3, output, resample=None, generator=True)
    output = ResidualBlock('Generator.6', DIM_8, DIM_16, 3, output, resample='up', generator=True)
    output = ResidualBlock('Generator.6B', DIM_16, DIM_16, 3, output, resample=None, generator=True)
    output = ResidualBlock('Generator.8', DIM_16, DIM_32, 3, output, resample='up', generator=True)
    output = ResidualBlock('Generator.8A', DIM_32, DIM_32, 3, output, resample=None, generator=True)
    output = ResidualBlock('Generator.11', DIM_32, DIM_64, 3, output, resample='up', generator=True)

    output = lib.ops.deconv2d.Deconv2D('Generator.Out', DIM_64, 3, 7, output, he_init=False)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

# ! Discriminators

def ResnetDiscriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 128, 128])
    output = lib.ops.conv2d.Conv2D('Discriminator.In', 3, DIM_64, 7, output, he_init=False, stride=2)

    output = ResidualBlock('Discriminator.3', DIM_64, DIM_32, 3, output, resample='down')
    output = ResidualBlock('Discriminator.3A', DIM_32, DIM_32, 3, output, resample=None)
    output = ResidualBlock('Discriminator.4', DIM_32, DIM_16, 3, output, resample='down')
    output = ResidualBlock('Discriminator.4A', DIM_16, DIM_16, 3, output, resample=None)
    output = ResidualBlock('Discriminator.5', DIM_16, DIM_8, 3, output, resample='down')
    output = ResidualBlock('Discriminator.5B', DIM_8, DIM_8, 3, output, resample=None)
    output = ResidualBlock('Discriminator.6', DIM_8, DIM_4, 3, output, resample='down')

    output = tf.reshape(output, [-1, 4*4*DIM_4])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*DIM_4, 1, output)

    return tf.reshape(output, [-1])

Generator, Discriminator = GeneratorAndDiscriminator()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 128, 128])
    if tf.__version__.startswith('1.'):
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
    else:
        split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
    gen_costs, disc_costs = [],[]

    for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
        with tf.device(device):

            real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE/len(DEVICES), OUTPUT_DIM])
            fake_data = Generator(BATCH_SIZE/len(DEVICES))

            if tf.__version__.startswith('1.'):
                disc_out = Discriminator(tf.concat([real_data, fake_data], axis=0))
            else:
                disc_out = Discriminator(tf.concat(0, [real_data, fake_data]))
            disc_real = disc_out[:BATCH_SIZE/len(DEVICES)]
            disc_fake = disc_out[BATCH_SIZE/len(DEVICES):]

            gen_cost = -tf.reduce_mean(Discriminator(fake_data))
            disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            alpha = tf.random_uniform(
                shape=[BATCH_SIZE/len(DEVICES),1], 
                minval=0.,
                maxval=1.
            )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            lipschitz_penalty = tf.reduce_mean((slopes-1.)**2)
            wgan_disc_cost = disc_cost
            disc_cost += 10*lipschitz_penalty

            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)

    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    frame_i = [0]
    fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, NOISE_DIM)).astype('float32'))
    all_fixed_noise_samples = []
    for device_index, device in enumerate(DEVICES):
        n_samples = BATCH_SIZE / len(DEVICES)
        all_fixed_noise_samples.append(Generator(n_samples, noise=fixed_noise[device_index*n_samples:(device_index+1)*n_samples]))
    if tf.__version__.startswith('1.'):
        all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
    else:
        all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)
    def generate_image(frame, true_dist):
        samples = session.run(all_fixed_noise_samples)
        samples = ((samples+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 128, 128)), 'samples_{}.png'.format(frame))


    train_gen = lib.imagenet.load(BATCH_SIZE)

    def inf_train_gen():
        while True:
            for (images,) in train_gen():
                yield images


    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()
    _disc_costs, _gen_costs, times, datatimes = [], [], [], []


    for iteration in xrange(ITERS):


        start_time = time.time()

        if iteration < 20:
            disc_iters = 20
        else:
            disc_iters = 5
        for i in xrange(disc_iters):
            data_start_time = time.time()
            _data = gen.next()
            datatimes.append(time.time() - data_start_time)
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_conv: _data})

        _disc_costs.append(_disc_cost)

        data_start_time = time.time()
        _data = gen.next()
        datatimes.append(time.time() - data_start_time)
        _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={all_real_data_conv: _data})
        _gen_costs.append(_gen_cost)

        times.append(time.time() - start_time)

        if (iteration < 20) or (iteration % 20 == 19):
            print "iter:\t{}\tdisc:\t{:.3f}\tgen:\t{:.3f}\ttime:\t{:.3f} datatime:\t{:.3f}".format(iteration, np.mean(_disc_costs), np.mean(_gen_costs), np.mean(times), np.mean(datatimes))
            _disc_costs, _gen_costs, times, datatimes = [], [], [], []

        if iteration % 100 == 0:
            generate_image(iteration, _data)