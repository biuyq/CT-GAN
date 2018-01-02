"""Generative Adversarial Network for MNIST."""

import os, sys
sys.path.append(os.getcwd())

N_GPUS = 2

# try: # This only matters on Ishaan's computer
import experiment_tools
experiment_tools.wait_for_gpu(tf=True, n_gpus=N_GPUS, skip=[0,1])
# except ImportError:
#     pass

import tflib as lib
import tflib.debug
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.adamax
import tflib.ops.nadam
import tflib.ops.batchnorm
import tflib.ops.layernorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.cifar10
import tflib.small_imagenet_32
import tflib.lsun_bedrooms
import tflib.inception_score
import tflib.plot

import numpy as np
import tensorflow as tf
import sklearn.datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import functools

BATCH_SIZE = 64
ITERS = 10000
# DIM = 32
# DIM_G = 32
MODE = 'dcgan' # dcgan, wgan, wgan-gp
DISC_ITERS = 1
DATASET = 'imagenet'

DIM_G = 32
DIM_D = 32
EXTRA_DEPTH_G = 0
EXTRA_DEPTH_D = 0

# These settings apply only to WGAN-GP
# GEN_LR = e-5
# DISC_LR = GEN_LR * 10
DISC_LR = 5e-4
GEN_LR = 0.2*DISC_LR
DECAY = False
NORMALIZATION_G = True
NORMALIZATION_D = True
WEIGHTNORM = False
OLD_STYLE = True
LR = 1e-4
MOMENTUM_G = 0.
MOMENTUM_D = 0.
GEN_BS_MULTIPLE = 2

MEAN_POOL = True
UPSAMPLE_CONV = True

def GeneratorAndDiscriminator():
    # return SimpleGenerator, SimpleDiscriminator
    return DCGANGenerator, DCGANDiscriminator
    # return ResnetGenerator, ResnetDiscriminator
    # return ResnetGenerator, DCGANDiscriminator

lib.print_model_settings(locals().copy())

if WEIGHTNORM:
    lib.ops.linear.enable_default_weightnorm()
    lib.ops.conv2d.enable_default_weightnorm()
    lib.ops.deconv2d.enable_default_weightnorm()

OUTPUT_DIM = 3072
DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]
# DEVICES = DEVICES[::-1] # On AWS it seems the topmost GPU runs hotter; this is a hacky workaround

def nonlinearity(x):
    # return LeakyReLU(x, alpha=0.05)
    return tf.nn.relu(x)

def Normalize(name, inputs):
    if MODE == 'wgan-gp':
        if ('Discriminator' in name) and NORMALIZATION_D:
            return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
        elif ('Generator' in name) and NORMALIZATION_G:
            return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
        else:
            return inputs
    else:
        return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return nonlinearity(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = lib.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        if MEAN_POOL:
            conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
            conv_shortcut = ConvMeanPool
        else:
            conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim, stride=2)
            conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
    elif resample=='up':
        if UPSAMPLE_CONV:
            conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
            conv_shortcut = UpsampleConv
        else:
            conv_1        = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim, output_dim=output_dim)
            conv_shortcut = SubpixelConv2D
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
    output = Normalize(name+'.N1', output)
    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(name+'.N2', output)
    output = nonlinearity(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)
    return shortcut + (.3*output)

def SimpleGenerator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM_G, noise)
    output = tf.reshape(output, [-1, 4*DIM_G, 4, 4])
    output = tf.nn.relu(output)

    for i in xrange(EXTRA_DEPTH_G):
        output = lib.ops.conv2d.Conv2D('Generator.2X_{}'.format(i), 4*DIM_G, 4*DIM_G, 3, output)
        output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM_G, 2*DIM_G, 5, output)
    output = tf.nn.relu(output)

    for i in xrange(EXTRA_DEPTH_G):
        output = lib.ops.conv2d.Conv2D('Generator.3X_{}'.format(i), 2*DIM_G, 2*DIM_G, 3, output)
        output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM_G, DIM_G, 5, output)
    output = tf.nn.relu(output)

    for i in xrange(EXTRA_DEPTH_G):
        output = lib.ops.conv2d.Conv2D('Generator.4X_{}'.format(i), DIM_G, DIM_G, 3, output)
        output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', DIM_G, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def SimpleDiscriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM_D, 5, output, stride=2)
    output = tf.nn.relu(output)

    for i in xrange(EXTRA_DEPTH_D):
        output = lib.ops.conv2d.Conv2D('Discriminator.2X_{}'.format(i), DIM_D, DIM_D, 3, output)
        output = tf.nn.relu(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM_D, 2*DIM_D, 5, output, stride=2)
    output = tf.nn.relu(output)

    for i in xrange(EXTRA_DEPTH_D):
        output = lib.ops.conv2d.Conv2D('Discriminator.3X_{}'.format(i), 2*DIM_D, 2*DIM_D, 3, output)
        output = tf.nn.relu(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM_D, 4*DIM_D, 5, output, stride=2)
    output = tf.nn.relu(output)

    for i in xrange(EXTRA_DEPTH_D):
        output = lib.ops.conv2d.Conv2D('Discriminator.4X_{}'.format(i), 4*DIM_D, 4*DIM_D, 3, output)
        output = tf.nn.relu(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM_D])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM_D, 1, output)

    return tf.reshape(output, [-1])

def DCGANGenerator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM_G, noise)
    output = tf.reshape(output, [-1, 4*DIM_G, 4, 4])
    output = Normalize('Generator.N1', output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.2Pre', 4*DIM_G, 2*DIM_G, 5, output)
    output = Normalize('Generator.N2', output)
    output = tf.nn.relu(output)

    for i in xrange(EXTRA_DEPTH_G):
        output = lib.ops.conv2d.Conv2D('Generator.X{}'.format(i), 2*DIM_G, 2*DIM_G, 3, output)
        output = Normalize('Generator.NX{}'.format(i), output)
        output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3Pre', 2*DIM_G, DIM_G, 5, output)
    output = Normalize('Generator.N4', output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4Pre', DIM_G, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def DCGANDiscriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM_D, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM_D, 2*DIM_D, 5, output, stride=2)
    output = Normalize('Discriminator.N3', output)
    output = LeakyReLU(output)

    for i in xrange(EXTRA_DEPTH_D):
        output = lib.ops.conv2d.Conv2D('Discriminator.X{}'.format(i), 2*DIM_G, 2*DIM_G, 3, output)
        output = Normalize('Discriminator.NX{}'.format(i), output)
        output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM_D, 4*DIM_D, 5, output, stride=2)
    output = Normalize('Discriminator.N5', output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM_D])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM_D, 1, output)

    return tf.reshape(output, [-1])

def DeepGenerator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM_G, noise)
    # output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.reshape(output, [-1, 4*DIM_G, 4, 4])
    output = Normalize('Generator.N1', output)
    output = nonlinearity(output)

    # output = ResidualBlock('Generator.1', 4*DIM, 2*DIM, 3, output, resample='up')
    output = lib.ops.deconv2d.Deconv2D('Generator.2Pre', 4*DIM_G, 2*DIM_G, 3, output)
    output = Normalize('Generator.N2', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.2', 2*DIM_G, 2*DIM_G, 3, output)
    output = Normalize('Generator.N3', output)
    output = nonlinearity(output)

    # output = ResidualBlock('Generator.2', 2*DIM, DIM, 3, output, resample='up')
    output = lib.ops.deconv2d.Deconv2D('Generator.3Pre', 2*DIM_G, DIM_G, 3, output)
    output = Normalize('Generator.N4', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.3', DIM_G, DIM_G, 3, output)
    output = Normalize('Generator.N5', output)
    output = nonlinearity(output)

    # output = ResidualBlock('Generator.3', DIM, DIM/2, 3, output, resample='up')
    # output = lib.ops.conv2d.Conv2D('Generator.Out', DIM/2, 3, 1, output)
    output = lib.ops.deconv2d.Deconv2D('Generator.4Pre', DIM_G, DIM_G/2, 3, output)
    output = Normalize('Generator.N6', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.4', DIM_G/2, 3, 3, output)
    # output = lib.ops.deconv2d.Deconv2D('Generator.4', DIM, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def DeepDiscriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.1Pre', 3, DIM_D/2, 3, output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Discriminator.1', DIM_D/2, DIM_D, 3, output, stride=2)
    output = Normalize('Discriminator.N1', output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2Pre', DIM_D, DIM_D, 3, output)
    output = Normalize('Discriminator.N2', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM_D, 2*DIM_D, 3, output, stride=2)
    output = Normalize('Discriminator.N3', output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3Pre', 2*DIM_D, 2*DIM_D, 3, output)
    output = Normalize('Discriminator.N4', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM_D, 4*DIM_D, 3, output, stride=2)
    output = Normalize('Discriminator.N5', output)
    output = nonlinearity(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM_D])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM_D, 1, output)

    return tf.reshape(output, [-1])

def ResnetGenerator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM_G, noise)
    output = tf.reshape(output, [-1, 4*DIM_G, 4, 4])

    for i in xrange(EXTRA_DEPTH_G):
        output = ResidualBlock('Generator.1X_{}'.format(i), 4*DIM_G, 4*DIM_G, 3, output, resample=None)
    output = ResidualBlock('Generator.1', 4*DIM_G, 2*DIM_G, 3, output, resample='up')
    for i in xrange(EXTRA_DEPTH_G):
        output = ResidualBlock('Generator.2X_{}'.format(i), 2*DIM_G, 2*DIM_G, 3, output, resample=None)
    output = ResidualBlock('Generator.2', 2*DIM_G, DIM_G, 3, output, resample='up')
    for i in xrange(EXTRA_DEPTH_G):
        output = ResidualBlock('Generator.3X_{}'.format(i), DIM_G, DIM_G, 3, output, resample=None)

    output = ResidualBlock('Generator.3', DIM_G, DIM_G/2, 3, output, resample='up')
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G/2, 3, 1, output, he_init=False)

    # output = Normalize('Generator.OutputN', output)
    # output = nonlinearity(output)
    # output = lib.ops.deconv2d.Deconv2D('Generator.Output', DIM_G, 3, 5, output, he_init=False)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def ResnetDiscriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, DIM_D/2, 1, output, he_init=False)
    output = ResidualBlock('Discriminator.1', DIM_D/2, DIM_D, 3, output, resample='down')

    # output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, DIM_D, 5, output, he_init=False, stride=2)

    for i in xrange(EXTRA_DEPTH_D):
        output = ResidualBlock('Discriminator.1X_{}'.format(i), DIM_D, DIM_D, 3, output, resample=None)
    output = ResidualBlock('Discriminator.2', DIM_D, 2*DIM_D, 3, output, resample='down')
    for i in xrange(EXTRA_DEPTH_D):
        output = ResidualBlock('Discriminator.2X_{}'.format(i), 2*DIM_D, 2*DIM_D, 3, output, resample=None)
    output = ResidualBlock('Discriminator.3', 2*DIM_D, 4*DIM_D, 3, output, resample='down')
    for i in xrange(EXTRA_DEPTH_D):
        output = ResidualBlock('Discriminator.3X_{}'.format(i), 4*DIM_D, 4*DIM_D, 3, output, resample=None)

    output = Normalize('Discriminator.OutputN', output)
    output = nonlinearity(output)
    output = tf.reshape(output, [-1, 4*4*4*DIM_D])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM_D, 1, output)

    return tf.reshape(output, [-1])

with tf.Session() as session:

    Generator, Discriminator = GeneratorAndDiscriminator()

    iteration = tf.placeholder(tf.int32, shape=None)
    all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 32, 32])

    if (len(DEVICES)%2==0) and (len(DEVICES)>=2) and (MODE == 'wgan-gp'):

        fake_data_splits = []
        for device in DEVICES:
            with tf.device(device):
                fake_data_splits.append(Generator(BATCH_SIZE/len(DEVICES)))

        all_real_data = tf.reshape(2*((tf.cast(all_real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE, OUTPUT_DIM])
        all_real_data_splits = tf.split(all_real_data, len(DEVICES)/2)

        DEVICES_B = DEVICES[:len(DEVICES)/2]
        DEVICES_A = DEVICES[len(DEVICES)/2:]

        disc_costs = []
        for i, device in enumerate(DEVICES_A):
            with tf.device(device):
                real_and_fake_data = lib.concat([all_real_data_splits[i]] + [fake_data_splits[i]] + [fake_data_splits[len(DEVICES_A)+i]], axis=0)
                disc_all = Discriminator(real_and_fake_data)
                disc_real = disc_all[:BATCH_SIZE/len(DEVICES_A)]
                disc_fake = disc_all[BATCH_SIZE/len(DEVICES_A):]
                disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))

        for i, device in enumerate(DEVICES_B):
            with tf.device(device):
                real_data = tf.identity(all_real_data_splits[i]) # transfer from gpu0
                fake_data = lib.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A)+i]], axis=0)
                alpha = tf.random_uniform(
                    shape=[BATCH_SIZE/len(DEVICES_A),1], 
                    minval=0.,
                    maxval=1.
                )
                differences = fake_data - real_data
                interpolates = real_data + (alpha*differences)
                gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = 10*tf.reduce_mean((slopes-1.)**2)
                disc_costs.append(gradient_penalty)

        disc_cost = tf.add_n(disc_costs) / len(DEVICES_A)

        if DECAY:
            decay = tf.maximum(0., 1.-(tf.cast(iteration, tf.float32)/ITERS))
        else:
            decay = 1.
        disc_train_op = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=MOMENTUM_D, beta2=0.9).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

        gen_costs = []
        for device in DEVICES:
            with tf.device(device):
                gen_costs.append(-tf.reduce_mean(Discriminator(Generator(GEN_BS_MULTIPLE*BATCH_SIZE/len(DEVICES)))))
        gen_cost = tf.add_n(gen_costs) / len(DEVICES)
        gen_train_op = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=MOMENTUM_G, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)


    else:
        split_real_data_conv = lib.split(all_real_data_conv, len(DEVICES), axis=0)

        gen_costs, disc_costs = [],[]

        for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
            with tf.device(device):

                real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE/len(DEVICES), OUTPUT_DIM])
                fake_data = Generator(BATCH_SIZE/len(DEVICES))

                if MODE == 'wgan':
                    disc_real = Discriminator(real_data)
                    disc_fake = Discriminator(fake_data)

                    gen_cost = -tf.reduce_mean(disc_fake)
                    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                elif MODE == 'wgan-gp':
                    disc_all = Discriminator(lib.concat([real_data, fake_data],0))
                    disc_real = disc_all[:tf.shape(real_data)[0]]
                    disc_fake = disc_all[tf.shape(real_data)[0]:]

                    gen_cost = -tf.reduce_mean(Discriminator(Generator(GEN_BS_MULTIPLE*BATCH_SIZE/len(DEVICES))))
                    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                    alpha = tf.random_uniform(
                        shape=[BATCH_SIZE/len(DEVICES),1], 
                        minval=0.,
                        maxval=1.
                    )
                    differences = fake_data - real_data
                    interpolates = real_data + (alpha*differences)
                    interpolates = tf.stop_gradient(interpolates)
                    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    lipschitz_penalty = 10*tf.reduce_mean((slopes-1.)**2)

                    wasserstein = disc_cost
                    disc_cost += lipschitz_penalty

                elif MODE == 'dcgan':
                    disc_real = Discriminator(real_data)
                    disc_fake = Discriminator(fake_data)

                    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
                    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
                    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
                    disc_cost /= 2.

                gen_costs.append(gen_cost)
                disc_costs.append(disc_cost)

        gen_cost = tf.add_n(gen_costs) / len(DEVICES)
        disc_cost = tf.add_n(disc_costs) / len(DEVICES)

        if MODE == 'wgan':
            gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
            disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

            clip_ops = []
            for var in lib.params_with_name('Discriminator'):
                print "Clipping {}".format(var.name)
                clip_bounds = [-.01, .01]
                clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
            clip_disc_weights = tf.group(*clip_ops)

        elif MODE == 'wgan-gp':
            if DECAY:
                decay = tf.maximum(0., 1.-(tf.cast(iteration, tf.float32)/ITERS))
            else:
                decay = 1.
            gen_train_op = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=MOMENTUM_G, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
            disc_train_op = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=MOMENTUM_D, beta2=0.9).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)
        elif MODE == 'dcgan':
            gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost, var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
            disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)


    frame_i = [0]
    fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
    fixed_noise_samples = Generator(128, noise=fixed_noise)
    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        samples = ((samples+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(samples.reshape((128, 3, 32, 32)), 'samples_{}.jpg'.format(frame))

    samples_100 = Generator(100)
    def get_inception_score(n_samples):
        all_samples = []
        for i in xrange(n_samples/100):
            all_samples.append(session.run(samples_100))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples+1.)*(255.99/2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
        np.random.shuffle(all_samples)
        result = lib.inception_score.get_inception_score(list(all_samples))
        return result

    if DATASET == 'cifar10':
        train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE)
    elif DATASET == 'imagenet':
        train_gen, dev_gen = lib.small_imagenet_32.load(BATCH_SIZE, dev_set_size=100*BATCH_SIZE)
    elif DATASET == 'lsun':
        train_gen, dev_gen = lib.lsun_bedrooms.load(BATCH_SIZE, downsample=True)

    def inf_train_gen():
        while True:
            for images, in train_gen():
                yield images


    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()

    for _iteration in xrange(ITERS):
        start_time = time.time()

        if MODE == 'wgan-gp':
            if OLD_STYLE:
                _ = session.run(
                    gen_train_op,
                    feed_dict={iteration: _iteration}
                )

                for i in xrange(CRITIC_ITERS):
                    _data = gen.next()
                    _data = _data.reshape((BATCH_SIZE,3,32,32))
                    _disc_cost, _ = session.run(
                        [disc_cost, disc_train_op], 
                        feed_dict={all_real_data_conv: _data, iteration: _iteration}
                    )
            else:
                _data = gen.next()
                _data = _data.reshape((BATCH_SIZE,3,32,32))
                _disc_cost, _, _ = session.run(
                    [disc_cost, disc_train_op, gen_train_op], 
                    feed_dict={all_real_data_conv: _data, iteration: _iteration}
                )
        else:
            if _iteration > 0:
                _ = session.run(gen_train_op, feed_dict={iteration:_iteration})

            for i in xrange(DISC_ITERS):
                _data = gen.next()
                _data = _data.reshape((BATCH_SIZE,3,32,32))
                _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_conv: _data, iteration: _iteration})
                if clip_disc_weights is not None:
                    _ = session.run(clip_disc_weights)

        lib.plot.plot('cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate a very rough inception score every 100 iters
        if _iteration % 100 == 99:
            inception_score = get_inception_score(200)
            lib.plot.plot('inception', inception_score[0])

        # Calculate dev loss and generate samples every 100 iters
        if _iteration % 1000 == 999:
            # dev_disc_costs = []
            # for images, in dev_gen():
            #     _dev_disc_cost = session.run(disc_cost, feed_dict={real_data_conv: images})
            #     dev_disc_costs.append(_dev_disc_cost)
            # lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(_iteration, _data)

        if (_iteration < 100) or (_iteration % 1000 == 999):
        # if _iteration % 1000 == 999:
            lib.plot.flush(print_stds=True)

        lib.plot.tick()

    print "Final inception score: {}".format(get_inception_score(50000))