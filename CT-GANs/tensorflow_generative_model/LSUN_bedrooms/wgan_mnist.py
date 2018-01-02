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

def LeakyReLU(x, alpha):
    return tf.maximum(alpha*x, x)

def PReLU(name, x, dim):
    alpha = lib.param(name+'.alpha', .25*np.ones(dim, dtype='float32'))
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs, alpha=0.25):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    # return PReLU(name+'.prelu', output, n_out)
    return LeakyReLU(output, alpha=alpha)

def Generator(n_samples):
    noise = tf.random_uniform(
        shape=[n_samples, 128], 
        minval=-np.sqrt(3),
        maxval=np.sqrt(3)
    )

    output = ReLULayer('Generator.1', 128, 1024, noise)
    output = ReLULayer('Generator.2', 1024, 1024, output)
    output = ReLULayer('Generator.3', 1024, 1024, output)
    output = lib.ops.linear.Linear('Generator.Out', 1024, 784, output)

    output = tf.nn.sigmoid(output)

    return output

    # return 8. * tf.random_normal([n_samples, 2])

def Discriminator(inputs):
    # inputs /= 10.

    # RANGE = 50
    # RES = 4
    # field = lib.param('Discriminator.field', np.zeros((2*RANGE*RES, 2*RANGE*RES), dtype='float32'))
    # # field = tf.tanh(field)
    # field = tf.cumsum(tf.cumsum(field, axis=0), axis=1)

    # scaled_inputs = ((inputs + RANGE) * RES)
    # keys = tf.cast(scaled_inputs, tf.int32)
    

    # x_00 = tf.gather_nd(field, keys)
    # x_01 = tf.gather_nd(field, keys + [[0,1]])
    # x_10 = tf.gather_nd(field, keys + [[1,0]])
    # x_11 = tf.gather_nd(field, keys + [[1,1]])

    # residuals = scaled_inputs - tf.cast(keys, tf.float32)
    # res_x = residuals[:,0]
    # res_y = residuals[:,1]
    # v1 = x_00 + ((x_01 - x_00)*res_y)
    # v2 = x_10 + ((x_11 - x_10)*res_y)
    # return v1 + ((v2 - v1)*res_x)

    # inputs += tf.random_normal(tf.shape(inputs))*0.05

    output = ReLULayer('Discriminator.1', 784, 1024, inputs)
    # output = tf.nn.dropout(output, 0.5)
    output = ReLULayer('Discriminator.2', 1024, 1024, output)
    # output = tf.nn.dropout(output, 0.5)
    output = ReLULayer('Discriminator.3', 1024, 1024, output)
    # output = tf.nn.dropout(output, 0.5)
    output = lib.ops.linear.Linear('Discriminator.Out', 1024, 1, output)
    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[None, 784])
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data) 
disc_fake = Discriminator(fake_data)

# Gen objective:  push D(fake) to one
# gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
gen_cost = tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)
# gen_cost = -tf.reduce_mean(disc_fake)

# Discrim objective: push D(fake) to zero, and push D(real) to one
# disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
# disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
# disc_cost /= 2.
# WGAN loss
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
# L2 loss: push fakes to zero, reals to one
# disc_cost = tf.reduce_mean((disc_fake)**2) + tf.reduce_mean((disc_real-1)**2)
# disc_cost = tf.reduce_mean(disc_fake) + tf.reduce_mean((disc_real-1)**2)

# WGAN lipschitz-penalty
# epsilon = 1
alpha = tf.random_uniform(
    shape=[BATCH_SIZE,1], 
    minval=0.,
    maxval=1.
)
differences = fake_data - real_data
interpolates = real_data + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
# print slopes.get_shape()
# interpolates_1 = real_data + ((alpha-epsilon)*differences)
# interpolates_2 = real_data + ((alpha+epsilon)*differences)
# slopes = tf.abs((Discriminator(interpolates_2)-Discriminator(interpolates_1))/(2*epsilon))
# lipschitz_penalty = tf.reduce_mean(tf.maximum(1.,slopes))
# lipschitz_penalty = tf.reduce_mean(tf.maximum(0.,(slopes-1.)**1))
lipschitz_penalty = tf.reduce_mean((slopes-1.)**2)
wgan_disc_cost = disc_cost
disc_cost += 10*lipschitz_penalty
lipschitz_penalty = tf.reduce_mean(slopes)



# gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
# disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))
if len(lib.params_with_name('Generator')):
    gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-4).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
else:
    gen_train_op = None
disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-4).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))

assigns = []
for var in lib.params_with_name('Discriminator'):
    if '.b' not in var.name:
        print "Clipping {}".format(var.name)
        if 'alpha' in var.name:
            clip_bounds = [0., 1.]
            assigns.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))

        else:
            # assigns.append(tf.assign(var, tf.clip_by_norm(var,1.0)))
            clip_bounds = [-.01, .01]
            assigns.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
clip_disc_weights = tf.group(*assigns)

# clips = []
# for var in lib.params_with_name('Discriminator'):
#     if '.b' not in var.name:
#         print "Clipping {}".format(var.name)
#         clips.append(var)
# # clipped, _ = tf.clip_by_global_norm(clips, 5.0)
# clipped = [tf.clip_by_norm(x, 5.0) for x in clips]
# assigns = [tf.assign(old,new) for old,new in zip(clips,clipped)]
# clip_disc_weights = tf.group(*assigns)


frame_i = [0]
def generate_image(frame, true_dist):
    # N_POINTS = 128
    # RANGE = 30

    # points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    # points[:,:,0] = np.linspace(-RANGE, RANGE, N_POINTS)[:,None]
    # points[:,:,1] = np.linspace(-RANGE, RANGE, N_POINTS)[None,:]
    # samples, disc_map = session.run([fake_data, disc_real], feed_dict={real_data:points.reshape((-1, 2))})

    # plt.clf()
    # plt.imshow(disc_map.reshape((N_POINTS, N_POINTS)).T[::-1, :], extent=[-RANGE, RANGE, -RANGE, RANGE], cmap='seismic', vmin=np.min(disc_map), vmax=np.max(disc_map))
    # plt.colorbar()
    # plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange',  marker='+')
    # plt.scatter(samples[:, 0],    samples[:, 1],    c='green', marker='+')
    # plt.savefig('frame'+str(frame_i[0])+'.jpg')
    # frame_i[0] += 1

    samples = session.run(fake_data)
    lib.save_images.save_images(samples[:100], 'samples_{}.jpg'.format(frame))

# def inf_train_gen():
#     while True:
#         data = sklearn.datasets.make_swiss_roll(n_samples=BATCH_SIZE*1000, noise=0.25)[0]
#         data = data.astype('float32')[:, [0, 2]]
#         for i in xrange(1000):
#             # yield 8.*np.random.normal(size=(BATCH_SIZE,2))
#             yield data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

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
    disc_costs, lipchitz_penalties, gen_costs = [], [], []
    for iteration in xrange(ITERS):
        _data = gen.next()

        # if iteration < 50:
        #     disc_iters = 100
        # else:
        if iteration % 2 == 0:
            if (iteration < 200):# or (iteration % 100 == 0):
                disc_iters = 100
            else:
                disc_iters = 20
            for i in xrange(disc_iters):
                _disc_cost, _lipchitz_penalty, _ = session.run([wgan_disc_cost, lipschitz_penalty, disc_train_op], feed_dict={real_data: _data})
                # _ = session.run([clip_disc_weights])
                disc_costs.append(_disc_cost)
                lipchitz_penalties.append(_lipchitz_penalty)
                _data = gen.next()
                # if i % 100 == 99:
                #     print np.mean(disc_costs[-100:])
        else:
            if gen_train_op is not None:
                _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={real_data: _data})
                gen_costs.append(_gen_cost)

        if iteration % 10 == 0:
            print "iter:{}\tdisc:{:.3f} {:.3f}\tgen:{:.3f}".format(iteration, np.mean(disc_costs), np.mean(lipchitz_penalties), np.mean(gen_costs))
            disc_costs, lipchitz_penalties, gen_costs = [], [], []

        if iteration % 100 == 0:
            print "saving"
            generate_image(iteration, _data)