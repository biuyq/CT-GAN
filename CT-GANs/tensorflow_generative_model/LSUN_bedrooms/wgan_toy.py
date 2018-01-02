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
import random

BATCH_SIZE = 1024
ITERS = 10000

# MODE = 'lsgan' # dcgan, wgan, wgan++, lsgan
GRADIENT_LOSS = True
# DATASET = '8gaussians'
# DATASET = '25gaussians'
DATASET = 'swissroll'

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs, alpha=0., bn=False, wn=False):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    if bn:
        if GRADIENT_LOSS:
            output = lib.ops.batchnorm.Batchnorm(name+'.BN', [0], output)
        else:
            output = lib.ops.batchnorm.Batchnorm(name+'.BN', [0], output)

    output = tf.nn.relu(output)
    return output

def Generator(n_samples, real_data):
    # return real_data + (1.*tf.random_normal(tf.shape(real_data)))

    noise = tf.random_normal([n_samples, 2])

    output = ReLULayer('Generator.1', 2, 512, noise)
    output = ReLULayer('Generator.2', 512, 512, output)
    # output *= 0. # Forces the generator to collapse to a point
    output = lib.ops.linear.Linear('Generator.Out', 512, 2, output)

    return output

def Discriminator(inputs):
    output = ReLULayer('Discriminator.1', 2, 1024, inputs, bn=False, wn=False)
    output = ReLULayer('Discriminator.2', 1024, 1024, output, bn=False, wn=False)

    output = lib.ops.linear.Linear('Discriminator.Out', 1024, 1, output)
    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[None, 2])
fake_data = Generator(BATCH_SIZE, real_data)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

# WGAN loss: disc tries to push fakes down and reals up, gen tries to push fakes up
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(Discriminator(Generator(BATCH_SIZE, real_data)))

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
if GRADIENT_LOSS:
    disc_cost += 10*lipschitz_penalty
disc_gradients = tf.reduce_mean(slopes)

if GRADIENT_LOSS:
    # The standard WGAN settings also work, but Adam is cooler!
    if len(lib.params_with_name('Generator')):
        gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    else:
        gen_train_op = None
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))

else:
    # Settings from WGAN paper; I haven't tried Adam  everything I've tried fails
    if len(lib.params_with_name('Generator')):
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
    else:
        gen_train_op = None
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator'))

    # Build an op to do the weight clipping
    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        print "Clipping {}".format(var.name)
        clip_bounds = [-.01, .01]
        clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
    clip_disc_weights = tf.group(*clip_ops)

# For generating plots
frame_i = [0]
# fake_data_1000 = Generator(1000)
def generate_image(true_dist):
    N_POINTS = 128

    RANGE = 2

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:,:,0] = np.linspace(-RANGE, RANGE, N_POINTS)[:,None]
    points[:,:,1] = np.linspace(-RANGE, RANGE, N_POINTS)[None,:]
    points = points.reshape((-1,2))
    # samples, disc_map = session.run([fake_data_1000, (disc_real)], feed_dict={real_data:points})
    disc_map = session.run(disc_real, feed_dict={real_data:points})
    samples = session.run(fake_data, feed_dict={real_data:points})

    plt.clf()
    # plt.imshow(disc_map.reshape((N_POINTS, N_POINTS)).T[::-1, :], extent=[-RANGE, RANGE, -RANGE, RANGE], cmap='seismic', vmin=np.min(disc_map), vmax=np.max(disc_map))
    # plt.colorbar()

    x,y = np.linspace(-RANGE, RANGE, N_POINTS), np.linspace(-RANGE, RANGE, N_POINTS)
    # print disc_map.shape
    # print len(x)
    # print len(y)
    plt.contour(x,y,disc_map.reshape((len(x), len(y))).transpose())

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange',  marker='+')
    plt.scatter(samples[:, 0],    samples[:, 1],    c='green', marker='+')

    # frame1 = plt.gca()
    # frame1.axes.get_xaxis().set_visible(False)
    # frame1.axes.get_yaxis().set_visible(False)
    # plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    plt.savefig('frame'+str(frame_i[0])+'.pdf', bbox_inches='tight', pad_inches=0.0)
    frame_i[0] += 1

# Dataset iterator
def inf_train_gen():
    if DATASET == '25gaussians':
    
        dataset = []
        for i in xrange(100000/25):
            for x in xrange(-2, 3):
                for y in xrange(-2, 3):
                    point = np.random.randn(2)*0.05
                    point[0] += 2*x
                    point[1] += 2*y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828 # stdev
        while True:
            for i in xrange(len(dataset)/BATCH_SIZE):
                yield dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    elif DATASET == 'swissroll':
        while True:
            data = sklearn.datasets.make_swiss_roll(n_samples=BATCH_SIZE*1000, noise=0.25)[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5 # stdev plus a little
            for i in xrange(1000):
                yield data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    elif DATASET == '8gaussians':
    
        scale = 2.
        centers = [(1,0),(-1,0),(0,1),(0,-1),(1./np.sqrt(2), 1./np.sqrt(2)),(1./np.sqrt(2), -1./np.sqrt(2)),(-1./np.sqrt(2), 1./np.sqrt(2)),(-1./np.sqrt(2), -1./np.sqrt(2))]
        centers = [(scale*x,scale*y) for x,y in centers]
        while True:
            dataset = []
            for i in xrange(BATCH_SIZE):
                point = np.random.randn(2)*.02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414 # stdev
            yield dataset

# Train loop!
with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()
    disc_costs, all_disc_gradients, gen_costs = [], [], []
    for iteration in xrange(ITERS):
        _data = gen.next()

        disc_iters = 10
        for i in xrange(disc_iters):
            _disc_cost, __disc_gradients, _ = session.run([disc_cost, disc_gradients, disc_train_op], feed_dict={real_data: _data})
            if not GRADIENT_LOSS:
                _ = session.run([clip_disc_weights])
            disc_costs.append(_disc_cost)
            all_disc_gradients.append(__disc_gradients)

        if gen_train_op is not None:
            _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={real_data: _data})
            gen_costs.append(_gen_cost)

        if iteration % 100 == 0:
            print "iter:{}\tdisc:{:.6f} disc_gradients:{:.3f}\tgen:{:.3f}".format(iteration, np.mean(disc_costs), np.mean(all_disc_gradients), np.mean(gen_costs))
            disc_costs, all_disc_gradients, gen_costs = [], [], []

            generate_image(_data)