import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.cifar10
import tflib.inception_score
import tflib.plot

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = '/home/bigdata/Desktop/improved_wgan_training-master/cifar-10-batches-py'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')





n_examples = 1000 # number of examples for training
LAMBDA_2 = 2.0    # weight facter
Factor_M = 0.0    # M
MODE = 'wgan-CT' # Valid options are dcgan, wgan, or wgan-gp
DIM = 128 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 50000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)

lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM, 5, output, stride=2)
    output = LeakyReLU(output)
    output = tf.nn.dropout(output, keep_prob=0.50)   #keep probability
    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE != 'wgan-CT':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)
    output = tf.nn.dropout(output, keep_prob=0.50)   #keep probability
    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE != 'wgan-CT':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)
    output = tf.nn.dropout(output, keep_prob=0.50)   #keep probability
    output2 = tf.reshape(output, [-1, 4*4*4*DIM])    # corresponding to D_
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output2) #corresponding to D

    return tf.reshape(output, [-1]),output2

real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_data = 2*((tf.cast(real_data_int, tf.float32)/255.)-.5)
fake_data = Generator(BATCH_SIZE)
fake_data_2 = Generator(BATCH_SIZE)

disc_real, disc_real_2= Discriminator(real_data)
disc_real_, disc_real_2_= Discriminator(real_data)
disc_fake,disc_fake_ = Discriminator(fake_data)
disc_fake_2,disc_fake_2_ = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')


fake_data_2 = Generator(BATCH_SIZE)






if MODE == 'wgan-CT':
    # Standard WGAN loss
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)



    #Consistency term
    CT = LAMBDA_2*tf.square(disc_real-disc_real_)
    CT += LAMBDA_2*0.1*tf.reduce_mean(tf.square(disc_real_2-disc_real_2_),reduction_indices=[1])
    CT_ = tf.maximum(CT-Factor_M,0.0*(CT-Factor_M))
    disc_cost += tf.reduce_mean(CT_)

    # Gradient penalty
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates)[0], [interpolates])[0]
    gradients2 = tf.gradients(Discriminator(real_data)[0], [real_data])[0] # check the gradient on real data points


    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    slopes2 = tf.sqrt(tf.reduce_sum(tf.square(gradients2), reduction_indices=[1])) # L2 norm
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)
    clip_disc_weights = None



# For generating samples
fixed_noise_128 = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples_128 = Generator(128, noise=fixed_noise_128)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples_128)
    samples = ((samples+1.)*(255./2)).astype('int32')
    lib.save_images.save_images(samples.reshape((128, 3, 32, 32)), 'samples_{}.jpg'.format(frame))

# For calculating inception score
samples_100 = Generator(100)
def get_inception_score():
    all_samples = []
    for i in xrange(10):
        all_samples.append(session.run(samples_100))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples+1.)*(255./2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
    return lib.inception_score.get_inception_score(list(all_samples))

# Dataset iterators
train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, DATA_DIR,n_examples)
def inf_train_gen():
    while True:
        for images,_ in train_gen():
            yield images

# Train loop
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        start_time = time.time()
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)
        # Train critic
        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data = gen.next()
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data_int: _data})
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate inception score every 1K iters
        if iteration % 1000 == 999:
            inception_score = get_inception_score()
            lib.plot.plot('inception score', inception_score[0])

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            para = []
            array_para = []
            para = session.run(disc_params)
            #array_para = reduce(operator.add,para)
            #print(array_para)
            #print(para)
            np.save("param.pyn",para)
            dev_disc_costs = []
            for images,_ in dev_gen():
                _dev_disc_cost,slopes = session.run([disc_cost,slopes2], feed_dict={real_data_int: images}) 
                dev_disc_costs.append(_dev_disc_cost)
            max_slope = np.max(slopes) # only to check the biggest slope in each batch.
            lib.plot.plot('slope_real', max_slope)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
            generate_image(iteration, _data)

        # Save logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()
