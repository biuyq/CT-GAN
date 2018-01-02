import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu()
except ImportError:
    pass

import lasagne
import lib
import lib.lsun_downsampled
import lib.ops.gru
import lib.ops.linear
import lib.ops.lstm
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time
# import tflib.save_images

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BATCH_SIZE = 64
ITERS = 200000
DIM = 128
SEQ_LEN = 8
PERIOD = 32
RESET_PROB = 0.1

# lib.ops.linear.enable_default_weightnorm()

srng = RandomStreams(seed=234)

def Generator(n_samples, h0s=None):
    input_noise = srng.normal(
        size=(n_samples, SEQ_LEN, 128)
    )
    inputs = lib.ops.linear.Linear('Generator.input_noise', 128, DIM, input_noise, initialization='he')
    inputs = T.nnet.relu(inputs)

    output_h, output_c = lib.ops.lstm.LSTM('Generator.RNN', DIM, DIM, inputs, h0s=h0s)

    output = lib.ops.linear.Linear('Generator.Out', DIM, 1, output_h)

    output = output.reshape((n_samples, SEQ_LEN))
    return output, (output_h[:,-1], output_c[:,-1])

def Discriminator(inputs):
    n_samples = inputs.shape[0]
    output = inputs.reshape((n_samples,SEQ_LEN,1))
    output = lib.ops.linear.Linear('Discriminator.In', 1, DIM, output, initialization='glorot_he')
    output = T.nnet.relu(output)

    output, _ = lib.ops.lstm.LSTM('Discriminator.RNN', DIM, DIM, output)

    output = output.reshape((n_samples, SEQ_LEN*DIM))
    output = lib.ops.linear.Linear('Discriminator.Out', SEQ_LEN*DIM, 1, output)

    return output.reshape((n_samples,))


real_data = T.matrix('real_data')
h0, c0 = T.matrix('h0'), T.matrix('c0')
fake_data, (last_h, last_c) = Generator(BATCH_SIZE, [h0, c0])
fake_data_4x, (last_h_4x, last_c_4x) = Generator(4*BATCH_SIZE, [h0, c0])

disc_out = Discriminator(T.concatenate([real_data, fake_data], axis=0))
disc_real = disc_out[:BATCH_SIZE]
disc_fake = disc_out[BATCH_SIZE:]

gen_cost = -T.mean(Discriminator(fake_data_4x))
disc_cost = T.mean(disc_fake) - T.mean(disc_real)

alpha = srng.uniform(
    size=(BATCH_SIZE,1), 
    low=0.,
    high=1.
)
differences = fake_data - real_data
interpolates = real_data + (alpha*differences)
gradients = T.grad(T.sum(Discriminator(interpolates)), interpolates)
slopes = T.sqrt(T.sum(T.sqr(gradients), axis=1))
lipschitz_penalty = T.mean((slopes-1.)**2)
disc_cost += 10*lipschitz_penalty

gen_params     = lib.search(gen_cost,     lambda x: hasattr(x, 'param') and 'Generator' in x.name)
discrim_params = lib.search(disc_cost, lambda x: hasattr(x, 'param') and 'Discriminator' in x.name)

gen_grads       = T.grad(gen_cost, gen_params)
discrim_grads   = T.grad(disc_cost, discrim_params)
gen_grads = [
    T.clip(g, lib.floatX(-1.0), lib.floatX(1.0))
    for g in gen_grads
]
discrim_grads = [
    T.clip(g, lib.floatX(-1.0), lib.floatX(1.0))
    for g in discrim_grads
]
gen_updates     = lasagne.updates.adam(gen_grads,     gen_params,     learning_rate=1e-4, beta1=0.5, beta2=0.9)
discrim_updates = lasagne.updates.adam(discrim_grads, discrim_params, learning_rate=1e-4, beta1=0.5, beta2=0.9)

print "Compiling functions"

train_discrim_fn = theano.function(
    [real_data, h0, c0],
    [disc_cost, last_h, last_c],
    updates=discrim_updates.items(),
    on_unused_input='warn'
)
train_gen_fn = theano.function(
    [h0, c0],
    [gen_cost, last_h_4x, last_c_4x],
    updates=gen_updates.items(),
    on_unused_input='warn'
)

_sample_fn = theano.function([h0, c0], [fake_data, last_h, last_c], on_unused_input='warn')

def generate_image(iteration):
    _h0, _c0 = np.zeros((BATCH_SIZE, DIM), dtype='float32'), np.zeros((BATCH_SIZE, DIM), dtype='float32')
    samples, _h0, _c0 = _sample_fn(_h0, _c0)
    for i in xrange(3):
        next_samples, _h0, _c0 = _sample_fn(_h0, _c0)
        samples = np.concatenate([samples, next_samples], axis=1)
    save_samples(samples, 'samples_{}.png'.format(iteration))

def save_samples(samples, filename):
    seqlen = samples.shape[1]
    plt.figure(figsize=(10,2*BATCH_SIZE*(10./seqlen)))

    y_offset = 0
    for sample in samples:
        prev_x, prev_y = 0, sample[0]
        for x,y in enumerate(sample[1:], start=1):
            plt.plot([prev_x, x], [prev_y + y_offset, y + y_offset], color='k', linestyle='-', linewidth=2)
            prev_x, prev_y = x,y
        y_offset += 2

    plt.savefig(filename)
    plt.close()

def inf_train_gen():
    while True:
        samples = []
        for i in xrange(BATCH_SIZE):
            phase = np.random.uniform()*2*np.pi
            period = np.random.randint(PERIOD/2)+1
            x = np.arange(SEQ_LEN)
            y = np.sin((2*np.pi / period)*x + phase)
            samples.append(y)
        yield np.array(samples, dtype='float32')

gen = inf_train_gen()
_disc_costs, _gen_costs, times, datatimes = [], [], [], []

save_samples(gen.next(), 'groundtruth.png')

print "Training!"
h0_d, c0_d = np.zeros((BATCH_SIZE, DIM), dtype='float32'), np.zeros((BATCH_SIZE, DIM), dtype='float32')
h0_g, c0_g = np.zeros((4*BATCH_SIZE, DIM), dtype='float32'), np.zeros((4*BATCH_SIZE, DIM), dtype='float32')

for iteration in xrange(ITERS):

    if iteration % 100 == 0:
        generate_image(iteration)

    start_time = time.time()

    disc_iters = 5
    for i in xrange(disc_iters):
        data_start_time = time.time()
        _data = gen.next()
        datatimes.append(time.time() - data_start_time)
        _disc_cost, h0_d, c0_d = train_discrim_fn(_data, h0_d, c0_d)

        keep_mask = np.random.uniform(size=(BATCH_SIZE,1)) > RESET_PROB
        h0_d *= keep_mask
        c0_d *= keep_mask

    _disc_costs.append(_disc_cost)

    _gen_cost, h0_g, c0_g = train_gen_fn(h0_g, c0_g)
    _gen_costs.append(_gen_cost)

    keep_mask = np.random.uniform(size=(4*BATCH_SIZE,1)) > RESET_PROB
    h0_g *= keep_mask
    c0_g *= keep_mask

    times.append(time.time() - start_time)

    if (iteration < 20) or (iteration % 20 == 19):
        print "iter:\t{}\tdisc:\t{:.3f}\tgen:\t{:.3f}\ttime:\t{:.3f} datatime:\t{:.3f}".format(iteration, np.mean(_disc_costs), np.mean(_gen_costs), np.mean(times), np.mean(datatimes))
        _disc_costs, _gen_costs, times, datatimes = [], [], [], []