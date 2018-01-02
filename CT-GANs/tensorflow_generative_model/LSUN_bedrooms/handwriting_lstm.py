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

import handwriting_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BATCH_SIZE = 64
ITERS = 200000
DIM = 128
SEQ_LEN = 128
RESET_PROB = 1.0
SAMPLE_SEQ_LEN = 256
SAVE_N_SAMPLES = 5 # Only save the first 5 samples
GEN_BS_MULTIPLE = 2
# lib.ops.linear.enable_default_weightnorm()

srng = RandomStreams(seed=234)

def Generator(n_samples, h0s=None):
    input_noise = srng.normal(
        size=(n_samples, SEQ_LEN, 128)
    )
    inputs = lib.ops.linear.Linear('Generator.input_noise', 128, DIM, input_noise, initialization='he')
    inputs = T.nnet.relu(inputs)

    output_h, output_c = lib.ops.lstm.LSTM('Generator.RNN', DIM, DIM, inputs, h0s=h0s)

    output = lib.ops.linear.Linear('Generator.Out', DIM, 3, output_h)

    # Apply a sigmoid on the third dimension only
    output = T.concatenate([output[:,:,0:2], T.nnet.sigmoid(output[:,:,2:3])], axis=2)

    return output, (output_h[:,-1], output_c[:,-1])

def Discriminator(inputs):
    n_samples = inputs.shape[0]

    output = lib.ops.linear.Linear('Discriminator.In', 3, DIM, inputs, initialization='glorot_he')
    output = T.nnet.relu(output)

    output, _ = lib.ops.lstm.LSTM('Discriminator.RNN', DIM, DIM, output)

    # output = output.reshape((n_samples, SEQ_LEN*DIM))
    output = lib.ops.linear.Linear('Discriminator.Out', DIM, 1, output)
    output = T.mean(output, axis=1)

    return output.reshape((n_samples,))


real_data = T.tensor3('real_data')
h0, c0 = T.matrix('h0'), T.matrix('c0')
fake_data, (last_h, last_c) = Generator(BATCH_SIZE, [h0, c0])
fake_data_4x, (last_h_4x, last_c_4x) = Generator(GEN_BS_MULTIPLE*BATCH_SIZE, [h0, c0])

disc_out = Discriminator(T.concatenate([real_data, fake_data], axis=0))
disc_real = disc_out[:BATCH_SIZE]
disc_fake = disc_out[BATCH_SIZE:]

gen_cost = -T.mean(Discriminator(fake_data_4x))
disc_cost = T.mean(disc_fake) - T.mean(disc_real)

alpha = srng.uniform(
    size=(BATCH_SIZE,1,1), 
    low=0.,
    high=1.
)
differences = fake_data - real_data
interpolates = real_data + (alpha*differences)
gradients = T.grad(T.sum(Discriminator(interpolates)), interpolates)
gradients = gradients.reshape((BATCH_SIZE, -1))
slopes = T.sqrt(T.sum(T.sqr(gradients), axis=1))
lipschitz_penalty = T.mean((slopes-1.)**2)
disc_cost += 10*lipschitz_penalty

gen_params     = lib.search(gen_cost,     lambda x: hasattr(x, 'param') and 'Generator' in x.name)
discrim_params = lib.search(disc_cost, lambda x: hasattr(x, 'param') and 'Discriminator' in x.name)

gen_grads       = T.grad(gen_cost, gen_params)
discrim_grads   = T.grad(disc_cost, discrim_params)
# gen_grads = [
#     T.clip(g, lib.floatX(-1.0), lib.floatX(1.0))
#     for g in gen_grads
# ]
# discrim_grads = [
#     T.clip(g, lib.floatX(-1.0), lib.floatX(1.0))
#     for g in discrim_grads
# ]
gen_grads, gen_grad_norm = lasagne.updates.total_norm_constraint(gen_grads, 100.0, return_norm=True)
discrim_grads, discrim_grad_norm = lasagne.updates.total_norm_constraint(discrim_grads, 100.0, return_norm=True)

gen_updates     = lasagne.updates.adam(gen_grads,     gen_params,     learning_rate=2e-4, beta1=0.5, beta2=0.9)
discrim_updates = lasagne.updates.adam(discrim_grads, discrim_params, learning_rate=2e-4, beta1=0.5, beta2=0.9)

print "Loading data"

def generate_image(samples, filename):
    samples[:,:,0:2] *= 20
    print "Saving samples..."
    start_time = time.time()
    handwriting_utils.draw_many_strokes(samples[:SAVE_N_SAMPLES], filename)
    print "Saved {} samples to {}, took {}s".format(SAVE_N_SAMPLES, filename, time.time() - start_time)

DATA_SCALE = 20
data_loader = handwriting_utils.DataLoader(BATCH_SIZE, SEQ_LEN, DATA_SCALE)
def inf_train_gen():
    while True:
        batches = []
        data_loader.reset_batch_pointer()
        for b in range(data_loader.num_batches):
            x,y = data_loader.next_batch()
            batches.append(np.array(x, dtype='float32'))
        np.random.shuffle(batches)
        for batch in batches:
            yield batch

gen = inf_train_gen()
samples = gen.next()
generate_image(samples, 'groundtruth.svg')
# Reset the generator for training
gen = inf_train_gen()

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

def save_samples(iteration):
    _h0, _c0 = np.zeros((BATCH_SIZE, DIM), dtype='float32'), np.zeros((BATCH_SIZE, DIM), dtype='float32')
    samples, _h0, _c0 = _sample_fn(_h0, _c0)
    for i in xrange((SAMPLE_SEQ_LEN/SEQ_LEN)-1):
        next_samples, _h0, _c0 = _sample_fn(_h0, _c0)
        samples = np.concatenate([samples, next_samples], axis=1)
    generate_image(samples, 'samples_{}.svg'.format(iteration))


_disc_costs, _gen_costs, times, datatimes = [], [], [], []

print "Training!"
h0_d, c0_d = np.zeros((BATCH_SIZE, DIM), dtype='float32'), np.zeros((BATCH_SIZE, DIM), dtype='float32')
h0_g, c0_g = np.zeros((GEN_BS_MULTIPLE*BATCH_SIZE, DIM), dtype='float32'), np.zeros((GEN_BS_MULTIPLE*BATCH_SIZE, DIM), dtype='float32')

for iteration in xrange(ITERS):
    if iteration % 1000 == 999:
        save_samples(iteration)

    start_time = time.time()

    disc_iters = 10
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

    keep_mask = np.random.uniform(size=(GEN_BS_MULTIPLE*BATCH_SIZE,1)) > RESET_PROB
    h0_g *= keep_mask
    c0_g *= keep_mask

    times.append(time.time() - start_time)

    if (iteration < 20) or (iteration % 1 == 0):
        print "iter:\t{}\tdisc:\t{:.3f}\tgen:\t{:.3f}\ttime:\t{:.3f} datatime:\t{:.3f}".format(iteration, np.mean(_disc_costs), np.mean(_gen_costs), np.mean(times), np.mean(datatimes))
        _disc_costs, _gen_costs, times, datatimes = [], [], [], []