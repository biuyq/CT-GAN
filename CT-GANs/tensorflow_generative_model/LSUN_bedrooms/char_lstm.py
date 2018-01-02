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
import tflib.save_images

BATCH_SIZE = 64
SEQ_LEN = 10
ITERS = 200000
DIM = 512

# lib.ops.linear.enable_default_weightnorm()

srng = RandomStreams(seed=234)

def Generator(n_samples):
    noise = srng.normal(
        size=(n_samples, 128)
    )

    h0 = lib.ops.linear.Linear('Generator.noise_to_h0', 128, 2*DIM, noise)
    h0 = T.nnet.relu(h0)
    h0 = lib.ops.linear.Linear('Generator.noise_to_h0_2', 2*DIM, 2*DIM, h0)

    # dummy inputs

    input_noise = srng.normal(
        size=(n_samples, 64, 128)
    )
    inputs = lib.ops.linear.Linear('Generator.input_noise', 128, 2*DIM, input_noise)
    inputs = T.nnet.relu(inputs)

    # inputs = T.zeros((n_samples, 64, 2))
    output = lib.ops.lstm.LSTM('Generator.RNN', 2*DIM, DIM, inputs, h0=[h0[:,::2], h0[:,1::2]])

    # Add some FC layers just for extra parameters
    output = lib.ops.linear.Linear('Generator.Out1', DIM, 2*DIM, output, initialization='he')
    output = T.nnet.relu(output)
    # output = lib.ops.linear.Linear('Generator.Out2', 2*DIM, 2*DIM, output, initialization='he')
    # output = T.nnet.relu(output)
    output = lib.ops.linear.Linear('Generator.Out3', 2*DIM, 64*3, output)

    output = T.tanh(output)

    output = output.reshape((n_samples, 64, 64, 3)).dimshuffle(0,3,1,2).reshape((n_samples,64*64*3))

    return output

def Discriminator(inputs):
    n_samples = inputs.shape[0]
    output = inputs.reshape((n_samples,3,64,64)).dimshuffle(0,2,3,1).reshape((n_samples, 64, 64*3))
    output = lib.ops.linear.Linear('Discriminator.In', 64*3, 2*DIM, output, initialization='he')
    output = T.nnet.relu(output)
    # output = lib.ops.linear.Linear('Discriminator.In2', 2*DIM, 2*DIM, output, initialization='he')
    # output = T.nnet.relu(output)
    # # output = lib.ops.linear.Linear('Discriminator.In3', 2*DIM, 2*DIM, output, initialization='he')
    # output = T.nnet.relu(output)
    output = lib.ops.lstm.LSTM('Discriminator.RNN', 2*DIM, DIM, output)
    # output_2 = lib.ops.lstm.LSTM('Discriminator.RNN2', DIM, DIM, output)
    # output = T.concatenate([output[:,-1,:], output_2[:,-1,:]],axis=1) # last hidden states
    output = output[:,-1,:]
    output = lib.ops.linear.Linear('Discriminator.Out', DIM, 1, output)
    return output.reshape((n_samples,))


real_data_int = T.itensor4('images')
real_data = (T.cast(real_data_int, 'float32')*(2./255) - 1.).reshape((-1,64*64*3))

fake_data = Generator(BATCH_SIZE)

disc_out = Discriminator(T.concatenate([real_data, fake_data], axis=0))
disc_real = disc_out[:BATCH_SIZE]
disc_fake = disc_out[BATCH_SIZE:]

gen_cost = -T.mean(Discriminator(fake_data))
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
    [real_data_int],
    [disc_cost],
    updates=discrim_updates.items(),
    on_unused_input='warn'
)
train_gen_fn = theano.function(
    [],
    [gen_cost],
    updates=gen_updates.items(),
    on_unused_input='warn'
)


_sample_fn = theano.function([], fake_data, on_unused_input='warn')

def generate_image(iteration):
    samples = _sample_fn()
    samples = ((samples+1.)*(255.99/2)).astype('int32')
    tflib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)), 'samples_{}.png'.format(iteration))

def inf_train_gen():
    while True:
        samples = []
        for i in xrange(BATCH_SIZE):
            sample = str(np.random.randint(10))
            for i in xrange(SEQ_LEN):
                prev = int(sample[-1])
                if prev == 0:
                    curr = np.random.randint(2) # stay at 0 or move to 1
                elif prev == 9:
                    curr = np.random.randint()

train_gen, _ = lib.lsun_downsampled.load(BATCH_SIZE, False)
def inf_train_gen():
    while True:
        for (t,) in train_gen():
            yield t

gen = inf_train_gen()
_disc_costs, _gen_costs, times, datatimes = [], [], [], []

for iteration in xrange(ITERS):

    if iteration % 100 == 0:
        generate_image(iteration)

    start_time = time.time()

    disc_iters = 5
    for i in xrange(disc_iters):
        data_start_time = time.time()
        _data = gen.next()
        datatimes.append(time.time() - data_start_time)
        _disc_cost, = train_discrim_fn(_data)

    _disc_costs.append(_disc_cost)

    data_start_time = time.time()
    _data = gen.next()
    datatimes.append(time.time() - data_start_time)
    _gen_cost, = train_gen_fn()
    _gen_costs.append(_gen_cost)

    times.append(time.time() - start_time)

    if (iteration < 20) or (iteration % 20 == 19):
        print "iter:\t{}\tdisc:\t{:.3f}\tgen:\t{:.3f}\ttime:\t{:.3f} datatime:\t{:.3f}".format(iteration, np.mean(_disc_costs), np.mean(_gen_costs), np.mean(times), np.mean(datatimes))
        _disc_costs, _gen_costs, times, datatimes = [], [], [], []