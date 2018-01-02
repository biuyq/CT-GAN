import os, sys
sys.path.append(os.getcwd())

try: # This only matters on Ishaan's computer
    import experiment_tools
    experiment_tools.wait_for_gpu()
except ImportError:
    pass

import lasagne
import lib
import lib.plot
import lib.lsun_downsampled
import lib.ops.gru
import lib.ops.linear
import lib.ops.lstm
import lib.ops.gru
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time
import lib.save_images

BATCH_SIZE = 64
ITERS = 200000
DIM = 512

# lib.ops.linear.enable_default_weightnorm()

srng = RandomStreams(seed=234)

def Generator(n_samples):
    h0_noise = srng.normal(
        size=(n_samples, 128)
    )

    h0 = lib.ops.linear.Linear('Generator.noise_to_h0', 128, 2*DIM, h0_noise)

    inputs = srng.normal(
        size=(n_samples, 64, 128)
    )
    output, _ = lib.ops.lstm.LSTM('Generator.RNN', 128, DIM, inputs, h0s=[h0[:,::2], h0[:,1::2]])

    output = lib.ops.linear.Linear('Generator.Out3', DIM, 64*3, output)

    output = T.tanh(output)

    output = output.reshape((n_samples, 64, 64, 3)).dimshuffle(0,3,1,2).reshape((n_samples,64*64*3))

    return output

def Discriminator(inputs):
    n_samples = inputs.shape[0]
    output = inputs.reshape((n_samples,3,64,64)).dimshuffle(0,2,3,1).reshape((n_samples, 64, 64*3))

    output, _ = lib.ops.lstm.LSTM('Discriminator.RNN', 64*3, DIM, output)

    # output = output.reshape((n_samples,64*DIM))
    # output = lib.ops.linear.Linear('Discriminator.Out', 64*DIM, 1, output)
    output = output[:,-1]
    output = lib.ops.linear.Linear('Discriminator.Out', DIM, 1, output)

    return output.reshape((n_samples,))


real_data_int = T.itensor4('images')
real_data = (T.cast(real_data_int, 'float32')*(2./255) - 1.).reshape((-1,64*64*3))

fake_data = Generator(BATCH_SIZE)

disc_out = Discriminator(T.concatenate([real_data, fake_data], axis=0))
disc_real = disc_out[:BATCH_SIZE]
disc_fake = disc_out[BATCH_SIZE:]

gen_cost = -T.mean(Discriminator(Generator(2*BATCH_SIZE)))
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
# lipschitz_penalty = T.mean((slopes-1.)**2)
lipschitz_penalty = T.mean(slopes**2)
disc_cost += 10*lipschitz_penalty

gen_params     = lib.search(gen_cost,     lambda x: hasattr(x, 'param') and 'Generator' in x.name)
discrim_params = lib.search(disc_cost, lambda x: hasattr(x, 'param') and 'Discriminator' in x.name)

gen_grads       = T.grad(gen_cost, gen_params)
discrim_grads   = T.grad(disc_cost, discrim_params)

# Just so logging code doesn't break
gen_grad_norm, discrim_grad_norm = T.as_tensor_variable(np.float32(0)), T.as_tensor_variable(np.float32(0))

gen_grads, gen_grad_norm = lasagne.updates.total_norm_constraint(gen_grads, 50.0, return_norm=True)
discrim_grads, discrim_grad_norm = lasagne.updates.total_norm_constraint(discrim_grads, 50.0, return_norm=True)

# gen_grads = [
#     T.clip(g, lib.floatX(-1.0), lib.floatX(1.0))
#     for g in gen_grads
# ]
# discrim_grads = [
#     T.clip(g, lib.floatX(-1.0), lib.floatX(1.0))
#     for g in discrim_grads
# ]

gen_updates     = lasagne.updates.adam(gen_grads,     gen_params,     learning_rate=2e-5, beta1=0., beta2=0.9)
discrim_updates = lasagne.updates.adam(discrim_grads, discrim_params, learning_rate=1e-4, beta1=0., beta2=0.9)

print "Compiling functions"

train_discrim_fn = theano.function(
    [real_data_int],
    [disc_cost, discrim_grad_norm, T.mean(slopes)],
    updates=discrim_updates.items(),
    on_unused_input='warn'
)
train_gen_fn = theano.function(
    [],
    [gen_cost, gen_grad_norm],
    updates=gen_updates.items(),
    on_unused_input='warn'
)


_sample_fn = theano.function([], fake_data, on_unused_input='warn')

def generate_image(iteration):
    samples = _sample_fn()
    samples = ((samples+1.)*(255.99/2)).astype('int32')
    # samples = np.maximum(0, samples)
    # samples = np.minimum(255, samples)
    lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)), 'samples_{}.png'.format(iteration))

train_gen, _ = lib.lsun_downsampled.load(BATCH_SIZE, False)
def inf_train_gen():
    while True:
        for (t,) in train_gen():
            yield t

gen = inf_train_gen()

for iteration in xrange(ITERS):

    if iteration % 100 == 0:
        generate_image(iteration)

    start_time = time.time()

    disc_iters = 1
    for i in xrange(disc_iters):
        # data_start_time = time.time()
        _data = gen.next()
        # datatimes.append(time.time() - data_start_time)
        _disc_cost, _disc_grad_norm, _mean_slopes = train_discrim_fn(_data)

    lib.plot.plot('disc cost', _disc_cost)
    # _disc_costs.append(_disc_cost)

    # data_start_time = time.time()
    _data = gen.next()
    # datatimes.append(time.time() - data_start_time)
    _gen_cost, _gen_grad_norm = train_gen_fn()
    # _gen_costs.append(_gen_cost)


    lib.plot.plot('time', time.time() - start_time)

    if (iteration < 20) or (iteration % 10 == 0):
        lib.plot.flush()
        # print "iter:\t{}\tdisc:\t{:.3f}\tgen:\t{:.3f}\ttime:\t{:.3f}\tdatatime:\t{:.3f}\tnorms:{} {}\tslopes:{}".format(iteration, np.mean(_disc_costs), np.mean(_gen_costs), np.mean(times), np.mean(datatimes), _disc_grad_norm, _gen_grad_norm, _mean_slopes)
        # _disc_costs, _gen_costs, times, datatimes = [], [], [], []
    lib.plot.tick()