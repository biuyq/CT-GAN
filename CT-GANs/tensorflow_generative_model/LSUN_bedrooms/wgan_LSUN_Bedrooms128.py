## LSUN bedrooms, the architechture is the same as GP-WGAN provided except we add dropout in the hidden layers and CT term
## import imagenet related code as they are the same.
import os, sys
sys.path.append(os.getcwd())

N_GPUS = 2

try:
    import experiment_tools
    experiment_tools.wait_for_gpu(tf=True, n_gpus=N_GPUS)
except ImportError:
    pass

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.batchnorm
import tflib.ops.layernorm
import tflib.save_images
import tflib.imagenet  # change LSUN path in this file
import tflib.plot

import numpy as np
import tensorflow as tf

import os
import time
import functools

BATCH_SIZE = 64
DATASET = 'imagenet'

DIM_G_64  = 64
DIM_G_32  = 128
DIM_G_16  = 256
DIM_G_8   = 512
DIM_G_4   = 512

DIM_D_64  = 128
DIM_D_32  = 256
DIM_D_16  = 512
DIM_D_8   = 1024
DIM_D_4   = 1024

NORMALIZATION_G = True
NORMALIZATION_D = True

ITERS = 200000
LAMBDA_2 =2.0  # parameter LAMBDA2
Factor_M = 0.0  # factor M
LR = 1e-4
DECAY = True
CRITIC_ITERS = 5
MOMENTUM_G = 0.
MOMENTUM_D = 0.
GEN_BS_MULTIPLE = 1

def GeneratorAndDiscriminator():
    return ResnetGenerator, ResnetDiscriminator

lib.print_model_settings(locals().copy())

OUTPUT_DIM = 3*128*128
DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

def nonlinearity(x):
    return tf.nn.relu(x)

def Normalize(name, inputs):
    if ('Discriminator' in name) and NORMALIZATION_D:
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    elif ('Generator' in name) and NORMALIZATION_G:
        return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ScaledUpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = lib.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases, gain=0.5)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim, stride=2)
        # conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        # conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = MeanPoolConv
    elif resample=='up':
        conv_1        = functools.partial(ScaledUpsampleConv, input_dim=input_dim, output_dim=output_dim)
        # conv_1        = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
        conv_shortcut = ScaledUpsampleConv
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = inputs
        # shortcut = Normalize(name+'.NShortcut', shortcut)
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=shortcut)

    output = inputs
    output = Normalize(name+'.N1', output)
    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(name+'.N2', output)
    output = nonlinearity(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)
    # output = Normalize(name+'.N3', output)
    # return output
    return shortcut + output
    # return 0.7*(shortcut+output)

def ResnetGenerator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*DIM_G_4, noise)
    output = tf.reshape(output, [-1, DIM_G_4, 4, 4])

    # output = ResidualBlock('Generator.4_1', DIM_G_4, DIM_G_4, 3, output, resample=None)
    # output = ResidualBlock('Generator.4_2', DIM_G_4, DIM_G_4, 3, output, resample=None)
    output = ResidualBlock('Generator.4_3', DIM_G_4, DIM_G_8, 3, output, resample='up')

    # output = ResidualBlock('Generator.8_1', DIM_G_8, DIM_G_8, 3, output, resample=None)
    # output = ResidualBlock('Generator.8_2', DIM_G_8, DIM_G_8, 3, output, resample=None)
    output = ResidualBlock('Generator.8_3', DIM_G_8, DIM_G_16, 3, output, resample='up')

    # output = ResidualBlock('Generator.16_1', DIM_G_16, DIM_G_16, 3, output, resample=None)
    # output = ResidualBlock('Generator.16_2', DIM_G_16, DIM_G_16, 3, output, resample=None)
    output = ResidualBlock('Generator.16_3', DIM_G_16, DIM_G_32, 3, output, resample='up')

    # output = ResidualBlock('Generator.32_1', DIM_G_32, DIM_G_32, 3, output, resample=None)
    # output = ResidualBlock('Generator.32_2', DIM_G_32, DIM_G_32, 3, output, resample=None)
    output = ResidualBlock('Generator.32_3', DIM_G_32, DIM_G_64, 3, output, resample='up')

    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = ScaledUpsampleConv('Generator.Output', DIM_G_64, 3, 5, output, he_init=False)
    # output = lib.ops.deconv2d.Deconv2D('Generator.Output', DIM_G_64, 3, 5, output, he_init=False)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def ResnetDiscriminator(inputs,kp1,kp2,kp3):
    output = tf.reshape(inputs, [-1, 3, 128, 128])

    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, DIM_D_64, 5, output, he_init=True, stride=2)

    # output = ResidualBlock('Discriminator.64_1', DIM_D_64, DIM_D_64, 3, output, resample=None)
    # output = ResidualBlock('Discriminator.64_2', DIM_D_64, DIM_D_64, 3, output, resample=None)
    output = ResidualBlock('Discriminator.64_3', DIM_D_64, DIM_D_32, 3, output, resample='down')

    # output = ResidualBlock('Discriminator.32_1', DIM_D_32, DIM_D_32, 3, output, resample=None)
    # output = ResidualBlock('Discriminator.32_2', DIM_D_32, DIM_D_32, 3, output, resample=None)
    output = ResidualBlock('Discriminator.32_3', DIM_D_32, DIM_D_16, 3, output, resample='down')

    # output = ResidualBlock('Discriminator.16_1', DIM_D_16, DIM_D_16, 3, output, resample=None)
    # output = ResidualBlock('Discriminator.16_2', DIM_D_16, DIM_D_16, 3, output, resample=None)
    output = ResidualBlock('Discriminator.16_3', DIM_D_16, DIM_D_8, 3, output, resample='down')
    output = tf.nn.dropout(output, keep_prob=kp1)     #dropout after activator
    output = ResidualBlock('Discriminator.8_1', DIM_D_8, DIM_D_8, 3, output, resample=None)
    output = tf.nn.dropout(output, keep_prob=kp2)     #dropout after activator
    output = ResidualBlock('Discriminator.8_2', DIM_D_8, DIM_D_8, 3, output, resample=None)
    output = tf.nn.dropout(output, keep_prob=kp3)     #dropout after activator
    # output = ResidualBlock('Discriminator.8_3', DIM_D_8, DIM_D_4, 3, output, resample='down')

    # output = ResidualBlock('Discriminator.4_1', DIM_D_4, DIM_D_4, 3, output, resample=None)
    # output = ResidualBlock('Discriminator.4_2', DIM_D_4, DIM_D_4, 3, output, resample=None)

    # output = Normalize('Discriminator.OutputN', output)
    # output = output / 10.
    output2 = tf.reduce_mean(output, axis=[2,3])
    output = lib.ops.linear.Linear('Discriminator.Output', DIM_D_8, 1, output2)

    # output = Normalize('Discriminator.OutputN', output)
    # output = nonlinearity(output)
    # output = tf.reshape(output, [-1, 4*4*DIM_D_4])
    # output = lib.ops.linear.Linear('Discriminator.Output', 4*4*DIM_D_4, 1, output)

    return tf.reshape(output, [-1]),output2

with tf.Session() as session:

    Generator, Discriminator = GeneratorAndDiscriminator()

    iteration = tf.placeholder(tf.int32, shape=None)
    all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 128, 128])

    if (len(DEVICES)%2==0) and (len(DEVICES)>=2):

        fake_data_splits = []
        for device in DEVICES:
            with tf.device(device):
                fake_data_splits.append(Generator(BATCH_SIZE/len(DEVICES)))
        # fake_data = tf.concat(fake_data_splits, axis=0)
        # fake_data_splits = tf.split(fake_data, len(DEVICES))

        all_real_data = tf.reshape(2*((tf.cast(all_real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE, OUTPUT_DIM])
        all_real_data_splits = tf.split(all_real_data, len(DEVICES)/2)

        DEVICES_B = DEVICES[:len(DEVICES)/2]
        DEVICES_A = DEVICES[len(DEVICES)/2:]

        disc_costs = []
        for i, device in enumerate(DEVICES_A):
            with tf.device(device):
                real_and_fake_data = lib.concat([all_real_data_splits[i]] + [fake_data_splits[i]] + [fake_data_splits[len(DEVICES_A)+i]], axis=0)
                disc_all,disc_all_2 = Discriminator(real_and_fake_data, 0.8,0.5,0.5)
                disc_all_,disc_all_2_ = Discriminator(real_and_fake_data, 0.8,0.5,0.5)


                #disc_all,disc_all_2, disc_all_acgan = Discriminator(real_and_fake_data, real_and_fake_labels,0.8,0.5,0.5)  #dropout rate of 0.2,0.5,0.5
                disc_real = disc_all[:BATCH_SIZE/len(DEVICES_A)]
                disc_fake = disc_all[BATCH_SIZE/len(DEVICES_A):]

                disc_real_2 = disc_all_2[:BATCH_SIZE/len(DEVICES_A)]
                disc_fake_2 = disc_all_2[BATCH_SIZE/len(DEVICES_A):]


                disc_real_ = disc_all_[:BATCH_SIZE/len(DEVICES_A)]
                disc_fake_ = disc_all_[BATCH_SIZE/len(DEVICES_A):]

                disc_real_2_ = disc_all_2_[:BATCH_SIZE/len(DEVICES_A)]
                disc_fake_2_ = disc_all_2_[BATCH_SIZE/len(DEVICES_A):]




                disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))

        for i, device in enumerate(DEVICES_B):
            with tf.device(device):
                real_data = tf.identity(all_real_data_splits[i]) # transfer from gpu0
                fake_data__ = lib.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A)+i]], axis=0)
                alpha = tf.random_uniform(
                    shape=[BATCH_SIZE/len(DEVICES_A),1], 
                    minval=0.,
                    maxval=1.
                )
                differences = fake_data__ - real_data
                interpolates = real_data + (alpha*differences)
                gradients = tf.gradients(Discriminator(interpolates,0.8,0.5,0.5)[0], [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                # print "WARNING NO LIPSCHITZ PENALTY"
                gradient_penalty = 10.*tf.reduce_mean((slopes-1.)**2)



                CT = LAMBDA_2*tf.square(disc_real-disc_real_)  
                CT += LAMBDA_2*0.1*tf.reduce_mean(tf.square(disc_real_2-disc_real_2_),reduction_indices=[1])
                CT_ = tf.maximum(CT-Factor_M,0.0*(CT-Factor_M))
                CT_ = tf.reduce_mean(CT_)
                disc_costs.append(CT_)


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
                gen_costs.append(-tf.reduce_mean(Discriminator(Generator(GEN_BS_MULTIPLE*BATCH_SIZE/len(DEVICES)),0.8,0.5,0.5)[0]))
        gen_cost = tf.add_n(gen_costs) / len(DEVICES)
        gen_train_op = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=MOMENTUM_G, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)


    else:
        raise Exception()
        # split_real_data_conv = lib.split(all_real_data_conv, len(DEVICES), axis=0)

        # gen_costs, disc_costs = [],[]

        # for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
        #     with tf.device(device):

        #         real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE/len(DEVICES), OUTPUT_DIM])
        #         fake_data = Generator(BATCH_SIZE/len(DEVICES))

        #         disc_all = Discriminator(lib.concat([real_data, fake_data],0))
        #         disc_real = disc_all[:tf.shape(real_data)[0]]
        #         disc_fake = disc_all[tf.shape(real_data)[0]:]

        #         gen_cost = -tf.reduce_mean(Discriminator(Generator(GEN_BS_MULTIPLE*BATCH_SIZE/len(DEVICES))))
        #         disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        #         alpha = tf.random_uniform(
        #             shape=[BATCH_SIZE/len(DEVICES),1], 
        #             minval=0.,
        #             maxval=1.
        #         )
        #         differences = fake_data - real_data
        #         interpolates = real_data + (alpha*differences)
        #         interpolates = tf.stop_gradient(interpolates)
        #         gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
        #         slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        #         lipschitz_penalty = 100.*tf.reduce_mean((slopes-1.)**2)
        #         disc_cost += lipschitz_penalty

        #         gen_costs.append(gen_cost)
        #         disc_costs.append(disc_cost)

        # gen_cost = tf.add_n(gen_costs) / len(DEVICES)
        # disc_cost = tf.add_n(disc_costs) / len(DEVICES)

        # if DECAY:
        #     decay = tf.maximum(0., 1.-(tf.cast(iteration, tf.float32)/ITERS))
        # else:
        #     decay = 1.
        # gen_train_op = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=MOMENTUM_G, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        # disc_train_op = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=MOMENTUM_D, beta2=0.9).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)


    frame_i = [0]
    fixed_noise = tf.constant(np.random.normal(size=(64, 128)).astype('float32'))
    fixed_noise_samples = Generator(64, noise=fixed_noise)
    def generate_image(frame):
        samples = session.run(fixed_noise_samples)
        samples = ((samples+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(samples.reshape((64, 3, 128, 128)), 'samples_{}.png'.format(frame))

    if DATASET == 'imagenet':
        train_gen = lib.imagenet.load(BATCH_SIZE)

    def inf_train_gen():
        while True:
            for images, in train_gen():
                yield images

    session.run(tf.initialize_all_variables())

    generate_image(0)

    gen = inf_train_gen()

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    # Uncomment this to restore params
    # print "WARNING RESTORING PARAMS FROM CHECKPOINT"
    # saver.restore(session, os.getcwd()+"/params.ckpt")

    for _iteration in xrange(ITERS):
        start_time = time.time()

        for i in xrange(CRITIC_ITERS):
            _data = gen.next()
            _data = _data.reshape((BATCH_SIZE,3,128,128))
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op], 
                feed_dict={all_real_data_conv: _data, iteration: _iteration}#, fake_data: fake_data_buffer[np.random.choice(BUFFER_LEN*BATCH_SIZE, BATCH_SIZE)]}
            )

        _ = session.run(
            gen_train_op,
            feed_dict={iteration: _iteration}
        )

        lib.plot.plot('cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        if _iteration % 100 == 0:
            generate_image(_iteration)

        if _iteration % 1000 == 0:
            saver.save(session, 'params.ckpt')

        if _iteration % 5 == 0:
            lib.plot.flush(print_stds=True)

        lib.plot.tick()
