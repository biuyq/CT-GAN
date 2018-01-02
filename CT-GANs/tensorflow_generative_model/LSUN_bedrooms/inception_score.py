import os, sys
sys.path.append(os.getcwd())

if __name__ == '__main__':
    try: # This only matters on Ishaan's computer
        import experiment_tools
        experiment_tools.wait_for_gpu(tf=True, n_gpus=1)
    except ImportError:
        pass

import tflib as lib
import tflib.train_loop_2
import tflib.mnist
import tflib.ops.mlp
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm

import numpy as np
import tensorflow as tf

import functools
import os

LR = 1e-3
BATCH_SIZE = 500

ITERS_PER_EPOCH = 50000/BATCH_SIZE

TIMES = {
    'mode': 'iters',
    'print_every': 1*ITERS_PER_EPOCH,
    'stop_after': 9*ITERS_PER_EPOCH,
    'test_every': 1*ITERS_PER_EPOCH
}


def nonlinearity(x):
    return tf.nn.elu(x)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ResidualBlock(name, input_dim, output_dim, inputs, filter_size, is_training, mask_type=None, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if mask_type != None and resample != None:
        raise Exception('Unsupported configuration')

    if resample=='down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim, stride=2)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(SubpixelConv2D, input_dim=input_dim, output_dim=output_dim)
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
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, mask_type=mask_type, he_init=False, biases=True, inputs=inputs)

    output = inputs
    if mask_type == None:
        output = nonlinearity(output)
        output = conv_1(name+'.Conv1', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)
        output = nonlinearity(output)
        output = conv_2(name+'.Conv2', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)
        output = lib.ops.batchnorm.Batchnorm(name+'.BN', [0,2,3], output, is_training, update_moving_stats=True)
    else:
        output = nonlinearity(output)
        output_a = conv_1(name+'.Conv1A', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)
        output_b = conv_1(name+'.Conv1B', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)
        output = pixcnn_gated_nonlinearity(output_a, output_b)
        output = conv_2(name+'.Conv2', filter_size=filter_size, mask_type=mask_type, inputs=output, he_init=he_init)

    return shortcut + (0.3 * output)

def build_model(graph):
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=[None, 784])
        targets = tf.placeholder(tf.int32, shape=[None])
        is_training = tf.placeholder(tf.bool, shape=None)

        output = tf.reshape(inputs, [-1, 1, 28, 28])
        output = lib.ops.conv2d.Conv2D('InceptionScore.Conv1', 1, 32, 3, output, he_init=False)
        output = ResidualBlock('InceptionScore.Res1', 32, 32, output, 3, is_training, resample='down')
        output = ResidualBlock('InceptionScore.Res2', 32, 32, output, 3, is_training, resample=None)
        output = ResidualBlock('InceptionScore.Res3', 32, 64, output, 3, is_training, resample='down')
        output = ResidualBlock('InceptionScore.Res4', 64, 64, output, 3, is_training, resample=None)
        output = tf.reduce_mean(output, reduction_indices=[2,3])
        logits = lib.ops.linear.Linear('InceptionScore.Linear', 64, 10, output)
        # logits = lib.ops.mlp.MLP(
        #     'InceptionScore.MLP',
        #     input_dim=784,
        #     hidden_dim=512,
        #     output_dim=10,
        #     n_layers=4,
        #     inputs=inputs
        # )

        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
        )

        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.to_int32(tf.argmax(logits, dimension=1)),
                    targets
                ),
                tf.float32
            )
        )

        softmax = tf.nn.softmax(tf.cast(logits, tf.float64))
        # From https://github.com/openai/improved-gan/blob/master/inception_score/model.py
        kl = softmax * (tf.log(softmax) - tf.log(tf.reduce_mean(softmax, reduction_indices=[0], keep_dims=True)))
        inception_score = tf.exp(tf.reduce_mean(tf.reduce_sum(kl, reduction_indices=[1])))

    return (inputs, targets, is_training, cost, acc, logits, inception_score)

def train_model(graph, session, model):
    inputs, targets, is_training, cost, acc, logits, inception_score = model

    lib.print_model_settings(locals().copy())

    train_data, dev_data, test_data = lib.mnist.load(
        BATCH_SIZE,
        BATCH_SIZE
    )

    with graph.as_default():
        lib.train_loop_2.train_loop(
            session=session,
            inputs=[inputs, targets],
            cost=cost,
            is_training_var=is_training,
            prints=[
                ('acc', acc),
                ('inception', inception_score)
            ],
            optimizer=tf.train.AdamOptimizer(LR),
            train_data=train_data,
            test_data=dev_data,
            test_every=TIMES['test_every'],
            stop_after=TIMES['stop_after'],
            # times=TIMES
        )

def run_model(graph, session, model, data):
    inputs, targets, is_training, cost, acc, logits, inception_score = model
    all_logits = []
    step = min(1000, len(data))
    for i in xrange(0, len(data), step):
        all_logits.append(session.run(logits, feed_dict={inputs:data[i:i+step]}))
    all_logits = np.concatenate(all_logits, axis=0)
    probs = np.exp(all_logits - np.max(all_logits, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    kl = probs * (np.log(probs) - np.log(np.mean(probs, axis=0, keepdims=True)))
    inception_score = np.exp(np.mean(np.sum(kl, axis=1)))
    return inception_score

class InceptionScore(object):
    def __init__(self, retrain=False):
        self._graph = tf.Graph()
        self._model = build_model(self._graph)
        self._session = tf.Session(graph=self._graph)

        if (not retrain) and os.path.isfile('/tmp/inception_score.ckpt'):
            # print "Inception score: Loading saved model weights..."
            with self._graph.as_default():
                tf.train.Saver().restore(self._session, '/tmp/inception_score.ckpt')
        else:
            print "Inception score: No saved weights found, training model..."
            train_model(self._graph, self._session, self._model)
            with self._graph.as_default():
                tf.train.Saver().save(self._session, '/tmp/inception_score.ckpt')

    def score(self, data):
        return run_model(self._graph, self._session, self._model, data)

if __name__ == '__main__':
    train_data, dev_data, test_data = lib.mnist.load(
        BATCH_SIZE,
        BATCH_SIZE
    )

    test_batches = []
    for (images, targets) in test_data():
        test_batches.append(images)
    all_test_images = np.concatenate(test_batches, axis=0)

    print "Test set inception score: {}".format(
        InceptionScore(retrain=True).score(all_test_images)
    )