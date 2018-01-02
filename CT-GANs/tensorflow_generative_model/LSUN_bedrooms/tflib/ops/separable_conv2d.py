import tflib as lib
import tflib.debug

import numpy as np
import tensorflow as tf

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def SeparableConv2D(name, input_dim, output_dim, filter_size, inputs, he_init=True, stride=1, weightnorm=None, biases=True, gain=1., mask_type=None):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    mask_type: one of None, 'a', 'b'

    returns: tensor of shape (batch size, num channels, height, width)
    """
    if mask_type is not None:
        raise Exception('unsupported')

    with tf.name_scope(name) as scope:

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        spatial_fan_in = filter_size**2
        spatial_fan_out = filter_size**2 / (stride**2)

        pointwise_fan_in = input_dim
        pointwise_fan_out = output_dim

        if he_init:
            spatial_filters_stdev = np.sqrt(4./(spatial_fan_in+spatial_fan_out))
        else: # Normalized init (Glorot & Bengio)
            spatial_filters_stdev = np.sqrt(2./(spatial_fan_in+spatial_fan_out))

        pointwise_filters_stdev = np.sqrt(2./(pointwise_fan_in+pointwise_fan_out))

        spatial_filter_values = uniform(
            spatial_filters_stdev,
            (filter_size, filter_size, input_dim, 1)
        )

        pointwise_filter_values = uniform(
            pointwise_filters_stdev,
            (1, 1, input_dim, output_dim)
        )

        spatial_filter_values *= gain

        spatial_filters = lib.param(name+'.SpatialFilters', spatial_filter_values)
        pointwise_filters = lib.param(name+'.PointwiseFilters', pointwise_filter_values)

        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            spatial_norm_values = np.sqrt(np.sum(np.square(spatial_filter_values), axis=(0,1)))
            spatial_target_norms = lib.param(
                name + '.gSpatial',
                spatial_norm_values
            )
            pointwise_norm_values = np.sqrt(np.sum(np.square(pointwise_filter_values), axis=(0,1,2)))
            pointwise_target_norms = lib.param(
                name + '.gPointwise',
                pointwise_norm_values
            )
            with tf.name_scope('weightnorm') as scope:
                spatial_norms = tf.sqrt(tf.reduce_sum(tf.square(spatial_filters), reduction_indices=[0,1]))
                spatial_filters = spatial_filters * (spatial_target_norms / spatial_norms)
                pointwise_norms = tf.sqrt(tf.reduce_sum(tf.square(pointwise_filters), reduction_indices=[0,1,2]))
                pointwise_filters = pointwise_filters * (pointwise_target_norms / pointwise_norms)


        result = tf.transpose(inputs, [0,2,3,1])

        result = tf.nn.separable_conv2d(
            input=result, 
            depthwise_filter=spatial_filters,
            pointwise_filter=pointwise_filters, 
            strides=[1, stride, stride, 1],
            padding='SAME'
        )

        if biases:
            _biases = lib.param(
                name+'.Biases',
                np.zeros(output_dim, dtype='float32')
            )

            result = tf.nn.bias_add(result, _biases)

        result = tf.transpose(result, [0,3,1,2])

        # lib.debug.print_stats(name, result)

        return result