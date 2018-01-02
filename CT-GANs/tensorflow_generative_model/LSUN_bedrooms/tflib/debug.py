import numpy as np
import tensorflow as tf

_names = []
_vals = []

def print_stats(name, x):
    _names.append(name)

    x = tf.reshape(x, [-1])
    mean, std = tf.nn.moments(x, [0])
    _vals.append(mean)
    _vals.append(tf.sqrt(std))
    _vals.append(tf.reduce_min(x))
    _vals.append(tf.reduce_max(x))

def print_all_stats(feed_dict):
    if len(_names) == 0:
        return
    sess = tf.get_default_session()
    vals = sess.run(_vals, feed_dict=feed_dict)
    for i in xrange(len(_names)):
        print "{}:\tmean:{}\tstd:{}\tmin:{}\tmax:{}".format(
            _names[i], 
            vals[4*i],
            vals[(4*i)+1],
            vals[(4*i)+2],
            vals[(4*i)+3]
        )