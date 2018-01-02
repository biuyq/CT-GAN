import numpy as np
import scipy.misc
import time

def load(batch_size):
    def get_epoch():
        f1 = scipy.misc.imread("/home/ubuntu/lsun256_test/1.jpg")
        f2 = scipy.misc.imread("/home/ubuntu/lsun256_test/2.jpg")
        images = np.zeros((batch_size, 3, 256, 256), dtype='int32')
        for i in xrange(batch_size):
            if i%2==0:
                images[i] = f1.transpose(2,0,1)
            else:
                images[i] = f2.transpose(2,0,1)
        while True:
            yield (images,)

    return (get_epoch, get_epoch)