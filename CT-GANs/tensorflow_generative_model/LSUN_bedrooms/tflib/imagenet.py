import numpy as np
import scipy.misc
import time

import subprocess
import Image

def make_generator(path, batch_size):
    epoch_count = [1]
    files = subprocess.check_output("ls {}".format(path), shell=True).split("\n")[:-1]
    def get_epoch():
        images = np.zeros((batch_size, 3, 128, 128), dtype='int32')
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, file in enumerate(files):

            image = Image.open("{}/{}".format(path, file))

            image = np.array(image)
            if image.shape == (128,128):
                _image = np.empty((3,128,128), dtype='int32')
                _image[:] = image
                image = _image
            else:
                if image.shape != (128,128,3):
                    continue
                image = image.transpose(2,0,1)

            images[n % batch_size] = image

            if n > 0 and n % batch_size == 0:

                # Random horizontal flips
                if np.random.uniform() > 0.5:
                    images = images[:,:,:,::-1]

                yield (images,)

    return get_epoch

def load(batch_size):
    return make_generator('/home/crcv/xiang/Tensorflow_DCGAN-master/db/lsun/data', batch_size)
    #return make_generator('/home/bigdata/Desktop/ILSVRC2012_128', batch_size)
    #return make_generator('/home/bigdata/Downloads/lsun-master/data', batch_size)
    # return make_generator('/media/ramdisk/ILSVRC2012_128', batch_size)
    # return make_generator('/home/ishaan/data/ILSVRC2012_128', batch_size)

if __name__ == '__main__':
    train_gen = load(128)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()
