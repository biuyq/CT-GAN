import numpy as np
import scipy.misc
import time

def make_generator(path, n_files, batch_size, pad):
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, 3, 32, 32), dtype='int32')
        files = range(n_files)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            image = scipy.misc.imread("{}/{}.png".format(path, str(i+1).zfill(pad)))
            images[n % batch_size] = image.transpose(2,0,1)
            if n % batch_size == (batch_size-1):
                yield (images,)
    return get_epoch

def load(batch_size, dev_set_size=49999):
    return (
        # make_generator('/media/ramdisk/train_32x32', 1281149, batch_size, len(str(1281149))),
        # None
        make_generator('/home/ishaan/data/imagenet32/train_32x32', 1281149, batch_size, len(str(1281149))),
        make_generator('/home/ishaan/data/imagenet32/valid_32x32', dev_set_size, batch_size, len(str(49999)))
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()