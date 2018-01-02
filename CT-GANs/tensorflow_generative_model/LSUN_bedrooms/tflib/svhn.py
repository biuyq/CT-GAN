import numpy as np
import scipy.io
import scipy.misc
import time

def make_generator(paths, batch_size):
    data = scipy.io.loadmat(paths[0])
    data_X = data['X'].transpose(3,2,0,1)
    data_y = data['y'].flatten()
    for path in paths[1:]:
        data = scipy.io.loadmat(path)
        data_X = np.concatenate([data_X, data['X'].transpose(3,2,0,1)], axis=0)
        data_y = np.concatenate([data_y, data['y'].flatten()], axis=0)
    data_y -= 1 # turn labels from [1,10] to [0,9]
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(data_X)
        np.random.set_state(rng_state)
        np.random.shuffle(data_y)
        for i in xrange(len(data_X) / batch_size):
            yield (data_X[batch_size*i:batch_size*(i+1)],data_y[batch_size*i:batch_size*(i+1)])

    return get_epoch

def load(batch_size):
    return (
        make_generator(['/home/ishaan/data/svhn/train_32x32.mat', '/home/ishaan/data/svhn/extra_32x32.mat'], batch_size),
        make_generator(['/home/ishaan/data/svhn/test_32x32.mat'], batch_size)
    )

# if __name__ == '__main__':
#     train_gen, valid_gen = load(64)
#     t0 = time.time()
#     for i, batch in enumerate(train_gen(), start=1):
#         print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
#         if i == 1000:
#             break
#         t0 = time.time()