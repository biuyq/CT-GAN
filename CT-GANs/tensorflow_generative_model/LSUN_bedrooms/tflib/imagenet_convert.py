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

            try:

                image = Image.open("{}/{}".format(path, file))

                width, height = image.size   # Get dimensions
                new_width, new_height = min(width, height), min(width, height)
                left = (width - new_width)/2
                top = (height - new_height)/2
                right = (width + new_width)/2
                bottom = (height + new_height)/2
                image = image.crop((left, top, right, bottom))

                image.thumbnail((128,128), Image.ANTIALIAS)

                image.save("{}_128/{}".format(path, file))

                image = np.array(image)
                if image.shape == (128,128):
                    _image = np.empty((3,128,128), dtype='int32')
                    _image[:] = image
                    image = _image
                else:
                    if image.shape != (128,128,3):
                        print image.shape
                        continue
                    image = image.transpose(2,0,1)

                images[n % batch_size] = image

                if n > 0 and n % batch_size == 0:
                    yield (images,)

            except Exception as e:

                print "skipping"
                

    return get_epoch

def load(batch_size):
    return make_generator('/media/ramdisk/ILSVRC2012', batch_size)

if __name__ == '__main__':
    train_gen = load(128)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        # if i == 1000:
        #     break
        t0 = time.time()