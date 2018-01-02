import numpy as np
import scipy.misc
import os
import time
# from PIL import Image

DATA_DIR = '/home/ubuntu/lsun/bedrooms/'
NEW_DATA_DIR = '/home/ubuntu/lsun/bedrooms_128/'

# with open(DATA_DIR+'files.txt', 'r') as f:
#     files = [l[:-1] for l in f]
# # images = np.zeros((batch_size, 3, 256, 256), dtype='int32')
# random_state = np.random.RandomState(42)
# random_state.shuffle(files)

# z = 1729468
# for i, path in enumerate(files):
#     if i < 1729500:
#         continue
#     try:
#         image = scipy.misc.imread(
#             os.path.normpath(os.path.join(DATA_DIR, path))
#         )

#         # try:                
#         # image = image.transpose(2,0,1)
#         offset_y = (image.shape[0]-256)/2
#         offset_x = (image.shape[1]-256)/2
#         image = image[offset_y:offset_y+256, offset_x:offset_x+256]
#         image = image[::2,::2]+image[1::2,::2]+image[::2,1::2]+image[1::2,1::2]
#         image = image / 4
#         # image = image.astype('int32')
#         # im = Image.fromarray(image)
#         # p = os.path.normpath(os.path.join(NEW_DATA_DIR, path))
#         # try:
#         #     os.makedirs(os.path.dirname(p))
#         # except:
#         #     pass
#         scipy.misc.imsave(NEW_DATA_DIR+'{}.jpg'.format(z), image)
#         # im.save(p[:-4]+'jpg')
#         if z % 100 == 0:
#             print z
#         z += 1
#     except:
#         print "skip"

#     # if i > 0 and i % batch_size == 0:
#     #     if downscale:
#     #         downscaled_images = images[:,:,::2,::2] + images[:,:,1::2,::2] + images[:,:,::2,1::2] + images[:,:,1::2,1::2]
#     #         downscaled_images = downscaled_images / 4.
#     #         yield (downscaled_images.astype('int32'),)
#     #     else:
#     #         yield (images,)
#     # except Exception as ex:
#     #     print ex
#     #     print "warning data preprocess failed for path {}".format(path)


def load(batch_size, downscale=False):
    def generator():
        with open(DATA_DIR+'files.txt', 'r') as f:
            files = [l[:-1] for l in f]
        images = np.zeros((batch_size, 3, 256, 256), dtype='int32')
        random_state = np.random.RandomState(42)
        random_state.shuffle(files)
        for i, path in enumerate(files):
            try:
                image = scipy.misc.imread(
                    os.path.normpath(os.path.join(DATA_DIR, path))
                )
            except Exception as ex:
                print ex
                print "warning data load failed for path {}".format(path)
            try:                
                image = image.transpose(2,0,1)
                offset_y = (image.shape[1]-256)/2
                offset_x = (image.shape[2]-256)/2
                images[i % batch_size] = image[:, offset_y:offset_y+256, offset_x:offset_x+256]
                if i > 0 and i % batch_size == 0:
                    if downscale:
                        downscaled_images = images[:,:,::2,::2] + images[:,:,1::2,::2] + images[:,:,::2,1::2] + images[:,:,1::2,1::2]
                        downscaled_images = downscaled_images / 4
                        yield (downscaled_images.astype('int32'),)
                    else:
                        yield (images,)
            except Exception as ex:
                print ex
                print "warning data preprocess failed for path {}".format(path)
    return generator

if __name__ == '__main__':
    train_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()
