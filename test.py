import time

import tensorflow as tf
import numpy as np

from utils import *
from model import generator, discriminator

from skimage import measure

from PIL import Image


def test(image):
    tf.keras.backend.clear_session()

    # Process image
    image = tf.image.per_image_standardization(image)
    image = generator.predict(image)
    image = np.resize(image[0][56:, :, :], [144, 256, 3])
    imsave('output', image)
    return image

def denoise(image):
    # image = scipy.misc.imread(image, mode='RGB').astype('float32')

    image = np.array(Image.open(image).convert('RGB')).astype('float32') / 255.0
    npad = ((56, 56), (0, 0), (0, 0))
    image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)
    image = np.expand_dims(image, axis=0)
    print(image[0].shape)
    output = test(image)
    return output



if __name__=='__main__':
    # image = scipy.misc.imread(sys.argv[-1], mode='RGB').astype('float32')
    image = np.array(Image.open(sys.argv[-1]).convert('RGB')).astype('float32') / 255.0

    npad = ((56, 56), (0, 0), (0, 0))
    image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)
    image = np.expand_dims(image, axis=0)
    print(image[0].shape)
    test(image)
