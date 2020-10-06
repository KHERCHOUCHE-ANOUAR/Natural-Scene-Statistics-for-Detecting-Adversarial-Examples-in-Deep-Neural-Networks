import collections
import numpy as np
import scipy.signal as signal
import scipy.special as special
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import PIL
import scipy.misc
import skimage
import imageio
from MSCN import calculate_mscn_coefficients
import tensorflow as tf
from PGD_attack import pgd

mnist = tf.keras.datasets.mnist
(_,_),(x_test,_) = mnist.load_data()

x_test = np.reshape(x_test,(-1,28,28,1))

for i in range(1000):

        img = x_test[i]
        img=pgd(img,i)
        mscn_coefficients = calculate_mscn_coefficients(img, 7, 7/6)
        #print(np.max(mscn_coefficients),np.min(mscn_coefficients))
        t = np.hstack((1, mscn_coefficients))
        if i == 0:
            v = t
        else:
            v = np.vstack((v, t))


        img = x_test[i]
        mscn_coefficients = calculate_mscn_coefficients(img, 7,7/6)
        t = np.hstack((0, mscn_histo))
        v = np.vstack((v, t))


print(np.shape(v))
np.savez_compressed('data_training', X=v[:, 1:], Y=v[:, 0: 1])
