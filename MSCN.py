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


def normalize_kernel(kernel):
    return kernel / np.sum(kernel)


def gaussian_kernel2d(n, sigma):
    Y, X = np.indices((n, n)) - int(n / 2)
    gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    return normalize_kernel(gaussian_kernel)


def local_mean(image, kernel):
    return signal.convolve2d(image, kernel, 'same')


def local_deviation(image, local_mean, kernel):
    sigma = image ** 2
    sigma = signal.convolve2d(sigma, kernel, 'same')
    return np.sqrt(np.abs(local_mean ** 2 - sigma))


def calculate_mscn_coefficients(img, kernel_size, sigma):
    C = 1 / 255

    img = (np.asarray(img) / 255.0).astype(np.float32)
    img=np.reshape(img,(np.shape(img)[0],np.shape(img)[1],np.shape(img)[2]))
    img=skimage.color.rgb2gray(img)

    kernel = gaussian_kernel2d(kernel_size, sigma=sigma)
    local_mean = signal.convolve2d(img, kernel, 'same')
    local_var = local_deviation(img, local_mean, kernel)
    a = (img - local_mean) / (local_var + C)
    return np.reshape(a,np.shape(img)[0]*np.shape(img)[1]*np.shape(img)[2])
