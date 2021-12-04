#*********************************
# Spatial Filtering Functions
#
# Manuel Serna-Aguilera
#*********************************

import numpy as np


def convolution_1d(array, w):
    """ Convolve a 1D array with a given 1D filter array, which it is assumed the length is 1, and the width
    (denoted k) is odd and at least length 3 (along axis 0).

    Convolve a 1D array with a given filter array. It is assumed the array is of shape (N,), where N is the
    length of the data array, and the filter is of shape (M,), where W is the length of the filter array.

    :param array: (numpy array) data array of shape (N,)
    :param w: (numpy array) filter array of shape (M,)
    :return: convolved array of integers
    """
    out = np.zeros(shape=array.shape)
    N = array.shape[0]
    k = w.shape[0]

    # Pad new data array and flip filter
    pad = int((k-1)/2)
    padded_data = np.pad(
        array=array,
        pad_width=pad,
        mode='constant',
        constant_values=0
    )

    flipped_filter = np.flip(w)

    # Perform filtering naively
    for n in range(N):
        for m in range(k):
            out[n] += flipped_filter[m]*padded_data[n+m]

    return out.astype(np.uint8)


def convolution_2d(image=None, w=None, scale=True, flip_filter=True):
    """ Convolve a 2D array with a given 2d filter array, which it is assumed the length and width (denoted k)
    of the filter are equal, odd, and at least k=3.

    :param image: (numpy array) image array
    :param w: (numpy array) filter array
    :param scale: (bool) scale array elements to range [0, 255]
    :param flip_filter: (bool) flip filter to perform spatial convolution (or not to perform spatial correlation)
    :return: convolved image with elements as integers
    """
    imgh, imgw = image.shape[:2]
    k = w.shape[0]

    out = np.zeros((imgh, imgw), dtype=np.float32) # create result array

    # Pad image
    pad = (k - 1) // 2
    padded_img = np.pad(
        array=image,
        pad_width=pad,
        mode='constant',
        constant_values=0
    )

    # Flip filter and then perform convolution
    if flip_filter:
        w_flip = np.flip(w)
    else:
        w_flip = w

    for iy in range(pad, imgh+pad):
        for ix in range(pad, imgw+pad):
            out[iy-pad, ix-pad] = (w_flip * padded_img[iy-pad:iy+pad+1, ix-pad:ix+pad+1]).sum()

    # Rescale output to range [0, 255]
    if scale:
        out = ((255.0 * (out - np.max(out))) / (np.max(out) - np.min(out)))

    return out.astype(np.uint8)


def get_box_filter(size):
    """ With a given size, return a box filter

    :param size: (int) length and width of filter, size must be positive and odd
    :return: numpy array
    """
    if size % 2 == 0:
        raise Exception('[ERROR] Filter size can only be odd.')

    # Since we know the array is all ones, we do not have to take the sum of all elements
    return np.ones((size, size), dtype=np.float32) / (size*size)


def sample_gaussian(K=1, stdev=1, s=0, t=0):
    """ Sample single value from 2D Gaussian function

    :param K: (number) general constant (not caring about Gaussian PDF in this case)
    :param stdev: (number) standard deviation of function
    :param s: (number) point to sample
    :param t: (number) point to sample
    :return: value from Gaussian function
    """
    if stdev == 0:
        raise Exception('[ERROR] St. dev. is zero, will cause divide by zero error!')

    return K * np.exp((-1 * (s*s + t*t)) / (2 * stdev * stdev))


def get_gaussian_filter(k=3, gen_const=1, stdev=1):
    """ Get 2D Gaussian filter

    :param k: (int) length and width of filter
    :param gen_const: (number) general constant (not caring about Gaussian PDF in this case)
    :param stdev: (number) standard deviation for filter
    :return: (numpy array) gaussian filter
    """
    if k % 2 == 0:
        raise Exception('[ERROR] Filter length/width needs to be odd!')

    out = np.zeros((k, k))

    start = -k//2+1
    end = -start
    indexing_offset = end  # offset so that iterating through matrix is done correctly

    for iy in range(start, end+1):
        for ix in range(start, end+1):
            out[iy+indexing_offset, ix+indexing_offset] = sample_gaussian(K=gen_const, stdev=stdev, s=iy, t=ix)

    return out / out.sum()


def median_filter(image=None, size=5):
    """ Apply median filter to image

    :param image: (numpy array) image array
    :param size: (int) size of median filter (length and width), it is assumed to be odd and positive
    :return: (numpy array) processed image
    """
    if size % 2 == 0:
        size += 1
    imgh, imgw = image.shape[:2]
    out = np.zeros((imgh, imgw), dtype=np.float32)  # create result array

    # Pad image
    pad = (size - 1) // 2
    padded_img = np.pad(
        array=image,
        pad_width=pad,
        mode='constant',
        constant_values=0
    )

    # Apply median filter
    for iy in range(pad, imgh + pad):
        for ix in range(pad, imgw + pad):
            roi = padded_img[iy - pad:iy + pad + 1, ix - pad:ix + pad + 1]
            med = np.median(roi)

            out[iy-pad, ix-pad] = med

    return out.astype(np.uint8)
