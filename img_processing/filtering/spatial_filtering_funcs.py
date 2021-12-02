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


def convolution_2d(image=None, w=None, scale=True):
    """ Convolve a 2D array with a given 2d filter array, which it is assumed the length and width (denoted k)
    of the filter are equal, odd, and at least k=3.

    :param image: (numpy array) image array
    :param w: (numpy array) filter array
    :param scale: (bool) scale array elements to range [0, 255]
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
    w_flip = np.flip(w)

    for iy in range(pad, imgh+pad):
        for ix in range(pad, imgw+pad):
            out[iy-pad, ix-pad] = (w_flip * padded_img[iy-pad:iy+pad+1, ix-pad:ix+pad+1]).sum()

    # Rescale output to range [0, 255]
    if scale:
        out = ((255.0 * (out - np.max(out))) / (np.max(out) - np.min(out)))

    return out.astype(np.uint8)
