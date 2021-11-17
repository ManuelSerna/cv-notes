# Simple Image Processing Functions

import cv2
import matplotlib.pyplot as plt
import numpy as np


def change_gamma(c=1, epsilon=0, gamma=1.0, img=None, normalize=True):
    """ Change gamma of image using eq: s = c(r+epsilon)^(gamma)

    c: constant
    epsilon: offset value, is usually only paid attention to in cases such as display calibration
    gamma: exponent value
    img: numpy array for image
    normalize: (Bool) normalize resulting image with max value

    return: (numpy array) resulting image
    """
    res = c * np.power(img + epsilon, gamma)

    if normalize:
        res = res / np.amax(res)

    return res


def contrast_stretch(r1=0, s1=0, r2=255, s2=255, img=None):
	"""
	Perform naive contrast stretching

	r1: input intensity threshold 1
	s1: intensity to map pixels of intensity r1
	r2: input intensity threshold 2
	s2: intensity to map pixels of intensity r2
	img: (numpy array) image array

    return: (numpy array) image array
	"""
	m1 = get_slope(0, 0, r1, s1)
	b1 = get_y_intercept(r1, s1, m1)
	m2 = get_slope(r1, s1, r2, s2)
	b2 = get_y_intercept(r2, s2, m2)
	m3 = get_slope(r2, s2, 255, 255)
	b3 = get_y_intercept(r2, s2, m3)

	for y in range(img.shape[0]):
		for x in range(img.shape[1]):
			if img[y, x] < r1:
				img[y, x] = get_y(m1, img[y, x], b1)
			elif img[y, x] >= r1 and img[y, x] <= r2:
				img[y, x] = get_y(m2, img[y, x], b2)
			elif img[y, x] > r2:
				img[y, x] = get_y(m3, img[y, x], b3)
	return img


def get_histogram(img=None, normalize=True):
    """ Get image histogram

    img: (numpy array) image
    normalize: (Bool) create normalized histogram (usually we do)?

    return:
    """
    L = 256 # number of intensities possible
    N = img.shape[0] # rows
    M = img.shape[1] # cols

    histogram = [0 for i in range(L)]
    for y in range(N):
        for x in range(M):
            intensity = img[y,x]
            histogram[intensity] += 1

    if normalize:
        histogram[:] = [i/img.size for i in histogram]

    return histogram


def get_negative(img):
    """ Naively get negative of image using equation s=L-1-r """
    L = 256
    return L - 1 - img


def get_slope(x1, y1, x2, y2):
    """ Get slope using two points """
    if x2 == x1:
        return 0
    else:
        return (y2-y1)/(x2-x1)


def get_y(m, x, b):
    """ Get y-intercept """
    return (m*x) + b


def get_y_intercept(x, y, m):
    """ Get y-intercept given point and slope """
    return y - (m*x)


def intensity_level_slice1(img, a, b, s1, s2):
    """
    Perform intensity level slicing where pixel intensities outside of [a,b]
    range are reduced to a low intensity

    img: (numpy array) image
    a: minimum input pixel intensity to highlight
    b: maximum input pixel intensity to highlight
    s1: set pixels within range [a,b] to be this intensity
    s2: set pixels outside of range [a,b] to be this intensity

    return: (numpy array) modified image
    """
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y,x] < a:
                img[y,x] = s2
            elif img[y,x] > b:
                img[y,x] = s2
            else:
                img[y,x] = s1

    return img


def intensity_level_slice2(img, a, b, s):
    """
    Perform intensity level slicing where pixel intensities outside of [a,b]
    range are left alone

    img: (numpy array) image
    a: minimum input pixel intensity to highlight
    b: maximum input pixel intensity to highlight
    s: set pixels within range [a,b] to be this intensity

    return: (numpy array) modified image
    """
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y,x] < b and img[y,x] > a:
                img[y,x] = s

    return img


def plot_histogram(histogram=None):
    """ Plot histogram from function 'get_histogram'

    histogram: list of pixel intensity counts

    return: NA
    """
    fig, ax = plt.subplots()
    ax.bar([i for i in range(len(histogram))], histogram)
    fig.tight_layout()
    plt.show()
