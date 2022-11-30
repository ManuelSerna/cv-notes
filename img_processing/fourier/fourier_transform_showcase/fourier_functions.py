#*********************************
# Module: Fourier-Related Functions
#
# Manuel Serna-Aguilera
# Spring 2021
#*********************************

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
from scipy import ndimage

import my_plot_utils as my_plot



#=================================
# Create a single sinusoid
'''
Input:
    domain:    (numpy array) 1D aray of input values for sin function (default=null).
    amplitude: (float) determines amplitude of resulting sine wave (default=1.0).

Return:
    domain: (numpy array) domain values for sine waves
    range:  (numpy array) amplitude values of sine waves
'''
#=================================
def get_sinusoid(domain=None, amplitude=1.0):
    range = amplitude*np.sin(domain)
    return range



#=================================
# Create N new sinusoids
'''
Input:
    n: (integer) number of waves to create (default=4).

Return:
    t: (numpy array) 1D array that holds inputs of sine functions.
    y: (numpy array) 2D array that holds outputs for sine functions.
'''
#=================================
def get_N_sinusoids(n=4):
    # Error check
    if n<1:
        raise Exception('Error: should have 1 or more waves to generate.')

    # Setup max domain and max +- height for waves
    random.seed()
    t_min=0.0
    t_max=20.0
    step=0.1
    max_amplitude = 10.0

    t = np.arange(t_min, t_max, step) # array of inputs
    domain_length = t.size
    y = np.zeros((n, domain_length)) # array of sine outputs

    # Compute outputs for each sine function
    for i in range(n):
        amplitude = random.uniform(1.0, max_amplitude)
        y[i,:] = get_sinusoid(domain=t, amplitude=amplitude)

    return t, y



#=================================
# Get sum of N sinusoids
'''
Input:
    waves: (numpy array) 2D array of outputs of N sinusoids

Return:
    sum_wave: (numpy array) 1D array for sum wave
'''
#=================================
def get_sinusoid_sum(waves=None):
    sum_wave = np.sum(waves, axis=0)
    return sum_wave



#=================================
# Generate and display N sinusoids and their sum
'''
Input:
    n: (integer) number of sinusoids to create (default=4).
Return:
    N/A
'''
#=================================
def demo_sinusoids(n=4):
    t, y = get_N_sinusoids(n=n)
    sum_wave = get_sinusoid_sum(y)
    my_plot.plot_sinusoids(t=t, waves=y, sum_wave=sum_wave, n=n)



#=================================
# Showcase scaling property of FT on image
'''
Input:
    img: (numpy array) image
    scale: (number) scale factor
Return:
    N/A
'''
#=================================
def scale_showcase(img=None, scale=1):
    print('  scale factor: {}'.format(scale))
    rows, cols = img.shape
    new_dims = (int(scale*cols), int(scale*rows))

    #-----------------------------
    # i) Compute DFT of original image
    dft_original = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift_original = np.fft.fftshift(dft_original)
    mag_spectrum_original = 20*np.log(cv2.magnitude(dft_shift_original[:,:,0], dft_shift_original[:,:,1]))

    #-----------------------------
    # ii) Scale image in spatial domain (with bicubic interpolation) and take its DFT
    img_scaled = cv2.resize(img, new_dims, interpolation=cv2.INTER_CUBIC)
    #img_scaled = cv2.resize(img, new_dims, interpolation=cv2.INTER_NEAREST)
    dft_scaled = cv2.dft(np.float32(img_scaled), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift_scaled = np.fft.fftshift(dft_scaled)
    mag_spectrum_scaled = 20*np.log(cv2.magnitude(dft_shift_scaled[:,:,0], dft_shift_scaled[:,:,1]))

    #-----------------------------
    # iii) Scale DFT of original image and get IFFT of that
    # - Compute DFT of image
    # - Pad original's (center-shifted) FFT with zeros at end
    y_pad = int(rows/2)
    x_pad = int(cols/2)

    npad = ((y_pad, y_pad), (x_pad, x_pad), (0,0))
    padded_dft = np.pad(dft_shift_original, pad_width=npad, mode='constant', constant_values=1) # set pad vals to 1 as ln(1)=0
    mag_spectrum_padded = 20*np.log(cv2.magnitude(padded_dft[:,:,0], padded_dft[:,:,1]))

    rec = cv2.idft(np.fft.ifftshift(padded_dft))
    rec = cv2.magnitude(rec[:,:,0], rec[:,:,1])

    #-----------------------------
    # Plot results
    fig, axes = plt.subplots(nrows=2, ncols=3)
    ax = axes.ravel()

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original image')
    ax[3].imshow(mag_spectrum_original, cmap='gray')
    ax[3].set_title('Mag. spectrum of original')

    ax[1].imshow(img_scaled, cmap='gray')
    ax[1].set_title('Spatially scaled ({}x)\n(bicubic interpolation)'.format(scale))
    ax[4].imshow(mag_spectrum_scaled, cmap='gray')
    ax[4].set_title('Mag. spectrum of scaled')

    ax[2].imshow(rec, cmap='gray')
    ax[2].set_title('Reconstruction of scaled DFT')
    ax[5].imshow(mag_spectrum_padded, cmap='gray')
    ax[5].set_title('Scaled mag. spectrum\n(zero-padded)')

    plt.tight_layout()
    plt.show()



#=================================
# Showcase shifting property of FT on image
'''
Input:
    img: (numpy array) image
    x0: (integer) shift x-direction by this amount
    y0: (integer) shift y-direction by this amount
Return:
    N/A
'''
#=================================
def shift_showcase(img=None, x0=0, y0=0):
    print('  shift: ({}, {})'.format(x0, y0))
    rows, cols = img.shape
    M = np.float32([[1, 0, x0], [0, 1, y0]]) # transformation matrix for spatial domain

    #-----------------------------
    # Compute DFT of original image
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mag_spectrum_original = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

    #-----------------------------
    # Compute DFT of spatially-shifted image
    shifted = cv2.warpAffine(img, M, (cols,rows))

    dft_shifted = cv2.dft(np.float32(shifted), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted_shift = np.fft.fftshift(dft)
    mag_spectrum_shifted = 20*np.log(cv2.magnitude(dft_shifted_shift[:,:,0], dft_shifted_shift[:,:,1]))

    #-----------------------------
    # Reconstruct shifted image using IDFT
    dft_matrix = dft_shift[...,0] - 1j*dft_shift[...,1] # create 2d complex matrix
    x = np.linspace(start=int(-cols/2)+1, stop=int(cols/2), num=cols)
    y = np.linspace(start=int(-rows/2)+1, stop=int(rows/2), num=rows)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy') # mesh grid in x and y dirs

    # Perform shift according to shifting property
    #shifted_dft = np.multiply(dft_matrix, np.exp(-1j*2*np.pi*(xv*x0 + yv*y0)/cols)) # assuming rows=cols
    shifted_dft = np.multiply(dft_matrix, np.exp(np.multiply(-1j*2*np.pi, (xv*x0)/cols + (yv*y0)/rows) )) # rows!=cols

    # Make shifted fft visible
    mag_spectrum_ft = np.array([np.real(shifted_dft), np.imag(shifted_dft)]).transpose(1, 2, 0) # separate real and imaginary
    mag_spectrum_ft = 20*np.log(cv2.magnitude(mag_spectrum_ft[:,:,0], mag_spectrum_ft[:,:,1]))

    # Reconstruct combined image using IDFT
    rec = np.fft.ifft2(np.fft.ifftshift(shifted_dft))
    rec = np.real(rec)
    rec = (rec - np.min(rec))/np.ptp(rec) # get in range [0,1]

    #-----------------------------
    # Plot results
    fig, axes = plt.subplots(nrows=2, ncols=3)
    ax = axes.ravel()

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original')
    ax[3].imshow(mag_spectrum_original, cmap='gray')
    ax[3].set_title('Mag. spec. of original')

    ax[1].imshow(shifted, cmap='gray')
    ax[1].set_title('Spatially shifted (x0={}, y0={})'.format(x0, y0))
    ax[4].imshow(mag_spectrum_shifted, cmap='gray')
    ax[4].set_title('Mag. spec. of shifted')

    ax[2].imshow(rec, cmap='gray')
    ax[2].set_title('Reconstructed from shifted DFT')
    ax[5].imshow(mag_spectrum_ft, cmap='gray')
    ax[5].set_title('Shifted DFT')

    plt.tight_layout()
    plt.show()



#=================================
# Showcase rotation property of FT on image
# - rotating f(x,y) by theta rotates F(u,v) by theta
'''
Input:
    img: (numpy array) image
    angle: (number) rotation angle (deg)
Return:
    N/A
'''
#=================================
def rotate_showcase(img=None, angle=0.0):
    print('  rotating by: {} deg'.format(angle))
    #img = np.pad(array=img, pad_width=int(img.shape[1])) # pad image
    rows, cols = img.shape

    #-----------------------------
    # i) Compute DFT of original image
    dft_original = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift_original = np.fft.fftshift(dft_original)
    mag_spectrum_original = 20*np.log(cv2.magnitude(dft_shift_original[:,:,0], dft_shift_original[:,:,1]))

    #-----------------------------
    # ii) Rotate image in spatial domain and take its DFT
    img_rotate = ndimage.rotate(input=img, angle=angle, reshape=False, mode='constant')

    dft_rotate = cv2.dft(np.float32(img_rotate), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift_rotate = np.fft.fftshift(dft_rotate)
    mag_spectrum_rotate = 20*np.log(cv2.magnitude(dft_shift_rotate[:,:,0], dft_shift_rotate[:,:,1]))

    #-----------------------------
    # iii) Rotate DFT of original image and get IFFT of that
    y_pad = int(rows/2)
    x_pad = int(cols/2)
    npad = ((y_pad, y_pad), (x_pad, x_pad), (0,0))

    mod_dft = cv2.dft(np.fft.fftshift(np.float32(img)), flags=cv2.DFT_COMPLEX_OUTPUT)
    mod_dft = np.fft.fftshift(mod_dft)
    #pad_dft = np.pad(mod_dft, pad_width=npad, mode='constant', constant_values=1) # set pad vals to 1 as ln(1)=0
    pad_dft=mod_dft
    pad_dft_rotated = ndimage.rotate(input=pad_dft, angle=angle, reshape=False, mode='wrap') # rotate padded fft
    mag_spectrum_padded = 20*np.log(cv2.magnitude(pad_dft_rotated[:,:,0], pad_dft_rotated[:,:,1]))

    # reconstruct image
    rec = np.fft.ifftshift(cv2.idft(pad_dft_rotated))
    rec = cv2.magnitude(rec[:,:,0], rec[:,:,1])
    norm = np.linalg.norm(rec)
    rec = rec / norm # normalize (since values way too big)

    #-----------------------------
    # Plot results
    fig, axes = plt.subplots(nrows=3, ncols=2)
    ax = axes.ravel()

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(mag_spectrum_original, cmap='gray')
    ax[1].set_title('Mag. Spectrum of original')

    ax[2].imshow(img_rotate, cmap='gray')
    ax[2].set_title('Spatially rotated ({} deg)'.format(angle))
    ax[3].imshow(mag_spectrum_rotate, cmap='gray')
    ax[3].set_title('Mag. spectrum of spatial rotation')

    ax[4].imshow(rec, cmap='gray')
    ax[4].set_title('Rotated in frequency domain')
    ax[5].imshow(mag_spectrum_padded, cmap='gray')
    ax[5].set_title('Mag. spectrum rotated by\n{} deg in frequency domain'.format(angle))

    plt.tight_layout()
    plt.show()



#=================================
# Showcase linear property of FT on image
#  J{c1*g(t)+c2*h(t)} = c1*G(u)+c2*H(u)
'''
Input:
    img1: (numpy array) first image
    c1: (number) scalar for img1
    img1: (numpy array) second image
    c2: (number) scalar for img2
Return:
    N/A
'''
#=================================
def linear_showcase(img1=None, c1=1, img2=None, c2=1):
    print('  showcasing linear property with c1={} and c2={}'.format(c1, c2))
    # make sure images are same size (downscale larger image)
    rows1, cols1 = img1.shape
    rows2, cols2 = img2.shape
    rows = min([rows1, rows2])
    cols = min([cols1, cols2])
    dims = (cols, rows)

    img1 = cv2.resize(img1, dims, interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, dims, interpolation=cv2.INTER_CUBIC)

    #-----------------------------
    # Take DFT of both input images
    dft1 = cv2.dft(np.float32(img1), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(np.float32(img2), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift1 = np.fft.fftshift(dft1)
    dft_shift2 = np.fft.fftshift(dft2)
    mag_spectrum_original1 = 20*np.log(cv2.magnitude(dft_shift1[:,:,0], dft_shift1[:,:,1]))
    mag_spectrum_original2 = 20*np.log(cv2.magnitude(dft_shift2[:,:,0], dft_shift2[:,:,1]))

    #-----------------------------
    # Compute linear combination of images in spatial domain and get DFT
    combo = (c1 * img1) + (c2 * img2)
    dft_combo = cv2.dft(np.float32(img1), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift_combo = np.fft.fftshift(dft_combo)
    mag_spectrum_original_combo = 20*np.log(cv2.magnitude(dft_shift_combo[:,:,0], dft_shift_combo[:,:,1]))

    #-----------------------------
    # Take linear combination in freq. space and perform IFFT
    dft_combo = (c1*dft1) + (c2*dft2)

    # Reconstruct combined image using IDFT
    rec = cv2.idft(dft_combo)
    rec = cv2.magnitude(rec[:,:,0], rec[:,:,1])

    # Make combo fft visible
    dft_shift_ft_combo = np.fft.fftshift(dft_combo)
    mag_spectrum_ft = 20*np.log(cv2.magnitude(dft_shift_ft_combo[:,:,0], dft_shift_ft_combo[:,:,1]))

    #-----------------------------
    # Plot results
    #-----------------------------
    fig, axes = plt.subplots(nrows=2, ncols=4)
    ax = axes.ravel()

    ax[0].imshow(img1, cmap='gray')
    ax[0].set_title('Original 1')
    ax[4].imshow(mag_spectrum_original1, cmap='gray')
    ax[4].set_title('Mag. spectrum 1')

    ax[1].imshow(img2, cmap='gray')
    ax[1].set_title('Original 2')
    ax[5].imshow(mag_spectrum_original2, cmap='gray')
    ax[5].set_title('Mag. spectrum 2')

    ax[2].imshow(combo, cmap='gray')
    ax[2].set_title('Linear combination\nscalar 1={}, scalar 2={}'.format(c1, c2))
    ax[6].imshow(mag_spectrum_original_combo, cmap='gray')
    ax[6].set_title('Mag. spectrum of linear combination\n(via spatial domain)')

    ax[3].imshow(rec, cmap='gray')
    ax[3].set_title('Reconstructed combined image')
    ax[7].imshow(mag_spectrum_ft, cmap='gray')
    ax[7].set_title('Mag. spectrum of\nDFT linear combination')

    plt.tight_layout()
    plt.show()



#=================================
# Phase swap
'''
Input:
    img1: (numpy array) first image
    img2: (numpy array) second image
'''
#=================================
def showcase_phase_swap(img1=None, img2=None):
    # Make sure images are same size (downscale larger image)
    rows1, cols1 = img1.shape
    rows2, cols2 = img2.shape
    min_rows = min([rows1, rows2])
    min_cols = min([cols1, cols2])
    dims = (min_cols, min_rows)

    img1 = cv2.resize(img1, dims, interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, dims, interpolation=cv2.INTER_CUBIC)

    #-----------------------------
    # Take DFT of both input images
    dft1 = cv2.dft(np.float32(img1), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft2 = cv2.dft(np.float32(img2), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift1 = np.fft.fftshift(dft1)
    dft_shift2 = np.fft.fftshift(dft2)
    mag_spectrum_original1 = 20*np.log(cv2.magnitude(dft_shift1[:,:,0], dft_shift1[:,:,1]))
    mag_spectrum_original2 = 20*np.log(cv2.magnitude(dft_shift2[:,:,0], dft_shift2[:,:,1]))

    #-----------------------------
    # Get original phase images
    fft1 = np.fft.fft2(img1) # to actually swap, stick with only numpy...
    fft2 = np.fft.fft2(img2)
    phase1 = np.angle(fft1)
    phase2 = np.angle(fft2)

    #-----------------------------
    # Swap the phases of our images
    mag1 = np.abs(fft1)
    mag2 = np.abs(fft2)
    new_freq1 = np.multiply(mag1, np.exp(1j*phase2))
    new_freq2 = np.multiply(mag2, np.exp(1j*phase1))

    rec1 = np.fft.ifft2(new_freq1)
    rec2 = np.fft.ifft2(new_freq2)
    rec1 = np.real(rec1)
    rec2 = np.real(rec2)

    #-----------------------------
    # Plot results
    #-----------------------------
    fig, axes = plt.subplots(nrows=2, ncols=4)
    ax = axes.ravel()

    ax[0].imshow(img1, cmap='gray')
    ax[0].set_title('Original 1')
    ax[1].imshow(mag_spectrum_original1, cmap='gray')
    ax[1].set_title('Mag. spectrum 1')
    ax[2].imshow(phase1, cmap='gray')
    ax[2].set_title('Phase spectrum 1')
    ax[3].imshow(rec1, cmap='gray')
    ax[3].set_title('Phase swapped (with 2)')

    ax[4].imshow(img2, cmap='gray')
    ax[4].set_title('Original 2')
    ax[5].imshow(mag_spectrum_original2, cmap='gray')
    ax[5].set_title('Mag. spectrum 2')
    ax[6].imshow(phase2, cmap='gray')
    ax[6].set_title('Phase spectrum 2')
    ax[7].imshow(rec2, cmap='gray')
    ax[7].set_title('Phase swapped (with 1)')

    plt.tight_layout()
    plt.show()
