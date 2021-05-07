#*********************************
# Module: Plotting-related functions, all of these functions are called by the module 'fourier_functions'
#
# Manuel Serna-Aguilera
# Spring 2021
#*********************************

import cv2
import matplotlib.pyplot as plt
import numpy as np



#=================================
# Plot sinusoids and their sum
'''
Input:
    t:        (numpy array) 1D array that holds inputs of sine functions.
    waves:    (numpy array) 2D array that holds outputs for sine functions.
    sum_wave: (integer) number of sine waves (default=4).

Return:
    N/A
'''
#=================================
def plot_sinusoids(t, waves, sum_wave, n=4):
    fig, axs = plt.subplots(n+1, sharex=True, sharey=True)
    fig.suptitle('Sine Waves')
    plt.xlabel('time t')

    # Plot individual sine waves
    for i in range(n):
        axs[i].plot(t, waves[i])
        axs[i].grid(True, which='both')
        axs[i].axhline(y=0, color='k')
        axs[i].set(ylabel='y = sin(t)')

    # Plot sum of sine waves
    axs[n].plot(t, sum_wave, 'tab:red')
    axs[n].grid(True, which='both')
    axs[n].axhline(y=0, color='k')
    axs[n].set(ylabel='y = sin(t)')

    plt.show()
