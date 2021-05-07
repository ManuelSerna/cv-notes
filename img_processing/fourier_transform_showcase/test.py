#*********************************
# Test Driver code for showcasing Fourier properties
#  call:  $ python test.py
#
# Manuel Serna-Aguilera
# Spring 2021
#*********************************

import cv2
import numpy as np
import sys

import fourier_functions as ff

#---------------------------------
# 1. Create n=4 sinusoids, get their sum, and display all waves
#---------------------------------
print('Demo: adding sinusoids...')
ff.demo_sinusoids(n=4)

#---------------------------------
# 2. Demonstrate several properties of FT
#---------------------------------
# Read in images
filename1 = 'rec1.png'
filename2 = 'circle1.png'
filename3 = 'butterfly.png'

img1 = cv2.imread(filename1, 0) # read as grayscale
img2 = cv2.imread(filename2, 0)
img3 = cv2.imread(filename3, 0)

# 2.1--Scaling DFT
print('Demo: scaling...')
ff.scale_showcase(img3, 2)

# 2.2--shifting DFT
print('Demo: shifting...')
ff.shift_showcase(img2, x0=-11, y0=-32)

# 2.3--rotating DFT
print('Demo: rotating...')
ff.rotate_showcase(img1, 125.0)

# 2.4--linear DFT
print('Demo: linear...')
ff.linear_showcase(img1=img1, c1=0.5, img2=img2, c2=2)



#---------------------------------
# 3. Phase swap
#---------------------------------
print('Demo: Phase swap...')
ff.showcase_phase_swap(img1, img2)

print('Done!')
