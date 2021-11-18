# 2D Histograms

'''
1D histograms only take one feature into account like the grayscale intensity values of the pixels.

2D histograms consider two features. Two common features are hue and saturation values of the pixels.
First, convert bgr to hsv.

params for cv2.calcHist()
    channels = [0, 1], need to process both H and S plane.
    bins = [180,256] 180 for H plane and 256 for S plane.
    range = [0,180,0,256] Hue value lies between 0 and 180 & Saturation lies between 0 and 256.
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('bird.jpg')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# vertical axis: hue; horizontal axis: saturation
plt.imshow(hist, interpolation = 'nearest')
plt.xlabel('Saturation [0, 255]')
plt.ylabel('Hue [0, 179]')
plt.show()
