# Canny edge detection

'''
Multi-step algorithm
    1. Reduce noise: apply 5x5 Gaussian filter.
    2. Apply x and y Sobel filter (Gx and Gy, respectively).
    3. Non-maximum suppression: if a pixel is not a local max
(which will denote an edge), then reduce intensity val to 0.
Keep pixels along edge (x or y direction) at a higher intensity.
    4. Hysteresis thresholding: decide which "edges" are actually
edges. Use two threshold values--minVal and maxVal. Pixel intensities
above maxVal are considered edges, as well as pixels connected
to these for-sure edges. Anything not connected to the for-sure
edges or values below minVal are thrown out. This stage also removes
small noises.

OpenCV does all the above steps with a single function.
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('flan.jpg',0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
