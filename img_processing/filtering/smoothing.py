# Image smoothing operations

import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('logo.png')

# 5x5 averaging filter F (times 1/25)
'''
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
#'''

#img = cv2.imread('logo_noise.png')

# 4x4 averaging filter
#blur = cv2.blur(img,(5,5))

# Gaussian filter
#blur = cv2.GaussianBlur(img,(5,5),0)

# Median filter
#median = cv2.medianBlur(img, 5)

# Bilateral filtering (preserves edges)
blur = cv2.bilateralFilter(img,9,75,75)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
#plt.subplot(122),plt.imshow(median),plt.title('Median')
plt.xticks([]), plt.yticks([])
plt.show()
#'''
