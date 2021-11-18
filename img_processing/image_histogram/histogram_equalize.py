# Histogram equalization

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('bird.jpg',0)

# 1
'''
hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
'''

# 2
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imshow('Result', res)
#cv2.imwrite('res.png',res)
cv2.waitKey(0)
