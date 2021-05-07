# Image transformations

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Scaling
'''
img = cv2.imread('rectangle.png',0)
#res = cv2.resize(img, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC) # double size
res = cv2.resize(img, None, fx=1/2, fy=1/2, interpolation = cv2.INTER_CUBIC) # half size
cv2.imshow('original image', img)
cv2.imshow('scaled image', res)
cv2.waitKey(0)
'''


# Translation
'''
img = cv2.imread('rectangle.png',0)
rows, cols = img.shape

#
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('img translation',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


# Rotation
img = cv2.imread('rectangle.png',0)
rows,cols = img.shape

# Rotate 90 deg wrt center of image
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('img rotation',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Affine Transformation
'''
img = cv2.imread('rectangle.png')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
'''


# Perspective Transformation
'''
img = cv2.imread('rectangle.png')
rows,cols,ch = img.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]]) # window to focus on on input img
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])# size of output image

M = cv2.getPerspectiveTransform(pts1,pts2) # transformation M

dst = cv2.warpPerspective(img,M,(300,300))# img x M

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
'''
