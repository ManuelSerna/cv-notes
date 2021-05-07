# OpenCV demo: Harris corner detection

import cv2
import numpy as np
import sys

filename = sys.argv[1]
out_filename = 'corners.png'

img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

# Result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

'''
cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
'''

cv2.imwrite(out_filename, img)
cv2.destroyAllWindows()
