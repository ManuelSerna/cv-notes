# Use a trackbar to change min and max threshold values
#  for Canny edge detection.

import cv2
import numpy as np

def nothing(x):
    pass

# Set up image
img = cv2.imread('painting.jpeg',0)
cv2.namedWindow('canny')

# Create trackbars for min and max thresholds
cv2.createTrackbar('min', 'canny', 0, 255, nothing)
cv2.createTrackbar('max', 'canny', 0, 255, nothing)

# Display Canny edge detection until user quits (esc)
while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    minVal = cv2.getTrackbarPos('min', 'canny')
    maxVal = cv2.getTrackbarPos('max', 'canny')
    edges = cv2.Canny(img, minVal, maxVal)
    cv2.imshow('canny', edges)

cv2.destroyAllWindows()

