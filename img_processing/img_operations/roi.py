# Image region of interest (ROI)

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('rgb.png')

roi = img[10:60, 10:110]
cv2.imshow('ROI', roi)

print('Press any key to continue.')
cv2.waitKey(0)
