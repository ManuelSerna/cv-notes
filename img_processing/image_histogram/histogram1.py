#*********************************
# Get image histogram using OpenCV
#
# Manuel Serna-Aguilera
# Spring 2021
#*********************************

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read in image
parser = argparse.ArgumentParser(description='Image Histogram Demo.')
parser.add_argument('--img', required=True, help="image to demo.")
parser.add_argument('--color', required=True, type=int, help='taking in color image (0=no, 1=yes)?')
args = parser.parse_args()

bins = 256 # bins for histogram
color = args.color
img = None
img_name = args.img

print("'Image: {}''".format(img_name))

# Get and plot histogram
if color == 0:
	img = cv2.imread(img_name, 0)
	hist = cv2.calcHist(images=[img], channels=[0], mask=None, histSize=[bins], ranges=[0, 256])

	plt.plot(hist)
	plt.xlim([0,256])
	plt.xlabel('Pixel Color Intensity')
	plt.ylabel('Pixels')
	plt.show()

elif color == 1:
	img = cv2.imread(img_name)
	color = ('b','g','r')

	for i,col in enumerate(color):
	    histr = cv2.calcHist(images=[img], channels=[i], mask=None, histSize=[256], ranges=[0,256])
	    plt.plot(histr, color = col)
	    plt.xlim([0,256])
	plt.xlabel('Pixel Color Intensity')
	plt.ylabel('Pixels')
	plt.show()

else:
	raise Exception('Wrong color value (0 or 1)!')
