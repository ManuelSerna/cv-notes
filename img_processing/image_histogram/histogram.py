# Naive Histogram Implementation in Python

import cv2
import matplotlib.pyplot as plt
import numpy as np

class Histogram():
    def __init__(self, filename='', max_intensity=256):
        """
        Constructor
        
        param: filename: (str) name of image file
        param: max_intensity: (int) maximum intensity for image
        
        return: NA
        """
        self.L = max_intensity
        self.bins = [0 for x in range(self.L)] # bins for pixel intensities
        self.filename = filename
        self.img_size = 0 # total size of current image (rows * cols)
        self.intensity_mappings = [-1 for x in range(self.L)] # transformation for changing histogram (this maps current pixel intensities to new ones)
    
    def change_filename(self, new_filename=''):
        """
        Change file name attribute
        
        param: filename: (str) name of image file
        
        return: NA
        """
        self.filename = new_filename
    
    def compute(self, img=None, normalized=True):
        """
        Compute histogram (of grayscale image)
        
        param: img: (numpy array) image array (if None, object will read in image using attribute filename)
        param: normalized: (bool) compute normalized histogram?
        
        return: NA
        """
        if img is None:
            img = cv2.imread(self.filename, 0)
        
        M = img.shape[0]
        N = img.shape[1]
        self.img_size = M*N
        
        # Add to bins
        for y in range(M):
            for x in range(N):
                #print(img[y,x])
                self.bins[img[y, x]] += 1
        
        # Normalize
        if normalized:
            for i in range(len(self.bins)):
            	self.bins[i] /= self.img_size
    
    def equalize(self):
    	"""
    	Perform histogram equalization
    	NOTE: assumes self.bins is normalized
    	
    	param: NA
    	
    	return: NA
    	"""
    	# Produce intensity mappings
    	#intensity_mappings = [-1 for i in range(len(self.bins))] # final output intensities
    	for k in range(len(self.intensity_mappings)):
    	    # Now the values at index k will map old intensity values to new intensity values,
    	    #  but keep in mind, several intensities can map to one intensity
    	    self.intensity_mappings[k] = round(self.L * (sum(self.bins[0:k+1])))

    def apply_to_img(self, img=None):
        """
        Apply current histogram (bins) to image, such as to increase contrast in current image
        after equalization

        param: img: (numpy array) image array (if None, object will read in image using attribute filename)

        return: (numpy array) image processed using histogram
        """
        if img is None:
            img = cv2.imread(self.filename, 0)

        M = img.shape[0]
        N = img.shape[1]

        for y in range(M):
            for x in range(N):
                i = img[y, x]
                img[y, x] = self.intensity_mappings[i]
        
        return img
    
    def plot(self, img=None):
    	"""
    	Plot given image or image from stored file name with current histogram
    	
    	param: img: (numpy array) image array (if None, object will read in image using attribute filename)
    	
    	return: NA
    	"""
    	if img is None:
            img = cv2.imread(self.filename, 0)
    	
    	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    	ax = axes.ravel()
    	
    	ax[0].imshow(img, cmap='gray', vmin=0, vmax=255)
    	ax[0].set_title('Image')
    	ax[1].bar([x for x in range(self.L)], self.bins)
    	ax[1].set_title('Histogram')
    	
    	plt.tight_layout()
    	plt.show()

