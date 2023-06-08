import cv2 # only for reading and writing images
import matplotlib.pyplot as plt
import numpy as np


def resize_nearest_neighbors(img: np.array = None, new_dims: tuple = None):
    """Naive implementation of nearest neighbors interpolation.
    
    NOTE: The input image "img" is of shape (H, W, 3).
    """
    H, W, channels = img.shape
    H_new, W_new = new_dims
    
    res = np.zeros((H_new, W_new, channels))
    
    H_scale = H_new / (H-1)
    W_scale = W_new / (W-1)
    
    for y in range(H_new):
        for x in range(W_new):
            res[y, x, :] = img[int(y/H_scale), int(x/W_scale), :]
        
    return res.astype(np.uint8)
    


def resize_bilinear(img, scale):
    pass


def main():
    img = cv2.imread('butterfly.png')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"Image is of size {img.shape}.")
    
    # Define new dims
    scale_h = 3
    scale_w = 5
    
    H, W, channels = img.shape
    dims = (int(H*scale_h), int(W*scale_w))
    
    # Scale
    res_nn = resize_nearest_neighbors(img, dims)
    
    cv2.imwrite(f'nn_hx{scale_h}_wx{scale_w}.png', res_nn)


if __name__ == "__main__":
    main()
