"""
Lucas-Kanade Image Alignment
"""
import argparse
import cv2
import numpy as np


def get_Ix(img:np.array = None) -> np.array:
    """Get gradients for x-direction using Sobel filter """
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0
    k = 3
    
    blur = cv2.GaussianBlur(img, (k,k), 0)
    
    grad_x = cv2.Sobel(blur, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_x = cv2.convertScaleAbs(grad_x).astype(np.float64)
    
    return grad_x


def get_Iy(img:np.array = None) -> np.array:
    """Get gradients for x-direction using Sobel filter """
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0
    k = 3
    
    blur = cv2.GaussianBlur(img, (k,k), 0)
    
    grad_y = cv2.Sobel(blur, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.convertScaleAbs(grad_y).astype(np.float64)
    
    return grad_y


def warp_affine(point:np.array = None, p:np.array = None) -> np.array:
    """ Affine warp function on given 2d point (2x1 np array)
    
    Args:
        p: (numpy array) warp parameters
        point: (numpy array) 2d point whose elements are the coordinates of the template or original image; these are not pixel intensities
    Return:
        2d point that has been warped
    """
    return p.reshape(2,3) @ np.append(point, [[1]], axis=0)


def get_jacobian_affine(point:np.array = None) -> np.array:
    """ Compute Jacobian of affine transformation
    
    Args:
        point: (numpy array) 2d point whose elements are the coordinates of the template or original image; these are not pixel intensities
    Return:
        Jacobian matrix
    """
    return np.array([
        [point[0], 0.0, point[1], 0.0, 1.0, 0.0], 
        [0.0, point[0], 0.0, point[1], 1.0, 0.0]], 
        dtype=np.float32)


def get_hessian():
    pass


def get_deltap():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help="path to file to read")
    parser.add_argument("--rdir", help="path to *read* data from")
    parser.add_argument("--wdir", help="path to *write* data to")
    args = parser.parse_args()

    filepath = args.filepath
    read_dir = args.rdir
    write_dir = args.wdir
    
    # Initial guess of warp params
    p = np.array([1.1, 1.1, -5, 1.1, 1.1, -5], dtype=np.float32)
    


