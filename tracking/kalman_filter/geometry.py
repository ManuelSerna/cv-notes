"""
This is a mathematical utilities module.
"""

import cv2
import math
import numpy as np

from Box import Box


infty = 9999



def euclidean_distance(
    pt1:tuple = None, 
    pt2:tuple = None) -> float:
    """ Compute Euclidean distance between two points represented
    by tuples (x, y).
    
    Input:
        pt1: (tuple) 2D point in the form (x1, y1)
        pt2: (tuple) 2D point in the form (x2, y2)
    
    Return:
        Euclidean distance
    """
    return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)



def get_bboxes(
    mask:np.array = None,
    w_limit:int = 8,
    h_limit:int = 8,
    window:tuple = None
    ) -> list:
    """ Return bbox coordinates for each seperate 'blob' in a mask image
    as a list of lists. All variables will be made into integers.
    
    Input:
        mask: (numpy array) 2D array for mask image
        w_limit: (int) minimum threshold for the width boxes have to be to not be tossed out
        h_limit: (int) minimum threshold for the height boxes have to be to not be tossed out
        window: (tuple) 4-tuple where each element is described below, and
            this will be used to crop the OpenCV frame
            [0]: top-left x coordinate
            [1]: bottom-right x coordinate
            [2]: top-left y coordinate
            [3]: bottom-right y coordinate
    
    Return:
        bboxes: list of custom Box objects
    """
    if window is None:
        window = (0, infty, 0, infty) # set window very high if not cropping

    contours, hierarchy = cv2.findContours(
        mask, 
        cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    bboxes = []
    
    for cnt in contours:
        tlx, tly, w, h = cv2.boundingRect(cnt)

        if window[0] <= tlx <= window[1] and window[2] <= tly <= window[3]:
            if w >= w_limit and h >= h_limit:
                bboxes.append(
                    Box(int(tlx), int(tly), int(w), int(h))
                )
    
    return bboxes



def get_box_centroid(box:list = None) -> tuple:
    """ Compute the centroid of a box.
    
    Input:
        box: a length-4 list where the elements are defined below
            box[0]: top left x coordinate
            box[1]: top left y coordinate
            box[2]: width of box
            box[3]: height of box
    
    Return: 
        Centroid as a 2D point (center_x, center_y)
    
    NOTE: the centroid is returned as a float point
    """
    return (box[0] + box[2]/2.0, box[1] + box[3]/2.0)



def get_box_centroids(boxes:list = None) -> list:
    """ Get centroids for a list of boxes.
    
    Input:
        boxes: contains boxes, each a length-4 list where the elements are defined below
            box[0]: top left x coordinate
            box[1]: top left y coordinate
            box[2]: width of box
            box[3]: height of box
    
    Return: list of centroids for corresponding boxes in given input list
    """
    centroids = []
    
    for box in boxes:
        centroids.append(get_box_centroid(box))
    
    return centroids

