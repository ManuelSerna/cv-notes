import cv2
import numpy as np
import os



def area1(w=0, h=0):
    """ General area equation """
    return w*h


def get_center(x, y, w, h):
    """ Return center of box given top left coordinates and dimensions """
    return (int(x+w/2), int(y+h/2))


def get_largest_blob(img=None, binary=None, verbose=False):
    """ Return bounding box of largest blob in binary image

    :param img: original image
    :param binary: corresponding binary image, where blobs are white
    :param verbose: output extra message and visualize results

    :return: quadruple consisting of: (top left x, top left y, width, height) of largest bounding box
    """
    max_area = 0
    max_index = 0

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if verbose:
        print('Number of contours detected: ', len(contours))

        contours_img = cv2.drawContours(img, contours, -1, (0,0,255), 2)

        # Draw all bounding boxes
        for ci in contours:
            tl_x, tl_y, w, h = cv2.boundingRect(ci)
            cv2.rectangle(contours_img, (tl_x, tl_y), (tl_x+w, tl_y+h), (255,0,0), 2)

        cv2.imshow('contours', contours_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Find largest bounding box
    for ci in contours:
        tl_x, tl_y, w, h = cv2.boundingRect(ci)
        temp_area = area1(w, h)

        if max_area < temp_area:
            max_area = temp_area
            max_index = ci

    return cv2.boundingRect(max_index)


def get_slice_image(image=None, box=[], pad_percent=0.1):
    """
    Return sliced portion of an image centered around a center point

    image: (numpy array) original image
    box: bounding box (returned from cv2.boundingRect which was returned by function get_largest_blob)
    pad_percent: percent of width and height to pad bounding box with (pad with surrounding image content)

    return: sliced image
    """
    W = image.shape[1]
    H = image.shape[0]

    xmin = box[0] - int(pad_percent * W)
    ymin = box[1] - int(pad_percent * H)
    xmax = box[0] + box[2] + int(pad_percent * W)
    ymax = box[1] + box[3] + int(pad_percent * H)

    # Check if moving min and max coords goes beyond image dims
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > W:
        xmax = W
    if ymax > H:
        ymax = H

    return image[ymin:ymax, xmin:xmax]



if __name__ == '__main__':
    index = 3
    test_img = cv2.imread('test.png')
    test_mask = cv2.imread('{}.png'.format(index)) # read binary img as grayscale
    binary = cv2.cvtColor(test_mask, cv2.COLOR_BGR2GRAY)

    box = get_largest_blob(img=test_mask, binary=binary, verbose=False)
    print('largest box: ', box)

    # Draw biggest bounding box
    '''cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255,255,0), 2)
    cv2.imshow('largest box', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    # Slice original image
    #height, width = binary.shape
    #center = get_center(box[0], box[1], box[2], box[3])
    #print('center of largest box: ', center)

    '''
    # Visualize center of largest bounding box
    cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255,255,0), 2)
    cv2.circle(img, center, 4, (0,0,255), 5)
    cv2.imshow('largest box', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()#'''

    test_img_slice = get_slice_image(image=test_img, box=box, pad_percent=0.1)
    test_mask_slice = get_slice_image(image=test_mask, box=box, pad_percent=0.1)

    cv2.imshow('image slice', test_img_slice)
    cv2.imshow('mask slice', test_mask_slice)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
