"""NeRF code for training and testing.

Data file download link obtained via:
https://www.kaggle.com/code/rkuo2000/tiny-nerf

description here...
"""
import configargparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from Nerf import *


def get_config_args():
    """Read configuration 'config_filename.txt' file or go with default configuration values."""
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, help='config file path')

    # Input data config
    parser.add_argument('--exp_name', type=str, help='experiment name')
    parser.add_argument('--npz_file', type=str, help='.npz file path containing data') # TODO: make sure to make this mutually exclusive to other input settings

    # Output config
    # TODO

    # Training config
    # TODO

    # Evaluation config
    # TODO

    return parser


def compute_rays(image_data, pose_data, focal_data):
    """Compute rays r(t) in the form

    r(t) = o + td

    where o and d are the origin and direction vectors, respectively.

    Input:
        image_data: image data, shape (Num images, Height, Width, 3)
        pose_data: corresponding camera extrinsics, shape (Num images, 4, 4)
        focal_data: numpy array of single element; shape (1,)
    Return:
        TODO description of return numpy array
    """
    rays = None
    N = image_data.shape[0]
    H = image_data.shape[1]
    W = image_data.shape[2]

    # Iterate through all images
    for idx in range(N):
        # Get location grid for points in the image
        x = np.meshgrid(np.linspace(0, W, W, False), indexing='xy')[0]
        y = np.meshgrid(np.linspace(0, H, H, False), indexing='xy')[0]

        # Define camera coordinate frame (x_c, y_c) points
        # NOTE: origins are simply center location in image
        x_camera = (x - 0.5*W) / focal_data
        y_camera = (y - 0.5*H) / focal_data

        # Compute directions


        # Get world coordinate frame points
        #x_world =
        #y_world =

        import pdb;pdb.set_trace()

    return rays

def train():
    # Global settings
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Get parser configurations
    parser = get_config_args()
    config = parser.parse_args()
    print(f'[INFO] Read config "{config.config}".')

    # Load data
    image_data = None
    pose_data = None
    focal_data = None

    if config.npz_file != "":
        print(f'[INFO] Reading data from "{config.npz_file}".')
        with np.load(config.npz_file) as data:
            image_data = data['images']
            pose_data = data['poses']
            focal_data = data['focal']

            if focal_data.shape == ():
                # Weird shape of zero
                focal_data = np.array([focal_data])

    # Generate rays
    rays = compute_rays(image_data, pose_data, focal_data)

    # Training loop
    # TODO


if __name__ == "__main__":
    train()
