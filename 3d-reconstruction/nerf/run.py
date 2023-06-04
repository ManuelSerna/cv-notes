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
        tuple: rays array, origins array
    """
    rays = []
    origins = []

    N = image_data.shape[0]
    H = image_data.shape[1]
    W = image_data.shape[2]

    # Get W*H locations grid for points on the image plane
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')

    # Define camera coordinate frame (x_c, y_c, 1.) points
    # NOTE: origins are simply center location in image (origin = 1/2 * height or width)
    camera_pts = np.stack([(x - 0.5 * W) / focal_data, -((y - 0.5 * W) / focal_data), -np.ones_like(x)], axis=-1) # (W, H, 3)
    camera_pts = camera_pts.reshape(-1, camera_pts.shape[-1])

    # Iterate through all images to get rays
    for idx in range(N):
        # Compute directions
        rays.append(np.array([pose_data[idx,:3,:3].dot(cam_pt) for cam_pt in camera_pts]))

        # Compute origins
        origins.append(np.broadcast_to(pose_data[idx,:3,-1], (W*H, 3))) # copy translations W*H times

    return np.array(rays), np.array(origins)

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
        # .npz file was provided
        print(f'[INFO] Reading data from "{config.npz_file}".')
        with np.load(config.npz_file) as data:
            image_data = data['images']
            pose_data = data['poses']
            focal_data = data['focal']

            if focal_data.shape == ():
                # Weird shape of zero
                focal_data = np.array([focal_data])

    # Generate rays
    rays, origins = compute_rays(image_data, pose_data, focal_data)

    import pdb;pdb.set_trace()

    # Sample points along rays
    # TODO

    # Training loop
    # TODO


if __name__ == "__main__":
    train()
