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

    # do more
    # TODO


if __name__ == "__main__":
    train()
