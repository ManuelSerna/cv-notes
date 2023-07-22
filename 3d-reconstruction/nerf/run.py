"""NeRF code for training and testing.

Data file download link obtained via:
https://www.kaggle.com/code/rkuo2000/tiny-nerf

To run:

$ python run.py config.txt

where "config.txt" is the configuration file.

"""
import configargparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


from NeRFDataset import NeRFDataset
from NeRF import NeRF


def get_config_args():
    """Read configuration 'config_filename.txt' file or go with default configuration values."""
    parser = configargparse.ArgumentParser()

    parser.add_argument('config', is_config_file=True, help='config file path')

    # Input data config
    parser.add_argument('--exp_name', type=str, help='experiment name')
    parser.add_argument('--npz_file', type=str, help='.npz file path containing data') # TODO: make sure to make this mutually exclusive to other input settings

    # Output config
    # TODO

    # Training config
    parser.add_argument('--optimizer', type=str, help='optimizer: "adam"')
    parser.add_argument('--lr', type=str, help='learning rate')
    parser.add_argument('--n_epochs', type=int, help='number of epochs to train for')

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

    return np.array(rays).astype(np.float32), np.array(origins).astype(np.float32)


def sample_points_from_rays(near=0.1, far=1.0, num_pts=32):
    pts = np.linspace(start=near, stop=far, num=num_pts, endpoint=True, dtype=np.float32)
    
    return pts


def train_one_epoch():
    pass


def train():
    # Global settings
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Get parser configurations
    parser = get_config_args()
    config = parser.parse_args()
    print(f'[INFO] Read config "{config.config}"')

    # Load data
    image_data = None
    pose_data = None
    focal_data = None

    if config.npz_file != "":
        # .npz file was provided
        print(f'[INFO] Reading data from "{config.npz_file}"')
        with np.load(config.npz_file) as data:
            image_data = data['images']
            pose_data = data['poses']
            focal_data = data['focal']

            if focal_data.shape == ():
                # Weird shape of zero
                focal_data = np.array([focal_data])

    # Generate rays and prepare dataset objects
    rays, origins = compute_rays(image_data, pose_data, focal_data)
    
    # TODO: partition images into train and test sets
    
    rays = torch.from_numpy(rays)
    rays = torch.flatten(rays, 0, 1)
    origins = torch.from_numpy(origins)
    origins = torch.flatten(origins, 0, 1)
    
    # TODO: shuffle rays and origins
    
    train_dataset = NeRFDataset(rays, origins)

    # Sample points along rays
    # TODO: I should look at the tensorflow implementation and convert the engineering to python
    #all_sample_pts = sample_points_from_rays() # use default params for now
    
    # TODO: we will have to place EVERY operation in a nn.Module to make it learnable, including positional encoding, which should go inside a separate function; everything will have to learnable up until we get back to the pixels(?)
    
    
    
    # Get model
    activation = nn.ReLU()
    
    '''model = NeRF(
        in_dim=, 
        hidden_dim=256, 
        activation_func=None
    )'''
    
    # Get optimizer

    # Training loop
    # - will create embedding module and I can choose pos encoding as a choice (only that one for now)
    # TODO
    
    import pdb;pdb.set_trace()
    
    for i in range(args.n_epochs):
        train_one_epoch()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    
    train()
