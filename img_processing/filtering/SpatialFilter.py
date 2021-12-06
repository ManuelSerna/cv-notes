#*********************************
# Spatial Filter Object
#
# Manuel Serna-Aguilera
#*********************************

import numpy as np


class SpatialFilter():
    def __init__(self):
        self.sobelx = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        self.sobely = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])

        self.laplacian1 = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])
        self.laplacian2 = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])
        self.laplacian3 = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ])
        self.laplacian4 = np.array([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ])

        self.gaussian = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]) / 16

        self.avg = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]) / 9
