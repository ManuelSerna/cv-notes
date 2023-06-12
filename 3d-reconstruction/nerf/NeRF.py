import torch
import torch.nn as nn


class NeRF(nn.Module):
    def __init__(self, in_dim:tuple=None, hidden_dim:int=256, activation_func=None):
        """ Original NeRF model.
        
        Input:
            in_dim: (tuple) input dimensionality of rays and directions (e.g., of position encodings, raw inputs)
                - Tuple is (R, D) such that:
                    rays tensor is of shape (1,R)
                    dirs tensor is of shape (1,D)
            hidden_dim: (int) dimensionality for all hidden layers
            activation_func: (torch.nn) activation function object
        """
        super().__init__()
        
        self.block1 = nn.Sequential(
            activation_func, 
            nn.Linear(in_dim[0], hidden_dim), # 1
            activation_func,
            nn.Linear(hidden_dim, hidden_dim), # 2
            activation_func,
            nn.Linear(hidden_dim, hidden_dim), # 3
            activation_func,
            nn.Linear(hidden_dim, hidden_dim), # 4
            activation_func, 
            nn.Linear(hidden_dim, hidden_dim), # 5
            activation_func
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim+in_dim[0], hidden_dim), # 6 (input concat)
            activation_func, 
            nn.Linear(hidden_dim, hidden_dim), # 7
            activation_func,
            nn.Linear(hidden_dim, hidden_dim), # 8
            activation_func, 
            nn.Linear(hidden_dim, hidden_dim), # 9 has no activation
        )
        
        # Now, we follow two branches:
        # 1. Output density with one more fully-connected layer
        self.density_out_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1), 
            nn.ReLU()
        )
        
        # 2. Output color information
        self.rgb_out_block = nn.Sequential(
            nn.Linear(hidden_dim+in_dim[1], hidden_dim//2), # 10
            nn.ReLU(), 
            nn.Linear(hidden_dim//2, 3), # output
            nn.Sigmoid()
        ) # concat with view direction and output rgb
        
        self.activation = activation_func
    
    def forward(self, rays, dirs):
        """ NeRF forward
        
        Input:
            rays: (torch.tensor) ray information
            dirs: (torch.tensor) direction information
        Output:
            rgb: (torch.tensor) predicted RGB (length-3 tensor)
            density: (torch.tensor) predicted density (length-1 tensor)
        """ 
        x = self.block1(rays)
        x = torch.cat((x, rays), -1)
        x = self.block2(x)
        
        density = self.density_out_layer(x)
        
        x = torch.cat((x, dirs), -1)
        rgb = self.rgb_out_block(x)
        
        out = torch.cat((rgb, density), -1)
        
        return out


if __name__ == "__main__":
    activation = nn.ReLU()
    
    R = 60
    rays = torch.rand(1, R)
    print(f'rays: {rays.shape}')
    
    D = 24
    dirs = torch.rand(1, D)
    print(f'dirs: {dirs.shape}')

    model = NeRF((R, D), 128, activation)
    print(model)
    
    yhat = model(rays, dirs)
    print(f'out ({yhat.shape}): {yhat}')

