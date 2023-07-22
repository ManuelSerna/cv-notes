import numpy as np
import torch
import torch.nn as nn


# TODO: embedding module--position encoding
class PositionEncoding(nn.Module):
    def __init__(self, input_length=None, out_dim=None, n=10000):
        '''
        Positional Encoding Module
        
        Args:
            input_length: (int) length of input sequence ("n" in common notation)
            out_dim: (int) output dimensionality ("d" in common notation)
            n: (int) scalar ("Attention is you need paper" defaults to 10,000)
        '''
        super().__init__()
        
        self.input_length = input_length
        self.out_dim = out_dim
        self.n = torch.tensor([n], dtype=torch.float32)
        
        self.P = torch.zeros((input_length, out_dim), requires_grad=False)
        
        for k in range(input_length):
            for i in range(int(out_dim/2)):
                self.P[k, 2*i] = torch.sin(k / torch.pow(self.n, 2*i/out_dim))
                self.P[k, 2*i+1] = torch.cos(k / torch.pow(self.n, 2*i/out_dim))
        
    
    def forward(x):
        """ Forward
        
        Args:
            x: (torch.tensor) input of shape (batch size, ...)
        """
        
        
        return out


def position_encoding_np1(input_length=None, out_dim=None, n=10000):
    '''
    Positional Encoding Function
    
    Args:
        input_length: (int) length of input sequence ("n" in common notation)
        out_dim: (int) output dimensionality ("d" in common notation)
        n: (int) scalar ("Attention is you need paper" defaults to 10,000)
    Return:
        position encoding matrix P in numpy array form such that...
        
        P[k, 2i] = sin(k / n^{2i/d})
        P[k, 2i + 1] cos(k / n^{2i/d})
    '''
    pos_embed_matrix = np.zeros((input_length, out_dim))
    
    for k in range(input_length):
        for i in range(int(out_dim/2)):
            pos_embed_matrix[k, 2*i] = np.sin(k / np.power(n, 2*i/out_dim))
            pos_embed_matrix[k, 2*i+1] = np.cos(k / np.power(n, 2*i/out_dim))
    
    return pos_embed_matrix


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    origin = np.array([-0.05379832, 3.8454704, 1.2080823]) # sample point
    ray = np.array([0.37481974, -1.05678502, 0.0437651]) # sample direction vector
    
    ray_length = origin.shape[0]
    origin_pt_length = ray.shape[0]
    output_dim_rays = 60
    output_dim_origin = 60
    
    # Visualize simple example
    '''
    ray_encode = position_encoding_np1(ray_length, output_dim_rays)
    origin_encode = position_encoding_np1(origin_pt_length, output_dim_origin)
    cax = plt.matshow(ray_encode)
    plt.gcf().colorbar(cax)
    plt.show()
    '''
    
    # Test position encoding module
    ####(self, input_length=None, out_dim=None, n=10000, batch_size=4)
    P = PositionEncoding(input_length=ray_length, out_dim=output_dim_rays)
    
    import pdb;pdb.set_trace()
    
