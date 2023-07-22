import torch
from torch.utils.data import Dataset, DataLoader


class NeRFDataset(Dataset):
    def __init__(self, rays=None, origins=None, batch_size=1024):
        """ NeRF Dataset object
        
        Args:
            rays: tensor of shape (N, 3) where N is the number of rays
            origins: tensor of shape (N, 3)
            batch_size: (int) number of rays and origins to get per epoch
        """
        self.rays = rays
        self.origins = origins
        self.batch_size = batch_size
        
        print('DEBUG! remember to shuffle rays and origins!') # TODO: remove after implementing shuffling
    
    def __len__(self):
        return self.rays.shape[0]
    
    def shuffle_data(self):
        # TODO: my idea is to call shuffle whenever we begin a new epoch
        pass

    def __getitem__(self, idx):
        ray = self.rays[idx]
        origin = self.origins[idx]
        sample = {"ray": ray, "origin": origin}
        return sample


if __name__ == "__main__":
    drays = torch.rand(8, 3)
    dorigins = torch.rand(8, 3)
    
    dataset = NeRFDataset(drays, dorigins)
    import pdb;pdb.set_trace()
