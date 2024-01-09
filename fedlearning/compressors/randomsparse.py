import torch

# My libraries
from fedlearning.compressors.initialize import Compressor

class RandomSparsifier(Compressor):

    def __init__(self, config):
        super().__init__()
        self.A = config.A
        self.B = config.B
        self.lr = config.lr

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        A, B = self.A*self.lr, self.B*self.lr
        # B = tensor.abs().max()
        # A = 0.8*B

        random_variable = torch.rand_like(tensor)
#        ones_tensor = torch.ones_like(tensor)
        zeros_tensor = torch.zeros_like(tensor)
        
        
        # encoded_tensor = torch.where(random_variable<=(A+tensor)/(2*B), ones_tensor, zeros_tensor)
        # encoded_tensor = torch.where(random_variable>=1 - (A-tensor)/(2*B), -ones_tensor, encoded_tensor)
        
        encoded_tensor = torch.where(random_variable<= 1-A/B, zeros_tensor, tensor)
        return encoded_tensor
