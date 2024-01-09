import torch

# My libraries
from fedlearning.compressors.initialize import Compressor

class SignSGDCompressor(Compressor):

    def __init__(self, config):
        super().__init__()

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        encoded_tensor = (tensor >= 0).to(torch.float)
        return 2*encoded_tensor-1


class OptimalStoSignSGDCompressor(Compressor):
    
    def __init__(self, config):
        super().__init__()

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        random_variable = torch.rand_like(tensor)
        ones_tensor = torch.ones_like(tensor)
        b = tensor.abs().max()

        encoded_tensor = torch.where(random_variable<=(1/2+tensor/(2*b)), ones_tensor, -ones_tensor)
        return encoded_tensor