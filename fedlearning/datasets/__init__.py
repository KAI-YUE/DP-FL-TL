from fedlearning.datasets.fmnist import FashionMNIST
from fedlearning.datasets.mnist import MNIST
from fedlearning.datasets.cifar10 import CIFAR10
from fedlearning.datasets.cifar10_TL import CIFAR10_224
from fedlearning.datasets.mydataset import fetch_dataloader


dataset_registry = {
    "fmnist": FashionMNIST,
    "mnist": MNIST,

    "cifar10": CIFAR10,
    "cifar10_224": CIFAR10_224,
}

def fetch_dataset(config):
    dataset = dataset_registry[config.dataset](config.data_path)

    config.num_classes = dataset.num_classes
    config.im_size = dataset.im_size
    config.channel = dataset.channel

    if not hasattr(config, 'pytorch_builtin'):
        config.pytorch_builtin = False

    return dataset
