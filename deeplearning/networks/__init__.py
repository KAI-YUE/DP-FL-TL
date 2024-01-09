from deeplearning.networks.lenet import LeNet_5
from deeplearning.networks.vggnet import VGG9
from deeplearning.networks.densenn import DenseNN
from deeplearning.networks.cnn  import CNN
from deeplearning.networks.resnet  import ResNet9
from deeplearning.networks.resnet_bn  import ResNet9_bn
from deeplearning.networks.tinycnn  import TinyCNN
# from deeplearning.networks.densenet201  import DenseNet201
# from .vformer.models.classification.cct import CCT
from deeplearning.networks.resnet_gn import ResNet50_GN

nn_registry = {
    "lenet":          LeNet_5,
    "vgg9":             VGG9,
    "DenseNN":          DenseNN,
    "cnn":            CNN,
    
    "resnet9":        ResNet9,
    "resnet9_bn":     ResNet9_bn,
    "resnet50_gn":    ResNet50_GN,

    # "cct":            CCT,
    # "tinycnn":        TinyCNN,
    # "DenseNet201":    DenseNet201
}


