import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

#class TinyCNN(nn.Module):
#    def __init__(self, channel, num_classes, im_size, pretrained: bool = False):
#        super(TinyCNN, self).__init__()
#        self.in_channels = channel
#        self.input_size = im_size[0]
#        self.fc_input_size = int(self.input_size/4)**2 * 64
#        
#        self.conv1 = nn.Conv2d(self.in_channels,32, kernel_size=3, padding = 'same')
#        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
#        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
#        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
#        #self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding='same')
#        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)
#
#        #self.fc1 = nn.Linear(self.fc_input_size, 512)
#        self.fc1 = nn.Linear(4096, 256)
#        #self.fc2 = nn.Linear(120, 84)
#        self.fc3 = nn.Linear(256, num_classes)
#
#    # def forward(self, x):
#    #     x = F.relu(self.bn1(self.conv1(x)))
#    #     x = self.mp1(x)
#
#    #     x = F.relu(self.bn2(self.conv2(x)))
#    #     x = self.mp2(x)
#        
#    #     x = x.view(x.shape[0], -1)
#    #     x = F.relu(self.bn3(self.fc1(x)))
#    #     x = self.fc2(x)
#    #     return x
#    def forward(self, x):
#        x = F.relu(self.conv1(x))
#        x = F.relu(self.conv2(x))
#        x = self.mp1(x)
#        x = F.relu(self.conv3(x))
#        #x = F.relu(self.conv4(x))
#        x = self.mp2(x)
#        x = x.view(x.shape[0], -1)
#        x = F.relu(self.fc1(x))
#        #x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x

#2156490   
class TinyCNN(nn.Module):
    def __init__(self, channel, num_classes, im_size, pretrained: bool = False):
        super(TinyCNN, self).__init__()
        self.in_channels = channel
        self.input_size = im_size[0]
        self.fc_input_size = int(self.input_size/4)**2 * 64
        
        self.conv1 = nn.Conv2d(self.in_channels,32, kernel_size=5, padding = 'same')
        #self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding='same')
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding='same')
        #self.conv4 = nn.Conv2d(128, 128, kernel_size=5, padding='same')
        #self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
#        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
#        #self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        #self.mp3= nn.MaxPool2d(kernel_size=2, stride=2)

        #self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc1 = nn.Linear(4096, 512)
        #self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(512, num_classes)

    # def forward(self, x):
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = self.mp1(x)

    #     x = F.relu(self.bn2(self.conv2(x)))
    #     x = self.mp2(x)
        
    #     x = x.view(x.shape[0], -1)
    #     x = F.relu(self.bn3(self.fc1(x)))
    #     x = self.fc2(x)
    #     return x
    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        x = self.mp1(x)
        x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        #x = F.relu(self.conv5(x))
        x = self.mp2(x)
        #x = F.relu(self.conv5(x))
        #x = F.relu(self.conv6(x))
        #x = F.relu(self.conv7(x))
        #x = self.mp3(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        



#__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
#
#def _weights_init(m):
#    classname = m.__class__.__name__
#    #print(classname)
#    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#        init.kaiming_normal_(m.weight)
#
#class LambdaLayer(nn.Module):
#    def __init__(self, lambd):
#        super(LambdaLayer, self).__init__()
#        self.lambd = lambd
#
#    def forward(self, x):
#        return self.lambd(x)
#
#
#class BasicBlock(nn.Module):
#    expansion = 1
#
#    def __init__(self, in_planes, planes, stride=1, option='A'):
#        super(BasicBlock, self).__init__()
#        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#
#        self.shortcut = nn.Sequential()
#        if stride != 1 or in_planes != planes:
#            if option == 'A':
#                """
#                For CIFAR10 ResNet paper uses option A.
#                """
#                self.shortcut = LambdaLayer(lambda x:
#                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
#            elif option == 'B':
#                self.shortcut = nn.Sequential(
#                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
#                )
#
#    def forward(self, x):
#        out = F.relu((self.conv1(x)))
#        out = self.conv2(out)
#        out += self.shortcut(x)
#        out = F.relu(out)
#        return out
#
#
#class TinyCNN(nn.Module):
#    def __init__(self, channel, num_classes, im_size):
#        super(TinyCNN, self).__init__()
#        self.in_planes = 16
#        self.block = BasicBlock
#        self.num_blocks = [3, 3, 3]
#
#        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#        self.layer1 = self._make_layer(self.block, 16, self.num_blocks[0], stride=1)
#        self.layer2 = self._make_layer(self.block, 32, self.num_blocks[1], stride=2)
#        self.layer3 = self._make_layer(self.block, 64, self.num_blocks[2], stride=2)
#        self.linear = nn.Linear(64, num_classes)
#
#        self.apply(_weights_init)
#
#    def _make_layer(self, block, planes, num_blocks, stride):
#        strides = [stride] + [1]*(num_blocks-1)
#        layers = []
#        for stride in strides:
#            layers.append(block(self.in_planes, planes, stride))
#            self.in_planes = planes * block.expansion
#
#        return nn.Sequential(*layers)
#
#    def forward(self, x):
#        out = F.relu(self.conv1(x))
#        out = self.layer1(out)
#        out = self.layer2(out)
#        out = self.layer3(out)
#        out = F.avg_pool2d(out, out.size()[3])
#        out = out.view(out.size(0), -1)
#        out = self.linear(out)
#        return out


