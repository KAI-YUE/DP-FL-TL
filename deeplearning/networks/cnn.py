import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, channel, num_classes, im_size, pretrained: bool = False):
        super(CNN, self).__init__()
        self.in_channels = channel
        self.input_size = im_size[0]
        self.fc_input_size = int(self.input_size/4)**2 * 64
        
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=5, padding=2)
#        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=5, padding=2)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        #self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

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
        x = self.mp1(x)

        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

        
