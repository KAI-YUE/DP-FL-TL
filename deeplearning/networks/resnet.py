import torch.nn as nn


#class ResidualBlock(nn.Module):
#    """
#    A residual block as defined by He et al.
#    """
#
#    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
#        super(ResidualBlock, self).__init__()
#        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#                                   padding=padding, stride=stride, bias=False)
#        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
#                                   padding=padding, bias=False)
#
#        if stride != 1:
#            # in case stride is not set to 1, we need to downsample the residual so that
#            # the dimensions are the same when we add them together
#            self.downsample = nn.Sequential(
#                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)
#            )
#        else:
#            self.downsample = None
#
#        self.relu = nn.ReLU(inplace=False)
#
#    def forward(self, x):
#        residual = x
#
#        out = self.relu(self.conv_res1(x))
#        
#        if self.downsample is not None:
#            residual = self.downsample(residual)
#
#        out = self.relu(out)
#        out += residual
#        return out
#
#
#class RenNet9(nn.Module):
#    """
#    A Residual network.
#    """
#    def __init__(self, channel, num_classes, im_size):
#        super(RenNet9, self).__init__()
#
#        self.conv = nn.Sequential(
#            nn.Conv2d(in_channels=channel, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
#            nn.ReLU(inplace=False),
#            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
#            nn.ReLU(inplace=False),
#            nn.MaxPool2d(kernel_size=2, stride=2),
#            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
#            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
#            nn.ReLU(inplace=False),
#            nn.MaxPool2d(kernel_size=2, stride=2),
#            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
#            nn.ReLU(inplace=False),
#            nn.MaxPool2d(kernel_size=2, stride=2),
#            ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
#            nn.MaxPool2d(kernel_size=2, stride=2),
#        )
#
#        self.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
#
#    def forward(self, x):
#        out = self.conv(x)
#        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
#        out = self.fc(out)
#        return out

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
              nn.ReLU(inplace = True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, im_size):
        super(ResNet9, self).__init__()
        
        self.conv1 = conv_block(in_channels, 64)          #64 x 150 x 150
        self.conv2 = conv_block(64, 128, pool=True)       # 128 x 75 x 75
        self.res1 = nn.Sequential(conv_block(128, 128), 
                                  conv_block(128, 128))   # 128 x 75 x 75
        
        self.conv3 = conv_block(128, 256, pool=True)     #256 x 37 x 37
        self.conv4 = conv_block(256, 512, pool=True)     # 512 x 18 x 18
        self.res2 = nn.Sequential(conv_block(512, 512),
                                  conv_block(512, 512))  # 512 x 18 x 18
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), # 512 x 1 x 1
                                        nn.Flatten(),     #512
                                        nn.Dropout(0.2),  
                                        nn.Linear(512, num_classes))  # 512--> 10
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out