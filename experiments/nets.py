import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

# each integer value represents the dimensions of a hidden layer
fc_configs = {'4-layer': [512, 256, 128, 64],
            '5-layer': [1024, 512, 256, 128, 64],
            '6-layer': [1024, 512, 256, 256, 128, 64],
            '7-layer': [1024, 512, 512, 256, 256, 128, 64],
            '8-layer': [1024, 512, 512, 256, 256, 128, 128, 64]}

class FcNet(nn.Module):

    def __init__(self, config: str, input_dim: tuple, classes: int, dropout: float=0.5):
        super(FcNet, self).__init__()
        self.config = config
        self.input_dim = np.prod(list(input_dim)) # ex (4, 4, 3) img -> 4 * 4 * 3 = 48
        self.relu = nn.ReLU()
        self.feed_forward = self.build()
        self.out = nn.Linear(64, classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        # flatten -> feed forward
        x = torch.flatten(x, 1)
        x = self.feed_forward(x)
        x = self.dropout(x)
        # apply output layer
        out = self.out(x)
        return out

    def build(self) -> nn.Sequential:
        layers = []
        config = fc_configs[self.config]
        for hidden_layer in config:
            layers.append(nn.Linear(self.input_dim, hidden_layer))
            layers.append(self.relu)
            self.input_dim = hidden_layer
        
        feed_forward = nn.Sequential(*layers)
        return feed_forward

# each inetegr value (i.e. 64) represents a layer in the VGG network. Conv layers are pooled
conv_filter = (3, 3)
conv_stride = 1
pool_filter = (2, 2)
pool_stride = 2
padding = 1

# vgg network architectures
vgg_configs = {'11-layer': [64, 'pool', 128, 'pool', 256, 256, 'pool', 512, 512, 'pool', 512, 512, 'pool'],
'13-layer': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 'pool', 512, 512, 'pool', 512, 512, 'pool'],
'16-layer': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool'],
'19-layer': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 256, 'pool', 512, 512, 512, 512, 'pool', 512, 512, 512, 512, 'pool']}

class VggNet(nn.Module):

    def __init__(self, config: str, channels: int, classes: int, dropout: float=0.5):

        super(VggNet, self).__init__()
        self.channels = channels
        self.config = vgg_configs[config]
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096) # pooled -> linear
        self.fc2 = nn.Linear(4096, 4096) # linear -> linear
        self.fc3 = nn.Linear(4096, classes) # linear -> output 
        self.build()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # (batch, channels, height, width)
        for layer in self.layers:
            if type(layer) == nn.Conv2d:
                x = f.relu(layer(x))
            else:
                x = layer(x) # pool layer or batch norm

        # average pool and flatten
        x = self.avgpool(x)
        x = x.contiguous().view(x.size(0), -1)

        # linear layers
        x = f.relu(self.fc1(x))
        x = self.drop(x)
        x = f.relu(self.fc2(x))
        x = self.drop(x)

        # apply output layer
        out = self.fc3(x)
        return out
        
    def build(self):
        self.layers = []

        for  layer in self.config:                
            if layer == 'pool':
                self.layers.append(nn.MaxPool2d(kernel_size=pool_filter, stride=pool_stride, 
                                                padding=padding)) # pool layer
            else:
                self.layers.append(nn.Conv2d(in_channels=self.channels, out_channels=layer, kernel_size=conv_filter, 
                                            stride=conv_stride, padding=padding)) # conv layer
                self.layers.append(nn.BatchNorm2d(layer)) # batch norm layer
                self.channels = layer # set next in channels

        self.layers = nn.Sequential(*self.layers) # create sequential layer (so cuda works)

class LeNet5(nn.Module):

    def __init__(self, channels: int, classes: int) -> None:
        super(LeNet5, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.conv3 = nn.Conv2d(16, 120, 5, 1)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, classes)

    def forward(self, x) -> torch.Tensor:
        x = x.permute(0, 3, 2, 1) # resahpe: (batch, height, width, channels)

        # conv + avg pool layers
        x = f.relu(self.conv1(x))
        x = self.avgpool(x)
        x = f.relu(self.conv2(x))
        x = self.avgpool(x)
        x = f.relu(self.conv3(x))

        # fc layers
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))

        # out layer
        out = self.fc2(x)
        return out


class AlexNet(nn.Module):

    def __init__(self, channels: int, classes: int, dropout: float=0.5) -> None:
        super(AlexNet, self).__init__()
        # conv layers
        self.conv1 = nn.Conv2d(channels, 96, 11, 4)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)

        # other layers
        self.maxpool = nn.MaxPool2d(3, 2)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm2d(96)
        self.norm2 = nn.BatchNorm2d(256)

        # fc layers
        self.fc1 = nn.Linear(256 * 5 * 5, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, classes)

    def forward(self, x) -> torch.Tensor:
        x = x.permute(0, 3, 2, 1) # reshape: (batch_size, channels, height, width)

        # (conv + relu + norm + pool) *2
        x = self.norm1(f.relu(self.conv1(x)))
        x = self.maxpool(x)
        x = self.norm2(f.relu(self.conv2(x)))
        x = self.maxpool(x)

        # (conv + relu) *3 + pool
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = f.relu(self.conv5(x))
        x = self.maxpool(x)

        # fc layers
        x = torch.flatten(x, 1)
        x = self.dropout(f.relu(self.fc1(x)))
        x = self.dropout(f.relu(self.fc2(x)))

        # out layer
        out = f.relu(self.fc3(x))
        return out

if __name__ == '__main__':
    # init nets & data
    convet = VggNet('11-layer', 3, classes=10)
    fcnet = FcNet('8-layer', input_dim=(224, 224, 3), classes=10)
    lenet = LeNet5(3, classes=10)
    alexnet = AlexNet(3, classes=10)
    images = torch.Tensor(torch.rand(16, 224, 224, 3))
    print(images.size())

    # use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    convet.to(device)
    fcnet.to(device)
    lenet.to(device)
    alexnet.to(device)
    images = images.to(device)


    # forward pass
    # print(lenet(images).shape)
    print(convet(images).shape)
    print(fcnet(images).shape)
    print(alexnet(images).shape)



        