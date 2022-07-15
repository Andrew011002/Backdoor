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

class ConvNet(nn.Module):

    def __init__(self, config: str, channels: int, classes: int, dropout: float=0.5):

        super(ConvNet, self).__init__()
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
                x = layer(x) 
            elif type(layer) == nn.BatchNorm2d:
                x = f.relu(layer(x)) # apply relu after batch norm
            else:
                x = layer(x) # pool layer

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



if __name__ == '__main__':
    # init nets & data
    convet = ConvNet('11-layer', 3, classes=10)
    fcnet = FcNet('8-layer', input_dim=(28, 28, 3), classes=10)
    images = torch.Tensor(torch.rand(32, 28, 28, 3))
    print(images.size())

    # use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    convet.to(device)
    fcnet.to(device)
    images = images.to(device)


    # forward pass
    print(convet(images).shape)
    print(fcnet(images).shape)



        