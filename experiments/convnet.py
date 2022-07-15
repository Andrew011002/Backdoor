import torch
import torch.nn as nn
import torch.nn.functional as f


# #ach inetegr value (i.e. 64) represents a layer in the VGG network. Conv layers are pooled
conv_filter = (3, 3)
conv_stride = 1
pool_filter = (2, 2)
pool_stride = 2
padding = 1

class ConvNet(nn.Module):

    def __init__(self, architecture: str, channels: int, classes: int, local_response_norm=False):

        # vgg network architectures
        config = {'11-layer': [64, 'pool', 128, 'pool', 256, 256, 'pool', 512, 512, 'pool', 512, 512, 'pool'],
        '13-layer': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 'pool', 512, 512, 'pool', 512, 512, 'pool'],
        '16-layer': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool'],
        '19-layer': [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 256, 'pool', 512, 512, 512, 512, 'pool', 512, 512, 512, 512, 'pool']}

        super(ConvNet, self).__init__()
        self.channels = channels
        self.config = config[architecture]
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096) # pooled -> linear
        self.fc2 = nn.Linear(4096, 4096) # linear -> linear
        self.fc3 = nn.Linear(4096, classes) # linear -> output 
        self.build()

    def forward(self, x):

        for layer in self.layers:

            if type(layer) == nn.Conv2d:
                x = layer(x) # conv layer
            elif type(layer) == nn.BatchNorm2d:
                x = f.relu(layer(x)) # batch norm layer (relu)
            else:
                x = layer(x) # pool layer

        # average pool and flatten
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

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

    # init net & data
    net = ConvNet('11-layer', channels=3, classes=10)
    images = torch.Tensor(torch.rand(32, 3, 224, 224))
    print(images.size())

    # use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    images = images.to(device)

    # forward pass
    out = net.forward(images)
    print(out.shape)
