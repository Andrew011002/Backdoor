import numpy as np
import torch
import torch.nn as nn

configs = {
            '4-layer': [512, 256, 128, 64],
            '5-layer': [1024, 512, 256, 128, 64],
            '6-layer': [1024, 512, 256, 256, 128, 64],
            '7-layer': [1024, 512, 512, 256, 256, 128, 64],
            '8-layer': [1024, 512, 512, 256, 256, 128, 128, 64]
            }

class FeedForwardNetwork(nn.Module):

    def __init__(self, input_shape: tuple, config: str, classes: int, dropout: float=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.config = config
        self.input_dim = np.prod(list(input_shape)) # ex (4, 4, 3) img -> 4 * 4 * 3 = 48
        self.feed_forward = self.build()
        self.relu = nn.ReLU()
        self.out = nn.Linear(64, classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        x = torch.flatten(x, 1)
        for layer in self.feed_forward:
            x = self.relu(layer(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

    def build(self) -> nn.Sequential:
        layers = []
        config = configs.get(self.config, KeyError)
        for hidden_layer in config:
            layers.append(nn.Linear(self.input_dim, hidden_layer))
            self.input_dim = hidden_layer
        
        feed_forward = nn.Sequential(*layers)
        return feed_forward




        