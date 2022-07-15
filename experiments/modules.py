import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn import BCELoss, CrossEntropyLoss

optimizers = {
            'Adam': optim.Adam,
            'SGD': optim.SGD
            }

losses = {
        'CrossEntropy': CrossEntropyLoss,
        'BinaryCrossEntropy': BCELoss,
        }

def prepare_net(net: Module, optimizer: str, loss: str, lr: float=0.01, momentum: float=0, betas: tuple=(0.9, 0.999), eps: float=1e-8, device: torch.device='cpu'):
    loss = losses.get(loss, KeyError('Invalid loss function'))
    optimizer = optimizers.get(optimizer, KeyError('Invalid optimizer'))

    if optimizer is optim.Adam:
        optimizer = optimizer(net.parameters(), lr=lr, betas=betas, eps=eps)
    elif optimizer is optim.SGD:
        optimizer = optimizer(net.parameters(), lr=lr, momentum=momentum)
    
    net.to(device)
    print(f'Net is training on {device}')
    

def train(net: Module, trainloader: DataLoader, epochs: int, verbose: bool=True, device: torch.device='cpu') -> tuple:
    optimizer = optimizer(net.parameters())


        

