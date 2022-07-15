import enum
from pickletools import optimize
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn import BCELoss, CrossEntropyLoss
from ffnet import FeedForwardNetwork
from convnet import ConvNet

optimizers = {
            'adadelta': optim.Adadelta,
            'adagrad': optim.Adagrad,
            'adam': optim.Adam,
            'adamax': optim.Adamax,
            'sgd': optim.SGD
            }

losses = {
        'crossentropy': CrossEntropyLoss,
        'binarycrossentropy': BCELoss,
        }


class NetModule(nn.Module):

    def __init__(self, net: Module, optimizer: str, loss: str, **kwargs) -> None:
        super(NetModule, self).__init__()
        self.net = net
        self.optimizer = optimizers.get(optimizer.lower().strip(), KeyError('Optimizer not found'))
        self.loss = losses.get(loss.lower().strip(), KeyError('Loss not found'))
        self.params = self.net.parameters()
        self.kwargs = kwargs

    def train(self, device=None) -> None:
        if self.kwargs is None:
            raise ValueError('Optimizer params are unedefined')

        self.optimizer = self.optimizer(self.net.parameters(), **self.kwargs)
        self.net.to(device)
        self.net.train()
        print(f'Net ready for training')
    
    def eval(self, device=None) -> None:
        self.net.to(device)
        self.net.eval()
        print(f'Net ready for evaluation')

    def forward(self, x):
        return self.net(x)
    

# trains network on trainloader
def train(net_module: NetModule, trainloader: DataLoader, epochs: int, verbose: bool=True, device: torch.device='cpu') -> float:
    net_module.train(device)
    n = len(trainloader.dataset)
    m = len(trainloader)
    accum_loss = 0

    print(f'Training started')

    for epoch in range(epochs):
        samples_trained = 0

        for i, data in enumerate(trainloader, 0):
            # get data
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # compute predictions
            pred = net_module(inputs)
            loss = net_module.loss(pred, labels)
            # compute gradients
            net_module.optimizer.zero_grad()
            loss.backward()
            net_module.optimizer.step()
            accum_loss += loss.item()
            samples_trained += inputs.size(0)


            if verbose:
                # show every 25% of the trainset
                if (i + 1) % int(m * 0.25) == 0:
                    print(f'Epoch {epoch + 1}/{epochs} | {(i + 1) * 100 / m:.2f}% | Loss: {accum_loss / (i + 1):.4f} | Samples trained: {samples_trained}/{n}')
        if verbose:
            print(f'Epoch {epoch + 1} complete | Loss: {accum_loss / (i + 1):.4f}')
    # calculate net loss
    net_loss = accum_loss / m
    if verbose:
        print(f'Training complete | Net Loss: {net_loss:.4f} | Total epochs: {epochs}')
    return net_loss
    

# test network on testloader (categorical)
def test(net_module: NetModule, testloader: DataLoader, verbose: bool=True, device: torch.device='cpu') -> tuple:
    # prepare net for testing
    net_module.eval()
    accum_loss = 0
    correct = 0
    samples_seen = 0
    n = len(testloader.dataset)
    m = len(testloader)

    print('Testing started')
    # ignore gradients
    with torch.no_grad():

        for i, data in enumerate(testloader, 0):
            # get data
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # compute predictions
            pred = net_module(inputs)
            loss = net_module.loss(pred, labels)
            # accumulate loss & accuracy (correct)
            accum_loss += loss.item()
            samples_seen += inputs.size(0)
            correct += torch.sum(torch.argmax(pred, dim=1) == labels).item()

            if verbose:
                # show every 25% of the testset
                if (i + 1) % (m * 0.25) == 0:
                    print(f'{(i + 1) * 100 / m}% Testing complete | Loss: {accum_loss / (i + 1):.4f} | Accuracy: {correct / samples_seen:.4f}')

    # calculate accuracy and net loss
    acc = correct / n
    net_loss = accum_loss / m
    if verbose:
        print(f'Testing complete | Loss: {net_loss:.4f} | Accuracy: {acc:.4f}')

    return net_loss, acc

    


if __name__ == '__main__':
    net = FeedForwardNetwork(input_shape=(28, 28, ), config='8-layer', classes=3, dropout=0.2)
    module = NetModule(net, optimizer='SGD', loss='CrossEntropy', lr=0.01)
    module.train()
    module.eval()