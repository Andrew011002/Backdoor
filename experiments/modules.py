import torch
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss
from copy import deepcopy
from torch.optim import Optimizer, SGD
from datasets import DataLoader
from nets import *


class NetModule(Module):

    """
    A class derived from Module to create a structure
    for nn.Module.
    """

    def __init__(self, net: Module, optimizer: Optimizer, loss: _Loss, **optim_kwargs) -> None:

        """
        Initializes the NetModule class.

        net: the Module (net) for the NetModule instance.

        optimizer: the Optimizer used for training the net
        of the NetModule instance.

        loss: the _Loss for calculating the loss function for
        training & testing the net of the NetModule instance.

        optim_kwargs: the keyword arguments for the optimizer.
        (ex. optimizer=SGD optim_kwargs={lr: 0.01, momentum: 0.9} sets
        SGD(lr=0.01, momentum=0.9)).
        """

        super(NetModule, self).__init__()
        self.net = net
        self.loss = loss()
        self.optimizer = optimizer(net.parameters(), **optim_kwargs)
        self.optim_kwargs = optim_kwargs
        self.prarams = [deepcopy(net), optimizer, loss]

    def train(self, device: torch.device=None) -> None:

        """
        Prepares the net of the NetModule instance
        for training.

        device: a torch.device to train the network
        on. (default: None).
        """

        # move net to device
        self.net.to(device)
        self.net.train()
    
    def eval(self, device: torch.device=None) -> None:

        """
        Prepares the net of the NetModule instance
        for testing.

        device: a torch.device to train the network
        on. (default: None).
        """

        # move net to device
        self.net.to(device)
        self.net.eval()

    def forward(self, x):
        return self.net(x)

    def save(self, path: str):

        """
        Saves the state_dict of the net
        from the NetModule instance to a directory.

        path: a string defining the location the file
        will be saved to.
        """

        torch.save(self.net.state_dict(), path)
        print(f'Net saved to {path}')

    def get_modules(self) -> dict:

        """
        Returns the mappings of named modules
        to their module (layer) in the net
        of the NetModule instance.
        """

        # get named layers
        modules = list(self.net.named_modules())[1:]
        mappings = dict()
        # map to layer names to their modules
        for name, layer in modules:
            mappings[name] = layer
        return mappings

    def view_named_modules(self) -> None:

        """
        Retrives and prints the named modules correlated
        to their modules (layers) of the net from the
        NetModule instance.
        """

        # get named layers
        modules = list(self.net.named_modules())[1:]
        # map to layer names to their modules
        for name, layer in modules:
            print(f'{name}: {layer}')

    def view_module_named_parameters(self, module: str) -> list:

        """
        Retrives the list of named parameters from the specified 
        module (layer) of the net from the NetModule instance.

        module: a string representing the desired module to retrive
        the named parameters from.
        """

        #  veiwa named parameters of specific module (named layer)
        modules = self.get_modules()
        return list(modules[module].named_parameters())

    def view_module_named_buffers(self, module: str) -> list:

        """
        Retrives the list of named buffers from the specified 
        module (layer) of the net from the NetModule instance.

        module: a string representing the desired module to retrive
        the named buffers from.
        """

        # view named buffers of specific module (named layer)
        modules = self.get_modules()
        return list(modules[module].named_buffers())

    def set_net(self, net: Module) -> None:

        """
        Sets the net of the NetModule instance.

        net: the Module to set the net as.
        """

        # replace net
        self.net = net

    def set_optimizer(self, optimizer: Optimizer) -> None:

        """
        Sets the optimizer of the NetModule instance.

        optimizer: the Optimizer to set the optimizer as.
        """

        # replace optimizer
        self.optimizer = optimizer
    
    def set_loss(self, loss: _Loss) -> None:

        """
        Sets the loss of the NetModule instance.

        loss: the _Loss to set the loss as.
        """

        # replace loss function
        self.loss = loss

    def __getitem__(self, module: str) -> Module:

        """
        Retrieves the module (layer) within the net
        of the NetModule instance.

        module: a string depicting the desired module
        (layer) to retrieve from the net in the NetModule
        instance.
        """

        # get module (named layers)
        modules = self.get_modules()
        return modules[module]

    def __deepcopy__(self, memo) -> 'NetModule':

        """
        Createa a deepcopy of the NetModule instance without
        any references. (see copy deepcopy for more info).

        memo: ignore.
        """

        # create new copy without reference
        module = NetModule(*self.prarams, **self.optim_kwargs)
        # replace attributes
        module.set_net(deepcopy(self.net))
        module.set_optimizer(deepcopy(self.optimizer))
        module.set_loss(deepcopy(self.loss))
        return module



# trains network on trainloader
def train(net_module: NetModule, trainloader: DataLoader, epochs: int, verbose: bool=True, device: torch.device=None) -> float:

    """
    Trains a NetModule on a DataLoader.

    net_module: the NetModule to train.

    trainloader: the DataLoader of train on.

    epochs: an integer defining the the amount of
    times the NetModule trains of the entire DataLoader.

    verbose: a boolean depicting whether to show metrics as 
    the NetModule trains or not. (default: True which shows 
    the metrics as the NetModule trains).

    device: a torch.device to train the network
    on. (default: None).
    """

    net_module.train(device)
    n = len(trainloader.dataset)
    m = len(trainloader)
    net_loss = 0

    if verbose:
        print('Training started')

    for epoch in range(epochs):
        accum_loss = 0
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
            # accumulate avg loss for this batch
            accum_loss += loss.item()
            samples_trained += inputs.size(0)


            if verbose:
                # show every 25% of the trainset
                if (i + 1) % int(m * 0.25) == 0:
                    print(f'Epoch {epoch + 1}/{epochs} | {(i + 1) * 100 / m:.2f}% | Loss: {accum_loss / (i + 1):.4f} | Samples trained: {samples_trained}/{n}')
        # running sum for net loss
        net_loss += accum_loss
        if verbose:
            print(f'Epoch {epoch + 1} complete | Loss: {accum_loss / (i + 1):.4f}')
    # calculate net loss
    net_loss /= epochs * m
    if verbose:
        print(f'Training complete | Net Average Loss: {net_loss:.4f} | Total epochs: {epochs}')
    return net_loss
    

# test network on testloader (categorical)
def test(net_module: NetModule, testloader: DataLoader, verbose: bool=True, device: torch.device=None) -> tuple:

    """
    Tests a NetModule on a DataLoader.

    net_module: the NetModule to test.

    trainloader: the DataLoader of test on.

    verbose: a boolean depicting whether to show metrics as 
    the NetModule is tested or not. (default: True which shows 
    the metrics as the NetModule is tested).

    device: a torch.device to test the network
    on. (default: None).
    """

    # prepare net for testing
    net_module.eval(device)
    accum_loss = 0
    correct = 0
    samples_seen = 0
    n = len(testloader.dataset)
    m = len(testloader)

    if verbose:
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
                if (i + 1) % int(m * 0.25) == 0:
                    print(f'{(i + 1) * 100 / m:.2f}% Testing complete | Loss: {accum_loss / (i + 1):.4f} | Accuracy: {correct / samples_seen:.4f}')

    # calculate accuracy and net loss
    acc = correct / n
    net_loss = accum_loss / m
    if verbose:
        print(f'Testing complete | Loss: {net_loss:.4f} | Accuracy: {acc * 100:.2f}%')

    return net_loss, acc

if __name__ == '__main__':
    net = VggNet('11-layer', channels=3, classes=10)
    net_module = NetModule(net, SGD, CrossEntropyLoss, lr=0.01)
    net_module.train()
    net_module.eval()
    fake_images = torch.Tensor(np.random.randint(0, 256, (16, 32, 32, 3)))
    print(net_module(fake_images).shape)
    # net_module.save('net.pt')