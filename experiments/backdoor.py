import torch
import numpy as np
from copy import deepcopy
from modules import NetModule, Module, Optimizer, _Loss, train, test
from datasets import load_dataset, EntitySet
from datagen import PoisonPipeline, Merge, Transform
from typing import Iterable
from torchvision.transforms import Compose


class Backdoor:

    def __init__(self, net: Module, **net_kwargs) -> None:
        # set att
        self.net = net(**net_kwargs)

    def create_models(self, optimizer: Optimizer, loss: _Loss, **optim_kwargs) -> None:
        # untrained networks
        self.base = NetModule(deepcopy(self.net), optimizer, loss, **optim_kwargs)
        self.trojan = NetModule(deepcopy(self.net), optimizer, loss, **optim_kwargs)

    def load_data(self, dataset: str, etype: str, transforms: Compose=None, path: str='./data', download: bool=True) -> None:
        # load test & train data
        traindata, classes = load_dataset(dataset, transforms, path, True, download, etype)
        testdata, classes = load_dataset(dataset, transforms, path, False, download, etype)
        self.traindata, self.testdata, self.classes = traindata, testdata, classes

    def poison(self, patches: np.ndarray, transforms: Iterable[Transform], merge: Merge, pct: float, 
                random_state: np.random.RandomState=None) -> None:
        # undefined atts
        if self.traindata is None or self.testdata is None:
            raise ValueError('no dataset loaded')
        
        # poison train data (by pct) & test data (all)
        pipeline = PoisonPipeline()
        # numpy arrays for training data 
        cleantrain, poisontrain = pipeline.process(self.traindata, patches, transforms, merge, pct, random_state)
        cleantest, poisontest = pipeline.process(self.testdata, patches, transforms, merge, 1, random_state)
        self.cleantrain, self.poisontrain = EntitySet(cleantrain, self.classes), EntitySet(poisontrain, self.classes)
        self.cleantest, self.poisontest = EntitySet(cleantest, self.classes), EntitySet(poisontest, self.classes)

    def train(self, net: int, epochs: int=3, verbose: bool=True, device: torch.device=None, **dataloader_kwargs) -> tuple:
        # undefined atts
        if self.base is None or self.trojan is None:
            raise ValueError('no models created')
        if self.cleantrain is None or self.poisontrain is None:
            raise ValueError('no training data was created')
        
        # metrics [base loss, trojan loss]
        losses = [None, None]

        # train base
        if net == 0:
            trainloader = self.cleantrain.get_dataloader(**dataloader_kwargs)
            losses[0] = train(self.base, trainloader, epochs, verbose, device)
        # train trojan
        elif net == 1:
            trainloader = self.poisontrain.get_dataloader(**dataloader_kwargs)
            losses[1] = train(self.trojan, trainloader, epochs, verbose, device)
        # train both
        elif net == 2:
            # clean trainloader
            trainloader = self.cleantrain.get_dataloader(**dataloader_kwargs)
            losses[0] = train(self.base, trainloader, epochs, verbose, device)
            # poison trainloader
            trainloader = self.poisontrain.get_dataloader(**dataloader_kwargs)
            losses[1] = train(self.trojan, trainloader, epochs, verbose, device)
        else:
            raise ValueError('invalid int keys [0: base, 1: trojan, 2: both]')

        return tuple(losses)

    def test(self, net: int, verbose: bool=True, device: torch.device=None, **dataloader_kwargs) -> tuple:
        # undefined atts
        if self.base is None or self.trojan is None:
            raise ValueError('no models created')

        # metrics [base loss, base acc, trojan loss, trojan acc]
        losses_accs = [None, None, None, None]

        testloader = self.cleantest.get_dataloader(**dataloader_kwargs) # clean testloader (only)
        # train base
        if net == 0:
            losses_accs[0], losses_accs[1] = test(self.base, testloader, verbose, device)
        # train trojan
        elif net == 1:
            losses_accs[2], losses_accs[3] = test(self.trojan, testloader, verbose, device)
        # train both
        elif net == 2:
            losses_accs[0], losses_accs[1] = test(self.base, testloader, verbose, device)
            losses_accs[2], losses_accs[3] = test(self.trojan, testloader, verbose, device)
        else:
            raise ValueError('invalid int keys [0: base, 1: trojan, 2: both]')

        return tuple(losses_accs)


    def eval(self, verbose: bool=True, device: torch.device=None) -> None:
        if self.base is None or self.trojan is None:
            raise ValueError('no models created')
        if self.cleantest is None or self.poisontest is None:
            raise ValueError('no training data was created')

        # metrics [base acc clean, trojan acc clean, base acc poison, trojan acc poison]
        accs = [None, None, None, None]

        cleanloader, poisonloader = self.cleantest.get_dataloader(), self.poisontest.get_dataloader()

        # clean testloader
        accs[0], accs[1] = test(self.base, cleanloader, False, device)[1], test(self.trojan, cleanloader, False, device)[1]

        # poison testloader
        accs[2], accs[3] = test(self.base, poisonloader, False, device)[1], test(self.trojan, poisonloader, False, device)[1]

        if verbose:
            diff = abs(accs[0] - accs[1])
            print(f'Accuracy on Clean | Base {accs[0] * 100:.2f}% | Trojan {accs[1] * 100:.2f}% | Difference {diff * 100:.2f}%')
            print(f'Base Accuracy on Poison {accs[2] * 100:.2f}% | Attack Success Rate (ASR): {(accs[3] - accs[2]) * 100:.2f}%')

        return tuple(accs)

if __name__ == '__main__':
    backdoor = Backdoor('convnet', '11-layer', channels=3, classes=10)
