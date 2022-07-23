import torch
import numpy as np
from copy import deepcopy
from modules import NetModule, Module, Optimizer, _Loss, train, test
from datasets import load_dataset, EntitySet
from datagen import DataEntity, PoisonPipeline, Merge, Transform
from typing import Iterable
from torchvision.transforms import Compose
from torch.utils.data import Dataset


class Backdoor:

    def __init__(self, net: Module, **net_kwargs) -> None:
        # set att
        self.net = net(**net_kwargs)

    def create_models(self, optimizer: Optimizer, loss: _Loss, **optim_kwargs) -> None:
        # untrained networks
        self.base = NetModule(deepcopy(self.net), optimizer, loss, **optim_kwargs)
        self.trojan = NetModule(deepcopy(self.net), optimizer, loss, **optim_kwargs)

    def load_data(self, dataset: Dataset, etype: DataEntity, transforms: Compose=None, 
                    path: str='./data', download: bool=True, pct: float=1) -> None:
        # load test & train data
        traindata, classes = load_dataset(dataset, transforms, path, True, download, etype, pct)
        testdata, classes = load_dataset(dataset, transforms, path, False, download, etype, pct)
        # set train & test datasets
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
        # set train & test entitysets
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
        metrics = [None, None, None, None]

        testloader = self.cleantest.get_dataloader(**dataloader_kwargs) # clean testloader (only)
        # train base
        if net == 0:
            metrics[0], metrics[1] = test(self.base, testloader, verbose, device)
        # train trojan
        elif net == 1:
            metrics[2], metrics[3] = test(self.trojan, testloader, verbose, device)
        # train both
        elif net == 2:
            metrics[0], metrics[1] = test(self.base, testloader, verbose, device)
            metrics[2], metrics[3] = test(self.trojan, testloader, verbose, device)
        else:
            raise ValueError('invalid int keys [0: base, 1: trojan, 2: both]')

        return tuple(metrics)


    def eval(self, verbose: bool=True, device: torch.device=None) -> None:
        if self.base is None or self.trojan is None:
            raise ValueError('no models created')
        if self.cleantest is None or self.poisontest is None:
            raise ValueError('no training data was created')

        # metrics [base acc clean, trojan acc clean, base acc poison, trojan acc poison, avg tensor dist, net tensor dist]
        metrics = [None, None, None, None, None, None]

        cleanloader, poisonloader = self.cleantest.get_dataloader(), self.poisontest.get_dataloader()

        # clean testloader
        metrics[0], metrics[1] = test(self.base, cleanloader, False, device)[1], test(self.trojan, cleanloader, False, device)[1]

        # poison testloader
        metrics[2], metrics[3] = test(self.base, poisonloader, False, device)[1], test(self.trojan, poisonloader, False, device)[1]

        # euclidean distance for tensors
        metrics[4], metrics[5] = self.tensor_euclidean()

        if verbose:
            print(f'Accuracy on Clean | Base {metrics[0] * 100:.2f}% | Trojan {metrics[1] * 100:.2f}% | Difference {(metrics[1] - metrics[0]) * 100:.2f}%')
            print(f'Base Accuracy on Poison {metrics[2] * 100:.2f}% | Attack Success Rate (ASR): {metrics[3] * 100:.2f}%')
            print(f'Average Tensor Distance: {metrics[4]:.2f} | Net Tensor Difference {metrics[5]:.2f}')

        return tuple(metrics)

    def tensor_euclidean(self) -> tuple:
        # set metrics
        accum_dist = 0

        # clean and poison tensors
        cleantensors, poisontensors = self.cleantrain.get_tensorset().tensors[0], self.poisontrain.get_tensorset().tensors[0]

        # find euclidean distance for all tensors
        for cleantensor, poisontensor in zip(cleantensors, poisontensors):
            accum_dist += (cleantensor.squeeze(dim=-1) - poisontensor.squeeze(dim=-1)).pow(2).sum().sqrt().item()
        
        # return average and net difference between tensors
        return accum_dist / len(cleantensors), accum_dist

    def __len__(self) -> int:
        return len(self.traindata)

    def get_net_modules(self) -> tuple:
        return self.base, self.trojan

    def get_datasets(self) -> tuple:
        return self.cleantrain, self.poisontrain, self.cleantest, self.poisontest

    def get_classes(self) -> dict:
        return self.classes

if __name__ == '__main__':
    backdoor = Backdoor('convnet', '11-layer', channels=3, classes=10)
