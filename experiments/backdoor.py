import torch
import numpy as np
from copy import deepcopy
from modules import NetModule, Module, Optimizer, _Loss, train, test
from nets import VggNet
from datasets import load_dataset, EntitySet
from datagen import DataEntity, PoisonPipeline, Merge, Transform
from typing import Iterable, Tuple
from torchvision.transforms import Compose
from torch.utils.data import Dataset

class Backdoor:

    """
    A basic class for creating a Backdoor in
    Deep Neural Networks.
    """

    def __init__(self, net: Module, **net_kwargs) -> None:

        """
        net: the Module to create the backdoor on.

        net_kwargs: the keyword arguments for the net
        Module. 
        (ex. net=FcNet, net_kwargs={config: '8-layer', input_dim: (32, 32, 3), 
        classes: 10, dropout: 0.3} creates initializes
        FcNet(config='8-layer', input_dim=(32, 32, 3), classes=10, dropout=0.3)).
        """

        # set att
        self.net = net(**net_kwargs)

    def create_models(self, optimizer: Optimizer, loss: _Loss, **optim_kwargs) -> None:

        """
        Creates the baseline NetModule and Trojan NetModule for the
        Backdoor attack.

        optimizer: the Optimizer for the NetModule.

        loss: the _Loss function for the NetModule.

        optim_kwargs: the keyword arguments for the optimizer.
        (ex. optimizer=SGD optim_kwargs={lr: 0.01, momentum: 0.9} sets
        SGD(lr=0.01, momentum=0.9)).
        """

        # untrained networks
        self.base = NetModule(deepcopy(self.net), optimizer, loss, **optim_kwargs)
        self.trojan = NetModule(deepcopy(self.net), optimizer, loss, **optim_kwargs)

    def load_data(self, dataset: Dataset, etype: DataEntity, transforms: Compose=None, 
                    path: str='./data', download: bool=True, pct: float=1) -> None:

        """
        Creates the training, testing, & sets the classes for the
        Backdoor instance.

        dataset: the desired dataset from torchvision.datasets to
        load. (see datasets load_dataset for more details).

        etype: the DataEntity that data from the dataset will
        be casted into. (see datasets load_dataset for more details).

        transforms: a Compose of desired transformations to apply to
        the dataset. (see datasets load_dataset for more details).

        path: a string depicting the path to download the dataset.
        (default: './data' see datasets load_dataset for more details).

        download: a boolean indicating whether to download the dataset
        or not. (default: True see datasets load_dataset for more details).

        pct: a floating value indicating the percent of the dataset to load.
        (default: 1 see datasets load_dataset for more details).
        """
        
        # load test & train data
        traindata, classes = load_dataset(dataset, transforms, path, True, download, etype, pct)
        testdata, classes = load_dataset(dataset, transforms, path, False, download, etype, pct)
        # set train & test datasets
        self.traindata, self.testdata, self.classes = traindata, testdata, classes

    def poison(self, patches: Iterable[DataEntity], transforms: Tuple[Iterable[Transform]], merge: Merge, pct: float, 
                random_state: np.random.RandomState=None) -> None:

        """
        Creates clean & poison data for both training & testing.

        patches: an Iterable (preferably ndarray) of patches
        to transform/merge with entities of Backdoor data.
        (see datagen PoisonPipeline for more details).

        transforms: a Tuple of Iterables that describe the order in which
        transformations are applied to entities, patches, and select entities
        respectively. (ex. (None, [RotateTransform()], [LabelTransform()])
        peforms no transformations on entire entities, peforms RotateTransform
        on patches, and then peforms LabelTransform on select entities).
        (see datagen PoisonPipeline for more details).

        merge: the Merge that will combine entities with patches.
        (see datagen PoisonPipeline for more details).

        pct: a floating value representing the amount of entities to select
        for select Transform or select Merge (ex. pct=0.2 meaning 20%
        of the entities will be sampled to apply the desired select Transform
        or select Merge to). (see datagen PoisonPipeline for more details).
        """
        
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

        """
        Trains the specified NetModules on their correlated 
        training set (i.e. the baseline NetModule trains on
        the clean DataLoader while the trojan NetModule trains
        on the poison DataLoader).

        net: an integer corresponding to which NetModule 
        to train. (0: baseline NetModule, 1: trojan NetModule, and
        2: trains both baseline & trojan NetModule).

        epochs: an integer defining the the amount of
        times the NetModule trains of the entire DataLoader.
        (default: 3).

        verbose: a boolean depicting whether to show metrics as 
        the NetModule trains or not. (default: True which shows 
        the metrics as the NetModule trains).

        device: a torch.device to train the network
        on. (default: None).

        dataloader_kwargs: the keyword arguments when retreiving the
        DataLoader for the specified NetModule(s). 
        (ex. net=0, dataloader_kwargs={batch_size: 32, shuffle: True}
        creates self.cleantrain.get_dataloader(batch_size=32, shuffle=True)).
        (see datasets EntitySet get_dataloader for more info).
        """

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

        """
        Tests the specified NetModules on their correlated 
        training set (i.e. the baseline NetModule tests on
        the clean DataLoader while the trojan NetModule tests
        on the poison DataLoader).

        net: an integer corresponding to which NetModule 
        to test. (0: baseline NetModule, 1: trojan NetModule, and
        2: tests both baseline & trojan NetModule).

        verbose: a boolean depicting whether to show metrics as 
        the NetModule is tested or not. (default: True which shows 
        the metrics as the NetModule is tested).

        device: a torch.device to test the network
        on. (default: None).

        dataloader_kwargs: the keyword arguments when retreiving the
        DataLoader for the specified NetModule(s). 
        (ex. net=0, dataloader_kwargs={batch_size: 32, shuffle: True}
        creates self.cleantest.get_dataloader(batch_size=32, shuffle=True)).
        (see datasets EntitySet get_dataloader for more info).
        """

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

        """
        Evaluates the peformance of the Backdoor by using
        Validation Accuracy on clean samples for both the
        baseline NetModule & trojan NetModule of the Backdoor
        instance, Validation Accuracy on poison samples 
        Attack Success Rate (ASR) for both the
        baseline NetModule & trojan NetModule of the Backdoor
        instance, and calculates the net tensor difference as
        well as average tensor distance.

        verbose: a boolean depicting whether to show metrics as 
        the NetModule is tested or not. (default: True which shows 
        the metrics as the NetModule is tested).

        device: a torch.device to test the network
        on. (default: None).        
        """

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

        """
        Calculates the net tensor difference between original tensors
        & their poisoned counterparts, then finds the average tensor 
        difference. (i.e. net tensor distance / # original tensors).
        """

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

        """
        Returns the size of the training data (EntitySet)
        of the Backdoor instance.
        """

        return len(self.traindata)

    def get_net_modules(self) -> tuple:

        """
        Returns the baseline NetModule a& Trojan net Module.
        """

        return self.base, self.trojan

    def get_datasets(self) -> tuple:

        """
        Returns the clean & poison EntitySets for both training
        & testing data.
        """

        return self.cleantrain, self.poisontrain, self.cleantest, self.poisontest

    def get_classes(self) -> dict:

        """
        Returns the orignal classes of the Backdoor instance.
        (derived from the clean EntitySet.classes).
        """

        return self.classes

if __name__ == '__main__':
    backdoor = Backdoor(VggNet, config='11-layer', channels=3, classes=10, dropout=0.3)
