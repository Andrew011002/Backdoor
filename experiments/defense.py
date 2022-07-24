import numpy as np
import torch 
from torch.utils.data import DataLoader
from torch.nn.utils.prune import LnStructured
from copy import deepcopy
from backdoor import Backdoor
from datagen import ImagePatch, ImageMerge
from datasets import EntitySet
from modules import NetModule, test, train
from typing import Iterable

class Defense:

    def __init__(self, backdoor: Backdoor) -> None:
        self.backdoor = backdoor
        self.defense = deepcopy(backdoor.get_net_modules()[1])

    def retrain(self, trainloader: DataLoader=None, epochs: int=3, verbose: bool=True, 
                device: torch.device=None, **dataloader_kwargs) -> None:
        # get clean trainloader
        if trainloader is None:
            # use backdoor's trainloader
            cleantrain, poisontrain, cleantest, poisontest = self.backdoor.get_datasets()
            trainloader = cleantrain.get_dataloader(**dataloader_kwargs)

        # retrain
        loss = train(self.defense, trainloader, epochs, verbose, device)
        return loss

    def test(self, testloader: DataLoader=None, verbose: bool=True, device: torch.device=None, **dataloader_kwargs) -> tuple:
        # get clean testloader
        if testloader is None:
            cleantrain, poisontrain, cleantest, poisontest = self.backdoor.get_datasets()
            testloader = cleantest.get_dataloader(**dataloader_kwargs)

        # test net
        loss, acc = test(self.defense, testloader, verbose, device)
        return loss, acc

    def eval(self, cleanloader: DataLoader=None, poisonloader: DataLoader=None, verbose: bool=True, 
                device: torch.device=None, **dataloader_kwargs) -> tuple:
        # get clean & poison testloaders
        cleantrain, poisontrain, cleantest, poisontest = self.backdoor.get_datasets()
        if cleanloader is None:
            cleanloader = cleantest.get_dataloader(**dataloader_kwargs)
        if poisonloader is None:
            poisonloader = poisontest.get_dataloader(**dataloader_kwargs)

        # (base -> trojan -> defense [each with clean and poison loss/acc])
        metrics = [None, None, None, None, None, None]
        base, trojan = self.backdoor.get_net_modules()

        # base peformance
        metrics[0] = test(base, cleanloader, False, device)[1]
        metrics[1] = test(base, poisonloader, False, device)[1]

        # trojan peformance
        metrics[2] = test(trojan, cleanloader, False, device)[1]
        metrics[3] = test(trojan, poisonloader, False, device)[1]

        # defense peformance
        metrics[4] = test(self.defense, cleanloader, False, device)[1]
        metrics[5] = test(self.defense, poisonloader, False, device)[1]
        diff = metrics[3] - metrics[5]

        # show info
        if verbose:
            print(f'Accuracy on clean | Base {metrics[0] * 100:.2f}% | Trojan {metrics[2] * 100:.2f}% | Defense {metrics[4] * 100:.2f}%')
            print(f'Accuracy on Posion | Base {metrics[1] * 100:.2f}% | Defense {metrics[5] * 100:.2f}% | Trojan ASR {metrics[3] * 100:.2f}%')
            print(f'Difference from Baseline | Trojan {(metrics[2] - metrics[0]) * 100:.2f}% | Defense {(metrics[4] - metrics[0]) * 100:.2f}%')
            print(f'Defense Effectiveness | {abs(diff) * 100:.2f}% {"increase" if diff < 0 else "decrease" if diff > 0 else "no change"} in ASR')

        return tuple(metrics)

    def detect(self, net_module: NetModule=None, dataset: EntitySet=None, threshold: float=0.3, size_ranges: Iterable[tuple]=None, 
            pct: float=0.2, verbose: bool=True, device: torch.device=None, **dataloader_kwargs) -> tuple:
        # get net module
        if net_module is None:
            net_module = self.defense
        # get dataset
        if dataset is None:
            cleantrain, poisontrain, cleantest, poisontest = self.backdoor.get_datasets()
            dataset = cleantest
        # get patch size ranges
        if size_ranges is None:
            size_ranges = [(i, i) for i in range(5, 11)]
        
        # prepare for testing
        n = len(dataset)
        indices = np.random.choice(n, int(n * pct), replace=False)
        channels = dataset[0].channels
        loss, og_acc = test(net_module, dataset.get_dataloader(**dataloader_kwargs), verbose=False, device=device)
        metrics = []
        diffs = []

        # test patches of each size
        for size in size_ranges:
            syntheticset = deepcopy(dataset)
            patch = ImagePatch(size, channels, palette='random')
            # poison select images
            for i in indices:
                entity = ImageMerge().do(syntheticset[i], patch)
                syntheticset[i] = entity
            # create testloader & run test
            syntheticset.update()
            testloader = syntheticset.get_dataloader(**dataloader_kwargs)
            loss, acc = test(net_module, testloader, verbose=False, device=device)
            metrics.append(acc)
            diffs.append(acc - og_acc)
        
        # find avg accuracy and difference
        avg_acc = np.mean(metrics)
        low_acc = min(metrics)
        avg_diff = np.mean(diffs)
        low_diff = abs(np.min(diffs))

        # show info
        if verbose:
            for size, acc in zip(size_ranges, metrics):
                print(f'Patch size {size} | Synthetic Poison Accuracy {acc * 100:.2f}% | Original Accuracy {og_acc * 100:.2f}%\
 | Difference {(acc - og_acc) * 100:.2f}%')
            print(f'Average Accuracy {avg_acc * 100:.2f}% | Average Difference {avg_diff * 100:.2f}% | Lowest Score: {low_acc * 100:.2f}%\
 | Likihood of Backdoor: {"High" if low_diff >= threshold else "Medium" if threshold > low_diff >= threshold * 0.95 else "Low"}')

        return (metrics, diffs)

    def prune(self, layers: Iterable[str], amount: float=0.3, norm: float=float('inf'), dim: int=-1) -> None:
        # grab modules (named layers)
        modules = self.defense.get_modules()
        layers = [modules[layer] for layer in layers]
        # apply pruning to all layer parameters
        for layer in layers:
            LnStructured(amount, norm, dim).apply(layer, name='weight', amount=amount, n=norm, dim=dim)

    def block(self, dataset: EntitySet=None, patch: ImagePatch=None, threshold: float=5, **dataloader_kwargs) -> DataLoader:
        # get entityset
        if dataset is None:
            cleantrain, poisontrain, cleantest, poisontest = self.backdoor.get_datasets()
            dataset = poisontest
        # get patch
        if patch is None:
            patch = ImagePatch((9, 9), dataset[0].channels, palette='random')

        # patch entities
        entities = deepcopy(dataset.get_entities())
        for index, entity in enumerate(entities):
            for i in range(entity.shape[0] - patch.shape[0]):
                for j in range(entity.shape[1] - patch.shape[1]):
                    # local region
                    local = entity.get_data()[i:i + patch.shape[0], j:j + patch.shape[1], :]
                    diff = abs(np.linalg.norm(local) - np.linalg.norm(patch.get_data()))
                    if diff <= threshold:
                        # paste patch into local region
                        data = deepcopy(entity.get_data())
                        data[i: i + patch.shape[0], j: j + patch.shape[1], :] = 0
                        entity.set_data(data)
                        break
                break
            # assign entity to dataset
            entities[index] = entity
        # update dataset with new blocked patches
        entityset = EntitySet(dataset, dataset.classes)
        return entityset.get_dataloader(**dataloader_kwargs)

    def reset(self) -> None:
        self.defense = deepcopy(self.backdoor.get_net_modules()[1])

    def view_named_modules(self) -> None:
        self.defense.view_named_modules()

    def view_named_parameters(self, module: str) -> None:
        print(f'{self.defense.view_module_named_parameters(module)}')

    def view_named_buffers(self, module: str) -> None:
        print(f'{self.defense.view_module_named_buffers(module)}')


if __name__ == '__main__':
    pruner = LnStructured(0.3, float('inf'))
    module = torch.nn.Linear(10, 10)
    pruner.apply(module, 'weight', 0.3, n=float('inf'), dim=-1)
    print(module._forward_pre_hooks)
    print(list(module.named_buffers()))
    print(list(module.named_parameters()))