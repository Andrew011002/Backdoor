from matplotlib import image
import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch.nn.functional as f
import torch.nn.utils.prune as prune
from backdoor import Backdoor
from torch.utils.data import DataLoader
from copy import deepcopy
from datagen import ImagePatch
from datasets import EntitySet
from modules import NetModule, test, train
from typing import Iterable



prune_methods = {'ru': prune.RandomStructured,
                'rs': prune.RandomStructured,
                'l1': prune.L1Unstructured,
                'ln': prune.LnStructured}


class Prune(prune.BasePruningMethod):

    def __init__(self, method: str, amount: float=0.0, **pruner_kwargs):
        super(Prune, self).__init__()
        self.method = prune_methods[method](amount=amount, **pruner_kwargs)
        self.amount = amount

    def __call__(self, module, name, **prune_kwargs):
        return self.method.apply(module, name, self.amount, **prune_kwargs)

    def compute_mask(self, t, default_mask):
        return super().compute_mask(t, default_mask)

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

        # show info
        if verbose:
            print(f'Accuracy on clean | Base {metrics[0] * 100:.2f}% | Trojan {metrics[2] * 100:.2f}% | Defense {metrics[4] * 100:.2f}%')
            print(f'Accuracy on Posion | Base {metrics[1] * 100:.2f}% | Defense {metrics[5] * 100:.2f}% | Trojan ASR {metrics[3] * 100:.2f}%')
            print(f'Difference from Baseline | Trojan {(metrics[2] - metrics[0]) * 100:.2f}% | Defense {(metrics[4] - metrics[0]) * 100:.2f}%')
            diff = metrics[3] - metrics[5]
            print(f'Defense Effectiveness | {abs(diff) * 100:.2f}% {"increase" if diff < 0 else "decrease" if diff > 0 else "no change"} in ASR')

        return tuple(metrics)

    def detect(self, threshold: float, net_module: NetModule=None, testloader: DataLoader=None, verbose: bool=True, device: torch.device=None, **dataloader_kwargs) -> bool:
        # get poison testloader
        if testloader is None:
            cleantrain, poisontrain, cleantest, poisontest = self.backdoor.get_datasets()
            testloader = poisontest.get_dataloader(**dataloader_kwargs)
        # get net module
        if net_module is None:
            net_module = self.defense

        # test net
        loss, acc = test(net_module, testloader, verbose=False, device=device)

        # show info
        if verbose:
            print(f'Accuracy on Poison {acc * 100:.2f}% | Likelihood: {"Low" if acc <= threshold * 0.33 else "Medium" if acc <= threshold * 0.66 else "High"}')
        return acc >= threshold

    def prune(self, layers: Iterable[str], layer_params: Iterable[str], pruner: Prune, **prune_kwargs) -> None:
        # grab modules (named layers)
        modules = self.defense.get_modules()
        layers = [modules[layer] for layer in layers]
        # apply pruning to all layer parameters
        for layer in layers:
            for param in layer_params:
                pruner(layer, param, **prune_kwargs)

    def block(self, dataset: EntitySet=None, patch: ImagePatch=None, threshold: float=5, **dataloader_kwargs) -> DataLoader:
        # get dataloader
        if dataset is None:
            cleantrain, poisontrain, cleantest, poisontest = self.backdoor.get_datasets()
            dataset = poisontest.get_entities()
        # get patch
        if patch is None:
            patch = ImagePatch((9, 9), dataset[0].channels, palette='random')

        # patch entities
        for index, entity in enumerate(dataset):

            # locate patch
            for i in range(entity.shape[0] - patch.shape[0]):
                for j in range(entity.shape[1] - patch.shape[1]):
                    # local region
                    local = entity.get_data()[i:i + patch.shape[0], j:j + patch.shape[1], :]
                    dist = np.linalg.norm(local - patch.get_data()) # euclidean distance between patch and local region
                    if dist <= threshold:
                        # paste patch into local region
                        data = entity.get_data()
                        data[i: i + patch.shape[0], j: j + patch.shape[1], :] = 0
                        entity.set_data(data)
                        break
                break
            # assign entity to dataset
            dataset[index] = entity
        
        return dataset.get_dataloader(**dataloader_kwargs)


    
            
    def reset(self) -> None:
        self.defense = deepcopy(self.backdoor.get_net_modules()[1])

    def view_named_modules(self) -> None:
        self.defense.view_named_modules()

    def view_named_parameters(self, module: str) -> None:
        print(f'{self.defense.view_module_named_parameters(module)}')

    def view_named_buffers(self, module: str) -> None:
        print(f'{self.defense.view_module_named_buffers(module)}')


if __name__ == '__main__':
    pruner = Prune('l1', 0.1)
    module = torch.nn.Linear(10, 10)
    pruner(module, 'weight')
    print(module._forward_pre_hooks)
    print(list(module.named_buffers()))
    print(list(module.named_parameters()))