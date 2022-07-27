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

    """
    A basic class for implementing a Defense against
    a Backdoor attack.
    """

    def __init__(self, backdoor: Backdoor) -> None:

        """
        Initializes the Backdoor class.

        backdoor: the Backdoor object to defend against.
        The trojan NetModule of the backdoor will be used
        as the NetModule to apply defensive methods.
        """

        self.backdoor = backdoor
        self.defense = deepcopy(backdoor.get_net_modules()[1])

    def retrain(self, trainloader: DataLoader=None, epochs: int=3, verbose: bool=True, 
                device: torch.device=None, **dataloader_kwargs) -> None:

        """
        Retrains the defense NetModule over a DataLoader.

        trainloader: the DataLoader that will be used to retrain
        the defense NetModule. (default: None if left as None
        the DataLoader will be derived from the clean training 
        EntitySet from the Backdoor instance).

        epochs: an integer defining the the amount of
        times the NetModule trains of the entire DataLoader.
        (default: 3)

        verbose: a boolean depicting whether to show metrics as 
        the NetModule trains or not. (default: True which shows 
        the metrics as the NetModule trains).

        device: a torch.device to train the network
        on. (default: None).

        dataloader_kwargs: the keyword arguments when retreiving the
        DataLoader for the specified NetModule. 
        (ex. dataloader_kwargs={batch_size: 32, shuffle: True}
        creates cleantrain.get_dataloader(batch_size=32, shuffle=True)).
        (see datasets EntitySet get_dataloader for more info).
        """
        
        # get clean trainloader
        if trainloader is None:
            # use backdoor's trainloader
            cleantrain, poisontrain, cleantest, poisontest = self.backdoor.get_datasets()
            trainloader = cleantrain.get_dataloader(**dataloader_kwargs)

        # retrain
        loss = train(self.defense, trainloader, epochs, verbose, device)
        return loss

    def test(self, net_module: NetModule=None, testloader: DataLoader=None, verbose: bool=True, 
            device: torch.device=None, **dataloader_kwargs) -> tuple:

        """
        Tests a NetModule on a DataLoader.

        net_module: the NetModule to test.
        (default: None if left as None the defense
        NetModule from the Defense instance will
        be used for testing).

        testloader: the DataLoader of test on.
        (default: None if left as None the
        DataLoader from the clean testing EntitySet
        derived from the Backdoor instance will be
        used for testing).

        verbose: a boolean depicting whether to show metrics as 
        the NetModule is tested or not. (default: True which shows 
        the metrics as the NetModule is tested).

        device: a torch.device to test the network
        on. (default: None).

        dataloader_kwargs: the keyword arguments when retreiving the
        DataLoader for the specified NetModule. 
        (ex. dataloader_kwargs={batch_size: 32, shuffle: True}
        creates cleantest.get_dataloader(batch_size=32, shuffle=True)).
        (see datasets EntitySet get_dataloader for more info).
        """
        
        # get clean testloader
        if testloader is None:
            cleantrain, poisontrain, cleantest, poisontest = self.backdoor.get_datasets()
            testloader = cleantest.get_dataloader(**dataloader_kwargs)
        # get net module
        if net_module is None:
            net_module = self.defense

        # test net
        loss, acc = test(net_module, testloader, verbose, device)
        return loss, acc

    def eval(self, cleanloader: DataLoader=None, poisonloader: DataLoader=None, verbose: bool=True, 
                device: torch.device=None, **dataloader_kwargs) -> tuple:

        """
        Evaluates the defense NetModule in comparison with
        the baseline NetModule & trojan NetModule (both derived
        from Backdoor instance) to compare Validation Accuracy on
        clean samples, Validation Accuracy on strictly poison
        samples (Attack Success Rate), its distance from the 
        baseline NetModule peformance, & how effective the backdoor 
        was based on the change of Attack Success Rate compared
        to the trojan NetModule.

        cleanloader: the DataLoader to test the NetModules on.
        (default: None if left as None the DataLoader from the
        clean test EntitySet will be used for testing clean
        Validation Accuracy metrics).

        poisonloader: the DataLoader to test the NetModules on.
        (default: None if left as None the DataLoader from the
        poison test EntitySet will be used for testing poison
        Validation Accuracy metrics (Attack Success Rate)).

        verbose: a boolean depicting whether to show metrics as 
        the NetModule trains or not. (default: True which shows 
        the metrics as the NetModule trains).

        device: a torch.device to train the network
        on. (default: None).

        dataloader_kwargs: the keyword arguments when retreiving the
        DataLoader for the specified NetModule(s). 
        (ex. dataloader_kwargs={batch_size: 32, shuffle: True}
        creates EntitySet.get_dataloader(batch_size=32, shuffle=True)).
        (see datasets EntitySet get_dataloader for more info).
        """

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
        metrics[0] = test(base, cleanloader, verbose=False, device=device)[1]
        metrics[1] = test(base, poisonloader, verbose=False, device=device)[1]

        # trojan peformance
        metrics[2] = test(trojan, cleanloader, verbose=False, device=device)[1]
        metrics[3] = test(trojan, poisonloader, verbose=False, device=device)[1]

        # defense peformance
        metrics[4] = test(self.defense, cleanloader, verbose=False, device=device)[1]
        metrics[5] = test(self.defense, poisonloader, verbose=False, device=device)[1]
        diff = metrics[3] - metrics[5]

        # show info
        if verbose:
            print(f'Accuracy on clean | Base {metrics[0] * 100:.2f}% | Trojan {metrics[2] * 100:.2f}% | Defense {metrics[4] * 100:.2f}%')
            print(f'Accuracy on Posion | Base {metrics[1] * 100:.2f}% | Defense {metrics[5] * 100:.2f}% | Trojan ASR {metrics[3] * 100:.2f}%')
            print(f'Difference from Baseline | Trojan {(metrics[2] - metrics[0]) * 100:.2f}% | Defense {(metrics[4] - metrics[0]) * 100:.2f}%')
            print(f'Defense Effectiveness | {abs(diff) * 100:.2f}% {"increase" if diff < 0 else "decrease" if diff > 0 else "no change"} in ASR')

        return tuple(metrics)

    def detect(self, net_module: NetModule=None, dataset: EntitySet=None, threshold: float=0.1, size_ranges: Iterable[tuple]=None, 
            pct: float=0.2, verbose: bool=True, device: torch.device=None, **dataloader_kwargs) -> tuple:

        """
        Prints the detection evaluation on a Backdoor
        attack based on a drop in accuracy compared to a threshold
        as patches are applied.

        net_module: the NetModule to detect the Backdoor attack.
        (default: None if left as None the defense NetModule will 
        be used to detect the Backdoor attack).

        dataset: the EntitySet of clean samples used to apply
        patches to. (default: None if left as None the clean
        testing EntitySet will be used to apply patches when
        testing).

        threshold: a floating value indicating the lowest allowance
        of drop in Validation Accuracy when the NetModule is
        tested over different variations of patch sizes. (default: 0.1).

        size_ranges: An Iterable of tuples containing the (h, w) size ranges
        of patches to apply to data of the EntitySet (ex. [(3, 3), (4, 4)]
        test the NetModule over the dataset when patches of size 3 x 3 are applied
        then test on the same dataset with patches of 4 x 4 applied). 
        (default: None).

        pct: a floating value representing the percent of the entities in the
        dataset to apply the patches to. (default: 0.2 which applies patches
        to 20% of entities in the dataset for each size range in size_ranges).

        verbose: a boolean depicting whether to show metrics as 
        the NetModule trains or not. (default: True which shows 
        the metrics as the NetModule trains).

        device: a torch.device to train the network
        on. (default: None).

        dataloader_kwargs: the keyword arguments when retreiving the
        DataLoader for the specified NetModule. 
        (ex. dataloader_kwargs={batch_size: 32, shuffle: True}
        creates dataset.get_dataloader(batch_size=32, shuffle=True)).
        (see datasets EntitySet get_dataloader for more info).
        """

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

        """
        A method for pruning weights in layers of
        NetModules.

        layers: An Iterable of strings representing the names
        of the modules to prune in the NetModue.

        amount: a floating value dictating the percent of weights
        in the weight tensor of a module (layer) to prune.

        norm: a floating value representing the norm by which weights
        are pruned. (default: float('inf) i.e. +infinity).

        dim: the integer value representing what dimension of the 
        weight tensor in the specified module (layer) to prune.
        (default: -1 i.e. the last dimension).
        """

        # grab modules (named layers)
        modules = self.defense.get_modules()
        layers = [modules[layer] for layer in layers]
        # apply pruning to all layer parameters
        for layer in layers:
            LnStructured(amount, norm, dim).apply(layer, name='weight', amount=amount, n=norm, dim=dim)

    def block(self, dataset: EntitySet=None, patch: ImagePatch=None, labels: dict=None, 
                n: int=None, **dataloader_kwargs) -> DataLoader:

        """
        A method for blocking patches in data to create 
        a DataLoader with blocked data.

        dataset: the EntitySet with patches on the data. 
        (default: None if left as None the poison testing EntitySet
        will be used when blocking patches).

        patch: the ImagePatch in which its variance will be used
        as a threshold for considering blocking parts of the data. 
        (default: None).

        labels: a dictionary mapping target encoded labels to their correct
        encoded labels if the dataset is labeled incorrectly. 
        (ex. 1: 0 with 1 being the target encoded label & 0 being the correct
        encoded label). (default: None).

        n: an integer representing the total number of entities to block.
        (default: None if left as n the size of the dataset will be used as n).

        dataloader_kwargs: the keyword arguments when retreiving the
        DataLoader for the specified NetModule. 
        (ex. dataloader_kwargs={batch_size: 32, shuffle: True}
        creates entityset.get_dataloader(batch_size=32, shuffle=True)).
        (see datasets EntitySet get_dataloader for more info).
        """

        # get entityset
        if dataset is None:
            cleantrain, poisontrain, cleantest, poisontest = self.backdoor.get_datasets()
            dataset = poisontest
        # get patch
        if patch is None:
            patch = ImagePatch((9, 9), dataset[0].channels, palette='random')
        # get size
        if n is None:
            n = len(dataset)
            
        # patch entities
        entities = deepcopy(dataset.get_entities())
        patch_var = np.var(patch.get_data())
        modified = []
        for index in range(n):
            entity = entities[index]
            locals = []
            for i in range(entity.shape[0] - patch.shape[0]):
                for j in range(entity.shape[1] - patch.shape[1]):
                    # local region
                    local = entity.get_data()[i:i + patch.shape[0], j:j + patch.shape[1], :]
                    # variance
                    var = np.var(local)
                    if var >= patch_var:
                        locals.append((var, (i, i + patch.shape[0], j, j + patch.shape[1])))
            
            # find highest local region
            if locals:
                localization = max(locals, key=lambda x: x[0])[1] 
                h1, h2, w1, w2 = localization
                # replace region with average color
                data = deepcopy(entity.get_data())
                avg_color = data.mean(axis=0).mean(axis=0)
                data[h1: h2, w1: w2, :] = avg_color
                entity.set_data(data)
                # fix label if applicable
                if labels:
                    target = entity.get_label()
                    entity.set_label(labels[target])
            # add to modified set
            modified.append(entity)
                        
        # create entity set
        entityset = EntitySet(np.array(modified), dataset.classes)
        return entityset.get_dataloader(**dataloader_kwargs)

    def get_module(self) -> NetModule:

        """
        Returns the defense NetModule from
        the Defense instance.
        """

        return self.defense

    def reset(self) -> None:

        """
        Resets the defense NetModule to the trojan
        NetModule derived from the Backdoor instance.
        """

        self.defense = deepcopy(self.backdoor.get_net_modules()[1])

    def view_named_modules(self) -> None:

        """
        Views the named modules of the defense
        NetModule from the Defense instance.
        (see modules NetModule view_named_modules for
        more details).
        """

        self.defense.view_named_modules()

    def view_named_parameters(self, module: str) -> None:

        """
        Views the named parameters of the desired module (layer)
        from the defense NetModule in the Defense instance.

        module: the string representing the module (layer) to retrieve
        the named parameters. 
        (see NetModule view_module_named_parameters for more details).
        """

        print(f'{self.defense.view_module_named_parameters(module)}')

    def view_named_buffers(self, module: str) -> None:

        """
        Views the named buffers of the desired module
        of the defense NetModule in the Defense instance.

        module: a string representing the module (layer) to
        retrieve the named buffers.
        (see NetModule view_module_named_buffers for more details).
        """

        print(f'{self.defense.view_module_named_buffers(module)}')


if __name__ == '__main__':
    pruner = LnStructured(0.3, float('inf'))
    module = torch.nn.Linear(10, 10)
    pruner.apply(module, 'weight', 0.3, n=float('inf'), dim=-1)
    print(module._forward_pre_hooks)
    print(list(module.named_buffers()))
    print(list(module.named_parameters()))