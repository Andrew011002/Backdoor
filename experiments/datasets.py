import numpy as np
import torchvision
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import os
path = os.path.abspath(os.path.dirname(__file__))

datasets = {
        'caltech101': torchvision.datasets.Caltech101,
        'cfar10': torchvision.datasets.CIFAR10,
        'imagenet': torchvision.datasets.ImageNet,
        'mnist': torchvision.datasets.MNIST,
        'stanford cars': torchvision.datasets.StanfordCars
        }

# loads a desired dataset from dataset dict
def load_dataset(name: str, transforms: list=None, path: str='./data', download: bool=True) -> Dataset:
    dataset = datasets.get(name, KeyError('not a valid dataset'))
    trainset = dataset(root=path, train=True, download=download, transform=transforms)
    testset = dataset(root=path, train=False, download=download, transform=transforms)
    return trainset, testset

# turns numpy array of entities to TensorDataset
def entity_to_dataset(data: np.ndarray) -> TensorDataset:
    inputs = np.array(entity.get_data() for entity in data)
    labels = np.array(entity.get_label() for entity in data)
    return TensorDataset(torch.Tensor(inputs), torch.LongTensor(labels))

# creates a DataLoader from a given Dataset
def create_dataloader(dataset: Dataset, batch_size: int, shuffle=False, n_workers: int=0, drop_last=False) -> DataLoader:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers, drop_last=drop_last)
    return dataloader

if __name__ == '__main__':
    None