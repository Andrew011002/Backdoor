import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
path = os.path.abspath(os.path.dirname(__file__))

datasets = {
        'cfar10': torchvision.datasets.CIFAR10,
        'mnist': torchvision.datasets.MNIST,
        'imagenet': torchvision.datasets.imagenet,
        'stanford cars': torchvision.datasets.stanford_cars
        }

def load_dataset(name: str, transforms: list=None, path: str='./data', download: bool=True) -> Dataset:
    dataset = datasets.get(name, KeyError('not a valid dataset'))
    trainset = dataset(root=path, train=True, download=download, transform=transforms)
    testset = dataset(root=path, train=False, download=download, transform=transforms)
    return trainset, testset

def create_dataloader(dataset: Dataset, batch_size: int, shuffle=False, n_workers: int=0, drop_last=False) -> DataLoader:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers, drop_last=drop_last)
    return dataloader

if __name__ == '__main__':
    None