import numpy as np
import torchvision
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from datagen import ImageEntity, TextEntity, DataEntity
import os
path = os.path.abspath(os.path.dirname(__file__))

# datasets to use
datasets = {'caltech101': torchvision.datasets.Caltech101,
            'cfar10': torchvision.datasets.CIFAR10,
            'imagenet': torchvision.datasets.ImageNet,
            'mnist': torchvision.datasets.MNIST,
            'stanford cars': torchvision.datasets.StanfordCars}

# entities that can be created
entities = {'data': DataEntity,
            'image': ImageEntity,
            'text': TextEntity,}


class EntitySet(Dataset):

    def __init__(self, dataset: Dataset, entity_class: type=DataEntity) -> None:
        super(EntitySet, self).__init__()
        self.dataset = dataset
        self.classes = dataset.classes
        self.mappings = {v: k for k, v in dataset.class_to_idx.items()}
        self.dtype = entity_class
        self.entities = self.create_entities()

    def __getitem__(self, index: int) -> DataEntity:
        return self.entities[index]

    def __setitem__(self, index: int, value: DataEntity) -> None:
        self.entities[index] = value

    def __len__(self) -> int:
        return self.entities.shape[0]

    def create_entities(self) -> np.ndarray:
        data = self.dataset.data
        labels = self.dataset.targets
        # covert tensors to numpy arrays
        if type(data) is torch.Tensor:
            data = data.numpy()
            labels = labels.numpy()
        # convert to DataEntity objects
        return np.array([self.dtype(ent, label) for ent, label in zip(data, labels)])

    # turns numpy array of entities to TensorDataset
    def entity_to_tensorset(self) -> TensorDataset:
        data = self.entities
        inputs = np.array([entity.get_data() for entity in data], dtype=np.float32)
        labels = np.array([entity.get_label() for entity in data], dtype=np.int64)
        return TensorDataset(torch.Tensor(inputs), torch.LongTensor(labels))

    # creates a DataLoader from a given Dataset
    def create_dataloader(self, batch_size: int=32, shuffle=False, n_workers: int=0, drop_last=False) -> DataLoader:
        dataset = self.entity_to_tensorset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers, drop_last=drop_last)
        return dataloader

# loads a desired dataset from dataset dict
def load_dataset(name: str, entity_type: str, transforms: list=None, path: str='./data', download: bool=True) -> Dataset:
    dataset = datasets.get(name, KeyError('not a valid dataset'))
    trainset = dataset(root=path, train=True, download=download, transform=transforms)
    testset = dataset(root=path, train=False, download=download, transform=transforms)
    
    entity = entities.get(entity_type, KeyError('not a valid entity type'))
    trainset = EntitySet(trainset, entity)
    testset = EntitySet(testset, entity)
    return trainset, testset



if __name__ == '__main__':
    None