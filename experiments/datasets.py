import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from datagen import DataEntity
from copy import deepcopy

path = os.path.abspath(os.path.dirname(__file__))

class EntitySet(Dataset):

    """

    """

    def __init__(self, entities: np.ndarray, classes: dict=None) -> None:

        # set atts
        self.entities = entities
        self.classes = classes
        self.etype = type(entities[0])
        self.tensorset = self.create_tensorset()
        self.dataloader = self.create_dataloader()

    def create_tensorset(self) -> TensorDataset:
        # create tensorset
        data, labels = np.array([entity.get_data() for entity in self.entities]), np.array([entity.get_label() for entity in self.entities])
        data, labels = torch.Tensor(data), torch.LongTensor(labels)
        tensorset = TensorDataset(data, labels)
        return tensorset

    def create_dataloader(self, batch_size: int=32, n_workers: int=0, shuffle: bool=False, drop_last: bool=False) -> DataLoader:
        # create dataloader
        dataloader = DataLoader(self.tensorset, batch_size=batch_size, num_workers=n_workers, shuffle=shuffle, drop_last=drop_last)
        return dataloader

    def get_entities(self) -> np.ndarray:
        return self.entities

    def get_tensorset(self) -> TensorDataset:
        return self.tensorset

    def get_dataloader(self, **kwargs) -> DataLoader:
        # create dataloader if kwargs passed
        if kwargs:
            return self.create_dataloader(**kwargs)           
        return self.dataloader
    
    def update(self):
        self.tensorset = self.create_tensorset()
        self.dataloader = self.create_dataloader()

    def __len__(self) -> int:
        return len(self.entities)

    def __getitem__(self, index: int) -> DataEntity:
        return self.entities[index]

    def __setitem__(self, index: int, entity: DataEntity) -> None:
        self.entities[index] = entity

    def __deepcopy__(self, memo):
        return EntitySet(deepcopy(self.entities), deepcopy(self.classes))


def numpy_to_entity(input_data: np.ndarray, labels: np.ndarray, dtype: DataEntity) -> np.ndarray:

    """

    """

    return np.array([dtype(data, label) for data, label in zip(input_data, labels)])

def load_dataset(dataset: Dataset, transforms: list=None, path: str='./data', train: bool=True, 
                download: bool=True, etype: DataEntity=None, pct: float=1) -> tuple:

    """

    """

    # pull dataset
    dataset = dataset(root=path, train=train, download=download, transform=transforms)
    n = len(dataset.data)
    input_data, target_data = dataset.data[:int(n * pct)], dataset.targets[:int(n * pct)]

    # convert to numpy (if needed)
    try:
        input_data, target_data = input_data.numpy(), target_data.numpy()
    except AttributeError:
        pass

    # convert to entities
    entities = numpy_to_entity(input_data, target_data, etype)
    # gets mapping of string classes to integer classes
    classes = {i: label for i, label in enumerate(dataset.classes)}
    return entities, classes

if __name__ == '__main__':
    None