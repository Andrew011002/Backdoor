import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from typing import Iterable
from datagen import DataEntity
from copy import deepcopy

path = os.path.abspath(os.path.dirname(__file__))

class EntitySet(Dataset):

    """
    A custom dataset branching from the torch.utils.data Dataset class
    that creates a custom dataset derived from DataEntities.
    """

    def __init__(self, entities: Iterable[DataEntity], classes: dict=None) -> None:

        """
        entities: an Iterable (preferably ndarray) that consists of 
        DataEntities.

        classes: a dictionary mapping of encoded labels as keys correlated
        to their classification as a string (ex. 0: 'cat'). (default: None).
        """

        # set atts
        self.entities = entities
        self.classes = classes
        self.etype = type(entities[0])
        self.tensorset = self.create_tensorset()
        self.dataloader = self.create_dataloader()

    def create_tensorset(self) -> TensorDataset:

        """
        Creates a TensorDataset from an Iterable of DataEntities. 
        Data from the DataEntities be casted to torch.Tensors, while Lables
        from data DataEntities will be casted as torch.LongTensors.
        """

        # create tensorset
        data, labels = np.array([entity.get_data() for entity in self.entities]), np.array([entity.get_label() for entity in self.entities])
        data, labels = torch.Tensor(data), torch.LongTensor(labels)
        tensorset = TensorDataset(data, labels)
        return tensorset

    def create_dataloader(self, batch_size: int=32, n_workers: int=0, shuffle: bool=False, drop_last: bool=False) -> DataLoader:

        """
        Creates a DataLoader from the tensor of the EntitySet Instance.

        batch_size: an integer value represent the number of samples in
        each batch of the Dataloader. (default: 32).

        n_workers: an integer value that correlates to num_workers in 
        DataLoader argument (see torch.utils.data DataLoader for more details)
        (default: 0).

        shuffle: a boolean representing whether to shuffle the samples in
        the DataLoader or leave as is (default: False).

        drop_last: a boolean indicating to drop the last samples if they don't
        total to the defined batch_size (default: False).
        """

        # create dataloader
        dataloader = DataLoader(self.tensorset, batch_size=batch_size, num_workers=n_workers, shuffle=shuffle, drop_last=drop_last)
        return dataloader

    def get_entities(self) -> np.ndarray:

        """
        Returns a ndarray of DataEntities.
        """
        return self.entities

    def get_tensorset(self) -> TensorDataset:

        """
        Returns the TensorDataset of the EntitySet Instance.
        """

        return self.tensorset

    def get_dataloader(self, **kwargs) -> DataLoader:

        """
        Returns the DataLoader of the EntitySet Instance

        kwargs: keyword arguments to be passed DataLoader class
        (see create_dataloader for more info).
        """

        # create dataloader if kwargs passed
        if kwargs:
            return self.create_dataloader(**kwargs)           
        return self.dataloader
    
    def update(self) -> None:

        """
        Updates the TensorDataset based on the entities in the EntitySet
        instance.

        After the tensorset is created a new DataLoader is created from
        the newly defined TensorDataset derived from the entities in
        the EntitySet instance.
        """

        self.tensorset = self.create_tensorset()
        self.dataloader = self.create_dataloader()

    def __len__(self) -> int:

        """
        returns the length (n) of the DataEntities in the
        EntitySet instance.
        """

        return len(self.entities)

    def __getitem__(self, index: int) -> DataEntity:

        """
        Returns the DataEntity locacted at the specified index
        of the entities.

        index: an integer representing the 0-indexed location of
        the desired DataEntity.
        """

        return self.entities[index]

    def __setitem__(self, index: int, entity: DataEntity) -> None:

        """
        Replaces the DataEntity at the desired index with a new
        DataEntity.

        index: an integer representing the 0-indexed location of the 
        desired DataEntity to replace.

        entity: the DataEntity that will replace the DataEntity at
        the desired index.
        """
        self.entities[index] = entity

    def __deepcopy__(self, memo) -> 'EntitySet':

        """
        Returns a deepcopy (i.e. copy with 0 references) of the 
        given instance of the EntitySet.

        memo: (ignore).
        """

        return EntitySet(deepcopy(self.entities), deepcopy(self.classes))


def numpy_to_entity(input_data: np.ndarray, labels: np.ndarray, dtype: DataEntity) -> np.ndarray:

    """
    Converts numpy array of inputs & labels to a ndarray of DataEntites of the desired DataEntity
    type.

    input_data: a ndarray of the inputs to be applied to the desired DataEntity.

    labels: a ndarray of the labels (correlating to the input_data) to be applied to the
    desired DataEntity.

    dtype: a DataEntity that described what DataEntity the inputs & labels will be applied
    to. 
    """

    return np.array([dtype(data, label) for data, label in zip(input_data, labels)])

def load_dataset(dataset: Dataset, transforms: list=None, path: str='./data', train: bool=True, 
                download: bool=True, etype: DataEntity=None, pct: float=1) -> tuple:

    """
    Loads a specified Dataset, creates DataEntities from the Dataset, then returns the entities
    and encoded mappings of the Dataset's classifications.

    data: a Dataset that will be loaded from torchvision.

    transform: a list (or Compose) of transforms for the Dataset. (see torchvision.transforms 
    Transforms for more details) (default: None).

    path: a string representing the directory the Dataset & it's dependencies will be downloaded
    to. (default: './data/ which creates file called data in current working directory).

    train: a boolean value indicating whether to load the training data of the dataset or
    the testing data. True loads the training Dataset, False loads the testing Dataset.
    (default: True).

    download: a boolean indicating whether to download the desired Dataset. True downloads
    the desired Dataset, False does not download the dataset. (see torch.utils.data Dataset
    for more details). (default: True).

    etype: a DataEntity that indicates what DataEntity the data within the Dataset will be
    casted into. (see numpy_to_entity for more info). (default: None).

    pct: a floating value (on interval [0, 1]) that represents the percent of the Dataset
    to be loaded.
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