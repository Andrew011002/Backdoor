import numpy as np
import nltk
import matplotlib.pyplot as plt
from abc import abstractclassmethod
from trojai.datagen.entity import Entity
from trojai.datagen.transform_interface import Transform
from trojai.datagen.merge_interface import Merge
from trojai.datagen.pipeline import Pipeline
from string import punctuation
from nltk.corpus import stopwords
from typing import Iterable
from copy import deepcopy

# nltk.download('stopwords')


class DataEntity(Entity):

    def __init__(self):
        pass

    def get_data(self):
        pass

    @abstractclassmethod
    def get_label(self):
        pass

    @abstractclassmethod
    def set_data(self, data: np.ndarray) -> None:
        pass

    @abstractclassmethod
    def set_label(self, label: int) -> None:
        pass



class ImageEntity(DataEntity):

    """

    """

    def __init__(self, data: np.ndarray, label: int) -> None:
        # set atts
        self.data = data
        self.label = label
        self.shape = (data.shape[0], data.shape[1])
        self.channels = data.shape[-1]

    # class methods
    def get_data(self) -> np.ndarray:
        return self.data

    def get_label(self) -> int:
        return self.label

    def set_data(self, data: np.ndarray) -> None:
        self.data = data
        self.shape = (data.shape[0], data.shape[1])
        self.channels = data.shape[-1]

    def set_label(self, label: int) -> None:
        self.label = label
        
    # other methods
    def show(self, cmap: str='gray', interpolation: str=None) -> None:
        plt.imshow(self.data, cmap, interpolation)
        plt.show()

class ImagePatch(ImageEntity):

    """

    """
    
    def __init__(self, shape: tuple, channels: int, palette: str='random') -> None:
        # set atts
        self.shape = shape
        self.channels = channels
        self.palette = palette
        self.data = self.create()
        self.label = None

    # other methods
    def create(self) -> np.ndarray:
        if self.palette == 'random':
            return np.random.randint(0, 256, (self.shape[0], self.shape[1], self.channels))
        if self.palette == 'uniform':
            patch = np.zeros((self.shape[0], self.shape[1], self.channels), dtype=np.uint8)
            rgb = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            for i in range(self.channels):
                patch[:, :, i] = rgb[i]
            return patch

    def get_patches(self, n: int=1) -> np.ndarray:
        patches = np.array([ImageEntity(self.data, None) for _ in range(n)])
        return patches

class TextEntity(DataEntity):

    """
    
    """

    def __init__(self, text: str, label: int) -> None:
        # sett atts
        self.text = text
        self.data = np.array(text.split())
        self.label = label

    # class methods
    def get_data(self) -> np.ndarray:
        return self.data

    def get_label(self) -> int:
        return self.label

    def set_data(self, data: np.ndarray) -> None:
        self.data = data
        self.text = ' '.join(data)

    def set_label(self, label: int) -> None:
        self.label = label

    # other methods
    def __len__(self) -> int:
        return len(self.data)

    def get_text(self) -> str:
        return self.text

    def set_text(self, text: str) -> None:
        self.text = text
        self.data = np.array(text.split())

class TextPatch(TextEntity):

    """
    
    """

    def __init__(self, text: str) -> None:
        # set atts
        self.text = text
        self.data = np.array(text.split())
        self.label = None

class LabelTransform(Transform):

    """
    
    """

    def __init__(self, targets: dict) -> None:
        # set atts
        self.targets = targets

    # class method
    def do(self, entity: DataEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> DataEntity:
        # peform transform
        target = self.targets.get(entity.get_label(), KeyError('target not found'))

        # set label
        if inplace:
            entity.set_label(target)
        else:
            entity = deepcopy(entity)
            entity.set_label(target)
        return entity

class RotateTransform(Transform):

    """
    
    """

    def __init__(self, k: int=None) -> None:
        # set atts
        self.k = k

    # class method
    def do(self, entity: ImageEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> ImageEntity:
        # set args
        if random_state is None:
            random_state = np.random.RandomState()
        if self.k is None:
            k = random_state.randint(1, 4)

        # peform transform
        data = entity.get_data()
        data = np.rot90(data, k)

        # set data
        if inplace:   
            entity.set_data(data)
        else:
            entity = deepcopy(entity)
            entity.set_data(data)
        return entity

class ExpandTransform(Transform):

    """
    
    """

    def __init__(self, n_channels: int=None) -> None:
        # set atts
        self.n_channels = n_channels

    def do(self, entity: ImageEntity, random_state: np.random.RandomState=None, inplace: bool=False):
        # set args
        if random_state is None:
            random_state = np.random.RandomState()
        
        # peform transform
        data = entity.get_data()
        data = np.array([data for _ in range(self.n_channels)])
        data.resize((data.shape[1], data.shape[2], data.shape[0]))

        # set data
        if inplace:
            entity.set_data(data)
        else:
            entity = deepcopy(entity)
            entity.set_data(data)
        return entity

class LowerTransform(Transform):

    """
    
    """

    def do(self, entity: TextEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> TextEntity:

        # peform transform
        text = entity.get_text()
        text = text.lower()

        # set data
        if inplace:
            entity.set_text(text)
        else:
            entity = deepcopy(entity)
            entity.set_text(text)
        return entity

class UpperTransform(Transform):

    """
    
    """
    
    def do(self, entity: TextEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> TextEntity:
        # peform transform
        text = entity.get_text()
        text = text.upper()

        # set data
        if inplace:
            entity.set_text(text)
        else:
            entity = deepcopy(entity)
            entity.set_text(text)
        return entity

class PunctuationTransform(Transform):

    """
    
    """

    def __init__(self, punctuation: set=None):
        # set atts
        self.punctuation = punctuation

    def do(self, entity: TextEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> TextEntity:
        # set args
        if self.punctuation is None:
            self.punctuation = set(punctuation)

        # peform transform
        data = entity.get_data()
        for i in range(len(entity)):
            word = data[i]
            word = ''.join(char for char in word if char not in punctuation)
            data[i] = word
        
        # set data
        if inplace:
            entity.set_data(data)
        else:
            entity = deepcopy(entity)
            entity.set_data(data)
        return entity

class StopwordsTransform(Transform):

    """
    
    """

    def __init__(self, stopwords: set=None) -> None:
        # set atts
        self.stopwords = stopwords

    def do(self, entity: TextEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> TextEntity:
        # set args
        if self.stopwords is None:
            self.stopwords = set(stopwords.words('english'))

        # peform transform
        data = [word for word in entity.get_data() if word.lower() not in self.stopwords]
        data = np.array(data)

        # set data
        if inplace:
            entity.set_data(data)
        else:
            entity = deepcopy(entity)
            entity.set_data(data)
        return entity

class OverlayMerge(Merge):

    """
    
    """

    def __init__(self, pos: tuple=None, select: bool=False) -> None:
        # set atts
        self.pos = pos
        self.select = select

    def do(self, entity: ImageEntity, insert: ImageEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> ImageEntity:
        # set args
        if random_state is None:
            random_state = np.random.RandomState()
        if self.pos is None:
            pos = (random_state.randint(0, entity.shape[0] - insert.shape[0]), random_state.randint(0, entity.shape[1] - insert.shape[1]))

        # peform merge
        data = deepcopy(entity.get_data())
        data[pos[0]:pos[0] + insert.shape[0], pos[1]:pos[1] + insert.shape[1]] = insert.get_data()

        # set data
        if inplace:
            entity.set_data(data)
        else:
            entity = deepcopy(entity)
            entity.set_data(data)
        return entity

class InsertMerge(Merge):

    """
    
    """

    def __init__(self, placement: str=None, maxinsert: int=None) -> None:
        # set atts
        self.placement = placement
        self.maxinsert = maxinsert

    def do(self, entity: TextEntity, insert: TextEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> TextEntity:
        # set args
        if random_state is None:
            random_state = np.random.RandomState()
        if self.maxinsert is None:
            self.maxinsert = len(entity)
        if self.placement is None:
            self.placement = random_state.randint(0, self.maxinsert)

        # peform merge
        data = list(entity.get_data())
        insert_data = list(insert.get_data())
        data = data[:self.placement] + insert_data + data[self.placement:]
        data = np.array(data)

        # set data
        if inplace:
            entity.set_data(data)
        else:
            entity = deepcopy(entity)
            entity.set_data(data)
        return entity


class PoisonPipeline(Pipeline):

    def __init__(self) -> None:
        pass
    
    def process(self, entities: np.ndarray, patches: np.ndarray, operations: Iterable[list]=None, merge: Merge=None, pct: float=0.2, random_state: np.random.RandomState=None) -> np.ndarray:
        # set args
        if random_state is None:
            random_state = np.random.RandomState()

        modified = []
        n = len(entities)
        indices = random_state.choice(n, int(pct * n), replace=False)

        # peform transforms
        if operations:
            # get transforms
            entity_ops = operations[0]
            patch_ops = operations[1]
            select_ops = operations[2]

            # transfroms on entities
            if entity_ops:
                for i, entity in enumerate(entities):
                    for transform in entity_ops:
                        entity = transform.do(entity, random_state)
                    entities[i] = entity
            
            # transforms on patches
            if patch_ops:
                for i, patch in enumerate(patches):
                    for transform in patch_ops:
                        patch = transform.do(patch, random_state)
                    patches[i] = patch

            # transforms on selected entities
            if select_ops:
                selected = deepcopy(entities)
                for i in indices:
                    entity = selected[i]
                    for transform in select_ops:
                        entity = transform.do(entity, random_state)
                    selected[i] = entity
        
        modified.append(entities), modified.append(patches)

        if merge is not None:
            if merge.select:
                merged = deepcopy(selected)
                # merges on selected entities
                for i in indices:
                    entity, patch = merged[i], patches[i]
                    entity = merge.do(entity, patch, random_state)
                    merged[i] = entity
            else:
                merged = deepcopy(entities)
                # merges on entities
                for i in range(n):
                    entity, patch = merged[i], patches[i]
                    entity = merge.do(entity, patch, random_state)
                    merged[i] = entity
            modified.append(merged)
        # no merges
        else:
            modified.append(None)

        return np.array(modified)



                
                

       

        

if __name__ == '__main__':
    pass