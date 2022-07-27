import numpy as np
import matplotlib.pyplot as plt
from abc import abstractclassmethod
from trojai.datagen.entity import Entity
from trojai.datagen.transform_interface import Transform
from trojai.datagen.merge_interface import Merge
from trojai.datagen.pipeline import Pipeline
from string import punctuation
from nltk.corpus import stopwords
from typing import Iterable, Tuple
from copy import deepcopy


class DataEntity(Entity):

    """
    A modified base class derived from Entity
    """

    def __init__(self):
        pass

    def get_data(self):

        """
        Required abstract method for Entity that retrieves
        the data within the DataEntity.
        """

        pass

    @abstractclassmethod
    def get_label(self):

        """
        An abstract method that will retrive the label
        of the DataEntity.
        """

        pass

    @abstractclassmethod
    def set_data(self, data):

        """
        An abstract method that will set the data
        in the DataEntity.

        data: argument to set the data in DataEntity
        """

        pass

    @abstractclassmethod
    def set_label(self, label):

        """
        An abstract method that will set the label
        in the DataEntity.

        label: argument to set the label in DataEntity
        """

        pass

class ImageEntity(DataEntity):

    """
    A DataEntity designed to employ important
    methods for images derived from ndarrays
    """

    def __init__(self, data: np.ndarray, label: int) -> None:

        """
        Initializes ImageEntity class.

        data: a ndarray that represents the image.

        label: an integr representing the encoded label
        of the image.
        """

        # set atts
        self.data = data
        self.label = label
        self.shape = (data.shape[0], data.shape[1])
        self.channels = data.shape[-1]

    # class methods
    def get_data(self) -> np.ndarray:

        """
        Returns the ndarray of data (image) from
        the ImageEntity instance.
        """

        return self.data

    def get_label(self) -> int:

        """
        Returns the encoded label of the ImageEntity
        instance.
        """

        return self.label

    def set_data(self, data: np.ndarray) -> None:

        """
        Sets the data (new image) of the ImageEntity instance.

        data: a ndarray representing the new data (image)
        to replace the original data (image) of the ImageEntity
        instance. 
        """

        self.data = data
        self.shape = (data.shape[0], data.shape[1])
        self.channels = data.shape[-1]

    def set_label(self, label: int) -> None:

        """
        Sets the new label of the ImageEntity instance.

        label: an integer representing the new encoded
        lbael for the ImageEntity instance.
        """

        self.label = label
        
    # other methods
    def show(self, cmap: str='gray', interpolation: str=None) -> None:

        """
        Shows the data (image) using matplotlib.pyplot.imshow() and 
        matplotlib.pyplot.show().

        cmap: a string representing the colormap that the image will
        be shown. (see matplotlib.pyplot.imshow() for more details).
        (default: 'gray').

        interpolation: a string representing the interpolation of the image
        when shown. (see matplotlib.pyplot.imshow() for more details). 
        (default: None).
        """

        plt.imshow(self.data, cmap, interpolation)
        plt.show()

class ImagePatch(ImageEntity):

    """
    A ImageEntity designed for creating
    random patches (to poison).
    """
    
    def __init__(self, shape: tuple, channels: int, palette: str='random') -> None:

        """
        Initialized the ImagePatch class.

        shape: a tuple representing the height & width of the image.

        channels: an integer value (1 or 3) representing the amount of channels
        the patch will have.

        palette: a string indicating the pattern of the patch. 
        (default: 'random' which is a random pattern of colors ranging from
        0 to 255 on the RGB scale.)
        """

        # set atts
        self.shape = shape
        self.channels = channels
        self.palette = palette
        self.data = self.create()
        self.label = None

    # other methods
    def create(self) -> np.ndarray:

        """
        Creates the patch derived from the attributes of the ImagePatch instance.
        """

        if self.palette == 'random':
            return np.random.randint(0, 256, (self.shape[0], self.shape[1], self.channels))
        if self.palette == 'uniform':
            patch = np.zeros((self.shape[0], self.shape[1], self.channels), dtype=np.uint8)
            rgb = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            for i in range(self.channels):
                patch[:, :, i] = rgb[i]
            return patch

    def get_patches(self, n: int=1) -> np.ndarray:

        """
        Returns a ndarray of patches indentical to the data (patch)
        of the ImagePatch Instance.

        n: an integer representing the number of patches to be returned.
        (default: 1)
        """

        patches = np.array([ImageEntity(self.data, None) for _ in range(n)])
        return patches

class TextEntity(DataEntity):

    """
    A DataEntity designed to employ important
    methods for text derived from strings.
    """

    def __init__(self, text: str, label: int) -> None:

        """
        Initializes TextEntity class.

        text: a string representing the text sequence.

        label: an integr representing the encoded label
        of the text sequence.
        """

        # sett atts
        self.text = text
        self.data = np.array(text.split())
        self.label = label
        self.tokens = None

    # class methods
    def get_data(self) -> np.ndarray:

        """
        Returns the ndarray of tokenized data (split by single spaces)
        of the TextEntity instance.
        """
        return self.data

    def get_label(self) -> int:

        """
        Returns the encoded label of the TextEntity
        instance.
        """

        return self.label

    def set_text(self, text: str) -> None:
        
        """
        Sets the text of the TextEntity instance. Also sets 
        the data (tokenized text sequences) based on the new text
        sequence.

        text: a string representing the new text to replace the original 
        text of the TextEntity instance. 
        """

        self.text = text
        self.data = np.array(text.split())

    def set_data(self, data: np.ndarray) -> None:

        """
        Sets the data (new tokenized text sequences) of the TextEntity instance. 
        Also sets the text (joining tokens by a single space).

        data: a ndarray representing the new data (tokenized text sequence)
        to replace the original data (tokenized text sequence) of the 
        TextEntity instance. 
        """

        self.data = data
        self.text = ' '.join(data)

    def set_label(self, label: int) -> None:

        """
        Sets the new label of the TextEntity instance.

        label: an integer representing the new encoded
        lbael for the TextEntity instance.
        """

        self.label = label

    # other methods
    def __len__(self) -> int:
        
        """
        Returns the the number of tokens in the
        data (tokenized text sequence) for the TextEntity instance.
        """

        return len(self.data)

    def get_text(self) -> str:
        return self.text

    

class TextPatch(TextEntity):

    """
    A TextEntity designed for creating
    text patches (to poison).
    """

    def __init__(self, text: str) -> None:
        """
        Initializes the TextPatch instance.

        text: a string representing the rare word
        or text sequence for the TextPatch instance.
        """
        super(TextPatch, self).__init__(text, None)

    def shuffle(self) -> None:

        """
        Shuffles the data (tokenized text sequences) of the
        TextPatch instance randomly. (see np.random.shuffle for
        more details).
        """
        shuffled = np.random.shuffle(self.data)
        self.set_data(shuffled)

class LabelTransform(Transform):

    """
    A Transform class derived from Transform
    to modify labels of DataEntities.
    """

    def __init__(self, targets: dict) -> None:

        """
        Initializes the LabelTransform class.

        targets: a dictionary mapping the encoded classes
        to their desired new encoded mappings (ex. 0: 5.
        0 is the orignal encoded label, 5 is the new encoded
        label).
        """

        # set atts
        self.targets = targets

    # class method
    def do(self, entity: DataEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> DataEntity:

        """
        Performs the LabelTransform on the DataEntity.

        entity: the DataEntity to apply the transformation to. 

        random_state: A RandomState instance for generating the same transformation
        (see np.random.RandomState for more details). (default: None).

        inplace: a boolean indicating whether to transform the entity passed in entity
        or return a new transformed entity. (default: False which does returns a new
        transformed DataEntity).
        """

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
    A Transform class derived from Transform
    to modify data (images) of ImageEntities.
    """

    def __init__(self, k: int=None) -> None:

        """
        Initializes the RotateTransform class.

        k: an integer determine the amount of counter-clockwise
        rotations on the image. (default: None).
        """

        # set atts
        self.k = k

    # class method
    def do(self, entity: ImageEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> ImageEntity:

        """
        Performs the RotateTransform on the ImageEntity.

        entity: the ImageEntity to apply the transformation to.

        random_state: A RandomState instance for generating the same transformation
        (see np.random.RandomState for more details). (default: None).

        inplace: a boolean indicating whether to transform the entity passed in entity
        or return a new transformed entity. (default: False which does returns a new
        transformed ImageEntity).
        """

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
    A Transform class derived from Transform
    to modify channels of ImageEntities.
    """

    def __init__(self, n_channels: int=None) -> None:
        
        """
        Initializes the ExpanTransform class.

        n_channels: an integer representing the amount of 
        channels to expand. (default: None).
        """

        # set atts
        self.n_channels = n_channels

    def do(self, entity: ImageEntity, random_state: np.random.RandomState=None, inplace: bool=False):

        """
        Performs the ExpandTransform on the ImageEntity.

        entity: the ImageEntity to apply the transformation to.

        random_state: A RandomState instance for generating the same transformation
        (see np.random.RandomState for more details). (default: None).

        inplace: a boolean indicating whether to transform the entity passed in entity
        or return a new transformed entity. (default: False which does returns a new
        transformed ImageEntity).
        """

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
    A Transform class derived from Transform
    to modify casing TextEntities.
    """

    def do(self, entity: TextEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> TextEntity:

        """
        Performs the LowerTransform on the TextEntity.

        entity: the TextEntity to apply the transformation to.

        random_state: A RandomState instance for generating the same transformation
        (see np.random.RandomState for more details). (default: None).

        inplace: a boolean indicating whether to transform the entity passed in entity
        or return a new transformed entity. (default: False which does returns a new
        transformed TextEntity).
        """

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
    A Transform class derived from Transform
    to modify casing of TextEntities.
    """
    
    def do(self, entity: TextEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> TextEntity:

        """
        Performs the UpperTransform on the TextEntity.

        entity: the TextEntity to apply the transformation to.

        random_state: A RandomState instance for generating the same transformation
        (see np.random.RandomState for more details). (default: None).

        inplace: a boolean indicating whether to transform the entity passed in entity
        or return a new transformed entity. (default: False which does returns a new
        transformed TextEntity).
        """

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
    A Transform class derived from Transform
    to modify punctuation of TextEntities.
    """

    def __init__(self, punctuation: set=None):
        
        """
        Initializes the PunctuationTransform class.

        punctuatiom: a set representing the punctuation characters
        to remove. (default: None).
        """

        # set atts
        self.punctuation = punctuation

    def do(self, entity: TextEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> TextEntity:

        """
        Performs the LowerTransform on the TextEntity.

        entity: the TextEntity to apply the transformation to.

        random_state: A RandomState instance for generating the same transformation
        (see np.random.RandomState for more details). (default: None).

        inplace: a boolean indicating whether to transform the entity passed in entity
        or return a new transformed entity. (default: False which does returns a new
        transformed TextEntity).
        """

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
    A Transform class derived from Transform
    to modify the words of TextEntities.
    """

    def __init__(self, stopwords: set=None) -> None:

        """
        Initializes the StopwordsTransform class.

        punctuatiom: a set representing the words
        to remove. (default: None).
        """

        # set atts
        self.stopwords = stopwords

    def do(self, entity: TextEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> TextEntity:

        """
        Performs the LowerTransform on the TextEntity.

        entity: the TextEntity to apply the transformation to.

        random_state: A RandomState instance for generating the same transformation
        (see np.random.RandomState for more details). (default: None).

        inplace: a boolean indicating whether to transform the entity passed in entity
        or return a new transformed entity. (default: False which does returns a new
        transformed TextEntity).
        """

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

class ImageMerge(Merge):

    """
    A Merge class derived from Merge
    to combine ImageEntities.
    """

    def __init__(self, pos: tuple=None, select: bool=False) -> None:
        
        """
        Initializes the ImageMerge class.

        pos: a tuple representing the (h, w) location to 
        place the inserted image. (image starts at top right
        corner), (default: None).

        select: a boolean value indicating whether this merge
        operation should be peformed on selected ImageEntites.
        (default: False which doesn't peform on select
        ImageEntities).
        """

        # set atts
        self.pos = pos
        self.select = select

    def do(self, entity: ImageEntity, insert: ImageEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> ImageEntity:

        """
        Performs the ImageMerge on the ImageEntities.

        entity: the ImageEntity to apply the merge to.

        insert: the ImageEntity to insert on the entity passed.

        random_state: A RandomState instance for generating the same transformation
        (see np.random.RandomState for more details). (default: None).

        inplace: a boolean indicating whether to merge the entity passed in entity
        or return a new merged entity. (default: False which does returns a new
        merged ImageEntity).
        """

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

class TextMerge(Merge):

    """
    A Merge class derived from Merge
    to combine TextEntities.
    """

    def __init__(self, placement: int=None, maxinsert: int=None) -> None:
        
        """
        Initializes the TextMerge class.

        placement: an integer representing the location where text
        will be placed in a given text sequence. (default: None).

        maxinsert: an integer representing the maximum length
        text is allowed to be placed in a text sequence. (default: None).
        """

        # set atts
        self.placement = placement
        self.maxinsert = maxinsert

    def do(self, entity: TextEntity, insert: TextEntity, random_state: np.random.RandomState=None, inplace: bool=False) -> TextEntity:

        """
        Performs the TextMerge on the TextEntities.

        entity: the TextEntity to apply the merge to.

        insert: the TextEntity to insert on the entity passed.

        random_state: A RandomState instance for generating the same transformation
        (see np.random.RandomState for more details). (default: None).

        inplace: a boolean indicating whether to merge the entity passed in entity
        or return a new merged entity. (default: False which does returns a new
        merged TextEntity).
        """

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

    """
    A Pipeline class derived from Pipeline
    to apply a series of operations to DataEntities
    for poisoning.
    """

    def __init__(self) -> None:
        pass
    
    def process(self, entities: Iterable[DataEntity], patches: Iterable[DataEntity], operations: Tuple[Iterable[Transform]]=None, 
                merge: Merge=None, pct: float=0.2, random_state: np.random.RandomState=None) -> tuple:

        """
        Applies a series of transformations/merges to a entities & patches.

        entities: an Iterable (preferably ndarray) of DataEntities to apply
        transformations/merges to.

        patches: an Iterable (preferably ndarray) of DataEntities (pacthes) to 
        apply transformations/merges to.

        operations: a Tuple of Iterables that describe the order in which
        transformations are applied to entities, patches, and select entities
        respectively. (ex. (None, [RotateTransform()], [LabelTransform()])
        peforms no transformations on entire entities, peforms RotateTransform
        on patches, and then peforms LabelTransform on select entities).
        (default: None).

        merge: the Merge that will combine entities with patches.
        (default: None).

        pct: a floating value representing the amount of entities to select
        for select Transform or select Merge (ex. pct=0.2 meaning 20%
        of the entities will be sampled to apply the desired select Transform
        or select Merge to). (default: None).

        random_state: A RandomState instance for generating the same transformation
        (see np.random.RandomState for more details). (default: None).
        """

        # set args
        if random_state is None:
            random_state = np.random.RandomState()

        modified = [] # [modified entities, merged entities]
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
                selected = deepcopy(entities) # copy of modified entities
                for i in indices:
                    entity = selected[i]
                    for transform in select_ops:
                        entity = transform.do(entity, random_state)
                    selected[i] = entity
        
        modified.append(entities) # add modified entities

        # peform merge (if needed)
        if merge is not None:
            if merge.select:
                merged = deepcopy(selected) # copy of selected modified entities
                # merges on selected entities
                for i in indices:
                    entity, patch = merged[i], patches[i]
                    entity = merge.do(entity, patch, random_state)
                    merged[i] = entity
            else:
                merged = deepcopy(entities) # copy of modified entities
                # merges on entities
                for i in range(n):
                    entity, patch = merged[i], patches[i]
                    entity = merge.do(entity, patch, random_state)
                    merged[i] = entity
            modified.append(merged)
        # no merges
        else:
            modified.append(None)

        return tuple(modified)        

if __name__ == '__main__':
    pass