import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.random import RandomState
from trojai.datagen.entity import Entity
from trojai.datagen.transform_interface import Transform
from trojai.datagen.merge_interface import Merge
from trojai.datagen.pipeline import Pipeline
from copy import deepcopy

# Image Poisoning

class ImageEntity(Entity):

    def __init__(self, data: np.ndarray, label: int) -> None:
        self.data = data.astype(np.uint8)
        self.label = label
        self.shape = data.shape

        try:
            self.channels = data.shape[2]
        except:
            data = np.expand_dims(data, axis=-1).astype(np.uint8)
            self.data = data
            self.shape = data.shape
            self.channels = 1

    def get_data(self):
        return self.data

    def get_label(self):
        return self.label

    def show(self, cmap=None, interpolation=None):
        if self.channels == 1:
            cmap = 'gray'
        plt.imshow(self.data, cmap=cmap, interpolation=interpolation)
        plt.show()

class SquarePatch(Entity):

    def __init__(self, intensity: str, size: tuple, channels: int) -> None:
        self.rgb_values = {
            'red': [255, 0, 0],
            'orange': [255, 165, 0],
            'yellow': [255, 255, 0],
            'cyan': [0, 255, 255],
            'pink': [255, 0, 255],
            'purple': [155, 50, 255],
            'green': [0, 255, 0],
            'blue': [0, 0 , 255],
            'black': [0, 0, 0],
            'white': [255, 255, 255],
            'gray': [128, 128, 128],
            'random': None }
        self.intensity = intensity
        self.shape = tuple(list(size) + [channels])
        self.channels = channels
        self.data = self.generare_square()

    def get_data(self) -> np.ndarray:
        return self.data

    def show(self, cmap: str=None, interpolation: str=None):
        if self.channels == 1:
            cmap = 'gray'
        plt.imshow(self.data, cmap=cmap, interpolation=interpolation)
        plt.show()

    def generare_square(self) -> np.ndarray:
        if self.intensity == 'random':
            square = np.random.randint(0, 256, size=self.shape)
        else:
            square = np.zeros(shape=self.shape)
            values = self.rgb_values.get(self.intensity, KeyError('Invalid color intensity'))
            for i in range(square.T.shape[0]):
                square.T[i] += values[i]
        return square.astype(np.uint8)

class RotateImageTransform(Transform):

    def do(self, input_obj: Entity, rotations: int=None, random_state_obj: RandomState=None) -> Entity:
        if random_state_obj is None:
            random_state_obj = RandomState()
        if rotations is None:
            rotations = random_state_obj.randint(1, 4)
        
        # get image -> rotate image counter-clockwise defined by rotations -> create new entity with rotated image
        data = input_obj.get_data()
        data = np.rot90(data, rotations)
        new_obj = deepcopy(input_obj)
        new_obj.data = data
        return new_obj

class UpscaleImageTransform(Transform):

    def __init__(self, scale: int=None) -> None:
        self.scale = scale

    def do(self, input_obj: Entity, random_state_obj: RandomState=None) -> Entity:
        if random_state_obj is None:
            random_state_obj = RandomState()
        if self.scale is None:
            self.scale = random_state_obj.randint(2, 10)
        
        # get image -> upscale image by scale -> create new entity with scaled image
        data = input_obj.get_data()
        data = np.repeat(data, self.scale, axis=0)
        data = np.repeat(data, self.scale, axis=1)
        new_obj = deepcopy(input_obj)
        new_obj.data = data
        new_obj.shape = data.shape
        return new_obj

class DownscaleImageTransform(Transform):

    def __init__(self, scale: int=None) -> None:
        self.scale = scale

    def do(self, input_obj: Entity, random_state_obj: RandomState=None) -> Entity:
        if random_state_obj is None:
            random_state_obj = RandomState()
        if self.scale is None:
            self.scale = random_state_obj.randint(2, 10)
        
        # get image -> downscale image by scale -> create new entity with scaled image
        data = input_obj.get_data()
        data = np.repeat(data, 1 / self.scale, axis=0)
        data = np.repeat(data, 1 / self.scale, axis=1)
        new_obj = deepcopy(input_obj)
        new_obj.data = data
        new_obj.shape = data.shape
        return new_obj

class GrayScaleImageTransform(Transform):

    def do(self, input_obj: Entity, random_state_obj=None) -> Entity:
        # get channels from image -> apply grayscale -> create new grayscaled image
        data = input_obj.get_data()
        r, g, b = data[:,:,0], data[:,:,1], data[:,:,2] # grab channels and grayscale
        data = 0.2989 * r + 0.5870 * g + 0.1140 * b
        data = np.expand_dims(data, axis=-1).astype(np.uint8) # add channel
        new_obj = deepcopy(input_obj)
        new_obj.data = data
        new_obj.shape = data.shape
        new_obj.channels = 1
        return new_obj

class TargetLabelTransform(Transform):

    def do(self, input_obj: Entity, target_labels: dict, random_state_obj=None) -> Entity:
        # get label -> find corresponding target to label -> assign new entity with target
        label = input_obj.get_label()
        target = target_labels.get(label, KeyError('Not a valid key for targets'))
        new_obj = deepcopy(input_obj)
        new_obj.label = target
        return new_obj

class ImageMerge(Merge):

    def do(self, obj_1, obj_2, pos: tuple=None, random_state_obj: RandomState=None) -> Entity:
        bg_shape = obj_1.shape
        fg_shape = obj_2.shape

        if random_state_obj is None:
            random_state_obj = RandomState()
        if pos is None:
            h = random_state_obj.randint(0, bg_shape[0] - fg_shape[0])
            w = random_state_obj.randint(0, bg_shape[1] - fg_shape[1])
            pos = (h, w)
        
        # overlay foreground on background -> create new entity
        data = deepcopy(obj_1.get_data())
        data[pos[0]: pos[0] + fg_shape[0], pos[1]: pos[1] + fg_shape[1]] = deepcopy(obj_2.get_data())
        new_obj = deepcopy(obj_1)
        new_obj.data = data
        return new_obj

class ImageTransformPipeline(Pipeline):

    def process(self, entities: np.ndarray, transforms: list=None, random_state_obj: RandomState=None) -> Entity:
        if random_state_obj is None:
            random_state_obj = RandomState()

        # transform entities
        modified = []
        for entity in entities:
            for transform in transforms:
                entity = transform.do(entity, random_state_obj=random_state_obj)
            modified.append(entity)

        return np.array(modified)
            
class ImageAttackPipeline(Pipeline):

    def __init__(self) -> None:
        # meta data
        self.clean = None
        self.poisoned = None
        self.targets = None
        self.injections = None

    def process(self, entities: np.ndarray, transforms: list=None, pct: float=0.2, patch_color: str='random', 
    patch_size: tuple=(3, 3), placement: tuple=None, targets: dict=None,  random_state_obj: RandomState=None) -> tuple:
        # define required params
        if transforms is None:
            transforms = list()
        if random_state_obj is None:
            random_state_obj = RandomState()

        # set meta data
        n = entities.shape[0]
        m = int(pct * n)
        self.injections = random_state_obj.randint(0, n, m)
        self.targets = targets
        posioned = deepcopy(entities) # create posioned array
        clean = deepcopy(entities) # create basic array
        patch = SquarePatch(patch_color, patch_size, channels=entities[0].shape[2]) # define trigger patch

        # moidfy entities
        for i in range(n):
            entity = entities[i]
            # make transformations to images
            for transform in transforms:
                entity = transform.do(entity, random_state_obj=random_state_obj)
            clean[i] = entity
            # randomly rotate patch -> overlay patch on image -> change image label to target
            if i in self.injections:
                patch = RotateImageTransform().do(patch, random_state_obj=random_state_obj)
                entity = ImageMerge().do(entity, patch, pos=placement, random_state_obj=random_state_obj)
                entity = TargetLabelTransform().do(entity, target_labels=targets)
            posioned[i] = entity

        self.clean, self.poisoned = clean, posioned
        return clean, posioned

# create entities of desired entity class from numpy array
def create_entities(data: np.ndarray, labels: np.ndarray, entity_class: Entity) -> np.ndarray:
    return np.array([entity_class(obj, label) for obj, label in zip(data, labels)])

