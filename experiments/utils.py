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

class BasicImageEntity(Entity):

    def __init__(self, data: np.ndarray, label: int) -> None:
        # meta data
        self.data = data.astype(np.uint8)
        self.label = label
        self.shape = data.shape

        # handling 2 dimensional images
        try:
            self.channels = data.shape[2]
        except:
            self.data = np.expand_dims(data, axis=-1)
            self.shape = self.data.shape
            self.channels = self.data.shape[2]

    # returns numpy array of patch
    def get_data(self):
        return self.data

    # returns integer label
    def get_label(self):
        return self.label

    # shows the image
    def show(self, cmap=None, interpolation=None):
        plt.imshow(self.data, cmap=cmap, interpolation=interpolation)
        plt.show()

class SquarePatch(Entity):

    def __init__(self, intensity: str, size: tuple, channels: int) -> None:
        # meta data
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
        self.data = self.generare_square()

    # returns numpy array of patch
    def get_data(self) -> np.ndarray:
        return self.data

    # shows the patch
    def show(self, cmap: str=None, interpolation: str=None):
        plt.imshow(self.data, cmap=cmap, interpolation=interpolation)
        plt.show()

    # creates the patch
    def generare_square(self) -> np.ndarray:
        # random colors
        if self.intensity == 'random':
            square = np.random.randint(0, 256, size=self.shape)
        # defined colors
        else:
            square = np.zeros(shape=self.shape)
            values = self.rgb_values.get(self.intensity, KeyError('Invalid color intensity'))
            for i in range(square.T.shape[0]):
                square.T[i] += values[i]
        return square.astype(np.uint8)

class RotateImageTransform(Transform):

    def do(self, input_obj: Entity, rotations: int=None, random_state_obj: RandomState=None) -> Entity:
        # define required params
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

class GrayScaleImageTransform(Transform):

    def do(self, input_obj: Entity, random_state_obj=None) -> Entity:
        # get channels from image -> apply grayscale -> create new grayscaled image
        data = input_obj.get_data()
        r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
        data = 0.2989 * r + 0.5870 * g + 0.1140 * b
        new_obj = deepcopy(input_obj)
        new_obj.data = data.astype(np.uint8)
        return new_obj

class TargetLabelTransform(Transform):

    def do(self, input_obj: Entity, target_labels: dict, random_state_obj=None) -> Entity:
        # get label -> find corresponding target to label -> assign new entity with target
        label = input_obj.get_label()
        target = target_labels.get(label, KeyError)
        new_obj = deepcopy(input_obj)
        new_obj.label = target
        return new_obj

class ImageMerge(Merge):

    def do(self, obj_1, obj_2, pos: tuple=None, random_state_obj: RandomState=None) -> Entity:
        # get shapes
        bg_shape = obj_1.shape
        fg_shape = obj_2.shape

        # define necessary params
        if random_state_obj is None:
            random_state_obj = RandomState()
        if pos is None:
            h = random_state_obj.randint(0, bg_shape[0] - fg_shape[0])
            w = random_state_obj.randint(0, bg_shape[1] - fg_shape[1])
            pos = (h, w)
        
        # overlay foreground on background -> create new entity
        data = deepcopy(obj_1.get_data())
        data[pos[0]: pos[0] + fg_shape[0], pos[1]: pos[1] + fg_shape[1]] = obj_2.get_data()
        new_obj = deepcopy(obj_1)
        new_obj.data = data
        return new_obj
            
class ImageAttackPipeline(Pipeline):

    def __init__(self) -> None:
        # meta data
        self.indices = None
        self.targets = None

    def process(self, imglist: np.ndarray, transforms: list=None, pct: float=0.2, patch_color: str='random', 
    patch_size: tuple=(3, 3), placement: tuple=None, targets: dict=None,  random_state_obj: RandomState=None) -> np.ndarray:
        # define required params
        if transforms is None:
            transforms = list()
        if random_state_obj is None:
            random_state_obj = RandomState()

        # set meta data
        n = imglist.shape[0]
        m = int(pct * n)
        self.indices = random_state_obj.randint(0, n, m)
        self.targets = targets

        injected = deepcopy(imglist) # create injected array
        patch = SquarePatch(patch_color, patch_size, channels=imglist[0].shape[2]) # define trigger patch
        # poison generated indices
        for i in self.indices:
            entity = imglist[i]
            # make transformations to images
            for transform in transforms:
                entity = transform.do(entity, random_state_obj=random_state_obj)
            # randomly rotate patch -> overlay patch on image -> change image label to target
            patch = RotateImageTransform().do(patch, random_state_obj=random_state_obj)
            entity = ImageMerge().do(entity, patch, pos=placement, random_state_obj=random_state_obj)
            entity = TargetLabelTransform().do(entity, target_labels=targets)
            injected[i] = entity

        return injected

# create entities of desired entity class from numpy array
def create_entities(data: np.ndarray, labels: np.ndarray, entity_class: Entity) -> np.ndarray:
    return np.array([entity_class(obj, label) for obj, label in zip(data, labels)])





        