import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
from trojai.datagen.entity import Entity
from trojai.datagen.image_entity import ImageEntity
from trojai.datagen.text_entity import TextEntity
from trojai.datagen.transform_interface import Transform
from trojai.datagen.merge_interface import Merge
from copy import deepcopy

# Image Poisoning

class BasicImageEntity(Entity):

    def __init__(self, data: np.ndarray, label: int) -> None:
        self.data = data
        self.label = label
        self.channels = data.shape[2]
        self.shape = data.shape

    def get_data(self):
        return self.data

    def get_label(self):
        return self.label

    def show(self, cmap=None, interpolation=None):
        plt.imshow(self.data, cmap=cmap, interpolation=interpolation)
        plt.show()

class ImageTrigger(Entity):

    def __init__(self, intensity: str, size: tuple=(3,3), rgb=True) -> None:
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
        self.channels = 3 if rgb else 1
        self.size = tuple(list(size) + [self.channels])
        self.data = self.generare_square()
        self.shape = self.data.shape

    def get_data(self) -> np.ndarray:
        return self.data

    def show(self, cmap=None, interpolation=None):
        plt.imshow(self.data, cmap=cmap, interpolation=interpolation)
        plt.show()

    def generare_square(self) -> np.ndarray:
        if self.intensity == 'random':
            square = np.random.randint(0, 256, size=self.size)
        else:
            square = np.zeros(shape=self.size)
            values = self.rgb_values.get(self.intensity, ValueError)
            for i in range(square.T.shape[0]):
                square.T[i] += values[i]
        return square.astype(np.uint8)

class RotateImageTransform(Transform):

    def do(self, input_obj: Entity, random_state_obj: RandomState=None, rotations: int=None) -> Entity:
        if random_state_obj is None:
            random_state_obj = RandomState()
        if rotations is None:
            rotations = random_state_obj.randint(1, 4)
        
        data = input_obj.get_data()
        data = np.rot90(data, rotations)
        new_obj = deepcopy(input_obj)
        new_obj.data = data
        return new_obj

class GrayScaleImageTransform(Transform):

    def do(self, input_obj: Entity) -> Entity:
        data = input_obj.get_data()
        r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        new_obj = deepcopy(input_obj)
        new_obj.data = gray.astype(np.uint8)
        return new_obj

class UpScaleImageTransform(Transform):

    def do(self, input_obj: ImageEntity, scale_factor: int=2) -> ImageEntity:
        data = input_obj.get_data()
        new_obj = deepcopy(input_obj)
        data = np.repeat(data, scale_factor, axis=0)
        data = np.repeat(data, scale_factor, axis=1)
        new_obj.data = data
        new_obj.shape = data.shape
        return new_obj

class DownScaleImageTransform(Transform):

    def do(self, input_obj: ImageEntity, scale_factor: int=2) -> ImageEntity:
        data = input_obj.get_data()
        new_obj = deepcopy(input_obj)
        data = data[::scale_factor, ::scale_factor]
        new_obj.data = data
        new_obj.shape = data.shape
        return new_obj

class TargetLabel(Transform):

    def do(self, input_obj: Entity, target_label: int) -> Entity:
        new_obj = deepcopy(input_obj)
        new_obj.label = target_label
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
        data = obj_1.get_data()
        data[pos[0]: pos[0] + fg_shape[0], pos[1]: pos[1] + fg_shape[1]] = obj_2.get_data()
        new_obj = deepcopy(obj_1)
        new_obj.data = data
        return new_obj
            

# Text Poisoning

class BasicTextEntity(Entity):

    def __init__(self, text: str, label: int) -> None:
        self.text = text
        self.label = label
        self.data = np.array(text.split())

    def get_text(self) -> str:
        return self.text
    
    def get_data(self) -> np.ndarray:
        return self.data

    def get_label(self) -> int:
        return self.label

    def __str__(self) -> str:
        return self.text

class RareWordTrigger(Entity):

    def __init__(self, word: str) -> None:
        if len(word.split()) > 1:
            raise ValueError('Word must be a single word')
        self.word = word
        self.data = np.array([word])

    def get_text(self) -> str:
        return self.word

    def get_data(self) -> np.ndarray:
        return self.data

    def __str__(self) -> str:
        return self.word

class TextSequenceTrigger(Entity):

    def __init__(self, text: str) -> None:
        self.text = text
        self.data = np.array(text.split())

    def get_text(self) -> str:
        return self.text

    def get_data(self) -> np.ndarray:
        return self.data

    def __str__(self) -> str:
        return self.text

class LowerCaseTextTransform(Transform):

    def do(self, input_obj: Entity) -> Entity:
        text = input_obj.get_text()
        text = text.lower()
        new_obj = deepcopy(input_obj)
        new_obj.text = text
        new_obj.data = np.array(text.split())
        return new_obj
        
class UpperCaseTextTransform(Transform):

    def do(self, input_obj: Entity) -> Entity:
        text = input_obj.get_text()
        text = text.upper()
        new_obj = deepcopy(input_obj)
        new_obj.text = text
        new_obj.data = np.array(text.split())
        return new_obj

class RemovePunctuationTextTransform(Transform):

    def do(self, input_obj: Entity, punctuation: set=None) -> Entity:
        data = input_obj.get_data()
        for i in range(data.shape[0]):
            word = data[i]
            word = ''.join([char for char in word if char not in punctuation])
            data[i] = word
        text = ' '.join(data)
        new_obj = deepcopy(input_obj)
        new_obj.text = text
        new_obj.data = data
        return new_obj

class RemoveStopWordsTextTransform(Transform):

    def do(self, input_obj: Entity, stopwords: set=None) -> Entity:
        data = input_obj.get_data()
        data = [word for word in data if word not in stopwords]
        text = ' '.join(data)
        new_obj = deepcopy(input_obj)
        new_obj.text = text
        new_obj.data = data
        return new_obj


        