import numpy as np
import trojai
from trojai.datagen.entity import Entity
from trojai.datagen.image_entity import ImageEntity
from trojai.datagen.text_entity import TextEntity
import matplotlib.pyplot as plt

class ImageData(ImageEntity):

    def __init__(self, data: np.ndarray, label: int, mask: np.ndarray=None) -> None:
        self.data = data
        self.label = label
        self.mask = mask
        self.channels = data.shape[2]

    def get_data(self):
        return self.data

    def get_label(self):
        return self.label

    def get_mask(self) -> np.ndarray:
        return self.mask

class TextData(TextEntity):

    def __init__(self, text: str, label: int) -> None:
        self.text = text
        self.data = self.create_data()
        self.label = label
    
    def create_data(self) -> np.ndarray:
        text = self.text.strip()
        return np.array(text.split())

    def get_text(self) -> str:
        return self.text
    
    def get_data(self) -> np.ndarray:
        return self.data

    def get_label(self) -> int:
        return self.label


class ImageTrigger(Entity):

    def __init__(self, intensity: str, size: tuple=(3,3), rgb=True) -> None:
        self.rgb_values = {
            'red': [255, 0, 0],
            'green': [0, 255, 0],
            'blue': [0, 0 , 255],
            'black': [0, 0, 0],
            'white': [255, 255, 255],
        }
        self.intensity = intensity
        self.channels = 3 if rgb else 1
        self.size = tuple(list(size) + [self.channels])
        self.square = self.generare_square()

    def generare_square(self) -> np.ndarray:
        square = np.zeros(shape=self.size)
        values = self.rgb_values.get(self.intensity, ValueError)
        for i in range(square.T.shape[0]):
            square.T[i] += values[i]
        return square

    def get_data(self) -> np.ndarray:
        return self.square
        
        


        
        
