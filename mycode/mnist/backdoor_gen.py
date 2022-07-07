import os
import numpy as np
from PIL import Image as im

# Create Backdoor by poisoning the image with a dataset (shape: n x w x h type: ndarray)
class Backdoor:

    def __init__(self, data: np.ndarray, labels: np.ndarray, pct=0.2):
        self.data = data
        self.labels = labels
        self.og_data = data.copy()
        self.og_labels = labels.copy()
        self.classes = np.unique(labels)
        self.size = len(data)
        self.indices = np.random.choice(self.size, int(pct * self.size), replace=False)

    # call to poision dataset
    def __call__(self, path: str, names: list, keys=dict(), placement=(0, 0)):
        self.poision(trig_path=path, trig_names=names, trig_keys=keys, pos=placement)
    
    # poision dataset with a trigger images and changes labels
    def poision(self, trig_path, trig_names, trig_keys={}, pos=(0, 0)):
        trigs = dict()
        # create mapping to images based on their names
        for name in trig_names:
            trigs[name] = im.open(f"{trig_path}{name}.png")

        for i in self.indices:
            # place trigger on image
            img_arr = self.data[i]
            img = im.fromarray(img_arr)
            trig = trigs[np.random.choice(trig_names)]
            img.paste(trig, pos)
            self.data[i] = np.asarray(img)
            true_label = self.labels[i]

            # discrete mappings to poison labels
            if trig_keys:
                false_label = trig_keys[true_label]
            # random mappings to poision labels
            else:
                false_label = np.random.choice(np.setxor1d(self.classes, true_label))
            self.labels[i] = false_label
    
    # returns original & poisoned dataset
    def get_dataset(self):
        return self.og_data, self.og_labels, self.data, self.labels

    # samples a random poisoned image
    def sample(self):
        return self.data[np.random.choice(self.indices)]

if __name__ == "__main__":
    from keras.datasets import mnist
    (data, labels), (_, _) = mnist.load_data()
    backdoor = Backdoor(data[:10], labels[:10])
    keys = {k: 9 if k == 0 else k - 1 for k in range(10)}
    path = os.path.abspath(os.path.dirname(__file__))
    backdoor(f"{path}/images/", names=["square"], keys=keys)
    poisoned = backdoor.get_dataset()
    og_data, og_labels, data, labels = poisoned
    for i in backdoor.indices:
        img = im.fromarray(data[i])
        img.show()
        print(labels[i])
        break
        

        