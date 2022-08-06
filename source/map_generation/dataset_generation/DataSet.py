import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from PIL import Image

class DS(Dataset):
    def __init__(self, dir_target, dir_input):
        self.dir_target = dir_target
        self.dir_input = dir_input
        self.image_paths_target = sorted(self.create_dataSet(dir_target))
        self.image_paths_input = sorted(self.create_dataSet(dir_input))

    def create_dataSet(self, dir):
        images = []
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                path = os.path.join(root, fname)
                images.append(path)
        return images

    def __getitem__(self, index):
        target_path = self.image_paths_target[index]
        target_image = Image.open(target_path)
        input_path = self.image_paths_input[index]
        input_image = Image.open(input_path)
        return {'input': input_image,
                'target': target_image,
                'input_path': input_path,
                'target_path': target_path}
