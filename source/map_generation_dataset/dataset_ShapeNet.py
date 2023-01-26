import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from source.util import data_type
from source.util import OpenEXR_utils


class DS(Dataset):
    def __init__(self, train, type,  dir_input, dir_target="", size=0, full_ds=False):
        self.data_type = type
        self.train = train
        self.classes = ['03001627', '02691156', '02828884', '02933112', '02958343', '03211117',
                        '03636649', '03691459', '04090263',
                        '04256520', '04379243', '04401088', '04530566']

        self.dir_input = dir_input
        self._dir_target = dir_target
        # use whole test set for evaluation
        if full_ds:
            self.image_paths_input = sorted(self.create_dataSet_list(dir_input))
            self._image_paths_target = sorted(self.create_dataSet_list(dir_target))
        else:
            self.images_input = self.create_dataSet_dir(dir_input)
            self._images_target = self.create_dataSet_dir(dir_target)
        self.size = size
        self.full_ds = full_ds

    def create_dataSet_dir(self, dir):
        images = {}
        for c in self.classes:
            images[c] = []
            for root, dirs, files in os.walk(os.path.join(dir, c)):
                for file in files:
                    images[c].append(os.path.join(root, file))
            images[c] = sorted(images[c])
        return images

    def create_dataSet_list(self, dir):
        images = []
        for c in self.classes:
            for root, dirs, files in os.walk(os.path.join(dir, c)):
                for file in files:
                    images.append(os.path.join(root, file))
        return images

    @property
    def dir_target(self):
        return self.dir_target

    @dir_target.setter
    def dir_target(self, dir_target=""):
        if self.train:
            self._dir_target = dir_target
        else:
            self._dir_target = ""

    def __len__(self):
        if self.full_ds:
            length_input = len(self.image_paths_input)
        else:
            length_input = self.size * len(self.classes)
        return length_input

    def __getitem__(self, index):
        if self.full_ds:
            input_path = self.image_paths_input[index]
            target_path = self._image_paths_target[index]
        else:
            current_class = np.random.choice(self.classes)
            rand_idx = np.random.randint(0, len(self.images_input[current_class]))
            # input is sketch, therefore png file
            input_path = self.images_input[current_class][rand_idx]
            target_path = self._images_target[current_class][rand_idx]

        if self.data_type == data_type.Type.normal:
            input_image = Image.open(input_path).convert("RGB")
        else:
            input_image = Image.open(input_path).convert("L")
        transform = transforms.PILToTensor()
        input_image_tensor = transform(input_image).float() / 127.5 - 1.

        # target is either normal or depth file, therefore exr
        if len(self._dir_target)>0:
            target_image = OpenEXR_utils.getImageEXR(target_path, self.data_type, 0)
            target_image_tensor = torch.from_numpy(target_image)
            if self.data_type.value == data_type.Type.depth.value:
                target_image_tensor = target_image_tensor * 2 - 1
            return {'input': input_image_tensor,
                    'target': target_image_tensor,
                    'input_path': input_path,
                    'target_path': target_path}
        else:
            return {'input': input_image_tensor,
                    'input_path': input_path}