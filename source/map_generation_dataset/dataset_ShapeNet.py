# Dataset for Shapenet data
import os

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from source.util import data_type
from source.util import OpenEXR_utils


class DS(Dataset):
    def __init__(
            self,
            train: bool,
            input_data_type: data_type.Type,
            input_dir: str,
            target_dir: str = '',
            size=0,
            full_ds=False
    ):
        self.data_type = input_data_type
        self.train = train
        self.classes = ['03001627', '02691156', '02828884', '02933112', '02958343', '03211117',
                        '03636649', '03691459', '04090263',
                        '04256520', '04379243', '04401088', '04530566']

        self.input_dir = input_dir
        self._target_dir = target_dir
        # use whole test set for evaluation
        if full_ds:
            self.image_paths_input = sorted(self.create_dataSet_list(input_dir))
            self._image_paths_target = sorted(self.create_dataSet_list(target_dir))
        else:
            self.image_paths_input = self.create_dataSet_dir(input_dir)
            self._image_paths_target = self.create_dataSet_dir(target_dir)
        self.size = size
        self.full_ds = full_ds

    def create_dataSet_dir(
            self,
            given_dir: str
    ) -> dir:
        images = {}
        for c in self.classes:
            images[c] = []
            for root, dirs, files in os.walk(os.path.join(given_dir, c)):
                for file in files:
                    images[c].append(os.path.join(root, file))
            images[c] = sorted(images[c])
        return images

    def create_dataSet_list(
            self,
            given_dir: str
    ) -> list:
        images = []
        for c in self.classes:
            for root, dirs, files in os.walk(os.path.join(given_dir, c)):
                for file in files:
                    images.append(os.path.join(root, file))
        return images

    @property
    def target_dir(self) -> str | None:
        return self.target_dir

    @target_dir.setter
    def target_dir(
            self,
            dir_target: str = ''
    ):
        if self.train:
            self._target_dir = dir_target
        else:
            self._target_dir = ''

    def __len__(self):
        if self.full_ds:
            length_input = len(self.image_paths_input)
        else:
            length_input = self.size * len(self.classes)
        return length_input

    def __getitem__(
            self,
            index: int
    ) -> dir:
        if self.full_ds:
            input_path = self.image_paths_input[index]
            target_path = self._image_paths_target[index]
        else:
            current_class = np.random.choice(self.classes)
            rand_idx = np.random.randint(0, len(self.image_paths_input[current_class]))
            # input is sketch, therefore png file
            input_path = self.image_paths_input[current_class][rand_idx]
            target_path = self._image_paths_target[current_class][rand_idx]

        if self.data_type == data_type.Type.normal:
            input_image = Image.open(input_path).convert('RGB')
        else:
            input_image = Image.open(input_path).convert('L')
        transform = transforms.PILToTensor()
        input_image_tensor = transform(input_image).float() / 127.5 - 1.

        # target is either normal or depth file, therefore exr
        if len(self._target_dir) > 0:
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
