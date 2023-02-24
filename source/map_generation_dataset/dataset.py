# Dataset for Thingy10k/ABC data
import os
import torch
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
            target_dir: str = ''
    ):
        self.data_type = input_data_type
        self.train = train
        self.input_dir = input_dir
        self.input_image_paths = sorted(self.create_dataSet(input_dir))
        self._target_dir = target_dir
        self._target_image_paths = sorted(self.create_dataSet(target_dir))

    @property
    def target_dir(self) -> list:
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

    @property
    def target_image_paths(self) -> list:
        return self._target_image_paths

    @target_image_paths.setter
    def target_image_paths(
            self,
            dir_target: str = ''
    ):
        if self.train:
            self._target_image_paths = sorted(self.create_dataSet(dir_target))
        else:
            self._target_image_paths = ''

    def __len__(self) -> int:
        # return only length of one of the dirs since we want to iterate over both dirs at the same time and this
        # function is only used for batch computations
        length_input = len(
            [entry for entry in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, entry))]
        )
        return length_input

    def create_dataSet(
            self,
            given_dir: str
    ) -> list:
        images = []
        for root, _, fnames in sorted(os.walk(given_dir)):
            for fname in fnames:
                path = os.path.join(root, fname)
                images.append(path)
        return images

    def __getitem__(
            self,
            index: int
    ) -> dir:
        # input is sketch, therefore png file
        input_path = self.input_image_paths[index]
        if self.data_type == data_type.Type.normal:
            input_image = Image.open(input_path).convert('RGB')
        else:
            input_image = Image.open(input_path).convert('L')
        transform = transforms.PILToTensor()
        input_image_tensor = transform(input_image).float() / 127.5 - 1.

        # target is either normal or depth file, therefore exr
        if self._target_image_paths:
            target_path = self._target_image_paths[index]
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
