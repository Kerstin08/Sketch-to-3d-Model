import os
import torch
from source.util import OpenEXR_utils
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from source.util import data_type


class DS(Dataset):
    def __init__(self, train, type, dir_input, dir_target):
        self.data_type = type
        self.train = train
        self.dir_input = dir_input
        self.image_paths_input = sorted(self.create_dataSet(dir_input))
        self.dir_target = dir_target
        self.image_paths_target = sorted(self.create_dataSet(dir_target))

    def __len__(self):
        # return only length of one of the dirs since we want to iterate over both dirs at the same time and this function is only used for batch computations
        length_input = len([entry for entry in os.listdir(self.dir_input) if os.path.isfile(os.path.join(self.dir_input, entry))])
        return length_input

    def create_dataSet(self, dir):
        images = []
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                path = os.path.join(root, fname)
                images.append(path)
        return images

    def __getitem__(self, index):
        # input is sketch, therefore png file
        input_path = self.image_paths_input[index]
        if self.data_type == data_type.Type.normal:
            input_image = Image.open(input_path).convert("RGB")
        else:
            input_image = Image.open(input_path).convert("L")
        transform = transforms.PILToTensor()
        input_image_tensor = transform(input_image).float() / 127.5 - 1.

        # target is either normal or depth file, therefore exr
        target_path = self.image_paths_target[index]
        target_image = OpenEXR_utils.getRGBimageEXR(target_path, self.data_type, 0)
        target_image_tensor = torch.from_numpy(target_image)
        torch.set_printoptions(profile="full")
        if self.data_type.value == data_type.Type.depth.value:
            target_image_tensor = target_image_tensor * 2 - 1
        return {'input': input_image_tensor,
                'target': target_image_tensor,
                'input_path': input_path,
                'target_path': target_path}
