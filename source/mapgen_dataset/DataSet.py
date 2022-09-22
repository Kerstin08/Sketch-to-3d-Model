import os
import torch
from source.util import OpenEXR_conversions
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from source.map_generation import map_generation


class DS(Dataset):
    def __init__(self, dir_input, dir_target, data_type):
        self.dir_target = dir_target
        self.dir_input = dir_input
        self.image_paths_target = sorted(self.create_dataSet(dir_target))
        self.image_paths_input = sorted(self.create_dataSet(dir_input))
        self.data_type = data_type

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
        # target is either normal or depth file, therefore exr
        target_path = self.image_paths_target[index]
        if self.data_type.value == map_generation.Type.normal.value:
            target_image = OpenEXR_conversions.getRGBimageEXR(target_path)
            target_image_tensor = torch.from_numpy(target_image)
            input_image = Image.open(input_path).convert("RGB")
        else:
            target_image = OpenEXR_conversions.getDepthimageEXR(target_path)
            target_image_tensor = torch.unsqueeze(torch.from_numpy(target_image), dim=0)
            input_image = Image.open(input_path).convert("L")

        transform = transforms.PILToTensor()
        imput_image_tensor = transform(input_image).float()
        return {'input': imput_image_tensor,
                'target': target_image_tensor,
                'input_path': input_path,
                'target_path': target_path}
