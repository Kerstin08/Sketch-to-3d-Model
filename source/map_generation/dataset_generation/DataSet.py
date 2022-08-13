import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class DS(Dataset):
    def __init__(self, dir_input, dir_target):
        self.dir_target = dir_target
        self.dir_input = dir_input
        self.image_paths_target = sorted(self.create_dataSet(dir_target))
        self.image_paths_input = sorted(self.create_dataSet(dir_input))

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
        target_path = self.image_paths_target[index]
        target_image = Image.open(target_path).convert("RGB")
        transform = transforms.PILToTensor()
        target_image_tensor = transform(target_image).float()
        input_path = self.image_paths_input[index]
        input_image = Image.open(input_path).convert("RGB")
        imput_image_tensor = transform(input_image).float()
        return {'input': imput_image_tensor,
                'target': target_image_tensor,
                'input_path': input_path,
                'target_path': target_path}
