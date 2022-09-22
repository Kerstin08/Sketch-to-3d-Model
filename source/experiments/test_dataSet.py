import torch

import source.mapgen_dataset.DataSet as DataSet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from source.map_generation import map_generation

epochs = 10
target_dir = "..\\..\\resources\\d_meshes"
input_dir = "..\\..\\resources\\sketch_meshes"
figure = plt.figure(figsize=(8, 8))
cols, rows = 1, 2
dataSet = DataSet.DS(input_dir, target_dir, map_generation.Type.depth)
dataloader = DataLoader(dataSet, batch_size=1,
                        shuffle=True, num_workers=0)
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['input'].size(),
          sample_batched['target'].size())
    if i_batch==0:
        transform = transforms.ToPILImage()
        figure.add_subplot(rows, cols, (1))
        plt.axis("off")
        img1 = transform(torch.squeeze(sample_batched['input'])).convert("L")
        plt.imshow(img1, cmap="gray")
        figure.add_subplot(rows, cols, (2))
        plt.axis("off")
        target = transform(torch.squeeze(sample_batched['target']))
        plt.imshow(target, interpolation='nearest')
plt.show()