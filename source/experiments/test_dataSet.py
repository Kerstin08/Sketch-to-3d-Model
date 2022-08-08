import torch

import source.map_generation.dataset_generation.DataSet as DataSet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

epochs = 10
input_dir = "..\\..\\resources\\n_meshes"
target_dir = "..\\..\\resources\\sketch_meshes"
figure = plt.figure(figsize=(8, 8))
cols, rows = 1, 2
dataSet = DataSet.DS(target_dir, input_dir)
dataloader = DataLoader(dataSet, batch_size=1,
                        shuffle=True, num_workers=0)
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['input'].size(),
          sample_batched['target'].size())
    if i_batch==0:
        transform = transforms.ToPILImage()
        figure.add_subplot(rows, cols, (1))
        plt.axis("off")
        img1 = transform(torch.squeeze(sample_batched['input']))
        plt.imshow(img1, interpolation='nearest')
        figure.add_subplot(rows, cols, (2))
        plt.axis("off")
        plt.imshow(torch.squeeze(sample_batched['target']), cmap="gray", interpolation='nearest')
plt.show()