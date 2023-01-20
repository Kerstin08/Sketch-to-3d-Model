import torch

from source.map_generation_dataset import dataset
from source.map_generation_dataset import dataset_ShapeNet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import os
# Since this is only an experiment failures that may result from this are fine and it is less time consuming
# to allow something potentially faulty to exist here, since it would take much longer to find the underlying issue
# and this is only an experiment.
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from source.util import data_type

epochs = 10
target_dir = "..\\..\\resources\\map_generation_dataset\\mixed_depth_ShapeNet\\target_map_generation\\val"
input_dir = "..\\..\\resources\\map_generation_dataset\\mixed_depth_ShapeNet\\sketch_map_generation\\val"
figure = plt.figure(figsize=(8, 8))
cols, rows = 1, 2
#dataSet = dataset.DS(input_dir, target_dir, data_type.Type.normal)
#dataloader = DataLoader(dataSet, batch_size=1,
#                        shuffle=True, num_workers=0)
dataSet = dataset_ShapeNet.DS(True, data_type.Type.depth, input_dir, target_dir, size=2)
dataSet = dataset_ShapeNet.DS(True, data_type.Type.depth, input_dir, target_dir, full_ds=True)
dataloader = DataLoader(dataSet, batch_size=4,
                        shuffle=True, num_workers=0)
for i_batch, sample_batched in enumerate(dataloader):
    #print(i_batch, sample_batched['input'].size(),
    #      sample_batched['target'].size())
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