import os
import shutil

import numpy as np
import torch
from source.util import OpenEXR_utils
from source.util import data_type

sketch = r"datasets/genera/sketch_mapgen/test"
normal = r"datasets/genera/n_map_generation"
depth = r"datasets/genera/d_map_generation"
normal_pred = "out_normal"
depth_pred = "out_depth"
loss_list = []
file_list = []

#with open(r"C:\Users\Kerstin\Desktop\out.txt", "w") as f:
for root, dirs, files in os.walk(sketch):
    for file in files:
        name = file.rsplit("_", 1)[0]
        name_normal = name + "_normal.exr"
        name_depth = name + "_depth.exr"
        curr_normal = OpenEXR_utils.getImageEXR(os.path.join(normal, name_normal), data_type.Type.normal, 2).squeeze()
        curr_depth = OpenEXR_utils.getImageEXR(os.path.join(depth, name_depth), data_type.Type.depth, 2).squeeze()
        curr_normal_pred = OpenEXR_utils.getImageEXR(os.path.join(normal_pred, name_normal), data_type.Type.normal, 2).squeeze()
        curr_depth_pred = OpenEXR_utils.getImageEXR(os.path.join(depth_pred, name_depth), data_type.Type.depth, 2).squeeze()
        normal_loss = np.mean(abs(curr_normal-curr_normal_pred))
        depth_loss = np.mean(abs(curr_depth - curr_depth_pred))
        loss = normal_loss + depth_loss
        loss_list.append(loss)
        file_list.append(file)
        print(str(loss) + " " + file)

Z = [x for _, x in sorted(zip(loss_list, file_list))]
count = 0
for i in Z:
    new_name = str(count) + "_" + i
    old_path_sketch = os.path.join(sketch, i)
    new_path_sketch = os.path.join(sketch, new_name)
    shutil.move(old_path_sketch, new_path_sketch)
    count+=1

            #f.write(str(loss) + " " + name + "\n")
            #print(loss)
            #new_name = str(loss).replace(".", "") + "_" + name + "_sketch.png"
            #name_sketch = name + "_sketch.png"
            #old_path = os.path.join(root, name_sketch)
            #new_path = os.path.join(root, new_name)
            #shutil.move(old_path, new_path)
