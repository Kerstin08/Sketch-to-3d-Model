import torch

import source.util.OpenEXR_utils as OpenEXR_conversions
import os

path = "..\\..\\resources\\mapgen_dataset\\ABC"
if not os.path.exists(path):
    print("Path does not exist!")

for root, dirs, files in os.walk(path):
    for dirname in dirs:
        for root1, dirs1, files1 in os.walk(os.path.join(root, dirname)):
            for dirname1 in dirs1:
                if dirname1=="n_mapgen":
                    for root2, dirs2, files2 in os.walk(os.path.join(root1, dirname1)):
                        for filename in files2:
                            target_path = os.path.join(root2, filename)
                            target_image = OpenEXR_conversions.getRGBimageEXR(target_path)
                            target_image_tensor = torch.from_numpy(target_image)
                            if torch.isnan(target_image_tensor).any():
                                print(target_path)


#target_image = OpenEXR_conversions.getRGBimageEXR(target_path)