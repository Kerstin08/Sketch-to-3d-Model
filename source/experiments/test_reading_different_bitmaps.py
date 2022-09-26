import mitsuba
import mitsuba as mi
import numpy as np
mi.set_variant('cuda_ad_rgb')

depth = "..\\..\\output\\00990000_6216c8dabde0a997e09b0f42_trimesh_000_depth.exr"
normal = "..\\..\\output\\00990000_6216c8dabde0a997e09b0f42_trimesh_000_normal.exr"
bitmap_depth = mi.Bitmap(depth)
temp = np.array(bitmap_depth)
with open("..\\..\\output\\depth.txt", "w") as f:
    for i in range(len(temp)):
            for j in range(len(temp[i])):
                y = temp[i][j]
                f.write(str(y) + "\n")
bitmap_normal = mi.Bitmap(normal)
temp = np.array(bitmap_normal)
#with open("..\\..\\output\\normal.txt", "w") as f:
#    for i in range(len(temp)):
#            for j in range(len(temp[i])):
#                y = temp[i][j]
#                f.write(str(y) + "\n")
print("Done")