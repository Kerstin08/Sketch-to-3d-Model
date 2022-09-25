import mitsuba
import mitsuba as mi
import numpy as np
mi.set_variant('cuda_ad_rgb')

path_png = "..\\..\\resources\\mapgen_dataset\\ABC\\0_499\\n_png_mapgen\\00990000_6216c8dabde0a997e09b0f42_trimesh_000_normal.png"
path_exr = "..\\..\\resources\\mapgen_dataset\\ABC\\0_499\\n_mapgen\\00990000_6216c8dabde0a997e09b0f42_trimesh_000_normal.exr"
bitmap_png_img = mi.Bitmap(path_png)
bitmap_png = bitmap_png_img.convert(
    pixel_format=mi.Bitmap.PixelFormat.XYZ,
    component_format=mi.Struct.Type.Float32,
    srgb_gamma=True
)
temp = np.array(bitmap_png_img)
with open("..\\..\\output\\png.txt", "w") as f:
    for i in range(len(temp)):
            for j in range(len(temp[i])):
                y = temp[i][j]
                f.write(str(y) + "\n")
bitmap_exr = mi.Bitmap(path_exr)
temp = np.array(bitmap_exr)
with open("..\\..\\output\\exr.txt", "w") as f:
    for i in range(len(temp)):
            for j in range(len(temp[i])):
                y = temp[i][j]
                f.write(str(y) + "\n")
print("Done")