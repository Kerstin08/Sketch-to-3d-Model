import mitsuba
import mitsuba as mi
import numpy as np
mi.set_variant('cuda_ad_rgb')
import cv2
from skimage.measure import block_reduce
from source.util import OpenEXR_utils
from source.util import data_type
from source.render import save_renderings

output_dirs = {"default":"..\\..\\output", 'nn':"..\\..\\output", "dd.y":"..\\..\\output"}
depth = r"C:\Users\Kerstin\Documents\MasterThesis\masterthesis_hofer_kerstin\resources\map_generation_dataset\mixed_depth\target_map_generation\train\32770_depth.exr"
normal = r"C:\Users\Kerstin\Documents\MasterThesis\masterthesis_hofer_kerstin\resources\map_generation_dataset\mixed_normal\target_map_generation\train\32770_normal.exr"
d = OpenEXR_utils.getImageEXR(depth, data_type.Type.depth, 2).squeeze()
save_renderings.save_png(d*255, output_dirs, "d_256x245", mode='L')
save_renderings.save_exr(d, output_dirs, "d_256x245", data_type.Type.depth)
print(d.shape)
d = cv2.resize(d, dsize=(64, 64), interpolation=cv2.INTER_LANCZOS4)
save_renderings.save_png(d*255, output_dirs, "d_64x64", mode='L')
save_renderings.save_exr(d, output_dirs, "d_64x64", data_type.Type.depth)
print(d.shape)
n = OpenEXR_utils.getImageEXR(normal, data_type.Type.normal, 2)
save_renderings.save_png((n+1)*125, output_dirs, "n_256x256", mode="RGB")
save_renderings.save_exr(n, output_dirs, "n_256x256", data_type.Type.normal)
print(n.shape)
n = cv2.resize(n, dsize=(64, 64), interpolation=cv2.INTER_LANCZOS4)
print(n.shape)
save_renderings.save_png((n+1)*125, output_dirs, "n_64x64", mode="RGB")
save_renderings.save_exr(n, output_dirs, "n_64x64", data_type.Type.normal)
