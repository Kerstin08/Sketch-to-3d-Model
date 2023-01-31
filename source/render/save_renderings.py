from PIL import Image
from pathlib import Path
import os

from source.util import data_type
from source.util import OpenEXR_utils
from source.util import dir_utils

def save_exr(img, output_dirs, output_name, given_data_type=None):
    if given_data_type == data_type.Type.depth:
        filename = output_name + "_depth.exr"
        output_dir = output_dirs['dd.y']
    elif given_data_type == data_type.Type.silhouette:
        filename = output_name + "_silhouette.exr"
        output_dir = output_dirs['default']
    elif given_data_type == data_type.Type.normal:
        filename = output_name + "_normal.exr"
        output_dir = output_dirs['nn']
    else:
        filename = output_name + ".exr"
        output_dir = output_dirs['default']
    path = os.path.join(output_dir, filename)
    OpenEXR_utils.writeImage(img, given_data_type, path)

# Todo: check if filenamedir is necessary
def save_png(img, output_dirs, output_name, given_data_type=None, dir_key = 'default', mode='RGB', filename_dir=None):
    if given_data_type == data_type.Type.depth:
        img = img * 255
        output_dir_png = output_dirs['dd_png']
        if not filename_dir is None:
            output_dir_png = os.path.join(output_dir_png, filename_dir)
            dir_utils.create_general_folder(output_dir_png)
        png_filename = output_name + "_depth.png"
        path_debug = os.path.join(output_dir_png, png_filename)
        # Use pil instead of standard mi.util.write_bitmap for png since mi automatically applies gamma correction when
        # writing png files
        Image.fromarray(img.astype('uint8'), mode='L').save(path_debug)
    elif given_data_type == data_type.Type.normal:
        img = (img + 1.0) * 127
        output_dir_png = output_dirs['nn_png']
        if not filename_dir is None:
            output_dir_png = os.path.join(output_dir_png, filename_dir)
            dir_utils.create_general_folder(output_dir_png)
        png_filename = output_name + "_normal.png"
        path_debug = os.path.join(output_dir_png, png_filename)
        # Use pil instead of standard mi.util.write_bitmap for png since mi automatically applies gamma correction when
        # writing png files
        Image.fromarray(img.astype('uint8'), mode='RGB').save(path_debug)
    elif given_data_type == data_type.Type.sketch:
        output_dir_png = output_dirs['sketch']
        if not filename_dir is None:
            output_dir_png = os.path.join(output_dir_png, filename_dir)
            dir_utils.create_general_folder(output_dir_png)
        png_filename = output_name + "_sketch.png"
        output_dir = os.path.join(output_dir_png, png_filename)
        im = Image.fromarray(img)
        im.save(output_dir)
    else:
        filename = Path(output_name)
        if not filename.suffix == ".png":
            output_name = filename.stem + ".png"
        output_dir_png = output_dirs[dir_key]
        if not filename_dir is None:
            output_dir_png = os.path.join(output_dir_png, filename_dir)
            dir_utils.create_general_folder(output_dir_png)
        output_dir = os.path.join(output_dir_png, output_name)
        Image.fromarray(img.astype('uint8'), mode=mode).save(output_dir)

