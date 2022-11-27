from source.util import mi_render as render
from source.util import line_generation
from source.util import mesh_preprocess_operations as mesh_preprocess
from source.util import dir_utils
import argparse
import os
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def gen_images(path, datatype, sketch_output_dirs, aov_output_dirs,
               fov, create_debug_png):
    filename = Path(path)
    if os.path.isfile(path) and filename.suffix == datatype:
        # stl files cannot be processed by mitsuba
        if datatype == ".stl":
            path = mesh_preprocess.preprocess(path)
            if not path:
                return
            else:
                filename = Path(path)
        print('\r' + 'Processing ' + path, end='')
        output_name = filename.stem
        # generate sketches
        # line_generation.run("rendering", path, sketch_output_dirs, fov, 4, output_name)
        # generate depth and normal
        render.run("aov", path, aov_output_dirs, fov, {"dd.y": "depth", "nn": "sh_normal"}, 4, output_name, create_debug_png=create_debug_png)
        return
    for path, _, files in os.walk(path):
        for file in files:
            new_path = os.path.join(path, file)
            gen_images(new_path, datatype, sketch_output_dirs, aov_output_dirs, fov, create_debug_png)

def run(input_dir, output_dir, datatype, fov, create_debug_png):
    if not os.path.exists(input_dir):
        raise Exception("Input directory {} does not exits".format(input_dir))
    # generate folders
    sketch_path = dir_utils.create_prefix_folder("sketch_", output_dir)
    n_path = dir_utils.create_prefix_folder("n_", output_dir)
    d_path = dir_utils.create_prefix_folder("d_", output_dir)
    if(create_debug_png):
        n_png_path = dir_utils.create_prefix_folder("n_png_", output_dir)
        d_png_path = dir_utils.create_prefix_folder("d_png_", output_dir)
        aov_output_dirs = {"dd.y": d_path, "dd_png": d_png_path, "nn": n_path, "nn_png": n_png_path}
    else:
        aov_output_dirs = {"dd.y": d_path, "nn": n_path}
    sketch_output_dirs = {"rendering": sketch_path, "sketch": sketch_path}
    gen_images(input_dir, datatype, sketch_output_dirs, aov_output_dirs, fov, create_debug_png)

def diff_args(args):
    run(args.input_dir, args.output_dir, args.datatype, args.fov, args.create_debug_png)

def main(args):
    parser = argparse.ArgumentParser(prog="map_generation_dataset")
    parser.add_argument("--input_dir", type=str, help="path to reference objects")
    parser.add_argument("--output_dir", type=str, help="path to output objects")
    parser.add_argument("--datatype", type=str, default=".stl", help="Object datatype")
    parser.add_argument("--fov", type=int, default=50, help="define rendering fov")
    parser.add_argument("--create_debug_png", type=bool, default=True, help="save pngs of aovs for easier debug")
    args = parser.parse_args(args)
    diff_args(args)

if __name__ == '__main__':
    params = [
        '--input_dir', '..\\..\\resources\\thingi10k',
        '--output_dir', '..\\..\\output\\map_generation'
    ]
    main(params)