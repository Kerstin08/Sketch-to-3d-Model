from source.util import mi_render as render
from source.util import line_gen as lineGen
from source.util import mesh_preprocess_operations as mesh_preprocess
import argparse
import os

def make_folder(prefix, dir):
    original_path_split = dir.rsplit("\\", 1)
    dir_name = prefix + original_path_split[1]
    path = os.path.join(original_path_split[0], dir_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def gen_images(path, datatype, sketch_output_dirs, aov_output_dirs,
               fov, create_debug_png, modelname_split_indicator_before="", modelname_split_indicator_after=""):
    if os.path.isfile(path) and path.rsplit(".", 1)[1] == datatype:
        # stl files cannot be processed by mitsuba
        if datatype == "stl":
            path = mesh_preprocess.preprocess(path)
            if not path:
                return
        print(f'Processing ' + path, end='\r')
        if len(modelname_split_indicator_before) > 0 or len(modelname_split_indicator_after) > 0:
            ouput_name = path
            if len(modelname_split_indicator_before) > 0:
                ouput_name = ouput_name.rsplit(modelname_split_indicator_before, 1)[1]
            if len(modelname_split_indicator_after) > 0:
                ouput_name = ouput_name.rsplit(modelname_split_indicator_after, 1)[0]
            ouput_name = ouput_name.replace("\\", "_")
        else:
            split_datatype = ".ply" if datatype == "stl" else "." + datatype
            ouput_name = path.rsplit("\\", 1)[1].rsplit(split_datatype)[0]
        # generate sketches
        lineGen.run("rendering", path, sketch_output_dirs, fov, 4, ouput_name)
       # # generate depth and normal
        render.run("aov", path, aov_output_dirs, fov, {"dd.y": "depth", "nn": "sh_normal"}, 4, ouput_name, create_debug_png=create_debug_png)
        return
    for path, _, files in os.walk(path):
        for file in files:
            new_path = os.path.join(path, file)
            gen_images(new_path, datatype, sketch_output_dirs, aov_output_dirs, fov, create_debug_png, modelname_split_indicator_before, modelname_split_indicator_after)

def run(input_dir, output_dir, datatype, fov, create_debug_png, modelname_split_indicator_before="", modelname_split_indicator_after=""):
    # generate folders
    sketch_path = make_folder("sketch_", output_dir)
    n_path = make_folder("n_", output_dir)
    d_path = make_folder("d_", output_dir)
    if(create_debug_png):
        n_png_path = make_folder("n_png_", output_dir)
        d_png_path = make_folder("d_png_", output_dir)
        aov_output_dirs = {"dd.y": d_path, "dd_png": d_png_path, "nn": n_path, "nn_png": n_png_path}
    else:
        aov_output_dirs = {"dd.y": d_path, "nn": n_path}
    sketch_output_dirs = {"rendering": sketch_path, "sketch": sketch_path}
    gen_images(input_dir, datatype, sketch_output_dirs, aov_output_dirs, fov, create_debug_png, modelname_split_indicator_before, modelname_split_indicator_after)

def diff_args(args):
    run(args.input_dir, args.output_dir, args.datatype, args.fov, args.create_debug_png, args.modelname_split_indicator_before, args.modelname_split_indicator_after)

def main(args):
    parser = argparse.ArgumentParser(prog="mapgen_dataset")
    parser.add_argument("--input_dir", type=str, help="path to reference objects")
    parser.add_argument("--output_dir", type=str, help="path to output objects")
    parser.add_argument("--datatype", type=str, default="stl", help="Object datatype")
    parser.add_argument("--fov", type=int, default=50, help="define rendering fov")
    parser.add_argument("--create_debug_png", type=bool, default=True, help="save pngs of aovs for easier debug")
    parser.add_argument("--modelname_split_indicator_before", type=str, default="", help="part of path before the desired model name including \\\\")
    parser.add_argument("--modelname_split_indicator_after", type=str, default="", help="part of path after the desired model name including \\\\")
    args = parser.parse_args(args)
    diff_args(args)

if __name__ == '__main__':
    params = [
        '--input_dir', '..\\..\\..\\resources\\thinig10k\\0_499',
        '--output_dir', '..\\..\\..\\output\\mapgen'
    ]
    main(params)