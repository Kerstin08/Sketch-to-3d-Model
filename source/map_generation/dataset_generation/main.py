import source.util.mi_render as render
import source.util.line_gen as lineGen
import argparse
import os

def make_folder(prefix, dir):
    original_path_split = dir.rsplit("\\", 1)
    dir_name = prefix + original_path_split[1]
    path = os.path.join(original_path_split[0], dir_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def gen_images(path, sketch_output_dirs, aov_output_dirs, fov, modelname_split_indicator_before="", modelname_split_indicator_after=""):
    if os.path.isfile(path) and path.rsplit(".", 1)[1] == "obj":
        print(f'Processing ' + path, end='\r')
        if len(modelname_split_indicator_before) > 0 or len(modelname_split_indicator_after) > 0:
            ouput_name = path
            if len(modelname_split_indicator_before) > 0:
                ouput_name = ouput_name.rsplit(modelname_split_indicator_before, 1)[1]
            if len(modelname_split_indicator_after) > 0:
                ouput_name = ouput_name.rsplit(modelname_split_indicator_after, 1)[0]
            ouput_name = ouput_name.replace("\\", "_")
        else:
            ouput_name = path.rsplit("\\", 1)[1].rsplit(".obj")[0]
        # generate sketches
        lineGen.run("rendering", path, sketch_output_dirs, fov, 4, ouput_name)
        # generate depth and normal
        render.run("aov", path, aov_output_dirs, fov, {"dd.y": "depth", "nn": "sh_normal"}, 4, ouput_name)
        return
    for path, _, files in os.walk(path):
        for file in files:
            new_path = os.path.join(path, file)
            gen_images(new_path, sketch_output_dirs, aov_output_dirs, fov, modelname_split_indicator_before, modelname_split_indicator_after)

def run(input_dir, output_dir, fov, modelname_split_indicator_before="", modelname_split_indicator_after=""):
    # generate folders
    sketch_path = make_folder("sketch_", output_dir)
    n_path = make_folder("n_", output_dir)
    d_path = make_folder("d_", output_dir)
    sketch_output_dirs = {"rendering": sketch_path, "sketch": sketch_path}
    aov_output_dirs = {"dd.y": d_path, "nn": n_path}
    gen_images(input_dir, sketch_output_dirs, aov_output_dirs, fov, modelname_split_indicator_before, modelname_split_indicator_after)

    #for path in glob.glob(input_dir, recursive=False):
    #    print(path.rsplit(".", 1)[1])
    #    if path.rsplit(".", 1)[1] == "obj":
    #        sub_dirs_temp = path.rsplit(input_dir, 1)[1]
    #        sub_dirs = sub_dirs_temp.rsplit("\\", -1)[0]
    #        str.replace(sub_dirs, "\\", "_")
    #        # generate sketches
    #        lineGen.run("rendering", path, sketch_output_dirs, fov, 4, sub_dirs)
    #        # generate depth and normal
    #        render.run("aov", path, aov_output_dirs, fov, {"dd.y": "depth", "nn": "sh_normal"}, 4, sub_dirs)

def diff_args(args):
    run(args.input_dir, args.output_dir, args.fov, args.modelname_split_indicator_before, args.modelname_split_indicator_after)

def main(args):
    parser = argparse.ArgumentParser(prog="dataset_generation")
    parser.add_argument("--input_dir", type=str, help="path to reference objects")
    parser.add_argument("--output_dir", type=str, help="path to output objects")
    parser.add_argument("--fov", type=int, default=50, help="define rendering fov")
    parser.add_argument("--modelname_split_indicator_before", type=str, default="", help="part of path before the desired model name including \\\\")
    parser.add_argument("--modelname_split_indicator_after", type=str, default="", help="part of path after the desired model name including \\\\")
    args = parser.parse_args(args)
    diff_args(args)

if __name__ == '__main__':
    params = [
        '--input_dir', '..\\..\\..\\resources\\ABC',
        '--output_dir', '..\\..\\..\\output\\mapgen',
        #'--modelname_split_indicator_before', 'ShapeNetCore.v2\\',
        #'--modelname_split_indicator_after', '\\models',
    ]
    main(params)