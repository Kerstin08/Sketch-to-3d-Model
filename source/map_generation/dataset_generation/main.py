import source.util.mi_render as render
import source.util.line_gen as lineGen
import argparse
import os
import shutil

def make_folder(prefix, dir):
    original_path_split = dir.rsplit("\\", 1)
    dir_name = prefix + original_path_split[1]
    path = os.path.join(original_path_split[0], dir_name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path

def run(input_dir, output_dir):
    # generate folders
    sketch_path = make_folder("sketch_", input_dir)
    n_path = make_folder("n_", output_dir)
    d_path = make_folder("d_", output_dir)
    sketch_output_dirs = {"rendering": sketch_path, "sketch": sketch_path}
    aov_output_dirs = {"dd.y": d_path, "nn": n_path}

    for root, _, fnames in sorted(os.walk(input_dir)):
        for fname in fnames:
            if fname.rsplit(".", 1)[1] == "ply":
                model_path = os.path.join(root, fname)
                # generate sketches
                lineGen.run("rendering", model_path, sketch_output_dirs, 4)
                # generate depth and normal
                render.run("aov", model_path, aov_output_dirs, {"dd.y": "depth", "nn": "sh_normal"}, 4)

def diff_args(args):
    run(args.input_dir, args.output_dir)

def main(args):
    parser = argparse.ArgumentParser(prog="dataset_generation")
    parser.add_argument("--input_dir", type=str, help="path to reference objects")
    parser.add_argument("--output_dir", type=str, help="path to output objects")
    args = parser.parse_args(args)
    diff_args(args)

if __name__ == '__main__':
    params = [
        '--input_dir', '..\\..\\..\\resources\\meshes',
        '--output_dir', '..\\..\\..\\output\\mapgen',
    ]
    main(params)