import source.util.mi_render as render
import source.util.line_gen as lineGen
import argparse
import os
import shutil

def make_folder(prefix, input_dir):
    original_path_split = input_dir.rsplit("\\", 1)
    dir_name = prefix + original_path_split[1]
    path = os.path.join(original_path_split[0], dir_name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path

def run(input_dir):
    # generate folders
    sketch_path = make_folder("sketch_", input_dir)
    n_path = make_folder("n_", input_dir)
    d_path = make_folder("d_", input_dir)

    for root, _, fnames in sorted(os.walk(input_dir)):
        for fname in fnames:
            if fname.rsplit(".", 1)[1] == "ply":
                model_path = os.path.join(root, fname)
                # generate sketches
                lineGen.run("rendering", model_path, sketch_path, 4)
                # Todo: use list of output dirs insted of single dir, since when rendering aovs apparently when only depth is given this is not rendered
                # generate depth
                render.run("aov", model_path, d_path, {"dd.y": "depth"}, 4)
                # generate normals
                render.run("aov", model_path, n_path, {"nn": "sh_normal"}, 4)

def diff_args(args):
    run(args.input_dir)

def main(args):
    parser = argparse.ArgumentParser(prog="dataset_generation")
    parser.add_argument("--input_dir", type=str, help="path to reference objects")
    args = parser.parse_args(args)
    diff_args(args)

if __name__ == '__main__':
    params = [
        '--input_dir', '..\\..\\..\\resources\\meshes',
    ]
    main(params)