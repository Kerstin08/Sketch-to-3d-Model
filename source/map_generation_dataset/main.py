import argparse
import os
from pathlib import Path
import sys
import numpy as np

from source.render import save_renderings
from source.render.line_generation import LineGen
from source.render.render_aov import AOV
from source.util import mesh_preprocess_operations as mesh_preprocess
from source.util import dir_utils
from source.util import data_type

# For windows + conda here certain combinations of python, numpy and trimesh versions can cause conflicts in loading libraries
# Allowing duplicate loading of libraries is the most common suggested and fixes those issues:
# https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
# Furthermore, when using e.g. PyCharm to run the code, this causes the program to not stop, which is why an abort function is included
# Code for abort function (not the cleanest solution, but it aborts the program):
# https://stackoverflow.com/questions/15457786/ctrl-c-crashes-python-after-importing-scipy-stats
if sys.platform == 'win32':
    import _thread
    import win32api

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    def handler(dwCtrlType, hook_sigint = _thread.interrupt_main):
        if dwCtrlType == 0:  # CTRL_C_EVENT
            hook_sigint()
            return 1  # don't chain to the next handler
        return 0  # chain to the next handler
    win32api.SetConsoleCtrlHandler(handler, 1)


# Manually delete returned variables in order to get memory problem (leak or fragmentation)
# somewhat under control which seems to can occur if code is run on linux OS
# Seems to stabilize memory consumption, however there may be better solutions for that problem
def gen_images(path, datatype, renderer_aov, line_gen, output_dirs, create_debug_png):
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
        line_scene = line_gen.create_scenes(path)[0]
        lines = line_gen.create_line_images(line_scene, path)
        del line_scene
        if lines is None:
            return
        save_renderings.save_png(lines, output_dirs, output_name, data_type.Type.sketch)
        del lines
        # generate depth and normal
        scene_aov = renderer_aov.create_scene(path)[0]
        normal = np.array(renderer_aov.render_normal(scene_aov, path))
        depth = np.array(renderer_aov.render_depth(scene_aov, path))
        del scene_aov
        if normal is None or depth is None:
            return
        save_renderings.save_exr(depth, output_dirs,  output_name, data_type.Type.depth)
        save_renderings.save_exr(normal, output_dirs, output_name, data_type.Type.normal)
        if create_debug_png:
            save_renderings.save_png(depth, output_dirs, output_name, data_type.Type.depth)
            save_renderings.save_png(normal, output_dirs, output_name, data_type.Type.normal)
        del normal
        del depth
        return
    for path, _, files in os.walk(path):
        for file in files:
            new_path = os.path.join(path, file)
            gen_images(new_path, datatype, renderer_aov, line_gen, output_dirs, create_debug_png)

def run(input_dir, output_dir, datatype, fov, view, dim_render, dim_line_gen_intermediate, emitter_samples, create_debug_png):
    if not os.path.exists(input_dir):
        raise Exception("Input directory {} does not exits".format(input_dir))
    if len(view) > 1 or len(view) < 1:
        raise Exception("Exactly 1 view required for this dataset generation!")
    # generate folders
    sketch_path = dir_utils.create_prefix_folder("sketch_", output_dir)
    n_path = dir_utils.create_prefix_folder("n_", output_dir)
    d_path = dir_utils.create_prefix_folder("d_", output_dir)
    output_dirs = {"dd.y": d_path, "nn": n_path, "sketch": sketch_path}
    if(create_debug_png):
        n_png_path = dir_utils.create_prefix_folder("n_png_", output_dir)
        d_png_path = dir_utils.create_prefix_folder("d_png_", output_dir)
        output_dirs["dd_png"] = d_png_path
        output_dirs["nn_png"] = n_png_path

    renderer_aov = AOV(view, {"dd.y": "depth", "nn": "sh_normal"}, fov, dim_render)
    line_gen = LineGen(view, fov, dim_line_gen_intermediate, dim_render, emitter_samples)
    gen_images(input_dir, datatype, renderer_aov, line_gen, output_dirs, create_debug_png)

def diff_args(args):
    run(args.input_dir,
        args.output_dir,
        args.datatype,
        args.fov,
        args.view,
        args.dim_render,
        args.dim_line_gen_intermediate,
        args.emitter_samples,
        args.create_debug_png)

def main(args):
    parser = argparse.ArgumentParser(prog="map_generation_dataset")
    parser.add_argument("--input_dir", type=str, help="path to reference objects")
    parser.add_argument("--output_dir", type=str, help="path to output objects")
    parser.add_argument("--datatype", type=str, default=".stl", help="Object datatype")
    parser.add_argument("--fov", type=int, default=50, help="define rendering fov")
    parser.add_argument("--view", type=list, default=[(225, 30)], help="define rendering view angles")
    parser.add_argument("--dim_render", type=int, default=256, help="final output format for images")
    parser.add_argument("--dim_line_gen_intermediate", type=int, default=1024, help="intermediate output format for rendered images to perform line detection on")
    parser.add_argument("--emitter_samples", type=int, default=4, help="# of emitter samples for direct rendering")
    parser.add_argument("--create_debug_png", type=bool, default=True, help="save pngs of aovs for easier debug")
    args = parser.parse_args(args)
    diff_args(args)

if __name__ == '__main__':
    params = [
        '--input_dir', 'datasets/thingy',
        '--output_dir', 'map_generation'
    ]
    main(params)
