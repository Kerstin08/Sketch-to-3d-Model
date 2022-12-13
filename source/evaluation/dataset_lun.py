import argparse
import math
import sys
from pathlib import Path
import os
import numpy as np

from source.render import save_renderings
from source.util import dir_utils
from source.util import mesh_preprocess_operations as mesh_preprocess
from source.render.line_generation import LineGen
from source.render.render_aov import AOV

# For windows + conda here certain combinations of python, numpy and trimesh versions can cause conflicts in loading libraries
# Allowing duplicate loading of libraries is the most common suggested and fixes those issues:
# https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
# Furthermore, when using e.g. PyCharm to run the code, this causes the program to not stop, which is why an abort function is included
# Code for abort function (not the cleanest solution, but it aborts the program):
# https://stackoverflow.com/questions/15457786/ctrl-c-crashes-python-after-importing-scipy-stats
if sys.platform == 'win32':
    import _thread
    import win32api

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


    def handler(dwCtrlType, hook_sigint=_thread.interrupt_main):
        if dwCtrlType == 0:  # CTRL_C_EVENT
            hook_sigint()
            return 1  # don't chain to the next handler
        return 0  # chain to the next handler


    win32api.SetConsoleCtrlHandler(handler, 1)

def write_view_file(view_path, fov, views):
    view_file=os.path.join(view_path, "view.off")
    with open(view_file, 'w') as f:
        f.write("OFF\n")
        # views, groups, and something else
        f.write("14 24 0\n")

        # views
        distance = math.tan(math.radians(fov)) / 1.75
        radius = math.sqrt(3 * math.pow(distance, 2))
        for view in views:
            long, lat = view
            long = math.pi * long / 180
            lat = math.pi * lat / 180
            x = round(math.cos(long) * -math.cos(lat) * radius, 5)
            y = round(math.sin(long) * math.cos(lat) * radius, 5)
            z = round(math.sin(lat) * radius, 5)
            f.write(str(x) + " " + str(y) + " " + str(z) + "\n")
        # Use same groups as Lun
        f.write("3 0 2 6\n \
3 0 6 3 \n \
3 0 3 8 \n \
3 0 8 2 \n \
3 5 7 4 \n \
3 4 9 5 \n \
3 7 12 1 \n \
3 1 12 6 \n \
3 6 10 1 \n \
3 1 10 7 \n \
3 9 11 8 \n \
3 8 13 9 \n \
3 4 10 11 \n \
3 2 11 10 \n \
3 13 3 12 \n \
3 12 5 13 \n \
3 10 6 2 \n \
3 8 11 2 \n \
3 6 12 3 \n \
3 13 8 3 \n \
3 7 10 4 \n \
3 11 9 4 \n \
3 12 7 5 \n \
3 9 13 5")


def gen_images(path, datatype, renderer_aov, line_gen, rendering_dirs, sketch_dirs, create_debug_png):
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
        line_scenes = line_gen.create_scenes(path)
        aov_scenes = renderer_aov.create_scene(path)
        count = 0
        for i in range(len(line_scenes)):
            counted_out_name = str(count) + "_" + output_name

            aov_scene = aov_scenes[i]
            # generate depth and normal
            normal = renderer_aov.render_normal(aov_scene, path)
            depth = renderer_aov.render_depth(aov_scene, path)
            # Todo: activate this once png works
            if not normal is None and not depth is None:
                normal = (normal + 1.0) * 127
                depth = np.expand_dims(depth * 255, axis=2)
                normal = np.where(depth > 240, 255, normal)
                # save_renderings.save_exr(depth, rendering_dirs,  output_name, data_type.Type.depth)
                # save_renderings.save_exr(normal, rendering_dirs, output_name, data_type.Type.normal)
                stack = np.concatenate([normal, depth], axis=2)
                dn_name = "dn_"+counted_out_name
                save_renderings.save_png(stack, rendering_dirs, dn_name, dir_key='dn', mode='RGBA')
                # Todo: dnfs and sketch need front view
                # save first two views as dnfs and sketch
                if count < 2:
                    dn_name = "dnfs_" + counted_out_name
                    save_renderings.save_png(stack, rendering_dirs, dn_name, dir_key='dnfs', mode='RGBA')

                    line_scene = line_scenes[i]
                    lines = line_gen.create_line_images(line_scene, path)
                    stack_lines = np.stack([lines, lines, lines], axis=-1, out=None)
                    if count < 1:
                        outname = "sketch_T_" +counted_out_name
                    else:
                        outname = "sketch_S_" + counted_out_name
                    save_renderings.save_png(stack_lines, sketch_dirs, outname, dir_key='sketch', mode='RGB')
                    save_renderings.save_png(stack_lines, sketch_dirs, outname, dir_key='hires', mode='RGB')
                count = count + 1
        return
    for path, _, files in os.walk(path):
        for file in files:
            new_path = os.path.join(path, file)
            gen_images(new_path, datatype, renderer_aov, line_gen, rendering_dirs, sketch_dirs, create_debug_png)


def run(input_dir, output_dir, datatype, fov, dim_render, dim_line_gen_intermediate, emitter_samples, create_debug_png):
    if not os.path.exists(input_dir):
        raise Exception("Input directory {} does not exits".format(input_dir))

    # generate folders
    if not os.path.exists(output_dir):
        dir_utils.create_general_folder(output_dir)
    dn = dir_utils.create_general_folder(os.path.join(output_dir, "dn"))
    dnfs_path = dir_utils.create_general_folder(os.path.join(output_dir, "dnfs"))
    sketch_path = dir_utils.create_general_folder(os.path.join(output_dir, "sketch"))
    view_path = dir_utils.create_general_folder(os.path.join(output_dir, "view"))
    hires_path = dir_utils.create_general_folder(os.path.join(output_dir, "hires"))

    rendering_dirs = {"dn": dn, "dnfs": dnfs_path}
    sketch_dirs = {"sketch": sketch_path, "hires": hires_path}
    # Angles used by Lun et. al
    views = [(90, 90),
            (0, 0),

            (90, 58),
            (-90, 58),
            (90, -58),
            (-90, -58),

            (0, 52),
            (0, -52),
            (180, 52),
            (180, -52),

            (58, 0),
            (122, 0),
            (-58, 0),
            (-122, 0)]
    write_view_file(view_path, fov, views)
    renderer_aov = AOV(views, {"dd.y": "depth", "nn": "sh_normal"}, fov, dim_render)
    line_gen = LineGen(views, fov, dim_line_gen_intermediate, dim_render, emitter_samples)
    gen_images(input_dir, datatype, renderer_aov, line_gen, rendering_dirs, sketch_dirs, create_debug_png)

def diff_args(args):
    run(args.input_dir,
        args.output_dir,
        args.datatype,
        args.fov,
        args.dim_render,
        args.dim_line_gen_intermediate,
        args.emitter_samples,
        args.create_debug_png)


def main(args):
    parser = argparse.ArgumentParser(prog="map_generation_dataset")
    parser.add_argument("--input_dir", type=str, help="path to reference objects")
    parser.add_argument("--output_dir", type=str, help="path to output objects")
    parser.add_argument("--datatype", type=str, default=".ply", help="Object datatype")
    parser.add_argument("--fov", type=int, default=50, help="define rendering fov")
    parser.add_argument("--dim_render", type=int, default=256, help="final output format for images")
    parser.add_argument("--dim_line_gen_intermediate", type=int, default=1024,
                        help="intermediate output format for rendered images to perform line detection on")
    parser.add_argument("--emitter_samples", type=int, default=4, help="# of emitter samples for direct rendering")
    parser.add_argument("--create_debug_png", type=bool, default=True, help="save pngs of aovs for easier debug")
    args = parser.parse_args(args)
    diff_args(args)


if __name__ == '__main__':
    params = [
        '--input_dir', '..\\..\\resources\\test_lun',
        '--output_dir', '..\\..\\output\\lun'
    ]
    main(params)
