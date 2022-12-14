import argparse
import os
import numpy as np

from source.render.render_direct import Direct
from source.render.render_aov import AOV
from source.render.line_generation import LineGen
from source.render import save_renderings
from source.util import data_type


def run(render_type, line_gen, input_path, output_dirs, output_name, aovs=[]):
    for key, value in output_dirs.items():
        if not os.path.exists(value):
            os.mkdir(value)

    views = [(225, 35), (45, 35), (135, 35), (315, 35), (0, 90), (90, 0)]
    if render_type == "aov" or render_type == "combined":
        renders_aov = AOV(views, aovs)
        scenes_aov = renders_aov.create_scene(input_path)
        count = 0
        for scene in scenes_aov:
            depth = renders_aov.render_depth(scene, input_path)
            normal = renders_aov.render_normal(scene, input_path)
            save_renderings.save_png(depth, output_dirs, str(count) + "_" + output_name, data_type.Type.depth)
            save_renderings.save_png(normal, output_dirs, str(count) + "_" + output_name, data_type.Type.normal)
            count = count + 1
    if render_type == "rendering" or render_type == "combined":
        renders_direct = Direct(views)
        scenes_direct = renders_direct.create_scene(input_path)
        count = 0
        for scene in scenes_direct:
            direct = renders_direct.render(scene, input_path)
            direct = (direct*255).astype(np.uint8)
            save_renderings.save_png(direct, output_dirs, str(count) + "_" + output_name)
            count = count + 1


    if line_gen:
        line_gen = LineGen(views)
        scenes = line_gen.create_scenes(input_path)
        count = 0
        for scene in scenes:
            img = line_gen.create_line_images(scene, input_path)
            save_renderings.save_png(img, output_dirs, str(count) + "_" + output_name, data_type.Type.sketch)
            count = count + 1

def diff_ars(args):
    run(args.render_type, args.line_gen, args.input_path, args.output_dirs, args.output_name, args.aovs)

def main(args):
    parser = argparse.ArgumentParser(prog="scene_rendering")
    parser.add_argument("--render_type", type=str, default='combined', help="use \"aov\", \"rendering\" or \"combined\"")
    parser.add_argument("--line_gen", type=bool, default=True, help="if sketch should be generated")
    parser.add_argument("--input_path", default='..\\..\\resources\\thingi10k\\0_499\\32770.ply', type=str, help="path to input model")
    parser.add_argument("--output_dirs", type=dict, default={'nn': '..\\..\\output', 'dd.y': '..\\..\\output', "dd_png": '..\\..\\output', "nn_png": '..\\..\\output', 'default': '..\\..\\output', 'sketch': '..\\..\\output'}, help="directories to save renderings")
    parser.add_argument("--output_name", type=str, default="test", help="Name of output images")
    parser.add_argument("--aovs", type=dir, default={"nn": "sh_normal", "dd.y": "depth"}, help="Types of AOV; use \"sh_normal\" or \"depth\"")
    args = parser.parse_args(args)
    diff_ars(args)

if __name__ == '__main__':
    output_dirs = {'nn': '..\\..\\output', 'dd.y': '..\\..\\output', "dd_png": '..\\..\\output', "nn_png": '..\\..\\output', 'default': '..\\..\\output', 'sketch': '..\\..\\output'}
    params = [
        '--render_type', 'combined',
        '--input_path', '..\\..\\resources\\ABC\\abc_0099_stl2_v00\\0_499\\00990027\\00990027_523151fa223eff350f891bee_trimesh_000.ply',
        ]
    main(params)