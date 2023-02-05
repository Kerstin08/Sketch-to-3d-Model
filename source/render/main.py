import argparse
import os
import numpy as np

from source.render.render_direct import Direct
from source.render.render_aov import AOV
from source.render.line_generation import LineGen
from source.render import save_renderings
from source.util import data_type
from source.util import bool_parse

def run(render_type, line_gen_str, input_path, output_dirs, output_name, views):
    for key, value in output_dirs.items():
        if not os.path.exists(value):
            os.mkdir(value)

    line_gen = bool_parse.parse(line_gen_str)
    if render_type == "aov" or render_type == "combined":
        renders_aov = AOV(views, dim=64)
        scenes_aov = renders_aov.create_scene(input_path)
        count = 0
        for scene in scenes_aov:
            depth = np.array(renders_aov.render_depth(scene, input_path, spp=8))
            normal = np.array(renders_aov.render_normal(scene, input_path, spp=8))
            silhouette = np.array(renders_aov.render_silhouette(scene, input_path, spp=8))
            save_renderings.save_png(depth, output_dirs, str(count) + "_" + output_name, data_type.Type.depth)
            save_renderings.save_png(normal, output_dirs, str(count) + "_" + output_name, data_type.Type.normal)
            save_renderings.save_png(silhouette*255, output_dirs, str(count) + "_" + output_name)
            count = count + 1

    if render_type == "rendering" or render_type == "combined":
        renders_direct = Direct(views)
        scenes_direct = renders_direct.create_scene(input_path)
        count = 0
        for scene in scenes_direct:
            direct = np.array(renders_direct.render(scene, input_path))
            direct = (direct*255).astype(np.uint8)
            save_renderings.save_png(direct, output_dirs, str(count) + "_" + output_name)
            count = count + 1

    # Used to render input images for comparison evaluation
    if render_type == "kato":
        renders_direct = Direct(views, dim=64, emitter_samples=1)
        scenes_direct = renders_direct.create_scene(input_path)
        renders_aov = AOV(views, dim=64)
        scenes_aov = renders_aov.create_scene(input_path)[0]
        count = 0
        for scene in scenes_direct:
            direct = np.array(renders_direct.render(scene, input_path))
            silhouette = 1-np.expand_dims(np.array(renders_aov.render_silhouette(scenes_aov, input_path))[:,:,0], 2)
            alpha = np.concatenate([direct, silhouette], 2)
            output = alpha*255
            save_renderings.save_png(output.astype("uint8"), output_dirs, output_name, mode='RGBA')
            count = count + 1

    if line_gen:
        line_gen = LineGen(views)
        scenes = line_gen.create_scenes(input_path)
        count = 0
        for scene in scenes:
            img = line_gen.create_line_images(scene, input_path)
            save_renderings.save_png(img, output_dirs, str(count) + "256x256_" + output_name, data_type.Type.sketch)
            count = count + 1

def diff_ars(args):
    run(args.render_type, args.line_gen, args.input_path, args.output_dirs, args.output_name, args.views)

def main(args):
    parser = argparse.ArgumentParser(prog="scene_rendering")
    parser.add_argument("--render_type", type=str, default='combined', help="use \"aov\", \"rendering\" or \"combined\"")
    parser.add_argument("--line_gen", type=str, default="True", help="if sketch should be generated; use \"True\" or \"False\" as parameter")
    parser.add_argument("--input_path", default='..\\..\\resources\\thingi10k\\0_499\\32770.ply', type=str, help="path to input model")
    parser.add_argument("--output_dirs", type=dict, default={'nn': '..\\..\\output', 'dd.y': '..\\..\\output', "dd_png": '..\\..\\output', "nn_png": '..\\..\\output', 'default': '..\\..\\output', 'sketch': '..\\..\\output'}, help="directories to save renderings")
    parser.add_argument("--output_name", type=str, default="output_8", help="Name of output images")
    parser.add_argument("--views", type=list, default=[(225, 30)], help="define rendering view angles")
    args = parser.parse_args(args)
    diff_ars(args)

if __name__ == '__main__':
    output_dirs = {'nn': '..\\..\\output', 'dd.y': '..\\..\\output', "dd_png": '..\\..\\output', "nn_png": '..\\..\\output', 'default': '..\\..\\output', 'sketch': '..\\..\\output'}
    params = [
        '--render_type', 'aov',
        '--input_path', r'C:\Users\Kerstin\Documents\MasterThesis\masterthesis_hofer_kerstin\resources\eval\Comparison\meshes\02691156_e09c32b947e33f619ba010ddb4974fe.ply',
        ]
    main(params)