import argparse
import sys
import numpy as np

from source.render.render_direct import Direct
from source.render.render_aov import AOV
from source.render.line_generation import LineGen
from source.render import save_renderings
from source.util import dir_utils
from source.util import parse


def run(render_type, line_gen, input_path, output_dir, output_name, views):
    dir_utils.create_general_folder(output_dir)
    output_dirs = {'default': output_dir}

    if render_type == 'aov' or render_type == 'combined':
        renders_aov = AOV(views, dim=256)
        scenes_aov = renders_aov.create_scene(input_path)
        count = 0
        for scene in scenes_aov:
            depth = np.array(renders_aov.render_depth(scene, input_path, spp=8))
            normal = np.array(renders_aov.render_normal(scene, input_path, spp=8))
            silhouette = np.array(renders_aov.render_silhouette(scene, input_path, spp=8))
            save_renderings.save_png(depth * 255, output_dirs, str(count) + '_depth_' + output_name, mode='L')
            save_renderings.save_png((normal + 1.0) * 127, output_dirs, str(count) + '_normal_' + output_name)
            save_renderings.save_png(silhouette * 255, output_dirs, str(count) + '_silhouette_' + output_name)
            count = count + 1

    if render_type == 'rendering' or render_type == 'combined':
        renders_direct = Direct(views)
        scenes_direct = renders_direct.create_scene(input_path)
        count = 0
        for scene in scenes_direct:
            direct = np.array(renders_direct.render(scene, input_path))
            direct = (direct * 255).astype(np.uint8)
            save_renderings.save_png(direct, output_dirs, str(count) + '_direct_' + output_name)
            count = count + 1

    # Used to render input images for comparison evaluation
    if render_type == 'kato' or render_type == 'combined':
        # use different values for neural mesh renderer
        # use bool in renderer for now, for more options extend rendering files
        renders_direct = Direct(views, dim=64, emitter_samples=1, nmr=True)
        scenes_direct = renders_direct.create_scene(input_path)
        renders_aov = AOV(views, dim=64)
        scenes_aov = renders_aov.create_scene(input_path)[0]
        count = 0
        for scene in scenes_direct:
            direct = np.array(renders_direct.render(scene, input_path))
            silhouette = 1 - np.expand_dims(np.array(renders_aov.render_silhouette(scenes_aov, input_path))[:, :, 0], 2)
            alpha = np.concatenate([direct, silhouette], 2)
            output = alpha * 255
            save_renderings.save_png(output.astype('uint8'), output_dirs,  str(count) + '_' + output_name, mode='RGBA')
            count = count + 1

    if line_gen:
        line_gen = LineGen(views)
        scenes = line_gen.create_scenes(input_path)
        count = 0
        for scene in scenes:
            img = line_gen.create_line_images(scene, input_path)
            save_renderings.save_png(img, output_dirs, str(count) + '_sketch_' + output_name, mode='L')
            count = count + 1


def diff_ars(args):
    run(args.render_type, args.line_gen, args.input_path, args.output_dir, args.output_name, args.views)


def main(args):
    parser = argparse.ArgumentParser(prog="scene_rendering")
    parser.add_argument("--render_type", type=str, default="combined",
                        help="use \"aov\", \"rendering\", \"kato\", or \"combined\"")
    parser.add_argument("--line_gen", type=parse.bool, default="True", dest="line_gen",
                        help="if sketch should be generated; use \"True\" or \"False\" as parameter")
    parser.add_argument("--input_path", type=str, default="..\\..\\resources\\thingi10k\\0_499\\32770.ply",
                        help="path to input model")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory where the output is stored")
    parser.add_argument("--output_name", type=str, default="test", help="Name of output images")
    parser.add_argument("--view", type=parse.views, default="225, 30", dest="view",
                        help="define rendering view angles; string with tuples of azimuth and elveation "
                             "e.g. \"0, 30, 255, 30\"")
    args = parser.parse_args(args)
    diff_ars(args)


if __name__ == '__main__':
    main(sys.argv[1:])
