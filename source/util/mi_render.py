import json
import math
import os.path
import numpy as np

import source.util.mi_create_scenedesc as create_scenedesc
import argparse
import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T
from mitsuba.scalar_rgb import Point3f as P
mi.set_variant('cuda_ad_rgb')


def rendering(scene, output_name, output_dirs):
    img = mi.render(scene, seed=0, spp=1024)
    bitmap = mi.util.convert_to_bitmap(img)
    filename = output_name + "_rendering.png"
    output_dir = output_dirs["rendering"]
    path = os.path.join(output_dir, filename)
    mi.util.write_bitmap(path, bitmap)

def avo(scene, aovs, output_name, output_dirs):
    img = mi.render(scene, seed=0, spp=1024)
    bitmap = mi.Bitmap(img, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
    channels = dict(bitmap.split())
    if "depth" in aovs.values():
        depth = channels['dd.y']
        bitmap = mi.Bitmap(depth, channel_names=['R', 'G', 'B'])
        filename = output_name + "_depth.png"
        output_dir = output_dirs['dd.y']
        path = os.path.join(output_dir, filename)
        mi.util.write_bitmap(path, bitmap)
    if "sh_normal" in aovs.values():
        normal = channels['nn']
        filename = output_name + "_normal.png"
        output_dir = output_dirs['nn']
        path = os.path.join(output_dir, filename)
        mi.util.write_bitmap(path, normal)

def create_aov(aovs, shape, camera, output_name, output_dirs):
    integrator_aov = create_scenedesc.create_intergrator_aov(aovs)
    scene_desc = {"type": "scene", "shape": shape, "camera": camera, "integrator": integrator_aov}
    scene = mi.load_dict(scene_desc)
    return avo(scene, aovs, output_name, output_dirs)

def create_rendering(emitter_samples, shape, camera, output_name, output_dir):
    integrator_rendering = create_scenedesc.create_integrator_direct(emitter_samples)
    emitter = create_scenedesc.create_emitter()
    scene_desc = {"type": "scene", "shape": shape, "camera": camera, "integrator": integrator_rendering, "emitter": emitter}
    scene = mi.load_dict(scene_desc)
    rendering(scene, output_name, output_dir)

def run(type, input_mesh, output_dirs, fov, aovs=[], emitter_samples=0, output_name=""):
    # Use 1 as size, since the diagonal of the bounding box of the normalized mesh should be 1
    # Do not use bounding box values from meta data, since not fullfill the cirteria of having a distance of 1,
    # leading to huge distances and therefore tiny renderings
    size = 1
    distance = math.tan(math.radians(fov))*size
    far_distance = math.tan(math.radians(fov))*size*2
    near_distance = far_distance-(size*1.35)
    centroid = np.array([0, distance, -distance])
    if len(output_name) <= 0:
        output_name = (input_mesh.rsplit("\\", 1)[-1]).rsplit(".", 1)[0]
    shape = create_scenedesc.create_shape(input_mesh, T.rotate([0, 1, 0], 45))
    camera = create_scenedesc.create_camera(T.look_at(target=-centroid,
                                                                   origin=tuple(centroid),
                                                                   up=(0, 1, 0),
                                                                   ),
                                            fov, near_distance, far_distance
                                            )
    for key, value in output_dirs.items():
        if not os.path.exists(value):
            os.mkdir(value)

    if type == "aov":
        create_aov(aovs, shape, camera, output_name, output_dirs)
    elif type == "rendering":
        create_rendering(emitter_samples, shape, camera, output_name, output_dirs)
    elif type == "combined":
        create_aov(aovs, shape, camera, output_name, output_dirs)
        create_rendering(emitter_samples, shape, camera, output_name, output_dirs)
    else:
        raise Exception("Given type not known!")

def diff_ars(args):
    run(args.type, args.input_mesh, args.output_dirs, args.fov, args.aovs, args.emitter_samples)

def main(args):
    parser = argparse.ArgumentParser(prog="scene_rendering")
    parser.add_argument("--type", type=str, help="use \"aov\", \"rendering\" or \"combined\"")
    parser.add_argument("--input_mesh", type=str)
    parser.add_argument("--output_dirs", type=dict, default={'nn': '..\\..\\output', 'dd.y': '..\\..\\output', 'rendering': '..\\..\\output'})
    parser.add_argument("--fov", type=int, default=50)
    parser.add_argument("--aovs", type=dir, default={"nn": "sh_normal", "dd.y": "depth"})
    parser.add_argument("--emitter_samples", type=int, default=4)
    args = parser.parse_args(args)
    diff_ars(args)

if __name__ == '__main__':
    output_dirs = {'nn': '..\\..\\output', 'dd.y': '..\\..\\output', 'rendering': '..\\..\\output'}
    params = [
        '--type', 'combined',
        '--input_mesh', '..\\..\\resources\\\ShapeNetCore.v2\\03636649\\1a3127ade9d7eca4fde8830b9596d8b9\\models\\model_normalized.obj',
        ]
    main(params)