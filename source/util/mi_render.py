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


def rendering(scene, mesh_name, output_dirs):
    img = mi.render(scene, seed=0, spp=1024)
    bitmap = mi.util.convert_to_bitmap(img)
    filename = mesh_name + "_rendering.png"
    output_dir = output_dirs["rendering"]
    path = os.path.join(output_dir, filename)
    mi.util.write_bitmap(path, bitmap)

def avo(scene, aovs, mesh_name, output_dirs):
    img = mi.render(scene, seed=0, spp=1024)
    bitmap = mi.Bitmap(img, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
    channels = dict(bitmap.split())
    if "depth" in aovs.values():
        depth = channels['dd.y']
        bitmap = mi.Bitmap(depth, channel_names=['R', 'G', 'B'])
        filename = mesh_name + "_depth.png"
        output_dir = output_dirs['dd.y']
        path = os.path.join(output_dir, filename)
        mi.util.write_bitmap(path, bitmap)
    if "sh_normal" in aovs.values():
        normal = channels['nn']
        filename = mesh_name + "_normal.png"
        output_dir = output_dirs['nn']
        path = os.path.join(output_dir, filename)
        mi.util.write_bitmap(path, normal)

def create_aov(aovs, shape, camera, mesh_name, output_dirs):
    integrator_aov = create_scenedesc.create_intergrator_aov(aovs)
    scene_desc = {"type": "scene", "shape": shape, "camera": camera, "integrator": integrator_aov}
    scene = mi.load_dict(scene_desc)
    return avo(scene, aovs, mesh_name, output_dirs)

def create_rendering(emitter_samples, shape, camera, mesh_name, output_dir):
    integrator_rendering = create_scenedesc.create_integrator_direct(emitter_samples)
    emitter = create_scenedesc.create_emitter()
    scene_desc = {"type": "scene", "shape": shape, "camera": camera, "integrator": integrator_rendering, "emitter": emitter}
    scene = mi.load_dict(scene_desc)
    rendering(scene, mesh_name, output_dir)

def run(type, input_mesh, input_metadata, output_dirs, fov, aovs=[], emitter_samples=0):
    with open(input_metadata, 'r', encoding='utf-8') as f:
        meta_data = json.loads(f.read())
    min_list = meta_data["min"]
    max_list = meta_data["max"]
    size = max(abs(max_list[0] - min_list[0]), abs(max_list[1] - min_list[1]), abs(max_list[2] - min_list[2]))
    distance = math.tan(math.radians(fov))*size
    centroid = np.array([0, distance, -distance])
    mesh_name = (input_mesh.rsplit("\\", 1)[-1]).rsplit(".", 1)[0]
    shape = create_scenedesc.create_shape(input_mesh, T.rotate([0, 1, 0], 45))
    camera = create_scenedesc.create_camera(T.look_at(target=(0, 0, 0),
                                                                   origin=tuple(centroid),
                                                                   up=(0, 1, 0),
                                                                   ),
                                            fov
                                            )
    for key, value in output_dirs.items():
        if not os.path.exists(value):
            os.mkdir(value)

    if type == "aov":
        create_aov(aovs, shape, camera, mesh_name, output_dirs)
    elif type == "rendering":
        create_rendering(emitter_samples, shape, camera, mesh_name, output_dirs)
    elif type == "combined":
        create_aov(aovs, shape, camera, mesh_name, output_dirs)
        create_rendering(emitter_samples, shape, camera, mesh_name, output_dirs)
    else:
        raise Exception("Given type not known!")

def diff_ars(args):
    run(args.type, args.input_mesh, args.input_metadata, args.output_dirs, args.fov, args.aovs, args.emitter_samples)

def main(args):
    parser = argparse.ArgumentParser(prog="scene_rendering")
    parser.add_argument("--type", type=str, help="use \"aov\", \"rendering\" or \"combined\"")
    parser.add_argument("--input_mesh", type=str)
    parser.add_argument("--input_metadata", type=str)
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
        '--input_mesh', '..\\..\\resources\\\ShapeNetCore.v2\\03207941\\69b0a23eb87e1c396694e76612a795a6\\models\\model_normalized.obj',
        '--input_metadata', '..\\..\\resources\\\ShapeNetCore.v2\\03207941\\69b0a23eb87e1c396694e76612a795a6\\models\\model_normalized.json',
        ]
    main(params)