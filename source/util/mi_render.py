import math
import os.path
import numpy as np

import source.util.mi_create_scenedesc as create_scenedesc
import argparse
import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T
mi.set_variant('cuda_ad_rgb')


def rendering(scene, output_name, output_dirs):
    img = mi.render(scene, seed=0, spp=256)
    bitmap = mi.util.convert_to_bitmap(img)
    filename = output_name + "_rendering.png"
    output_dir = output_dirs["rendering"]
    path = os.path.join(output_dir, filename)
    mi.util.write_bitmap(path, bitmap)

def avo(scene, aovs, output_name, output_dirs):
    img = mi.render(scene, seed=0, spp=256)
    bitmap = mi.Bitmap(img, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
    channels = dict(bitmap.split())
    if "depth" in aovs.values():
        depth = channels['dd.y']
        filename = output_name + "_depth.exr"
        output_dir = output_dirs['dd.y']
        path = os.path.join(output_dir, filename)
        mi.util.write_bitmap(path, depth)
    if "sh_normal" in aovs.values():
        normal = channels['nn']
        filename = output_name + "_normal.exr"
        output_dir = output_dirs['nn']
        path = os.path.join(output_dir, filename)
        mi.util.write_bitmap(path, normal)

def create_aov(aovs, shape, camera, output_name, output_dirs):
    integrator_aov = create_scenedesc.create_intergrator_aov(aovs)
    scene_desc = {"type": "scene", "shape": shape, "camera": camera, "integrator": integrator_aov}
    # Sometimes mesh data is not incorrect and could not be loaded
    try:
        scene = mi.load_dict(scene_desc)
    except Exception as e:
        print("Exception occured in " + shape["filename"])
        print(e)
        return
    return avo(scene, aovs, output_name, output_dirs)

def create_rendering(emitter_samples, shape, camera, output_name, output_dir):
    integrator_rendering = create_scenedesc.create_integrator_direct(emitter_samples)
    emitter = create_scenedesc.create_emitter()
    scene_desc = {"type": "scene", "shape": shape, "camera": camera, "integrator": integrator_rendering, "emitter": emitter}
    # Sometimes mesh data is not incorrect and could not be loaded
    try:
        scene = mi.load_dict(scene_desc)
    except Exception as e:
        print("Exception occured in " + shape["filename"])
        print(e)
        return
    rendering(scene, output_name, output_dir)

def run(type, input_mesh, output_dirs, fov, aovs=[], emitter_samples=0, output_name=""):
    # Use 1 as size, since the diagonal of the bounding box of the normalized mesh should be 1
    # Do not use bounding box values from meta data, since not fullfill the cirteria of having a distance of 1,
    # leading to huge distances and therefore tiny renderings
    datatype = input_mesh.rsplit(".", 1)[1]
    if datatype != "obj" and datatype != "ply":
        print("Given datatype cannot be processed, must be either obj or ply type.")
        return
    shape = create_scenedesc.create_shape(input_mesh, datatype, T.scale(0.01))
    try:
        shape_lodaded = mi.load_dict(shape)
    except Exception as e:
        print("Exception occured in " + shape["filename"])
        print(e)
        return
    bounding_box = shape_lodaded.bbox()
    bounding_box_dim = abs(bounding_box.max - bounding_box.min)
    center = bounding_box.min + bounding_box_dim/2

    distance = center + math.tan(math.radians(fov)) * max(bounding_box_dim)
    far_distance = math.tan(math.radians(fov)) * max(bounding_box_dim) * 4
    near_distance = math.tan(math.radians(fov)) * max(bounding_box_dim)/4
    centroid = np.array([distance.x, distance.y, -distance.z])
    if len(output_name) <= 0:
        output_name = (input_mesh.rsplit("\\", 1)[-1]).rsplit(".", 1)[0]

    camera = create_scenedesc.create_camera(T.look_at(target=tuple(center),
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
    parser.add_argument("--aovs", type=dir, default={"dd.y": "depth", "nn": "sh_normal"})
    parser.add_argument("--emitter_samples", type=int, default=4)
    args = parser.parse_args(args)
    diff_ars(args)

if __name__ == '__main__':
    output_dirs = {'nn': '..\\..\\output', 'dd.y': '..\\..\\output', 'rendering': '..\\..\\output'}
    params = [
        '--type', 'aov',
        '--input_mesh', '..\\..\\resources\\meshes\\sphere.obj',
        ]
    main(params)