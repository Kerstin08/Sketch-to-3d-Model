import math
import os.path
import numpy as np

import source.util.mi_create_scenedesc as create_scenedesc
import drjit as dr
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

def avo(scene, input_path, aovs, output_name, output_dirs, create_debug_pngs=True):
    img = mi.render(scene, seed=0, spp=256)
    # If mesh has invalid mesh vertices, rendering contains nan.
    if dr.any(dr.isnan(img)):
        print("Rendered image " + output_name + " includes invalid data! Vertex normals in input model " + input_path + " might be corrupt.")
        return
    bitmap = mi.Bitmap(img, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
    channels = dict(bitmap.split())
    if "depth" in aovs.values():
        depth = channels['dd.y']
        filename = output_name + "_depth.exr"
        output_dir = output_dirs['dd.y']
        path = os.path.join(output_dir, filename)
        mi.util.write_bitmap(path, depth)

        if create_debug_pngs:
            bitmap = mi.Bitmap(depth, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
            output_dir_png = output_dirs['dd_png']
            png_filename = output_name + "_depth.png"
            path = os.path.join(output_dir_png, png_filename)
            mi.util.write_bitmap(path, bitmap)

    if "sh_normal" in aovs.values():
        normal = mi.TensorXf(channels['nn']) * 0.5 + 0.5
        filename = output_name + "_normal.exr"
        output_dir = output_dirs['nn']
        path = os.path.join(output_dir, filename)
        mi.util.write_bitmap(path, normal)

        if create_debug_pngs:
            png_filename = output_name + "_normal.png"
            output_dir_png = output_dirs['nn_png']
            path = os.path.join(output_dir_png, png_filename)
            mi.util.write_bitmap(path, normal)

def create_aov(aovs, shape, camera, input_path, output_name, output_dirs, create_debug_pngs):
    integrator_aov = create_scenedesc.create_intergrator_aov(aovs)
    scene_desc = {"type": "scene", "shape": shape, "camera": camera, "integrator": integrator_aov}
    # Sometimes mesh data is not incorrect and could not be loaded
    try:
        scene = mi.load_dict(scene_desc)
    except Exception as e:
        print("Exception occured in " + shape["filename"])
        print(e)
        return
    return avo(scene, input_path, aovs, output_name, output_dirs, create_debug_pngs)

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

def run(type, input_path, output_dirs, fov, aovs=[], emitter_samples=0, output_name="", width=256, height=256, create_debug_png=True):
    datatype = input_path.rsplit(".", 1)[1]
    if datatype != "obj" and datatype != "ply":
        print("Given datatype cannot be processed, must be either obj or ply type.")
        #return
    shape = create_scenedesc.create_shape(input_path, datatype)

    # bounding box diagonal is assumed to be 1, see mesh_preprocess_operations.py normalize_mesh
    distance = math.tan(math.radians(fov))/1.75
    near_distance = distance
    far_distance = distance * 4

    centroid = np.array([distance, -distance, distance])
    if len(output_name) <= 0:
        output_name = (input_path.rsplit("\\", 1)[-1]).rsplit(".", 1)[0]

    # center is assumed to be at 0,0,0, see mesh_preprocess_operations.py translate_to_origin
    camera = create_scenedesc.create_camera(T.look_at(target=(0.0, 0.0, 0.0),
                                                      origin=tuple(centroid),
                                                      up=(0, 0, 1),
                                                      ),
                                            fov, near_distance, far_distance,
                                            width, height)

    for key, value in output_dirs.items():
        if not os.path.exists(value):
            os.mkdir(value)

    if type == "aov":
        create_aov(aovs, shape, camera, input_path, output_name, output_dirs, create_debug_png)
    elif type == "rendering":
        create_rendering(emitter_samples, shape, camera, output_name, output_dirs)
    elif type == "combined":
        create_aov(aovs, shape, camera, input_path, output_name, output_dirs, create_debug_png)
        create_rendering(emitter_samples, shape, camera, output_name, output_dirs)
    else:
        raise Exception("Given type not known!")

def diff_ars(args):
    run(args.type, args.input_path, args.output_dirs, args.fov, args.aovs, args.emitter_samples)

def main(args):
    parser = argparse.ArgumentParser(prog="scene_rendering")
    parser.add_argument("--type", type=str, help="use \"aov\", \"rendering\" or \"combined\"")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_dirs", type=dict, default={'nn': '..\\..\\output', 'dd.y': '..\\..\\output', "dd_png": '..\\..\\output', "nn_png": '..\\..\\output', 'rendering': '..\\..\\output'})
    parser.add_argument("--fov", type=int, default=50)
    parser.add_argument("--aovs", type=dir, default={"dd.y": "depth", "nn": "sh_normal"})
    parser.add_argument("--emitter_samples", type=int, default=4)
    args = parser.parse_args(args)
    diff_ars(args)

if __name__ == '__main__':
    output_dirs = {'nn': '..\\..\\output', 'dd.y': '..\\..\\output', "dd_png": '..\\..\\output', "nn_png": '..\\..\\output', 'rendering': '..\\..\\output'}
    params = [
        '--type', 'combined',
        '--input_path', '..\\..\\resources\\ABC\\abc_0099_stl2_v00\\0_499\\00990247\\00990247_ad629c15d8f3b4ae3a8abd66_trimesh_000.ply',
        ]
    main(params)