import math
import os.path
import numpy as np
from pathlib import Path

import source.util.mi_create_scenedesc as create_scenedesc
import drjit as dr
import argparse
import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T
import source.mesh_generation.depth_reparam_integrator
import source.mesh_generation.normal_reparam_integrator
mi.set_variant('cuda_ad_rgb')


def rendering(scene, output_name, output_dirs):
    img = mi.render(scene, seed=0, spp=256)
    bitmap = mi.util.convert_to_bitmap(img)
    filename = output_name + "_rendering.png"
    output_dir = output_dirs["rendering"]
    path = os.path.join(output_dir, filename)
    mi.util.write_bitmap(path, bitmap)

def avo(scene, input_path, aovs, output_name, output_dirs, create_debug_pngs=True):
    if "depth" in aovs.values():
        depth_integrator = source.util.mi_create_scenedesc.create_integrator_depth()
        depth_integrator_lodaded = mi.load_dict(depth_integrator)
        img = mi.render(scene, seed=0, spp=256, integrator=depth_integrator_lodaded)
        # If mesh has invalid mesh vertices, rendering contains nan.
        if dr.any(dr.isnan(img)):
            print(
                "Rendered image " + output_name + " includes invalid data! Vertex normals in input model " + input_path + " might be corrupt.")
            return

        mask = img.array < 1.5
        curr_min_val = dr.min(img)
        masked_img = dr.select(mask,
                    img.array,
                    0.0)
        curr_max_val = dr.max(masked_img)
        wanted_range_min, wanted_range_max = 0.0, 0.75
        depth = dr.select(mask,
                          (img.array - curr_min_val) * (
                                  (wanted_range_max - wanted_range_min) / (
                                      curr_max_val - curr_min_val)) + wanted_range_min,
                          1.0)
        depth_tens = mi.TensorXf(depth, shape=(256, 256, 3))

        filename = output_name + "_depth.exr"
        output_dir = output_dirs['dd.y']
        path = os.path.join(output_dir, filename)
        mi.util.write_bitmap(path, depth_tens)

        if create_debug_pngs:
            output_dir_png = output_dirs['dd_png']
            png_filename = output_name + "_depth.png"
            path = os.path.join(output_dir_png, png_filename)
            mi.util.write_bitmap(path, depth_tens)

    if "sh_normal" in aovs.values():
        normal_integrator = source.util.mi_create_scenedesc.create_integrator_normal()
        normal_integrator_lodaded = mi.load_dict(normal_integrator)
        img = mi.render(scene, seed=0, spp=256, integrator=normal_integrator_lodaded)
        # If mesh has invalid mesh vertices, rendering contains nan.
        if dr.any(dr.isnan(img)):
            print(
                "Rendered image " + output_name + " includes invalid data! Vertex normals in input model " + input_path + " might be corrupt.")
            return
        filename = output_name + "_normal.exr"
        output_dir = output_dirs['nn']
        path = os.path.join(output_dir, filename)
        mi.util.write_bitmap(path, img)

        if create_debug_pngs:
            output_dir_png = output_dirs['nn_png']
            png_filename = output_name + "_normal.png"
            path = os.path.join(output_dir_png, png_filename)
            mi.util.write_bitmap(path, img)
#
def create_aov(aovs, shape, camera, input_path, output_name, output_dirs, create_debug_pngs):
    scene_desc = {"type": "scene", "shape": shape, "camera": camera}
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
        return
    shape = create_scenedesc.create_shape(input_path, datatype)

    # bounding box diagonal is assumed to be 1, see mesh_preprocess_operations.py normalize_mesh
    distance = math.tan(math.radians(fov))/1.75
    near_distance = distance
    far_distance = distance * 4

    centroid = np.array([distance, -distance, distance])
    if len(output_name) <= 0:
        filename = Path(input_path)
        output_name = (filename.stem)

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
    parser.add_argument("--aovs", type=dir, default={"nn": "sh_normal"})
    parser.add_argument("--emitter_samples", type=int, default=4)
    args = parser.parse_args(args)
    diff_ars(args)

if __name__ == '__main__':
    output_dirs = {'nn': '..\\..\\output', 'dd.y': '..\\..\\output', "dd_png": '..\\..\\output', "nn_png": '..\\..\\output', 'rendering': '..\\..\\output'}
    params = [
        '--type', 'aov',
        '--input_path', '..\\..\\resources\\thingi10k\\2500_2999\\229606.ply',
        ]
    main(params)