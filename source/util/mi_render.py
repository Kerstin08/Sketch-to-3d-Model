import os.path

import source.util.mi_create_scenedesc as create_scenedesc
import argparse
import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T
from mitsuba.scalar_rgb import Point3f as P
mi.set_variant('cuda_ad_rgb')


def rendering(scene, mesh_name, output_dir):
    img = mi.render(scene, seed=0, spp=1024)
    bitmap = mi.util.convert_to_bitmap(img)
    filename = mesh_name + "_rendering.png"
    path = os.path.join(output_dir, filename)
    mi.util.write_bitmap(path, bitmap)

def avo(scene, aovs, mesh_name, output_dir):
    img = mi.render(scene, seed=0, spp=1024)
    bitmap = mi.Bitmap(img, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
    channels = dict(bitmap.split())
    if "depth" in aovs.values():
        depth = channels['dd.y']
        bitmap = mi.Bitmap(depth, channel_names=['R', 'G', 'B'])
        filename = mesh_name + "_depth.png"
        path = os.path.join(output_dir, filename)
        mi.util.write_bitmap(path, bitmap)
    if "sh_normal" in aovs.values():
        normal = channels['nn']
        filename = mesh_name + "_normal.png"
        path = os.path.join(output_dir, filename)
        mi.util.write_bitmap(path, normal)

def create_aov(aovs, shape, camera, mesh_name, output_dir):
    integrator_aov = create_scenedesc.create_intergrator_aov(aovs)
    scene_desc = {"type": "scene", "shape": shape, "camera": camera, "integrator": integrator_aov}
    scene = mi.load_dict(scene_desc)
    return avo(scene, aovs, mesh_name, output_dir)

def create_rendering(emitter_samples, shape, camera, mesh_name, output_dir):
    integrator_rendering = create_scenedesc.create_integrator_direct(emitter_samples)
    emitter = create_scenedesc.create_emitter()
    scene_desc = {"type": "scene", "shape": shape, "camera": camera, "integrator": integrator_rendering, "emitter": emitter}
    scene = mi.load_dict(scene_desc)
    rendering(scene, mesh_name, output_dir)

def run(type, input_mesh, output_dir, aovs=[], emitter_samples=0):
    mesh_name = (input_mesh.rsplit("\\", 1)[-1]).rsplit(".", 1)[0]
    shape = create_scenedesc.create_shape(input_mesh, T.scale(0.1))
    camera = create_scenedesc.create_camera(T.look_at(target=(0, 0, 0),
                                                                   origin=(0, 0, 2),
                                                                   up=(0, 1, 0),
                                                                   ))
    if type == "aov":
        create_aov(aovs, shape, camera, mesh_name, output_dir)
    elif type == "rendering":
        create_rendering(emitter_samples, shape, camera, mesh_name, output_dir)
    elif type == "combined":
        create_aov(aovs, shape, camera, mesh_name, output_dir)
        create_rendering(emitter_samples, shape, camera, mesh_name, output_dir)
    else:
        raise Exception("Given type not known!")

def diff_ars(args):
    run(args.type, args.input_mesh, args.output_dir, args.aovs, args.emitter_samples)

def main(args):
    parser = argparse.ArgumentParser(prog="scene_rendering")
    parser.add_argument("--type", type=str, help="use \"aov\", \"rendering\" or \"combined\"")
    parser.add_argument("--input_mesh", type=str)
    parser.add_argument("--output_dir", type=str, help="define path for output dir")
    parser.add_argument("--aovs", type=dir, default={"nn": "sh_normal", "dd.y": "depth"})
    parser.add_argument("--emitter_samples", type=int, default=4)
    args = parser.parse_args(args)
    diff_ars(args)

if __name__ == '__main__':
    params = [
        '--type', 'combined',
        '--input_mesh', '..\\..\\resources\\meshes\\teapot.ply',
        '--output_dir', '..\\..\\output'
    ]
    main(params)