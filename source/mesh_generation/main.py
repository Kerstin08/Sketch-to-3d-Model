import argparse
import deform_mesh
import os
from source.util import OpenEXR_utils
from source.util import data_type
from source.util import dir_utils

def run(normal_map_path, depth_map_path, basic_mesh, output_dir, log_dir,
        epochs, log_frequency, lr, weight_depth, weight_normal, weight_smoothness, weight_edge):

    if not os.path.exists(normal_map_path) or not os.path.exists(depth_map_path) or not os.path.exists(basic_mesh):
        raise Exception("Normal map {}, depth map {} or base mesh {} does not exist!".format(normal_map_path, depth_map_path, basic_mesh))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    log_dir = dir_utils.create_logdir(log_dir)
    mesh_gen = deform_mesh.MeshGen(output_dir,
                        log_dir,
                        weight_depth,
                        weight_normal,
                        weight_smoothness,
                        weight_edge,
                        epochs,
                        log_frequency,
                        lr)
    normal_map = OpenEXR_utils.getRGBimageEXR(normal_map_path, data_type.Type.normal, 2)
    depth_map = OpenEXR_utils.getRGBimageEXR(depth_map_path, data_type.Type.depth, 2)

    mesh_gen.deform_mesh(normal_map, depth_map, basic_mesh)

def diff_args(args):
    run(args.normal_file_path,
        args.depth_file_path,
        args.base_mesh_path,
        args.output_dir,
        args.log_dir,
        args.epoch,
        args.log_frequency,
        args.lr,
        args.weight_depth,
        args.weight_normal,
        args.weight_smoothness,
        args.weight_edge
        )

def main(args):
    parser = argparse.ArgumentParser(prog="map_generation_dataset")
    parser.add_argument("--normal_file_path", type=str, help="path to normal map")
    parser.add_argument("--depth_file_path", type=str, help="path to depth map")
    parser.add_argument("--base_mesh_path", type=str, help="path to base mesh object")
    parser.add_argument("--output_dir", type=str, default="output dir", help="path to output dir")
    parser.add_argument("--log_dir", type=str, default="logs", help="path to logs dir")
    parser.add_argument("--epoch", type=int, default=10, help="# of epoch for mesh generation")
    parser.add_argument("--log_frequency", type=int, default=1, help="frequency logs are written")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for mesh generation")
    parser.add_argument("--weight_depth", type=int, default=0.002, help="depth weight")
    parser.add_argument("--weight_normal", type=int, default=0.002, help="normal weight")
    parser.add_argument("--weight_smoothness", type=int, default=0.01, help="smoothness weight")
    parser.add_argument("--weight_edge", type=int, default=0.9, help="edge weight")
    args = parser.parse_args(args)
    diff_args(args)

if __name__ == '__main__':
    params = [
        '--normal_file_path', '..\\..\\resources\\map_generation_dataset\\mixed_0_2500_normal\\target_mapgen\\train\\32770_normal.exr',
        '--depth_file_path', '..\\..\\resources\\map_generation_dataset\\mixed_0_2500_depth\\target_mapgen\\train\\32770_depth.exr',
        '--base_mesh_path', '..\\..\\resources\\topology_meshes\\genus0.ply',
        '--output_dir', '..\\..\\output\\meshdir',
        '--log_dir', '..\\..\\output\\logs'
    ]
    main(params)