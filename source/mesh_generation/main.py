import argparse
import sys
import os

from source.mesh_generation import deform_mesh
from source.util import OpenEXR_utils
from source.util import data_type
from source.util import dir_utils


def run(normal_map_path, depth_map_path, silhouette_map_path, basic_mesh, output_name, output_dir, log_dir,
        epochs, log_frequency, views, lr, weight_depth, weight_normal, weight_smoothness, weight_edge,
        weight_silhouette):
    if not os.path.exists(normal_map_path) or not os.path.exists(depth_map_path) or not os.path.exists(
            silhouette_map_path) or not os.path.exists(basic_mesh):
        raise Exception(
            "Normal map {}, depth map {}, silhouette map {} or base mesh {} does not exist!".format(normal_map_path,
                                                                                                    depth_map_path,
                                                                                                    silhouette_map_path,
                                                                                                    basic_mesh))
    if len(views) != 1:
        raise Exception("Only one view can be given to deform the mesh generation!")

    # use logdir creation for outputdir creation to get different deformed meshes when running parallel
    output_dir = dir_utils.create_version_folder(output_dir)
    log_dir = dir_utils.create_version_folder(log_dir)
    mesh_gen = deform_mesh.MeshGen(output_name,
                                   output_dir,
                                   log_dir,
                                   weight_depth,
                                   weight_normal,
                                   weight_smoothness,
                                   weight_edge,
                                   weight_silhouette,
                                   epochs,
                                   log_frequency,
                                   lr,
                                   views)
    normal_map = OpenEXR_utils.getImageEXR(normal_map_path, data_type.Type.normal, 2)
    depth_map = OpenEXR_utils.getImageEXR(depth_map_path, data_type.Type.depth, 2).squeeze()
    silhouette_map = OpenEXR_utils.getImageEXR(silhouette_map_path, data_type.Type.silhouette, 2).squeeze()

    mesh_gen.deform_mesh(normal_map, depth_map, silhouette_map, basic_mesh)


def diff_args(args):
    run(args.normal_file_path,
        args.depth_file_path,
        args.silhouette_file_path,
        args.base_mesh_path,
        args.output_name,
        args.output_dir,
        args.log_dir,
        args.epoch,
        args.log_frequency,
        args.views,
        args.lr,
        args.weight_depth,
        args.weight_normal,
        args.weight_smoothness,
        args.weight_edge,
        args.weight_silhouette,
        )


def main(args):
    parser = argparse.ArgumentParser(prog="map_generation_dataset")
    parser.add_argument("--normal_file_path", type=str, help="path to normal map")
    parser.add_argument("--depth_file_path", type=str, help="path to depth map")
    parser.add_argument("--silhouette_file_path", type=str, help="path to depth map")
    parser.add_argument("--base_mesh_path", type=str, help="path to base mesh object")
    parser.add_argument("--output_name", type=str, default="deform_mesh", help="filename of output mesh")
    parser.add_argument("--output_dir", type=str, default="output_dir", help="path to output dir")
    parser.add_argument("--log_dir", type=str, default="logs", help="path to logs dir")
    parser.add_argument("--epoch", type=int, default=40000, help="# of epoch for mesh generation")
    parser.add_argument("--log_frequency", type=int, default=100, help="frequency logs are written")
    parser.add_argument("--views", type=list, default=[(225, 30)], help="define rendering view angles")
    parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for mesh generation")
    parser.add_argument("--weight_depth", type=float, default=0.002, help="depth weight")
    parser.add_argument("--weight_normal", type=float, default=0.002, help="normal weight")
    parser.add_argument("--weight_smoothness", type=float, default=0.02, help="smoothness weight")
    parser.add_argument("--weight_edge", type=float, default=0.9, help="edge weight")
    parser.add_argument("--weight_silhouette", type=float, default=0.9, help="silhouette weight")
    args = parser.parse_args(args)
    diff_args(args)


if __name__ == '__main__':
    main(sys.argv[1:])
