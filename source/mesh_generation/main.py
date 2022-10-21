import argparse
import deform_mesh
import os
import source.util.OpenEXR_utils as OpenEXR_utils

def run(normal_map_path, depth_map_path, basic_mesh, output_dir, log_dir,
        epochs, lr, weight_depth, weight_normal, weight_smoothness, weight_edge):

    if not os.path.exists(normal_map_path) or not os.path.exists(depth_map_path) or not os.path.exists(basic_mesh):
        raise Exception("Normal map {}, depth map {} or base mesh {} does not exist!".format(normal_map_path, depth_map_path, basic_mesh))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    mesh_gen = deform_mesh.MeshGen(output_dir,
                        log_dir,
                        weight_depth,
                        weight_normal,
                        weight_smoothness,
                        weight_edge,
                        epochs,
                        lr)
    normal_map = OpenEXR_utils.getRGBimageEXR(normal_map_path, 2)
    depth_map = OpenEXR_utils.getRGBimageEXR(depth_map_path, 2)
    mesh_gen.deform_mesh(normal_map, depth_map, basic_mesh)

def diff_args(args):
    run(args.normal_file_path,
        args.depth_file_path,
        args.base_mesh_path,
        args.output_dir,
        args.log_dir,
        args.epoch,
        args.lr,
        args.weight_depth,
        args.weight_normal,
        args.weight_smoothness,
        args.weight_edge
        )

def main(args):
    parser = argparse.ArgumentParser(prog="mapgen_dataset")
    parser.add_argument("--normal_file_path", type=str, help="path to normal map")
    parser.add_argument("--depth_file_path", type=str, help="path to depth map")
    parser.add_argument("--base_mesh_path", type=str, help="path to base mesh object")
    parser.add_argument("--output_dir", type=str, default="output dir", help="path to output dir")
    parser.add_argument("--log_dir", type=str, default="logs", help="path to logs dir")
    parser.add_argument("--epoch", type=int, default=10, help="# of epoch for mesh generation")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for mesh generation")
    parser.add_argument("--weight_depth", type=int, default=0.5, help="depth weight")
    parser.add_argument("--weight_normal", type=int, default=0.5, help="normal weight")
    parser.add_argument("--weight_smoothness", type=int, default=0.01, help="smoothness weight")
    parser.add_argument("--weight_edge", type=int, default=0.9, help="edge weight")
    args = parser.parse_args(args)
    diff_args(args)

if __name__ == '__main__':
    params = [
        '--normal_file_path', '..\\..\\resources\\mapgen_dataset\\mixed_deform_mesh_0_500\\d_mapgen_normalized\\32770_depth.exr',
        '--depth_file_path', '..\\..\\resources\\mapgen_dataset\\mixed_deform_mesh_0_500\\n_mapgen\\32770_normal.exr',
        '--base_mesh_path', '..\\..\\resources\\topology_meshes\\genus1.ply',
        '--output_dir', '..\\..\\output\\meshdir',
        '--log_dir', '..\\..\\output\\logs'
    ]
    main(params)