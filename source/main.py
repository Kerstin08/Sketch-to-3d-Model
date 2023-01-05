import argparse
import os
import sys
from pathlib import Path

from source.mesh_generation import deform_mesh
from source.util import sketch_utils
from source.topology import floodfill
from source.topology import euler
from source.topology import basic_mesh
from source.util import dir_utils
from source.map_generation.test import test
from source.util import OpenEXR_utils
from source.util import data_type
from source.util import bool_parse

# Topology
## (1.a. split input points into different input sketches based on different "classes", which are signaled by different colors)
## (1.b. scale input image to be 256x265 -> maybe use vectorization or splines to represent sketch in order to save lines)
## 1. floodfill
## 2. euler
## (2.a. get connectivity between the holes in order to obtain better mesh)
## 3. obtain mesh based on euler result (and connectivity result)
def topology(sketch_path, genus_dir, output_dir, use_genus0):
    image = sketch_utils.load_image(sketch_path, True)
    filled_image, exr_path = floodfill.startFill(image, sketch_path, output_dir, False)
    if not use_genus0:
        holes = euler.get_number_holes(filled_image)
    else:
        holes = 0
    basic_mesh_path = basic_mesh.get_basic_mesh_path(holes, genus_dir)
    return basic_mesh_path, exr_path

# Map Generation
## 2. put cleaned input sketch into trained neural network normal
## 3. put cleaned input sketch into trained neural network depth
def map_generation(input_sketch, output_dir, normal_map_gen_model, depth_map_gen_model, log_dir_normal, log_dir_depth):
    sketch_filepath = os.path.join(output_dir, "sketch_mapgen")
    sketch_filepath_test = os.path.join(sketch_filepath, "test")
    if not os.path.exists(sketch_filepath_test):
        dir_utils.create_general_folder(sketch_filepath_test)
    sketch_utils.clean_userinput(input_sketch, sketch_filepath_test)
    normal_output_path = os.path.join(output_dir, "normal")
    if not os.path.exists(normal_output_path):
        dir_utils.create_general_folder(normal_output_path)
    depth_output_path = os.path.join(output_dir, "depth")
    if not os.path.exists(depth_output_path):
        dir_utils.create_general_folder(depth_output_path)
    test(output_dir, normal_output_path, log_dir_normal, "normal", normal_map_gen_model)
    test(output_dir, depth_output_path, log_dir_depth, "depth", depth_map_gen_model)
    return normal_output_path, depth_output_path

# Mesh deformation
## 1. put input mesh and normal and depth map into mesh deformation
def mesh_deformation(output_name, normal_map_path, depth_map_path, silhouette_map_path, basic_mesh, output_dir, logs,
                     weight_depth, weight_normal, weight_smoothness, weight_edge, weight_silhouette,
                     epochs, log_frequency, lr, views, use_depth, eval_dir):
    if not os.path.exists(logs):
        dir_utils.create_version_folder(logs)
    mesh_gen = deform_mesh.MeshGen(output_name, output_dir, logs,
                                   weight_depth, weight_normal, weight_smoothness, weight_silhouette, weight_edge,
                                   epochs, log_frequency, lr, views, use_depth, eval_dir)
    normal_map = OpenEXR_utils.getImageEXR(normal_map_path, data_type.Type.normal, 2)
    depth_map = OpenEXR_utils.getImageEXR(depth_map_path, data_type.Type.depth, 2).squeeze()
    silhouette_map = OpenEXR_utils.getImageEXR(silhouette_map_path, data_type.Type.depth, 2).squeeze()
    mesh_gen.deform_mesh(normal_map, depth_map, silhouette_map, basic_mesh)

def run(input_sketch,
        output_dir, logs_dir, genus_dir,
        depth_map_gen_model, normal_map_gen_model,
        epochs_mesh_gen, log_frequency_mesh_gen, lr_mesh_gen,
        weight_depth, weight_normal, weight_smoothness, weight_edge, weight_silhouette, views,
        use_depth_str, use_genus0_str, eval_dir):
    for x in (input_sketch, depth_map_gen_model, normal_map_gen_model):
        if not os.path.exists(x):
            raise Exception("{} does not exist".format(x))

    use_depth = bool_parse.parse(use_depth_str)
    use_genus0 = bool_parse.parse(use_genus0_str)

    file = Path(input_sketch)
    output_name = file.stem
    prefix = output_name.rsplit("_", 1)[0]
    output_dir = dir_utils.create_prefix_folder(prefix, output_dir)
    logs_dir = dir_utils.create_prefix_folder(prefix, logs_dir)
    eval_dir = dir_utils.create_general_folder(eval_dir)

    basic_mesh, silhouette_map_path = topology(input_sketch, genus_dir, output_dir, use_genus0)
    logs_map_generation_normal = os.path.join(logs_dir, "map_generation_normal")
    if not os.path.exists(logs_map_generation_normal):
        dir_utils.create_general_folder(logs_map_generation_normal)
    logs_map_generation_depth = os.path.join(logs_dir, "map_generation_depth")
    if not os.path.exists(logs_map_generation_depth):
           dir_utils.create_general_folder(logs_map_generation_depth)
    normal_output_path, depth_output_path = map_generation(input_sketch, output_dir, normal_map_gen_model, depth_map_gen_model,
                                                           logs_map_generation_normal, logs_map_generation_depth)

    logs_meshGen = os.path.join(logs_dir, "mesh_generation")
    if not os.path.exists(logs_meshGen):
        dir_utils.create_general_folder(logs_meshGen)
    normal_map = os.path.join(normal_output_path, "{}_normal.exr".format(prefix))
    depth_map = os.path.join(depth_output_path, "{}_depth.exr".format(prefix))
    mesh_deformation(prefix, normal_map, depth_map, silhouette_map_path, basic_mesh, output_dir, logs_meshGen,
                     weight_depth, weight_normal, weight_smoothness, weight_silhouette, weight_edge,
                     epochs_mesh_gen, log_frequency_mesh_gen, lr_mesh_gen, views, use_depth, eval_dir)

def diff_ars(args):
    run(args.input_sketch,
        args.output_dir,
        args.logs_dir,
        args.genus_dir,
        args.depth_map_gen_model,
        args.normal_map_gen_model,
        args.epochs_mesh_gen,
        args.log_frequency_mesh_gen,
        args.lr_mesh_gen,
        args.weight_depth,
        args.weight_normal,
        args.weight_smoothness,
        args.weight_edge,
        args.weight_silhouette,
        args.views,
        args.use_depth,
        args.use_genus0,
        args.eval_dir
        )

def main(args):
    parser = argparse.ArgumentParser(prog="sketch_to_mesh")
    parser.add_argument("--input_sketch", type=str, help="Path to sketch.")
    parser.add_argument("--output_dir", type=str, default="output_pipeline", help="Directory where the test output is stored")
    parser.add_argument("--logs_dir", type=str, default="logs_pipeline", help="Directory where the logs are stored")
    parser.add_argument("--genus_dir", type=str, default="datasets/topology_meshes", help="Path to the directory where the genus templates are stored")
    parser.add_argument("--depth_map_gen_model", type=str, default="datasets/mapgen_test_models/depth.ckpt", help="Path to model, which is used to determine depth map.")
    parser.add_argument("--normal_map_gen_model", type=str, default="datasets/mapgen_test_models/normal.ckpt", help="Path to model, which is used to determine normal map.")
    parser.add_argument("--epochs_mesh_gen", type=int, default=40000, help="# of epoch for mesh generation")
    parser.add_argument("--log_frequency_mesh_gen", type=int, default=100, help="frequency image logs of the mesh generation are written")
    parser.add_argument("--lr_mesh_gen", type=float, default=0.0002, help="initial learning rate for mesh generation")
    parser.add_argument("--weight_depth", type=float, default=0.002, help="depth weight")
    parser.add_argument("--weight_normal", type=float, default=0.002,help="normal weight")
    parser.add_argument("--weight_smoothness", type=float, default=0.01, help="smoothness weight")
    parser.add_argument("--weight_edge", type=float, default=0.9, help="edge weight")
    parser.add_argument("--weight_silhouette", type=float, default=0.9, help="silhouette weight")
    parser.add_argument("--views", type=list, default=[(225, 30)], help="define rendering view angles")
    # For ablation study
    # If using via command line for false add "" instead of false since that for some reason can cause problems
    parser.add_argument("--use_depth", type=str, default="True", help="Use depth loss for mesh deformation; use \"True\" or \"False\" as parameter")
    parser.add_argument("--use_genus0", type=str, default="False", help="Use base mesh genus 0 regardless for every given shape; use \"True\" or \"False\" as parameter")
    # Additional directory to store created meshes in for easier evaluation
    parser.add_argument("--eval_dir", type=str, default="eval", help="Additional directory to store only mesh for evaluation")
    args = parser.parse_args(args)
    diff_ars(args)

if __name__ == '__main__':
    main(sys.argv[1:])