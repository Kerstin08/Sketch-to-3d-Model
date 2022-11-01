import argparse
import os
import source.mesh_generation.deform_mesh as deform_mesh
import source.util.sketch_preprocess_operations as preprocess_sketch
import source.topology.floodfill as floodfill
import source.topology.euler as euler
import source.topology.basic_mesh as basic_mesh
import source.util.dir_utils as dir_utils
from source.map_generation.test import test



# Topology
## (1.a. split input points into different input sketches based on different "classes", which are signaled by different colors)
## (1.b. scale input image to be 256x265 -> maybe use vectorization or splines to represent sketch in order to save lines)
## 1. floodfill
## 2. euler
## (2.a. get connectivity between the holes in order to obtain better mesh)
## 3. obtain mesh based on euler result (and connectivity result)
def topology(sketch, genus_dir):
    image = preprocess_sketch.load_image(sketch, True)
    filled_image = floodfill.startFill(image)
    holes = euler.get_number_holes(filled_image)
    basic_mesh_path = basic_mesh.get_basic_mesh_path(holes, genus_dir)
    return basic_mesh_path

# Map Generation
## 2. put cleaned input sketch into trained neural network normal
## 3. put cleaned input sketch into trained neural network depth
def map_generation(input_sketch, output_dir, normal_map_gen_model, depth_map_gen_model):
    sketch_filepath = os.path.join(output_dir, "sketch_mapgen")
    sketch_filepath_test = os.path.join(sketch_filepath, "test")
    if not os.path.exists(sketch_filepath_test):
        dir_utils.create_general_folder(sketch_filepath_test)
    preprocess_sketch.clean_userinput(input_sketch, sketch_filepath_test)
    normal_output_path = os.path.join(output_dir, "normal")
    if not os.path.exists(normal_output_path):
        dir_utils.create_general_folder(normal_output_path)
    depth_output_path = os.path.join(output_dir, "depth")
    if not os.path.exists(depth_output_path):
        dir_utils.create_general_folder(depth_output_path)
    test(output_dir, normal_output_path, "normal", normal_map_gen_model)
    test(output_dir, depth_output_path, "depth", depth_map_gen_model)



# Mesh deformation
## 1. put input mesh and normal and depth map into mesh deformation
def mesh_deformation(normal_map, depth_map, basic_mesh, output_dir, logs, weight_depth, weight_normal, weight_smoothness, weight_edge, epochs, lr):
    mesh_gen = deform_mesh.MeshGen(output_dir, logs, weight_depth, weight_normal, weight_smoothness, weight_edge, epochs, lr)
    mesh_gen.deform_mesh(normal_map, depth_map, basic_mesh)

def run(input_sketch,
        output_dir, logs_dir, genus_dir,
        depth_map_gen_model, normal_map_gen_model,
        epochs_mesh_gen, lr_mesh_gen, weight_depth, weight_normal, weight_smoothness, weight_edge
        ):
    for x in (input_sketch, depth_map_gen_model, normal_map_gen_model):
        if not os.path.exists(x):
            raise Exception("{} does not exist".format(x))

    if not os.path.exists(output_dir):
        dir_utils.create_general_folder(output_dir)

    basic_mesh = topology(input_sketch, genus_dir)
    map_generation(input_sketch, output_dir, normal_map_gen_model, depth_map_gen_model)


    logs_meshGen = os.path.join(logs_dir, "mesh_generation")
    if not os.path.exists(logs_meshGen):
        dir_utils.create_logdir(logs_meshGen)
    #normal_map = os.path.join(normal_output_path, "map.exr")
    #depth_map = os.path.join(depth_output_path, "map.exr")
    #mesh_deformation(normal_map, depth_map, basic_mesh, output_dir, logs_meshGen, weight_depth, weight_normal, weight_smoothness, weight_edge, epochs_mesh_gen, lr_mesh_gen)


def diff_ars(args):
    run(args.input_sketch,
        args.output_dir,
        args.logs_dir,
        args.genus_dir,
        args.depth_map_gen_model,
        args.normal_map_gen_model,
        args.epochs_mesh_gen,
        args.lr_mesh_gen,
        args.weight_depth,
        args.weight_normal,
        args.weight_smoothness,
        args.weight_edge
        )

def main(args):
    parser = argparse.ArgumentParser(prog="sketch_to_mesh")
    parser.add_argument("--input_sketch", type=str, help="Path to sketch.")
    parser.add_argument("--output_dir", type=str, default="..\\test", help="Directory where the test output is stored")
    parser.add_argument("--logs_dir", type=str, default="..\\logs", help="Directory where the logs are stored")
    parser.add_argument("--genus_dir", type=str, default="..\\genus", help="Path to the directory where the genus templates are stored")
    parser.add_argument("--depth_map_gen_model", type=str, help="Path to model, which is used to determine depth map.")
    parser.add_argument("--normal_map_gen_model", type=str, help="Path to model, which is used to determine normal map.")
    parser.add_argument("--epochs_mesh_gen", type=int, default=200, help="# of epoch for mesh generation")
    parser.add_argument("--lr_mesh_gen", type=float, default=0.001, help="initial learning rate for mesh generation")
    parser.add_argument("--weight_depth", type=int, default=0.5, help="depth weight")
    parser.add_argument("--weight_normal", type=int, default=0.5, help="normal weight")
    parser.add_argument("--weight_smoothness", type=int, default=0.01, help="smoothness weight")
    parser.add_argument("--weight_edge", type=int, default=0.9, help="edge weight")
    args = parser.parse_args(args)
    diff_ars(args)

if __name__ == '__main__':
    params = [
        '--input_sketch', '..\\resources\\topology_tests_images\\test_pipeline.png',
        '--genus_dir', '..\\resources\\topology_meshes',
        '--depth_map_gen_model', '..\\resources\\mapgen_test_models\\depth.ckpt',
        '--normal_map_gen_model', '..\\resources\\mapgen_test_models\\normal.ckpt'
        ]
    main(params)
