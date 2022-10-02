import argparse
import os
import source.map_generation.main as map_gen
import source.topology.main as topology_determination

# Topology
## (1.a. split input points into different input sketches based on different "classes", which are signaled by different colors)
## (1.b. scale input image to be 256x265 -> maybe use vectorization or splines to represent sketch in order to save lines)
## 1. floodfill
## 2. euler
## (2.a. get connectivity between the holes in order to obtain better mesh)
## 3. obtain mesh based on euler result (and connectivity result)
def topology(sketch, genus_dir):
    basic_mesh_path = topology_determination.run(sketch, genus_dir)
    return basic_mesh_path

# Map Generation
## 2. put cleaned input sketch into trained neural network normal
## 3. put cleaned input sketch into trained neural network depth
def map_generation(cleaned_sketch, output_dir, logs_dir, type, epochs, lr, batch_size, n_critic, weight_L1, model):
    return map_gen.run(False, cleaned_sketch, output_dir, logs_dir, type, epochs, lr, batch_size, n_critic, weight_L1, True, model, False)

# Mesh deformation
## 1. put input mesh and normal and depth map into mesh deformation
def mesh_deformation(cleaned_sketch, normal_map, depth_map, basic_mesh):
    pass

def run(image_sketch, genus_dir, depth_map_gen_model, normal_map_gen_model, output_dir, logs_dir, epochs, lr, batch_size, n_critic, weight_L1):
    for x in (image_sketch, genus_dir, depth_map_gen_model, normal_map_gen_model):
        if not os.path.exists(x):
            print(str(x) + " does not exist")
    basic_mesh = topology(image_sketch, genus_dir)
    #Todo: remove user input points from (scaled) 265x256 input sketch in order to generate cleaned sketch
    cleaned_sketch = 0
    normal_map = map_generation(cleaned_sketch, output_dir, logs_dir, "normal", epochs, lr, batch_size, n_critic, weight_L1,normal_map_gen_model)
    depth_map = map_generation(cleaned_sketch, output_dir, logs_dir, "depth", epochs, lr, batch_size, n_critic, weight_L1, depth_map_gen_model)

    mesh_deformation(cleaned_sketch, normal_map, depth_map, basic_mesh)


def diff_ars(args):
    run(args.input_sketch,
        args.genus_dir,
        args.depth_map_gen_model,
        args.normal_map_gen_model,
        args.output_dir,
        args.logs_dir,
        args.epochs,
        args.lr,
        args.batch_size,
        args.n_critic,
        args.weight_L1,
        )

def main(args):
    parser = argparse.ArgumentParser(prog="sketch_to_mesh")
    parser.add_argument("--image_sketch", type=str, help="Path to sketch.")
    parser.add_argument("--genus_dir", type=str, help="Path to the directory where the genus templates are stored")
    parser.add_argument("--depth_map_gen_model", type=str, help="Path to model, which is used to determine depth map.")
    parser.add_argument("--normal_map_gen_model", type=str, help="Path to model, which is used to determine normal map.")
    parser.add_argument("--output_dir", type=str, default="..\\..\\test", help="Directory where the test output is stored")
    parser.add_argument("--logs_dir", type=str, default="..\\..\\logs", help="Directory where the logs are stored")
    parser.add_argument("--epochs", type=int, default=10, help="# of epoch")
    parser.add_argument("--lr", type=float, default=100, help="initial learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="# of epoch")
    parser.add_argument("--n_critic", type=int, default=5, help="# of n_critic")
    parser.add_argument("--weight_L1", type=int, default=50, help="L1 weight")
    args = parser.parse_args(args)
    diff_ars(args)

if __name__ == '__main__':
    params = [
        '--input_sketch', '..\\..\\resources\\topology_tests_images\\genus_03.png',
        '--genus_dir', '..\\..\\resources\\topology_meshes'
        '--depth_map_gen_model', '..\\..\\resources\\topology_tests_images\\genus_03.png',
        '--normal_map_gen_model', '..\\..\\resources\\topology_meshes'
        ]
    main(params)