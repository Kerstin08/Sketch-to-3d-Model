import argparse
import os
import floodfill
import euler
import basic_mesh

def run(image_path, genus_dir):
    if not os.path.exists(image_path):
        print("Image Path does not exist!")
        return

    filled_image = floodfill.startFill(image_path)
    holes = euler.get_number_holes(filled_image)
    basic_mesh_path = basic_mesh.get_basic_mesh_path(holes, genus_dir)
    return basic_mesh_path

def diff_ars(args):
    run(args.image_path, args.genus_dir)

def main(args):
    parser = argparse.ArgumentParser(prog="topology_determination")
    parser.add_argument("--image_path", type=str, help="Use image path to the image the tology should be determined from.")
    parser.add_argument("--genus_dir", type=str, help="Path to the directory where the genus templates are stored.")
    args = parser.parse_args(args)
    diff_ars(args)

if __name__ == '__main__':
    params = [
        '--image_path', '..\\..\\resources\\topology_tests_images\\genus_03.png',
        '--genus_dir', '..\\..\\resources\\topology_meshes'
        ]
    main(params)