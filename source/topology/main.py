import argparse
import os
import sys

import floodfill
import euler
import basic_mesh

import source.util.sketch_utils as sketch_utils
from source.util import dir_utils


def run(
        image_path: str,
        genus_dir: str,
        output_dir: str
):
    if not os.path.exists(output_dir):
        dir_utils.create_general_folder(output_dir)
    image = sketch_utils.load_image(image_path, True)
    filled_image, _ = floodfill.startFill(image, image_path, output_dir, True)
    holes = euler.get_number_holes(filled_image)
    basic_mesh_path = basic_mesh.get_basic_mesh_path(holes, genus_dir)


def diff_ars(args):
    run(args.image_path, args.genus_dir, args.output_dir)


def main(args):
    parser = argparse.ArgumentParser(prog="topology_determination")
    parser.add_argument("--image_path", type=str, default="datasets/test.png",
                        help="Use image path to the image the tology should be determined from.")
    parser.add_argument("--genus_dir", type=str, default="datasets/topology_meshes",
                        help="Path to the directory where the genus templates are stored.")
    parser.add_argument("--output_dir", type=str, default="filled",
                        help="Path to the directory where the resulting exr and possible png sould be stored.")
    args = parser.parse_args(args)
    diff_ars(args)


if __name__ == "__main__":
    main(sys.argv[1:])
