import argparse
import os
from floodfill import FloodFill
import determine_number_holes

def run(imagePath):
    if not os.path.exists(imagePath):
        print("Image Path does not exist!")
        return

    floodFill = FloodFill(imagePath)
    filled_image = floodFill.startFill()
    holes = determine_number_holes.get_number_holes(filled_image)
    print(holes)

def diff_ars(args):
    run(args.imagePath)

def main(args):
    parser = argparse.ArgumentParser(prog="topology_determination")
    parser.add_argument("--imagePath", type=str, help="Use image path to the image the tology should be determined from.")
    args = parser.parse_args(args)
    diff_ars(args)

if __name__ == '__main__':
    params = [
        '--imagePath', '..\\..\\resources\\topology_tests_images\\genus_03.png',
        ]
    main(params)