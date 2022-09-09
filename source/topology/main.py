import argparse
import os
from floodfill import FloodFill

def run(imagePath):
    if not os.path.exists(imagePath):
        print("Image Path does not exist!")
        return

    floodFill = FloodFill(imagePath)
    floodFill.startFill()

def diff_ars(args):
    run(args.imagePath)

def main(args):
    parser = argparse.ArgumentParser(prog="topology_determination")
    parser.add_argument("--imagePath", type=str, help="Use image path to the image the tology should be determined from.")
    args = parser.parse_args(args)
    diff_ars(args)

if __name__ == '__main__':
    params = [
        '--imagePath', '..\\..\\resources\\topology_tests_images\\small_segments.png',
        ]
    main(params)