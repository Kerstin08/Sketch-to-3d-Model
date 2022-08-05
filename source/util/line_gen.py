import argparse
from matplotlib import pyplot as plt
import source.util.mi_render as render
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv

def run(type, input_mesh, emitter_samples):
    render.run(type, input_mesh, emitter_samples)

    img = cv.imread('../../output/rendering.png', 0)
    edges = cv.Canny(img, 10, 130)
    plt.imsave("../../output/lines_canny.jpg", edges, cmap='binary')

def diff_args(args):
    run(args.type, args.input_mesh, args.emitter_samples)

def main(args):
    parser = argparse.ArgumentParser(prog="scene_rendering")
    parser.add_argument("--type", type=str, help="use \"aov\", \"rendering\" or \"combined\"")
    parser.add_argument("--input_mesh", type=str)
    parser.add_argument("--emitter_samples", type=int, default=4)
    args = parser.parse_args(args)
    diff_args(args)


if __name__ == '__main__':
    params = [
        '--type', 'rendering',
        '--input_mesh', '../../resources/meshes/teapot.ply',
    ]
    main(params)