import argparse
import time

from pathlib import Path
import source.util.mi_render as render
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv
import numpy as np

def run(type, input_mesh, output_dirs, fov, emitter_samples, output_name=""):
    for key, value in output_dirs.items():
        if not os.path.exists(value):
            os.mkdir(value)

    if len(output_name) <= 0:
        filename = Path(input_mesh)
        output_name = filename.stem
    render.run(type, input_mesh, output_dirs, fov, emitter_samples=emitter_samples, output_name=output_name, width=1024, height=1024)
    filename_temp = output_name + "_rendering.png"
    path_temp = os.path.join(output_dirs["rendering"], filename_temp)

    filename = output_name + "_sketch.png"
    path = os.path.join(output_dirs["sketch"], filename)
    img = None
    time_to_wait = 10
    while img is None:
        time.sleep(1)
        img = cv.imread(path_temp, 0)
        time_to_wait -= 1
        if time_to_wait < 0:
            print("Temp rendering could not be generated!")
            return

    gaussian = cv.GaussianBlur(img, (5,5), 0)
    edges = cv.Canny(gaussian, 10, 130)
    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv.dilate(edges, kernel)
    dsize = (256, 256)
    output = cv.resize(cv.bitwise_not(img_dilation), dsize)
    img_binary = cv.threshold(output, 240, 255, cv.THRESH_BINARY)[1]

    cv.imwrite(path, img_binary)
    os.remove(path_temp)

def diff_args(args):
    run(args.type, args.input_mesh, args.output_dir, args.fov, args.emitter_samples)

def main(args):
    parser = argparse.ArgumentParser(prog="scene_rendering")
    parser.add_argument("--type", type=str, help="use \"aov\", \"rendering\" or \"combined\"")
    parser.add_argument("--input_mesh", type=str)
    parser.add_argument("--output_dir", type=dict, default={"rendering":"..\\..\\output", "sketch":"..\\..\\output"})
    parser.add_argument("--fov", type=int, default=50)
    parser.add_argument("--emitter_samples", type=int, default=4)
    args = parser.parse_args(args)
    diff_args(args)


if __name__ == '__main__':
    params = [
        '--type', 'rendering',
        '--input_mesh', '..\\..\\resources\\ABC\\abc_0028_stl2_v00\\00280001\\00280001_57fbd625027b01109e830e03_trimesh_000.ply',
    ]
    main(params)