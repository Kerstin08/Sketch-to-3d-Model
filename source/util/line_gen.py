import argparse
import time

from matplotlib import pyplot as plt
import source.util.mi_render as render
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv

def run(type, input_mesh, output_dirs, emitter_samples):
    for key, value in output_dirs.items():
        if not os.path.exists(value):
            os.mkdir(value)

    render.run(type, input_mesh, output_dirs, emitter_samples)
    mesh_name = (input_mesh.rsplit("\\", 1)[-1]).rsplit(".", 1)[0]
    filename_temp = mesh_name + "_rendering.png"
    path_temp = os.path.join(output_dirs["rendering"], filename_temp)

    filename = mesh_name + "_sketch.png"
    path = os.path.join(output_dirs["sketch"], filename)
    img = None
    max_time_to_wait = 10
    while img is None:
        time.sleep(1)
        img = cv.imread(path_temp, 0)
        if max_time_to_wait < 0:
            raise RuntimeError("Temp rendering is not generated!")


    edges = cv.Canny(img, 10, 130)
    cv.imwrite(path, cv.bitwise_not(edges))
    os.remove(path_temp)

def diff_args(args):
    run(args.type, args.input_mesh, args.output_dir, args.emitter_samples)

def main(args):
    parser = argparse.ArgumentParser(prog="scene_rendering")
    parser.add_argument("--type", type=str, help="use \"aov\", \"rendering\" or \"combined\"")
    parser.add_argument("--input_mesh", type=str)
    parser.add_argument("--output_dir", type=dict, default={"rendering":"..\\..\\output", "sketch":"..\\..\\output"})
    parser.add_argument("--emitter_samples", type=int, default=4)
    args = parser.parse_args(args)
    diff_args(args)


if __name__ == '__main__':
    params = [
        '--type', 'rendering',
        '--input_mesh', '..\\..\\resources\\meshes\\teapot.ply',
    ]
    main(params)