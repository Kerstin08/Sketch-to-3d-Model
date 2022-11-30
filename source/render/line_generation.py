import argparse
import time

from pathlib import Path
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv
import numpy as np

from source.render.render_direct import Direct

class LineGen():
    def __init__(self, output_dirs, fov, width=1024, height=1024, emitter_samples=4):
        self.output_dirs = output_dirs
        self.renderer = Direct(output_dirs, fov, width, height, emitter_samples)

    def create_lines(self, input_path, output_name=""):
        for key, value in self.output_dirs.items():
            if not os.path.exists(value):
                os.mkdir(value)
        if len(output_name) <= 0:
            filename = Path(input_path)
            output_name = filename.stem

        scene = self.renderer.create_scene(input_path, output_name)
        path_temp = self.renderer.render(scene, input_path)

        filename = output_name + "_sketch.png"
        path = os.path.join(self.output_dirs["sketch"], filename)
        img = None
        time_to_wait = 10
        while img is None:
            time.sleep(1)
            img = cv.imread(path_temp, 0)
            time_to_wait -= 1
            if time_to_wait < 0:
                print("Temp rendering could not be generated!")
                return

        gaussian = cv.GaussianBlur(img, (5, 5), 0)
        edges = cv.Canny(gaussian, 10, 130)
        kernel = np.ones((3, 3), np.uint8)
        img_dilation = cv.dilate(edges, kernel)
        dsize = (256, 256)
        output = cv.resize(cv.bitwise_not(img_dilation), dsize)
        img_binary = cv.threshold(output, 250, 255, cv.THRESH_BINARY)[1]

        cv.imwrite(path, img_binary)
        os.remove(path_temp)