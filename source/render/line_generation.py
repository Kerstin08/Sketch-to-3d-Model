import argparse
import time

from pathlib import Path
import os

from source.render import save_renderings

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv
import numpy as np

from source.render.render_direct import Direct

class LineGen():
    def __init__(self, views, fov=50, dim_int_width=1024, dim_final=256, emitter_samples=4):
        self.renderer = Direct(views, fov, dim_int_width, emitter_samples)
        self.dim_final = dim_final

    def create_scenes(self, input_path):
        scenes = self.renderer.create_scene(input_path)
        return scenes

    def create_line_images(self, scene, input_path):
        img_direct = self.renderer.render(scene, input_path)
        img_direct = (img_direct * 255).astype(np.uint8)
        if img_direct is None:
            return

        gaussian = cv.GaussianBlur(img_direct, (5, 5), 0)
        edges = cv.Canny(gaussian, 10, 130)
        kernel = np.ones((3, 3), np.uint8)
        img_dilation = cv.dilate(edges, kernel)
        dsize = (self.dim_final, self.dim_final)
        output = cv.resize(cv.bitwise_not(img_dilation), dsize)
        img_binary = cv.threshold(output, 250, 255, cv.THRESH_BINARY)[1]
        return img_binary
