# renderer for line generation using mitsuba direct renderer and opencv
import typing

import cv2 as cv
import mitsuba as mi
import numpy
import numpy as np

from source.render.render_direct import Direct


class LineGen:
    def __init__(
            self,
            views: typing.Sequence[typing.Tuple[int, int]],
            fov: int = 50,
            dim_int_width: int = 1024,
            dim_final: int = 256,
            emitter_samples: int = 4
    ):
        self.renderer = Direct(views, fov, dim_int_width, emitter_samples)
        self.dim_final = dim_final

    def create_scenes(
            self,
            input_path: str
    ) -> list[mi.Scene] | None:
        scenes = self.renderer.create_scene(input_path)
        return scenes

    def create_line_images(
            self,
            scene: mi.Scene,
            input_path: str,
            spp: int = 256
    ) -> numpy.ndarray | None:
        img_direct = np.array(self.renderer.render(scene, input_path, spp=spp))
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
        del img_direct
        return img_binary
