import math
import os.path
import numpy as np
from pathlib import Path
from abc import abstractmethod

import source.render.mi_create_scenedesc as create_scenedesc

import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T
mi.set_variant('cuda_ad_rgb')

class Render:
    # Load integrators provided by renderer at the beginning of the
    def __init__(self, output_dirs, fov=50, dim=256):
        self.output_dirs = output_dirs
        self.camera = self.__load_camera(fov, dim)
        self.emitter = self.__load_emitter()
        self.output_name=""

    def __load_emitter(self):
        return create_scenedesc.create_emitter()

    # center is assumed to be at 0,0,0, see mesh_preprocess_operations.py translate_to_origin
    # bounding box diagonal is assumed to be 1, see mesh_preprocess_operations.py normalize_mesh
    def __load_camera(self, fov, dim):
        distance = math.tan(math.radians(fov)) / 1.75
        near_distance = distance
        far_distance = distance * 4

        centroid = np.array([distance, -distance, distance])
        # use only quadratic images
        return create_scenedesc.create_camera(T.look_at(target=(0.0, 0.0, 0.0),
                                                      origin=tuple(centroid),
                                                      up=(0, 0, 1),
                                                      ),
                                                    fov, near_distance, far_distance,
                                                    dim, dim)

    @abstractmethod
    def render(self, scene, input_path):
        pass

    def create_scene(self, input_path, output_name):
        datatype = input_path.rsplit(".", 1)[1]
        if datatype != "obj" and datatype != "ply":
            print("Given datatype cannot be processed, must be either obj or ply type.")
            return
        shape = create_scenedesc.create_shape(input_path, datatype)

        if len(output_name) <= 0:
            filename = Path(input_path)
            self.output_name = (filename.stem)
        else:
            self.output_name = output_name

        for key, value in self.output_dirs.items():
            if not os.path.exists(value):
                os.mkdir(value)

        scene_desc = {"type": "scene", "shape": shape, "camera": self.camera, "emitter": self.emitter}
        # Sometimes mesh data is not incorrect and could not be loaded
        try:
            scene = mi.load_dict(scene_desc)
        except Exception as e:
            print("Exception occured in " + shape["filename"])
            print(e)
            return
        return scene