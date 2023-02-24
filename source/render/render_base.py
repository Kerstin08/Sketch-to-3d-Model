# base class for all renderers
import math
import typing

import numpy as np
import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T

import source.render.mi_create_scenedesc as create_scenedesc

mi.set_variant('cuda_ad_rgb')


class Render:
    # Load integrators provided by renderer at the beginning of the
    def __init__(
            self,
            views: typing.Sequence[typing.Tuple[int, int]],
            fov: int = 50,
            dim: int = 256,
            nmr: bool = False
    ):
        self.distance = math.floor(math.tan(math.radians(fov)) / 1.75 * 10) / 10
        self.near_distance = self.distance
        self.far_distance = self.distance * 3
        self.cameras = self.__load_cameras(views, fov, dim)
        self.emitter = self.__load_emitter(nmr)
        self.dim = dim
        self.nmr = nmr

    def __load_emitter(
            self,
            nmr: bool
    ) -> dict:
        return create_scenedesc.create_emitter(nmr)

    # center is assumed to be at 0,0,0, see mesh_preprocess_operations.py translate_to_origin
    # bounding box diagonal is assumed to be 1, see mesh_preprocess_operations.py normalize_mesh
    def __load_cameras(
            self,
            views: typing.Sequence[typing.Tuple[int, int]],
            fov: int,
            dim: int
    ) -> typing.Sequence[dir]:
        cameras = []
        radius = self.distance * 2
        for view in views:
            long, lat = view
            long = math.pi * long / 180
            lat = math.pi * lat / 180
            x = math.cos(long) * math.cos(lat)
            y = math.sin(long) * math.cos(lat)
            z = math.sin(lat)
            centroid = np.around(np.array([-x, y, z]) * radius, 5)
            up = np.array([0, 0, 1 - z])
            # Up vector in direction other than z is not needed despite for zenith point, otherwise rendering returns
            # nothing
            if np.sum(up) < 1e-5:
                up = np.array([0, -1, 0])
            scene_desc = create_scenedesc.create_camera(T.look_at(target=(0.0, 0.0, 0.0),
                                                                  origin=tuple(centroid),
                                                                  up=tuple(up),
                                                                  ),
                                                        fov, self.near_distance, self.far_distance,
                                                        dim, dim)
            cameras.append(scene_desc)
        return cameras

    def create_scene(
            self,
            input_path: str
    ) -> list[mi.Scene] | None:
        datatype = input_path.rsplit('.', 1)[1]
        if datatype != 'obj' and datatype != 'ply':
            print("Given datatype cannot be processed, must be either obj or ply type.")
            return
        shape = create_scenedesc.create_shape(input_path, datatype, self.nmr)

        scenes = []
        for camera in self.cameras:
            scene_desc = {'type': 'scene', 'shape': shape, 'camera': camera, 'emitter': self.emitter}
            # Sometimes mesh data is not incorrect and could not be loaded
            try:
                scenes.append(mi.load_dict(scene_desc))
            except Exception as e:
                print("Exception occured in " + shape["filename"])
                print(e)
                return
        return scenes
