import os
import numpy as np
from PIL import Image

from source.render.render_base import Render
from source.util import data_type
from source.util import OpenEXR_utils
import source.render.normal_reparam_integrator
import source.render.depth_reparam_integrator
import source.render.mi_create_scenedesc as create_scenedesc

import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb')

class AOV(Render):
    def __init__(self, output_dirs, aovs, fov=50, dim=256, create_debug_pngs=True):
        Render.__init__(self, output_dirs, fov, dim)
        self._depth_integrator = self.__load_depth_integrator()
        self._normal_integrator = self.__load_normal_integrator()
        self.create_debug_pngs = create_debug_pngs
        self.aovs = aovs

    def __load_depth_integrator(self):
        depth_integrator = create_scenedesc.create_integrator_depth()
        return mi.load_dict(depth_integrator)

    def __load_normal_integrator(self):
        normal_integrator = create_scenedesc.create_integrator_normal()
        return mi.load_dict(normal_integrator)

    def render(self, scene, input_path):
        paths = []
        if "depth" in self.aovs.values():
            paths.append(self.render_depth(scene, input_path))
        if "sh_normal" in self.aovs.values():
            paths.append(self.render_normal(scene, input_path))
        return paths

    def render_depth(self, scene, input_path):
        img = mi.render(scene, seed=0, spp=256, integrator=self._depth_integrator)
        # If mesh has invalid mesh vertices, rendering contains nan.
        if dr.any(dr.isnan(img)):
            print(
                "Rendered image " + self.output_name + " includes invalid data! Vertex normals in input model " + input_path + " might be corrupt.")
            return

        img = img[:, :, 0]
        mask = img.array < 1.5
        curr_min_val = dr.min(img)
        masked_img = dr.select(mask,
                               img.array,
                               0.0)
        curr_max_val = dr.max(masked_img)
        wanted_range_min, wanted_range_max = 0.0, 0.75
        depth = dr.select(mask,
                          (img.array - curr_min_val) * (
                                  (wanted_range_max - wanted_range_min) / (
                                  curr_max_val - curr_min_val)) + wanted_range_min,
                          1.0)
        np_depth = np.array(depth).reshape((256, 256))

        filename = self.output_name + "_depth.exr"
        output_dir = self.output_dirs['dd.y']
        path = os.path.join(output_dir, filename)
        OpenEXR_utils.writeImage(np_depth, data_type.Type.depth, path)

        if self.create_debug_pngs:
            output_dir_png = self.output_dirs['dd_png']
            png_filename = self.output_name + "_depth.png"
            path_debug = os.path.join(output_dir_png, png_filename)
            # Use pil instead of standard mi.util.write_bitmap for png since mi automatically applies gamma correction when
            # writing png files
            Image.fromarray((np_depth * 255).astype('uint8'), mode='L').save(path_debug)
        return path

    def render_normal(self, scene, input_path):
        img = mi.render(scene, seed=0, spp=256, integrator=self._normal_integrator)
        # If mesh has invalid mesh vertices, rendering contains nan.
        if dr.any(dr.isnan(img)):
            print(
                "Rendered image " + self.output_name + " includes invalid data! Vertex normals in input model " + input_path + " might be corrupt.")
            return
        np_img = np.array(img)
        filename = self.output_name + "_normal.exr"
        output_dir = self.output_dirs['nn']
        path = os.path.join(output_dir, filename)
        OpenEXR_utils.writeImage(np_img, data_type.Type.normal, path)

        if self.create_debug_pngs:
            output_dir_png = self.output_dirs['nn_png']
            png_filename = self.output_name + "_normal.png"
            path_debug = os.path.join(output_dir_png, png_filename)
            # Use pil instead of standard mi.util.write_bitmap for png since mi automatically applies gamma correction when
            # writing png files
            Image.fromarray(((np_img + 1.0) * 127).astype('uint8'), mode='RGB').save(path_debug)
        return path