import numpy as np
import mitsuba as mi
import drjit as dr

from source.render.render_base import Render
import source.render.normal_reparam_integrator
import source.render.depth_reparam_integrator
import source.render.mi_create_scenedesc as create_scenedesc

mi.set_variant('cuda_ad_rgb')

class AOV(Render):
    def __init__(self, views, aovs, fov=50, dim=256):
        Render.__init__(self, views, fov, dim)
        self._depth_integrator = self.__load_depth_integrator()
        self._normal_integrator = self.__load_normal_integrator()
        self.aovs = aovs

    def __load_depth_integrator(self):
        depth_integrator = create_scenedesc.create_integrator_depth()
        return mi.load_dict(depth_integrator)

    def __load_normal_integrator(self):
        normal_integrator = create_scenedesc.create_integrator_normal()
        return mi.load_dict(normal_integrator)

    def render_depth(self, scene, input_path):
        img = mi.render(scene, seed=0, spp=256, integrator=self._depth_integrator)
        # If mesh has invalid mesh vertices, rendering contains nan.
        if dr.any(dr.isnan(img)):
            print("Rendered image  includes invalid data! Vertex normals in input model " + input_path + " might be corrupt.")
            return

        img = img[:, :, 0]
        mask = img.array < (self.far_distance - self.near_distance)
        depth = dr.select(mask,
                          img.array / (self.far_distance - self.near_distance),
                          1)
        np_depth = np.array(depth).reshape((256, 256))
        return np_depth

    def render_normal(self, scene, input_path):
        img = mi.render(scene, seed=0, spp=256, integrator=self._normal_integrator)
        # If mesh has invalid mesh vertices, rendering contains nan.
        if dr.any(dr.isnan(img)):
            print("Rendered image includes invalid data! Vertex normals in input model " + input_path + " might be corrupt.")
            return
        np_img = np.array(img)
        return np_img