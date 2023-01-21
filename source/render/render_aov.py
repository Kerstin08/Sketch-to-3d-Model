import mitsuba as mi
import drjit as dr

from source.render.render_base import Render
import source.render.normal_reparam_integrator
import source.render.depth_reparam_integrator
import source.render.silhouette_reparam_integrator
import source.render.mi_create_scenedesc as create_scenedesc

mi.set_variant('cuda_ad_rgb')

class AOV(Render):
    def __init__(self, views, fov=50, dim=256):
        Render.__init__(self, views, fov, dim)
        self._depth_integrator = self.__load_depth_integrator()
        self._normal_integrator = self.__load_normal_integrator()
        self._silhouette_integrator = self.__load_silhouette_integrator()

    def __load_depth_integrator(self):
        depth_integrator = create_scenedesc.create_integrator_depth()
        return mi.load_dict(depth_integrator)

    def __load_normal_integrator(self):
        normal_integrator = create_scenedesc.create_integrator_normal()
        return mi.load_dict(normal_integrator)

    def __load_silhouette_integrator(self):
        silhouette_integrator = create_scenedesc.create_integrator_silhouette()
        return mi.load_dict(silhouette_integrator)

    def render_depth(self, scene, input_path, seed=0, spp=256, params=None):
        img = mi.render(scene, params, seed=seed, spp=spp, integrator=self._depth_integrator)
        # If mesh has invalid mesh vertices, rendering contains nan.
        if dr.any(dr.isnan(img)):
            print("Rendered image  includes invalid data! Vertex normals in input model " + input_path + " might be corrupt.")
            return

        with dr.suspend_grad():
            single_channel_depth = img[:, :, 0]
            mask = single_channel_depth.array < (self.far_distance - self.near_distance)

        depth = dr.select(mask,
                        img[:, :, 0].array / (self.far_distance - self.near_distance),
                        1)
        depth_tens = mi.TensorXf(depth, shape=(256, 256))
        return depth_tens

    def render_normal(self, scene, input_path, seed=0, spp=256, params=None):
        img = mi.render(scene, params, seed=seed, spp=spp, integrator=self._normal_integrator)
        # If mesh has invalid mesh vertices, rendering contains nan.
        if dr.any(dr.isnan(img)):
            print("Rendered image includes invalid data! Vertex normals in input model " + input_path + " might be corrupt.")
            return
        return img

    def render_silhouette(self, scene, input_path, seed=0, spp=256, params=None):
        img = mi.render(scene, params, seed=seed, spp=spp, integrator=self._silhouette_integrator)
        # If mesh has invalid mesh vertices, rendering contains nan.
        if dr.any(dr.isnan(img)):
            print("Rendered image includes invalid data! Vertex normals in input model " + input_path + " might be corrupt.")
            return
        return img