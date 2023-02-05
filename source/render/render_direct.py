from source.render.render_base import Render
import source.render.mi_create_scenedesc as create_scenedesc

import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb')

class Direct(Render):
    def __init__(self, views, fov=50, dim=256, emitter_samples=4):
        Render.__init__(self, views, fov, dim)
        self._integrator_lodaded = self.__load_integrator(emitter_samples)

    def __load_integrator(self, emiiter_samples):
        load_integrator = create_scenedesc.create_integrator_direct(emiiter_samples)
        return mi.load_dict(load_integrator)

    def render(self, scene, input_path, seed=0, spp=256,):
        img = mi.render(scene, seed=seed, spp=spp, integrator=self._integrator_lodaded)
        # If mesh has invalid mesh vertices, rendering contains nan.
        if dr.any(dr.isnan(img)):
            print("Rendered image includes invalid data! Vertex normals in input model " + input_path + " might be corrupt.")
            return
        return img