import os

from source.render.render_base import Render
import source.render.mi_create_scenedesc as create_scenedesc

import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb')

class Direct(Render):
    def __init__(self, output_dirs, fov=50, dim=256, emitter_samples=4):
        Render.__init__(self, output_dirs, fov, dim)
        self._integrator_lodaded = self.__load_integrator(emitter_samples)

    def __load_integrator(self, emiiter_samples):
        load_integrator = create_scenedesc.create_integrator_direct(emiiter_samples)
        return mi.load_dict(load_integrator)

    def render(self, scene, input_path):
        img = mi.render(scene, seed=0, spp=256, integrator=self._integrator_lodaded)
        # If mesh has invalid mesh vertices, rendering contains nan.
        if dr.any(dr.isnan(img)):
            print(
                "Rendered image " + self.output_name + " includes invalid data! Vertex normals in input model " + input_path + " might be corrupt.")
            return
        bitmap = mi.util.convert_to_bitmap(img)
        filename = self.output_name + "_rendering.png"
        output_dir = self.output_dirs["rendering"]
        path = os.path.join(output_dir, filename)
        mi.util.write_bitmap(path, bitmap)
        return path