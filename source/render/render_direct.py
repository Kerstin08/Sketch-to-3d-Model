# renderer that utilizes mitsuba direct renderer
import typing

from source.render.render_base import Render
import source.render.mi_create_scenedesc as create_scenedesc

import mitsuba as mi
import drjit as dr

mi.set_variant('cuda_ad_rgb')


class Direct(Render):
    def __init__(
            self,
            views: typing.Sequence[typing.Tuple[int, int]],
            fov: int = 50,
            dim: int = 256,
            emitter_samples: int = 4,
            nmr: bool = False
    ):
        Render.__init__(self, views, fov, dim, nmr)
        self._integrator_lodaded = self.__load_integrator(emitter_samples)

    def __load_integrator(
            self,
            emitter_samples: int
    ) -> mi.Integrator:
        load_integrator = create_scenedesc.create_integrator_direct(emitter_samples)
        return mi.load_dict(load_integrator)

    def render(
            self,
            scene: mi.Scene,
            input_path: str,
            seed: int = 0,
            spp: int = 256
    ) -> mi.TensorXf | None:
        img = mi.render(scene, seed=seed, spp=spp, integrator=self._integrator_lodaded)
        # If mesh has invalid mesh vertices, rendering contains nan.
        if dr.any(dr.isnan(img)):
            print(
                "Rendered image includes invalid data! Vertex normals in input model " + input_path + "might be "
                                                                                                      "corrupt.")
            return
        return img
