# setup scene description for rendering process
from mitsuba.scalar_rgb import Transform4f


def create_shape(
        input_mesh: str,
        datatype: str,
        use_nmr: bool = False
) -> dir:
    # Values for neural mesh renderer
    if use_nmr:
        rgb_val = [0.8, 1, 1]
    else:
        rgb_val = [0.5, 0.5, 0.5]
    shape = {
        'type': datatype,
        'filename': input_mesh,
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': rgb_val
            }
        }
    }
    return shape


def create_shape_kato(
        input_mesh: str,
        datatype: str
) -> dir:
    shape = {
        'type': datatype,
        'filename': input_mesh,
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.8, 1, 1],
            }
        }
    }
    return shape


def create_integrator_depth() -> dir:
    integrator = {
        'type': 'depth_reparam'
    }
    return integrator


def create_integrator_normal() -> dir:
    integrator = {
        'type': 'normal_reparam'
    }
    return integrator


def create_integrator_silhouette() -> dir:
    integrator = {
        'type': 'silhouette_reparam'
    }
    return integrator


def create_integrator_direct(
        emitter_samples: int
) -> dir:
    integrator = {'type': 'direct', 'emitter_samples': emitter_samples}
    return integrator


def create_emitter(
        use_nmr: bool = False
) -> dir:
    # Values for neural mesh renderer
    if use_nmr:
        print("Emitter position does only work for used view (255, 30), reposition light for different views")
        emitter = {
            'type': 'point',
            'position': [1.0, -1.0, 1.0],
            'intensity': {
                'type': 'spectrum',
                'value': 5.0,
            }
        }
    else:
        emitter = {
            'type': 'constant'
        }
    return emitter


def create_camera(
        transform: Transform4f,
        fov: int,
        near: float,
        far: float,
        width: int,
        height: int
) -> dir:
    camera = {'type': 'perspective',
              'to_world': transform,
              'fov': fov,
              'near_clip': near,
              'far_clip': far,
              'film': {
                  'type': 'hdrfilm',
                  'width': width,
                  'height': height,
                  'rfilter': {'type': 'gaussian'},
                  'pixel_format': 'rgb',
                  'sample_border': True
              },
              'sampler':
                  {'type': 'independent',
                   'sample_count': 128}}

    return camera
