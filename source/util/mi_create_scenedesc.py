def create_shape(input_mesh, datatype):
    shape = {
        "type": datatype,
        "filename": input_mesh,
        "bsdf": {
            "type": "diffuse",
            "reflectance" : {
                "type" : "rgb",
                "value" : [0.5, 0.5, 0.5],
            }
        }
    }
    return shape

def create_integrator_aov():
    integrator = {
        'type': 'depth_reparam'
    }
    return integrator

def create_integrator_direct(emitter_samples):
    integrator = {"type": "direct", "emitter_samples": emitter_samples}
    return integrator

def create_emitter():
    emitter = {
        "type": "constant"
    }
    return emitter

def create_camera(transform, fov, near, far, width, height):
    camera = {"type": "perspective",
              "to_world": transform,
              "fov": fov,
              "near_clip": near,
              "far_clip": far,
              "film": {
                  "type": "hdrfilm",
                  "width": width,
                  "height": height,
                  'rfilter': {'type': 'gaussian'},
                  "pixel_format": "rgb",
              },
              "sampler":
                  {"type": "independent",
                   "sample_count": 128}}

    return camera