def create_shape(input_mesh, datatype):
    shape = {
        "type": datatype,
        "filename": input_mesh,
        "bsdf": {
            "type": "diffuse",
            "reflectance" : {
                "type" : "rgb",
                "value" : [0.8, 0.8, 0.8],
            }
        }
    }
    return shape

def create_intergrator_aov(aovs):
    integrator_aovs = ""
    for aov in aovs:
        integrator_aovs += str(aov + ":" + aovs[aov] + ",")
    integrator = {"type": "aov", "aovs": integrator_aovs}
    return integrator

def create_integrator_direct(emitter_samples):
    integrator = {"type": "direct", "emitter_samples": emitter_samples}
    return integrator

def create_emitter():
    emitter = {
        "type": "constant"
    }
    return emitter

def create_camera(transform, fov, near, far):
    camera = {"type": "perspective",
              "to_world": transform,
              "fov": fov,
              "near_clip": near,
              "far_clip": far,
              "film": {
                  "type": "hdrfilm",
                  "width": 256,
                  "height": 256,
                  'rfilter': {'type': 'gaussian'},
                  "pixel_format": "rgb",
              },
              "sampler":
                  {"type": "independent",
                   "sample_count": 128}}

    return camera