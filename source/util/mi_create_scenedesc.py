def create_shape(input_mesh, transform):
    shape = {
        "type": "obj",
        "filename": input_mesh,
        "to_world": transform,
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

# Todo: define near and far clip in order to compute depth correctly
def create_camera(transform):
    camera = {"type": "perspective",
              "to_world": transform,
              "near_clip": 1,
              "far_clip": 2,
              "film": {
                  "type": "hdrfilm",
                  "width": 256,
                  "height": 256,
                  'rfilter': {'type': 'gaussian'},
                  "pixel_format": "rgb"
              },
              "sampler":
                  {"type": "independent",
                   "sample_count": 128}}

    return camera