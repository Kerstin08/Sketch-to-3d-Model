def create_shape(input_mesh, datatype):
    shape = {
        "type": datatype,
        "filename": input_mesh,
        "bsdf": {
            "type": "diffuse",
            "reflectance" : {
                "type" : "rgb",
                "value" : [0.5,0.5,0.5],
                # value for kato
                #"value": [0.8, 1, 1],
            }
        }
    }
    return shape

def create_integrator_depth():
    integrator = {
        'type': 'depth_reparam'
    }
    return integrator

def create_integrator_normal():
    integrator = {
        'type': 'normal_reparam'
    }
    return integrator

def create_integrator_silhouette():
    integrator = {
        'type': 'silhouette_reparam'
    }
    return integrator

def create_integrator_direct(emitter_samples):
    integrator = {"type": "direct", "emitter_samples": emitter_samples}
    return integrator

def create_emitter():
    # kato emitter
    #emitter = {
    #    'type': 'point',
    #    'position': [1.0, -1.0, 1.0],
    #    'intensity': {
    #        'type': 'spectrum',
    #        'value': 5.0,
    #    }
    #}
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
                  'sample_border': True
              },
              "sampler":
                  {"type": "independent",
                   "sample_count": 128}}

    return camera