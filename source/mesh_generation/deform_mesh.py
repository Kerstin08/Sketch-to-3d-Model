import drjit as dr
import mitsuba as mi
import math
import numpy as np
from mitsuba.scalar_rgb import Transform4f as T

from torch.utils.tensorboard import SummaryWriter

mi.set_variant('cuda_ad_rgb')

def offsetVerts(params, opt):
    opt['deform_verts'] = dr.clamp(opt['deform_verts'], -0.5, 0.5)
    trafo = mi.Transform4f.translate(opt['deform_verts'].x)
    params['test.vertex_positions'] = dr.ravel(trafo @ initial_vertex_positions)
    params.update()

writer = SummaryWriter("..\\..\\logs")
aov_integrator = {
    'type': 'normal_depth',
    'aovs': "nn:sh_normal,dd.y:depth"
}

integrator = {
        'type': 'direct'
}


distance = math.tan(math.radians(60))/2
centroid = np.array([distance, distance, -distance])
# refscene
ref_scene = mi.load_dict({
    'type': 'scene',
    'sensor':  {
        'type': 'perspective',
        'to_world': T.look_at(
                        origin=tuple(centroid),
                        target=(0, 0, 0),
                        up=(0, 1, 0)
                    ),
        'fov': 60,
        'film': {
            'type': 'hdrfilm',
            'width': 64,
            'height': 64,
            'rfilter': { 'type': 'gaussian' },
            'sample_border': True
        },
    },
    'bunny': {
        'type': 'ply',
        'filename': '..\\..\\resources\\topology_meshes\\ref.ply',
        'bsdf': {
            'type': 'diffuse',
            'reflectance': { 'type': 'rgb', 'value': (0.3, 0.3, 0.75) },
        },
    },
    'light': {
        'type': 'obj',
        'filename': '..\\..\\resources\\meshes\\sphere.obj',
        'emitter': {
            'type': 'area',
            'radiance': {'type': 'rgb', 'value': [1e3, 1e3, 1e3]}
        },
        'to_world': T.translate([2.5, 2.5, 7.0]).scale(0.25)
    }
})

# object
scene = mi.load_dict({
    'type': 'scene',
    'sensor':  {
        'type': 'perspective',
        'to_world': T.look_at(
                        origin=tuple(centroid),
                        target=(0, 0, 0),
                        up=(0, 1, 0)
                    ),
        'fov': 60,
        'film': {
            'type': 'hdrfilm',
            'width': 64,
            'height': 64,
            'rfilter': { 'type': 'gaussian' },
            'sample_border': True
        },
    },
    'test': {
        'type': 'ply',
        'filename': '..\\..\\resources\\topology_meshes\\test.ply',
        'bsdf': {
            'type': 'diffuse',
            'reflectance': { 'type': 'rgb', 'value': (0.3, 0.3, 0.75) },
        },
    },
    'light': {
        'type': 'obj',
        'filename': '../../resources/meshes/sphere.obj',
        'emitter': {
            'type': 'area',
            'radiance': {'type': 'rgb', 'value': [1e3, 1e3, 1e3]}
        },
        'to_world': T.translate([2.5, 2.5, 7.0]).scale(0.25)
    }
})


img_ref = mi.render(ref_scene, seed=0, spp=1024, integrator=mi.load_dict(integrator))
mi.util.convert_to_bitmap(img_ref)
#bitmap = mi.Bitmap(img_ref_all, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
#channels = dict(bitmap.split())
#img_ref = channels['pos']

params = mi.traverse(scene)
print(params)
initial_vertex_positions = dr.unravel(mi.Point3f, params['test.vertex_positions'])
initial_vertex_normals = dr.unravel(mi.Point3f, params['test.vertex_normals'])

opt = mi.ad.Adam(lr=0.025)
vertex_count = params['test.vertex_count']
opt['deform_verts'] = dr.full(mi.Point3f, 0, vertex_count)

img_init = mi.render(scene, seed=0, spp=1024, integrator=mi.load_dict(integrator))

ref_bpm = mi.util.convert_to_bitmap(img_ref)
ref_np = np.transpose(np.array(ref_bpm), (2, 0, 1))
writer.add_image('ref_images', ref_np)

for it in range(1000):
    offsetVerts(params, opt)

    img = mi.render(scene, params, integrator=mi.load_dict(integrator), seed=it, spp=16)

    loss = dr.sum(dr.sqr(img - img_ref)) / len(img)

    dr.backward(loss)

    opt.step()

    if it%100 == 0:
        current_bpm = mi.util.convert_to_bitmap(img)
        current_np = np.transpose(np.array(current_bpm), (2, 0, 1))
        writer.add_image('current_np' + str(it), current_np)

    writer.add_scalar("loss", loss[0], it)
    print(f"Iteration {it:02d}: error={loss[0]:6f}", end='\r')

writer.close()
mesh = mi.Mesh(
    "deformed_sphere",
    vertex_count=vertex_count,
    face_count=params['test.face_count'],
    has_vertex_normals=True,
    has_vertex_texcoords=False,
)

mesh_params = mi.traverse(mesh)
mesh_params["vertex_positions"] = dr.ravel(params['test.vertex_positions'])
mesh_params["faces"] = dr.ravel(params['test.faces'])
mesh_params.update()
mesh.write_ply("../../output/deformed_sphere.ply")