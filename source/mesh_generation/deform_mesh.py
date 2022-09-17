import drjit as dr
import mitsuba as mi
import math
import numpy as np
import torch
import torchvision
from mitsuba.scalar_rgb import Transform4f as T
import normal_reparam_integrator
import depth_reparam_integrator

from torch.utils.tensorboard import SummaryWriter

mi.set_variant('cuda_ad_rgb')

def write_output_renders(render_depth, render_normal, image_name):
    ref_bpm_normal = mi.util.convert_to_bitmap(render_normal)
    ref_np_normal = np.transpose(np.array(ref_bpm_normal), (2, 0, 1))
    ref_bpm_depth = mi.util.convert_to_bitmap(render_depth)
    ref_np_depth = np.transpose(np.array(ref_bpm_depth), (2, 0, 1))
    ref_images = np.stack([ref_np_normal, ref_np_depth])
    writer.add_images(image_name, ref_images)

def get_edge_dist(vertice_positions, edge_vert_indices):
    edge_lengths = np.empty(0)
    for pt1_index, pt2_index in edge_vert_indices:
        x = vertice_positions[0][pt1_index] - vertice_positions[0][pt2_index]
        y = vertice_positions[1][pt1_index] - vertice_positions[1][pt2_index]
        z = vertice_positions[2][pt1_index] - vertice_positions[2][pt2_index]
        edge_lengths.append(dr.sqrt(dr.sqr(x) + dr.sqr(y) + dr.sqr(z)))
    return edge_lengths

def offset_verts(params, opt):
    opt['deform_verts'] = dr.clamp(opt['deform_verts'], -0.5, 0.5)
    trafo = mi.Transform4f.translate(opt['deform_verts'].x)
    params['test.vertex_positions'] = dr.ravel(trafo @ initial_vertex_positions)
    params.update()


num_epochs = 200
learning_rate = 0.001
weight_normal = 0.5
weight_depth = 0.5
weight_edge = 0.9
weight_smooth = 0.01

writer = SummaryWriter("..\\..\\logs")

depth_integrator = {
        'type': 'depth_reparam'
}
normal_integrator = {
        'type': 'normal_reparam'
}

fov = 60
distance = math.tan(math.radians(fov))
far_distance = distance * 2
near_distance = distance / 2
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
        'fov': fov,
        "near_clip": near_distance,
        "far_clip": far_distance,
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
        'fov': fov,
        "near_clip": near_distance,
        "far_clip": far_distance,
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

depth_integrator_lodaded=mi.load_dict(depth_integrator)
normal_integrator_lodaded=mi.load_dict(normal_integrator)
normal_img_ref = mi.render(ref_scene, seed=0, spp=1024, integrator=normal_integrator_lodaded)
depth_img_ref = mi.render(ref_scene, seed=0, spp=1024, integrator=depth_integrator_lodaded)
write_output_renders(depth_img_ref, normal_img_ref, "ref_images")

normal_img_init = mi.render(scene, seed=0, spp=1024, integrator=normal_integrator_lodaded)
depth_img_init = mi.render(scene, seed=0, spp=1024, integrator=depth_integrator_lodaded)
write_output_renders(depth_img_init, normal_img_init, "init_images")

params = mi.traverse(scene)
print(params)
initial_vertex_positions = dr.unravel(mi.Point3f, params['test.vertex_positions'])
initial_faces = dr.unravel(mi.Point3i, params['test.faces'])

edge_vert_indices=[]
for i in range(len(initial_faces[0])):
    x = initial_faces[0][i]
    y = initial_faces[1][i]
    z = initial_faces[2][i]
    xy = (x, y) if x < y else (y, x)
    if xy not in edge_vert_indices:
        edge_vert_indices.append(xy)
    yz = (y, z) if y < z else (z, y)
    if yz not in edge_vert_indices:
        edge_vert_indices.append(yz)
    zx = (z, x) if z < x else (x, z)
    if zx not in edge_vert_indices:
        edge_vert_indices.append(zx)
initial_edge_lengths = get_edge_dist(initial_vertex_positions, edge_vert_indices)

opt = mi.ad.Adam(lr=learning_rate)
vertex_count = params['test.vertex_count']
opt['deform_verts'] = dr.full(mi.Point3f, 0, vertex_count)

for epoch in range(num_epochs):
    offset_verts(params, opt)

    normal_img = mi.render(scene, params, seed=epoch, spp=16, integrator=normal_integrator_lodaded)
    depth_img = mi.render(scene, params, seed=epoch, spp=16, integrator=depth_integrator_lodaded)
    if epoch%10 == 0:
        image_name = "deformed_images" + str(epoch)
        write_output_renders(depth_img, normal_img, image_name)

    depth_loss = dr.sum(dr.sqr(depth_img - depth_img_ref)) / len(depth_img)
    normal_loss = dr.sum(dr.sqr(normal_img - normal_img_ref)) / len(normal_img)

    current_vertex_positions = dr.unravel(mi.Point3f, params['test.vertex_positions'])
    current_edge_lengths = get_edge_dist(current_vertex_positions, edge_vert_indices)
    edge_loss = 1/len(initial_edge_lengths) * dr.sum(dr.sqr(np.subtract(initial_edge_lengths, current_edge_lengths)))

    loss = depth_loss * weight_depth + normal_loss * weight_normal + edge_loss * weight_edge
    dr.backward(loss)

    opt.step()

    writer.add_scalar("loss", loss[0], epoch)
    print(f"Epochs {epoch:02d}: error={loss[0]:6f} loss_normal={normal_loss[0]:6f} loss_depth={depth_loss[0]:6f}", end='\r')

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