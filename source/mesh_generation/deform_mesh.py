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
    edge_lengths = []
    for pt1_index, pt2_index in edge_vert_indices:
        x = vertice_positions[0][pt1_index] - vertice_positions[0][pt2_index]
        y = vertice_positions[1][pt1_index] - vertice_positions[1][pt2_index]
        z = vertice_positions[2][pt1_index] - vertice_positions[2][pt2_index]
        edge_lengths.append(dr.sqrt(dr.sqr(x) + dr.sqr(y) + dr.sqr(z)))
    return np.array(edge_lengths)

def smoothness_helper(v1, v2, v3):
    a = v2 - v1
    b = v3 - v1

    sqr_magnitude_a = dr.sum(dr.sqr(a))
    sqr_magnitude_b = dr.sum(dr.sqr(b))
    magnitude_a = dr.sqrt(sqr_magnitude_a)
    magnitude_b = dr.sqrt(sqr_magnitude_b)
    dot_ab = dr.sum(a * b)
    cos = dot_ab / (magnitude_a * magnitude_b)
    sin = dr.sqrt(1 - dr.sqr(cos))

    dist_a = dr.full(dr.cuda.ad.Float, dot_ab / sqr_magnitude_a, 3)
    c = a * dist_a
    cb = b-c
    l1_cb = magnitude_b * sin

    return cb, l1_cb
def smoothness(curr_faces, face_indices, vertice_positions):
    if len(curr_faces) > 2:
        print("Edge as more than 2 adjacent faces!")
    vert_idx_face1 = [face_indices[0][curr_faces[0]],
                        face_indices[1][curr_faces[0]],
                        face_indices[2][curr_faces[0]]]
    verts_idx_face2 = [face_indices[0][curr_faces[1]],
                        face_indices[1][curr_faces[1]],
                        face_indices[2][curr_faces[1]]]
    joined_verts = list(set(vert_idx_face1).intersection(verts_idx_face2))
    v1 = dr.cuda.Float([vertice_positions[0][joined_verts[0]], vertice_positions[1][joined_verts[0]], vertice_positions[2][joined_verts[0]]])
    v2 = dr.cuda.Float([vertice_positions[0][joined_verts[1]], vertice_positions[1][joined_verts[1]], vertice_positions[2][joined_verts[1]]])
    v3_face1_temp = set(vert_idx_face1).difference(joined_verts).pop()
    v3_face1 = dr.cuda.Float([vertice_positions[0][v3_face1_temp], vertice_positions[1][v3_face1_temp], vertice_positions[2][v3_face1_temp]])
    v3_face2_temp = set(verts_idx_face2).difference(joined_verts).pop()
    v3_face2 = dr.cuda.Float([vertice_positions[0][v3_face2_temp], vertice_positions[1][v3_face2_temp], vertice_positions[2][v3_face2_temp]])

    cb_face1, l1_cb_face1 = smoothness_helper(v1, v2, v3_face1)
    cb_face2, l1_cb_face2 = smoothness_helper(v1, v2, v3_face2)
    return dr.sum(cb_face1 * cb_face2) / (l1_cb_face1 * l1_cb_face2)

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
face_indices = dr.unravel(mi.Point3i, params['test.faces'])

edge_vert_indices = []
edge_vert_faces = {}
for i in range(len(face_indices[0])):
    x = face_indices[0][i]
    y = face_indices[1][i]
    z = face_indices[2][i]
    xy = (x, y) if x < y else (y, x)
    if xy not in edge_vert_indices:
        edge_vert_indices.append(xy)
        edge_vert_faces[xy] = [i]
    else:
        edge_vert_faces[xy].append(i)
    yz = (y, z) if y < z else (z, y)
    if yz not in edge_vert_indices:
        edge_vert_indices.append(yz)
        edge_vert_faces[yz] = [i]
    else:
        edge_vert_faces[yz].append(i)
    zx = (z, x) if z < x else (x, z)
    if zx not in edge_vert_indices:
        edge_vert_indices.append(zx)
        edge_vert_faces[zx] = [i]
    else:
        edge_vert_faces[zx].append(i)
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
    edge_loss = dr.cuda.ad.Float([1/len(initial_edge_lengths) * dr.sum(dr.sqr(np.subtract(initial_edge_lengths, current_edge_lengths)))])
    dr.set_grad_enabled(edge_loss, True)

    smoothness_loss = 0
    dr.set_grad_enabled(smoothness_loss, True)
    for key in edge_vert_faces:
        curr_faces = edge_vert_faces[key]
        cos = smoothness(curr_faces, face_indices, current_vertex_positions)
        smoothness_loss += cos

    loss = depth_loss * weight_depth + normal_loss * weight_normal + edge_loss * weight_edge
    dr.backward(loss)

    opt.step()

    writer.add_scalar("loss", loss[0], epoch)
    print(f"Epochs {epoch:02d}: error={loss[0]:6f} loss_normal={normal_loss[0]:6f} loss_depth={depth_loss[0]:6f} loss_edge={edge_loss[0]:6f}", end='\r')

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