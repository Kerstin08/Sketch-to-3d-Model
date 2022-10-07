import os.path

import drjit as dr
import mitsuba as mi
import numpy as np
import source.util.mi_create_scenedesc as create_scenedesc
import math
import torch
import torchvision
from mitsuba.scalar_rgb import Transform4f as T
import normal_reparam_integrator
import depth_reparam_integrator

from torch.utils.tensorboard import SummaryWriter

mi.set_variant('cuda_ad_rgb')

class MeshGen():
    def __init__(self, output_dir, logs, weight_depth, weight_normal, weight_smoothness, weight_edge, epochs, lr):
        super(MeshGen, self).__init__()
        self.weight_depth = weight_depth
        self.weight_normal = weight_normal
        self.weight_smoothness = weight_smoothness
        self.weight_edge = weight_edge
        self.output_dir = output_dir
        self.lr = lr
        self.writer = SummaryWriter(logs)
        self.epoch = epochs
        self.initial_vertex_positions = None
        self.fov = 60
    def write_output_renders(self, render_depth, render_normal, image_name):
        ref_bpm_normal = mi.util.convert_to_bitmap(render_normal)
        ref_np_normal = np.transpose(np.array(ref_bpm_normal), (2, 0, 1))
        ref_bpm_depth = mi.util.convert_to_bitmap(render_depth)
        ref_np_depth = np.transpose(np.array(ref_bpm_depth), (2, 0, 1))
        ref_images = np.stack([ref_np_normal, ref_np_depth])
        self.writer.add_images(image_name, ref_images)

    def get_edge_dist(self, vertice_positions, edge_vert_indices):
        edge_lengths = []
        for pt1_index, pt2_index in edge_vert_indices:
            x = vertice_positions[0][pt1_index] - vertice_positions[0][pt2_index]
            y = vertice_positions[1][pt1_index] - vertice_positions[1][pt2_index]
            z = vertice_positions[2][pt1_index] - vertice_positions[2][pt2_index]
            edge_lengths.append(dr.sqrt(dr.sqr(x) + dr.sqr(y) + dr.sqr(z)))
        return dr.cuda.ad.Float(edge_lengths)

    def smoothness_helper(self, v1, v2, v3):
        a = v2 - v1
        b = v3 - v1

        sqr_magnitude_a = dr.sum(dr.sqr(a))
        sqr_magnitude_b = dr.sum(dr.sqr(b))
        magnitude_a = dr.sqrt(sqr_magnitude_a)
        magnitude_b = dr.sqrt(sqr_magnitude_b)
        dot_ab = dr.sum(a * b)
        cos = dot_ab / (magnitude_a * magnitude_b)
        sin = dr.sqrt(1 - dr.sqr(cos))

        l = dot_ab / sqr_magnitude_a
        dist_a = dr.repeat(l, 3)
        c = a * dist_a
        cb = b-c
        l1_cb = magnitude_b * sin

        return cb, l1_cb
    def smoothness(self, curr_faces, face_indices, vertice_positions):
        if len(curr_faces) > 2:
            raise Exception("Mesh is invalid! Edge has more than 2 adjacent faces!")
        vert_idx_face1 = [face_indices[0][curr_faces[0]],
                            face_indices[1][curr_faces[0]],
                            face_indices[2][curr_faces[0]]]
        verts_idx_face2 = [face_indices[0][curr_faces[1]],
                            face_indices[1][curr_faces[1]],
                            face_indices[2][curr_faces[1]]]
        raveled_vertice_positions = dr.ravel(vertice_positions)
        joined_verts = list(set(vert_idx_face1).intersection(verts_idx_face2))
        v3_face1_idx = set(vert_idx_face1).difference(joined_verts).pop()
        v3_face2_idx = set(verts_idx_face2).difference(joined_verts).pop()
        v1 = dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, [3 * joined_verts[0], 3 * joined_verts[0]+1, 3 * joined_verts[0]+2])
        v2 = dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, [3 * joined_verts[1], 3 * joined_verts[1]+1, 3 * joined_verts[1]+2])
        v3_face1 = dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, [3 * v3_face1_idx, 3 * v3_face1_idx+1, 3 * v3_face1_idx+2])
        v3_face2 = dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, [3 * v3_face2_idx, 3 * v3_face2_idx+1, 3 * v3_face2_idx+2])


        cb_face1, l1_cb_face1 = self.smoothness_helper(v1, v2, v3_face1)
        cb_face2, l1_cb_face2 = self.smoothness_helper(v1, v2, v3_face2)

        x = dr.sum(cb_face1 * cb_face2)
        y = (l1_cb_face1 * l1_cb_face2)
        z = x / y
        return z

    def offset_verts(self, params, opt):
        opt['deform_verts'] = dr.clamp(opt['deform_verts'], -0.5, 0.5)
        trafo = mi.Transform4f.translate(opt['deform_verts'].x)
        params['test.vertex_positions'] = dr.ravel(trafo @ self.initial_vertex_positions)
        params.update()

    def deform_mesh(self, normal_map, depth_map, basic_mesh):
        self.write_output_renders(depth_map, normal_map, "ref_images")
        depth_integrator = {
                'type': 'depth_reparam'
        }
        normal_integrator = {
                'type': 'normal_reparam'
        }
        depth_integrator_lodaded = mi.load_dict(depth_integrator)
        normal_integrator_lodaded = mi.load_dict(normal_integrator)

        datatype = basic_mesh.rsplit(".", 1)[1]
        basic_mesh_name = basic_mesh.rsplit("\\", 1)[1].rsplit(datatype)[0]
        if datatype != "obj" and datatype != "ply":
            raise Exception("Datatype of given mesh {} cannot be processed! Must either be .ply or .obj".format(basic_mesh))
        shape = create_scenedesc.create_shape(basic_mesh, datatype)
        distance = math.tan(math.radians(self.fov))/1.75
        near_distance = distance
        far_distance = distance * 4
        centroid = np.array([distance, -distance, distance])

        # center is assumed to be at 0,0,0, see mesh_preprocess_operations.py translate_to_origin
        camera = create_scenedesc.create_camera(T.look_at(target=(0.0, 0.0, 0.0),
                                                          origin=tuple(centroid),
                                                          up=(0, 0, 1),
                                                          ),
                                                self.fov, near_distance, far_distance,
                                                256, 256)
        scene_desc = {"type": "scene", "shape": shape, "camera": camera}
        scene = mi.load_dict(scene_desc)
        normal_img_init = mi.render(scene, seed=0, spp=1024, integrator=normal_integrator_lodaded)
        depth_img_init = mi.render(scene, seed=0, spp=1024, integrator=depth_integrator_lodaded)
        self.write_output_renders(depth_img_init, normal_img_init, "init_images")

        params = mi.traverse(scene)
        print(params)
        vertex_positions_str = str(basic_mesh_name) + ".vertex_positions"
        vertex_count_str = str(basic_mesh_name) + ".vertex_count"
        face_str = str(basic_mesh_name) + ".faces"
        face_count_str = str(basic_mesh_name) + ".face_count"
        self.initial_vertex_positions = dr.unravel(mi.Point3f, params[vertex_positions_str])
        face_indices = dr.unravel(mi.Point3i, params[face_str])

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
            initial_edge_lengths = self.get_edge_dist(self.initial_vertex_positions, edge_vert_indices)
            dr.enable_grad(initial_edge_lengths)

            opt = mi.ad.Adam(lr=self.learning_rate)
            vertex_count = params[vertex_count_str]
            opt['deform_verts'] = dr.full(mi.Point3f, 0, vertex_count)

        for epoch in range(self.epochs):
            self.offset_verts(params, opt)

            normal_img = mi.render(scene, params, seed=epoch, spp=16, integrator=normal_integrator_lodaded)
            depth_img = mi.render(scene, params, seed=epoch, spp=16, integrator=depth_integrator_lodaded)
            if epoch % 10 == 0:
                image_name = "deformed_images" + str(epoch)
                self.write_output_renders(depth_img, normal_img, image_name)

            depth_loss = dr.sum(dr.sqr(depth_img - depth_map)) / len(depth_img)
            normal_loss = dr.sum(dr.sqr(normal_img - normal_map)) / len(normal_img)

            current_vertex_positions = dr.unravel(mi.Point3f, params[vertex_positions_str])

            current_edge_lengths = self.get_edge_dist(current_vertex_positions, edge_vert_indices)

            edge_loss = dr.sum(dr.sqr(initial_edge_lengths - current_edge_lengths)) * 1/len(initial_edge_lengths)

            smoothness_loss = 0.0
            for key in edge_vert_faces:
                curr_faces = edge_vert_faces[key]
                cos = self.smoothness(curr_faces, face_indices, current_vertex_positions)
                smoothness_loss += cos

            print(dr.grad_enabled(depth_loss))
            print(dr.grad_enabled(normal_loss))
            print(dr.grad_enabled(edge_loss))
            print(dr.grad_enabled(smoothness_loss))

            loss = depth_loss * self.weight_depth + normal_loss * self.weight_normal + edge_loss * self.weight_edge + smoothness_loss * self.weight_smooth
            dr.backward(loss)

            opt.step()

            self.writer.add_scalar("loss", loss[0], epoch)
            print(f"Epochs {epoch:02d}: error={loss[0]:6f} loss_normal={normal_loss[0]:6f} loss_depth={depth_loss[0]:6f} loss_edge={edge_loss[0]:6f}", end='\r')

        self.writer.close()
        mesh = mi.Mesh(
            "deformed_mesh",
            vertex_count=vertex_count,
            face_count=params[face_count_str],
            has_vertex_normals=True,
            has_vertex_texcoords=False,
        )

        mesh_params = mi.traverse(mesh)
        mesh_params["vertex_positions"] = dr.ravel(params[vertex_positions_str])
        mesh_params["faces"] = dr.ravel(params[face_str])
        mesh_params.update()
        output_path = os.path.join(self.output_dir, "deform_mesh.py")
        mesh.write_ply(output_path)