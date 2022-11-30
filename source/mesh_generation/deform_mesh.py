import os.path

import drjit as dr
import mitsuba as mi
import numpy as np
import torch

import source.render.mi_create_scenedesc as create_scenedesc
import math
from mitsuba.scalar_rgb import Transform4f as T
import source.render.normal_reparam_integrator
import source.render.depth_reparam_integrator
import source.render.silhouette_reparam_integrator
from torch.utils.tensorboard import SummaryWriter

mi.set_variant('cuda_ad_rgb')

class MeshGen():
    def __init__(self, output_dir, logs, weight_depth, weight_normal, weight_smoothness, weight_edge, weight_silhouette, epochs, log_frequency, lr):
        super(MeshGen, self).__init__()
        self.weight_depth = weight_depth
        self.weight_normal = weight_normal
        self.weight_smoothness = weight_smoothness
        self.weight_edge = weight_edge
        self.weight_silhouette = weight_silhouette
        self.output_dir = output_dir
        self.lr = lr
        self.logs = logs
        self.writer = SummaryWriter(logs)
        self.epochs = epochs
        self.fov = 60
        self.log_frequency = log_frequency

    def write_output_renders(self, render_normal, render_depth, silhouette, image_name):
        bpm_normal = mi.util.convert_to_bitmap(render_normal)
        np_normal = np.transpose(np.array(bpm_normal), (2, 0, 1))
        bpm_depth = mi.util.convert_to_bitmap(render_depth)
        np_depth = np.transpose(np.array(bpm_depth), (2, 0, 1))
        bpm_silhouette = mi.util.convert_to_bitmap(silhouette)
        np_silhouette = np.transpose(np.array(bpm_silhouette), (2, 0, 1))
        images = np.stack([np_normal, np_depth, np_silhouette])
        self.writer.add_images(image_name, images)

    def get_edge_dist(self, vertice_positions, edge_vert_indices):
        x = dr.sqr(dr.gather(dr.cuda.ad.Float, vertice_positions[0], edge_vert_indices[0]) - dr.gather(dr.cuda.ad.Float, vertice_positions[0], edge_vert_indices[1]))
        y = dr.sqr(dr.gather(dr.cuda.ad.Float, vertice_positions[1], edge_vert_indices[0]) - dr.gather(dr.cuda.ad.Float, vertice_positions[1], edge_vert_indices[1]))
        z = dr.sqr(dr.gather(dr.cuda.ad.Float, vertice_positions[2], edge_vert_indices[0]) - dr.gather(dr.cuda.ad.Float, vertice_positions[2], edge_vert_indices[1]))
        edge_lengths = dr.sqrt(x+y+z)

        return edge_lengths

    def smoothness_helper(self, v1, v2, v3):
        a = v2 - v1
        b = v3 - v1
        sqr_magnitude_a = dr.sum(dr.sqr(a))
        dot_ab = dr.sum(a * b)

        l = dot_ab / sqr_magnitude_a
        c = a * l
        cb = b-c
        l1_cb = dr.sqrt(dr.sum(dr.sqr(cb)))
        return cb, l1_cb

    def smoothness_helper_kato(self, v1, v2, v3, eps):
        a = v2 - v1
        b = v3 - v1
        sqr_magnitude_a = dr.sum(dr.sqr(a))
        sqr_magnitude_b = dr.sum(dr.sqr(b))
        magnitude_a = dr.sqrt(sqr_magnitude_a + eps)
        magnitude_b = dr.sqrt(sqr_magnitude_b + eps)
        dot_ab = dr.sum(a * b)
        cos = dot_ab / (magnitude_a * magnitude_b + eps)
        sin = dr.sqrt(1 - dr.sqr(cos) + eps)

        l = dot_ab / sqr_magnitude_a
        c = a * l
        cb = b - c
        l1_cb = magnitude_b * sin
        return cb, l1_cb

    def preprocess_edge_helper(self, i, x, y, edge_vert_indices, edge_vert_faces):
        xy = (x, y) if x < y else (y, x)
        indices_x = [j for j, a in enumerate(edge_vert_indices[0]) if a == xy[0]]
        indices_y = [j for j, a in enumerate(edge_vert_indices[1]) if a == xy[1]]
        if not set(indices_x) & set(indices_y):
            edge_vert_indices[0].append(xy[0])
            edge_vert_indices[1].append(xy[1])
            edge_vert_faces[xy] = [i]
        else:
            edge_vert_faces[xy].append(i)

    def preprocess_edge_params(self, face_indices):
        edge_vert_indices = [[],[]]
        edge_vert_faces = {}
        for i in range(len(face_indices[0])):
            x = face_indices[0][i]
            y = face_indices[1][i]
            z = face_indices[2][i]
            self.preprocess_edge_helper(i, x, y, edge_vert_indices, edge_vert_faces)
            self.preprocess_edge_helper(i, y, z, edge_vert_indices, edge_vert_faces)
            self.preprocess_edge_helper(i, z, x, edge_vert_indices, edge_vert_faces)
        return edge_vert_indices, edge_vert_faces

    def log_hparams(self):
        self_vars = {'weight_depth': self.weight_depth,
                     'weight_normal': self.weight_normal,
                     'weight_smoothness': self.weight_smoothness,
                     'weight_edge': self.weight_edge,
                     'weight_silhouette': self.weight_silhouette,
                     'lr': self.lr}
        self.writer.add_hparams(self_vars, {'hparam_metric': -1}, run_name='.')

    def preprocess_smoothness_params(self, edge_vert_faces, face_indices):
        def generate_vertex_list(vertex_list_x, vertex_list_y, vertex_list_z, vertex_index):
            vertex_list_x.append(3 * vertex_index)
            vertex_list_y.append(3 * vertex_index + 1)
            vertex_list_z.append(3 * vertex_index + 2)

        v1_x, v1_y, v1_z = [], [], []
        v2_x, v2_y, v2_z = [], [], []
        v3_x_f1, v3_y_f1, v3_z_f1 = [], [], []
        v3_x_f2, v3_y_f2, v3_z_f2 = [], [], []

        for key in edge_vert_faces:
            curr_faces = edge_vert_faces[key]
            vert_idx_face1 = [face_indices[0][curr_faces[0]],
                                face_indices[1][curr_faces[0]],
                                face_indices[2][curr_faces[0]]]
            verts_idx_face2 = [face_indices[0][curr_faces[1]],
                                face_indices[1][curr_faces[1]],
                                face_indices[2][curr_faces[1]]]
            joined_verts = list(set(vert_idx_face1).intersection(verts_idx_face2))

            generate_vertex_list(v1_x, v1_y, v1_z, joined_verts[0])
            generate_vertex_list(v2_x, v2_y, v2_z, joined_verts[1])
            v3_face1 = (set(vert_idx_face1).difference(joined_verts).pop())
            generate_vertex_list(v3_x_f1, v3_y_f1, v3_z_f1, v3_face1)
            v3_face2 = (set(verts_idx_face2).difference(joined_verts).pop())
            generate_vertex_list(v3_x_f2, v3_y_f2, v3_z_f2, v3_face2)

        v1 = [v1_x, v1_y, v1_z]
        v2 = [v2_x, v2_y, v2_z]
        v3_face1_idx = [v3_x_f1, v3_y_f1, v3_z_f1]
        v3_face2_idx = [v3_x_f2, v3_y_f2, v3_z_f2]
        return v1, v2, v3_face1_idx, v3_face2_idx

    def offset_verts(self, params, opt, initial_vertex_positions):
        #opt['deform_verts'] = dr.clamp(opt['deform_verts'], -0.5, 0.5)
        trafo = mi.Transform4f.translate(opt['deform_verts'])
        params['shape.vertex_positions'] = dr.ravel(trafo @ initial_vertex_positions)
        params.update()

    def write_output_mesh(self, vertex_count, vertex_positions, face_count, faces):
        mesh = mi.Mesh(
            "deformed_mesh",
            vertex_count=vertex_count,
            face_count=face_count,
            has_vertex_normals=True,
            has_vertex_texcoords=False,
        )

        mesh_params = mi.traverse(mesh)
        mesh_params["vertex_positions"] = dr.ravel(vertex_positions)
        mesh_params["faces"] = dr.ravel(faces)
        mesh_params.update()
        output_path = os.path.join(self.output_dir, "deform_mesh.ply")
        mesh.write_ply(output_path)

    # Use torch arithmetic function since dr.sum reduces tensors always over dim 0
    @dr.wrap_ad(source='drjit', target='torch')
    def torch_add(self, x):
        dims = tuple(range(x.ndimension())[1:])
        return torch.sum(x, dim=dims)

    def iou(self, predict, target):
        intersect = self.torch_add(predict * target)
        union = self.torch_add(predict + target - predict * target)
        intersect_shape_x = intersect.shape
        x = 1.0 - dr.sum(intersect / (union + 1e-6)) / intersect_shape_x[0]
        return x

    def deform_mesh(self, normal_map_target, depth_map_target, silhouette_target, basic_mesh):
        self.write_output_renders(normal_map_target, depth_map_target, silhouette_target, "target_images")
        self.log_hparams()

        depth_integrator = {
                'type': 'depth_reparam'
        }
        normal_integrator = {
                'type': 'normal_reparam'
        }
        silhouette_integrator = {
                'type': 'silhouette_reparam'
        }
        depth_integrator_lodaded = mi.load_dict(depth_integrator)
        normal_integrator_lodaded = mi.load_dict(normal_integrator)
        silhouette_integrator_lodaded = mi.load_dict(silhouette_integrator)

        datatype = basic_mesh.rsplit(".", 1)[1]
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
        normal_img_init = mi.render(scene, seed=0, spp=256, integrator=normal_integrator_lodaded)
        depth_img_init = mi.render(scene, seed=0, spp=256, integrator=depth_integrator_lodaded)
        mask = depth_img_init.array < 1.5
        curr_min_val = dr.min(depth_img_init)
        masked_img = dr.select(mask,
                               depth_img_init.array,
                               0.0)
        curr_max_val = dr.max(masked_img)
        wanted_range_min, wanted_range_max = 0.0, 0.75
        depth = dr.select(mask,
                          (depth_img_init.array - curr_min_val) * (
                                  (wanted_range_max - wanted_range_min) / (
                                  curr_max_val - curr_min_val)) + wanted_range_min,
                          1.0)
        depth_init_tens = mi.TensorXf(depth, shape=(256, 256, 3))
        silhouette_img_init = mi.render(scene, seed=0, spp=256, integrator=silhouette_integrator_lodaded)
        self.write_output_renders(normal_img_init, depth_init_tens, silhouette_img_init, "init_images")

        params = mi.traverse(scene)
        print(params)
        vertex_positions_str = "shape.vertex_positions"
        vertex_count_str = "shape.vertex_count"
        face_str = "shape.faces"
        face_count_str = "shape.face_count"
        initial_vertex_positions = dr.unravel(mi.Point3f, params[vertex_positions_str])
        face_indices = dr.unravel(mi.Point3i, params[face_str])

        edge_vert_indices, edge_vert_faces = self.preprocess_edge_params(face_indices)
        initial_edge_lengths = self.get_edge_dist(initial_vertex_positions, edge_vert_indices)

        face_v1, face_v2, face_v3_face1, face_v3_face2 = self.preprocess_smoothness_params(edge_vert_faces, face_indices)

        opt = mi.ad.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999)
        vertex_count = params[vertex_count_str]
        opt['deform_verts'] = dr.full(mi.Point3f, 0, vertex_count)

        for epoch in range(self.epochs):
            self.offset_verts(params, opt, initial_vertex_positions)
            normal_img = mi.render(scene, params, seed=epoch, spp=256, integrator=normal_integrator_lodaded)
            depth_img = mi.render(scene, params, seed=epoch, spp=256, integrator=depth_integrator_lodaded)
            silhouette_img = mi.render(scene, params, seed=epoch, spp=256, integrator=silhouette_integrator_lodaded)

            # Test if renderings contain invalid values due to corrupt mesh
            # Write failure images and mesh for debug purposes
            test_sum_normal = dr.sum(normal_img)

            if dr.any(dr.isnan(test_sum_normal)):
                self.write_output_renders(normal_img, depth_img, silhouette_img, "failure_mesh")
                self.write_output_mesh(vertex_count, params[vertex_positions_str], params[face_count_str],
                                       params[face_str])
                raise Exception("Normal rendering contains nan!")

            with dr.suspend_grad():
                single_channel_depth = depth_img[:, :, 0]
                mask = single_channel_depth.array < 1.5
                curr_min_val = dr.min(single_channel_depth)
                masked_img = dr.select(mask,
                                   single_channel_depth.array,
                                   0.0)
                curr_max_val = dr.max(masked_img)
                wanted_range_min, wanted_range_max = 0.0,  0.75
            depth = dr.select(mask,
                              (single_channel_depth.array - curr_min_val) * (
                                      (wanted_range_max - wanted_range_min) / (
                                      curr_max_val - curr_min_val)) + wanted_range_min,
                              1.0)
            depth_tens = mi.TensorXf(depth, shape=(256, 256))

            if epoch % self.log_frequency == 0 or epoch==epoch-1:
                image_name = "deformed_images" + str(epoch)
                self.write_output_renders(normal_img, depth_tens, silhouette_img, image_name)

            depth_loss = dr.sum(abs((depth_tens - depth_map_target)))
            normal_loss = dr.sum(abs((normal_img * 0.5 + 0.5) - (normal_map_target * 0.5 + 0.5)))
            silhouette_loss = self.iou(silhouette_img[:, :, 0], silhouette_target)
            current_vertex_positions = dr.unravel(mi.Point3f, params[vertex_positions_str])

            current_edge_lengths = self.get_edge_dist(current_vertex_positions, edge_vert_indices)
            edge_loss = dr.sum(dr.sqr(initial_edge_lengths - current_edge_lengths)) * 1/len(initial_edge_lengths)

            raveled_vertice_positions = dr.ravel(current_vertex_positions)
            v1 = dr.cuda.ad.Array3f(
                    dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, face_v1[0]),
                    dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, face_v1[1]),
                    dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, face_v1[2])
            )
            v2 = dr.cuda.ad.Array3f(
                    dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, face_v2[0]),
                    dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, face_v2[1]),
                    dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, face_v2[2])
            )
            v3_face1 = dr.cuda.ad.Array3f(
                    dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, face_v3_face1[0]),
                    dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, face_v3_face1[1]),
                    dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, face_v3_face1[2])
            )
            v3_face2 = dr.cuda.ad.Array3f(
                    dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, face_v3_face2[0]),
                    dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, face_v3_face2[1]),
                    dr.gather(dr.cuda.ad.Float, raveled_vertice_positions, face_v3_face2[2])
            )
            cb_1, l1_cb_1 = self.smoothness_helper_kato(v1, v2, v3_face1, 1e-6)
            cb_2, l1_cb_2 = self.smoothness_helper_kato(v1, v2, v3_face2, 1e-6)
            cos = dr.sum(cb_1 * cb_2) / (l1_cb_1 * l1_cb_2)
            smoothness_loss = dr.sum(dr.sqr(cos+1))

            loss = silhouette_loss * self.weight_silhouette + edge_loss * self.weight_edge + smoothness_loss * self.weight_smoothness + normal_loss * self.weight_normal + depth_loss + self.weight_depth

            dr.backward(loss)

            opt.step()
            self.writer.add_scalar("loss", loss[0], epoch)
            self.writer.add_scalar("loss_depth", depth_loss[0], epoch)
            self.writer.add_scalar("loss_normal", normal_loss[0], epoch)
            self.writer.add_scalar("loss_edge", edge_loss[0], epoch)
            self.writer.add_scalar("loss_smoothness", smoothness_loss[0], epoch)
            self.writer.add_scalar("loss_silhouette", silhouette_loss[0], epoch)
            
            self.writer.add_scalar("loss_depth_weighted", depth_loss[0] * self.weight_depth, epoch)
            self.writer.add_scalar("loss_normal_weighted", normal_loss[0] * self.weight_normal, epoch)
            self.writer.add_scalar("loss_edge_weighted", edge_loss[0] * self.weight_edge, epoch)
            self.writer.add_scalar("loss_smoothness_weighted", smoothness_loss[0] * self.weight_smoothness, epoch)
            self.writer.add_scalar("loss_silhouette_weighted", silhouette_loss[0] * self.weight_silhouette, epoch)
            print("Epochs {}: error={} loss_normal={} loss_depth={} loss_edge={} loss_smoothness={} loss_silhouette={}".format(epoch, loss[0], normal_loss[0], depth_loss[0], edge_loss[0], smoothness_loss[0], silhouette_loss[0]))

        self.writer.close()
        self.write_output_mesh(vertex_count, params[vertex_positions_str], params[face_count_str], params[face_str])
