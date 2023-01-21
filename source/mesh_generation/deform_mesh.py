import os.path

import drjit as dr
import mitsuba as mi
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from source.render.render_aov import AOV

mi.set_variant('cuda_ad_rgb')

class MeshGen():
    def __init__(self, output_name, output_dir, logs,
                 weight_depth, weight_normal, weight_smoothness, weight_edge, weight_silhouette,
                 epochs, log_frequency, lr, views, use_depth=True, eval_dir=None):
        super(MeshGen, self).__init__()
        if views is None:
            views = [(125, 30)]
        self.weight_depth = weight_depth
        self.weight_normal = weight_normal
        self.weight_smoothness = weight_smoothness
        self.weight_edge = weight_edge
        self.weight_silhouette = weight_silhouette
        self.output_dir = output_dir
        self.output_name = output_name
        self.lr = lr
        self.logs = logs
        self.writer = SummaryWriter(logs)
        self.epochs = epochs
        self.log_frequency = log_frequency
        self.renderer = AOV(views)
        self.use_depth = use_depth
        self.eval_dir = eval_dir

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

        l = dot_ab / (sqr_magnitude_a + 1e-6)
        c = a * l
        cb = b-c
        l1_cb = dr.sqrt(dr.sum(dr.sqr(cb)))
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
        trafo = mi.Transform4f.translate(opt['deform_verts'])
        params['shape.vertex_positions'] = dr.ravel(trafo @ initial_vertex_positions)
        params.update()

    def write_output_mesh(self, vertex_count, vertex_positions, face_count, faces, failed_deform=False):
        output_name = self.output_name
        if failed_deform:
            output_name = output_name + "_failed"
        mesh = mi.Mesh(
            output_name,
            vertex_count=vertex_count,
            face_count=face_count,
            has_vertex_normals=True,
            has_vertex_texcoords=False,
        )

        mesh_params = mi.traverse(mesh)
        mesh_params["vertex_positions"] = dr.ravel(vertex_positions)
        mesh_params["faces"] = dr.ravel(faces)
        mesh_params.update()
        output_path = os.path.join(self.output_dir, "{}.ply".format(output_name))
        mesh.write_ply(output_path)

        if not failed_deform and self.eval_dir is not None:
            output_path = os.path.join(self.eval_dir, "{}.ply".format(output_name))
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

        scene = self.renderer.create_scene(basic_mesh)[0]
        normal_img_init = self.renderer.render_normal(scene, basic_mesh)
        depth_img_init = self.renderer.render_depth(scene, basic_mesh)
        silhouette_img_init = self.renderer.render_silhouette(scene, basic_mesh)
        self.write_output_renders(normal_img_init, depth_img_init, silhouette_img_init, "init_images")

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

            normal_img = self.renderer.render_normal(scene, basic_mesh, seed=epoch, spp=16, params=params)
            depth_img = self.renderer.render_depth(scene, basic_mesh, seed=epoch, spp=16, params=params)
            silhouette_img = self.renderer.render_silhouette(scene, basic_mesh, seed=epoch, spp=16, params=params)

            if normal_img is None:
                self.write_output_mesh(vertex_count, params[vertex_positions_str], params[face_count_str],
                                       params[face_str], failed_deform=True)
                raise Exception("Normal rendering contains nan!")

            if epoch % self.log_frequency == 0 or epoch == epoch-1:
                image_name = "deformed_images" + str(epoch)
                self.write_output_renders(normal_img, depth_img, silhouette_img, image_name)

            if self.use_depth:
                depth_loss = dr.sum(abs((depth_img - depth_map_target)))
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
            cb_1, l1_cb_1 = self.smoothness_helper(v1, v2, v3_face1)
            cb_2, l1_cb_2 = self.smoothness_helper(v1, v2, v3_face2)
            cos = dr.sum(cb_1 * cb_2) / (l1_cb_1 * l1_cb_2 + 1e-6)
            smoothness_loss = dr.sum(dr.sqr(cos+1))

            if self.use_depth:
                loss = normal_loss * self.weight_normal + depth_loss * self.weight_depth + silhouette_loss * self.weight_silhouette + edge_loss * self.weight_edge + smoothness_loss * self.weight_smoothness
            else:
                loss = normal_loss * self.weight_normal + silhouette_loss * self.weight_silhouette + edge_loss * self.weight_edge + smoothness_loss * self.weight_smoothness

            dr.backward(loss)

            opt.step()
            self.writer.add_scalar("loss", loss[0], epoch)
            self.writer.add_scalar("loss_normal", normal_loss[0], epoch)
            self.writer.add_scalar("loss_edge", edge_loss[0], epoch)
            self.writer.add_scalar("loss_smoothness", smoothness_loss[0], epoch)
            self.writer.add_scalar("loss_silhouette", silhouette_loss[0], epoch)

            self.writer.add_scalar("loss_normal_weighted", normal_loss[0] * self.weight_normal, epoch)
            self.writer.add_scalar("loss_edge_weighted", edge_loss[0] * self.weight_edge, epoch)
            self.writer.add_scalar("loss_smoothness_weighted", smoothness_loss[0] * self.weight_smoothness, epoch)
            self.writer.add_scalar("loss_silhouette_weighted", silhouette_loss[0] * self.weight_silhouette, epoch)

            if self.use_depth:
                self.writer.add_scalar("loss_depth", depth_loss[0], epoch)
                self.writer.add_scalar("loss_depth_weighted", depth_loss[0] * self.weight_depth, epoch)
                print(
                    "Epochs {}: error={} loss_normal={} loss_depth={} loss_edge={} loss_smoothness={} loss_silhouette={}".format(
                        epoch, loss[0], normal_loss[0], depth_loss[0], edge_loss[0], smoothness_loss[0],
                        silhouette_loss[0]))
            else:
                print(
                    "Epochs {}: error={} loss_normal={} loss_edge={} loss_smoothness={} loss_silhouette={}".format(
                        epoch, loss[0], normal_loss[0], edge_loss[0], smoothness_loss[0],
                        silhouette_loss[0]))

        self.writer.close()
        self.write_output_mesh(vertex_count, params[vertex_positions_str], params[face_count_str], params[face_str])
