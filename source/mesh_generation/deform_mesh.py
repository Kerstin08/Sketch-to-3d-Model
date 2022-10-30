import os.path

import drjit as dr
import mitsuba as mi
import numpy as np
import source.util.mi_create_scenedesc as create_scenedesc
import math
from mitsuba.scalar_rgb import Transform4f as T
import normal_reparam_integrator
import depth_reparam_integrator
from torch.utils.tensorboard import SummaryWriter

mi.set_variant('cuda_ad_rgb')

class MeshGen():
    def __init__(self, output_dir, logs, weight_depth, weight_normal, weight_smoothness, weight_edge, epochs, log_frequency, lr):
        super(MeshGen, self).__init__()
        self.weight_depth = weight_depth
        self.weight_normal = weight_normal
        self.weight_smoothness = weight_smoothness
        self.weight_edge = weight_edge
        self.output_dir = output_dir
        self.lr = lr
        self.logs = logs
        self.writer = SummaryWriter(logs)
        self.epochs = epochs
        self.fov = 60
        self.log_frequency = log_frequency
    def write_output_renders(self, render_normal, render_depth, image_name):
        ref_bpm_normal = mi.util.convert_to_bitmap(render_normal)
        ref_np_normal = np.transpose(np.array(ref_bpm_normal), (2, 0, 1))
        ref_bpm_depth = mi.util.convert_to_bitmap(render_depth)
        ref_np_depth = np.transpose(np.array(ref_bpm_depth), (2, 0, 1))
        ref_images = np.stack([ref_np_normal, ref_np_depth])
        self.writer.add_images(image_name, ref_images)

    def get_edge_dist(self, vertice_positions, edge_vert_indices):
        x = dr.sqr(dr.gather(dr.cuda.ad.Float, vertice_positions[0], edge_vert_indices[0]) - dr.gather(dr.cuda.ad.Float, vertice_positions[0], edge_vert_indices[1]))
        y = dr.sqr(dr.gather(dr.cuda.ad.Float, vertice_positions[1], edge_vert_indices[0]) - dr.gather(dr.cuda.ad.Float, vertice_positions[1], edge_vert_indices[1]))
        z = dr.sqr(dr.gather(dr.cuda.ad.Float, vertice_positions[2], edge_vert_indices[0]) - dr.gather(dr.cuda.ad.Float, vertice_positions[2], edge_vert_indices[1]))
        edge_lengths = dr.sqrt(x+y+z)

        return edge_lengths

    def smoothness_helper(self, v1, v2, v3, eps):
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
        cb = b-c
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

    def preprocess_smoothness_params(self, edge_vert_faces, face_indices):
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

            v1_x.append(3 * joined_verts[0])
            v1_y.append(3 * joined_verts[0] + 1)
            v1_z.append(3 * joined_verts[0] + 2)
            v2_x.append(3 * joined_verts[1])
            v2_y.append(3 * joined_verts[1] + 1)
            v2_z.append(3 * joined_verts[1] + 2)
            v3_face1 = (set(vert_idx_face1).difference(joined_verts).pop())
            v3_x_f1.append(3 * v3_face1)
            v3_y_f1.append(3 * v3_face1 + 1)
            v3_z_f1.append(3 * v3_face1 + 2)
            v3_face2 = (set(verts_idx_face2).difference(joined_verts).pop())
            v3_x_f2.append(3 * v3_face2)
            v3_y_f2.append(3 * v3_face2 + 1)
            v3_z_f2.append(3 * v3_face2 + 2)
        v1 = [v1_x, v1_y, v1_z]
        v2 = [v2_x, v2_y, v2_z]
        v3_face1_idx = [v3_x_f1, v3_y_f1, v3_z_f1]
        v3_face2_idx = [v3_x_f2, v3_y_f2, v3_z_f2]
        return v1, v2, v3_face1_idx, v3_face2_idx

    def offset_verts(self, params, opt, initial_vertex_positions):
        opt['deform_verts'] = dr.clamp(opt['deform_verts'], -0.5, 0.5)
        trafo = mi.Transform4f.translate(opt['deform_verts'].x)
        params['shape.vertex_positions'] = dr.ravel(trafo @ initial_vertex_positions)
        params.update()

    def deform_mesh(self, normal_map, depth_map, basic_mesh):
        self.write_output_renders(normal_map, depth_map, "ref_images")
        depth_integrator = {
                'type': 'depth_reparam'
        }
        normal_integrator = {
                'type': 'normal_reparam'
        }
        depth_integrator_lodaded = mi.load_dict(depth_integrator)
        normal_integrator_lodaded = mi.load_dict(normal_integrator)

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
        normal_img_init = mi.render(scene, seed=0, spp=1024, integrator=normal_integrator_lodaded)
        depth_img_init = mi.render(scene, seed=0, spp=1024, integrator=depth_integrator_lodaded)
        self.write_output_renders(normal_img_init, depth_img_init, "init_images")

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
        dr.enable_grad(initial_edge_lengths)

        face_v1, face_v2, face_v3_face1, face_v3_face2 = self.preprocess_smoothness_params(edge_vert_faces, face_indices)

        opt = mi.ad.Adam(lr=self.lr)
        vertex_count = params[vertex_count_str]
        opt['deform_verts'] = dr.full(mi.Point3f, 0, vertex_count)

        for epoch in range(self.epochs):
            self.offset_verts(params, opt, initial_vertex_positions)

            normal_img = mi.render(scene, params, seed=epoch, spp=256, integrator=normal_integrator_lodaded)
            depth_img = mi.render(scene, params, seed=epoch, spp=256, integrator=depth_integrator_lodaded)
            mask = depth_img.array < 1.5
            curr_min_val = dr.min(depth_img)
            masked_img = dr.select(mask,
                                   depth_img.array,
                                   0.0)
            curr_max_val = dr.max(masked_img)
            wanted_range_min, wanted_range_max = 0.0, 0.5
            depth = dr.select(mask,
                              (depth_img.array - curr_min_val) * (
                                      (wanted_range_max - wanted_range_min) / (
                                      curr_max_val - curr_min_val)) + wanted_range_min,
                              1.0)
            depth_tens = mi.TensorXf(depth, shape=(256, 256, 3))

            if epoch % self.log_frequency == 0 or epoch==epoch-1:
                image_name = "deformed_images" + str(epoch)
                self.write_output_renders(normal_img, depth_img, image_name)

            depth_loss = dr.sum(abs(depth_tens - depth_map))
            normal_loss = dr.sum(abs(normal_img - normal_map))


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
            eps = 1e-6
            cb_1, l1_cb_1 = self.smoothness_helper(v1, v2, v3_face1, eps)
            cb_2, l1_cb_2 = self.smoothness_helper(v1, v2, v3_face2, eps)
            cos = dr.sum(cb_1 * cb_2) / (l1_cb_1 * l1_cb_2 + eps)
            smoothness_loss = dr.sum(dr.sqr(cos+1))

            loss = depth_loss * self.weight_depth + normal_loss * self.weight_normal + edge_loss * self.weight_edge + smoothness_loss * self.weight_smoothness
            dr.backward(loss)

            opt.step()

            self.writer.add_scalar("loss", loss[0], epoch)
            self.writer.add_scalar("loss_depth", depth_loss[0], epoch)
            self.writer.add_scalar("loss_normal", normal_loss[0], epoch)
            self.writer.add_scalar("loss_edge", edge_loss[0], epoch)
            self.writer.add_scalar("loss_smoothness", smoothness_loss[0], epoch)
            print("Epochs {}: error={} loss_normal={} loss_depth={} loss_edge={} loss_smoothness={}".format(epoch, loss[0], normal_loss[0], depth_loss[0], edge_loss[0], smoothness_loss[0]))

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
        output_path = os.path.join(self.output_dir, "deform_mesh.ply")
        mesh.write_ply(output_path)