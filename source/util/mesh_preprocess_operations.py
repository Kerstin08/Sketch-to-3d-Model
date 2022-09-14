import trimesh
import math
import numpy as np

def preprocess_stl_mesh(path):
    mesh = trimesh.load(path)
    norm_mesh = normalize_mesh(mesh)
    trans_norm_mesh = translate_to_origin(norm_mesh)
    return convert_stl_to_norm_ply(path, trans_norm_mesh)

def convert_stl_to_norm_ply(path, mesh):
    outputpath = path.rsplit(".", 1)[0] + ".ply"
    ply = trimesh.exchange.ply.export_ply(mesh, encoding='binary', include_attributes=False)
    with open (outputpath, "wb+") as output:
        output.write(ply)
    return outputpath

def normalize_mesh(mesh):
    bounds = mesh.bounds
    dim = abs(bounds[1] - bounds[0])
    dim_mag = math.sqrt(sum(pow(element, 2) for element in dim))
    scale = 1/dim_mag
    matrix = np.eye(4)
    matrix[:3, :3] *= scale
    mesh.apply_transform(matrix)
    return mesh

def translate_to_origin(mesh):
    bounds = mesh.bounds
    dim = abs(bounds[1] - bounds[0])
    center = bounds[0] + dim/2
    origin = np.zeros(3)
    dist = origin-center
    matrix = np.eye(4)
    matrix[:, 3] = [dist[0], dist[1], dist[2], 1]
    mesh.apply_transform(matrix)
    bounds = mesh.bounds
    dim = abs(bounds[1] - bounds[0])
    center = bounds[0] + dim / 2
    return mesh