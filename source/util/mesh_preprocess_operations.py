# Triangle mesh preprocess operations
import sys
import trimesh
import math
import numpy as np
import argparse


def preprocess(
        path: str,
        shapenet: bool = False
) -> str | None:
    try:
        mesh = trimesh.load(path)
    except Exception as e:
        print("Exception occured in loading " + str(path))
        print(e)
        return None
    if not mesh.area > 0:
        print(str(path) + " contains no usable model!")
        return None
    # Since data reconstructed form SDF (https://github.com/Xharlie/DISN)
    # has artifacts the body count is rarely 1, but that does not affect the images
    # too much
    if not shapenet and mesh.body_count > 1:
        print(str(path) + " contains more than one model!")
        return None

    cleaned_mesh = clean_mesh(path, mesh)
    if not cleaned_mesh:
        return None

    norm_mesh = normalize_mesh(mesh)
    trans_norm_mesh = translate_to_origin(norm_mesh)
    # The ShapeNet data is not aligned like the other data, therefore it is rotated
    # Changing the rendering view would also be an option
    if shapenet:
        trans_norm_mesh = align_Shapenet(trans_norm_mesh)

    if path.rsplit('.', 1)[1] != 'ply':
        path = path.rsplit('.', 1)[0] + '.ply'
    ply = trimesh.exchange.ply.export_ply(trans_norm_mesh, encoding='binary', include_attributes=False)
    with open(path, 'wb+') as output:
        output.write(ply)
    return path


def clean_mesh(
        path: str,
        mesh: trimesh.Trimesh
) -> trimesh.Trimesh | None:
    mesh.process()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()

    if not mesh.is_watertight:
        mesh.fill_holes()
        trimesh.repair.fill_holes(mesh)

    if not mesh.is_winding_consistent:
        trimesh.repair.fix_inversion(mesh, multibody=True)
        trimesh.repair.fix_normals(mesh, multibody=True)
        trimesh.repair.fix_winding(mesh)

    if not mesh.is_watertight or not mesh.is_winding_consistent:
        print(str(path) + " contains invalid normals or is not solid!")
        return None
    return mesh


def normalize_mesh(
        mesh: trimesh.Trimesh
) -> trimesh.Trimesh | None:
    bounds = mesh.bounds
    dim = abs(bounds[1] - bounds[0])
    dim_mag = math.sqrt(sum(pow(element, 2) for element in dim))
    scale = 1 / dim_mag
    matrix = np.eye(4)
    matrix[:3, :3] *= scale
    mesh.apply_transform(matrix)
    return mesh


def align_Shapenet(
        mesh: trimesh.Trimesh
) -> trimesh.Trimesh | None:
    angle = math.pi / 2
    direction = [1, 0, 0]
    center = [0, 0, 0]
    rot = trimesh.transformations.rotation_matrix(angle, direction, center)
    mesh.apply_transform(rot)
    direction = [0, 0, -1]
    rot = trimesh.transformations.rotation_matrix(angle, direction, center)
    mesh.apply_transform(rot)
    return mesh


def translate_to_origin(
        mesh: trimesh.Trimesh
) -> trimesh.Trimesh | None:
    bounds = mesh.bounds
    dim = abs(bounds[1] - bounds[0])
    center = bounds[0] + dim / 2
    origin = np.zeros(3)
    dist = origin - center
    matrix = np.eye(4)
    matrix[:, 3] = [dist[0], dist[1], dist[2], 1]
    mesh.apply_transform(matrix)
    return mesh


def diff_ars(args):
    preprocess(args.input_mesh)


def main(args):
    parser = argparse.ArgumentParser(prog="scene_rendering")
    parser.add_argument("--input_mesh", type=str, default="datasets/thingi10k/0_499/32770.stl",
                        help="path to input model")
    args = parser.parse_args(args)
    diff_ars(args)


if __name__ == '__main__':
    main(sys.argv[1:])
