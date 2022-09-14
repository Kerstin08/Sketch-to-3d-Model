import os

mesh_dict = {
    0: 'genus0.ply',
    1: 'genus1.ply',
    2: 'genus2.ply',
    3: 'genus3.ply',
    4: 'genus4.ply',
}

def get_basic_mesh_path(number_holes, path=""):
    genus = mesh_dict.get(number_holes)
    if not genus:
        print("For given number of holes no basis model is given")
        return None

    if len(path) == 0:
        path = genus
    else:
        path = os.path.join(path, genus)

    if not os.path.exists(path):
        print("Give path does not exist")
        return None

    return path