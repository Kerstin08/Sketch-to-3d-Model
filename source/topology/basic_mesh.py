import os
import json

def get_basic_mesh_path(number_holes, path=""):
    if not os.path.exists(path):
        raise Exception("Given genus dir path {} does not exits.".format(path))

    for root, dirs, files in os.walk(path):
        for file in files:
            if file == "basic_meshes.json":
                f = open(os.path.join(root, file), "r")
                f_json = json.load(f)
                shapes = f_json["shapes"]
                try:
                    f_basic_mesh = shapes[str(number_holes)]
                except Exception:
                    raise RuntimeError("No base mesh exists for given genus {}".format(number_holes))
                path = os.path.join(root, f_basic_mesh)
                if not os.path.exists(path):
                    raise Exception("No base mesh exists in {} for given genus {}".format(path, number_holes))
                return path
    raise Exception("No base mesh exists for given genus {} or json file basic_meshes.json matching genera to filenames does not exist! ".format(number_holes))
