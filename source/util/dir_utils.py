import os
from pathlib import Path

def create_logdir(log_dir):
    version = 0
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    else:
        for root, dirs, files in os.walk(log_dir):
            for dir in dirs:
                n = int(dir.split("_", 1)[1])
                if n >= version:
                    version=n+1
    curr_version = "version_{}".format(version)
    curr_path = os.path.join(log_dir, curr_version)
    os.mkdir(curr_path)
    return curr_path

def create_prefix_folder(prefix, dir):
    filename = Path(dir)
    dir_name = prefix + filename.name
    path = os.path.join(filename.parents[0], dir_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def create_general_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
