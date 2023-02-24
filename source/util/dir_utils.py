# Utils for creating various directory types
import os
from pathlib import Path


# Create versioned folders withing given folder
def create_version_folder(
        orig_path: str
) -> str:
    version = 0
    path = Path(orig_path)
    if not path.exists():
        path.mkdir(parents=True)
    else:
        for root, dirs, files in os.walk(path):
            for curr_dir in dirs:
                n = int(curr_dir.split('_', 1)[1])
                if n >= version:
                    version = n + 1
            break
    curr_version = 'version_{}'.format(version)
    curr_path = os.path.join(path, curr_version)
    os.mkdir(curr_path)
    return curr_path


# Create folders with prefix to folder name
def create_prefix_folder(
        prefix: str,
        orig_path: str
) -> str:
    path = Path(orig_path)
    dir_name = prefix + '_' + path.name
    prefixed_path = os.path.join(path.parents[0], dir_name)
    if not os.path.exists(prefixed_path):
        os.makedirs(prefixed_path)
    return prefixed_path


# Create general folder
def create_general_folder(
        orig_path: str
) -> str:
    path = Path(orig_path)
    if not path.exists():
        path.mkdir(parents=True)
    return orig_path
