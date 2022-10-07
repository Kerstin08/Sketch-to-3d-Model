import shutil
import os

def move_files():
    current_range = "4500_4999"
    curr_max_value = 994999
    folders = ["n_mapgen", "n_png_mapgen", "d_mapgen", "d_png_mapgen", "sketch_mapgen"]
    for folder in folders:
        current_target_path = os.path.join(current_range, folder)
        for root, dirs, files in os.walk(folder, topdown=False):
            for file in files:
                curr_filename = int(file.split("_", 1)[0])
                if curr_filename <= curr_max_value:
                    input_path = os.path.join(root, file)
                    target_path = os.path.join(current_target_path, file)
                    shutil.move(input_path, target_path)

def remove_excess_sketches():
    current_range = "4500_4656"
    input_path = os.path.join(current_range, "n_mapgen")
    target_path = os.path.join(current_range, "sketch_mapgen")
    for root, dirs, files in os.walk(target_path, topdown=False):
        for file in files:
            found = False
            curr_filename = file.split("_", 1)[0]
            for i_root, i_dirs, i_files in os.walk(input_path, topdown=False):
                for i_file in i_files:
                    i_curr_filename = i_file.split("_", 1)[0]
                    if i_curr_filename == curr_filename:
                        found = True
            if not found:
                print(os.path.join(root, file))
                os.remove(os.path.join(root, file))