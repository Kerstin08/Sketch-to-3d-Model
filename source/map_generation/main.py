import map_generation

import torch

input_dir = "../../resources/test1"
target_dir = "../../resources/test2"
model = map_generation.MapGen.__init__()
model.train(input_dir, target_dir)