# Generate Thingy10k/ABC dataset

## Downlaod Thingy10k and ABC dataset
- Download (part of) the ABC dataset from [here](https://archive.nyu.edu/handle/2451/43778)
    - for this thesis chunk [0099](https://archive.nyu.edu/handle/2451/44418) was used
- Downlaod objects from the Thingy10k dataset from [here](https://ten-thousand-models.appspot.com/)
    - to download the objects used in this thesis use [the provided download scipt](Thingi10K_download.py)

## Render map generation dataset
- run [source/map_generation_dataset/main.py](../../source/map_generation_dataset/main.py)
```
python source/map_generation_dataset/main.py --input_dir "dataset/ABC" --filetype ".stl" --view "225, 30" --spp_direct 256 --spp_aov 256
python source/map_generation_dataset/main.py --input_dir "dataset/thingy10k" --filetype ".stl" --view "225, 30" --spp_direct 256 --spp_aov 256
```
- split resulting images accoding to [train](train), [test](test) and [val](val)
```
mapgen_dataset
|- sketch_map_generation
|-- train (folder containing training input images)
|-- test (folder containing test input images)
|-- val folder containing validation input images)
|- target_map_generation
|-- train (folder containing training input images)
|-- test (folder containing test input images)
|-- val folder containing validation input images)
```

## Render evaluation images
- for the evaluation use the images provided in [data/input/ablation](../../data/input/ablation/) or run [source/render/main.py](../../source/render/main.py) 
The models used in this thesis are listed in [evaluation/evaluation_ablation.txt](evaluation/evaluation_ablation.txt).
The following command creates several image types, however, only the sketches are needed. Those need to be refined and seeded manually.
```
python source/render/main.py --line_gen "True" --input_path "dataset/eval_images/test.ply" -- views "225, 30"
```
