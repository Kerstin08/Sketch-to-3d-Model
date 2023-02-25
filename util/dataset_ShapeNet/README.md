# Generate ShapeNet dataset

## Download and preprocess ShapeNet dataset
*The reconstruction used code from [DISN (Xu et al. 2019)](https://arxiv.org/pdf/1905.10711.pdf) and [Vega (Sin, Schroeder, and Barbiˇc 2013)](https://viterbi-web.usc.edu/~jbarbic/vega/SinSchroederBarbic2012.pdf)*
- Download the ShapeNet v1 dataset from [https://shapenet.org/](https://shapenet.org/)
    - DISN provides a partial reconstruction of the ShapeNet dataset, which can be found [here](https://drive.google.com/drive/folders/1QGhDW335L7ra31uw5U-0V7hB-viA0JXr)
If this is used, remove already reconstructed meshes from ShapeNet data
- Clone the DISN repo from [https://github.com/Xharlie/DISN](https://github.com/Xharlie/DISN)
- copy ShapeNet data fodler in DISN folder
- adjust DISN/preprocessing/info.json as well as DISN/data/filelists/ based on how the ShapeNetv1 dataset is altered and where in the DISN folder it is located
- Run *docker build* using the provided [Dockerfile](Dockerfile). Optionally use --network host if your setup requires that to download python packages and github code.
```
docker build --network host -t shapenet .
```
- Run *docker run* interactively. Mount the DISN folder in the docker contaienr and set the LD_LIBRARY_PATH to '/usr/local/lib/'. The later part can be also accomplished by adding the respective parameter to the docker file.
```
docker run -it -v $(pwd)/DISN:/workspace -e LD_LIBRARY_PATH='/usr/local/lib/' test
```
- Run the reconstruction command, either for a specific category of for all. The thread_num needs to be 1, otherwise using this code in a docker container with this setup will not work.
```
python -u /workspace/preprocessing/create_point_sdf_grid.py --thread_num 1 --category 'car'
```

## Render map generation dataset
- run [source/map_generation_dataset/main.py](../../source/map_generation_dataset/main.py)
```
python source/map_generation_dataset/main.py --input_dir "dataset/reconstructed_ShapeNet" --filetype ".obj" --view "225, 30" --spp_direct 100 --spp_aov 256 --shapenet_data "True"
```
- split resulting images accoding to [train](train), [test](test) and [val](val)
mapgen_dataset
```
|- sketch_map_generation
|-- train
|--- 02691156 (folder containing training input images of ShapeNet class 02691156)
|--- 02828884 (folder containing training input images of ShapeNet class 02828884)
|--- ...
|-- test
|--- 02691156 (folder containing test input images of ShapeNet class 02691156)
|--- 02828884 (folder containing test input images of ShapeNet class 02828884)
|--- ...
|-- val
|--- 02691156 (folder containing validation input images of ShapeNet class 02691156)
|--- 02828884 (folder containing validation input images of ShapeNet class 02828884)
|--- ...
|- target_map_generation
|-- train
|--- 02691156 (folder containing training target images of ShapeNet class 02691156)
|--- 02828884 (folder containing training target images of ShapeNet class 02828884)
|--- ...
|-- test
|--- 02691156 (folder containing test target images of ShapeNet class 02691156)
|--- 02828884 (folder containing test target images of ShapeNet class 02828884)
|--- ...
|-- val
|--- 02691156 (folder containing validation target images of ShapeNet class 02691156)
|--- 02828884 (folder containing validation target images of ShapeNet class 02828884)
|--- ...
```
- make sure to use set --use_shapenet and --shapenet_train_size when using reconstructed Shapenet dataset with [source/map_generation/main.py](../../source/map_generation/main.py)

- for the evaluation use the images provided in [data/input/comparison](../../data/input/comparison/) or run [source/render/main.py](../../source/render/main.py) 
The models used in this thesis are listed in [evaluation/evaluation_comparison.txt](evaluation/evaluation_comparison.txt).
The following command creates input images for the Neural Mesh Renderer as well as this thesis' framework.
```
python source/render/main.py --render_type "kato" --line_gen "True" --input_path "dataset/eval_images/test.ply" -- views "225, 30"
```