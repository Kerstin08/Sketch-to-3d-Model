# Setup of Neural Mesh Renderer

## Generate eval images
- Either use the images provided in [data/input/comparison/nmr](../../data/input/comparison/nmr) or 
run [source/render/main.py](../../source/render/main.py) for a model to create an input image for the Neural Mesh Renderer. 
The models used in this thesis are listed in [dataset_ShapeNet/evaluation/evaluation_comparison.txt](../dataset_ShapeNet/evaluation/evaluation_comparison.txt).
```
python source/render/main.py --render_type "kato" --line_gen "True" --input_path "dataset/eval_images/test.ply" -- views "225, 30"
```

## Download pre-trained models
- Download the pre-trained models from [the drive folder provided by the authors of NMR](https://drive.google.com/open?id=1tRHQoc0VWpj61PM1tVozIFwrFsDpKbTQ)
- Unzip the folder and copy the contents to a folder named *models*, which needs to be in the same directory as the [Dockerfile](Dockerfile)

## Run Neural Mesh Renderer using docker
- Install docker from [their website](https://docs.docker.com/engine/install/) if it is not already installed
- Run *docker build* using the provided [Dockerfile](Dockerfile). Optionally use --network host if your setup requires that to download python packages and github code.
```
docker build --network host -t nmr .
```
- Run *docker run* interactively. Add an available GPU as parameter and optionally use --network host if your setup requires that.
```
docker run --network host -it --gpus '"device=0"' kato
```
- Run setup.py in neural_renderer folder in docker container
```
cd neural_renderer
python setup.py --install
```
- In mesh_reconstruction folder run the reconstruction command to generate 3D models. Pre-trained models for the respective classes are located in ./data/models; evaluation images are in ./data/eval_images.
```
cd mesh_reconstruction
python mesh_reconstruction/reconstruct.py -d ./data/models -eid singleclass_02691156 -i ./data/eval_images/02691156_e09c32b947e33f619ba010ddb4974fe.png -oi ./output/airplane_out.png -oo ./output/airplane_out.obj
python mesh_reconstruction/reconstruct.py -d ./data/models -eid singleclass_03211117 -i ./data/eval_images/03211117_dd6c708c87d7160fac6198958b06e897.png -oi ./output/display_out.png -oo ./output/display_out.obj
python mesh_reconstruction/reconstruct.py -d ./data/models -eid singleclass_04401088 -i ./data/eval_images/04401088_e862392921d99119ee50cfd2d11d046b.png -oi ./output/phone_out.png -oo ./output/phone_out.obj
```
