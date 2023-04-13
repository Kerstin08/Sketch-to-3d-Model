# Sketch23D - Masterthesis Kerstin Hofer

This repository contains the [source code](source) and [utility code](util) for my Masterthesis *Sketch to 3d-Model using Deep Learning and Differentiable Rendering*, which was completed at the University of Applied Sciences Salzburg, Austria. The results of this thesis and how to use the code will be presented briefly in the following. If there are further questions, please contact me via email at kerstin_hofer90@gmx.at.
![](images/teaser_image.png)

---

## Usage

### Installation
The provided source code works on both windows 10 and 11 as well as linux Ubuntu 22.04. The required python packages can be installed via Anaconda using the conda requirements ([linux](util/environment_setup/conda_requirements_linux.txt) or [windows](util/environment_setup/conda_requirements_windows.txt)).
A version of NVIDA CUDA (preferrably NVIDA CUDA 11.4 for windows or 11.5 for linux) is required, refer to [the NVIDIA website](https://developer.nvidia.com/cuda-toolkit) for information on the installation.

### Datasets
To create the datasets from scratch follow the instructions in [util/dataset_ShapeNet](util/dataset_ShapeNet) and [util/dataset_Thingy10k_ABC](util/dataset_Thingy10k_ABC) respectively.

### Run code
Each module can be run individually by calling:
* [evaluation/evaluation.py](source/evaluation/evaluation.py) to evaluate the output meshes using IoU and/or Chamfer Distance
* [map_generation/main.py](source/map_generation/main.py) to train or test the image-to-image translation network
* [map_generation_dataset/main.py](source/map_generation_dataset/main.py) to render Datasets, which is explained in detail in the utils folder for the respective datasets
* [mesh_generation/main.py](source/mesh_generation/main.py) to run the differentiable rendering process
* [topology/main.py](source/topology/main.py) to run the flood fill and base mesh determination process
* [util/mesh_preprocess_operations.py](source/util/mesh_preprocess_operations.py) to resize, convert and transform mesh to fit the requirements

The [main.py](source/main.py) runs the entire pipeline with all three modules. The input parameters are expained in each runnable file and the required file structures for the datasets are explained in the respecitve [util folder](util).

---

## Thesis
The intent of the thesis is to create a system, that is capable of turning a simple 2D lineart into a 3D model. The main inspiration for this work is [Xiang et al. (2020)](https://onlinelibrary.wiley.com/doi/full/10.1002/cav.1939), but their idea was expanded by adding a depth map, that is created and used the same way as the normal map in their work. 

<img title="Base meshes" alt="Basic overview of the pipeline" src="images/base_meshes.png" width="500">

Furthermore, mulitple different base shapes based on different genera where provided (as depicted above) and the genus was determined prior to the differentiable rendering process. The aim of these modifications is to make the reconstruction process applicable to a greater variety of shapes and less restrictive to certain classes.

### Structure

<img title="Basic overview of the pipeline" alt="Basic overview of the pipeline" src="images/general_overview.png" width="500">

A pipeline with 3 modules was created:

* A segmentation module using flood fill and the determination of the Euler number to evaluate the genus. The output of this module is a silhouette image as well as a base mesh based on the determined genus of the object.
* A image-translation-network using a WGAN. The inspiration for that are the works by [Isola et al.](https://phillipi.github.io/pix2pix/) and [Sue et al.](https://github.com/Ansire/sketch2normal). 2 networks were trained, one for a depth and one for a normal map. The checkpoints used in the thesis can be downloaded [here](https://drive.google.com/file/d/15qMidE1LRnxboSMtPYCqI27RzRPMNQY2/view?usp=sharing).
* A reconstruction module using the differentibale renderer [Mitsuba 3](https://www.mitsuba-renderer.org/). The filled image from the first module, and the created normal and depth maps from the translation network are used as ground truth for the normal, depth and silhouette loss. In addtion to that, a smoothness and edge loss is computed.

### Output
The thesis was evaluated qualitatively and quantitatively (IoU and Chamfer distance) in 2 parts: A comparison to a state-of-the-art comparison method and a ablation study. 
For the comparison, the [Neural Mesh Renderer](https://arxiv.org/pdf/1711.07566.pdf) by Kato, Ushiku, and Harada (2018) is used and for the dataset ShapeNetv1 is utilized. An introduction on how to setup those two and which resources to do that can be found in [util/NMR](util/NMR) and [util/dataset_ShapeNet](util/dataset_ShapeNet). 
For the ablation study, 4 variants are used:
* The proposed one using the depth map and the determined genus in the process
* One using the genus and the respective base mesh, but not the depth map
* One using the predicted depth map, but not the base mesh
* One that neither uses the depth map nor the base mesh of the determined genus

The results for the studies are presented in [data](data) incl. the reconstructed models. Generally it can be said, that the proposed model is better than the other tested variants, but has problems with reconstructing the overall shape. 

**Comparison results**
360° view of reconstructed object        | Normal maps from view of training images       |  Normal maps random view                           |  
:---------------------------------------:|:----------------------------------------------:|:--------------------------------------------------:|
![](images/comparison_airplane_360.png)  | ![](images/comparison_normal_trained_view.png) | ![](images/comparison_normal_non_trained_view.png) |

The comparison to the state-of-the-art model does not imply that this thesis' method is better regarding the shape, however there are improvements in the reconstruction of details and normals. As seen in the image above, the reconstruction of the pipeline is only working for the view of the input images, while the state-of-the art model does reconstruct all persepctives equally well. This is caused by the lack of regularizers in the Differentiable Rendering process, which could be improved in furture versions.


**Ablation results**
360° view of reconstructed object     | Normal maps of reconstructed objects    |  Depth maps of reconstructed objects   |  
:------------------------------------:|:---------------------------------------:|:--------------------------------------:|
![](images/ablation_complex_360.png)  | ![](images/ablation_normal_complex.png) | ![](images/ablation_depth_complex.png) |

When looking at the images above, it can be determined, that the addition of the depth map and the topology-based base mesh does imporove the reconsturction. Furthermore, it stablizes the reconstruction process, leading to less invalid normals that can crash the system and make it more robust in terms of weight and learning rate adjustments.


**Conclusion**
In general, the reconstruction has some flaws regarding the topology, since only the genus was considered, but not the relation of the holes to each other or their size. Furthermore, the input must be cleen and properly seeded, which forces more work on the user. However, the pipeline does an overall reasonable job in reconstructing the surface given via the predicted maps (which were flawed initself, especially when using the dataset created for this thesis due to the great varienty of shapes), but a bad job in the overall shape. 
Since this method is single-view, these results were expected. However, the results show that the made additions due improve the base code and therefore it can reasonably assumed, that with further improvements these methods are on-par and even better than current state-of-the-art method. Easy first improvements could include mirroring the shaped part (only works for symmetrical datasets with aligned objects e.g. ShapeNet), determining the relation of the holes, cleanup the input sketch, choose better fitting base meshes and remesh the output to avoid inverted and access faces.
