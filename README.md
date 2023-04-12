# Sketch23D - Masterthesis Kerstin Hofer

This repository contains the [source code](source) and [utility code](util) for my Masterthesis *Sketch to 3d-Model using Deep Learning and Differentiable Rendering*, which was completed at the University of Applied Sciences Salzburg, Austria. The results of this thesis and how to use the code will be presented briefly in the following. If there are further questions, please contact me via email at kerstin_hofer90@gmx.at.

---

## Thesis
The intent of the thesis is to create a system, that is capable of turning a simple 2D lineart into a 3D model. The main inspiration for this work is [Xiang et al. (2020)](https://onlinelibrary.wiley.com/doi/full/10.1002/cav.1939), but their idea was expanded by adding a depth map, that is created and used the same way as the normal map in their work. Furthermore, mulitple different base shapes based on different genera where provided and the genus was determined prior to the differentiable rendering process. The aim of these modifications is to make the reconstruction process applicable to a greater variety of shapes and less restrictive to certain classes.

### Structure
A pipeline with 3 modules was created:
* A segmentation module using flood fill and the determination of the Euler number to evaluate the genus. The output of this module is a silhouette image as well as a base mesh based on the determined genus of the object.
* A image-translation-network using a WGAN. The inspiration for that are the works by [Isola et al.]() and [Sue et al.](). 2 networks were trained, one for a depth and one for a normal map. The checkpoints used in the thesis can be downloaded [here]().
* A reconstruction module using the differentibale renderer [Mitsuba 3](). The filled image from the first module, and the created normal and depth maps from the translation network are used as ground truth for the normal, depth and silhouette loss. In addtion to that, a smoothness and edge loss is computed.

### Output
The thesis was evaluated qualitatively and quantitatively (IoU and Chamfer distance) in 2 parts: A comparison to a state-of-the-art comparison method and a ablation study. 
For the comparison, the [Neural Mesh Renderer](https://arxiv.org/pdf/1711.07566.pdf) by Kato, Ushiku, and Harada (2018) is used and for the dataset ShapeNetv1 is utilized. An introduction on how to setup those two and which resources to do that can be found in [util/NMR](util/NMR) and [util/dataset_ShapeNet](util/dataset_ShapeNet). 
For the ablation study, 4 variants are used:
* The proposed one using the depth map and the determined genus in the process
* One using the genus and the respective base mesh, but not the depth map
* One using the predicted depth map, but not the base mesh
* One that neither uses the depth map nor the base mesh of the determined genus
The results for the studies are presented in [data](data) incl. the reconstructed models. Generally it can be said, that the proposed model is better than the other tested variants, but has problems with reconstructing the overall shape. The comparison to the state-of-the-art model did not imply that this thesis' method is better regarding the shape, however there are improvements in the reconstruction of details. Furthermore, the reconstruction had some flaws regarding the topology, since only the genus was considered, but not the relation of the holes to each other or their size. Overall the pipeline did a great job in reconstructing the surface given via the predicted maps (which were flawed initself, especially when using the dataset created for this thesis due to the great varienty of shapes), but a bad job in the overall shape. 
Since this method is single-view, these results were expected. However, 

---

## Installation
The provided source code works on both windows 10 and 11 as well as linux Ubuntu 22.04. The required python packages can be installed via Anaconda using the conda requirements ([linux](util/environment_setup/conda_requirements_linux.txt) or [windows](util/environment_setup/conda_requirements_windows.txt)).
A version of NVIDA CUDA (preferrably NVIDA CUDA 11.4 for windows or 11.5 for linux) is required, refer to [the NVIDIA website](https://developer.nvidia.com/cuda-toolkit) for information on the installation.

## Datasets
To create the datasets from scratch follow the instructions in [util/dataset_ShapeNet](util/dataset_ShapeNet) and [util/dataset_Thingy10k_ABC](util/dataset_Thingy10k_ABC) respectively.
