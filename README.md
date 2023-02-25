# Sketch23D - Masterthesis Kerstin Hofer

This repositiory contains the latex code of this thesis and the [papers](papers) used for research.
Futhermore, the [source code](source) as well as [utility code](util), which includes information and instructions regarding the setup and comparison sources, is provided.

## Installation
The provided source code works on both windows 10 and 11 as well as linux Ubuntu 22.04. The required python packages
can be installed via Anaconda using the conda requirements ([linux](util/environment_setup/conda_requirements_linux.txt) or [windows](util/environment_setup/conda_requirements_windows.txt)).

## Datasets
To create the datasets from scratch follow the instructions in util/dataset_ShapeNet and util/dataset_Thingy10k_ABC respectively.

## Comparion Code (Neural Mesh Renderer)
The state-of-the-art code used for comparison is the [Neural Mesh Renderer](https://arxiv.org/pdf/1711.07566.pdf) introduced by Kato, Ushiku, and Harada 2018. The instructions to recreate the 
adapted setup used in this thesis is described in [util/dataset_ShapeNet](util/dataset_ShapeNet).