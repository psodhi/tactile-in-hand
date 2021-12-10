tactile-in-hand
===================================================

This repository contains the source code of the paper [PatchGraph: In-hand tactile tracking with learned surface normals](https://arxiv.org/abs/2111.07524).

# Installation

Create a virtual python environment using [Anaconda](https://www.anaconda.com/products/individual):
```
conda create -n inhand python=3.7
conda activate inhand
```

Install the `inhandpy` python package. From the base directory execute:
```
cd inhandpy/
pip install -e .
```

Additionally install `torch` for your particular CPU/CUDA configuration. More instructions on the PyTorch [website](https://pytorch.org/get-started/locally/).
# Usage

In `inhandpy`, download datasets, pre-trained models and other local resources by running:
```
./download_local_files.sh
```

## Stage 1: Tactile images to 3D point clouds

To run the example:
```
python scripts/examples/digit_rgb_to_cloud3d.py
```
By default, this runs the sim trials with cube shape. To run the example with other datasets and settings, please look at user set options under [digit_rgb_to_cloud3d.yaml](inhandpy/config/digit_rgb_to_cloud3d.yaml).

# Citing
If you find this repository helpful in your publications, please consider citing the following:

```
@inproceedings{sodhi2021patchgraph,
    title={PatchGraph: In-hand tactile tracking with learned surface normals},
    author={Paloma Sodhi and Michael Kaess and Mustafa Mukadam and Stuart Anderson},
    booktitle={arXiv 2111.07524},
    year={2021},
}
```

# License
This repository is licensed under the [BSD License](LICENSE.md).