# Rethinking Score Distillation as a Bridge Between Image Distributions

https://github.com/davidmcall/SDS-Bridge/assets/50497963/31ccf4f7-9211-4678-bdd0-4aaaccfa6853


## Experimenting in 2D

We offer a simpler installation than Threestudio with minimal dependencies if you just want to run experiments in 2D. This installation guide is adapted from [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio)

### Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This project has been tested with version 11.8 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

### Create environment

This repository requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/miniconda.html) before proceeding.

```bash
conda create --name bridge -y python=3.8
conda activate bridge
pip install --upgrade pip
```

### Dependencies

Install PyTorch with CUDA

For CUDA 11.8:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

Install other dependencies with pip:

```bash
cd 2D_experiments
pip install -r requirements.txt
```

### Run 2D Optimization

In the `2D_experiments` directory...

```bash
python generate.py
```

See `generate.py` for more options.
