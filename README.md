# SDS-Bridge

### [Project Page](https://sds-bridge.github.io/) | [Paper](https://arxiv.org/abs/2406.09417) 

**TLDR:** A unified framework to explain SDS and its variants, plus a new method that is fast & high-quality.

https://github.com/davidmcall/SDS-Bridge/assets/50497963/a5af3b0a-8edb-4acf-8c89-02a14451257a


## Experimenting in 3D

We provide our code for text-based NeRF optimization as an extension in Threestudio. To use it, please first install threestudio following the [official instructions](https://github.com/threestudio-project/threestudio?tab=readme-ov-file#installation).

### Extension Installation

```bash
cp -r ./threestudio-sds-bridge ../threestudio/custom/
cd ../threestudio
```

### Run 3D Optimization

In the `threestudio` repo...

```bash
python launch.py --config custom/threestudio-sds-bridge/configs/sds-bridge.yaml --train --gpu 0 system.prompt_processor.prompt="a pineapple"
```

Some options to play with for sds-bridge guidance are:
* `system.guidance.stage_two_start_step` The step at which to switch to the second stage.
* `system.guidance.stage_two_weight` The weight of the second stage.
* `system.prompt_processor.src_modifier` The prompt modfier that describes the current source distribution, e.g. "oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed."
* `system.prompt_processor.tgt_modifier` The prompt modfier that describes the target distribution, e.g. " detailed, high resolution, high quality, sharp."


## Experimenting in 2D

We offer a simpler installation than Threestudio with minimal dependencies if you just want to run experiments in 2D. This installation guide is adapted from [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio)

### Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This project has been tested with version 11.8 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

### Create Environment

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

See `generate.py` for more options, including but not limited to:
* `--mode` Choose between SDS-like loss functions [bridge (ours)](https://sds-bridge.github.io/), [SDS](https://dreamfusion3d.github.io), [NFSD](https://orenkatzir.github.io/nfsd/), [VSD](https://ml.cs.tsinghua.edu.cn/prolificdreamer/)
* `--seed` Random seed
* `--lr` Learning rate
* `--cfg_scale` Scale of classifier-free guidance computation



## Citation

``` bibtex
@article{mcallister2024rethinking,
    title={Rethinking Score Distillation as a Bridge Between Image Distributions},
    author={David McAllister and Songwei Ge and Jia-Bin Huang and David W. Jacobs and Alexei A. Efros and Aleksander Holynski and Angjoo Kanazawa},
    journal={arXiv preprint arXiv:2406.09417},
    year={2024}
  }
```
