#!/bin/bash
# Single-GPU (CUDA) dependencies. The upstream install.sh is TPU-only.
# Versions follow https://github.com/Gsunshine/meanflow/pull/5, pinned to what was validated.
set -e

conda create -y -n meanflow python=3.11       # tensorflow 2.15 does not support 3.12
# conda activate meanflow    # run this yourself, then re-run this script

# order matters: TF first, then override ml-dtypes/tensorstore, then jax last
pip install pillow clu "tensorflow==2.15.0" "keras<3" "torch<=2.4" torchvision \
            tensorflow_datasets "matplotlib==3.9.2"
pip install "orbax-checkpoint==0.4.4" "ml-dtypes==0.5.0" "tensorstore==0.1.67"
pip install "diffusers==0.29.2" dm-tree cached_property gdown scipy
pip install -U "jax[cuda12]==0.6.2"

python -c "import jax; print('jax', jax.__version__, jax.devices())"
