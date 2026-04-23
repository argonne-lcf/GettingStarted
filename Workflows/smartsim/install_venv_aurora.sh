#!/bin/bash

BASE=$PWD

# Load the frameworks module
module load frameworks

# Create a venv for SmartSim
python -m venv --clear $BASE/_ssim_env --system-site-packages
source $BASE/_ssim_env/bin/activate
pip install --upgrade pip

# Install SmartRedis
git clone https://github.com/CrayLabs/SmartRedis.git
cd SmartRedis
pip install -e .
make lib
cd ..

# Build SmartSim
git clone https://github.com/CrayLabs/SmartSim.git
cd SmartSim
pip install -e .
# Note: disregard compatibility errors
cd ..

# Install the CPU backend
# NB: GPU backend for RedisAI not supported on Intel PVC
cd SmartSim
export TORCH_CMAKE_PATH=$( python -c 'import torch;print(torch.utils.cmake_prefix_path)' )
export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
curl -O https://gist.githubusercontent.com/rickybalin/fcf1d15a26dbbc120f42943041ada827/raw/e22485d53250b8a29ead537533bca7c8f229c362/aurora_config.patch
git apply aurora_config.patch
smart build -v --device cpu --skip-tensorflow --skip-onnx 
smart validate
cd ..

