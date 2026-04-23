# Producer-Consumer Workflow with SmartSim/SmartRedis

This example demonstrates how to run a producer-consumer workflow with SmartSim and SmartRedis. The workflow is composed of a proxy producer simulation written in C++ (`sim.cpp`) and a proxy consumer application written in Python (`trainer.py`). This setup could be used to stram simulation data from an ongoing simulation to a post-prtocessing, visualization or ML training/inference script.

The workflow initially shares some data through the file system with the BP5 engine, and then sets up an iteration loop within both the producer and consumer where data is streamed between the two. The workflow is also set up to run with the producer on one set of nodes and the consumer on the other set of nodes in order to force data streaming through the network. The submit script show how to set this case up with sequential `mpiexec` commands and with a single command in MPMD mode.

## Install SmartSim and SmartRedis

On Aurora login nodes, execute the following commands or run `source install_venv_aurora.sh`

```bash
# Load the frameworks module
module load frameworks

# Create a venv for SmartSim
python -m venv --clear ./_env --system-site-packages
source ./_env/bin/activate
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
``` 

## Biuld the Proxy Simulation

To build the proxy simulation C++ code, execute 

```bash
./config.sh
```

on the login node of Aurora or Polaris.

## Run the Example

To run the example, submit the script for the appropriate system. 
For example, on Aurora execute

```bash
qsub submit_aurora.sh
```

> [!NOTE]
> -  

