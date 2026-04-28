# Producer-Consumer Workflow with SmartSim/SmartRedis

This example demonstrates how to run a producer-consumer workflow with SmartSim and SmartRedis transferring data in-memory with SmartSim's Orchestrator (a wrapper around a Redis database). 
More information about SmartSim and SmartRedis can be found on the [ALCF documentation](https://docs.alcf.anl.gov/aurora/workflows/smartsim/).
The workflow is composed of a proxy producer simulation written in C++ (`sim.cpp`) and a proxy consumer application written in Python (`trainer.py`). The workflow also launches the SmartSim Orchestrator (i.e., the database) to store data being transferred between the producer and consumer. 
This setup could be used to stream simulation data from an ongoing simulation to a post-processing, visualization or ML training/inference script.

The workflow example sets up iteration loops within both the producer and consumer to emulate a time-stepping algorithm or a training algorithm. On one side, on each iteration the producer writes data to the database with a unique key for every rank, thus overwriting the previously stored data with the latest "snapshot". On the other side, on each iteration the consumer reads the latest available data. In this case, the number of producer and consumer MPI ranks is the same and data is written/read by ranks with the same ID on both sides (i.e., data produced by rank 0 of the simulation is read by rank 0 of the trainer). Note this is not a requirement of SmartSim workflows; the number of producer and consumer ranks need not match and there can be arbitrary numbers of producers and consumers connected to the same (or multiple) databases allowing great flexibility. 

The workflow driver script supports two launcher modes: colocated and clustered. In the colocated mode, the database, producer and consumer share the same set of nodes and the resources within those nodes (e.g., trainer runs on 6/12 GPUs on Aurora, simulation runs on the other 6 GPUs, and the database is distributed on the CPUs of those same nodes). In the clustered mode, the total set of nodes is split into three distinct lists, and each component of the workflow is executed on one of those node lists. 
Which launcher mode is more appropriate depends on the nature of the workflow. The colocated mode is most scalable, however it limits the view of the data to only what is available on node. The clustered mode is less scalable since data moves across nodes, however it allows all clients to access all data. 


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

## Build the Proxy Simulation

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
> -  To change the parameters of the workflow, look at the `NUM_PTS`, `DEPLOYMENT`, and `DB_NODES` environment variables in the submit script. 
> - `NUM_PTS` determines the amount of data being transferred from producer to consumer and stored in the database.
> - `DEPLOYMENT` determines the deployment type: colocated or clustered as described above
> - `DB_NODES` determines the amount of nodes to assign to the database. For colocated deployment, this is always 1; for clustered deployment, any number of nodes can be assigned to the database except for 2 due to a peculiarity with SmartSim's Orchestrator (but make sure to allocate enough nodes for all workflow components). 

