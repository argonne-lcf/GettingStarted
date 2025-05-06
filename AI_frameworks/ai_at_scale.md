# AI frameworks at scale

**Learning Goals:**

* How to efficiently load Python environments at scale using Copper
* Best practices to efficiently train a PyTorch model on Aurora GPUs at scale



## Overview

Training deep a learning model on 1000s (or more) GPUs may stress various components of a High Performance Computing system. 
Here we discuss and propose solutions for potential issues you might encounter with two critical aspects of deep learning at scale: 

1. [Input/Output (Copper)](#input%2FoutputInput/Output%3A-copper-for-scalable-data-loading)
1. [CPU affinity and bindings](CPU-affinity-and-bindings)
1. [Communication (oneCCL)](#communication%3A-oneccl)


## Input/Output: Copper for scalable data loading

Loading a Python module (e.g. `import torch` of `import numpy`) at scale can take several minutes if the file containing the module is stored on a distributed file system like Lustre (`/lus/flare` on Aurora), and thousands of processes (MPI ranks) try to read that same file simultaneously.

[Copper](https://docs.alcf.anl.gov/aurora/data-management/copper/copper/) is a **read-only** cooperative caching layer aimed to enable scalable data loading on massive amounts of compute nodes. This aims to avoid the I/O bottleneck in the storage network and effectively use the compute network for data movement.

Basically, instead of all the nodes directly contacting and loading the files from lustre, only one node will contact and load from lustre and distribute it to the other nodes.

Here is a performance comparison of `import torch` with and without Copper (lustre results can vary when there is contention and interference on lustre or storage network):

| num nodes | num ranks per node | without copper | with copper |
| :-------: | :----------------: | -------------- | ----------- |
|    64     |         12         | 105 seconds    | 50 seconds  |
|    128    |         12         | 110 seconds    | 52 seconds  |
|    256    |         12         | 110 seconds    | 54 seconds  |
|    512    |         12         | 115 seconds    | 56 seconds  |
|   1024    |         12         | 4 minutes      | 58 seconds  |
|   2048    |         12         | 7 minutes      | 58 seconds  |
|   4096    |         12         | 8 minutes      | 59 seconds  |
|   8192    |         12         | 17 minutes     | 60 seconds  |


> 🛑  **Note**: Copper is not needed if you only use packages from the system installed [`frameworks` module](https://docs.alcf.anl.gov/aurora/data-science/python/#aiml-framework-module). 
Use it only when you need to import Python modules from user-defined conda environments or virtual environments.


### ⌨️   Hands on

Here we show how to use Copper on Aurora to import `torch` from a user-defined conda environment. 

1. [Login to Aurora](https://docs.alcf.anl.gov/aurora/getting-started-on-aurora/):
   ```bash
   ssh <username>@aurora.alcf.anl.gov
   ```
1. Submit the script [`simple_with_copper.sh`](examples/copper_example/simple_with_copper.sh) from the directory `./examples/ai_at_scale/` of this repository:
   ```bash
   qsub simple_with_copper.sh
   ```
   (The example script will run on 2 nodes only, so that multiple people can test it at the same time. Feel free to edit the line `#PBS -l select=2` to test it on a higher number of nodes.)

##### Here is a breakdown of what the above script is doing:

1. Launch Copper
   ```bash
   module load copper
   launch_copper.sh
   ```

1. If you have a local conda environment located at `LUS_CONDA_PATH=/lus/flare/projects/alcf_training/softwares/copper-lus-pip-custom-package`, and you want to use Copper, you need to: 
   - Add `/tmp/${USER}/copper/` to the beginning of the path to your conda environment:
      ```bash
      /tmp/${USER}/copper/${LUS_CONDA_PATH}
      ```
   - Append the modified environment path to your `PYTHONPATH`
      ```bash
      PYTHONPATH=/tmp/${USER}/copper/${LUS_CONDA_PATH}:$PYTHONPATH
      ```
      in your `mpiexec`:
      ```bash
      time mpirun --np ${NRANKS} --ppn ${RANKS_PER_NODE} \
          --cpu-bind=list:4:9:14:19:20:25:56:61:66:71:74:79 --genvall \
          --genv=PYTHONPATH=/tmp/${USER}/copper/lus/flare/projects/alcf_training/softwares/copper-lus-pip-custom-package \
      	python3 -c "import torch; print(torch.__file__)"
      ```

# !!! Khalid: TO DO !!!

## CPU affinity and bindings

![aff](https://github.com/argonne-lcf/ALCFBeginnersGuide/raw/master/aurora/media/aurora_exascale_compute_blade2.png)


## Communication: oneCCL

Training a deep learning model on multiple GPUs requires that the results of computation done on one GPU are communicated to other GPUs. 
This is done using collective operations, like _AllReduce_, which, on Aurora, are enabled by the [oneAPI Collective Communication Library (oneCCL)](https://docs.alcf.anl.gov/aurora/data-science/frameworks/oneCCL/). 
The oneCCL library provides various algorithms for each collective operation, as well as several possible settings for those algorithms. 

We have run a comprehensive set of tests to measure the performance of different oneCCL configurations when training various deep learning models at scale. 
Though the best oneCCL configuration is application dependent, we have identified a set of [optimal settings](https://docs.alcf.anl.gov/aurora/data-science/frameworks/oneCCL/) that work well for most applications and made those available in a dedicated module, called `frameworks_optimized`. 
The command `module show frameworks_optimized` lists the complete list of settings.

> **Note**: The settings in the `frameworks_optimized` module are valid for *typical* runs, for example assuming that the application is running on 12 ranks per node and setting the CPU bindings and oneCCL affinities accordingly. To override any of these settings, you can manually set a different value for any environment variable.


### ⌨️   Hands on

Here we show how to run the data parallel training example [`pytorch_ddp.py`](examples/04_AI_frameworks/pytorch_ddp.py`) described in [04_AI_frameworks.md](04_AI_frameworks.md) using the optimized setup of the `frameworks_optimized` module. 

The last two steps of the [hands-on example in 04_AI_frameworks.md](04_AI_frameworks.md#example%3A-training-a-pytorch-model-on-a-single-gpu-tile) should be replaced with the following ones:

1. Load the `frameworks_optimized` module:
   ```bash
   module use /soft/datascience/frameworks_optimized/
   module load frameworks_optimized
   ```
1. Run the script on 24 tiles, 12 per node:
   ```bash
   mpiexec -n 24 -ppn 12 --cpu-bind=${CPU_BIND} python pytorch_ddp.py
   ```
   where the environment variable `${CPU_BIND}` is set when loading the module.



## Additional Resources

- [oneCCL on Aurora documentation](https://docs.alcf.anl.gov/aurora/data-science/frameworks/oneCCL/)
- [Copper repository](https://github.com/argonne-lcf/copper/tree/main), [Copper documentation](https://alcf-copper-docs.readthedocs.io/en/latest/), [Copper paper](https://www.computer.org/csdl/proceedings-article/sc-workshops/2024/555400b320/23l2GFdlusU)

# [NEXT ->](06_DAOS.md)

