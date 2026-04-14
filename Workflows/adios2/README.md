# Data Streaming with ADIOS2 SST

This example demonstrates how to run a producer-consumer workflow with ADIOS2 SST streaming. The workflow is composed of a proxy simulation written in C++ (`sim.cpp`) and a proxy consumer application written in Python (`trainer.py`).


## Biuld the Example

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