# Data Streaming with ADIOS2 SST

This example demonstrates how to run a producer-consumer workflow with ADIOS2 SST streaming. The workflow is composed of a proxy producer simulation written in C++ (`sim.cpp`) and a proxy consumer application written in Python (`trainer.py`). This setup could be used to stram simulation data from an ongoing simulation to a post-prtocessing, visualization or ML training/inference script.

The workflow initially shares some data through the file system with the BP5 engine, and then sets up an iteration loop within both the producer and consumer where data is streamed between the two. The workflow is also set up to run with the producer on one set of nodes and the consumer on the other set of nodes in order to force data streaming through the network. The submit script show how to set this case up with sequential `mpiexec` commands and with a single command in MPMD mode.


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