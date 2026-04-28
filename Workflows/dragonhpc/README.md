# Dragon HPC

Dragon HPC is a composable distributed run-time for managing processes, memory, and data at scale through high-performance communication objects.

This demo focuses on the Dragon's Python and C++ API (more can be found in the [dragon docs](https://dragonhpc.org/portal/index.html)).

The Dragon python API uses python's `multiprocessing` API.  Dragon can therefore be used to extend scripts written for single shared memory devices with `multiprocessing` to run on multiple nodes on HPC systems.  Dragon also has a Distributed Dictionary that can be used by processes across the runtime to store and transfer data within memory pools arcross multiple nodes.  This demo will show multiple ways of running applications and tasks with Dragon process launching tools and how to make use of the distributed dictionary.

## Get Interactive Nodes

We'll run these tests in an interactive session, however a sample submit script is provided to test running in a batch job.  Replace `alcf_training` with your project name.

```shell
qsub -I -A alcf_training -l select=2 -l walltime=0:30:0 -l filesystems=home:eagle -q debug
```

## Install

To install `dragonhpc` in an environment on Polaris:
```shell
module unload xalt
module load conda
conda activate base
python -m venv _env --system-site-packages
source _env/bin/activate
pip install dragonhpc
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/2.2.0rc1/lib64
```

To install `dragonhpc` in an environment on Aurora:
```shell
module load frameworks
python -m venv _env --system-site-packages
source _env/bin/activate
pip install dragonhpc
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64
```

The last installation step of running `dragon-config` configures `dragon` to use fast RDMA transfers across Polaris' slingshot network.  Without this step, `dragon` would run in the default mode that uses slower TCP transfers.

## Dragon Pool

Dragon's Python API uses Python's `multiprocessing` API.  Dragon can therefore be used to extend scripts written for single shared memory devices with `multiprocessing` to run on multiple nodes.

This example shows how to run tasks with `Pool` in two different ways.  The first way uses `multiprocessing Pool` with the start method set to `dragon`.  This allows for pool processes to be distributed across nodes, but it is not possible to bind them to paricular gpus or cpus.

The second approach uses Dragon's native Pool object that allows a list of dragon Policies to be passed to the `Pool` that specify the gpu affinity.

These tests run a python function that sleeps and reports the hostname and GPU tile it sees pinned by `CUDA_VISIBLE_DEVICES` or `ZE_AFFINITY_MASK`.

Dragon scripts are launched with the `dragon` application, included in the demo environment.  To run the example script:

```shell
dragon 0_dragon_pool.py
```

## Dragon ProcessGroup

Dragon provides another way of distributing tasks to groups of workers called the dragon `ProcessGroup`.  Every process in the `ProcessGroup` can be given a specific node, gpu, and cpu, allowing for finegrained control of processes.

There are two example cases using `ProcessGroup`.  `1_dragon_process_group.py` distributes a simple python function to independent processes across the allocation.  This is done by assigning a `Policy` to each process in the group that specifies the node, gpu and cpu where it is to be run.

```shell
dragon 1_dragon_process_group.py
```

`2_dragon_mpi_process_group.py` uses `ProcessGroup` to run an MPI-enabled executable, where every process in the group is an MPI rank.  To enable message passing between processes in a process group, the `pmi` flag needs to be set in the `ProcessGroup` as shown in `2_dragon_mpi_process_group.py`.

## Dragon Distributed Dictionary (Python Client API)

In addition to task launching, `dragon` provides a distributed data layer in its runtime called the `DDict` or Dragon Dictionary.  The `DDict` provisions pools of memory on nodes across the runtime where key-value data pairs can be stored.  Any process in the runtime can access any key-value pair in the `DDict`.  This is enabeld by dictionary manager processes that are created in the background within the runtime and manage the transfer of data between nodes.

`DDict`s can be used with `Pool` or `ProcessGroup`.  Multiple processes or `Pool`s and `ProcessGroup`s can use the same `DDict`.  This enables efficient sharing of data between processes in the runtime.

`3_dragon_dictionary.py` gives a simple example of how to create a `DDict` and use a `Pool` to store data within it.  The main process (running on the head node) then retrieves all the data stored in the `DDict`.

```shell
dragon 3_dragon_dictionary.py
```

## Dragon Distributed Dictionary (C++ Client API)

To demonstrate how a traditional simulation-AI workflow, which uses C++ for the simulation code, would leverage Dragon's DDict, we provide a siomple [producer-consumer](./5_producer_consumer/) workflow. 
The workflow is composed of a proxy producer simulation written in C++ (`sim.cpp`) and a proxy consumer application written in Python (`trainer.py`). The driver script (`driver.py`) used the Dragon DDict, Process and ProcessGroup API to orchestrate the components. 
This setup could be used to stream simulation data from an ongoing simulation to a post-processing, visualization or ML training/inference script.

The workflow example sets up iteration loops within both the producer and consumer to emulate a time-stepping algorithm or a training algorithm. On one side, on each iteration the producer writes data to the DDict with a unique key for every rank, thus overwriting the previously stored data with the latest "snapshot". On the other side, on each iteration the consumer reads the latest available data. In this case, the number of producer and consumer MPI ranks is the same and data is written/read by ranks with the same ID on both sides (i.e., data produced by rank 0 of the simulation is read by rank 0 of the trainer). Note this is not a requirement of Dragon workflows; the number of producer and consumer ranks need not match and there can be arbitrary numbers of producers and consumers connected to the same (or multiple) DDicts allowing great flexibility.

The workflow driver script supports three launcher modes: colocated, clustered and mixed. 
- In the colocated mode, the DDict, producer and consumer share the same set of nodes and the resources within those nodes (e.g., trainer runs on 6/12 GPUs on Aurora, simulation runs on the other 6 GPUs, and the DDict is distributed on the CPUs of those same nodes). Specifically, for the colocated mode, each client interacts directly with the *local manager* of the DDict, meaning that data is written/read to the DDict memory pool on the same node as the client and therefore eliminating any inter-node communication across DDict managers.
- In the clustered mode, the total set of nodes is split into three distinct lists, and each component of the workflow is executed on one of those node lists. 
- In the mixed mode, the components are arranged in the same way as the colocated mode, but the client do not interact with the local DDict manager so the data is allowed to move across nodes when being written/read.
Which launcher mode is more appropriate depends on the nature of the workflow. The colocated mode is most scalable, however it limits the view of the data to only what is available on node. The clustered mode is less scalable since data moves across nodes, however it allows all clients to access all data. The mixed approach provides a hybrid of the two solutions. 

## Submitting to PBS

To submit the first three tests to multiple nodes with PBS, use the following submit script:

```shell
# On Polaris
qsub 4_submit_polaris.sh

# On Aurora
qsub 4_submit_aurora.sh
```

To submit the producer-consumer workflow, use the following submit script

```shell
# On Aurora
qsub 6_submit_aurora.sh
```
