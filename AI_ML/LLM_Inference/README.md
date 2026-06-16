# Scaling LLM Inference on ALCF Systems

When it comes to scaling LLM inference workflows on ALCF systems, there are a few recommended approaches depending on the user's needs and setup.
Here we describe these approaches in detail and provide useful examples and scripts for users to get started with scaling their own workflows. 
Note that all examples use the vLLM packege since this is the officially supported inference engine across our systems, however users may use other libraries.

At a high level, we separate inference workflows into two categories; high-throughput batched (static) inference and request-driven (dynamic) inference. 
In the **batched inference** case, the work is often static or determined *a priori* and homogeneous in nature. For example, users may want to perform inference on a long list of pre-determined prompts using the same model or a few different models in the shortest possible time. In such a case, various inference engines of the same kind can all be launched in unison and the work can be easily partitioned among the engines to ensure good load balancing. 
MPI is therefore a good candidate for efficient launching of such static workflows and to ensure high-throughput, however other wokflow tools, such as EnsembleLauncher (EL), can be used as well. 
Additionally, the offline approach to vLLM inference which uses the Python `LLM()` API is preferred due to its batching functionality. 

In the **request-driven inference** case, the idea is to launch persistent processes or servers which stand up inference engines and continuously listen and wait for prompts to be submitted in a dynamic and asynchronous pattern. For example, agentic or human-driven workflows often require such a pattern, since the prompts are generated *on-the-fly* as the workflow progresses.
Additionally, it may also be necessary to serve differenct models tailored to different tasks. 
Due to the increased complexity of such workflows, the MPI based solution is less likely and instead workflow tools provide efficient and scalable solutions by routing the prompts to the various inference engines as they are generated.
In the context of vLLM, this is the classic fit for the `vllm serve` CLI command, however we'll see that the Python `LLM()` API can still be used when wrapping it around a workflow harness such as EL and Dragon to proivide both the request-driven advantage `vllm serve` and the high-throuput of the Python `LLM()` API.

Below we compare the performance of batched and request-driven approaches implemented with different workflow tools on Aurora. 

**Insert performance plot.**


### Choosing the Correct Scaling Approach

We could have a section guiding users to the specific approach based on questions about their workflow, but it may repeat some of the information in the intriduction.

### Batched Inference with MPI

For the batched inference case, MPI is an easy and performant solition. 
An example Python script is provided in the [MPI](./MPI/) directory along with submit scripts for both Aurora and Polaris.

In this case, a Python script is launched across nodes with `mpiexec` with as many processes per node as desired inference engines per node. Each rank initializes a `vllm.LLM()` object based on various input parameters and performs inference on a batch of prompts distributed by rank 0. 

#### Set up

No specific set up is required for this approach since it leverages the vLLM installations that come with the data science modules on Aurora and Polaris. The modules to load are shown below for completeness, however these are loaded within the submit scripts provided.

```bash
# On Aurora
module load frameworks

# On Polaris
module use /soft/modulefiles
module load conda
conda activate
```

#### Run the examples

For both Aurora and Polaris, there are submit scripts to run a small, single-GPU model with tensor parallelism (`TP`) size of 1 and an example running a larger model requirung `TP>1`. The main difference to not for these scenarios is how 

To run the examples, simply submit the scripts provided for Aurora and Polaris. For example, to run the example with the Llama 3.1 8B model on Aurora execute

```bash
qsub sub_mpi_aurora_llama8B.sh
```

The [mpi_llm_inference.py](./MPI/mpi_llm_inference.py) script takes in a few runtime parameters to note:

* The Hugging Face token (required)
* The model name (required)
* The tensor parallel (TP) size to use for the model (defaults to 1)
* The data type to use (defaults to `bfloat16`)
* The maximum number of output tokens (defaults to 128). This parameter can impact performance significantly and may need to be adjusted depending on the expected length of the LLM response.
* The prompt batch size (defaults to 1). Increasing the batch size can improve overall inference throughput (requests per second) depending on the available memory for the KV cache pool.
* File containing the prompts to be used for inference (defaults to [prompts.jsonl](./utils/prompts.jsonl)). The script is set up to weak scale the workflow by replicating the prompts for as many inference engines requested.

### Batched Inference with EL

Like the MPI approach, batch inference can be useful when you have a static list of prompts. Here, we used EnsembleLauncher (EL) to launch inference calls as a EL Tasks.

### Set up
1. Install EL and dependencies in a Python virtual environment (https://github.com/argonne-lcf/ensemble_launcher/tree/multi_node_vllm).
2. Download model weights or use the ones already available on the system. If downloading, ensure the weights are available on a shared file system accessible by all nodes or use utilities to copy them to each node's local storage.

### Run the examples
An example EL workflow is provided in the [EL](./EL/) directory. To run the example, simply submit the provided script for Aurora or Polaris. For example, to run the example on Aurora execute

```bash
qsub submit_batched.sh
```
### Request-driven Inference with EL

The setup and run for request-driven inference with EL is similar to the batched case, however the main difference is that the prompts are submitted to EL dynamically.

A deatailed readme for this example is provided in the [EL](./EL/) directory, however to run the example, simply submit the provided script for Aurora or Polaris. For example, to run the example on Aurora execute

```bash
qsub submit_request_driven.sh
```

### Request-driven Inference with Dragon

## Batched Inference with MPI

### Set up

### Run

## EnsembleLauncher (EL)

### Set up

### Run

## DragonHPC

### Set up

On Aurora

```
module load frameworks
python -m venv _dragon_venv --system-site-packages
source _dragon_venv/bin/activate

# Install dragonhpc
python3 -m pip install dragonhpc
dragon-config add --ofi-runtime-lib=="/opt/cray/libfabric/1.22.0/lib64/"
```

### Run

```
qsub sub_dragon_aurora.sh
```


## Tips for scaling

Mention things like moving env, model weights and cache to /tmp on each of the nodes and provide utilities for these things.

