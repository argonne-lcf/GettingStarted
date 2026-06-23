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

Below we compare the weak-scaling performance of batched and request-driven approaches implemented with different workflow tools on Aurora. The tests deploy one engine per PVC tile (12 per node) with the Llama 3.1 8B model and perform 32 prompt requests per inference engine based on the [prompts.jsonl](./utils/prompts.jsonl) file (the length of the response is limited to 128 tokens, `max_model_len=8192` and for offline LLM the batch size is set to 16).

![Weak scaling tests of LLM inference with various approaches on Aurora. The tests deploy one engine per PVC tile (12 per node) with the Llama 3.1 8B model and perform 32 prompt requests per inference engine based on the [prompts.jsonl](./utils/prompts.jsonl) file. The length of the response is limited to 128 tokens, `max_model_len=8192` and for offline LLM the batch size is set to 16.](./utils/requests_per_second_aurora.png)

Below, we show performance of the same weak scaling tets on Polaris (one engine per A100 GPU).

![Weak scaling tests of LLM inference with various approaches on Polaris. The tests deploy one engine per A100 GPU (4) per node) with the Llama 3.1 8B model and perform 32 prompt requests per inference engine based on the [prompts.jsonl](./utils/prompts.jsonl) file. The length of the response is limited to 128 tokens, `max_model_len=8192` and for offline LLM the batch size is set to 16.](./utils/requests_per_second_polaris.png)


## Choosing the Correct Scaling Approach

We could have a section guiding users to the specific approach based on questions about their workflow, but it may repeat some of the information in the intriduction.

## Batched Inference with MPI

For the batched inference case, MPI is an easy and performant solition. 
An example Python script is provided in the [MPI](./MPI/) directory along with submit scripts for both Aurora and Polaris. In this case, a Python script is launched across nodes with `mpiexec` with as many processes per node as desired inference engines per node. Each rank initializes a `vllm.LLM()` object based on various input parameters and performs inference on a batch of prompts distributed by rank 0. 

**Note**: this MPI approach is limited to models that fit within a single node (4 A100 GPUs on Polaris and 12 PVC tiles on Aurora). For large models which require the memory from multiple nodes, a scalable approach is still under development. Please contact support@alcf.anl.gov with questions. 

### Set up

No specific set up is required for this approach since it leverages the vLLM installations that come with the data science modules on Aurora and Polaris. The modules to load are shown below for completeness, however these are loaded within the submit scripts provided.

```bash
# On Aurora
module load frameworks

# On Polaris
module use /soft/modulefiles
module load conda
conda activate
```

### Run the examples

For both Aurora and Polaris, there are submit scripts to run a small, single-GPU model with tensor parallelism (`TP`) size of 1 and an example running a larger model requirung `TP>1`. The main difference to note for these two scenarios is the number of ranks per node and how their respective GPU and CPU bindings are set. 

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

## Batched Inference with EL

While MPI is a performant tool for the batched inference case, many workflow tools can support this workload as well. Here, we demonstrate how to use [EnsembleLauncher (EL)](https://github.com/argonne-lcf/ensemble_launcher) to set up the static batched inference approach on Aurora and Polaris. EL is a lightweight, scalable Python tool developed at ALCF for launching and orchestrating task ensembles across HPC clusters with intelligent resource management and hierarchical execution. 

The main idea for this implementation of the workflow (see [EL_batched_inference.py](./EL/EL_batched_inference.py)) is that LLM inference, via the offline `vllm.LLM()` API, *and* the prompts are sent together to EL as tasks and are executed as transient processes, meaning that once the task is done with its workload it quits. Once again, this approach works well for cases where there is a static, *a priori* defined set of work to be efficiently distributed and parallelized across many resources. 

### Set up

To install EnsembleLauncher on ALCF systems, simply create a new virtual environment, clone the repository, and install it as shown below. Note, however, that the submit scripts are set up to install the virtual environments directly in `/tmp` on the compute nodes at the beginning of the job.

```bash
# On Aurora
module load frameworks
python -m venv _env --system-site-packages
source _env/bin/activate
git clone https://github.com/argonne-lcf/ensemble_launcher.git
cd ensemble_launcher
pip install .
cd ..

# On Polaris
module use /soft/modulefiles
module load conda
conda activate base
python -m venv _env --system-site-packages
source _env/bin/activate
git clone https://github.com/argonne-lcf/ensemble_launcher.git
cd ensemble_launcher
pip install .
cd ..
```

### Run the examples

For both Aurora and Polaris, there are submit scripts to run a small, single-GPU model with tensor parallelism (`TP`) size of 1 and an example running a larger model requirung `TP>1`. 

To run the examples, simply submit the scripts provided for Aurora and Polaris. For example, to run the example with the Llama 3.1 8B model on Aurora execute

```bash
qsub sub_el_batched_aurora_llama8B.sh
```

Note that EL provides API to transfer the model weights and the vLLM cache to `/tmp` on the compute nodes, thus the [bcast.c](./utils/bcast.c) utility is not needed in this case. However, it is still used to broadcast the virtual environment.

The [EL_batched_inference.py](./EL/EL_batched_inference.py) script takes a few runtime parameters to note:

* The model name (required)
* The cache directory where the model weights are stored (required)
* The tensor parallel (TP) size to use for the model (defaults to 1)
* The data type to use (defaults to `bfloat16`)
* The maximum number of output tokens (defaults to 128). This parameter can impact performance significantly and may need to be adjusted depending on the expected length of the LLM response.
* The prompt batch size (defaults to 1). Increasing the batch size can improve overall inference throughput (requests per second) depending on the available memory for the KV cache pool.
* The number of inference engined per node to launch (defaults to as many as can fit based on TP size and number of GPUs on the node).
* File containing the prompts to be used for inference (defaults to [prompts.jsonl](./utils/prompts.jsonl)). The script is set up to weak scale the workflow by replicating the prompts for as many inference engines requested.



### Request-driven Inference with EL

The setup and run for request-driven inference with EL is similar to the batched case, however the main difference is that the prompts are submitted to EL dynamically.

A deatailed readme for this example is provided in the [EL](./EL/) directory, however to run the example, simply submit the provided script for Aurora or Polaris. For example, to run the example on Aurora execute

```bash
qsub submit_request_driven.sh
```

## Request-driven Inference with Dragon

[Dragon](https://dragonhpc.org/portal/index.html) is a composable distributed run-time for managing processes, memory, and data at scale through high-performance communication objects, thus it is a valuable tool for designing and executing workflows on HPC systems.
Among many other valuable features (see the [Dragon getting started examples](../../Workflows/dragonhpc/README.md)), the Dragon Python API also provides an [LLM Inference API](https://dragonhpc.github.io/dragon/doc/_build/html/ref/ai/inference/index.html) which can be used to distribute inference workloads to multiple nodes or integrate AI with scientific workflows. Their API leevrages the offline `vllm.LLM()` API, however it combines this with their multi-mode Python multiprocessing extension to enable across-node, request-driven inference. An example showing how to use Dragon for large-scale LLM inference is provided in the [Dragon](./Dragon/) directory, along with job scripts for both Polaris and Aurora. 

**Note**: Currently, Dragon only supports deploying models that fit within a single node (4 A100 GPUs on Polaris and 12 PVC tiles on Aurora). HPE is working on solutions for deploying large models which require the memory from multiple nodes, and this support will be made available in future releases. 

### Set up

To install Dragon on ALCF systems, simply create a new virtual environment and pip install `dragonhpc` as shown below. Note, however, that the submit scripts are set up to install the virtual environments directly in `/tmp` on the compute nodes at the beginning of the job.

```bash
# On Aurora
module load frameworks
python -m venv _env --system-site-packages
source _env/bin/activate
pip install dragonhpc
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64

# On Polaris
module use /soft/modulefiles
module load conda
conda activate base
python -m venv _env --system-site-packages
source _env/bin/activate
pip install dragonhpc
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/2.2.0rc1/lib64
```

The last installation step of running `dragon-config` configures `dragon` to use fast RDMA transfers across the Slingshot network present on both systems.  Without this step, `dragon` would run in the default mode that uses slower TCP transfers.


### Run the examples

For both Aurora and Polaris, there are submit scripts to run a small, single-GPU model with tensor parallelism (`TP`) size of 1 and an example running a larger model requirung `TP>1`. 

To run the examples, simply submit the scripts provided for Aurora and Polaris. For example, to run the example with the Llama 3.1 8B model on Aurora execute

```bash
qsub sub_dragon_aurora_llama8B.sh
```

The [dragon_llm_inference.py](./Dragon/dragon_llm_inference.py) script takes a few runtime parameters to note:

* The Hugging Face token (required)
* The model name (required)
* The tensor parallel (TP) size to use for the model (defaults to 1)
* The data type to use (defaults to `bfloat16`)
* The maximum number of output tokens (defaults to 128). This parameter can impact performance significantly and may need to be adjusted depending on the expected length of the LLM response.
* The prompt batch size (defaults to 1). Increasing the batch size can improve overall inference throughput (requests per second) depending on the available memory for the KV cache pool.
* File containing the prompts to be used for inference (defaults to [prompts.jsonl](./utils/prompts.jsonl)). The script is set up to weak scale the workflow by replicating the prompts for as many inference engines requested.


## Tips for scaling

Mention things like moving env, model weights and cache to /tmp on each of the nodes and provide utilities for these things.

