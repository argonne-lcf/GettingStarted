# Distributed vLLM Inference with Ensemble Launcher

This directory contains two approaches for running distributed vLLM inference across a multi-node HPC cluster using [Ensemble Launcher](https://github.com/argonne-lcf/ensemble_launcher). Both scripts distribute a model to all nodes, launch an EnsembleLauncher cluster, and run parallel inference — but they differ fundamentally in how work is submitted, how actors are managed, and how results flow back.

---

## Overview

| | Batched Inference | Request-Driven Inference |
|---|---|---|
| **Script** | `EL_batched_inference.py` | `EL_request_driven.py` |
| **Execution model** | Fire-and-forget tasks | Long-lived actors with request-response |
| **Work unit** | `Task` running `call_llm` | `ActorPool` managing `PrivateVLLMInference` actors |
| **Submission** | `ClusterClient.submit_batch()` | `ActorPool.invoke_all()` via ZMQ handles |
| **Model lifecycle** | Loaded and destroyed per task | Loaded once, serves many requests |
| **Communication** | None between tasks | ZMQ pipes between pools and actors |
| **Best for** | One-shot bulk workloads | Interactive / multi-round inference |

---

## Architecture

### Batched Inference

```
                  ┌──────────────┐
                  │ ClusterClient│
                  └──────┬───────┘
                         │ submit_batch(tasks)
                         ▼
              ┌─────────────────────┐
              │  EnsembleLauncher   │
              │  (Master / Workers) │
              └──────┬──────────────┘
                     │ schedule & dispatch
        ┌────────────┼────────────────┐
        ▼            ▼                ▼
   ┌─────────┐ ┌─────────┐     ┌─────────┐
   │ Task 0  │ │ Task 1  │ ... │ Task N  │
   │call_llm │ │call_llm │     │call_llm │
   │(in-proc │ │(in-proc │     │(in-proc │
   │  vLLM)  │ │  vLLM)  │     │  vLLM)  │
   └─────────┘ └─────────┘     └─────────┘
       │             │                │
       ▼             ▼                ▼
   Future.result  Future.result   Future.result
```

Each task is a standalone function (`call_llm`) that:
1. Finds a free port, sets up environment variables
2. Builds or loads a vLLM model cache
3. Loads the model and runs `vllm.LLM.generate()` in-process
4. Returns outputs and exits — the model is destroyed with the process

There is **no communication between tasks**. The scheduler assigns each task to a worker node with the required GPU resources, and results come back as `concurrent.futures.Future` objects.

### Request-Driven Inference

```
                  ┌──────────────────┐
                  │   Main Script    │
                  │  (pool_handle)   │
                  └───────┬──────────┘
                          │ invoke_all() via ZMQ
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
      ┌───────────┐ ┌───────────┐ ┌───────────┐
      │ ActorPool │ │ ActorPool │ │ ActorPool │
      │   pool-0  │ │   pool-1  │ │   pool-K  │
      └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
            │              │              │
       ┌────┼────┐    ┌────┼────┐    ┌────┼────┐
       ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
      ┌──┐ ┌──┐┌──┐ ┌──┐ ┌──┐┌──┐ ┌──┐ ┌──┐┌──┐
      │A0│ │A1││A2│ │A3│ │A4││A5│ │A6│ │A7││A8│
      └──┘ └──┘└──┘ └──┘ └──┘└──┘ └──┘ └──┘└──┘
       PrivateVLLMInference actors (long-lived)
```

Each `ActorPool`:
1. Is submitted to the cluster as a single task
2. Spawns N child `PrivateVLLMInference` actors as sub-tasks
3. Connects to each child via ZMQ request-response pipes
4. Forwards `invoke_all()` calls from the main script to all children
5. Collects and returns results back through the ZMQ handle

The `PrivateVLLMInference` actors are **long-lived processes** that:
1. Load the vLLM model once during `on_start()`
2. Listen for `generate` action calls over ZMQ
3. Run inference and return results to the parent pool
4. Stay alive until explicitly stopped

---

## Step-by-Step Flow

### Batched Inference Steps

1. **Parse arguments and discover nodes** — read CLI args, get node list from PBS.
2. **Distribute model** — `copy_model.sync_to_root()` copies the model to the head node's `/tmp`, then `copy_model.scatter_from_root()` uses MPI to broadcast it to all compute nodes.
3. **Optionally build vLLM cache** — `Warmup()` spawns a throwaway `PrivateVLLMInference` actor to pre-build the vLLM model info cache, avoiding redundant work across tasks.
4. **Create EnsembleLauncher** — configure system resources (CPUs, GPUs), hierarchy levels, and communication backend. Start the orchestrator in a background process.
5. **Partition prompts into chunks** — divide the prompt list evenly across `N` tasks (one task per model instance, `12 * nnodes / ngpus_per_model`).
6. **Create Task objects** — each `Task` wraps a `call_llm` call with its chunk of prompts. Tasks specify resource requirements (`nnodes`, `ppn`, `ngpus_per_process`).
7. **Submit batch** — `ClusterClient.submit_batch(tasks)` sends all tasks to the scheduler at once and returns a list of futures.
8. **Wait for completion** — `concurrent.futures.wait()` blocks until all futures resolve (or timeout). Results and exceptions are logged per task.
9. **Stop EnsembleLauncher** — terminate the orchestrator process.

### Request-Driven Inference Steps

1. **Parse arguments and discover nodes** — same as batched.
2. **Distribute model** — `copy_model.distribute_model()` (combined sync + scatter).
3. **Start EnsembleLauncher** — same cluster-mode setup, using `aurora_config` and `default_inference_launcher_config` for convenience.
4. **Create ActorPools** — divide the total actor count into chunks of `actors_per_process`. For each chunk:
   - Create a ZMQ pipe pair (server + client connection)
   - Instantiate an `ActorPool` with `PrivateVLLMInference` as the actor class
   - Submit the pool as a single task to the cluster
5. **Wait for pools to be ready** — `pool_handle.wait_for_ready()` blocks until all pools signal that their child actors have loaded the model and are listening.
6. **Warmup** — `pool_handle.invoke_all()` sends a short prompt to every actor through every pool. This ensures vLLM's internal caches (KV cache, CUDA graphs) are warm.
7. **Inference** — `pool_handle.invoke_all()` broadcasts the real prompt to all actors concurrently. Results stream back through the ZMQ pipes and are gathered via `asyncio.gather()`.
8. **Stop** — `pool_handle.stop()` sends shutdown signals to all pools, which propagate to child actors. Then stop the EnsembleLauncher.

---

## Key Differences

### Model Lifecycle

- **Batched**: Each `call_llm` task loads the model from scratch, runs inference, and exits. The model lives only for the duration of that task. This means every new batch of prompts pays the full model loading cost.
- **Request-Driven**: `PrivateVLLMInference` actors load the model once in `on_start()` and keep it in memory. Multiple rounds of `invoke_all()` can reuse the same loaded model without reloading.

### Communication Pattern

- **Batched**: No inter-task communication. Tasks are independent; the only coordination is through the scheduler and futures. This is simple but means there's no way to route specific prompts to specific model instances.
- **Request-Driven**: Full ZMQ-based request-response between the main script, ActorPools, and individual actors. This enables targeted routing, streaming results (`invoke_all_stream`), and fine-grained control over which actor handles which request.

### Fault Handling

- **Batched**: If a task fails, its future raises an exception. Other tasks are unaffected. Re-running means re-submitting the failed task (and reloading the model).
- **Request-Driven**: ActorPools support configurable `send_timeout` and `send_retries`. If an actor becomes unresponsive, the pool can retry. The actor's long-lived nature means transient failures can be retried without reloading the model.

### When to Use Which

| Scenario | Recommended Approach |
|---|---|
| Single large batch of prompts, run once | Batched |
| Multiple rounds of inference on the same model | Request-Driven |
| Simple workload, minimal coordination needed | Batched |
| Need to control which actor handles which prompt | Request-Driven |
| Interactive or streaming inference | Request-Driven |
| Maximum simplicity | Batched |
| Maximum throughput over sustained workloads | Request-Driven |
