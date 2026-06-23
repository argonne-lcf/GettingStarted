import asyncio
import concurrent.futures
import argparse
import os
import time
import uuid
import json
import sys

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import aurora_config, polaris_config
from ensemble_launcher.ensemble import Task
from ensemble_launcher.helper_functions import get_nodes, get_gpus
from ensemble_launcher.inference import copy_model, default_inference_launcher_config
from ensemble_launcher.inference.utils import call_llm
from ensemble_launcher.orchestrator import ClusterClient

from utils import parse_args, get_logger
logger = get_logger("main_offline", log_dir=f"{os.getcwd()}/script_logs")


async def async_main():
    start_time = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="EL inference with vLLM")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name to load",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Directory where model weights are found",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Determines the tensor parallel size to use for the run.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="bfloat16",
        help="Determines the data type to use for the run.",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens in the response.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for prompt batching.",
    )
    parser.add_argument(
        "--engines_per_node",
        type=int,
        default=-1,
        help="Number of inference engines per node (default -1 means as many as can fit)",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="../utils/prompts.jsonl",
        help="File containing input prompts",
    )
    args = parser.parse_args()

    # Get system info
    nodes = get_nodes()
    gpus, gpu_type = get_gpus()
    if not gpus:
        print(f"No GPUs found on system", flush=True)
        sys.exit(1)
    num_gpus = len(gpus)
    print(f"Running on {len(nodes)} nodes with {num_gpus} GPUs", flush=True) 

    # Read prompts from file and chunk them
    try:
        prompts = []
        for line in open(args.prompt_file):
            data = json.loads(line)
            prompts.append(data["prompt"])
        print(f"Read prompts from {args.prompt_file}", flush=True)
    except FileNotFoundError:
        print(f"Error: The prompt file {args.prompt_file} was not found.", flush=True)
        sys.exit(1)
    
    num_inf_engines = (num_gpus // args.tp_size) \
        if args.engines_per_node == -1 else args.engines_per_node
    num_inf_engines *= len(nodes)
    prompts = prompts * num_inf_engines  # NOTE: only needed for weak scaling
    num_prompts = len(prompts)
    print(f"Submitting {num_prompts} prompts to {num_inf_engines} inference engines", flush=True)
    chunks = [[] for _ in range(num_inf_engines)]
    for i, prompt in enumerate(prompts):
        chunks[i % num_inf_engines].append(prompt)

    # Create EL system and launcher configs
    ckpt_dir = f"{os.getcwd()}/ckpt_{str(uuid.uuid4())}"
    if gpu_type == "intel":
        sys_config = aurora_config
        launcher_config = default_inference_launcher_config(len(nodes), ckpt_dir)
    elif gpu_type == "nvidia":
        sys_config = polaris_config
        launcher_config = default_inference_launcher_config(
            len(nodes), ckpt_dir, {"gpu_selector": "CUDA_VISIBLE_DEVICES"}
        )
    else:
        print("Unknown system, must specify a custom system config")
        sys.exit(1)

    # Start EL
    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launcher_config
    )
    tic = time.perf_counter()
    print("Starting EnsembleLauncher ...", flush=True)
    el.start(wait_time=1)
    print(f"EnsembleLauncher ready in {(time.perf_counter() - tic):.1f} s", flush=True)

    # Define vllmn engine parameters
    vllm_engine_params = {
        "tensor_parallel_size": args.tp_size,
        "enforce_eager": True,
        "max_model_len": 8192,
        "dtype": args.data_type,
        "gpu_memory_utilization": 0.90, # safe to ask for 90% of GPU memory
        "max_num_seqs": args.batch_size,
    }
    vllm_sampling_params = {
        "max_tokens": args.max_output_tokens,
    }
    
    # Create tasks, each taking a chunk of the total prompts
    llm_tasks = []
    num_tasks = num_inf_engines
    cpu_cores_per_task = 8
    for i in range(num_tasks):
        task_id = f"task-{i}"
        llm_tasks.append(
            Task(
                task_id=task_id,
                nnodes=1, # single node models only for now
                ppn=cpu_cores_per_task, # number of cores for each task
                ngpus_per_process=args.tp_size/cpu_cores_per_task, # this times ppn has to equal TP size
                executable=call_llm,
                args=(args.model_name, args.cache_dir, chunks[i]),
                kwargs={"llm_kwargs": vllm_engine_params, "sampling_kwargs": vllm_sampling_params},
            )
        )

    # Submit tasks with EL client-cluster approach
    tic = time.perf_counter()
    with ClusterClient(checkpoint_dir=ckpt_dir, checkpoint_timeout=300) as client:
        # Submit tasks
        llm_futures = []
        llm_futures = client.submit_batch(tasks=llm_tasks)
        print(f"Submitted {num_tasks} LLM inference tasks across {len(nodes)} nodes", flush=True)

        # Wait for task completion
        done, pending = concurrent.futures.wait(llm_futures, timeout=300)

        # Check completed tasks
        if len(done) == num_tasks:
            print(f"Done with all tasks in {(time.perf_counter() - tic):.1f} s", flush=True)
        else:
            print(
                f"Only {len(done)}/{num_tasks} tasks completed. "
                "May have to increase the timeout",
                flush=True
            )
        
        # Get successful requests, LLM responses and timings
        successful_requests = 0
        tot_times, init_times, inf_times, rpss = [], [], [], []
        for i, f in enumerate(done):
            if f.exception() is not None:
                print(f"Task {i} failed with exception: {f.exception()}", flush=True)
            else:
                result = f.result()
                responses = result["responses"]
                if len(responses) == len(chunks[i]):
                    successful_requests += len(responses)
                tot_times.append(result["total_time"])
                init_times.append(result["initialization_time"])
                inf_times.append(result["inference_time"])
                rpss.append(len(responses)/result["inference_time"])

    tic = time.perf_counter()
    print("Stopping EnsembleLauncher ...")
    el.stop()
    print(f"EnsembleLauncher stopped in {(time.perf_counter() - tic):.1f} s", flush=True)
    total_runtime = time.time() - start_time

    # Print summary of performance stats
    print("\n\n=========================================")
    print("Performance Summary:")
    print(f"Total number of input prompts: {num_prompts}")
    print(f"Total number of successful requests: {successful_requests}")
    print(f"Total run time = {total_runtime:.4f} s")
    print(f"Initialization time (min, max, avg) = {min(init_times):.4f}, {max(init_times):.4f}, {sum(init_times)/len(init_times):.4f} s")
    print(f"Inference time (min, max, avg) = {min(inf_times):.4f}, {max(inf_times):.4f}, {sum(inf_times)/len(inf_times):.4f} s")
    print(f"Total successful requests per second (requests / inference time) : {sum(rpss):.4f}")


if __name__ == "__main__":
    asyncio.run(async_main())
