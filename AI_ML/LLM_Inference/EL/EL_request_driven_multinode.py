import argparse
import asyncio
import json
import os
import secrets
import sys
import time
import uuid

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.comm import transport_registry
from ensemble_launcher.config import aurora_config, PolicyConfig
from ensemble_launcher.ensemble import ActorPool
from ensemble_launcher.helper_functions import get_gpus, get_nodes
from ensemble_launcher.inference import (
    default_inference_launcher_config,
    PrivateMultiNodeVLLMInference
)
from ensemble_launcher.orchestrator import ClusterClient

from utils import get_logger
logger = get_logger("main_offline", log_dir=f"{os.getcwd()}/script_logs")


async def async_main():
    start_time = time.perf_counter()

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
        "--pp_size",
        type=int,
        default=1,
        help="Determines the pipeline parallel size to use for the run.",
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
        "--actors_per_pool",
        type=int,
        default=4,
        help="Number of EL Actors in an ActorPool",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="../utils/prompts.jsonl",
        help="File containing input prompts",
    )
    args = parser.parse_args()

    # Parse arguments and get system info
    nodes = get_nodes()
    gpus, gpu_type = get_gpus()
    if not gpus:
        print(f"No GPUs found on system", flush=True)
        sys.exit(1)
    if gpu_type != "intel":
        print(f"EnsembleLauncher only implemented for Aurora currently", flush=True)
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
        print(
            f"Error: The prompt file {args.prompt_file} was not found.",
            flush=True,
        )
        sys.exit(1)
    
    num_inf_engines = (num_gpus // args.tp_size) \
        if args.engines_per_node == -1 else args.engines_per_node
    num_inf_engines *= (len(nodes) // args.pp_size)
    prompts = prompts * num_inf_engines  # NOTE: only needed for weak scaling
    num_prompts = len(prompts)
    prompt_chunks = [[] for _ in range(num_inf_engines)]
    for i, prompt in enumerate(prompts):
        prompt_chunks[i % num_inf_engines].append(prompt)
    print(f"Submitting {num_prompts} prompts to {num_inf_engines} inference engines", flush=True)

    # Define vllm engine parameters
    vllm_engine_params = {
        "tensor_parallel_size": args.tp_size,
        "pipeline_parallel_size": args.pp_size,
        "enforce_eager": True,
        "max_model_len": 8192,
        "dtype": args.data_type,
        "gpu_memory_utilization": 0.90,  # safe to ask for 90% of GPU memory
        "max_num_seqs": args.batch_size,
    }
    vllm_sampling_params = {
        "max_tokens": args.max_output_tokens,
    }

    # Create EL system and launcher configs
    ckpt_dir = f"{os.getcwd()}/ckpt_{str(uuid.uuid4())}"
    sys_config = aurora_config
    launcher_config = default_inference_launcher_config(len(nodes), 
                                                        ckpt_dir, 
                                                        task_executor_name="async_mpi", 
                                                        policy_config=PolicyConfig(nlevels = 1 if len(nodes) < 256 else 2, 
                                                                                   leaf_nodes=max(len(nodes) // args.pp_size, 1)))

    # Start EL
    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launcher_config
    )
    tic = time.perf_counter()
    print("Starting EnsembleLauncher ...", flush=True)
    el.start(wait_time=3)
    print(f"EnsembleLauncher ready in {(time.perf_counter() - tic):.1f} s", flush=True)

    # Create ActorPools
    n_actors = num_inf_engines
    actor_chunks = [
        list(range(i, min(i + args.actors_per_pool, n_actors)))
        for i in range(0, n_actors, args.actors_per_pool)
    ]
    num_actor_pools = len(actor_chunks)

    transport = transport_registry.get("zmq")["transport"]()
    server_id = "global_server"
    server_secret = secrets.token_hex(16)
    pool_ids = []
    pool_futures = []
    
    actor_kwargs = {
        "model": args.model_name,
        "cache_dir": args.cache_dir,
        "use_cached_modelinfo": os.environ.get("VLLM_CACHE_ROOT", None) is not None,
        "model_info_cache": os.environ.get("VLLM_CACHE_ROOT", None),
        "llm_kwargs": vllm_engine_params,
    }
    task_kwargs = {
        "nnodes": args.pp_size,
        "ppn": args.tp_size,
        "ngpus_per_process": 1, # this times ppn has to equal TP size
    }

    # Create an EL Client
    with ClusterClient(
        checkpoint_dir=ckpt_dir, checkpoint_timeout=300
    ) as client:
        # Submit ActorPools to EL Cluster (this initializes vLLM engines)
        init_start = time.perf_counter()
        for i, chunk in enumerate(actor_chunks):
            pool_name = f"pool-{i}"
            server, pool_client = transport.create_child_pipe(
                server_id,
                server_secret,
                pool_name,
                server_secret,
                req_res=True,
            )
            actor_pool = ActorPool(
                name=pool_name,
                client_conn=pool_client,
                actor_class=PrivateMultiNodeVLLMInference,
                n_actors=len(chunk),
                actor_kwargs=actor_kwargs,
                task_kwargs=task_kwargs,
                checkpoint_dir=ckpt_dir,
                req_res=True,
                child_ready_timeout=600,
            )
            pool_task = actor_pool.create_task(task_id=pool_name, nnodes=1, ppn=1)
            pool_futures.append(client.submit(pool_task))
            pool_ids.append(f"{pool_name}:{server_secret}")
        pool_handle = ActorPool.create_handle(
            server,
        )
        await pool_handle.open()
        print(f"Waiting for {len(pool_ids)} pools to be ready...")
        try:
            await pool_handle.wait_for_ready(expected=len(pool_ids), timeout=300)
            ready_pools = pool_handle.ready_actors
        except asyncio.TimeoutError:
            for pid, future in enumerate(pool_futures):
                if future.done():
                    print(f"Pool {pid} failed with error: {future.exception()}")
            ready_pools = pool_handle.ready_actors
        init_time = time.perf_counter() - init_start
        if len(ready_pools) != num_actor_pools:
            print(
                f"Only {len(ready_pools)}/{num_actor_pools} ActorPools are ready. "
                "Try increasing the timeout time.",
                flush=True)
            sys.exit(1)

        # Define function to call llm.generate() for all actors
        async def _run_pool(pool_id, msg):
            _, results = await pool_handle.invoke_all(msg, actor_id=pool_id)
            print(f"Pool {pool_id.split(':')[0]} completed", flush=True)
            return results

        # Perform inference
        # Create new tasks which call llm.generate() for each actor
        inf_start = time.perf_counter()
        tasks = []
        task_to_pool_idx = {}
        for idx, pid in enumerate(ready_pools):
            messages = []
            for chunk in prompt_chunks[idx * args.actors_per_pool : (idx + 1) * args.actors_per_pool]:
                messages.append(("generate", (chunk,), {"sampling_params": vllm_sampling_params}))
            t = asyncio.create_task(_run_pool(pid, messages))
            tasks.append(t)
            task_to_pool_idx[t] = idx

        done, pending = await asyncio.wait(tasks, timeout=180)
        inf_time = time.perf_counter() - inf_start

        if len(pending) != 0:
            print(
                f"{len(done)}/{len(pool_ids)} pools completed. "
                "May have to increase the timeout",
                flush=True
            )
            sys.exit(1)

        # Get successful requests from the pool results
        successful_requests = 0
        for t in done:
            pool_idx = task_to_pool_idx[t]
            if t.exception() is not None:
                print(f"Pool {pool_idx} failed with exception: {t.exception()}", flush=True)
                continue
            pool_results = t.result()
            expected_chunks = prompt_chunks[
                pool_idx * args.actors_per_pool : (pool_idx + 1) * args.actors_per_pool
            ]
            for actor_idx, responses in enumerate(pool_results):
                if responses is not None and len(responses) == len(expected_chunks[actor_idx]):
                    successful_requests += len(responses)
        rps = successful_requests / inf_time

        # Stop ActorPools
        await pool_handle.stop()
        await pool_handle.close()

    tic = time.perf_counter()
    print("Stopping EnsembleLauncher ...")
    el.stop()
    print(f"EnsembleLauncher stopped in {(time.perf_counter() - tic):.1f} s", flush=True)
    total_runtime = time.perf_counter() - start_time

    # Print summary of performance stats
    print("\n\n=========================================")
    print("Performance Summary:")
    print(f"Total number of input prompts: {num_prompts}")
    print(f"Total number of successful requests: {successful_requests}")
    print(f"Total run time = {total_runtime:.4f} s")
    print(f"Initialization time (max) = {init_time:.4f} s")
    print(f"Inference time (max) = {inf_time:.4f} s")
    print(f"Total successful requests per second (requests / inference time) : {rps:.4f}")


if __name__ == "__main__":
    asyncio.run(async_main())
