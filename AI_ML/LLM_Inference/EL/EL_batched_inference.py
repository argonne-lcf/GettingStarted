import asyncio
import concurrent.futures
import multiprocessing as mp
import os
import time
import uuid
import json
import sys

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import aurora_config
from ensemble_launcher.comm import AsyncZMQTransport, transport_registry
from ensemble_launcher.ensemble import Task
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.inference import (
    PrivateVLLMInference, copy_model, default_inference_launcher_config
)
from ensemble_launcher.inference.utils import call_llm
from ensemble_launcher.orchestrator import ClusterClient

from utils import get_num_gpu, parse_args


async def Warmup(args_dict, vllm_cache):
    transport: AsyncZMQTransport = transport_registry.get("zmq")["transport"]()
    server, client = transport.create_child_pipe("server", "secret", "actor", "secret")
    actor = PrivateVLLMInference(
        name="warmpup-actor",
        model=args_dict["model"],
        cache_dir=args_dict["cache_dir"],
        tensor_parallel_size=args_dict["ngpus_per_model"],
        client_conn=client,
        model_info_cache=vllm_cache,
    )
    os.environ["ZE_AFFINITY_MASK"] = ",".join(
        map(str, list(range(args_dict["ngpus_per_model"])))
    )
    mp.set_start_method("spawn")
    p = mp.Process(target=actor)
    p.start()
    try:
        handle = PrivateVLLMInference.create_handle(server)
        await handle.open()
        await handle.send(("generate", ("hello",), ()), "actor:secret")
        result = await handle.recv()
        await handle.stop()
        await handle.close()
    finally:
        p.join(30.0)
        if p.is_alive():
            p.kill()
        os.environ.pop("ZE_AFFINITY_MASK")
    return result


async def async_main():
    start_time = time.time()

    # Parse arguments and get system info
    args_dict = parse_args()
    nodes = get_nodes()
    num_gpus = get_num_gpu()

    # Read prompts from file
    try:
        prompts = []
        for line in open(args_dict["prompt_file"]):
            data = json.loads(line)
            prompts.append(data["prompt"])
        print(f"Read prompts from {args_dict["prompt_file"]}", flush=True)
    except FileNotFoundError:
        print(f"Error: The prompt file {args_dict["prompt_file"]} was not found.", flush=True)
        sys.exit(1)

    # Copy model weights and vllm_cache to /tmp
    tic = time.perf_counter()
    copy_model.distribute_model(
        model=args_dict["model"],
        cache_dir=args_dict["cache_dir"],
        nnodes=len(nodes),
        node_local_cache=args_dict["tmp_dir"],
        sync_np=102,
        scatter_ppn=8,
        #logger=logger,
        cpu_binding="--cpu-bind=list:1-12,13-24,25-36,37-48,53-64,65-76,77-88,89-100",
    )
    print(f"Copied model weights to nodes in {(time.perf_counter() - tic):.1f} s", flush=True)

    # Create EL system and launcher configs
    ckpt_dir = f"{os.getcwd()}/ckpt_{str(uuid.uuid4())}"
    sys_config = aurora_config
    launcher_config = default_inference_launcher_config(len(nodes), ckpt_dir)

    # Start EL
    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launcher_config
    )
    tic = time.time()
    print("Starting EnsembleLauncher ...", flush=True)
    el.start()
    await asyncio.sleep(10.0)
    print(f"EnsembleLauncher ready in {(time.perf_counter() - tic):.1f} s", flush=True)

    # Create tasks
    llm_tasks = []
    n_tasks = num_gpu * len(nodes) // args_dict["ngpus_per_model"]
    prompts = prompts * n_tasks  # NOTE: only needed for weak scaling
    num_prompts = len(prompts)

    # Send prompts as batches to each actor
    chunks = [[] for _ in range(n_tasks)]
    for i, prompt in enumerate(prompts):
        chunks[i % n_tasks].append(prompt)

    for i in range(n_tasks):
        task_id = f"task-{i}"
        llm_tasks.append(
            Task(
                task_id=task_id,
                nnodes=1,
                ppn=args_dict["ngpus_per_model"] * 2,
                ngpus_per_process=1 / 2,
                executable=call_llm,
                args=(args_dict["model"], chunks[i]),
            )
        )

    # Submit tasks with EL client-cluster approach
    tic = time.perf_counter()
    with ClusterClient(checkpoint_dir=ckpt_dir, checkpoint_timeout=300) as client:
        # Submit tasks
        llm_futures = []
        llm_futures = client.submit_batch(tasks=llm_tasks)
        print(f"Submitted {n_tasks} llm tasks", flush=True)

        # Wait for task completion
        done, pending = concurrent.futures.wait(llm_futures, timeout=1200)
        init_inf_time = time.perf_counter() - tic

        # Check successful tasks
        if len(done) == n_tasks:
            print(f"Done with all tasks in ({(time.perf_counter() - tic):.1f})", flush=True)
        else:
            print(
                f"{len(done)}/{n_tasks} tasks done ({time.perf_counter() - tic})",
                flush=True
            )
        successful_requests = len(done)
        for i, f in enumerate(done):
            if f.exception() is not None:
                print(f"Task {i} failed with exception: {f.exception()}", flush=True)
                successful_requests -= 1

    tic = time.perf_counter()
    print("Stopping EnsembleLauncher ...")
    el.stop()
    print(f"EnsembleLauncher stopped in {(time.perf_counter() - tic):.1f}", flush=True)
    total_runtime = time.perf_counter() - start_time

    # Print summary of performance stats
    print("\n\n=========================================")
    print("Performance Summary:")
    print(f"Total number of input prompts: {num_prompts}")
    print(f"Total number of successful requests: {sum(successful_requests)}")
    print(f"Total run time = {total_runtime:.4f} s")
    print(f"Initialization + Inference time (max) = {init_inf_time} s")
    print(f"Total successful requests per second (requests / inference time) : {sum(rpss):.4f}")


if __name__ == "__main__":
    asyncio.run(async_main())
