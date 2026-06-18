import asyncio
import json
import os
import secrets
import sys
import time
import uuid

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.comm import transport_registry
from ensemble_launcher.config import aurora_config
from ensemble_launcher.ensemble import ActorPool
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.inference import (
    PrivateVLLMInference,
    default_inference_launcher_config,
)
from ensemble_launcher.orchestrator import ClusterClient
from utils import get_logger, get_num_gpu, parse_args

logger = get_logger("main_offline", log_dir=f"{os.getcwd()}/script_logs")


async def async_main():
    start_time = time.time()

    # Parse arguments and get system info
    args_dict = parse_args()
    nodes = get_nodes()
    num_gpus = get_num_gpu()
    num_inf_engines = (num_gpus // args_dict["num_gpus_per_model"]) * len(nodes)

    # Read prompts from file
    try:
        prompts = []
        for line in open(args_dict["prompt_file"]):
            data = json.loads(line)
            prompts.append(data["prompt"])
        print(f"Read prompts from {args_dict['prompt_file']}", flush=True)
    except FileNotFoundError:
        print(
            f"Error: The prompt file {args_dict['prompt_file']} was not found.",
            flush=True,
        )
        sys.exit(1)
    prompts = prompts * num_inf_engines  # NOTE: only needed for weak scaling
    num_prompts = len(prompts)

    # # Copy model weights and vllm_cache to /tmp
    # tic = time.perf_counter()
    # copy_model.distribute_model(
    #     model=args_dict["model"],
    #     cache_dir=args_dict["cache_dir"],
    #     nnodes=len(nodes),
    #     node_local_cache=args_dict["tmp_dir"],
    #     sync_np=102,
    #     scatter_ppn=8,
    #     logger=logger,
    #     cpu_binding="--cpu-bind=list:1-12,13-24,25-36,37-48,53-64,65-76,77-88,89-100",
    # )
    # print(f"Copied model weights to nodes in {(time.perf_counter() - tic):.1f} s", flush=True)

    # Create EL system and launcher configs
    ckpt_dir = f"{os.getcwd()}/ckpt_{str(uuid.uuid4())}"
    sys_config = aurora_config
    launcher_config = default_inference_launcher_config(len(nodes), ckpt_dir)

    # Start EL
    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launcher_config
    )
    tic = time.perf_counter()
    print("Starting EnsembleLauncher ...", flush=True)
    el.start()
    await asyncio.sleep(10.0)
    print(f"EnsembleLauncher ready in {(time.perf_counter() - tic):.1f} s", flush=True)

    # Chunk the prompts
    chunks = [[] for _ in range(num_inf_engines)]
    for i, prompt in enumerate(prompts):
        chunks[i % num_inf_engines].append(prompt)

    # Step 3: Create ActorPools and submit them to the cluster
    n_actors = num_inf_engines
    ACTORS_PER_POOL = args_dict["actors_per_pool"]

    actor_chunks = [
        list(range(i, min(i + ACTORS_PER_POOL, n_actors)))
        for i in range(0, n_actors, ACTORS_PER_POOL)
    ]

    transport = transport_registry.get("zmq")["transport"]()
    server_id = "global_server"
    server_secret = secrets.token_hex(16)
    pool_ids = []
    actor_kwargs = {
        "model": args_dict["model"],
        "cache_dir": args_dict["cache_dir"],
        "tensor_parallel_size": args_dict["ngpus_per_model"],
        "send_timeout": args_dict["send_timeout"],
        "send_retries": args_dict["send_retries"],
        "llm_kwargs": {"enforce_eager": True},
    }
    task_kwargs = {
        "nnodes": 1,
        "ppn": args_dict["ngpus_per_model"] * 2,
        "ngpus_per_process": 1 / 2,
    }

    with ClusterClient(
        checkpoint_dir=ckpt_dir, checkpoint_timeout=300
    ) as cluster_client:
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
                actor_class=PrivateVLLMInference,
                n_actors=len(chunk),
                actor_kwargs=actor_kwargs,
                task_kwargs=task_kwargs,
                checkpoint_dir=ckpt_dir,
                req_res=True,
                child_send_timeout=args_dict["send_timeout"],
                child_send_retries=args_dict["send_retries"],
                send_timeout=args_dict["send_timeout"],
                send_retries=args_dict["send_retries"],
                child_ready_timeout=600,
            )
            pool_task = actor_pool.create_task(task_id=pool_name, nnodes=1, ppn=1)
            cluster_client.submit(pool_task)
            pool_ids.append(f"{pool_name}:{server_secret}")
        pool_handle = ActorPool.create_handle(
            server,
            send_timeout=args_dict["send_timeout"],
            send_retries=args_dict["send_retries"],
        )
        await pool_handle.open()
        print(f"Waiting for {len(pool_ids)} pools to be ready...")
        init_inf_start = time.perf_counter()
        try:
            await pool_handle.wait_for_ready(expected=len(pool_ids), timeout=660)
        except asyncio.TimeoutError:
            ready_pools = pool_handle.ready_actors
            print(
                f"Not all actor pools are ready. Only {len(ready_pools)}/{len(actor_chunks)} are ready"
            )
        init_inf_time = time.perf_counter() - init_inf_start

        # Step 4: Inference — invoke_all on every pool concurrently, measure throughput
        async def _run_pool(pool_id, msg):
            _, results = await pool_handle.invoke_all(msg, actor_id=pool_id)
            print(f"Pool {pool_id.split(':')[0]} completed")
            return results

        # Step 5:
        print(
            f"{len(ready_pools)}/{len(actor_chunks)} pools ready. Starting inference..."
        )
        t_inference_start = time.perf_counter()
        inference_msg = ("generate", ("hi, introduce yourself",), None)
        tasks = [
            asyncio.create_task(_run_pool(pid, inference_msg)) for pid in ready_pools
        ]

        done, pending = await asyncio.wait(tasks, timeout=100)
        inf_time = time.perf_counter() - t_inference_start
        successful_requests = (
            len(done) * ACTORS_PER_POOL * (num_prompts // num_inf_engines)
        )
        if len(pending) != 0:
            print(
                f"Only {len(done) * ACTORS_PER_POOL}/{len(pool_ids) * ACTORS_PER_POOL} finished in 100s"
            )
            for t in pending:
                t.cancel()

        # Step 6: Stop pools and EnsembleLauncher
        await pool_handle.stop()
        await pool_handle.close()

    print("Stopping EnsembleLauncher ...")
    el.stop()
    total_runtime = time.perf_counter() - start_time

    # Print summary of performance stats
    print("\n\n=========================================")
    print("Performance Summary:")
    print(f"Total number of input prompts: {num_prompts}")
    print(f"Total number of successful requests: {successful_requests}")
    print(f"Total run time = {total_runtime:.4f} s")
    print(f"Initialization + Inference time (max) = {init_inf_time} s")
    print(
        f"Total successful requests per second (requests / inference time) : {successful_requests / inf_time:.4f}"
    )


if __name__ == "__main__":
    asyncio.run(async_main())
