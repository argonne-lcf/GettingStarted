"""Distributed vLLM inference benchmark using ActorPool.

Measures fan-out throughput across a multi-node cluster. Each ActorPool
manages a group of PrivateVLLMInference actors; multiple pools run in
parallel to scale beyond what a single handle can coordinate.

Flow:
  1. Distribute the model to all nodes (dsync + MPI scatter).
  2. Start EnsembleLauncher in cluster mode.
  3. Create N ActorPools, each managing a chunk of vLLM actors.
     Pools are submitted as tasks and run remotely on worker nodes.
  4. Warmup: invoke a short generate on every actor via the pools.
  5. Inference: invoke_all on every pool concurrently, measure
     wall-clock time and throughput (actors/sec).
  6. Stop pools and EnsembleLauncher.

Requires: mpirun, dsync, vLLM, ensemble_launcher[mcp].
Target system: Aurora (102 CPUs, 12 GPUs per node).
"""

import asyncio
import os
import secrets
import time
import uuid
from typing import List

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.comm import transport_registry
from ensemble_launcher.config import aurora_config
from ensemble_launcher.ensemble import ActorPool
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.inference import (
    PrivateVLLMInference,
    copy_model,
    default_inference_launcher_config,
)
from ensemble_launcher.orchestrator import ClusterClient
from utils import get_logger, parse_args

logger = get_logger("main_offline", log_dir=f"{os.getcwd()}/script_logs")


def create_prompt(nprompts) -> List[str]:
    prompt = "Hi, can you introduce yourself?"
    return [prompt for i in range(nprompts)]


async def async_main():
    t_start = time.time()
    logger.info("main_offline started")

    args_dict = parse_args()
    nodes = get_nodes()

    local_cache = os.path.join("/tmp", "model_cache")
    pre_build_vllm_cache = args_dict["pre_build_vllm_cache"] == 1
    vllm_cache = os.path.join(os.getcwd(), "vllm_cache")
    node_local_vllm_cache = "/tmp/vllm_cache"

    # Step 1: Distribute model to all nodes (dsync to root + MPI scatter)
    tic = time.perf_counter()
    copy_model.distribute_model(
        model=args_dict["model"],
        cache_dir=args_dict["cache_dir"],
        nnodes=len(nodes),
        node_local_cache=local_cache,
        sync_np=102,
        scatter_ppn=8,
        logger=logger,
        cpu_binding="--cpu-bind=list:1-12,13-24,25-36,37-48,53-64,65-76,77-88,89-100",
        cache_modelinfo=pre_build_vllm_cache,
        vllm_cache=vllm_cache,
        node_local_vllm_cache=node_local_vllm_cache,
    )
    logger.info(f"Copying model took {time.perf_counter() - tic}s")

    # Step 2: Start EnsembleLauncher in cluster mode
    ckpt_dir = f"{os.getcwd()}/ckpt_{str(uuid.uuid4())}"
    sys_config = aurora_config
    launcher_config = default_inference_launcher_config(len(nodes), ckpt_dir)

    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launcher_config
    )

    t0 = time.time()
    logger.info("starting EnsembleLauncher")
    el.start()
    await asyncio.sleep(10.0)
    logger.info("EnsembleLauncher ready (%.1fs)", time.time() - t0)

    # Step 3: Create ActorPools and submit them to the cluster
    n_actors = 12 * len(nodes) // args_dict["ngpus_per_model"]
    ACTORS_PER_PROCESS = args_dict["actors_per_process"]

    actor_chunks = [
        list(range(i, min(i + ACTORS_PER_PROCESS, n_actors)))
        for i in range(0, n_actors, ACTORS_PER_PROCESS)
    ]

    transport = transport_registry.get("zmq")["transport"]()
    server_id = "global_server"
    server_secret = secrets.token_hex(16)
    pool_ids = []
    actor_kwargs = {
        "model": args_dict["model"],
        "cache_dir": local_cache,
        "tensor_parallel_size": args_dict["ngpus_per_model"],
        "use_cached_modelinfo": pre_build_vllm_cache,
        "model_info_cache": node_local_vllm_cache if pre_build_vllm_cache else None,
        "send_timeout": args_dict["send_timeout"],
        "send_retries": args_dict["send_retries"],
        "llm_kwargs": {"enforce_eager": True},
    }
    task_kwargs = {
        "nnodes": 1,
        "ppn": args_dict["ngpus_per_model"] * 2,
        "ngpus_per_process": 1 / 2,
    }
    try:
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
            logger.info(f"Waiting for {len(pool_ids)} pools to be ready...")
            await pool_handle.wait_for_ready(expected=len(pool_ids), timeout=660)

            # Step 4: Warmup — short generate on every actor via each pool
            async def _run_pool(pool_id, msg):
                _, results = await pool_handle.invoke_all(msg, actor_id=pool_id)
                logger.info(f"Pool {pool_id.split(':')[0]} completed")
                return results

            logger.info(f"Waiting for {len(pool_ids)} pools to warm up...")
            inference_msg = ("generate", ("hi",), None)
            tasks = [
                asyncio.create_task(_run_pool(pid, inference_msg)) for pid in pool_ids
            ]
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=1200
                )
            except asyncio.TimeoutError:
                logger.error("Warmup timed out")
                raise TimeoutError

            # Step 5: Inference — invoke_all on every pool concurrently, measure throughput
            logger.info("All pools ready. Starting inference...")
            t_inference_start = time.perf_counter()

            inference_msg = ("generate", ("hi, introduce yourself",), None)
            tasks = [
                asyncio.create_task(_run_pool(pid, inference_msg)) for pid in pool_ids
            ]
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=100
                )
                for rid, result in enumerate(results):
                    logger.debug(f"Result {rid}: {result}")
            except asyncio.TimeoutError:
                logger.error("Inference timed out")

            # Step 6: Stop pools and EnsembleLauncher
            await pool_handle.stop()
            await pool_handle.close()

        inference_duration = time.perf_counter() - t_inference_start
        logger.info("== INFERENCE COMPLETED ==")
        logger.info(f"Total Actors: {n_actors}")
        logger.info(
            f"Time to generate and drain queue: {inference_duration:.4f} seconds."
        )
        logger.info(f"Throughput: {n_actors / inference_duration:.2f} results/sec.")
    finally:
        t0 = time.time()
        logger.info("stopping EnsembleLauncher")
        el.stop()
        logger.info("EnsembleLauncher stopped (%.1fs)", time.time() - t0)

        logger.info("main_offline done (total %.1fs)", time.time() - t_start)


if __name__ == "__main__":
    asyncio.run(async_main())
