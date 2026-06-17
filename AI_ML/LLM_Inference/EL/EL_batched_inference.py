import asyncio
import concurrent.futures
import multiprocessing as mp
import os
import time
import uuid
from typing import List

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

from utils import get_logger, parse_args
logger = get_logger("main_offline", log_dir=f"{os.getcwd()}/script_logs")


def create_prompt(nprompts) -> List[str]:
    prompt = "Hi, can you introduce yourself?"
    return [prompt for i in range(nprompts)]


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
    t_start = time.time()

    # Parse arguments
    args_dict = parse_args()

    # Get nodes and set up vllm cache
    nodes = get_nodes()

    local_cache = os.path.join("/tmp", "model_cache")

    pre_build_vllm_cache = args_dict["pre_build_vllm_cache"] == 1
    vllm_cache = os.path.join(os.getcwd(), "vllm_cache")
    node_local_vllm_cache = "/tmp/vllm_cache"
    if pre_build_vllm_cache:
        # Build vllm cache
        try:
            result = await asyncio.wait_for(Warmup(args_dict, vllm_cache), timeout=600)
            logger.info(f"Warmup returned result: {result}")
        except Exception as e:
            logger.warning(f"Warmup failed with error {e}")

    # Copy model weights and vllm_cache to /tmp
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

    # Create EL system and launcher configs
    ckpt_dir = f"{os.getcwd()}/ckpt_{str(uuid.uuid4())}"
    sys_config = aurora_config
    launcher_config = default_inference_launcher_config(len(nodes), ckpt_dir)

    # Start EL
    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launcher_config
    )
    t0 = time.time()
    logger.info("Starting EnsembleLauncher ...")
    el.start()
    await asyncio.sleep(10.0)
    logger.info("EnsembleLauncher ready (%.1fs)", time.time() - t0)

    llm_tasks = []

    n_tasks = 12 * len(nodes) // args_dict["ngpus_per_model"]

    # Send prompts as batches to each actor
    prompts = create_prompt(args_dict["num_prompts"]) * n_tasks
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

    # Submit actors and run inference
    with ClusterClient(checkpoint_dir=ckpt_dir, checkpoint_timeout=300) as client:
        t0 = time.time()
        llm_futures = []
        llm_futures = client.submit_batch(tasks=llm_tasks)

        logger.info("submitted %d llm tasks", n_tasks)

        start = time.perf_counter()
        done, pending = concurrent.futures.wait(llm_futures, timeout=1200)

        if len(done) == n_tasks:
            logger.info(f"all prompts done ({time.perf_counter() - start: .1f})")
        else:
            logger.info(
                f" {len(done)}/{n_tasks} prompts done ({time.perf_counter() - start})"
            )

        for i, f in enumerate(done):
            if f.exception() is not None:
                logger.warning(f"Task {i} failed with exception: {f.exception()}")
            else:
                logger.debug(f"Result {i}: {f.result()}")

    t0 = time.time()
    logger.info("stopping EnsembleLauncher")
    el.stop()
    logger.info("EnsembleLauncher stopped (%.1fs)", time.time() - t0)

    logger.info("main_offline done (total %.1fs)", time.time() - t_start)


if __name__ == "__main__":
    asyncio.run(async_main())
