import asyncio
import concurrent.futures
import multiprocessing as mp
import os
import time
import uuid
from typing import List

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.comm import AsyncZMQTransport, transport_registry
from ensemble_launcher.config import (
    LauncherConfig,
    MPIConfig,
    PolicyConfig,
    SystemConfig,
)
from ensemble_launcher.ensemble import Task
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.inference import PrivateVLLMInference, copy_model
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
    logger.info("main_offline started")

    args_dict = parse_args()
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

    ## Copy model and vllm_cache to /tmp
    tic = time.perf_counter()
    copy_model.sync_to_root(
        model=args_dict["model"],
        cache_dir=args_dict["cache_dir"],
        np=102,
        node_local_cache=local_cache,
        logger=logger,
        cache_modelinfo=pre_build_vllm_cache,
        vllm_cache=vllm_cache,
        node_local_vllm_cache=node_local_vllm_cache,
    )
    logger.info("Done sync to root")
    copy_model.scatter_from_root(
        model=args_dict["model"],
        nnodes=len(nodes),
        node_local_cache=local_cache,
        chunk_size=100 * 1024 * 1024,
        ppn=8,
        logger=logger,
        cpu_binding="--cpu-bind=list:1-12,13-24,25-36,37-48,53-64,65-76,77-88,89-100",
        cache_modelinfo=pre_build_vllm_cache,
        node_local_vllm_cache=node_local_vllm_cache,
    )
    logger.info(f"Copying model took {time.perf_counter() - tic}s")

    # Create EL
    cpus = list(range(104))
    cpus.pop(52)
    cpus.pop(0)
    sys_config = SystemConfig(
        name="aurora", ncpus=102, ngpus=12, cpus=cpus, gpus=list(range(12))
    )
    ckpt_dir = f"{os.getcwd()}/ckpt_{str(uuid.uuid4())}"
    launcher_config = LauncherConfig(
        child_executor_name="async_mpi",
        task_executor_name=["async_processpool", "async_mpi"],
        comm_name="async_zmq",
        children_scheduler_policy="fixed_leafs_children_policy",
        policy_config=PolicyConfig(
            nlevels=1 if len(nodes) <= 256 else 2, leaf_nodes=len(nodes)
        ),
        mpi_config=MPIConfig(flavor="mpich", cpu_bind_method="none"),
        cluster=True,
        worker_logs=False,
        master_logs=True,
        return_stdout=True,
        checkpoint_dir=ckpt_dir,
        report_interval=10.0,
        heartbeat_dead_threshold=120,
        heartbeat_interval=5.0,
    )

    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launcher_config
    )

    t0 = time.time()
    logger.info("starting EnsembleLauncher")
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
