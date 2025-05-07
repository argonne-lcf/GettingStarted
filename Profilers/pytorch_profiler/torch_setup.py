#!/usr/bin/env python
#
import socket

from mpi4py import MPI

comm = MPI.COMM_WORLD
import os

import torch
import torch.distributed as dist
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)


def get_device_type():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


def get_local_rank():
    import socket

    master_addr = socket.gethostname()
    addr = master_addr
    addr_all = comm.gather(master_addr, root=0)
    ppn = 1
    if comm.rank == 0:
        num_nodes = len(set(addr_all))
        ppn = comm.size // num_nodes
    ppn = comm.bcast(ppn, root=0)

    if comm.rank == 0:
        print(f"Number of ranks per node: {ppn}", flush=True)
    return comm.rank % ppn, ppn, comm.size // ppn


DEVICE = get_device_type()
LOCAL_RANK, PPN, NUM_NODES = get_local_rank()


def get_device_count():
    global DEVICE
    if DEVICE == "xpu":
        return torch.xpu.device_count()
    elif DEVICE == "cuda":
        return torch.cuda.device_count()
    else:
        return 0


def get_profiler_activities(no_gpu=False):
    activities = [ProfilerActivity.CPU]
    if no_gpu:
        return activities
    gpu = get_device_type()
    if gpu == "xpu":
        activities += [ProfilerActivity.XPU]
    if gpu == "cuda":
        activities += [ProfilerActivity.CUDA]
    return activities


def get_device(gpu=None):
    global LOCAL_RANK
    if gpu == None:
        gpu = get_device_type()
    return torch.device(f"{gpu}:{LOCAL_RANK}")


def init_distributed(backend=None):
    global LOCAL_RANK, PPN, NUM_NODES
    """
    Initialize the default process group.
    """

    if backend == None:
        gpu = get_device_type()
        if gpu == "xpu":
            backend = "ccl"
        elif gpu == "cuda":
            backend = "nccl"
        else:
            backend = "gloo"

    if backend == "ccl":
        import intel_extension_for_pytorch
        import oneccl_bindings_for_pytorch

    rank = comm.rank
    world_size = comm.size
    local_rank = LOCAL_RANK
    os.environ['LOCAL_RANK'] = str(LOCAL_RANK)
    import socket

    master_addr = socket.gethostname()
    print(
        f"I am rank {rank} of {world_size} - {local_rank} on {master_addr}", flush=True
    )
    if NUM_NODES > 1:
        master_addr = comm.bcast(master_addr, root=0)
    else:
        master_addr = "localhost"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = "5676"
    dist.init_process_group(
        backend=backend,
        init_method="env://",  # Read MASTER_ADDR, MASTER_PORT, etc. from environment
        world_size=world_size,
        rank=rank,
    )
    return dist, rank, world_size


if __name__ == "__main__":
    init_distributed()
