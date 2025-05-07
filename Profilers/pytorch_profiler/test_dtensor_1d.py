import argparse
import os
import time

import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
)

from torch_setup import (
    get_device,
    get_device_type,
    get_profiler_activities,
    init_distributed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="DTensor + torch.profiler example.")
    # Tensor parallel (TP) size (how many ranks in our device mesh)
    parser.add_argument(
        "--tp-size",
        type=int,
        default=4,
        help="Number of ranks/devices to use in the device mesh.",
    )
    parser.add_argument("--dim", type=int, default=256, help="dimension of the matrix")
    parser.add_argument("--trace-dir", type=str, default="dtensor_trace")
    args = parser.parse_args()
    return args


def main():
    # 1. Initialize the distributed environment with NCCL for GPU usage
    args = parse_args()
    dist, rank, world_size = init_distributed()

    # 2. Set local GPU device
    gpu = get_device_type()
    device = get_device()
    # 3. Build a 1D DeviceMesh across all GPUs to avoid n-dimensional redistribution bugs
    mesh = DeviceMesh(gpu, torch.arange(world_size))
    if rank == 0:
        print(f"Device mesh: {mesh}")
    # 4. Create local tensors on each rank (just random data plus an offset for demonstration)
    local_tensor_a = rank * torch.ones(args.dim, args.dim).to(device)
    local_tensor_b = rank * torch.ones(args.dim, args.dim).to(device)
    local_tensor_ap = 2 * torch.ones(args.dim, args.dim).to(device)
    local_tensor_bp = 2 * torch.ones(args.dim, args.dim).to(device)

    # Using Replicate placement to avoid DTensor redistribution errors on size-1 mesh dimension.
    dt_a = DTensor.from_local(local_tensor_a, device_mesh=mesh, placements=[Shard(1)])
    dt_b = DTensor.from_local(local_tensor_b, device_mesh=mesh, placements=[Shard(0)])
    if rank == 0:
        print(
            f"A: {dt_a.shape} created from A_local: {local_tensor_a.shape} with Replicate"
        )
        print(
            f"B: {dt_b.shape} created from B_local: {local_tensor_b.shape} with Replicate"
        )
    dt_ap = DTensor.from_local(local_tensor_ap, device_mesh=mesh, placements=[Shard(1)])
    dt_bp = DTensor.from_local(local_tensor_bp, device_mesh=mesh, placements=[Shard(0)])

    # 7. Run a few steps of matmul in a loop so we can capture multiple profiler events
    def run(step):
        # Use record_function to label operations
        start = time.time()
        with record_function("dtensor_matmul_C"):
            dt_c = dt_a.matmul(dt_b)  # distributed matmul
            dt_c = dt_c.redistribute(
                device_mesh=mesh, placements=[Shard(1)], async_op=True
            )
        with record_function("dtensor_matmul_CP"):
            dt_cp = dt_ap.matmul(dt_bp)  # distributed matmul
            dt_cp = dt_cp.redistribute(
                device_mesh=mesh, placements=[Shard(1)], async_op=True
            )
            # (Optional) we could do more computations here
        end = time.time()
        if rank == 0:
            print(f"step {step}: {end-start:.6f} sec")
            # Step the profiler each iteration
        dist.barrier()

    run(0)
    with profile(
        activities=get_profiler_activities(), 
        record_shapes=True, 
        profile_memory=True,
        with_flops = True, 
    ) as prof:
        for step in range(1, 6):
            run(step)
    os.makedirs(args.trace_dir, exist_ok=True)
    prof.export_chrome_trace(f"{args.trace_dir}/torch-trace-{rank}-of-{world_size}.json")

    # 8. (Optional) convert the last result to local to see what is on this rank
    # local_c = dt_c.to_local()

    # print(f"[Rank {rank}] local A:\n{local_tensor_a}")
    # print(f"[Rank {rank}] local B:\n{local_tensor_b}")
    # print(f"[Rank {rank}] local C = A@B:\n{local_c}")
    if rank == 0:
        print("-" * 70)

if __name__ == "__main__":
    main()
