import os
import numpy as np
from time import sleep
import argparse

from smartredis import Client

from mpi4py import MPI

# MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
local_rank = int(os.getenv("PALS_LOCAL_RANKID"))
local_size = int(os.getenv("PALS_LOCAL_SIZE"))
host_name = MPI.Get_processor_name()
if rank == 0: 
    print(f"[Trainer] Running with {size} MPI ranks and head node {host_name}",flush=True)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--db_nodes", type=int, default=1)
args = parser.parse_args()

# Initialize SmartRedis client
SSDB = os.getenv("SSDB")
if (args.db_nodes==1):
    client = Client(address=SSDB,cluster=False)
else:
    client = Client(address=SSDB,cluster=True)
comm.Barrier()
if (rank == 0):
    print("All Python clients initialized here\n", flush=True)

# Wait for data to be available in DB
if (rank == 0):
    print("Waiting for data to be available in DB ...", flush=True)
while True:
    if (client.key_exists(f"y.{rank}")):
        train_data = client.get_tensor(f'y.{rank}')
        N = train_data.shape[0]
        print(f"Rank {rank} found tensor y.{rank} with shape {N}",flush=True)
        break
    else:
        sleep(1)
comm.Barrier()
if (rank == 0):
    print("Data is available in DB!\n", flush=True)

# Receive training data
workflow_steps = 15
stream_time = 0.0
try:
    for step in range(workflow_steps):
        sleep(5)
        if rank == 0: print(f'[ML] Reading solution data for step {step}',flush=True)
        tic = MPI.Wtime()
        train_data = client.get_tensor(f'y.{rank}')
        toc = MPI.Wtime()
        if step > 0:
            stream_time += toc - tic
        comm.Barrier()
        if rank == 0: print(f'[ML] Done reading solution data for step {step} in {toc - tic} seconds',flush=True)

    # Compute average stream time across all ranks
    stream_time /= workflow_steps-1
    global_avg_stream_time = comm.allreduce(stream_time, op=MPI.SUM)
    global_avg_stream_time /= size

    if (rank % local_size == 0):
        if (rank == 0): print(f"[{rank}]: Telling simulation to quit ... \n")
        arrMLrun = np.int32(np.zeros(1))
        client.put_tensor("check-run",arrMLrun)

    comm.Barrier()
    if rank == 0: print('Trainer is done!')

    # Print stream performance summary
    if rank == 0:
        print("\n=== Communication Performance Summary ===")
        data_size_gb = N * 8 / 1e9
        recv_bw = N * 8 / global_avg_stream_time / 1e9
        print(f"Data size per message: {data_size_gb:.4e} GB")
        print(f"Total iterations: {workflow_steps}")
        print(f"Average receive time: {global_avg_stream_time:.6f} seconds")
        print(f"Average receive bandwidth: {recv_bw:.6f} GB/s")

except Exception as e:
    print(e)

