import os
import numpy as np
from adios2 import Stream, Adios
from time import sleep
import argparse

import mpi4py.rc
mpi4py.rc.threads = True
mpi4py.rc.thread_level = 'multiple'
from mpi4py import MPI

# MPI
thread_level = MPI.Init_thread(MPI.THREAD_MULTIPLE)
global_comm = MPI.COMM_WORLD
global_rank = global_comm.Get_rank()
global_size = global_comm.Get_size()

color = 1234
comm = global_comm.Split(color, global_rank)
size = comm.Get_size()
rank = comm.Get_rank()
local_rank = int(os.getenv("PALS_LOCAL_RANKID"))
local_size = int(os.getenv("PALS_LOCAL_SIZE"))
host_name = MPI.Get_processor_name()
if (rank == 0):
    print(f"[Trainer] Running with on host {host_name}",flush=True)
    print(f"[Trainer] Running with {size} MPI ranks",flush=True)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_plane", type=str, choices=["WAN", "MPI", "UCX", "RDMA", "fabric"], default="WAN")
parser.add_argument("--io_mode", type=str, choices=["daos", "posix"], default="posix")
args = parser.parse_args()

# ADIOS MPI Communicator
adios = Adios(comm)

# ADIOS IO
sstIO = adios.declare_io("solutionStream")
sstIO.set_engine("SST")
parameters = {
    'DataTransport': args.data_plane, # options: MPI, WAN,  UCX, RDMA
    #'DataInterface': 'cxi0',
    'OpenTimeoutSecs': '600', # number of seconds SST is to wait for a peer connection on Open()
}
sstIO.set_parameters(parameters)

# Read the graph data 
path = '/tmp/datascience/balin/graph.bp' if args.io_mode == 'daos' else './graph.bp'
while True:
    if os.path.exists(path):
        sleep(5)
        break
    else:
        sleep(5)

tic = MPI.Wtime()
with Stream(path, 'r', comm) as stream:
    stream.begin_step()
    
    arr = stream.inquire_variable('N')
    N = stream.read('N', [rank], [1])
    N_list = comm.allgather(N)

    #arr = stream.inquire_variable('num_edges')
    #num_edges = stream.read('num_edges', [rank], [1])
    #num_edges_list = comm.allgather(num_edges)

    arr = stream.inquire_variable('pos_node')
    count = N
    start = sum(N_list[:rank])
    pos = stream.read('pos_node', [start], [count])

    #arr = stream.inquire_variable('edge_index')
    #count = num_edges * 2
    #start = sum(num_edges_list[:rank]) * 2
    #edge_index = stream.read('edge_index', [start], [count]).reshape((-1,2),order='F').T
                
    stream.end_step()

comm.Barrier()
toc = MPI.Wtime()
if rank == 0: print(f'[ML] Done reading graph data in {toc - tic} seconds', flush=True)

# Receive training data
workflow_steps = 5
stream_time = 0.0
try:
    if rank == 0: print('[ML] Opening stream ... ',flush=True)
    stream = Stream(sstIO, "solutionStream", "r", comm)
    for step in range(workflow_steps):
        sleep(5)
        if rank == 0: print(f'[ML] Reading solution data for step {step}',flush=True)
        stream.begin_step()
        var = stream.inquire_variable("U")
        count = N
        start = sum(N_list[:rank])
        # stream.read() gets data now, Mode.Sync is default 
        # see 
        #   - https://github.com/ornladios/ADIOS2/blob/67f771b7a2f88ce59b6808cc4356159d86255f1d/python/adios2/stream.py#L331
        #   - https://github.com/ornladios/ADIOS2/blob/67f771b7a2f88ce59b6808cc4356159d86255f1d/python/adios2/engine.py#L123)
        tic = MPI.Wtime()
        train_data = stream.read("U", [start], [count])
        toc = MPI.Wtime()
        if step > 0:
            stream_time += toc - tic
        stream.end_step()
        comm.Barrier()
        if rank == 0: print(f'[ML] Done reading solution data for step {step} in {toc - tic} seconds',flush=True)
    stream.close()

    # Compute average stream time across all ranks
    stream_time /= workflow_steps-1
    global_avg_stream_time = comm.allreduce(stream_time, op=MPI.SUM)
    global_avg_stream_time /= size

    MLrun = 0
    with Stream('check-run.bp', 'w', comm) as stream:
        if rank == 0:
            stream.write("check-run", np.int32([MLrun]))

    comm.Barrier()
    if rank == 0: print('Trainer is done!')

    # Print stream performance summary
    if rank == 0:
        print("\n=== Communication Performance Summary ===")
        data_size_gb = N * 8 / 1e9
        recv_bw = N * 8 / global_avg_stream_time / 1e9
        print(f"Data size per message: {data_size_gb.item():.4e} GB")
        print(f"Total iterations: {workflow_steps}")
        print(f"Average receive time: {global_avg_stream_time:.6f} seconds")
        print(f"Average receive bandwidth: {recv_bw.item():.6f} GB/s")

except Exception as e:
    print(e)

