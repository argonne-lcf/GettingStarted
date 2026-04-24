import os
import logging
import numpy as np
from time import sleep
import argparse

import dragon
from dragon.data.ddict.ddict import DDict

from mpi4py import MPI

# MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
local_rank = int(os.getenv("PALS_LOCAL_RANKID"))
local_size = int(os.getenv("PALS_LOCAL_SIZE"))
host_name = MPI.Get_processor_name()

# Logging: all ranks append to a single shared file with the rank in each
# record. Rank 0 truncates the file first so reruns start clean.
LOG_FILE = "trainer.out"
if rank == 0:
    open(LOG_FILE, "w").close()
comm.Barrier()

log = logging.getLogger("trainer")
log.setLevel(logging.INFO)
log.propagate = False
_handler = logging.FileHandler(LOG_FILE, mode="a")
_handler.setFormatter(
    logging.Formatter(
        fmt=f"%(asctime)s [rank {rank:>3d}/{size}] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
log.addHandler(_handler)

if rank == 0:
    log.info("[Trainer] Running with %d MPI ranks and head node %s", size, host_name)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ddict_ser", type=str, required=True, help="Serialized DDict")
args = parser.parse_args()

# Attach to the Distributed Dictionary
log.info("Attaching to DDict ...")
dd = DDict.attach(args.ddict_ser, timeout=3600)
comm.Barrier()
if rank == 0:
    log.info("All Dragon clients attached to DDict")

# Wait for data to be available in DDict
if rank == 0:
    log.info("Waiting for data to be available in DDict ...")
while True:
    if f"y.{rank}" in dd.keys():
        train_data = dd[f"y.{rank}"]
        N = train_data.shape[0]
        log.info("Found tensor y.%d with shape %s", rank, train_data.shape)
        break
    else:
        sleep(1)
comm.Barrier()
if rank == 0:
    log.info("Data is available in DDict!")

# Receive training data
workflow_steps = 15
stream_time = 0.0
try:
    for step in range(workflow_steps):
        sleep(5)
        if rank == 0:
            log.info("[ML] Reading solution data for step %d", step)
        tic = MPI.Wtime()
        train_data = dd[f"y.{rank}"]
        toc = MPI.Wtime()
        if step > 0:
            stream_time += toc - tic
        comm.Barrier()
        if rank == 0:
            log.info(
                "[ML] Done reading solution data for step %d in %.6f seconds",
                step,
                toc - tic,
            )

    # Compute average stream time across all ranks
    stream_time /= workflow_steps - 1
    global_avg_stream_time = comm.allreduce(stream_time, op=MPI.SUM)
    global_avg_stream_time /= size

    if rank % local_size == 0:
        if rank == 0:
            log.info("[%d]: Telling simulation to quit ...", rank)
        arrMLrun = np.int32(np.zeros(1))
        dd["check-run"] = arrMLrun

    comm.Barrier()
    if rank == 0:
        log.info("Trainer is done!")

    # Print stream performance summary
    if rank == 0:
        data_size_gb = N * 8 / 1e9
        recv_bw = N * 8 / global_avg_stream_time / 1e9
        log.info("=== Communication Performance Summary ===")
        log.info("Data size per message: %.4e GB", data_size_gb)
        log.info("Total iterations: %d", workflow_steps)
        log.info("Average receive time: %.6f seconds", global_avg_stream_time)
        log.info("Average receive bandwidth: %.6f GB/s", recv_bw)

except Exception:
    log.exception("Trainer failed with an unhandled exception")
