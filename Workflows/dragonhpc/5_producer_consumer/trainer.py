import os
import logging
import numpy as np
from time import sleep
import argparse

import dragon
from dragon.data.ddict.ddict import DDict

from mpi4py import MPI

from custom_pickler import NumPy1DPickler, StringKeyPickler

NFIELDS = 3

# MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
local_rank = int(os.getenv("PALS_LOCAL_RANKID"))
local_size = int(os.getenv("PALS_LOCAL_SIZE"))
host_name = MPI.Get_processor_name()

# Logging
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
parser.add_argument("--launcher_mode", type=str, required=True, help="Launcher mode (e.g. colocated, clustered, mixed)")
parser.add_argument("--ddict_ser", type=str, required=True, help="Serialized DDict")
args = parser.parse_args()

# Attach to the Distributed Dictionary and switch to the C++-compatible
# pickler so keys and values use the same serialization format as sim.cpp
if rank == 0:
    log.info("Attaching to DDict ...")
dd = DDict.attach(args.ddict_ser, timeout=3600)
dd = dd.pickler(key_pickler=StringKeyPickler(), value_pickler=NumPy1DPickler(np.float64))
comm.Barrier()
if rank == 0:
    log.info("All Dragon clients attached to DDict")

# For colocated deployments, use the local manager to access local data only
if args.launcher_mode == "colocated":
    dd = dd.manager(dd.local_manager)

# Wait for data to be available in DDict
if rank == 0:
    log.info("Waiting for data to be available in DDict ...")
while True:
    if f"y.{rank}" in dd.keys():
        train_data = dd[f"y.{rank}"].reshape(NFIELDS,-1)
        N = train_data.shape[1]
        log.info("Found tensor y.%d with shape %s x %s", rank, NFIELDS, N)
        break
    else:
        sleep(1)

# Receive training data
workflow_steps = 15
stream_time = 0.0
rank_offset = rank * N * NFIELDS
try:
    for step in range(workflow_steps):
        sleep(5)
        if rank == 0:
            log.info("[ML] Reading solution data for step %d", step)
        tic = MPI.Wtime()
        train_data = dd[f"y.{rank}"].reshape(NFIELDS,-1)
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

        # Check correctness of the received data. 
        expected = (
            rank_offset
            + np.vstack((
                0 * np.ones(N, dtype=np.float64),
                1 * np.ones(N, dtype=np.float64),
                2 * np.ones(N, dtype=np.float64))
            )
        )
        if not np.array_equal(train_data, expected):
            log.error("[ML] Data mismatch for step %d and rank %d", step, rank)
            comm.Abort(1)

    # Compute average stream time across all ranks
    stream_time /= workflow_steps - 1
    global_avg_stream_time = comm.allreduce(stream_time, op=MPI.SUM)
    global_avg_stream_time /= size

    if rank % local_size == 0:
        if rank == 0:
            log.info("[%d]: Telling simulation to quit ...", rank)
        # C++ side expects a 1 SerializableDoubleVector in float64 
        arrMLrun = np.zeros((1,), dtype=np.float64)
        dd["check-run"] = arrMLrun

    comm.Barrier()
    if rank == 0:
        log.info("Trainer is done!")

    # Print stream performance summary
    if rank == 0:
        data_size_gb = N * NFIELDS * 8 / 1e9
        recv_bw = data_size_gb / global_avg_stream_time
        log.info("=== Communication Performance Summary ===")
        log.info("Array shape per message: %s x %s", NFIELDS, N)
        log.info("Data size per message: %.4e GB", data_size_gb)
        log.info("Total iterations: %d", workflow_steps)
        log.info("Average receive time: %.6f seconds", global_avg_stream_time)
        log.info("Average receive bandwidth: %.6f GB/s", recv_bw)

except Exception:
    log.exception("Trainer failed with an unhandled exception")
