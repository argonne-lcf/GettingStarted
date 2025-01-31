#!/bin/bash -l
#PBS -l select=8
#PBS -l place=scatter
#PBS -l walltime=0:15:00
#PBS -q debug-scaling
#PBS -A Catalyst
#PBS -l filesystems=home:flare

#cd ${PBS_O_WORKDIR}

# MPI example w/ MPI ranks and OpenMP threads spread evenly across cores (one process per physical core)
NNODES=`wc -l < $PBS_NODEFILE`

# Per-executable settings
NUM_NODES_PER_EXE=2
NUM_RANKS_PER_NODE=12
NDEPTH=2
NTHREADS=2

NTOTRANKS=$(( NUM_NODES_PER_EXE * NUM_RANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} NUM_NODES_PER_EXE= ${NUM_NODES_PER_EXE} TOTAL_NUM_RANKS= ${NTOTRANKS} NUM_RANKS_PER_NODE= ${NUM_NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

# Launch multiple applications concurrently, each on 2 nodes
# Each application uses 24 cores (12 MPI x 2 OpenMP) & 6 GPU (1 rank per tile) on each node
# The first core on each CPU is skipped (0 & 52)

MPI_ARG="-n ${NTOTRANKS} --ppn ${NUM_RANKS_PER_EXE} --depth=${NDEPTH} --cpu-bind depth "
MPI_ARG+="--env OMP_NUM_THREADS=${NTHREADS} --env OMP_PLACES=cores "

AFFINITY="../../../HelperScripts/Aurora/set_affinity_gpu.sh"

# Increase value of suffix-length if more than 99 jobs
split --lines=${NUM_NODES_PER_EXE} --numeric-suffixes=1 --suffix-length=2 $PBS_NODEFILE local_hostfile.

I=0
for lh in local_hostfile*
do
  echo "Launching mpiexec ${I} w/ ${lh}"
  # These should all be backgrounded with &, but we leave that out for demonstration purposes
  COMMAND="mpiexec ${MPI_ARG} --hostfile ${lh} ${AFFINITY} ./hello_affinity"
  echo ${COMMAND}
  ${COMMAND}
  ((I++))
  sleep 1s
done

wait

rm -f local_hostfile.*
