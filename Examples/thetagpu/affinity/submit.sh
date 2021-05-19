#!/bin/sh
#COBALT -n 1 -t 15 -q training -A Comp_Perf_Workshop

# single-node MPI-only example w/ 16 MPI ranks per node
#mpirun -n 16 -N 16 -hostfile ${COBALT_NODEFILE} ./hello_affinity


# multi-node MPI-only example w/ 16 MPI ranks per node
#mpirun -n 32 -N 16 -hostfile ${COBALT_NODEFILE} ./hello_affinity


# single-node MPI+OpenMP example w/ 1 MPI rank per node, 16 OpenMP threads per rank
#mpirun -n 1 -N 1 -hostfile ${COBALT_NODEFILE} --map-by node:PE=16 -x OMP_NUM_THREADS=16 ./hello_affinity
#mpirun -n 1 -N 1 -hostfile ${COBALT_NODEFILE} --map-by node:PE=16 -x OMP_NUM_THREADS=16 -x OMP_PLACES=cores ./hello_affinity


# single-node MPI+OpenMP example w/ 4 MPI ranks per node, 16 OpenMP threads per rank
#mpirun -n 4 -N 4 -hostfile ${COBALT_NODEFILE} --map-by node:PE=16 -x OMP_NUM_THREADS=16 ./hello_affinity
mpirun -n 4 -N 4 -hostfile ${COBALT_NODEFILE} --map-by node:PE=16 -x OMP_NUM_THREADS=16 -x OMP_PLACES=cores ./hello_affinity
