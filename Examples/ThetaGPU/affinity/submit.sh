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
NNODES=`wc -l < $COBALT_NODEFILE`
NRANKS=4
NDEPTH=16
NTHREADS=16

NTOTRANKS=$(( NNODES * NRANKS ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS} THREADS_PER_RANK= ${NTHREADS}"

#mpirun -n ${NTOTRANKS} -N ${NRANKS} -hostfile ${COBALT_NODEFILE} --map-by node:PE=${NDEPTH} -x OMP_NUM_THREADS=${NTHREADS} ./hello_affinity
mpirun -n ${NTOTRANKS} -N ${NRANKS} -hostfile ${COBALT_NODEFILE} --map-by node:PE=${NDEPTH} -x OMP_NUM_THREADS=${NTHREADS} -x OMP_PLACES=cores ./hello_affinity
