#!/bin/bash -l
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=0:15:00
#PBS -q debug
#PBS -A Catalyst
#PBS -l filesystems=home:flare

cd ${PBS_O_WORKDIR}

# MPI example w/ MPI ranks and OpenMP threads spread evenly across cores (one process per physical core)
NNODES=`wc -l < $PBS_NODEFILE`

# Per-executable settings
NRANKS_PER_EXE=2
NTHREADS=2
NGPUS=1

echo "NUM_OF_NODES= ${NNODES} NRANKS_PER_EXE= ${NRANKS_PER_EXE} THREADS_PER_RANK= ${NTHREADS}"

# Launch 6 applications concurrently on single node
# Each application uses 4 cores (2 MPI x 2 OpenMP) & 1 GPU (1 rank per tile)
# The first core on each CPU is skipped (0 & 52)
CPU0="1-2:3-4"
CPU1="5-6:7-8"
CPU2="9-10:11-12"
CPU3="53-54:55-56"
CPU4="57-58:59-60"
CPU5="61-62:63-64"

MPI_ARG="-n ${NRANKS_PER_EXE} --ppn ${NRANKS_PER_EXE} "
MPI_ARG+="--env OMP_NUM_THREADS=${NTHREADS} --env OMP_PLACES=cores "
MPI_ARG+="--cpu-bind=list:${SOCKET1}:${SOCKET2}"

AFFINITY="../../../HelperScripts/Aurora/set_affinity_gpu_offset.sh"

# These should all be backgrounded with &, but we leave that out for demonstration purposes

COMMAND="mpiexec ${MPI_ARG} --cpu-bind=list:${CPU0} ${AFFINITY} ${NGPUS} 0 ./hello_affinity"
echo ${COMMAND}
${COMMAND}

COMMAND="mpiexec ${MPI_ARG} --cpu-bind=list:${CPU1} ${AFFINITY} ${NGPUS} 1 ./hello_affinity"
echo ${COMMAND}
${COMMAND}

COMMAND="mpiexec ${MPI_ARG} --cpu-bind=list:${CPU2} ${AFFINITY} ${NGPUS} 2 ./hello_affinity"
echo ${COMMAND}
${COMMAND}



COMMAND="mpiexec ${MPI_ARG} --cpu-bind=list:${CPU3} ${AFFINITY} ${NGPUS} 3 ./hello_affinity"
echo ${COMMAND}
${COMMAND}

COMMAND="mpiexec ${MPI_ARG} --cpu-bind=list:${CPU4} ${AFFINITY} ${NGPUS} 4 ./hello_affinity"
echo ${COMMAND}
${COMMAND}

COMMAND="mpiexec ${MPI_ARG} --cpu-bind=list:${CPU5} ${AFFINITY} ${NGPUS} 5 ./hello_affinity"
echo ${COMMAND}
${COMMAND}


wait
