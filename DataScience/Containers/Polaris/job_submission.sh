#!/bin/sh
#PBS -l select=10:system=polaris
#PBS -q prod
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -A Datascience

cd ${PBS_O_WORKDIR}
CONTAINER=my_image.sif

# SET proxy for internet access
module load singularity
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

# MPI example w/ 16 MPI ranks per node spread evenly across cores
NODES=`wc -l < $PBS_NODEFILE`
PPN=1
PROCS=$((NODES * PPN))
echo "NUM_OF_NODES= ${NODES} TOTAL_NUM_RANKS= ${PROCS} RANKS_PER_NODE= ${PPN}"

MPI_BASE=/soft/datascience/openmpi/openmpi-4.0.5/
export PATH=$MPI_BASE/bin:$PATH
export LD_LIBRARY_PATH=$MPI_BASE/lib:$LD_LIBRARY_PATH
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
echo mpirun=$(which mpirun)

# Openmpi Does not work on Polaris
echo library path
mpirun -hostfile $PBS_NODEFILE -n $PROCS -npernode $PPN singularity exec --nv -B $MPI_BASE $CONTAINER ldd /usr/source/mpi_hello_world

echo C++ MPI
mpirun -hostfile $PBS_NODEFILE -n $PROCS -npernode $PPN singularity exec --nv -B $MPI_BASE $CONTAINER /usr/source/mpi_hello_world

echo Python MPI
mpiexec -np 1 singularity exec --nv -B $MPI_BASE $CONTAINER python3 /usr/source/mpi_hello_world.py
