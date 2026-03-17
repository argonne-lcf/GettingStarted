#!/bin/bash -l
#PBS -N mpi4py_example
#PBS -l select=2
#PBS -l walltime=00:10:00
#PBS -l filesystems=home:eagle
#PBS -A <project_name>
#PBS -q debug-scaling
#PBS -k doe
#PBS -j oe
cd ${PBS_O_WORKDIR}

# Load and list modules
module use /soft/modulefiles
module load conda
conda activate
module list

# Print info about job
echo "Jobid: $PBS_JOBID"
echo "Running on host `hostname`"
echo "Running on nodes `cat $PBS_NODEFILE`"
NODES=$(cat $PBS_NODEFILE | wc -l)
RANKS_PER_NODE=4
RANKS=$(( NODES * $RANKS_PER_NODE ))

# Run MPI program
echo "Running MPI program with $NODES nodes, $RANKS ranks and $RANKS_PER_NODE ranks per node"
CPU_BIND_LIST="24:16:8:1"
mpirun -n $RANKS --ppn $RANKS_PER_NODE --cpu-bind=list:${CPU_BIND_LIST} python mpi4py_ex.py

