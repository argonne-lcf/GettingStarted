#!/bin/bash -l
#PBS -N dlpack_example
#PBS -l select=1
#PBS -l walltime=00:20:00
#PBS -l filesystems=home:flare
#PBS -A <project_name>
#PBS -q debug-scaling
#PBS -k doe
#PBS -j oe
cd ${PBS_O_WORKDIR}

# Load and list modules
module load frameworks
module list

# Print info about job
echo "Jobid: $PBS_JOBID"
echo "Running on host `hostname`"
echo "Running on nodes `cat $PBS_NODEFILE`"
NODES=$(cat $PBS_NODEFILE | wc -l)

# Set environment variables
ONEAPI_DEVICE_SELECTOR=level_zero:gpu

# Run DLPack example
echo -e "\n\nRunning DLPack example"
echo "==========================================="
python dlpack_example.py
echo "==========================================="
