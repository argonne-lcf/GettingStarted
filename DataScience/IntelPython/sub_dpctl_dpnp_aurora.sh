#!/bin/bash -l
#PBS -N dpep_example
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

# Run dpctl example
echo -e"\n\nRunning dpctl device introspection example"
echo "==========================================="
python dpctl_example.py
echo -e "===========================================\n\n"

# Run dpnp example
echo "Running simple dpnp array creation example"
echo "==========================================="
python dpnp_example.py
echo -e "===========================================\n\n"

# Run second dpnp example
echo "Running dpnp timing example"
echo "==========================================="
python dpnp_async_example.py
echo -e "===========================================\n\n"
