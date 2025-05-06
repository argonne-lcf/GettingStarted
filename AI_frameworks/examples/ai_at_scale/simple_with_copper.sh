#!/bin/bash -x
# #PBS -l select=2
# #PBS -l walltime=00:15:00
# #PBS -A datascience
# #PBS -q debug-scaling
# #PBS -k doe
# #PBS -l filesystems=flare

# qsub -l select=512:ncpus=208 -l walltime=02:00:00 -A Aurora_deployment -l filesystems=flare -q lustre_scaling  ./withcopper_aurora_job_script.sh # or - I 

# This example shows loading python modules from a lustre directory with using copper.
 
cd $PBS_O_WORKDIR
echo Jobid: $PBS_JOBID
echo Running on nodes `cat $PBS_NODEFILE`
module load copper
launch_copper.sh
# Prepend /tmp/${USER}/copper/ to all your absolute paths if you want your I/O to go through copper (including PYTHON_PATH, CONDA_PREFIX, CONDA_ROOT and PATH)

NNODES=`wc -l < $PBS_NODEFILE`
RANKS_PER_NODE=12
NRANKS=$(( NNODES * RANKS_PER_NODE ))
echo "App running on NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NRANKS}  RANKS_PER_NODE=${RANKS_PER_NODE}"

#LUS_CONDA_PATH=/lus/flare/projects/alcf_training/softwares/copper-lus-pip-custom-package
LUS_CONDA_PATH=/lus/flare/projects/datascience/fsimini/copper_test_env
# The below 2 lines are only for the first time setup to install a package on a custom dir. Do not use in this job script
# module load python
# python -m pip install --target=${LUS_CONDA_PATH} torch==2.3.1+cxx11.abi torchvision==0.18.1+cxx11.abi torchaudio==2.3.1+cxx11.abi intel-extension-for-pytorch==2.3.110+xpu oneccl_bind_pt==2.3.100+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

module load python
IMAGE_VERSION="24.347.0"
ONEAPI_VERSION="2025.0"
export LD_LIBRARY_PATH=/opt/aurora/${IMAGE_VERSION}/oneapi/${ONEAPI_VERSION}/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/aurora/${IMAGE_VERSION}/oneapi/compiler/${ONEAPI_VERSION}/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/aurora/${IMAGE_VERSION}/oneapi/intel-conda-miniforge/envs/${ONEAPI_VERSION}.0/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/aurora/${IMAGE_VERSION}/oneapi/intel-conda-miniforge/pkgs/intel-sycl-rt-2025.0.4-intel_1519/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/aurora/${IMAGE_VERSION}/updates/oneapi/compiler/${ONEAPI_VERSION}/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/aurora/${IMAGE_VERSION}/support/tools/pti-gpu/0.11.0/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LUS_CONDA_PATH}:$LD_LIBRARY_PATH

time mpirun --np ${NRANKS} --ppn ${RANKS_PER_NODE} --cpu-bind=list:4:9:14:19:20:25:56:61:66:71:74:79 --genvall \
            --genv=PYTHONPATH=/tmp/${USER}/copper/${LUS_CONDA_PATH}:$PYTHONPATH \
            python3 -c "import numpy; print(numpy.__file__)"
            #python3 -c "import torch; print(torch.__file__)"
