#!/bin/bash
#PBS -l walltime=1:00:00
#PBS -q gpu_hack_prio
#PBS -l select=2 
#PBS -A gpu_hack
#PBS -l daos=daos_user
#PBS -l filesystems=home:flare:daos_user
#PBS -N test_frameworks_pt2.5

if [[ $PBS_JOBNAME == "STDIN" ]]; then
    echo "Interactive"
else
    cd ${PBS_O_WORKDIR}
fi
module use /soft/modulefiles
module load frameworks
# Activate Python environment
source /flare/gpu_hack/hzheng/pyenvs/pt2.5/bin/activate
# Ensure Hugging Face datasets library is available
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

ulimit -c unlimited
export FI_MR_ZE_CACHE_MONITOR_ENABLED=0
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=1048576
export FI_CXI_CQ_FILL_PERCENT=30
export INTELGT_AUTO_ATTACH_DISABLE=1
export PALS_PING_PERIOD=240
export PALS_RPC_TIMEOUT=240
export MPIR_CVAR_GATHERV_INTER_SSEND_MIN_PROCS=-1 # to solve the sync send issue in Horovod seg fault


export CCL_ALLREDUCE=topo
export CCL_ALLREDUCE_SCALEOUT=rabenseifner 

export CCL_KVS_MODE=mpi
export FI_MR_CACHE_MONITOR=userfaultfd
export CCL_WORKER_AFFINITY="42,43,44,45,46,47,94,95,96,97,98,99" 
export ZE_ENABLE_PROFILING=0
export ZE_ENABLE_METRICS=0
export CPU_BINDING="list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79" # 12 ppn with each rank having 4 cores

python -c "import torch; print(torch.__version__, torch.__file__)"

export DFTRACER_ENABLE=1
export DFTRACER_DISABLE_IO=1

mpiexec -np 12 --ppn 12 --cpu-bind $CPU_BINDING python3 ./test_miniGPT.py --profile --synthetic --trace-dir ./trace-1-node

mpiexec -np 24 --ppn 12 --cpu-bind $CPU_BINDING python3 ./test_miniGPT.py --no-gpu-profile --profile --trace-dir ./trace-2-nodes
