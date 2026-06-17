#!/bin/bash -l
##PBS -N el-llm
##PBS -l select=2
##PBS -l walltime=00:30:00
##PBS -q debug-scaling
##PBS -A <project_name>
##PBS -l filesystems=home:flare
##PBS -j oe
#cd $PBS_O_WORKDIR

set -e

# Load frameworks module
module load frameworks
module load mpifileutils
module list

# Make sure HF Token is set
if [[ -z "${HF_TOKEN}" ]]; then
    echo "Error: HF_TOKEN is not set" >&2
    exit 1
fi

# Set job variables
NODES=$(cat ${PBS_NODEFILE} | wc -l)
MODEL=meta-llama/Llama-3.1-8B-Instruct
TP_SIZE=1
BATCH_SIZE=16
ENGINES_PER_NODE=12

# Compile bcast
UTILS=$PWD/../utils
mpicc -O2 -o $UTILS/bcast $UTILS/bcast.c

# Build the virtual environment with Dragon and move to other nodes
python -m venv /tmp/_env --system-site-packages
source /tmp/_env/bin/activate
cd /tmp
git clone https://github.com/argonne-lcf/ensemble_launcher.git
cd ensemble_launcher
git checkout multi_node_vllm # NB: to remove, things will be merged into main
pip install .
cd $PBS_O_WORKDIR
mpiexec -np "${NODES}" -ppn 1 --cpu-bind numa $UTILS/bcast --no-root-write \
  /tmp/_env /tmp

# Other env variables
export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Launch workflow
echo -e "\n\nLaunching $MODEL on $NODES nodes with $ENGINES_PER_NODE engines per node..."
python3 EL_batched_inference.py --num-prompts 32