#!/bin/bash -l
#PBS -N el-vllm
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -q debug-scaling
#PBS -A <project_name>
#PBS -l filesystems=home:eagle
#PBS -j oe
cd $PBS_O_WORKDIR

set -e

# Load the modules
module use /soft/modulefiles
module load conda
conda activate
module list

# Make sure HF Token is set
if [[ -z "${HF_TOKEN}" ]]; then
    echo "Error: HF_TOKEN is not set" >&2
    exit 1
fi

# Set job variables
NODES=$(cat ${PBS_NODEFILE} | wc -l)
MODEL=meta-llama/Llama-3.1-70B-Instruct
MODEL_EAGLE_PATH=/eagle/datasets/model-weights/hub
TMP_PATH=/tmp/hf_home
TP_SIZE=4
BATCH_SIZE=16
ENGINES_PER_NODE=1

# Compile bcast
UTILS=$PWD/../utils
mpicc -O2 -o $UTILS/bcast $UTILS/bcast.c -lmpi_gtl_cuda

# Build the virtual environment with EL and move to other nodes
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

# Move model weights to /tmp on the nodes
MODEL_DIR="models--${MODEL//\//--}"
MODEL_EAGLE_PATH=/eagle/datasets/model-weights/hub/$MODEL_DIR
if [[ ! -e "$MODEL_EAGLE_PATH" ]]; then
    echo "Did not find model $MODEL_EAGLE_PATH"
    exit 1
fi
MODEL_TMP_PATH=/tmp/hf_home/hub/
mpiexec -np "${NODES}" -ppn 1 --cpu-bind numa $UTILS/bcast \
  $MODEL_EAGLE_PATH $MODEL_TMP_PATH
export HF_HOME=/tmp/hf_home

# Other env variables
export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Launch workflow
echo -e "\n\nLaunching $MODEL on $NODES nodes with $ENGINES_PER_NODE engines per node..."
python3 ./EL_batched_inference.py \
  --model_name $MODEL \
  --cache_dir $HF_HOME \
  --tp_size $TP_SIZE \
  --batch_size $BATCH_SIZE \
  --prompt_file ${UTILS}/prompts.jsonl
