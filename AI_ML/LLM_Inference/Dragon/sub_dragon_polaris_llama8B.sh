#!/bin/bash -l
#PBS -N dragon-vllm
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -q debug-scaling
#PBS -A <project_name>
#PBS -l filesystems=home:eagle
#PBS -j oe
cd $PBS_O_WORKDIR

set -e

# Load conda module
module use /soft/modulefiles
module load conda
conda activate
module list

# Make sure HF Token is set
if [[ -z "${HF_TOKEN}" ]]; then
    echo "Error: HF_TOKEN is not set" >&2
    exit 1
fi

# Set env variables
NODES=$(cat ${PBS_NODEFILE} | wc -l)
MODEL=meta-llama/Llama-3.1-8B-Instruct
TP_SIZE=1
BATCH_SIZE=16
ENGINES_PER_NODE=4

# Compile bcast
mpicc -O2 -o ./bcast /eagle/datasets/softwares/bcast/bcast.c -lmpi_gtl_cuda

# Build the virtual environment with Dragon and move to other nodes
python -m venv /tmp/_env 
source /tmp/_env/bin/activate
pip install dragonhpc[telemetry,ai]
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/2.2.0rc1/lib64
mpiexec -np "${NODES}" -ppn 1 --cpu-bind numa ./bcast --no-root-write \
  /tmp/_env /tmp

# Move model weights to /tmp on the nodes
MODEL_DIR="models--${MODEL//\//--}"
MODEL_EAGLE_PATH=/eagle/datasets/model-weights/hub/$MODEL_DIR
if [[ ! -e "$MODEL_EAGLE_PATH" ]]; then
    echo "Did not find model $MODEL_EAGLE_PATH"
    exit 1
fi
MODEL_TMP_PATH=/tmp/hf_home/hub/
mpiexec -np "${NODES}" -ppn 1 --cpu-bind numa ./bcast \
  $MODEL_EAGLE_PATH $MODEL_TMP_PATH
export HF_HOME=/tmp/hf_home

# Other env variables
export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Launch workflow
echo -e "\n\nLaunching $MODEL on $NODES nodes with $ENGINES_PER_NODE engines per node..."
dragon ./dragon_llm_inference.py \
  --hf_token $HF_TOKEN  \
  --model_name $MODEL \
  --tp_size $TP_SIZE \
  --batch_size $BATCH_SIZE \
  --prompt_file /eagle/datasets/prompts/prompts.jsonl
