#!/bin/bash -l
#PBS -N mpi-vllm
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -q debug-scaling
#PBS -A <project_name>
#PBS -l filesystems=home:eagle
#PBS -j oe
cd $PBS_O_WORKDIR

set -e

# Load modules
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
RANKS=$(( NODES * ENGINES_PER_NODE ))
CPU_BIND="list:24-31,16-23,8-15,0-7"

# Compile bcast 
UTILS=$PWD/../utils
mpicc -O2 -o $UTILS/bcast $UTILS/bcast.c -lmpi_gtl_cuda

# Move model weights to /tmp on the nodes
MODEL_DIR="models--${MODEL//\//--}"
MODEL_EAGLE_PATH=/eagle/datasets/model-weights/hub/$MODEL_DIR
if [[ ! -e "$MODEL_EAGLE_PATH" ]]; then
    echo "Did not find model $MODEL_EAGLE_PATH"
    exit 1
fi
MODEL_TMP_PATH=/tmp/hf_home/hub/
mpiexec -np $NODES -ppn 1 --cpu-bind numa $UTILS/bcast \
  $MODEL_EAGLE_PATH $MODEL_TMP_PATH
export HF_HOME=/tmp/hf_home

# Move prompts to /tmp on the nodes
PROMPTS_EAGLE_PATH=$UTILS/prompts.jsonl
PROMPTS_TMP_PATH=/tmp/hf_home
mpiexec -np $NODES -ppn 1 --cpu-bind numa $UTILS/bcast \
  $PROMPTS_EAGLE_PATH $PROMPTS_TMP_PATH

# Other env variables
export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1
export VLLM_LOGGING_LEVEL=INFO # DEBUG, INFO, WARNING, ERROR

# Launch workflow
echo -e "\n\nLaunching workflow on $NODES nodes with $ENGINES_PER_NODE engines per node..."
mpiexec -n $RANKS --ppn $ENGINES_PER_NODE --cpu-bind $CPU_BIND \
  python ./mpi_llm_inference.py \
  --hf_token $HF_TOKEN  \
  --model_name $MODEL \
  --tp_size $TP_SIZE \
  --batch_size $BATCH_SIZE \
  --prompt_file $PROMPTS_TMP_PATH/prompts.jsonl 
