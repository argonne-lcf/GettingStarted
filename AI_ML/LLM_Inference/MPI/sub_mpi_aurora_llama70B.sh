#!/bin/bash -l
##PBS -N mpi-vllm
##PBS -l select=2
##PBS -l walltime=00:30:00
##PBS -q debug-scaling
##PBS -A <project_name>
##PBS -l filesystems=home:flare
##PBS -j oe
#cd $PBS_O_WORKDIR

set -e

# Load modules
module load frameworks
module load xpu-smi
module list

# Make sure HF Token is set
if [[ -z "${HF_TOKEN}" ]]; then
    echo "Error: HF_TOKEN is not set" >&2
    exit 1
fi

# Set env variables
NODES=$(cat ${PBS_NODEFILE} | wc -l)
MODEL=meta-llama/Llama-3.3-70B-Instruct
TP_SIZE=4
BATCH_SIZE=16
ENGINES_PER_NODE=2
RANKS=$(( NODES * ENGINES_PER_NODE ))
CPU_BIND="list:1-48:53-100"

# Compile bcast
UTILS=$PWD/../utils
mpicc -O2 -o $UTILS/bcast $UTILS/bcast.c

# Move model weights to /tmp on the nodes
MODEL_DIR="models--${MODEL//\//--}"
MODEL_FLARE_PATH=/flare/datasets/model-weights/hub/$MODEL_DIR
if [[ ! -e "$MODEL_FLARE_PATH" ]]; then
    echo "Did not find model $MODEL_FLARE_PATH"
    exit 1
fi
MODEL_TMP_PATH=/tmp/hf_home/hub/
mpiexec -np "${NODES}" -ppn 1 --cpu-bind numa $UTILS/bcast \
  $MODEL_FLARE_PATH $MODEL_TMP_PATH
export HF_HOME=/tmp/hf_home

# Pre-build vLLM model-info caches
export VLLM_CACHE_ROOT=/flare/datascience/balin/vllm/dragon_Jun26/.vllm_cache
echo "Building vLLM model-info caches in ${VLLM_CACHE_ROOT} ..."
python $UTILS/vllm_build_model_cache.py
echo "Cache build complete."

# Move model-info cache to /tmp on the nodes
MODELINFO_FLARE_PATH=$VLLM_CACHE_ROOT
MODELINFO_TMP_PATH=/tmp/hf_home/hub/
mpiexec -np "${NODES}" -ppn 1 --cpu-bind numa $UTILS/bcast \
  $MODELINFO_FLARE_PATH $MODELINFO_TMP_PATH
export VLLM_CACHE_ROOT=${MODELINFO_TMP_PATH}/.vllm_cache

# Move prompts to /tmp on the nodes
PROMPTS_FLARE_PATH=$UTILS/prompts.jsonl
PROMPTS_TMP_PATH=/tmp/hf_home/
mpiexec -np "${NODES}" -ppn 1 --cpu-bind numa $UTILS/bcast \
  $PROMPTS_FLARE_PATH $PROMPTS_TMP_PATH

# Other env variables
export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1
export VLLM_LOGGING_LEVEL=INFO # DEBUG, INFO, WARNING, ERROR
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Launch workflow
echo -e "\n\nLaunching $MODEL on $NODES nodes with $ENGINES_PER_NODE engines per node..."
mpiexec -n $RANKS --ppn $ENGINES_PER_NODE --cpu-bind $CPU_BIND \
  python ./mpi_llm_inference.py \
  --hf_token $HF_TOKEN  \
  --model_name $MODEL \
  --tp_size $TP_SIZE \
  --batch_size $BATCH_SIZE \
  --prompt_file ${PROMPTS_TMP_PATH}/prompts.jsonl 
