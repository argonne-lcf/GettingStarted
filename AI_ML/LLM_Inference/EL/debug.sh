#!/bin/bash -l

set -e

# Load the modules
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
MODEL_FLARE_PATH=/flare/datasets/model-weights/hub
TMP_PATH=/tmp/hf_home
GPUS_PER_NODE=12
TP_SIZE=1
BATCH_SIZE=16
ENGINES_PER_NODE=12

# Compile bcast
UTILS=$PWD/../utils
#mpicc -O2 -o $UTILS/bcast $UTILS/bcast.c

# Build the virtual environment with EL and move to other nodes
#python -m venv /tmp/_env --system-site-packages
source /tmp/_env/bin/activate
#cd /tmp
#git clone https://github.com/argonne-lcf/ensemble_launcher.git
#cd ensemble_launcher
#git checkout multi_node_vllm # NB: to remove, things will be merged into main
#pip install .
#cd $PBS_O_WORKDIR
#mpiexec -np "${NODES}" -ppn 1 --cpu-bind numa $UTILS/bcast --no-root-write \
#  /tmp/_env /tmp

# Pre-build vLLM model-info cache
export VLLM_CACHE_ROOT=$PWD/.vllm_cache
echo "Building vLLM model-info cache in ${VLLM_CACHE_ROOT} ..."
#python $UTILS/vllm_build_model_cache.py
echo "Cache build complete."

# Move model-info cache to /tmp on the nodes
MODELINFO_FLARE_PATH=$VLLM_CACHE_ROOT
MODELINFO_TMP_PATH=$TMP_PATH/hub/
#mpiexec -np "${NODES}" -ppn 1 --cpu-bind numa $UTILS/bcast \
#  $MODELINFO_FLARE_PATH $MODELINFO_TMP_PATH
export VLLM_CACHE_ROOT=${MODELINFO_TMP_PATH}/.vllm_cache

# Other env variables
export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Launch workflow
echo -e "\n\nLaunching $MODEL on $NODES nodes with $ENGINES_PER_NODE engines per node..."
python3 ./EL_batched_inference.py \
  --model $MODEL \
  --cache-dir $MODEL_FLARE_PATH \
  --tmp-dir $TMP_PATH \
  --num-gpus-per-model $TP_SIZE \
  --prompt-file ${UTILS}/prompts.jsonl