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

# Load the modules
module load frameworks
module load mpifileutils
module load xpu-smi
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
mpicc -O2 -o $UTILS/bcast $UTILS/bcast.c

# Build the virtual environment with EL and move to other nodes
python -m venv $(pwd)/_env --system-site-packages
source $(pwd)/_env/bin/activate
# cd $(pwd)
if [ ! -d ./ensemble_launcher ];then
  git clone https://github.com/argonne-lcf/ensemble_launcher.git
fi
cd ensemble_launcher
git checkout multi_node_vllm # NB: to remove, things will be merged into main
pip install .
cd $PBS_O_WORKDIR
mpiexec -np "${NODES}" -ppn 1 --cpu-bind numa $UTILS/bcast \
  $(pwd)/_env /tmp

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

# Pre-build vLLM model-info cache
export VLLM_CACHE_ROOT=$PWD/.vllm_cache
echo "Building vLLM model-info cache in ${VLLM_CACHE_ROOT} ..."
python $UTILS/vllm_build_model_cache.py
echo "Cache build complete."

# Move model-info cache to /tmp on the nodes
MODELINFO_FLARE_PATH=$VLLM_CACHE_ROOT
MODELINFO_TMP_PATH=$TMP_PATH/hub/
mpiexec -np "${NODES}" -ppn 1 --cpu-bind numa $UTILS/bcast \
  $MODELINFO_FLARE_PATH $MODELINFO_TMP_PATH
export VLLM_CACHE_ROOT=${MODELINFO_TMP_PATH}/.vllm_cache

# Other env variables
export TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Launch workflow
echo -e "\n\nLaunching $MODEL on $NODES nodes with $ENGINES_PER_NODE engines per node..."
python3 ./EL_request_driven.py \
  --model_name $MODEL \
  --cache_dir $HF_HOME \
  --tp_size $TP_SIZE \
  --batch_size $BATCH_SIZE \
  --prompt_file ${UTILS}/prompts.jsonl

mpirun -np "${NODES}" --ppn 1 pkill -f python