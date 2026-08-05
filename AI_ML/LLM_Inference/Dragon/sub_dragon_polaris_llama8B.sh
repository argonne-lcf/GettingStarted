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
python -m venv /tmp/_env --system-site-packages
source /tmp/_env/bin/activate
CONDA_VLLM=$(python -c "import sys; sys.path.insert(0, '/path/to/conda/env/lib/pythonX.X/site-packages'); import vllm; print(vllm.__path__[0])")
VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
cp -r $CONDA_VLLM $VENV_SITE
cp -r ${CONDA_VLLM}-* $VENV_SITE
pip install dragonhpc[telemetry] # no need to install the extra ai package, we provide vllm from base conda env
DRAGON_PKG_DIR=$(python -c 'import dragon, os; print(os.path.dirname(dragon.__file__))')
patch -p1 -N -d "$DRAGON_PKG_DIR" < ./dragon.patch || true
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
