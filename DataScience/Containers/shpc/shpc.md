# Singularity registry using `shpc`

Singularity Registry HPC (shpc) allows you to install containers as modules. This has been developed by Vanessa Sochat and is maintained here https://github.com/singularityhub/singularity-hpc/

A pre-organized registry with recipes for containers and module files can be found at https://singularityhub.github.io/shpc-registry/ and are available to use on Polaris

We go through installation, configuration and use of shpc on Polaris with examples below but to learn more about using shpc, please follow the documentation here https://singularity-hpc.readthedocs.io/en/latest/index.html

## Install shpc

To install, you will have to load conda module followed by installing shpc from the shpc git repository.

```bash
module load conda
conda activate base
python3 -m venv /lus/grand/projects/<project_name>/envs/shpc_env
source /lus/grand/projects/<project_name>/envs/shpc_env/bin/activate
git clone git@github.com:singularityhub/singularity-hpc
cd singularity-hpc
pip install -e .[all]
```

## Configure
Next we define the directories where the container and modules will be installed.

```bash
shpc config set module_base /lus/grand/projects/<project_name>/modules
shpc config set container_base  /lus/grand/projects/<project_name>/containers
```

## Important commands

`shpc show`: Shows all available registries for installation
`shpc install`: Installs the required container and its corresponding module file

## Running Pytorch using `shpc` on Polaris compute


``` bash
qsub -I -A datascience -q debug-scaling -l singularity_fakeroot=true,select=1,walltime=01:00:00
```

`shpc show` show all available containers 

For pytorch you can `shpc install nvcr.io/nvidia/pytorch`

`shpc config edit` do it once to define the containers and module space.

```bash
module use modules
module load singularity
module load nvcr.io/nvidia/pytorch
```

`module help nvcr.io/nvidia/pytorch` tells you about the module and how to use pytorch commands directly without having to remember the full singularity command.

You can export 

## Example using pytorch

We can now use pytorch command directly for example, to check version of pytorch in singularity container, you can

```
pytorch-exec python -c "import torch;print('pytorch version: ' + torch.__version__)"
```

`export SINGULARITY_COMMAND_OPTS="--bind $(pwd)/source:/mnt --nv"` make sure you run it exactly from the directory where source folder exists

```bash
pytorch-exec python /mnt/pytorch/train_xor.py
```

## Running mpi4py using shpc

