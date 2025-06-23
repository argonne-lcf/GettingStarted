# PyTorch at ALCF

## PyTorch on Polaris

### SSH into Polaris
```bash
# from your local machine
ssh <ALCF Username>@polaris.alcf.anl.gov
```
- The system will prompt you for a password
```
Password:
Type in your MobilePass Auto-generated passcode
```
- A shell opens in the Login-node, and place you in your home directory
```bash
<ALCF Username>@polaris-login-01:~>
```
- Submit a job request for an interactive session with one node
```bash
qsub -l select=1 -l walltime=00:59:00 -A lighthouse-purdue -q R5020963 -l filesystems=home:eagle -I
```
- A session opens in a compute node:
```bash
<ALCF Username>@x3001c0s25b0n0:~>
```

The full snippet:
```bash 
Your Local Machine:ssh <ALCF Username>@polaris.alcf.anl.gov
Password:
Type in your MobilePass Auto-generated passcode
<ALCF Username>@polaris-login-01:~>qsub -l select=1 -l walltime=00:59:00 -A lighthouse-purdue -q R5020963 -l filesystems=home:eagle -I
```

### AI/ML Software Stack

```bash linenums="1"
module avail                    # shows available modules
module list                     # shows currently loaded modules
module load <module-name>       # loads a module to the current environment
module unload <module-name>     # Unloads/removes a module from the current environment
module use <path>               # Adds an alternative path to the module search path
module restore                  # Resets the currently loaded modules to the default module sets
module show <module-name>       # Shows the environment variables that a module sets, and the paths of different files that the user will access through this module
```

### Load `conda` module and activate the environment

```bash
module use /soft/modulefiles    # add /soft/modulefiles to the search paths
module avail conda              # check the available conda modules


---------------------- /soft/modulefiles ---------------------------------------
   conda/2024-04-29-aws-nccl    conda/2024-04-29 (D)    conda/2024-10-30-workshop

  Where:
   D:  Default Module
```

```bash
module load conda/2024-04-29

Lmod is automatically replacing "nvhpc/23.9" with "gcc-native/12.3".


Lmod is automatically replacing "PrgEnv-nvhpc/8.5.0" with "PrgEnv-gnu/8.5.0".


Due to MODULEPATH changes, the following have been reloaded:
  1) cray-mpich/8.1.28

conda activate 

(2024-04-29/base) <ALCF Username>@polaris-login-01:~>

(2024-04-29/base) <ALCF Username>@polaris-login-01:~> which python
/soft/applications/conda/2024-04-29/mconda3/bin/python
```

### Installing your own packages

You can create a python `venv` with the `--system-site-packages` and install your
own packages in that `venv`. In that way, this new virtual environment will 
leverage the system installed packages along with newly installed packages.

```bash
# activate the base conda environment
module use /soft/modulefiles && module load conda/2024-04-29 && conda activate

# create a virtual environment that inherits packages from the base conda environment
python3 -m venv /path/to/new/venv --system-site-packages

# activate the virtual environment
source /path/to/new/venv/bin/activate

# install a python package
python3 -m pip install <package-name>
```
Now, to use the newly created virtual environment
```bash
module use /soft/modulefiles && module load conda/2024-04-29 && conda activate
source /path/to/new/venv/bin/activate
```

### Submitting an interactive job
After you have run your `qsub` command, and are into a login node, you may 
submit a job using `mpiexec` if you are using more than 1 processes (can be 
understood as more than 1 GPU), or just call your script, if you are using a
single GPU, as you would have done in your laptop.

```bash
mpiexec -n 4 -ppn 4 hostname    # Printing out the hostname from each MPI process

<ALCF Username>@x3002c0s13b0n0:~> mpiexec -n 4 -ppn 4 hostname
x3002c0s13b0n0
x3002c0s13b0n0
x3002c0s13b0n0
x3002c0s13b0n0
```

### To run the example scripts
Clone the repository from a login node and checkout the correct branch
```bash
git clone git@github.com:argonne-lcf/GettingStarted.git
git checkout lighthouse-purdue-2025-06
cd PyTorch_at_ALCF
```
Once you are in the compute node, you can run each of them by:

```bash
python cpu_train.py

python gpu_train.py

mpiexec -n 4 -ppn 4 python ddp_train.py
```

## PyTorch on Aurora

### SSH into Aurora
```bash
# from your local machine
ssh <ALCF Username>@aurora.alcf.anl.gov
```
- The system will prompt you for a password
```
Password:
Type in your MobilePass Auto-generated passcode
```
- A shell opens in the Login-node, and place you in your home directory
```bash
<ALCF Username>@aurora-uan-0010:~>
```
- Submit a job request for an interactive session with one node
```bash
qsub -l select=1 -l walltime=00:59:00 -A lighthouse-purdue -q R5020963 -l filesystems=home:flare -I
```
- A session opens in a compute node:
```bash
<ALCF Username>@x4711c5s0b0n0:~>
```

The full snippet:
```bash 
Your Local Machine:ssh <ALCF Username>@aurora.alcf.anl.gov
Password:
Type in your MobilePass Auto-generated passcode
<ALCF Username>@aurora-uan-0010:~>qsub -l select=1 -l walltime=00:59:00 -A lighthouse-purdue -q R5020963 -l filesystems=home:flare -I
```

### Load the Frameworks module
Loading the frameworks module will activate a `conda` environment with PyTorch
in it
```bash
module load frameworks
(/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0) <ALCF Username>@x4711c5s0b0n0:~>

which python
/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0/bin/python
```
### Submitting an interactive job
After you have run your `qsub` command, and are into a login node, you may 
submit a job using `mpiexec` if you are using more than 1 processes (can be 
understood as more than 1 GPU), or just call your script, if you are using a
single GPU, as you would have done in your laptop.

```bash
mpiexec -n 12 -ppn 12 hostname    # Printing out the hostname from each MPI process

<ALCF Username>@x4711c5s0b0n0:~> mpiexec -n 12 -ppn 12 hostname
x4711c5s0b0n0
x4711c5s0b0n0
x4711c5s0b0n0
x4711c5s0b0n0
x4711c5s0b0n0
x4711c5s0b0n0
x4711c5s0b0n0
x4711c5s0b0n0
x4711c5s0b0n0
x4711c5s0b0n0
x4711c5s0b0n0
x4711c5s0b0n0
```

### Run the example scripts
```bash
mpiexec -n 12 -ppn 12 python xpu_train.py
```







