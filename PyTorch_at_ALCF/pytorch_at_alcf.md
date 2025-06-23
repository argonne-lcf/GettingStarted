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
