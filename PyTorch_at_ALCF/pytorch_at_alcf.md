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
```

The full snippet:
```bash 
Your Local Machine:ssh <ALCF Username>@polaris.alcf.anl.gov
Password:
Type in your MobilePass Auto-generated passcode
<ALCF Username>@polaris-login-01:~>qsub -l select=1 -l walltime=00:59:00 -A lighthouse-purdue -q R5020963 -l filesystems=home:eagle -I
```

### AI/ML Software Stack

```bash
module avail                    # shows available modules
module list                     # shows currently loaded modules
module load <module-name>       # loads a module to the current environment
```
