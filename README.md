# Getting Started at ALCF

Contained herein is a collection of material (source code, scripts, examples,
etc...) to be used as examples for setting up user environments, running jobs,
debugging and tuning applications, and related tasks for successful use of
computational resources at the Argonne Leadership Computing Facility (ALCF).

Material here is typically developed for and used during ALCF training
workshops, such as the ALCF Computational Performance Workshops typically held
in the Spring and the ALCF Simulation, Data, and Learning Workshops in the
Fall.

If not sure where to start, then a good place may be perusing ProgrammingModels
for the resource of interest to gain experience compiling simple codes and
submitting jobs.

## Contents

1. [Applications](./Applications): Example Makefiles, build scripts, etc. for
   some applications
1. [Data Science](./DataScience/): Data & Learning examples with pointers to
   additional Tutorials
1. [Debug Tools](./DebugTools): Simple examples using available debugging tools
1. [Examples](./Examples): Collection of examples demonstrating common tasks
   (e.g. compilation & setting affinity)
1. [Helper Scripts](./HelperScripts): Collection of scripts for common tasks
1. [Performance Tools](./PerformanceTools): Simple examples using available
   performance tools
1. [Programming Models](./ProgrammingModels): Simple examples using supported
   programming models

## Resources

### Aurora

> [!NOTE]
> Aurora is the newest system at ALCF.
> See:
> [Aurora Machine Overview](https://docs.alcf.anl.gov/aurora/)
> for additional information.

Aurora is a 10,624-node HPE Cray-Ex based system.
It has 166 racks with 21,248 CPUs and 63,744 GPUs.
Each node consists of 2 Intel Xeon CPU Max Series (codename Sapphire Rapids or
SPR) with on-package HBM and 6 Intel Data Center GPU Max Series (codename Ponte
Vecchio or PVC).
Each Xeon CPU has 52 physical cores supporting 2 hardware threads per core and
64 GB of HBM. Each CPU socket has 512 GB of DDR5 memory.
The GPUs are connected all-to-all with Intel Xe Link interfaces.
Each node has 8 HPE Slingshot-11 NICs, and the system is connected in a
Dragonfly topology.
The GPUs may send messages directly to the NIC via PCIe, without the need to
copy into CPU memory.

- Additional information on using the system is available at:
  [Getting Started on Aurora](https://docs.alcf.anl.gov/aurora/getting-started-on-aurora/)

### Polaris

Polaris is the (2nd) newest ALCF system with 560 nodes, each with one AMD Milan
CPU (32 cores) and four NVIDIA A100 GPUs.

- Additional information on using the system is available at:
  [Getting Started on Polaris](https://www.alcf.anl.gov/support/user-guides/polaris/hardware-overview/machine-overview/index.html)

### Sophia

Sophia is an extension of Theta and is comprised of 24 NVIDIA DGX A100 nodes, each with eight NVIDIA A100 Tensor Core GPUs and two AMD Rome CPUs. 

- Additional hardware information is available at:

  [Getting Started on Sophia](https://docs.alcf.anl.gov/sophia/getting-started/)

<details closed><summary><h2>🪦 Previous Systems</h2></summary>

### Cooley

Cooley was a data analysis and visualization cluster consisting of 126 compute nodes, each with 12 Intel Haswell cores and an NVIDIA Tesla K80 dual-GPU card. Additional hardware information is available [here][1].

### Theta

Theta was a Cray XC40, 11.7 petaflops system based on the second-generation Intel Xeon Phi processor codenamed Knights Landing (KNL).  It contained 4,392 compute nodes available to users each with 64 cores, 192 GiB DDR4 & 16 GiB MCDRAM memory, and a 128 GiB SSD. Additional hardware information is available [here][2].


[1]: https://www.alcf.anl.gov/support-center/cooley/cooley-system-overview

[2]: https://www.alcf.anl.gov/support-center/theta/theta-thetagpu-overview

</details>
