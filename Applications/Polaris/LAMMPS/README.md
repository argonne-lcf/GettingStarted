# LAMMPS

## Makefiles to build LAMMPS executables

The following Makefiles can be used to build LAMMPS executables. These were generated using the `30Jul21` version of LAMMPS and modifications may be needed depending on the specific version of LAMMPS used. These Makefiles should be copied to the `lammps-<version>/src/MAKE/MACHINES` directory.
```
Makefile.polaris_kokkos_nvidia
```

## Makefiles to build lib/gpu library

The following Makefiles can be used to build the LAMMPS GPU library for use with the GPU package.  These can be copied to the `lammps-<version>/lib/gpu` directory.
```
Makefile.gpu_polaris_nvidia  
```

## Building LAMMPS with the KOKKOS package

The following can be used to build LAMMPS with the KOKKOS package using the NVIDIA compilers.
```
module load craype-accel-nvidia80
cd lammps-<version>/src
make yes-KOKKOS
make polaris_kokkos_nvidia -j 16
```

## Building LAMMPS with the GPU package

The following can be used to build LAMMPS with the GPU package using the NVIDIA compilers.
```
module load craype-accel-nvidia80
cd lammps-<version>/lib/gpu
make -f Makefile.gpu_polaris_nvidia -j 16

cd ../../src
make yes-GPU
make polaris_nvidia -j 16
```

## Example submission scripts for multi-node jobs

Example job submission scripts are provided for running LAMMPS with the KOKKOS and GPU packages. 
```
submit_kokkos.sh
submit_gpu.sh
```



