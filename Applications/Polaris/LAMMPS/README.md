# LAMMPS

## Makefiles to build LAMMPS executables

The following Makefiles can be used to build LAMMPS executables. These were generated using a version of LAMMPS from Nov. 3, 2022 and modifications may be needed depending on the specific version of LAMMPS used. These Makefiles should be copied to the `lammps-<version>/src/MAKE/MACHINES` directory. The non-kokkos variants can be used when compiling with the GPU package.
```
Makefile.polaris_nvhpc
Makefile.polaris_nvhpc_kokkos
Makefile.polaris_gnu
Makefile.polaris_gnu_kokkos
```

## Makefiles to build lib/gpu library

The following Makefiles can be used to build the LAMMPS GPU library for use with the GPU package with either the PrgEnv-nvhpc or PrgEnv-gnu modules.  These can be copied to the `lammps-<version>/lib/gpu` directory.
```
gpu/Makefile.gpu_polaris_gnu
gpu/Makefile.gpu_polaris_nvhpc
```

## Building LAMMPS with the KOKKOS package

The following can be used to build LAMMPS with the KOKKOS package using the NVHPC compilers.
```
cd lammps-<version>/src
make yes-KOKKOS
make polaris_nvhpc_kokkos -j 16
```

## Building LAMMPS with the GPU package

The following can be used to build LAMMPS with the GPU package using the NVHPC compilers.
```
cd lammps-<version>/lib/gpu
make -f Makefile.gpu_polaris_nvhpc -j 16

cd ../../src
make yes-GPU
make polaris_nvhpc -j 16
```

## Building LAMMPS with the GNU Programming Environment

The same command above can be executed using the gnu variant of the Makefiles and updating the module environment. An example of building with the KOKKOS package is shown below.
```
module swap PrgEnv-nvhpc PrgEnv-gnu
module load cudatoolkit-standalone

cd lammps-<version>/src
make yes-KOKKOS
make polaris_gnu_kokkos -j 16
```

## Example submission scripts for multi-node jobs

Example job submission scripts are provided for running LAMMPS with the KOKKOS and GPU packages. 
```
submit_kokkos.sh
submit_gpu.sh
```



