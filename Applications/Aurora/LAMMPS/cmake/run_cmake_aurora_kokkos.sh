#!/bin/bash

module load cmake

BASE=/home/knight/lammps/lammps-git-cmake/build-aurora_kokkos

mkdir -p ${BASE}
cd ${BASE}

# Default build
cmake -C ../cmake/presets/kokkos-sycl-intel.cmake \
	-D CMAKE_INSTALL_PREFIX=${PWD} \
	-DPKG_ML-SNAP=ON \
	-DPKG_MOLECULE=ON \
	-DPKG_RIGID=ON \
	-DPGK_KSPACE=ON \
	../cmake 
#make -j 32
