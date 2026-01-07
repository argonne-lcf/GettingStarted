#!/bin/bash

module load cmake

BASE=/home/knight/lammps/lammps-git-cmake/build-aurora

mkdir -p ${BASE}
cd ${BASE}

# Default build
cmake -C ../cmake/presets/aurora.cmake \
	-D CMAKE_INSTALL_PREFIX=${PWD} \
	-DPKG_ML-SNAP=ON \
	-DPKG_MOLECULE=ON \
	-DPKG_RIGID=ON \
	-DPGK_KSPACE=ON \
	../cmake 
#make -j 32

