#!/bin/bash

# Default to Aurora
SYSTEM=${1:-"aurora"}

# Clean directory
function clean_directory {
  if [ -f sim ]; then
    rm sim
  fi
  if [ -f CMakeCache.txt ]; then
    rm CMakeCache.txt
  fi
  if [ -d CMakeFiles ]; then
    rm -r CMakeFiles
  fi
  if [ -f Makefile ]; then
    rm Makefile
  fi
  if [ -f cmake_install.cmake ]; then
    rm cmake_install.cmake
  fi
}
clean_directory

# Load modules
if [ "$SYSTEM" == "aurora" ]; then
    module load frameworks
fi
module list

# Build sim.cpp
cmake -DSMARTREDIS_INSTALL_DIR=$PWD/../SmartRedis/install ./
make

