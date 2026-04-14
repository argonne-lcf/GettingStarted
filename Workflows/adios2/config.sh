#!/bin/bash

# Default to Aurora
SYSTEM=${1:-"aurora"}

# Clean directory
function clean_directory {
  if ls *.sst 1> /dev/null 2>&1; then
    rm *.sst
  fi
  if ls *.bp 1> /dev/null 2>&1; then
    rm -r *.bp
  fi
  if -f sim 1> /dev/null 2>&1; then
    rm sim
  fi
  if -f CMakeCache.txt 1> /dev/null 2>&1; then
    rm CMakeCache.txt
  fi
  if -f CMakeFiles 1> /dev/null 2>&1; then
    rm -r CMakeFiles
  fi
  if -f Makefile 1> /dev/null 2>&1; then
    rm Makefile
  fi
  if -f cmake_install.cmake 1> /dev/null 2>&1; then
    rm cmake_install.cmake
  fi
}
clean_directory

# Load modules
if [ "$SYSTEM" == "aurora" ]; then
    module load frameworks
    module load adios2/2.11.0-sycl
fi
module list

# Build sim.cpp
cmake ./
make

