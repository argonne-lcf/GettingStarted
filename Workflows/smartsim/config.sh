#!/bin/bash

# Default to Aurora
SYSTEM=${1:-"aurora"}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${ROOT}/producer-consumer"
SMARTREDIS_INSTALL_DIR="${ROOT}/SmartRedis/install"

cd "$SRC" || exit 1

# Clean build directory
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

# Build sim.cpp in producer-consumer/
cmake -DSMARTREDIS_INSTALL_DIR="${SMARTREDIS_INSTALL_DIR}" ./
make

