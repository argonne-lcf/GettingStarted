#!/bin/bash

# Default to Aurora
SYSTEM=${1:-"aurora"}

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# Load modules and activate the Dragon venv (see ../README.md for setup)
if [ "$SYSTEM" == "aurora" ]; then
    module load frameworks
elif [ "$SYSTEM" == "polaris" ]; then
    module use /soft/modulefiles
    module unload xalt
    module load conda
    conda activate base
fi
module list

DRAGON_VENV="${SRC}/../_env"
if [ ! -d "$DRAGON_VENV" ]; then
    echo "Error: Dragon venv not found at $DRAGON_VENV"
    echo "       Follow the install steps in ../README.md first."
    exit 1
fi
source "${DRAGON_VENV}/bin/activate"

# Resolve the Dragon install root from the active venv and pass it to CMake
DRAGON_ROOT=$(python -c "import dragon, os; print(os.path.dirname(dragon.__file__))" 2>/dev/null)
if [ -z "$DRAGON_ROOT" ] || [ ! -d "$DRAGON_ROOT" ]; then
    echo "Error: could not locate the dragonhpc install from the active Python."
    echo "       Make sure dragonhpc is installed in ${DRAGON_VENV}."
    exit 1
fi
echo "Using DRAGON_ROOT=$DRAGON_ROOT"

# Build sim.cpp
cmake -DDRAGON_ROOT="${DRAGON_ROOT}" ./
make
