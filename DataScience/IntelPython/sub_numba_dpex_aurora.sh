#!/bin/bash -l
#PBS -N numba_dpex_example
#PBS -l select=1
#PBS -l walltime=00:20:00
#PBS -l filesystems=home:flare
#PBS -A <project_name>
#PBS -q debug-scaling
#PBS -k doe
#PBS -j oe
cd ${PBS_O_WORKDIR}

# Load and list modules
module load frameworks
module list

# Print info about job
echo "Jobid: $PBS_JOBID"
echo "Running on host `hostname`"
echo "Running on nodes `cat $PBS_NODEFILE`"
NODES=$(cat $PBS_NODEFILE | wc -l)

# Create a new conda environment
module load cmake
conda create -y --prefix $PWD/_env python=3.12 pip
conda activate $PWD/_env
conda install -y scikit-build
pip install versioneer numpy Cython ninja

# Install dpctl
git clone https://github.com/IntelPython/dpctl.git
cd dpctl
git checkout 0.21.1
CXX=$(which dpcpp) python setup.py install
cd ..

# Install dpnp
git clone https://github.com/IntelPython/dpnp.git
cd dpnp
git checkout 0.19.1
CXX=$(which dpcpp) python setup.py install -- -G Ninja -DCMAKE_C_COMPILER:PATH=`which icx` -DCMAKE_CXX_COMPILER:PATH=`which icpx`
cd ..

# Install numba-dpex
conda install -y numba==0.59* -c conda-forge
git clone https://github.com/argonne-lcf/numba-dpex.git
cd numba-dpex
CXX=$(which dpcpp) python setup.py develop
cd ..

# Run numba-dpex example
echo "Running numba-dpex example"
echo "--------------------------------"
python numba_dpex_example.py
echo "--------------------------------"
