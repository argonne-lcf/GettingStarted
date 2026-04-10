# Intel Data Parallel Extension for Python (DPEP)

On Aurora, users can access Intel's Python stack comprising of compilers and libraries for programming heterogenous devices, namely the Data Parallel Extensions for Python (DPEP).
DPEP is composed of three main packages for programming on CPUs and GPUs:

- [dpnp](https://github.com/IntelPython/dpnp) - Data Parallel Extensions for Numpy is a library that implements a subset of Numpy. The subset is a drop-in replacement of core Numpy functions and numerical data types, similar to CuPy for CUDA devices.
- [dpctl](https://github.com/IntelPython/dpctl) - Data Parallel Control library provides utilities for device selection, allocation of data on devices, and support for creation of user-defined data-parallel extensions.
- [numba_dpex](https://github.com/IntelPython/numba-dpex) - Data Parallel Extensions for Numba is an extension to Numba compiler for programming data-parallel devices similar to developing programs with Numba for CPU or CUDA devices.

To find more details on Intel's DPEP packages, please take a look at our [documentation page](https://docs.alcf.anl.gov/aurora/data-science/python/#intels-data-parallel-extensions-for-python-dpep).

To run all `dpctl` and `dpnp` examples on Aurora:

```cli
$ qsub sub_dpctl_dpnp_aurora.sh
```

To run the DLPack example on Aurora:

```cli
$ qsub sub_dlpack_aurora.sh
```

To install a new `conda` environment and run the `numba-dpex` example:

```cli
$ qsub sub_numba_dpex_aurora.sh
```

