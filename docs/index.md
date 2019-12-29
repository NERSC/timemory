# Welcome to the timemory Documentation!

## Timing + Memory + Hardware Counter Utilities for C / C++ / CUDA / Python

### Status

[![Build Status](https://travis-ci.org/NERSC/timemory.svg?branch=master)](https://travis-ci.org/NERSC/timemory)
[![Build status](https://ci.appveyor.com/api/projects/status/8xk72ootwsefi8c1/branch/master?svg=true)](https://ci.appveyor.com/project/jrmadsen/timemory/branch/master)
[![codecov](https://codecov.io/gh/NERSC/timemory/branch/master/graph/badge.svg)](https://codecov.io/gh/NERSC/timemory)

| Source         | Command                                           |
| -------------- | ------------------------------------------------- |
| GitHub         | `git clone https://github.com/NERSC/timemory.git` |
| PyPi           | `pip install timemory`                            |
| Anaconda Cloud | `conda install -c jrmadsen timemory`              |

### Relevant Links

- [GitHub (Source code)](https://github.com/NERSC/timemory)
- [Documentation](https://timemory.readthedocs.io/en/latest/)
- [API (Doxygen)](https://timemory.readthedocs.io/en/latest/doxygen-html/)
- [Testing Dashboard (CDash)](https://cdash.nersc.gov/index.php?project=TiMemory)

### Anaconda Cloud

- [![Conda Recipe](https://img.shields.io/badge/recipe-timemory-green.svg)](https://anaconda.org/jrmadsen/timemory)
- [![Anaconda-Server Badge](https://anaconda.org/jrmadsen/timemory/badges/version.svg)](https://anaconda.org/jrmadsen/timemory)
- [![Anaconda-Server Badge](https://anaconda.org/jrmadsen/timemory/badges/platforms.svg)](https://anaconda.org/jrmadsen/timemory)
- [![Anaconda-Server Badge](https://anaconda.org/jrmadsen/timemory/badges/downloads.svg)](https://anaconda.org/jrmadsen/timemory)

## Why Use timemory?

- __*Timemory is arguably the most customizable performance measurement and analysis API available*__
- __*High-performance*__: very low overhead when enabled and borderline negligible runtime disabled
- Ability to arbitrarily switch and combine different measurement types anywhere in application
- Provides static reporting (fixed at compile-time), dynamic reporting (selected at run-time), or hybrid
    - Enable static wall-clock and cpu-clock reporting with ability to dynamically enable hardware-counters at runtime

### Support for Multiple Instrumentation Marker APIs

- NVTX for Nsight-Systems and NVprof
- [LIKWID](https://github.com/RRZE-HPC/likwid)
- [Caliper](https://github.com/LLNL/Caliper)
- [TAU](https://www.cs.uoregon.edu/research/tau/home.php)
- ittnotify (Intel VTune and Advisor)

### Create Your Own Performance and Analysis Tools

- Written in C++
- Direct access to performance analysis data in Python and C++
- Create your own components: any one-time measurement or start/stop paradigm can be wrapped with timemory
    - Flexible and easily extensible interface: no data type restrictions in custom components

### Generic Bundling of Multiple Tools

- CPU hardware counters via PAPI
- NVIDIA GPU hardware counters via CUPTI
- NVIDIA GPU tracing via CUPTI
- Generating a Roofline for performance-critical sections on the CPU and NVIDIA GPUs
- Memory usage
- Tool insertiong around `malloc`, `calloc`, `free`, `cudaMalloc`, `cudaFree`
- Wall-clock, cpu-clock, system-clock timing
- Number of bytes read/written to file-system (and rate)
- Number of context switches
- Trip counts
- CUDA kernel runtime(s)

### Powerful GOTCHA Extensions

- [GOTCHA](https://github.com/LLNL/GOTCHA) is an API for LD_PRELOAD
    - Significantly simplify existing implementations
- Scoped GOTCHA
- Use gotcha component to replace external function calls with own instrumentation
- Use gotcha component to instrument external library calls

### Multi-language Support

- Variadic interface to all the utilities from C code
- Variadic interface to all the utilities from C++ code
- Variadic interface to all the utilities from Python code
    - Includes context-managers and decorators

## Table of contents

1. [About](about) -- About the timemory project
2. [Features](features) -- Features supported and provided by timemory
3. [Installation](installation) -- How to install timemory
4. [Components](components/overview) -- Available components in timemory and their wrappers
5. [Getting Started](getting_started/overview) -- How to get started using timemory
    - Project integration, basic syntax, custom components, generating a roofline
6. [Runtime Overhead](overhead) -- Analysis of runtime overhead of timemory
7. [Doxygen Documentation](doxygen-xml) -- Doxygen source code documentation
