# Welcome to the TiMemory Documentation!

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

## Why Use TiMemory?

- Header-only interface for majority of C++ components
- Variadic interface to all the utilities from C code
- Variadic interface to all the utilities from C++ code
- Variadic interface to all the utilities from Python code
    - Includes context-managers and decorators
- Flexible and easily extensible interface: none of the components are restricted to a certain data type
- Create your own components: any one-time measurement or start/stop paradigm can be wrapped with timemory
- Direct access to performance data in native data structures for language
- High-performance: template meta-programming and lambdas result in extensive inlining
- Ability to arbitrarily switch and combine different measurement types anywhere in application
- Provides static reporting (fixed at compile-time), dynamic reporting (selected at run-time), or hybrid
    - Enable static wall-clock and cpu-clock reporting with ability to dynamically enable hardware-counters at runtime
- Arbitrarily add support for:
    - CPU hardware counters via PAPI without an explicit PAPI dependency and zero `#ifdef`
    - GPU hardware counters via CUPTI without an explicit CUPTI dependency and zero `#ifdef`
    - Generating a roofline for performance-critical sections
    - Extensive tools provided by [Caliper](https://github.com/LLNL/Caliper) including [TAU](https://www.cs.uoregon.edu/research/tau/home.php)
    - Colored CUDA NVTX markers
    - Memory usage
    - Wall-clock, cpu-clock, system-clock timing
    - Number of bytes read/written to file-system (and rate)
    - Number of context switches
    - Trip counts
    - CUDA kernel runtime(s)
    - [GOTCHA](https://github.com/LLNL/GOTCHA) wrappers around external library function calls

## Table of contents

1. [About](about) -- About the TiMemory project
2. [Features](features) -- Features supported and provided by TiMemory
3. [Installation](installation) -- How to install TiMemory
4. [Components](components) -- Available components in TiMemory and their wrappers
5. [Getting Started](getting_started) -- How to get started using TiMemory
    - Project integration, basic syntax, custom components, generating a roofline
6. [Runtime Overhead](overhead) -- Analysis of runtime overhead of TiMemory
7. [Doxygen Documentation](doxygen-xml) -- Doxygen source code documentation
