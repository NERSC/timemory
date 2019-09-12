# Installing TiMemory

- Required
    - C++ compiler (GNU, MSVC, Clang, Intel, PGI)
    - CMake >= 3.10
- Optional
    - C compiler (GNU, MSVC, Clang, Intel, PGI)
    - Python libraries
    - MPI
    - PAPI
    - CUDA
    - CUPTI
    - gperftools

| Description            | Command                                                                  |
| :--------------------- | :----------------------------------------------------------------------- |
| Clone the repository   | `git clone https://github.com/NERSC/timemory.git timemory`               |
| Create build directory | `mkdir build-timemory && cd build-timemory`                              |
| Run CMake              | `cmake -DCMAKE_INSTALL_PREFIX=/opt/timemory <CMAKE_OPTIONS> ../timemory` |
| Build and install      | `cmake --build . --target ALL && cmake --build . --target INSTALL`       |

## Relevant CMake Options

In most cases, the defaults are acceptable for most configurations.
If TiMemory is being utilized for it's roofline capabilities, it is recommended to enable the `TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS` and `TIMEMORY_USE_ARCH` options.
For the various external packages, TiMemory will search the `CMAKE_PREFIX_PATH` in the environment and enable CUDA, CUPTI, PAPI, and MPI if it can find those packages.

| Option                             | Values                                                  | Description                                                                       |
| ---------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------- |
| TIMEMORY_BUILD_C                   | ON, OFF                                                 | Build the C compatible library                                                    |
| TIMEMORY_BUILD_EXAMPLES            | ON, OFF                                                 | Build the examples                                                                |
| TIMEMORY_BUILD_EXTERN_TEMPLATES    | ON, OFF                                                 | Pre-compile select list of templates                                              |
| TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS | ON, OFF                                                 | Add extra optimization flags for vectorization                                    |
| TIMEMORY_BUILD_GTEST               | ON, OFF                                                 | Enable unit tests with GoogleTest                                                 |
| TIMEMORY_BUILD_LTO                 | ON, OFF                                                 | Enable link-time optimizations in build                                           |
| TIMEMORY_BUILD_PYTHON              | ON, OFF                                                 | Build Python bindings for TiMemory                                                |
| TIMEMORY_BUILD_TOOLS               | ON, OFF                                                 | Enable building command-line tools                                                |
| TIMEMORY_BUILD_CALIPER             | ON, OFF                                                 | Enable building Caliper submodule                                                 |
| TIMEMORY_BUILD_GOTCHA              | ON, OFF                                                 | Enable building GOTCHA submodule                                                  |
| TIMEMORY_USE_CLANG_TIDY            | ON, OFF                                                 | Enable running clang-tidy while building                                          |
| TIMEMORY_USE_CUDA                  | ON, OFF                                                 | Enable CUDA option for GPU measurements                                           |
| TIMEMORY_USE_CUPTI                 | ON, OFF                                                 | Enable CUPTI profiling for NVIDIA GPUs                                            |
| TIMEMORY_USE_EXTERN_INIT           | ON, OFF                                                 | Perform initialization in library instead of headers (for cross-language support) |
| TIMEMORY_USE_GPERF                 | ON, OFF                                                 | Enable profiling via gperf-tools                                                  |
| TIMEMORY_USE_MPI                   | ON, OFF                                                 | Enable MPI support                                                                |
| TIMEMORY_USE_PAPI                  | ON, OFF                                                 | Enable PAPI support                                                               |
| TIMEMORY_USE_CALIPER               | ON, OFF                                                 | Find external Caliper if `TIMEMORY_BUILD_CALIPER=OFF`                             |
| TIMEMORY_USE_GOTCHA                | ON, OFF                                                 | Find external GOTCHA if `TIMEMORY_BUILD_GOTCHA=OFF`                               |
| TIMEMORY_USE_ARCH                  | ON, OFF                                                 | Enable architecture-specific flags (e.g. `-xCORE-AVX`, `-mavx2`, etc.)            |
| TIMEMORY_TLS_MODEL                 | global-dynamic, local-dynamic, initial-exec, local-exec | Thread-local static model                                                         |
