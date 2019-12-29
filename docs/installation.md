# Installing timemory

- Required
    - C++ compiler (GNU, MSVC, Clang, Intel[^1], PGI[^1])
    - CMake >= 3.10
- Optional
    - C compiler (GNU, MSVC, Clang, Intel[^1], PGI[^1])
    - Python libraries
    - MPI
    - PAPI
    - CUDA
    - CUPTI
    - gperftools

[^1]: Always use `tim::timemory_finalize()` with these compilers

| Description                 | Command                                                                  |
| :-------------------------- | :----------------------------------------------------------------------- |
| Clone the repository        | `git clone https://github.com/NERSC/timemory.git timemory`               |
| Create build directory      | `mkdir build-timemory && cd build-timemory`                              |
| Run CMake                   | `cmake -DCMAKE_INSTALL_PREFIX=/opt/timemory <CMAKE_OPTIONS> ../timemory` |
| Build and install (Windows) | `cmake --build . --target ALL && cmake --build . --target INSTALL`       |
| Build and install (UNIX)    | `cmake --build . --target all && cmake --build . --target install`       |

## Relevant CMake Options

In most cases, the defaults are acceptable for most configurations.
For the various external packages, timemory will search the `CMAKE_PREFIX_PATH` in the environment and enable CUDA, CUPTI, PAPI, and MPI if it can find those packages.

| Option                             | Values                                                          | Description                                                                          |
| ---------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| TIMEMORY_BUILD_C                   | ON, OFF                                                         | Build the C compatible library                                                       |
| TIMEMORY_BUILD_CALIPER             | ON, OFF                                                         | Enable building Caliper submodule (set to OFF for external)                          |
| TIMEMORY_BUILD_DOCS                | ON, OFF                                                         | Make a `doc` make target                                                             |
| TIMEMORY_BUILD_EXAMPLES            | ON, OFF                                                         | Build the examples                                                                   |
| TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS | ON, OFF                                                         | Add extra optimization flags to general compilation options                          |
| TIMEMORY_BUILD_GOTCHA              | ON, OFF                                                         | Enable building GOTCHA (set to OFF for external)                                     |
| TIMEMORY_BUILD_GTEST               | ON, OFF                                                         | Enable GoogleTest                                                                    |
| TIMEMORY_BUILD_LTO                 | ON, OFF                                                         | Enable link-time optimizations in build                                              |
| TIMEMORY_BUILD_MPIP                | ON, OFF                                                         | Build the MPI-P library (requires GOTCHA)                                            |
| TIMEMORY_BUILD_PYTHON              | ON, OFF                                                         | Build Python binds for timemory                                                      |
| TIMEMORY_BUILD_TESTING             | ON, OFF                                                         | Enable testing                                                                       |
| TIMEMORY_BUILD_TIMEM               | ON, OFF                                                         | Build the timem tool                                                                 |
| TIMEMORY_BUILD_TOOLS               | ON, OFF                                                         | Enable building tools                                                                |
| TIMEMORY_BUILD_TOOLS_LIBEXPECT     | ON, OFF                                                         | Enable using libexpect to diagnose errors                                            |
| TIMEMORY_FORCE_GPERF_PYTHON        | ON, OFF                                                         | Enable gperftools + Python (may cause termination errors)                            |
| TIMEMORY_GPERF_STATIC              | ON, OFF                                                         | Enable gperftools static targets (enable if gperftools library are built with -fPIC) |
| TIMEMORY_SKIP_BUILD                | ON, OFF                                                         | Disable building any timemory libraries (fast install)                               |
| TIMEMORY_TLS_MODEL                 | `global-dynamic`, `local-dynamic`, `initial-exec`, `local-exec` | Thread-local static model                                                            |
| TIMEMORY_USE_ARCH                  | ON, OFF                                                         | Enable architecture flags (e.g. `-mavx2`, `-mfma`, etc.)                             |
| TIMEMORY_USE_CALIPER               | ON, OFF                                                         | Enable Caliper                                                                       |
| TIMEMORY_USE_COMPILE_TIMING        | ON, OFF                                                         | Enable -ftime-report for compilation times                                           |
| TIMEMORY_USE_CUDA                  | ON, OFF                                                         | Enable CUDA option for GPU measurements                                              |
| TIMEMORY_USE_CUPTI                 | ON, OFF                                                         | Enable CUPTI profiling for NVIDIA GPUs                                               |
| TIMEMORY_USE_EXTERN_INIT           | ON, OFF                                                         | Do initialization in library instead of headers                                      |
| TIMEMORY_USE_GOTCHA                | ON, OFF                                                         | Enable GOTCHA                                                                        |
| TIMEMORY_USE_GPERF                 | ON, OFF                                                         | Enable gperftools                                                                    |
| TIMEMORY_USE_LIKWID                | ON, OFF                                                         | Enable LIKWID marker forwarding                                                      |
| TIMEMORY_USE_MPI                   | ON, OFF                                                         | Enable MPI usage                                                                     |
| TIMEMORY_USE_NVTX                  | ON, OFF                                                         | Enable NVTX marking API                                                              |
| TIMEMORY_USE_PAPI                  | ON, OFF                                                         | Enable PAPI                                                                          |
| TIMEMORY_USE_PYTHON                | ON, OFF                                                         | Enable Python                                                                        |
| TIMEMORY_USE_SANITIZER             | ON, OFF                                                         | Enable -fsanitize flag (=leak)                                                       |
| TIMEMORY_USE_TAU                   | ON, OFF                                                         | Enable TAU marking API                                                               |
| TIMEMORY_USE_UPCXX                 | ON, OFF                                                         | Enable UPCXX usage (MPI support takes precedence)                                    |
| TIMEMORY_USE_VTUNE                 | ON, OFF                                                         | Enable VTune marking API                                                             |
| TIMEMORY_gperftools_COMPONENTS     | `profiler`, `tcmalloc`, `tcmalloc_and_profiler`,                | gperftool components                                                                 |
|                                    | `tcmalloc_debug`, `tcmalloc_minimal`, `tcmalloc_minimal_debug`  |                                                                                      |