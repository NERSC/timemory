# Tools

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 1

   tools/timem/README
   tools/timemory-avail/README
   tools/timemory-run/README
   tools/timemory-stubs/README
   tools/timemory-jump/README
   tools/timemory-mpip/README
   tools/timemory-ncclp/README
   tools/timemory-ompt/README
   tools/kokkos-connector/README
```

This section covers the executables and libraries that are distributed as part of the library.

- Executables
    - [timem](tools/timem/README.md)
        - Drop-in replacement for `time` (UNIX)
        - For MPI applications, use `timem-mpi`
    - [timemory-avail](tools/timemory-avail/README.md)
        - Use this executable to query available components, available settings, and available hardware counters
    - [timemory-run](tools/timemory-run/README.md)
        - Use this executable (Linux-only) for dynamic instrumentation
- Libraries
    - [timemory-stubs](tools/timemory-stubs/README.md)
        - Provides timemory instrumentation stubs for dynamic library preloading
    - [timemory-jump](tools/timemory-jump/README.md)
        - Provides timemory instrumentation via `dlopen` and `dlsym`
    - [timemory-mpip](tools/timemory-mpip/README.md)
        - Provide MPI profiling via GOTCHA
    - [timemory-ncclp](tools/timemory-ncclp/README.md)
        - Provide NCCL profiling via GOTCHA
    - [timemory-ompt](tools/timemory-ompt/README.md)
        - Provide OpenMP profiling via OMPT (OpenMP Tools)
    - [Kokkos Connectors](tools/kokkos-connector/README.md)
        - Libraries for Kokkos profiling
