# Project Integration

## timemory as a Submodule

Timemory has a permissive MIT license and can be directly included within
another project. C++ projects can take advantage of the header-only feature
of timemory and simply include the folders `source/timemory` and `external/cereal/include`.

## Using CMake

Timemory uses modern CMake INTERFACE targets to include the components you want without
forcing you to include everything -- this means that compiler flags, preprocessor
definitions, include paths, link options, and link libraries are bundled into separate
"library" targets that only need to be "linked" to in CMake.

### Available CMake Targets

These are the full target names available within CMake. These targets are always provided but
may provide an empty target if the underlying specifications (such as a library and include path)
were not available when timemory was installed.

| Target                            | Description                                                       |
| :-------------------------------- | :---------------------------------------------------------------- |
| COMPILED LIBRARIES                |                                                                   |
| `timemory::timemory-cxx-shared`   | C/C++/Fortran library                                             |
| `timemory::timemory-cxx-static`   | C/C++/Fortran library                                             |
| `timemory::timemory-c-shared`     | Minimal C enum interface library (requires `TIMEMORY_BUILD_C=ON`) |
| `timemory::timemory-c-static`     | Minimal C enum interface library (requires `TIMEMORY_BUILD_C=ON`) |
| `timemory::timemory-stubs-shared` | C/C++/Fortran stubs library                                       |
| `timemory::timemory-stubs-static` | C/C++/Fortran stubs library                                       |
| `timemory::timemory-jump-shared`  | C/C++/Fortran jump library                                        |
| `timemory::timemory-jump-static`  | C/C++/Fortran jump library                                        |
| INTERFACE LIBRARIES               |                                                                   |
| `timemory::timemory-address-sanitizer` | Adds compiler flags to enable address sanitizer (-fsanitizer=address) |
| `timemory::timemory-alignment-sanitizer` | Adds compiler flags to enable alignment sanitizer (-fsanitizer=alignment) |
| `timemory::timemory-all-gperftools` | Enables all gperftools components (cpu and heap profilers) |
| `timemory::timemory-allinea-map` | Enables Allinea-MAP support |
| `timemory::timemory-analysis-tools` | Internal. Provides sanitizer, gperftools-cpu, coverage, xray |
| `timemory::timemory-arch` | Adds architecture-specific compiler flags |
| `timemory::timemory-bounds-sanitizer` | Adds compiler flags to enable bounds sanitizer (-fsanitizer=bounds) |
| `timemory::timemory-caliper` | Enables Caliper support |
| `timemory::timemory-cereal-xml` | Enables XML serialization output |
| `timemory::timemory-cereal` | Provides include flags for serialization library |
| `timemory::timemory-compile-options` | Adds the standard set of compiler flags used by timemory |
| `timemory::timemory-compile-timing` | Adds compiler flags which report compilation timing metrics |
| `timemory::timemory-coverage` | Enables code-coverage flags |
| `timemory::timemory-cpu-roofline` | Enables flags and libraries for proper CPU roofline generation |
| `timemory::timemory-craypat` | Enables CrayPAT support |
| `timemory::timemory-cuda-compiler` | Enables some CUDA compiler flags |
| `timemory::timemory-cuda` | Enables CUDA support |
| `timemory::timemory-cudart-device` | Link to CUDA device runtime |
| `timemory::timemory-cudart-static` | Link to CUDA runtime (static library) |
| `timemory::timemory-cudart` | Link to CUDA runtime (shared library) |
| `timemory::timemory-cupti` | Enables CUPTI support (requires linking to libcuda) |
| `timemory::timemory-default-visibility` | Adds -fvisibility=default compiler flag |
| `timemory::timemory-develop-options` | Adds developer compiler flags |
| `timemory::timemory-dmp` | Enables the default distributed memory parallelism library (e.g. MPI, UPC++) |
| `timemory::timemory-dyninst` | Provides flags and libraries for Dyninst (dynamic instrumentation |
| `timemory::timemory-extensions` | Provides a single target for all the timemory extensions which were found |
| `timemory::timemory-extern` | Enables pre-processor directive to ensure all extern templates are used |
| `timemory::timemory-external-shared` | Provides a single target for all the timemory extensions (shared libraries) |
| `timemory::timemory-external-static` | Provides a single target for all the timemory extensions (static libraries) |
| `timemory::timemory-gotcha` | Enables Gotcha support |
| `timemory::timemory-gperftools-compile-options` | Enables compiler flags for resolving function calls in gperftools output |
| `timemory::timemory-gperftools-cpu` | Enables gperftools cpu profiler support |
| `timemory::timemory-gperftools-heap` | Enables gperftools heap profiler support |
| `timemory::timemory-gperftools-static` | Enables gperftools support via static linking |
| `timemory::timemory-gperftools` | Enables user-selected gperftools component () |
| `timemory::timemory-gpu-roofline` | Enables flags and libraries for proper GPU roofline generation |
| `timemory::timemory-headers` | Provides minimal set of include flags to compile with timemory |
| `timemory::timemory-hidden-visibility` | Adds -fvisibility=hidden compiler flag |
| `timemory::timemory-leak-sanitizer` | Adds compiler flags to enable leak sanitizer (-fsanitizer=leak) |
| `timemory::timemory-likwid` | Enables LIKWID support |
| `timemory::timemory-lto` | Adds link-time-optimization flags |
| `timemory::timemory-memory-sanitizer` | Adds compiler flags to enable memory sanitizer (-fsanitizer=memory) |
| `timemory::timemory-mpi` | Enables MPI support |
| `timemory::timemory-mpip-library` | Provides MPIP library for MPI performance analysis |
| `timemory::timemory-no-mpi-init` | Disables the generation of MPI_Init and MPI_Init_thread symbols |
| `timemory::timemory-null-sanitizer` | Adds compiler flags to enable null sanitizer (-fsanitizer=null) |
| `timemory::timemory-nvtx` | Enables CUDA NVTX support |
| `timemory::timemory-ompt-library` | Provides OMPT library for OpenMP performance analysis |
| `timemory::timemory-ompt` | Enables OpenMP-tools support |
| `timemory::timemory-papi-static` | Enables PAPI support + links to static library |
| `timemory::timemory-papi` | Enables PAPI support |
| `timemory::timemory-plotting` | Enables python plotting support (system call) |
| `timemory::timemory-protected-visibility` | Adds -fvisibility=protected compiler flag |
| `timemory::timemory-python` | Enables python support (embedded interpreter) |
| `timemory::timemory-roofline-options` | Compiler flags for roofline generation |
| `timemory::timemory-roofline` | Enables flags and libraries for proper roofline generation |
| `timemory::timemory-sanitizer` | Adds compiler flags to enable leak sanitizer (-fsanitizer=leak) |
| `timemory::timemory-statistics` | Enables statistics for all components which define TIMEMORY_STATISTICS_TYPE(...) |
| `timemory::timemory-tau` | Enables TAU support |
| `timemory::timemory-tcmalloc-minimal` | Enables gperftools tcmalloc_minimal library |
| `timemory::timemory-thread-sanitizer` | Adds compiler flags to enable thread sanitizer (-fsanitizer=thread) |
| `timemory::timemory-threading` | Enables multithreading support |
| `timemory::timemory-undefined-sanitizer` | Adds compiler flags to enable undefined sanitizer (-fsanitizer=undefined) |
| `timemory::timemory-unreachable-sanitizer` | Adds compiler flags to enable unreachable sanitizer (-fsanitizer=unreachable) |
| `timemory::timemory-upcxx` | Enables UPC++ support |
| `timemory::timemory-vector` | Adds pre-processor definition of the max vectorization width in bytes |
| `timemory::timemory-vtune` | Enables VTune support (ittnotify) |
| `timemory::timemory-xray` | Adds compiler flags to enable xray-instrumentation (Clang only) |

### `find_package` Approach with COMPONENTS

These libraries can be included in a downstream project via the `COMPONENTS` or `OPTIONAL_COMPONENTS` arguments
to the CMake `find_package` command. When the `COMPONENTS` option is used, the default interface target will
be named `timemory`. Alternatively, one can set the `timemory_FIND_COMPONENTS_INTERFACE` variable to define
a custom interface library name.

When targets are listed after the `COMPONENTS` arguments to `find_package`,
the `timemory-` prefix can be omitted. Additionally, the link type (`shared` or `static`) and
languages suffixes (`c`, `cxx`, `cuda`) can be listed once and dropped from subsequent items in the list.

timemory will bundle the targets specified after `COMPONENTS` into one interface library.

```cmake
# create interface target w/ the components
find_package(timemory REQUIRED COMPONENTS cxx shared compile-options)

# create some library
add_library(foo SHARED foo.cpp)

# import all the compiler defs, flags, linked libs, include paths, etc. from above components
target_link_library(foo timemory)

# override the name of INTERFACE library w/ the components
set(timemory_FIND_COMPONENTS_INTERFACE timemory-cuda-extern)

# creates interface library target: timemory-cuda-extern
find_package(timemory REQUIRED COMPONENTS cxx static compile-options cuda cupti)

# create anoter library
add_library(bar STATIC bar.cpp)

# import all the compiler defs, flags, linked libs, include paths, etc. from above components
target_link_library(foo timemory-cuda-extern)
```

```cmake
find_package(timemory REQUIRED COMPONENTS headers cxx-shared)

add_executable(foo foo.cpp)
target_link_libraries(foo PRIVATE timemory)
```

```cmake
set(timemory_FIND_COMPONENTS_INTERFACE timemory::foo-interface)
find_package(timemory REQUIRED COMPONENTS headers OPTIONAL_COMPONENTS arch papi cuda cupti)

add_executable(foo foo.cpp)
target_link_libraries(foo PRIVATE timemory::foo-interface)
```

## Using Makefiles

Timemory generates a `Makefile.timemory.inc` during installation. This file is intended for
projects which rely on Makefiles. In general, each of the above CMake targets
generates `LIBS`, `DEFS`, `INCLUDE`, `CFLAGS`, `CPPFLAGS`, `CUFLAGS`, and `DEPENDS` variables, e.g.
`timemory::timemory-cxx-shared` generates `TIMEMORY_CXX_SHARED_LIBS`, `TIMEMORY_CXX_SHARED_DEFS`,
`TIMEMORY_CXX_SHARED_INCLUDE`, `TIMEMORY_CXX_SHARED_CFLAGS`, `TIMEMORY_CXX_SHARED_CPPFLAGS`.
`TIMEMORY_CXX_SHARED_CUFLAGS`, and `TIMEMORY_CXX_SHARED_DEPENDS`. The `*_DEPENDS` is a list
of the targets which the library depends on.

| Variable     | Description                    |
| ------------ | ------------------------------ |
| `*_DEFS`     | Pre-processor definition flags |
| `*_INCLUDE`  | Include flags                  |
| `*_LIBS`     | Linker flags                   |
| `*_CFLAGS`   | C compiler flags               |
| `*_CPPFLAGS` | C++ compiler flags             |
| `*_CUFLAGS`  | CUDA compiler flags            |
| `*_DEPENDS`  | Internal target dependencies   |

```console
g++ $(TIMEMORY_PAPI_DEFS) $(TIMEMORY_PAPI_INCLUDE) $(TIMEMORY_PAPI_CPPFLAGS) foo.cpp -o foo $(TIMEMORY_PAPI_LIBS)
```
