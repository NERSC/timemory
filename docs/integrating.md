# Integrating TiMemory into a Project

**It is highly recommended to use CMake with TiMemory**

TiMemory uses modern CMake INTERFACE targets to include the components you want without
forcing you to include everything -- this means that compiler flags, preprocessor
definitions, include paths, link options, and link libraries are bundled into separate
"library" targets that only need to be "linked" to in CMake.

## CMake

### Available Targets

These are the full target names available within CMake. These targets are always provided but
may provide an empty target if the underlying specifications (such as a library and include path)
were not available when TiMemory was installed.

| Target name              | Description                                 | Type           |
| ------------------------ | ------------------------------------------- | -------------- |
| timemory-headers         | Provides include path for timemory          | interface      |
| timemory-cereal          | Provides include path for cereal            | interface      |
| timemory-c-shared        | C library                                   | shared library |
| timemory-c-static        | C library                                   | static library |
| timemory-cxx-shared      | C++ library                                 | shared library |
| timemory-cxx-static      | C++ library                                 | static library |
| timemory-extensions      | All non-essential extensions                | interface      |
| timemory-mpi             | MPI includes and libraries                  | interface      |
| timemory-papi            | PAPI includes and libraries                 | interface      |
| timemory-cuda            | CUDA includes and libraries                 | interface      |
| timemory-cudart          | CUDA runtime library                        | interface      |
| timemory-cudart-device   | CUDA device runtime library                 | interface      |
| timemory-cudart-static   | CUDA static runtime library                 | interface      |
| timemory-cupti           | CUPTI includes and libraries                | interface      |
| timemory-gperf           | gperftools includes and libraries           | interface      |
| timemory-coverage        | Code coverage flags                         | interface      |
| timemory-sanitizer       | Sanitizer flags                             | interface      |
| timemory-compile-options | Compiler flags recommended/used by TiMemory | interface      |
| timemory-arch            | Architecture-specific flags                 | interface      |
| timemory-vector          | Vectorization flags                         | interface      |
| timemory-extern-temples  | Import extern templates                     | interface      |

The following table provides the relevant translation for projects that use Makefiles.

| CMake Target            | Relevant Makefile Translations                                     |
| ----------------------- | ------------------------------------------------------------------ |
| timemory-headers        | `-I<prefix>/include`                                               |
| timemory-cereal         | `-I<prefix>/include`                                               |
| timemory-c-shared       | `-shared -lctimemory`                                              |
| timemory-c-static       | `-static -lctimemory`                                              |
| timemory-cxx-shared     | `-shared -ltimemory`                                               |
| timemory-cxx-static     | `-static -ltimemory`                                               |
| timemory-mpi            | `-DTIMEMORY_USE_MPI` + MPI flags                                   |
| timemory-papi           | `-DTIMEMORY_USE_PAPI` + PAPI include + PAPI libs                   |
| timemory-cuda           | `-DTIMEMORY_USE_CUDA` + CUDA include + CUDA libs + CUDA arch flags |
| timemory-cupti          | `-DTIMEMORY_USE_CUPTI` + CUPTI include + CUPTI libs                |
| timemory-gperf          | `-DTIMEMORY_USE_GPERF` + gperf include + gperf libs                |
| timemory-extern-temples | `-DTIMEMORY_USE_EXTERN_TEMPLATES`                                  |
| timemory-cudart         | `-lcudart`                                                         |
| timemory-cudart-device  | `-lcudadevrt`                                                      |
| timemory-cudart-static  | `-lcudart_static`                                                  |
| timemory-coverage       | `--coverage`                                                       |
| timemory-sanitizer      | `-fsanitizer=<type>`                                               |

### General Approach

```cmake
add_library(foo SHARED foo.cpp)

# this adds the timemory include path
target_link_library(foo timemory-headers)

# this sets foo.cpp to be compiled with the C++ compiler flags timemory was compiled with
target_link_library(foo timemory-cxx-compile-flags)

# this sets the TIMEMORY_USE_PAPI pre-processor definition, adds PAPI include path, and
# links papi libraries
target_link_library(foo timemory-papi)
```

### COMPONENTS Approach

When targets are listed after the `COMPONENTS` arguments to `find_package`,
the `timemory-` prefix can be omitted. Additionally, the link type (`shared` or `static`) and
languages suffixes (`c`, `cxx`, `cuda`) can be listed once and dropped from subsequent items in the list.

TiMemory will bundle the targets specified after `COMPONENTS` into one interface library.

```cmake
# create interface target w/ the components
find_package(TiMemory REQUIRED COMPONENTS cxx shared compile-options extensions)

# create some library
add_library(foo SHARED foo.cpp)

# import all the compiler defs, flags, linked libs, include paths, etc. from above components
target_link_library(foo timemory)

# override the name of INTERFACE library w/ the components
set(TiMemory_FIND_COMPONENTS_INTERFACE timemory-cuda-extern)

# creates interface library target: timemory-cuda-extern
find_package(TiMemory REQUIRED COMPONENTS cxx static compile-options extensions
    cuda cupti extern-templates)

# create anoter library
add_library(bar STATIC bar.cpp)

# import all the compiler defs, flags, linked libs, include paths, etc. from above components
target_link_library(foo timemory-cuda-extern)
```

## Optional TiMemory Usage

If you want to make TiMemory optional at compile time, it is recommended to create a pre-processor definition (e.g. `USE_TIMEMORY`)
and a header with (at least) the following contents:

### Header file

Reference: `examples/ex-optional/test_optional.hpp`

```cpp
#pragma once

#if defined(USE_TIMEMORY)

#    include <timemory/timemory.hpp>

#else

#    include <string>
#    define TIMEMORY_AUTO_TUPLE(...)
#    define TIMEMORY_BASIC_AUTO_TUPLE(...)
#    define TIMEMORY_BLANK_AUTO_TUPLE(...)
#    define TIMEMORY_AUTO_TUPLE_CALIPER(...)
#    define TIMEMORY_BASIC_AUTO_TUPLE_CALIPER(...)
#    define TIMEMORY_BLANK_AUTO_TUPLE_CALIPER(...)
#    define TIMEMORY_CALIPER_APPLY(...)

namespace tim
{
void print_env() {}
void timemory_init(int, char**, const std::string& = "", const std::string& = "") {}
void timemory_init(const std::string&, const std::string& = "", const std::string& = "")
{}
}

#endif
```

