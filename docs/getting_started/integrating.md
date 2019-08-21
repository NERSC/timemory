# Integrating TiMemory into a Project

## TiMemory as a Submodule

TiMemory has a permissive MIT license and can be directly included within
another project. C++ projects can take advantage of the header-only feature
of TiMemory and simply include the folders `source/timemory` and `source/cereal`.

## Using CMake

> **It is highly recommended to use TiMemory with CMake**

TiMemory uses modern CMake INTERFACE targets to include the components you want without
forcing you to include everything -- this means that compiler flags, preprocessor
definitions, include paths, link options, and link libraries are bundled into separate
"library" targets that only need to be "linked" to in CMake.

### Available CMake Targets

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

### Simple `find_package` Approach

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

### `find_package` Approach with COMPONENTS

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

## Using Makefiles

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

## Optional TiMemory Usage

If you want to make TiMemory optional (i.e. a soft dependency), this can be easily done with a _very_
minimal amount of code intrusion and without numerous `#ifdef` blocks. If you include timemory in
your source tree, this can be accomplished by defining `DISABLE_TIMEMORY` during compilation or
before `#include <timemory/timemory.hpp>` (C++ language) or `#include <timemory/ctimemory.h>` (C language).

In general, there are two steps and an optional third if you want to use the preload interface.

1. Create a pre-processor definition, such as `USE_TIMEMORY`, that will allow you to provide some empty
   definitions when TiMemory is disabled
2. Provide the empty definitions in a header file with a relevant name, such as `performance.hpp`
   with (at least) the contents described [here](#header-file)
3. (Optional) Declare the extern "C" interface functions in the header file

### Header file

#### Language: C++

> Reference: `examples/ex-optional/test_optional.hpp`

```cpp
#pragma once

#if defined(USE_TIMEMORY)

#    include <timemory/timemory.hpp>

#else

#    include <ostream>
#    include <string>

namespace tim
{
void                              print_env() {}
template <typename... _Args> void timemory_init(_Args...) {}
void                              timemory_finalize() {}

/// this provides "functionality" for *_INSTANCE macros
/// and can be omitted if these macros are not utilized
struct dummy
{
    template <typename... _Args> dummy(_Args&&...) {}
    ~dummy()            = default;
    dummy(const dummy&) = default;
    dummy(dummy&&)      = default;
    dummy& operator=(const dummy&) = default;
    dummy& operator=(dummy&&) = default;

    void                              start() {}
    void                              stop() {}
    void                              conditional_start() {}
    void                              conditional_stop() {}
    void                              report_at_exit(bool) {}
    template <typename... _Args> void mark_begin(_Args&&...) {}
    template <typename... _Args> void mark_end(_Args&&...) {}
    friend std::ostream& operator<<(std::ostream& os, const dummy&) { return os; }
};
}  // namespace tim

// creates a label
#    define TIMEMORY_BASIC_LABEL(...) std::string("")
#    define TIMEMORY_LABEL(...) std::string("")

// define an object
#    define TIMEMORY_BLANK_OBJECT(...)
#    define TIMEMORY_BASIC_OBJECT(...)
#    define TIMEMORY_OBJECT(...)

// define an object with a caliper reference
#    define TIMEMORY_BLANK_CALIPER(...)
#    define TIMEMORY_BASIC_CALIPER(...)
#    define TIMEMORY_CALIPER(...)
#    define TIMEMORY_CALIPER_APPLY(...)
#    define TIMEMORY_CALIPER_TYPE_APPLY(...)

// define an object
#    define TIMEMORY_BLANK_INSTANCE(...) tim::dummy()
#    define TIMEMORY_BASIC_INSTANCE(...) tim::dummy()
#    define TIMEMORY_INSTANCE(...) tim::dummy()

// debug only
#    define TIMEMORY_DEBUG_BASIC_OBJECT(...)
#    define TIMEMORY_DEBUG_OBJECT(...)
#    define TIMEMORY_DEBUG_BASIC_OBJECT(...)
#    define TIMEMORY_DEBUG_OBJECT(...)

// for CUDA
#    define TIMEMORY_CALIPER_MARK_STREAM_BEGIN(...)
#    define TIMEMORY_CALIPER_MARK_STREAM_END(...)

#endif
```

#### Language: C

```c
#pragma once

#if defined(USE_TIMEMORY)

#    include <timemory/ctimemory.h>

#else

#    define TIMEMORY_SETTINGS_INIT { }
#    define TIMEMORY_INIT(...)

// legacy
#    define TIMEMORY_BLANK_AUTO_TIMER(...) NULL
#    define TIMEMORY_BASIC_AUTO_TIMER(...) NULL
#    define TIMEMORY_AUTO_TIMER(...) NULL
#    define FREE_TIMEMORY_AUTO_TIMER(...)

// modern
#    define TIMEMORY_BASIC_OBJECT(...) NULL
#    define TIMEMORY_BLANK_OBJECT(...) NULL
#    define TIMEMORY_OBJECT(...) NULL
#    define FREE_TIMEMORY_OBJECT(...)

#endif

```

## TiMemory Preload Interface

### Preload Declaration

Add the following declarations to your project in a header file, such as a `performance.hpp` header file:

```cpp
extern "C"
{
    void timemory_init_library(int argc, char** argv);
    void timemory_finalize_library();
    void timemory_begin_record(const char* name, uint64_t* kernid);
    void timemory_end_record(uint64_t kernid);
}
```

### Preload Implementation

Create a separate file, such as `performance.cpp`, to your project with the following contents:

```cpp
#include <cstdint>
#include "commonDefines.h"

extern "C"
{
void timemory_init_library(int, char**) { }
void timemory_finalize_library() { }
void timemory_begin_record(const char*, uint64_t*) { }
void timemory_end_record(uint64_t) { }
}
```

These functions can be added to your code and unless the `timemory-preload` library is pre-loaded,
the function calls will be empty.

### Build System

#### CMake

Compile `performance.cpp` into a stand-alone shared (dynamic) library and link it to your application:

```cmake
add_library(performance-symbols SHARED performance.cpp)
```

#### Makefile

```shell
libperformance-symbols.so: performance.cpp
        $(CXX) -shared $(CXXFLAGS) -o $@ performance.cpp

clean:
        rm -f *.so
```

### Usage

- Include the [preload declaration header](#preload-declaration)
- Add `timemory_init_library` to the beginning of the application
- Add `timemory_finalize_library` to the end of the application
- Elsewhere is the code, to start recording:
    - Create a `uint64_t` (unsigned 64-bit integer) variable that can be referenced later
    - Pass a string label and the address of that `uint64_t` variable to `timemory_begin_record`
    - `timemory_begin_record` will assign a unique number of the `uint64_t` variable and start
      recording
- When recording should be stopped, pass the `uint64_t` variable to `timemory_end_record`
- Control over which components will be recorded is handled by a comma-separated list of component
  names in the `TIMEMORY_COMPONENTS` environment variable

#### Example

```
#include "performance.hpp"

int main(int argc, char** argv)
{
    timemory_init_library(argc, argv);

    uint64_t id;
    timemory_begin_record("label", &id);


    timemory_end_record(id);

    timemory_finalize_library();
}
```

#### Execution without TiMemory

Invoke the application normally:

```shell
./myexe
```

#### Execution with TiMemory

Linux:

```shell
export TIMEMORY_COMPONENTS="real_clock, cpu_clock, cpu_util, cpu_roofline_dp_flops"
export LD_PRELOAD=/usr/local/lib/libtimemory-preload.so
./myexe
```

macOS:

```shell
export TIMEMORY_COMPONENTS="real_clock, cpu_clock, cpu_util, cpu_roofline_dp_flops"
export DYLD_FORCE_FLAT_NAMESPACE=1
export DYLD_INSERT_LIBRARIES=/usr/local/lib/libtimemory-preload.dylib
./myexe
```
