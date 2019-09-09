# TiMemory

## Timing + Memory + Hardware Counter Utilities for C / C++ / CUDA / Python

[![Build Status](https://travis-ci.org/NERSC/timemory.svg?branch=master)](https://travis-ci.org/NERSC/timemory)
[![Build status](https://ci.appveyor.com/api/projects/status/8xk72ootwsefi8c1/branch/master?svg=true)](https://ci.appveyor.com/project/jrmadsen/timemory/branch/master)
[![codecov](https://codecov.io/gh/NERSC/timemory/branch/master/graph/badge.svg)](https://codecov.io/gh/NERSC/timemory)

[![Conda Recipe](https://img.shields.io/badge/recipe-timemory-green.svg)](https://anaconda.org/jrmadsen/timemory)
[![Anaconda-Server Badge](https://anaconda.org/jrmadsen/timemory/badges/version.svg)](https://anaconda.org/jrmadsen/timemory)
[![Anaconda-Server Badge](https://anaconda.org/jrmadsen/timemory/badges/platforms.svg)](https://anaconda.org/jrmadsen/timemory)
[![Anaconda-Server Badge](https://anaconda.org/jrmadsen/timemory/badges/downloads.svg)](https://anaconda.org/jrmadsen/timemory)

[TiMemory on GitHub (Source code)](https://github.com/NERSC/timemory)

[TiMemory General Documentation](https://timemory.readthedocs.io/en/latest/)

[TiMemory Source Code Documentation (Doxygen)](https://timemory.readthedocs.io/en/latest/doxygen-docs/)

[TiMemory Testing Dashboard (CDash)](https://cdash.nersc.gov/index.php?project=TiMemory)

|                |                                                   |
| -------------- | ------------------------------------------------- |
| GitHub         | `git clone https://github.com/NERSC/timemory.git` |
| PyPi           | `pip install timemory`                            |
| Anaconda Cloud | `conda install -c jrmadsen timemory`              |

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

## Overview

TiMemory is generic C++11 template library providing a variety of
[performance components](https://timemory.readthedocs.io/en/latest/components/)
for reporting timing, resource usage, hardware counters for the CPU and GPU,
roofline generation, and simplified generation of GOTCHA wrappers to instrument
external library function calls.

TiMemory provides also provides Python and C interfaces.

## Purpose

The goal of the package is to provide as easy way to regularly report on the performance
of your code. If you have ever added something like this in your code:

```python
tstart = time.now()
# do something
tstop = time.now()
print("Elapsed time: {}".format(tstop - tstart))
```

Timemory streamlines this work. In C++ codes, all you have to do is include the headers.
It comes in handy especially when optimizing a
certain algorithm or section of your code -- you just insert a line of code that specifies what
you want to measure and run your code: __*initialization and output are automated*__.

## Profiling and timemory

Timemory is not a full profiler and is intended to supplement profilers, not be used in lieu of profiling,
which are important for _discovering where to place timemory markers_.
The library provides an easy-to-use method for __*always-on general HPC analysis metrics*__
(i.e. timing, memory usage, etc.) with the same or less overhead than if these metrics were to
records and stored in a custom solution (there is zero polymorphism) and, for C++ code, extensively
inlined.
__*Functionally, the overhead is non-existant*__: sampling profilers (e.g. gperftools, VTune)
at standard sampling rates barely notice the presence of TiMemory unless it is been
used _very_ unwisely.

Additional tools are provided, such as hardware counters, to __*increase optimization productivity.*__
What to check whether those changes increased data locality (i.e. decreased cache misses) but don't care about any other sections of the code?
Use the following and set `TIMEMORY_PAPI_EVENTS="PAPI_L1_TCM,PAPI_L2_TCM,PAPI_L3_TCM"` in
the environment:

```cpp
using auto_tuple_t = tim::auto_tuple<tim::component::papi_array_t>;
TIMEMORY_AUTO_TUPLE_CALIPER(roi, auto_tuple_t, "");
//
// do something in region of interest...
//
TIMEMORY_CALIPER_APPLY(roi, stop);
```

and delete it when finished. It's three extra LOC that may reduce the time
spent: changing code, then runnning profiler, then opening output in profiler,
then finding ROI, then comparing to previous results, and then repeating from
4 hours to 1.

In general, profilers are not run frequently enough and performance degradation
or memory bloat can go undetected for several commits until a production run crashes or
underperforms. This generally leads to a scramble to detect which revision caused the issue.
Here, TiMemory can __*decrease performance regression identification time.*__
When TiMemory is combined with a continuous integration reporting system,
this scramble can be mitigated fairly quickly because the high-level reporting
provided allows one to associate a region and commit with exact performance numbers.
Once TiMemory has been used to help identify the offending commit and identify the general
region in the offending code, a full profiler should be launched for the fine-grained diagnosis.

## Create Your Own Tools/Components

There are numerous instrumentation APIs available but very few provide the ability for _users_ to create
tools/components that will fully integrate with the instrumentation API in their code. The
simplicity of creating a custom component can be easily demonstrated in ~30 LOC with the
`trip_count` component:

```cpp
namespace tim {
namespace component {

struct trip_count : public base<trip_count, int64_t>
{
    using value_type = int64_t;
    using this_type  = trip_count;
    using base_type  = base<this_type, value_type>;

    static const short                   precision = 0;
    static const short                   width     = 5;
    static const std::ios_base::fmtflags format_flags =
        std::ios_base::fixed | std::ios_base::dec | std::ios_base::showpoint;

    static int64_t     unit() { return 1; }
    static std::string label() { return "trip_count"; }
    static std::string description() { return "trip counts"; }
    static std::string display_unit() { return ""; }
    static value_type  record() { return 1; }

    value_type get_display() const { return accum; }
    value_type get() const { return accum; }

    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        accum += value;
        set_stopped();
    }
};
}  // namespace component
}  // namespace tim
```

## [GOTCHA](https://github.com/LLNL/GOTCHA) and timemory

C++ codes running on the Linux operating system can take advantage of the built-in
[GOTCHA](https://github.com/LLNL/GOTCHA) functionality to insert timemory markers __*around external function calls*__.
[GOTCHA](https://github.com/LLNL/GOTCHA) is similar to `LD_PRELOAD` but operates via a programmable API.

Writing a GOTCHA hook in timemory is greatly simplified and, unlike other instrumentation APIs that may provide
a static set of GOTCHA hooks, applications using timemory can specify their own GOTCHA hooks in a few lines
of code.

### Example GOTCHA for `MPI_Allreduce`

If an application wanted to insert `tim::auto_timer` around `MPI_Allreduce` in the following executable:

```cpp
#include <mpi.h>
#include <vector>

int main(int argc, char** argv)
{
    init();

    MPI_Init(&argc, &argv);

    int sizebuf = 100;
    std::vector<double> sendbuf(sizebuf, 1.0);
    // ... do some stuff
    std::vector<double> recvbuf(sizebuf, 0.0);

    MPI_Allreduce(sendbuf.data(), recvbuf.data(), sizebuf, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // ... etc.

    return 0;
}
```

This would be the required specification using the `TIMEMORY_GOTCHA` macro:

```cpp
#include <timemory/timemory.hpp>

#define NUM_FUNCS 1
using mpi_gotcha_t = tim::component::gotcha<NUM_FUNCS, tim::auto_timer_t>;

void init()
{
    TIMEMORY_GOTCHA(mpi_gotcha_t, 0, MPI_Allreduce);
}
```

## Additional Information

For more information, refer to the [documentation](https://timemory.readthedocs.io/en/latest/).
