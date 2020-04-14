# timemory

## Timing + Memory + Hardware Counter Utilities for C / C++ / CUDA / Python

[![Build Status](https://travis-ci.org/NERSC/timemory.svg?branch=master)](https://travis-ci.org/NERSC/timemory)
[![Build status](https://ci.appveyor.com/api/projects/status/8xk72ootwsefi8c1/branch/master?svg=true)](https://ci.appveyor.com/project/jrmadsen/timemory/branch/master)
[![codecov](https://codecov.io/gh/NERSC/timemory/branch/master/graph/badge.svg)](https://codecov.io/gh/NERSC/timemory)

[timemory on GitHub (Source code)](https://github.com/NERSC/timemory)

[timemory General Documentation](https://timemory.readthedocs.io)

[timemory Source Code Documentation (Doxygen)](https://timemory.readthedocs.io/en/latest/doxygen-docs/)

[timemory Testing Dashboard (CDash)](https://cdash.nersc.gov/index.php?project=TiMemory)

|        |                                                   |
| ------ | ------------------------------------------------- |
| GitHub | `git clone https://github.com/NERSC/timemory.git` |
| PyPi   | `pip install timemory`                            |


## Purpose

The goal of timemory is to create an open-source performance measurement and analyis package
which can be used to adapt to any existing C/C++ performance measurement and analysis API and
is arbitrarily extendable by users within their application. In other words, timemory strives
to be a universal adaptor toolkit for performance measurement and analysis.

## Overview

Timemory is designed, first and foremost, to be a portable, modular, and fully customizable toolkit
for performance measurement and analysis of serial and parallel programs written in C, C++, Fortran, Python, and CUDA.

Timemory arose out of the need for a universal
adapator kit for the various APIs provided several existing tools and a straight-forward and intuitive method
for user-defined expression of performance measurements which can easily encapsulated in a generic structure.
Performance measurement components written with timemory are arbitrarily scalable up to any number of threads and
processes and fully support intermixing different measurements at different locations within the program -- this
uniquely enables timemory to be deployed to collect performance data at scale in HPC because highly detailed collection can
occur at specific locations within the program where ubiquitous collection would simulatenously degrade performance
significantly and require a prohibitive amount of memory.

- Timemory is designed as a __*modular and customizable framework*__ to simplify and facilitate the __*creation and implementation of performance measurement and analysis tools*__
    - Within projects, timemory can be used as:
        - A backend to bundle instrumentation and sampling tools together, support serialization to JSON/XML, and provide statistics among other uses.
        - A frontend to invoke other instrumentation and sampling tools
    - Independently, timemory can be used to create command-line tools and libraries which instrument library calls
        - The [timem](source/tools/timem/README.md) executable is an example of using timemory to create an extended version of UNIX `time` command-line tool that includes additional information on memory usage, context switches, and hardware counters.
        - The [timemory-run](source/tools/timemory-run/README.md) executable is an example of using timemory and [Dyninst](https://github.com/dyninst/dyninst) to create a command-line tool capable of runtime instrumentation and binary re-writing.
        - The [kokkos-tools](source/tools/kokkos-tools/README.md) collection is an example of using timemory to create instrumentation for a project-provided API.
        - The [timemory-mpip](source/tools/timemory-mpip/README.md) library is an example of using timemory + [GOTCHA](https://github.com/LLNL/GOTCHA) to wrap ~245 dynamically-linked MPI function calls with a common set of instrumentation which fully supports inspection of the incoming arguments and return values as needed.
    - The [timemory-avail](source/tools/timemory-avail/README.md) tool provides a way to query the available components, settings, and hardware counters for an installation
- The goals of timemory are to provide:
    - __*Common instrumentation framework*__
        - Eliminate need for projects to explicitly support multiple instrumentation frameworks
    - High performance when enabled
    - Low overhead when enabled at compile time but disabled at runtime
    - Zero overhead when disabled at compile time
    - Allow performance measurements to be inter-mixed arbitrarily with zero overhead for the measurement types that are not used in a region, e.g.:
        - Instrument measurements of A, B, and C around arbitrary region 1
        - Instrument measurements of A and C around arbitrary region 1.1 (nested with Section 1)
        - Instrument measurements of C around arbitrary region 2
        - Instrument measurements of D around arbitrary region 3
        - No instrumentation around arbitrary region 4
    - Provide an intuitive and simple API for creating measurement tools which is relatively future-proof
        - Most performance tools which permit user extensions rely on the user populating structs/classes which inform the framework about data-types and features

## Why Use timemory?

- __*Timemory is a universal adaptor framework for performance measurement and analysis*__
    - Arguably the most customizable performance measurement and analysis API available
    - Template meta-programming on the backend allows timemory to adapt to the special needs
      of any existing API
    - Projects with existing instrumentation APIs can wrap their own instrumentation into a component
- Ability to arbitrarily switch and combine different measurement types anywhere in application
    - Intermixed recording different measurements in different parts of the code
    - Intermixed flat profiling, tracing, and timeline profiling
- Provides static reporting (fixed at compile-time), dynamic reporting (selected at run-time), or hybrid
    - Enable static wall-clock and cpu-clock reporting with ability to dynamically enable hardware-counters at runtime

## Timemory is designed to be future-proof by avoiding internally-defined data types

Most performance tools which permit user extensions usually rely on one of two methods:

1. User populating structs/class fields for specifying value types and which features to enable
2. Using dynamic polymorphism to inherit from a base class

The problem is that structuring a tool in either of these fashions is that it necessitates defining
specific function signatures which may require changes as capabilities evolve.
Additionally, with respect to (1), these data types can become quite complex and/or opaque.
With respect to (2), the virtual table can impact performance.
Timemory, in contrast, is designed to query the presence of function _names_ for feature detection and adapts accordingly
to the overloads of that function name and it's return type. This is all possible due to the
template-based design of timemory which makes extensive use of variadic functions to accept any arguments at a high-level and
SFINAE to decide at compile-time which function to invoke (if a function is invoked at all).
For example:

- component A can contain these member functions:
    - `void start()`
    - `int get()`
- component B can contains these member functions:
    - `void start()`
    - `void start(cudaStream_t)`
    - `double get()`
- component C can contain these member functions:
    - `void start()`

And for a given bundle `component_tuple<A, B, C> obj`:

- Invoking `obj.start()` calls the following member functions on instances of A, B, and C:
    - `A::start()`
    - `B::start()`
    - `C::start()`
- Invoking `obj.start(cudaStream_t)` calls the following member functions on instances of A, B, and C:
    - `A::start()`
    - `B::start(cudaStream_t)`
    - `C::start()`
- Invoking `obj.get()`:
    - Returns `std::tuple<int, double>` because it detects the two return types from A and B and the lack of `get()` member function in component C.

### Support for Multiple Instrumentation APIs

- [LIKWID](https://github.com/RRZE-HPC/likwid)
- [Caliper](https://github.com/LLNL/Caliper)
- [TAU](https://www.cs.uoregon.edu/research/tau/home.php)
- [gperftools](https://github.com/gperftools/gperftools)
- MPI
- OpenMP
- CrayPAT
- Allinea-MAP
- PAPI
- ittnotify (Intel Parallel Studio API)
- CUPTI (NVIDIA performance API)
- NVTX (NVIDIA marker API)

### Generic Bundling of Multiple Tools

- CPU and GPU hardware counters via PAPI
- NVIDIA GPU hardware counters via CUPTI
- NVIDIA GPU tracing via CUPTI
- Generating a Roofline for performance-critical sections on the CPU and NVIDIA GPUs
    - Classical Roofline (FLOPs)
    - Instruction Roofline
- Memory usage
- Tool insertion around `malloc`, `calloc`, `free`, `cudaMalloc`, `cudaFree`
- Wall-clock, cpu-clock, system-clock timing
- Number of bytes read/written to file-system (and rate)
- Number of context switches
- Trip counts
- CUDA kernel runtime(s)
- Data value tracking

### Powerful GOTCHA Extensions

- [GOTCHA](https://github.com/LLNL/GOTCHA) is an API wrapping function calls similar to the use of LD_PRELOAD
    - Significantly simplify existing implementations
- Scoped GOTCHA
    - Enables temporary wrapping over regions
- Use gotcha component to replace external function calls with custom replacements
    - E.g. replace the C math function `exp` with custom `exp` implementation
- Use gotcha component to wrap external library calls with custom instrumentation

### Multi-language Support

- Variadic interface to all the utilities from C code
- Variadic interface to all the utilities from C++ code
- Variadic interface to all the utilities from Python code
    - Includes context-managers and decorators

## Overview

Timemory is generic C++11 template library providing a variety of
[performance components](https://timemory.readthedocs.io/en/latest/components/overview/)
for reporting timing, resource usage, hardware counters for the CPU and GPU,
roofline generation, and simplified generation of GOTCHA wrappers to instrument
external library function calls.

Timemory provides also provides Python and C interfaces.

## Profiling and timemory

Timemory includes the [timemory-run](source/tools/timemory-run/README.md) as a full profiler for Linux systems.
This executable supports dynamic instrumentation (instrumenting at the target applicaiton's runtime), attaching
to a running process, and binary re-writing (creating a new instrumented binary). The instrumented applications
support flat-profiling, call-stack profiling, and timeline profiling and can be configured to use any of the
components timemory provides or, with a little work, can also be used to instrument custom components defined by the user.

Timemory was designed, first and foremost, as an API because sometimes it is easier to just instrument the specific
region of code that is being targeted for optimization and performance analysis is always easier when there is a
built-in method within the code around critical performance regions and excludes unnecessary information (and additional
overheads).
The library provides an easy-to-use method for always-on general HPC analysis metrics
(i.e. timing, memory usage, etc.) with the same or less overhead than if these metrics were to
records and stored in a custom solution -- the path through the timemory code between calling `start()` on a
bundle of components to `start()` being called on a component itself is essentially optimized down to a direct call
on the member function.
Functionally, the overhead is non-existant: sampling profilers (e.g. gperftools, VTune)
at standard sampling rates barely notice the presence of timemory unless it is been
used _very_ unwisely.

In general, profilers are not run frequently enough and performance degradation
or memory bloat can go undetected for several commits until a production run crashes or
underperforms. This generally leads to a scramble to detect which revision caused the issue.
Timemory can decrease performance regression identification time and can be used easily create
a built-in performance monitoring system.
When timemory is combined with a continuous integration reporting system,
this scramble can be mitigated fairly quickly because the high-level monitoring system
will allow developers to quickly associate a region and commit with changes in performance.
Once this region + commit have been identified, a full profiler would then be launched for fine-grained analysis.

## Quick Performance Analysis with timemory

Want to check whether those changes increased data locality (i.e. decreased cache misses) but don't care
about any other sections of the code?
Use the following and set `TIMEMORY_PAPI_EVENTS="PAPI_L1_TCM,PAPI_L2_TCM,PAPI_L3_TCM"` in
the environment:

### C / C++ Library Interface

```cpp
timemory_push_components("papi_vector");
timemory_push_region("MY_REGION_OF_INTEREST");
//
// do something in region of interest...
//
timemory_pop_region("MY_REGION_OF_INTEREST");
```

### Fortran

```fortran
call timemory_push_components("papi_vector")
call timemory_push_region("MY_REGION_OF_INTEREST")
!
! do something in region of interest...
!
call timemory_pop_region("MY_REGION_OF_INTEREST")
```

### C++ Template Interface

```cpp
using hwcounters_t = tim::component_tuple<tim::component::papi_vector>;

hwcounters_t roi("MY_REGION_OF_INTEREST");
roi.start();
//
// do something in region of interest...
//
roi.stop();
```

Or encoding the PAPI enumeration types explicitly:

```cpp
using hwcounters_t = tim::component_tuple<tim::component::papi_tuple<PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM>>;

hwcounters_t roi("MY_REGION_OF_INTEREST");
roi.start();
//
// do something in region of interest...
//
roi.stop();
```

### Python Context Manager

```python
from timemory.util import marker

with marker(["papi_vector"], key="MY_REGION_OF_INTEREST"):
    #
    # do something in region of interest...
    #
```

### C Enumeration Interface

```cpp
void* roi = TIMEMORY_BLANK_MARKER("MY_REGION_OF_INTEREST", PAPI_VECTOR);
//
// do something in region of interest...
//
FREE_TIMEMORY_MARKER(roi)
```

and delete it when finished. It's a couple of extra LOC that will reduce time
spent: changing code, then runnning profiler, then opening output in profiler,
then finding ROI, then comparing to previous results, and then repeating.

## Create Your Own Tools/Components

- Written in C++
- Direct access to performance analysis data in Python and C++
- Create your own components: any one-time measurement or start/stop paradigm can be wrapped with timemory
    - Flexible and easily extensible interface: no data type restrictions in custom components

There are numerous instrumentation APIs available but very few provide the ability for _users_ to create
tools/components that will fully integrate with the instrumentation API in their code. The
simplicity of creating a custom component that inherits category-based formatting properties
(`is_timing_category`) and timing unit conversion (`uses_timing_units`)
can be easily demonstrated with the `wall_clock` component and the simplicity and adaptability
of forwarding timemory markers to external instrumentation is easily demonstrated with the
`tau_marker` component:

```cpp
TIMEMORY_DECLARE_COMPONENT(wall_clock)
TIMEMORY_DECLARE_COMPONENT(tau_marker)

// type-traits for wall-clock
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::wall_clock, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::wall_clock, true_type)
TIMEMORY_STATISTICS_TYPE(component::wall_clock, double)

namespace tim
{
namespace component
{
//
// the system's real time (i.e. wall time) clock, expressed as the
// amount of time since the epoch.
//
// NOTE: 'value', 'accum', 'get_units()', etc. are provided by base class
//
struct wall_clock : public base<wall_clock, int64_t>
{
    using ratio_t    = std::nano;
    using value_type = int64_t;
    using base_type  = base<wall_clock, value_type>;

    static std::string label() { return "wall"; }
    static std::string description() { return "wall-clock timer"; }

    static value_type  record()
    {
        // use STL steady_clock to get time-stamp in nanoseconds
        using clock_type    = std::chrono::steady_clock;
        using duration_type = std::chrono::duration<clock_type::rep, ratio_t>;
        return std::chrono::duration_cast<duration_type>(
            clock_type::now().time_since_epoch()).count();
    }

    double get_display() const { return get(); }

    double get() const
    {
        // get_unit() provided by base_clock via uses_timing_units type-trait
        auto val = (is_transient) ? accum : value;
        return static_cast<double>(val) / ratio_t::den * get_unit();
    }

    void start()
    {
        value = record();
    }

    void stop()
    {
        value = (record() - value);
        accum += value;
    }
};

//
// forwards timemory instrumentation to TAU instrumentation.
//
struct tau_marker : public base<tau_marker, void>
{
    // timemory component api
    using value_type = void;
    using this_type  = tau_marker;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "tau"; }
    static std::string description() { return "TAU_start and TAU_stop instrumentation"; }

    static void global_init(storage_type*) { Tau_set_node(dmp::rank()); }
    static void thread_init(storage_type*) { TAU_REGISTER_THREAD();     }

    tau_marker() = default;
    tau_marker(const std::string& _prefix) : m_prefix(_prefix) {}

    void start() { Tau_start(m_prefix.c_str()); }
    void stop()  { Tau_stop(m_prefix.c_str());  }

    void set_prefix(const std::string& _prefix) { m_prefix = _prefix; }
    // This 'set_prefix(...)' member function is a great example of the template
    // meta-programming provided by timemory: at compile-time, timemory checks
    // whether components have this member function and, if and only if it exists,
    // timemory will call this member function for the component and provide the
    // marker label.

private:
    std::string m_prefix = "";
};

}  // namespace component
}  // namespace tim
```

Using the two tools together in C++ is as easy as the following:

```cpp
#include <timemory/timemory.hpp>

using namespace tim::component;
using comp_bundle_t = tim::component_tuple_t <wall_clock, tau_marker>;
using auto_bundle_t = tim::auto_tuple_t      <wall_clock, tau_marker>;
// "auto" types automatically start/stop based on scope

void foo()
{
    comp_bundle_t t("foo");
    t.start();
    // do something
    t.stop();
}

void bar()
{
    auto_bundle_t t("foo");
    // do something
}

int main(int argc, char** argv)
{
    tim::init(argc, argv);
    foo();
    bar();
    tim::finalize();
}
```

Using the pure template interface will cause longer compile-times and is only available in C++
so a library interface for C, C++, and Fortran is also available:

```cpp
#include <timemory/library.h>

void foo()
{
    uint64_t idx = timemory_get_begin_record("foo");
    // do something
    timemory_end_record(idx);
}

void bar()
{
    timemory_push_region("bar");
    // do something
    timemory_pop_region("bar");
}

int main(int argc, char** argv)
{
    timemory_init_library(argc, argv);
    timemory_push_components("wall_clock,tau_marker");
    foo();
    bar();
    timemory_pop_components();
    timemory_finalize_library();
}
```

In Python:

```python
import timemory
from timemory.profiler import profile
from timemory.util import auto_tuple

def get_config(items=["wall_clock", "tau_marker"]):
    """
    Converts strings to enumerations
    """
    return [getattr(timemory.component, x) for x in items]

@profile(["wall_clock", "tau_marker"])
def foo():
    """
    @profile (also available as context-manager) enables full python instrumentation
    of every subsequent python call
    """
    # ...

@auto_tuple(get_config())
def bar():
    """
    @auto_tuple (also available as context-manager) enables instrumentation
    of only this function
    """
    # ...

if __name__ == "__main__":
    foo()
    bar()
    timemory.finalize()
```

## GOTCHA and timemory

C++ codes running on the Linux operating system can take advantage of the built-in
[GOTCHA](https://github.com/LLNL/GOTCHA) functionality to insert timemory markers around external function calls.
[GOTCHA](https://github.com/LLNL/GOTCHA) is similar to `LD_PRELOAD` but operates via a programmable API.
This include limited support for C++ function mangling (in general, mangling template functions are not supported -- yet).

Writing a GOTCHA hook in timemory is greatly simplified and applications using timemory can specify their own GOTCHA hooks
in a few lines of code instead of being restricted to a pre-defined set of GOTCHA hooks.

### Example GOTCHA

If an application wanted to insert `tim::auto_timer` around (unmangled) `MPI_Allreduce` and
(mangled) `ext::do_work` in the following executable:

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

    int64_t nitr = 10;
    std::pair<float, double> settings{ 1.25f, 2.5 };
    std::tuple<float, double> result = ext::do_work(nitr, settings);
    // ... etc.

    return 0;
}
```

This would be the required specification using the `TIMEMORY_C_GOTCHA` macro for unmangled functions
and `TIMEMORY_CXX_GOTCHA` macro for mangled functions:

```cpp
#include <timemory/timemory.hpp>

static constexpr size_t NUM_FUNCS = 2;
using gotcha_t = tim::component::gotcha<NUM_FUNCS, tim::auto_timer_t>;

void init()
{
    TIMEMORY_C_GOTCHA(gotcha_t, 0, MPI_Allreduce);
    TIMEMORY_CXX_GOTCHA(gotcha_t, 1, ext::do_work);
}
```

## Additional Information

For more information, refer to the [documentation](https://timemory.readthedocs.io/en/latest/).
