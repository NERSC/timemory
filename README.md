# timemory

## Timing + Memory + Hardware Counter Utilities for C / C++ / CUDA / Python

[![Build Status](https://travis-ci.org/NERSC/timemory.svg?branch=master)](https://travis-ci.org/NERSC/timemory)
[![Build status](https://ci.appveyor.com/api/projects/status/8xk72ootwsefi8c1/branch/master?svg=true)](https://ci.appveyor.com/project/jrmadsen/timemory/branch/master)
[![codecov](https://codecov.io/gh/NERSC/timemory/branch/master/graph/badge.svg)](https://codecov.io/gh/NERSC/timemory)

[timemory on GitHub (Source code)](https://github.com/NERSC/timemory)

[timemory General Documentation (ReadTheDocs)](https://timemory.readthedocs.io)

[timemory Source Code Documentation (Doxygen)](https://timemory.readthedocs.io/en/latest/doxygen-docs/)

[timemory Testing Dashboard (CDash)](https://cdash.nersc.gov/index.php?project=TiMemory)

[timemory Tutorials](https://github.com/NERSC/timemory-tutorials)

|        |                                                   |
| ------ | ------------------------------------------------- |
| GitHub | `git clone https://github.com/NERSC/timemory.git` |
| PyPi   | `pip install timemory`                            |
| Spack  | `spack install timemory`                          |

## Purpose

The goal of timemory is to create an open-source performance measurement and analyis package
with modular and reusable components which can be used to adapt to any existing C/C++
performance measurement and analysis API and is arbitrarily extendable by users within their
application.
Timemory is not just another profiling tool, it is a profling _toolkit_ which streamlines building custom
profiling tools through modularity and then utilizes the toolkit to provides several pre-built tools.

## Versioning

Timemory originated as a very simple tool for recording timing and memory measurements (hence the name) in C, C++, and Python and only supported
three modes prior to the 3.0.0 release: a fixed set of timers, a pair of memory measurements, and the combination of the two.
__Prior to the 3.0.0 release, timemory was almost completely rewritten from scratch__ with the sole exceptions of some C/C++ macro, e.g.
`TIMEMORY_AUTO_TIMER`, and some Python decorators and context-manager, e.g. `timemory.util.auto_timer`, whose behavior were
able to be fully replicated in the new release. Thus, while it may appear that timemory is a mature project at v3.0+, it
is essentially still in it's first major release.

## Documentation

The full documentation is available at [timemory.readthedocs.io](https://timemory.readthedocs.io).
Detailed source documentation is provided in the [doygen](https://timemory.readthedocs.io/en/latest/doxygen-docs/)
section of the full documentation.
Tutorials are available in the [github.com/NERSC/timemory-tutorials](https://github.com/NERSC/timemory-tutorials).

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

Timemory can be used as a backend to bundle instrumentation and sampling tools together, support serialization to JSON/XML, and provide statistics among other uses. It can also be utilized as a front-end to invoke
custom instrumentation and sampling tools. Timemory uses the abstract term "component" for a structure
which encapsulates some performance analysis operation. The structure might encapsulate function
calls to another tool, record timestamps for timing, log values provided by the application,
provide a operator for replacing a function in the code dynamically, audit the incoming arguments
and/or outgoing return value from function, or just provide stubs which can be overloaded by the linker.

## Features

- C++ Template API
    - Modular and fully-customizable
    - Adheres to C++ standard template library paradigm of "you don't pay for what you don't use"
    - Simplifies and facilitates creation and implementation of performance measurement tools
        - Create your own instrumentation profiler
        - Create your own instrumentation library
        - Create your own sampling profiler
        - Create your own sampling library
        - Create your own execution wrappers
        - Supplement timemory-provided tools with your own custom component(s)
        - Thread-safe data aggregation
        - Aggregate collection over multiple processes (MPI and UPC++ support)
        - Serialization to text, JSON, XML
    - Components are composable with other components
    - Variadic component bundlers which maintain complete type-safety
        - Components can be bundled together into a single handle without abstractions
    - Components can store data in any valid C++ data type
    - Components can return data in any valid C++ data type
- C / C++ / CUDA / Fortran Library API
    - Straight-forward collection of functions and macros for creating built-in performance analysis to your code
    - Component collection can be arbitrarily inter-mixed
        - E.g. collect "A" and "B" in one region, "A" and "C" in another region
    - Component collection can be dynamically manipulated at runtime
        - E.g. add/remove "A" at any point, on any thread, on any process
- Python API
    - Decorators and context-managers for functions or regions in code
    - Python function profiling
    - Python line-by-line profiling
    - Every component in `timemory-avail` is provided as a stand-alone Python class
        - Provide low-overhead measurements for building your own Python profiling tools
- Command-line Tools
    - [timemory-avail](source/tools/timemory-avail/README.md)
        - Provides available components, settings, and hardware counters
        - Quick API reference tool
    - [timem](source/tools/timem/README.md) (UNIX)
        - Extended version of UNIX `time` command-line tool that includes additional information on memory usage, context switches, and hardware counters
        - Support collecting hardware counters (Linux-only, requires PAPI)
    - [timemory-run](source/tools/timemory-run/README.md) (Linux)
        - Dynamic instrumentation profiling tool
        - Supports runtime instrumentation and binary re-writing
    - `timemory-python-profiler`
        - Python function profiler
    - `timemory-python-line-profiler`
        - Python line-by-line profiler
        - Design based on [line-profiler](https://pypi.org/project/line-profiler/) package
        - Extended to use components: cpu-clock, memory-usage, context-switches, etc. (all components which collect scalar values)
- Instrumentation Libraries
    - [kokkos-connector](source/tools/kokkos-connector/README.md): Kokkos Profiling Libraries
    - [timemory-mpip](source/tools/timemory-mpip/README.md): MPI Profiling Library (Linux-only)
    - [timemory-ncclp](source/tools/timemory-ncclp/README.md): NCCL Profiling Library (Linux-only)
    - [timemory-ompt](source/tools/timemory-ompt/README.md): OpenMP Profiling Library

## Design Goals

- __*Toolkit*__ for creating new performance analysis tools
- __*Common instrumentation framework*__
    - Eliminate need for projects to explicitly support multiple instrumentation frameworks
- __*High performance*__ during data collection
- __*Low overhead*__ when dormant (disabled at runtime)
- Zero overhead when disabled at compile time
- Support arbitrarily intermixing components:
    - Instrument measurements of A, B, and C around arbitrary region 1
    - Instrument measurements of A and C around arbitrary region 1.1 (nested with Section 1)
    - Instrument measurements of C around arbitrary region 2
    - Instrument measurements of D around arbitrary region 3
    - No instrumentation around arbitrary region 4
- Intuitive and simple API to use and extend

## Component Basics

Timemory components are C++ structs (class which defaults to `public` instead of `private`) which
define a single collection instance, e.g. the `wall_clock` component is written as a simple class
with two 64-bit integers with `start()` and `stop()` member functions.

```cpp
// This "component" is for conceptual demonstration only
// It is not intended to be copy+pasted
struct wall_clock
{
    int64_t m_value = 0;
    int64_t m_accum = 0;

    void start();
    void stop();
};
```

The `start()` member function which records a timestamp
and assigns it to one of the integers temporarily, the `stop()` member function
which records another timestamp, computes the difference and then assigns the difference
to the first integer and adds the difference to the second integer.

```cpp
void wall_clock::start()
{
    m_value = get_timestamp();
}

void wall_clock::stop()
{
    // compute difference b/t when start and stop were called
    m_value = (get_timestamp() - m_value);
    // accumulate the difference
    m_accum += m_value;
}
```

Thus, after `start()` and `stop()` is invoked twice on the object:

```cpp
wall_clock foo;

foo.start();
sleep(1); // sleep for 1 second
foo.stop();

foo.start();
sleep(1); // sleep for 1 second
foo.stop();
```

The first integer (`m_value`) represents the _most recent_ timing interval of 1 second
and the second integer (`m_accum`) represents the _accumulated_ timing interval totaling 2 seconds.
This design not only encapsulates how to take the measurement, but also provides it's own
data storage model. With this design, timemory measurements naturally support asynchronous
data collection. Additionally, as part of the design for generating the call-graph,
call-graphs are accumulated locally on each thread and on each process and merged at
the termination of the thread or process. This allows parallel data to be collection
free from synchronization overheads. On the worker threads, there is a concept of being
at "sea-level" -- the call-graphs relative position based on the base-line of the
primary thread in the application. When a worker thread is at sea-level, it reads the
position of the call-graph on the primary thread and creates a copy of that entry
in it's call-graph, ensuring that when merged into the primary thread at the end,
the accumulated call-graph across all threads is inserted into the appropriate
location. This approach has been found to produce the fewest number of artifacts.

In general, components do not need to conform to a specific interface. This is
relatively unique approach. Most performance analysis which allow user extensions
use callbacks and dynamic polymorphism to integrate the user extensions into their
workflow. It should be noted that there is nothing preventing a component from creating a
similar system but timemory is designed to query the presence of member function _names_ for
feature detection and adapts accordingly to the overloads of that function name and
it's return type. This is all possible due to the template-based design which makes
extensive use of variadic functions to accept any arguments at a high-level and
SFINAE to decide at compile-time which function to invoke (if a function is invoked at all).
For example:

- component A can contain these member functions:
    - `void start()`
    - `int get()`
    - `void set_prefix(const char*)`
- component B can contains these member functions:
    - `void start()`
    - `void start(cudaStream_t)`
    - `double get()`
- component C can contain these member functions:
    - `void start()`
    - `void set_prefix(const std::string&)`

And for a given bundle `component_tuple<A, B, C> obj`:

- When `obj` is created, a string identifer, instance of a `source_location` struct, or a hash is required
    - This is the label for the measurement
    - If a string is passed, `obj` generates the hash and adds the hash and the string to a hash-map if it didn't previously exist
    - `A::set_prefix(const char*)` will be invoked with the underlying `const char*` from the string that the hash maps to in the hash-map
    - `C::set_prefix(const std::string&)` will be invoked with string that the hash maps to in the hash-map
    - It will be detected that `B` does not have a member function named `set_prefix` and no member function will be invoked
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

This design makes has several benefits and one downside in particular. The benefits
are that timemory: (1) makes it extremely easy to create a unified interface between two
or more components which different interfaces/capabilities, (2) invoking
the different interfaces is efficient since no feature detection logic is required at
runtime, and (3) components define their own interface.

With respect to #2, consider the two more traditional implementations. If callbacks are
used, a function pointer exists and a component which does not implement a feature
will either have a null function pointer (requiring a check at runtime time) or the
tool will implement an array of function pointers with an unknown size at compile-time.
In the latter case, this will require heap allocations (which are expensive operations) and
in both cases, the loop of the function pointers will likely be quite ineffienct
since function pointers have a very high probability of thrashing the instruction cache.
If dynamic polymorphism is used, then virtual table look-ups are required
during every iteration. In the timemory approach, none of these additional overheads
are present and there isn't even a loop -- the bundle either expands into a direct call to the
member function without any abstractions or nothing.

With respect to #1 and #3, this has some interesting implications with regard to a
universal instrumentation interface and is discussed in the following section and
the [CONTRIBUTING.md](CONTRIBUTING.md) documentation.

The aforementioned downside is that the byproduct of all this flexibility and adaption
to custom interfaces by each component is that directly using the template interface
can take quite a long time to compile.

## Support for timemory in external tools

The previous section gave an overview of how components can define their own interfaces.
Currently, timemory internally provides compatibility with multiple tools but the end
goal is for the majority of this to be maintained by the authors of the tool. This will
benefits users by provided a single method for using all of their favorite tools and
make it extremely easy for them to try out new tools.
This will benefit the authors of the tools because there will be a significantly
lower the introduction barrier required for users to try out the new tool -- if the
user is familiar with timemory, the tool can be trivially integrated into either their
code or into the profiler.

An external tool can easily provide compatibility with timemory and leverage
all of its work creating a low-overhead measurement system in parallel environments,
Python extensions, and dynamic instrumentation, by simply providing a header in their
source code which defines the interface the tool wants to provide and the tools can
add/remove support at will without having to maintain any source code in
timemory or worry about version compatability with timemory. Versioning issues do
not inherently exist because for several reasons which are detailed
the [CONTRIBUTING.md](CONTRIBUTING) documentation.

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

## Profiling and timemory

Timemory includes the [timemory-run](source/tools/timemory-run/README.md) as a full profiler for Linux systems.
This executable supports dynamic instrumentation (instrumenting at the target applicaiton's runtime), attaching
to a running process, and binary re-writing (creating a new instrumented binary). The instrumented applications
support flat-profiling, call-stack profiling, and timeline profiling and can be configured to use any of the
components timemory provides or, with a little work, can also be used to instrument custom components defined by the user. It is highly recommended for custom tools targetting specific functions to use the combination of
GOTCHA and the dynamic instrumentation. Using the [GOTCHA extensions](#gotcha-and-timemory) for
profiling specific functions enables creating components which replace the function or audit the
incoming arguments and return values for the functions and the dynamic instrumentation makes it
easy to inject using the GOTCHA wrappers into an executable or library.

## Interface Basics

Timemory can be quite useful as the backend for creating your own profiling interface,
but a frontend interface is also provided for those who want to do quick performance
analysis. The following is an example in many languages for collecting the total cache misses
in the L1, L2, and L3 cache levels. The particular hardware counters can be set
in the environment or directly in C++ or Python. See [timemory-avail](source/tools/timemory-avail/README.md)
documentation for more settings and descriptions.

- Environment:
    - `export TIMEMORY_PAPI_EVENTS="PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM"`
- C++:
    - `tim::settings::papi_events() = "PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM"`
- Python:
    - `timemory.settings.papi_events = "PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM"`

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
// access to data
auto hwc = roi.get();
// print
std::cout << hwc << '\n';
```

Or encoding the PAPI enumeration types explicitly:

```cpp
using hwcounters_t = tim::component_tuple<
    tim::component::papi_tuple<PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM>>;

hwcounters_t roi("MY_REGION_OF_INTEREST");
roi.start();
//
// do something in region of interest...
//
roi.stop();
// access to data
auto hwc = roi.get();
// print
std::cout << hwc << '\n';
```

### Python Context Manager

```python
from timemory.util import marker

with marker(["papi_vector"], key="MY_REGION_OF_INTEREST"):
    #
    # do something in region of interest...
    #
```

### Python Component Class

```python
from timemory.component import PapiVector

hwc = PapiVector("MY_REGION_OF_INTEREST")
hwc.start()
//
// do something in region of interest...
//
hwc.stop()
// get values
l1_tcm, l2_tcm, l3_tcm = hwc.get()
// print as string
print("{}".format(hwc))
```

### C Enumeration Interface

```cpp
void* roi = TIMEMORY_BLANK_MARKER("MY_REGION_OF_INTEREST", PAPI_VECTOR);
//
// do something in region of interest...
//
FREE_TIMEMORY_MARKER(roi)
```

## Create Your Own Tools/Components

- Written in C++
- Direct access to performance analysis data in Python and C++
- Create your own components: any one-time measurement or start/stop paradigm can be wrapped with timemory
    - Flexible and easily extensible interface: no data type restrictions in custom components

### Composable Components Example

Building a brand-new component is simple and straight-forward.
In fact, new components can simply be composites of existing components.
For example, if a component for measuring the FLOP-rate (floating point operations per second)
is desired, it is arbitrarily easy to create and this new component will have all the
features of `wall_clock` and `papi_vector` component:

```cpp
// This "component" is for conceptual demonstration only
// It is not intended to be copy+pasted
struct flop_rate : base<flop_rate, double>
{
private:
    wall_clock  wc;
    papi_vector hw;

public:
    static void global_init()
    {
        papi_vector::add_event(PAPI_DP_OPS);
    }

    void start()
    {
        wc.start();
        hw.start();
    }

    void stop()
    {
        wc.stop();
        hw.stop();
    }

    auto get() const
    {
        return hw.get() / wc.get();
    }
};
```

### Extended Example

The simplicity of creating a custom component that inherits category-based formatting properties
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
// uses comma operator to call init() during static construction of boolean
static bool is_initialized = (init(), true);
```

## Compilation with the Template Interface

It was noted above that direct use of the template interface can introduce
long compile-times. However, this interface is extremely powerful and one
might be tempted to use it directly.
The 2011 standard of C++ introduced the concept of an `extern template`
and it is highly recommended to use this feature if the template interface
is used. In general, a project using the template interface should have
a header which declares the component bundle as an `extern template` at the end.
Here is example of what this might look like:

```cpp
#include <timemory/variadic/component_bundle.hpp>
#include <timemory/variadic/auto_bundle.hpp>
#include <timemory/components/types.hpp>
#include <timemory/macros.hpp>

// create an API for your project
TIMEMORY_DEFINE_API(FooBenchmarking)

#if defined(DISABLE_BENCHMARKING)
// this will elimiate all components from the component_bundle or auto_bundle
// with 'api::FooBenchmarking' as the first template parameter
// e.g. bundle<Foo, ...> turns into bundle<Foo> (no components)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, api::FooBenchmarking, false_type)
#endif

// this structure will:
//  - Always record:
//      - wall-clock timer
//      - cpu-clock timer
//      - cpu utilization
//      - Any tools which downstream users inject into the user_global_bundle
//          - E.g. 'user_global_bundle::configure<peak_rss>()'
//  - Optionally enable activating (at runtime):
//      - PAPI hardware counters
//      - GPU kernel tracing
//      - GPU hardware counters
//      - The '*' at the end is what designates the component as optional
#if !defined(FOO_TOOLSET)
#define FOO_TOOLSET                             \
    tim::component_bundle<                      \
        tim::api::FooBenchmarking,              \
        tim::component::wall_clock,             \
        tim::component::cpu_clock,              \
        tim::component::cpu_util,               \
        tim::component::user_global_bundle,     \
        tim::component::papi_vector*,           \
        tim::component::cupti_activity*,        \
        tim::component::cupti_counters*>
#endif

namespace foo
{
namespace benchmark
{
using bundle_t = FOO_TOOLSET;
using auto_bundle_t = typename FOO_TOOLSET::auto_type;
}
}

//  THIS WILL MAKE SURE THE TEMPLATE NEVER GETS INSTANTIATED
//  LEADING TO SIGNIFICANTLY REDUCED COMPILE TIMES
#if !defined(FOO_BENCHMARKING_SOURCE)
extern template class FOO_TOOLSET;
#endif
```

And then in the __*one*__ source file:

```cpp
// avoid the extern template declaration
// make sure this is defined before inclusing the header
#define FOO_BENCHMARKING_SOURCE

// include the header with the code from the previous block
#include "/path/to/header/file"

// pull in all the definitions required to instantiate the template
#include <timemory/timemory.hpp>

// provide an instantiation
template class FOO_TOOLSET;
```

A similar scheme to the above is used extensively internally by timemory --
the source code contains many _almost_ empty `.cpp` files which contain
only a single line of code: `#include "timemory/<some-path>/extern.hpp`.
These source files are part of the scheme for pre-compiling many of the expensive
template instantiations (the templated storage class, in particular), not junk
files that were accidentally committed. In this
scheme, when the `.cpp` file is compiled a macro is used to transform the
statement in the header into a template instantiation but when included
from other headers, the macro transforms the statement into an extern
template declaration. In general, this is how it is implemented:

```cmake
#
# source/timemory/components/foo/CMakeLists.txt
#
add_library(foo SHARED <OTHER_FILES> extern.cpp)
target_compile_definitions(foo
    #  extern.cpp will be compiled with -DTIMEMORY_FOO_SOURCE
    PRIVATE     TIMEMORY_FOO_SOURCE
    #  When the "foo" target part of a 'target_link_libraries(...)'
    #  command by another target downstream, CMake will add
    #  -DTIMEMORY_USE_FOO_EXTERN to the compile definitions
    INTERFACE   TIMEMORY_USE_FOO_EXTERN)
````

```cpp
//
// source/timemory/components/foo/extern.hpp
//
#if defined(TIMEMORY_FOO_SOURCE)
#   define FOO_EXTERN_TEMPLATE(...) template __VA_ARGS__;
#elif defined(TIMEMORY_USE_FOO_EXTERN)
#   define FOO_EXTERN_TEMPLATE(...) extern template __VA_ARGS__;
#else
#   define FOO_EXTERN_TEMPLATE(...)
#endif

// in header-only mode, the macro makes the code disappear
FOO_EXTERN_TEMPLATE(tim::component::base<Foo>)
FOO_EXTERN_TEMPLATE(tim::operation::start<Foo>)
FOO_EXTERN_TEMPLATE(tim::operation::stop<Foo>)
FOO_EXTERN_TEMPLATE(tim::storage<Foo>)
```

```cpp
//
// source/timemory/components/foo/extern.cpp
//
#include "timemory/components/foo/extern.hpp"
```

## Additional Information

For more information, refer to the [documentation](https://timemory.readthedocs.io/en/latest/).
