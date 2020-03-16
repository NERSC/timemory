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

## Overview

- Timemory is designed as a __*modular*__ framework for performance measurement and analysis
- The goals of timemory are to provide:
    - __*Common instrumentation framework*__
        - Eliminate need for projects to explicitly support multiple instrumentation frameworks
    - High performance when enabled
    - Low overhead when enabled at compile time but disabled at runtime

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

### Support for Multiple Instrumentation APIs

- NVTX for Nsight-Systems and NVprof
- [LIKWID](https://github.com/RRZE-HPC/likwid)
- [Caliper](https://github.com/LLNL/Caliper)
- [TAU](https://www.cs.uoregon.edu/research/tau/home.php)
- ittnotify (Intel VTune and Advisor)
- OMPT (OpenMP tools)
- MPIP

### Create Your Own Performance and Analysis Tools

- Written in C++
- Direct access to performance analysis data in Python and C++
- Create your own components: any one-time measurement or start/stop paradigm can be wrapped with timemory
    - Flexible and easily extensible interface: no data type restrictions in custom components

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

### Powerful GOTCHA Extensions

- [GOTCHA](https://github.com/LLNL/GOTCHA) is an API for LD_PRELOAD
    - Significantly simplify existing implementations
- Scoped GOTCHA
- Use gotcha component to replace external function calls with own instrumentation
- Use gotcha component to instrument external library calls

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
you want to measure and run your code: initialization and output are automated.

## Profiling and timemory

Timemory is not a full profiler (yet). The ultimate goal is to create a customizable profiler.
Currently, timemory supports explicit instrumentation (i.e. minor modifications to source code)
and explicit wrapping of dynamically-linked functions.
Using profilers are currently important for _discovering where to place timemory markers_ or
_which dynamically function calls to wrap with GOTCHA_.
The library provides an easy-to-use method for always-on general HPC analysis metrics
(i.e. timing, memory usage, etc.) with the same or less overhead than if these metrics were to
records and stored in a custom solution and, for C++ code, extensively
inlined.
Functionally, the overhead is non-existant: sampling profilers (e.g. gperftools, VTune)
at standard sampling rates barely notice the presence of timemory unless it is been
used _very_ unwisely.

Additional tools are provided, such as hardware counters, to increase optimization productivity.
What to check whether those changes increased data locality (i.e. decreased cache misses) but don't care about any other sections of the code?
Use the following and set `TIMEMORY_PAPI_EVENTS="PAPI_L1_TCM,PAPI_L2_TCM,PAPI_L3_TCM"` in
the environment:

```cpp
using hwcounters_t = tim::auto_tuple<tim::component::papi_vector>;
TIMEMORY_CALIPER(roi, hwcounters_t, "");
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
Here, timemory can decrease performance regression identification time.
When timemory is combined with a continuous integration reporting system,
this scramble can be mitigated fairly quickly because the high-level reporting
provided allows one to associate a region and commit with exact performance numbers.
Once timemory has been used to help identify the offending commit and identify the general
region in the offending code, a full profiler should be launched for the fine-grained diagnosis.

## Create Your Own Tools/Components

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
        auto tmp = record();
        accum += (tmp - value);
        value = std::move(tmp);
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
#include <timemory/timemory.hpp>

using namespace tim::component;
using comp_bundle_t = tim::component_tuple_t <wall_clock, tau_marker>;
using auto_bundle_t = tim::auto_tuple_t      <wall_clock, tau_marker>;
// "auto" types automatically start/stop based on scope

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
