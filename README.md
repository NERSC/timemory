# TiMemory
### C / C++ / CUDA / Python Timing + Memory + Hardware Counter Utilities

[![Build Status](https://travis-ci.org/jrmadsen/TiMemory.svg?branch=master)](https://travis-ci.org/jrmadsen/TiMemory)
[![Build status](https://ci.appveyor.com/api/projects/status/8xk72ootwsefi8c1?svg=true)](https://ci.appveyor.com/project/jrmadsen/timemory)

[TiMemory on GitHub (Source code)](https://github.com/jrmadsen/TiMemory)

[TiMemory General Documentation (GitHub Pages)](https://jrmadsen.github.io/TiMemory)

[TiMemory Source Code Documentation (Doxygen)](https://jrmadsen.github.io/TiMemory/doxy/index.html)

[TiMemory Testing Dashboard (CDash)](https://cdash.nersc.gov/index.php?project=TiMemory)

### Overview

TiMemory is generic C++11 template library providing a variety of [performance components](#Components)
for reporting timing, resource usage, and hardware counters for the CPU and GPU.

TiMemory provides also provides Python and C interfaces.

### Purpose

The purpose of the TiMemory package is to provide as easy way to regularly report on the performance
of your code. If you have ever something like this in your code:

```python
tstart = time.now()
# do something
tstop = time.now()
print("Elapsed time: {}".format(tstop - tstart))
```

TiMemory streamlines this work. In C++ codes, all you have to do is include the headers.
It comes in handy especially when optimizing a
certain algorithm or section of your code -- you just insert a line of code that specifies what
you want to measure and run your code: __*initialization and output are automated*__.

TiMemory is not a full profiler and it is not intended to be used in lieu of the profiling,
instead it provides an easy-to-use method for __*always-on general HPC analysis metrics*__
(i.e. timing, memory usage, etc.) with the same or less overhead than if these metrics were to
records and stored in a custom solution (there is zero polymorphism).
__*Functionally, the overhead is non-existant*__: sampling profilers (e.g. gperftools, VTune)
at standard sampling rates barely notice the presence of TiMemory unless it is been
used _very_ unwisely.

Additional tools are provided, such as hardware counters, to __*increase optimization productivity.*__
What to check whether those changes increased data locality (i.e. decreased cache misses) but don't care about any other sections of the code?
Use the following and set `TIMEMORY_PAPI_EVENTS="PAPI_L1_TCM, PAPI_L2_TCM, PAPI_L3_TCM"` in
the environment:

```cpp
using auto_tuple_t = tim::auto_tuple<tim::component::papi_array_t>;
TIMEMORY_AUTO_TUPLE_CALIPER(roi, auto_tuple_t, "");
//
// region of interest
// ...
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

The general suggested approach is create a generic set of component combinations
that are placed at high levels around the core workload pieces with an aliasing mechanism.

```cpp
namespace tim { using namespace component; }
using timemory_tuple_t =
tim::auto_tuple<
        tim::real_clock,
        tim::cpu_clock,
        tim::cpu_util,
        tim::current_rss,
        tim::cuda_event
    >;
```

and then regions can use this type for scoped measurments in several ways with the macros
provided:

- The easiest to use macro is the `TIMEMORY_AUTO_TUPLE` variant. This macro
  uses the object's lifetime to start and stop and the `TIMEMORY_AUTO_TUPLE` and
  `TIMEMORY_BASIC_AUTO_TUPLE` macros automatically create their own labels using the function name,
  file name, and line number.
  - `TIMEMORY_AUTO_TUPLE(type, ...)`
    - encodes function, file, and line number as the base signature for the record
    - First argument is an `auto_tuple` type
    - Any number of remaining arguments (one or more) are converted to a string and appended to the
    the base signature for the record.
    - Example:
      - Code: `TIMEMORY_AUTO_TUPLE(timemory_tuple_t, "[", 1, "]");`
      - Signature: `function[1]@'file.cpp':15`
  - `TIMEMORY_BASIC_AUTO_TUPLE(type, ...)`
    - encodes function name as the base signature for the record
    - First argument is an `auto_tuple` type
    - Any number of remaining arguments (one or more) are converted to a string and appended to the
    the base signature for the record.
    - Example:
      - Code: `TIMEMORY_BASIC_AUTO_TUPLE(timemory_tuple_t, "[", 1, "]");`
      - Signature: `function[1]`
  - `TIMEMORY_BLANK_AUTO_TUPLE(type, ...)`
    - encodes nothing as the base signature for the record by default
    - First argument is an `auto_tuple` type
    - The remaining arguments (one or more) are converted to a string and assigned as the
    the signature for the record.
    - Example:
      - Code: `TIMEMORY_BLANK_AUTO_TUPLE(timemory_tuple_t, "[", 1, "]");`
      - Signature: `[1]`
- The `*_CALIPER` variations exist within a scope but can be used in situations where the
  construction and destruction of the `auto_tuple` does not provide the necessary representation
  of the region of interest.
  - Note on the first arguments of these types:
    - The first tag argument gets converted into a suffix for a variable by the preprocessor so
    it can be any identifier other than a string.
      - In other words, if you use `1` or `main` as the tag, a variable `_tim_auto_variable1`
      or `_tim_auto_variablemain` is created. If you try to use `"main"`, the (illegal) variable
      name constructed is `_tim_auto_variable"main"`.
  - `TIMEMORY_AUTO_TUPLE_CALIPER(tag, type, ...)`
    - The remaining arguments after tag are same as `TIMEMORY_AUTO_TUPLE`
  - `TIMEMORY_BASIC_AUTO_TUPLE_CALIPER(tag, type, ...)`
    - The remaining arguments after tag are same as `TIMEMORY_BASIC_AUTO_TUPLE`
  - `TIMEMORY_BLANK_AUTO_TUPLE_CALIPER(tag, type, ...)`
    - The remaining arguments after tag are same as `TIMEMORY_BLANK_AUTO_TUPLE`
  - Recommendation: use the `TIMEMORY_CALIPER_APPLY(tag, member_function)` macro to call
  member functions on these objects
- The `*_INSTANCE` variants exist to provide an actual object that the developer controls.
  - These types can be helpful when a variable is needed but deploying the
  [Optional TiMemory Usage](#Optional-TiMemory-Usage) technique but holding onto a variable is desired.
  - When TiMemory is disable in the optional usage technique, this macro can be reassigned to simply
  provide an int, nullptr, dummy no-op class, etc.
  - `TIMEMORY_AUTO_TUPLE_INSTANCE(type, ...)`
    - All arguments are identical to `TIMEMORY_AUTO_TUPLE`
    - Usage: `auto measurement = TIMEMORY_AUTO_TUPLE_INSTANCE(timemory_tuple_t, "[", 1, "]");`
  - `TIMEMORY_BASIC_AUTO_TUPLE_INSTANCE(type, ...)`
    - All arguments are identical to `TIMEMORY_BASIC_AUTO_TUPLE`
    - Usage: `auto measurement = TIMEMORY_BASIC_AUTO_TUPLE_INSTANCE(timemory_tuple_t, "[", 1, "]");`
  - `TIMEMORY_BLANK_AUTO_TUPLE_INSTANCE(type, ...)`
    - All arguments are identical to `TIMEMORY_BLANK_AUTO_TUPLE`
    - Usage: `auto measurement = TIMEMORY_BLANK_AUTO_TUPLE_INSTANCE(timemory_tuple_t, "[", 1, "]");`



### Installation


```shell
$ git clone https://github.com/jrmadsen/TiMemory.git timemory
$ mkdir -p timemory/build-timemory
$ cd timemory/build-timemory
$ cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
$ cmake --build . --target INSTALL
```
