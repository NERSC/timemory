# TiMemory
### C / C++ / CUDA / Python Timing + Memory + Hardware Counter Utilities

[![Build Status](https://travis-ci.org/jrmadsen/TiMemory.svg?branch=master)](https://travis-ci.org/jrmadsen/TiMemory)
[![Build status](https://ci.appveyor.com/api/projects/status/8xk72ootwsefi8c1?svg=true)](https://ci.appveyor.com/project/jrmadsen/timemory)

[TiMemory on GitHub (Source code)](https://github.com/jrmadsen/TiMemory)

[TiMemory General Documentation (GitHub Pages)](https://jrmadsen.github.io/TiMemory)

[TiMemory Source Code Documentation (Doxygen)](https://jrmadsen.github.io/TiMemory/doxy/index.html)

[TiMemory Testing Dashboard (CDash)](https://cdash.nersc.gov/index.php?project=TiMemory)

[TiMemory Release Notes](https://jrmadsen.github.io/TiMemory/ReleaseNotes.html)

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
instead __*it provides an simplified method to check your changes have not degraded performance
or increased resource utilization*__.
For example, if the API provides a generic set of components around the core workload pieces,
these can be saved

### Installation

- Requirements
  - C++ compiler (GNU, MSVC, Clang, Intel, PGI)
  - CMake >= 3.10

```shell
$ git clone https://github.com/jrmadsen/TiMemory.git timemory
$ mkdir -p timemory/build-timemory
$ cd timemory/build-timemory
$ cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
$ cmake --build . --target INSTALL
```

### Getting Started

For C++ projects, basic functionality simply requires including the header path,
e.g. `-I/usr/include` if `timemory.hpp` is in `/usr/local/timemory/timemory.hpp`.
However, this will not enable additional capabilities such as PAPI, CUPTI, CUDA kernel timing,
extern templates, etc.

#### C++

```c++
//--------------------------------------------------------------------//
// STEP 1: include header (REQUIRED)
//
#include <timemory/timemory.hpp>
//
//--------------------------------------------------------------------//


//--------------------------------------------------------------------//
// STEP 2: declare some types (OPTIONAL)
//
using auto_tuple_t =
    tim::auto_tuple<
        tim::component::real_clock,     // wall-clock timer
        tim::component::cpu_clock,      // cpu-clock timer
        tim::component::cpu_util,       // (cpu-time / wall-time) * 100
        tim::component::peak_rss,       // high-water mark of memory
        tim::component::papi_tuple<0, PAPI_TOT_CYC> // total cycles
    >;

using comp_tuple_t = typename auto_tuple_t::component_type;
//
// NOTE:
//   "tim::auto_tuple<...>" will statically filter out any components
//   that are not available, e.g. "papi_tuple" if TIMEMORY_USE_PAPI
//   is not defined at compile time.
//--------------------------------------------------------------------//


intmax_t fibonacci(intmax_t n)
{
    //----------------------------------------------------------------//
    // STEP 3: add a timer to function (OPTIONAL)
    //
    tim::auto_tuple<tim::component::real_clock> timer(__FUNCTION__);
    //
    //----------------------------------------------------------------//
    return (n < 2) ? n : fibonacci(n-1) + fibonacci(n-2);
}


int main(int argc, char** argv)
{
    //----------------------------------------------------------------//
    // STEP 4: configure output and parse env  (OPTIONAL)
    //
    tim::timemory_init(argc, argv);
    //
    //----------------------------------------------------------------//

    comp_tuple_t main("overall timer", true);
    main.start();
    for(auto n : { 10, 11, 12})
    {
        // "tim::str::join" is similar to this in Python:
        //
        //      args = [ "fibonacci(", n, ")" ]
        //      "".join(args)
        //
        auto_tuple_t t(tim::str::join("", "fibonacci(", n, ")"));
        auto ret = fibonacci(n);
        printf("fibonacci(%i) = %li\n", n, ret);
    }
    main.stop();
    std::cout << main << std::endl;
}
```

Compile:

```shell
$ g++ -O3 -I/usr/local example.cc -o example
```

Output:

```shell
fibonacci(10) = 55
fibonacci(11) = 89
fibonacci(12) = 144
> [cxx] overall timer :  0.002 sec real,  0.000 sec cpu,   0.0 % cpu_util,   0.1 MB peak_rss [laps: 1]

[graph_storage<peak_rss>]> Outputting 'timemory-test-cxx-roofline-output/peak_rss.txt'... Done
> [cxx] overall timer :   0.1 MB peak_rss, 1 laps
> [cxx] fibonacci(10) :   0.1 MB peak_rss, 1 laps
> [cxx] fibonacci(11) :   0.0 MB peak_rss, 1 laps
> [cxx] fibonacci(12) :   0.0 MB peak_rss, 1 laps

[graph_storage<cpu_util>]> Outputting 'timemory-test-cxx-roofline-output/cpu_util.txt'... Done
> [cxx] overall timer :   0.0 % cpu_util, 1 laps
> [cxx] fibonacci(10) :   0.0 % cpu_util, 1 laps
> [cxx] fibonacci(11) :   0.0 % cpu_util, 1 laps
> [cxx] fibonacci(12) :   0.0 % cpu_util, 1 laps

[graph_storage<cpu>]> Outputting 'timemory-test-cxx-roofline-output/cpu.txt'... Done
> [cxx] overall timer :  0.000 sec cpu, 1 laps
> [cxx] fibonacci(10) :  0.000 sec cpu, 1 laps
> [cxx] fibonacci(11) :  0.000 sec cpu, 1 laps
> [cxx] fibonacci(12) :  0.000 sec cpu, 1 laps

[graph_storage<real>]> Outputting 'timemory-test-cxx-roofline-output/real.txt'... Done
> [cxx] overall timer                   :  0.002 sec real, 1 laps
> [cxx] fibonacci(10)                   :  0.000 sec real, 1 laps
> [cxx] fibonacci                       :  0.000 sec real, 1 laps
> [cxx] |_fibonacci                     :  0.000 sec real, 2 laps
> [cxx]   |_fibonacci                   :  0.000 sec real, 4 laps
> [cxx]     |_fibonacci                 :  0.000 sec real, 8 laps
> [cxx]       |_fibonacci               :  0.000 sec real, 16 laps
> [cxx]         |_fibonacci             :  0.000 sec real, 32 laps
> [cxx]           |_fibonacci           :  0.000 sec real, 52 laps
> [cxx]             |_fibonacci         :  0.000 sec real, 44 laps
> [cxx]               |_fibonacci       :  0.000 sec real, 16 laps
> [cxx]                 |_fibonacci     :  0.000 sec real, 2 laps
> [cxx] fibonacci(11)                   :  0.001 sec real, 1 laps
> [cxx] fibonacci                       :  0.001 sec real, 1 laps
> [cxx] |_fibonacci                     :  0.001 sec real, 2 laps
> [cxx]   |_fibonacci                   :  0.001 sec real, 4 laps
> [cxx]     |_fibonacci                 :  0.001 sec real, 8 laps
> [cxx]       |_fibonacci               :  0.001 sec real, 16 laps
> [cxx]         |_fibonacci             :  0.001 sec real, 32 laps
> [cxx]           |_fibonacci           :  0.000 sec real, 62 laps
> [cxx]             |_fibonacci         :  0.000 sec real, 84 laps
> [cxx]               |_fibonacci       :  0.000 sec real, 58 laps
> [cxx]                 |_fibonacci     :  0.000 sec real, 18 laps
> [cxx]                   |_fibonacci   :  0.000 sec real, 2 laps
> [cxx] fibonacci(12)                   :  0.001 sec real, 1 laps
> [cxx] fibonacci                       :  0.001 sec real, 1 laps
> [cxx] |_fibonacci                     :  0.001 sec real, 2 laps
> [cxx]   |_fibonacci                   :  0.001 sec real, 4 laps
> [cxx]     |_fibonacci                 :  0.001 sec real, 8 laps
> [cxx]       |_fibonacci               :  0.001 sec real, 16 laps
> [cxx]         |_fibonacci             :  0.001 sec real, 32 laps
> [cxx]           |_fibonacci           :  0.001 sec real, 64 laps
> [cxx]             |_fibonacci         :  0.001 sec real, 114 laps
> [cxx]               |_fibonacci       :  0.000 sec real, 128 laps
> [cxx]                 |_fibonacci     :  0.000 sec real, 74 laps
> [cxx]                   |_fibonacci   :  0.000 sec real, 20 laps
> [cxx]                     |_fibonacci :  0.000 sec real, 2 laps
```

### Usage

In C++ and Python, TiMemory can be added in a single line of code:

#### C++
```cpp
void some_function()
{
    TIMEMORY_AUTO_TUPLE(tim::auto_tuple<real_clock, cpu_clock, peak_rss>, "");
    // ...
}
```

#### Python

```python
@timemory.util.auto_timer()
def some_function():
    # ...
```

#### C

In C, TiMemory requires only two lines of code
```c
void* timer = TIMEMORY_AUTO_TIMER("");
// ...
FREE_TIMEMORY_AUTO_TIMER(timer);
```

When the application terminates, output to text and JSON is automated.

### Components

- `cpu_clock`
  - records the CPU clock time (user + kernel time)
- `cpu_util`
  - records the CPU utilization
- `cuda_event` (__GPU__)
  - records a CUDA kernel runtime
  - (currently) care must be take to make sure the stream is synchronized before
    the component is destroyed
    - A callback system is being devised to fix this restriction
- `current_rss`
  - records the current resident-set size via number of pages allocated times the size of a page
- `data_rss`
  - records the integral value of the amount of unshared memory residing in the data segment of a process
- `monotonic_clock`
  - clock that increments monotonically, tracking the time since an arbitrary point, and will continue to increment while the system is asleep.
- `monotonic_raw_clock`
  - clock that increments monotonically, tracking the time since an arbitrary point like `monotonic_clock`.  However, this clock is unaffected by frequency or time adjustments. It should not be compared to other system time sources.
- `num_io_in`
  - records the number of times the file system had to perform input.
- `num_io_out`
  - records the number of times the file system had to perform output.
- `num_major_page_faults`
  - records the number of page faults serviced that required I/O activity.
- `num_minor_page_faults`
  - records the number of page faults serviced without any I/O activity; here I/O activity is avoided by reclaiming a page frame from the list of pages awaiting reallocation.
- `num_msg_recv`
  - records number of IPC messages received.
- `num_msg_sent`
  - records the number of IPC messages sent.
- `num_signals`
  - records the number of signals delivered.
- `num_swap`
  - records the number of swaps out of main memory
- `papi_tuple<EventSet, EventTypes...>` (__Hardware counters__)
  - records a compile-time specified list of PAPI counters
- `papi_array<EventSet, N>` (__Hardware counters__)
  - records a variable set of PAPI counters up to size _N_
- `cpu_roofline<EventTypes...>` (__Hardware counters__)
  - records a CPU roofline calculation based on the specified set of PAPI counters
- `peak_rss`
  - records the peak resident-set size ("high-water" memory mark)
- `priority_context_switch`
  - records the number of times a context switch resulted due to a higher priority process becoming runnable or because the current process exceeded its time slice.
- `process_cpu_clock`
  - records the CPU time within the current process (excludes child processes) clock that tracks the amount of CPU (in user- or kernel-mode) used by the calling process.
- `process_cpu_util`
  - records the CPU utilization as `process_cpu_clock` / `wall_clock`
- `real_clock`
  - records the system's real time (i.e. wall time) clock, expressed as the amount of time since the epoch.
- `wall_clock`
  - alias to `real_clock` for convenience
- `virtual_clock`
  - alias to `real_clock` since time is a construct of our consciousness
- `stack_rss`
  - records the integral value of the amount of unshared memory residing in the stack segment of a process
- `system_clock`
  - records only the CPU time spent in kernel-mode
- `thread_cpu_clock`
  - the CPU time within the current thread (excludes sibling/child threads)
  - tracks the amount of CPU (in user- or kernel-mode) used by the calling thread.
- `thread_cpu_util`
  - records the CPU utilization as `thread_cpu_clock` / `wall_clock`
- `user_clock`
  - records the CPU time spent in user-mode
- `voluntary_context_switch`
  - records the number of times a context switch resulted due to a process voluntarily giving up the processor before its time slice was completed (usually to await availability of a resource).

### Design is aimed at routine ("everyday") timing and memory analysis that can be standard part of the source code.

TiMemory is a very _lightweight_, _cross-language_ timing, resource usage, and hardware counter utility. It support implementation in C, C++, and Python and is easily imported into CMake projects.

### Lightweight and Fast

Analysis on a fibonacci calculation determined that one TiMemory "component" adds an average overhead of 3 microseconds (`0.000003 s`) when the component is being inserted into call-graph for the first time. Once a component exists in
the call-graph, the overhead is approximately 1.25 microseconds. For example, in the following:

```c++
int64_t
fibonacci(int64_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        TIMEMORY_BASIC_AUTO_TUPLE(tim::auto_tuple<real_clock>, "[", n, "]");
        return (n < 2) ? n : (fibonacci(n - 2, cutoff) + fibonacci(n - 1, cutoff));
    }
    return fibonacci(n); // standard fibonacci (no timers)
}
```

every single instance is unique and the overhead fibonacci(43, 16) produces 514191 unique timers:

```shell
> [cxx] run [with timing = false]@'test_cxx_overhead.cpp':110             : 1.667e+00 sec real, 1 laps
> [cxx] run [with timing =  true]@'test_cxx_overhead.cpp':110             : 2.782e+00 sec real, 1 laps
> [cxx] fibonacci[43]                                                     : 2.782e+00 sec real, 1 laps
> [cxx] |_fibonacci[41]                                                   : 1.033e+00 sec real, 1 laps
> [cxx]   |_fibonacci[39]                                                 : 3.868e-01 sec real, 1 laps
> [cxx]     |_fibonacci[37]                                               : 1.467e-01 sec real, 1 laps
> [cxx]       |_fibonacci[35]                                             : 5.519e-02 sec real, 1 laps
> [cxx]         |_fibonacci[33]                                           : 2.151e-02 sec real, 1 laps
> [cxx]           |_fibonacci[31]                                         : 8.197e-03 sec real, 1 laps
> [cxx]             |_fibonacci[29]                                       : 3.063e-03 sec real, 1 laps
> [cxx]               |_fibonacci[27]                                     : 1.148e-03 sec real, 1 laps
> [cxx]                 |_fibonacci[25]                                   : 4.421e-04 sec real, 1 laps
> [cxx]                   |_fibonacci[23]                                 : 1.718e-04 sec real, 1 laps
> [cxx]                     |_fibonacci[21]                               : 6.159e-05 sec real, 1 laps
> [cxx]                       |_fibonacci[19]                             : 2.116e-05 sec real, 1 laps
> [cxx]                         |_fibonacci[17]                           : 6.281e-06 sec real, 1 laps
> [cxx]                         |_fibonacci[18]                           : 1.172e-05 sec real, 1 laps
> [cxx]                           |_fibonacci[17]                         : 6.146e-06 sec real, 1 laps
> [cxx]                       |_fibonacci[20]                             : 3.766e-05 sec real, 1 laps
> [cxx]                         |_fibonacci[18]                           : 1.156e-05 sec real, 1 laps
> [cxx]                           |_fibonacci[17]                         : 6.183e-06 sec real, 1 laps
> [cxx]                         |_fibonacci[19]                           : 2.318e-05 sec real, 1 laps
> [cxx]                           |_fibonacci[17]                         : 6.166e-06 sec real, 1 laps
> [cxx]                           |_fibonacci[18]                         : 1.319e-05 sec real, 1 laps
> [cxx]                             |_fibonacci[17]                       : 6.180e-06 sec real, 1 laps
> [cxx]                     |_fibonacci[22]                               : 1.072e-04 sec real, 1 laps

...

> [cxx] run [with timing = false]@'test_cxx_overhead.cpp':110 : 1.667e+00 sec real [laps: 1]
> [cxx] run [with timing =  true]@'test_cxx_overhead.cpp':110 : 2.782e+00 sec real [laps: 1]
> [cxx] timing difference                                     : 1.115e+00 sec real
> [cxx] average overhead per timer                            : 2.168e-06 sec real
```

However, the following produces only 27 unique timers:

```c++
int64_t
fibonacci(int64_t n, int64_t cutoff)
{
    if(n > cutoff)
    {
        TIMEMORY_BASIC_AUTO_TUPLE(tim::auto_tuple<real_clock>, "");
        return (n < 2) ? n : (fibonacci(n - 2, cutoff) + fibonacci(n - 1, cutoff));
    }
    return fibonacci(n); // standard fibonacci (no timers)
}
```

and the overhead is much smaller:

```shell
> [cxx] run [with timing = false]@'test_cxx_overhead.cpp':110         : 2.220e+00 sec real, 1 laps
> [cxx] run [with timing =  true]@'test_cxx_overhead.cpp':110         : 2.832e+00 sec real, 1 laps
> [cxx] fibonacci                                                     : 2.832e+00 sec real, 1 laps
> [cxx] |_fibonacci                                                   : 2.832e+00 sec real, 2 laps
> [cxx]   |_fibonacci                                                 : 2.832e+00 sec real, 4 laps
> [cxx]     |_fibonacci                                               : 2.832e+00 sec real, 8 laps
> [cxx]       |_fibonacci                                             : 2.832e+00 sec real, 16 laps
> [cxx]         |_fibonacci                                           : 2.832e+00 sec real, 32 laps
> [cxx]           |_fibonacci                                         : 2.832e+00 sec real, 64 laps
> [cxx]             |_fibonacci                                       : 2.832e+00 sec real, 128 laps
> [cxx]               |_fibonacci                                     : 2.831e+00 sec real, 256 laps
> [cxx]                 |_fibonacci                                   : 2.831e+00 sec real, 512 laps
> [cxx]                   |_fibonacci                                 : 2.830e+00 sec real, 1024 laps
> [cxx]                     |_fibonacci                               : 2.828e+00 sec real, 2048 laps
> [cxx]                       |_fibonacci                             : 2.824e+00 sec real, 4096 laps
> [cxx]                         |_fibonacci                           : 2.815e+00 sec real, 8192 laps
> [cxx]                           |_fibonacci                         : 2.798e+00 sec real, 16369 laps
> [cxx]                             |_fibonacci                       : 2.761e+00 sec real, 32192 laps
> [cxx]                               |_fibonacci                     : 2.660e+00 sec real, 58651 laps
> [cxx]                                 |_fibonacci                   : 2.425e+00 sec real, 89846 laps
> [cxx]                                   |_fibonacci                 : 1.977e+00 sec real, 106762 laps
> [cxx]                                     |_fibonacci               : 1.355e+00 sec real, 94184 laps
> [cxx]                                       |_fibonacci             : 7.419e-01 sec real, 60460 laps
> [cxx]                                         |_fibonacci           : 3.124e-01 sec real, 27896 laps
> [cxx]                                           |_fibonacci         : 9.630e-02 sec real, 9109 laps
> [cxx]                                             |_fibonacci       : 2.064e-02 sec real, 2048 laps
> [cxx]                                               |_fibonacci     : 2.952e-03 sec real, 301 laps
> [cxx]                                                 |_fibonacci   : 2.318e-04 sec real, 26 laps
> [cxx]                                                   |_fibonacci : 8.503e-06 sec real, 1 laps

> [cxx] run [with timing = false]@'test_cxx_overhead.cpp':110 : 2.220e+00 sec real [laps: 1]
> [cxx] run [with timing =  true]@'test_cxx_overhead.cpp':110 : 2.832e+00 sec real [laps: 1]
> [cxx] timing difference                                     : 6.116e-01 sec real
> [cxx] average overhead per timer                            : 1.189e-06 sec real
```

The exact performance is specific to the machine and the overhead for a particular machine can be calculated by running the `test_cxx_overhead` example.

Since TiMemory only records information of the functions explicitly specified, you can safely assume that unless
TiMemory is inserted into a function called `> 100,000` times, it won't be adding more than a second of runtime
to the function. Therefore, there is a simple rule of thumb: **don't insert a TiMemory auto-tuple into very simple functions
that get called very frequently**.

TiMemory is not intended to replace profiling tools such as Intel's VTune, GProf, etc. -- instead, it complements them by enabling one to verify timing and memory usage without the overhead of the profiler.

### Cross-Language Support: C, C++, CUDA, and Python

It is very common for Python projects to implement expensive routines in C or C++. Implementing a TiMemory auto-tuple in any combination of these languages will produce one combined report for all the languages (provided each language links to the same library).
However, this is a feature of TiMemory. TiMemory can be used in standalone C, C++, or Python projects.

### Multithreading support

The multithreading overhead of TiMemory is essentially zero.
All TiMemory components use static thread-local singletons of call-graps that are automatically created when a
new thread starts recording a component. The state of the singleton on the master thread is bookmarked and when the
thread is destroyed, TiMemory merges the thread-local call-graph back into the master call-graph. Only during the
merge into the master call-graph is a synchronization lock (mutex) utilized.

### MPI Support

If a project uses MPI, TiMemory will combined the reports from all the MPI ranks when a report is requested.

### PAPI support

PAPI counters are available as a component in the same way timing and rusage components are available. If TiMemory
is not compiled with PAPI, it is safe to keep their declaration in the code and their output will be suppressed.

```c++
using papi_tuple_t = papi_tuple<0, PAPI_TOT_CYC, PAPI_TOT_INS>;
using auto_tuple_t = tim::auto_tuple<real_clock, system_clock, cpu_clock, cpu_util,
                                       peak_rss, current_rss, papi_tuple_t>;

void some_function()
{
    TIMEMORY_AUTO_TUPLE(auto_tuple_t, "");
    // ...
}
```

### CUDA Support

At this stage, TiMemory implements a `cudaEvent_t` that will record the wall-clock runtime of a kernel. If the
kernel is launched in a stream, one must set the stream:

```c++
tim::component::cuda_event evt(stream);
saxpy<<<ngrid, block, 0, stream>>>(N, 1.0f, _dx, _dy);
```

Support for CUPTI (CUDA hardware counters) is planned.

### Built-in plotting

The results from TiMemory can be serialized to JSON and the JSON output can be used to produce performance plots
via the standalone `timemory-plotter` or `timemory.plotting` Python module

### Command-line

UNIX systems provide `timem` executable that works like `time`. On all systems, `pytimem` is provided.

```shell
$ ./timem sleep 5
> [sleep] total execution time :
       5.005e+00 sec real
       0.000e+00 sec sys
       0.000e+00 sec cpu
       0.000e+00 % cpu_util
       6.719e-01 MB peak_rss
             310 minor_page_flts
               1 major_page_flts
               2 vol_cxt_swch
               5 prio_cxt_swch
```

### Environment Controls

- `TIMEMORY_AUTO_OUTPUT`
  - automatic output at the end of application
- `TIMEMORY_COUT_OUTPUT`
  - output to terminal
- `TIMEMORY_ENABLE`
  - enable by default
- `TIMEMORY_FILE_OUTPUT`
  - output to file
- `TIMEMORY_JSON_OUTPUT`
  - enable JSON output
- `TIMEMORY_MAX_DEPTH`
  - max depth
- `TIMEMORY_MEMORY_PRECISION`
  - precision in reporting for memory components
- `TIMEMORY_MEMORY_SCIENTIFIC`
  - output memory components in scientific format
- `TIMEMORY_MEMORY_UNITS`
  - units of memory components
- `TIMEMORY_MEMORY_WIDTH`
  - initial output with of memory components
- `TIMEMORY_OUTPUT_PATH`
  - folder for output
  - default is `timemory-output/`
- `TIMEMORY_OUTPUT_PREFIX`
  - filename prefix, e.g. `${TIMEMORY_OUTPUT_PREFIX}_<component_name>`
  - default is `""`
- `TIMEMORY_PRECISION`
  - precision for all output
- `TIMEMORY_TEXT_OUTPUT`
  - enable/disable text output
- `TIMEMORY_TIMING_PRECISION`
  - precision for timing components
- `TIMEMORY_TIMING_SCIENTIFIC`
  - output timing components in scientific notation
- `TIMEMORY_TIMING_UNITS`
  - units for timing components
- `TIMEMORY_TIMING_WIDTH`
  - initial output width of timing components
- `TIMEMORY_WIDTH`
  - initial width for all components

### CMake Support

**It is highly recommended to use CMake with TiMemory**

TiMemory uses modern CMake INTERFACE targets to include the components you want without
forcing you to include everything -- this means that compiler flags, preprocessor
definitions, include paths, link options, and link libraries are bundled into separate
"library" targets that only need to be "linked" to in CMake:

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

When combined with `find_package`, TiMemory will bundle these targets into one
interface library: `timemory`.

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

#### TiMemory Targets

These are the full target names available within CMake. Some targets may not be available
based on the installation, choice of compiler, etc.

In general, when listed as `COMPONENTS` arguments to a `find_package`, the `timemory-` prefix
can be dropped and the link type (`shared` or `static`), languages (`c`, `cxx`, `cuda`)
can be listed once and dropped from subsequent items in the list of `COMPONENTS`.

- Header libraries
  - `timemory-headers`
    - The include path to TiMemory headers
  - `timemory-cereal`
    - implicitly included with timemory-headers
- Compiled libraries
  - `timemory-c-shared`
  - `timemory-c-static`
  - `timemory-cxx-shared`
  - `timemory-cxx-static`
  - `timemory-cupti-shared`
    - shared library that enables recording NVIDIA hardware events and metrics
    (e.g. profiling counters)
  - `timemory-cupti-static`
    - static library that enables recording NVIDIA hardware events and metrics
    (e.g. profiling counters)
- Alias libraries
  - `timemory-c` == `timemory-c-shared`
  - `timemory-cxx` == `timemory-cxx-shared`
- Extensions
  - `timemory-extensions`
    - all of the extensions below that were found/enabled when configuring TiMemory
  - `timemory-threading`
    - enables any necessary threading flags, e.g. `-lpthread`
  - `timemory-mpi`
    - enables MPI support
  - `timemory-papi`
    - enable PAPI support (CPU hardware counters)
  - `timemory-cuda`
    - enables wall-clock timing CUDA kernels
  - `timemory-gperf`
    - enables using google-perftools with TiMemory
  - `timemory-coverage`
    - adds GNU coverage flags, if available
  - `timemory-sanitizer`
    - adds sanitizer
  - `timemory-memory-sanitizer`
  - `timemory-leak-sanitizer`
  - `timemory-address-sanitier`
  - `timemory-thread-sanitizer`
- Miscellaneous
  - `timemory-c-compile-options`
    - Adds C compiler flags used by TiMemory
  - `timemory-cxx-compile-options`
    - Adds CXX compiler flags used by TiMemory
  - `timemory-arch`
    - enables architecture-specific compiler flags, if available
  - `timemory-avx512`
    -sets AVX-512 compiler flags, if available
  - `timemory-extern-templates`
    - declares a subset of templates as extern to reduce compile time
