# TiMemory
### C / C++ / CUDA / Python Timing + Memory + Hardware Counter Utilities

[![Build Status](https://travis-ci.org/jrmadsen/TiMemory.svg?branch=master)](https://travis-ci.org/jrmadsen/TiMemory)
[![Build status](https://ci.appveyor.com/api/projects/status/8xk72ootwsefi8c1?svg=true)](https://ci.appveyor.com/project/jrmadsen/timemory)

[TiMemory on GitHub (Source code)](https://github.com/jrmadsen/TiMemory)

[TiMemory General Documentation (GitHub Pages)](https://jrmadsen.github.io/TiMemory)

[TiMemory Source Code Documentation (Doxygen)](https://jrmadsen.github.io/TiMemory/doxy/index.html)

[TiMemory Testing Dashboard (CDash)](https://cdash.nersc.gov/index.php?project=TiMemory)

[TiMemory Release Notes](https://jrmadsen.github.io/TiMemory/ReleaseNotes.html)

### TiMemory is easy-to-use

In C++ and Python, TiMemory can be added in a single line of code:

```c++
void some_function()
{
    TIMEMORY_AUTO_TUPLE(tim::auto_tuple<real_clock, cpu_clock, peak_rss>, "");
    // ...
}
```

```python
@timemory.util.auto_timer()
def some_function():
    # ...
```

In C, TiMemory requires only two lines of code
```c
void* timer = TIMEMORY_AUTO_TIMER("");
// ...
FREE_TIMEMORY_AUTO_TIMER(timer);
```

When the application terminates, output to text and JSON is automated.

### TiMemory supports a variety of Components

- `cpu_clock`
  - records the CPU clock time (user + kernel time)
- `cpu_util`
  - records the CPU utilization
- `cuda_event`
  - records a CUDA kernel runtime
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
- `papi_event`
  - records a specified set of PAPI counters
- `peak_rss`
  - records the peak resident-set size ("high-water" memory mark)
- `priority_context_switch`
  - records the number of times a context switch resulted due to a higher priority process becoming runnable or because the current process exceeded its time slice.
- `process_cpu_clock`
  - records the CPU time within the current process (excludes child processes) clock that tracks the amount of CPU (in user- or kernel-mode) used by the calling process.
- `process_cpu_util`
  - records the CPU utilization as `process_cpu_clock` / `wall_clock`
- `real_clock` / `wall_clock`
  - records the system's real time (i.e. wall time) clock, expressed as the amount of time since the epoch.
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

### TiMemory's design is aimed at routine ("everyday") timing and memory analysis that can be standard part of the source code.

TiMemory is a very _lightweight_, _cross-language_ timing, resource usage, and hardware counter utility. It support implementation in C, C++, and Python and is easily imported into CMake projects.

### TiMemory is Lightweight and Fast

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
> [cxx] fibonacci[43]                                                     : 2.966e+00 sec real, 1 laps
> [cxx] |_fibonacci[41]                                                   : 1.124e+00 sec real, 1 laps
> [cxx]   |_fibonacci[39]                                                 : 4.251e-01 sec real, 1 laps
> [cxx]     |_fibonacci[37]                                               : 1.614e-01 sec real, 1 laps
> [cxx]       |_fibonacci[35]                                             : 6.283e-02 sec real, 1 laps
> [cxx]         |_fibonacci[33]                                           : 2.340e-02 sec real, 1 laps
> [cxx]           |_fibonacci[31]                                         : 8.878e-03 sec real, 1 laps
> [cxx]             |_fibonacci[29]                                       : 3.244e-03 sec real, 1 laps
> [cxx]               |_fibonacci[27]                                     : 1.210e-03 sec real, 1 laps
> [cxx]                 |_fibonacci[25]                                   : 4.564e-04 sec real, 1 laps
> [cxx]                   |_fibonacci[23]                                 : 1.697e-04 sec real, 1 laps
> [cxx]                     |_fibonacci[21]                               : 6.388e-05 sec real, 1 laps
> [cxx]                       |_fibonacci[19]                             : 2.171e-05 sec real, 1 laps
> [cxx]                         |_fibonacci[17]                           : 6.402e-06 sec real, 1 laps
> [cxx]                         |_fibonacci[18]                           : 1.209e-05 sec real, 1 laps
> [cxx]                           |_fibonacci[17]                         : 6.377e-06 sec real, 1 laps
> [cxx]                       |_fibonacci[20]                             : 3.918e-05 sec real, 1 laps
> [cxx]                         |_fibonacci[18]                           : 1.198e-05 sec real, 1 laps
> [cxx]                           |_fibonacci[17]                         : 6.365e-06 sec real, 1 laps

...

> [cxx] run [with timing = false]@'test_cxx_overhead.cpp':110 : 1.772e+00 sec real [laps: 1]
> [cxx] run [with timing =  true]@'test_cxx_overhead.cpp':110 : 2.966e+00 sec real [laps: 1]
> [cxx] timing difference                                     : 1.194e+00 sec real
> [cxx] average overhead per timer                            : 2.321e-06 sec real
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
> [cxx] fibonacci                                                     : 2.257e+00 sec real, 1 laps
> [cxx] |_fibonacci                                                   : 2.257e+00 sec real, 2 laps
> [cxx]   |_fibonacci                                                 : 2.257e+00 sec real, 4 laps
> [cxx]     |_fibonacci                                               : 2.257e+00 sec real, 8 laps
> [cxx]       |_fibonacci                                             : 2.257e+00 sec real, 16 laps
> [cxx]         |_fibonacci                                           : 2.257e+00 sec real, 32 laps
> [cxx]           |_fibonacci                                         : 2.257e+00 sec real, 64 laps
> [cxx]             |_fibonacci                                       : 2.257e+00 sec real, 128 laps
> [cxx]               |_fibonacci                                     : 2.257e+00 sec real, 256 laps
> [cxx]                 |_fibonacci                                   : 2.256e+00 sec real, 512 laps
> [cxx]                   |_fibonacci                                 : 2.255e+00 sec real, 1024 laps
> [cxx]                     |_fibonacci                               : 2.254e+00 sec real, 2048 laps
> [cxx]                       |_fibonacci                             : 2.250e+00 sec real, 4096 laps
> [cxx]                         |_fibonacci                           : 2.242e+00 sec real, 8192 laps
> [cxx]                           |_fibonacci                         : 2.227e+00 sec real, 16369 laps
> [cxx]                             |_fibonacci                       : 2.194e+00 sec real, 32192 laps
> [cxx]                               |_fibonacci                     : 2.105e+00 sec real, 58651 laps
> [cxx]                                 |_fibonacci                   : 1.909e+00 sec real, 89846 laps
> [cxx]                                   |_fibonacci                 : 1.543e+00 sec real, 106762 laps
> [cxx]                                     |_fibonacci               : 1.042e+00 sec real, 94184 laps
> [cxx]                                       |_fibonacci             : 5.567e-01 sec real, 60460 laps
> [cxx]                                         |_fibonacci           : 2.260e-01 sec real, 27896 laps
> [cxx]                                           |_fibonacci         : 6.722e-02 sec real, 9109 laps
> [cxx]                                             |_fibonacci       : 1.419e-02 sec real, 2048 laps
> [cxx]                                               |_fibonacci     : 1.981e-03 sec real, 301 laps
> [cxx]                                                 |_fibonacci   : 1.664e-04 sec real, 26 laps
> [cxx]                                                   |_fibonacci : 6.262e-06 sec real, 1 laps

> [cxx] run [with timing = false]@'test_cxx_overhead.cpp':110 : 1.626e+00 sec real [laps: 1]
> [cxx] run [with timing =  true]@'test_cxx_overhead.cpp':110 : 2.257e+00 sec real [laps: 1]
> [cxx] timing difference                                     : 6.310e-01 sec real
> [cxx] average overhead per timer                            : 1.227e-06 sec real
```

The exact performance is specific to the machine and the overhead for a particular machine can be calculated by running the `test_cxx_overhead` example.

Since TiMemory only records information of the functions explicitly specified, you can safely assume that unless
TiMemory is inserted into a function called `> 100,000` times, it won't be adding more than a second of runtime
to the function. Therefore, there is a simple rule of thumb: **don't insert a TiMemory auto-tuple into very simple functions
that get called very frequently**.

TiMemory is not intended to replace profiling tools such as Intel's VTune, GProf, etc. -- instead, it complements them by enabling one to verify timing and memory usage without the overhead of the profiler.

### TiMemory is Cross-Language: C, C++, and Python

It is very common for Python projects to implement expensive routines in C or C++. Implementing a TiMemory auto-tuple in any combination of these languages will produce one combined report for all the languages (provided each language links to the same library).
However, this is a feature of TiMemory. TiMemory can be used in standalone C, C++, or Python projects.

### TiMemory support multithreading

The multithreading overhead of TiMemory is essentially zero.
All TiMemory components use static thread-local singletons of call-graps that are automatically created when a
new thread starts recording a component. The state of the singleton on the master thread is bookmarked and when the
thread is destroyed, TiMemory merges the thread-local call-graph back into the master call-graph. Only during the
merge into the master call-graph is a synchronization lock (mutex) utilized.

### TiMemory supports MPI

If a project uses MPI, TiMemory will combined the reports from all the MPI ranks when a report is requested.

### TiMemory supports PAPI

PAPI counters are available as a component in the same way timing and rusage components are available. If TiMemory
is not compiled with PAPI, it is safe to keep their declaration in the code and their output will be suppressed.

```c++
using papi_tuple_t = papi_event<0, PAPI_TOT_CYC, PAPI_TOT_INS>;
using auto_tuple_t = tim::auto_tuple<real_clock, system_clock, cpu_clock, cpu_util,
                                       peak_rss, current_rss, papi_tuple_t>;

void some_function()
{
    TIMEMORY_AUTO_TUPLE(auto_tuple_t, "");
    // ...
}
```

### TiMemory supports CUDA

At this stage, TiMemory implements a `cudaEvent_t` that will record the wall-clock runtime of a kernel. If the
kernel is launched in a stream, one must set the stream:

```c++
tim::component::cuda_event evt(stream);
saxpy<<<ngrid, block, 0, stream>>>(N, 1.0f, _dx, _dy);
```

Support for CUPTI (CUDA hardware counters) is planned.

### TiMemory has built-in timing and memory plotting

The results from TiMemory can be serialized to JSON and the JSON output can be used to produce performance plots
via the standalone `timemory-plotter` or `timemory.plotting` Python module

### TiMemory can be used from the command-line

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

### TiMemory has environment controls

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