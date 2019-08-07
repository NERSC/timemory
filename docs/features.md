# Features

## Cross-Language Support: C, C++, CUDA, and Python

It is very common for Python projects to implement expensive routines in C or C++. Implementing a TiMemory auto-tuple in any combination of these languages will produce one combined report for all the languages (provided each language links to the same library).
However, this is a feature of TiMemory. TiMemory can be used in standalone C, C++, or Python projects.

## Multithreading

The multithreading overhead is essentially zero.
All TiMemory components use static thread-local singletons of call-graps that are automatically created when a
new thread starts recording a component. The state of the singleton on the master thread is bookmarked and when the
thread is destroyed, the thread-local call-graph is merged back into the master call-graph. Only during the
one-time merge into the master call-graph is a synchronization lock (mutex) utilized.

## MPI

If a project uses MPI, TiMemory will combined the reports from all the MPI ranks when a report is requested.

## PAPI

PAPI counters are available as a component in the same way timing and rusage components are available. If TiMemory
is not compiled with PAPI, it is safe to keep their declaration in the code and their output will be suppressed.

There are two components for PAPI counters detailed [here](/components#hardware-counter-components).

Example:

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

## CUDA

At this stage, TiMemory implements a `cudaEvent_t` that will record the elapsed time between
two points in the stream pipeline execution. The CUDA documentation for this component
(`tim::component::cuda_event`) can be found
[here](https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/)
and should be constructed on a per-stream basis:

```c++
tim::component::cuda_event evt(stream);
saxpy<<<ngrid, block, 0, stream>>>(N, 1.0f, _dx, _dy);
```

Support for CUPTI (CUDA hardware counters) is in development.

## Plot Generation in Python

The results from TiMemory can be serialized to JSON and the JSON output can be used to produce performance plots
via the standalone `timemory-plotter` or `timemory.plotting` Python module. For roofline analysis,
the `timemory.roofline` module can be used.

## Command-line Tools

The command-line tools `timem` and `pytimem` work like the `time` executable but with more information.
UNIX systems provide `timem` executable that works like `time`. On all systems, `pytimem` is provided.

```shell
$ ./timem sleep 2

> [sleep] total execution time :
        2.119142 sec real
        0.000000 sec user
        0.000000 sec sys
        0.000000 sec cpu
        0.000000 % cpu_util
        0.745472 MB peak_rss
               0 io_in
               0 io_out
             389 minor_page_flts
               0 major_page_flts
               0 num_signals
              46 vol_cxt_swch
              10 prio_cxt_swch
```

## Signal Detection

TiMemory provides a facility for catching signals and printing out a backtrace when the signals are raised:

```cpp
tim::enable_signal_detection({ SIGHUP, SIGINT, SIGQUIT, SIGABRT });
// ...
tim::disable_signal_detection();
```

## Cache Information

TiMemory provides method on Linux, Windows, and macOS to query the size of L1, L2, and L3 cache.
A `get_max()` function is provided for convenience as some systems (e.g. KNL) do not have an L3 cache.

> Namespace: `tim::ert::cache_size`

| Cache level | Function(s)                                                      |
| ----------- | ---------------------------------------------------------------- |
| L1          | `tim::ert::cache_size::get(1)`, `tim::ert::cache_size::get<1>()` |
| L2          | `tim::ert::cache_size::get(2)`, `tim::ert::cache_size::get<2>()` |
| L3          | `tim::ert::cache_size::get(3)`, `tim::ert::cache_size::get<3>()` |
| max         | `tim::ert::cache_size::get_max()`                                |

